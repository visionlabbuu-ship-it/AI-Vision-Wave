"""
=============================================================================
FUSION WEIGHT EXPERIMENT
=============================================================================
Simulate various Spectrum:Camera weight ratios using real audit log data
to find the optimal SPECTRUM_WEIGHT_SCALE that minimises false positives
caused by the spectrum sensor overriding correct camera predictions.

Ground truth strategy:
  - When Camera and Spectrum AGREE → that IS the true class (high confidence).
  - When they DISAGREE → Camera is treated as ground truth (85-90 % empirical
    accuracy vs spectrum's higher FP rate).

The experiment replays every audit row through the Bayesian fusion with
SPECTRUM_WEIGHT_SCALE swept from 0.0 (camera-only) to 2.0 in fine steps.

Outputs:
  - Per-weight accuracy, camera-override rate, spectrum-override rate
  - Confusion matrix at current production weight
  - Recommended weight with the best accuracy
  - Bar chart comparing weight scales (if matplotlib available)

Usage:
    python fusion_weight_experiment.py
"""

import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

# ── Import fusion components from robot.py ──
# We replicate the core fusion math here so we can sweep the weight parameter
# without modifying production code.

FUSION_CLASSES = ["Glass", "Metal", "Paper", "Plastic"]
NUM_CLASSES = len(FUSION_CLASSES)

# Confusion matrices (same as robot.py)
CM_YOLO = np.array([
    [67,  8,  0, 25],
    [ 2, 66, 19, 13],
    [ 0, 10, 87,  3],
    [ 8, 23, 14, 55]
], dtype=float)

CM_SPEC = np.array([
    [56,  9,  4, 31],
    [ 1, 78, 14,  7],
    [ 0,  3, 93,  4],
    [ 3, 13, 18, 66]
], dtype=float)

FUSION_ALPHA = 0.001
FUSION_GAMMA = 3.5

_true_counts = CM_YOLO.sum(axis=1)
_total_samples = _true_counts.sum()
FUSION_PRIOR = (_true_counts + FUSION_ALPHA) / (_total_samples + FUSION_ALPHA * NUM_CLASSES)
LIKELIHOOD_YOLO = (CM_YOLO + FUSION_ALPHA) / (_true_counts[:, None] + FUSION_ALPHA * NUM_CLASSES)
LIKELIHOOD_SPEC = (CM_SPEC + FUSION_ALPHA) / (CM_SPEC.sum(axis=1)[:, None] + FUSION_ALPHA * NUM_CLASSES)


def simulate_fusion(yolo_class, yolo_conf, spec_class, spec_conf, spectrum_weight_scale):
    """
    Run gamma-Bayesian fusion with a given spectrum weight scale.
    Returns the predicted class string.
    """
    if yolo_class not in FUSION_CLASSES:
        return yolo_class
    if spec_class not in FUSION_CLASSES:
        return yolo_class

    yolo_conf_norm = yolo_conf if yolo_conf <= 1.0 else yolo_conf / 100.0
    spec_conf_norm = spec_conf if spec_conf <= 1.0 else spec_conf / 100.0

    w_yolo = yolo_conf_norm ** FUSION_GAMMA
    w_spec = (spec_conf_norm ** FUSION_GAMMA) * spectrum_weight_scale

    idx_yolo = FUSION_CLASSES.index(yolo_class)
    idx_spec = FUSION_CLASSES.index(spec_class)

    log_scores = []
    for c in range(NUM_CLASSES):
        p_yolo = LIKELIHOOD_YOLO[c, idx_yolo]
        p_spec = LIKELIHOOD_SPEC[c, idx_spec]
        p_prior = FUSION_PRIOR[c]
        score = (
            w_yolo * np.log(p_yolo + 1e-12) +
            w_spec * np.log(p_spec + 1e-12) +
            np.log(p_prior + 1e-12)
        )
        log_scores.append(score)

    return FUSION_CLASSES[np.argmax(log_scores)]


def load_audit_data(audit_dir):
    """Load all audit CSVs from the given directory into one DataFrame."""
    pattern = os.path.join(audit_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No CSV files found in {audit_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['_source_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Skip {os.path.basename(f)}: {e}")
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def run_experiment():
    audit_dir = os.path.join(os.path.dirname(__file__),
                             "detection_logs", "audits", "Old")

    print("=" * 75)
    print("  FUSION WEIGHT EXPERIMENT — Spectrum:Camera Ratio Sweep")
    print("=" * 75)
    print()

    # ── Load data ──
    print(f"Loading audit data from: {audit_dir}")
    df = load_audit_data(audit_dir)
    if df.empty:
        print("ERROR: No audit data found.")
        return

    # Filter valid rows
    required = ['Cam_Class', 'Cam_Conf', 'Spec_Class', 'Spec_Conf']
    for col in required:
        df = df[df[col].notna()]
    df = df[df['Cam_Class'].isin(FUSION_CLASSES)]
    df = df[df['Spec_Class'].isin(FUSION_CLASSES)]
    df['Cam_Conf'] = df['Cam_Conf'].astype(float)
    df['Spec_Conf'] = df['Spec_Conf'].astype(float)

    n = len(df)
    print(f"Valid audit rows: {n}")
    print()

    # ── Ground truth assignment ──
    # When camera & spectrum agree → consensus = true class
    # When they disagree → camera = true class (higher empirical accuracy)
    agree_mask = df['Cam_Class'] == df['Spec_Class']
    n_agree = agree_mask.sum()
    n_disagree = n - n_agree
    df['True_Class'] = df['Cam_Class']  # camera as ground truth baseline
    print(f"Agreement rows (cam==spec): {n_agree}/{n} ({100*n_agree/n:.1f}%)")
    print(f"Disagreement rows:          {n_disagree}/{n} ({100*n_disagree/n:.1f}%)")
    print()

    # ── Baseline: camera-only accuracy ──
    cam_correct = (df['Cam_Class'] == df['True_Class']).sum()
    cam_acc = 100.0 * cam_correct / n
    print(f"Camera-only accuracy (baseline):  {cam_acc:.1f}% ({cam_correct}/{n})")

    # ── Baseline: spectrum-only "accuracy" (vs camera truth) ──
    spec_correct = (df['Spec_Class'] == df['True_Class']).sum()
    spec_acc = 100.0 * spec_correct / n
    print(f"Spectrum-only accuracy (vs cam):  {spec_acc:.1f}% ({spec_correct}/{n})")
    print()

    # ── Disagree subset analysis ──
    disagree_df = df[~agree_mask]
    if len(disagree_df) > 0:
        print(f"─── Disagreement analysis ({len(disagree_df)} rows) ───")
        # In disagreement rows, camera IS ground truth, so spec is always "wrong"
        print(f"  Spectrum false-positive rate in disagree rows: 100%")
        print(f"  Most common spec predictions when wrong:")
        spec_wrong = disagree_df['Spec_Class'].value_counts()
        for cls, cnt in spec_wrong.items():
            pct = 100 * cnt / len(disagree_df)
            print(f"    {cls}: {cnt} ({pct:.1f}%)")
        print()
        print(f"  Camera classes being overridden:")
        cam_overridden = disagree_df['Cam_Class'].value_counts()
        for cls, cnt in cam_overridden.items():
            pct = 100 * cnt / len(disagree_df)
            print(f"    {cls}: {cnt} ({pct:.1f}%)")
        print()

    # ── Sweep SPECTRUM_WEIGHT_SCALE ──
    scales = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
              0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]

    print("─" * 75)
    print(f"{'Weight':>8} │ {'Accuracy':>8} │ {'Correct':>7} │ {'Cam Wins':>9} │ "
          f"{'Spec Wins':>10} │ {'Spec FP':>8} │ {'Changed':>7}")
    print("─" * 75)

    results = []
    for scale in scales:
        correct = 0
        cam_wins = 0       # fusion agrees with camera (not spectrum)
        spec_wins = 0      # fusion agrees with spectrum (not camera)
        spec_fp = 0        # spectrum caused a false positive (overrode correct camera)
        changed = 0        # fusion differs from camera-only

        for _, row in df.iterrows():
            fused = simulate_fusion(
                row['Cam_Class'], row['Cam_Conf'],
                row['Spec_Class'], row['Spec_Conf'],
                scale
            )
            true = row['True_Class']
            cam = row['Cam_Class']
            spec = row['Spec_Class']

            if fused == true:
                correct += 1
            if fused != cam:
                changed += 1
            if cam != spec:
                if fused == cam:
                    cam_wins += 1
                elif fused == spec:
                    spec_wins += 1
                    if fused != true:
                        spec_fp += 1

        acc = 100.0 * correct / n
        results.append({
            'scale': scale,
            'accuracy': acc,
            'correct': correct,
            'cam_wins': cam_wins,
            'spec_wins': spec_wins,
            'spec_fp': spec_fp,
            'changed': changed,
        })

        marker = ""
        if scale == 0.5:
            marker = " ◄── CURRENT"
        elif acc == max(r['accuracy'] for r in results):
            marker = " ★"

        print(f"{scale:>8.2f} │ {acc:>7.1f}% │ {correct:>7d} │ {cam_wins:>9d} │ "
              f"{spec_wins:>10d} │ {spec_fp:>8d} │ {changed:>7d}{marker}")

    print("─" * 75)
    print()

    # ── Find best weight ──
    best = max(results, key=lambda r: r['accuracy'])
    current = next(r for r in results if r['scale'] == 0.5)

    print(f"CURRENT production weight:  SPECTRUM_WEIGHT_SCALE = 0.50")
    print(f"  Accuracy: {current['accuracy']:.1f}%  |  Spec overrides: {current['spec_wins']}  |  Spec FP: {current['spec_fp']}")
    print()
    print(f"BEST weight found:          SPECTRUM_WEIGHT_SCALE = {best['scale']:.2f}")
    print(f"  Accuracy: {best['accuracy']:.1f}%  |  Spec overrides: {best['spec_wins']}  |  Spec FP: {best['spec_fp']}")
    print()

    delta = best['accuracy'] - current['accuracy']
    if delta > 0:
        print(f"  ▲ Improvement: +{delta:.1f}% accuracy by changing weight to {best['scale']:.2f}")
    elif delta < 0:
        print(f"  Current weight is already optimal (or near-optimal)")
    else:
        print(f"  No change in accuracy — multiple weights tie at {best['accuracy']:.1f}%")
    print()

    # ── Confusion matrix at current weight ──
    print("═" * 45)
    print("  CONFUSION MATRIX (current weight = 0.50)")
    print("═" * 45)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for _, row in df.iterrows():
        fused = simulate_fusion(row['Cam_Class'], row['Cam_Conf'],
                                row['Spec_Class'], row['Spec_Conf'], 0.5)
        true = row['True_Class']
        if true in FUSION_CLASSES and fused in FUSION_CLASSES:
            ti = FUSION_CLASSES.index(true)
            fi = FUSION_CLASSES.index(fused)
            cm[ti][fi] += 1

    header = "True \\ Pred │ " + " │ ".join(f"{c:>8s}" for c in FUSION_CLASSES) + " │ Acc"
    print(header)
    print("─" * len(header))
    for i, cls in enumerate(FUSION_CLASSES):
        row_total = cm[i].sum()
        row_acc = 100 * cm[i][i] / row_total if row_total > 0 else 0
        vals = " │ ".join(f"{cm[i][j]:>8d}" for j in range(NUM_CLASSES))
        print(f"{cls:>10s} │ {vals} │ {row_acc:5.1f}%")
    print()

    # ── Confusion matrix at best weight ──
    if best['scale'] != 0.5:
        print("═" * 45)
        print(f"  CONFUSION MATRIX (best weight = {best['scale']:.2f})")
        print("═" * 45)
        cm2 = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for _, row in df.iterrows():
            fused = simulate_fusion(row['Cam_Class'], row['Cam_Conf'],
                                    row['Spec_Class'], row['Spec_Conf'], best['scale'])
            true = row['True_Class']
            if true in FUSION_CLASSES and fused in FUSION_CLASSES:
                ti = FUSION_CLASSES.index(true)
                fi = FUSION_CLASSES.index(fused)
                cm2[ti][fi] += 1

        header = "True \\ Pred │ " + " │ ".join(f"{c:>8s}" for c in FUSION_CLASSES) + " │ Acc"
        print(header)
        print("─" * len(header))
        for i, cls in enumerate(FUSION_CLASSES):
            row_total = cm2[i].sum()
            row_acc = 100 * cm2[i][i] / row_total if row_total > 0 else 0
            vals = " │ ".join(f"{cm2[i][j]:>8d}" for j in range(NUM_CLASSES))
            print(f"{cls:>10s} │ {vals} │ {row_acc:5.1f}%")
        print()

    # ── SECONDARY ANALYSIS: Agreement-only ground truth ──
    # Use only rows where camera & spectrum agree as "confirmed" ground truth
    # This avoids the bias of using camera as truth and gives a fairer picture
    agree_df = df[agree_mask].copy()
    n_agree_total = len(agree_df)
    if n_agree_total > 10:
        print("═" * 75)
        print(f"  SECONDARY: Agreement-only analysis ({n_agree_total} confirmed rows)")
        print("═" * 75)
        print(f"  (Both sensors agree → high confidence these are correct labels)")
        print()

        # For agreement rows, both sensors say the same thing.
        # The question: does fusion still get it right, or does fusion math distort?
        agree_df['True_Class'] = agree_df['Cam_Class']  # both agree anyway

        print(f"{'Weight':>8} │ {'Accuracy':>8} │ {'Correct':>7}/{n_agree_total}")
        print("─" * 40)
        for scale in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            correct_a = 0
            for _, row in agree_df.iterrows():
                fused = simulate_fusion(
                    row['Cam_Class'], row['Cam_Conf'],
                    row['Spec_Class'], row['Spec_Conf'], scale)
                if fused == row['True_Class']:
                    correct_a += 1
            acc_a = 100.0 * correct_a / n_agree_total
            marker = " ◄── CURRENT" if scale == 0.5 else ""
            print(f"{scale:>8.2f} │ {acc_a:>7.1f}% │ {correct_a:>7d}/{n_agree_total}{marker}")
        print()

    # ── TERTIARY ANALYSIS: Labeled file ground truth ──
    # Files like glass_25_metal_25.csv tell us which classes were placed
    # We can check if fusion correctly identifies those classes
    labeled_files = {
        'glass_25_metal_25.csv':          {'Glass': 25, 'Metal': 25},
        'glass_metal_50.csv':             {'Glass': 25, 'Metal': 25},
        'metal_25_plastic_25.csv':        {'Metal': 25, 'Plastic': 25},
        'paper_19.csv':                   {'Paper': 19},
        'paper_metal_50.csv':             {'Paper': 25, 'Metal': 25},
        'paper_plastic_50.csv':           {'Paper': 25, 'Plastic': 25},
        'paper_plastic_glass_50.csv':     {'Paper': 17, 'Plastic': 17, 'Glass': 16},
        'paper_plastic_metal_31.csv':     {'Paper': 10, 'Plastic': 10, 'Metal': 11},
        'plastic_metal_33.csv':           {'Plastic': 17, 'Metal': 16},
        'plastic_metal_37.csv':           {'Plastic': 19, 'Metal': 18},
        'm_22_pl_3_pp_25.csv':            {'Metal': 22, 'Plastic': 3, 'Paper': 25},
    }

    labeled_rows = df[df['_source_file'].isin(labeled_files.keys())].copy()
    if len(labeled_rows) > 10:
        print("═" * 75)
        print(f"  TERTIARY: Labeled-file class distribution check ({len(labeled_rows)} rows)")
        print("═" * 75)
        print()
        for fname, expected in labeled_files.items():
            fdf = labeled_rows[labeled_rows['_source_file'] == fname]
            if len(fdf) == 0:
                continue
            exp_classes = set(expected.keys())
            # Class distributions at various weights
            print(f"  {fname} (expected: {expected})")
            for scale in [0.0, 0.5, 1.0]:
                fused_classes = []
                for _, row in fdf.iterrows():
                    fused_classes.append(simulate_fusion(
                        row['Cam_Class'], row['Cam_Conf'],
                        row['Spec_Class'], row['Spec_Conf'], scale))
                dist = pd.Series(fused_classes).value_counts().to_dict()
                # Count classes that shouldn't be there (false positives)
                fp_count = sum(v for k, v in dist.items() if k not in exp_classes)
                print(f"    w={scale:.1f}: {dist}  (FP={fp_count})")
            print()

    # ── Try to plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        xs = [r['scale'] for r in results]
        accs = [r['accuracy'] for r in results]
        fps = [r['spec_fp'] for r in results]

        ax1.plot(xs, accs, 'b-o', markersize=5, linewidth=2, label='Fusion Accuracy')
        ax1.axhline(y=cam_acc, color='g', linestyle='--', alpha=0.7, label=f'Camera-only ({cam_acc:.1f}%)')
        ax1.axvline(x=0.5, color='r', linestyle=':', alpha=0.7, label='Current (0.50)')
        ax1.axvline(x=best['scale'], color='orange', linestyle='--', alpha=0.7,
                     label=f'Best ({best["scale"]:.2f})')
        ax1.set_xlabel('SPECTRUM_WEIGHT_SCALE')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Fusion Accuracy vs Spectrum Weight')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.bar(xs, fps, color='salmon', width=0.03, edgecolor='darkred')
        ax2.axvline(x=0.5, color='r', linestyle=':', alpha=0.7, label='Current (0.50)')
        ax2.set_xlabel('SPECTRUM_WEIGHT_SCALE')
        ax2.set_ylabel('Spectrum False Positives')
        ax2.set_title('Spectrum FP Count vs Weight')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__),
                                "detection_logs", "audits", "fusion_weight_experiment.png")
        plt.savefig(out_path, dpi=150)
        print(f"Chart saved: {out_path}")
    except ImportError:
        print("(matplotlib not available — skipping chart)")
    except Exception as e:
        print(f"(Chart error: {e})")

    print()
    print("To apply the recommended weight, update modules/robot.py:")
    print(f'  SPECTRUM_WEIGHT_SCALE = {best["scale"]}')
    print()


if __name__ == "__main__":
    run_experiment()
