import numpy as np

# ----------------------------------
# 1) Define Classes
# ----------------------------------
classes = ["Plastic", "Metal", "Paper", "Glass"]
num_classes = len(classes)

# ----------------------------------
# 2) Confusion Matrices (True x Pred)
# ----------------------------------

cm_yolo = np.array([
    [223, 13, 19, 3],
    [26, 216, 8, 0],
    [14, 9, 228, 0],
    [96, 6, 8, 139]
], dtype=float)

cm_spec = np.array([
    [17, 10, 4, 13],
    [11, 18, 11, 2],
    [5, 9, 30, 0],
    [13, 0, 1, 34]
], dtype=float)

# ----------------------------------
# 3) Laplace Smoothing Parameter
# ----------------------------------
alpha = 1.0

# ----------------------------------
# 4) Compute Prior (with smoothing optional)
# ----------------------------------
true_counts = cm_yolo.sum(axis=1)
total_samples = true_counts.sum()

prior = (true_counts + alpha) / (total_samples + alpha * num_classes)

# ----------------------------------
# 5) Compute Likelihood Tables
#    P(Pred | True)
# ----------------------------------

likelihood_yolo = (cm_yolo + alpha) / (true_counts[:, None] + alpha * num_classes)
likelihood_spec = (cm_spec + alpha) / (cm_spec.sum(axis=1)[:, None] + alpha * num_classes)

# ----------------------------------
# 6) Bayesian Fusion (Log-space)
# ----------------------------------

def bayesian_fusion(pred_yolo, pred_spec, verbose=True):

    idx_yolo = classes.index(pred_yolo)
    idx_spec = classes.index(pred_spec)

    log_scores = []

    if verbose:
        print("\n==============================")
        print("YOLO prediction:", pred_yolo)
        print("Spectrum prediction:", pred_spec)
        print("==============================\n")

    for c in range(num_classes):

        class_name = classes[c]

        p_yolo = likelihood_yolo[c, idx_yolo]
        p_spec = likelihood_spec[c, idx_spec]
        p_prior = prior[c]

        # log-space computation (spectrum weight halved: camera leads fusion)
        log_score = np.log(p_yolo) + 0.5 * np.log(p_spec) + np.log(p_prior)
        log_scores.append(log_score)

        if verbose:
            print(f"Assume True = {class_name}")
            print(f"  P(YOLO={pred_yolo} | True={class_name}) = {p_yolo:.6f}")
            print(f"  P(Spec={pred_spec} | True={class_name}) = {p_spec:.6f}")
            print(f"  Prior P({class_name}) = {p_prior:.6f}")
            print(f"  log-score = log({p_yolo:.6f}) + log({p_spec:.6f}) + log({p_prior:.6f})")
            print(f"            = {log_score:.6f}")
            print("----------------------------------")

    log_scores = np.array(log_scores)
    best_idx = np.argmax(log_scores)

    if verbose:
        print("\nFinal Log Posterior Scores:")
        for i, cls in enumerate(classes):
            print(f"{cls}: {log_scores[i]:.6f}")

        print("\nFinal Decision:", classes[best_idx])
        print("==============================\n")

    return classes[best_idx], log_scores


# ----------------------------------
# 7) Example
# ----------------------------------
YOLO_Pred = "Glass"
Spec_Pred = "Glass"
final_class, scores = bayesian_fusion(YOLO_Pred,Spec_Pred, verbose=True)
print("Final fused class:", final_class)