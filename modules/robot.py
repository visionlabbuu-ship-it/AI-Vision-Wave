"""
=============================================================================
ROBOT MODULE
=============================================================================
Delta robot and slider control via serial communication.
"""

import serial
import threading
import queue
import time
import numpy as np
from datetime import datetime

from .config import (
    DEFAULT_DELTA_PORT, DEFAULT_SLIDER_PORT, CLASS_NAMES,
    THROW_TARGETS, X_DELTA_MM, Y_DELTA_MM, X_BELT_MM,
    BASE_Z, STANDBY_POSITION,
    BELT_X_GRID, BELT_Y_GRID, ROBOT_X_GRID, ROBOT_Y_GRID,
    CLASS_Z_OFFSET_MM,
    ROBOT_WORKSPACE_DEPTH_CM, ROI_HEIGHT_CM, ROBOT_WORKSPACE_OFFSET_CM,
    TRACK_APPROACH_HOVER_MM, TRACK_APPROACH_SPEED, TRACK_DESCEND_SPEED,
    SCAN_W_OFFSET_DEG, SCAN_W_MIN_DEG, SCAN_W_MAX_DEG, SCAN_W_HOME_CMD,
    PICK_SURFACE_PENETRATION_MM,
    STACK_BOTTOM_Y_ADVANCE_CM,
    PLACE_Z_HEIGHT, PLACE_Z_RELEASE,
    ROBOT_MOVE_TIME_S,
)


def bilinear_interpolate(belt_x_mm, belt_y_mm):
    """
    2D bilinear interpolation from belt coordinates to robot coordinates.
    
    IMPORTANT: This is a delta robot (3-arm parallel linkage). Robot X and Y
    are kinematically coupled — the same belt X requires different robot X
    values depending on belt Y, and vice versa. This is why we interpolate
    over a 2D grid (not two separate 1D lookups). Each grid point stores a
    paired (robot_x, robot_y) that was calibrated together.
    
    Args:
        belt_x_mm: Belt X position in mm (0-200, left to right)
        belt_y_mm: Belt Y position in mm (0-200, top to bottom of pick zone)
    
    Returns:
        (robot_x, robot_y) in mm — these are a coupled pair, not independent axes
    """
    # Clamp to grid bounds
    x = np.clip(belt_x_mm, BELT_X_GRID[0], BELT_X_GRID[-1])
    y = np.clip(belt_y_mm, BELT_Y_GRID[0], BELT_Y_GRID[-1])
    
    # Find grid cell indices
    # X direction (columns)
    if x <= BELT_X_GRID[0]:
        col = 0
        tx = 0.0
    elif x >= BELT_X_GRID[-1]:
        col = len(BELT_X_GRID) - 2
        tx = 1.0
    else:
        for i in range(len(BELT_X_GRID) - 1):
            if BELT_X_GRID[i] <= x <= BELT_X_GRID[i + 1]:
                col = i
                tx = (x - BELT_X_GRID[i]) / (BELT_X_GRID[i + 1] - BELT_X_GRID[i])
                break
    
    # Y direction (rows)
    if y <= BELT_Y_GRID[0]:
        row = 0
        ty = 0.0
    elif y >= BELT_Y_GRID[-1]:
        row = len(BELT_Y_GRID) - 2
        ty = 1.0
    else:
        for i in range(len(BELT_Y_GRID) - 1):
            if BELT_Y_GRID[i] <= y <= BELT_Y_GRID[i + 1]:
                row = i
                ty = (y - BELT_Y_GRID[i]) / (BELT_Y_GRID[i + 1] - BELT_Y_GRID[i])
                break
    
    # Get 4 corner values for bilinear interpolation
    # Robot X
    rx00 = ROBOT_X_GRID[row][col]
    rx10 = ROBOT_X_GRID[row][col + 1]
    rx01 = ROBOT_X_GRID[row + 1][col]
    rx11 = ROBOT_X_GRID[row + 1][col + 1]
    
    # Robot Y  
    ry00 = ROBOT_Y_GRID[row][col]
    ry10 = ROBOT_Y_GRID[row][col + 1]
    ry01 = ROBOT_Y_GRID[row + 1][col]
    ry11 = ROBOT_Y_GRID[row + 1][col + 1]
    
    # Bilinear interpolation
    robot_x = (1 - tx) * (1 - ty) * rx00 + tx * (1 - ty) * rx10 + (1 - tx) * ty * rx01 + tx * ty * rx11
    robot_y = (1 - tx) * (1 - ty) * ry00 + tx * (1 - ty) * ry10 + (1 - tx) * ty * ry01 + tx * ty * ry11
    
    return robot_x, robot_y


def inverse_bilinear_interpolate(robot_x, robot_y):
    """
    Approximate inverse mapping from robot coordinates to belt coordinates.
    Uses iterative search over the grid to find the belt position that maps
    closest to the given robot position.
    
    Args:
        robot_x: Robot X position in mm
        robot_y: Robot Y position in mm
    
    Returns:
        (belt_x_mm, belt_y_mm) approximate belt coordinates
    """
    best_dist = float('inf')
    best_bx, best_by = 100.0, 100.0  # Default to center
    
    # Search over belt grid with fine steps
    steps = 20
    for i in range(steps + 1):
        for j in range(steps + 1):
            bx = BELT_X_GRID[0] + (BELT_X_GRID[-1] - BELT_X_GRID[0]) * i / steps
            by = BELT_Y_GRID[0] + (BELT_Y_GRID[-1] - BELT_Y_GRID[0]) * j / steps
            rx, ry = bilinear_interpolate(bx, by)
            dist = (rx - robot_x) ** 2 + (ry - robot_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_bx, best_by = bx, by
    
    return best_bx, best_by


def map_angle_to_robot_r(angle_deg):
    """Map object angle to robot R (rotation) value for PICKING."""
    angle_deg = angle_deg % 180
    if angle_deg > 90:
        angle_deg -= 180
    angle_deg = abs(angle_deg)
    r_val = 130.0 - (1.222 * angle_deg)
    return np.clip(r_val, 20.0, 130.0)


def map_angle_to_scan_w(pca_angle_deg):
    """
    Map object PCA orientation angle to motor W value for SPECTRUM SCANNING.

    Motor coordinate system:
        W=0   -> aligned with belt X (left-right)  = HOME
        W=90  -> aligned with belt Y (belt travel direction)
        W=180 -> opposite of belt X
        Range: 0 to 180 degrees

    PCA angle 0 = object long axis along image X (= belt X).
    PCA angles range -180..+180 but orientation is 180-symmetric
    (pointing left vs right is the same axis), so we fold into 0..180
    which maps directly to the motor range.
    """
    # Fold PCA angle into 0..180 (orientation is 180-symmetric)
    a = pca_angle_deg % 180  # Result is 0..179.99
    w = a + SCAN_W_OFFSET_DEG
    return float(np.clip(w, SCAN_W_MIN_DEG, SCAN_W_MAX_DEG))


# =============================================================================
# ADAPTIVE BAYESIAN FUSION FOR YOLO + SPECTRUM
# =============================================================================
# Updated from new_purpose_combined.py — gamma-powered Bayesian method with
# confusion matrices from real Manual_Inspection data collection.
# Contact gate retained for production safety.

# Classes for fusion (must match model outputs)
FUSION_CLASSES = ["Glass", "Metal", "Paper", "Plastic"]
NUM_FUSION_CLASSES = len(FUSION_CLASSES)

# Confusion matrices (True x Pred) from real data collection
# Row = True class, Col = Predicted class [Glass, Metal, Paper, Plastic]
CM_YOLO = np.array([
    [67,  8,  0, 25],    # Glass
    [ 2, 66, 19, 13],    # Metal
    [ 0, 10, 87,  3],    # Paper
    [ 8, 23, 14, 55]     # Plastic
], dtype=float)

CM_SPEC = np.array([
    [56,  9,  4, 31],    # Glass
    [ 1, 78, 14,  7],    # Metal
    [ 0,  3, 93,  4],    # Paper
    [ 3, 13, 18, 66]     # Plastic
], dtype=float)

# Hyperparameters (optimised from offline grid search on real data)
FUSION_ALPHA = 0.001     # Near-zero Laplace smoothing — trust empirical data
FUSION_GAMMA = 3.5       # Confidence power — aggressively suppresses low-conf
SPECTRUM_WEIGHT_SCALE = 0.1  # Scale factor for spectrum weight (reduced: camera leads fusion)

# Compute prior (with smoothing)
_true_counts = CM_YOLO.sum(axis=1)
_total_samples = _true_counts.sum()
FUSION_PRIOR = (_true_counts + FUSION_ALPHA) / (_total_samples + FUSION_ALPHA * NUM_FUSION_CLASSES)

# Compute likelihood tables P(Pred | True)
LIKELIHOOD_YOLO = (CM_YOLO + FUSION_ALPHA) / (_true_counts[:, None] + FUSION_ALPHA * NUM_FUSION_CLASSES)
LIKELIHOOD_SPEC = (CM_SPEC + FUSION_ALPHA) / (CM_SPEC.sum(axis=1)[:, None] + FUSION_ALPHA * NUM_FUSION_CLASSES)

# Contact threshold for spectrum sensor validity
CONTACT_THRESHOLD = 220.0


def adaptive_fusion(yolo_class, yolo_conf, spec_class, spec_probs, spectral_raw, log_func=None):
    """
    Gamma-powered Bayesian fusion of YOLO and Spectrum predictions.
    
    New method (from new_purpose_combined.py):
    - Weights = confidence^gamma  (gamma=3.5 aggressively suppresses low-conf)
    - Confusion-matrix likelihoods with near-zero smoothing (alpha=0.001)
    - Updated CMs from real Manual_Inspection data collection
    
    Production safety retained:
    - Contact gate: spectrum weight=0 if sensor not touching object
    
    Args:
        yolo_class: YOLO model prediction (class name string)
        yolo_conf: YOLO confidence (0-1 or 0-100)
        spec_class: Spectrum model prediction (class name string)
        spec_probs: Spectrum probability distribution (dict or array) — used for spec_conf extraction
        spectral_raw: Raw 18-channel spectral data (for contact detection)
        log_func: Optional logging function
    
    Returns:
        (final_class, fusion_info): Best class and debug info dict
    """
    log = log_func or (lambda x: None)
    
    # Handle unknown YOLO class — return as-is
    if yolo_class not in FUSION_CLASSES:
        log(f"[FUSION] Unknown YOLO class '{yolo_class}', using as-is")
        return yolo_class, {'contact': 'N/A', 'w_yolo': 1.0, 'w_spec': 0.0, 'method': 'unknown_class'}
    
    # Normalise YOLO confidence to 0-1
    yolo_conf_norm = yolo_conf / 100.0 if yolo_conf > 1 else yolo_conf
    
    # Extract spectrum confidence from probs dict / array
    if isinstance(spec_probs, dict):
        spec_conf_norm = max(spec_probs.values()) / 100.0 if max(spec_probs.values()) > 1 else max(spec_probs.values())
    elif spec_probs is not None:
        spec_conf_norm = float(np.max(spec_probs))
        if spec_conf_norm > 1:
            spec_conf_norm /= 100.0
    else:
        log(f"[FUSION] No spectrum probs, using YOLO: {yolo_class}")
        return yolo_class, {'contact': 'NO_DATA', 'w_yolo': 1.0, 'w_spec': 0.0, 'method': 'yolo_only'}
    
    # =========================================
    # 1) CONTACT GATE (Physics-based)
    # =========================================
    contact_factor = 1.0
    contact_status = "CONTACT"
    total_intensity = 0
    if spectral_raw is not None and len(spectral_raw) >= 18:
        total_intensity = np.sum(spectral_raw)
        if total_intensity < CONTACT_THRESHOLD:
            contact_factor = 0.0
            contact_status = "NO_CONTACT"
    else:
        contact_factor = 0.0
        contact_status = "NO_DATA"
    
    # If no contact, use YOLO only
    if contact_factor == 0.0:
        log(f"[FUSION] {contact_status} (intensity={total_intensity:.0f}) - using YOLO: {yolo_class}")
        return yolo_class, {'contact': contact_status, 'intensity': round(total_intensity, 1),
                            'w_yolo': 1.0, 'w_spec': 0.0, 'method': 'yolo_only_no_contact'}
    
    # =========================================
    # 2) GAMMA-POWERED WEIGHTS
    # =========================================
    w_yolo = yolo_conf_norm ** FUSION_GAMMA
    w_spec = (spec_conf_norm ** FUSION_GAMMA) * SPECTRUM_WEIGHT_SCALE  # Halved: camera leads
    
    # =========================================
    # 3) BAYESIAN FUSION (LOG SPACE)
    # =========================================
    idx_yolo = FUSION_CLASSES.index(yolo_class)
    idx_spec = FUSION_CLASSES.index(spec_class) if spec_class in FUSION_CLASSES else -1
    
    log_scores = []
    for c in range(NUM_FUSION_CLASSES):
        p_yolo = LIKELIHOOD_YOLO[c, idx_yolo]
        p_spec = LIKELIHOOD_SPEC[c, idx_spec] if idx_spec != -1 else 1.0
        p_prior = FUSION_PRIOR[c]
        
        log_score = (
            w_yolo * np.log(p_yolo + 1e-12) +
            w_spec * np.log(p_spec + 1e-12) +
            np.log(p_prior + 1e-12)
        )
        log_scores.append(log_score)
    
    log_scores = np.array(log_scores)
    best_idx = np.argmax(log_scores)
    final_class = FUSION_CLASSES[best_idx]
    
    # Build debug info
    fusion_info = {
        'contact': contact_status,
        'intensity': round(total_intensity, 1),
        'w_yolo': round(w_yolo, 4),
        'w_spec': round(w_spec, 4),
        'gamma': FUSION_GAMMA,
        'alpha': FUSION_ALPHA,
        'method': 'gamma_bayesian',
        'scores': {FUSION_CLASSES[i]: round(log_scores[i], 4) for i in range(NUM_FUSION_CLASSES)},
    }
    
    log(f"[FUSION] {contact_status} | YOLO={yolo_class}({yolo_conf_norm:.2f}) Spec={spec_class}({spec_conf_norm:.2f}) | "
        f"w_yolo={w_yolo:.4f} w_spec={w_spec:.4f} (g={FUSION_GAMMA}) -> {final_class}")
    
    return final_class, fusion_info


# Keep old function for backward compatibility
def bayesian_fusion(pred_yolo, pred_spec, log_func=None):
    """Legacy simple Bayesian fusion (for backward compatibility)."""
    return adaptive_fusion(
        yolo_class=pred_yolo,
        yolo_conf=0.9,
        spec_class=pred_spec,
        spec_probs={pred_spec: 0.9} if pred_spec in FUSION_CLASSES else None,
        spectral_raw=[300] * 18,
        log_func=log_func
    )


# =============================================================================
# 10 FUSION METHODS — for pipeline audit / comparison
# =============================================================================
# Mirrors the methods from Manual_Inspection.py / fusion_methods_summary.md
# Each function: (cam_cls, cam_conf, spec_cls, spec_conf) → predicted class
# cam_conf / spec_conf may be 0-100 or 0-1 — normalised internally.

_FUSION_CLS_IDX = {c: i for i, c in enumerate(FUSION_CLASSES)}

def _norm(c):
    """Ensure confidence is 0-1."""
    return c / 100.0 if c > 1.0 else float(c)


def fusion_majority(cam_cls, cam_conf, spec_cls, spec_conf):
    """1. Majority Voting — agree → use it; disagree → higher confidence wins."""
    if cam_cls == spec_cls:
        return cam_cls
    return cam_cls if _norm(cam_conf) >= _norm(spec_conf) else spec_cls


def fusion_weighted(cam_cls, cam_conf, spec_cls, spec_conf,
                    w_cam=0.6, w_spec=0.3):
    """2. Weighted Voting — cam*0.6 vs spec*0.3 (camera leads, spectrum halved)."""
    if cam_cls == spec_cls:
        return cam_cls
    sc = _norm(cam_conf) * w_cam
    ss = _norm(spec_conf) * w_spec
    return cam_cls if sc >= ss else spec_cls


def fusion_max_confidence(cam_cls, cam_conf, spec_cls, spec_conf):
    """3. Max Confidence — sensor with higher raw confidence wins."""
    return cam_cls if _norm(cam_conf) >= _norm(spec_conf) else spec_cls


def fusion_min_error(cam_cls, cam_conf, spec_cls, spec_conf):
    """4. Min Error (CM-based) — pick sensor with fewer mis-classifications for its class."""
    ci = _FUSION_CLS_IDX.get(cam_cls)
    si = _FUSION_CLS_IDX.get(spec_cls)
    if ci is None:
        return spec_cls
    if si is None:
        return cam_cls
    cam_err = 100.0 - CM_YOLO[ci, ci] if ci < CM_YOLO.shape[0] else 100.0
    spe_err = 100.0 - CM_SPEC[si, si] if si < CM_SPEC.shape[0] else 100.0
    return cam_cls if cam_err <= spe_err else spec_cls


def fusion_stacking(cam_cls, cam_conf, spec_cls, spec_conf):
    """5. Stacking (CM-based) — sensor with higher TP rate for its predicted class."""
    ci = _FUSION_CLS_IDX.get(cam_cls)
    si = _FUSION_CLS_IDX.get(spec_cls)
    if ci is None:
        return spec_cls
    if si is None:
        return cam_cls
    cam_tp = CM_YOLO[ci, ci] / max(CM_YOLO[ci].sum(), 1)
    spe_tp = CM_SPEC[si, si] / max(CM_SPEC[si].sum(), 1)
    return cam_cls if cam_tp >= spe_tp else spec_cls


def fusion_dempster_shafer(cam_cls, cam_conf, spec_cls, spec_conf):
    """7. Dempster-Shafer — average CM-row belief vectors weighted by confidence."""
    ci = _FUSION_CLS_IDX.get(cam_cls)
    si = _FUSION_CLS_IDX.get(spec_cls)
    nc = _norm(cam_conf)
    ns = _norm(spec_conf)
    belief = np.zeros(NUM_FUSION_CLASSES)
    if ci is not None:
        row = CM_YOLO[ci]
        belief += (row / max(row.sum(), 1)) * nc
    if si is not None:
        row = CM_SPEC[si]
        belief += (row / max(row.sum(), 1)) * ns
    return FUSION_CLASSES[int(np.argmax(belief))]


def fusion_rule_based(cam_cls, cam_conf, spec_cls, spec_conf):
    """8. Rule-Based — Camera for Glass/Metal, Spectrum for Paper/Plastic."""
    if cam_cls == spec_cls:
        return cam_cls
    if cam_cls in ('Glass', 'Metal'):
        return cam_cls
    if spec_cls in ('Paper', 'Plastic'):
        return spec_cls
    return cam_cls if _norm(cam_conf) >= _norm(spec_conf) else spec_cls


def fusion_neural_net(cam_cls, cam_conf, spec_cls, spec_conf):
    """9. Neural Net (simulated) — soft probability vectors averaged."""
    nc = _norm(cam_conf)
    ns = _norm(spec_conf)
    vec = np.zeros(NUM_FUSION_CLASSES)
    ci = _FUSION_CLS_IDX.get(cam_cls)
    si = _FUSION_CLS_IDX.get(spec_cls)
    if ci is not None:
        vec[ci] += nc
        remaining = (1 - nc) / max(NUM_FUSION_CLASSES - 1, 1)
        for j in range(NUM_FUSION_CLASSES):
            if j != ci:
                vec[j] += remaining
    if si is not None:
        vec[si] += ns
        remaining = (1 - ns) / max(NUM_FUSION_CLASSES - 1, 1)
        for j in range(NUM_FUSION_CLASSES):
            if j != si:
                vec[j] += remaining
    return FUSION_CLASSES[int(np.argmax(vec))]


def fusion_ensemble(cam_cls, cam_conf, spec_cls, spec_conf):
    """10. Ensemble — majority vote across methods 1-5, 7-9."""
    from collections import Counter
    votes = [
        fusion_majority(cam_cls, cam_conf, spec_cls, spec_conf),
        fusion_weighted(cam_cls, cam_conf, spec_cls, spec_conf),
        fusion_max_confidence(cam_cls, cam_conf, spec_cls, spec_conf),
        fusion_min_error(cam_cls, cam_conf, spec_cls, spec_conf),
        fusion_stacking(cam_cls, cam_conf, spec_cls, spec_conf),
        fusion_dempster_shafer(cam_cls, cam_conf, spec_cls, spec_conf),
        fusion_rule_based(cam_cls, cam_conf, spec_cls, spec_conf),
        fusion_neural_net(cam_cls, cam_conf, spec_cls, spec_conf),
    ]
    winner, _ = Counter(votes).most_common(1)[0]
    return winner


# Registry: (column_name, function)
# Method 6 (Bayesian / adaptive_fusion) is handled separately — it's already
# the primary fusion in _execute_pick.
AUDIT_FUSION_METHODS = [
    ("Majority",        fusion_majority),
    ("Weighted",        fusion_weighted),
    ("MaxConfidence",   fusion_max_confidence),
    ("MinError",        fusion_min_error),
    ("Stacking",        fusion_stacking),
    # 6. Bayesian = adaptive_fusion (already computed as Final_Class)
    ("DempsterShafer",  fusion_dempster_shafer),
    ("RuleBased",       fusion_rule_based),
    ("NeuralNet",       fusion_neural_net),
    ("Ensemble",        fusion_ensemble),
]


def run_all_fusions(cam_cls, cam_conf, spec_cls, spec_conf):
    """
    Run all 9 non-Bayesian fusion methods and return an OrderedDict.
    Bayesian is expected to be passed separately as the primary 'Final_Class'.
    
    Returns: dict  {method_name: predicted_class, ...}
    """
    results = {}
    for name, func in AUDIT_FUSION_METHODS:
        try:
            results[name] = func(cam_cls, cam_conf, spec_cls, spec_conf)
        except Exception:
            results[name] = "Error"
    return results


class SerialController:
    """Base class for serial-controlled devices."""
    
    def __init__(self, log_func=None):
        self.ser = None
        self.connected = False
        self.log = log_func or print
    
    def connect(self, port, baudrate=115200, timeout=0.1):
        """Connect to serial device."""
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            self.connected = True
            self.log(f"[SERIAL] Connected to {port}")
            return True
        except Exception as e:
            self.log(f"[SERIAL] Failed to connect to {port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial device."""
        if self.ser:
            self.ser.close()
        self.connected = False
    
    def send(self, cmd):
        """Send command to device."""
        if self.connected and self.ser:
            try:
                self.ser.write((cmd + '\n').encode())
                return True
            except Exception as e:
                self.log(f"[SERIAL] Send error: {e}")
                return False
        return False
    
    def read_response(self, timeout=0.5):
        """Read response from device."""
        if self.connected and self.ser:
            try:
                self.ser.timeout = timeout
                return self.ser.readline().decode().strip()
            except:
                return None
        return None


class DeltaController(SerialController):
    """Controller for Delta robot arm."""
    
    def __init__(self, log_func=None):
        super().__init__(log_func)
        # Track current robot position (mm)
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = -250.0
    
    def move_to(self, x, y, z, w=0, f=15000):
        """Move to position (x, y, z) with rotation w and speed f."""
        self.last_x = float(x)
        self.last_y = float(y)
        self.last_z = float(z)
        cmd = f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} W{w:.2f} F{f}"
        return self.send(cmd)
    
    def set_vacuum(self, on):
        """Turn vacuum on/off."""
        cmd = "M3" if on else "M5"
        return self.send(cmd)
    
    def home(self):
        """Home the robot."""
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = -250.0
        return self.send("G28")
    
    def home_scan_motor(self):
        """
        Home the scan motor (W axis) by driving to hard stop.
        
        Sends W=-180 which forces the motor to its physical lower limit
        (always reaches 0 regardless of current position), then sends
        W=0 to reset the reference.
        """
        self.log("[MOTOR] Homing scan motor (W axis)...")
        # Drive to hard stop — motor physically cannot go below 0
        self.move_to(self.last_x, self.last_y, self.last_z, w=SCAN_W_HOME_CMD)
        time.sleep(0.5)  # Wait for motor to reach hard stop
        # Reset to 0 (home position)
        self.move_to(self.last_x, self.last_y, self.last_z, w=0)
        time.sleep(0.1)
        self.log("[MOTOR] Scan motor homed (W=0)")
        return True
    
    def go_standby(self):
        """Move to standby position."""
        x, y, z = STANDBY_POSITION
        return self.move_to(x, y, z)


class SliderController(SerialController):
    """Controller for linear slider."""
    
    def move_to(self, pos):
        """Move slider to position."""
        self.send("M321 3600")
        time.sleep(0.01)
        return self.send(f"M322 {pos:.1f}")
    
    def home(self):
        """Home the slider."""
        return self.send("M320")


class RobotManager:
    """
    Robot task manager - handles pick and place operations.
    Runs in separate thread to not block main loop.
    
    Includes duplicate pick prevention — skips picks that happen too quickly
    after a previous pick (likely the same object dispatched twice).
    """
    
    # Minimum time between consecutive picks (seconds).
    # If a new pick is dispatched within this window of the last pick, skip it.
    DUPLICATE_TIME_WINDOW_S = 0.0
    
    # Sensor offset from suction center (cm) - sensor is 3cm away from pick head
    SENSOR_OFFSET_CM = 3.0
    
    def __init__(self, delta, slider, spectrum_manager=None, log_func=None, 
                 ui_queue=None, db_manager=None, session_id=None, detection_logger=None,
                 tracker=None):
        self.delta = delta
        self.slider = slider
        self.spectrum = spectrum_manager
        self.log = log_func or print
        self.ui_queue = ui_queue
        self.db_mgr = db_manager
        self.session_id = session_id or "default"
        self.detection_logger = detection_logger  # CSV logger for detection results
        self.tracker = tracker  # Reference to SimpleTracker for real-time position queries
        
        self.task_queue = queue.Queue()
        self.running = False
        self.is_busy = False  # True while executing a pick cycle
        self._thread = None
        
        # Position calibration
        self.offsets = {'x': 0, 'y': 0, 'z': 0, 'latency': 0}
        
        # Belt speed for movement prediction (cm/s)
        self.belt_speed_cm_s = 6.0  # Updated from UI via set_offsets
        
        # Robot approach time (seconds) - time from pick command to vacuum on
        # This is used to predict where object will be when robot arrives
        self.robot_approach_time_s = 0.5  # Tune based on actual robot speed
        
        # Track recent pick timestamps to prevent duplicates
        self.last_pick_time = 0.0
        
        # Callback to update simulation robot position during tracking approach
        self.on_robot_pos_update = None  # Called with (x, y, z) during approach
        
        # Dynamic place targets — editable from UI, defaults from config
        self.place_targets = dict(THROW_TARGETS)  # Copy so UI edits don't mutate config
        self.place_z = PLACE_Z_HEIGHT
        self.place_z_release = PLACE_Z_RELEASE
        
        # Per-class Z offsets (mm) — editable from UI, defaults from config
        self.class_z_offsets = dict(CLASS_Z_OFFSET_MM)
        
        # Stack-bottom offsets — editable from UI
        self.stack_bottom_y_advance_cm = STACK_BOTTOM_Y_ADVANCE_CM  # Extra Y (cm) for belt drift
        self.stack_bottom_z_extra_mm = -10.0                        # Extra Z (mm), negative = deeper
        
        # === Main-loop-driven tracking (shared state) ===
        # The main loop writes real-time position here every frame,
        # the robot thread reads it — eliminates thread desync.
        import threading as _th
        self._pick_lock = _th.Lock()
        self._pick_phase = 'IDLE'        # IDLE | APPROACH | DESCEND | MECHANICAL
        self._pick_obj_id = None          # Object ID being picked
        self._live_pos = {                # Written by main loop every frame
            'belt_x': 10.0,
            'ws_y': 0.0,
            'height_cm': 0.0,
            'timestamp': 0.0,
            'valid': False,
        }
        self._descend_approved = False    # Main loop sets True when position is good
    
    def set_offsets(self, x=0, y=0, z=0, latency=0, belt_speed=None):
        """Set position offsets and belt speed."""
        self.offsets = {'x': x, 'y': y, 'z': z, 'latency': latency}
        if belt_speed is not None:
            self.belt_speed_cm_s = belt_speed
    
    def get_z_height_by_y(self, robot_y, top_z=-380, bottom_z=-370):
        """
        Interpolate Z height based on robot Y position.
        top_z: Z at top workspace (robot_y = max)
        bottom_z: Z at bottom workspace (robot_y = min)
        """
        min_y = min([min(row) for row in ROBOT_Y_GRID])
        max_y = max([max(row) for row in ROBOT_Y_GRID])
        y_clamped = np.clip(robot_y, min_y, max_y)
        ratio = (y_clamped - min_y) / (max_y - min_y) if max_y > min_y else 0
        return bottom_z + (top_z - bottom_z) * (1 - ratio)

    def test_grid_positions(self, z_height=-380, dwell_time=1.0):
        """
        Test all 9 calibration grid positions.
        Moves robot to each position in order: TL, TC, TR, ML, MC, MR, BL, BC, BR.
        
        Args:
            z_height: Z height to move to at each position (default -380mm, pick height)
            dwell_time: Time to pause at each position (seconds)
        
        Returns:
            True if all positions visited successfully
        """
        if not self.delta or not self.delta.connected:
            self.log("[ROBOT] Delta not connected - cannot test grid")
            return False
        
        # Grid labels for logging
        labels = [
            ["TL (Top-Left)", "TC (Top-Center)", "TR (Top-Right)"],
            ["ML (Mid-Left)", "MC (Mid-Center)", "MR (Mid-Right)"],
            ["BL (Bot-Left)", "BC (Bot-Center)", "BR (Bot-Right)"]
        ]
        
        # Belt coordinates for each position (in mm)
        belt_coords = [
            [(0, 0), (100, 0), (200, 0)],       # Top row (belt Y = 0)
            [(0, 100), (100, 100), (200, 100)], # Middle row (belt Y = 100)
            [(0, 200), (100, 200), (200, 200)]  # Bottom row (belt Y = 200)
        ]
        
        self.log("[ROBOT] === Starting Grid Calibration Test ===")
        self.log(f"[ROBOT] Z height: {z_height}mm, Dwell time: {dwell_time}s")
        
        # Move to standby first
        self.delta.move_to(0, 0, -250)
        time.sleep(0.5)
        
        for row in range(3):
            for col in range(3):
                label = labels[row][col]
                belt_x, belt_y = belt_coords[row][col]
                
                # Use bilinear interpolation to get robot coordinates
                robot_x, robot_y = bilinear_interpolate(belt_x, belt_y)
                
                # Interpolate Z height based on Y
                z_adj = self.get_z_height_by_y(robot_y, top_z=z_height, bottom_z=z_height+10)
                
                self.log(f"[ROBOT] Moving to {label}: Belt({belt_x}, {belt_y})mm -> Robot({robot_x:.1f}, {robot_y:.1f})mm, Z={z_adj:.1f}")
                
                # Move to position
                self.delta.move_to(robot_x, robot_y, z_adj)
                time.sleep(dwell_time)
        
        # Return to standby
        self.log("[ROBOT] === Grid Test Complete - Returning to Standby ===")
        self.delta.move_to(0, 0, -250)
        
        return True

    def _is_duplicate_pick(self):
        """
        Check if this pick is too soon after the last one (likely a duplicate dispatch).
        
        Returns True if it's a duplicate (should skip).
        """
        current_time = time.time()
        time_gap = current_time - self.last_pick_time
        
        if time_gap < self.DUPLICATE_TIME_WINDOW_S:
            self.log(f"[ROBOT] Duplicate detected: only {time_gap:.2f}s since last pick")
            return True
        
        return False
    
    def _record_pick(self):
        """Record the timestamp of this pick for duplicate detection."""
        self.last_pick_time = time.time()

    def is_idle(self):
        """Check if robot is idle (not executing a pick and no queued tasks)."""
        return not self.is_busy and self.task_queue.empty()
    
    def add_task(self, task):
        """Add a pick task to queue."""
        self.task_queue.put(task)
    
    def start_manager(self):
        """Start the robot manager thread."""
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def _run(self):
        """Main task processing loop."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.1)
                self.is_busy = True
                try:
                    self.execute_pick(task)
                finally:
                    self.is_busy = False
                    self._set_phase('IDLE')
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.is_busy = False
                self._set_phase('IDLE')
                self.log(f"[ROBOT] Task error: {e}")
    
    def smooth_move_to_pick(self, start_x, start_y, target_x, target_y, z, steps=15):
        """Move smoothly from current position to target position before picking."""
        for i in range(steps + 1):
            t = i / steps
            x = start_x * (1 - t) + target_x * t
            y = start_y * (1 - t) + target_y * t
            self.delta.move_to(x, y, z)
            time.sleep(0.05)

    def _compute_pick_z(self, height, class_name, is_stack_bottom=False):
        """
        Compute the pick Z height from object height and class.
        
        Formula:
          pz = BASE_Z + height_mm + global_z_offset + class_z_offset
               + (STACK_BOTTOM_EXTRA_MM if bottom of a stack)
        
        The safety clamp prevents the gripper from going more than
        PICK_SURFACE_PENETRATION_MM below the offset-adjusted target.
        User offsets are always respected — the clamp only catches
        extreme values that would slam into the belt surface.
        
        Returns:
            (pz, effective_height): pick Z in mm and the height used
        """
        MAX_VALID_HEIGHT_CM = 30.0
        if height < 0 or height > MAX_VALID_HEIGHT_CM:
            height = 0
        
        height_mm = height * 10  # cm -> mm
        class_z_off = self.class_z_offsets.get(class_name, 0.0)
        global_z_off = self.offsets['z']
        
        # Bottom object in a stack sits lower after the top is removed.
        # Use UI-adjustable offset (negative = deeper).
        bottom_off = self.stack_bottom_z_extra_mm if is_stack_bottom else 0
        
        # Target Z: base + object height + user offsets + stack bottom offset
        pz = BASE_Z + height_mm + global_z_off + class_z_off + bottom_off
        
        # Safety clamp: never go below the BELT SURFACE (BASE_Z) minus
        # a small penetration margin.  This prevents crashing into the belt
        # if height is wrong, but does NOT override user Z offsets for
        # reaching the object surface.
        belt_floor_z = BASE_Z - PICK_SURFACE_PENETRATION_MM
        if pz < belt_floor_z:
            pz = belt_floor_z
        
        pz = float(np.clip(pz, -450, -150))
        
        bot_str = ' [STACK-BOT -10mm]' if is_stack_bottom else ''
        self.log(f"[Z-CALC] h={height:.1f}cm class={class_name} "
                 f"base={BASE_Z:.0f} + h_mm={height_mm:.0f} + z_off={global_z_off:.1f} "
                 f"+ cls_off={class_z_off:.1f}{bot_str} = {pz:.0f}mm")
        
        return pz, height
    
    def _belt_to_robot(self, belt_x_cm, ws_y_cm):
        """
        Convert belt coordinates (cm) to clamped robot coordinates (mm).
        
        Args:
            belt_x_cm: X position on belt in cm (0-20)
            ws_y_cm: Workspace-relative Y in cm (0-20)
            
        Returns:
            (px, py): Robot position in mm, with offsets applied and clamped
        """
        x_mm = belt_x_cm * 10.0   # 0-200mm
        y_mm = ws_y_cm * 10.0     # 0-200mm
        robot_x, robot_y = bilinear_interpolate(x_mm, y_mm)
        px = float(np.clip(robot_x + self.offsets['x'], -110, 110))
        py = float(np.clip(robot_y + self.offsets['y'], -110, 110))
        return px, py

    def get_pick_phase(self):
        """Thread-safe read of current pick phase."""
        with self._pick_lock:
            return self._pick_phase
    
    def get_pick_obj_id(self):
        """Thread-safe read of the object ID currently being picked."""
        with self._pick_lock:
            return self._pick_obj_id
    
    def feed_live_position(self, belt_x, ws_y, height_cm):
        """
        Called by main loop every frame to feed real-time object position.
        Thread-safe write — robot thread reads this during approach.
        """
        with self._pick_lock:
            self._live_pos['belt_x'] = belt_x
            self._live_pos['ws_y'] = ws_y
            self._live_pos['height_cm'] = height_cm
            self._live_pos['timestamp'] = time.time()
            self._live_pos['valid'] = True
    
    def approve_descend(self):
        """Main loop signals that the object position is good for descent."""
        with self._pick_lock:
            self._descend_approved = True
    
    def _read_live_pos(self):
        """Thread-safe read of the latest position fed by main loop."""
        with self._pick_lock:
            return dict(self._live_pos)
    
    def _set_phase(self, phase):
        """Thread-safe phase transition."""
        with self._pick_lock:
            self._pick_phase = phase

    def execute_pick(self, task):
        """
        Execute a pick-and-place with PREDICTIVE direct pick.
        
        FLOW (single rotation, no hover):
        1. PREDICT: Use latest camera position + belt speed to predict
           where the object WILL be when the robot physically arrives.
        2. PICK: Vacuum ON, move directly to predicted (X, Y, pick_Z)
           with W aligned to object orientation.  One G-code command.
        3. LIFT + SCAN: Lift straight up (same W), spectrum reads while
           sensor is already aligned with object's long axis.
        4. THROW: Move to place bin (same W), release vacuum.
        """
        obj_id = task.get('id', 0)
        class_name = task.get('class_name', 'Unknown')
        camera_conf = task.get('confidence', 0.9)
        belt_x = task.get('belt_x', 10.0)
        height = task.get('height', 0)
        angle = task.get('angle', 0)
        belt_y_dispatched = task.get('belt_y', 10.0)  # workspace-relative Y at dispatch
        is_stack_bottom = task.get('is_stack_bottom', False)
        
        # Check for duplicate pick
        if self._is_duplicate_pick():
            self.log(f"[ROBOT] Skipping ID:{obj_id} - duplicate (too soon after last pick)")
            return False
        
        self.log(f"[ROBOT] Processing ID:{obj_id} - {class_name} at X={belt_x:.1f}cm, Y_ws={belt_y_dispatched:.1f}cm"
                 f"{' [STACK-BOTTOM]' if is_stack_bottom else ''}")
        self._record_pick()
        
        # Calculate rotation - single rotation for entire pick cycle.
        # Aligns the spectrum sensor with the object's long axis (PCA angle).
        # Motor W=0 = belt X, same reference as PCA angle 0.
        obj_rotation = map_angle_to_scan_w(angle)
        
        # Compute pick Z from object height
        pz, effective_height = self._compute_pick_z(height, class_name, is_stack_bottom)
        
        if not self.delta.connected:
            self.log("[ROBOT] Delta not connected!")
            self._set_phase('IDLE')
            return False
        
        ws_depth = ROBOT_WORKSPACE_DEPTH_CM  # 20cm
        
        # === PHASE 1: PREDICT - where will the object be when robot arrives? ===
        t_dispatch = time.time()
        
        with self._pick_lock:
            self._pick_obj_id = obj_id
            self._pick_phase = 'APPROACH'
            self._descend_approved = False
            self._live_pos['valid'] = False
        
        # Wait briefly for main loop to feed fresh position (up to 0.15s)
        wait_start = time.time()
        final_belt_x = belt_x
        final_ws_y = belt_y_dispatched
        final_height = height
        
        while (time.time() - wait_start) < 0.15:
            live = self._read_live_pos()
            if live['valid']:
                final_belt_x = live['belt_x']
                final_ws_y = live['ws_y']
                if live['height_cm'] > 0:
                    final_height = live['height_cm']
                break
            time.sleep(0.01)
        
        # Recompute pick Z with latest height
        pz, effective_height = self._compute_pick_z(final_height, class_name, is_stack_bottom)
        
        # Predict where object will be when robot physically arrives.
        # Single travel time - no hover step, robot goes directly to pick Z.
        PICK_TRAVEL_TIME_S = 0.35   # Empirical: time for robot to reach pick position
        predict_y = final_ws_y + (self.belt_speed_cm_s * PICK_TRAVEL_TIME_S)
        
        # Stack-bottom compensation: after picking the top object, the robot
        # went through lift+scan+throw+return — the bottom object drifted
        # further along the belt during that cycle.
        if is_stack_bottom:
            predict_y += self.stack_bottom_y_advance_cm
            self.log(f"[ROBOT] Stack-bottom Y advance: +{self.stack_bottom_y_advance_cm:.1f}cm -> pred_y={predict_y:.1f}cm")
        
        predict_y = max(0, min(ws_depth, predict_y))
        
        # Bail if object would be past workspace exit
        if predict_y >= ws_depth - 0.5:
            self.log(f"[ROBOT] ID:{obj_id} predicted ws_y={predict_y:.1f}cm - too close to exit, skip")
            self._set_phase('IDLE')
            return False
        
        # Convert to robot coordinates
        px, py = self._belt_to_robot(final_belt_x, predict_y)
        
        self.log(f"[ROBOT] >>> DIRECT PICK <<< ID:{obj_id}, "
                 f"ws_y={final_ws_y:.1f} -> pred={predict_y:.1f}cm "
                 f"({PICK_TRAVEL_TIME_S}s*{self.belt_speed_cm_s:.1f}), "
                 f"robot({px:.1f}, {py:.1f}, pz={pz:.0f})mm, W={obj_rotation:.1f}")
        
        # === PHASE 2: PICK - vacuum on, move directly to pick position ===
        self._set_phase('DESCEND')
        
        # Turn on vacuum before moving - suction builds during travel
        self.delta.set_vacuum(True)
        
        t_before_move = time.time()
        
        # Single move: go directly to pick position + pick Z.
        # No hover step - robot aligns rotation and descends in one command.
        self.delta.move_to(px, py, pz, w=obj_rotation, f=TRACK_DESCEND_SPEED)
        
        if self.on_robot_pos_update:
            self.on_robot_pos_update(px, py, pz)
        
        # Wait for robot to physically arrive at pick position
        time.sleep(max(0.25, PICK_TRAVEL_TIME_S))
        
        t_after_pick = time.time()
        
        # ── TIMING REPORT ──
        total_s = t_after_pick - t_dispatch
        belt_drift_cm = self.belt_speed_cm_s * total_s
        self.log(f"[TIMING] ID:{obj_id}  total={total_s:.3f}s  "
                 f"belt_drift={belt_drift_cm:.1f}cm @ {self.belt_speed_cm_s:.1f}cm/s  "
                 f"predicted={self.belt_speed_cm_s * PICK_TRAVEL_TIME_S:.1f}cm")
        
        # Wait for latency compensation
        if self.offsets['latency'] > 0:
            time.sleep(self.offsets['latency'])
        
        # === PHASE 3: MECHANICAL - lift, scan, throw ===
        self._set_phase('MECHANICAL')
        
        # Short settle wait for suction grip
        time.sleep(0.1)
        
        # Lift with object - same rotation, sensor already aligned
        self.delta.move_to(0, 0, -250, w=obj_rotation)
        
        # === SPECTRUM SCAN (live sensor OR replay payload) ===
        # Sensor aligned with object orientation via single obj_rotation
        final_class = class_name  # Default to YOLO prediction
        spectrum_pred = None
        spectrum_probs = None
        raw_data = None  # Initialize raw_data for logging
        conf = 0.0  # Initialize confidence

        # Replay payload from MP4 sidecar (18 channels). If present, keep this as
        # the decision/logging spectrum source so replay stays deterministic.
        replay_spec = task.get('replay_spectrum_raw')
        using_replay_spec = False
        if isinstance(replay_spec, (list, tuple)) and len(replay_spec) == 18:
            try:
                raw_data = [float(v) for v in replay_spec]
            except Exception:
                raw_data = None
            using_replay_spec = raw_data is not None

            if raw_data is not None:
                if self.spectrum and self.spectrum.is_ready:
                    spectrum_pred, conf, _, spectrum_probs = self.spectrum.predict(raw_data)
                else:
                    # Fallback when ML model is unavailable in replay:
                    # trust recorded class/conf from sidecar if provided.
                    spectrum_pred = task.get('replay_spectrum_class')
                    conf = float(task.get('replay_spectrum_conf', 0.0) or 0.0)
                    if spectrum_pred in FUSION_CLASSES:
                        probs = {c: 0.0 for c in FUSION_CLASSES}
                        probs[spectrum_pred] = max(0.0, min(1.0, conf / 100.0 if conf > 1 else conf))
                        spectrum_probs = probs

                if spectrum_pred:
                    self.log(f"[SPECTRUM][REPLAY] Prediction: {spectrum_pred} ({conf:.1f}%)")

        # Live hardware scan path (existing behavior when not replay)
        if raw_data is None and self.spectrum and self.spectrum.is_ready:
            time.sleep(0.1)  # Wait for stable position
            raw_data = self.spectrum.read_sensor()
            if raw_data:
                spectrum_pred, conf, _, spectrum_probs = self.spectrum.predict(raw_data)
                if spectrum_pred:
                    self.log(f"[SPECTRUM] Raw prediction: {spectrum_pred} ({conf:.1f}%)")

        # Replay realism mode: still trigger REAL hardware scan for IO/power realism,
        # but do NOT use this probe to overwrite replay-based decision values.
        if using_replay_spec and self.spectrum and self.spectrum.is_ready:
            try:
                time.sleep(0.1)  # Wait for stable position before real probe
                probe_raw = self.spectrum.read_sensor()
                if probe_raw:
                    probe_pred, probe_conf, _, _ = self.spectrum.predict(probe_raw)
                    if probe_pred:
                        self.log(f"[SPECTRUM][LIVE-PROBE] {probe_pred} ({probe_conf:.1f}%) [ignored]")
            except Exception as e:
                self.log(f"[SPECTRUM][LIVE-PROBE] error: {e}")

        # Common UI + fusion path for both live and replay spectrum
        if spectrum_pred:
            if self.ui_queue:
                self.ui_queue.put(('spectrum', str(raw_data)))
                self.ui_queue.put(('spectrum_result', {
                    'yolo_class': class_name,
                    'yolo_conf': camera_conf,
                    'spectrum_class': spectrum_pred,
                    'spectrum_conf': conf,
                }))

            # === ADAPTIVE BAYESIAN FUSION ===
            # Combine YOLO (vision) and Spectrum predictions with quality-based weighting
            final_class, fusion_info = adaptive_fusion(
                yolo_class=class_name,
                yolo_conf=camera_conf,
                spec_class=spectrum_pred,
                spec_probs=spectrum_probs,
                spectral_raw=raw_data,
                log_func=self.log
            )

            if final_class != class_name and final_class != spectrum_pred:
                self.log(f"[FUSION] Override: YOLO={class_name}, Spec={spectrum_pred} -> Final={final_class}")

        # === LOG DETECTION RESULTS ===
        # Log to CSV (30 columns: 12 classification + 18 spectral raw values)
        if self.detection_logger:
            # Get spectrum confidence if available
            spec_conf = conf if spectrum_pred else 0.0
            
            self.detection_logger.log_detection(
                obj_id=obj_id,
                camera_class=class_name,
                camera_conf=camera_conf,
                spectrum_class=spectrum_pred if spectrum_pred else "N/A",
                spectrum_conf=spec_conf,
                final_class=final_class,
                spectrum_raw=raw_data  # Pass raw 18-channel spectral data
            )
        
        # === AUDIT LOG (all 10 fusion methods) ===
        # Send full fusion comparison row to UI for audit log collection
        if self.ui_queue and spectrum_pred:
            audit_methods = run_all_fusions(class_name, camera_conf,
                                           spectrum_pred, conf)
            self.ui_queue.put(('audit_row', {
                'obj_id': obj_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cam_class': class_name,
                'cam_conf': round(camera_conf, 2),
                'spec_class': spectrum_pred,
                'spec_conf': round(conf, 2),
                'final_class': final_class,   # Bayesian (method 6)
                **audit_methods,               # Methods 1-5, 7-10
            }))

        # Detailed replay-sync row for MP4 experiment metadata export.
        if self.ui_queue:
            self.ui_queue.put(('pick_replay', {
                'obj_id': obj_id,
                'event_epoch_s': time.time(),
                'camera_class': class_name,
                'camera_conf': round(float(camera_conf), 4),
                'spectrum_class': spectrum_pred if spectrum_pred else "N/A",
                'spectrum_conf': round(float(conf), 4) if spectrum_pred else 0.0,
                'final_class': final_class,
                'belt_x_cm': round(float(final_belt_x), 4),
                'ws_y_cm': round(float(predict_y), 4),
                'height_cm': round(float(effective_height), 4),
                'robot_x_mm': round(float(px), 4),
                'robot_y_mm': round(float(py), 4),
                'pick_z_mm': round(float(pz), 4),
                'angle_deg': round(float(angle), 4),
                'scan_w_deg': round(float(obj_rotation), 4),
                'source': 'replay' if replay_spec else 'live',
                'spectrum_raw': raw_data if raw_data is not None else [],
            }))
        
        # === PLACE SEQUENCE ===
        # Use dynamic place targets (editable from UI), fall back to config defaults
        targets = getattr(self, 'place_targets', None) or THROW_TARGETS
        place_z = getattr(self, 'place_z', -300.0)
        release_z = getattr(self, 'place_z_release', -320.0)
        tx, ty = targets.get(final_class, (0, -120))
        self.delta.move_to(tx, ty, place_z, w=obj_rotation)
        self.delta.set_vacuum(False)
        self.delta.move_to(tx, ty, release_z)
        self.delta.move_to(tx, ty, place_z)
        time.sleep(max(0.0, ROBOT_MOVE_TIME_S))

        # Placement is complete after release, clearance, and the configured
        # movement-completion wait (the controller does not expose motion ACKs).
        if self.ui_queue:
            self.ui_queue.put(('timing_laid', {
                'obj_id': obj_id,
                'class_name': final_class,
                'event_epoch_s': time.time(),
            }))
        
        # Return to standby
        self.delta.go_standby()
        
        # Update stats
        if self.ui_queue:
            self.ui_queue.put(('sorted', final_class))
        
        # Log to database
        if self.db_mgr:
            self.db_mgr.log_detection(
                self.session_id,
                {'id': obj_id, 'class_name': final_class, 'vision_class_name': class_name,
                 'height_cm': height, 'status': 'Sorted'}
            )
        
        self.log(f"[ROBOT] ID:{obj_id} sorted to {final_class}")
        self._set_phase('IDLE')
        return True
    
    def stop(self):
        """Stop the robot manager."""
        self.running = False
