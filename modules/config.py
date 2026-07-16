"""
=============================================================================
CONFIGURATION MODULE
=============================================================================
Central configuration for the sorting system.
"""

import cv2
import numpy as np
import os
import glob

# ==========================================================================================
# CAMERA SETTINGS
# ==========================================================================================
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FPS = 30

# Robot camera (USB) — records robot picking actions
ROBOT_CAM_DEVICE = "/dev/video6"          # V4L2 device path
ROBOT_CAM_WIDTH = 640
ROBOT_CAM_HEIGHT = 480
ROBOT_CAM_FPS = 30

# ==========================================================================================
# YOLO SETTINGS
# ==========================================================================================
MODEL_PATH = "yolov26s_fixed.pt"
CONFIDENCE_THRESHOLD = 0.7

# ==========================================================================================
# CALIBRATION
# ==========================================================================================
CALIBRATION_FILE = "calibration_data.json"
OFFSETS_FILE = "offsets.json"

# ==========================================================================================
# BELT / ROI SETTINGS (cm)
# ==========================================================================================
ROI_HEIGHT_CM = 30
ROI_WIDTH_CM = 20
ENTRY_PATH_CM = 21.5
EXIT_PATH_CM = 21.5
CONVEYOR_SPEED_CM_S = 7.0  # Default belt speed (cm/s)

# Registration line position (cm from ROI start)
REGISTRATION_LINE_CM = 15.0

# Exit line position (cm from ROI start) — second timing checkpoint for speed measurement.
# Objects crossing this line after registration provide a measured transit time,
# enabling vision-based belt speed calculation instead of relying on the manual slider.
EXIT_LINE_CM = ROI_HEIGHT_CM   # 30cm — at the ROI exit boundary

# Speed measurement rolling window — how many per-object speed samples to keep.
# The median of recent samples becomes the measured belt speed.
SPEED_MEASUREMENT_WINDOW = 10

# Robot timing for reachability calculation
# Estimated time for robot to complete one pick cycle (approach + pick + lift + scan + throw)
ROBOT_PICK_CYCLE_TIME_S = 2.5  # seconds per pick operation
ROBOT_MOVE_TIME_S = 0.5  # Additional time to move between X positions (seconds)

# Robot workspace is AFTER the ROI exit, not inside the ROI
# Distance from registration line to middle of workspace = 37cm
# Calculation: REG at 15cm, ROI ends at 30cm, so 15cm to ROI exit
#              ROI exit is 14.50cm from workspace start
#              Workspace is 20cm deep, middle at 10cm
#              37 - 15 = 22cm from ROI exit to workspace middle => offset = 22-10 = 12cm
ROBOT_WORKSPACE_OFFSET_CM = 14.50  # Distance from ROI exit to workspace start (14.50cm gap)
ROBOT_WORKSPACE_DEPTH_CM = 20.0   # How deep the workspace is (Y direction)
ROBOT_WORKSPACE_WIDTH_CM = 20.0   # Same as belt width (X direction)

# Minimum depth into workspace before dispatching a pick (cm)
# How far into the workspace an object must travel before it becomes pickable.
# 0.0 = pick immediately at workspace edge (robot starts as soon as object enters).
# Higher values add a buffer so the robot picks objects deeper in the workspace.
MIN_PICK_WORKSPACE_Y_CM = 2.0

# ==========================================================================================
# PICK-POINT CONSOLIDATION
# ==========================================================================================
# Merge tracked objects whose centroids are too close together
# (same physical object detected as multiple entities).
# Prevents the robot from receiving duplicate pick commands for one object.
PICK_CONSOLIDATION_DIST_CM = 1.5   # Objects closer than 1.5cm are consolidated
PICK_CONSOLIDATION_DIST_PX = 30    # Pixel-space fallback (used before belt coords available)

# ==========================================================================================
# ROI MASK COVERAGE
# ==========================================================================================
# Minimum fraction of an object's mask pixels that must lie inside the ROI polygon
# for it to be detected/displayed.
# NOTE: 1.0 (100%) was too strict — objects entering the ROI were invisible
#       until fully inside, causing "one by one" detection behavior.
ROI_MIN_MASK_COVERAGE = 0.85  # 85%

# ==========================================================================================
# EXTENDED DETECTION ZONE — Include exit zone in detection
# ==========================================================================================
# By default, detection only happens inside the ROI (30 cm).
# When enabled, the detection mask is extended to include the exit zone (21.5 cm)
# so objects continue to be detected as they leave the ROI and travel through the gap
# toward the robot workspace.  This improves continuity tracking for stacked objects
# that may partially enter the ROI and get missed with ROI-only detection.
EXTENDED_DETECTION_ZONE = True           # Master toggle — extend detection to exit zone
EXIT_ZONE_MIN_MASK_COVERAGE = 0.50       # Lower coverage ok for exit zone (objects partially leave frame)

# ==========================================================================================
# CROSS-CLASS DUPLICATE SUPPRESSION (Mask NMS across all classes)
# ==========================================================================================
# YOLO sometimes detects the same physical object twice under different class labels
# (e.g., one mask as "Glass" and another overlapping mask as "Metal").
# Standard NMS only suppresses within the same class, so these duplicates survive.
# This performs mask-IoU NMS *across* classes — if two detections overlap heavily,
# the lower-confidence one is removed.
DUPLICATE_MASK_NMS_ENABLED = True    # Master toggle — guards against re-registration of the same object
DUPLICATE_MASK_IOU_THRESHOLD = 0.30  # IoU above this → same physical object (30%, lowered for async gaps)

# ==========================================================================================
# DEPTH CLUSTERING — Split merged YOLO masks using depth discontinuities
# ==========================================================================================
# When YOLO merges two adjacent objects into a single mask, depth clustering
# analyses the depth distribution inside that mask and splits it into
# sub-masks at depth valleys.
# DISABLED: depth noise on belt causes false splits that feed bad picks to the robot.
# Keep the code for experimentation (detection_experiment.py) — flip True to re-enable.
DEPTH_CLUSTER_ENABLED = False        # Master toggle for depth-based mask splitting
DEPTH_CLUSTER_MIN_MASK_PX = 800      # Ignore masks smaller than this (px) — too small to split
DEPTH_CLUSTER_DEPTH_GAP_MM = 15      # Minimum depth gap between histogram peaks to split (mm)
DEPTH_CLUSTER_MIN_CLUSTER_PX = 200   # Sub-cluster must have at least this many pixels
DEPTH_CLUSTER_MAX_SPLITS = 4         # Maximum number of sub-objects from one mask
DEPTH_CLUSTER_HIST_BINS = 50         # Number of histogram bins for depth analysis
DEPTH_CLUSTER_MORPH_OPEN_PX = 5      # Morphological opening kernel size (clean noise)
DEPTH_CLUSTER_SPATIAL_CONNECT = True  # Require spatial connectivity (not just depth similarity)

# ==========================================================================================
# WATERSHED MASK JOINING — rejoin split masks that belong to the same object
# ==========================================================================================
# When an object is partially occluded (e.g., by an object stacked on top),
# YOLO may produce two separate mask fragments for the bottom object.
# This uses the OpenCV watershed algorithm to attempt to rejoin fragments
# that are spatially close and at similar depth.
# The join boundary is drawn as a transparent grid line so the user can
# still see where the original split was.
# DISABLED: depth noise causes false joins, merging distinct objects into one pick target.
WATERSHED_JOIN_ENABLED = False         # Master toggle
WATERSHED_MAX_GAP_PX = 40             # Max pixel gap between fragment edges to consider joining
WATERSHED_DEPTH_TOLERANCE_MM = 20     # Fragments must be within this depth range to join
WATERSHED_MIN_FRAGMENT_PX = 150       # Ignore fragments smaller than this
WATERSHED_BOUNDARY_ALPHA = 0.35       # Transparency of the watershed grid boundary
WATERSHED_REQUIRE_SAME_CLASS = True    # Watershed only joins same-class fragments (class resolution is handled separately at registration via IoU)
WATERSHED_COLOR_SIM_ENABLED = True     # Use colour histogram similarity to gate joins
WATERSHED_COLOR_SIM_THRESHOLD = 0.45   # Min histogram correlation (0-1, higher=stricter) — 0.45 is lenient
WATERSHED_ASPECT_RATIO_TOL = 3.0       # Max aspect-ratio factor between fragments (skip if one is way wider)
WATERSHED_DEBUG = True                 # Print join decisions to console for troubleshooting

# ==========================================================================================
# MASK ALPHA FADING — smooth fade-in/out to reduce flickering
# ==========================================================================================
# Instead of masks appearing/disappearing instantly each frame, tracked
# objects carry a per-object alpha value that fades smoothly.
# When detection is lost the mask fades out; when re-detected it fades in.
MASK_FADE_ENABLED = True               # Master toggle
MASK_FADE_IN_RATE = 0.25              # Alpha increase per frame when detected (0→1)
MASK_FADE_OUT_RATE = 0.08             # Alpha decrease per frame when ghost   (1→0)
MASK_FADE_MIN_ALPHA = 0.0             # Minimum alpha before mask is hidden
MASK_FADE_MAX_ALPHA = 1.0             # Fully opaque ceiling

# Temporal mask EMA blending — smooths jittery edges between frames.
# Each frame: blended = old_mask * ALPHA + new_mask * (1-ALPHA)
# Higher ALPHA = smoother but more lag; 0 = raw YOLO mask (jittery).
MASK_TEMPORAL_SMOOTH = 0.5            # EMA factor (0=off, 0.5=balanced, 0.7=very smooth)
MASK_MORPH_SMOOTH_PX = 5              # Morphological close/open kernel radius for edge polish

# ==========================================================================================
# TEMPORAL IDENTITY TRACKING — preserve IDs when masks merge/split
# ==========================================================================================
TEMPORAL_TRACKING_ENABLED = True     # Master toggle for temporal reasoning
TEMPORAL_MERGE_DIST_PX = 60         # Distance to associate a disappearing object with a merge event
TEMPORAL_MAX_MERGE_FRAMES = 30      # Max frames an identity is held through a merge
TEMPORAL_SPLIT_MATCH_DEPTH_MM = 20  # Depth tolerance to re-identify a split sub-object

# ==========================================================================================
# QUEUE MANAGEMENT
# ==========================================================================================
QUEUE_EXIT_BUFFER_CM = 10.0          # Extra cm past workspace end before removal

# ==========================================================================================
# STACKING DETECTION (mask-based adjacency)
# ==========================================================================================
# Stacking is detected by checking if object masks are adjacent or overlapping.
# Each mask is dilated by this many pixels before checking overlap.
# Larger values = more generous adjacency detection (catches nearby objects).
# Typical: 15-25px for objects that are touching/close on the belt.
STACK_MASK_DILATE_PX = 20

# Mask IoU threshold — pair must exceed this to be considered related at all.
# IoU = intersection_pixels / union_pixels of dilated masks.
# Any overlap above this = at least "adjacent".
STACK_MASK_IOU_THRESHOLD = 0.02  # 2% overlap of dilated masks → adjacent

# Stacking IoU threshold — if IoU >= this the object is INSIDE / ON TOP of
# the other object's region.  Below this but above STACK_MASK_IOU_THRESHOLD
# means the objects are merely close (adjacent), not stacked.
STACK_IOU_STACKING_MIN = 0.10   # 10% overlap → stacking (one on top of other)

# Depth-based stacking confirmation (secondary check).
# Even with high IoU, if height difference is < this the objects are at the
# same level — still labelled as stacking by IoU but without height ordering.
STACK_HEIGHT_DIFF_MIN_CM = 2.0

# Extra Y advance (cm) added to predicted pick position for the bottom object
# in a stack.  After picking the top, the robot lifts+scans+throws and comes
# back — during that cycle the bottom object drifts further along the belt.
# This empirical offset compensates for that drift.
STACK_BOTTOM_Y_ADVANCE_CM = 2.0

# Pick priority scoring weights
# Final score = w_urgency*urgency + w_height*height_bonus + w_isolation*isolation
#             - w_stack_risk*stack_penalty
PICK_PRIORITY_WEIGHTS = {
    'urgency':     1.0,   # Higher belt_y = closer to exit = more urgent
    'height':      0.3,   # Taller objects get slight bonus (pick from top)
    'isolation':   0.5,   # Objects NOT in a stack group get bonus
    'stack_risk':  0.8,   # Penalty for objects in tight overlap (IoU)
}

# ==========================================================================================
# TRACK-TO-PICK (real-time approach)
# ==========================================================================================
# During approach phase, robot tracks the object in real-time (like TRACK mode)
# then descends to grab when the object is at the right Y position.
TRACK_APPROACH_HOVER_MM = 50    # Hover height above pick Z during approach (mm)
TRACK_APPROACH_SPEED = 15000    # G-code feedrate during tracking approach
TRACK_DESCEND_SPEED  = 20000    # G-code feedrate during descent (faster — robot drops Z + tracks Y)
PICK_SURFACE_PENETRATION_MM = 5 # Max depth below object top surface the gripper is allowed to go (mm)
                                # Prevents crushing/flinging objects when Z offset is too aggressive.
                                # Gripper aims at: object_top - PENETRATION.  Vacuum handles grip.

# ==========================================================================================
# TRACKING SETTINGS
# ==========================================================================================
MAX_TRACKING_DISTANCE_PX = 80
MAX_DISAPPEARED_FRAMES = 15
GHOST_GRACE_FRAMES = 3             # Keep mask/contour for N frames after detection loss
                                    # (async detector can miss 1-2 frames — clearing mask
                                    #  immediately breaks stacking IoU and causes re-registration)

# ==========================================================================================
# CLASS DEFINITIONS
# ==========================================================================================
CLASS_NAMES = {
    0: "Glass",
    1: "Metal",
    2: "Paper",
    3: "Plastic"
}

CLASS_COLORS = {
    0: (0, 255, 255),    # Glass - Yellow
    1: (192, 192, 192),  # Metal - Silver
    2: (0, 165, 255),    # Paper - Orange
    3: (0, 255, 0)       # Plastic - Green
}

# Minimum object height per class (cm)
# Used to compensate for RealSense depth errors on transparent/reflective objects
# If detected height is below this, use this minimum instead
CLASS_MIN_HEIGHT_CM = {
    "Glass": 5.0,    # Glass is often transparent, depth unreliable
    "Metal": 3.0,    # Metal can be reflective
    "Paper": 0.1,    # Paper is flat, can be very thin
    "Plastic": 4.0,  # Some plastics are transparent
}
DEFAULT_MIN_HEIGHT_CM = 1.0  # Default for unknown classes

# Per-class Z offset (mm) added to pick Z during suction.
# Positive = higher (less pressure), Negative = lower (more pressure).
# - Paper: goes negative because paper deflects away from suction nozzle
# - Metal: goes positive to reduce impact force and protect the robot
# - Glass: goes positive to avoid cracking and reduce force
# - Plastic: slight positive, less rigid than metal/glass
CLASS_Z_OFFSET_MM = {
    "Glass":    5.0,   # Fragile — stay a bit higher
    "Metal":   10.0,   # Hard surface — reduce pressure significantly
    "Paper":  -10.0,   # Deflects down — push lower for better grip
    "Plastic":  0.0,   # Neutral default
}

DEFAULT_COLOR = (255, 0, 255)  # Magenta

# ==========================================================================================
# DEPTH COLORMAP OPTIONS
# ==========================================================================================
COLORMAP_OPTIONS = [
    cv2.COLORMAP_JET,
    cv2.COLORMAP_TURBO,
    cv2.COLORMAP_VIRIDIS,
    cv2.COLORMAP_PLASMA,
    cv2.COLORMAP_INFERNO,
    cv2.COLORMAP_HOT
]
COLORMAP_NAMES = ["JET", "TURBO", "VIRIDIS", "PLASMA", "INFERNO", "HOT"]

# ==========================================================================================
# SPECTRUM SENSOR CHANNELS
# ==========================================================================================
SENSOR_CHANNELS_SPECTRAL = [
    "410nm (A)", "435nm (B)", "460nm (C)", "485nm (D)", "510nm (E)", "535nm (F)",
    "560nm (G)", "585nm (H)", "610nm (R)", "645nm (I)", "680nm (S)", "705nm (J)",
    "730nm (T)", "760nm (U)", "810nm (V)", "860nm (W)", "900nm (K)", "940nm (L)"
]

# ==========================================================================================
# SPECTRUM SCAN MOTOR (W AXIS) CALIBRATION
# ==========================================================================================
# Motor W=0 deg is HOME position, aligned with belt X-axis (left-right).
# Motor W=90 deg is aligned with belt Y-axis (belt travel direction).
# Range: 0-180 degrees.
#
# HOMING: Send W=-180 to force motor to its physical hard stop (always
# reaches 0 regardless of current position), then reset that as 0.
#
# PCA orientation angle 0 deg = object long axis along belt X = motor 0 deg.
# Since orientation is 180-symmetric, PCA angle maps directly to 0-180 motor range.
# Adjust SCAN_W_OFFSET_DEG if sensor is slightly misaligned from motor 0.
SCAN_W_OFFSET_DEG  = 0.0       # Fine-tune offset (degrees)
SCAN_W_MIN_DEG     = 0.0       # Motor minimum W (degrees) = home
SCAN_W_MAX_DEG     = 180.0     # Motor maximum W (degrees)
SCAN_W_HOME_CMD    = -180.0    # Value sent to force motor to hard stop for homing

# ==========================================================================================
# ROBOT PARAMETERS
# ==========================================================================================
BASE_MODEL_PATH = "./Model"
DEFAULT_DELTA_PORT = '/dev/ttyUSB0'
DEFAULT_SLIDER_PORT = '/dev/ttyUSB1'

# Robot throw/place targets (x, y) in mm — DEFAULT values
# These can be overridden at runtime from the UI and saved to PLACE_TARGETS_FILE.
THROW_TARGETS = {
    "Glass": (0.0, 150),
    "Metal": (0.0, -150.0),
    "Paper": (150.0, 80.0),
    "Plastic": (-150.0, -70.0)
}
PLACE_Z_HEIGHT = -300.0       # Z height for place (mm)
PLACE_Z_RELEASE = -320.0      # Z height for release push (mm)
PLACE_TARGETS_FILE = "place_targets.json"

# Robot position mapping - 2D Grid (3x3)
# Belt coordinates: X = 0-200mm (left to right), Y = 0-200mm (top to bottom of pick zone)
# Robot coordinates are in mm, measured at each grid point
#
# IMPORTANT - DELTA ROBOT KINEMATICS:
# This is a 3-arm delta robot, NOT a Cartesian gantry. The 3 motors work together
# to position the end effector, which means:
#   - Robot X and Y are COUPLED — moving to a different belt X also changes the
#     required robot Y (and vice versa), because the arm geometry creates
#     non-linear, angle-dependent reach patterns.
#   - You CANNOT calibrate X and Y independently (e.g., with separate 1D arrays).
#     A single belt position maps to a UNIQUE (robot_x, robot_y) PAIR that must
#     be calibrated together at each grid point.
#   - This is why we use a 2D grid with bilinear interpolation: each grid point
#     stores the PAIRED (robot_x, robot_y) measured by physically jogging the
#     robot to that belt position.
#
# To recalibrate: jog the robot to each of the 9 belt grid points, record both
# the robot X AND Y at each point, and update ROBOT_X_GRID and ROBOT_Y_GRID together.
#
# Grid layout (belt perspective):
#   TL (0,0)    TC (100,0)   TR (200,0)    <- Top row (belt Y = 0mm)
#   ML (0,100)  MC (100,100) MR (200,100)  <- Middle row (belt Y = 100mm)  
#   BL (0,200)  BC (100,200) BR (200,200)  <- Bottom row (belt Y = 200mm)

# Belt grid coordinates (mm)
BELT_X_GRID = [0.0, 100.0, 200.0]    # Left, Center, Right
BELT_Y_GRID = [0.0, 100.0, 200.0]    # Top, Middle, Bottom

# Robot X positions (mm) for each grid point [row][col]
# Row 0 = Top (belt Y=0), Row 1 = Middle (belt Y=100), Row 2 = Bottom (belt Y=200)
# NOTE: ROBOT_X_GRID and ROBOT_Y_GRID are PAIRED — each (row, col) entry in both
# grids was measured together. Do NOT update one without updating the other.
ROBOT_X_GRID = [
    [-30.0,  -70.0, -100.0],   # Top row:    TL, TC, TR
    [ 35.0,    0.0,  -35.0],   # Middle row: ML, MC, MR
    [100.0,   80.0,   50.0],   # Bottom row: BL, BC, BR
]

# Robot Y positions (mm) for each grid point [row][col]
ROBOT_Y_GRID = [
    [  80.0,   60.0,    0.0],   # Top row:    TL, TC, TR
    [  45.0,    0.0,  -45.0],   # Middle row: ML, MC, MR
    [   0.0,  -50.0, -100.0],   # Bottom row: BL, BC, BR
]

# Legacy 1D arrays (for backward compatibility if needed)
X_DELTA_MM = [35.0, 0.0, -35.0]   # Robot X positions (middle row)
Y_DELTA_MM = [45.0, 0.0, -45.0]   # Robot Y positions (middle row)
X_BELT_MM = [0.0, 100.0, 200.0]   # Belt X positions (0-200mm = 0-20cm)

BASE_Z = -425.0              # Real measured belt-contact Z (calibrated 2026-04-08)
STANDBY_POSITION = (0, 0, -250)

# ==========================================================================================
# MODEL DISCOVERY
# ==========================================================================================
def discover_models():
    """
    Scan for YOLO model files (.pt, .engine) in the project directory and
    its parent directory.  Returns (paths, display_names) — both as lists.

    Priority rules:
    - .engine (TensorRT) is preferred over .pt (PyTorch) when both exist
      for the same model stem (e.g. yolov8s.engine beats yolov8s.pt).
    - When the same basename exists in both directories the local (project)
      copy is preferred so the list stays clean.
    - Display names show format suffix: "yolov8s.engine ⚡" for TensorRT,
      plain name for .pt files.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Spectrum_pipeline/
    parent_dir  = os.path.dirname(project_dir)                                 # pipeline_new/
    extensions = ("*.pt", "*.engine")

    # Collect per-directory, project dir first so it wins on dedup
    by_basename = {}                       # basename -> abs path
    for d in [project_dir, parent_dir]:
        for ext in extensions:
            for p in glob.glob(os.path.join(d, ext)):
                bn = os.path.basename(p)
                if bn not in by_basename:  # first seen (project dir) wins
                    by_basename[bn] = p

    # Group by stem — prefer .engine over .pt for same model
    by_stem = {}  # stem -> abs path (best format)
    for bn, path in by_basename.items():
        stem = os.path.splitext(bn)[0]
        ext  = os.path.splitext(bn)[1]
        if stem not in by_stem:
            by_stem[stem] = path
        else:
            existing_ext = os.path.splitext(by_stem[stem])[1]
            # .engine takes priority over .pt
            if ext == '.engine' and existing_ext == '.pt':
                by_stem[stem] = path

    # Also keep both formats available — user may want to switch
    # Final list: grouped (engine preferred) + any extras not covered
    all_paths = set(by_stem.values())
    # Add back any files that were eclipsed (so .pt is still selectable)
    for bn, path in by_basename.items():
        all_paths.add(path)

    # Sort: current default first, then .engine before .pt, then alphabetically
    default_abs = os.path.abspath(MODEL_PATH)
    models = sorted(all_paths,
                    key=lambda p: (
                        p != default_abs,                          # default first
                        not p.endswith('.engine'),                  # .engine before .pt
                        os.path.basename(p).lower()                # alphabetical
                    ))
    names  = [os.path.basename(p) for p in models]
    return models, names

# Pre-built list for UI combos
AVAILABLE_MODELS, AVAILABLE_MODEL_NAMES = discover_models()
