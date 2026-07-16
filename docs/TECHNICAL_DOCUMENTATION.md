# Waste Sorting Delta Robot System — Technical Documentation

**Project:** Automated Conveyor Belt Waste Sorting System  
**Platform:** NVIDIA Jetson  
**Last Updated:** April 7, 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture & Module Breakdown](#2-architecture--module-breakdown)
3. [Hardware Specifications](#3-hardware-specifications)
4. [Belt & Coordinate System](#4-belt--coordinate-system)
5. [Detection Pipeline](#5-detection-pipeline)
6. [Tracking & Queuing Strategy](#6-tracking--queuing-strategy)
7. [Stacking Detection Algorithm](#7-stacking-detection-algorithm)
8. [Smart Pick Priority Scoring](#8-smart-pick-priority-scoring)
9. [Robot Control & Pick Execution](#9-robot-control--pick-execution)
10. [Position Mapping (Belt → Robot)](#10-position-mapping-belt--robot)
11. [Adaptive Bayesian Fusion](#11-adaptive-bayesian-fusion)
12. [Spectrum Sensor Integration](#12-spectrum-sensor-integration)
13. [Configuration Reference](#13-configuration-reference)
14. [Data Logging](#14-data-logging)
15. [User Interface (DearPyGui)](#15-user-interface-dearpygui)
16. [Manual Inspection Tool](#16-manual-inspection-tool)
17. [Watershed Mask Joining](#17-watershed-mask-joining)
18. [Mask Alpha Fading](#18-mask-alpha-fading)
19. [IoU Class Resolution at Registration](#19-iou-class-resolution-at-registration)
20. [Hardware Disconnect](#20-hardware-disconnect)
21. [Detection Experiment Tool](#21-detection-experiment-tool)
22. [Display Lag Fix (Smoothing Removal)](#22-display-lag-fix-smoothing-removal)
23. [Robot Freeze Root Cause Analysis](#23-robot-freeze-root-cause-analysis)
24. [Known Threading & Safety Issues](#24-known-threading--safety-issues)
25. [TensorRT (.engine) Support](#25-tensorrt-engine-support)
26. [Synchronous Detection Refactor](#26-synchronous-detection-refactor)
27. [Mask Temporal Smoothing](#27-mask-temporal-smoothing)
28. [Integrated Calibration (Tab 5)](#28-integrated-calibration-tab-5)
29. [Vision-Based Belt Speed Measurement](#29-vision-based-belt-speed-measurement)
30. [Central Dashboard Sync Architecture](#30-central-dashboard-sync-architecture)

---

## 1. System Overview

This system uses a **3-arm delta robot** mounted above a **conveyor belt** to automatically detect, classify, pick, and sort waste objects into 4 categories:

| Class ID | Material | Color Code (BGR) |
|----------|----------|-------------------|
| 0 | Glass | (0, 255, 255) Yellow |
| 1 | Metal | (192, 192, 192) Silver |
| 2 | Paper | (0, 165, 255) Orange |
| 3 | Plastic | (0, 255, 0) Green |

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **Vision** | Intel RealSense RGB-D camera + YOLOv26s-fixed instance segmentation |
| **Spectral** | AS7265X 18-channel NIR/Vis/UV sensor (410–940nm) |
| **Classification** | Adaptive Bayesian Fusion (YOLO + Spectrum) |
| **Tracking** | Centroid-based spatial tracker (belt coordinate space) |
| **Robot** | 3-arm Delta robot (G-code/serial) + linear slider |
| **UI** | Dear PyGui with 5-tab interface |
| **Platform** | NVIDIA Jetson |

### End-to-End Flow (Summary)

```
Camera → YOLO Detection → Tracker → Registration Queue (time-anchor stored) →
Smart Pick (priority scored) → Track-to-Pick (real-time tracking + predictive descend) →
Vacuum Grip → Spectrum Scan → Bayesian Fusion → Sort → Log
```

---

## 2. Architecture & Module Breakdown

### File Structure

```
Spectrum_pipeline/
├── index.py                    # Main application (SortingApp, DearPyGui UI)
├── Manual_Inspection.py        # Standalone manual inspection tool (capture, scan, label, confusion matrix)
├── modules/
│   ├── config.py               # Central configuration constants
│   ├── tracker.py              # SimpleTracker: tracking + queuing + stacking + time-anchor
│   ├── robot.py                # RobotManager: pick execution + Bayesian fusion
│   ├── camera.py               # CameraStream: RealSense threading
│   ├── detector.py             # ObjectDetector: YOLO + homography + measurements
│   ├── spectrum.py             # SpectrumSensor: AS7265X + CatBoost ML
│   ├── database.py             # DetectionLogger (CSV) + DatabaseManager (SQLite)
│   ├── dashboard_sync.py       # Machine-side summary sync worker + outbox
│   ├── dashboard_central.py    # Central dashboard DB and query helpers
│   └── voting_logic.py         # Voting-based fusion alternative
├── dashboard_app.py            # Central Flask dashboard
├── run_dashboard_sync.py       # Sync worker CLI
├── as7265x_sparkfun_python.py  # AS7265X I2C driver library (Python port of SparkFun Arduino lib)
├── detection_experiment.py      # Standalone detection comparison tool (DearPyGui)
├── calibration_tool.py         # Standalone ROI + floor plane calibration (backup)
├── delta_control.py            # Standalone Tkinter robot control GUI
├── Spectrum_Control.py         # Standalone spectrum dashboard + SHAP
├── realsense_depth_segmentation.py  # Standalone OpenCV viewer (experiment)
├── calibration_data.json       # Camera calibration output
├── offsets.json                # Robot pick offset fine-tuning
├── place_targets.json          # Per-class place bin positions
├── yolov26s_fixed.pt           # YOLOv26s-fixed segmentation model (default — extra dataset training)
├── yolov26s.pt                 # YOLOv26s segmentation model (original)
├── yolov8s.pt                  # YOLOv8s segmentation model (available)
├── Model/                      # CatBoost spectrum classification model
├── docs/                       # Technical & user documentation
└── _legacy/                    # Archived old program versions (Final_*, Master_*, Merge_*, etc.)
```

### Module Responsibilities

| Module | Class | Lines | Purpose |
|--------|-------|-------|----------|
| `index.py` | `SortingApp` | ~4420 | Main app: 5-tab UI, pipeline orchestration, workspace simulation, integrated calibration |
| `Manual_Inspection.py` | — | ~2306 | Standalone manual inspection: capture, YOLO, spectrum, voting, labeling, confusion matrix |
| `detection_experiment.py` | — | ~960 | Standalone detection comparison tool: model hot-swap, stacking analysis |
| `config.py` | — | ~442 | All constants: camera, belt, robot, stacking, classes, mask smoothing |
| `tracker.py` | `SimpleTracker` | ~1710 | Object tracking, queue management, stacking detection, time-anchor prediction, priority pick, mask smoothing |
| `robot.py` | `RobotManager`, `DeltaController`, `SliderController` | ~1016 | Robot control, direct predictive pick, Bayesian fusion |
| `camera.py` | `CameraStream`, `VideoPlaybackStream` | ~200 | Threaded RealSense frame capture + .bag video playback |
| `detector.py` | `ObjectDetector` | ~1428 | YOLO inference (synchronous), homography, height/size measurement, calibration loading |
| `spectrum.py` | `SpectrumManager` | ~200 | 18-channel spectral reading + CatBoost prediction |
| `database.py` | `DetectionLogger`, `DatabaseManager` | ~292 | CSV (30 columns) + SQLite logging |
| `dashboard_sync.py` | `MachineSyncService` | new | 1-minute summary aggregation, outbox persistence, retry upload |
| `dashboard_central.py` | `CentralDashboardStore` | new | central dashboard schema, API upsert, web query shaping |

---

## 3. Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **Robot type** | 3-arm delta, G-code controlled |
| **Delta serial** | 115200 baud, `/dev/ttyACM0` |
| **Slider serial** | 115200 baud, `/dev/ttyUSB1` |
| **Delta XY range** | ±110 mm (clamped) |
| **Delta Z range** | −450 to −150 mm |
| **Belt surface Z** | −415 mm |
| **Standby position** | (0, 0, −250) mm |
| **Vacuum control** | M3 (on) / M5 (off) via G-code |
| **Rotation range** | 20°–130° (W parameter) |
| **Camera** | Intel RealSense (RGB-D), 640×480 @ 30fps |
| **Spectrum sensor** | SparkFun AS7265X, I2C bus 1, 18 channels (410–940nm) |
| **Conveyor belt** | 20 cm wide × 30 cm ROI, ~5.5–6.0 cm/s |
| **Pick cycle time** | ~2.5 s (approach + pick + scan + sort) |
| **Duplicate prevention** | 1.0 s minimum between picks |

### Robot Throw Targets (mm)

| Class | Robot X | Robot Y |
|-------|---------|---------|
| Glass | 0.0 | 150.0 |
| Metal | 0.0 | −150.0 |
| Paper | 150.0 | 80.0 |
| Plastic | −150.0 | −70.0 |

### Class-Specific Minimum Heights (cm)

| Class | Min Height | Reason |
|-------|-----------|--------|
| Glass | 5.0 | RealSense fails on transparent surfaces |
| Metal | 3.0 | Reflective surfaces cause depth errors |
| Paper | 0.1 | Flat, reliable depth |
| Plastic | 4.0 | Semi-transparent, unreliable depth |

---

## 4. Belt & Coordinate System

### Physical Layout (Y axis, cm from ROI start)

```
  0 cm ─────── ROI Start (camera detection begins)
  │
  │  Camera detects objects here
  │  Objects get tracked, masks analyzed
  │
 15 cm ─────── Registration Line
  │             Objects enter picking queue here
  │             Stacking groups detected at this moment
  │             Reachability estimated
  │
 30 cm ─────── ROI Exit (camera detection ends)
  │
  │  12 cm GAP (blind zone — no camera coverage)
  │  Objects tracked by belt speed × time
  │
 42 cm ─────── Workspace Entry (robot can reach)
  │
  │  Robot workspace (20 cm deep)
  │  Smart pick dispatched when object ≥ 8 cm in (≥50 cm belt_y)
  │
 62 cm ─────── Workspace Exit (robot can no longer reach)
  │
  │  10 cm buffer before removal from memory
  │
 72 cm ─────── Queue Exit Limit (object removed from queue + memory)
```

### Coordinate Spaces

| Space | Range | Unit | Used By |
|-------|-------|------|---------|
| **Pixel** | 0–640 × 0–480 | px | Camera, YOLO, display |
| **Belt** | X: 0–20, Y: 0–30+ | cm | Tracker, queue, registration |
| **Workspace** | X: 0–20, Y: 0–20 | cm | Robot pick dispatch (Y relative to workspace entry) |
| **Robot** | X: ±110, Y: ±110, Z: −450 to −150 | mm | Delta arm G-code |

### Coordinate Conversions

- **Pixel → Belt cm**: Homography matrix (from calibration) + `pixel_y_to_belt_y_cm()`, `pixel_x_to_belt_x_cm()`
- **Belt cm → Robot mm**: 3×3 bilinear interpolation grid (`belt_to_robot_bilinear()`)
- **Robot mm → Belt cm**: Iterative inverse search (`inverse_belt_to_robot()`)

---

## 5. Detection Pipeline

### Per-Frame Flow

```
1. CameraStream captures aligned RGB + Depth at 30fps (daemon thread)
2. YOLO segmentation runs synchronously on the main thread (no lag)
3. Per detection:
   a. Resize mask to 640×480, binarize at 0.5 threshold
   b. Find contours → compute minAreaRect
   c. Centroid from mask moments
   d. ROI mask coverage check (≥90% of mask pixels must be inside ROI)
   e. Height from RealSense depth (median of valid mask pixels vs floor plane)
   f. Physical size from mask + depth + camera intrinsics
   g. Orientation from PCA on mask pixels
   h. Belt coordinates via homography
4. Cross-class mask NMS (suppress duplicate detections across classes, IoU ≥ 0.40) [toggleable]
5. Watershed mask joining (rejoin same-class fragments split by occlusion) [toggleable]
6. Returns list of detection dicts
```

> **Note (March 2026):** All three separation stages (Depth Clustering, Cross-Class NMS,
> Watershed Joining) are now **disabled by default** (raw YOLO output) and can be
> toggled live from the Vision tab UI in both `index.py` and `detection_experiment.py`.
> The `ObjectDetector` class holds instance-level toggle attributes (`use_depth_cluster`,
> `use_cross_nms`, `use_watershed`) that the UI updates at runtime.
>
> **Note (March 2026):** Detection is now **synchronous** — YOLO runs directly on the
> main DearPyGui thread each frame. The old `ThreadedDetector` wrapper (which caused
> 1–2 frame lag between detection results and displayed frame) has been removed.
> See [Section 26: Synchronous Detection Refactor](#26-synchronous-detection-refactor).

### Detection Dict Structure

```python
{
    'centroid': (cx, cy),            # Pixel coordinates
    'class_id': int,                 # 0=Glass, 1=Metal, 2=Paper, 3=Plastic
    'confidence': float,             # YOLO confidence score
    'mask': np.array,                # Binary mask (480×640, uint8)
    'min_area_box': np.array,        # Rotated bounding box corners
    'height_cm': float,              # Height above belt surface
    'width_cm': float,               # Physical width (2D footprint)
    'obj_height_cm': float,          # Physical height (2D footprint)
    'angle': float,                  # PCA orientation angle
    'belt_x_cm': float,             # Belt X coordinate
    'belt_y_cm': float,             # Belt Y coordinate
    'depth_mm': float,              # Average depth in mm
    'is_watershed_joined': bool,     # True if this detection was merged by watershed
    'joined_count': int,             # Number of fragments joined (if watershed)
    'watershed_boundary': np.array,  # Boundary mask between joined fragments
}
```

---

## 6. Tracking & Queuing Strategy

### SimpleTracker (`modules/tracker.py`)

The tracker operates in **belt coordinate space (cm)** for stability — pixel positions jitter but belt coordinates are smooth.

### Tracking Algorithm

1. **Match** each new detection to existing objects by:
   - Same class ID
   - Euclidean distance in belt cm < 10.0 cm threshold
   - Closest match wins (greedy)
2. **Update** matched objects:
   - Centroid and bounding box updated directly from detection (smoothing disabled: α=0.0)
   - Fresh mask, height, angle from detection
   - **Non-queued:** belt_x, belt_y updated from detection
   - **Queued:** belt_x AND belt_y are frozen at registration (no more detection updates)
   - Mask alpha fade-in when re-detected

   > **Note (March 2026):** Centroid smoothing (EMA α) and box smoothing were previously
   > 0.7 and 0.75 respectively. Both have been set to **0.0** to eliminate display lag
   > between raw mask positions and smoothed centroid/box positions. See
   > [Section 22: Display Lag Fix](#22-display-lag-fix-smoothing-removal) for details.
3. **Register** unmatched detections as new objects (if belt_y < registration + 10cm)
   - Tracker-level cross-class duplicate guard: checks mask IoU against all existing objects
4. **Ghost tracking:** Undetected objects get mask_alpha faded out, last_valid_mask preserved
5. **Remove** objects that disappear for > 15 frames (non-queued) or past exit limit (queued)

### Object Lifecycle

```
New Detection → Tracking (ID assigned) → Registration Line Crossing →
  ↓ IoU Class Resolution (choose single best class)
  ↓ Position & Class Locked
Queued (stacking analyzed) → Workspace Entry → Smart Pick Dispatch →
Robot Picks → mark_picked() → Removed from queue
        ↓ (if unreachable)
    Marked 'Unreachable' → Rides belt out → Auto-removed at exit limit
        ↓ (if IoU overlap with another crossing object)
    Marked 'Absorbed' → Not queued (duplicate of same physical object)
```

### IoU Class Resolution at Registration (NEW)

When objects cross the registration line, `_resolve_class_at_registration()` compares
every pair of crossing objects via mask IoU. If IoU ≥ threshold (0.40), the two
detections represent the **same physical object** detected under different YOLO classes.

The winner is chosen by a **composite score**:

```python
score = normalized_mask_area × 0.6  +  confidence × 0.4
```

- **60% weight on mask area** (object resolution) — the detection covering more of the
  object is more likely the correct class.
- **40% weight on YOLO confidence** — still matters but doesn't dominate.

The loser is marked `status='Absorbed'` and not queued. Its height is transferred to
the winner if taller.

### Registration Locking (NEW)

When an object is queued at the registration line, the following are **permanently frozen**:

| Field | Stored As | Purpose |
|-------|-----------|--------|
| Belt X position | `reg_belt_x` | Locked pick X — no more detection updates |
| Belt Y position | `reg_belt_y` | Time-anchor reference for Y prediction |
| Registration time | `reg_time` | Time-anchor for drift-free Y prediction |
| Class ID | `registered_class_id` | Locked class — won't change from later detections |
| Class name | `registered_class_name` | Human-readable locked class |

`get_smart_pick()` and `get_next_pick()` use these locked values, ensuring the robot
always picks based on the registration-time snapshot, not fluctuating detection data.

### Mask Alpha Fading (NEW)

Instead of masks appearing/disappearing instantly (causing flicker), each tracked
object carries a per-object `mask_alpha` value:

| Event | Behaviour |
|-------|-----------|
| Object detected | `mask_alpha += MASK_FADE_IN_RATE (0.25)` per frame, up to 1.0 |
| Object lost (ghost) | `mask_alpha -= MASK_FADE_OUT_RATE (0.08)` per frame, down to 0.0 |
| Ghost rendering | Uses `last_valid_mask` (preserved from last detection) |

Rendered by `_draw_tracked_masks()` in `index.py` which draws from tracked objects
instead of raw detections, applying per-object fade alpha.

### Time-Anchor Position Prediction

Instead of accumulating small belt speed × dt increments each frame (which drifts over time), the system uses a **time-anchor** system:

1. When an object crosses the **registration line**, the tracker stores:
   - `reg_time`: timestamp of registration
   - `reg_belt_y`: belt Y position at that moment (~15 cm)
   - `reg_belt_x`: belt X position at that moment (locked)

2. All subsequent Y position predictions use a single multiplication:
   ```python
   predicted_y = reg_belt_y + (current_time - reg_time) × belt_speed
   ```

3. This is **drift-free** — no error accumulation, no matter how long the object has been tracked.

4. The helper `_get_anchor_belt_y(obj)` implements this with fallback to incremental prediction for objects without anchors.

Used by: `advance_queued_objects()`, `get_smart_pick()`, `get_realtime_object_position()`, `_process_tracking()` in index.py, and the workspace simulation display.

### Queue Management

- Objects enter queue at **registration line** (15 cm belt_y)
- IoU class resolution applied before queuing (same physical object → single best class)
- Position (X and Y) and class are **locked** at registration — no further detection updates
- belt_y is predicted via **time-anchor** (reg_y + elapsed × speed) — drift-free
- Queue is ordered by registration time (FIFO), but pick selection uses **priority scoring**
- Objects removed from queue when belt_y > 72 cm (workspace end + 10 cm buffer)

---

## 7. Stacking Detection Algorithm

### Problem

Objects on the belt can be **stacked** (one on top of another) or **adjacent** (side by side, touching). The robot must:
- **Physical stacks**: Pick the top object first (tallest-first enforcement)
- **Adjacent objects**: Penalize but don't hard-block (they can be picked carefully)

### Algorithm: `_detect_mask_stack_groups()` — 3 Techniques Combined

#### Step 1: Mask Dilation + IoU Scoring

```python
# Dilate each object's binary mask by 20px elliptical kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
dilated_mask = cv2.dilate(mask, kernel, iterations=1)

# Compute pairwise IoU
intersection = cv2.bitwise_and(mask_a, mask_b)
union = cv2.bitwise_or(mask_a, mask_b)
iou = count_nonzero(intersection) / count_nonzero(union)
```

- IoU ≥ 0.02 (2%) → objects are in contact
- IoU < 0.02 → objects are separate

#### Step 2: Depth/Height Difference (Physical Stack Confirmation)

```
height_diff = abs(height_a - height_b)

If IoU ≥ threshold AND height_diff ≥ 2.0 cm → 'physical_stack'
If IoU ≥ threshold AND height_diff < 2.0 cm → 'adjacent'
If IoU < threshold                          → 'none'
```

#### Step 3: Union-Find Grouping

Transitive closure: if A↔B and B↔C overlap, they form group {A, B, C}.  
Each group sorted by height (tallest first = on top = pick first).

### Per-Pair Data Stored (`_pair_info`)

```python
{
    (id_a, id_b): {
        'iou': 0.035,              # IoU score
        'height_diff': 3.2,        # Height difference in cm
        'stack_type': 'physical_stack',  # or 'adjacent' or 'none'
        'inter_pixels': 1240,      # Intersection pixel count
        'union_pixels': 35400,     # Union pixel count
    }
}
```

### Per-Object Metadata (set at registration)

```python
obj['max_iou'] = 0.035           # Highest IoU with any neighbor
obj['stack_type'] = 'physical_stack'  # Worst-case relationship type
obj['stack_group'] = [5, 3, 7]   # Group IDs, sorted tallest-first
```

---

## 8. Smart Pick Priority Scoring

### `get_smart_pick()` — Production Version

Combines **real-time position tracking** with **priority scoring**.

#### Step 1: Collect Candidates

Only objects currently **inside the workspace** (belt_y ≥ 45 cm, i.e., workspace entry + 3 cm) with real-time position:

```python
real_belt_y = _get_anchor_belt_y(obj)  # Time-anchor based (drift-free)
# Equivalent to: reg_belt_y + (now - reg_time) × belt_speed
```

#### Step 2: Hard-Block Physical Stacks

Objects with `stack_type == 'physical_stack'` are **hard-blocked** if a taller group member is still a candidate. This enforces tallest-first picking for true physical stacks.

#### Step 3: Priority Scoring (Soft-NMS Style)

For all non-blocked candidates:

$$\text{Score} = W_u \cdot \text{urgency} + W_h \cdot \text{height} + W_i \cdot \text{isolation} - W_s \cdot \text{stack\_penalty}$$

| Factor | Formula | Weight | Meaning |
|--------|---------|--------|---------|
| Urgency | `real_belt_y / max_belt_y` | 1.0 | Closer to exit = more urgent |
| Height | `min(height_cm / 20, 1.0)` | 0.3 | Taller = on top = pick first |
| Isolation | `1.0 / group_size` | 0.5 | Standalone objects preferred |
| Stack Risk | `max_iou` | 0.8 | High overlap = risk of knocking neighbors |

#### Step 4: Select Best

Highest-scoring candidate is dispatched to the robot.

### Why Not Pure FIFO?

- FIFO ignores **stacking constraints** (lower object can't be picked first)
- FIFO ignores **urgency** (an object about to exit should be prioritized)
- Adjacent objects with high IoU risk knocking neighbors — soft penalty reduces this

---

## 9. Robot Control & Pick Execution

### Architecture: Serial Controllers

| Class | Port | Purpose |
|-------|------|--------|
| `SerialController` | — | Base class: `send()` fire-and-forget, `read_response(timeout=0.5)` |
| `DeltaController` | `/dev/ttyACM0` | `move_to(x,y,z,w,f)`, `set_vacuum(on)` (M3/M5), `home()` (G28), `go_standby()` |
| `SliderController` | `/dev/ttyUSB1` | `move_to(pos)` (M321+M322), `home()` (M320) |

All G-code commands are **fire-and-forget** — `send()` writes to serial and returns
immediately without waiting for an "ok" response from the firmware.

### RobotManager Thread

`RobotManager` runs a **daemon thread** with a `queue.Queue` task queue:

```python
def _run(self):
    while self.running:
        task = self.task_queue.get(timeout=0.1)  # blocks until task available
        self.is_busy = True
        self.execute_pick(task)
        self.is_busy = False
        self._set_phase('IDLE')
```

Shared state protected by `_pick_lock` (threading.Lock):
- `_pick_phase`: IDLE → APPROACH → DESCEND → MECHANICAL → IDLE
- `_pick_obj_id`: ID of object currently being picked
- `_live_pos`: dict fed by main loop every frame (belt_x, ws_y, height, timestamp)
- `_descend_approved`: flag from main loop signaling position is good

### Direct Predictive Pick Cycle (`execute_pick()`)

The current pick strategy is a **two-stage predictive pick** — no slow real-time
tracking approach. The robot predicts where the object WILL be and moves directly.

#### Phase 1: PREDICT (belt-Y forward prediction)

```
1. Set phase to APPROACH, record _pick_obj_id
2. Wait up to 150ms for main loop to feed fresh live position
3. HOVER PREDICTION:
   hover_predict_y = current_ws_y + (belt_speed × HOVER_TRAVEL_TIME_S)
   HOVER_TRAVEL_TIME_S = 0.40s (empirical robot travel time)
4. Bail if predicted Y ≥ workspace_depth - 0.5cm (too close to exit)
5. Convert to robot coordinates via bilinear interpolation
6. Compute hover_z = pick_z + HOVER_CLEARANCE_MM (10mm)
```

#### Phase 2: PICK (hover above, then corrected drop)

```
1. Set phase to DESCEND
2. Vacuum ON (M3) — suction builds during travel
3. Move to hover position (px, py, hover_z) at TRACK_DESCEND_SPEED
4. Wait for robot arrival: sleep(max(0.3, HOVER_TRAVEL_TIME_S))
5. CORRECTION: Read latest _live_pos (reject if age > 200ms)
6. Forward-predict for correction move:
   drop_ws_y = current_ws_y + (belt_speed × CORRECTION_MOVE_TIME_S)
   CORRECTION_MOVE_TIME_S = 0.10s (~100ms diagonal correction)
7. Move to corrected position at pick_z
8. Wait for correction move: sleep(CORRECTION_MOVE_TIME_S + 0.05)
```

#### Timing Report (logged per pick)

```
[TIMING] ID:5  total=0.623s  (wait=0.032 + hover=0.401 + drop=0.190)
         belt_drift=3.7cm @ 6.0cm/s  predicted=3.0cm  error=0.7cm
```

#### Phase 3: MECHANICAL (no more tracking needed)

```
1. Set phase to MECHANICAL
2. Optional latency compensation sleep (from offsets['latency'])
3. Suction settle: sleep(0.35s)
4. Lift to scan position (0, 0, -250) with scan_rotation
5. Spectrum scan (if sensor connected):
   a. sleep(0.1s) for stable position
   b. read_sensor() → 18 calibrated spectral channels
   c. predict() → CatBoost classification
   d. adaptive_fusion() → Bayesian combination with YOLO
6. Log to CSV + SQLite
7. Place: move to class-specific target, vacuum OFF (M5)
8. Release sequence: move_to(tx, ty, place_z) → vacuum OFF → 
   move_to(tx, ty, release_z) → move_to(tx, ty, place_z)
9. Return to standby (0, 0, -250)
10. Set phase to IDLE
```

### Pick Z Calculation

```python
pick_z = BASE_Z + (height_cm * 10) + global_z_offset + class_z_offset + bottom_offset
pick_z = clamp(pick_z, BASE_Z, -150)  # Safety: never below belt floor
```

| Component | Value | Description |
|-----------|-------|-------------|
| `BASE_Z` | −415 mm | Belt surface reference |
| `height_cm * 10` | varies | Raw measured object height → mm |
| `global_z_offset` | UI slider | User-adjustable Z offset |
| `class_z_offset` | per-class | From `CLASS_Z_OFFSET_MM` config |
| `STACK_BOTTOM_EXTRA_MM` | −10 mm | Extra depth for stack-bottom picks |
| `PICK_SURFACE_PENETRATION_MM` | 5 mm | Max depth below surface (safety clamp) |

**Per-class Z offsets:**

| Class | Z Offset (mm) | Reason |
|-------|--------------|--------|
| Glass | +5.0 | Fragile — stay higher |
| Metal | +10.0 | Hard surface — reduce pressure |
| Paper | −10.0 | Deflects down — push lower for grip |
| Plastic | 0.0 | Neutral default |

**Safety clamp:** pick_z is clamped to never go below `BASE_Z` (belt floor level),
preventing the gripper from pushing through the belt surface.

> **Note:** Per-class minimum height overrides (`CLASS_MIN_HEIGHT_CM`) have been removed.
> The system now relies entirely on the raw depth sensor measurement + per-class Z offsets.

### Vacuum Timing

- Vacuum turns ON **before** hover move (suction builds during travel)
- Settle time after drop: **0.35 seconds** at pick position
- Total vacuum-on-before-grip time: ~0.40s (hover travel) + 0.10s (correction) + 0.35s (settle) ≈ **0.85s**

### Robot Timing Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `HOVER_TRAVEL_TIME_S` | 0.40 s | `execute_pick()` | Empirical time for robot to reach hover position |
| `CORRECTION_MOVE_TIME_S` | 0.10 s | `execute_pick()` | Time for final diagonal correction move |
| `HOVER_CLEARANCE_MM` | 10 mm | `execute_pick()` | Height above pick_z for safe horizontal travel |
| `TRACK_DESCEND_SPEED` | 20000 | `config.py` | G-code feedrate during descent |
| `PICK_SURFACE_PENETRATION_MM` | 5 mm | `config.py` | Max depth below object surface |
| `STACK_BOTTOM_EXTRA_MM` | −10 mm | `_compute_pick_z()` | Extra depth for picking bottom of stack |

---

## 10. Position Mapping (Belt → Robot)

### The Challenge

Delta robots have **coupled kinematics** — the X and Y axes are not independent. Moving to belt position (10cm, 15cm) doesn't map to a simple (X_offset, Y_offset) in robot space. Both robot X and Y depend on both belt X and belt Y.

### Solution: 3×3 Bilinear Interpolation Grid

9 physical calibration points measured manually:

```
Belt Grid:  X = [0, 100, 200] mm    Y = [0, 100, 200] mm

ROBOT_X_GRID (mm):                  ROBOT_Y_GRID (mm):
[[-30,  -70, -100],                 [[ 80,   60,    0],
 [ 35,    0,  -35],                  [ 45,    0,  -45],
 [100,   80,   50]]                  [  0,  -50, -100]]
```

### Bilinear Interpolation

```python
def belt_to_robot_bilinear(belt_x_mm, belt_y_mm):
    # Find which grid cell the point falls in
    # Bilinear interpolate both ROBOT_X_GRID and ROBOT_Y_GRID
    # Apply X/Y/Z offsets from offsets.json
    # Clamp to ±110mm
    return (robot_x, robot_y)
```

### Offset Fine-Tuning (`offsets.json`)

| Offset | Default | Unit | Purpose |
|--------|---------|------|---------|
| X | 0.0 | mm | Lateral correction |
| Y | 0.0 | mm | Along-belt correction |
| Z | −20.0 | mm | Height correction |
| Latency | −0.7 | s | Belt movement compensation (negative = anticipate) |

---

## 11. Adaptive Bayesian Fusion

### Purpose

Combines YOLO vision classification with AS7265X spectrum classification for higher accuracy.

### Algorithm

```
1. CONTACT GATE: total spectral intensity < 220 → no object contact → use YOLO only

2. Build likelihoods from confusion matrices:
   P(YOLO_pred | true_class) from YOLO confusion matrix
   P(Spec_pred | true_class) from Spectrum confusion matrix

3. Compute dynamic weights based on:
   - Margin: top-2 probability gap (higher margin = more confident)
   - Entropy: normalized Shannon entropy (lower = more certain)
   - Agreement: do both sensors agree?

4. Bayesian posterior:
   P(class | YOLO, Spec) ∝ P(YOLO_pred|class) × P(Spec_pred|class) × P(class)

5. Return: best class, probabilities, fusion metadata
```

### Confusion Matrix Sizes

- YOLO: 4×4, ~258 samples per class
- Spectrum: 4×4, ~44 samples per class
- Laplace smoothing applied to avoid zero probabilities

---

## 12. Spectrum Sensor Integration

### AS7265X 18-Channel Sensor

| Channel | Wavelength | Die |
|---------|-----------|-----|
| A | 410 nm | UV |
| B | 435 nm | UV |
| C | 460 nm | UV |
| D | 485 nm | Vis |
| E | 510 nm | Vis |
| F | 535 nm | Vis |
| G | 560 nm | Vis |
| H | 585 nm | Vis |
| R | 610 nm | Vis |
| I | 645 nm | NIR |
| S | 680 nm | NIR |
| J | 705 nm | NIR |
| T | 730 nm | NIR |
| U | 760 nm | NIR |
| V | 810 nm | NIR |
| W | 860 nm | NIR |
| K | 900 nm | NIR |
| L | 940 nm | NIR |

### Read Sequence

```
1. Set gain to 64× (sensitive mode)
2. Set integration to 25 cycles
3. Turn on IR LED at 12.5mA
4. Take measurement
5. Read all 18 calibrated channel values
6. Turn off LED
7. Run CatBoost ML model → (class, confidence%, probabilities)
```

### ML Pipeline

- **Model**: CatBoost (loaded from `Model/` directory via joblib)
- **Preprocessing**: StandardScaler normalization
- **Labels**: LabelEncoder (Glass/Metal/Paper/Plastic)
- **Output**: Class prediction + per-class probability distribution

---

## 13. Configuration Reference

### `modules/config.py` — Complete Constants

#### Camera
| Constant | Value |
|----------|-------|
| `IMAGE_WIDTH` | 640 |
| `IMAGE_HEIGHT` | 480 |
| `FPS` | 30 |

#### YOLO
| Constant | Value |
|----------|-------|
| `MODEL_PATH` | `"yolov26s_fixed.pt"` |
| `CONFIDENCE_THRESHOLD` | 0.5 |

#### Belt / ROI
| Constant | Value | Unit |
|----------|-------|------|
| `ROI_HEIGHT_CM` | 30 | cm |
| `ROI_WIDTH_CM` | 20 | cm |
| `ENTRY_PATH_CM` | 21.5 | cm |
| `EXIT_PATH_CM` | 21.5 | cm |
| `CONVEYOR_SPEED_CM_S` | 6.0 | cm/s |
| `REGISTRATION_LINE_CM` | 15.0 | cm |

#### Robot Workspace
| Constant | Value | Unit |
|----------|-------|------|
| `ROBOT_WORKSPACE_OFFSET_CM` | 12.0 | cm |
| `ROBOT_WORKSPACE_DEPTH_CM` | 20.0 | cm |
| `ROBOT_WORKSPACE_WIDTH_CM` | 20.0 | cm |
| `MIN_PICK_WORKSPACE_Y_CM` | 3.0 | cm |

#### Robot Timing
| Constant | Value |
|----------|-------|
| `ROBOT_PICK_CYCLE_TIME_S` | 2.5 |
| `ROBOT_MOVE_TIME_S` | 0.5 |

#### Stacking Detection
| Constant | Value | Description |
|----------|-------|-------------|
| `STACK_MASK_DILATE_PX` | 20 | Mask dilation kernel radius |
| `STACK_MASK_IOU_THRESHOLD` | 0.02 | Minimum IoU for stack detection |
| `STACK_HEIGHT_DIFF_MIN_CM` | 2.0 | Height diff for physical stack |

#### Watershed Mask Joining
| Constant | Value | Description |
|----------|-------|-------------|
| `WATERSHED_JOIN_ENABLED` | `False` | Master toggle (disabled by default) |
| `WATERSHED_REQUIRE_SAME_CLASS` | `True` | Only join fragments of same class |
| `WATERSHED_MAX_GAP_PX` | 40 | Max dilated gap for proximity test |
| `WATERSHED_DEPTH_TOLERANCE_MM` | 20 | Max depth difference between fragments |
| `WATERSHED_MIN_FRAGMENT_PX` | 150 | Ignore fragments smaller than this |
| `WATERSHED_BOUNDARY_ALPHA` | 0.35 | Visual overlay alpha for join boundary |
| `WATERSHED_COLOR_SIM_ENABLED` | `True` | Enable HSV colour histogram gate |
| `WATERSHED_COLOR_SIM_THRESHOLD` | 0.45 | Min histogram correlation to pass |
| `WATERSHED_ASPECT_RATIO_TOL` | 3.0 | Max aspect ratio for fragment shape |
| `WATERSHED_DEBUG` | `True` | Print per-pair join decision log |

#### Mask Alpha Fading
| Constant | Value | Description |
|----------|-------|-------------|
| `MASK_FADE_ENABLED` | `True` | Master toggle |
| `MASK_FADE_IN_RATE` | 0.25 | Alpha increase per frame (detected) |
| `MASK_FADE_OUT_RATE` | 0.08 | Alpha decrease per frame (ghost) |
| `MASK_FADE_MIN_ALPHA` | 0.0 | Fully transparent |
| `MASK_FADE_MAX_ALPHA` | 1.0 | Fully opaque |

#### Per-Class Z Offsets
| Constant | Value | Description |
|----------|-------|-------------|
| `CLASS_Z_OFFSET_MM` | `{0: +5, 1: +10, 2: -10, 3: 0}` | Per-class pick Z adjustment (Glass/Metal/Paper/Plastic) |

> **Deprecated:** `CLASS_MIN_HEIGHT_CM` and `DEFAULT_MIN_HEIGHT_CM` are still defined in config.py
> but are **no longer imported or used** by robot.py or index.py. Raw sensor height is used directly.

#### Priority Weights
| Weight | Value |
|--------|-------|
| `urgency` | 1.0 |
| `height` | 0.3 |
| `isolation` | 0.5 |
| `stack_risk` | 0.8 |

#### Direct Predictive Pick (in `execute_pick()`)
| Constant | Value |
|----------|-------|
| `HOVER_TRAVEL_TIME_S` | 0.40 s |
| `CORRECTION_MOVE_TIME_S` | 0.10 s |
| `HOVER_CLEARANCE_MM` | 10 mm |
| `TRACK_DESCEND_SPEED` | 20000 |
| `PICK_SURFACE_PENETRATION_MM` | 5 mm |
| `STACK_BOTTOM_EXTRA_MM` | −10 mm |

#### Robot Hardware
| Constant | Value |
|----------|-------|
| `DELTA_PORT` | `/dev/ttyACM0` |
| `SLIDER_PORT` | `/dev/ttyUSB1` |
| `BASE_Z` | −415.0 mm |
| `STANDBY_POS` | (0, 0, −250) mm |

---

## 14. Data Logging

### CSV Logger (`DetectionLogger`)

30-column CSV files saved to `detection_logs/`:

| Columns 1–12 | Classification |
|---------------|---------------|
| ID, Camera_{Glass,Plastic,Metal,Paper,Class}, Spectrum_{Glass,Plastic,Metal,Paper,Class}, Final_Class |

| Columns 13–30 | Spectral Raw Values |
|----------------|---------------------|
| One column per AS7265X channel (410nm through 940nm) |

Also saves cropped object images per detection.

### SQLite Database (`DatabaseLogger`)

Table columns: `session_id`, `object_id`, `final_class`, `vision_class`, `spectrum_class`, `height_cm`, `belt_x_cm`, `belt_y_cm`, `status`, `timestamp`, + 18 spectral channels.

---

## 15. User Interface (DearPyGui)

### 5-Tab Layout (1920×1080 viewport)

#### Tab 1: Vision (Main Sorting View)

**Left column (660px):**
- Video feed (640×480) with YOLO detection overlay
- Tracking dashboard table (ID, Class, X, Y, Status — up to 10 objects)
- System log (30 messages, color-coded)

**Right column (600px):**
- Control buttons: START, STOP, AUTO PICK, CALIBRATE, RELOAD CAL, DUMMY OBJ, TRACK
  - Buttons sized ~20% larger for touch-screen use
- Robot connection: port fields, CONNECT, DISCONNECT, HOME
- Live parameters: Belt Speed, YOLO Confidence, Approach Time
- **Detection Pipeline toggles:** 3 checkboxes for Depth Clustering / Cross-Class NMS / Watershed Joining
  - Pipeline status label: `Pipeline: DC > NMS > WS` or `Pipeline: RAW (no post-processing)`
  - Also drawn on the video feed overlay (bottom-left corner)
- Offsets: X, Y, Z, Latency in 2×2 grid (240px wide inputs), Save button inline
- **Spectrum prediction display** (compact 2-line layout):
  - Line 1: YOLO class + confidence, Spectrum class + confidence
  - Line 2: Final fused class (color-coded), sensor status
- Status indicators: [X] red / [V] green / [~] yellow (ASCII, compatible with default font)
- Robot Workspace Simulation (500×700 drawlist, 2D top-down view):
  - Full ROI display from Y=0 (detection zone + registration line + ROI + gap + workspace)
  - Detection zone (0–15cm) with dimmed pre-queue object markers
  - Dashed yellow registration line at 15cm
  - Workspace divided into TOP/MID/BOT sections with colored fills
  - Objects labeled: `ID:Class elapsed_time` (e.g., "3:Plastic 2.1s")
  - Red robot dot with trail + white crosshair (smoothed position)
  - Green target crosshair showing pick dispatch target with distance line
  - TRACK mode: orange ring around tracked object
- Sorting statistics per class

#### Tab 2: Robot Control

Side-by-side layout: Robot camera feed on the left (~710 px, ≥10 % larger than the
main 640×480 vision feed), all control panels on the right.

- **Left column:** Robot USB camera live preview (640×480), camera status + enable/reconnect buttons, current position readout (X, Y, Z, R)
- **Right column, top row:** Manual jog controls (X±, Y±, Z±, R± with step size) + Direct position input (X, Y, Z, R, Speed → GO), vacuum ON/OFF, preset positions, pick test, smooth move demos
- **Right column, bottom row:** Place positions editor (per-class XY, Z heights, apply/save/load/reset) + Workspace grid (3×3 belt→robot coordinate mapping, apply/save/go-to/set-from-bot)

#### Tab 3: Spectrum Scan

- Manual scan button + prediction display
- LED control (IR, White, UV toggles)
- Raw 18-channel spectral data display

#### Tab 4: Video & Recording

- .bag file loader with play/stop/unload controls
- .bag recording toggle (records live RealSense streams)
- Experiment recording (.mp4 screen capture + linked robot USB camera)
- Robot camera preview and status
- Playback loops automatically for repeated testing

#### Tab 5: Calibration (Integrated)

- Full calibration workflow integrated from standalone `calibration_tool.py`
- 5-step guided process: Enable → Flip/Orient → Confirm ROI → Floor Calibration → Save
- Uses the **existing camera feed** (no second RealSense pipeline)
- **ROI zones and YOLO detection are bypassed** during calibration for a clean checkerboard view
- Real-time checkerboard overlay with orientation preview
- RANSAC floor plane fitting + floor depth map generation
- Saves to `calibration_data.json` and immediately applies via `detector.load_calibration()`
- See [Section 28: Integrated Calibration](#28-integrated-calibration-tab-5) for full details

#### Tracking Dashboard

The dashboard table (below the video feed on Tab 1) shows up to 10 tracked objects:

| Column | Description |
|--------|-------------|
| **ID** | Unique tracking ID |
| **Class** | Detected material class |
| **Conf** | YOLO confidence (0.0–1.0) |
| **Belt X** | Position across belt width (cm) |
| **Belt Y** | Position along belt travel (cm) |
| **H(cm)** | Object height above belt surface — color-coded: 🔴 <1cm, 🟡 1–2cm, 🟢 ≥2cm |
| **Status** | Current state: Tracking, Queued, In Workspace, Picking, Unreachable |

---

## 16. Manual Inspection Tool

### `Manual_Inspection.py` (~1344 lines)

A standalone DearPyGui application for manual waste object inspection, labeling, and model evaluation.

### Features

| Feature | Description |
|---------|-------------|
| **Capture** | Captures current camera frame + YOLO detection |
| **Spectrum Scan** | Reads AS7265X sensor, runs CatBoost prediction |
| **Voting/Fusion** | Combines YOLO + Spectrum predictions with voting logic |
| **CSV Logging** | Logs each inspected object (camera class, spectrum class, final class, real class, raw spectrum) |
| **Label Tool** | Lets user assign ground-truth `Real_Class` label to each logged entry |
| **Confusion Matrix** | Compares `Final_Class` vs `Real_Class` across all logged entries, shows per-class precision/recall/F1 |
| **Clear Queue** | Resets inspection queue |

### Use Case

Used for offline model evaluation: inspect objects one-by-one, record predictions from both sensors, assign ground-truth labels, then generate a confusion matrix to measure system accuracy.

---

## 17. Watershed Mask Joining

### Problem

When an object is partially occluded (e.g., by another object stacked on top), YOLO may
produce **two separate mask fragments** for the bottom object instead of one complete mask.

### Solution: `watershed_join_masks()` in `detector.py`

After YOLO detection and cross-class mask NMS, the watershed algorithm attempts to
rejoin fragments that belong to the same physical object.

### Join Criteria (5 Gates)

All enabled gates must pass for two fragments to be joined:

| Gate | Check | Config Constant |
|------|-------|----------------|
| 1. Same class | Fragments must have same YOLO class_id | `WATERSHED_REQUIRE_SAME_CLASS = True` |
| 2. Similar depth | Depth difference ≤ 20mm | `WATERSHED_DEPTH_TOLERANCE_MM = 20` |
| 3. Spatial proximity | Dilated masks overlap (gap ≤ 40px) | `WATERSHED_MAX_GAP_PX = 40` |
| 4. Colour similarity | HSV histogram correlation ≥ 0.45 | `WATERSHED_COLOR_SIM_THRESHOLD = 0.45` |
| 5. Aspect ratio | Shape ratio ≤ 3.0 | `WATERSHED_ASPECT_RATIO_TOL = 3.0` |

### Algorithm

```
1. Union-find grouping of fragments passing all 5 gates
2. For each group with >1 member:
   a. Combine masks into union region
   b. Create eroded markers (one label per original fragment)
   c. Dilate union to create watershed fill region
   d. Run OpenCV cv2.watershed() on the colour image
   e. Build joined mask from positive marker regions
   f. Store boundary mask (marker == -1) for visual overlay
   g. Recalculate centroid, contour, angle, depth, belt coords
3. Merged detection inherits highest-confidence member's class
```

### Design Note

Watershed **only joins same-class fragments** (`WATERSHED_REQUIRE_SAME_CLASS = True`).
Cross-class duplicate resolution is handled separately at registration time by
[IoU Class Resolution](#19-iou-class-resolution-at-registration). This keeps
responsibilities cleanly separated.

### Visual Feedback

- Watershed boundaries drawn as semi-transparent white grid lines (alpha = 0.35)
- `WATERSHED_DEBUG = True` prints detailed per-pair join decisions to console

---

## 18. Mask Alpha Fading

### Problem

When YOLO detection flickers (object detected one frame, lost the next), masks
appear and disappear abruptly, causing visual noise.

### Solution: Per-Object Alpha Fading

Each tracked object carries a `mask_alpha` value (0.0 to 1.0) that changes smoothly:

| State | Behaviour | Rate |
|-------|-----------|------|
| Detected | Alpha increases toward 1.0 | +0.25/frame |
| Ghost (undetected) | Alpha decreases toward 0.0 | −0.08/frame |

### Implementation

- **`tracker.py`**: Stores `mask_alpha`, `last_valid_mask`, `watershed_boundary` per object.
  Fade-in on match, fade-out on ghost transition.
- **`index.py`**: `_draw_tracked_masks()` renders from tracked objects (not raw detections),
  applying per-object alpha. Ghost objects use `last_valid_mask` for fading out.

### Config Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MASK_FADE_ENABLED` | `True` | Master toggle |
| `MASK_FADE_IN_RATE` | 0.25 | Alpha increase per frame when detected |
| `MASK_FADE_OUT_RATE` | 0.08 | Alpha decrease per frame when ghost |
| `MASK_FADE_MIN_ALPHA` | 0.0 | Minimum alpha (fully transparent) |
| `MASK_FADE_MAX_ALPHA` | 1.0 | Maximum alpha (fully opaque) |

---

## 19. IoU Class Resolution at Registration

### Problem

YOLO sometimes assigns **multiple class labels** to the same physical object (e.g., a bottle
detected as both "Glass" and "Plastic" with overlapping masks). Without resolution, both
would be queued as separate pick targets.

### Solution: `_resolve_class_at_registration()` in `tracker.py`

Called during `check_registration_crossing()` Phase 1b, before any objects are queued.

### Algorithm

```
1. Compute composite score for each crossing object:
      score = normalized_mask_area × 0.6  +  confidence × 0.4
   (larger mask = better object resolution = more likely correct class)

2. Sort by score descending (best first)

3. For each pair (i, j) where i has higher score:
   a. Compute mask IoU between objects i and j
   b. If IoU ≥ DUPLICATE_MASK_IOU_THRESHOLD (0.40):
      - Object j is the same physical object as i
      - Absorb j: mark status='Absorbed', transfer height if taller
      - Keep i: inherits the correct class (best object resolution)

4. Return filtered list of crossing IDs (duplicates removed)
```

### Separation from Watershed

| Feature | Watershed | IoU Class Resolution |
|---------|-----------|---------------------|
| **When** | During detection (detector.py) | At registration line (tracker.py) |
| **What** | Rejoins same-class mask fragments | Resolves different-class duplicates |
| **Class rule** | Same class required | Cross-class comparison |
| **Output** | Single merged mask | Single winning object (losers absorbed) |

### Debug Output

```
[REG-IoU] #5 (Metal 3.2x4.1cm score=0.55) overlaps #3 (Glass 8.5x9.2cm score=0.82)
         IoU=0.52 → absorb #5, keep #3
```

---

## 20. Hardware Disconnect

### DISCONNECT Button (`index.py`)

Added alongside CONNECT and HOME in the Robot Connection UI panel.

### `_on_disconnect()` Behaviour

```
1. Stop system if running (calls _on_stop())
2. Delta robot:
   a. Turn off vacuum (safety: M5 command)
   b. Close serial connection
3. Slider: close serial connection
4. Spectrum sensor:
   a. Disable all LEDs (IR, White, UV)
   b. Close I2C bus (SMBus)
   c. Reset hardware_ready and is_ready flags
5. Camera: stop RealSense pipeline and thread
6. Update UI status text to "Disconnected" (orange)
```

### Spectrum `disconnect_hardware()` (`modules/spectrum.py`)

New method that safely shuts down the AS7265X sensor:
- Disables all LED bulbs before closing
- Closes the SMBus I2C connection
- Resets `sensor`, `hardware_ready`, `is_ready` to None/False

Each component checks its connection state before disconnecting, so pressing
DISCONNECT when nothing is connected simply shows "Nothing connected".

---

## 21. Detection Experiment Tool

### `detection_experiment.py` (~960 lines)

A standalone DearPyGui application for **comparing YOLO models and separation logic
settings** on a live RealSense camera feed. Useful for tuning detection parameters
without running the full sorting pipeline.

### Features

| Feature | Description |
|---------|-------------|
| **Model hot-swap** | Combo box to switch between any `.pt` model in the project directory at runtime |
| **Separation logic toggles** | Enable/disable Depth Clustering, Cross-Class NMS, Watershed Joining independently |
| **Confidence slider** | Adjust YOLO confidence threshold live (0.1–1.0) |
| **Live stats** | FPS, inference time (ms), detection count, per-class breakdown |
| **Stacking analysis** | Pairwise dilated-mask IoU analysis with TOP/BOT/CLOSE labels |
| **Connecting lines** | Red lines for stacking pairs, orange for adjacent pairs, IoU % at midpoint |
| **Tunable sliders** | Dilate px (1–50), Adjacent IoU threshold, Stacking IoU threshold |
| **Pipeline status bar** | Shows active stages: `Pipeline: DC > NMS > WS > STACK` |
| **Screenshot** | Save current annotated frame as PNG |
| **Freeze** | Pause/resume video feed for inspection |
| **Depth map** | Toggle depth visualization overlay |

### Stacking/Adjacent Detection (Live)

The experiment tool computes pairwise mask overlap for all detected objects every frame:

```
1. Dilate each object's binary mask by configurable kernel size
2. Compute IoU for every pair of detections
3. Classify each pair:
   - IoU ≥ stacking_threshold (default 10%) → STACKING (red line)
     Taller object labeled "TOP", shorter labeled "BOT"
   - IoU ≥ adjacent_threshold (default 2%) → ADJACENT (orange line)
     Both objects labeled "CLOSE"
   - IoU < adjacent_threshold → no relationship
4. Draw connecting lines + IoU% at midpoint
5. Update live stats: stacking groups, stacking pairs, adjacent pairs
```

### Usage

```bash
python detection_experiment.py
```

Requires RealSense camera connected. Uses the same `modules/detector.py` pipeline
as the main application but without tracker, robot, or spectrum integration.

---

## 22. Display Lag Fix (Smoothing Removal)

### Problem

The main pipeline display showed a visible **lag between mask outlines and centroid/box
positions** — masks appeared to be "behind" where the tracking markers were drawn.

### Root Cause Analysis (3 contributing factors)

1. **Async ThreadedDetector (1–2 frame delay)**
   `ThreadedDetector` runs YOLO in a background thread. Detection results from frame N−1
   are drawn on frame N. This is a ~33ms delay at 30fps — acceptable and uniform across
   all display elements.

2. **Centroid EMA Smoothing (α=0.7) — 4–5mm steady-state lag** ⚠️
   The tracker applied exponential moving average to centroids. At belt speed 6 cm/s and
   30fps, each detection centroid moves ~2mm/frame. EMA with α=0.7 creates a steady-state
   lag of ~4–5mm — visible as the X marker trailing behind the mask edge.

3. **Raw mask vs smoothed centroid mismatch** ⚠️
   Masks are always drawn from the raw detection (no smoothing possible on binary masks),
   but centroids/boxes were smoothed. This created a visual disconnect — the mask shows
   where the object IS, while the marker shows where it WAS.

### Fix Applied

In `modules/tracker.py`:

```python
centroid_smoothing = 0.0   # was 0.7
box_smoothing = 0.0        # was 0.75
```

Setting both to 0.0 means centroids and boxes update instantly from detection data,
matching the raw mask positions exactly. The remaining 1-frame async delay from
ThreadedDetector is uniform across mask/centroid/box — no visible misalignment.

### Impact

- **No functional impact**: Belt coordinates come from detection data, not smoothed centroids.
  All pick calculations, time-anchoring, and queue logic are unaffected.
- **Visual improvement**: Masks, centroids, and bounding boxes now align perfectly.
- The detection experiment tool (`detection_experiment.py`) never had this issue because
  it runs YOLO synchronously — results are drawn on the same frame they were computed for.

---

## 23. Robot Freeze Root Cause Analysis

A systematic investigation of all code paths that can cause the delta robot to
**appear frozen** (stuck at a position, unresponsive to new commands). Findings are
ranked by severity.

### 🔴 CRITICAL Issues

#### 23.1 No Serial Lock — Concurrent Serial Writes

**Location:** `SerialController.send()` (`robot.py:355`)

`send()` has **no threading lock**. The robot thread calls `delta.move_to()` during
`execute_pick()`, while the main DPG thread can simultaneously call `delta.move_to()` from:

| Caller | Thread | When |
|--------|--------|------|
| `_jog_robot()` | Main (DPG) | User presses jog button |
| `_go_to_position()` | Main (DPG) | User clicks GO |
| `_go_place_pos()` | Main (DPG) | User tests place position |
| `_go_to_grid_pos()` | Main (DPG) | User tests grid position |
| `_go_preset()` | Main (DPG) | User clicks preset button |
| `_process_tracking()` | Main (DPG) | TRACK mode — sends every frame! |
| `delta.set_vacuum()` | Main (DPG) | Vacuum toggle buttons |
| `_pick_test()` | Spawned thread | Pick test (4 corners) |
| `_demo_smooth_*()` | Main (DPG) | Demo functions |
| `execute_pick()` | Robot thread | During auto-pick cycle |

**Freeze mechanism:** Two threads calling `ser.write()` simultaneously → interleaved
bytes → corrupt G-code → firmware receives garbage → firmware enters unknown state
or ignores all subsequent commands → **robot appears frozen**.

**Fix:** Add a `threading.Lock` to `SerialController.send()`.

#### 23.2 Spectrum Sensor I2C Block During Pick

**Location:** `execute_pick()` → `self.spectrum.read_sensor()` (`robot.py:944`)

`read_sensor()` calls:
- `setGain()`, `setIntegrationCycles()` — each does virtual register read/write
- `enableBulb()` / `disableBulb()` — virtual register writes
- `takeMeasurements()` — polling loop (up to `maxWaitTime` ms)
- `getCalibratedA()` through `getCalibratedL()` — **18 calibrated reads**, each doing
  **4 × `virtualReadRegister()`** = **72 virtual register reads**

Each `virtualReadRegister()` has a polling loop with timeout = `maxWaitTime`.
With integration cycle 25: `maxWaitTime = int(25 × 2.8 × 1.5) + 1 = 106ms`.

**Worst case:** If I2C bus glitches, every virtual register read blocks for 106ms.
72 reads × 106ms = **up to 7.6 seconds blocking** on the robot thread.

During this time the robot is stuck in MECHANICAL phase with vacuum ON, unable to
proceed to the place sequence. The system appears "frozen."

**Fix:** Wrap spectrum read in a timeout thread or add I2C health check.

#### 23.3 No Timeout on `execute_pick()` — Entire Cycle Can Block Forever

**Location:** `_run()` loop (`robot.py:610`)

The `_run()` loop calls `self.execute_pick(task)` with **no timeout wrapper**.
If any sub-step hangs (I2C blocks, serial hangs), the robot thread blocks forever.

**Consequences:**
- `is_idle()` never returns True → no new picks dispatched
- Robot physically stuck at last commanded position
- No watchdog, no abort mechanism, no recovery

**Fix:** Add a watchdog timer that kills the pick cycle after N seconds.

### 🟠 HIGH Issues

#### 23.4 `_on_stop()` Does NOT Stop the Robot Thread

**Location:** `index.py:1910`

When the user presses STOP, `_on_stop()` sets `self.is_running = False` and
`self.auto_pick = False`, but **never calls `self.robot_manager.stop()`**. The robot
thread keeps running. If a pick is in progress, it continues the full cycle
(hover → pick → spectrum scan → place → standby).

The user thinks the system is stopped and may try to jog the robot → both threads
send serial commands → corrupt G-code → freeze (see §23.1).

**Fix:** Call `self.robot_manager.stop()` in `_on_stop()`, and have `execute_pick()`
check `self.running` at each phase transition.

#### 23.5 TRACK Mode + AUTO PICK Serial Race

**Location:** `_process_tracking()` (`index.py:2571`) + `execute_pick()` (`robot.py`)

Although the UI prevents enabling both simultaneously, rapid mode toggling can create
a brief window where TRACK mode sends `delta.move_to()` every frame from the main
thread while the robot thread's `execute_pick()` is still mid-cycle.

**Fix:** Hard mutex — check `robot_manager.is_busy` before sending TRACK commands.

#### 23.6 `_pick_test()` Spawns Unprotected Thread

**Location:** `index.py:1636`

`_pick_test()` spawns a new daemon thread that calls `delta.move_to()` and
`delta.set_vacuum()` directly. If AUTO PICK is active, both the pick test thread
AND the robot manager thread send serial commands simultaneously.

**Fix:** Check `is_busy` before starting pick test, or use the serial lock.

### 🟡 MEDIUM Issues

#### 23.7 Serial `send()` Returns `False` Silently

**Location:** `SerialController.send()` (`robot.py:355`)

If `ser.write()` throws an exception (e.g., USB disconnect), `send()` returns `False`,
but **no caller checks the return value**. All subsequent commands silently fail.
The robot is stuck at its last position with vacuum potentially stuck ON.

**Fix:** Check return value; set `self.connected = False` on serial exception.

#### 23.8 Fire-and-Forget Buffer Overflow

**Location:** All `DeltaController` methods

The firmware receives G-code but there's no "ok" response check. TRACK mode sends
up to 30 G-codes/second (one per frame). If the firmware's serial input buffer
overflows, commands get silently dropped and the robot stops mid-motion.

**Fix:** Add flow control (wait for "ok") or rate-limit commands.

#### 23.9 Disconnect During Active Pick

**Location:** `_on_disconnect()` in `index.py`

`_on_disconnect()` closes the serial port while the robot thread may still be
writing to it. This causes `serial.SerialException` in the robot thread.

**Fix:** Stop the robot thread before closing the serial port.

### 🟢 LOW Issues

#### 23.10 Arbitrary Latency Sleep

`time.sleep(self.offsets['latency'])` in `execute_pick()` is user-controlled. A very
high value blocks the robot thread for that duration during every pick cycle.

**Fix:** Clamp the latency offset to a reasonable range (0–2 seconds).

#### 23.11 `smooth_move_to_pick()` / `test_grid_positions()` Blocking Loops

Both contain `time.sleep()` loops with no `self.running` escape check. Not currently
in the production pick flow but callable from UI/code.

**Fix:** Add `if not self.running: break` inside loops.

### Summary: Freeze Likelihood by Phase

| Pick Phase | Most Likely Freeze Cause |
|------------|-------------------------|
| **Moving to hover** | Serial collision (§23.1) if user presses buttons during pick |
| **Hovering / correction** | Serial collision, or belt exits workspace (handled — returns False) |
| **At pick position** | I2C spectrum block (§23.2) if sensor unresponsive |
| **Moving to place** | Serial collision if user toggles vacuum button |
| **Returning to standby** | Serial collision |
| **Between picks** | `_on_stop()` not stopping thread (§23.4) → user jogs → collision |

---

## 24. Known Threading & Safety Issues

### Thread Map

| Thread | Created By | Purpose | Shares Serial? |
|--------|-----------|---------|---------------|
| **Main (DPG)** | Application start | UI, camera, detection dispatch, drawing | Yes — jog, preset, TRACK mode |
| **Robot Manager** | `start_manager()` | Pick cycle execution | Yes — `execute_pick()` |
| **ThreadedDetector** | `start_detection()` | YOLO inference | No |
| **CameraStream** | `CameraStream.__init__` | RealSense frame capture | No |
| **Pick Test** | `_pick_test()` | 4-corner pick test (temporary) | Yes — `delta.move_to()` |
| **Manual Spectrum** | `_manual_spectrum_scan()` | I2C sensor read (temporary) | No (I2C, not serial) |

### Shared Resources Without Protection

| Resource | Accessed By | Protection | Risk |
|----------|------------|------------|------|
| `delta.ser` (serial port) | Main + Robot + Pick Test threads | **None** | 🔴 Corrupt G-code |
| `self.robot_pos` (dict) | Main (read/write) + Robot (write via callback) | **None** | 🟡 Display glitch |
| `spectrum.sensor` (I2C) | Robot thread (during pick) + Manual scan thread | **None** (but double-tap guard exists) | 🟡 I2C collision |
| `_live_pos` (dict) | Main (write) + Robot (read) | `_pick_lock` ✅ | Safe |
| `_pick_phase` / `_pick_obj_id` | Main (read) + Robot (write) | `_pick_lock` ✅ | Safe |
| `task_queue` | Main (put) + Robot (get) | `queue.Queue` ✅ | Safe |

### Safety Recommendations

1. **Add `threading.Lock` to `SerialController`** — all serial writes go through a single lock
2. **Add watchdog timer** — if `execute_pick()` exceeds 10 seconds, force-abort and return to standby
3. **Stop robot thread on STOP button** — call `robot_manager.stop()` in `_on_stop()`
4. **Check `is_busy` before manual commands** — jog/preset/go buttons should warn if robot is mid-pick
5. **Wrap spectrum I2C in timeout** — use `threading.Timer` or `signal.alarm` to abort hung reads
6. **Clamp latency offset** — limit to 0–2 seconds in UI slider range

---

## 25. TensorRT (.engine) Support

### 25.1 Overview

YOLO models can be exported from PyTorch `.pt` format to NVIDIA TensorRT `.engine`
format for ~2× faster inference on Jetson hardware. This is achieved via FP16
(half-precision) quantisation, which halves memory bandwidth with negligible
accuracy loss for object detection/segmentation.

**Benchmark (Jetson AGX Orin, 640×640, FP16):**

| Model           | Format       | Avg (ms) | FPS  | Speedup |
|-----------------|-------------|----------|------|---------|
| yolov8s         | PyTorch .pt  | 31.6     | 32   | —       |
| yolov8s         | TensorRT FP16| 16.9     | 59   | 1.87×   |

### 25.2 Export Tool

A standalone export tool is provided: **`export_tensorrt.py`**

```bash
# Interactive DearPyGui tool
python export_tensorrt.py

# CLI: export a single model
python export_tensorrt.py yolov8s.pt

# CLI: export all .pt models missing .engine counterparts
python export_tensorrt.py --all

# CLI: re-export even if .engine already exists
python export_tensorrt.py --all --force

# CLI: benchmark all available models (.pt and .engine)
python export_tensorrt.py --benchmark
```

Export takes **5–10 minutes per model** on Jetson AGX Orin. The `.engine` file is
saved alongside the `.pt` with the same stem name.

### 25.3 Code Integration

#### config.py — Default MODEL_PATH

```python
MODEL_PATH = "yolov26s_fixed.pt"
```

The default model is `yolov26s_fixed.pt` (extra dataset training). If a corresponding
`.engine` file exists, the model discovery function prefers it for faster inference. All `.pt` and `.engine`
files in the project directory are auto-discovered and selectable from the UI.

#### config.py — discover_models()

The model discovery function groups files by stem and **prefers `.engine`** over `.pt`
when both exist. Both formats remain selectable in the UI combo box. Engine files
are sorted first.

#### detector.py — task='segment' for .engine

TensorRT `.engine` files cannot auto-detect the YOLO task type. The `load_model()`
and `switch_model()` methods detect `.engine` extension and pass `task='segment'`
explicitly:

```python
if self.model_path.endswith('.engine'):
    self.model = YOLO(self.model_path, task='segment')
else:
    self.model = YOLO(self.model_path)
```

**Without `task='segment'`, masks will always be `None`** even when the engine was
built from a segmentation model. This is a critical Ultralytics behaviour.

### 25.4 Important Notes

1. **Device-specific**: `.engine` files are compiled for the specific GPU. If you
   change hardware (e.g., from Jetson Orin to desktop GPU), you must re-export.
2. **CUDA compute capability**: The Jetson AGX Orin is compute 8.7. Engines built
   for different compute capabilities will fail to load.
3. **onnxslim/onnxruntime-gpu warnings**: The export logs may show warnings about
   missing `onnxslim` or `onnxruntime-gpu`. These are non-critical — the ONNX→TRT
   conversion path works without them.
4. **Model task**: Only `SegmentationModel` (task='segment') models should be
   exported. The export tool validates this and skips non-segmentation models.
5. **Warmup**: First inference after loading is always slower due to CUDA context
   initialisation. Both `load_model()` and `switch_model()` perform a warmup pass.

---

## 26. Synchronous Detection Refactor

### Problem

The original pipeline used `ThreadedDetector` — a background thread wrapper around
`ObjectDetector.detect()`. YOLO inference ran on frame N while the main thread displayed
frame N+1, causing a **1–2 frame detection lag**. This manifested as:

- Masks appearing slightly behind actual object positions
- Ghost detections lingering when objects moved quickly
- Belt-Y prediction errors amplified by the async offset

### Why ThreadedDetector Existed

`ThreadedDetector` was created during the Tkinter era of the application. Tkinter's
mainloop freezes on any blocking call, so YOLO inference (30–80ms) had to run in a
background thread. DearPyGui does **not** have this limitation — it renders
independently of user callbacks.

### Fix Applied (March 2026)

Removed `ThreadedDetector` usage from `index.py`. YOLO now runs **synchronously**
in `_update_frame()`:

```python
# OLD: async (1-2 frame lag)
self.threaded_detector.set_frame(color_image, depth_frame)
detections = self.threaded_detector.get_detections()  # stale!

# NEW: synchronous (zero lag)
detections = self.detector.detect(color_image, depth_frame)
```

### Impact

- **Zero detection lag** — masks and detections match the displayed frame exactly
- **No threading complexity** — no race conditions between detection and display
- **~30–80ms per frame** — acceptable for 30fps pipeline on Jetson
- `ThreadedDetector` class still exists in `detector.py` but is unused

---

## 27. Mask Temporal Smoothing

### Problem

YOLO segmentation masks flicker frame-to-frame — mask boundaries jitter even for
stationary objects, causing visual noise and unstable contour edges.

### Solution: 3-Layer Smoothing Pipeline

Applied in `SimpleTracker.update()` when a tracked object receives a new detection:

#### Layer 1: Temporal EMA Blending

```python
smoothed = α × new_mask + (1 − α) × old_mask
```

- `MASK_TEMPORAL_SMOOTH = 0.5` — 50/50 blend between current and previous mask
- Prevents single-frame mask shape spikes
- Applied via floating-point accumulation, then thresholded back to binary

#### Layer 2: Morphological Polish

```python
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
```

- `MASK_MORPH_SMOOTH_PX = 5` — 11×11 elliptical kernel
- Close fills small holes inside the mask
- Open removes small noise speckles outside the mask

#### Layer 3: Contour Smoothing

```python
contours = cv2.findContours(mask)
approx = cv2.approxPolyDP(contour, 0.5% of perimeter)
```

- Keeps only the largest contour (removes fragments)
- `approxPolyDP` with 0.5% epsilon simplifies jagged edges
- Redraws the mask from the smoothed contour

### Mask Shift Correction (warpAffine)

When the belt moves between frames, the old mask is in a stale position. Before
blending, the previous mask is **shifted** by the belt movement delta using
`cv2.warpAffine()`, aligning it with the new detection position before EMA averaging.

### Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `MASK_TEMPORAL_SMOOTH` | 0.5 | EMA blend factor (0=keep old, 1=use new only) |
| `MASK_MORPH_SMOOTH_PX` | 5 | Morphological kernel radius |
| `ROI_MIN_MASK_COVERAGE` | 0.85 | Minimum mask coverage within ROI to accept detection |

---

## 28. Integrated Calibration (Tab 5)

### Background

Previously, camera calibration required running a separate standalone tool:

```bash
python calibration_tool.py
```

This opened its own RealSense pipeline and OpenCV window, creating potential
**camera device conflicts** if the main application was also running. The calibration
button in `index.py` was just a placeholder that printed a message.

### Integration (March 2026)

The full calibration workflow from `calibration_tool.py` (938 lines) has been
integrated into `index.py` as **Tab 5: Calibration**. The standalone tool remains
as a backup but is no longer needed for normal operation.

### Key Design Decisions

1. **Same camera instance** — uses `self.camera` (the existing RealSense feed),
   avoiding device conflicts
2. **ROI bypass during calibration** — when `cal_mode` is active, `_update_frame()`
   skips YOLO detection and ROI zone drawing, providing a clean raw camera feed
   for checkerboard detection
3. **Immediate apply** — after saving, `detector.load_calibration()` is called
   immediately so the new calibration takes effect without restarting

### 5-Step Guided Workflow

| Step | Action | What Happens |
|------|--------|--------------|
| 1 | **Enable** | Activates calibration overlay; resets all state |
| 2 | **Flip / Orient** | Toggle V-Flip / H-Flip to match belt direction |
| 3 | **Confirm** | Locks ROI corners + computes homography matrix |
| 4 | **Floor** | RANSAC floor plane fitting + depth map from current depth frame |
| 5 | **Save** | Writes `calibration_data.json` + applies to detector immediately |

### Checkerboard Parameters

| Parameter | Value |
|-----------|-------|
| Squares X × Y | 8 × 10 |
| Inner corners | 7 × 9 |
| Square size | 2.5 cm |
| Target ROI | 20 × 30 cm |
| Entry/Exit zones | 21.5 cm each |

### ROI and Detection Bypass

During calibration (`cal_mode == True`):

- **YOLO detection is skipped** — `is_tracking and not cal_mode` guard
- **ROI zone overlays are not drawn** — `_draw_roi_zones()` only called when `cal_mode` is False
- **Tracked masks and tracking markers are not drawn** — no visual clutter
- **Calibration overlay is drawn instead** — checkerboard corners, ROI preview,
  entry/exit zones, orientation labels

This ensures the checkerboard is visible without any existing ROI zones or
detection overlays interfering.

### Floor Calibration

- Collects 3D points from the depth frame using `rs2_deproject_pixel_to_point()`
- **RANSAC plane fitting**: 100 iterations, 1 cm inlier threshold
- Floor depth map: downsampled 4×, median blur + morphological close for hole filling
- Plane equation stored as `(a, b, c, d)` coefficients

### Calibration Output (`calibration_data.json`)

```json
{
  "roi": { "corners_px": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] },
  "transforms": {
    "homography": [[3×3 matrix]],
    "homography_inv": [[3×3 matrix]]
  },
  "floor_plane": { "coefficients": [a, b, c, d] },
  "floor_depth_map": [[downsampled depth values]],
  "zones": {
    "entry_corners_px": [[4 corners]],
    "exit_corners_px": [[4 corners]]
  },
  "camera": { "intrinsics": { "fx", "fy", "ppx", "ppy" } }
}
```

### Methods Added to SortingApp

| Method | Purpose |
|--------|---------|
| `_cal_log()` | Log to calibration panel + main log |
| `_cal_toggle()` | Enable/disable calibration mode |
| `_cal_flip_v()` / `_cal_flip_h()` | Toggle orientation flips |
| `_cal_get_oriented_corners()` | Apply flips to 4 outer corners |
| `_cal_preview_roi()` | Compute preview ROI from checkerboard |
| `_cal_compute_entry_exit()` | Calculate entry/exit zone corners |
| `_cal_confirm()` | Lock ROI + compute homography |
| `_cal_floor()` | RANSAC floor plane + depth map |
| `_cal_save()` | Write JSON + apply to detector |
| `_cal_draw_overlay()` | Per-frame checkerboard detection + visual overlay |

---

## 29. Vision-Based Belt Speed Measurement

### Problem

Belt speed was previously 100% manual — a UI slider value used for all time-anchor predictions. With 35cm of extrapolation (registration at 15cm → pick at ~50cm) over ~5.8 seconds, a 10% speed error translates to **3.5cm position error** at pick time. The system had no way to verify or correct the slider value.

### Solution: Two-Checkpoint Transit Timing

Objects are camera-tracked through the entire ROI (0–30cm). Two timing checkpoints create a measured speed:

| Checkpoint | Position | Event |
|-----------|----------|-------|
| **Registration line** | 15cm | Object queued, `reg_time` + `reg_belt_y` stored |
| **Exit line** | 30cm (ROI exit) | Object's `camera_belt_y_cm` crosses exit, `exit_time` + `exit_belt_y` stored |

**Per-object speed:** `measured_speed = (exit_y - reg_y) / (exit_time - reg_time)`

**System median:** Rolling window of last 10 per-object speeds, median filters outliers.

### Anchor + Speed Priority

`_get_anchor_belt_y()` now uses a priority chain for both anchor point and speed:

| Priority | Anchor | Speed | When |
|----------|--------|-------|------|
| 1 (best) | Exit point | Per-object measured | After ROI exit crossing |
| 2 | Registration | Per-object or system median | Before exit but speed available |
| 3 | Registration | UI slider | No measured speed yet |
| 4 (fallback) | Stored belt_y | UI slider | No anchor at all |

### Key Design: First Object Benefits Too

Unlike approaches where the first object calibrates speed for subsequent ones, here the **same object's own transit** provides its speed. The exit crossing re-anchors prediction from a point only 12cm from workspace entry (vs 27cm from registration), dramatically reducing extrapolation error.

| Scenario | Old System | New System |
|----------|-----------|------------|
| Extrapolation distance | 35cm (reg → pick) | 12cm (exit → WS entry) + 8cm (to mid-WS) |
| Speed source | Manual slider | Vision-measured per-object |
| First object accuracy | Same as slider accuracy | Measured from own transit |
| Error at pick (10% speed error) | 3.5cm | 0.8cm |

### Implementation

**config.py:**
- `EXIT_LINE_CM = ROI_HEIGHT_CM` (30cm — ROI exit boundary)
- `SPEED_MEASUREMENT_WINDOW = 10` (rolling deque size)

**tracker.py:**
- `__init__`: `self._speed_measurements = deque(maxlen=10)`, `self.measured_belt_speed = None`
- `update()`: Queued objects store `camera_belt_y_cm` (camera-observed Y, separate from locked `belt_y_cm`)
- `check_exit_crossing()`: Detects exit crossing, computes per-object speed, updates rolling median
- `get_effective_belt_speed()`: Returns measured median or UI slider fallback
- `_get_anchor_belt_y()`: 4-level priority chain (exit anchor > reg anchor, measured speed > slider)

**index.py:**
- `check_exit_crossing()` called after `check_registration_crossing()` each frame
- `_process_picking()`: Uses `effective_speed` (measured or slider) for robot dispatch
- `_feed_pick_tracking()`, `_process_tracking()`, `_update_workspace_simulation()`, `_draw_tracked_masks()`: All refactored to use centralized `_get_anchor_belt_y()` instead of inlined time-anchor logic
- Simulation: Exit line drawn as dashed green, measured speed displayed at exit line and in "Measured: X.XX cm/s (N samples)" label near belt speed slider
- Dashboard: Measured speed indicator updated every 5 frames

### Unified Speed Consumers

All 6 speed consumer locations in `index.py` are unified to prefer measured speed over the UI slider:

```python
ui_speed = dpg.get_value("in_speed")
belt_speed = self.tracker.measured_belt_speed or ui_speed
```

| Consumer | Function | What It Controls |
|----------|----------|------------------|
| Frame update | `_update_frame()` | Exit crossing checks, per-frame tracking calls |
| Pick tracking | `_feed_pick_tracking()` | Ghost position prediction during active pick |
| Tracking mode | `_process_tracking()` | Robot real-time follow speed in TRACK mode |
| Drawing | `_draw_tracking()` | Visual overlay positions for tracked objects |
| Simulation | `_update_workspace_simulation()` | Workspace sim object advancement |
| Pick dispatch | `_process_picking()` | Robot `set_offsets(belt_speed=…)` at pick time |

**Robot.py** receives the effective speed via `set_offsets(belt_speed=effective_speed)` at dispatch. Its `execute_pick()` uses `self.belt_speed_cm_s` for forward prediction: `predict_y = final_ws_y + belt_speed × PICK_TRAVEL_TIME_S`. This ensures the robot moves to where the object **will be** when the arm arrives, not where it was when the pick was dispatched.

This unification guarantees that when belt speed changes, every part of the system — tracking, picking, simulation, and ghost prediction — reacts identically. No consumer reads the raw slider while others use measured speed.

### Console Output

```
[SPEED] #3 (Plastic) reg_y=15.0 → exit_y=30.2cm in 2.54s = 5.98 cm/s  (system median: 5.98 cm/s)
[SPEED] #4 (Metal)   reg_y=15.1 → exit_y=30.5cm in 2.58s = 5.97 cm/s  (system median: 5.97 cm/s)
```

### Sanity Guards

- Transit time must be ≥ 100ms (rejects same-frame false crossings)
- Speed must be 1.0–30.0 cm/s (rejects obvious outliers)
- Rolling median (not mean) resists single bad measurements
- UI slider remains as fallback when no measurements exist

---

## 30. Central Dashboard Sync Architecture

The customer deployment path separates reporting from machine control:

- `index.py` keeps controlling the sorter and writing local machine data.
- `run_dashboard_sync.py` polls the machine database every 60 seconds, builds a minute summary, stores pending payloads in a local outbox, and posts them to the central dashboard API.
- `dashboard_app.py` stores all web-facing data in `central_dashboard.db` and never queries machine databases directly.

### Central Dashboard Tables

- `machines`: machine registry and latest metadata
- `minute_stats`: 1 row per `machine_id + minute_bucket`
- `machine_live_status`: latest per-machine status snapshot
- `sync_audit`: ingest audit trail

### Machine Sync Outbox

The sync worker keeps its own SQLite file, `machine_sync.db`, with:

- `sync_state`: last successfully synced minute per machine
- `sync_outbox`: queued payloads, retry counts, and last error

This design means the machine continues sorting even if the dashboard or network is unavailable. Failed sync payloads remain queued and are retried during the next cycle.

---

*End of Technical Documentation*
