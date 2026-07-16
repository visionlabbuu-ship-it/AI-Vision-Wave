# Waste Sorting Delta Robot — Project Summary

> **Last updated:** 9 April 2026  
> **Platform:** NVIDIA Jetson AGX Orin  
> **Application:** `index.py` (~5 100 lines, Dear PyGui)

---

## 1. What the System Does

An automated waste-sorting station that **identifies, tracks, and picks** objects
from a moving conveyor belt using a 3-arm delta robot. Each object is classified
as **Glass, Metal, Paper, or Plastic** and placed into the correct bin.

### End-to-End Flow

```
Camera → YOLO Segmentation → Depth Height → Centroid Tracking
   → Belt Speed Measurement → Registration Queue
   → Smart Priority Scoring → Predictive Pick
   → Spectrum Scan (mid-air) → Bayesian Fusion → Place to Bin
```

---

## 2. Hardware

| Component | Model / Spec | Connection |
|-----------|-------------|------------|  
| **Computer** | NVIDIA Jetson AGX Orin | — |
| **Camera** | Intel RealSense D-series RGB-D | USB 3, 640×480 @ 30 fps |
| **Delta Robot** | 3-arm parallel kinematics, ±110 mm reach | Serial `/dev/ttyACM0` (G-code) |
| **Linear Slider** | Positions robot over belt | Serial `/dev/ttyUSB1` (custom M-code) |
| **Spectrum Sensor** | SparkFun AS7265X Triad (18-channel, 410–940 nm) | I²C (SMBus) |
| **Conveyor Belt** | 20 cm wide, ~5.5–6.0 cm/s | Motor via Jetson GPIO |
| **Vacuum Gripper** | Suction cup on delta end-effector | G-code M3 ON / M5 OFF |
| **Robot USB Camera** | Generic USB webcam (robot-mounted) | USB, 640×480 |

---

## 3. Software Architecture

### Runtime Split For Customer Delivery

The delivered system is now split into three runtime roles:

- `index.py`: machine control runtime on each sorting machine
- `run_dashboard_sync.py`: machine-side 1-minute sync worker that pushes summaries to the web server
- `dashboard_app.py`: central web dashboard backed by its own database

This separation keeps robot control independent from dashboard or network availability while allowing many machines to report to one central dashboard.

### Core Modules (`modules/`)

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `config.py` | ~470 | All constants, ROI geometry, robot grids, model discovery |
| `camera.py` | ~100 | RealSense pipeline wrapper (colour + depth streams) |
| `detector.py` | ~850 | YOLO inference, depth clustering, cross-class NMS, watershed mask joining |
| `tracker.py` | ~850 | Centroid-based spatial tracking, time-anchor prediction, belt speed measurement, stacking detection, mask smoothing |
| `robot.py` | ~1 100 | Serial controllers (Delta + Slider), RobotManager thread, predictive pick cycle, Bayesian fusion |
| `spectrum.py` | ~400 | AS7265X I²C driver, CatBoost ML classification |
| `logger.py` | ~150 | CSV + SQLite detection logging |
| `dashboard_sync.py` | new | Machine-side summary aggregation, outbox queue, retry upload |
| `dashboard_central.py` | new | Central dashboard DB schema, ingest upsert, and query shaping |

### Main Application

| File | Purpose |
|------|---------|
| `index.py` | Full Dear PyGui application — camera loop, UI, tracking, pick dispatch |
| `dashboard_app.py` | Central Flask dashboard for multi-machine monitoring |
| `run_dashboard_sync.py` | CLI sync worker for 1-minute dashboard uploads |
| `as7265x_sparkfun_python.py` | Low-level AS7265X register driver (I²C) |
| `calibration_tool.py` | Standalone calibration backup (integrated into Tab 5) |
| `motor_direct_jetson.py` | Conveyor motor control via Jetson GPIO |
| `detection_experiment.py` | Standalone model comparison / parameter tuning tool |
| `Manual_Inspection.py` | Standalone manual labelling + confusion-matrix tool |
| `export_tensorrt.py` | PyTorch → TensorRT FP16 export + benchmark |
| `fusion_weight_experiment.py` | Spectrum weight sweep experiment |

### ML Models

| File | Type | Notes |
|------|------|-------|
| `yolov26s_fixed.pt` | YOLOv26s segmentation (default) | Extended-dataset training |
| `yolov26s.pt` | YOLOv26s segmentation | Original training |
| `yolov11s.pt` | YOLOv11s segmentation | Older model |
| `yolov8s.pt` / `.engine` | YOLOv8s segmentation | TensorRT export available |
| `Model/` directory | CatBoost + scaler + encoder | Spectrum ML pipeline |

### Customer Launch Scripts

| Script | Purpose |
|-------|---------|
| `scripts/install_customer.sh` | Create customer venv and install dashboard/sync dependencies |
| `scripts/run_machine.sh` | Launch the machine runtime with the provisioned Python interpreter |
| `scripts/run_dashboard.sh` | Launch the central dashboard web app |
| `scripts/run_sync_worker.sh` | Launch the recurring sync worker |

---

## 4. Detection Pipeline

```
Raw Frame (640×480)
  │
  ├─ YOLO Segmentation (confidence ≥ 0.5)
  │    └─ Outputs: class, confidence, mask, bounding box
  │
  ├─ Depth Clustering (optional toggle)
  │    └─ Splits detections with bimodal depth histograms
  │
  ├─ Cross-Class Mask NMS (optional toggle)
  │    └─ Resolves same-pixel multi-class overlaps
  │
  └─ Watershed Mask Joining (optional toggle, off by default)
       └─ Rejoins same-class mask fragments from occlusion
```

Each pipeline stage is individually togglable from the UI. The active stages are
displayed as a status label (e.g., `Pipeline: DC > NMS`).

---

## 5. Belt Coordinate System & ROI

```
  Camera top (belt enters)
  ┌──────────────────────┐  Y = 0 cm
  │   Detection Zone     │
  │   (objects appear)    │
  ├──── Registration ─────┤  Y = 15 cm  ← objects queued here
  │   Tracking Zone       │
  │   (camera-visible)    │
  └──────────────────────┘  Y = 30 cm  ← ROI exit, speed measured
         ~ 14.5 cm gap (not visible)
  ┌──────────────────────┐  Y ≈ 42 cm  ← robot workspace entry
  │   Robot Workspace     │
  │   (20 cm deep)        │
  └──────────────────────┘  Y ≈ 62 cm  ← workspace exit (objects lost)
```

- **Belt width:** 20 cm (0–200 mm in robot space)
- **Belt speed:** ~5.5–6.0 cm/s (vision-measured or manual slider)
- **Registration line:** 15 cm — objects are formally queued and time-stamped here
- **ROI exit:** 30 cm — per-object belt speed measured at this crossing

---

## 6. Tracking & Belt Speed Measurement

### Centroid Tracker (`SimpleTracker`)

- Matches new detections to existing objects via minimum Euclidean centroid distance
- **Time-anchor prediction**: Each object stores `(reg_belt_y, reg_time)` at
  registration and `(exit_belt_y, exit_time)` at ROI exit; future position is
  extrapolated from the most recent anchor

### Vision-Based Belt Speed

When an object crosses the ROI exit line its transit time gives a per-object speed:

$$v = \frac{y_{\text{exit}} - y_{\text{reg}}}{t_{\text{exit}} - t_{\text{reg}}}$$

A rolling median of the last 10 measurements becomes the **system measured speed**,
used by all consumers (tracking, simulation, pick prediction). A UI checkbox
("Dynamic Speed Estimation") lets the operator toggle between measured and fixed
(slider) speed. Sanity guards: transit ≥ 100 ms, speed 1–30 cm/s.

---

## 7. Stacking Detection

Three combined techniques identify overlapping objects:

1. **Dilated mask IoU** — each mask dilated by 20 px; pairwise IoU ≥ 2 % = contact
2. **Height difference** — ≥ 2.0 cm height diff + contact → `physical_stack`; < 2.0 cm → `adjacent`
3. **Union-find grouping** — transitive closure (A↔B, B↔C → group {A, B, C}), sorted tallest-first

Physical stacks enforce **tallest-first picking** (hard block on lower objects).
Adjacent objects receive a soft penalty in the priority scorer.

---

## 8. Smart Pick Priority Scoring

All objects inside the workspace are scored:

$$\text{Score} = 1.0 \times \text{urgency} + 0.3 \times \text{height} + 0.5 \times \text{isolation} - 0.8 \times \text{stack\_risk}$$

| Factor | Meaning |
|--------|---------|
| Urgency | Closer to workspace exit → higher priority |
| Height | Taller objects (on top of stacks) preferred |
| Isolation | Standalone objects preferred over clustered |
| Stack risk | High pairwise IoU penalised (risk of knocking neighbours) |

Physical-stack members are hard-blocked until all taller group members are removed.

---

## 9. Predictive Pick Cycle

The robot does **not** follow objects in real-time. It predicts where the object
**will be** and moves directly.

| Phase | Action | Duration |
|-------|--------|----------|
| **PREDICT** | Forward-predict belt-Y for hover arrival | — |
| **APPROACH** | Vacuum ON → move to predicted hover position | ~400 ms |
| **DESCEND** | Read live position → correction-predict → drop to pick Z | ~100 ms |
| **MECHANICAL** | Suction settle → lift → spectrum scan → fusion → place → release → standby | ~1.5 s |

### Pick Z Calculation

```
pick_z = BASE_Z + (height_cm × 10) + global_z_offset + class_z_offset + bottom_offset
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| `BASE_Z` | −425.0 mm | Calibrated belt-contact Z |
| Class Z offsets | Glass +5, Metal +5, Paper −10, Plastic 0 mm | Per-class tuning |
| Stack-bottom Z extra | −10 mm (default, UI-adjustable) | Extra depth for bottom objects |
| Stack-bottom Y advance | 2.0 cm (default, UI-adjustable) | Belt drift compensation during pick |
| Surface penetration | 5 mm max | Safety clamp — never below belt floor |

---

## 10. Bayesian Fusion (YOLO + Spectrum)

At the scan position the vacuum-held object is read by the 18-channel spectrum
sensor. CatBoost predicts a material class. The two predictions are fused:

1. **Contact gate** — total spectral intensity < 220 → no object → YOLO only
2. **Confusion-matrix likelihoods** — P(pred | true_class) for each sensor
3. **Dynamic weights** — based on margin (top-2 gap) and entropy per sensor
4. **Gamma-powered Bayesian posterior**

$$P(\text{class} \mid \text{YOLO}, \text{Spec}) \propto P(\text{YOLO} \mid \text{class})^{w_v} \times P(\text{Spec} \mid \text{class})^{w_s} \times P(\text{class})$$

- `FUSION_GAMMA = 3.5` — sharpens high-confidence predictions
- `SPECTRUM_WEIGHT_SCALE = 0.1` — heavily down-weights spectrum (camera leads)

> **Why 0.1?** A sweep experiment over 956 audit rows showed the spectrum sensor
> disagrees with camera 62.9 % of the time (heavily biased toward Metal), causing
> a 12.6 % accuracy drop at the default weight. Reducing to 0.1 preserves the
> camera's ~100 % accuracy while still allowing spectrum to contribute when both
> sensors agree.

---

## 11. Position Mapping (Belt → Robot)

A **3×3 bilinear interpolation grid** (9 manually measured calibration points) maps
belt coordinates to robot X/Y. Both robot axes depend on both belt axes due to
coupled delta kinematics. Fine-tuning via `offsets.json`:

| Offset | Default | Unit | Purpose |
|--------|---------|------|---------|
| X | 0.0 | mm | Lateral correction |
| Y | 0.0 | mm | Along-belt correction |
| Z | 0.0 | mm | Height correction |
| Latency | 0.0 | s | Belt-movement time compensation |

Per-class Z offsets and stack-bottom offsets are also saved in `offsets.json`.

---

## 12. User Interface (5 Tabs)

### Tab 1 — Vision (Main Sorting View)

- **Left:** 640×480 live feed with YOLO overlays, crosshair pick points, mask
  fade-in/fade-out, pipeline status badge
- **Left bottom:** Tracking dashboard (up to 10 objects: ID, class, X, Y, height, status)
- **Right:** START/STOP/AUTO PICK buttons, belt speed slider + Dynamic Speed checkbox,
  YOLO confidence, pipeline toggles (Depth Clustering / NMS / Watershed),
  X/Y/Z/Latency offsets, spectrum prediction display, robot workspace simulation
  (top-down 2D view with objects, robot dot, target crosshair)

### Tab 2 — Robot Control

- **Left column (~710 px):** Robot USB camera feed, position readout (X, Y, Z, R)
- **Right column:** Manual jog (X±/Y±/Z±/R± with step size), direct GO position,
  vacuum ON/OFF, preset positions, pick test, smooth demos,
  place positions editor (per-class XY + Z, apply/save/load/reset),
  3×3 workspace grid (belt↔robot coordinate calibration),
  **Stack-bottom controls**: Y advance (cm) + Z extra (mm) — both UI-adjustable

### Tab 3 — Spectrum Scan

- Manual scan trigger, LED toggles (IR / White / UV), raw 18-channel display

### Tab 4 — Video & Recording

- `.bag` file loader with play/stop/unload + recording toggle
- Experiment recording (.mp4 screen capture + linked robot camera)

### Tab 5 — Calibration

- 5-step guided process: Enable → Flip/Orient → Confirm ROI → Floor (RANSAC) → Save
- Uses the live camera feed (no second pipeline), bypasses YOLO/ROI during calibration
- Outputs `calibration_data.json` (homography, floor plane, intrinsics)

---

## 13. Data Logging

| Destination | Content |
|-------------|---------|
| **CSV** (`detection_logs/`) | 30 columns: ID, camera/spectrum/final classes + confidences, 18 spectral channels |
| **SQLite** | Same data in structured DB (session, object_id, classes, height, position, spectrum) |
| **Cropped images** | Per-detection RGB crop saved alongside CSV |

---

## 14. TensorRT Support

`.pt` models can be exported to TensorRT FP16 `.engine` for ~2× faster inference
on Jetson. Tool: `export_tensorrt.py` (interactive GUI or CLI). Key rule: `.engine`
files must pass `task='segment'` explicitly to Ultralytics YOLO loader.

**Benchmark:** YOLOv8s — 32 FPS (PyTorch) → 59 FPS (TensorRT FP16) on Orin.

---

## 15. Calibration

### Camera Calibration (`calibration_data.json`)

- **Checkerboard:** 8×10 squares (7×9 inner corners), 2.5 cm square size
- **Homography:** 3×3 perspective matrix mapping pixel → belt-cm
- **Floor plane:** RANSAC from depth frame (100 iterations, 1 cm inlier threshold)
- **Floor depth map:** Down-sampled 4×, median blur + morphological close

### Robot Coordinate Calibration

- 3×3 grid of belt positions → robot positions, stored as `ROBOT_X_GRID` / `ROBOT_Y_GRID`
- Editable live from Tab 2 (apply / save / go-to / set-from-current-position)

---

## 16. Key Algorithms at a Glance

| Algorithm | Location | Purpose |
|-----------|----------|---------|
| Time-anchor Y prediction | `tracker.py` | Drift-free belt position from timestamp × speed |
| Vision belt speed | `tracker.py` | Per-object transit speed, rolling median |
| Dilated-mask stacking | `tracker.py` | IoU + height diff → physical/adjacent grouping |
| Smart priority scoring | `tracker.py` | Urgency + height + isolation − stack_risk |
| Bilinear grid interpolation | `robot.py` | Belt (X,Y) → Robot (X,Y) |
| Predictive pick | `robot.py` | Forward-predict belt drift during robot travel |
| Gamma Bayesian fusion | `robot.py` | Combine YOLO + spectrum with confidence weighting |
| Watershed mask join | `detector.py` | Rejoin same-class fragments from occlusion |
| IoU class resolution | `tracker.py` | De-duplicate cross-class overlapping masks |
| Mask temporal smoothing | `tracker.py` | EMA + morphology + contour approx |
| Mask alpha fading | `tracker.py` + `index.py` | Smooth appear/disappear for flickering detections |
| RANSAC floor fitting | `index.py` (cal) | Floor plane from depth point cloud |

---

## 17. Known Issues & Safety Notes

### Threading

| Risk | Severity | Status |
|------|----------|--------|
| No serial lock — concurrent writes corrupt G-code | 🔴 Critical | Documented, not yet fixed |
| Spectrum I²C can block robot thread up to 7.6 s | 🔴 Critical | Documented |
| STOP button doesn't stop robot thread | 🟠 High | Documented |
| `serial.send()` failures silently ignored | 🟡 Medium | Documented |

### Safety

- Pick Z is clamped to never go below `BASE_Z` (belt floor)
- No minimum-height guard for very flat objects (< 0.5 cm)
- Latency offset is user-controlled with no max clamp

### Recommendations

1. Add `threading.Lock` to `SerialController.send()`
2. Add watchdog timer (10 s) on `execute_pick()`
3. Call `robot_manager.stop()` in `_on_stop()`
4. Wrap spectrum I²C reads in a timeout wrapper
5. Clamp latency offset to 0–2 s

---

## 18. File Structure

```
Spectrum_pipeline/
├── index.py                        # Main application (Dear PyGui, ~5 100 lines)
├── as7265x_sparkfun_python.py      # AS7265X I²C register driver
├── calibration_tool.py             # Standalone calibration backup
├── detection_experiment.py         # Model comparison tool
├── Manual_Inspection.py            # Manual labelling + confusion matrix
├── export_tensorrt.py              # TensorRT export + benchmark
├── fusion_weight_experiment.py     # Spectrum weight sweep experiment
├── motor_direct_jetson.py          # Conveyor motor GPIO control
├── offsets.json                    # Runtime X/Y/Z/latency/class offsets
├── calibration_data.json           # Camera homography + floor plane
├── place_targets.json              # Per-class bin coordinates
├── yolov26s_fixed.pt              # Default YOLO model
├── modules/
│   ├── config.py                   # All constants + model discovery
│   ├── camera.py                   # RealSense pipeline
│   ├── detector.py                 # YOLO + depth clustering + NMS + watershed
│   ├── tracker.py                  # Centroid tracker + belt speed + stacking
│   ├── robot.py                    # Serial controllers + pick cycle + fusion
│   ├── spectrum.py                 # AS7265X ML pipeline (CatBoost)
│   ├── logger.py                   # CSV + SQLite logging
│   └── dashboard.py                # Plotly Dash web dashboard
├── Model/                          # CatBoost model + scaler + encoder
├── docs/                           # Documentation
│   ├── TECHNICAL_DOCUMENTATION.md  # Full technical reference (~1 870 lines)
│   ├── USER_WORKFLOW_GUIDE.md      # Operator guide (~630 lines)
│   ├── Next_part.md                # Motor connection note
│   └── PROJECT_SUMMARY.md          # ← This file
├── detection_logs/                 # CSV detection logs + cropped images
├── manual_inspection_logs/         # Manual inspection CSV data
├── manual_inspection_images/       # Manual inspection captures
├── recordings/                     # .bag and .mp4 recordings
├── templates/                      # Dash HTML templates
├── _legacy/                        # Archived old versions
└── experiment_screenshots/         # Detection experiment captures
```

---

## 19. Quick Reference

| What | Value |
|------|-------|
| Default model | `yolov26s_fixed.pt` |
| YOLO confidence threshold | 0.5 |
| Camera resolution | 640 × 480 @ 30 fps |
| Belt speed (typical) | 5.5–6.0 cm/s |
| ROI size | 20 × 30 cm |
| Registration line | 15 cm |
| Workspace offset from ROI exit | ~12 cm |
| Workspace depth | 20 cm |
| BASE_Z (belt contact) | −425.0 mm |
| Suction dwell | 0.05 s |
| Hover clearance | 10 mm |
| Hover travel time | 0.40 s |
| Correction move time | 0.10 s |
| Fusion gamma | 3.5 |
| Spectrum weight scale | 0.1 |
| Stack-bottom Y advance | 2.0 cm (default) |
| Stack-bottom Z extra | −10 mm (default) |
| Material classes | Glass · Metal · Paper · Plastic |
| Delta port | `/dev/ttyACM0` |
| Slider port | `/dev/ttyUSB1` |

---

*This summary was auto-generated from TECHNICAL_DOCUMENTATION.md, USER_WORKFLOW_GUIDE.md,
and the live codebase on 9 April 2026.*
