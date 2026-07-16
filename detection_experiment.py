"""
=============================================================================
DETECTION EXPERIMENT TOOL
=============================================================================
Standalone DearPyGui application for comparing YOLO models and separation
logic settings on a live RealSense camera feed.

No robot, no tracker, no belt — pure detection only.

Features:
  • Live model hot-swap (all .pt / .engine files auto-discovered)
  • Toggle each separation stage independently:
      - Depth Clustering (split merged masks)
      - Cross-Class NMS  (remove duplicate detections)
      - Watershed Joining (rejoin split fragments)
  • Adjustable confidence threshold
  • Adjustable depth clustering gap (mm)
  • Live detection count, FPS, inference time
  • Per-class detection count breakdown
  • Screenshot (saves annotated frame to experiment_screenshots/)
  • Freeze / resume feed to inspect a frame
  • Show/hide: masks, edges, contours, height labels, ROI overlay
  • Show depth colormap alongside colour feed
"""

import cv2
import numpy as np
import time
import os
import threading
import queue
from datetime import datetime
from collections import deque

import dearpygui.dearpygui as dpg

# ── project imports ──────────────────────────────────────────────────────────
from modules.config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, FPS, CALIBRATION_FILE,
    CLASS_NAMES, CLASS_COLORS, DEFAULT_COLOR,
    CONFIDENCE_THRESHOLD,
    DUPLICATE_MASK_NMS_ENABLED, DUPLICATE_MASK_IOU_THRESHOLD,
    DEPTH_CLUSTER_ENABLED, DEPTH_CLUSTER_MIN_MASK_PX,
    DEPTH_CLUSTER_DEPTH_GAP_MM, DEPTH_CLUSTER_MIN_CLUSTER_PX,
    DEPTH_CLUSTER_MAX_SPLITS, DEPTH_CLUSTER_HIST_BINS,
    DEPTH_CLUSTER_MORPH_OPEN_PX, DEPTH_CLUSTER_SPATIAL_CONNECT,
    WATERSHED_JOIN_ENABLED,
    AVAILABLE_MODELS, AVAILABLE_MODEL_NAMES,
    COLORMAP_OPTIONS, COLORMAP_NAMES,
    ROI_MIN_MASK_COVERAGE,
    WATERSHED_BOUNDARY_ALPHA,
    STACK_MASK_DILATE_PX, STACK_MASK_IOU_THRESHOLD, STACK_IOU_STACKING_MIN,
)
from modules.detector import (
    ObjectDetector,
    cross_class_mask_nms,
    depth_cluster_mask,
    watershed_join_masks,
    get_orientation_pca,
)
from modules.camera import CameraStream


# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SCREENSHOT_DIR = "experiment_screenshots"
W, H = IMAGE_WIDTH, IMAGE_HEIGHT


# =============================================================================
# EXPERIMENT APP
# =============================================================================
class DetectionExperiment:
    """DearPyGui application for detection-only experiments."""

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
    def __init__(self):
        dpg.create_context()
        self._setup_theme()

        # ── state ──
        self.running = False
        self.frozen = False           # pause feed for inspection
        self.frozen_color = None
        self.frozen_depth_frame = None

        # ── camera ──
        self.camera = CameraStream()

        # ── detector ──
        self.detector = ObjectDetector()
        self.detector.load_model()
        self.detector.load_calibration(CALIBRATION_FILE)

        # ── separation toggles (start with project defaults) ──
        self.use_depth_cluster = DEPTH_CLUSTER_ENABLED
        self.use_cross_nms    = DUPLICATE_MASK_NMS_ENABLED
        self.use_watershed    = WATERSHED_JOIN_ENABLED

        # ── stacking analysis toggle ──
        self.show_stacking  = True

        # ── display toggles ──
        self.show_masks     = True
        self.show_contours  = True
        self.show_labels    = True
        self.show_height    = True
        self.show_centroids = True
        self.show_roi       = True
        self.show_depth     = False   # side-by-side depth map
        self.mask_alpha     = 0.35
        self.colormap_idx   = 0

        # ── stacking analysis results (per frame) ──
        self.stack_labels = {}        # det_idx -> 'TOP'/'BOT'/'CLOSE'
        self.stack_pairs  = []        # [(idx_a, idx_b, iou, stack_type), ...]
        self.stack_group_count = 0
        self.stacking_count = 0
        self.adjacent_count = 0

        # ── perf tracking ──
        self.fps_history = deque(maxlen=60)
        self.infer_ms_history = deque(maxlen=60)
        self.last_frame_time = time.time()
        self.last_detections = []
        self.frame_count = 0

        # ── screenshot ──
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

        # ── build UI ──
        self._build_ui()

        dpg.create_viewport(title="Detection Experiment", width=1540, height=860)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("exp_main", True)

    # ------------------------------------------------------------------
    # THEME
    # ------------------------------------------------------------------
    def _setup_theme(self):
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 4)
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 25, 30))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (40, 40, 50))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (50, 60, 80))
        dpg.bind_theme(self.global_theme)

    # ------------------------------------------------------------------
    # UI LAYOUT
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ── textures ──
        with dpg.texture_registry(show=False):
            blank = np.zeros((H, W, 4), dtype=np.float32)
            dpg.add_dynamic_texture(W, H, blank.flatten(), tag="exp_tex_color")
            dpg.add_dynamic_texture(W, H, blank.flatten(), tag="exp_tex_depth")

        with dpg.window(tag="exp_main", no_title_bar=True, no_move=True):
            # ── top bar ──
            with dpg.group(horizontal=True):
                dpg.add_text("DETECTION EXPERIMENT", color=(0, 220, 255))
                dpg.add_spacer(width=30)
                dpg.add_text("FPS: --", tag="exp_fps", color=(180, 255, 180))
                dpg.add_spacer(width=20)
                dpg.add_text("Infer: -- ms", tag="exp_infer", color=(255, 220, 150))
                dpg.add_spacer(width=20)
                dpg.add_text("Detections: 0", tag="exp_det_count", color=(255, 255, 255))

            dpg.add_separator()

            # ── main content: left = video, right = controls ──
            with dpg.group(horizontal=True):

                # ===== LEFT: video feeds =====
                with dpg.child_window(width=660, height=-1):
                    dpg.add_text("Camera Feed", color=(0, 200, 255))
                    dpg.add_image("exp_tex_color", width=W, height=H, tag="exp_img_color")

                    # Depth feed (toggled)
                    dpg.add_spacer(height=4)
                    dpg.add_text("Depth Map", color=(0, 200, 255), tag="exp_depth_label", show=False)
                    dpg.add_image("exp_tex_depth", width=W, height=H, tag="exp_img_depth", show=False)

                # ===== RIGHT: control panel =====
                with dpg.child_window(width=-1, height=-1):
                    self._build_controls()

    # ------------------------------------------------------------------
    def _build_controls(self):
        """Build the right-side control panel."""

        # ── MODEL ──
        with dpg.collapsing_header(label="Model", default_open=True):
            dpg.add_combo(
                items=AVAILABLE_MODEL_NAMES,
                default_value=AVAILABLE_MODEL_NAMES[0] if AVAILABLE_MODEL_NAMES else "",
                tag="exp_model_combo",
                callback=self._on_model_change,
                width=-1,
            )
            dpg.add_text("", tag="exp_model_status", color=(120, 255, 120))

        dpg.add_spacer(height=6)

        # ── CONFIDENCE ──
        with dpg.collapsing_header(label="Confidence", default_open=True):
            dpg.add_slider_float(
                label="Threshold",
                default_value=CONFIDENCE_THRESHOLD,
                min_value=0.1, max_value=1.0,
                tag="exp_conf",
                callback=self._on_conf_change,
                width=-1,
            )

        dpg.add_spacer(height=6)

        # ── SEPARATION LOGIC ──
        with dpg.collapsing_header(label="Separation Logic", default_open=True):
            dpg.add_checkbox(
                label="Depth Clustering  (split merged masks)",
                default_value=self.use_depth_cluster,
                tag="exp_depth_cluster",
                callback=self._on_toggle,
            )
            with dpg.group(horizontal=False, indent=20):
                dpg.add_slider_int(
                    label="Depth Gap (mm)",
                    default_value=int(DEPTH_CLUSTER_DEPTH_GAP_MM),
                    min_value=5, max_value=50,
                    tag="exp_depth_gap",
                    width=-1,
                )
                dpg.add_slider_int(
                    label="Min Mask (px)",
                    default_value=int(DEPTH_CLUSTER_MIN_MASK_PX),
                    min_value=200, max_value=3000,
                    tag="exp_depth_min_px",
                    width=-1,
                )
                dpg.add_slider_int(
                    label="Min Cluster (px)",
                    default_value=int(DEPTH_CLUSTER_MIN_CLUSTER_PX),
                    min_value=50, max_value=1000,
                    tag="exp_depth_min_cluster",
                    width=-1,
                )
                dpg.add_slider_int(
                    label="Max Splits",
                    default_value=int(DEPTH_CLUSTER_MAX_SPLITS),
                    min_value=1, max_value=8,
                    tag="exp_depth_max_splits",
                    width=-1,
                )

            dpg.add_spacer(height=4)

            dpg.add_checkbox(
                label="Cross-Class NMS  (remove duplicates)",
                default_value=self.use_cross_nms,
                tag="exp_cross_nms",
                callback=self._on_toggle,
            )
            with dpg.group(horizontal=False, indent=20):
                dpg.add_slider_float(
                    label="NMS IoU Thresh",
                    default_value=DUPLICATE_MASK_IOU_THRESHOLD,
                    min_value=0.1, max_value=0.9,
                    tag="exp_nms_iou",
                    width=-1,
                )

            dpg.add_spacer(height=4)

            dpg.add_checkbox(
                label="Watershed Joining  (rejoin fragments)",
                default_value=self.use_watershed,
                tag="exp_watershed",
                callback=self._on_toggle,
            )

        dpg.add_spacer(height=6)

        # ── STACKING / ADJACENT ANALYSIS ──
        with dpg.collapsing_header(label="Stacking & Adjacent Detection", default_open=True):
            dpg.add_checkbox(
                label="Enable stacking / adjacent analysis",
                default_value=self.show_stacking,
                tag="exp_show_stacking",
                callback=self._on_toggle,
            )
            with dpg.group(horizontal=False, indent=20):
                dpg.add_slider_int(
                    label="Dilate (px)",
                    default_value=int(STACK_MASK_DILATE_PX),
                    min_value=5, max_value=60,
                    tag="exp_stack_dilate",
                    width=-1,
                )
                dpg.add_slider_float(
                    label="Adjacent IoU",
                    default_value=STACK_MASK_IOU_THRESHOLD,
                    min_value=0.005, max_value=0.20, format="%.3f",
                    tag="exp_stack_adj_iou",
                    width=-1,
                )
                dpg.add_slider_float(
                    label="Stacking IoU",
                    default_value=STACK_IOU_STACKING_MIN,
                    min_value=0.02, max_value=0.50, format="%.2f",
                    tag="exp_stack_stk_iou",
                    width=-1,
                )
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_text("Groups:", color=(180, 180, 180))
                dpg.add_text("0", tag="exp_stack_groups", color=(255, 255, 255))
                dpg.add_spacer(width=12)
                dpg.add_text("Stacking:", color=(255, 100, 100))
                dpg.add_text("0", tag="exp_stack_stacking", color=(255, 255, 255))
                dpg.add_spacer(width=12)
                dpg.add_text("Adjacent:", color=(255, 200, 100))
                dpg.add_text("0", tag="exp_stack_adjacent", color=(255, 255, 255))

        dpg.add_spacer(height=6)

        # ── DISPLAY ──
        with dpg.collapsing_header(label="Display", default_open=True):
            dpg.add_checkbox(label="Masks",     default_value=self.show_masks,     tag="exp_show_masks",     callback=self._on_toggle)
            dpg.add_checkbox(label="Contours",   default_value=self.show_contours,  tag="exp_show_contours",  callback=self._on_toggle)
            dpg.add_checkbox(label="Labels",     default_value=self.show_labels,    tag="exp_show_labels",    callback=self._on_toggle)
            dpg.add_checkbox(label="Heights",    default_value=self.show_height,    tag="exp_show_height",    callback=self._on_toggle)
            dpg.add_checkbox(label="Centroids",  default_value=self.show_centroids, tag="exp_show_centroids", callback=self._on_toggle)
            dpg.add_checkbox(label="ROI Overlay",default_value=self.show_roi,       tag="exp_show_roi",       callback=self._on_toggle)
            dpg.add_checkbox(label="Depth Map",  default_value=self.show_depth,     tag="exp_show_depth",     callback=self._on_toggle_depth)
            dpg.add_slider_float(
                label="Mask Alpha",
                default_value=self.mask_alpha,
                min_value=0.0, max_value=1.0,
                tag="exp_mask_alpha",
                width=-1,
            )
            dpg.add_combo(
                items=COLORMAP_NAMES,
                default_value=COLORMAP_NAMES[0],
                label="Colormap",
                tag="exp_colormap",
                callback=self._on_colormap_change,
                width=-1,
            )

        dpg.add_spacer(height=6)

        # ── DETECTION BREAKDOWN ──
        with dpg.collapsing_header(label="Detection Breakdown", default_open=True):
            for cls_id, cls_name in CLASS_NAMES.items():
                color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
                # Convert BGR → RGB for DPG
                r, g, b = color[2], color[1], color[0]
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{cls_name}:", color=(r, g, b))
                    dpg.add_text("0", tag=f"exp_cls_{cls_id}", color=(255, 255, 255))

        dpg.add_spacer(height=6)

        # ── ACTIONS ──
        with dpg.collapsing_header(label="Actions", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", tag="exp_btn_start", callback=self._on_start, width=120)
                dpg.add_button(label="Stop",  tag="exp_btn_stop",  callback=self._on_stop,  width=120)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Freeze", tag="exp_btn_freeze", callback=self._on_freeze, width=120)
                dpg.add_button(label="Screenshot", callback=self._on_screenshot, width=120)

        dpg.add_spacer(height=6)

        # ── LOG ──
        with dpg.collapsing_header(label="Log", default_open=False):
            dpg.add_child_window(tag="exp_log_child", height=200, border=True)

    # ------------------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------------------
    def _on_model_change(self, sender, app_data):
        """Hot-swap model when user picks a different one."""
        idx = AVAILABLE_MODEL_NAMES.index(app_data) if app_data in AVAILABLE_MODEL_NAMES else -1
        if idx < 0:
            return
        path = AVAILABLE_MODELS[idx]
        self._log(f"Switching model -> {app_data} ...")
        dpg.set_value("exp_model_status", "Loading...")
        dpg.configure_item("exp_model_status", color=(255, 200, 50))

        def _swap():
            ok = self.detector.switch_model(path)
            if ok:
                dpg.set_value("exp_model_status", f"[OK] {app_data}")
                dpg.configure_item("exp_model_status", color=(120, 255, 120))
                self._log(f"Model loaded: {app_data}")
            else:
                dpg.set_value("exp_model_status", f"[X] Failed")
                dpg.configure_item("exp_model_status", color=(255, 80, 80))
                self._log(f"Failed to load {app_data}", "ERROR")

        threading.Thread(target=_swap, daemon=True).start()

    def _on_conf_change(self, sender, app_data):
        self.detector.conf_threshold = app_data

    def _on_toggle(self, sender, app_data):
        """Generic checkbox toggle reader."""
        self.use_depth_cluster = dpg.get_value("exp_depth_cluster")
        self.use_cross_nms     = dpg.get_value("exp_cross_nms")
        self.use_watershed     = dpg.get_value("exp_watershed")
        self.show_masks        = dpg.get_value("exp_show_masks")
        self.show_contours     = dpg.get_value("exp_show_contours")
        self.show_labels       = dpg.get_value("exp_show_labels")
        self.show_height       = dpg.get_value("exp_show_height")
        self.show_centroids    = dpg.get_value("exp_show_centroids")
        self.show_roi          = dpg.get_value("exp_show_roi")
        if dpg.does_item_exist("exp_show_stacking"):
            self.show_stacking = dpg.get_value("exp_show_stacking")

    def _on_toggle_depth(self, sender, app_data):
        self.show_depth = dpg.get_value("exp_show_depth")
        dpg.configure_item("exp_depth_label", show=self.show_depth)
        dpg.configure_item("exp_img_depth", show=self.show_depth)

    def _on_colormap_change(self, sender, app_data):
        idx = COLORMAP_NAMES.index(app_data) if app_data in COLORMAP_NAMES else 0
        self.colormap_idx = idx

    def _on_start(self, sender=None, app_data=None):
        if self.running:
            return
        self._log("Starting camera...")
        if self.camera.start_camera():
            self.running = True
            # Pass intrinsics to detector
            self.detector.intrinsics = self.camera.depth_intrinsics
            self._log("Camera started", "SUCCESS")
        else:
            self._log("Camera failed to start!", "ERROR")

    def _on_stop(self, sender=None, app_data=None):
        self.running = False
        self.frozen = False
        try:
            self.camera.running = False
            self.camera.pipeline.stop()
        except Exception:
            pass
        self._log("Camera stopped")

    def _on_freeze(self, sender=None, app_data=None):
        self.frozen = not self.frozen
        label = "Resume" if self.frozen else "Freeze"
        dpg.configure_item("exp_btn_freeze", label=label)
        self._log("Feed frozen" if self.frozen else "Feed resumed")

    def _on_screenshot(self, sender=None, app_data=None):
        if self._last_display is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"exp_{ts}.png")
            cv2.imwrite(path, self._last_display)
            self._log(f"Screenshot saved: {path}", "SUCCESS")
        else:
            self._log("No frame to save", "ERROR")

    def _log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        color_map = {"INFO": (200, 200, 200), "SUCCESS": (100, 255, 100),
                     "ERROR": (255, 80, 80), "WARN": (255, 200, 50)}
        color = color_map.get(level, (200, 200, 200))
        if dpg.does_item_exist("exp_log_child"):
            dpg.add_text(f"[{ts}] {msg}", color=color, parent="exp_log_child")
        print(f"[{ts}] [{level}] {msg}")

    # ------------------------------------------------------------------
    # CUSTOM DETECT (respects live toggles)
    # ------------------------------------------------------------------
    def _detect_with_settings(self, color_image, depth_frame):
        """
        Run YOLO and apply separation stages based on live UI toggles.
        Returns list of detection dicts.
        """
        model = self.detector.model
        if model is None:
            return []

        conf = dpg.get_value("exp_conf") if dpg.does_item_exist("exp_conf") else CONFIDENCE_THRESHOLD
        results = model(color_image, conf=conf, verbose=False)
        detections = []

        if not results or not results[0].masks:
            return detections

        depth_image_raw = np.asanyarray(depth_frame.get_data())

        for i, box in enumerate(results[0].boxes):
            if box.conf < conf:
                continue

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            raw_mask = results[0].masks.data[i].cpu().numpy()
            mask = cv2.resize(raw_mask, (W, H))
            mask = (mask > 0.5).astype(np.uint8)

            if mask.sum() < 100:
                continue

            # ROI coverage filter
            if self.detector._roi_mask is not None:
                total_px = int(mask.sum())
                if total_px == 0:
                    continue
                inside_px = int((mask & (self.detector._roi_mask > 0)).sum())
                coverage = inside_px / total_px
                if coverage < ROI_MIN_MASK_COVERAGE:
                    continue

            # === DEPTH CLUSTERING ===
            if self.use_depth_cluster:
                gap_mm = dpg.get_value("exp_depth_gap") if dpg.does_item_exist("exp_depth_gap") else DEPTH_CLUSTER_DEPTH_GAP_MM
                min_px = dpg.get_value("exp_depth_min_px") if dpg.does_item_exist("exp_depth_min_px") else DEPTH_CLUSTER_MIN_MASK_PX
                min_cl = dpg.get_value("exp_depth_min_cluster") if dpg.does_item_exist("exp_depth_min_cluster") else DEPTH_CLUSTER_MIN_CLUSTER_PX
                max_sp = dpg.get_value("exp_depth_max_splits") if dpg.does_item_exist("exp_depth_max_splits") else DEPTH_CLUSTER_MAX_SPLITS
                sub_masks = depth_cluster_mask(
                    mask, depth_image_raw,
                    depth_gap_mm=gap_mm,
                    min_cluster_px=min_cl,
                    max_splits=max_sp,
                )
            else:
                valid_d = depth_image_raw[mask > 0]
                valid_d = valid_d[(valid_d > 100) & (valid_d < 2000)]
                med_d = float(np.median(valid_d)) if len(valid_d) > 0 else 0
                sub_masks = [(mask, med_d)]

            is_split = len(sub_masks) > 1

            for sub_idx, (sub_mask, sub_depth_mm) in enumerate(sub_masks):
                M = cv2.moments(sub_mask)
                if M["m00"] <= 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                largest = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest)
                box_pts = cv2.boxPoints(rect).astype(np.int32)

                height_cm = self.detector.get_height_at_point(depth_frame, cx, cy)
                angle = get_orientation_pca(sub_mask)

                belt_x_cm, belt_y_cm = self.detector.pixel_to_belt_cm(cx, cy)

                masked_depths = depth_image_raw[sub_mask > 0]
                valid_depths = masked_depths[(masked_depths > 100) & (masked_depths < 2000)]
                avg_depth_mm = float(np.mean(valid_depths)) if len(valid_depths) > 0 else 0

                detections.append({
                    'centroid': (cx, cy),
                    'class_id': class_id,
                    'class_name': CLASS_NAMES.get(class_id, f"Class_{class_id}"),
                    'mask': sub_mask,
                    'contour': largest,
                    'min_area_box': box_pts,
                    'height_cm': height_cm,
                    'angle': angle,
                    'belt_x_cm': belt_x_cm,
                    'belt_y_cm': belt_y_cm,
                    'depth_mm': avg_depth_mm,
                    'confidence': confidence,
                    'is_depth_split': is_split,
                    'split_index': sub_idx if is_split else -1,
                })

        # === CROSS-CLASS NMS ===
        if self.use_cross_nms and len(detections) > 1:
            nms_iou = dpg.get_value("exp_nms_iou") if dpg.does_item_exist("exp_nms_iou") else DUPLICATE_MASK_IOU_THRESHOLD
            detections = cross_class_mask_nms(detections, nms_iou)

        # === WATERSHED JOINING ===
        if self.use_watershed and len(detections) > 1:
            detections = watershed_join_masks(detections, depth_image_raw, color_image)
            # Recalc belt coords for joined
            for det in detections:
                if det.get('is_watershed_joined'):
                    cx, cy = det['centroid']
                    det['belt_x_cm'], det['belt_y_cm'] = self.detector.pixel_to_belt_cm(cx, cy)
                    det['height_cm'] = self.detector.get_height_at_point(depth_frame, cx, cy)

        return detections

    # ------------------------------------------------------------------
    # STACKING / ADJACENT ANALYSIS
    # ------------------------------------------------------------------
    def _compute_stacking_labels(self, detections):
        """
        Compute pairwise stacking / adjacent relationships from detection masks.
        Uses the same algorithm as tracker._detect_mask_stack_groups() but
        operates on raw detection dicts (no tracker state needed).

        Populates:
            self.stack_labels   — {det_idx: 'TOP'|'BOT'|'CLOSE'}
            self.stack_pairs    — [(idx_a, idx_b, iou, stack_type), ...]
            self.stack_group_count, self.stacking_count, self.adjacent_count
        """
        self.stack_labels = {}
        self.stack_pairs  = []
        self.stack_group_count = 0
        self.stacking_count = 0
        self.adjacent_count = 0

        n = len(detections)
        if n < 2:
            return

        # Read live slider values
        dilate_px = dpg.get_value("exp_stack_dilate") if dpg.does_item_exist("exp_stack_dilate") else STACK_MASK_DILATE_PX
        adj_iou   = dpg.get_value("exp_stack_adj_iou") if dpg.does_item_exist("exp_stack_adj_iou") else STACK_MASK_IOU_THRESHOLD
        stk_iou   = dpg.get_value("exp_stack_stk_iou") if dpg.does_item_exist("exp_stack_stk_iou") else STACK_IOU_STACKING_MIN

        # Dilate masks
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        dilated = []
        for det in detections:
            m = det.get('mask')
            if m is not None and m.any():
                dilated.append(cv2.dilate(m, kernel, iterations=1))
            else:
                dilated.append(None)

        # Union-Find for group counting
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Pairwise analysis
        for i in range(n):
            if dilated[i] is None:
                continue
            for j in range(i + 1, n):
                if dilated[j] is None:
                    continue

                inter = np.count_nonzero(dilated[i] & dilated[j])
                union_px = np.count_nonzero(dilated[i] | dilated[j])
                if union_px == 0:
                    continue
                iou = inter / union_px

                if iou < adj_iou:
                    continue  # unrelated

                hi = detections[i].get('height_cm', 0) or 0
                hj = detections[j].get('height_cm', 0) or 0

                if iou >= stk_iou:
                    # STACKING — one on top of the other
                    stack_type = 'physical_stack'
                    self.stacking_count += 1
                    if hi >= hj:
                        self.stack_labels[i] = 'TOP'
                        self.stack_labels.setdefault(j, 'BOT')
                    else:
                        self.stack_labels[j] = 'TOP'
                        self.stack_labels.setdefault(i, 'BOT')
                else:
                    # ADJACENT — close but not stacked
                    stack_type = 'adjacent'
                    self.adjacent_count += 1
                    self.stack_labels.setdefault(i, 'CLOSE')
                    self.stack_labels.setdefault(j, 'CLOSE')

                union(i, j)
                self.stack_pairs.append((i, j, iou, stack_type))

        # Count groups (groups with ≥2 members)
        groups = {}
        for idx in range(n):
            r = find(idx)
            groups.setdefault(r, []).append(idx)
        self.stack_group_count = sum(1 for g in groups.values() if len(g) > 1)

    # ------------------------------------------------------------------
    # DRAW OVERLAY
    # ------------------------------------------------------------------
    def _draw_overlay(self, frame, detections):
        """Draw detection overlays on a BGR frame.  Returns annotated BGR frame."""
        overlay = frame.copy()
        alpha = dpg.get_value("exp_mask_alpha") if dpg.does_item_exist("exp_mask_alpha") else self.mask_alpha

        for det in detections:
            cx, cy = det['centroid']
            cls_id = det['class_id']
            color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
            mask = det.get('mask')

            # ── mask fill ──
            if self.show_masks and mask is not None:
                colored = np.zeros_like(overlay)
                colored[mask > 0] = color
                region = mask > 0
                overlay[region] = cv2.addWeighted(
                    overlay[region], 1 - alpha,
                    colored[region], alpha, 0)

                # Watershed boundary
                ws_bnd = det.get('watershed_boundary')
                if ws_bnd is not None and ws_bnd.any():
                    bnd = ws_bnd > 0
                    overlay[bnd] = cv2.addWeighted(
                        overlay[bnd], 1 - WATERSHED_BOUNDARY_ALPHA,
                        np.full_like(overlay[bnd], [255, 255, 255], dtype=np.uint8),
                        WATERSHED_BOUNDARY_ALPHA, 0)

            # ── contours ──
            if self.show_contours and det.get('contour') is not None:
                cv2.drawContours(overlay, [det['contour']], -1, (255, 255, 255), 2)
                cv2.drawContours(overlay, [det['contour']], -1, color, 1)

            # ── centroid ──
            if self.show_centroids:
                cv2.circle(overlay, (cx, cy), 5, (255, 255, 255), -1)
                cv2.circle(overlay, (cx, cy), 4, color, -1)

            # ── label ──
            if self.show_labels:
                cls_name = CLASS_NAMES.get(cls_id, f"C{cls_id}")
                conf = det.get('confidence', 0)
                label = f"{cls_name} {conf:.0%}"
                if self.show_height and det.get('height_cm', 0) > 0:
                    label += f"  H:{det['height_cm']:.1f}cm"
                if det.get('is_depth_split'):
                    label += "  [SPLIT]"
                if det.get('is_watershed_joined'):
                    label += f"  [WS x{det.get('joined_count', 2)}]"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                lx, ly = cx + 8, cy - 10
                cv2.rectangle(overlay, (lx - 2, ly - th - 4), (lx + tw + 4, ly + 4), (0, 0, 0), -1)
                cv2.putText(overlay, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ── stacking / adjacent labels + connecting lines ──
        if self.show_stacking and self.stack_labels:
            # Draw connecting lines between related pairs
            for idx_a, idx_b, iou, stype in self.stack_pairs:
                ca = detections[idx_a]['centroid']
                cb = detections[idx_b]['centroid']
                if stype == 'physical_stack':
                    line_color = (0, 0, 255)    # Red — stacking
                else:
                    line_color = (0, 180, 255)  # Orange — adjacent
                cv2.line(overlay, ca, cb, (255, 255, 255), 3)  # white outline
                cv2.line(overlay, ca, cb, line_color, 2)
                # IoU label at midpoint
                mx = (ca[0] + cb[0]) // 2
                my = (ca[1] + cb[1]) // 2
                iou_txt = f"{iou:.1%}"
                (tw2, th2), _ = cv2.getTextSize(iou_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(overlay, (mx - 2, my - th2 - 2), (mx + tw2 + 2, my + 2), (0, 0, 0), -1)
                cv2.putText(overlay, iou_txt, (mx, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, line_color, 1, cv2.LINE_AA)

            # Draw TOP / BOT / CLOSE labels centered on each detection
            label_colors = {
                'TOP': (0, 255, 255),    # Yellow
                'BOT': (0, 100, 255),    # Orange-red
                'CLOSE': (255, 200, 100) # Light blue
            }
            for det_idx, slabel in self.stack_labels.items():
                if det_idx >= len(detections):
                    continue
                det = detections[det_idx]
                contour = det.get('contour')
                if contour is None:
                    continue
                bx, by, bw, bh = cv2.boundingRect(contour)
                diag = (bw * bw + bh * bh) ** 0.5
                font_scale = max(0.4, min(1.0, diag / 120.0))
                thickness = 2 if font_scale >= 0.6 else 1
                (tw3, th3), _ = cv2.getTextSize(slabel, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                tx = bx + (bw - tw3) // 2
                ty = by + (bh + th3) // 2
                lc = label_colors.get(slabel, (255, 255, 255))
                cv2.putText(overlay, slabel, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 2)
                cv2.putText(overlay, slabel, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, lc, thickness)

        # ── ROI overlay ──
        if self.show_roi and self.detector.roi_corners is not None:
            cv2.polylines(overlay, [self.detector.roi_corners.astype(np.int32)],
                          True, (0, 255, 0), 2)

        # ── stats text ──
        n = len(detections)
        cv2.putText(overlay, f"Detections: {n}", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ── separation pipeline status bar ──
        tags = []
        if self.use_depth_cluster:
            tags.append("DC")
        if self.use_cross_nms:
            tags.append("NMS")
        if self.use_watershed:
            tags.append("WS")
        if self.show_stacking:
            tags.append("STACK")
        pipeline_str = "Pipeline: " + (" > ".join(tags) if tags else "RAW (no post-processing)")
        cv2.putText(overlay, pipeline_str, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        return overlay

    # ------------------------------------------------------------------
    # FRAME LOOP
    # ------------------------------------------------------------------
    _last_display = None  # for screenshot

    def _update_frame(self):
        """Called every DPG frame."""
        if not self.running:
            return

        # ── get frame (or use frozen) ──
        if self.frozen:
            color_image = self.frozen_color
            depth_frame = self.frozen_depth_frame
        else:
            result = self.camera.get_latest()
            if result is None:
                return
            color_image, depth_frame, timestamp = result
            self.frozen_color = color_image.copy()
            self.frozen_depth_frame = depth_frame

        if color_image is None or depth_frame is None:
            return

        # ── detect ──
        t0 = time.time()
        detections = self._detect_with_settings(color_image, depth_frame)
        t1 = time.time()
        infer_ms = (t1 - t0) * 1000
        self.infer_ms_history.append(infer_ms)
        self.last_detections = detections

        # ── FPS ──
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        fps = 1.0 / max(dt, 1e-6)
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history)
        avg_infer = np.mean(self.infer_ms_history)

        dpg.set_value("exp_fps", f"FPS: {avg_fps:.1f}")
        dpg.set_value("exp_infer", f"Infer: {avg_infer:.0f} ms")
        dpg.set_value("exp_det_count", f"Detections: {len(detections)}")

        # ── per-class counts ──
        cls_counts = {cid: 0 for cid in CLASS_NAMES}
        for det in detections:
            cid = det['class_id']
            if cid in cls_counts:
                cls_counts[cid] += 1
        for cid, cnt in cls_counts.items():
            tag = f"exp_cls_{cid}"
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, str(cnt))

        # ── stacking / adjacent analysis ──
        if self.show_stacking:
            self._compute_stacking_labels(detections)
        else:
            self.stack_labels = {}
            self.stack_pairs = []
            self.stack_group_count = 0
            self.stacking_count = 0
            self.adjacent_count = 0

        # Update stacking stats in UI
        if dpg.does_item_exist("exp_stack_groups"):
            dpg.set_value("exp_stack_groups", str(self.stack_group_count))
        if dpg.does_item_exist("exp_stack_stacking"):
            dpg.set_value("exp_stack_stacking", str(self.stacking_count))
        if dpg.does_item_exist("exp_stack_adjacent"):
            dpg.set_value("exp_stack_adjacent", str(self.adjacent_count))

        # ── draw color overlay ──
        display = self._draw_overlay(color_image, detections)
        self._last_display = display

        # BGR → RGBA float32 for DPG
        rgba = cv2.cvtColor(display, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
        dpg.set_value("exp_tex_color", rgba.flatten())

        # ── depth colormap ──
        if self.show_depth:
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cmap = COLORMAP_OPTIONS[self.colormap_idx]
            depth_color = cv2.applyColorMap(depth_norm, cmap)
            depth_rgba = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
            dpg.set_value("exp_tex_depth", depth_rgba.flatten())

        self.frame_count += 1

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------
    def run(self):
        """Main loop."""
        self._log("Detection Experiment ready.  Press [Start] to begin.")
        while dpg.is_dearpygui_running():
            self._update_frame()
            dpg.render_dearpygui_frame()

        # Cleanup
        self.running = False
        try:
            self.camera.running = False
            self.camera.pipeline.stop()
        except Exception:
            pass
        dpg.destroy_context()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = DetectionExperiment()
    app.run()
