#!/usr/bin/env python3
"""
=============================================================================
INDEX.PY - WASTE SORTING SYSTEM
=============================================================================
Main application that combines:
- RealSense camera + YOLO detection (better detection from realsense_depth_segmentation.py)
- Simple spatial tracking with picking queue
- Robot control (Delta + Slider)
- Spectrum sensor for material verification
- Dear PyGui UI
Author: Pipeline Team
Date: 2026
=============================================================================
"""

import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import time
import queue
import threading
import json
import os
import glob
import csv
import random
import uuid
import pyrealsense2 as rs
from datetime import datetime, timedelta, timezone
import psutil

try:
    from jtop import jtop
except Exception:
    jtop = None

# Import modules
from modules.config import *
from modules.tracker import SimpleTracker
from modules.camera import (
    CameraStream,
    VideoPlaybackStream,
    MP4PlaybackStream,
    MP4LiveDepthPlaybackStream,
)
from modules.detector import ObjectDetector
from modules.robot import DeltaController, SliderController, RobotManager
from modules.spectrum import SpectrumManager
from modules.database import DatabaseManager, DetectionLogger
from modules.robot import bilinear_interpolate, inverse_bilinear_interpolate


class SortingApp:
    """Main sorting application with Dear PyGui UI."""

    def _get_belt_speed(self):
        """Return the effective belt speed based on the dynamic estimation checkbox.

        If 'Dynamic Speed Estimation' is ON  → prefer vision-measured speed,
        fall back to the UI slider value.
        If OFF → always use the fixed UI slider value.
        """
        ui_speed = (
            dpg.get_value("in_speed")
            if dpg.does_item_exist("in_speed")
            else CONVEYOR_SPEED_CM_S
        )
        use_dynamic = (
            dpg.get_value("chk_dynamic_speed")
            if dpg.does_item_exist("chk_dynamic_speed")
            else True
        )
        if use_dynamic:
            return self.tracker.measured_belt_speed or ui_speed
        return ui_speed

    def _process_picking(self):
        """
        Smart dispatch with height-aware stacking detection + direct pick.
        
        Flow:
        1. Wait until robot is IDLE
        2. Use get_smart_pick() to find the best target:
           - Detects stacked objects (close X/Y, different heights)
           - Picks tallest first (it's on top)
           - Among equal height: picks most urgent (closest to exit)
        3. Dispatch to robot — robot predicts where the object will be
           and sends a single direct pick command (no slow tracking)
        4. After pick completes, robot is idle -> loop finds next target immediately
        """
        if not hasattr(self, 'robot_manager') or self.robot_manager is None:
            return
        if not self.delta.connected:
            return
        
        # Only dispatch when robot is IDLE
        if not self.robot_manager.is_idle():
            return
        
        # Get belt speed and offsets from UI
        approach_time = dpg.get_value("in_approach_time") if dpg.does_item_exist("in_approach_time") else 0.5
        
        # Use dynamic or fixed belt speed based on checkbox
        effective_speed = self._get_belt_speed()
        
        # Update offsets on robot manager (use effective speed for prediction)
        self.robot_manager.set_offsets(
            x=dpg.get_value("off_x"),
            y=dpg.get_value("off_y"),
            z=dpg.get_value("off_z"),
            latency=dpg.get_value("off_lat"),
            belt_speed=effective_speed
        )
        self.robot_manager.robot_approach_time_s = approach_time
        
        # Push per-class Z offsets from UI to robot manager
        for cls_name in ["Glass", "Metal", "Paper", "Plastic"]:
            tag = f"czoff_{cls_name}"
            if dpg.does_item_exist(tag):
                self.robot_manager.class_z_offsets[cls_name] = dpg.get_value(tag)
        
        # Push stack-bottom offsets from UI
        if dpg.does_item_exist("sbot_y_advance"):
            self.robot_manager.stack_bottom_y_advance_cm = dpg.get_value("sbot_y_advance")
        if dpg.does_item_exist("sbot_z_extra"):
            self.robot_manager.stack_bottom_z_extra_mm = dpg.get_value("sbot_z_extra")
        
        # === SMART PICK: height-aware stacking detection ===
        pick_data = self.tracker.get_smart_pick(belt_speed_cm_s=effective_speed)
        if pick_data is None:
            return
        
        obj_id = pick_data['id']
        is_stacked = pick_data.get('is_stacked', False)
        
        replay_row = None
        if getattr(self, "_loaded_video_kind", None) == "mp4" and getattr(self, "replay_enabled", False):
            replay_row = self._next_replay_pick_row()

        pick_x = float(pick_data.get('belt_x', 10.0))
        pick_y = float(pick_data.get('belt_y', 10.0))
        pick_h = float(pick_data.get('height_cm', 0.0))
        if replay_row is not None:
            if replay_row.get('belt_x_cm') is not None:
                pick_x = float(replay_row['belt_x_cm'])
            if replay_row.get('ws_y_cm') is not None:
                pick_y = float(replay_row['ws_y_cm'])
            if replay_row.get('height_cm') is not None:
                pick_h = float(replay_row['height_cm'])

        self.log(f"[DISPATCH] ID:{obj_id} ({pick_data['class_name']}) "
                 f"X={pick_x:.1f}, ws_Y={pick_y:.1f}cm, "
                 f"H={pick_h:.1f}cm"
                 f"{' [STACKED-TOP]' if is_stacked else ''}")
        
        # Build task
        is_bottom = pick_data.get('is_stack_bottom', False)
        task = {
            'id': obj_id,
            'class_name': pick_data.get('class_name', 'Unknown'),
            'confidence': pick_data.get('confidence', 0),
            'belt_x': pick_x,
            'belt_y': pick_y,
            'height': pick_h,
            'angle': pick_data.get('angle', 0),
            'is_stack_bottom': is_bottom,
        }
        if replay_row is not None:
            task['replay_spectrum_raw'] = replay_row.get('spectrum_raw', [])
            task['replay_spectrum_class'] = replay_row.get('spectrum_class')
            task['replay_spectrum_conf'] = replay_row.get('spectrum_conf')
        
        # Mark object as being picked
        if obj_id in self.tracker.objects:
            self.tracker.objects[obj_id]['status'] = 'Picking'
        self.tracker.mark_picked(obj_id)
        
        # Store pick target for simulation green marker
        self.sim_pick_target = {
            'belt_x': task['belt_x'],
            'belt_y_ws': task['belt_y'],
            'belt_y_abs': task['belt_y'] + ROI_HEIGHT_CM + ROBOT_WORKSPACE_OFFSET_CM,
            'obj_id': obj_id,
            'class_name': task['class_name'],
            'dispatch_time': time.time(),
        }
        
        # Dispatch — robot predicts forward + picks directly
        self.robot_manager.add_task(task)
        self.log(f"Pick dispatched: ID {obj_id} ({task['class_name']}) "
                 f"H={task['height']:.1f}cm - direct predictive pick")

    def _feed_pick_tracking(self):
        """
        Feed real-time object position to robot during pick.
        
        Called every frame.  The robot reads _live_pos once at dispatch to
        get the freshest time-anchor position, then predicts forward and
        sends a single direct pick command.  We keep feeding so the robot
        has up-to-date data at that moment.
        """
        phase = self.robot_manager.get_pick_phase()
        if phase not in ('APPROACH', 'DESCEND'):
            return
        
        obj_id = self.robot_manager.get_pick_obj_id()
        if obj_id is None:
            return
        
        belt_speed = self._get_belt_speed()
        current_time = time.time()
        ws_entry_y = ROI_HEIGHT_CM + ROBOT_WORKSPACE_OFFSET_CM
        ws_depth = ROBOT_WORKSPACE_DEPTH_CM
        
        # Get real-time position using time-anchor (same logic as _process_tracking)
        obj = self.tracker.objects.get(obj_id)
        if obj is None:
            return
        
        belt_x = obj.get('last_known_x_cm', obj.get('belt_x_cm', 10.0))
        # Use the robust height: registered (frozen at queue time) > stable median > raw
        height_cm = (obj.get('registered_height_cm')
                     or obj.get('stable_height_cm')
                     or obj.get('height_cm')
                     or obj.get('obj_height_cm', 0))
        
        # Time-anchor prediction with measured speed (drift-free)
        # Uses exit anchor + measured speed when available (most accurate),
        # falls back to reg anchor + system median, then UI slider.
        real_belt_y = self.tracker._get_anchor_belt_y(obj, belt_speed_cm_s=belt_speed)
        
        # Convert to workspace-relative Y
        ws_y = real_belt_y - ws_entry_y
        ws_y = max(0, min(ws_depth, ws_y))
        
        # Feed position to robot thread
        self.robot_manager.feed_live_position(belt_x, ws_y, height_cm)

    def _draw_tracked_masks(self, frame):
        """Draw filled mask overlay + contour edges + live overlap labels.

        Filled masks are drawn with 0.3 alpha, tinted with the object's
        class colour.  Contour edges are drawn on top as thin outlines.

        Labels are computed every frame from actual mask overlap:
        - STACKING: IoU ≥ 10%  → one object is inside / on top of another
            Shows "TOP" on the taller object, "BOT" on the shorter one
            (uses depth-based height to decide who is on top)
        - ADJACENT: any overlap (IoU ≥ 2%) but < 10% → close, not stacked
            Shows "CLOSE" on both objects
        - No overlap → no label, just the contour edge

        Text auto-scales with contour size.  Color matches the object's class.
        """
        from modules.config import (STACK_MASK_IOU_THRESHOLD, STACK_IOU_STACKING_MIN,
                                    STACK_MASK_DILATE_PX)

        MASK_FILL_ALPHA = 0.30  # Translucent class-coloured fill

        overlay = frame.copy()

        # Dilation kernel — same as tracker uses for generous adjacency
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (STACK_MASK_DILATE_PX * 2 + 1, STACK_MASK_DILATE_PX * 2 + 1)
        )

        # --- Pass 1: collect masks and contours (no edge drawing) ---
        obj_data = {}  # obj_id -> {mask, dilated, contour, color, class_id, height, bbox}
        for obj_id, obj in self.tracker.objects.items():
            is_ghost = obj.get('ghost', False)

            mask = obj.get('mask')
            used_last_valid = False
            if mask is None and MASK_FADE_ENABLED:
                mask = obj.get('last_valid_mask')
                used_last_valid = True
            if mask is None or not mask.any():
                continue

            # ── Shift stale mask to follow belt movement ──
            # When using last_valid_mask, the pixel data is frozen at the
            # position where the object was last detected.  Translate the
            # mask downward (in pixel Y) by the belt distance travelled
            # since capture so it visually follows the object through
            # the exit zone and beyond.
            if used_last_valid:
                mask_belt_y = obj.get('last_valid_mask_belt_y', 0)
                current_belt_y = obj.get('belt_y_cm', mask_belt_y)
                dy_cm = current_belt_y - mask_belt_y
                if abs(dy_cm) > 0.1:
                    dy_px = dy_cm * self.detector.px_per_cm
                    h, w = mask.shape[:2]
                    M = np.float32([[1, 0, 0], [0, 1, dy_px]])
                    mask = cv2.warpAffine(mask, M, (w, h),
                                          flags=cv2.INTER_NEAREST,
                                          borderValue=0)
                    # If mask shifted entirely off-frame, skip
                    if not mask.any():
                        continue

            fade = obj.get('mask_alpha', 1.0) if MASK_FADE_ENABLED else 1.0
            if fade <= MASK_FADE_MIN_ALPHA:
                continue

            class_id = obj.get('class_id', -1)
            color = CLASS_COLORS.get(class_id, DEFAULT_COLOR)

            contour = obj.get('contour')
            if contour is None or used_last_valid:
                # Recompute contour from (possibly shifted) mask
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    contour = max(cnts, key=cv2.contourArea)
            # Smooth contour with polygon approximation for cleaner edges
            if contour is not None and len(contour) > 6:
                epsilon = 0.008 * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)

            # ── Filled mask overlay (class colour, 0.3 alpha) ──
            alpha = MASK_FILL_ALPHA * fade  # respect fade-in/out
            region = mask > 0
            if np.any(region):
                colored = np.zeros_like(overlay)
                colored[region] = color
                overlay[region] = cv2.addWeighted(
                    overlay[region], 1 - alpha,
                    colored[region], alpha, 0)

                # Contour edge on top of fill
                if contour is not None:
                    cv2.drawContours(overlay, [contour], -1, (255, 255, 255), 2)
                    cv2.drawContours(overlay, [contour], -1, color, 1)

                # Watershed boundary (white grid)
                ws_bnd = obj.get('watershed_boundary')
                if ws_bnd is not None and isinstance(ws_bnd, np.ndarray) and ws_bnd.any():
                    bnd = ws_bnd > 0
                    overlay[bnd] = cv2.addWeighted(
                        overlay[bnd], 0.65,
                        np.full_like(overlay[bnd], [255, 255, 255], dtype=np.uint8),
                        0.35, 0)

            if not is_ghost:
                height_cm = (obj.get('stable_height_cm')
                             or obj.get('height_cm')
                             or obj.get('obj_height_cm', 0)) or 0
                bbox = cv2.boundingRect(contour) if contour is not None else None
                dilated = cv2.dilate(mask, dilate_kernel, iterations=1)
                obj_data[obj_id] = {
                    'mask': mask, 'dilated': dilated,
                    'contour': contour, 'color': color,
                    'class_id': class_id, 'height': height_cm, 'bbox': bbox,
                }

        # --- Pass 2: compute live pairwise overlap ---
        ids = list(obj_data.keys())
        labels = {}  # obj_id -> label string ('TOP', 'BOT', 'CLOSE')
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                da, db = obj_data[a]['dilated'], obj_data[b]['dilated']

                # Use dilated masks so nearby (not yet touching) objects
                # are detected as adjacent
                inter = np.count_nonzero(da & db)
                union_px = np.count_nonzero(da | db)
                if union_px == 0:
                    continue
                iou = inter / union_px

                if iou < STACK_MASK_IOU_THRESHOLD:
                    continue  # no relationship

                ha, hb = obj_data[a]['height'], obj_data[b]['height']

                if iou >= STACK_IOU_STACKING_MIN:
                    # STACKING — object is inside / on top of another
                    if ha >= hb:
                        labels[a] = 'TOP'
                        labels.setdefault(b, 'BOT')
                    else:
                        labels[b] = 'TOP'
                        labels.setdefault(a, 'BOT')
                else:
                    # ADJACENT — close but not stacked (don't overwrite TOP/BOT)
                    labels.setdefault(a, 'CLOSE')
                    labels.setdefault(b, 'CLOSE')

        # --- Pass 3: draw class labels + status labels ---
        #   - Isolated objects   → class name at normal size (centered)
        #   - Adjacent/stacking  → status (TOP/BOT/CLOSE) large + class name small below
        for obj_id, d in obj_data.items():
            color = d['color']
            bbox = d['bbox']
            if bbox is None:
                continue
            bx, by, bw, bh = bbox
            class_id = d['class_id']
            class_name = CLASS_NAMES.get(class_id, '?')

            status = labels.get(obj_id)  # None for isolated

            # Scale font with contour size — clamp [0.35, 1.0]
            diag = (bw * bw + bh * bh) ** 0.5
            base_scale = max(0.35, min(1.0, diag / 120.0))

            if status:
                # ── STACKING / ADJACENT ──
                # Status label: primary (bigger)
                st_scale = base_scale
                st_thick = 2 if st_scale >= 0.6 else 1
                (stw, sth), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX,
                                                 st_scale, st_thick)
                stx = bx + (bw - stw) // 2
                sty = by + bh // 2  # slightly above center

                cv2.putText(overlay, status, (stx, sty),
                            cv2.FONT_HERSHEY_SIMPLEX, st_scale,
                            (255, 255, 255), st_thick + 2)
                cv2.putText(overlay, status, (stx, sty),
                            cv2.FONT_HERSHEY_SIMPLEX, st_scale,
                            color, st_thick)

                # Class name: secondary (smaller, below status)
                cls_scale = base_scale * 0.55
                cls_thick = 1
                (cw, ch), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX,
                                               cls_scale, cls_thick)
                cx = bx + (bw - cw) // 2
                cy = sty + sth + 4  # right below status text

                cv2.putText(overlay, class_name, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, cls_scale,
                            (255, 255, 255), cls_thick + 1)
                cv2.putText(overlay, class_name, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, cls_scale,
                            color, cls_thick)
            else:
                # ── ISOLATED ──
                # Class name only, normal size, centered
                cls_scale = base_scale * 0.7
                cls_thick = 2 if cls_scale >= 0.5 else 1
                (cw, ch), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX,
                                               cls_scale, cls_thick)
                cx = bx + (bw - cw) // 2
                cy = by + (bh + ch) // 2

                cv2.putText(overlay, class_name, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, cls_scale,
                            (255, 255, 255), cls_thick + 1)
                cv2.putText(overlay, class_name, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, cls_scale,
                            color, cls_thick)

        return overlay

    def _draw_roi_zones(self, frame):
        """Draw translucent filled ROI zones and registration line on the vision feed.

        Regions:
          Entry zone  — blue   (255, 180,  50)  objects approaching ROI
          ROI         — green  ( 50, 220,  80)  active detection zone
          Exit zone   — orange (  0, 140, 255)  objects leaving toward workspace
          Reg. line   — cyan   (255, 255,   0)  registration / queue commit line
        """
        ZONE_ALPHA = 0.1
        overlay = frame.copy()

        # Entry zone (blue-ish)
        if self.detector.entry_corners is not None:
            pts = self.detector.entry_corners.astype(np.int32)
            cv2.fillPoly(overlay, [pts], (255, 180, 50))

        # ROI (green)
        if self.detector.roi_corners is not None:
            pts = self.detector.roi_corners.astype(np.int32)
            cv2.fillPoly(overlay, [pts], (50, 220, 80))

        # Exit zone (orange)
        if self.detector.exit_corners is not None:
            pts = self.detector.exit_corners.astype(np.int32)
            cv2.fillPoly(overlay, [pts], (0, 140, 255))

        # Blend overlay onto frame
        cv2.addWeighted(overlay, ZONE_ALPHA, frame, 1 - ZONE_ALPHA, 0, frame)

        # Thin zone outlines (on top of the blend, fully opaque)
        if self.detector.entry_corners is not None:
            cv2.polylines(frame, [self.detector.entry_corners.astype(np.int32)],
                          True, (255, 180, 50), 1)
        if self.detector.roi_corners is not None:
            cv2.polylines(frame, [self.detector.roi_corners.astype(np.int32)],
                          True, (50, 220, 80), 1)
        if self.detector.exit_corners is not None:
            cv2.polylines(frame, [self.detector.exit_corners.astype(np.int32)],
                          True, (0, 140, 255), 1)

        # Registration line (cyan/yellow dashed)
        if self.detector.roi_corners is not None:
            reg_y_px = int(self.detector.entry_start_y_px
                           + REGISTRATION_LINE_CM * self.detector.px_per_cm)
            x_min = int(self.detector.x_min_px)
            x_max = int(self.detector.x_max_px)
            cv2.line(frame, (x_min, reg_y_px), (x_max, reg_y_px), (0, 255, 255), 2)
            cv2.putText(frame, "REG", (x_max + 4, reg_y_px + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        return frame

    def _draw_tracking(self, frame):
        """Draw pick-point centroids on ALL tracked objects.

        - Every detected object gets a small centroid dot (class-coloured)
          with its tracker ID label so the operator can see where the robot
          would aim.
        - The ACTIVE pick target (being picked or next-to-pick) gets a
          larger crosshair so it stands out clearly.
        - Ghost objects still have their centroids predicted (needed by
          pick tracking and simulation) but are drawn as a faint ring
          instead of a solid dot.
        """
        belt_speed = self._get_belt_speed()
        current_time = time.time()

        # Determine the active pick target (being picked or next-to-pick)
        pick_obj_id = None
        if hasattr(self, 'robot_manager') and self.robot_manager is not None:
            pick_obj_id = self.robot_manager.get_pick_obj_id()
        # If robot is idle, peek at next-to-pick from queue
        if pick_obj_id is None and hasattr(self, 'tracker'):
            peek = self.tracker.get_smart_pick(belt_speed_cm_s=belt_speed)
            if peek is not None:
                pick_obj_id = peek['id']

        for obj_id, obj in self.tracker.objects.items():
            is_ghost = obj.get('ghost', False)

            if is_ghost:
                # Still predict ghost position (needed by sim + pick tracking)
                belt_x = obj.get('last_known_x_cm', obj.get('belt_x_cm', 10.0))
                belt_y = self.tracker._get_anchor_belt_y(obj, belt_speed_cm_s=belt_speed)
                result = self.detector.belt_cm_to_pixel(belt_x, belt_y)
                if result is None:
                    continue
                px, py = result
                obj['centroid'] = (px, py)
                obj['smoothed_centroid'] = (px, py)
            else:
                centroid = obj.get('smoothed_centroid', obj.get('centroid', None))
                if centroid is None:
                    continue
                px, py = int(centroid[0]), int(centroid[1])

            if px < 0 or px >= IMAGE_WIDTH or py < 0 or py >= IMAGE_HEIGHT:
                continue

            class_id = obj.get('class_id', -1)
            bgr_color = CLASS_COLORS.get(class_id, DEFAULT_COLOR)

            is_pick_target = (obj_id == pick_obj_id)

            if is_pick_target:
                # ── Active pick target: large crosshair ──
                r = 14
                cv2.line(frame, (px - r, py), (px + r, py), (255, 255, 255), 3)
                cv2.line(frame, (px, py - r), (px, py + r), (255, 255, 255), 3)
                cv2.line(frame, (px - r, py), (px + r, py), bgr_color, 2)
                cv2.line(frame, (px, py - r), (px, py + r), bgr_color, 2)
                cv2.circle(frame, (px, py), 5, (255, 255, 255), -1)
                cv2.circle(frame, (px, py), 4, bgr_color, -1)
                # ID label above crosshair
                cv2.putText(frame, f"#{obj_id}", (px + 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2)
                cv2.putText(frame, f"#{obj_id}", (px + 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            bgr_color, 1)
            elif is_ghost:
                # ── Ghost: faint ring (no fill) ──
                cv2.circle(frame, (px, py), 6, bgr_color, 1)
            else:
                # ── Normal object: crosshair + ID ──
                r = 10
                cv2.line(frame, (px - r, py), (px + r, py), (255, 255, 255), 2)
                cv2.line(frame, (px, py - r), (px, py + r), (255, 255, 255), 2)
                cv2.line(frame, (px - r, py), (px + r, py), bgr_color, 1)
                cv2.line(frame, (px, py - r), (px, py + r), bgr_color, 1)
                cv2.circle(frame, (px, py), 2, bgr_color, -1)
                # ID label offset to the right
                cv2.putText(frame, f"#{obj_id}", (px + 8, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (255, 255, 255), 2)
                cv2.putText(frame, f"#{obj_id}", (px + 8, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            bgr_color, 1)

        return frame

    def _update_log_ui(self):
        """Update log display in the UI."""
        # Clear log child window
        dpg.delete_item("log_child", children_only=True)
        # Add log messages
        for timestamp, msg, level in self.log_messages[-30:]:  # Show last 30 messages
            color = (255, 255, 255)
            if level == "ERROR":
                color = (255, 100, 100)
            elif level == "SUCCESS":
                color = (100, 255, 100)
            dpg.add_text(f"[{timestamp}] {msg}", color=color, parent="log_child")

    def _update_stats(self):
        """Update sorting statistics display."""
        # Update statistics for each class
        for cls_name in CLASS_NAMES.values():
            count = self.sorted_counts.get(cls_name, 0)
            dpg.set_value(f"stat_{cls_name}", str(count))
        # Update overall detected count
        stats_text = f"Total Detected: {self.total_detected}"
        dpg.set_value("txt_stats", stats_text)

    def _init_timing_sessions(self):
        """Load saved timing runs and prepare a fresh run for the operator."""
        self.timing_log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "object_detect_lay_timing.csv",
        )
        self.timing_runs = []
        max_run_number = 0

        if os.path.exists(self.timing_log_path):
            grouped = {}
            try:
                with open(self.timing_log_path, "r", newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        run_id = row.get("run_id", "")
                        if not run_id:
                            continue
                        run = grouped.setdefault(run_id, {
                            "run_id": run_id,
                            "target_count": int(row.get("target_count", 1) or 1),
                            "objects": [],
                        })
                        detected_epoch = self._parse_timing_epoch(row.get("detected_at"))
                        laid_epoch = self._parse_timing_epoch(row.get("laid_at"))
                        run["objects"].append({
                            "index": int(row.get("object_index", 0) or 0),
                            "robot_object_id": row.get("robot_object_id", ""),
                            "class_name": row.get("camera_class", ""),
                            "detected_epoch": detected_epoch,
                            "laid_epoch": laid_epoch,
                        })
                        try:
                            max_run_number = max(
                                max_run_number,
                                int(run_id.rsplit("-", 1)[-1]),
                            )
                        except (TypeError, ValueError):
                            pass
                self.timing_runs = list(grouped.values())
            except Exception as exc:
                self.log(f"Could not load timing log: {exc}", "ERROR")

        self.timing_next_run_number = max_run_number + 1
        self.timing_target_count = 1
        self.timing_current_run = self._new_timing_run()
        self._save_timing_sessions()

    @staticmethod
    def _parse_timing_epoch(value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value).timestamp()
        except (TypeError, ValueError):
            return None

    def _new_timing_run(self):
        run = {
            "run_id": f"RUN-{self.timing_next_run_number:04d}",
            "target_count": self.timing_target_count,
            "objects": [],
        }
        self.timing_next_run_number += 1
        return run

    @staticmethod
    def _timing_clock_text(epoch):
        if epoch is None:
            return "--"
        return datetime.fromtimestamp(epoch).strftime("%H:%M:%S.%f")[:-3]

    @staticmethod
    def _timing_iso_text(epoch):
        if epoch is None:
            return ""
        return datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="milliseconds")

    def _timing_total_seconds(self, run):
        objects = run.get("objects", [])
        target_count = int(run.get("target_count", 1))
        if len(objects) < target_count:
            return None
        relevant = objects[:target_count]
        if not relevant or relevant[0].get("detected_epoch") is None:
            return None
        if any(obj.get("laid_epoch") is None for obj in relevant):
            return None
        return relevant[-1]["laid_epoch"] - relevant[0]["detected_epoch"]

    def _timing_run_status(self, run):
        objects = run.get("objects", [])
        target_count = int(run.get("target_count", 1))
        if self._timing_total_seconds(run) is not None:
            return "Complete"
        if objects:
            return f"In progress ({len(objects)}/{target_count} verified)"
        return "Waiting for first verification"

    def _save_timing_sessions(self):
        """Rewrite the compact CSV so deleting the active run is deterministic."""
        fieldnames = [
            "run_id", "target_count", "status", "object_index",
            "robot_object_id", "camera_class", "detected_at", "laid_at",
            "detect_to_lay_s", "total_time_s",
        ]
        runs = list(self.timing_runs)
        if self.timing_current_run.get("objects"):
            runs.append(self.timing_current_run)

        try:
            with open(self.timing_log_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for run in runs:
                    total_s = self._timing_total_seconds(run)
                    status = self._timing_run_status(run)
                    for obj in run.get("objects", []):
                        detected = obj.get("detected_epoch")
                        laid = obj.get("laid_epoch")
                        cycle_s = laid - detected if detected is not None and laid is not None else None
                        writer.writerow({
                            "run_id": run["run_id"],
                            "target_count": run["target_count"],
                            "status": status,
                            "object_index": obj.get("index", ""),
                            "robot_object_id": obj.get("robot_object_id", ""),
                            "camera_class": obj.get("class_name", ""),
                            "detected_at": self._timing_iso_text(detected),
                            "laid_at": self._timing_iso_text(laid),
                            "detect_to_lay_s": f"{cycle_s:.3f}" if cycle_s is not None else "",
                            "total_time_s": f"{total_s:.3f}" if total_s is not None else "",
                        })
        except Exception as exc:
            self.log(f"Could not save timing log: {exc}", "ERROR")

    def _on_timing_count_change(self, sender=None, app_data=None, user_data=None):
        value = app_data or (
            dpg.get_value("combo_timing_count")
            if dpg.does_item_exist("combo_timing_count")
            else "1 obj"
        )
        try:
            requested = int(str(value).split()[0])
        except (TypeError, ValueError):
            requested = 1

        current_objects = self.timing_current_run.get("objects", [])
        if current_objects:
            current = self.timing_current_run["target_count"]
            if dpg.does_item_exist("combo_timing_count"):
                dpg.set_value("combo_timing_count", f"{current} obj")
            self.log("Object count cannot change after verification has started", "ERROR")
            return

        self.timing_target_count = max(1, min(3, requested))
        self.timing_current_run["target_count"] = self.timing_target_count
        self._update_timing_ui()

    def _on_timing_next(self, sender=None, app_data=None, user_data=None):
        if self.timing_current_run.get("objects"):
            self.timing_runs.append(self.timing_current_run)
        self.timing_target_count = int(self.timing_current_run.get("target_count", 1))
        self.timing_current_run = self._new_timing_run()
        self._save_timing_sessions()
        self._update_timing_ui()
        self.log(f"Timing run changed to {self.timing_current_run['run_id']}", "SUCCESS")

    def _on_timing_delete(self, sender=None, app_data=None, user_data=None):
        run_id = self.timing_current_run["run_id"]
        self.timing_current_run = {
            "run_id": run_id,
            "target_count": self.timing_target_count,
            "objects": [],
        }
        self._save_timing_sessions()
        self._update_timing_ui()
        self.log(f"Timing data deleted for {run_id}")

    def _handle_timing_detected(self, payload):
        """Record camera verification at the registration line / queue commit."""
        run = self.timing_current_run
        robot_object_id = str(payload.get("obj_id", ""))
        existing = next(
            (obj for obj in run["objects"] if str(obj["robot_object_id"]) == robot_object_id),
            None,
        )
        if existing is not None:
            return
        if len(run["objects"]) >= run["target_count"]:
            self.log(
                f"Timing run {run['run_id']} is complete; press NEXT before the next object",
                "ERROR",
            )
            return

        run["objects"].append({
            "index": len(run["objects"]) + 1,
            "robot_object_id": robot_object_id,
            "class_name": payload.get("class_name", ""),
            "detected_epoch": float(payload.get("event_epoch_s", time.time())),
            "laid_epoch": None,
        })
        self._save_timing_sessions()
        self._update_timing_ui()

    def _record_new_queue_detections(self, queue_before):
        """Capture objects that crossed the registration line this frame."""
        for obj_id in self.tracker.picking_queue:
            if obj_id in queue_before:
                continue
            obj = self.tracker.objects.get(obj_id)
            if not obj or not obj.get("in_queue", False):
                continue
            self._handle_timing_detected({
                "obj_id": obj_id,
                "class_name": obj.get(
                    "registered_class_name",
                    obj.get("class_name", ""),
                ),
                "event_epoch_s": obj.get("reg_time", time.time()),
            })

    def _handle_timing_laid(self, payload):
        run = self.timing_current_run
        robot_object_id = str(payload.get("obj_id", ""))
        obj = next(
            (item for item in run["objects"] if str(item["robot_object_id"]) == robot_object_id),
            None,
        )
        if obj is None:
            for previous_run in reversed(self.timing_runs):
                previous_obj = next(
                    (
                        item for item in previous_run["objects"]
                        if str(item["robot_object_id"]) == robot_object_id
                        and item.get("laid_epoch") is None
                    ),
                    None,
                )
                if previous_obj is not None:
                    run = previous_run
                    obj = previous_obj
                    break
        if obj is None or obj.get("laid_epoch") is not None:
            return
        obj["laid_epoch"] = float(payload.get("event_epoch_s", time.time()))
        if payload.get("class_name"):
            obj["class_name"] = payload["class_name"]
        self._save_timing_sessions()
        if run is self.timing_current_run:
            self._update_timing_ui()

    def _update_timing_ui(self):
        if not dpg.does_item_exist("txt_timing_run_id"):
            return

        run = self.timing_current_run
        dpg.set_value("txt_timing_run_id", run["run_id"])
        dpg.set_value("txt_timing_status", self._timing_run_status(run))
        total_s = self._timing_total_seconds(run)
        dpg.set_value(
            "txt_timing_total",
            f"{total_s:.3f} s" if total_s is not None else "--",
        )

        for index in range(1, 4):
            row_tag = f"timing_row_{index}"
            dpg.configure_item(row_tag, show=index <= run["target_count"])
            obj = run["objects"][index - 1] if index <= len(run["objects"]) else None
            dpg.set_value(f"timing_obj_{index}_id", obj["robot_object_id"] if obj else "--")
            dpg.set_value(f"timing_obj_{index}_class", obj["class_name"] if obj else "--")
            detected = obj.get("detected_epoch") if obj else None
            laid = obj.get("laid_epoch") if obj else None
            dpg.set_value(f"timing_obj_{index}_detected", self._timing_clock_text(detected))
            dpg.set_value(f"timing_obj_{index}_laid", self._timing_clock_text(laid))
            cycle_s = laid - detected if detected is not None and laid is not None else None
            dpg.set_value(
                f"timing_obj_{index}_cycle",
                f"{cycle_s:.3f} s" if cycle_s is not None else "--",
            )

    def _update_video_tab_status(self):
        """Periodic sync for the Video & Recording tab status texts."""
        try:
            # Live-update .mp4 recording duration
            if self.mp4_recording:
                elapsed = self.record_frame_count / max(1.0, float(self._mp4_record_fps))
                dpg.set_value("txt_rec_mp4_status",
                              f"Recording: {os.path.basename(self.mp4_filename or '')}  "
                              f"({elapsed:.1f}s, {self.record_frame_count} frames)")
            # Live-update playback frame counter
            if self.video_source is not None:
                fc = getattr(self.video_source, 'frame_count', 0)
                lc = getattr(self.video_source, 'loop_count', 0)
                pass_idx = getattr(self.video_source, 'current_pass_index', lc + 1)
                pass_target = getattr(self.video_source, 'max_passes', None)
                name = os.path.basename(self._loaded_bag_path or '')
                if pass_target:
                    loop_txt = f"  (pass {pass_idx}/{pass_target})"
                else:
                    loop_txt = f"  (loop #{lc})" if lc > 0 else ""
                dpg.set_value("txt_vt_status",
                              f"> Playing - {name}  [{fc} frames{loop_txt}]")
        except Exception:
            pass
    
    def _process_ui_queue(self):
        """Drain the UI queue — robot thread posts results here.
        
        Messages:
          ('spectrum', raw_str)     — raw spectral data string from a pick scan
          ('sorted', class_name)    — pick completed, increment sorted counter
          ('fusion', info_str)      — fusion result summary (for log)
        """
        try:
            while not self.ui_queue.empty():
                msg = self.ui_queue.get_nowait()
                if not isinstance(msg, tuple) or len(msg) < 2:
                    continue
                msg_type = msg[0]
                payload = msg[1]

                if msg_type == 'spectrum':
                    # Update the Spectrum tab's raw data display only
                    if dpg.does_item_exist("txt_spectrum_raw"):
                        dpg.set_value("txt_spectrum_raw", payload)

                elif msg_type == 'spectrum_result':
                    # Update Vision tab's predicted class display
                    info = payload  # dict with yolo_class, yolo_conf, spectrum_class, spectrum_conf
                    yolo_str = f"{info['yolo_class']} ({info['yolo_conf']:.0f}%)"
                    spec_str = f"{info['spectrum_class']} ({info['spectrum_conf']:.0f}%)"
                    if dpg.does_item_exist("txt_spec_yolo"):
                        dpg.set_value("txt_spec_yolo", yolo_str)
                    if dpg.does_item_exist("txt_spec_pred"):
                        dpg.set_value("txt_spec_pred", spec_str)

                elif msg_type == '_manual_scan_result':
                    # Result from threaded manual spectrum scan
                    self._handle_manual_scan_result(payload)

                elif msg_type == 'sorted':
                    # Increment sorted counter for the class
                    class_name = payload
                    if class_name in self.sorted_counts:
                        self.sorted_counts[class_name] += 1
                    else:
                        self.sorted_counts[class_name] = 1
                    # Update Vision tab's final class display
                    if dpg.does_item_exist("txt_spec_final"):
                        dpg.set_value("txt_spec_final", class_name)
                        # Color-code by class
                        cls_id = next((k for k, v in CLASS_NAMES.items() if v == class_name), None)
                        color = CLASS_COLORS.get(cls_id, (255, 255, 0)) if cls_id is not None else (255, 255, 0)
                        dpg.configure_item("txt_spec_final", color=color)

                elif msg_type == 'timing_laid':
                    self._handle_timing_laid(payload)

                elif msg_type == 'fusion':
                    self.log(payload)

                elif msg_type == 'audit_row':
                    # Collect audit row when audit log is active
                    if self.audit_collecting:
                        self.audit_buffer.append(payload)
                        n = len(self.audit_buffer)
                        if dpg.does_item_exist("txt_audit_status"):
                            dpg.set_value("txt_audit_status",
                                          f"Collecting... {n} row{'s' if n != 1 else ''}")

                elif msg_type == 'pick_replay':
                    # Collect per-pick sync row while recording MP4.
                    if self.mp4_recording and self._mp4_record_start_epoch is not None:
                        try:
                            event_epoch = float(payload.get('event_epoch_s', time.time()))
                            video_time_s = max(0.0, event_epoch - float(self._mp4_record_start_epoch))
                            event_utc = datetime.fromtimestamp(
                                event_epoch, timezone.utc
                            ).isoformat().replace("+00:00", "Z")
                            row = {
                                'record_start_utc': self._mp4_record_start_utc or '',
                                'event_epoch_s': round(event_epoch, 6),
                                'event_utc': event_utc,
                                'video_time_s': round(video_time_s, 4),
                                'obj_id': payload.get('obj_id'),
                                'camera_class': payload.get('camera_class'),
                                'camera_conf': payload.get('camera_conf'),
                                'spectrum_class': payload.get('spectrum_class'),
                                'spectrum_conf': payload.get('spectrum_conf'),
                                'final_class': payload.get('final_class'),
                                'belt_x_cm': payload.get('belt_x_cm'),
                                'ws_y_cm': payload.get('ws_y_cm'),
                                'height_cm': payload.get('height_cm'),
                                'robot_x_mm': payload.get('robot_x_mm'),
                                'robot_y_mm': payload.get('robot_y_mm'),
                                'pick_z_mm': payload.get('pick_z_mm'),
                                'angle_deg': payload.get('angle_deg'),
                                'scan_w_deg': payload.get('scan_w_deg'),
                                'source': payload.get('source', 'live'),
                            }
                            raw = payload.get('spectrum_raw') or []
                            for i in range(18):
                                key = f'spectrum_raw_{i+1:02d}'
                                row[key] = raw[i] if i < len(raw) else ''
                            self._mp4_sync_rows.append(row)
                        except Exception:
                            pass

        except Exception:
            pass

    def _update_status_indicators(self):
        """Update status indicator dots in title bar."""
        # Camera status (green [V] if running, red [X] if not)
        if self.is_running and hasattr(self.camera, 'running') and self.camera.running:
            dpg.set_value("status_indicator_cam", "[V]")
            dpg.configure_item("status_indicator_cam", color=(0, 255, 100))
        else:
            dpg.set_value("status_indicator_cam", "[X]")
            dpg.configure_item("status_indicator_cam", color=(255, 80, 80))
        
        # Robot status (green [V] if connected, red [X] if not)
        if self.delta.connected:
            dpg.set_value("status_indicator_robot", "[V]")
            dpg.configure_item("status_indicator_robot", color=(0, 255, 100))
        else:
            dpg.set_value("status_indicator_robot", "[X]")
            dpg.configure_item("status_indicator_robot", color=(255, 80, 80))
        
        # Spectrum status (green [V] if ready, yellow [~] if initialized, red [X] if not)
        if self.spectrum.is_ready:
            dpg.set_value("status_indicator_spectrum", "[V]")
            dpg.configure_item("status_indicator_spectrum", color=(0, 255, 100))
        elif hasattr(self.spectrum, 'sensor') and self.spectrum.sensor is not None:
            dpg.set_value("status_indicator_spectrum", "[~]")
            dpg.configure_item("status_indicator_spectrum", color=(255, 200, 0))
        else:
            dpg.set_value("status_indicator_spectrum", "[X]")
            dpg.configure_item("status_indicator_spectrum", color=(255, 80, 80))
    
    def _update_video_tab_status(self):
        """Periodically sync the Video & Recording tab status texts."""
        try:
            # Live-update experiment recording timer
            if self.mp4_recording:
                elapsed = self.record_frame_count / max(1.0, float(self._mp4_record_fps))
                robot_tag = ""
                if self.robot_cam_connected and self.robot_cam_writer is not None:
                    robot_tag = f" | Robot: {self.robot_cam_frame_count}f"
                dpg.set_value("txt_rec_mp4_status",
                              f"Recording: {os.path.basename(self.mp4_filename or '')} "
                              f"({elapsed:.1f}s, {self.record_frame_count}f{robot_tag})")

            # Live-update playback frame counter
            if self.video_source is not None and hasattr(self.video_source, 'frame_count'):
                vs = self.video_source
                pass_idx = getattr(vs, "current_pass_index", getattr(vs, "loop_count", 0) + 1)
                pass_target = getattr(vs, "max_passes", None)
                if pass_target:
                    loop_tag = f" | Pass {pass_idx}/{pass_target}"
                else:
                    loop_tag = f" | Loop #{vs.loop_count}" if vs.loop_count > 0 else ""
                replay_tag = ""
                if self.replay_enabled:
                    replay_tag = f" | Replay {self.replay_pick_idx}/{len(self.replay_pick_rows)}"
                dpg.set_value("txt_vt_status",
                              f"> Playing - {vs.frame_count} frames{loop_tag}{replay_tag}")
                dpg.configure_item("txt_vt_status", color=(0, 255, 100))

            # Robot cam connection status
            if self.robot_cam_connected:
                dpg.set_value("txt_robot_cam_conn", "Connected")
                dpg.configure_item("txt_robot_cam_conn", color=(0, 255, 100))
            else:
                dpg.set_value("txt_robot_cam_conn", "Disconnected")
                dpg.configure_item("txt_robot_cam_conn", color=(255, 80, 80))

            if dpg.does_item_exist("txt_benchmark_status"):
                if self.benchmark_active:
                    mode = "REC" if self.benchmark_mode == "rec_mp4" else "PLAY"
                    name = os.path.basename(self.benchmark_csv_path or "")
                    dpg.set_value("txt_benchmark_status", f"Benchmark {mode}: {name}")
                    dpg.configure_item("txt_benchmark_status", color=(255, 180, 100))
                else:
                    dpg.set_value("txt_benchmark_status", "Benchmark: idle")
                    dpg.configure_item("txt_benchmark_status", color=(120, 120, 120))
        except Exception:
            pass

    def _is_benchmark_enabled(self):
        if dpg.does_item_exist("chk_benchmark_enable"):
            try:
                return bool(dpg.get_value("chk_benchmark_enable"))
            except Exception:
                return bool(self.benchmark_enabled_default)
        return bool(self.benchmark_enabled_default)

    @staticmethod
    def _bench_to_float(v):
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(str(v).strip())
        except Exception:
            return None

    @staticmethod
    def _bench_p95(values):
        if not values:
            return None
        arr = sorted(values)
        idx = max(0, min(len(arr) - 1, int(len(arr) * 0.95 + 0.5) - 1))
        return arr[idx]

    @staticmethod
    def _bench_mean(values):
        if not values:
            return None
        return sum(values) / float(len(values))

    @staticmethod
    def _bench_first_number(obj):
        if isinstance(obj, bool):
            return None
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, str):
            try:
                return float(obj)
            except Exception:
                return None
        if isinstance(obj, dict):
            for v in obj.values():
                n = SortingApp._bench_first_number(v)
                if n is not None:
                    return n
            return None
        if isinstance(obj, (list, tuple)):
            for v in obj:
                n = SortingApp._bench_first_number(v)
                if n is not None:
                    return n
            return None
        return None

    @staticmethod
    def _bench_estimate_power_w(stats):
        if not isinstance(stats, dict):
            return None
        preferred = [
            "Power TOT",
            "Power TOT POWER",
            "Power Total",
            "power_total",
        ]
        value = None
        for key in preferred:
            if key in stats:
                value = SortingApp._bench_first_number(stats.get(key))
                if value is not None:
                    break
        if value is None:
            for key, v in stats.items():
                lk = str(key).lower()
                if "power" in lk and "avg" not in lk and "current" not in lk:
                    value = SortingApp._bench_first_number(v)
                    if value is not None:
                        break
        if value is None:
            return None
        if value > 500.0:
            return value / 1000.0
        return value

    @staticmethod
    def _bench_items(obj):
        if obj is None:
            return []
        if hasattr(obj, "items"):
            try:
                return list(obj.items())
            except Exception:
                return []
        return []

    @staticmethod
    def _bench_is_mapping(obj):
        return obj is not None and hasattr(obj, "get") and hasattr(obj, "items")

    @staticmethod
    def _bench_kb_to_mb(value):
        value = SortingApp._bench_to_float(value)
        if value is None:
            return None
        return value / 1024.0

    @staticmethod
    def _bench_safe_pct(numerator, denominator):
        numerator = SortingApp._bench_to_float(numerator)
        denominator = SortingApp._bench_to_float(denominator)
        if numerator is None or denominator is None or denominator <= 0:
            return None
        return (numerator / denominator) * 100.0

    @staticmethod
    def _bench_safe_div(numerator, denominator):
        numerator = SortingApp._bench_to_float(numerator)
        denominator = SortingApp._bench_to_float(denominator)
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    @staticmethod
    def _bench_bool_to_int(value):
        if value is None or value == "":
            return ""
        return 1 if bool(value) else 0

    @staticmethod
    def _bench_find_metric_by_name(mapping, needles):
        for key, value in SortingApp._bench_items(mapping):
            lk = str(key).lower()
            if any(token in lk for token in needles):
                if isinstance(value, dict):
                    metric = SortingApp._bench_to_float(value.get("temp"))
                    if metric is not None and metric > -200:
                        return metric
                metric = SortingApp._bench_to_float(value)
                if metric is not None and metric > -200:
                    return metric
        return None

    @staticmethod
    def _bench_build_fieldnames():
        return [
            "run_id",
            "benchmark_version",
            "sample_idx",
            "timestamp_epoch",
            "timestamp_utc",
            "record_start_utc",
            "elapsed_s",
            "interval_s",
            "benchmark_mode",
            "source_kind",
            "source_name",
            "playback_loop_index",
            "playback_loop_target",
            "device_model",
            "jetpack_version",
            "l4t_version",
            "nvpmodel_mode",
            "jetson_clocks_enabled",
            "gpu_util_pct",
            "gpu_freq_khz",
            "gpu_freq_max_khz",
            "gpu_gpc_freq_mean_khz",
            "gpu_mem_used_mb",
            "gpu_headroom_pct",
            "cpu_util_mean_pct",
            "cpu_util_peak_pct",
            "cpu_freq_mean_khz",
            "cpu_freq_peak_khz",
            "cpu_cores_online_count",
            "cpu_app_util_pct",
            "cpu_headroom_pct",
            "ram_util_pct",
            "ram_used_mb",
            "ram_free_mb",
            "ram_shared_mb",
            "ram_cache_mb",
            "ram_headroom_pct",
            "swap_util_pct",
            "emc_util_pct",
            "emc_freq_khz",
            "power_total_w",
            "power_total_mean_w",
            "power_total_p95_w",
            "power_top_rail_1_name",
            "power_top_rail_1_w",
            "power_top_rail_2_name",
            "power_top_rail_2_w",
            "power_top_rail_3_name",
            "power_top_rail_3_w",
            "temp_cpu_c",
            "temp_gpu_c",
            "temp_max_c",
            "infer_fps_mean",
            "infer_fps_p95",
            "infer_fps_min",
            "infer_latency_mean_ms",
            "infer_latency_p95_ms",
            "infer_latency_max_ms",
            "infer_det_mean",
            "infer_frames_count",
            "eff_fps_per_w",
            "eff_det_per_w",
            "eff_latency_per_w",
        ]

    @staticmethod
    def _bench_extract_cpu_metrics(cpu_info):
        metrics = {
            "cpu_util_mean_pct": "",
            "cpu_util_peak_pct": "",
            "cpu_freq_mean_khz": "",
            "cpu_freq_peak_khz": "",
            "cpu_cores_online_count": "",
            "cpu_headroom_pct": "",
        }
        if not SortingApp._bench_is_mapping(cpu_info):
            return metrics
        cpu_cores = cpu_info.get("cpu") or []
        busy_vals = []
        freq_vals = []
        online_count = 0
        for core in cpu_cores:
            if not isinstance(core, dict):
                continue
            if core.get("online"):
                online_count += 1
            user = SortingApp._bench_to_float(core.get("user")) or 0.0
            nice = SortingApp._bench_to_float(core.get("nice")) or 0.0
            system = SortingApp._bench_to_float(core.get("system")) or 0.0
            idle = SortingApp._bench_to_float(core.get("idle"))
            busy = user + nice + system
            if busy <= 0.0 and idle is not None:
                busy = max(0.0, 100.0 - idle)
            if core.get("online") and (busy > 0.0 or idle is not None):
                busy_vals.append(busy)
            freq = ((core.get("freq") or {}).get("cur")
                    if isinstance(core.get("freq"), dict) else None)
            freq = SortingApp._bench_to_float(freq)
            if core.get("online") and freq is not None and freq > 0:
                freq_vals.append(freq)
        util_mean = SortingApp._bench_mean(busy_vals)
        metrics["cpu_util_mean_pct"] = util_mean if util_mean is not None else ""
        metrics["cpu_util_peak_pct"] = max(busy_vals) if busy_vals else ""
        metrics["cpu_freq_mean_khz"] = SortingApp._bench_mean(freq_vals) if freq_vals else ""
        metrics["cpu_freq_peak_khz"] = max(freq_vals) if freq_vals else ""
        metrics["cpu_cores_online_count"] = online_count if online_count > 0 else ""
        metrics["cpu_headroom_pct"] = max(0.0, 100.0 - util_mean) if util_mean is not None else ""
        return metrics

    @staticmethod
    def _bench_extract_memory_metrics(memory_info):
        metrics = {
            "ram_util_pct": "",
            "ram_used_mb": "",
            "ram_free_mb": "",
            "ram_shared_mb": "",
            "ram_cache_mb": "",
            "ram_headroom_pct": "",
            "swap_util_pct": "",
            "emc_util_pct": "",
            "emc_freq_khz": "",
        }
        if not SortingApp._bench_is_mapping(memory_info):
            return metrics
        ram = memory_info.get("RAM") if SortingApp._bench_is_mapping(memory_info.get("RAM")) else {}
        swap = memory_info.get("SWAP") if SortingApp._bench_is_mapping(memory_info.get("SWAP")) else {}
        emc = memory_info.get("EMC") if SortingApp._bench_is_mapping(memory_info.get("EMC")) else {}
        ram_util = SortingApp._bench_safe_pct(ram.get("used"), ram.get("tot"))
        metrics["ram_util_pct"] = ram_util if ram_util is not None else ""
        ram_used_mb = SortingApp._bench_kb_to_mb(ram.get("used"))
        ram_free_mb = SortingApp._bench_kb_to_mb(ram.get("free"))
        ram_shared_mb = SortingApp._bench_kb_to_mb(ram.get("shared"))
        ram_cache_mb = SortingApp._bench_kb_to_mb(ram.get("cached"))
        metrics["ram_used_mb"] = ram_used_mb if ram_used_mb is not None else ""
        metrics["ram_free_mb"] = ram_free_mb if ram_free_mb is not None else ""
        metrics["ram_shared_mb"] = ram_shared_mb if ram_shared_mb is not None else ""
        metrics["ram_cache_mb"] = ram_cache_mb if ram_cache_mb is not None else ""
        metrics["ram_headroom_pct"] = max(0.0, 100.0 - ram_util) if ram_util is not None else ""
        swap_util = SortingApp._bench_safe_pct(swap.get("used"), swap.get("tot"))
        metrics["swap_util_pct"] = swap_util if swap_util is not None else ""
        emc_util = SortingApp._bench_to_float(emc.get("val"))
        metrics["emc_util_pct"] = emc_util if emc_util is not None else ""
        emc_freq = SortingApp._bench_to_float(emc.get("cur"))
        metrics["emc_freq_khz"] = emc_freq if emc_freq is not None else ""
        return metrics

    @staticmethod
    def _bench_extract_gpu_metrics(gpu_info, process_gpu_mem_mb=None):
        metrics = {
            "gpu_util_pct": "",
            "gpu_freq_khz": "",
            "gpu_freq_max_khz": "",
            "gpu_gpc_freq_mean_khz": "",
            "gpu_mem_used_mb": process_gpu_mem_mb if process_gpu_mem_mb is not None else "",
            "gpu_headroom_pct": "",
        }
        gpu_entries = [value for _, value in SortingApp._bench_items(gpu_info)
                       if SortingApp._bench_is_mapping(value)]
        if not gpu_entries:
            return metrics
        gpu_entry = gpu_entries[0]
        status = gpu_entry.get("status") if SortingApp._bench_is_mapping(gpu_entry.get("status")) else {}
        freq = gpu_entry.get("freq") if SortingApp._bench_is_mapping(gpu_entry.get("freq")) else {}
        gpu_util = SortingApp._bench_to_float(status.get("load"))
        gpc_vals = []
        for item in freq.get("GPC", []) if isinstance(freq.get("GPC"), list) else []:
            value = SortingApp._bench_to_float(item)
            if value is not None:
                gpc_vals.append(value)
        metrics["gpu_util_pct"] = gpu_util if gpu_util is not None else ""
        gpu_freq_khz = SortingApp._bench_to_float(freq.get("cur"))
        gpu_freq_max_khz = SortingApp._bench_to_float(freq.get("max"))
        metrics["gpu_freq_khz"] = gpu_freq_khz if gpu_freq_khz is not None else ""
        metrics["gpu_freq_max_khz"] = gpu_freq_max_khz if gpu_freq_max_khz is not None else ""
        metrics["gpu_gpc_freq_mean_khz"] = SortingApp._bench_mean(gpc_vals) if gpc_vals else ""
        metrics["gpu_headroom_pct"] = max(0.0, 100.0 - gpu_util) if gpu_util is not None else ""
        return metrics

    @staticmethod
    def _bench_extract_power_metrics(power_info, power_history_w):
        metrics = {
            "power_total_w": "",
            "power_total_mean_w": "",
            "power_total_p95_w": "",
            "power_top_rail_1_name": "",
            "power_top_rail_1_w": "",
            "power_top_rail_2_name": "",
            "power_top_rail_2_w": "",
            "power_top_rail_3_name": "",
            "power_top_rail_3_w": "",
        }
        if not SortingApp._bench_is_mapping(power_info):
            mean_w = SortingApp._bench_mean(power_history_w)
            metrics["power_total_mean_w"] = mean_w if mean_w is not None else ""
            metrics["power_total_p95_w"] = SortingApp._bench_p95(power_history_w) if power_history_w else ""
            return metrics
        total = power_info.get("tot") if SortingApp._bench_is_mapping(power_info.get("tot")) else {}
        rails = power_info.get("rail") if SortingApp._bench_is_mapping(power_info.get("rail")) else {}
        total_w = SortingApp._bench_safe_div(total.get("power"), 1000.0)
        avg_w = SortingApp._bench_safe_div(total.get("avg"), 1000.0)
        metrics["power_total_w"] = total_w if total_w is not None else ""
        metrics["power_total_mean_w"] = avg_w if avg_w is not None else (
            SortingApp._bench_mean(power_history_w) if power_history_w else ""
        )
        metrics["power_total_p95_w"] = SortingApp._bench_p95(power_history_w) if power_history_w else ""
        rail_rows = []
        for name, rail in rails.items():
            if not SortingApp._bench_is_mapping(rail):
                continue
            power_w = SortingApp._bench_safe_div(rail.get("power"), 1000.0)
            if power_w is None:
                continue
            rail_rows.append((str(name), power_w))
        rail_rows.sort(key=lambda item: item[1], reverse=True)
        for idx, item in enumerate(rail_rows[:3], start=1):
            metrics[f"power_top_rail_{idx}_name"] = item[0]
            metrics[f"power_top_rail_{idx}_w"] = item[1]
        return metrics

    @staticmethod
    def _bench_extract_temperature_metrics(temp_info):
        metrics = {
            "temp_cpu_c": "",
            "temp_gpu_c": "",
            "temp_max_c": "",
        }
        if not SortingApp._bench_is_mapping(temp_info):
            return metrics
        temp_vals = []
        for _, sensor in SortingApp._bench_items(temp_info):
            if not SortingApp._bench_is_mapping(sensor):
                continue
            temp_c = SortingApp._bench_to_float(sensor.get("temp"))
            if temp_c is not None and temp_c > -200:
                temp_vals.append(temp_c)
        metrics["temp_cpu_c"] = SortingApp._bench_find_metric_by_name(temp_info, ("cpu", "soc"))
        metrics["temp_gpu_c"] = SortingApp._bench_find_metric_by_name(temp_info, ("gpu",))
        metrics["temp_max_c"] = max(temp_vals) if temp_vals else ""
        return metrics

    @staticmethod
    def _bench_extract_process_metrics(processes, current_pid):
        metrics = {
            "cpu_app_util_pct": "",
            "gpu_mem_used_mb": "",
        }
        if not isinstance(processes, list):
            return metrics
        for proc in processes:
            if not isinstance(proc, (list, tuple)) or len(proc) < 10:
                continue
            pid = proc[0]
            if pid != current_pid:
                continue
            cpu_pct = SortingApp._bench_to_float(proc[6])
            gpu_mem_mb = SortingApp._bench_kb_to_mb(proc[8])
            metrics["cpu_app_util_pct"] = cpu_pct if cpu_pct is not None else ""
            metrics["gpu_mem_used_mb"] = gpu_mem_mb if gpu_mem_mb is not None else ""
            break
        return metrics

    def _benchmark_push_cam_sample(self, infer_ms, fps_inst, det_count):
        if not self.benchmark_active:
            return
        with self._benchmark_lock:
            self._benchmark_window.append({
                "infer_ms": float(infer_ms) if infer_ms is not None else None,
                "fps": float(fps_inst) if fps_inst is not None else None,
                "det_count": float(det_count) if det_count is not None else None,
            })

    def _get_mp4_playback_pass_limit(self):
        if not dpg.does_item_exist("chk_vt_mp4_loop_limit"):
            return None
        try:
            enabled = bool(dpg.get_value("chk_vt_mp4_loop_limit"))
        except Exception:
            enabled = False
        if not enabled:
            return None
        try:
            value = int(dpg.get_value("in_vt_mp4_loop_count"))
        except Exception:
            value = 1
        return max(1, value)

    def _create_record_subdir(self, prefix):
        """Create one folder per recording/playback run."""
        os.makedirs("recordings", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("recordings", f"{prefix}_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir, ts

    def _start_benchmark_capture(self, mode, source_kind, source_file, output_dir=None):
        if self.benchmark_active:
            return
        if not self._is_benchmark_enabled():
            return

        if jtop is None:
            self.log("[BENCH] jtop not available - benchmark capture skipped", "ERROR")
            return

        if not output_dir:
            output_dir = "recordings"
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        src_name = os.path.splitext(os.path.basename(source_file or "session"))[0]
        self.benchmark_csv_path = os.path.join(output_dir, f"bench_{ts}_{mode}_{src_name}.csv")
        self.benchmark_mode = mode
        self.benchmark_source_kind = source_kind
        self._benchmark_window = []
        self._benchmark_stop_evt.clear()
        self._benchmark_started_epoch = time.time()
        self._benchmark_started_utc = datetime.fromtimestamp(
            self._benchmark_started_epoch, timezone.utc
        ).isoformat().replace("+00:00", "Z")
        self._benchmark_last_error = None
        self._benchmark_run_id = uuid.uuid4().hex[:12]
        self._benchmark_power_history_w = []

        fields = self._bench_build_fieldnames()

        try:
            self._benchmark_fh = open(self.benchmark_csv_path, "w", newline="", encoding="utf-8")
            self._benchmark_writer = csv.DictWriter(self._benchmark_fh, fieldnames=fields)
            self._benchmark_writer.writeheader()
            self._benchmark_fh.flush()
            os.fsync(self._benchmark_fh.fileno())
        except Exception as e:
            self.log(f"[BENCH] Cannot open benchmark CSV: {e}", "ERROR")
            self._benchmark_fh = None
            self._benchmark_writer = None
            self.benchmark_csv_path = None
            self.benchmark_mode = None
            self.benchmark_source_kind = None
            return

        self.benchmark_active = True
        self._benchmark_thread = threading.Thread(
            target=self._benchmark_worker_loop,
            args=(source_file,),
            daemon=True
        )
        self._benchmark_thread.start()
        self.log(f"[BENCH] Started ({mode}) every {self.benchmark_interval_s:.0f}s -> {self.benchmark_csv_path}", "SUCCESS")

    def _stop_benchmark_capture(self, reason=""):
        if not self.benchmark_active:
            return
        self._benchmark_stop_evt.set()
        th = self._benchmark_thread
        if th is not None and th.is_alive():
            th.join(timeout=2.0)
        self._benchmark_thread = None
        self.benchmark_active = False
        self.benchmark_mode = None
        self.benchmark_source_kind = None
        if self._benchmark_fh is not None:
            try:
                self._benchmark_fh.flush()
                os.fsync(self._benchmark_fh.fileno())
            except Exception:
                pass
            try:
                self._benchmark_fh.close()
            except Exception:
                pass
        self._benchmark_fh = None
        self._benchmark_writer = None
        if reason:
            self.log(f"[BENCH] Stopped: {reason}")
        else:
            self.log("[BENCH] Stopped")

    def _benchmark_worker_loop(self, source_file):
        process = psutil.Process()
        process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        sample_idx = 0
        jetson = None
        try:
            jetson = jtop()
            jetson.start()
            t0 = time.time()
            while not self._benchmark_stop_evt.is_set() and not jetson.ok():
                if time.time() - t0 > 3.0:
                    break
                time.sleep(0.1)

            next_tick = time.monotonic() + float(self.benchmark_interval_s)
            while not self._benchmark_stop_evt.is_set():
                sleep_s = max(0.0, next_tick - time.monotonic())
                if self._benchmark_stop_evt.wait(timeout=sleep_s):
                    break
                next_tick += float(self.benchmark_interval_s)

                now = time.time()
                sample_idx += 1
                stats = jetson.stats if (jetson is not None and jetson.ok()) else {}
                board = getattr(jetson, "board", {}) if jetson is not None else {}
                nvpmodel = getattr(jetson, "nvpmodel", "") if jetson is not None else ""
                cpu_info = getattr(jetson, "cpu", {}) if jetson is not None and jetson.ok() else {}
                memory_info = getattr(jetson, "memory", {}) if jetson is not None and jetson.ok() else {}
                gpu_info = getattr(jetson, "gpu", {}) if jetson is not None and jetson.ok() else {}
                power_info = getattr(jetson, "power", {}) if jetson is not None and jetson.ok() else {}
                temp_info = getattr(jetson, "temperature", {}) if jetson is not None and jetson.ok() else {}
                processes = getattr(jetson, "processes", []) if jetson is not None and jetson.ok() else []

                with self._benchmark_lock:
                    window = self._benchmark_window
                    self._benchmark_window = []

                infer_vals = [w["infer_ms"] for w in window if w.get("infer_ms") is not None]
                fps_vals = [w["fps"] for w in window if w.get("fps") is not None and w.get("fps") > 0]
                det_vals = [w["det_count"] for w in window if w.get("det_count") is not None]
                fps_mean = self._bench_mean(fps_vals)
                det_mean = self._bench_mean(det_vals)
                infer_mean = self._bench_mean(infer_vals)
                power_w = ""
                process_metrics = self._bench_extract_process_metrics(processes, process.pid)
                cpu_metrics = self._bench_extract_cpu_metrics(cpu_info)
                if process_metrics.get("cpu_app_util_pct") not in ("", None):
                    cpu_metrics["cpu_app_util_pct"] = process_metrics["cpu_app_util_pct"]
                else:
                    cpu_metrics["cpu_app_util_pct"] = process.cpu_percent(interval=None)
                ram_metrics = self._bench_extract_memory_metrics(memory_info)
                gpu_metrics = self._bench_extract_gpu_metrics(
                    gpu_info,
                    process_gpu_mem_mb=process_metrics.get("gpu_mem_used_mb"),
                )
                if gpu_metrics.get("gpu_util_pct") in ("", None) and isinstance(stats, dict):
                    gpu_fallback = self._bench_to_float(stats.get("GPU"))
                    if gpu_fallback is not None:
                        gpu_metrics["gpu_util_pct"] = gpu_fallback
                        gpu_metrics["gpu_headroom_pct"] = max(0.0, 100.0 - gpu_fallback)
                if ram_metrics.get("ram_util_pct") in ("", None) and isinstance(stats, dict):
                    ram_percent = self._bench_to_float(stats.get("RAM"))
                    if ram_percent is not None and ram_percent <= 1.0:
                        ram_percent *= 100.0
                    if ram_percent is not None:
                        ram_metrics["ram_util_pct"] = ram_percent
                        ram_metrics["ram_headroom_pct"] = max(0.0, 100.0 - ram_percent)
                temp_metrics = self._bench_extract_temperature_metrics(temp_info)
                if temp_metrics.get("temp_gpu_c") in ("", None) and isinstance(stats, dict):
                    temp_gpu_fallback = self._bench_to_float(stats.get("Temp gpu"))
                    temp_metrics["temp_gpu_c"] = temp_gpu_fallback if temp_gpu_fallback is not None else ""
                power_metrics = self._bench_extract_power_metrics(power_info, self._benchmark_power_history_w)
                power_w = power_metrics.get("power_total_w")
                if power_w in ("", None):
                    power_w = self._bench_estimate_power_w(stats)
                    power_metrics["power_total_w"] = power_w if power_w is not None else ""
                if power_w not in ("", None):
                    self._benchmark_power_history_w.append(float(power_w))
                    power_metrics["power_total_p95_w"] = self._bench_p95(self._benchmark_power_history_w)
                    if power_metrics.get("power_total_mean_w") in ("", None):
                        power_metrics["power_total_mean_w"] = self._bench_mean(self._benchmark_power_history_w)
                eff_fps_per_w = self._bench_safe_div(fps_mean, power_w)
                detections_per_s = None
                if det_mean is not None and fps_mean is not None:
                    detections_per_s = det_mean * fps_mean
                eff_det_per_w = self._bench_safe_div(detections_per_s, power_w)
                eff_latency_per_w = None
                if infer_mean not in (None, 0) and power_w not in ("", None):
                    eff_latency_per_w = self._bench_safe_div(1000.0 / infer_mean, power_w)

                jetson_clocks = getattr(jetson, "jetson_clocks", None) if jetson is not None else None
                jetson_clocks_status = getattr(jetson_clocks, "status", jetson_clocks)
                playback_loop_index = ""
                playback_loop_target = ""
                if self.benchmark_mode == "playback" and self.video_source is not None:
                    loop_count = int(getattr(self.video_source, "loop_count", 0) or 0)
                    playback_loop_index = int(getattr(self.video_source, "current_pass_index", loop_count + 1))
                    max_passes = getattr(self.video_source, "max_passes", None)
                    if max_passes is not None:
                        playback_loop_target = int(max_passes)

                row = {
                    "run_id": self._benchmark_run_id or "",
                    "benchmark_version": "v2",
                    "sample_idx": sample_idx,
                    "timestamp_epoch": round(now, 6),
                    "timestamp_utc": datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z"),
                    "record_start_utc": self._benchmark_started_utc or "",
                    "elapsed_s": round(max(0.0, now - float(self._benchmark_started_epoch or now)), 3),
                    "interval_s": float(self.benchmark_interval_s),
                    "benchmark_mode": self.benchmark_mode or "",
                    "source_kind": self.benchmark_source_kind or "",
                    "source_name": os.path.basename(source_file or ""),
                    "playback_loop_index": playback_loop_index,
                    "playback_loop_target": playback_loop_target,
                    "device_model": str(((board or {}).get("hardware", {}) or {}).get("Model", "")),
                    "jetpack_version": str(((board or {}).get("hardware", {}) or {}).get("Jetpack", "")),
                    "l4t_version": str(((board or {}).get("hardware", {}) or {}).get("L4T", "")),
                    "nvpmodel_mode": str(nvpmodel or stats.get("nvp model", "")),
                    "jetson_clocks_enabled": self._bench_bool_to_int(jetson_clocks_status),
                    "infer_fps_mean": fps_mean if fps_mean is not None else "",
                    "infer_fps_p95": self._bench_p95(fps_vals) if fps_vals else "",
                    "infer_fps_min": min(fps_vals) if fps_vals else "",
                    "infer_latency_mean_ms": infer_mean if infer_mean is not None else "",
                    "infer_latency_p95_ms": self._bench_p95(infer_vals) if infer_vals else "",
                    "infer_latency_max_ms": max(infer_vals) if infer_vals else "",
                    "infer_det_mean": det_mean if det_mean is not None else "",
                    "infer_frames_count": len(window),
                    "eff_fps_per_w": eff_fps_per_w if eff_fps_per_w is not None else "",
                    "eff_det_per_w": eff_det_per_w if eff_det_per_w is not None else "",
                    "eff_latency_per_w": eff_latency_per_w if eff_latency_per_w is not None else "",
                }
                row.update(gpu_metrics)
                row.update(cpu_metrics)
                row.update(ram_metrics)
                row.update(power_metrics)
                row.update(temp_metrics)

                if self._benchmark_writer is not None and self._benchmark_fh is not None:
                    self._benchmark_writer.writerow(row)
                    self._benchmark_fh.flush()
                    os.fsync(self._benchmark_fh.fileno())
        except Exception as e:
            self._benchmark_last_error = str(e)
            self.log(f"[BENCH] Worker error: {e}", "ERROR")
        finally:
            if jetson is not None:
                try:
                    jetson.close()
                except Exception:
                    pass

    def _maybe_auto_stop_finished_playback(self):
        if self.video_source is None:
            return
        if getattr(self.video_source, "running", True):
            return
        q = getattr(self.video_source, "frame_queue", None)
        if q is not None:
            try:
                if not q.empty():
                    return
            except Exception:
                pass
        auto_stop = dpg.get_value("chk_vt_auto_stop") if dpg.does_item_exist("chk_vt_auto_stop") else True
        if auto_stop:
            self.log("[VIDEO] Playback reached end of file")
            self._stop_video_playback()

    def __init__(self):
        # Initialize DearPyGui
        dpg.create_context()
        
        # Setup global theme
        self._setup_global_theme()
        
        # State variables
        self.is_running = False
        self.is_tracking = False
        self.auto_pick = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Statistics
        self.sorted_counts = {name: 0 for name in CLASS_NAMES.values()}
        self.total_detected = 0
        
        # Queue for thread communication (robot thread -> UI)
        self.ui_queue = queue.Queue()
        
        # Timing
        self.last_frame_time = time.time()
        self.last_update_time = time.time()
        
        # Store last detections to prevent flickering
        self.last_detections = []
        
        # Track robot position for jog control
        self.robot_pos = {'x': 0.0, 'y': 0.0, 'z': -250.0, 'r': 90.0}
        
        # Smoothed robot position for simulation display (in belt cm)
        self.sim_robot_x = 10.0  # Belt center X (cm)
        self.sim_robot_y = 10.0  # Belt center Y within workspace (cm)
        self.sim_robot_in_workspace = False  # Whether robot is in pick zone
        self.sim_robot_trail = []  # List of recent positions for trail effect
        self.sim_robot_smooth_factor = 0.15  # Lerp factor (0=no move, 1=instant)
        
        # Pick target visualization (shown on simulation as green marker)
        # Stores the dispatched pick coordinates so simulation can show WHERE robot should go
        self.sim_pick_target = None  # dict: {belt_x, belt_y_ws, belt_y_abs, obj_id, class_name, time}
        
        # TRACK mode: robot hovers above and follows an object through workspace
        self.track_mode = False          # Whether tracking mode is active
        self.track_target_id = None      # Object ID being tracked (None = auto-select first in WS)
        self.track_hover_margin_mm = 50  # Hover this many mm ABOVE the object's pick Z

        # Video playback state ("better dummy" — real detections from recorded .bag)
        self.video_source = None         # VideoPlaybackStream when active, else None
        self._live_camera = None         # Holds the live CameraStream while video is active
        self._loaded_bag_path = None     # Path to loaded .bag (persists after stop for replay)
        self._loaded_video_kind = None   # 'bag' | 'mp4' for loaded playback file
        self.replay_pick_rows = []       # Parsed sidecar rows for mp4 replay
        self.replay_pick_idx = 0         # Next replay row index
        self.replay_enabled = False      # True when replay sidecar rows are loaded
        self._replay_last_loop_count = 0

        # Recording state
        self.bag_recording = False       # True while recording .bag
        self.bag_filename = None         # Current .bag path
        self.mp4_recording = False       # True while recording .mp4 screen capture
        self.mp4_writer = None           # cv2.VideoWriter
        self.mp4_filename = None         # Current .mp4 path
        self.record_frame_count = 0      # Frames written to .mp4
        self._mp4_record_start_epoch = None   # Epoch when .mp4 recording started
        self._mp4_record_start_utc = None     # UTC timestamp of first recorded frame
        self._mp4_sync_rows = []              # Sync rows collected during recording
        self._mp4_record_fps = float(FPS)     # Output fps for mp4 recorder
        self._mp4_source_t0 = None            # Source timestamp of first recorded frame

        # Benchmark capture state (CSV every 5s, flushed immediately)
        self.benchmark_interval_s = 5.0
        self.benchmark_enabled_default = True
        self.benchmark_active = False
        self.benchmark_mode = None            # 'rec_mp4' | 'playback'
        self.benchmark_source_kind = None     # 'live' | 'mp4' | 'bag'
        self.benchmark_csv_path = None
        self._benchmark_fh = None
        self._benchmark_writer = None
        self._benchmark_thread = None
        self._benchmark_stop_evt = threading.Event()
        self._benchmark_lock = threading.Lock()
        self._benchmark_window = []           # per-frame camera samples since last CSV flush
        self._benchmark_started_epoch = None
        self._benchmark_started_utc = None
        self._benchmark_last_error = None
        self._benchmark_run_id = None
        self._benchmark_power_history_w = []

        # Robot camera (USB) state — records robot picking (linked to experiment recording)
        self.robot_cam = None                 # cv2.VideoCapture for USB camera
        self.robot_cam_writer = None          # cv2.VideoWriter for robot cam (linked to mp4 recording)
        self.robot_cam_filename = None        # Current robot cam .mp4 path
        self.robot_cam_frame_count = 0        # Frames written to robot cam .mp4
        self.robot_cam_connected = False      # Whether USB camera opened OK
        self.robot_cam_preview = True         # Live preview enabled (toggle from UI)

        # ── Audit log state ──
        self.audit_collecting = False         # True while audit log is being collected
        self.audit_buffer = []                # In-memory buffer of audit row dicts
        self.audit_start_time = None          # Timestamp when collection started

        # ── Calibration state ──
        self.cal_mode = False                 # True when calibration overlay is active
        self.cal_checkerboard_corners = None  # Last detected checkerboard corners
        self.cal_checkerboard_found = False   # Whether checkerboard was found this frame
        self.cal_flip_vertical = False        # Swap entry/exit direction
        self.cal_flip_horizontal = False      # Mirror left/right
        self.cal_confirmed = False            # True after user confirms calibration
        self.cal_homography = None            # Confirmed homography matrix
        self.cal_roi_corners = None           # Confirmed ROI corners (px)
        self.cal_floor_plane = None           # Floor plane coefficients (a,b,c,d)
        self.cal_floor_depth_map = None       # Floor depth reference map
        self.cal_corrected_corners = None     # Orientation-corrected 4 corners
        self.cal_depth_frame = None           # Last depth frame (for floor calibration)

        # Checkerboard parameters (matching calibration_tool.py)
        self.CAL_SQUARES_X = 8
        self.CAL_SQUARES_Y = 10
        self.CAL_SQUARE_SIZE_CM = 2.5
        self.CAL_CORNERS_X = 7   # inner corners
        self.CAL_CORNERS_Y = 9

        # Auto startup sequence at launch: connect -> home -> start -> auto pick
        self.auto_boot_on_launch = True
        self._auto_boot_pending = True
        self._auto_boot_started = False
        self._auto_boot_done = False
        self._auto_boot_frame_delay = 20
        
        # Initialize components
        self._init_components()

        # Operator timing runs (1-3 verified and placed objects per run)
        self._init_timing_sessions()
        
        # Setup UI
        self._setup_ui()
        self._update_timing_ui()
        
        # Load calibration
        if self.detector.load_calibration():
            self.log(f"Calibration loaded: ROI {self.detector.roi_width_cm}x{self.detector.roi_height_cm}cm", "SUCCESS")
        else:
            self.log("No calibration found! Use the Calibration tab to calibrate", "ERROR")
        self._load_offsets()
        self._load_grid_and_place()
        
        # Create viewport and start
        dpg.create_viewport(title="Waste Sorting System Prototype Version", width=1920, height=1080)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
    
    def _init_components(self):
        """Initialize all system components."""
        self.log_messages = []
        
        # Camera
        self.log("Initializing camera...")
        self.camera = CameraStream()
        
        # Detector
        self.log("Initializing detector...")
        self.detector = ObjectDetector()
        if not self.detector.load_model():
            self.log("WARNING: YOLO model failed to load!", "ERROR")
        
        # Tracker
        self.tracker = SimpleTracker()
        
        # Robot controllers
        self.log("Initializing robot controllers...")
        self.delta = DeltaController(log_func=self.log)
        self.slider = SliderController(log_func=self.log)
        
        # Spectrum sensor
        self.log("Initializing spectrum sensor...")
        self.spectrum = SpectrumManager(log_func=self.log)
        self.spectrum.initialize_hardware()
        self.spectrum.load_models()
        
        # Database
        self.db = DatabaseManager()
        
        # Detection logger (CSV with 9 columns: ID, Camera classes, Spectrum classes)
        self.detection_logger = DetectionLogger(log_dir="detection_logs")
        
        # Robot manager — with tracker reference for real-time tracking during approach
        self.robot_manager = RobotManager(
            self.delta, self.slider, self.spectrum,
            log_func=self.log, ui_queue=self.ui_queue,
            db_manager=self.db, session_id=self.session_id,
            detection_logger=self.detection_logger,
            tracker=self.tracker
        )
        # Callback: robot updates simulation position during tracking approach
        self.robot_manager.on_robot_pos_update = self._on_robot_approach_pos_update

        # Robot camera (USB) — for recording picking actions
        self._init_robot_camera()

    def _init_robot_camera(self):
        """Open the USB robot camera (/dev/video6) for live preview & recording."""
        try:
            self.log("Initializing robot camera...")
            cap = cv2.VideoCapture(ROBOT_CAM_DEVICE, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, ROBOT_CAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ROBOT_CAM_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, ROBOT_CAM_FPS)
                self.robot_cam = cap
                self.robot_cam_connected = True
                self.log(f"Robot camera opened: {ROBOT_CAM_DEVICE} "
                         f"({ROBOT_CAM_WIDTH}x{ROBOT_CAM_HEIGHT}@{ROBOT_CAM_FPS}fps)", "SUCCESS")
            else:
                cap.release()
                self.robot_cam = None
                self.robot_cam_connected = False
                self.log(f"Robot camera not available at {ROBOT_CAM_DEVICE}", "WARN")
        except Exception as e:
            self.robot_cam = None
            self.robot_cam_connected = False
            self.log(f"Robot camera init error: {e}", "ERROR")
        # Sync toggle button state if it exists
        if dpg.does_item_exist("btn_robot_cam_toggle"):
            if self.robot_cam_connected:
                self.robot_cam_preview = True
                dpg.configure_item("btn_robot_cam_toggle", label="Disable Preview")
                dpg.bind_item_theme("btn_robot_cam_toggle", self._robot_cam_btn_on)
            else:
                self.robot_cam_preview = False
                dpg.configure_item("btn_robot_cam_toggle", label="Enable Preview")
                dpg.bind_item_theme("btn_robot_cam_toggle", self._robot_cam_btn_off)
    
    def log(self, msg, level="INFO"):
        """Log a message to the UI."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_messages.append((timestamp, msg, level))
        # Keep only last 100 messages
        if len(self.log_messages) > 100:
            self.log_messages = self.log_messages[-100:]
        print(f"[{timestamp}] {msg}")
    
    def _setup_ui(self):
        """Setup Dear PyGui UI."""
        # Register texture for video feed
        with dpg.texture_registry(show=False):
            dummy = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.float32)
            dpg.add_dynamic_texture(IMAGE_WIDTH, IMAGE_HEIGHT, dummy.flatten(), tag="video_texture")
            # Robot camera texture (same resolution)
            robot_dummy = np.zeros((ROBOT_CAM_HEIGHT, ROBOT_CAM_WIDTH, 4), dtype=np.float32)
            dpg.add_dynamic_texture(ROBOT_CAM_WIDTH, ROBOT_CAM_HEIGHT,
                                    robot_dummy.flatten(), tag="robot_cam_texture")
        
        # Main window
        with dpg.window(tag="main_window", no_title_bar=True, no_move=True):
            # Title with status indicators
            with dpg.group(horizontal=True):
                dpg.add_text("WASTE SORTING SYSTEM PROTOTYPE VERSION", color=(0, 220, 255))
                dpg.add_spacer(width=50)
                dpg.add_text("[X]", tag="status_indicator_cam", color=(255, 80, 80))
                dpg.add_text("Camera", color=(150, 150, 150))
                dpg.add_spacer(width=20)
                dpg.add_text("[X]", tag="status_indicator_robot", color=(255, 80, 80))
                dpg.add_text("Robot", color=(150, 150, 150))
                dpg.add_spacer(width=20)
                dpg.add_text("[X]", tag="status_indicator_spectrum", color=(255, 80, 80))
                dpg.add_text("Spectrum", color=(150, 150, 150))
            dpg.add_separator()
            
            # Tab bar for different pages
            with dpg.tab_bar():
                # === TAB 1: VISION (Main sorting view) ===
                with dpg.tab(label="Vision"):
                    with dpg.child_window(height=205, border=True):
                        with dpg.group(horizontal=True):
                            dpg.add_text("OBJECT TIMING RUN", color=(0, 255, 255))
                            dpg.add_spacer(width=20)
                            dpg.add_text("Run ID:", color=(150, 150, 150))
                            dpg.add_text(
                                self.timing_current_run["run_id"],
                                tag="txt_timing_run_id",
                                color=(255, 220, 80),
                            )
                            dpg.add_spacer(width=20)
                            dpg.add_combo(
                                ["1 obj", "2 obj", "3 obj"],
                                default_value="1 obj",
                                tag="combo_timing_count",
                                width=110,
                                callback=self._on_timing_count_change,
                            )
                            dpg.add_button(
                                label="NEXT",
                                width=100,
                                callback=self._on_timing_next,
                            )
                            dpg.add_button(
                                label="DELETE",
                                width=100,
                                callback=self._on_timing_delete,
                            )
                            dpg.add_spacer(width=20)
                            dpg.add_text("Status:", color=(150, 150, 150))
                            dpg.add_text(
                                "Waiting for first verification",
                                tag="txt_timing_status",
                                color=(100, 220, 255),
                            )
                            dpg.add_spacer(width=20)
                            dpg.add_text("TOTAL:", color=(150, 150, 150))
                            dpg.add_text(
                                "--",
                                tag="txt_timing_total",
                                color=(100, 255, 140),
                            )

                        with dpg.table(
                            header_row=True,
                            row_background=True,
                            borders_innerH=True,
                            borders_outerH=True,
                            borders_innerV=True,
                            borders_outerV=True,
                            height=130,
                        ):
                            dpg.add_table_column(label="Object", width_fixed=True, init_width_or_weight=65)
                            dpg.add_table_column(label="Robot ID", width_fixed=True, init_width_or_weight=90)
                            dpg.add_table_column(label="Final Class", width_fixed=True, init_width_or_weight=120)
                            dpg.add_table_column(label="Detect at Registration", width_fixed=True, init_width_or_weight=180)
                            dpg.add_table_column(label="Lay Complete Time", width_fixed=True, init_width_or_weight=180)
                            dpg.add_table_column(label="Detect -> Lay", width_fixed=True, init_width_or_weight=130)
                            for index in range(1, 4):
                                with dpg.table_row(tag=f"timing_row_{index}"):
                                    dpg.add_text(f"Object {index}")
                                    dpg.add_text("--", tag=f"timing_obj_{index}_id")
                                    dpg.add_text("--", tag=f"timing_obj_{index}_class")
                                    dpg.add_text("--", tag=f"timing_obj_{index}_detected")
                                    dpg.add_text("--", tag=f"timing_obj_{index}_laid")
                                    dpg.add_text(
                                        "--",
                                        tag=f"timing_obj_{index}_cycle",
                                        color=(100, 255, 140),
                                    )

                    # Main layout - 3 child windows side by side
                    with dpg.group(horizontal=True):
                        # === CHILD WINDOW 1: Camera Display & Log ===
                        with dpg.child_window(width=720, height=-1, border=True):
                            dpg.add_text("CAMERA & SYSTEM LOG", color=(0, 255, 255))
                            dpg.add_separator()
                            
                            # Video feed
                            dpg.add_text("Vision Feed:", color=(180, 180, 180))
                            dpg.add_image("video_texture")
                            
                            dpg.add_separator()
                            
                            # Dashboard table
                            dpg.add_text("Tracking Dashboard:", color=(180, 180, 180))
                            with dpg.table(tag="dash_table", header_row=True, row_background=True,
                                          scrollY=True, height=160):
                                dpg.add_table_column(label="ID", width=35)
                                dpg.add_table_column(label="Class", width=55)
                                dpg.add_table_column(label="H (cm)", width=50)
                                dpg.add_table_column(label="X (cm)", width=45)
                                dpg.add_table_column(label="Y (cm)", width=45)
                                dpg.add_table_column(label="Status", width=60)
                            
                            dpg.add_separator()
                            
                            # Log area
                            dpg.add_text("System Log:", color=(180, 180, 180))
                            dpg.add_child_window(tag="log_child", height=-1, border=False)
                        
                        dpg.add_spacer(width=10)
                        
                        # === CHILD WINDOW 2: Control Buttons & Parameters ===
                        with dpg.child_window(width=600, height=-1, border=True):
                            dpg.add_text("CONTROLS & PARAMETERS", color=(0, 255, 255))
                            dpg.add_separator()
                            
                            # Control buttons
                            dpg.add_text("Main Controls:", color=(180, 180, 180))
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="START", width=168, height=60,
                                              tag="btn_start", callback=self._on_start)
                                dpg.bind_item_theme("btn_start", self._create_start_btn_theme())
                                dpg.add_button(label="STOP", width=168, height=60,
                                              tag="btn_stop", callback=self._on_stop, enabled=False)
                                dpg.bind_item_theme("btn_stop", self._create_stop_btn_theme())
                                dpg.add_button(label="AUTO PICK", width=168, height=60,
                                              tag="btn_autopick", callback=self._on_autopick)
                                dpg.bind_item_theme("btn_autopick", self._create_autopick_btn_theme())
                            
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="CALIBRATE", width=144, height=42,
                                              tag="btn_calibrate", callback=self._on_calibrate)
                                dpg.add_button(label="RELOAD CAL", width=144, height=42,
                                              tag="btn_reload_cal", callback=self._on_reload_calibration)
                                dpg.add_button(label="DUMMY OBJ", width=144, height=42,
                                              callback=self._on_inject_dummy,
                                              tag="btn_dummy")
                                dpg.bind_item_theme("btn_dummy", self._create_dummy_btn_theme())
                                dpg.add_button(label="TRACK", width=120, height=42,
                                              callback=self._on_toggle_track,
                                              tag="btn_track")
                                dpg.bind_item_theme("btn_track", self._create_track_btn_theme())
                            
                            dpg.add_separator()
                            
                            # Robot connection
                            dpg.add_text("Robot Connection:", color=(180, 180, 180))
                            with dpg.group(horizontal=True):
                                dpg.add_input_text(width=150, default_value=DEFAULT_DELTA_PORT,
                                                  tag="in_delta_port", label="Delta")
                                dpg.add_input_text(width=150, default_value=DEFAULT_SLIDER_PORT,
                                                  tag="in_slider_port", label="Slider")
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="CONNECT", width=120, height=36, callback=self._on_connect)
                                dpg.add_button(label="DISCONNECT", width=120, height=36, callback=self._on_disconnect)
                                dpg.add_button(label="HOME", width=96, height=36, callback=self._on_home)
                                dpg.add_text("", tag="txt_robot_status")
                            
                            dpg.add_separator()
                            
                            # Parameters
                            dpg.add_text("System Parameters:", color=(180, 180, 180))
                            
                            # -- Model selector --
                            dpg.add_combo(AVAILABLE_MODEL_NAMES,
                                          default_value=AVAILABLE_MODEL_NAMES[0] if AVAILABLE_MODEL_NAMES else MODEL_PATH,
                                          label="YOLO Model", tag="combo_model",
                                          width=300, popup_align_left=True,
                                          callback=self._on_model_change)
                            dpg.add_text(f"({len(AVAILABLE_MODEL_NAMES)} models found)",
                                         tag="txt_model_status", color=(150, 150, 150))
                            
                            dpg.add_slider_float(label="Belt Speed (cm/s)", default_value=CONVEYOR_SPEED_CM_S,
                                                min_value=1.0, max_value=15.0, tag="in_speed", width=200)
                            dpg.add_checkbox(
                                label="Dynamic Speed Estimation",
                                default_value=True,
                                tag="chk_dynamic_speed",
                            )
                            dpg.add_text("Measured: --", tag="txt_measured_speed",
                                         color=(100, 200, 100))
                            dpg.add_slider_float(label="YOLO Confidence", default_value=CONFIDENCE_THRESHOLD,
                                                min_value=0.1, max_value=1.0, tag="in_conf", width=200)
                            dpg.add_slider_float(label="Approach Time (s)", default_value=0.5,
                                                min_value=0.0, max_value=2.0, tag="in_approach_time", width=200,
                                                format="%.2f s")
                            dpg.add_text(f"Workspace: REG -> Pick = {REGISTRATION_LINE_CM + (ROI_HEIGHT_CM - REGISTRATION_LINE_CM) + ROBOT_WORKSPACE_OFFSET_CM + 10:.1f}cm", 
                                        color=(150, 150, 150))
                            
                            dpg.add_separator()
                            
                            # ── Separation Logic Toggles ──
                            dpg.add_text("Detection Pipeline:", color=(180, 180, 180))
                            dpg.add_checkbox(
                                label="Depth Clustering (split merged masks)",
                                default_value=DEPTH_CLUSTER_ENABLED,
                                tag="chk_depth_cluster",
                                callback=self._on_separation_toggle,
                            )
                            dpg.add_checkbox(
                                label="Cross-Class NMS (remove duplicates)",
                                default_value=DUPLICATE_MASK_NMS_ENABLED,
                                tag="chk_cross_nms",
                                callback=self._on_separation_toggle,
                            )
                            dpg.add_checkbox(
                                label="Watershed Joining (rejoin fragments)",
                                default_value=WATERSHED_JOIN_ENABLED,
                                tag="chk_watershed",
                                callback=self._on_separation_toggle,
                            )
                            # Build initial pipeline status from config defaults
                            _init_tags = []
                            if DEPTH_CLUSTER_ENABLED:
                                _init_tags.append("DC")
                            if DUPLICATE_MASK_NMS_ENABLED:
                                _init_tags.append("NMS")
                            if WATERSHED_JOIN_ENABLED:
                                _init_tags.append("WS")
                            _init_pl = "Pipeline: " + (" > ".join(_init_tags) if _init_tags else "RAW (no post-processing)")
                            dpg.add_text(_init_pl, tag="txt_pipeline_status",
                                         color=(0, 220, 255) if _init_tags else (255, 220, 50))
                            
                            dpg.add_separator()
                            
                            with dpg.group(horizontal=True):
                                dpg.add_text("Position Offsets:", color=(180, 180, 180))
                                dpg.add_spacer(width=10)
                                dpg.add_button(label="Save Offsets", width=120, height=28,
                                              callback=self._save_offsets)
                            with dpg.group(horizontal=True):
                                dpg.add_input_float(width=240, tag="off_x", label="X", default_value=0, step=0.5)
                                dpg.add_spacer(width=10)
                                dpg.add_input_float(width=240, tag="off_y", label="Y", default_value=0, step=0.5)
                            with dpg.group(horizontal=True):
                                dpg.add_input_float(width=240, tag="off_z", label="Z", default_value=0, step=0.5)
                                dpg.add_spacer(width=10)
                                dpg.add_input_float(width=240, tag="off_lat", label="Latency", default_value=-0.7, step=0.1)
                            
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_text("Per-Class Z Offset (mm):", color=(180, 180, 180))
                                dpg.add_spacer(width=10)
                                dpg.add_text("(+)higher  (-)lower", color=(120, 120, 120))
                            with dpg.group(horizontal=True):
                                dpg.add_input_float(width=115, tag="czoff_Glass",  label="Glass",  default_value=5.0,  step=1.0)
                                dpg.add_spacer(width=5)
                                dpg.add_input_float(width=115, tag="czoff_Metal",  label="Metal",  default_value=10.0, step=1.0)
                            with dpg.group(horizontal=True):
                                dpg.add_input_float(width=115, tag="czoff_Paper",  label="Paper",  default_value=-10.0, step=1.0)
                                dpg.add_spacer(width=5)
                                dpg.add_input_float(width=115, tag="czoff_Plastic", label="Plastic", default_value=0.0,  step=1.0)
                            
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_text("Stack-Bottom Offsets:", color=(180, 180, 180))
                                dpg.add_spacer(width=10)
                                dpg.add_text("(compensate for belt drift after picking top)", color=(120, 120, 120))
                            with dpg.group(horizontal=True):
                                dpg.add_input_float(width=180, tag="sbot_y_advance",
                                                    label="Y Advance (cm)", default_value=2.0, step=0.5)
                                dpg.add_spacer(width=10)
                                dpg.add_input_float(width=180, tag="sbot_z_extra",
                                                    label="Z Extra (mm)", default_value=-10.0, step=1.0)
                            
                            dpg.add_separator()
                            
                            # Spectrum prediction display
                            dpg.add_text("Spectrum Sensor:", color=(180, 180, 180))
                            with dpg.group(horizontal=True):
                                dpg.add_text("YOLO:", color=(150, 150, 150))
                                dpg.add_text("---", tag="txt_spec_yolo", color=(0, 200, 255))
                                dpg.add_spacer(width=15)
                                dpg.add_text("Spectrum:", color=(150, 150, 150))
                                dpg.add_text("---", tag="txt_spec_pred", color=(0, 255, 100))
                            with dpg.group(horizontal=True):
                                dpg.add_text("Final:", color=(150, 150, 150))
                                dpg.add_text("---", tag="txt_spec_final", color=(255, 255, 0))
                                dpg.add_spacer(width=15)
                                dpg.add_text("Status:", color=(150, 150, 150))
                                status = "Ready" if self.spectrum.is_ready else "Not Ready"
                                dpg.add_text(status, tag="txt_spectrum_status",
                                            color=(0, 255, 0) if self.spectrum.is_ready else (255, 100, 100))
                        
                        dpg.add_spacer(width=10)
                        
                        # === CHILD WINDOW 3: Robot Workspace Visualization ===
                        with dpg.child_window(width=-1, height=-1, border=True):
                            dpg.add_text("ROBOT WORKSPACE VISUALIZATION", color=(0, 255, 255))
                            dpg.add_separator()
                            
                            # Workspace simulation - fixed size for reliability
                            dpg.add_drawlist(width=500, height=700, tag="workspace_drawlist")
                            
                            dpg.add_separator()
                            
                            # Sorting statistics
                            dpg.add_text("Sorting Statistics:", color=(180, 180, 180))
                            dpg.add_text("", tag="txt_stats")
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                for cls_name in CLASS_NAMES.values():
                                    with dpg.group():
                                        color = CLASS_COLORS.get(
                                            list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(cls_name)],
                                            DEFAULT_COLOR
                                        )
                                        dpg.add_text(f"{cls_name}:", color=color)
                                        dpg.add_text("0", tag=f"stat_{cls_name}")
                
                # === TAB 2: ROBOT CONTROL ===
                with dpg.tab(label="Robot Control"):
                    # ── Full-height side-by-side: Camera (left) | Controls (right) ──
                    with dpg.group(horizontal=True):

                        # ============================================================
                        # LEFT COLUMN — Robot Camera Feed  (~710 px, ~10 % wider than
                        # the 640 px main camera texture so the image is at least as
                        # large).  Height = -1 → fills the tab.
                        # ============================================================
                        with dpg.child_window(width=710, height=-1, border=True):
                            dpg.add_text("ROBOT CAMERA", color=(0, 255, 255))
                            dpg.add_separator()
                            dpg.add_spacer(height=2)
                            # Camera image (640×480 native — drawn at texture size,
                            # the 710 px wrapper gives ~10 % extra breathing room)
                            dpg.add_image("robot_cam_texture")
                            dpg.add_spacer(height=6)
                            dpg.add_separator()
                            # Camera status & buttons
                            with dpg.group(horizontal=True):
                                dpg.add_text("Status:", color=(180, 180, 180))
                                dpg.add_spacer(width=6)
                                dpg.add_text("Disconnected", tag="txt_robot_cam_conn",
                                             color=(255, 80, 80))
                            dpg.add_spacer(height=6)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Enable Preview", width=140, height=32,
                                              tag="btn_robot_cam_toggle",
                                              callback=lambda s,a,u: self._toggle_robot_cam_preview())
                                self._robot_cam_btn_on = self._create_robot_cam_on_theme()
                                self._robot_cam_btn_off = self._create_robot_cam_off_theme()
                                dpg.bind_item_theme("btn_robot_cam_toggle", self._robot_cam_btn_off)
                                dpg.add_button(label="Reconnect", width=120, height=32,
                                              tag="btn_robot_cam_reconnect",
                                              callback=lambda s,a,u: self._reconnect_robot_cam())
                                dpg.bind_item_theme("btn_robot_cam_reconnect",
                                                    self._create_robot_cam_reconnect_theme())

                            dpg.add_spacer(height=8)
                            dpg.add_separator()

                            # Current position display (below camera)
                            dpg.add_text("Current Position:", color=(150, 150, 150))
                            with dpg.group(horizontal=True):
                                dpg.add_text("X:", color=(255, 100, 100))
                                dpg.add_text("0.00", tag="pos_x")
                                dpg.add_text("  Y:", color=(100, 255, 100))
                                dpg.add_text("0.00", tag="pos_y")
                                dpg.add_text("  Z:", color=(100, 100, 255))
                                dpg.add_text("0.00", tag="pos_z")
                                dpg.add_text("  R:", color=(255, 255, 100))
                                dpg.add_text("0.00", tag="pos_r")

                        dpg.add_spacer(width=6)

                        # ============================================================
                        # RIGHT COLUMN — All control panels (fills remaining width)
                        # ============================================================
                        with dpg.child_window(width=-1, height=-1, border=False):

                            # ── Row 1: Jog + Direct Position side by side ──
                            with dpg.group(horizontal=True):
                                # Panel 1 — Manual Jog Controls
                                with dpg.child_window(width=380, height=420, border=True):
                                    dpg.add_text("MANUAL JOG CONTROL", color=(0, 255, 255))
                                    dpg.add_separator()

                                    dpg.add_text("Jog Step Size:", color=(150, 150, 150))
                                    dpg.add_radio_button(items=["1", "5", "10", "50", "100"],
                                                         tag="jog_step", default_value="10",
                                                         horizontal=True)

                                    dpg.add_separator()
                                    dpg.add_text("X Axis", color=(255, 100, 100))
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="X-", width=80, height=40, callback=lambda: self._jog_robot('x', -1))
                                        dpg.add_text("", tag="pos_x_jog", color=(255, 255, 255))
                                        dpg.add_button(label="X+", width=80, height=40, callback=lambda: self._jog_robot('x', 1))

                                    dpg.add_text("Y Axis", color=(100, 255, 100))
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="Y-", width=80, height=40, callback=lambda: self._jog_robot('y', -1))
                                        dpg.add_text("", tag="pos_y_jog", color=(255, 255, 255))
                                        dpg.add_button(label="Y+", width=80, height=40, callback=lambda: self._jog_robot('y', 1))

                                    dpg.add_text("Z Axis", color=(100, 100, 255))
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="Z-", width=80, height=40, callback=lambda: self._jog_robot('z', -1))
                                        dpg.add_text("", tag="pos_z_jog", color=(255, 255, 255))
                                        dpg.add_button(label="Z+", width=80, height=40, callback=lambda: self._jog_robot('z', 1))

                                    dpg.add_text("R Axis (Rotation)", color=(255, 255, 100))
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="R-", width=80, height=40, callback=lambda: self._jog_robot('r', -1))
                                        dpg.add_text("", tag="pos_r_jog", color=(255, 255, 255))
                                        dpg.add_button(label="R+", width=80, height=40, callback=lambda: self._jog_robot('r', 1))

                                dpg.add_spacer(width=6)

                                # Panel 2 — Direct Position & Actions
                                with dpg.child_window(width=-1, height=420, border=True):
                                    dpg.add_text("DIRECT POSITION CONTROL", color=(0, 255, 255))
                                    dpg.add_separator()

                                    dpg.add_input_float(label="Target X", tag="target_x", default_value=0, width=150)
                                    dpg.add_input_float(label="Target Y", tag="target_y", default_value=0, width=150)
                                    dpg.add_input_float(label="Target Z", tag="target_z", default_value=-250, width=150)
                                    dpg.add_input_float(label="Target R", tag="target_r", default_value=90, width=150)
                                    dpg.add_input_int(label="Speed (F)", tag="target_speed", default_value=15000, width=150)

                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="GO TO POSITION", width=150, height=40,
                                                      callback=self._go_to_position)
                                        dpg.add_button(label="HOME", width=100, height=40,
                                                      callback=self._on_home)

                                    dpg.add_separator()
                                    dpg.add_text("VACUUM CONTROL", color=(0, 255, 255))
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="VACUUM ON", width=120, height=35,
                                                      callback=lambda: self.delta.set_vacuum(True))
                                        dpg.add_button(label="VACUUM OFF", width=120, height=35,
                                                      callback=lambda: self.delta.set_vacuum(False))

                                    dpg.add_separator()
                                    dpg.add_text("PRESET POSITIONS", color=(0, 255, 255))
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="STANDBY", width=100, height=35,
                                                      callback=lambda: self._go_preset('standby'))
                                        dpg.add_button(label="PICK POS", width=100, height=35,
                                                      callback=lambda: self._go_preset('pick'))
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="SCAN POS", width=100, height=35,
                                                      callback=lambda: self._go_preset('scan'))
                                        dpg.add_button(label="DROP POS", width=100, height=35,
                                                      callback=lambda: self._go_preset('drop'))

                            dpg.add_spacer(height=4)

                            # ── Row 2: Place Positions + Grid + Pick Test side by side ──
                            with dpg.group(horizontal=True):
                                # Panel 3 — Place Positions
                                with dpg.child_window(width=380, height=-1, border=True):
                                    dpg.add_text("PLACE POSITIONS (mm)", color=(0, 255, 255))
                                    dpg.add_separator()
                                    dpg.add_text("Set robot XY for each class bin:", color=(150, 150, 150))

                                    # Per-class place target editors
                                    for cls_name in ["Glass", "Metal", "Paper", "Plastic"]:
                                        default_x, default_y = THROW_TARGETS.get(cls_name, (0, 0))
                                        cls_color = {
                                            "Glass": (0, 255, 255),
                                            "Metal": (192, 192, 192),
                                            "Paper": (0, 165, 255),
                                            "Plastic": (0, 255, 0),
                                        }.get(cls_name, (200, 200, 200))

                                        with dpg.group(horizontal=True):
                                            dpg.add_text(f"{cls_name}:", color=cls_color)
                                            dpg.add_input_float(
                                                label="X", tag=f"place_{cls_name}_x",
                                                default_value=default_x, width=80, step=0)
                                            dpg.add_input_float(
                                                label="Y", tag=f"place_{cls_name}_y",
                                                default_value=default_y, width=80, step=0)
                                            dpg.add_button(label="Go",
                                                tag=f"place_{cls_name}_go", width=30,
                                                callback=lambda s, a, u: self._go_place_pos(u),
                                                user_data=cls_name)
                                            dpg.add_button(label="Set",
                                                tag=f"place_{cls_name}_set", width=30,
                                                callback=lambda s, a, u: self._set_place_from_robot(u),
                                                user_data=cls_name)

                                    dpg.add_separator()
                                    dpg.add_text("Place Z Heights (mm):", color=(150, 150, 150))
                                    with dpg.group(horizontal=True):
                                        dpg.add_input_float(label="Place Z", tag="place_z_height",
                                            default_value=PLACE_Z_HEIGHT, width=80, step=0)
                                        dpg.add_input_float(label="Release Z", tag="place_z_release",
                                            default_value=PLACE_Z_RELEASE, width=80, step=0)

                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="APPLY", width=80, height=30,
                                            callback=self._apply_place_targets)
                                        dpg.add_button(label="SAVE", width=80, height=30,
                                            callback=self._save_place_targets)
                                        dpg.add_button(label="LOAD", width=80, height=30,
                                            callback=self._load_place_targets)
                                        dpg.add_button(label="RESET", width=80, height=30,
                                            callback=self._reset_place_targets)

                                    dpg.add_separator()
                                    dpg.add_text("PICK TEST", color=(255, 200, 0))
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="PICK TEST (4 CORNERS)", width=180, height=40,
                                                      callback=self._pick_test)
                                        dpg.add_button(label="STOP", width=80, height=40,
                                                      callback=self._on_home)
                                    dpg.add_separator()
                                    dpg.add_text("SMOOTH MOVE DEMO", color=(255, 200, 0))
                                    dpg.add_button(label="SMOOTH TR->BL", width=200, height=40,
                                        callback=self._demo_smooth_two_points)
                                    dpg.add_separator()
                                    dpg.add_text("SMOOTH AXIS DEMO", color=(255, 200, 0))
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="SMOOTH Y", width=130, height=36,
                                            callback=self._demo_smooth_y_axis)
                                        dpg.add_button(label="SMOOTH X", width=130, height=36,
                                            callback=self._demo_smooth_x_axis)

                                dpg.add_spacer(width=6)

                                # Panel 4 — Workspace Grid
                                with dpg.child_window(width=-1, height=-1, border=True):
                                    dpg.add_text("WORKSPACE GRID (Robot XY mm)", color=(0, 255, 255))
                                    dpg.add_text("Belt -> Robot coordinate mapping:", color=(150, 150, 150))
                                    dpg.add_separator()

                                    # 3x3 Grid display: TL TC TR / ML MC MR / BL BC BR
                                    grid_labels = [
                                        ["TL", "TC", "TR"],
                                        ["ML", "MC", "MR"],
                                        ["BL", "BC", "BR"],
                                    ]
                                    grid_colors = [
                                        (255, 120, 120),  # Top row - reddish
                                        (255, 255, 120),  # Mid row - yellowish
                                        (120, 200, 255),  # Bot row - bluish
                                    ]
                                    for row in range(3):
                                        with dpg.group(horizontal=True):
                                            for col in range(3):
                                                lbl = grid_labels[row][col]
                                                rx = ROBOT_X_GRID[row][col]
                                                ry = ROBOT_Y_GRID[row][col]
                                                bx = BELT_X_GRID[col]
                                                by = BELT_Y_GRID[row]
                                                tag_x = f"grid_{lbl}_x"
                                                tag_y = f"grid_{lbl}_y"

                                                with dpg.child_window(width=145, height=70,
                                                                      border=True, no_scrollbar=True):
                                                    dpg.add_text(
                                                        f"{lbl} (B:{bx:.0f},{by:.0f})",
                                                        color=grid_colors[row])
                                                    with dpg.group(horizontal=True):
                                                        dpg.add_input_float(
                                                            tag=tag_x, default_value=rx,
                                                            width=58, step=0, format="%.1f")
                                                        dpg.add_input_float(
                                                            tag=tag_y, default_value=ry,
                                                            width=58, step=0, format="%.1f")
                                                    if col < 2:
                                                        dpg.add_spacer(width=2)

                                    dpg.add_separator()
                                    with dpg.group(horizontal=True):
                                        dpg.add_button(label="APPLY GRID", width=105, height=30,
                                            callback=self._apply_grid_positions)
                                        dpg.add_button(label="SAVE GRID", width=105, height=30,
                                            callback=self._save_grid_positions)
                                        dpg.add_button(label="GO TO", width=80, height=30,
                                            callback=self._go_to_grid_pos_popup)
                                        dpg.add_button(label="SET FROM BOT", width=105, height=30,
                                            callback=self._set_grid_from_robot_popup)
                
                # === TAB 3: SPECTRUM SENSOR ===
                with dpg.tab(label="Spectrum Scan"):
                    with dpg.group(horizontal=True):
                        # Left side - Manual Scan
                        with dpg.child_window(width=400, height=-1, border=True):
                            dpg.add_text("MANUAL SPECTRUM SCAN", color=(0, 255, 255))
                            dpg.add_separator()
                            
                            dpg.add_text("Sensor Status:", color=(150, 150, 150))
                            with dpg.group(horizontal=True):
                                status = "Ready" if self.spectrum.is_ready else "Not Ready"
                                dpg.add_text(status, tag="txt_spectrum_status_tab",
                                            color=(0, 255, 0) if self.spectrum.is_ready else (255, 100, 100))
                            
                            dpg.add_separator()
                            dpg.add_button(label="SCAN NOW", width=200, height=50,
                                          callback=self._manual_spectrum_scan)
                            
                            dpg.add_separator()
                            dpg.add_text("Last Prediction:", color=(150, 150, 150))
                            dpg.add_text("---", tag="txt_spectrum_pred", color=(0, 255, 0))
                            dpg.add_text("Confidence:", color=(150, 150, 150))
                            dpg.add_text("---", tag="txt_spectrum_conf", color=(255, 255, 0))
                            
                            dpg.add_separator()
                            dpg.add_text("LED Control:", color=(0, 255, 255))
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="IR ON", width=80, callback=lambda: self._toggle_led('ir', True))
                                dpg.add_button(label="IR OFF", width=80, callback=lambda: self._toggle_led('ir', False))
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="WHITE ON", width=80, callback=lambda: self._toggle_led('white', True))
                                dpg.add_button(label="WHITE OFF", width=80, callback=lambda: self._toggle_led('white', False))
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="UV ON", width=80, callback=lambda: self._toggle_led('uv', True))
                                dpg.add_button(label="UV OFF", width=80, callback=lambda: self._toggle_led('uv', False))
                        
                        dpg.add_spacer(width=20)
                        
                        # Right side - Raw Data Display
                        with dpg.child_window(width=500, height=-1, border=True):
                            dpg.add_text("RAW SPECTRUM DATA", color=(0, 255, 255))
                            dpg.add_separator()
                            dpg.add_input_text(multiline=True, height=400, width=-1,
                                              tag="txt_spectrum_raw", readonly=True)
                            
                            dpg.add_separator()
                            dpg.add_text("Channel Values:", color=(0, 255, 255))
                            dpg.add_input_text(multiline=True, height=150, width=-1,
                                              tag="txt_spectrum_channels", readonly=True)
                
                # === TAB 4: VIDEO & RECORDING ===
                with dpg.tab(label="Video & Recording"):
                    with dpg.group(horizontal=True):
                        # --- Left: Playback + Recording ---
                        with dpg.child_window(width=700, height=-1, border=True):
                            # ── Video Playback Section ──
                            dpg.add_text("VIDEO PLAYBACK (.bag / .mp4)", color=(0, 255, 255))
                            dpg.add_separator()
                            dpg.add_text("Load a recorded .bag or .mp4 file and replay it through\n"
                                         "the full detection + tracking pipeline.\n"
                                         "For .mp4, if *_replay.csv/xlsx exists it will replay H/X/Y+Spectrum.",
                                         color=(150, 150, 150))
                            dpg.add_spacer(height=8)

                            # File selection
                            dpg.add_text("Loaded file:", color=(180, 180, 180))
                            dpg.add_text("(none)", tag="txt_vt_bag_path",
                                         color=(160, 120, 255))
                            dpg.add_spacer(height=4)

                            with dpg.group(horizontal=True):
                                dpg.add_button(label="LOAD .bag/.mp4", width=160, height=50,
                                              tag="btn_vt_load",
                                              callback=lambda s,a,u: self._open_video_dialog())
                                self._vt_load_theme = self._create_video_btn_theme()
                                dpg.bind_item_theme("btn_vt_load", self._vt_load_theme)

                                dpg.add_button(label="[>] PLAY", width=160, height=50,
                                              tag="btn_vt_play",
                                              callback=lambda s,a,u: self._vt_play(),
                                              enabled=False)
                                dpg.bind_item_theme("btn_vt_play", self._create_play_btn_theme())

                                dpg.add_button(label="[=] STOP", width=160, height=50,
                                              tag="btn_vt_stop",
                                              callback=lambda s,a,u: self._vt_stop(),
                                              enabled=False)
                                dpg.bind_item_theme("btn_vt_stop", self._create_stop_btn_theme())

                            dpg.add_spacer(height=4)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="UNLOAD", width=120, height=36,
                                              tag="btn_vt_unload",
                                              callback=lambda s,a,u: self._vt_unload(),
                                              enabled=False)

                            dpg.add_spacer(height=4)
                            dpg.add_text("Playback Status:", color=(180, 180, 180))
                            dpg.add_text("Idle - no video loaded", tag="txt_vt_status",
                                         color=(120, 120, 120))

                            dpg.add_spacer(height=4)
                            dpg.add_text("OPTIONS", color=(0, 255, 255))
                            dpg.add_checkbox(label="Auto START + Track on Play",
                                            tag="chk_vt_auto_start", default_value=True)
                            dpg.add_checkbox(label="Auto STOP on video stop",
                                            tag="chk_vt_auto_stop", default_value=True)
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Limit MP4 playback passes",
                                                tag="chk_vt_mp4_loop_limit",
                                                default_value=False)
                                dpg.add_input_int(tag="in_vt_mp4_loop_count",
                                                  default_value=8,
                                                  min_value=1,
                                                  min_clamped=True,
                                                  width=90,
                                                  step=1,
                                                  step_fast=4)
                            dpg.add_checkbox(label="Enable Benchmark Capture (5s CSV flush)",
                                            tag="chk_benchmark_enable",
                                            default_value=self.benchmark_enabled_default)
                            dpg.add_text("Benchmark: idle", tag="txt_benchmark_status",
                                         color=(120, 120, 120))

                            dpg.add_spacer(height=8)
                            dpg.add_separator()
                            dpg.add_spacer(height=4)

                            # ── Recording Section ──
                            dpg.add_text("RECORDING", color=(0, 255, 255))
                            dpg.add_separator()
                            dpg.add_text("Record live camera data (.bag) or raw camera feed\n"
                                         "to .mp4 with replay metadata sidecar.",
                                         color=(150, 150, 150))
                            dpg.add_spacer(height=8)

                            # .bag recording (raw RGB + Depth)
                            dpg.add_text(".bag Recording (RGB + Depth):", color=(255, 180, 100))
                            dpg.add_text("Records raw RealSense streams.",
                                         color=(120, 120, 120))
                            dpg.add_spacer(height=4)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="(o) REC .bag", width=180, height=50,
                                              tag="btn_rec_bag",
                                              callback=lambda s,a,u: self._toggle_bag_recording())
                                self._rec_bag_theme_off = self._create_rec_bag_theme()
                                self._rec_bag_theme_on = self._create_rec_bag_active_theme()
                                dpg.bind_item_theme("btn_rec_bag", self._rec_bag_theme_off)

                            dpg.add_text("Not recording", tag="txt_rec_bag_status",
                                         color=(120, 120, 120))

                            dpg.add_spacer(height=8)
                            dpg.add_separator()
                            dpg.add_spacer(height=4)

                            # Experiment recording (screen + robot cam linked)
                            dpg.add_text("Experiment Recording (.mp4):", color=(255, 180, 100))
                            dpg.add_text("Records raw vision feed + robot camera and saves\n"
                                         "linked replay CSV/Excel (H/X/Y + 18ch spectrum).",
                                         color=(120, 120, 120))
                            dpg.add_spacer(height=4)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="(o) REC Experiment", width=200, height=50,
                                              tag="btn_rec_mp4",
                                              callback=lambda s,a,u: self._toggle_mp4_recording())
                                self._rec_mp4_theme_off = self._create_rec_mp4_theme()
                                self._rec_mp4_theme_on = self._create_rec_bag_active_theme()
                                dpg.bind_item_theme("btn_rec_mp4", self._rec_mp4_theme_off)

                            dpg.add_text("Not recording", tag="txt_rec_mp4_status",
                                         color=(120, 120, 120))
                            dpg.add_spacer(height=4)
                            dpg.add_text("", tag="txt_rec_robot_cam_note",
                                         color=(120, 120, 120))

                            dpg.add_spacer(height=4)
                            dpg.add_text("Saved recordings → ./recordings/",
                                         color=(120, 120, 120))

                            dpg.add_spacer(height=8)
                            dpg.add_separator()
                            dpg.add_spacer(height=4)

                            # ── Audit Log Collector ──
                            dpg.add_text("Pipeline Audit Log:", color=(255, 180, 100))
                            dpg.add_text("Captures every pick's fusion results across\n"
                                         "all 10 methods for offline analysis.",
                                         color=(120, 120, 120))
                            dpg.add_spacer(height=4)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="  Start Audit Log", width=200, height=50,
                                              tag="btn_audit_log",
                                              callback=lambda s,a,u: self._toggle_audit_log())
                                self._audit_theme_off = self._create_audit_btn_theme()
                                self._audit_theme_on = self._create_rec_bag_active_theme()
                                dpg.bind_item_theme("btn_audit_log", self._audit_theme_off)

                            dpg.add_text("Idle — not collecting", tag="txt_audit_status",
                                         color=(120, 120, 120))

                # === TAB 5: CALIBRATION ===
                with dpg.tab(label="Calibration", tag="tab_calibration"):
                    with dpg.group(horizontal=True):
                        # --- Left: Calibration Controls ---
                        with dpg.child_window(width=500, height=-1, border=True):
                            dpg.add_text("ROI & FLOOR CALIBRATION", color=(0, 255, 255))
                            dpg.add_separator()
                            dpg.add_text(
                                "Calibrate the conveyor belt ROI using a checkerboard\n"
                                "pattern. The camera must be running (press START first).\n"
                                "Place an 8x10 checkerboard (2.5cm squares) on the belt.",
                                color=(150, 150, 150))
                            dpg.add_spacer(height=8)

                            # Enable / Disable calibration mode
                            dpg.add_text("Step 1: Enable Calibration Overlay", color=(255, 200, 100))
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="ENABLE CALIBRATION", width=200, height=45,
                                              tag="btn_cal_enable",
                                              callback=lambda: self._cal_toggle(True))
                                dpg.add_button(label="DISABLE", width=120, height=45,
                                              tag="btn_cal_disable",
                                              callback=lambda: self._cal_toggle(False),
                                              enabled=False)
                            dpg.add_spacer(height=4)
                            dpg.add_text("Checkerboard: not detected", tag="txt_cal_board_status",
                                         color=(255, 80, 80))

                            dpg.add_spacer(height=10)
                            dpg.add_separator()
                            dpg.add_text("Step 2: Adjust Orientation", color=(255, 200, 100))
                            dpg.add_text(
                                "Flip the ROI direction if entry/exit are swapped,\n"
                                "or mirror left/right if needed.",
                                color=(120, 120, 120))
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="FLIP VERTICAL (swap entry/exit)",
                                              width=260, height=36,
                                              tag="btn_cal_flip_v",
                                              callback=self._cal_flip_v, enabled=False)
                                dpg.add_button(label="FLIP HORIZONTAL (mirror L/R)",
                                              width=260, height=36,
                                              tag="btn_cal_flip_h",
                                              callback=self._cal_flip_h, enabled=False)
                            with dpg.group(horizontal=True):
                                dpg.add_text("V-Flip: OFF", tag="txt_cal_flip_v", color=(120, 120, 120))
                                dpg.add_spacer(width=20)
                                dpg.add_text("H-Flip: OFF", tag="txt_cal_flip_h", color=(120, 120, 120))

                            dpg.add_spacer(height=10)
                            dpg.add_separator()
                            dpg.add_text("Step 3: Confirm Calibration", color=(255, 200, 100))
                            dpg.add_text(
                                "Lock in the ROI corners and homography transform.\n"
                                "The yellow preview will turn green when confirmed.",
                                color=(120, 120, 120))
                            dpg.add_button(label="CONFIRM CALIBRATION", width=240, height=40,
                                          tag="btn_cal_confirm",
                                          callback=self._cal_confirm, enabled=False)
                            dpg.add_text("Not confirmed", tag="txt_cal_confirm_status",
                                         color=(255, 80, 80))

                            dpg.add_spacer(height=10)
                            dpg.add_separator()
                            dpg.add_text("Step 4: Calibrate Floor", color=(255, 200, 100))
                            dpg.add_text(
                                "Fit a floor plane from depth data and create a\n"
                                "floor depth map for accurate height measurement.\n"
                                "Remove all objects from the belt before pressing.",
                                color=(120, 120, 120))
                            dpg.add_button(label="CALIBRATE FLOOR", width=200, height=40,
                                          tag="btn_cal_floor",
                                          callback=self._cal_floor, enabled=False)
                            dpg.add_text("Floor: not calibrated", tag="txt_cal_floor_status",
                                         color=(255, 80, 80))

                            dpg.add_spacer(height=10)
                            dpg.add_separator()
                            dpg.add_text("Step 5: Save & Apply", color=(255, 200, 100))
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="SAVE CALIBRATION", width=200, height=45,
                                              tag="btn_cal_save",
                                              callback=self._cal_save, enabled=False)
                                dpg.add_button(label="RELOAD FROM FILE", width=180, height=45,
                                              callback=self._on_reload_calibration)
                            dpg.add_text("", tag="txt_cal_save_status", color=(120, 120, 120))

                        dpg.add_spacer(width=20)

                        # --- Right: Calibration Info & Instructions ---
                        with dpg.child_window(width=-1, height=-1, border=True):
                            dpg.add_text("CALIBRATION INFO", color=(0, 255, 255))
                            dpg.add_separator()

                            dpg.add_text("Checkerboard Specs:", color=(180, 180, 180))
                            dpg.add_text(
                                "  Pattern: 8 x 10 squares (7 x 9 inner corners)\n"
                                "  Square size: 2.5 cm x 2.5 cm\n"
                                "  Total board: 20 cm x 25 cm",
                                color=(120, 120, 120))

                            dpg.add_spacer(height=8)
                            dpg.add_text("Target ROI:", color=(180, 180, 180))
                            dpg.add_text(
                                f"  Width: {ROI_WIDTH_CM} cm\n"
                                f"  Height: {ROI_HEIGHT_CM} cm\n"
                                f"  Entry zone: {ENTRY_PATH_CM} cm\n"
                                f"  Exit zone: {EXIT_PATH_CM} cm",
                                color=(120, 120, 120))

                            dpg.add_spacer(height=8)
                            dpg.add_separator()
                            dpg.add_text("Current Calibration:", color=(180, 180, 180))
                            has_cal = self.detector.homography is not None
                            dpg.add_text(
                                f"  Loaded: {'YES' if has_cal else 'NO'}\n"
                                f"  ROI: {self.detector.roi_width_cm}x{self.detector.roi_height_cm} cm"
                                if has_cal else "  No calibration loaded",
                                tag="txt_cal_current_info",
                                color=(0, 255, 0) if has_cal else (255, 100, 100))

                            dpg.add_spacer(height=8)
                            dpg.add_separator()
                            dpg.add_text("Instructions:", color=(255, 200, 100))
                            dpg.add_text(
                                "1. Press START to begin camera feed\n"
                                "2. Place checkerboard flat on conveyor belt\n"
                                "3. Click ENABLE CALIBRATION — yellow ROI preview appears\n"
                                "4. Use FLIP buttons if entry/exit or L/R are wrong\n"
                                "5. Click CONFIRM — ROI turns green\n"
                                "6. Remove objects from belt, click CALIBRATE FLOOR\n"
                                "7. Click SAVE CALIBRATION — writes calibration_data.json\n"
                                "8. Calibration is applied immediately (no restart needed)",
                                color=(150, 150, 150))

                            dpg.add_spacer(height=8)
                            dpg.add_separator()
                            dpg.add_text("Calibration Log:", color=(180, 180, 180))
                            dpg.add_child_window(tag="cal_log_child", height=-1, border=False)

                # === TAB 6: ANALYTICS DASHBOARD ===
                with dpg.tab(label="Dashboard", tag="tab_dashboard"):
                    self._build_dashboard_tab()

        # Add tooltips for buttons (must be after all tabs are created)
        with dpg.tooltip("btn_start"):
            dpg.add_text("Start camera feed and object detection")
        with dpg.tooltip("btn_stop"):
            dpg.add_text("Stop all operations")
        with dpg.tooltip("btn_autopick"):
            dpg.add_text("Toggle automatic object picking & sorting")
        with dpg.tooltip("btn_calibrate"):
            dpg.add_text("Switch to Calibration tab")
        with dpg.tooltip("btn_reload_cal"):
            dpg.add_text("Reload calibration data from file")
        with dpg.tooltip("btn_dummy"):
            dpg.add_text("Inject test object for debugging")
        with dpg.tooltip("btn_track"):
            dpg.add_text("Toggle robot tracking mode (follows objects)")
    
    # ========== TAB 6: ANALYTICS DASHBOARD ==========

    def _build_dashboard_tab(self):
        """Build the analytics dashboard tab (Tab 6) UI layout."""
        # --- Internal state for dashboard ---
        self._dash_selected_date = datetime.now().strftime("%Y-%m-%d")
        self._dash_csv_cache = {}          # {date_str: list of row dicts}
        self._dash_last_refresh = 0        # monotonic time of last full refresh
        self._dash_class_colors = {
            "Glass":   (255, 215, 0),      # gold
            "Metal":   (192, 192, 192),    # silver
            "Plastic": (0, 180, 255),      # blue
            "Paper":   (80, 200, 80),      # green
        }

        # === Top row: Date navigation ===
        with dpg.group(horizontal=True):
            dpg.add_button(label="<  Prev Day", width=120, height=32,
                           callback=lambda: self._dash_change_date(-1))
            dpg.add_text("", tag="dash_date_label", color=(0, 220, 255))
            dpg.add_button(label="Next Day  >", width=120, height=32,
                           callback=lambda: self._dash_change_date(+1))
            dpg.add_spacer(width=30)
            dpg.add_button(label="Today", width=80, height=32,
                           callback=lambda: self._dash_go_today())
            dpg.add_spacer(width=30)
            dpg.add_text("Live Session ▸", color=(150, 150, 150))
            dpg.add_text("--", tag="dash_session_id", color=(100, 220, 100))

        dpg.add_separator()
        dpg.add_spacer(height=4)

        # === Row 1: Stat cards ===
        with dpg.group(horizontal=True):
            card_w, card_h = 220, 100
            # Card 1: Total Detected
            with dpg.child_window(width=card_w, height=card_h, border=True):
                dpg.add_text("Total Detected", color=(150, 150, 150))
                dpg.add_text("0", tag="dash_total", color=(0, 220, 255))
                dpg.add_text("objects sorted today", color=(100, 100, 100))
            dpg.add_spacer(width=8)
            # Card 2: Pick Success Rate
            with dpg.child_window(width=card_w, height=card_h, border=True):
                dpg.add_text("Pick Success Rate", color=(150, 150, 150))
                dpg.add_text("-- %", tag="dash_success_rate", color=(80, 220, 120))
                dpg.add_text("completed / total picks", color=(100, 100, 100))
            dpg.add_spacer(width=8)
            # Card 3: Belt Speed
            with dpg.child_window(width=card_w, height=card_h, border=True):
                dpg.add_text("Belt Speed", color=(150, 150, 150))
                dpg.add_text("-- cm/s", tag="dash_belt_speed", color=(255, 200, 80))
                dpg.add_text("measured / slider", color=(100, 100, 100))
            dpg.add_spacer(width=8)
            # Card 4: Sessions Today
            with dpg.child_window(width=card_w, height=card_h, border=True):
                dpg.add_text("Sessions", color=(150, 150, 150))
                dpg.add_text("0", tag="dash_sessions_count", color=(200, 150, 255))
                dpg.add_text("log files for this day", color=(100, 100, 100))

        dpg.add_spacer(height=8)

        # === Row 2: Charts side-by-side ===
        with dpg.group(horizontal=True):
            # --- Left: Class distribution (pie-like bar chart) ---
            with dpg.child_window(width=480, height=340, border=True):
                dpg.add_text("Class Distribution", color=(0, 255, 255))
                dpg.add_separator()
                with dpg.plot(label="##pie_plot", width=-1, height=-1, tag="dash_pie_plot",
                              no_mouse_pos=True):
                    dpg.add_plot_legend()
                    ax_x = dpg.add_plot_axis(dpg.mvXAxis, label="", tag="dash_pie_x",
                                             no_tick_labels=True)
                    ax_y = dpg.add_plot_axis(dpg.mvYAxis, label="Count", tag="dash_pie_y")
                    # Pre-create bar series for each class
                    for i, cls in enumerate(["Glass", "Metal", "Plastic", "Paper"]):
                        clr = self._dash_class_colors[cls]
                        dpg.add_bar_series([i], [0], label=cls, weight=0.6,
                                           tag=f"dash_bar_{cls}", parent=ax_y)
                        dpg.bind_item_theme(f"dash_bar_{cls}",
                                            self._create_bar_theme(clr))
                    dpg.set_axis_limits(ax_x, -0.5, 3.5)

            dpg.add_spacer(width=8)

            # --- Right: Hourly trend chart ---
            with dpg.child_window(width=-1, height=340, border=True):
                dpg.add_text("Hourly Detection Trend", color=(0, 255, 255))
                dpg.add_separator()
                with dpg.plot(label="##trend_plot", width=-1, height=-1, tag="dash_trend_plot",
                              no_mouse_pos=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Hour", tag="dash_trend_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Count", tag="dash_trend_y")
                    for cls in ["Glass", "Metal", "Plastic", "Paper"]:
                        clr = self._dash_class_colors[cls]
                        dpg.add_line_series([], [], label=cls,
                                            tag=f"dash_line_{cls}",
                                            parent="dash_trend_y")
                        dpg.bind_item_theme(f"dash_line_{cls}",
                                            self._create_line_theme(clr))

        dpg.add_spacer(height=8)

        # === Row 3: Session log table ===
        with dpg.child_window(width=-1, height=-1, border=True):
            dpg.add_text("Session Log (CSV Files)", color=(0, 255, 255))
            dpg.add_separator()
            with dpg.table(tag="dash_session_table", header_row=True,
                           resizable=True, borders_innerH=True,
                           borders_outerH=True, borders_innerV=True,
                           borders_outerV=True, row_background=True,
                           scrollY=True):
                dpg.add_table_column(label="Session", width_fixed=True, init_width_or_weight=180)
                dpg.add_table_column(label="Objects")
                dpg.add_table_column(label="Glass")
                dpg.add_table_column(label="Metal")
                dpg.add_table_column(label="Plastic")
                dpg.add_table_column(label="Paper")

        # Initial render
        self._dash_refresh_date_label()
        self._dash_load_and_render()

    # --- Dashboard helper: bar theme ---
    def _create_bar_theme(self, color):
        """Create a DearPyGui theme for a bar series with the given color."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvBarSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, (*color, 200),
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (*color, 255),
                                    category=dpg.mvThemeCat_Plots)
        return theme

    # --- Dashboard helper: line theme ---
    def _create_line_theme(self, color):
        """Create a DearPyGui theme for a line series with the given color."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (*color, 255),
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0,
                                    category=dpg.mvThemeCat_Plots)
        return theme

    # --- Dashboard: date navigation ---
    def _dash_change_date(self, delta_days):
        """Move selected date forward or backward."""
        dt = datetime.strptime(self._dash_selected_date, "%Y-%m-%d")
        dt += timedelta(days=delta_days)
        self._dash_selected_date = dt.strftime("%Y-%m-%d")
        self._dash_refresh_date_label()
        self._dash_load_and_render()

    def _dash_go_today(self):
        """Jump to today's date."""
        self._dash_selected_date = datetime.now().strftime("%Y-%m-%d")
        self._dash_refresh_date_label()
        self._dash_load_and_render()

    def _dash_refresh_date_label(self):
        """Update the date label text."""
        today = datetime.now().strftime("%Y-%m-%d")
        lbl = self._dash_selected_date
        if lbl == today:
            lbl += "  (Today)"
        dpg.set_value("dash_date_label", f"  {lbl}  ")

    # --- Dashboard: CSV data loading ---
    def _dash_load_csv_for_date(self, date_str):
        """Load all detection_log CSVs matching a date. Returns list of row dicts.

        Filename pattern: detection_logs/detections/detection_log_YYYYMMDD_HHMMSS.csv
        """
        date_compact = date_str.replace("-", "")   # "20260316"
        pattern = os.path.join("detection_logs", "detections", f"detection_log_{date_compact}_*.csv")
        files = sorted(glob.glob(pattern))

        rows = []
        for fpath in files:
            # Extract session timestamp from filename
            basename = os.path.splitext(os.path.basename(fpath))[0]
            # detection_log_20260316_181641 -> session = "20260316_181641"
            parts = basename.replace("detection_log_", "")
            session_id = parts  # e.g. "20260316_181641"

            try:
                with open(fpath, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row['_session'] = session_id
                        row['_file'] = fpath
                        rows.append(row)
            except Exception as e:
                print(f"[DASHBOARD] Error reading {fpath}: {e}")

        return rows, files

    # --- Dashboard: full render from CSV data ---
    def _dash_load_and_render(self):
        """Load CSV data for selected date and update all dashboard widgets."""
        date_str = self._dash_selected_date
        rows, files = self._dash_load_csv_for_date(date_str)

        # --- Stat cards ---
        total = len(rows)
        dpg.set_value("dash_total", str(total))
        dpg.set_value("dash_sessions_count", str(len(files)))
        dpg.set_value("dash_session_id", self.session_id)

        # Class counts
        class_counts = {"Glass": 0, "Metal": 0, "Plastic": 0, "Paper": 0}
        session_data = {}  # session_id -> {cls: count}
        hourly_data = {}   # hour_int -> {cls: count}

        for row in rows:
            cls = row.get("Final_Class", row.get("Camera_Class", "Unknown"))
            if cls in class_counts:
                class_counts[cls] += 1

            sid = row.get("_session", "unknown")
            if sid not in session_data:
                session_data[sid] = {"Glass": 0, "Metal": 0, "Plastic": 0, "Paper": 0, "_total": 0}
            if cls in session_data[sid]:
                session_data[sid][cls] += 1
            session_data[sid]["_total"] += 1

            # Hourly: parse hour from session id (YYYYMMDD_HHMMSS)
            try:
                hour = int(sid.split("_")[1][:2])
            except (IndexError, ValueError):
                hour = 0
            if hour not in hourly_data:
                hourly_data[hour] = {"Glass": 0, "Metal": 0, "Plastic": 0, "Paper": 0}
            if cls in hourly_data[hour]:
                hourly_data[hour][cls] += 1

        # --- Update bar chart (class distribution) ---
        for i, cls in enumerate(["Glass", "Metal", "Plastic", "Paper"]):
            dpg.set_value(f"dash_bar_{cls}", [[i], [class_counts[cls]]])
        max_cls_count = max(class_counts.values()) if any(class_counts.values()) else 1
        dpg.set_axis_limits("dash_pie_y", 0, max_cls_count * 1.2)
        # Set tick labels for X axis
        dpg.set_axis_limits("dash_pie_x", -0.5, 3.5)
        dpg.set_axis_ticks("dash_pie_x",
                           (("Glass", 0), ("Metal", 1), ("Plastic", 2), ("Paper", 3)))

        # --- Update line chart (hourly trend) ---
        if hourly_data:
            hours_sorted = sorted(hourly_data.keys())
            all_hours = list(range(hours_sorted[0], hours_sorted[-1] + 1))
        else:
            all_hours = list(range(0, 24))

        max_hourly = 1
        for cls in ["Glass", "Metal", "Plastic", "Paper"]:
            xs = [float(h) for h in all_hours]
            ys = [float(hourly_data.get(h, {}).get(cls, 0)) for h in all_hours]
            dpg.set_value(f"dash_line_{cls}", [xs, ys])
            if ys:
                max_hourly = max(max_hourly, max(ys))
        dpg.set_axis_limits("dash_trend_x", all_hours[0] - 0.5, all_hours[-1] + 0.5)
        dpg.set_axis_limits("dash_trend_y", 0, max_hourly * 1.2)
        # Hour tick labels
        tick_labels = tuple((f"{h:02d}:00", h) for h in all_hours)
        dpg.set_axis_ticks("dash_trend_x", tick_labels)

        # --- Update session table ---
        dpg.delete_item("dash_session_table", children_only=True, slot=1)
        for sid in sorted(session_data.keys(), reverse=True):
            sd = session_data[sid]
            with dpg.table_row(parent="dash_session_table"):
                dpg.add_text(sid)
                dpg.add_text(str(sd["_total"]))
                dpg.add_text(str(sd["Glass"]), color=self._dash_class_colors["Glass"])
                dpg.add_text(str(sd["Metal"]), color=self._dash_class_colors["Metal"])
                dpg.add_text(str(sd["Plastic"]), color=self._dash_class_colors["Plastic"])
                dpg.add_text(str(sd["Paper"]), color=self._dash_class_colors["Paper"])

    def _update_analytics_live(self):
        """Update live stat cards on the dashboard (called periodically from main loop).
        
        Only updates the live indicators (belt speed, success rate, session id)
        — the CSV-based charts only refresh on date change or every 60 seconds.
        """
        if not dpg.does_item_exist("dash_belt_speed"):
            return

        # Belt speed card
        ms = self.tracker.measured_belt_speed
        ui_spd = dpg.get_value("in_speed") if dpg.does_item_exist("in_speed") else 6.0
        if ms is not None:
            dpg.set_value("dash_belt_speed", f"{ms:.2f} cm/s")
            dpg.configure_item("dash_belt_speed", color=(100, 220, 100))
        else:
            dpg.set_value("dash_belt_speed", f"{ui_spd:.1f} cm/s (slider)")
            dpg.configure_item("dash_belt_speed", color=(255, 200, 80))

        # Session id
        dpg.set_value("dash_session_id", self.session_id)

        # Live pick success rate — from sorted_counts
        total_sorted = sum(self.sorted_counts.values())
        total_det = self.total_detected
        if total_det > 0:
            rate = (total_sorted / total_det) * 100
            dpg.set_value("dash_success_rate", f"{rate:.1f} %")
            color = (80, 220, 120) if rate >= 80 else (255, 220, 50) if rate >= 50 else (255, 80, 80)
            dpg.configure_item("dash_success_rate", color=color)
        else:
            dpg.set_value("dash_success_rate", "-- %")

        # Auto-refresh CSV data for today every ~60 seconds
        now = time.monotonic()
        today = datetime.now().strftime("%Y-%m-%d")
        if self._dash_selected_date == today and (now - self._dash_last_refresh) > 60:
            self._dash_last_refresh = now
            self._dash_load_and_render()

    def _load_offsets(self):
        """Load offset values from file."""
        if os.path.exists(OFFSETS_FILE):
            try:
                with open(OFFSETS_FILE, 'r') as f:
                    d = json.load(f)
                dpg.set_value("off_x", d.get("x", 0.0))
                dpg.set_value("off_y", d.get("y", 0.0))
                dpg.set_value("off_z", d.get("z", 0.0))
                dpg.set_value("off_lat", d.get("latency", -0.5))
                # Per-class Z offsets
                cz = d.get("class_z_offsets", {})
                for cls_name in ["Glass", "Metal", "Paper", "Plastic"]:
                    tag = f"czoff_{cls_name}"
                    if dpg.does_item_exist(tag):
                        dpg.set_value(tag, cz.get(cls_name, CLASS_Z_OFFSET_MM.get(cls_name, 0.0)))
                # Stack-bottom offsets
                if dpg.does_item_exist("sbot_y_advance"):
                    dpg.set_value("sbot_y_advance", d.get("stack_bottom_y_advance", 2.0))
                if dpg.does_item_exist("sbot_z_extra"):
                    dpg.set_value("sbot_z_extra", d.get("stack_bottom_z_extra", -10.0))
                self.log("Offsets loaded")
            except:
                pass
    
    def _save_offsets(self):
        """Save offset values to file."""
        try:
            d = {
                "x": dpg.get_value("off_x"),
                "y": dpg.get_value("off_y"),
                "z": dpg.get_value("off_z"),
                "latency": dpg.get_value("off_lat"),
                "class_z_offsets": {
                    cls: dpg.get_value(f"czoff_{cls}")
                    for cls in ["Glass", "Metal", "Paper", "Plastic"]
                },
                "stack_bottom_y_advance": dpg.get_value("sbot_y_advance"),
                "stack_bottom_z_extra": dpg.get_value("sbot_z_extra"),
            }
            with open(OFFSETS_FILE, 'w') as f:
                json.dump(d, f)
            self.log("Offsets saved")
        except Exception as e:
            self.log(f"Failed to save offsets: {e}", "ERROR")
    
    # ========== ROBOT CONTROL TAB FUNCTIONS ==========
    
    # --- Place Target Functions ---
    
    def _apply_place_targets(self):
        """Read place target XY from UI and push to robot_manager."""
        targets = {}
        for cls_name in ["Glass", "Metal", "Paper", "Plastic"]:
            x = dpg.get_value(f"place_{cls_name}_x")
            y = dpg.get_value(f"place_{cls_name}_y")
            targets[cls_name] = (x, y)
        
        pz = dpg.get_value("place_z_height")
        prz = dpg.get_value("place_z_release")
        
        if hasattr(self, 'robot_manager') and self.robot_manager is not None:
            self.robot_manager.place_targets = targets
            self.robot_manager.place_z = pz
            self.robot_manager.place_z_release = prz
        
        self.log(f"Place targets applied: {targets}")
    
    def _save_place_targets(self):
        """Save place targets to JSON file."""
        self._apply_place_targets()  # Ensure latest UI values are applied
        try:
            data = {}
            for cls_name in ["Glass", "Metal", "Paper", "Plastic"]:
                data[cls_name] = {
                    "x": dpg.get_value(f"place_{cls_name}_x"),
                    "y": dpg.get_value(f"place_{cls_name}_y"),
                }
            data["_place_z"] = dpg.get_value("place_z_height")
            data["_release_z"] = dpg.get_value("place_z_release")
            
            with open(PLACE_TARGETS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            self.log("Place targets saved", "SUCCESS")
        except Exception as e:
            self.log(f"Failed to save place targets: {e}", "ERROR")
    
    def _load_place_targets(self):
        """Load place targets from JSON file."""
        if not os.path.exists(PLACE_TARGETS_FILE):
            self.log("No place_targets.json found", "ERROR")
            return
        try:
            with open(PLACE_TARGETS_FILE, 'r') as f:
                data = json.load(f)
            for cls_name in ["Glass", "Metal", "Paper", "Plastic"]:
                if cls_name in data:
                    dpg.set_value(f"place_{cls_name}_x", data[cls_name]["x"])
                    dpg.set_value(f"place_{cls_name}_y", data[cls_name]["y"])
            if "_place_z" in data:
                dpg.set_value("place_z_height", data["_place_z"])
            if "_release_z" in data:
                dpg.set_value("place_z_release", data["_release_z"])
            self._apply_place_targets()
            self.log("Place targets loaded", "SUCCESS")
        except Exception as e:
            self.log(f"Failed to load place targets: {e}", "ERROR")
    
    def _reset_place_targets(self):
        """Reset place targets to config defaults."""
        for cls_name in ["Glass", "Metal", "Paper", "Plastic"]:
            dx, dy = THROW_TARGETS.get(cls_name, (0, 0))
            dpg.set_value(f"place_{cls_name}_x", dx)
            dpg.set_value(f"place_{cls_name}_y", dy)
        dpg.set_value("place_z_height", PLACE_Z_HEIGHT)
        dpg.set_value("place_z_release", PLACE_Z_RELEASE)
        self._apply_place_targets()
        self.log("Place targets reset to defaults")
    
    def _go_place_pos(self, cls_name):
        """Move robot to the place position for a given class (for testing)."""
        if not self.delta.connected:
            self.log("Robot not connected!", "ERROR")
            return
        x = dpg.get_value(f"place_{cls_name}_x")
        y = dpg.get_value(f"place_{cls_name}_y")
        z = dpg.get_value("place_z_height")
        self.delta.move_to(x, y, z)
        self.robot_pos.update({'x': x, 'y': y, 'z': z})
        self._update_position_display()
        self.log(f"Moving to {cls_name} place: ({x:.1f}, {y:.1f}, {z:.1f})")
    
    def _set_place_from_robot(self, cls_name):
        """Set place position for a class from the robot's current position."""
        rx = self.delta.last_x if self.delta.connected else self.robot_pos['x']
        ry = self.delta.last_y if self.delta.connected else self.robot_pos['y']
        dpg.set_value(f"place_{cls_name}_x", rx)
        dpg.set_value(f"place_{cls_name}_y", ry)
        self.log(f"Set {cls_name} place from robot: ({rx:.1f}, {ry:.1f})")
    
    # --- Workspace Grid Functions ---
    
    def _apply_grid_positions(self):
        """Read grid XY from UI and update the live ROBOT_X_GRID / ROBOT_Y_GRID."""
        grid_labels = [
            ["TL", "TC", "TR"],
            ["ML", "MC", "MR"],
            ["BL", "BC", "BR"],
        ]
        for row in range(3):
            for col in range(3):
                lbl = grid_labels[row][col]
                ROBOT_X_GRID[row][col] = dpg.get_value(f"grid_{lbl}_x")
                ROBOT_Y_GRID[row][col] = dpg.get_value(f"grid_{lbl}_y")
        self.log("Workspace grid updated (live)")
    
    def _save_grid_positions(self):
        """Save workspace grid + place targets together to a single file."""
        self._apply_grid_positions()
        self._apply_place_targets()
        try:
            data = {
                "robot_x_grid": [list(row) for row in ROBOT_X_GRID],
                "robot_y_grid": [list(row) for row in ROBOT_Y_GRID],
            }
            # Include place targets in same file
            place = {}
            for cls_name in ["Glass", "Metal", "Paper", "Plastic"]:
                place[cls_name] = {
                    "x": dpg.get_value(f"place_{cls_name}_x"),
                    "y": dpg.get_value(f"place_{cls_name}_y"),
                }
            place["_place_z"] = dpg.get_value("place_z_height")
            place["_release_z"] = dpg.get_value("place_z_release")
            data["place_targets"] = place
            
            with open(PLACE_TARGETS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            self.log("Grid + place targets saved", "SUCCESS")
        except Exception as e:
            self.log(f"Failed to save grid: {e}", "ERROR")
    
    def _load_grid_and_place(self):
        """Load workspace grid + place targets from file (called at startup)."""
        if not os.path.exists(PLACE_TARGETS_FILE):
            return
        try:
            with open(PLACE_TARGETS_FILE, 'r') as f:
                data = json.load(f)
            
            grid_labels = [
                ["TL", "TC", "TR"],
                ["ML", "MC", "MR"],
                ["BL", "BC", "BR"],
            ]
            
            # Load grid positions
            if "robot_x_grid" in data and "robot_y_grid" in data:
                for row in range(3):
                    for col in range(3):
                        ROBOT_X_GRID[row][col] = data["robot_x_grid"][row][col]
                        ROBOT_Y_GRID[row][col] = data["robot_y_grid"][row][col]
                        lbl = grid_labels[row][col]
                        if dpg.does_item_exist(f"grid_{lbl}_x"):
                            dpg.set_value(f"grid_{lbl}_x", ROBOT_X_GRID[row][col])
                            dpg.set_value(f"grid_{lbl}_y", ROBOT_Y_GRID[row][col])
                self.log("Workspace grid loaded from file")
            
            # Load place targets
            pt = data.get("place_targets", data)  # Support both formats
            for cls_name in ["Glass", "Metal", "Paper", "Plastic"]:
                if cls_name in pt:
                    x_val = pt[cls_name].get("x", 0)
                    y_val = pt[cls_name].get("y", 0)
                    if dpg.does_item_exist(f"place_{cls_name}_x"):
                        dpg.set_value(f"place_{cls_name}_x", x_val)
                        dpg.set_value(f"place_{cls_name}_y", y_val)
            if "_place_z" in pt and dpg.does_item_exist("place_z_height"):
                dpg.set_value("place_z_height", pt["_place_z"])
            if "_release_z" in pt and dpg.does_item_exist("place_z_release"):
                dpg.set_value("place_z_release", pt["_release_z"])
            
            self._apply_place_targets()
            self.log("Place targets loaded from file")
        except Exception as e:
            self.log(f"Failed to load grid/place file: {e}", "ERROR")
    
    def _go_to_grid_pos_popup(self):
        """Show a popup to pick which grid position to move to."""
        if dpg.does_item_exist("grid_go_popup"):
            dpg.delete_item("grid_go_popup")
        
        with dpg.window(label="Go To Grid Position", tag="grid_go_popup",
                        width=250, height=180, no_resize=True, modal=True):
            dpg.add_text("Select grid position:", color=(200, 200, 200))
            labels = ["TL", "TC", "TR", "ML", "MC", "MR", "BL", "BC", "BR"]
            for row_start in range(0, 9, 3):
                with dpg.group(horizontal=True):
                    for i in range(row_start, row_start + 3):
                        lbl = labels[i]
                        dpg.add_button(label=lbl, width=65, height=35,
                            callback=lambda s, a, u: self._go_to_grid_pos(u),
                            user_data=lbl)
            dpg.add_separator()
            dpg.add_button(label="Close", width=100,
                callback=lambda: dpg.delete_item("grid_go_popup"))
    
    def _go_to_grid_pos(self, label):
        """Move robot to a specific grid position."""
        if not self.delta.connected:
            self.log("Robot not connected!", "ERROR")
            return
        rx = dpg.get_value(f"grid_{label}_x")
        ry = dpg.get_value(f"grid_{label}_y")
        z = dpg.get_value("target_z") if dpg.does_item_exist("target_z") else -300
        self.delta.move_to(rx, ry, z)
        self.robot_pos.update({'x': rx, 'y': ry, 'z': z})
        self._update_position_display()
        self.log(f"Moving to grid {label}: ({rx:.1f}, {ry:.1f}, {z:.1f})")
        if dpg.does_item_exist("grid_go_popup"):
            dpg.delete_item("grid_go_popup")
    
    def _set_grid_from_robot_popup(self):
        """Show popup to pick which grid position to set from robot's current pos."""
        if dpg.does_item_exist("grid_set_popup"):
            dpg.delete_item("grid_set_popup")
        
        with dpg.window(label="Set Grid From Robot", tag="grid_set_popup",
                        width=250, height=180, no_resize=True, modal=True):
            dpg.add_text("Set which position?", color=(200, 200, 200))
            labels = ["TL", "TC", "TR", "ML", "MC", "MR", "BL", "BC", "BR"]
            for row_start in range(0, 9, 3):
                with dpg.group(horizontal=True):
                    for i in range(row_start, row_start + 3):
                        lbl = labels[i]
                        dpg.add_button(label=lbl, width=65, height=35,
                            callback=lambda s, a, u: self._set_grid_from_robot(u),
                            user_data=lbl)
            dpg.add_separator()
            dpg.add_button(label="Close", width=100,
                callback=lambda: dpg.delete_item("grid_set_popup"))
    
    def _set_grid_from_robot(self, label):
        """Set a grid position from the robot's current XY."""
        rx = self.delta.last_x if self.delta.connected else self.robot_pos['x']
        ry = self.delta.last_y if self.delta.connected else self.robot_pos['y']
        dpg.set_value(f"grid_{label}_x", rx)
        dpg.set_value(f"grid_{label}_y", ry)
        self.log(f"Grid {label} set from robot: ({rx:.1f}, {ry:.1f})")
        if dpg.does_item_exist("grid_set_popup"):
            dpg.delete_item("grid_set_popup")
    
    # --- Original Robot Control Functions ---
    
    def _jog_robot(self, axis, direction):
        """Jog robot in specified axis by step size."""
        if not self.delta.connected:
            self.log("Robot not connected!", "ERROR")
            return
        
        step = float(dpg.get_value("jog_step")) * direction
        
        # Apply jog to internal position tracking
        if axis == 'x':
            self.robot_pos['x'] += step
        elif axis == 'y':
            self.robot_pos['y'] += step
        elif axis == 'z':
            self.robot_pos['z'] += step
        elif axis == 'r':
            self.robot_pos['r'] += step
        
        # Clamp values
        self.robot_pos['x'] = np.clip(self.robot_pos['x'], -110, 110)
        self.robot_pos['y'] = np.clip(self.robot_pos['y'], -110, 110)
        self.robot_pos['z'] = np.clip(self.robot_pos['z'], -450, -250)
        self.robot_pos['r'] = np.clip(self.robot_pos['r'], 0, 180)
        
        # Move robot
        self.delta.move_to(
            self.robot_pos['x'], 
            self.robot_pos['y'], 
            self.robot_pos['z'], 
            w=self.robot_pos['r']
        )
        
        # Update all position displays
        self._update_position_display()
        
        self.log(f"Jog {axis.upper()}{'+' if direction > 0 else '-'}: X={self.robot_pos['x']:.1f} Y={self.robot_pos['y']:.1f} Z={self.robot_pos['z']:.1f} R={self.robot_pos['r']:.1f}")
    
    def _update_position_display(self):
        """Update all position display elements."""
        # Header display
        dpg.set_value("pos_x", f"{self.robot_pos['x']:.2f}")
        dpg.set_value("pos_y", f"{self.robot_pos['y']:.2f}")
        dpg.set_value("pos_z", f"{self.robot_pos['z']:.2f}")
        dpg.set_value("pos_r", f"{self.robot_pos['r']:.2f}")
        
        # Jog button inline displays
        dpg.set_value("pos_x_jog", f"  {self.robot_pos['x']:>7.2f}  ")
        dpg.set_value("pos_y_jog", f"  {self.robot_pos['y']:>7.2f}  ")
        dpg.set_value("pos_z_jog", f"  {self.robot_pos['z']:>7.2f}  ")
        dpg.set_value("pos_r_jog", f"  {self.robot_pos['r']:>7.2f}  ")
    
    def _go_to_position(self):
        """Move robot to target position."""
        if not self.delta.connected:
            self.log("Robot not connected!", "ERROR")
            return
        
        x = dpg.get_value("target_x")
        y = dpg.get_value("target_y")
        z = dpg.get_value("target_z")
        r = dpg.get_value("target_r")
        f = dpg.get_value("target_speed")
        

        self.delta.move_to(x, y, z, w=r, f=f)
        
        # Sync internal position tracking
        self.robot_pos['x'] = x
        self.robot_pos['y'] = y
        self.robot_pos['z'] = z
        self.robot_pos['r'] = r
        
        # Update all position displays
        self._update_position_display()
        
        self.log(f"Moving to X={x:.1f} Y={y:.1f} Z={z:.1f} R={r:.1f} F={f}")
    
    def _go_preset(self, preset):
        """Move robot to preset position."""
        if not self.delta.connected:
            self.log("Robot not connected!", "ERROR")
            return
        
        presets = {
            'standby': (0, 0, -250, 90),
            'pick': (0, 0, -400, 90),
            'scan': (0, 0, -250, 90),
            'drop': (0, -120, -300, 90)
        }
        
        if preset in presets:
            x, y, z, r = presets[preset]
            self.delta.move_to(x, y, z, w=r)
            
            # Sync internal position tracking
            self.robot_pos['x'] = x
            self.robot_pos['y'] = y
            self.robot_pos['z'] = z
            self.robot_pos['r'] = r
            
            # Update all position displays
            self._update_position_display()
            
            self.log(f"Moving to {preset.upper()} position")
    
    def _pick_test(self):
        """Test pick at all 4 corners of robot workspace: TL, TR, BL, BR.
        Moves to each corner at hover Z, descends to -400, turns on vacuum,
        lifts back up, turns off vacuum, then moves to next corner."""
        if not self.delta.connected:
            self.log("Robot not connected!", "ERROR")
            return

        self.log("Starting PICK TEST (4 corners)...")

        PICK_Z   = -400
        HOVER_Z  = -300
        DWELL_S  = 0.8     # Vacuum hold time at pick Z

        # 4 corners in belt coordinates (mm)
        corners = [
            ("TL", 0,   0),
            ("TR", 200, 0),
            ("BR", 200, 200),
            ("BL", 0,   200),
        ]

        import threading
        def run_test():
            try:
                for label, belt_x, belt_y in corners:
                    robot_x, robot_y = bilinear_interpolate(belt_x, belt_y)

                    # 1. Move to corner at hover height
                    self.delta.move_to(robot_x, robot_y, HOVER_Z)
                    self.log(f"[PICK TEST] {label}  Belt({belt_x},{belt_y}) "
                             f"-> Robot({robot_x:.1f},{robot_y:.1f})  hover Z={HOVER_Z}")
                    time.sleep(0.5)

                    # 2. Descend to pick Z
                    self.delta.move_to(robot_x, robot_y, PICK_Z)
                    self.log(f"[PICK TEST] {label}  descend Z={PICK_Z}")
                    time.sleep(0.3)

                    # 3. Vacuum ON
                    self.delta.set_vacuum(True)
                    self.log(f"[PICK TEST] {label}  vacuum ON")
                    time.sleep(DWELL_S)

                    # 4. Lift back to hover
                    self.delta.move_to(robot_x, robot_y, HOVER_Z)
                    time.sleep(0.3)

                    # 5. Vacuum OFF
                    self.delta.set_vacuum(False)
                    self.log(f"[PICK TEST] {label}  vacuum OFF, lifting")
                    time.sleep(0.3)

                # Return to standby
                self.delta.move_to(0, 0, -250)
                self.log("[PICK TEST] Complete. Returned to standby.")
                self.robot_pos['x'] = 0
                self.robot_pos['y'] = 0
                self.robot_pos['z'] = -250
                self._update_position_display()
            except Exception as e:
                self.delta.set_vacuum(False)   # Safety: vacuum off on error
                self.log(f"[PICK TEST] Error: {e}", "ERROR")

        threading.Thread(target=run_test, daemon=True).start()
    
    def _demo_smooth_two_points(self):
        """Move robot smoothly from Top-Right (TR) to Bottom-Left (BL) in small steps."""
        from modules.robot import bilinear_interpolate
        if not self.delta.connected:
            self.log("Robot not connected!", "ERROR")
            return
        # Start and end positions (belt coordinates in mm)
        start = (200, 0)   # TR
        end = (0, 200)     # BL
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            belt_x = start[0] * (1 - t) + end[0] * t
            belt_y = start[1] * (1 - t) + end[1] * t
            robot_x, robot_y = bilinear_interpolate(belt_x, belt_y)
            self.delta.move_to(robot_x, robot_y, -380)
            self.log(f"Smooth move: Belt({belt_x:.1f}, {belt_y:.1f}) -> Robot({robot_x:.1f}, {robot_y:.1f}, -380)")
            time.sleep(0.08)
        self.delta.move_to(0, 0, -250)
        self.log("Smooth two-point demo complete. Returned to standby.")
    
    def _demo_smooth_y_axis(self):
        """Smoothly move robot through Y axis points (TR->BR, TC->BC, TL->BL)."""
        from modules.robot import bilinear_interpolate
        if not self.delta.connected:
            self.log("Robot not connected!", "ERROR")
            return
        y_pairs = [((200, 0), (200, 200)), ((100, 0), (100, 200)), ((0, 0), (0, 200))]
        steps = 20
        for start, end in y_pairs:
            for i in range(steps + 1):
                t = i / steps
                belt_x = start[0] * (1 - t) + end[0] * t
                belt_y = start[1] * (1 - t) + end[1] * t
                robot_x, robot_y = bilinear_interpolate(belt_x, belt_y)
                self.delta.move_to(robot_x, robot_y, -380)
                self.log(f"Smooth Y: Belt({belt_x:.1f}, {belt_y:.1f}) -> Robot({robot_x:.1f}, {robot_y:.1f}, -380)")
                time.sleep(0.08)
        self.delta.move_to(0, 0, -250)
        self.log("Smooth Y axis demo complete. Returned to standby.")

    def _demo_smooth_x_axis(self):
        """Smoothly move robot through X axis points (TL->TR, CL->CR, BL->BR)."""
        from modules.robot import bilinear_interpolate
        if not self.delta.connected:
            self.log("Robot not connected!", "ERROR")
            return
        x_pairs = [((0, 0), (200, 0)), ((0, 100), (200, 100)), ((0, 200), (200, 200))]
        steps = 20
        for start, end in x_pairs:
            for i in range(steps + 1):
                t = i / steps
                belt_x = start[0] * (1 - t) + end[0] * t
                belt_y = start[1] * (1 - t) + end[1] * t
                robot_x, robot_y = bilinear_interpolate(belt_x, belt_y)
                self.delta.move_to(robot_x, robot_y, -380)
                self.log(f"Smooth X: Belt({belt_x:.1f}, {belt_y:.1f}) -> Robot({robot_x:.1f}, {robot_y:.1f}, -380)")
                time.sleep(0.08)
        self.delta.move_to(0, 0, -250)
        self.log("Smooth X axis demo complete. Returned to standby.")
    
    # ========== SPECTRUM TAB FUNCTIONS ==========
    
    def _manual_spectrum_scan(self):
        """Perform manual spectrum scan (non-blocking).

        The I2C sensor read (takeMeasurements) can block for 1-3 seconds.
        Running it on the DPG main thread would freeze the UI, so we
        dispatch the heavy I/O to a daemon thread and post results back
        via the ui_queue which _process_ui_queue() drains every frame.
        """
        if not self.spectrum.is_ready:
            self.log("Spectrum ML models not loaded!", "ERROR")
            dpg.set_value("txt_spectrum_pred", "Models Not Ready")
            return
        
        if not self.spectrum.hardware_ready:
            self.log("Spectrum sensor hardware not connected!", "ERROR")
            dpg.set_value("txt_spectrum_pred", "Hardware Not Connected")
            return
        
        # Prevent double-tap
        if getattr(self, '_spectrum_scanning', False):
            return
        self._spectrum_scanning = True
        
        dpg.set_value("txt_spectrum_pred", "Scanning...")
        dpg.set_value("txt_spectrum_conf", "---")
        self.log("Starting manual spectrum scan...")
        
        def _scan_worker():
            try:
                raw_data = self.spectrum.read_sensor()
                if raw_data:
                    # Predict
                    pred, conf, _, class_probs = self.spectrum.predict(raw_data)
                    # Post results to ui_queue — main thread will pick them up
                    self.ui_queue.put(('_manual_scan_result', {
                        'raw_data': raw_data,
                        'pred': pred,
                        'conf': conf,
                        'class_probs': class_probs,
                    }))
                else:
                    self.ui_queue.put(('_manual_scan_result', None))
            except Exception as e:
                self.log(f"Spectrum scan error: {e}", "ERROR")
                self.ui_queue.put(('_manual_scan_result', None))
            finally:
                self._spectrum_scanning = False
        
        threading.Thread(target=_scan_worker, daemon=True).start()

    def _handle_manual_scan_result(self, result):
        """Process manual spectrum scan result on the main DPG thread."""
        if result is None:
            self.log("Failed to read spectrum sensor!", "ERROR")
            dpg.set_value("txt_spectrum_pred", "Read Error")
            dpg.set_value("txt_spectrum_conf", "---")
            return
        
        raw_data = result['raw_data']
        pred = result['pred']
        conf = result['conf']
        class_probs = result.get('class_probs', {})
        
        # Display raw data
        dpg.set_value("txt_spectrum_raw", str(raw_data))
        
        # Format channel values
        channels = [
            "410nm (A)", "435nm (B)", "460nm (C)", "485nm (D)", "510nm (E)", "535nm (F)",
            "560nm (G)", "585nm (H)", "610nm (R)", "645nm (I)", "680nm (S)", "705nm (J)",
            "730nm (T)", "760nm (U)", "810nm (V)", "860nm (W)", "900nm (K)", "940nm (L)"
        ]
        channel_text = "\n".join([f"{ch}: {val:.2f}" for ch, val in zip(channels, raw_data)])
        dpg.set_value("txt_spectrum_channels", channel_text)
        
        # Display prediction
        dpg.set_value("txt_spectrum_pred", pred)
        dpg.set_value("txt_spectrum_conf", f"{conf:.1f}%")
        
        # Also update main spectrum display on Vision tab
        dpg.set_value("txt_spectrum", str(raw_data))
        
        # Display class probabilities if available
        if class_probs:
            prob_text = "  ".join([f"{cls}: {p*100:.1f}%" for cls, p in class_probs.items()])
            self.log(f"Scan: {pred} ({conf:.1f}%) - {prob_text}", "SUCCESS")
        else:
            self.log(f"Scan result: {pred} ({conf:.1f}%)", "SUCCESS")
    
    def _toggle_led(self, led_type, on):
        """Toggle spectrum sensor LED."""
        if not self.spectrum.sensor:
            self.log("Spectrum sensor not available!", "ERROR")
            return
        
        try:
            from as7265x_sparkfun_python import AS7265x_LED_WHITE, AS7265x_LED_IR, AS7265x_LED_UV
            
            led_map = {
                'ir': AS7265x_LED_IR,
                'white': AS7265x_LED_WHITE,
                'uv': AS7265x_LED_UV
            }
            
            led = led_map.get(led_type)
            if led is not None:
                if on:
                    self.spectrum.sensor.enableBulb(led)
                    self.log(f"{led_type.upper()} LED ON")
                else:
                    self.spectrum.sensor.disableBulb(led)
                    self.log(f"{led_type.upper()} LED OFF")
        except Exception as e:
            self.log(f"LED control error: {e}", "ERROR")

    def _on_start(self):
        """Start the system."""
        if self.is_running:
            return
        
        self.log("Starting system...")
        
        # Start camera
        if not self.camera.connected:
            if not self.camera.start_camera():
                self.log("Failed to start camera!", "ERROR")
                return
        
        # Set camera intrinsics to detector
        if self.camera.depth_intrinsics:
            self.detector.set_intrinsics(self.camera.depth_intrinsics)
            self.log("Camera intrinsics set")
        
        # Start robot manager
        self.robot_manager.start_manager()
        
        self.is_running = True
        self.is_tracking = True
        self.last_frame_time = time.time()
        
        dpg.configure_item("btn_start", enabled=False)
        dpg.configure_item("btn_stop", enabled=True)
        
        self.log("System started", "SUCCESS")
    
    def _on_stop(self):
        """Stop the system. If video was playing, stop it but keep it loaded for replay."""
        self.log("Stopping system...")
        
        # Stop any active recordings first
        if self.bag_recording:
            self._stop_bag_recording()
        if self.mp4_recording:
            self._stop_mp4_recording()
        # Auto-export audit log if collecting
        if self.audit_collecting:
            self._toggle_audit_log()  # stop & export
        
        # If video playback was active, use the video stop flow
        if self.video_source is not None:
            self._stop_video_playback()
            self.log("System stopped")
            return
        
        self.is_running = False
        self.is_tracking = False
        self.auto_pick = False
        
        # Disable track mode
        if self.track_mode:
            self.track_mode = False
            self.track_target_id = None
            dpg.configure_item("btn_track", label="TRACK")
        
        dpg.configure_item("btn_start", enabled=True)
        dpg.configure_item("btn_stop", enabled=False)
        dpg.configure_item("btn_autopick", label="AUTO PICK")
        
        self.log("System stopped")
    
    def _on_autopick(self):
        """Toggle auto-pick mode."""
        # Prevent auto-pick during track mode
        if not self.auto_pick and self.track_mode:
            self.log("Cannot enable AUTO PICK while TRACK mode is active", "ERROR")
            return
        self.auto_pick = not self.auto_pick
        label = "STOP PICK" if self.auto_pick else "AUTO PICK"
        dpg.configure_item("btn_autopick", label=label)
        self.log(f"Auto-pick: {'ON' if self.auto_pick else 'OFF'}")

    def _run_auto_boot_sequence(self):
        """One-shot launch sequence: Connect -> Home -> Start -> Auto Pick."""
        if self._auto_boot_done or self._auto_boot_started:
            return
        self._auto_boot_started = True
        self.log("[AUTO BOOT] Running startup sequence...")
        try:
            if not self.delta.connected or not self.slider.connected:
                self._on_connect()
            if self.delta.connected:
                self._on_home()
            if not self.is_running:
                self._on_start()
            if self.is_running and not self.auto_pick:
                self._on_autopick()
            self._auto_boot_done = True
            self.log("[AUTO BOOT] Sequence complete", "SUCCESS")
        except Exception as e:
            self.log(f"[AUTO BOOT] Sequence failed: {e}", "ERROR")
    
    def _setup_global_theme(self):
        """Setup modern global theme with rounded corners and better colors."""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # Rounded corners
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 5, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 5, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4, category=dpg.mvThemeCat_Core)
                # Padding and spacing
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12, category=dpg.mvThemeCat_Core)
                # Colors - darker, more modern palette
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 25, 28), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (32, 32, 36), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Border, (60, 60, 66), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (45, 45, 50), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (55, 55, 62), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (65, 65, 75), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (20, 20, 23), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (30, 30, 35), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 120, 180), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 140, 210), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (70, 160, 240), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Header, (45, 105, 160), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (55, 125, 190), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (65, 145, 220), category=dpg.mvThemeCat_Core)
        dpg.bind_theme(global_theme)
    
    def _create_start_btn_theme(self):
        """Create a green theme for the start button."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 140, 60))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (30, 180, 80))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 220, 100))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
        return theme
    
    def _create_stop_btn_theme(self):
        """Create a red theme for the stop button."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (160, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (200, 40, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (240, 50, 50))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
        return theme
    
    def _create_autopick_btn_theme(self):
        """Create a blue theme for the auto pick button."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (30, 100, 180))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (40, 120, 220))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 140, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
        return theme
    
    def _create_dummy_btn_theme(self):
        """Create a yellow theme for the dummy button."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (140, 120, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (180, 160, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (220, 200, 0))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
        return theme
    
    def _on_inject_dummy(self):
        """
        Inject a dummy object into the tracker for testing.
        
        Creates a fake object positioned just before the registration line
        (at ~13cm belt Y) with a random X position across the belt width.
        The dummy uses a safe pick height of 3cm and gets a random waste class.
        
        It enters the full pipeline naturally:
          1. Appears in tracker at belt_y ~13cm (just before reg line at 15cm)
          2. Next frame's advance_queued_objects moves it past registration
          3. check_registration_crossing queues it
          4. Continues advancing through gap and into workspace
          5. Robot picks it when it's deep enough (MIN_PICK_WORKSPACE_Y_CM)
        """
        if not hasattr(self, 'tracker') or self.tracker is None:
            self.log("[DUMMY] Tracker not initialized")
            return
        
        # Random belt X position (2-18cm, avoiding very edges)
        dummy_x = random.uniform(2.0, 18.0)
        
        # Place just past registration line so check_registration_crossing
        # queues it on the very next frame. The dummy won't get real camera
        # detections so its belt_y won't advance until it's queued (at which
        # point advance_queued_objects takes over).
        dummy_y = REGISTRATION_LINE_CM + 0.5  # 15.5cm (just past reg line at 15cm)
        
        # Random waste class
        class_ids = list(CLASS_NAMES.keys())
        dummy_class_id = random.choice(class_ids)
        dummy_class_name = CLASS_NAMES[dummy_class_id]
        
        # Safe height: 3cm (won't crash into belt)
        dummy_height = 3.0
        
        # Inject directly into tracker's object dict
        obj_id = self.tracker.next_id
        self.tracker.objects[obj_id] = {
            'id': obj_id,
            'centroid': (320, 240),          # Dummy pixel pos (center-ish)
            'smoothed_centroid': (320, 240),
            'raw_centroid': (320, 240),
            'class_id': dummy_class_id,
            'class_name': dummy_class_name,
            'mask': None,
            'min_area_box': None,
            'smoothed_box': None,
            'height_cm': dummy_height,       # 3cm safe height
            'width_cm': 5.0,
            'obj_height_cm': dummy_height,
            'angle': random.uniform(0, 180),
            'confidence': 0.99,              # High confidence ("perfect" detection)
            'belt_y_cm': dummy_y,            # Just before registration line
            'belt_x_cm': dummy_x,
            'last_known_x_cm': dummy_x,
            'last_update_time': time.time(),
            'in_queue': False,
            'queue_position': -1,
            'status': 'Tracking',
            'disappeared': 0,
            'detected_this_frame': True,
        }
        self.tracker.next_id += 1
        
        self.log(f"[DUMMY] Injected ID:{obj_id} - {dummy_class_name} at X={dummy_x:.1f}cm, Y={dummy_y:.1f}cm, H=3cm")

    # ── Video & Recording tab ─────────────────────────────────────────────

    def _guess_replay_sidecar_paths(self, video_path):
        """Candidate sidecar paths for MP4 replay metadata."""
        base, _ = os.path.splitext(video_path)
        return [
            f"{base}_replay.csv",
            f"{base}_replay.xlsx",
            f"{base}_sync.csv",
            f"{base}_sync.xlsx",
        ]

    def _load_replay_sidecar(self, video_path):
        """Load MP4 replay sidecar with H/X/Y and 18ch spectrum."""
        self.replay_pick_rows = []
        self.replay_pick_idx = 0
        self.replay_enabled = False

        def _f(row, keys):
            for k in keys:
                if k in row and row[k] not in (None, ""):
                    try:
                        return float(row[k])
                    except Exception:
                        return None
            return None

        def _s(row, keys):
            for k in keys:
                v = row.get(k)
                if v is not None and v != "":
                    return str(v)
            return None

        sidecar_path = None
        rows = []
        for p in self._guess_replay_sidecar_paths(video_path):
            if os.path.exists(p):
                sidecar_path = p
                break
        if sidecar_path is None:
            return False

        try:
            if sidecar_path.lower().endswith(".csv"):
                with open(sidecar_path, "r", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            else:
                import pandas as pd
                rows = pd.read_excel(sidecar_path).to_dict(orient="records")
        except Exception as e:
            self.log(f"[VIDEO] Failed to load replay sidecar: {e}", "ERROR")
            return False

        parsed = []
        for row in rows:
            spec_raw = []
            for i in range(1, 19):
                keys = [
                    f"spectrum_raw_{i:02d}",
                    f"spectrum_{i:02d}",
                    f"spec_{i:02d}",
                    f"ch{i:02d}",
                    f"ch{i}",
                ]
                val = _f(row, keys)
                spec_raw.append(val if val is not None else 0.0)

            parsed.append({
                'video_time_s': _f(row, ['video_time_s', 'time_s', 'video_second']),
                'belt_x_cm': _f(row, ['belt_x_cm', 'belt_x', 'x_cm', 'x']),
                'ws_y_cm': _f(row, ['ws_y_cm', 'belt_y_cm', 'belt_y', 'y_cm', 'y']),
                'height_cm': _f(row, ['height_cm', 'height', 'h_cm', 'h']),
                'spectrum_class': _s(row, ['spectrum_class', 'spec_class']),
                'spectrum_conf': _f(row, ['spectrum_conf', 'spec_conf']),
                'spectrum_raw': spec_raw,
            })

        if not parsed:
            return False

        self.replay_pick_rows = parsed
        self.replay_pick_idx = 0
        self.replay_enabled = True
        self.log(f"[VIDEO] Replay sidecar loaded: {os.path.basename(sidecar_path)} "
                 f"({len(parsed)} rows)", "SUCCESS")
        return True

    def _next_replay_pick_row(self):
        """Get next replay metadata row aligned by pick order."""
        if not self.replay_enabled or not self.replay_pick_rows:
            return None
        if self.replay_pick_idx >= len(self.replay_pick_rows):
            return self.replay_pick_rows[-1]
        row = self.replay_pick_rows[self.replay_pick_idx]
        self.replay_pick_idx += 1
        return row

    def _export_mp4_sync_sidecar(self):
        """Export per-pick sync sidecar (.csv + best-effort .xlsx)."""
        if not self.mp4_filename or not self._mp4_sync_rows:
            return

        base, _ = os.path.splitext(self.mp4_filename)
        csv_path = f"{base}_replay.csv"
        xlsx_path = f"{base}_replay.xlsx"

        fields = list(self._mp4_sync_rows[0].keys())
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self._mp4_sync_rows)
            self.log(f"[REC] Replay sidecar CSV saved: {csv_path}", "SUCCESS")
        except Exception as e:
            self.log(f"[REC] Failed to save replay CSV: {e}", "ERROR")
            return

        try:
            import pandas as pd
            pd.DataFrame(self._mp4_sync_rows).to_excel(xlsx_path, index=False)
            self.log(f"[REC] Replay sidecar Excel saved: {xlsx_path}", "SUCCESS")
        except Exception as e:
            self.log(f"[REC] Excel export skipped: {e}")

    def _open_video_dialog(self):
        """Open a file dialog to pick a .bag or .mp4 recording."""
        rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        if not os.path.isdir(rec_dir):
            rec_dir = os.getcwd()

        if dpg.does_item_exist("video_file_dialog"):
            dpg.delete_item("video_file_dialog")

        with dpg.file_dialog(
            label="Select .bag / .mp4 Recording",
            callback=self._on_video_file_selected,
            cancel_callback=lambda: None,
            width=700, height=400,
            default_path=rec_dir,
            tag="video_file_dialog",
            modal=True,
        ):
            dpg.add_file_extension(".bag", color=(0, 255, 127, 255))
            dpg.add_file_extension(".mp4", color=(255, 190, 110, 255))
            dpg.add_file_extension(".*")

    def _on_video_file_selected(self, sender, app_data):
        """Callback when user picks a .bag/.mp4 file from the dialog."""
        selections = app_data.get("selections", {})
        if not selections:
            return
        video_path = list(selections.values())[0]
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in (".bag", ".mp4"):
            self.log(f"[VIDEO] Unsupported file type: {video_path}", "ERROR")
            return

        # Store path and update Video tab UI
        self._loaded_bag_path = video_path
        self._loaded_video_kind = "mp4" if ext == ".mp4" else "bag"
        self.replay_enabled = False
        self.replay_pick_rows = []
        self.replay_pick_idx = 0
        self._replay_last_loop_count = 0
        self.log(f"[VIDEO] Loaded {self._loaded_video_kind.upper()}: "
                 f"{os.path.basename(video_path)} - press > PLAY to start")
        dpg.set_value("txt_vt_bag_path", os.path.basename(video_path))
        dpg.configure_item("txt_vt_bag_path", color=(100, 220, 160))
        dpg.configure_item("btn_vt_play", enabled=True)
        dpg.configure_item("btn_vt_unload", enabled=True)
        dpg.set_value("txt_vt_status", f"Ready - {os.path.basename(video_path)}")
        dpg.configure_item("txt_vt_status", color=(100, 220, 160))

    def _start_video_playback(self, bag_path):
        """Start (or restart) video playback with tracking."""
        video_path = bag_path
        ext = os.path.splitext(video_path)[1].lower()
        is_mp4 = ext == ".mp4"

        # Stop current system if running
        if self.is_running:
            self._on_stop_system_only()

        # Stash live camera once. For MP4 playback we keep it running to supply live depth.
        if self._live_camera is None:
            if isinstance(self.camera, CameraStream):
                self._live_camera = self.camera
            else:
                self._live_camera = CameraStream()

        if is_mp4:
            if not self._live_camera.connected:
                if not self._live_camera.start_camera():
                    self.log("[VIDEO] Live depth camera not available - fallback to synthetic depth", "ERROR")
        else:
            # Non-MP4 playback does not need live depth stream.
            if self._live_camera.connected:
                self._live_camera.stop()

        # Stop any previous video stream
        if self.video_source is not None:
            self.video_source.stop()

        # Create new video stream
        mp4_pass_limit = self._get_mp4_playback_pass_limit() if is_mp4 else None
        if is_mp4:
            if self._live_camera is not None and self._live_camera.connected:
                vs = MP4LiveDepthPlaybackStream(video_path, self._live_camera)
                self.log("[VIDEO] MP4 replay using LIVE depth camera")
            else:
                vs = MP4PlaybackStream(video_path)
                # Fallback path: synthetic depth baseline for MP4 replay
                if getattr(self.detector, "floor_depth_map", None) is not None:
                    try:
                        vs.set_depth_template_from_meters(self.detector.floor_depth_map)
                    except Exception:
                        pass
        else:
            vs = VideoPlaybackStream(video_path)
        # MP4 playback can run a fixed number of passes when requested.
        if is_mp4 and mp4_pass_limit is not None:
            vs.max_passes = mp4_pass_limit
            vs.loop = mp4_pass_limit > 1
        elif is_mp4 and self._is_benchmark_enabled():
            vs.max_passes = 1
            vs.loop = False
        else:
            if is_mp4 and hasattr(vs, "max_passes"):
                vs.max_passes = None
            vs.loop = True
        if not vs.start_camera():
            self.log(f"[VIDEO] Failed to open {ext} file!", "ERROR")
            return

        self.video_source = vs
        self.camera = vs
        self._loaded_video_kind = "mp4" if is_mp4 else "bag"
        self._replay_last_loop_count = 0

        # Set intrinsics from the recording
        if vs.depth_intrinsics:
            self.detector.set_intrinsics(vs.depth_intrinsics)

        if is_mp4:
            self._load_replay_sidecar(video_path)
            if not self.replay_enabled:
                self.log("[VIDEO] No replay sidecar found for MP4; using live-computed H/X/Y/spectrum flow")
        else:
            self.replay_enabled = False
            self.replay_pick_rows = []
            self.replay_pick_idx = 0

        # Start tracking
        self.is_running = True
        self.is_tracking = True
        self.last_frame_time = time.time()

        # Reset tracker for clean start
        self.tracker = SimpleTracker()
        self.last_detections = []

        dpg.configure_item("btn_start", enabled=False)
        dpg.configure_item("btn_stop", enabled=True)

        # Update Video tab buttons
        dpg.configure_item("btn_vt_play", enabled=False)
        dpg.configure_item("btn_vt_stop", enabled=True)
        dpg.configure_item("btn_vt_load", enabled=False)
        dpg.configure_item("btn_vt_unload", enabled=False)
        dpg.set_value("txt_vt_status", f"> Playing - {os.path.basename(video_path)}")
        dpg.configure_item("txt_vt_status", color=(0, 255, 100))

        # Disable live recording while video is playing (can't record playback)
        dpg.configure_item("btn_rec_bag", enabled=False)
        dpg.configure_item("btn_rec_mp4", enabled=False)

        # Optional: auto-enable auto-pick on playback start
        auto_start = dpg.get_value("chk_vt_auto_start") if dpg.does_item_exist("chk_vt_auto_start") else True
        if auto_start and not self.auto_pick:
            self._on_autopick()

        self._start_benchmark_capture(
            mode="playback",
            source_kind=("mp4" if is_mp4 else "bag"),
            source_file=video_path,
            output_dir=self._create_record_subdir("playback")[0]
        )

        if is_mp4 and mp4_pass_limit is not None:
            loop_msg = f"{mp4_pass_limit} passes"
        else:
            loop_msg = "single-pass" if not vs.loop else "loop"
        self.log(f"[VIDEO] > Playing ({loop_msg}) - {os.path.basename(video_path)}", "SUCCESS")

    def _stop_video_playback(self):
        """Stop video playback and restore live camera. Video stays loaded for replay."""
        self._stop_benchmark_capture("playback stopped")
        if self.video_source is not None:
            if self.is_running:
                self._on_stop_system_only()
            self.video_source.stop()
            self.video_source = None

        # Restore live camera
        if self._live_camera is not None:
            self.camera = self._live_camera
            self._live_camera = None
        else:
            self.camera = CameraStream()

        dpg.configure_item("btn_start", enabled=True)
        dpg.configure_item("btn_stop", enabled=False)

        # Update Video tab — back to "ready to play" state
        dpg.configure_item("btn_vt_play", enabled=True)
        dpg.configure_item("btn_vt_stop", enabled=False)
        dpg.configure_item("btn_vt_load", enabled=True)
        dpg.configure_item("btn_vt_unload", enabled=True)
        dpg.configure_item("btn_rec_bag", enabled=True)
        dpg.configure_item("btn_rec_mp4", enabled=True)

        if self._loaded_bag_path:
            dpg.set_value("txt_vt_status",
                          f"Stopped - {os.path.basename(self._loaded_bag_path)} (press > to replay)")
            dpg.configure_item("txt_vt_status", color=(255, 200, 80))
        else:
            dpg.set_value("txt_vt_status", "Stopped")
            dpg.configure_item("txt_vt_status", color=(120, 120, 120))

        self.log("[VIDEO] Stopped - press PLAY to restart, or load a new file")

    def _on_stop_system_only(self):
        """Stop tracking/running without touching video state or button labels."""
        self.is_running = False
        self.is_tracking = False
        self.auto_pick = False
        if self.track_mode:
            self.track_mode = False
            self.track_target_id = None
            dpg.configure_item("btn_track", label="TRACK")
        dpg.configure_item("btn_autopick", label="AUTO PICK")

    def _create_video_btn_theme(self):
        """Create a purple/magenta theme for the LOAD VIDEO button."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (100, 40, 140))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (130, 60, 180))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (160, 80, 200))
        return theme

    def _create_play_btn_theme(self):
        """Create a green theme for the ▶ PLAY button."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 120, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (30, 160, 60))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 200, 80))
        return theme

    def _create_rec_bag_theme(self):
        """Idle .bag record button (dark orange)."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (140, 80, 10))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (170, 100, 20))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (200, 120, 30))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
        return theme

    def _create_rec_bag_active_theme(self):
        """Active recording button (bright red)."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (230, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 70, 70))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
        return theme

    def _create_rec_mp4_theme(self):
        """Idle .mp4 record button (dark peach)."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (140, 90, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (170, 110, 55))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (200, 130, 70))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
        return theme

    def _create_audit_btn_theme(self):
        """Idle audit log button (teal)."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 100, 120))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (30, 130, 150))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 160, 180))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
        return theme

    def _create_robot_cam_on_theme(self):
        """Robot cam preview ON (green)."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 120, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (30, 150, 65))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 180, 80))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
        return theme

    def _create_robot_cam_off_theme(self):
        """Robot cam preview OFF (dim grey)."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (70, 70, 70))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (130, 130, 130))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
        return theme

    def _create_robot_cam_reconnect_theme(self):
        """Reconnect button (orange)."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (140, 100, 20))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (170, 120, 30))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (200, 140, 40))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
        return theme

    def _toggle_robot_cam_preview(self):
        """Toggle robot camera live preview on/off."""
        self.robot_cam_preview = not self.robot_cam_preview
        if self.robot_cam_preview:
            dpg.configure_item("btn_robot_cam_toggle", label="Disable Preview")
            dpg.bind_item_theme("btn_robot_cam_toggle", self._robot_cam_btn_on)
            self.log("[ROBOT CAM] Preview enabled")
        else:
            dpg.configure_item("btn_robot_cam_toggle", label="Enable Preview")
            dpg.bind_item_theme("btn_robot_cam_toggle", self._robot_cam_btn_off)
            # Clear the texture to black
            black = np.zeros((ROBOT_CAM_HEIGHT, ROBOT_CAM_WIDTH, 4), dtype=np.float32)
            dpg.set_value("robot_cam_texture", black.flatten())
            self.log("[ROBOT CAM] Preview disabled")

    def _reconnect_robot_cam(self):
        """Release and re-open the USB robot camera."""
        self.log("[ROBOT CAM] Reconnecting...")
        # Release existing capture
        if self.robot_cam is not None:
            try:
                self.robot_cam.release()
            except Exception:
                pass
            self.robot_cam = None
            self.robot_cam_connected = False
        # Re-init
        self._init_robot_camera()
        # Auto-enable preview on reconnect
        if self.robot_cam_connected and not self.robot_cam_preview:
            self.robot_cam_preview = True
            dpg.configure_item("btn_robot_cam_toggle", label="Disable Preview")
            dpg.bind_item_theme("btn_robot_cam_toggle", self._robot_cam_btn_on)

    # ── Audit log collector ──────────────────────────────────────────────

    def _toggle_audit_log(self):
        """Toggle audit log collection on/off. On stop → export CSV."""
        if not self.audit_collecting:
            # START collecting
            self.audit_collecting = True
            self.audit_buffer = []
            self.audit_start_time = datetime.now()
            dpg.configure_item("btn_audit_log", label="  Stop & Export")
            dpg.bind_item_theme("btn_audit_log", self._audit_theme_on)
            dpg.set_value("txt_audit_status", "Collecting... 0 rows")
            dpg.configure_item("txt_audit_status", color=(0, 255, 100))
            self.log("[AUDIT] Audit log collection started")
        else:
            # STOP collecting → export
            self.audit_collecting = False
            dpg.configure_item("btn_audit_log", label="  Start Audit Log")
            dpg.bind_item_theme("btn_audit_log", self._audit_theme_off)
            n = len(self.audit_buffer)
            if n == 0:
                dpg.set_value("txt_audit_status", "No data collected — nothing exported")
                dpg.configure_item("txt_audit_status", color=(255, 180, 60))
                self.log("[AUDIT] Stopped — no rows collected")
                return
            path = self._export_audit_csv()
            dpg.set_value("txt_audit_status",
                          f"Exported {n} rows → {os.path.basename(path)}")
            dpg.configure_item("txt_audit_status", color=(100, 200, 255))
            self.log(f"[AUDIT] Exported {n} rows to {path}", "SUCCESS")

    def _export_audit_csv(self):
        """Write the audit buffer to a CSV file sorted by Object_ID."""
        from modules.robot import AUDIT_FUSION_METHODS

        audit_dir = os.path.join("detection_logs", "audits")
        os.makedirs(audit_dir, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(audit_dir, f"audit_{ts}.csv")

        # Column order: ID, timestamp, cam/spec, Bayesian (final), then all other methods, then agreement score
        method_cols = [name for name, _ in AUDIT_FUSION_METHODS]
        fieldnames = [
            'Object_ID', 'Timestamp',
            'Cam_Class', 'Cam_Conf',
            'Spec_Class', 'Spec_Conf',
            'Final_Class_Bayesian',
        ] + method_cols + ['Agreement_Score']

        # Sort rows by Object_ID
        sorted_rows = sorted(self.audit_buffer, key=lambda r: r.get('obj_id', 0))

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted_rows:
                bayesian = row.get('final_class', '')
                # Count how many methods agree with Bayesian
                agree = sum(1 for name in method_cols if row.get(name) == bayesian)
                # +1 for Bayesian itself
                total_methods = len(method_cols) + 1
                out = {
                    'Object_ID': row.get('obj_id', ''),
                    'Timestamp': row.get('timestamp', ''),
                    'Cam_Class': row.get('cam_class', ''),
                    'Cam_Conf': row.get('cam_conf', ''),
                    'Spec_Class': row.get('spec_class', ''),
                    'Spec_Conf': row.get('spec_conf', ''),
                    'Final_Class_Bayesian': bayesian,
                    'Agreement_Score': f"{agree + 1}/{total_methods}",
                }
                for name in method_cols:
                    out[name] = row.get(name, '')
                writer.writerow(out)

        self.log(f"[AUDIT] CSV saved: {path}")
        return path

    # ── Video tab actions ────────────────────────────────────────────────

    def _vt_play(self):
        """Play button on the Video tab — start video playback with auto-sync."""
        if self._loaded_bag_path is None:
            self.log("[VIDEO] No file loaded", "ERROR")
            return
        self._start_video_playback(self._loaded_bag_path)

    def _vt_stop(self):
        """Stop button on the Video tab — stop video playback."""
        if self.video_source is not None:
            self._stop_video_playback()

    def _vt_unload(self):
        """Unload the video file entirely."""
        if self.video_source is not None:
            self._stop_video_playback()
        self._loaded_bag_path = None
        self._loaded_video_kind = None
        self.replay_enabled = False
        self.replay_pick_rows = []
        self.replay_pick_idx = 0
        dpg.set_value("txt_vt_bag_path", "(none)")
        dpg.configure_item("txt_vt_bag_path", color=(160, 120, 255))
        dpg.configure_item("btn_vt_play", enabled=False)
        dpg.configure_item("btn_vt_stop", enabled=False)
        dpg.configure_item("btn_vt_unload", enabled=False)
        dpg.set_value("txt_vt_status", "Idle - no video loaded")
        dpg.configure_item("txt_vt_status", color=(120, 120, 120))
        self.log("[VIDEO] File unloaded")

    # ── Recording ────────────────────────────────────────────────────────

    def _toggle_bag_recording(self):
        """Toggle .bag recording on the live camera."""
        if self.video_source is not None:
            self.log("[REC] Cannot record .bag during video playback", "ERROR")
            return

        if not self.bag_recording:
            self._start_bag_recording()
        else:
            self._stop_bag_recording()

    def _start_bag_recording(self):
        """Start recording live RealSense streams to a .bag file."""
        if not self.camera.connected:
            self.log("[REC] Camera not running - start the system first", "ERROR")
            return
        if self.bag_recording:
            return

        rec_dir, timestamp = self._create_record_subdir("bag")
        self.bag_filename = os.path.join(rec_dir, f"rec_{timestamp}.bag")

        self.log("[REC] Restarting pipeline with .bag recording...")
        if self.camera.start_recording(self.bag_filename):
            self.bag_recording = True
            dpg.configure_item("btn_rec_bag", label="[=] STOP .bag")
            dpg.bind_item_theme("btn_rec_bag", self._rec_bag_theme_on)
            dpg.set_value("txt_rec_bag_status",
                          f"Recording: {os.path.basename(self.bag_filename)}")
            dpg.configure_item("txt_rec_bag_status", color=(255, 60, 60))
            self.log(f"[REC] .bag STARTED: {self.bag_filename}", "SUCCESS")
        else:
            self.log("[REC] Failed to start .bag recording!", "ERROR")

    def _stop_bag_recording(self):
        """Stop .bag recording by restarting pipeline without recording."""
        if not self.bag_recording:
            return

        self.log("[REC] Stopping .bag recording (restarting pipeline)...")
        if self.camera.stop_recording():
            self.bag_recording = False
            size_mb = 0
            if self.bag_filename and os.path.exists(self.bag_filename):
                size_mb = os.path.getsize(self.bag_filename) / (1024 * 1024)
            dpg.configure_item("btn_rec_bag", label="(o) REC .bag")
            dpg.bind_item_theme("btn_rec_bag", self._rec_bag_theme_off)
            dpg.set_value("txt_rec_bag_status",
                          f"Saved: {os.path.basename(self.bag_filename)} ({size_mb:.1f} MB)")
            dpg.configure_item("txt_rec_bag_status", color=(100, 220, 160))
            self.log(f"[REC] .bag STOPPED: {self.bag_filename} ({size_mb:.1f} MB)", "SUCCESS")
        else:
            self.log("[REC] Failed to stop .bag recording!", "ERROR")

    def _toggle_mp4_recording(self):
        """Toggle .mp4 raw camera recording."""
        if self.video_source is not None and not self.mp4_recording:
            self.log("[REC] Cannot record .mp4 during video playback", "ERROR")
            return
        if not self.mp4_recording:
            self._start_mp4_recording()
        else:
            self._stop_mp4_recording()

    def _start_mp4_recording(self):
        """Start experiment recording — raw vision feed + robot cam + replay sidecar data."""
        if not self.is_running:
            self.log("[REC] System not running - start first", "ERROR")
            return
        if self.mp4_recording:
            return

        rec_dir, timestamp = self._create_record_subdir("exp")

        # ── Screen (vision) writer ──
        self.mp4_filename = os.path.join(rec_dir, f"exp_{timestamp}_screen.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._mp4_record_fps = float(getattr(self.video_source, "source_fps", FPS) or FPS)
        if self._mp4_record_fps <= 0:
            self._mp4_record_fps = float(FPS)
        self.mp4_writer = cv2.VideoWriter(
            self.mp4_filename, fourcc, self._mp4_record_fps, (IMAGE_WIDTH, IMAGE_HEIGHT))
        self.mp4_recording = True
        self.record_frame_count = 0
        self._mp4_record_start_epoch = None
        self._mp4_record_start_utc = None
        self._mp4_sync_rows = []
        self._mp4_source_t0 = None

        # ── Robot camera writer (linked by same timestamp) ──
        robot_note = ""
        if self.robot_cam_connected and self.robot_cam is not None:
            self.robot_cam_filename = os.path.join(rec_dir, f"exp_{timestamp}_robot.mp4")
            self.robot_cam_writer = cv2.VideoWriter(
                self.robot_cam_filename, fourcc, ROBOT_CAM_FPS,
                (ROBOT_CAM_WIDTH, ROBOT_CAM_HEIGHT))
            self.robot_cam_frame_count = 0
            robot_note = f"  + Robot cam: {os.path.basename(self.robot_cam_filename)}"
            self.log(f"[REC] Robot cam recording STARTED: {self.robot_cam_filename}", "SUCCESS")
        else:
            self.robot_cam_writer = None
            self.robot_cam_filename = None
            robot_note = "  (Robot camera not connected - screen only)"

        dpg.configure_item("btn_rec_mp4", label="[=] STOP Experiment")
        dpg.bind_item_theme("btn_rec_mp4", self._rec_mp4_theme_on)
        dpg.set_value("txt_rec_mp4_status",
                      f"Recording: {os.path.basename(self.mp4_filename)}")
        dpg.configure_item("txt_rec_mp4_status", color=(255, 60, 60))
        dpg.set_value("txt_rec_robot_cam_note", robot_note)
        dpg.configure_item("txt_rec_robot_cam_note", color=(255, 180, 100))
        self.log(f"[REC] Experiment recording STARTED: {self.mp4_filename} "
                 f"@ {self._mp4_record_fps:.2f}fps", "SUCCESS")
        self._start_benchmark_capture(
            mode="rec_mp4",
            source_kind="live",
            source_file=self.mp4_filename,
            output_dir=rec_dir
        )

    def _stop_mp4_recording(self):
        """Stop experiment recording — releases both screen and robot cam writers."""
        if not self.mp4_recording:
            return
        self._stop_benchmark_capture("recording stopped")

        # ── Release screen writer ──
        if self.mp4_writer:
            self.mp4_writer.release()
            self.mp4_writer = None
        self.mp4_recording = False

        size_mb = 0
        if self.mp4_filename and os.path.exists(self.mp4_filename):
            size_mb = os.path.getsize(self.mp4_filename) / (1024 * 1024)
        duration_s = self.record_frame_count / max(1.0, float(self._mp4_record_fps))

        # ── Release robot cam writer ──
        robot_summary = ""
        if self.robot_cam_writer is not None:
            self.robot_cam_writer.release()
            self.robot_cam_writer = None
            r_size_mb = 0
            if self.robot_cam_filename and os.path.exists(self.robot_cam_filename):
                r_size_mb = os.path.getsize(self.robot_cam_filename) / (1024 * 1024)
            r_dur = self.robot_cam_frame_count / max(1, ROBOT_CAM_FPS)
            robot_summary = (f"  + Robot: {os.path.basename(self.robot_cam_filename)} "
                             f"({r_dur:.1f}s, {r_size_mb:.1f} MB)")
            self.log(f"[REC] Robot cam STOPPED: {self.robot_cam_filename} "
                     f"({self.robot_cam_frame_count}f, {r_dur:.1f}s, {r_size_mb:.1f} MB)", "SUCCESS")

        self._export_mp4_sync_sidecar()
        self._mp4_record_start_epoch = None
        self._mp4_record_start_utc = None
        self._mp4_sync_rows = []
        self._mp4_source_t0 = None

        dpg.configure_item("btn_rec_mp4", label="(o) REC Experiment")
        dpg.bind_item_theme("btn_rec_mp4", self._rec_mp4_theme_off)
        dpg.set_value("txt_rec_mp4_status",
                      f"Saved: {os.path.basename(self.mp4_filename)} "
                      f"({duration_s:.1f}s, {size_mb:.1f} MB)")
        dpg.configure_item("txt_rec_mp4_status", color=(100, 220, 160))
        dpg.set_value("txt_rec_robot_cam_note", robot_summary)
        dpg.configure_item("txt_rec_robot_cam_note", color=(100, 220, 160))
        self.log(f"[REC] Experiment recording STOPPED: {self.mp4_filename} "
                 f"({self.record_frame_count}f, {duration_s:.1f}s, {size_mb:.1f} MB)", "SUCCESS")

    def _process_robot_camera_frame(self):
        """Read one frame from the USB robot camera, flip it, display + record."""
        if not self.robot_cam_connected or self.robot_cam is None:
            return
        # Skip reading entirely when preview is off AND not recording
        if not self.robot_cam_preview and not self.mp4_recording:
            return
        try:
            ret, frame = self.robot_cam.read()
            if not ret or frame is None:
                return

            # Camera is mounted upside-down — flip vertically (around x-axis)
            frame = cv2.flip(frame, 0)

            # Recording overlay (tied to experiment recording)
            if self.mp4_recording and self.robot_cam_writer is not None:
                elapsed = self.robot_cam_frame_count / max(1, ROBOT_CAM_FPS)
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(frame, (15, 15), 7, (0, 0, 255), -1)
                cv2.putText(frame, f"REC {elapsed:.1f}s", (30, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            # Update DPG texture (only when preview is enabled)
            if self.robot_cam_preview:
                robot_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                robot_data = robot_rgba.astype(np.float32) / 255.0
                dpg.set_value("robot_cam_texture", robot_data.flatten())

            # Write to .mp4 if experiment recording is active
            if self.mp4_recording and self.robot_cam_writer is not None:
                self.robot_cam_writer.write(frame)
                self.robot_cam_frame_count += 1

        except Exception:
            pass  # Don't crash main loop if USB camera hiccups

    def _on_robot_approach_pos_update(self, px, py, pz):
        """
        Callback from RobotManager during tracking approach phase.
        Updates simulation robot position so the red dot follows the robot
        even during the approach (before the actual pick).
        Thread-safe: called from robot thread, only updates simple values.
        """
        self.robot_pos['x'] = px
        self.robot_pos['y'] = py
        self.robot_pos['z'] = pz

    def _create_track_btn_theme(self):
        """Create a teal/green theme for the track button."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 120, 80))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 160, 110))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 200, 140))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
        return theme
    
    def _on_toggle_track(self):
        """Toggle TRACK mode: robot follows objects through workspace."""
        self.track_mode = not self.track_mode
        self.track_target_id = None  # Reset target on toggle
        
        if self.track_mode:
            dpg.configure_item("btn_track", label="TRACKING")
            # Disable auto-pick while tracking (they conflict)
            if self.auto_pick:
                self.auto_pick = False
                dpg.configure_item("btn_autopick", label="AUTO PICK")
                self.log("Auto-pick disabled (TRACK mode active)")
            self.log("TRACK mode ON - robot will follow objects through workspace", "SUCCESS")
        else:
            dpg.configure_item("btn_track", label="TRACK")
            # Return robot to standby when tracking stops
            if self.delta.connected:
                self.delta.go_standby()
            self.log("TRACK mode OFF - robot returning to standby")
    
    def _process_tracking(self):
        """
        TRACK mode: robot hovers above and follows an object through the workspace.
        
        Called every frame when track_mode is True.
        - Finds the target object (auto-selects first queued object entering workspace)
        - Converts its real-time belt position to robot coordinates
        - Sends move_to command so robot follows smoothly
        
        Since the robot is very fast, 1 move command per frame is sufficient.
        The robot will physically follow the object from WS entry to WS exit.
        """
        if not self.delta.connected:
            return
        
        belt_speed = self._get_belt_speed()
        current_time = time.time()
        ws_entry_y = ROI_HEIGHT_CM + ROBOT_WORKSPACE_OFFSET_CM       # 42cm
        ws_exit_y = ws_entry_y + ROBOT_WORKSPACE_DEPTH_CM            # 62cm
        
        # --- Find or validate tracking target ---
        target_obj = None
        
        # Check if current target is still valid
        if self.track_target_id is not None:
            obj = self.tracker.objects.get(self.track_target_id)
            if obj and obj.get('in_queue', False):
                belt_y = obj.get('belt_y_cm', 0)
                # Object still in range (allow slightly past exit for smooth finish)
                if belt_y <= ws_exit_y + 2:
                    target_obj = obj
                else:
                    # Object exited workspace — clear target
                    self.log(f"[TRACK] ID:{self.track_target_id} exited workspace")
                    self.track_target_id = None
            else:
                # Object gone or no longer queued
                self.track_target_id = None
        
        # Auto-select: pick the queued object with lowest belt_y (earliest to enter WS)
        if target_obj is None:
            best_id = None
            best_y = float('inf')
            for obj_id, obj in self.tracker.objects.items():
                if not obj.get('in_queue', False):
                    continue
                belt_y = obj.get('belt_y_cm', 0)
                # Must be approaching or inside workspace
                if belt_y >= REGISTRATION_LINE_CM and belt_y <= ws_exit_y + 2:
                    if belt_y < best_y:
                        best_y = belt_y
                        best_id = obj_id
            
            if best_id is not None:
                self.track_target_id = best_id
                target_obj = self.tracker.objects[best_id]
                self.log(f"[TRACK] Now tracking ID:{best_id} at belt_y={best_y:.1f}cm")
        
        if target_obj is None:
            return  # Nothing to track
        
        # --- Get real-time position (time-anchor based with measured speed) ---
        belt_x = target_obj.get('last_known_x_cm', target_obj.get('belt_x_cm', 10.0))
        
        # Use time-anchor prediction with measured speed (drift-free)
        real_belt_y = self.tracker._get_anchor_belt_y(target_obj, belt_speed_cm_s=belt_speed)
        
        # Only move robot when object is in workspace range
        if real_belt_y < ws_entry_y:
            return  # Not yet in workspace — don't move robot
        
        # Convert to workspace-relative Y (0-20cm)
        ws_y = real_belt_y - ws_entry_y
        ws_y = max(0, min(ROBOT_WORKSPACE_DEPTH_CM, ws_y))
        
        # Convert belt cm to mm for bilinear interpolation
        x_mm = belt_x * 10.0       # 0-200mm
        y_mm = ws_y * 10.0          # 0-200mm
        
        # Get robot coordinates
        robot_x, robot_y = bilinear_interpolate(x_mm, y_mm)
        
        # Apply offsets
        off_x = dpg.get_value("off_x") if dpg.does_item_exist("off_x") else 0
        off_y = dpg.get_value("off_y") if dpg.does_item_exist("off_y") else 0
        off_z = dpg.get_value("off_z") if dpg.does_item_exist("off_z") else 0
        px = float(np.clip(robot_x + off_x, -110, 110))
        py = float(np.clip(robot_y + off_y, -110, 110))
        
        # Calculate Z from object's REAL height (same formula as execute_pick)
        # BASE_Z is the belt surface (-415mm), object height raises it, then hover margin above
        obj_height_cm = target_obj.get('height_cm', target_obj.get('obj_height_cm', 3.0))
        if obj_height_cm < 0 or obj_height_cm > 30.0:
            obj_height_cm = 0
        # Pick Z + hover margin = hovers above without touching
        pz = BASE_Z + (obj_height_cm * 10) + off_z + self.track_hover_margin_mm
        pz = float(np.clip(pz, -450, -150))
        
        # Send move command (robot is fast — 1 command per frame is smooth)
        self.delta.move_to(px, py, pz, f=15000)
        
        # Update internal robot_pos for display consistency
        self.robot_pos['x'] = px
        self.robot_pos['y'] = py
        self.robot_pos['z'] = pz

    # ── Model switching ────────────────────────────────────────────────────
    def _on_model_change(self, sender, app_data):
        """Callback when user selects a different YOLO model from the combo."""
        selected_name = app_data
        # Resolve to absolute path
        idx = AVAILABLE_MODEL_NAMES.index(selected_name) if selected_name in AVAILABLE_MODEL_NAMES else -1
        if idx < 0:
            return
        new_path = AVAILABLE_MODELS[idx]

        dpg.set_value("txt_model_status", f"Loading {selected_name} ...")
        dpg.configure_item("txt_model_status", color=(255, 255, 0))
        self.log(f"Switching model to {selected_name} ...")

        ok = self.detector.switch_model(new_path)

        if ok:
            dpg.set_value("txt_model_status", f"[OK] {selected_name}")
            dpg.configure_item("txt_model_status", color=(0, 255, 100))
            self.log(f"Model switched to {selected_name}", "SUCCESS")
        else:
            dpg.set_value("txt_model_status", "[X] Failed - kept previous")
            dpg.configure_item("txt_model_status", color=(255, 80, 80))
            self.log(f"Failed to load {selected_name}", "ERROR")

    # ── Separation logic toggles ───────────────────────────────────────────
    def _on_separation_toggle(self, sender=None, app_data=None):
        """Callback when user toggles a separation logic checkbox."""
        dc  = dpg.get_value("chk_depth_cluster")
        nms = dpg.get_value("chk_cross_nms")
        ws  = dpg.get_value("chk_watershed")

        # Push to detector (used by the detection thread)
        self.detector.use_depth_cluster = dc
        self.detector.use_cross_nms     = nms
        self.detector.use_watershed     = ws

        # Build pipeline status string
        tags = []
        if dc:
            tags.append("DC")
        if nms:
            tags.append("NMS")
        if ws:
            tags.append("WS")
        pipeline_str = "Pipeline: " + (" > ".join(tags) if tags else "RAW (no post-processing)")
        dpg.set_value("txt_pipeline_status", pipeline_str)

        # Color: cyan when stages active, yellow when RAW
        if tags:
            dpg.configure_item("txt_pipeline_status", color=(0, 220, 255))
        else:
            dpg.configure_item("txt_pipeline_status", color=(255, 220, 50))

        self.log(f"Detection pipeline: {pipeline_str}")

    def _on_connect(self):
        """Connect to robot."""
        delta_port = dpg.get_value("in_delta_port")
        slider_port = dpg.get_value("in_slider_port")
        
        delta_ok = self.delta.connect(delta_port)
        slider_ok = self.slider.connect(slider_port)
        
        if delta_ok and slider_ok:
            dpg.set_value("txt_robot_status", "Connected")
            dpg.configure_item("txt_robot_status", color=(0, 255, 0))
        else:
            dpg.set_value("txt_robot_status", "Connection Failed")
            dpg.configure_item("txt_robot_status", color=(255, 100, 100))
    
    def _on_disconnect(self):
        """Disconnect all hardware (robot, spectrum sensor, camera)."""
        disconnected = []

        # Stop system first if running
        if self.is_running:
            self._on_stop()

        # Delta robot
        if self.delta.connected:
            self.delta.set_vacuum(False)  # Safety: turn off vacuum
            self.delta.disconnect()
            disconnected.append("Delta")

        # Slider
        if self.slider.connected:
            self.slider.disconnect()
            disconnected.append("Slider")

        # Spectrum sensor
        if self.spectrum and self.spectrum.hardware_ready:
            self.spectrum.disconnect_hardware()
            disconnected.append("Spectrum")

        # Camera
        if self.camera.connected:
            self.camera.stop()
            disconnected.append("Camera")

        if disconnected:
            dpg.set_value("txt_robot_status", "Disconnected")
            dpg.configure_item("txt_robot_status", color=(255, 180, 0))
            self.log(f"Disconnected: {', '.join(disconnected)}")
        else:
            dpg.set_value("txt_robot_status", "Nothing connected")
            dpg.configure_item("txt_robot_status", color=(180, 180, 180))
            self.log("No hardware was connected")

    def _on_home(self):
        """Home the robot."""
        if self.delta.connected:
            self.delta.home()
            self.delta.home_scan_motor()
            
            # Reset position tracking to home position
            self.robot_pos = {'x': 0.0, 'y': 0.0, 'z': -250.0, 'r': 90.0}
            
            # Update all position displays
            self._update_position_display()
            
            self.log("Robot homing...")
    
    def _on_calibrate(self):
        """Switch to the Calibration tab."""
        # Find the tab bar and select the calibration tab
        if dpg.does_item_exist("tab_calibration"):
            dpg.set_value(dpg.get_item_parent("tab_calibration"), "tab_calibration")
            self.log("Switched to Calibration tab")
    
    def _on_reload_calibration(self):
        """Reload calibration data from file."""
        if self.detector.load_calibration():
            self.log("Calibration reloaded successfully", "SUCCESS")
            if self.detector.roi_corners is not None:
                self.log(f"ROI: {self.detector.roi_width_cm}x{self.detector.roi_height_cm}cm")
        else:
            self.log("Failed to reload calibration", "ERROR")

    # ── Calibration tab methods ────────────────────────────────────────────

    def _cal_log(self, msg, color=(200, 200, 200)):
        """Add a message to the calibration log panel."""
        if dpg.does_item_exist("cal_log_child"):
            dpg.add_text(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}",
                         parent="cal_log_child", color=color)
        self.log(f"[CAL] {msg}")

    def _cal_toggle(self, enable):
        """Enable or disable calibration overlay mode."""
        self.cal_mode = enable
        if enable:
            if not self.is_running:
                self._cal_log("Camera not running! Press START first.", (255, 80, 80))
                self.cal_mode = False
                return
            # Reset calibration state for fresh start
            self.cal_confirmed = False
            self.cal_checkerboard_corners = None
            self.cal_checkerboard_found = False
            self.cal_homography = None
            self.cal_roi_corners = None
            self.cal_floor_plane = None
            self.cal_floor_depth_map = None
            self.cal_corrected_corners = None
            self.cal_depth_frame = None

            dpg.configure_item("btn_cal_enable", enabled=False)
            dpg.configure_item("btn_cal_disable", enabled=True)
            dpg.configure_item("btn_cal_flip_v", enabled=True)
            dpg.configure_item("btn_cal_flip_h", enabled=True)
            dpg.configure_item("btn_cal_confirm", enabled=True)
            dpg.set_value("txt_cal_confirm_status", "Not confirmed")
            dpg.configure_item("txt_cal_confirm_status", color=(255, 80, 80))
            dpg.set_value("txt_cal_floor_status", "Floor: not calibrated")
            dpg.configure_item("txt_cal_floor_status", color=(255, 80, 80))
            dpg.configure_item("btn_cal_floor", enabled=False)
            dpg.configure_item("btn_cal_save", enabled=False)
            self._cal_log("Calibration mode ENABLED — place checkerboard on belt", (0, 255, 100))
        else:
            dpg.configure_item("btn_cal_enable", enabled=True)
            dpg.configure_item("btn_cal_disable", enabled=False)
            dpg.configure_item("btn_cal_flip_v", enabled=False)
            dpg.configure_item("btn_cal_flip_h", enabled=False)
            dpg.configure_item("btn_cal_confirm", enabled=False)
            dpg.configure_item("btn_cal_floor", enabled=False)
            dpg.configure_item("btn_cal_save", enabled=False)
            self._cal_log("Calibration mode DISABLED", (200, 200, 200))

    def _cal_flip_v(self):
        """Toggle vertical flip (swap entry/exit direction)."""
        self.cal_flip_vertical = not self.cal_flip_vertical
        self.cal_confirmed = False
        dpg.set_value("txt_cal_flip_v",
                      f"V-Flip: {'ON' if self.cal_flip_vertical else 'OFF'}")
        dpg.configure_item("txt_cal_flip_v",
                           color=(0, 200, 255) if self.cal_flip_vertical else (120, 120, 120))
        dpg.set_value("txt_cal_confirm_status", "Not confirmed (flip changed)")
        dpg.configure_item("txt_cal_confirm_status", color=(255, 200, 0))
        dpg.configure_item("btn_cal_floor", enabled=False)
        dpg.configure_item("btn_cal_save", enabled=False)
        self._cal_log(f"Vertical flip: {'ON' if self.cal_flip_vertical else 'OFF'}", (0, 200, 255))

    def _cal_flip_h(self):
        """Toggle horizontal flip (mirror left/right)."""
        self.cal_flip_horizontal = not self.cal_flip_horizontal
        self.cal_confirmed = False
        dpg.set_value("txt_cal_flip_h",
                      f"H-Flip: {'ON' if self.cal_flip_horizontal else 'OFF'}")
        dpg.configure_item("txt_cal_flip_h",
                           color=(0, 200, 255) if self.cal_flip_horizontal else (120, 120, 120))
        dpg.set_value("txt_cal_confirm_status", "Not confirmed (flip changed)")
        dpg.configure_item("txt_cal_confirm_status", color=(255, 200, 0))
        dpg.configure_item("btn_cal_floor", enabled=False)
        dpg.configure_item("btn_cal_save", enabled=False)
        self._cal_log(f"Horizontal flip: {'ON' if self.cal_flip_horizontal else 'OFF'}", (0, 200, 255))

    def _cal_get_oriented_corners(self, corners):
        """Get the 4 outer corners with user-controlled orientation (flip support)."""
        if corners is None:
            return None
        corners_grid = corners.reshape(self.CAL_CORNERS_Y, self.CAL_CORNERS_X, 2)
        c00 = corners_grid[0, 0]
        c0n = corners_grid[0, -1]
        cn0 = corners_grid[-1, 0]
        cnn = corners_grid[-1, -1]

        if self.cal_flip_vertical:
            tl, tr = cn0, cnn
            bl, br = c00, c0n
        else:
            tl, tr = c00, c0n
            bl, br = cn0, cnn

        if self.cal_flip_horizontal:
            tl, tr = tr, tl
            bl, br = br, bl

        return {'top_left': tl.copy(), 'top_right': tr.copy(),
                'bottom_left': bl.copy(), 'bottom_right': br.copy()}

    def _cal_preview_roi(self, corners):
        """Calculate preview ROI corners from detected checkerboard."""
        oriented = self._cal_get_oriented_corners(corners)
        if oriented is None:
            return None

        tl = oriented['top_left']
        tr = oriented['top_right']
        bl = oriented['bottom_left']

        x_span_px = np.linalg.norm(tr - tl)
        x_span_cm = (self.CAL_CORNERS_X - 1) * self.CAL_SQUARE_SIZE_CM
        px_per_cm_x = x_span_px / x_span_cm

        y_span_px = np.linalg.norm(bl - tl)
        y_span_cm = (self.CAL_CORNERS_Y - 1) * self.CAL_SQUARE_SIZE_CM
        px_per_cm_y = y_span_px / y_span_cm

        x_dir = (tr - tl) / x_span_px
        y_dir = (bl - tl) / y_span_px

        roi_tl = tl - x_dir * (self.CAL_SQUARE_SIZE_CM * px_per_cm_x) \
                     - y_dir * (self.CAL_SQUARE_SIZE_CM * px_per_cm_y)
        roi_tr = roi_tl + x_dir * (ROI_WIDTH_CM * px_per_cm_x)
        roi_bl = roi_tl + y_dir * (ROI_HEIGHT_CM * px_per_cm_y)
        roi_br = roi_tr + y_dir * (ROI_HEIGHT_CM * px_per_cm_y)

        return np.array([roi_tl, roi_tr, roi_br, roi_bl], dtype=np.float32)

    def _cal_compute_entry_exit(self, roi_corners):
        """Calculate entry and exit zone corners from ROI corners."""
        if roi_corners is None:
            return None, None
        roi_tl, roi_tr, roi_br, roi_bl = roi_corners

        left_dir = roi_bl - roi_tl
        left_norm = left_dir / np.linalg.norm(left_dir)
        right_dir = roi_br - roi_tr
        right_norm = right_dir / np.linalg.norm(right_dir)

        roi_h_px = np.linalg.norm(left_dir)
        px_per_cm = roi_h_px / ROI_HEIGHT_CM

        entry_px = ENTRY_PATH_CM * px_per_cm
        entry_corners = np.array([
            roi_tl - left_norm * entry_px,
            roi_tr - right_norm * entry_px,
            roi_tr, roi_tl
        ], dtype=np.float32)

        exit_px = EXIT_PATH_CM * px_per_cm
        exit_corners = np.array([
            roi_bl, roi_br,
            roi_br + right_norm * exit_px,
            roi_bl + left_norm * exit_px
        ], dtype=np.float32)

        return entry_corners, exit_corners

    def _cal_confirm(self):
        """Confirm calibration — lock ROI corners and compute homography."""
        if self.cal_checkerboard_corners is None or not self.cal_checkerboard_found:
            self._cal_log("No checkerboard detected! Cannot confirm.", (255, 80, 80))
            return

        corners = self.cal_checkerboard_corners
        oriented = self._cal_get_oriented_corners(corners)
        if oriented is None:
            self._cal_log("Failed to orient corners", (255, 80, 80))
            return

        tl = oriented['top_left']
        tr = oriented['top_right']
        bl = oriented['bottom_left']
        br = oriented['bottom_right']
        self.cal_corrected_corners = oriented

        # Compute homography from the 4 outer corners of the inner grid
        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        inner_w = (self.CAL_CORNERS_X - 1) * self.CAL_SQUARE_SIZE_CM  # 15cm
        inner_h = (self.CAL_CORNERS_Y - 1) * self.CAL_SQUARE_SIZE_CM  # 20cm
        dst_pts = np.array([[0, 0], [inner_w, 0], [inner_w, inner_h], [0, inner_h]],
                           dtype=np.float32)
        self.cal_homography = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Extend to full ROI size (20×30cm)
        x_span_px = np.linalg.norm(tr - tl)
        x_span_cm = (self.CAL_CORNERS_X - 1) * self.CAL_SQUARE_SIZE_CM
        px_per_cm_x = x_span_px / x_span_cm
        y_span_px = np.linalg.norm(bl - tl)
        y_span_cm = (self.CAL_CORNERS_Y - 1) * self.CAL_SQUARE_SIZE_CM
        px_per_cm_y = y_span_px / y_span_cm

        x_dir = (tr - tl) / x_span_px
        y_dir = (bl - tl) / y_span_px

        roi_tl = tl - x_dir * (self.CAL_SQUARE_SIZE_CM * px_per_cm_x) \
                     - y_dir * (self.CAL_SQUARE_SIZE_CM * px_per_cm_y)
        roi_tr = roi_tl + x_dir * (ROI_WIDTH_CM * px_per_cm_x)
        roi_bl = roi_tl + y_dir * (ROI_HEIGHT_CM * px_per_cm_y)
        roi_br = roi_tr + y_dir * (ROI_HEIGHT_CM * px_per_cm_y)

        self.cal_roi_corners = np.array([roi_tl, roi_tr, roi_br, roi_bl], dtype=np.float32)

        # Recompute homography for full ROI
        full_dst = np.array([[0, 0], [ROI_WIDTH_CM, 0],
                             [ROI_WIDTH_CM, ROI_HEIGHT_CM], [0, ROI_HEIGHT_CM]],
                            dtype=np.float32)
        self.cal_homography = cv2.getPerspectiveTransform(self.cal_roi_corners, full_dst)

        self.cal_confirmed = True
        dpg.set_value("txt_cal_confirm_status", "CONFIRMED ✓")
        dpg.configure_item("txt_cal_confirm_status", color=(0, 255, 0))
        dpg.configure_item("btn_cal_floor", enabled=True)
        dpg.configure_item("btn_cal_save", enabled=True)

        flip_info = ""
        if self.cal_flip_vertical:
            flip_info += " [V-FLIP]"
        if self.cal_flip_horizontal:
            flip_info += " [H-FLIP]"
        self._cal_log(f"Calibration CONFIRMED!{flip_info}", (0, 255, 0))
        self._cal_log(f"  ROI corners (px): {self.cal_roi_corners.astype(int).tolist()}", (0, 200, 150))

    def _cal_floor(self):
        """Calibrate floor plane from current depth data."""
        if self.cal_roi_corners is None:
            self._cal_log("Confirm calibration first!", (255, 80, 80))
            return
        if self.cal_depth_frame is None:
            self._cal_log("No depth frame available — is camera running?", (255, 80, 80))
            return

        intrinsics = self.camera.depth_intrinsics
        if intrinsics is None:
            self._cal_log("No camera intrinsics available!", (255, 80, 80))
            return

        self._cal_log("Calibrating floor plane...", (255, 255, 0))

        # Compute entry/exit zones
        entry_corners, exit_corners = self._cal_compute_entry_exit(self.cal_roi_corners)

        # Create combined belt mask
        mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        cv2.fillPoly(mask, [self.cal_roi_corners.astype(np.int32)], 255)
        if entry_corners is not None:
            cv2.fillPoly(mask, [entry_corners.astype(np.int32)], 255)
        if exit_corners is not None:
            cv2.fillPoly(mask, [exit_corners.astype(np.int32)], 255)

        # Collect 3D points from depth
        depth_frame = self.cal_depth_frame
        points_3d = []
        for y in range(0, IMAGE_HEIGHT, 4):
            for x in range(0, IMAGE_WIDTH, 4):
                if mask[y, x] == 0:
                    continue
                depth = depth_frame.get_distance(x, y)
                if depth <= 0 or depth > 2.0:
                    continue
                pt = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                points_3d.append(pt)

        if len(points_3d) < 100:
            self._cal_log(f"Only {len(points_3d)} valid depth points — need ≥100", (255, 80, 80))
            return

        points_3d = np.array(points_3d)

        # RANSAC plane fitting
        best_plane = None
        best_inliers = 0
        for _ in range(100):
            idx = np.random.choice(len(points_3d), 3, replace=False)
            p1, p2, p3 = points_3d[idx]
            v1, v2 = p2 - p1, p3 - p1
            normal = np.cross(v1, v2)
            nlen = np.linalg.norm(normal)
            if nlen < 1e-6:
                continue
            normal /= nlen
            d = -np.dot(normal, p1)
            distances = np.abs(np.dot(points_3d, normal) + d)
            inliers = int(np.sum(distances < 0.01))
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (float(normal[0]), float(normal[1]), float(normal[2]), float(d))

        self.cal_floor_plane = best_plane
        self._cal_log(f"Floor plane fitted: {best_inliers}/{len(points_3d)} inliers", (0, 255, 0))

        # Create floor depth map
        depth_data = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001
        floor_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        valid = (mask > 0) & (depth_data > 0.1) & (depth_data < 2.0)
        floor_map[valid] = depth_data[valid]

        # Smooth and fill holes
        floor_map = cv2.medianBlur(floor_map, 5)
        invalid = (mask > 0) & (floor_map == 0)
        if np.sum(invalid) > 0:
            kernel = np.ones((11, 11), np.uint8)
            filled = cv2.morphologyEx(floor_map, cv2.MORPH_CLOSE, kernel)
            floor_map[invalid] = filled[invalid]

        valid_count = int(np.sum((mask > 0) & (floor_map > 0)))
        total_count = int(np.sum(mask > 0))
        coverage = valid_count / total_count * 100 if total_count > 0 else 0

        self.cal_floor_depth_map = floor_map
        self._cal_log(f"Floor depth map: {valid_count}/{total_count} px ({coverage:.1f}% coverage)", (0, 255, 0))

        dpg.set_value("txt_cal_floor_status", f"Floor: OK ({best_inliers} inliers, {coverage:.0f}% map)")
        dpg.configure_item("txt_cal_floor_status", color=(0, 255, 0))

    def _cal_save(self):
        """Save calibration data to calibration_data.json and apply immediately."""
        if self.cal_homography is None or self.cal_roi_corners is None:
            self._cal_log("Confirm calibration first!", (255, 80, 80))
            return

        entry_corners, exit_corners = self._cal_compute_entry_exit(self.cal_roi_corners)
        intrinsics = self.camera.depth_intrinsics

        # Prepare floor depth map for storage (downsample 4x)
        floor_depth_map_data = None
        if self.cal_floor_depth_map is not None:
            downsampled = self.cal_floor_depth_map[::4, ::4]
            floor_depth_map_data = {
                "data": downsampled.tolist(),
                "original_shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
                "downsample_factor": 4,
                "description": "Floor depth in meters at each pixel (downsampled 4x)"
            }

        cal_data = {
            "timestamp": datetime.now().isoformat(),
            "checkerboard": {
                "squares_x": self.CAL_SQUARES_X,
                "squares_y": self.CAL_SQUARES_Y,
                "square_size_cm": self.CAL_SQUARE_SIZE_CM
            },
            "roi": {
                "width_cm": ROI_WIDTH_CM,
                "height_cm": ROI_HEIGHT_CM,
                "corners_px": self.cal_roi_corners.tolist()
            },
            "zones": {
                "entry_path_cm": ENTRY_PATH_CM,
                "exit_path_cm": EXIT_PATH_CM,
                "entry_corners_px": entry_corners.tolist() if entry_corners is not None else None,
                "exit_corners_px": exit_corners.tolist() if exit_corners is not None else None
            },
            "transforms": {
                "homography": self.cal_homography.tolist(),
                "homography_inv": np.linalg.inv(self.cal_homography).tolist()
            },
            "floor_plane": {
                "coefficients": list(self.cal_floor_plane) if self.cal_floor_plane else None,
                "description": "ax + by + cz + d = 0"
            },
            "floor_depth_map": floor_depth_map_data,
            "camera": {
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "intrinsics": {
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "ppx": intrinsics.ppx,
                    "ppy": intrinsics.ppy,
                    "coeffs": list(intrinsics.coeffs)
                } if intrinsics else None
            }
        }

        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(cal_data, f, indent=2)
            self._cal_log(f"Saved to {CALIBRATION_FILE}", (0, 255, 0))

            # Reload into detector immediately (no restart needed)
            if self.detector.load_calibration():
                self._cal_log("Applied to detector — active immediately!", (0, 255, 0))
                dpg.set_value("txt_cal_save_status",
                              f"Saved & applied ({datetime.now().strftime('%H:%M:%S')})")
                dpg.configure_item("txt_cal_save_status", color=(0, 255, 0))
                # Update the info panel
                dpg.set_value("txt_cal_current_info",
                              f"  Loaded: YES\n  ROI: {self.detector.roi_width_cm}x{self.detector.roi_height_cm} cm")
                dpg.configure_item("txt_cal_current_info", color=(0, 255, 0))
            else:
                self._cal_log("Saved but reload failed!", (255, 200, 0))
                dpg.set_value("txt_cal_save_status", "Saved but reload failed")
                dpg.configure_item("txt_cal_save_status", color=(255, 200, 0))
        except Exception as e:
            self._cal_log(f"Save failed: {e}", (255, 80, 80))
            dpg.set_value("txt_cal_save_status", f"Error: {e}")
            dpg.configure_item("txt_cal_save_status", color=(255, 80, 80))

    def _cal_draw_overlay(self, frame, depth_frame):
        """Detect checkerboard and draw calibration overlay on the video frame.

        Called from _update_frame when cal_mode is True.
        Returns the annotated frame.
        """
        # Store depth frame for floor calibration
        self.cal_depth_frame = depth_frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pattern = (self.CAL_CORNERS_X, self.CAL_CORNERS_Y)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)

        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            self.cal_checkerboard_corners = corners
            self.cal_checkerboard_found = True
            # Draw the checkerboard corners
            cv2.drawChessboardCorners(frame, pattern, corners, True)
            dpg.set_value("txt_cal_board_status", "Checkerboard: DETECTED ✓")
            dpg.configure_item("txt_cal_board_status", color=(0, 255, 0))
        else:
            self.cal_checkerboard_found = False
            dpg.set_value("txt_cal_board_status", "Checkerboard: not detected")
            dpg.configure_item("txt_cal_board_status", color=(255, 80, 80))

        # Draw ROI preview or confirmed ROI
        display_roi = None
        is_preview = True
        if self.cal_confirmed and self.cal_roi_corners is not None:
            display_roi = self.cal_roi_corners
            is_preview = False
        elif found:
            display_roi = self._cal_preview_roi(corners)

        if display_roi is not None:
            roi_int = display_roi.astype(np.int32)
            roi_color = (0, 255, 255) if is_preview else (0, 255, 0)
            thickness = 2 if is_preview else 3
            cv2.polylines(frame, [roi_int], True, roi_color, thickness)

            # Corner labels
            labels = ['TL (ENTRY)', 'TR (ENTRY)', 'BR (EXIT)', 'BL (EXIT)']
            for i, (corner, label) in enumerate(zip(roi_int, labels)):
                cv2.circle(frame, tuple(corner), 6, roi_color, -1)
                cv2.putText(frame, label, (corner[0] + 8, corner[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, roi_color, 1)

            # Flow direction arrows
            top_c = ((roi_int[0] + roi_int[1]) / 2).astype(int)
            bot_c = ((roi_int[2] + roi_int[3]) / 2).astype(int)
            flow = bot_c - top_c
            fl = np.linalg.norm(flow)
            if fl > 0:
                for t_i in range(3):
                    t = 0.25 + t_i * 0.25
                    a_s = top_c + (flow * (t - 0.1)).astype(int)
                    a_e = top_c + (flow * (t + 0.05)).astype(int)
                    cv2.arrowedLine(frame, tuple(a_s), tuple(a_e), (0, 200, 255), 2, tipLength=0.5)

            # Entry / Exit zone overlays
            entry_c, exit_c = self._cal_compute_entry_exit(display_roi)
            zone_alpha = 0.08 if is_preview else 0.12
            overlay = frame.copy()
            if entry_c is not None:
                cv2.fillPoly(overlay, [entry_c.astype(np.int32)], (255, 255, 0))
                ec = np.mean(entry_c, axis=0).astype(int)
                cv2.putText(frame, "ENTRY", (ec[0] - 25, ec[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if exit_c is not None:
                cv2.fillPoly(overlay, [exit_c.astype(np.int32)], (0, 255, 255))
                xc = np.mean(exit_c, axis=0).astype(int)
                cv2.putText(frame, "EXIT", (xc[0] - 20, xc[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.addWeighted(overlay, zone_alpha, frame, 1 - zone_alpha, 0, frame)

            # Status label
            status = "CONFIRMED" if not is_preview else "PREVIEW"
            cv2.putText(frame, f"CAL: {status}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

        # Flip indicators on frame
        if self.cal_flip_vertical or self.cal_flip_horizontal:
            flip_txt = ""
            if self.cal_flip_vertical:
                flip_txt += "[V-FLIP] "
            if self.cal_flip_horizontal:
                flip_txt += "[H-FLIP]"
            cv2.putText(frame, flip_txt, (10, IMAGE_HEIGHT - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        return frame

    def _update_frame(self):
        """Process one frame."""
        frame_data = self.camera.get_latest()
        if not frame_data:
            return
        
        color_image, depth_frame, timestamp = frame_data
        display_frame = color_image.copy()

        # Reset replay index when playback loops.
        if self.replay_enabled and self.video_source is not None:
            loop_now = int(getattr(self.video_source, "loop_count", 0))
            if loop_now != self._replay_last_loop_count:
                self._replay_last_loop_count = loop_now
                self.replay_pick_idx = 0
                self.log(f"[VIDEO] Replay sidecar reset for loop #{loop_now}")
        
        # Get belt speed — dynamic or fixed based on checkbox
        belt_speed = self._get_belt_speed()
        conf_threshold = dpg.get_value("in_conf")
        self.detector.conf_threshold = conf_threshold
        
        # Calculate time delta (prefer source timestamp for playback fidelity)
        current_time = float(timestamp) if timestamp else time.time()
        dt = current_time - self.last_frame_time
        if dt < 0 or dt > 1.0:
            dt = 0.0
        self.last_frame_time = current_time
        
        if self.is_tracking and not self.cal_mode:
            # Synchronous detection — no lag, results match current frame
            t_infer0 = time.perf_counter()
            detections = self.detector.detect(color_image, depth_frame)
            infer_ms = (time.perf_counter() - t_infer0) * 1000.0
            fps_inst = (1.0 / dt) if dt > 1e-6 else None
            self._benchmark_push_cam_sample(infer_ms, fps_inst, len(detections))
            self.last_detections = detections
            
            # Update tracker with fresh detections
            self.tracker.update(
                detections,
                belt_y_calculator=self.detector.pixel_y_to_belt_y_cm,
                x_to_cm_func=self.detector.pixel_x_to_belt_x_cm
            )
            
            # Check registration crossing. A new queue entry is the camera's
            # verification event and owns the Detect timestamp.
            queue_before_registration = set(self.tracker.picking_queue)
            self.tracker.check_registration_crossing(
                REGISTRATION_LINE_CM, 
                color_image, 
                belt_speed_cm_s=belt_speed
            )
            
            # Check exit crossing — vision-based speed measurement.
            # Objects crossing the ROI exit provide a measured transit time,
            # enabling per-object speed calculation for precise predictions.
            self.tracker.check_exit_crossing()
            
            # Consolidate queue — merge same-class objects too close together
            self.tracker.consolidate_queue()
            self._record_new_queue_detections(queue_before_registration)
            
            # Advance queued objects by belt movement
            belt_advancement = belt_speed * dt
            # Exit limit: ROI + offset to workspace + workspace depth + buffer
            exit_limit = ROI_HEIGHT_CM + ROBOT_WORKSPACE_OFFSET_CM + ROBOT_WORKSPACE_DEPTH_CM + 5
            self.tracker.advance_queued_objects(belt_advancement, exit_limit)
            
            # ALWAYS draw detections (use tracked objects for smooth mask fading)
            # Instead of raw YOLO detections (which flicker), draw from tracked
            # objects whose masks fade in/out smoothly.
            display_frame = self._draw_tracked_masks(display_frame)
            
            # Draw X markers for tracked objects
            display_frame = self._draw_tracking(display_frame)
            
            # Auto-pick logic (disabled during TRACK mode)
            if self.auto_pick and not self.track_mode:
                self._process_picking()
            
            # Feed real-time position to robot during pick approach
            # (main-loop-driven tracking — same quality as TRACK mode)
            if self.auto_pick and hasattr(self, 'robot_manager') and self.robot_manager is not None:
                self._feed_pick_tracking()
            
            # TRACK mode: robot follows object through workspace
            if self.track_mode:
                self._process_tracking()
        
        # ── Calibration overlay (when calibration mode is active) ──
        # Skip ROI zones during calibration — calibration is what DEFINES them
        if self.cal_mode:
            display_frame = self._cal_draw_overlay(display_frame, depth_frame)
        else:
            # Draw ROI zones (entry / detection / exit) and registration line
            display_frame = self._draw_roi_zones(display_frame)
        
        # ── Pipeline status overlay on video feed ──
        _tags = []
        if self.detector.use_depth_cluster:
            _tags.append("DC")
        if self.detector.use_cross_nms:
            _tags.append("NMS")
        if self.detector.use_watershed:
            _tags.append("WS")
        _pl_str = "Pipeline: " + (" > ".join(_tags) if _tags else "RAW")
        _pl_color = (255, 220, 0) if not _tags else (0, 220, 255)  # BGR
        cv2.putText(display_frame, _pl_str, (10, IMAGE_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display_frame, _pl_str, (10, IMAGE_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, _pl_color, 1, cv2.LINE_AA)

        # Recording / playback status overlays on vision feed
        rec_y = 15
        if self.bag_recording:
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(display_frame, (15, rec_y), 7, (0, 0, 255), -1)
            cv2.putText(display_frame, f"REC .bag", (30, rec_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            rec_y += 20
        if self.mp4_recording:
            elapsed = self.record_frame_count / max(1.0, float(self._mp4_record_fps))
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(display_frame, (15, rec_y), 7, (0, 100, 255), -1)
            cv2.putText(display_frame, f"REC EXP {elapsed:.1f}s", (30, rec_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 2)
            rec_y += 20
        if self.video_source is not None:
            cv2.putText(display_frame, "PLAYBACK", (IMAGE_WIDTH - 120, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
        
        # Update video texture
        display_rgba = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
        data = display_rgba.astype(np.float32) / 255.0
        dpg.set_value("video_texture", data.flatten())

        # Write .mp4 frame from raw camera image with source-time sync.
        # This keeps output duration close to real x1 even if detection loop drops frames.
        if self.mp4_recording and self.mp4_writer is not None:
            src_ts = float(timestamp) if timestamp else time.time()
            if self._mp4_source_t0 is None:
                self._mp4_source_t0 = src_ts
                self._mp4_record_start_epoch = src_ts
                self._mp4_record_start_utc = datetime.fromtimestamp(
                    src_ts, timezone.utc
                ).isoformat().replace("+00:00", "Z")
                self.mp4_writer.write(color_image)
                self.record_frame_count += 1
            else:
                elapsed_src = max(0.0, src_ts - self._mp4_source_t0)
                target_count = int(elapsed_src * float(self._mp4_record_fps)) + 1
                # Duplicate latest frame to fill timing gaps when processing is slower.
                while self.record_frame_count < target_count:
                    self.mp4_writer.write(color_image)
                    self.record_frame_count += 1

        # ── Robot Camera: read, flip, display, record ──
        self._process_robot_camera_frame()

        self._update_workspace_simulation()

    def _update_workspace_simulation(self):
        """Draw robot workspace simulation showing objects traveling from registration line through ROI, gap, into workspace."""
        # Check if drawlist exists
        if not dpg.does_item_exist("workspace_drawlist"):
            return
            
        dpg.delete_item("workspace_drawlist", children_only=True)
        dl = "workspace_drawlist"

        # --- Layout constants ---
        # Belt section we visualize: from ROI top (0cm) to workspace exit (62cm)
        # ROI top = 0cm (where camera first sees objects)
        # Registration line = 15cm (where objects get queued)
        # ROI exit = ROI_HEIGHT_CM = 30cm
        # Workspace entry = ROI exit + gap = 30 + 12 = 42cm 
        # Workspace exit  = entry + depth = 42 + 20 = 62cm
        roi_top_y_cm = 0                                                     # 0
        reg_line_y_cm = REGISTRATION_LINE_CM                                 # 15
        roi_exit_y_cm = ROI_HEIGHT_CM                                        # 30
        gap_cm = ROBOT_WORKSPACE_OFFSET_CM                                   # 12
        ws_entry_y_cm = roi_exit_y_cm + gap_cm                               # 42
        ws_exit_y_cm = ws_entry_y_cm + ROBOT_WORKSPACE_DEPTH_CM              # 62
        sim_start_y_cm = roi_top_y_cm                                        # 0 (full ROI)
        total_y_cm = ws_exit_y_cm - sim_start_y_cm                           # 62 (0→62)
        belt_width_cm = ROBOT_WORKSPACE_WIDTH_CM                             # 20

        # Drawing area - fixed dimensions matching drawlist size
        margin_left = 40
        margin_top = 30
        margin_right = 60
        margin_bottom = 30
        
        # Fixed drawlist dimensions
        available_width = 500
        available_height = 700
        
        draw_w = available_width - margin_left - margin_right
        draw_h = available_height - margin_top - margin_bottom
        
        x0 = margin_left
        y0 = margin_top

        # Conversion helpers: belt cm -> pixel
        def belt_x_to_px(bx):
            # Mirror X: camera left (bx=0) -> simulation right (robot perspective)
            return x0 + int((belt_width_cm - bx) / belt_width_cm * draw_w)

        def belt_y_to_px(by):
            # by is absolute belt coordinate; map reg_line..ws_exit -> 0..draw_h
            return y0 + int((by - sim_start_y_cm) / total_y_cm * draw_h)

        # --- Belt direction arrow ---
        arrow_x = x0 - 25
        dpg.draw_arrow([arrow_x, y0 + draw_h - 20], [arrow_x, y0 + 20],
                       color=(200, 200, 200, 255), thickness=2, size=8, parent=dl)
        dpg.draw_text([arrow_x - 15, y0 + draw_h // 2 - 5], "Belt",
                      color=(180, 180, 180, 255), size=10, parent=dl)

        # --- Detection zone (ROI top -> registration line) ---
        det_y0 = belt_y_to_px(roi_top_y_cm)
        det_y1 = belt_y_to_px(reg_line_y_cm)
        dpg.draw_rectangle([x0, det_y0], [x0 + draw_w, det_y1],
                           color=(60, 60, 80, 255), fill=(30, 30, 50, 25),
                           thickness=1, parent=dl)
        dpg.draw_text([x0 + draw_w // 2 - 35, (det_y0 + det_y1) // 2 - 5],
                      f"Detection {reg_line_y_cm:.0f}cm",
                      color=(120, 120, 160, 255), size=11, parent=dl)

        # --- Registration line (dashed) ---
        reg_px_y = belt_y_to_px(reg_line_y_cm)
        for dx in range(0, draw_w, 12):
            dpg.draw_line([x0 + dx, reg_px_y], [x0 + min(dx + 7, draw_w), reg_px_y],
                         color=(255, 255, 0, 200), thickness=2, parent=dl)
        dpg.draw_text([x0 + draw_w + 5, reg_px_y - 6], "REG Line",
                      color=(255, 255, 0, 255), size=10, parent=dl)

        # --- ROI zone (registration line -> ROI exit) ---
        roi_y0 = belt_y_to_px(reg_line_y_cm)
        roi_y1 = belt_y_to_px(roi_exit_y_cm)
        dpg.draw_rectangle([x0, roi_y0], [x0 + draw_w, roi_y1],
                           color=(100, 100, 0, 255), fill=(50, 50, 0, 25),
                           thickness=1, parent=dl)
        dpg.draw_text([x0 + draw_w // 2 - 25, (roi_y0 + roi_y1) // 2 - 5],
                      f"ROI {roi_exit_y_cm - reg_line_y_cm:.0f}cm",
                      color=(200, 200, 0, 255), size=11, parent=dl)

        # --- Gap zone (ROI exit -> workspace entry) ---
        gap_y0 = belt_y_to_px(roi_exit_y_cm)
        gap_y1 = belt_y_to_px(ws_entry_y_cm)
        dpg.draw_rectangle([x0, gap_y0], [x0 + draw_w, gap_y1],
                           color=(80, 80, 80, 255), fill=(40, 40, 40, 35),
                           thickness=1, parent=dl)
        dpg.draw_text([x0 + draw_w + 5, gap_y0 + 2], "ROI Exit",
                      color=(255, 200, 0, 255), size=10, parent=dl)
        
        # Exit line (speed measurement checkpoint) — dashed green
        exit_line_px_y = belt_y_to_px(EXIT_LINE_CM)
        for dx in range(0, draw_w, 12):
            dpg.draw_line([x0 + dx, exit_line_px_y], [x0 + min(dx + 7, draw_w), exit_line_px_y],
                         color=(100, 220, 100, 180), thickness=1, parent=dl)
        # Show measured speed near exit line
        ms = self.tracker.measured_belt_speed
        speed_txt = f"EXIT (Spd: {ms:.1f})" if ms else "EXIT (Spd: --)"
        dpg.draw_text([x0 + draw_w + 5, exit_line_px_y - 14], speed_txt,
                      color=(100, 220, 100, 255), size=10, parent=dl)
        
        dpg.draw_text([x0 + draw_w // 2 - 20, (gap_y0 + gap_y1) // 2 - 5],
                      f"Gap {gap_cm:.0f}cm",
                      color=(150, 150, 150, 255), size=11, parent=dl)

        # --- Workspace zone ---
        ws_y0 = belt_y_to_px(ws_entry_y_cm)
        ws_y1 = belt_y_to_px(ws_exit_y_cm)
        dpg.draw_rectangle([x0, ws_y0], [x0 + draw_w, ws_y1],
                           color=(0, 255, 255, 255), thickness=2, parent=dl)
        dpg.draw_text([x0 + draw_w + 5, ws_y0 + 2], "WS Entry",
                      color=(0, 255, 255, 255), size=10, parent=dl)
        dpg.draw_text([x0 + draw_w + 5, ws_y1 - 12], "WS Exit",
                      color=(0, 255, 255, 255), size=10, parent=dl)

        # Workspace vertical grid lines (3 columns)
        for i in range(1, 3):
            xi = x0 + i * draw_w // 3
            dpg.draw_line([xi, ws_y0], [xi, ws_y1],
                         color=(80, 80, 80, 255), thickness=1, parent=dl)

        # --- Workspace section lines: Top / Middle / Bottom ---
        # Divide workspace into 3 equal sections (each ~6.67cm of 20cm)
        ws_third = ROBOT_WORKSPACE_DEPTH_CM / 3.0   # ~6.67cm per section
        section_names = ["TOP", "MID", "BOT"]
        section_colors = [
            (255, 120, 120, 200),  # Top - red-ish
            (255, 255, 120, 200),  # Middle - yellow-ish
            (120, 200, 255, 200),  # Bottom - blue-ish
        ]
        section_fills = [
            (255, 80, 80, 25),     # Top - faint red
            (255, 255, 80, 25),    # Middle - faint yellow
            (80, 180, 255, 25),    # Bottom - faint blue
        ]

        for i in range(3):
            sec_top_cm = ws_entry_y_cm + i * ws_third
            sec_bot_cm = ws_entry_y_cm + (i + 1) * ws_third
            sec_top_px = belt_y_to_px(sec_top_cm)
            sec_bot_px = belt_y_to_px(sec_bot_cm)

            # Faint fill for each section
            dpg.draw_rectangle([x0 + 1, sec_top_px], [x0 + draw_w - 1, sec_bot_px],
                               color=(0, 0, 0, 0), fill=section_fills[i],
                               thickness=0, parent=dl)

            # Divider line between sections (skip first — that's WS Entry already)
            if i > 0:
                dpg.draw_line([x0, sec_top_px], [x0 + draw_w, sec_top_px],
                             color=section_colors[i], thickness=1, parent=dl)

            # Section label (centered in section)
            label_y = (sec_top_px + sec_bot_px) // 2 - 5
            dpg.draw_text([x0 + 5, label_y], section_names[i],
                         color=section_colors[i], size=11, parent=dl)

        # Workspace center crosshair
        cx = x0 + draw_w // 2
        cy = (ws_y0 + ws_y1) // 2
        dpg.draw_line([cx - 8, cy], [cx + 8, cy], color=(0, 180, 180, 120), thickness=1, parent=dl)
        dpg.draw_line([cx, cy - 8], [cx, cy + 8], color=(0, 180, 180, 120), thickness=1, parent=dl)

        # --- Draw tracked objects (all: queued + pre-queue in detection zone) ---
        belt_speed = self._get_belt_speed()
        current_time = time.time()

        for obj_id, obj in self.tracker.objects.items():
            is_queued = obj.get('in_queue', False)

            belt_x = obj.get('last_known_x_cm', obj.get('belt_x_cm', None))
            if belt_x is None:
                continue

            # Time-anchor based Y prediction with measured speed (drift-free)
            real_y = self.tracker._get_anchor_belt_y(obj, belt_speed_cm_s=belt_speed)
            if real_y is None:
                belt_y = obj.get('belt_y_cm', None)
                if belt_y is None:
                    continue
                real_y = belt_y

            # Show objects from top of ROI onward
            if real_y < roi_top_y_cm or real_y > ws_exit_y_cm + 2:
                continue

            # Convert to pixel coordinates
            px = belt_x_to_px(belt_x)
            py = belt_y_to_px(real_y)

            # Clamp to drawing area
            px = max(x0, min(x0 + draw_w, px))
            py = max(y0, min(y0 + draw_h, py))

            # Get class color (dimmed for pre-queue objects in detection zone)
            class_id = obj.get('class_id', -1)
            class_name = obj.get('class_name', CLASS_NAMES.get(class_id, '?'))
            r, g, b = CLASS_COLORS.get(class_id, DEFAULT_COLOR)
            is_ghost = obj.get('ghost', False)
            alpha = (140 if is_ghost else 255) if is_queued else 120
            obj_color = (r, g, b, alpha)

            status = obj.get('status', 'Tracking' if not is_queued else 'Queued')

            # (Mask polygon removed — performance-focused simulation uses
            #  simple markers only; masks are shown on the video feed.)

            if is_ghost:
                # Ghost: draw dashed rectangle outline instead of X mark
                ghost_sz = 10
                dpg.draw_rectangle(
                    [px - ghost_sz, py - ghost_sz],
                    [px + ghost_sz, py + ghost_sz],
                    color=(r, g, b, 140), thickness=1, parent=dl)
                dpg.draw_circle([px, py], 2, color=(r, g, b, 140),
                               fill=(r, g, b, 140), parent=dl)
            else:
                # Draw X mark at centroid (smaller for pre-queue)
                sz = 8 if is_queued else 5
                dpg.draw_line([px - sz, py - sz], [px + sz, py + sz],
                             color=obj_color, thickness=2, parent=dl)
                dpg.draw_line([px + sz, py - sz], [px - sz, py + sz],
                             color=obj_color, thickness=1, parent=dl)
                # Draw small circle for visibility
                dpg.draw_circle([px, py], 3, color=obj_color, fill=obj_color, parent=dl)

            # Label: ID + class + height + status + speed info
            h_cm = obj.get('height_cm', 0) or 0
            label = f"{obj_id}:{class_name} H:{h_cm:.1f}"
            if status == 'Picking':
                label += " [PICK]"
            obj_spd = obj.get('measured_speed')
            if obj_spd is not None:
                label += f" v={obj_spd:.1f}"
            reg_t = obj.get('reg_time')
            if reg_t is not None:
                dt_sec = current_time - reg_t
                label += f" {dt_sec:.1f}s"
            dpg.draw_text([px + 10, py - 8], label,
                         color=obj_color, size=10, parent=dl)

            # Show cluster/stack tag below main label
            group = obj.get('stack_group')
            s_type = obj.get('stack_type', 'none')
            if group and len(group) > 1:
                grp_sorted = sorted(
                    group,
                    key=lambda oid: self.tracker.objects.get(oid, {}).get('height_cm', 0) or 0,
                    reverse=True
                )
                rank = grp_sorted.index(obj_id) + 1 if obj_id in grp_sorted else 0
                is_top = (rank == 1)
                if s_type == 'physical_stack':
                    tag = f"STACK {'TOP' if is_top else 'BOT'} {rank}/{len(group)}"
                    tag_color = (0, 255, 255, 255) if is_top else (255, 100, 100, 255)
                else:
                    tag = f"CLUSTER {rank}/{len(group)}"
                    tag_color = (255, 200, 0, 255)
                dpg.draw_text([px + 10, py + 4], tag,
                             color=tag_color, size=9, parent=dl)

                # Draw connecting line to other group members
                for other_id in group:
                    if other_id == obj_id or other_id not in self.tracker.objects:
                        continue
                    o_obj = self.tracker.objects[other_id]
                    o_bx = o_obj.get('last_known_x_cm', o_obj.get('belt_x_cm'))
                    if o_bx is None:
                        continue
                    o_reg_time = o_obj.get('reg_time')
                    o_reg_y = o_obj.get('reg_belt_y')
                    if o_reg_time is not None and o_reg_y is not None:
                        o_real_y = o_reg_y + ((current_time - o_reg_time) * belt_speed)
                    else:
                        o_real_y = o_obj.get('belt_y_cm', 0)
                    o_px = belt_x_to_px(o_bx)
                    o_py = belt_y_to_px(o_real_y)
                    dpg.draw_line([px, py], [o_px, o_py],
                                color=(255, 255, 0, 80), thickness=1, parent=dl)

            # Highlight if this is the TRACK target
            if self.track_mode and obj_id == self.track_target_id:
                # Orange ring around tracked object
                dpg.draw_circle([px, py], 14,
                               color=(255, 165, 0, 255), thickness=2, parent=dl)
                dpg.draw_text([px + 10, py + 8], "TRACK",
                             color=(255, 165, 0, 255), size=10, parent=dl)

            # Draw object edge if available
            edge_pts = obj.get('edge_pts', None)
            if edge_pts and len(edge_pts) > 1:
                sim_edge = []
                for ex, ey in edge_pts:
                    ex_px = belt_x_to_px(ex)
                    ey_px = belt_y_to_px(ey + (real_y - belt_y))  # offset edge by same advancement
                    sim_edge.append([ex_px, ey_px])
                dpg.draw_polyline(sim_edge, color=(255, 255, 0, 255), thickness=1, parent=dl)

        # --- Queue info text ---
        queue_len = len(self.tracker.picking_queue)
        dpg.draw_text([x0, y0 + draw_h + 5],
                      f"Queue: {queue_len} objects | Belt: {belt_speed:.1f} cm/s",
                      color=(200, 200, 200, 255), size=12, parent=dl)

        # --- Robot position (red dot) ---
        # Get current robot position in robot coordinates (mm)
        robot_rx = self.delta.last_x
        robot_ry = self.delta.last_y
        robot_rz = self.delta.last_z

        # Convert robot coordinates back to belt coordinates (mm)
        belt_mm_x, belt_mm_y = inverse_bilinear_interpolate(robot_rx, robot_ry)
        # Convert belt mm to belt cm
        target_belt_x_cm = belt_mm_x / 10.0  # 0-20cm
        target_belt_y_cm = belt_mm_y / 10.0   # 0-20cm within workspace

        # Show robot dot when:
        # 1. Robot is actively busy (executing a pick cycle), OR
        # 2. Robot Z is low enough to be in/near workspace (Z < -280)
        robot_is_active = (hasattr(self, 'robot_manager') and 
                          self.robot_manager is not None and 
                          self.robot_manager.is_busy)
        robot_z_in_range = robot_rz < -280
        robot_in_ws = robot_is_active or robot_z_in_range

        if robot_in_ws:
            self.sim_robot_in_workspace = True
            # Smooth interpolation toward target
            smooth = self.sim_robot_smooth_factor
            self.sim_robot_x += (target_belt_x_cm - self.sim_robot_x) * smooth
            self.sim_robot_y += (target_belt_y_cm - self.sim_robot_y) * smooth

            # Add to trail (keep last 20 positions)
            self.sim_robot_trail.append((self.sim_robot_x, self.sim_robot_y))
            if len(self.sim_robot_trail) > 20:
                self.sim_robot_trail = self.sim_robot_trail[-20:]

            # Map workspace-local belt cm to absolute belt Y for pixel conversion
            # workspace belt_y_cm (0-20) maps to absolute belt Y (ws_entry..ws_exit)
            abs_belt_y = ws_entry_y_cm + self.sim_robot_y

            rpx = belt_x_to_px(self.sim_robot_x)
            rpy = belt_y_to_px(abs_belt_y)

            # Clamp to drawing area (allow full belt area, not just workspace)
            rpx = max(x0, min(x0 + draw_w, rpx))
            rpy = max(y0, min(y0 + draw_h, rpy))

            # Draw trail (fading)
            for i, (tx, ty) in enumerate(self.sim_robot_trail):
                alpha = int(40 + (180 * i / len(self.sim_robot_trail)))
                abs_ty = ws_entry_y_cm + ty
                tpx = belt_x_to_px(tx)
                tpy = belt_y_to_px(abs_ty)
                tpx = max(x0, min(x0 + draw_w, tpx))
                tpy = max(y0, min(y0 + draw_h, tpy))
                dpg.draw_circle([tpx, tpy], 2,
                               color=(255, 50, 50, alpha), fill=(255, 50, 50, alpha),
                               parent=dl)

            # Draw robot dot (large red circle with outline)
            dot_color = (255, 0, 0, 255) if robot_is_active else (200, 80, 80, 200)
            fill_color = (255, 50, 50, 200) if robot_is_active else (180, 60, 60, 150)
            dpg.draw_circle([rpx, rpy], 8,
                           color=dot_color, fill=fill_color,
                           thickness=2, parent=dl)
            # Draw crosshair on dot
            dpg.draw_line([rpx - 5, rpy], [rpx + 5, rpy],
                         color=(255, 255, 255, 200), thickness=1, parent=dl)
            dpg.draw_line([rpx, rpy - 5], [rpx, rpy + 5],
                         color=(255, 255, 255, 200), thickness=1, parent=dl)
            # Label with status
            status_label = "Robot [PICK]" if robot_is_active else "Robot"
            dpg.draw_text([rpx + 12, rpy - 6], status_label,
                         color=(255, 100, 100, 255), size=10, parent=dl)
        else:
            # Robot not in workspace - fade out trail gradually
            if self.sim_robot_trail:
                self.sim_robot_trail = self.sim_robot_trail[1:]  # Fade trail slowly
                
                # Draw remaining trail
                for i, (tx, ty) in enumerate(self.sim_robot_trail):
                    alpha = int(20 + (80 * i / max(len(self.sim_robot_trail), 1)))
                    abs_ty = ws_entry_y_cm + ty
                    tpx = belt_x_to_px(tx)
                    tpy = belt_y_to_px(abs_ty)
                    tpx = max(x0, min(x0 + draw_w, tpx))
                    tpy = max(y0, min(y0 + draw_h, tpy))
                    dpg.draw_circle([tpx, tpy], 2,
                                   color=(255, 50, 50, alpha), fill=(255, 50, 50, alpha),
                                   parent=dl)
            
            self.sim_robot_in_workspace = False
            # Clear pick target when robot returns to standby
            if self.sim_pick_target is not None:
                target_age = current_time - self.sim_pick_target.get('dispatch_time', 0)
                if target_age > 3.0:
                    self.sim_pick_target = None
            dpg.draw_text([x0 + draw_w // 2 - 30, ws_y1 + 3], "Robot: Standby",
                         color=(150, 100, 100, 200), size=10, parent=dl)

        # --- Pick target marker (green crosshair showing where robot SHOULD go) ---
        if self.sim_pick_target is not None:
            target = self.sim_pick_target
            target_age = current_time - target.get('dispatch_time', current_time)
            
            # Show target for up to 5 seconds after dispatch (covers full pick cycle)
            if target_age < 5.0:
                t_belt_x = target['belt_x']
                t_belt_y_abs = target['belt_y_abs']
                
                # Advance target Y by belt movement since dispatch
                t_belt_y_abs += belt_speed * target_age
                
                # Convert to pixels
                tgt_px = belt_x_to_px(t_belt_x)
                tgt_py = belt_y_to_px(t_belt_y_abs)
                tgt_px = max(x0, min(x0 + draw_w, tgt_px))
                tgt_py = max(y0, min(y0 + draw_h, tgt_py))
                
                # Fade out over last second
                alpha = 255 if target_age < 4.0 else int(255 * (5.0 - target_age))
                green = (0, 255, 100, alpha)
                green_dim = (0, 200, 80, max(40, alpha // 2))
                
                # Draw target crosshair (green, larger than object markers)
                sz = 12
                dpg.draw_line([tgt_px - sz, tgt_py], [tgt_px + sz, tgt_py],
                             color=green, thickness=2, parent=dl)
                dpg.draw_line([tgt_px, tgt_py - sz], [tgt_px, tgt_py + sz],
                             color=green, thickness=2, parent=dl)
                # Green diamond outline
                dpg.draw_line([tgt_px, tgt_py - sz], [tgt_px + sz, tgt_py],
                             color=green_dim, thickness=1, parent=dl)
                dpg.draw_line([tgt_px + sz, tgt_py], [tgt_px, tgt_py + sz],
                             color=green_dim, thickness=1, parent=dl)
                dpg.draw_line([tgt_px, tgt_py + sz], [tgt_px - sz, tgt_py],
                             color=green_dim, thickness=1, parent=dl)
                dpg.draw_line([tgt_px - sz, tgt_py], [tgt_px, tgt_py - sz],
                             color=green_dim, thickness=1, parent=dl)
                
                # Label with coordinates
                label_text = f"TARGET ID:{target['obj_id']} X:{t_belt_x:.1f} Y:{target['belt_y_ws']:.1f}cm"
                dpg.draw_text([tgt_px + 14, tgt_py - 12], label_text,
                             color=green, size=10, parent=dl)
                
                # Draw connecting line from robot dot to target (if robot visible)
                if robot_in_ws:
                    dpg.draw_line([rpx, rpy], [tgt_px, tgt_py],
                                 color=(100, 255, 100, max(40, alpha // 3)),
                                 thickness=1, parent=dl)
                    
                    # Show distance between robot and target
                    dist_x = abs(self.sim_robot_x - t_belt_x)
                    dist_y_ws = abs(self.sim_robot_y - target['belt_y_ws'])
                    dist = (dist_x**2 + dist_y_ws**2)**0.5
                    mid_px = (rpx + tgt_px) // 2
                    mid_py = (rpy + tgt_py) // 2
                    dpg.draw_text([mid_px + 5, mid_py - 5], f"{dist:.1f}cm",
                                 color=(100, 255, 100, max(40, alpha // 2)),
                                 size=9, parent=dl)
            else:
                # Target expired
                self.sim_pick_target = None
    
    def _update_dashboard(self):
        """Update dashboard table with height column and warnings.

        Height colour coding:
          RED    — height < 1.0 cm  (sensor probably reading through transparent object)
          YELLOW — height 1.0-2.0 cm  (suspiciously low, might be inaccurate)
          GREEN  — height ≥ 2.0 cm  (normal / healthy reading)
        This lets the operator spot glass objects with near-zero height and
        intervene before the robot attempts a bad pick.
        """
        # Clear table
        dpg.delete_item("dash_table", children_only=True)
        # Rebuild columns
        dpg.add_table_column(label="ID", width=35, parent="dash_table")
        dpg.add_table_column(label="Class", width=55, parent="dash_table")
        dpg.add_table_column(label="H (cm)", width=50, parent="dash_table")
        dpg.add_table_column(label="X (cm)", width=45, parent="dash_table")
        dpg.add_table_column(label="Y (cm)", width=45, parent="dash_table")
        dpg.add_table_column(label="Status", width=60, parent="dash_table")
        # Add rows
        for obj_id, obj in list(self.tracker.objects.items())[:10]:  # Limit to 10 rows
            # Best available height: registered > stable > raw
            height = (obj.get('registered_height_cm')
                      or obj.get('stable_height_cm')
                      or obj.get('height_cm')
                      or obj.get('obj_height_cm', 0)) or 0

            # Colour-code height for operator awareness
            if height < 1.0:
                h_color = (255, 60, 60)      # RED — near-zero, likely bad read
                h_text = f"{height:.1f} !"
            elif height < 2.0:
                h_color = (255, 220, 50)     # YELLOW — suspiciously low
                h_text = f"{height:.1f} ?"
            else:
                h_color = (80, 220, 120)     # GREEN — normal
                h_text = f"{height:.1f}"

            with dpg.table_row(parent="dash_table"):
                dpg.add_text(str(obj_id))
                dpg.add_text(obj.get('class_name', '?'))
                dpg.add_text(h_text, color=h_color)
                dpg.add_text(f"{obj.get('belt_x_cm', 0):.1f}")
                dpg.add_text(f"{obj.get('belt_y_cm', 0):.1f}")
                dpg.add_text(obj.get('status', 'Tracking'))
        
        # Update measured speed indicator
        if dpg.does_item_exist("txt_measured_speed"):
            ms = self.tracker.measured_belt_speed
            n = len(self.tracker._speed_measurements)
            use_dynamic = (
                dpg.get_value("chk_dynamic_speed")
                if dpg.does_item_exist("chk_dynamic_speed")
                else True
            )
            if ms is not None:
                if use_dynamic:
                    dpg.set_value("txt_measured_speed", f"Measured: {ms:.2f} cm/s ({n} samples) [ACTIVE]")
                    dpg.configure_item("txt_measured_speed", color=(100, 220, 100))
                else:
                    dpg.set_value("txt_measured_speed", f"Measured: {ms:.2f} cm/s ({n} samples) [IGNORED]")
                    dpg.configure_item("txt_measured_speed", color=(200, 200, 100))
            else:
                dpg.set_value("txt_measured_speed", "Measured: --")
                dpg.configure_item("txt_measured_speed", color=(150, 150, 150))
    
    def run(self):
        """Main application loop."""
        frame_count = 0
        
        while dpg.is_dearpygui_running():
            if (self.auto_boot_on_launch and self._auto_boot_pending and
                    not self._auto_boot_done and frame_count >= self._auto_boot_frame_delay):
                self._auto_boot_pending = False
                self._run_auto_boot_sequence()

            if self.is_running:
                self._update_frame()
                self._maybe_auto_stop_finished_playback()
            else:
                # Update workspace simulation even when not running
                if frame_count % 5 == 0:
                    self._update_workspace_simulation()
                # Keep robot camera live preview active even when pipeline is stopped
                self._process_robot_camera_frame()
            
            # Update UI elements periodically
            frame_count += 1
            
            # Drain ui_queue every frame for responsive manual scan + robot results
            self._process_ui_queue()
            
            if frame_count % 5 == 0:  # Every 5 frames
                self._update_dashboard()
                self._update_stats()
                self._update_status_indicators()
                self._update_video_tab_status()
            if frame_count % 30 == 0:  # Every 30 frames
                self._update_log_ui()
                self._update_analytics_live()
            
            dpg.render_dearpygui_frame()
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        self.log("Shutting down...")
        
        self.is_running = False
        self._stop_benchmark_capture("application shutdown")

        # Stop any active recordings
        if self.bag_recording:
            try:
                self.camera.stop_recording()
            except Exception:
                pass
            self.bag_recording = False
        if self.mp4_recording:
            if self.mp4_writer:
                self.mp4_writer.release()
                self.mp4_writer = None
            self._export_mp4_sync_sidecar()
            self.mp4_recording = False
            self._mp4_record_start_epoch = None
            self._mp4_record_start_utc = None
            self._mp4_sync_rows = []
            self._mp4_source_t0 = None

        # Stop robot camera recording & release
        if self.robot_cam_writer is not None:
            self.robot_cam_writer.release()
            self.robot_cam_writer = None
        if self.robot_cam is not None:
            try:
                self.robot_cam.release()
            except Exception:
                pass
            self.robot_cam = None
            self.robot_cam_connected = False

        if self.video_source is not None:
            self.video_source.stop()
            self.video_source = None
        self.camera.stop()
        if self._live_camera is not None:
            self._live_camera.stop()
        self.robot_manager.stop()
        self.delta.disconnect()
        self.slider.disconnect()
        self.db.close()
        self.detection_logger.close()
        
        dpg.destroy_context()


if __name__ == "__main__":
    app = SortingApp()
    app.run()
