"""
=============================================================================
TRACKER MODULE
=============================================================================
Simple centroid-based tracker for conveyor belt objects.
Uses spatial proximity only - no color or height matching.
Implements smooth tracking with LAST KNOWN X position for picking.
"""

import numpy as np
import cv2
import time
from collections import deque
from .config import (
    CLASS_NAMES, MAX_TRACKING_DISTANCE_PX, MAX_DISAPPEARED_FRAMES, GHOST_GRACE_FRAMES,
    REGISTRATION_LINE_CM, ROI_HEIGHT_CM,
    EXIT_LINE_CM, SPEED_MEASUREMENT_WINDOW,
    ROBOT_WORKSPACE_OFFSET_CM, ROBOT_WORKSPACE_DEPTH_CM,
    ROBOT_PICK_CYCLE_TIME_S, ROBOT_MOVE_TIME_S,
    STACK_MASK_DILATE_PX, STACK_MASK_IOU_THRESHOLD, STACK_IOU_STACKING_MIN,
    STACK_HEIGHT_DIFF_MIN_CM,
    PICK_PRIORITY_WEIGHTS,
    MIN_PICK_WORKSPACE_Y_CM,
    PICK_CONSOLIDATION_DIST_CM, PICK_CONSOLIDATION_DIST_PX,
    QUEUE_EXIT_BUFFER_CM,
    TEMPORAL_TRACKING_ENABLED, TEMPORAL_MERGE_DIST_PX,
    TEMPORAL_MAX_MERGE_FRAMES, TEMPORAL_SPLIT_MATCH_DEPTH_MM,
    DUPLICATE_MASK_NMS_ENABLED, DUPLICATE_MASK_IOU_THRESHOLD,
    MASK_FADE_ENABLED, MASK_FADE_IN_RATE, MASK_FADE_OUT_RATE,
    MASK_FADE_MIN_ALPHA, MASK_FADE_MAX_ALPHA,
    MASK_TEMPORAL_SMOOTH, MASK_MORPH_SMOOTH_PX,
)


def _height_mode(hist):
    """Return the mode of height samples, rounded to 0.1 cm.

    Rounding clusters near-identical readings (e.g. 4.01, 3.99 → 4.0)
    so the most-frequent *true* height wins.  Outliers from mask bleed
    are outvoted and ignored entirely.
    """
    if not hist:
        return 0.0
    from collections import Counter
    rounded = [round(h, 1) for h in hist]
    (mode_val, _count), = Counter(rounded).most_common(1)
    return float(mode_val)


class SimpleTracker:
    """
    Simple centroid-based tracker for conveyor belt objects.
    Tracks objects using BELT COORDINATES (cm) for matching - more stable than pixels.
    
    Key features:
    - Matches objects in belt coordinate space (cm) for less flickering
    - Smooth centroid and bounding box tracking with EMA
    - Stores LAST KNOWN X from detection for accurate picking
    - Removes objects from display when they exit the belt
    """
    
    def __init__(self, max_distance=MAX_TRACKING_DISTANCE_PX, max_disappeared=MAX_DISAPPEARED_FRAMES):
        self.next_id = 1
        self.objects = {}  # id -> object dict
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.picking_queue = []  # List of object IDs ready for picking
        
        # Belt speed for position prediction (cm/s)
        self.belt_speed_cm_s = 5.5  # Default, updated from UI
        
        # === Vision-based belt speed measurement ===
        # Per-object transit time (reg→exit) provides measured speed samples.
        # Rolling median of recent samples replaces the manual slider value.
        self._speed_measurements = deque(maxlen=SPEED_MEASUREMENT_WINDOW)
        self.measured_belt_speed = None  # None until first measurement
        
        # Belt coordinate matching threshold (cm) - like Merge_v2.py
        self.max_belt_distance_cm = 10.0
        
        # Smoothing factors — DISABLED for display accuracy.
        # Raw detection positions are now used directly so that mask,
        # centroid, and bounding box are all from the same detection frame.
        # Previously 0.7/0.75 caused visible lag on moving belt objects.
        self.centroid_smoothing = 0.0  # 0 = no smoothing (raw detection)
        self.box_smoothing = 0.0       # 0 = no smoothing (raw detection)
        
        # --- Temporal merge/split tracking ---
        # Dormant identities: objects that disappeared near a surviving object.
        # They are held in case the merged mask splits again later.
        self.dormant_objects = {}
        
        # Depth signature per tracked object (rolling median depth in mm)
        self.depth_signatures = {}
        
        # Stats
        self.merge_events = 0
        self.split_events = 0
    
    def _smooth_value(self, old_val, new_val, alpha):
        """Apply exponential moving average smoothing."""
        if old_val is None:
            return new_val
        return old_val * alpha + new_val * (1 - alpha)
    
    def _smooth_position(self, old_pos, new_pos):
        """Apply smoothing to a position tuple."""
        if old_pos is None:
            return new_pos
        ox, oy = old_pos
        nx, ny = new_pos
        sx = self._smooth_value(ox, nx, self.centroid_smoothing)
        sy = self._smooth_value(oy, ny, self.centroid_smoothing)
        return (int(sx), int(sy))
    
    def _smooth_box(self, old_box, new_box):
        """Apply smoothing to min area box corners."""
        if old_box is None or new_box is None:
            return new_box
        if len(old_box) != len(new_box):
            return new_box
        
        smoothed = []
        for old_pt, new_pt in zip(old_box, new_box):
            sx = self._smooth_value(old_pt[0], new_pt[0], self.box_smoothing)
            sy = self._smooth_value(old_pt[1], new_pt[1], self.box_smoothing)
            smoothed.append([int(sx), int(sy)])
        return np.array(smoothed, dtype=np.int32)
    
    def update(self, detections, belt_y_calculator=None, x_to_cm_func=None):
        """
        Update tracker with new detections.
        
        IMPROVED: Uses proper distance-sorted greedy matching with exclusion sets
        to ensure each detection matches at most one tracked object and vice-versa.
        Includes temporal merge/split tracking to preserve IDs through mask merges.
        
        Args:
            detections: List of dicts with 'centroid', 'class_id', 'mask', etc.
            belt_y_calculator: Function to convert pixel Y to belt Y cm
            x_to_cm_func: Function to convert pixel X to belt X cm
        
        Returns:
            Updated objects dict
        """
        # Mark all as potentially disappeared
        for obj_id in self.objects:
            self.objects[obj_id]['detected_this_frame'] = False
        
        # =====================================================================
        # MATCHING: Build a cost matrix (distance) between every detection and
        # every existing tracked object, then assign greedily by shortest
        # distance while ensuring each tracked object is matched to at most ONE
        # detection and vice-versa.
        # =====================================================================
        matched_obj_ids = set()      # Tracked objects already claimed
        matched_det_indices = set()  # Detections already claimed
        
        # Build candidate pairs: (distance, det_index, obj_id)
        candidates = []
        for di, det in enumerate(detections):
            cx, cy = det['centroid']
            class_id = det['class_id']
            det_belt_x = det.get('belt_x_cm', 0)
            det_belt_y = det.get('belt_y_cm', 0)
            
            for obj_id, obj in self.objects.items():
                if obj['class_id'] != class_id:
                    continue
                
                # Use belt coordinates for matching (more stable than pixels)
                obj_belt_x = obj.get('belt_x_cm', 0)
                obj_belt_y = obj.get('belt_y_cm', 0)
                dist_cm = np.sqrt((det_belt_x - obj_belt_x)**2 + (det_belt_y - obj_belt_y)**2)
                
                # Ghost objects (undetected but predicted) get a larger
                # matching radius so they can re-acquire detection
                max_dist = self.max_belt_distance_cm
                if obj.get('ghost', False):
                    max_dist *= 2.0
                
                if dist_cm < max_dist:
                    candidates.append((dist_cm, di, obj_id))
        
        # Sort by distance — shortest first (greedy optimal)
        candidates.sort(key=lambda x: x[0])
        
        # Assign: each detection → at most one object, each object → at most one detection
        for dist, di, obj_id in candidates:
            if di in matched_det_indices or obj_id in matched_obj_ids:
                continue  # Already used
            
            # Apply the match
            det = detections[di]
            cx, cy = det['centroid']
            obj = self.objects[obj_id]
            
            # Smooth the centroid position for display
            old_centroid = obj.get('smoothed_centroid', obj['centroid'])
            smoothed_centroid = self._smooth_position(old_centroid, (cx, cy))
            obj['smoothed_centroid'] = smoothed_centroid
            obj['centroid'] = smoothed_centroid  # Use smoothed for display
            
            # Store raw detection centroid
            obj['raw_centroid'] = (cx, cy)
            
            # Smooth the bounding box
            old_box = obj.get('smoothed_box')
            new_box = det.get('min_area_box')
            obj['smoothed_box'] = self._smooth_box(old_box, new_box)
            obj['min_area_box'] = obj['smoothed_box']  # Use smoothed for display
            
            # Update other detection data — with temporal mask smoothing
            new_mask = det.get('mask')
            if new_mask is not None and MASK_TEMPORAL_SMOOTH > 0:
                prev_mask_f = obj.get('_mask_float')  # float32 EMA accumulator
                new_f = new_mask.astype(np.float32)
                if prev_mask_f is not None and prev_mask_f.shape == new_f.shape:
                    blended = prev_mask_f * MASK_TEMPORAL_SMOOTH + new_f * (1.0 - MASK_TEMPORAL_SMOOTH)
                else:
                    blended = new_f
                obj['_mask_float'] = blended
                # Threshold back to binary + morphological polish
                smooth_mask = (blended > 0.4).astype(np.uint8)
                if MASK_MORPH_SMOOTH_PX > 0:
                    k = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (MASK_MORPH_SMOOTH_PX * 2 + 1, MASK_MORPH_SMOOTH_PX * 2 + 1))
                    smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, k)
                    smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_OPEN, k)
                obj['mask'] = smooth_mask
            else:
                obj['mask'] = new_mask
            obj['height_cm'] = det.get('height_cm')
            obj['width_cm'] = det.get('width_cm')
            obj['obj_height_cm'] = det.get('obj_height_cm')
            obj['depth_mm'] = det.get('depth_mm', 0)
            obj['angle'] = det.get('angle', 0)
            obj['confidence'] = det.get('confidence', 0)
            obj['disappeared'] = 0
            obj['detected_this_frame'] = True
            obj['ghost'] = False

            # --- Height history: track per-frame height, compute robust mode ---
            raw_h = det.get('height_cm') or 0
            if raw_h > 0:
                hist = obj.setdefault('height_history', [])
                hist.append(raw_h)
                # Keep last 90 samples (~3 sec at 30fps) to avoid stale data
                if len(hist) > 90:
                    hist[:] = hist[-90:]
                obj['stable_height_cm'] = _height_mode(hist)

            # --- Mask alpha fade-in ---
            if MASK_FADE_ENABLED:
                cur_alpha = obj.get('mask_alpha', MASK_FADE_MAX_ALPHA)
                obj['mask_alpha'] = min(cur_alpha + MASK_FADE_IN_RATE, MASK_FADE_MAX_ALPHA)
            # Store last valid mask/boundary for ghost rendering
            if det.get('mask') is not None:
                obj['last_valid_mask'] = det['mask']
                obj['last_valid_mask_belt_y'] = obj.get('belt_y_cm', 0)
                obj['contour'] = det.get('contour')
            obj['watershed_boundary'] = det.get('watershed_boundary')
            
            # Update belt coordinates from detection —
            # BUT only for non-queued objects.  Once registered at the
            # reg line the pick position (X and Y) is locked.
            if not obj.get('in_queue', False):
                obj['belt_y_cm'] = det.get('belt_y_cm', belt_y_calculator(cy) if belt_y_calculator else 0)
                last_x = det.get('belt_x_cm', x_to_cm_func(cx) if x_to_cm_func else 0)
                obj['last_known_x_cm'] = last_x
                obj['belt_x_cm'] = last_x
            else:
                # Queued objects: keep camera-observed belt_y for exit crossing
                # detection (speed measurement), but don't touch the locked
                # belt_y_cm used for pick prediction.
                camera_y = det.get('belt_y_cm', belt_y_calculator(cy) if belt_y_calculator else 0)
                obj['camera_belt_y_cm'] = camera_y
            obj['last_update_time'] = time.time()
            
            matched_obj_ids.add(obj_id)
            matched_det_indices.add(di)
        
        # Collect unmatched detections
        unmatched_detections = [
            det for di, det in enumerate(detections) if di not in matched_det_indices
        ]
        
        # Register new objects from unmatched detections
        for det in unmatched_detections:
            cx, cy = det['centroid']
            
            belt_y = det.get('belt_y_cm', belt_y_calculator(cy) if belt_y_calculator else 0)
            belt_x = det.get('belt_x_cm', x_to_cm_func(cx) if x_to_cm_func else 0)
            
            # Don't register new objects that are already past registration line
            if belt_y > REGISTRATION_LINE_CM + 10:  # 10cm grace
                continue
            
            # --- TRACKER-LEVEL CROSS-CLASS DUPLICATE GUARD ---
            # Even after detector NMS, a duplicate might sneak through on a
            # subsequent frame.  Before registering, check if any EXISTING
            # tracked object has heavy mask overlap with this new detection
            # (regardless of class).  If so, skip registration.
            if DUPLICATE_MASK_NMS_ENABLED:
                new_mask = det.get('mask')
                is_dup = False
                if new_mask is not None and new_mask.any():
                    new_px = int(new_mask.sum())
                    for eid, eobj in self.objects.items():
                        emask = eobj.get('mask')
                        if emask is None or not emask.any():
                            continue
                        inter = int(cv2.bitwise_and(new_mask, emask).sum())
                        if inter == 0:
                            continue
                        union = int(cv2.bitwise_or(new_mask, emask).sum())
                        if union == 0:
                            continue
                        iou = inter / union
                        if iou >= DUPLICATE_MASK_IOU_THRESHOLD:
                            is_dup = True
                            # Optionally upgrade confidence if new is higher
                            if det.get('confidence', 0) > eobj.get('confidence', 0):
                                eobj['class_id'] = det['class_id']
                                eobj['class_name'] = CLASS_NAMES.get(det['class_id'], f"Class_{det['class_id']}")
                                eobj['confidence'] = det.get('confidence', 0)
                            break
                if is_dup:
                    continue
            
            init_height = det.get('height_cm', 0) or 0
            self.objects[self.next_id] = {
                'id': self.next_id,
                'centroid': (cx, cy),
                'smoothed_centroid': (cx, cy),
                'raw_centroid': (cx, cy),
                'class_id': det['class_id'],
                'class_name': CLASS_NAMES.get(det['class_id'], f"Class_{det['class_id']}"),
                'mask': det.get('mask'),
                'min_area_box': det.get('min_area_box'),
                'smoothed_box': det.get('min_area_box'),
                'height_cm': init_height,
                'width_cm': det.get('width_cm'),
                'obj_height_cm': det.get('obj_height_cm'),
                'depth_mm': det.get('depth_mm', 0),
                'height_history': [init_height] if init_height > 0 else [],
                'stable_height_cm': init_height,
                'angle': det.get('angle', 0),
                'confidence': det.get('confidence', 0),
                'belt_y_cm': belt_y,
                'belt_x_cm': belt_x,
                'last_known_x_cm': belt_x,
                'last_update_time': time.time(),
                'in_queue': False,
                'queue_position': -1,
                'stack_group': None,
                'status': 'Tracking',
                'disappeared': 0,
                'detected_this_frame': True,
                'ghost': False,
                'mask_alpha': MASK_FADE_MAX_ALPHA if not MASK_FADE_ENABLED else MASK_FADE_IN_RATE,
                'last_valid_mask': det.get('mask'),
                'last_valid_mask_belt_y': belt_y,
                'watershed_boundary': det.get('watershed_boundary'),
            }
            self.next_id += 1
        
        # Update disappeared count, mark ghosts, and remove old objects
        to_remove = []
        for obj_id, obj in self.objects.items():
            if not obj['detected_this_frame']:
                obj['disappeared'] += 1
                obj['ghost'] = True
                # --- Mask alpha fade-out ---
                if MASK_FADE_ENABLED:
                    cur_alpha = obj.get('mask_alpha', MASK_FADE_MAX_ALPHA)
                    obj['mask_alpha'] = max(cur_alpha - MASK_FADE_OUT_RATE, MASK_FADE_MIN_ALPHA)

                # --- Advance ghost belt_y for non-queued objects ---
                # The async ThreadedDetector can miss 1-2 frames. During that
                # gap the belt moves but the ghost's belt_y stays frozen.
                # When detection returns, the distance between the ghost's
                # stale belt_y and the new detection's belt_y can exceed the
                # matching threshold → treated as new object → duplicate.
                # Fix: predict belt_y forward so the ghost position stays in
                # sync with the belt, keeping it within matching range.
                if not obj.get('in_queue', False):
                    last_t = obj.get('last_update_time', 0)
                    if last_t > 0:
                        elapsed = time.time() - last_t
                        obj['belt_y_cm'] = obj.get('belt_y_cm', 0) + self.belt_speed_cm_s * elapsed
                        obj['last_update_time'] = time.time()

                # Only clear stale mask/box AFTER grace period.  The async
                # detector regularly misses 1-2 frames; clearing the mask
                # immediately destroys the IoU-based duplicate guard —
                # a new detection of the same object can't compute IoU
                # against a None mask, so it registers as a duplicate.
                if obj['disappeared'] > GHOST_GRACE_FRAMES:
                    obj['mask'] = None
                    obj['min_area_box'] = None
                    obj['smoothed_box'] = None
                    obj['contour'] = None
                # NOTE: We always keep 'last_valid_mask' for ghost rendering
                if obj['in_queue']:
                    # Only remove queued objects if past workspace end + buffer
                    if obj.get('belt_y_cm', 0) > (ROI_HEIGHT_CM + ROBOT_WORKSPACE_OFFSET_CM
                                                    + ROBOT_WORKSPACE_DEPTH_CM + QUEUE_EXIT_BUFFER_CM):
                        to_remove.append(obj_id)
                elif obj['disappeared'] > self.max_disappeared:
                    to_remove.append(obj_id)
            else:
                obj['ghost'] = False
        
        # === TEMPORAL MERGE/SPLIT DETECTION ===
        if TEMPORAL_TRACKING_ENABLED:
            self._detect_merge_events(to_remove)
            self._try_revive_dormant(unmatched_detections, belt_y_calculator, x_to_cm_func)
            self._age_dormant_objects()
        
        for obj_id in to_remove:
            if obj_id in self.picking_queue:
                self.picking_queue.remove(obj_id)
            # Remove from objects (dormant copy already saved if merge detected)
            if obj_id in self.objects:
                del self.objects[obj_id]
            self.depth_signatures.pop(obj_id, None)
        
        # Re-ID: compact IDs after removals so display stays clean (1, 2, 3...)
        if to_remove:
            self.reindex_ids()
        
        # Update depth signatures for all active objects
        for obj_id, obj in self.objects.items():
            if obj.get('detected_this_frame') and obj.get('depth_mm'):
                depth_val = obj['depth_mm']
                if depth_val > 0:
                    prev = self.depth_signatures.get(obj_id, depth_val)
                    self.depth_signatures[obj_id] = prev * 0.7 + depth_val * 0.3
        
        return self.objects
    
    # =========================================================================
    # TEMPORAL MERGE/SPLIT TRACKING
    # =========================================================================
    
    def _detect_merge_events(self, to_remove):
        """
        Detect when a disappearing object was likely merged into a nearby
        surviving detection. Create a dormant identity to hold its ID.
        """
        for obj_id in to_remove:
            obj = self.objects.get(obj_id)
            if obj is None:
                continue
            
            cx, cy = obj['centroid']
            class_id = obj['class_id']
            
            best_absorber = None
            best_dist = float('inf')
            
            for other_id, other_obj in self.objects.items():
                if other_id == obj_id:
                    continue
                if other_obj['class_id'] != class_id:
                    continue
                if not other_obj.get('detected_this_frame', False):
                    continue
                
                ox, oy = other_obj['centroid']
                dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                
                if dist < TEMPORAL_MERGE_DIST_PX and dist < best_dist:
                    best_dist = dist
                    best_absorber = other_id
            
            if best_absorber is not None:
                self.dormant_objects[obj_id] = {
                    'last_centroid': (cx, cy),
                    'class_id': class_id,
                    'depth_mm': self.depth_signatures.get(obj_id, 0),
                    'merged_into': best_absorber,
                    'frames_dormant': 0,
                    'belt_x_cm': obj.get('belt_x_cm', 0),
                    'belt_y_cm': obj.get('belt_y_cm', 0),
                    'height_cm': obj.get('height_cm', 0),
                    'height_history': list(obj.get('height_history', [])),
                    'stable_height_cm': obj.get('stable_height_cm', obj.get('height_cm', 0)),
                    'in_queue': obj.get('in_queue', False),
                    'queue_position': obj.get('queue_position', -1),
                    'reg_time': obj.get('reg_time'),
                    'reg_belt_y': obj.get('reg_belt_y'),
                }
                self.merge_events += 1
                print(f"[TEMPORAL] MERGE: #{obj_id} -> dormant (absorbed by #{best_absorber}, "
                      f"dist={best_dist:.0f}px)")
    
    def _try_revive_dormant(self, unmatched_detections, belt_y_calculator, x_to_cm_func):
        """
        When a new unmatched detection appears near where a dormant object
        was last seen (or near the absorber), try to revive the dormant identity.
        """
        if not self.dormant_objects or not unmatched_detections:
            return
        
        revived = []
        used_detections = set()
        
        for dormant_id, dorm in list(self.dormant_objects.items()):
            absorber_id = dorm['merged_into']
            absorber_obj = self.objects.get(absorber_id)
            
            if absorber_obj is None:
                continue
            
            ax, ay = absorber_obj['centroid']
            
            best_det_idx = None
            best_score = float('inf')
            
            for di, det in enumerate(unmatched_detections):
                if di in used_detections:
                    continue
                if det['class_id'] != dorm['class_id']:
                    continue
                
                dx, dy = det['centroid']
                dist_to_absorber = np.sqrt((dx - ax)**2 + (dy - ay)**2)
                if dist_to_absorber > TEMPORAL_MERGE_DIST_PX * 1.5:
                    continue
                
                det_depth = det.get('depth_mm', 0)
                dorm_depth = dorm['depth_mm']
                if det_depth > 0 and dorm_depth > 0:
                    depth_diff = abs(det_depth - dorm_depth)
                    if depth_diff > TEMPORAL_SPLIT_MATCH_DEPTH_MM:
                        continue
                    score = dist_to_absorber + depth_diff * 0.5
                else:
                    score = dist_to_absorber
                
                if score < best_score:
                    best_score = score
                    best_det_idx = di
            
            if best_det_idx is not None:
                det = unmatched_detections[best_det_idx]
                cx, cy = det['centroid']
                belt_y = belt_y_calculator(cy) if belt_y_calculator else 0
                belt_x = x_to_cm_func(cx) if x_to_cm_func else 0
                
                # Revive the dormant object with its ORIGINAL ID
                revive_h = det.get('height_cm') or 0
                # Carry over height history from dormant if available
                old_hist = dorm.get('height_history', [])
                new_hist = list(old_hist) + ([revive_h] if revive_h > 0 else [])
                self.objects[dormant_id] = {
                    'id': dormant_id,
                    'centroid': (cx, cy),
                    'smoothed_centroid': (cx, cy),
                    'raw_centroid': (cx, cy),
                    'class_id': det['class_id'],
                    'class_name': CLASS_NAMES.get(det['class_id'], f"Class_{det['class_id']}"),
                    'mask': det.get('mask'),
                    'min_area_box': det.get('min_area_box'),
                    'smoothed_box': det.get('min_area_box'),
                    'height_cm': revive_h,
                    'depth_mm': det.get('depth_mm', 0),
                    'height_history': new_hist,
                    'stable_height_cm': _height_mode(new_hist) if new_hist else revive_h,
                    'belt_y_cm': belt_y if not dorm['in_queue'] else dorm['belt_y_cm'],
                    'belt_x_cm': belt_x,
                    'last_known_x_cm': belt_x,
                    'last_update_time': time.time(),
                    'in_queue': dorm['in_queue'],
                    'queue_position': dorm['queue_position'],
                    'reg_time': dorm.get('reg_time'),
                    'reg_belt_y': dorm.get('reg_belt_y'),
                    'stack_group': None,
                    'status': 'Queued' if dorm['in_queue'] else 'Tracking',
                    'disappeared': 0,
                    'detected_this_frame': True,
                    'ghost': False,
                    'mask_alpha': MASK_FADE_MAX_ALPHA if not MASK_FADE_ENABLED else MASK_FADE_IN_RATE,
                    'last_valid_mask': det.get('mask'),
                    'last_valid_mask_belt_y': det.get('belt_y_cm', 0),
                    'watershed_boundary': det.get('watershed_boundary'),
                }
                
                if dorm['in_queue'] and dormant_id not in self.picking_queue:
                    self.picking_queue.append(dormant_id)
                
                used_detections.add(best_det_idx)
                revived.append(dormant_id)
                self.split_events += 1
                
                print(f"[TEMPORAL] SPLIT/REVIVE: #{dormant_id} revived "
                      f"(was merged into #{dorm['merged_into']}, "
                      f"dormant {dorm['frames_dormant']} frames)")
        
        for rid in revived:
            del self.dormant_objects[rid]
        
        for di in sorted(used_detections, reverse=True):
            unmatched_detections.pop(di)
    
    def _age_dormant_objects(self):
        """Age dormant objects and expire those that have been dormant too long."""
        expired = []
        for dormant_id, dorm in self.dormant_objects.items():
            dorm['frames_dormant'] += 1
            if dorm['frames_dormant'] > TEMPORAL_MAX_MERGE_FRAMES:
                expired.append(dormant_id)
        
        for did in expired:
            if self.dormant_objects[did].get('in_queue') and did in self.picking_queue:
                self.picking_queue.remove(did)
            print(f"[TEMPORAL] EXPIRED: #{did} dormant identity expired "
                  f"after {TEMPORAL_MAX_MERGE_FRAMES} frames")
            del self.dormant_objects[did]
            self.depth_signatures.pop(did, None)
    
    def estimate_reachability(self, obj, belt_speed_cm_s):
        """
        Estimate if the robot can reach this object before it exits the workspace.
        
        Calculates the time available vs time required:
        - Time available = (distance to workspace exit) / belt_speed
        - Time required = queue_wait_time + pick_cycle_time
        
        Args:
            obj: Object dict with belt position
            belt_speed_cm_s: Current belt speed
            
        Returns:
            tuple: (can_reach: bool, time_margin_s: float, reason: str)
        """
        if belt_speed_cm_s <= 0:
            belt_speed_cm_s = 6.0  # Safety default
        
        belt_y = obj.get('belt_y_cm', 0)
        
        # Calculate distance from current position to workspace exit
        workspace_entry = ROI_HEIGHT_CM + ROBOT_WORKSPACE_OFFSET_CM
        workspace_exit = workspace_entry + ROBOT_WORKSPACE_DEPTH_CM
        distance_to_exit = workspace_exit - belt_y
        
        if distance_to_exit <= 0:
            return False, 0, "Already past workspace"
        
        # Time until object exits workspace
        time_available = distance_to_exit / belt_speed_cm_s
        
        # Time required: count objects ahead in queue * cycle time + this pick
        objects_ahead = len(self.picking_queue)
        time_for_queue = objects_ahead * ROBOT_PICK_CYCLE_TIME_S
        time_for_this_pick = ROBOT_PICK_CYCLE_TIME_S + ROBOT_MOVE_TIME_S
        time_required = time_for_queue + time_for_this_pick
        
        # Calculate margin (positive = reachable, negative = will miss)
        time_margin = time_available - time_required
        
        can_reach = time_margin >= 0
        
        if not can_reach:
            reason = f"Need {time_required:.1f}s but only {time_available:.1f}s available"
        else:
            reason = f"OK: {time_margin:.1f}s margin"
        
        return can_reach, time_margin, reason
    
    def _detect_mask_stack_groups(self, obj_ids):
        """
        Detect stacking groups using mask IoU + depth-based stacking confirmation.
        
        IMPROVED approach (3 techniques combined):
        1. Mask dilation + IoU scoring  — quantifies HOW MUCH masks overlap
        2. Depth/height difference      — confirms PHYSICAL stacking vs side-by-side
        3. Union-Find grouping          — transitive closure (A↔B, B↔C → {A,B,C})
        
        Each pair gets:
          - iou_score:  intersection / union of dilated masks (0.0–1.0)
          - height_diff: absolute height difference in cm
          - stack_type:  'physical_stack' | 'adjacent' | 'none'
        
        Returns:
            list of lists — each sub-list sorted by height (tallest first).
        Also stores/updates self._pair_info for debug visualization.
        New pairs are added; existing pairs are updated — never wiped.
        """
        if not hasattr(self, '_pair_info'):
            self._pair_info = {}
        
        if len(obj_ids) <= 1:
            return [[oid] for oid in obj_ids]
        
        # Collect + dilate masks
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (STACK_MASK_DILATE_PX * 2 + 1, STACK_MASK_DILATE_PX * 2 + 1)
        )
        masks = {}
        for oid in obj_ids:
            obj = self.objects.get(oid)
            if obj is None:
                continue
            # Skip ghost objects — their masks were cleared when they went
            # undetected, so they can't participate in mask-IoU checks
            if obj.get('ghost', False):
                continue
            mask = obj.get('mask')
            if mask is not None and mask.any():
                masks[oid] = cv2.dilate(mask, kernel, iterations=1)
        
        # Union-Find
        parent = {oid: oid for oid in obj_ids}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        
        # Pairwise mask IoU + depth check
        oid_list = [oid for oid in obj_ids if oid in masks]
        for i in range(len(oid_list)):
            for j in range(i + 1, len(oid_list)):
                id_a, id_b = oid_list[i], oid_list[j]
                mask_a, mask_b = masks[id_a], masks[id_b]
                
                # Compute IoU
                intersection = cv2.bitwise_and(mask_a, mask_b)
                union_mask = cv2.bitwise_or(mask_a, mask_b)
                inter_pixels = np.count_nonzero(intersection)
                union_pixels = np.count_nonzero(union_mask)
                
                if union_pixels == 0:
                    continue
                
                iou_score = inter_pixels / union_pixels
                
                # Height difference (depth-based stacking confirmation)
                h_a = self.objects[id_a].get('stable_height_cm', self.objects[id_a].get('height_cm', 0)) or 0
                h_b = self.objects[id_b].get('stable_height_cm', self.objects[id_b].get('height_cm', 0)) or 0
                height_diff = abs(h_a - h_b)
                
                # Classify relationship using two-tier IoU:
                #   IoU >= 10%  → physical_stack (object inside/on top of another)
                #   IoU >= 2%   → adjacent (close but not overlapping much)
                #   IoU < 2%    → none (unrelated)
                if iou_score >= STACK_IOU_STACKING_MIN:
                    stack_type = 'physical_stack'  # One is INSIDE / ON TOP of the other
                    union(id_a, id_b)
                elif iou_score >= STACK_MASK_IOU_THRESHOLD:
                    stack_type = 'adjacent'  # Close but not stacked
                    union(id_a, id_b)
                else:
                    stack_type = 'none'
                
                # Store pair info for debug
                pair_key = (min(id_a, id_b), max(id_a, id_b))
                self._pair_info[pair_key] = {
                    'iou': iou_score,
                    'height_diff': height_diff,
                    'stack_type': stack_type,
                    'inter_pixels': inter_pixels,
                    'union_pixels': union_pixels,
                }
        
        # Collect groups
        groups_map = {}
        for oid in obj_ids:
            root = find(oid)
            groups_map.setdefault(root, []).append(oid)
        
        # Sort each group by height (tallest first = on top = pick first)
        result = []
        for group in groups_map.values():
            group.sort(
                key=lambda oid: self.objects[oid].get('stable_height_cm',
                    self.objects[oid].get('height_cm', 0)) or 0
                    if oid in self.objects else 0,
                reverse=True,
            )
            result.append(group)
        
        return result
    
    def _resolve_class_at_registration(self, crossing_ids):
        """
        IoU-based class resolution for objects about to be registered.

        When YOLO produces overlapping masks with different class labels for
        the same physical object, multiple tracked objects may exist.  This
        method compares every pair of crossing objects via mask IoU.  If the
        IoU exceeds the duplicate threshold the two represent the same
        physical object — only the **best** detection is kept (scored by a
        combination of mask coverage / object resolution AND confidence);
        the other is absorbed.

        The winner is chosen by a composite score:
            score = mask_area_px * 0.6  +  confidence * 0.4
        A larger mask means the detection captured more of the real object
        (better "object resolution"), so it's more likely the correct class.

        Returns:
            Filtered list of crossing object IDs with duplicates removed.
        """
        if len(crossing_ids) < 2:
            return list(crossing_ids)

        # --- Compute composite score for each crossing object ---
        # Normalise mask area across the candidate set so area and confidence
        # contribute equally.
        area_map = {}
        for oid in crossing_ids:
            m = self.objects[oid].get('mask')
            area_map[oid] = int(m.sum()) if m is not None else 0
        max_area = max(area_map.values()) or 1

        def _score(oid):
            obj = self.objects[oid]
            norm_area = area_map[oid] / max_area          # 0-1
            conf = obj.get('confidence', 0)               # 0-1
            return norm_area * 0.6 + conf * 0.4

        # Sort by composite score descending — best detection first
        sorted_ids = sorted(crossing_ids, key=_score, reverse=True)

        keep = set(sorted_ids)  # start with all, remove losers

        for i in range(len(sorted_ids)):
            oid_a = sorted_ids[i]
            if oid_a not in keep:
                continue
            obj_a = self.objects[oid_a]
            mask_a = obj_a.get('mask')
            if mask_a is None or not mask_a.any():
                continue

            for j in range(i + 1, len(sorted_ids)):
                oid_b = sorted_ids[j]
                if oid_b not in keep:
                    continue
                obj_b = self.objects[oid_b]
                mask_b = obj_b.get('mask')
                if mask_b is None or not mask_b.any():
                    continue

                inter = int(cv2.bitwise_and(mask_a, mask_b).sum())
                if inter == 0:
                    continue
                union = int(cv2.bitwise_or(mask_a, mask_b).sum())
                if union == 0:
                    continue
                iou = inter / union

                if iou >= DUPLICATE_MASK_IOU_THRESHOLD:
                    # oid_a wins (higher composite score) — absorb oid_b
                    cls_a = CLASS_NAMES.get(obj_a['class_id'], '?')
                    cls_b = CLASS_NAMES.get(obj_b['class_id'], '?')
                    score_a = _score(oid_a)
                    score_b = _score(oid_b)
                    w_a = obj_a.get('width_cm') or 0
                    h_a = obj_a.get('obj_height_cm') or 0
                    w_b = obj_b.get('width_cm') or 0
                    h_b = obj_b.get('obj_height_cm') or 0
                    print(f"[REG-IoU] #{oid_b} ({cls_b} {w_b:.1f}x{h_b:.1f}cm score={score_b:.2f}) "
                          f"overlaps #{oid_a} ({cls_a} {w_a:.1f}x{h_a:.1f}cm score={score_a:.2f}) "
                          f"IoU={iou:.2f} -> absorb #{oid_b}, keep #{oid_a}")
                    # Transfer height if the duplicate is taller
                    elev_a = obj_a.get('stable_height_cm', obj_a.get('height_cm', 0)) or 0
                    elev_b = obj_b.get('stable_height_cm', obj_b.get('height_cm', 0)) or 0
                    if elev_b > elev_a:
                        obj_a['height_cm'] = elev_b
                        obj_a['stable_height_cm'] = elev_b
                    obj_b['merged_into_pick'] = oid_a
                    obj_b['in_queue'] = False
                    obj_b['status'] = 'Absorbed'
                    keep.discard(oid_b)

        # Return in original order, filtered
        return [oid for oid in crossing_ids if oid in keep]

    def check_registration_crossing(self, registration_y_cm, frame=None, belt_speed_cm_s=None):
        """
        Check if any objects have crossed the registration line.
        Add to picking queue when they cross (registers ONCE per object).
        
        IMPROVED:
        - IoU class resolution: if the same object was detected under
          multiple classes, keep only the highest-confidence one.
        - Consolidation check (skip duplicates near existing queue entries)
        - Height-ordered insertion: within each cluster group, queue tallest first
        - Reachability check for robot timing
        
        Args:
            registration_y_cm: Y position of registration line in cm
            frame: Current color frame (unused, kept for API compatibility)
            belt_speed_cm_s: Current belt speed for reachability calculation
        """
        if belt_speed_cm_s is not None:
            self.belt_speed_cm_s = belt_speed_cm_s
        
        # --- Phase 1: Collect newly crossing objects ---
        newly_crossing = []
        for obj_id, obj in self.objects.items():
            if obj['in_queue'] or obj.get('status') == 'Unreachable':
                continue
            if obj.get('status') == 'Absorbed':
                continue
            belt_y = obj.get('belt_y_cm', 0)
            if belt_y >= registration_y_cm:
                newly_crossing.append(obj_id)
        
        if not newly_crossing:
            return
        
        # --- Phase 1b: IoU class resolution ---
        # If the same physical object was detected as multiple classes,
        # keep only the highest-confidence one.
        newly_crossing = self._resolve_class_at_registration(newly_crossing)
        
        if not newly_crossing:
            return
        
        # --- Phase 2: Detect stacking groups among crossing + nearby queued ---
        nearby_queued = []
        for obj_id in self.picking_queue:
            if obj_id not in self.objects:
                continue
            if self.objects[obj_id].get('belt_y_cm', 999) < registration_y_cm + 10:
                nearby_queued.append(obj_id)
        
        all_candidates = newly_crossing + nearby_queued
        stack_groups = self._detect_mask_stack_groups(all_candidates)
        
        group_for = {}
        for group in stack_groups:
            for oid in group:
                group_for[oid] = group
        
        # --- Phase 3: Queue with consolidation + height-sorted insertion ---
        # 3a: Filter out duplicates, collect valid candidates
        valid_crossing = []
        for obj_id in newly_crossing:
            obj = self.objects[obj_id]
            x_cm = obj.get('last_known_x_cm', obj.get('belt_x_cm', 0))
            belt_y = obj.get('belt_y_cm', 0)
            class_name = CLASS_NAMES.get(obj['class_id'], f"Class_{obj['class_id']}")
            
            # Reachability check
            can_reach, time_margin, reason = self.estimate_reachability(obj, self.belt_speed_cm_s)
            if not can_reach:
                obj['status'] = 'Unreachable'
                print(f"[SKIP] Object {obj_id} ({class_name}) - {reason}")
                continue
            
            # --- CONSOLIDATION CHECK ---
            # Compare against BOTH existing queue entries AND other newly
            # validated objects (same-frame duplicates).  Cross-class: we do
            # NOT require matching class_id because the same physical object
            # can be detected under different labels.
            duplicate_of = None
            
            # Check against existing queue
            for qid in self.picking_queue:
                if qid not in self.objects:
                    continue
                qobj = self.objects[qid]
                dx = abs(qobj.get('belt_x_cm', 0) - x_cm)
                dy = abs(qobj.get('belt_y_cm', 0) - belt_y)
                dist_cm = np.sqrt(dx**2 + dy**2)
                if dist_cm < PICK_CONSOLIDATION_DIST_CM:
                    duplicate_of = qid
                    break
                qcx, qcy = qobj.get('centroid', (0, 0))
                ocx, ocy = obj.get('centroid', (0, 0))
                dist_px = np.sqrt((qcx - ocx)**2 + (qcy - ocy)**2)
                if dist_px < PICK_CONSOLIDATION_DIST_PX:
                    duplicate_of = qid
                    break
            
            # Check against other objects validated THIS frame (same-frame dups)
            if duplicate_of is None:
                for vid in valid_crossing:
                    if vid not in self.objects:
                        continue
                    vobj = self.objects[vid]
                    vx = vobj.get('last_known_x_cm', vobj.get('belt_x_cm', 0))
                    vy = vobj.get('belt_y_cm', 0)
                    dx = abs(vx - x_cm)
                    dy = abs(vy - belt_y)
                    dist_cm = np.sqrt(dx**2 + dy**2)
                    if dist_cm < PICK_CONSOLIDATION_DIST_CM:
                        duplicate_of = vid
                        break
                    vcx, vcy = vobj.get('centroid', (0, 0))
                    ocx, ocy = obj.get('centroid', (0, 0))
                    dist_px = np.sqrt((vcx - ocx)**2 + (vcy - ocy)**2)
                    if dist_px < PICK_CONSOLIDATION_DIST_PX:
                        duplicate_of = vid
                        break
            
            if duplicate_of is not None:
                existing = self.objects[duplicate_of]
                # Keep the higher-confidence detection's class
                if obj.get('confidence', 0) > existing.get('confidence', 0):
                    existing['class_id'] = obj['class_id']
                    existing['class_name'] = CLASS_NAMES.get(obj['class_id'], f"Class_{obj['class_id']}")
                    existing['confidence'] = obj.get('confidence', 0)
                new_h = obj.get('stable_height_cm', obj.get('height_cm', 0)) or 0
                old_h = existing.get('stable_height_cm', existing.get('height_cm', 0)) or 0
                if new_h > old_h:
                    existing['height_cm'] = new_h
                    existing['stable_height_cm'] = new_h
                    # Merge height histories
                    h1 = existing.get('height_history', [])
                    h2 = obj.get('height_history', [])
                    merged = h1 + h2
                    if merged:
                        existing['height_history'] = merged[-90:]
                        existing['stable_height_cm'] = _height_mode(existing['height_history'])
                obj['in_queue'] = False
                obj['merged_into_pick'] = duplicate_of
                dup_cls = CLASS_NAMES.get(obj['class_id'], '?')
                ext_cls = CLASS_NAMES.get(existing['class_id'], '?')
                print(f"[CONSOLIDATE] #{obj_id} ({dup_cls}) too close to #{duplicate_of} ({ext_cls}) "
                      f"- absorbed, not queued")
                continue
            
            valid_crossing.append(obj_id)
        
        # 3b: Sort valid_crossing by HEIGHT within each cluster group (tallest first)
        seen_groups = set()
        ordered_crossing = []
        
        for obj_id in valid_crossing:
            grp = group_for.get(obj_id, [obj_id])
            grp_key = tuple(sorted(grp))
            if grp_key in seen_groups:
                continue
            seen_groups.add(grp_key)
            
            group_members = [oid for oid in grp if oid in valid_crossing]
            group_members.sort(
                key=lambda oid: self.objects[oid].get('stable_height_cm',
                    self.objects[oid].get('height_cm', 0)) or 0
                    if oid in self.objects else 0,
                reverse=True
            )
            ordered_crossing.extend(group_members)
        
        # Add any valid_crossing members not in a group (shouldn't happen, but safe)
        for oid in valid_crossing:
            if oid not in ordered_crossing:
                ordered_crossing.append(oid)
        
        # 3c: Queue in the height-sorted order
        pair_info = getattr(self, '_pair_info', {})
        for obj_id in ordered_crossing:
            obj = self.objects[obj_id]
            x_cm = obj.get('last_known_x_cm', obj.get('belt_x_cm', 0))
            belt_y = obj.get('belt_y_cm', 0)
            class_name = CLASS_NAMES.get(obj['class_id'], f"Class_{obj['class_id']}")
            
            grp = group_for.get(obj_id, [obj_id])
            obj['stack_group'] = grp
            
            # Determine this object's max IoU and stack type
            max_iou = 0.0
            best_stack_type = 'none'
            for other_id in grp:
                if other_id == obj_id:
                    continue
                pk = (min(obj_id, other_id), max(obj_id, other_id))
                pi = pair_info.get(pk)
                if pi and pi['iou'] > max_iou:
                    max_iou = pi['iou']
                    best_stack_type = pi['stack_type']
            
            obj['max_iou'] = max_iou
            obj['stack_type'] = best_stack_type
            
            obj['in_queue'] = True
            obj['status'] = 'Queued'
            obj['queue_position'] = len(self.picking_queue)
            
            # === LOCK POSITION & CLASS AT REGISTRATION ===
            # Position is frozen at the registration line — no more updates
            # from subsequent detections.  Y is predicted via time anchor.
            reg_y = obj.get('belt_y_cm', 0)
            reg_x = obj.get('last_known_x_cm', obj.get('belt_x_cm', 0))
            reg_time = time.time()
            obj['reg_time'] = reg_time
            obj['reg_belt_y'] = reg_y
            obj['reg_belt_x'] = reg_x
            # Lock class — the class assigned at registration is final
            obj['registered_class_id'] = obj['class_id']
            obj['registered_class_name'] = obj.get('class_name', CLASS_NAMES.get(obj['class_id'], '?'))
            # Lock height — median of all observations is the most reliable
            obj['registered_height_cm'] = obj.get('stable_height_cm',
                                                   obj.get('height_cm', 0)) or 0
            
            self.picking_queue.append(obj_id)
            
            if len(grp) > 1:
                grp_heights = [(oid, self.objects[oid].get('stable_height_cm',
                                    self.objects[oid].get('height_cm', 0)) or 0)
                               for oid in grp if oid in self.objects]
                grp_heights.sort(key=lambda x: x[1], reverse=True)
                rank = next((i for i, (oid, _) in enumerate(grp_heights) if oid == obj_id), 0)
                h_str = ", ".join(f"#{oid}={h:.1f}cm" for oid, h in grp_heights)
                print(f"[QUEUE] #{obj_id} {class_name} X={x_cm:.1f} Y={belt_y:.1f} "
                      f"IoU={max_iou:.3f} type={best_stack_type} "
                      f"CLUSTER rank {rank+1}/{len(grp_heights)} [{h_str}]")
            else:
                print(f"[QUEUE] #{obj_id} {class_name} X={x_cm:.1f} Y={belt_y:.1f}")
        
        # --- Phase 4: Update stack_group on nearby-queued objects too ---
        for obj_id in nearby_queued:
            if obj_id in self.objects:
                grp = group_for.get(obj_id, self.objects[obj_id].get('stack_group'))
                if grp:
                    self.objects[obj_id]['stack_group'] = grp
    
    def check_exit_crossing(self, exit_y_cm=None):
        """
        Check if any queued objects have crossed the exit line (ROI exit).
        
        This is the second timing checkpoint for vision-based speed measurement.
        When an object's camera-observed belt_y crosses the exit line:
        1. Record exit_time and exit_belt_y
        2. Compute per-object measured speed from (exit_y - reg_y) / (exit_time - reg_time)
        3. Add to rolling speed measurements for system-wide median
        4. Re-anchor the time prediction from exit point with measured speed
        
        This means even the FIRST object benefits: its own measured speed
        (from reg→exit transit) is used for the exit→workspace prediction.
        
        Args:
            exit_y_cm: Y position of exit line in cm (default: EXIT_LINE_CM)
        """
        if exit_y_cm is None:
            exit_y_cm = EXIT_LINE_CM
        
        for obj_id, obj in self.objects.items():
            if not obj.get('in_queue', False):
                continue
            if obj.get('exit_time') is not None:
                continue  # Already crossed exit
            
            # Use camera-observed position (updated in update() for queued objects)
            camera_y = obj.get('camera_belt_y_cm')
            if camera_y is None:
                continue  # Not detected this frame
            
            if camera_y >= exit_y_cm:
                exit_time = time.time()
                reg_time = obj.get('reg_time')
                reg_y = obj.get('reg_belt_y')
                
                if reg_time is None or reg_y is None:
                    continue
                
                transit_time = exit_time - reg_time
                transit_dist = camera_y - reg_y  # Use actual camera Y, not fixed exit_y
                
                if transit_time > 0.1:  # Sanity: at least 100ms transit
                    obj_speed = transit_dist / transit_time
                    
                    # Sanity check: speed should be reasonable (1-30 cm/s)
                    if 1.0 <= obj_speed <= 30.0:
                        # Store per-object measured speed
                        obj['exit_time'] = exit_time
                        obj['exit_belt_y'] = camera_y
                        obj['measured_speed'] = obj_speed
                        
                        # Add to rolling system-wide measurements
                        self._speed_measurements.append(obj_speed)
                        
                        # Update system median (robust to outliers)
                        self.measured_belt_speed = float(
                            np.median(list(self._speed_measurements)))
                        
                        class_name = CLASS_NAMES.get(obj['class_id'], '?')
                        print(f"[SPEED] #{obj_id} ({class_name}) "
                              f"reg_y={reg_y:.1f} → exit_y={camera_y:.1f}cm "
                              f"in {transit_time:.2f}s = {obj_speed:.2f} cm/s  "
                              f"(system median: {self.measured_belt_speed:.2f} cm/s)")
                    else:
                        print(f"[SPEED] #{obj_id} rejected: {obj_speed:.2f} cm/s "
                              f"(outside 1-30 range)")
    
    def get_effective_belt_speed(self):
        """
        Return the best available belt speed (cm/s).
        
        Priority: measured (vision-based median) > UI slider fallback.
        """
        if self.measured_belt_speed is not None:
            return self.measured_belt_speed
        return self.belt_speed_cm_s
    
    def _get_anchor_belt_y(self, obj, belt_speed_cm_s=None):
        """
        Get object's predicted belt Y using time-anchor with measured speed.
        
        PRIORITY (best anchor + best speed):
        1. EXIT ANCHOR + MEASURED SPEED: If the object has crossed the exit
           line, re-anchor from exit_y/exit_time with its own measured speed.
           This is the most accurate — short extrapolation from a late checkpoint
           with a speed derived from the object's own transit.
        2. REG ANCHOR + MEASURED SPEED: Object has reg anchor but hasn't crossed
           exit yet.  Use measured speed (per-object or system median) if available.
        3. REG ANCHOR + UI SLIDER: No measured speed yet — fall back to manual.
        4. FALLBACK: No anchor at all — incremental prediction from stored Y.
        
        Returns: predicted belt_y_cm (float)
        """
        slider_speed = belt_speed_cm_s if belt_speed_cm_s else self.belt_speed_cm_s
        now = time.time()
        
        # --- Determine best speed to use ---
        # Per-object measured speed > system median > UI slider
        obj_speed = obj.get('measured_speed')
        if obj_speed is not None:
            speed = obj_speed
        elif self.measured_belt_speed is not None:
            speed = self.measured_belt_speed
        else:
            speed = slider_speed
        
        # --- Determine best anchor to use ---
        exit_time = obj.get('exit_time')
        exit_y = obj.get('exit_belt_y')
        
        if exit_time is not None and exit_y is not None:
            # BEST: Re-anchor from exit point (shorter extrapolation distance)
            elapsed = now - exit_time
            return exit_y + (elapsed * speed)
        
        reg_time = obj.get('reg_time')
        reg_y = obj.get('reg_belt_y')
        
        if reg_time is not None and reg_y is not None:
            elapsed = now - reg_time
            return reg_y + (elapsed * speed)
        
        # FALLBACK: no anchor — incremental
        last_update = obj.get('last_update_time', now)
        time_since = now - last_update
        return obj.get('belt_y_cm', 0) + (time_since * speed)

    def advance_queued_objects(self, distance_cm, exit_limit_cm):
        """
        Update belt_y_cm for queued objects.
        
        PRIMARY: Uses time-anchor prediction (reg_y + elapsed * speed).
        FALLBACK: Incremental advance for objects without anchors.
        
        The stored belt_y_cm is kept in sync with the anchor prediction
        so that other code reading belt_y_cm directly still works.
        """
        to_remove = []
        current_time = time.time()
        
        for obj_id, obj in self.objects.items():
            if obj['in_queue']:
                # Use time-anchor prediction (drift-free)
                anchor_y = self._get_anchor_belt_y(obj)
                obj['belt_y_cm'] = anchor_y  # Keep stored Y in sync
                obj['last_update_time'] = current_time
            
            # Remove if past exit limit
            if obj.get('belt_y_cm', 0) > exit_limit_cm:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            if obj_id in self.picking_queue:
                self.picking_queue.remove(obj_id)
                print(f"[REMOVED] Object {obj_id} exited belt")
            if obj_id in self.objects:
                del self.objects[obj_id]
        
        # Update queue positions
        for i, obj_id in enumerate(self.picking_queue):
            if obj_id in self.objects:
                self.objects[obj_id]['queue_position'] = i
    
    def get_picking_data(self):
        """Get data for robot picking using LAST KNOWN X position."""
        picking_data = []
        for obj_id in self.picking_queue:
            if obj_id in self.objects:
                obj = self.objects[obj_id]
                # Use last_known_x_cm (last detected X before going invisible)
                x_cm = obj.get('last_known_x_cm', obj.get('belt_x_cm', 0))
                picking_data.append({
                    'id': obj_id,
                    'class_id': obj['class_id'],
                    'class_name': obj.get('class_name', CLASS_NAMES.get(obj['class_id'], "Unknown")),
                    'x_cm': x_cm,  # Last known X position
                    'y_cm': obj.get('belt_y_cm', 0),  # Current Y position
                    'height_cm': (obj.get('registered_height_cm')
                                   or obj.get('stable_height_cm')
                                   or obj.get('height_cm', 0)),
                    'angle': obj.get('angle', 0),
                    'status': obj.get('status', 'Queued')
                })
        return picking_data
    
    def get_priority_pick(self, belt_speed_cm_s=None):
        """
        Get the highest-urgency object to pick with REAL-TIME position.
        
        "Last-moment dispatch" strategy:
        - Objects in the picking queue are continuously tracked as they move
          through the workspace (their belt_y_cm is advanced every frame).
        - This method is only called when the robot is IDLE.
        - It picks the object that is CLOSEST TO EXITING the workspace
          (highest belt_y = highest urgency), because that object has the
          least time remaining.
        - The returned position is the object's CURRENT real-time position,
          which is already very accurate since we dispatch at the last moment.
        - execute_pick only needs to add a small approach-time offset (~0.5s).
        
        Args:
            belt_speed_cm_s: Current belt speed for position prediction
        
        Returns:
            dict with object data or None if no pickable objects
        """
        if belt_speed_cm_s is not None:
            self.belt_speed_cm_s = belt_speed_cm_s
        
        # Calculate workspace boundaries in belt coordinates
        workspace_entry_y = ROI_HEIGHT_CM + ROBOT_WORKSPACE_OFFSET_CM   # 42cm
        workspace_exit_y = workspace_entry_y + ROBOT_WORKSPACE_DEPTH_CM  # 62cm
        
        # Minimum belt Y before we consider picking
        min_pick_y = workspace_entry_y + MIN_PICK_WORKSPACE_Y_CM
        
        current_time = time.time()
        best_obj_id = None
        best_y = -1
        best_real_y = -1
        
        for obj_id in self.picking_queue:
            if obj_id not in self.objects:
                continue
            
            obj = self.objects[obj_id]
            
            # Skip if already being picked by robot
            if obj.get('status') == 'Picking':
                continue
            
            # Calculate real-time position based on time since last update
            last_update = obj.get('last_update_time', current_time)
            time_since_update = current_time - last_update
            position_advance = time_since_update * self.belt_speed_cm_s
            
            # Real-time belt Y position
            stored_y = obj.get('belt_y_cm', 0)
            real_belt_y = stored_y + position_advance
            
            # Skip objects that haven't reached the minimum pick zone yet
            # (too close to workspace entry — let them travel deeper)
            if real_belt_y < min_pick_y:
                continue
            
            # Skip objects that have already passed the workspace
            if real_belt_y > workspace_exit_y:
                continue
            
            # Highest Y = closest to exit = most urgent = highest priority
            if real_belt_y > best_y:
                best_y = real_belt_y
                best_real_y = real_belt_y
                best_obj_id = obj_id
        
        if best_obj_id is None:
            return None
        
        obj = self.objects[best_obj_id]
        x_cm = obj.get('last_known_x_cm', obj.get('belt_x_cm', 10))
        
        # Convert absolute belt Y to workspace-relative (0-20cm)
        workspace_y = best_real_y - workspace_entry_y
        workspace_y = max(0, min(ROBOT_WORKSPACE_DEPTH_CM, workspace_y))
        
        return {
            'id': best_obj_id,
            'class_id': obj['class_id'],
            'class_name': obj.get('class_name', CLASS_NAMES.get(obj['class_id'], "Unknown")),
            'belt_x': x_cm,
            'belt_y': workspace_y,  # Y position within workspace (0-20cm) RIGHT NOW
            'real_belt_y': best_real_y,  # Actual belt position for logging
            'height_cm': (obj.get('registered_height_cm')
                          or obj.get('stable_height_cm')
                          or obj.get('height_cm', 0)),
            'angle': obj.get('angle', 0),
            'confidence': obj.get('confidence', 0.9)
        }
    
    def get_smart_pick(self, belt_speed_cm_s=None):
        """
        Priority-scored pick selection using mask IoU + depth + urgency.
        
        Stacking vs Adjacent (IoU-based):
        - Stacking (IoU >= 10%): object inside/on top of another.
          HARD-BLOCK: must pick the taller (TOP) object first to prevent
          the top object from falling when the bottom is removed.
        - Adjacent (IoU >= 2% but < 10%): objects near each other but
          independent.  No hard-block — pick order based on priority score.
        
        Priority = w_urgency*urgency + w_height*height + w_isolation*isolation
                   - w_stack_risk*stack_penalty
        
        Keeps production features: workspace boundaries, real-time Y, reachability.
        
        Args:
            belt_speed_cm_s: Current belt speed for real-time position calculation
            
        Returns:
            dict with object data + priority scores, or None
        """
        if belt_speed_cm_s is not None:
            self.belt_speed_cm_s = belt_speed_cm_s
        
        W = PICK_PRIORITY_WEIGHTS
        workspace_entry_y = ROI_HEIGHT_CM + ROBOT_WORKSPACE_OFFSET_CM   # 42cm
        workspace_exit_y = workspace_entry_y + ROBOT_WORKSPACE_DEPTH_CM  # 62cm
        min_pick_y = workspace_entry_y + MIN_PICK_WORKSPACE_Y_CM
        
        current_time = time.time()
        
        # --- Step 1: Collect all pickable objects with real-time positions ---
        candidates = {}  # obj_id -> candidate dict
        for obj_id in self.picking_queue:
            if obj_id not in self.objects:
                continue
            obj = self.objects[obj_id]
            if obj.get('status') == 'Picking':
                continue
            
            # Real-time Y position (time-anchor based — drift-free)
            real_belt_y = self._get_anchor_belt_y(obj)
            
            # Must be in pickable workspace range
            if real_belt_y < min_pick_y or real_belt_y > workspace_exit_y:
                continue
            
            x_cm = obj.get('last_known_x_cm', obj.get('belt_x_cm', 10))
            # Use registered (frozen) height if available, else stable median,
            # else raw height — this prevents mask-bleed from corrupting pick Z
            height_cm = (obj.get('registered_height_cm')
                         or obj.get('stable_height_cm')
                         or obj.get('height_cm')
                         or obj.get('obj_height_cm', 0)) or 0
            
            candidates[obj_id] = {
                'id': obj_id,
                'x_cm': x_cm,
                'real_belt_y': real_belt_y,
                'height_cm': height_cm,
                'obj': obj,
            }
        
        if not candidates:
            return None
        
        # --- Step 2: Hard-block STACKED objects (tallest-first enforcement) ---
        # For every candidate in a stack_group with >= 2 members:
        #   If ANY member of the group has stack_type == 'physical_stack',
        #   the entire group enforces tallest-first picking.
        #   A shorter member is blocked if a taller member is still a candidate.
        #
        # This is self-contained — uses only data stored on each object at
        # registration time (stack_group, stack_type, registered_height_cm).
        # Does NOT depend on _pair_info which can be reset/stale.
        blocked_ids = set()
        
        # First, identify which groups contain at least one physical_stack member
        stacking_groups = set()  # frozenset of group tuples that have stacking
        for obj_id, cand in candidates.items():
            obj = cand['obj']
            group = obj.get('stack_group')
            if not group or len(group) <= 1:
                continue
            if obj.get('stack_type') == 'physical_stack':
                stacking_groups.add(frozenset(group))
        
        # Now enforce tallest-first within each stacking group
        for obj_id, cand in candidates.items():
            obj = cand['obj']
            group = obj.get('stack_group')
            if not group or len(group) <= 1:
                continue
            
            # Only enforce for groups that have physical stacking
            grp_key = frozenset(group)
            if grp_key not in stacking_groups:
                continue
            
            my_height = cand['height_cm']
            for other_id in group:
                if other_id == obj_id:
                    continue
                if other_id not in candidates:
                    continue  # Already picked / gone / not in workspace
                
                other_height = candidates[other_id]['height_cm']
                # Block if someone taller exists; on tie, lower ID goes first
                if other_height > my_height or (other_height == my_height and other_id < obj_id):
                    blocked_ids.add(obj_id)
                    print(f"[HARD-BLOCK] #{obj_id} (h={my_height:.1f}cm) blocked by "
                          f"#{other_id} (h={other_height:.1f}cm) — must pick taller first")
                    break
        
        # --- Step 3: Priority scoring for non-blocked candidates ---
        # Normalize urgency by max real_belt_y among candidates
        max_y = max((c['real_belt_y'] for c in candidates.values()), default=1.0)
        max_y = max(max_y, 1.0)
        
        scores = {}  # obj_id -> priority score
        for obj_id, cand in candidates.items():
            if obj_id in blocked_ids:
                continue
            obj = cand['obj']
            
            # Urgency: normalized real belt_y (higher = closer to exit = more urgent)
            urgency = cand['real_belt_y'] / max_y
            
            # Height bonus: taller objects slightly preferred
            height_bonus = min(cand['height_cm'] / 20.0, 1.0)
            
            # Isolation: 1.0 if standalone, less if in a group
            group = obj.get('stack_group', [obj_id])
            isolation = 1.0 if len(group) <= 1 else 1.0 / len(group)
            
            # Stack risk: high IoU = high risk of knocking adjacent objects
            stack_penalty = obj.get('max_iou', 0.0) or 0.0
            
            score = (W['urgency'] * urgency
                     + W['height'] * height_bonus
                     + W['isolation'] * isolation
                     - W['stack_risk'] * stack_penalty)
            
            scores[obj_id] = round(score, 3)
            obj['pick_priority'] = scores[obj_id]
        
        # --- Step 4: Pick highest-scoring candidate ---
        if not scores:
            return None
        
        best_id = max(scores, key=scores.get)
        best_cand = candidates[best_id]
        obj = best_cand['obj']
        
        workspace_y = best_cand['real_belt_y'] - workspace_entry_y
        workspace_y = max(0, min(ROBOT_WORKSPACE_DEPTH_CM, workspace_y))
        
        group = obj.get('stack_group', [best_id])
        is_stacked = len(group) > 1
        
        # Detect if this is a BOTTOM object in a physical stack.
        # If it was in a stacking group but all taller members are gone
        # (already picked), this is the bottom — it sits lower on the belt
        # now that the top has been removed.
        is_stack_bottom = False
        if is_stacked and obj.get('stack_type') == 'physical_stack':
            # Check if any taller group member is still a candidate
            my_h = best_cand['height_cm']
            has_taller = any(
                candidates[oid]['height_cm'] > my_h
                for oid in group if oid != best_id and oid in candidates
            )
            if not has_taller:
                # All taller members gone → I'm the bottom being picked
                is_stack_bottom = True
        
        return {
            'id': best_id,
            'class_id': obj.get('registered_class_id', obj['class_id']),
            'class_name': obj.get('registered_class_name', obj.get('class_name', CLASS_NAMES.get(obj['class_id'], "Unknown"))),
            'belt_x': obj.get('reg_belt_x', best_cand['x_cm']),
            'belt_y': workspace_y,
            'real_belt_y': best_cand['real_belt_y'],
            'height_cm': best_cand['height_cm'],
            'angle': obj.get('angle', 0),
            'confidence': obj.get('confidence', 0.9),
            'priority': scores[best_id],
            'is_stacked': is_stacked,
            'is_stack_bottom': is_stack_bottom,
            'stack_type': obj.get('stack_type', 'none'),
            'max_iou': obj.get('max_iou', 0.0),
            'stack_group': group if is_stacked else None,
            'blocked_ids': blocked_ids,
            'all_scores': scores,
        }

    def consolidate_queue(self):
        """
        Periodic consolidation — scan the entire picking queue and merge any
        two objects whose centroids are within PICK_CONSOLIDATION_DIST_CM of
        each other.
        
        Cross-class aware: does NOT require matching class_id, because the
        same physical object may be detected under different labels.
        Keeps the object that has been tracked longer (lower ID = earlier),
        and upgrades to the higher-confidence detection's class label.
        Should be called every frame from the main loop.
        """
        if len(self.picking_queue) < 2:
            return
        
        to_remove = set()
        queue_list = list(self.picking_queue)
        
        for i in range(len(queue_list)):
            oid_a = queue_list[i]
            if oid_a in to_remove or oid_a not in self.objects:
                continue
            obj_a = self.objects[oid_a]
            
            for j in range(i + 1, len(queue_list)):
                oid_b = queue_list[j]
                if oid_b in to_remove or oid_b not in self.objects:
                    continue
                obj_b = self.objects[oid_b]
                
                dx = abs(obj_a.get('belt_x_cm', 0) - obj_b.get('belt_x_cm', 0))
                dy = abs(obj_a.get('belt_y_cm', 0) - obj_b.get('belt_y_cm', 0))
                dist_cm = np.sqrt(dx**2 + dy**2)
                
                if dist_cm < PICK_CONSOLIDATION_DIST_CM:
                    keeper, discard = (oid_a, oid_b) if oid_a < oid_b else (oid_b, oid_a)
                    to_remove.add(discard)
                    
                    k_obj = self.objects[keeper]
                    d_obj = self.objects[discard]
                    
                    # Keep higher-confidence class label
                    if (d_obj.get('confidence', 0) > k_obj.get('confidence', 0)):
                        k_obj['class_id'] = d_obj['class_id']
                        k_obj['class_name'] = CLASS_NAMES.get(
                            d_obj['class_id'], f"Class_{d_obj['class_id']}")
                        k_obj['confidence'] = d_obj.get('confidence', 0)
                    
                    new_h = d_obj.get('stable_height_cm', d_obj.get('height_cm', 0)) or 0
                    old_h = k_obj.get('stable_height_cm', k_obj.get('height_cm', 0)) or 0
                    if new_h > old_h:
                        k_obj['height_cm'] = new_h
                        k_obj['stable_height_cm'] = new_h
                    
                    d_obj['in_queue'] = False
                    d_obj['merged_into_pick'] = keeper
                    
                    k_cls = CLASS_NAMES.get(k_obj['class_id'], '?')
                    d_cls = CLASS_NAMES.get(d_obj['class_id'], '?')
                    print(f"[CONSOLIDATE] Queue cleanup: #{discard} ({d_cls}) "
                          f"merged into #{keeper} ({k_cls}) (dist={dist_cm:.1f}cm)")
        
        if to_remove:
            self.picking_queue = [oid for oid in self.picking_queue if oid not in to_remove]
            for oid in to_remove:
                self.objects.pop(oid, None)
                self.depth_signatures.pop(oid, None)
            for qi, oid in enumerate(self.picking_queue):
                if oid in self.objects:
                    self.objects[oid]['queue_position'] = qi
            self.reindex_ids()
    
    def reindex_ids(self):
        """
        Compact/renumber all tracked object IDs to sequential 1, 2, 3, ...
        
        After consolidation removes duplicates, IDs can have gaps (e.g. #1, #4, #7).
        This method renumbers everything so the display shows clean sequential IDs.
        
        Updates all internal references:
        - self.objects dict keys + each obj['id']
        - self.picking_queue
        - self.dormant_objects (merged_into references)
        - self.depth_signatures
        - _pair_info keys
        - stack_group lists inside each object
        """
        if not self.objects:
            self.next_id = 1
            return
        
        sorted_old_ids = sorted(self.objects.keys())
        id_map = {}
        for new_id, old_id in enumerate(sorted_old_ids, start=1):
            id_map[old_id] = new_id
        
        # Check if already sequential — skip if nothing to do
        if all(old == new for old, new in id_map.items()):
            self.next_id = len(self.objects) + 1
            return
        
        # Rebuild self.objects with new IDs
        new_objects = {}
        for old_id, new_id in id_map.items():
            obj = self.objects[old_id]
            obj['id'] = new_id
            
            grp = obj.get('stack_group')
            if grp:
                obj['stack_group'] = [id_map.get(g, g) for g in grp]
            
            mip = obj.get('merged_into_pick')
            if mip and mip in id_map:
                obj['merged_into_pick'] = id_map[mip]
            
            new_objects[new_id] = obj
        self.objects = new_objects
        
        # Rebuild picking_queue
        self.picking_queue = [id_map[oid] for oid in self.picking_queue if oid in id_map]
        
        # Rebuild depth_signatures
        new_sigs = {}
        for old_id, sig in self.depth_signatures.items():
            if old_id in id_map:
                new_sigs[id_map[old_id]] = sig
        self.depth_signatures = new_sigs
        
        # Rebuild dormant_objects
        new_dormant = {}
        for old_id, dorm in self.dormant_objects.items():
            new_id = id_map.get(old_id, old_id)
            mi = dorm.get('merged_into')
            if mi and mi in id_map:
                dorm['merged_into'] = id_map[mi]
            new_dormant[new_id] = dorm
        self.dormant_objects = new_dormant
        
        # Rebuild _pair_info
        old_pair_info = getattr(self, '_pair_info', {})
        if old_pair_info:
            new_pair_info = {}
            for (id_a, id_b), info in old_pair_info.items():
                na = id_map.get(id_a, id_a)
                nb = id_map.get(id_b, id_b)
                new_pair_info[(min(na, nb), max(na, nb))] = info
            self._pair_info = new_pair_info
        
        self.next_id = len(self.objects) + 1

    def get_realtime_object_position(self, obj_id, belt_speed_cm_s=None):
        """
        Get an object's real-time position for tracking during approach.
        
        Called repeatedly by execute_pick during the tracking approach phase.
        Returns the object's current belt coordinates or None if object is gone.
        
        Args:
            obj_id: Object ID to track
            belt_speed_cm_s: Current belt speed
            
        Returns:
            dict {belt_x, belt_y_cm, ws_y, height_cm, class_name} or None
        """
        if obj_id not in self.objects:
            return None
        
        obj = self.objects[obj_id]
        speed = belt_speed_cm_s if belt_speed_cm_s else self.belt_speed_cm_s
        
        # Time-anchor based prediction (drift-free)
        real_belt_y = self._get_anchor_belt_y(obj, belt_speed_cm_s=speed)
        
        ws_entry = ROI_HEIGHT_CM + ROBOT_WORKSPACE_OFFSET_CM
        ws_y = real_belt_y - ws_entry
        ws_y = max(0, min(ROBOT_WORKSPACE_DEPTH_CM, ws_y))
        
        belt_x = obj.get('last_known_x_cm', obj.get('belt_x_cm', 10.0))
        
        return {
            'belt_x': belt_x,
            'belt_y_cm': real_belt_y,
            'ws_y': ws_y,
            'height_cm': (obj.get('registered_height_cm')
                          or obj.get('stable_height_cm')
                          or obj.get('height_cm')
                          or obj.get('obj_height_cm', 0)),
            'class_name': obj.get('class_name', 'Unknown'),
            'angle': obj.get('angle', 0),
        }

    def get_next_pick(self):
        """Get the next object to pick (first in queue) with locked registration data."""
        if self.picking_queue and self.picking_queue[0] in self.objects:
            obj = self.objects[self.picking_queue[0]].copy()
            # Use registration-locked X position
            if 'reg_belt_x' in obj:
                obj['belt_x_cm'] = obj['reg_belt_x']
                obj['last_known_x_cm'] = obj['reg_belt_x']
            elif 'last_known_x_cm' in obj:
                obj['belt_x_cm'] = obj['last_known_x_cm']
            # Use registration-locked class
            if 'registered_class_id' in obj:
                obj['class_id'] = obj['registered_class_id']
                obj['class_name'] = obj.get('registered_class_name', obj.get('class_name', 'Unknown'))
            return obj
        return None
    
    def mark_picked(self, obj_id):
        """Mark an object as picked and remove from queue."""
        if obj_id in self.objects:
            self.objects[obj_id]['status'] = 'Picked'
        if obj_id in self.picking_queue:
            self.picking_queue.remove(obj_id)
        # Update queue positions
        for i, oid in enumerate(self.picking_queue):
            if oid in self.objects:
                self.objects[oid]['queue_position'] = i
