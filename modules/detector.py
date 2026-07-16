"""
=============================================================================
DETECTOR MODULE
=============================================================================
YOLO-based object detection with segmentation and height calculation.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import threading
import queue
import time
import json
import os
import math

from .config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, CALIBRATION_FILE,
    IMAGE_WIDTH, IMAGE_HEIGHT, ROI_HEIGHT_CM, ROI_WIDTH_CM,
    ENTRY_PATH_CM, EXIT_PATH_CM, CLASS_NAMES, CLASS_COLORS, DEFAULT_COLOR,
    ROI_MIN_MASK_COVERAGE,
    EXTENDED_DETECTION_ZONE, EXIT_ZONE_MIN_MASK_COVERAGE,
    DUPLICATE_MASK_NMS_ENABLED, DUPLICATE_MASK_IOU_THRESHOLD,
    DEPTH_CLUSTER_ENABLED, DEPTH_CLUSTER_MIN_MASK_PX,
    DEPTH_CLUSTER_DEPTH_GAP_MM, DEPTH_CLUSTER_MIN_CLUSTER_PX,
    DEPTH_CLUSTER_MAX_SPLITS, DEPTH_CLUSTER_HIST_BINS,
    DEPTH_CLUSTER_MORPH_OPEN_PX, DEPTH_CLUSTER_SPATIAL_CONNECT,
    WATERSHED_JOIN_ENABLED, WATERSHED_MAX_GAP_PX,
    WATERSHED_DEPTH_TOLERANCE_MM, WATERSHED_MIN_FRAGMENT_PX,
    WATERSHED_BOUNDARY_ALPHA,
    WATERSHED_REQUIRE_SAME_CLASS, WATERSHED_COLOR_SIM_ENABLED,
    WATERSHED_COLOR_SIM_THRESHOLD, WATERSHED_ASPECT_RATIO_TOL,
    WATERSHED_DEBUG,
)


def get_orientation_pca(mask):
    """Calculate object orientation using PCA on mask."""
    y, x = np.nonzero(mask)
    if len(x) < 50:
        return 0.0
    pts = np.array([x, y], dtype=np.float64).transpose()
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
    vec_primary = eigenvectors[0]
    angle = math.atan2(vec_primary[1], vec_primary[0]) * 180 / math.pi
    return angle


# =============================================================================
# CROSS-CLASS MASK NMS — Remove duplicate detections of the same object
# =============================================================================

def cross_class_mask_nms(detections, iou_threshold=DUPLICATE_MASK_IOU_THRESHOLD):
    """
    Suppress duplicate detections that describe the same physical object
    but were assigned different class labels by YOLO.

    Standard YOLO NMS only works within each class. This function performs
    mask-IoU NMS *across all classes* so a bottle detected as both "Glass"
    and "Metal" is reduced to a single detection (the higher-confidence one).

    Algorithm:
      1. Sort detections by confidence (highest first).
      2. For each kept detection, compute mask IoU with every remaining
         candidate.  If IoU >= threshold → suppress the candidate.

    Args:
        detections: list of detection dicts (must contain 'mask' and 'confidence')
        iou_threshold: IoU above which two masks are considered the same object

    Returns:
        Filtered list of detections with cross-class duplicates removed.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending (keep best first)
    indexed = sorted(enumerate(detections), key=lambda x: x[1].get('confidence', 0), reverse=True)

    keep_flags = [True] * len(detections)

    for rank_i in range(len(indexed)):
        orig_i, det_i = indexed[rank_i]
        if not keep_flags[orig_i]:
            continue
        mask_i = det_i.get('mask')
        if mask_i is None:
            continue

        for rank_j in range(rank_i + 1, len(indexed)):
            orig_j, det_j = indexed[rank_j]
            if not keep_flags[orig_j]:
                continue
            mask_j = det_j.get('mask')
            if mask_j is None:
                continue

            # Compute mask IoU
            inter = np.count_nonzero(cv2.bitwise_and(mask_i, mask_j))
            if inter == 0:
                continue
            union = np.count_nonzero(cv2.bitwise_or(mask_i, mask_j))
            if union == 0:
                continue
            iou = inter / union

            if iou >= iou_threshold:
                # Suppress the lower-confidence detection
                keep_flags[orig_j] = False

    return [det for det, keep in zip(detections, keep_flags) if keep]


# =============================================================================
# DEPTH CLUSTERING — Split merged YOLO masks via depth discontinuities
# =============================================================================

def depth_cluster_mask(mask_binary, depth_image_mm,
                       depth_gap_mm=DEPTH_CLUSTER_DEPTH_GAP_MM,
                       min_cluster_px=DEPTH_CLUSTER_MIN_CLUSTER_PX,
                       max_splits=DEPTH_CLUSTER_MAX_SPLITS,
                       hist_bins=DEPTH_CLUSTER_HIST_BINS,
                       morph_open_px=DEPTH_CLUSTER_MORPH_OPEN_PX,
                       spatial_connect=DEPTH_CLUSTER_SPATIAL_CONNECT):
    """
    Attempt to split a single YOLO mask into multiple sub-masks using depth
    histogram valley detection.

    Strategy:
    1. Extract valid depth values inside the mask
    2. Build a depth histogram and smooth it
    3. Find peaks in the histogram — each peak = a candidate object
    4. Find valleys between peaks — if valley is deep enough, split there
    5. Assign each pixel to its nearest peak's depth range
    6. Optionally enforce spatial connectivity (connected components)
    7. Return list of sub-masks (each is a binary mask)

    Args:
        mask_binary: np.uint8 HxW binary mask (1 = object)
        depth_image_mm: np.uint16 HxW depth in millimeters
        depth_gap_mm: Minimum depth gap between peaks to justify a split
        min_cluster_px: Minimum pixel count for a valid sub-cluster
        max_splits: Maximum number of sub-objects to produce
        hist_bins: Number of histogram bins
        morph_open_px: Morphological opening kernel size
        spatial_connect: Whether to enforce spatial connectivity

    Returns:
        list of (sub_mask, median_depth_mm) tuples.
        If no split is warranted, returns [(original_mask, median_depth)].
    """
    # Extract depth values inside the mask
    mask_pixels = mask_binary > 0
    total_px = int(mask_pixels.sum())

    if total_px < DEPTH_CLUSTER_MIN_MASK_PX:
        valid_d = depth_image_mm[mask_pixels]
        valid_d = valid_d[(valid_d > 100) & (valid_d < 2000)]
        med_d = float(np.median(valid_d)) if len(valid_d) > 0 else 0
        return [(mask_binary, med_d)]

    depths_in_mask = depth_image_mm[mask_pixels].astype(np.float32)

    # Filter invalid depths
    valid_mask_local = (depths_in_mask > 100) & (depths_in_mask < 2000)
    valid_depths = depths_in_mask[valid_mask_local]

    if len(valid_depths) < DEPTH_CLUSTER_MIN_MASK_PX:
        med_d = float(np.median(valid_depths)) if len(valid_depths) > 0 else 0
        return [(mask_binary, med_d)]

    # --- Step 1: Build depth histogram ---
    d_min, d_max = float(valid_depths.min()), float(valid_depths.max())
    depth_span = d_max - d_min

    # If depth span is tiny, no point splitting
    if depth_span < depth_gap_mm * 1.5:
        med_d = float(np.median(valid_depths))
        return [(mask_binary, med_d)]

    # Histogram with adaptive bin count
    actual_bins = min(hist_bins, max(10, int(depth_span / 2)))
    hist, bin_edges = np.histogram(valid_depths, bins=actual_bins, range=(d_min, d_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth histogram (Gaussian-like smoothing)
    kernel_size = max(3, actual_bins // 8)
    if kernel_size % 2 == 0:
        kernel_size += 1
    hist_smooth = np.convolve(hist, np.ones(kernel_size) / kernel_size, mode='same')

    # --- Step 2: Find peaks (local maxima) ---
    peaks = []
    for i in range(1, len(hist_smooth) - 1):
        if hist_smooth[i] > hist_smooth[i - 1] and hist_smooth[i] > hist_smooth[i + 1]:
            if hist_smooth[i] > total_px * 0.02:
                peaks.append(i)

    if len(peaks) == 0:
        peaks = [int(np.argmax(hist_smooth))]

    # If only 1 peak, try to check if distribution is bimodal
    if len(peaks) == 1:
        peak_idx = peaks[0]
        peak_val = hist_smooth[peak_idx]
        threshold = peak_val * 0.2
        for i in range(1, len(hist_smooth) - 1):
            if i == peak_idx:
                continue
            if hist_smooth[i] > threshold:
                if hist_smooth[i] > hist_smooth[i - 1] and hist_smooth[i] > hist_smooth[i + 1]:
                    if abs(bin_centers[i] - bin_centers[peak_idx]) > depth_gap_mm:
                        peaks.append(i)

    peaks.sort(key=lambda i: bin_centers[i])

    if len(peaks) < 2:
        med_d = float(np.median(valid_depths))
        return [(mask_binary, med_d)]

    # Limit to max_splits + 1 peaks
    if len(peaks) > max_splits:
        peaks.sort(key=lambda i: hist_smooth[i], reverse=True)
        peaks = peaks[:max_splits]
        peaks.sort(key=lambda i: bin_centers[i])

    # --- Step 3: Find valleys (split points) between consecutive peaks ---
    valleys = []
    for k in range(len(peaks) - 1):
        p1, p2 = peaks[k], peaks[k + 1]
        peak_gap = bin_centers[p2] - bin_centers[p1]
        if peak_gap < depth_gap_mm:
            continue

        region = hist_smooth[p1:p2 + 1]
        valley_local_idx = int(np.argmin(region))
        valley_idx = p1 + valley_local_idx

        min_peak = min(hist_smooth[p1], hist_smooth[p2])
        valley_val = hist_smooth[valley_idx]

        if valley_val < min_peak * 0.6:
            valleys.append(valley_idx)

    if len(valleys) == 0:
        med_d = float(np.median(valid_depths))
        return [(mask_binary, med_d)]

    # --- Step 4: Assign pixels to clusters based on depth ranges ---
    split_depths = [d_min] + [bin_centers[v] for v in valleys] + [d_max + 1]

    h, w = mask_binary.shape
    full_depth = depth_image_mm.astype(np.float32)

    sub_masks = []
    for si in range(len(split_depths) - 1):
        lo = split_depths[si]
        hi = split_depths[si + 1]

        depth_band = ((full_depth >= lo) & (full_depth < hi) & mask_pixels).astype(np.uint8)

        if morph_open_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (morph_open_px * 2 + 1, morph_open_px * 2 + 1))
            depth_band = cv2.morphologyEx(depth_band, cv2.MORPH_OPEN, kernel)

        if spatial_connect:
            num_labels, labels, stats, centroids_cc = cv2.connectedComponentsWithStats(
                depth_band, connectivity=8)
            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area >= min_cluster_px:
                    component_mask = (labels == lbl).astype(np.uint8)
                    comp_depths = full_depth[component_mask > 0]
                    comp_depths = comp_depths[(comp_depths > 100) & (comp_depths < 2000)]
                    med_d = float(np.median(comp_depths)) if len(comp_depths) > 0 else 0
                    sub_masks.append((component_mask, med_d))
        else:
            px_count = int(depth_band.sum())
            if px_count >= min_cluster_px:
                band_depths = full_depth[depth_band > 0]
                band_depths = band_depths[(band_depths > 100) & (band_depths < 2000)]
                med_d = float(np.median(band_depths)) if len(band_depths) > 0 else 0
                sub_masks.append((depth_band, med_d))

    # --- Step 5: Validate results ---
    if len(sub_masks) < 2:
        med_d = float(np.median(valid_depths))
        return [(mask_binary, med_d)]

    if len(sub_masks) > max_splits:
        sub_masks.sort(key=lambda x: int(x[0].sum()), reverse=True)
        sub_masks = sub_masks[:max_splits]

    return sub_masks


# =============================================================================
# WATERSHED MASK JOINING — rejoin fragments of the same object
# =============================================================================

def watershed_join_masks(detections, depth_image_mm, color_image,
                         max_gap_px=WATERSHED_MAX_GAP_PX,
                         depth_tol_mm=WATERSHED_DEPTH_TOLERANCE_MM,
                         min_frag_px=WATERSHED_MIN_FRAGMENT_PX):
    """
    Attempt to join detection fragments that likely belong to the same
    physical object but were split by an occluding object on top.

    Strategy:
      1. For each pair of detections, check:
         a) Spatially close (dilated masks overlap within max_gap_px)
         b) Similar depth (within depth_tol_mm)
         c) Optionally: same class (disabled by default — YOLO often assigns
            different classes to split fragments)
         d) Optionally: colour histogram similarity — fragments from the same
            object share similar colour distribution regardless of class label
         e) Optionally: aspect-ratio compatibility — reject if one fragment is
            vastly different shape from the other
      2. If a join candidate passes all enabled gates, use OpenCV watershed
         on the union bounding region to merge the fragments.
      3. The watershed boundary pixels are stored as 'watershed_boundary'
         on the merged detection so they can be drawn as a transparent grid.
      4. Centroid is recalculated on the joined mask.

    Returns:
        Modified detections list (some entries merged, boundary info added).
    """
    if len(detections) < 2:
        return detections

    n = len(detections)
    if WATERSHED_DEBUG:
        cls_summary = {}
        for d in detections:
            cid = d.get('class_id', -1)
            cls_summary[CLASS_NAMES.get(cid, str(cid))] = cls_summary.get(CLASS_NAMES.get(cid, str(cid)), 0) + 1
        print(f"[WATERSHED] Evaluating {n} detections: {cls_summary}")
    merged_into = list(range(n))  # Union-find parent

    def find_root(i):
        while merged_into[i] != i:
            merged_into[i] = merged_into[merged_into[i]]
            i = merged_into[i]
        return i

    # --- helper: colour histogram similarity between two masked regions ------
    def _colour_similarity(mask_a, mask_b, img_bgr):
        """Return correlation score (0–1) between colour histograms of two masked regions."""
        hists = []
        for m in (mask_a, mask_b):
            # Convert mask to uint8 255 for calcHist
            m8 = (m * 255).astype(np.uint8) if m.max() <= 1 else m
            # Hue-Saturation histogram in HSV space (invariant to brightness)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], m8, [30, 32], [0, 180, 0, 256])
            cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
            hists.append(h)
        corr = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)
        return max(0.0, corr)  # clamp to 0-1

    # --- helper: aspect-ratio compatibility ----------------------------------
    def _aspect_ratio(mask):
        """Return aspect ratio (>=1) of the minimum-area bounding rect."""
        pts = cv2.findNonZero(mask)
        if pts is None or len(pts) < 5:
            return 1.0
        rect = cv2.minAreaRect(pts)
        w, h = rect[1]
        if w == 0 or h == 0:
            return 1.0
        return max(w, h) / min(w, h)

    # Precompute dilated masks for proximity check
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (max_gap_px, max_gap_px))
    dilated = []
    for det in detections:
        m = det.get('mask')
        if m is not None and m.sum() >= min_frag_px:
            dilated.append(cv2.dilate(m, kernel, iterations=1))
        else:
            dilated.append(None)

    # Find join candidates
    for i in range(n):
        if dilated[i] is None:
            continue
        di = detections[i]
        depth_i = di.get('depth_mm', 0)
        cls_i = di.get('class_id', -1)
        name_i = CLASS_NAMES.get(cls_i, str(cls_i))

        for j in range(i + 1, n):
            if dilated[j] is None:
                continue
            dj = detections[j]
            cls_j = dj.get('class_id', -1)
            name_j = CLASS_NAMES.get(cls_j, str(cls_j))
            depth_j = dj.get('depth_mm', 0)

            # Gate 1: same class (optional — OFF by default)
            if WATERSHED_REQUIRE_SAME_CLASS and cls_i != cls_j:
                if WATERSHED_DEBUG:
                    print(f"  [WS] SKIP ({name_i} #{i} <-> {name_j} #{j}): class mismatch (same-class required)")
                continue

            # Gate 2: similar depth
            d_diff = abs(depth_i - depth_j)
            if d_diff > depth_tol_mm:
                if WATERSHED_DEBUG:
                    print(f"  [WS] SKIP ({name_i} #{i} <-> {name_j} #{j}): depth gap {d_diff:.0f}mm > {depth_tol_mm}mm")
                continue

            # Gate 3: spatial proximity (dilated masks overlap)
            overlap = cv2.bitwise_and(dilated[i], dilated[j]).sum()
            if overlap == 0:
                if WATERSHED_DEBUG:
                    print(f"  [WS] SKIP ({name_i} #{i} <-> {name_j} #{j}): no spatial overlap (gap>{max_gap_px}px)")
                continue

            # Gate 4: colour histogram similarity (optional)
            if WATERSHED_COLOR_SIM_ENABLED:
                m_i = di.get('mask')
                m_j = dj.get('mask')
                if m_i is not None and m_j is not None:
                    sim = _colour_similarity(m_i, m_j, color_image)
                    if sim < WATERSHED_COLOR_SIM_THRESHOLD:
                        if WATERSHED_DEBUG:
                            print(f"  [WS] SKIP ({name_i} #{i} <-> {name_j} #{j}): colour sim {sim:.2f} < {WATERSHED_COLOR_SIM_THRESHOLD}")
                        continue
                    if WATERSHED_DEBUG:
                        print(f"  [WS] PASS colour ({name_i} #{i} <-> {name_j} #{j}): sim={sim:.2f}")

            # Gate 5: aspect-ratio compatibility (optional)
            if WATERSHED_ASPECT_RATIO_TOL > 0:
                ar_i = _aspect_ratio(di.get('mask'))
                ar_j = _aspect_ratio(dj.get('mask'))
                ratio = max(ar_i, ar_j) / max(min(ar_i, ar_j), 0.01)
                if ratio > WATERSHED_ASPECT_RATIO_TOL:
                    if WATERSHED_DEBUG:
                        print(f"  [WS] SKIP ({name_i} #{i} <-> {name_j} #{j}): aspect ratio {ratio:.1f} > {WATERSHED_ASPECT_RATIO_TOL}")
                    continue

            # All gates passed → mark for merge
            if WATERSHED_DEBUG:
                print(f"  [WS] JOIN ({name_i} #{i} <-> {name_j} #{j}): depth_gap={d_diff:.0f}mm, overlap={overlap}")
            ri, rj = find_root(i), find_root(j)
            if ri != rj:
                merged_into[rj] = ri

    # Group detections by root
    groups = {}
    for i in range(n):
        r = find_root(i)
        groups.setdefault(r, []).append(i)

    join_count = sum(1 for g in groups.values() if len(g) > 1)
    if WATERSHED_DEBUG and join_count > 0:
        print(f"[WATERSHED] {join_count} group(s) to merge out of {n} detections")

    # Build output: merge groups with >1 member via watershed
    result = []
    used = set()

    for root, members in groups.items():
        if len(members) == 1:
            result.append(detections[members[0]])
            used.add(members[0])
            continue

        # --- Watershed join ---
        # Combine masks into a union mask
        combined_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        best_det = None
        best_conf = -1
        total_area = 0
        for idx in members:
            m = detections[idx].get('mask')
            if m is not None:
                combined_mask = cv2.bitwise_or(combined_mask, m)
                total_area += int(m.sum())
            if detections[idx].get('confidence', 0) > best_conf:
                best_conf = detections[idx].get('confidence', 0)
                best_det = detections[idx]
            used.add(idx)

        if best_det is None or combined_mask.sum() < min_frag_px:
            for idx in members:
                result.append(detections[idx])
            continue

        # Create markers for watershed: each original fragment gets a label
        markers = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.int32)
        for label_id, idx in enumerate(members, start=1):
            m = detections[idx].get('mask')
            if m is not None:
                # Erode slightly so markers are safely inside each fragment
                erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded = cv2.erode(m, erode_k, iterations=1)
                if eroded.sum() > 10:
                    markers[eroded > 0] = label_id
                else:
                    markers[m > 0] = label_id

        # The gap between fragments is unknown territory (marker = 0)
        # Dilate the combined mask to create the watershed region
        ws_region_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (max_gap_px, max_gap_px))
        ws_region = cv2.dilate(combined_mask, ws_region_kernel, iterations=1)

        # Prepare 3-channel image for watershed (use actual color image)
        ws_img = color_image.copy()
        # Mask out areas outside the watershed region so watershed only fills gaps
        ws_img[ws_region == 0] = 0

        # Run watershed
        try:
            cv2.watershed(ws_img, markers)
        except cv2.error:
            # Watershed failed — keep originals
            for idx in members:
                result.append(detections[idx])
            continue

        # Build joined mask: all positive marker regions within ws_region
        joined_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        joined_mask[(markers > 0) & (ws_region > 0)] = 1

        # Watershed boundaries (marker == -1)
        boundary_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        boundary_mask[(markers == -1) & (ws_region > 0)] = 1

        # Recalculate centroid on the joined mask
        M = cv2.moments(joined_mask)
        if M["m00"] <= 0:
            for idx in members:
                result.append(detections[idx])
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Recalculate contour
        contours, _ = cv2.findContours(joined_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            for idx in members:
                result.append(detections[idx])
            continue
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box_pts = cv2.boxPoints(rect).astype(np.int32)

        # Recalculate angle
        angle = get_orientation_pca(joined_mask)

        # Average depth
        joined_depths = depth_image_mm[joined_mask > 0]
        valid_d = joined_depths[(joined_depths > 100) & (joined_depths < 2000)]
        avg_depth = float(np.mean(valid_d)) if len(valid_d) > 0 else best_det.get('depth_mm', 0)

        # Build merged detection (inherit from best-confidence original)
        merged = dict(best_det)
        merged['mask'] = joined_mask
        merged['contour'] = largest
        merged['min_area_box'] = box_pts
        merged['centroid'] = (cx, cy)
        merged['angle'] = angle
        merged['depth_mm'] = avg_depth
        merged['watershed_boundary'] = boundary_mask
        merged['is_watershed_joined'] = True
        merged['joined_count'] = len(members)

        if WATERSHED_DEBUG:
            member_names = [CLASS_NAMES.get(detections[idx].get('class_id', -1), '?') for idx in members]
            print(f"[WATERSHED] Merged {len(members)} fragments -> {CLASS_NAMES.get(merged['class_id'], '?')} "
                  f"(members: {member_names}, centroid=({cx},{cy}), depth={avg_depth:.0f}mm)")

        result.append(merged)

    return result


class ObjectDetector:
    """
    YOLO-based object detector with depth and calibration support.
    Provides detections with belt coordinates, height, and orientation.
    """
    
    def __init__(self, model_path=MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        
        # Calibration data
        self.roi_corners = None
        self.entry_corners = None
        self.exit_corners = None
        self.homography = None
        self.homography_inv = None
        self.floor_depth_map = None
        self.floor_plane = None
        self.floor_depth = None  # Simple floor depth fallback
        
        # Camera intrinsics (set from camera)
        self.intrinsics = None
        
        # Belt coordinate helpers
        self.roi_width_cm = ROI_WIDTH_CM
        self.roi_height_cm = ROI_HEIGHT_CM
        self.entry_path_cm = ENTRY_PATH_CM
        self.exit_path_cm = EXIT_PATH_CM
        self.x_min_px = 0
        self.x_max_px = IMAGE_WIDTH
        self.entry_start_y_px = 0
        self.exit_end_y_px = IMAGE_HEIGHT
        self.px_per_cm = 10.0
        self.x_direction = 1  # 1 = normal, -1 = reversed
        
        # ROI mask cache for coverage filtering
        self._roi_mask = None
        self._exit_zone_mask = None       # Exit zone only mask
        self._extended_roi_mask = None    # Combined ROI + exit zone mask
        
        # Separation logic toggles (live UI overrides)
        self.use_depth_cluster = DEPTH_CLUSTER_ENABLED
        self.use_cross_nms     = DUPLICATE_MASK_NMS_ENABLED
        self.use_watershed     = WATERSHED_JOIN_ENABLED

        # Display options
        self.show_masks = True
        self.show_edges = True
        self.show_height = True
        self.show_x_marks = True
        self.mask_alpha = 0.3
        self.x_mark_size = 12
    
    def load_model(self):
        """Load YOLO model. Passes task='segment' for .engine files."""
        try:
            print(f"[DETECTOR] Loading YOLO model: {self.model_path}")
            # TensorRT .engine files cannot auto-detect task — must specify 'segment'
            if self.model_path.endswith('.engine'):
                self.model = YOLO(self.model_path, task='segment')
            else:
                self.model = YOLO(self.model_path)
            # Warmup
            self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            fmt = 'TensorRT' if self.model_path.endswith('.engine') else 'PyTorch'
            print(f"[DETECTOR] Model loaded successfully ({fmt})")
            return True
        except Exception as e:
            print(f"[DETECTOR] Failed to load model: {e}")
            return False
    
    def switch_model(self, new_model_path):
        """
        Hot-swap the YOLO model to a different weights file.
        Returns True on success, False on failure (keeps old model).
        Passes task='segment' for .engine (TensorRT) files.
        """
        old_path = self.model_path
        old_model = self.model
        try:
            print(f"[DETECTOR] Switching model: {old_path} -> {new_model_path}")
            self.model_path = new_model_path
            # TensorRT .engine files cannot auto-detect task — must specify 'segment'
            if new_model_path.endswith('.engine'):
                self.model = YOLO(new_model_path, task='segment')
            else:
                self.model = YOLO(new_model_path)
            self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            fmt = 'TensorRT' if new_model_path.endswith('.engine') else 'PyTorch'
            print(f"[DETECTOR] Model switched successfully to {os.path.basename(new_model_path)} ({fmt})")
            return True
        except Exception as e:
            print(f"[DETECTOR] Failed to switch model: {e}  - reverting")
            self.model_path = old_path
            self.model = old_model
            return False
    
    def load_calibration(self, filepath=CALIBRATION_FILE):
        """Load calibration data from JSON file (format from calibration_tool.py)."""
        if not os.path.exists(filepath):
            print(f"[DETECTOR] Calibration file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load ROI corners (calibration_tool.py format: roi.corners_px)
            if 'roi' in data and 'corners_px' in data['roi']:
                corners = data['roi']['corners_px']
                self.roi_corners = np.array(corners, dtype=np.float32)
                print(f"[DETECTOR] Loaded ROI corners from roi.corners_px")
            elif 'roi_corners_pixel' in data:
                self.roi_corners = np.array(data['roi_corners_pixel'], dtype=np.float32).reshape(4, 2)
            
            # Load entry/exit zone corners (calibration_tool.py format: zones.entry_corners_px)
            if 'zones' in data:
                zones = data['zones']
                if 'entry_corners_px' in zones and zones['entry_corners_px']:
                    self.entry_corners = np.array(zones['entry_corners_px'], dtype=np.float32)
                    print(f"[DETECTOR] Loaded entry zone corners")
                if 'exit_corners_px' in zones and zones['exit_corners_px']:
                    self.exit_corners = np.array(zones['exit_corners_px'], dtype=np.float32)
                    print(f"[DETECTOR] Loaded exit zone corners")
                if 'entry_path_cm' in zones:
                    self.entry_path_cm = zones['entry_path_cm']
                if 'exit_path_cm' in zones:
                    self.exit_path_cm = zones['exit_path_cm']
            
            # Load ROI dimensions
            if 'roi' in data:
                if 'width_cm' in data['roi']:
                    self.roi_width_cm = data['roi']['width_cm']
                if 'height_cm' in data['roi']:
                    self.roi_height_cm = data['roi']['height_cm']
            
            # Load floor depth map (calibration_tool.py format: floor_depth_map with downsampling)
            if 'floor_depth_map' in data and data['floor_depth_map'] is not None:
                fdm = data['floor_depth_map']
                if isinstance(fdm, dict) and 'data' in fdm:
                    # Downsampled format from calibration_tool.py
                    downsampled = np.array(fdm['data'], dtype=np.float32)
                    factor = fdm.get('downsample_factor', 4)
                    orig_shape = fdm.get('original_shape', [IMAGE_HEIGHT, IMAGE_WIDTH])
                    # Upscale to original size
                    self.floor_depth_map = cv2.resize(downsampled, (orig_shape[1], orig_shape[0]), 
                                                      interpolation=cv2.INTER_LINEAR)
                    print(f"[DETECTOR] Loaded floor depth map (upscaled from {downsampled.shape})")
                else:
                    self.floor_depth_map = np.array(fdm, dtype=np.float32)
                    print(f"[DETECTOR] Loaded floor depth map")
            
            # Load floor plane coefficients
            if 'floor_plane' in data and data['floor_plane'].get('coefficients'):
                self.floor_plane = tuple(data['floor_plane']['coefficients'])
                print(f"[DETECTOR] Loaded floor plane coefficients")
            
            # Load homography (calibration_tool.py format: transforms.homography)
            if 'transforms' in data and data['transforms'].get('homography'):
                self.homography = np.array(data['transforms']['homography'], dtype=np.float32)
                if data['transforms'].get('homography_inv'):
                    self.homography_inv = np.array(data['transforms']['homography_inv'], dtype=np.float32)
                else:
                    self.homography_inv = np.linalg.inv(self.homography)
                print(f"[DETECTOR] Loaded homography transform")
            elif 'homography' in data:
                self.homography = np.array(data['homography'], dtype=np.float32)
                self.homography_inv = np.linalg.inv(self.homography)
            
            # Calculate belt coordinate helpers from ROI corners
            if self.roi_corners is not None:
                roi_tl, roi_tr = self.roi_corners[0], self.roi_corners[1]
                roi_br, roi_bl = self.roi_corners[2], self.roi_corners[3]
                
                # Get X bounds from all corners (handles any orientation)
                all_x = [roi_tl[0], roi_tr[0], roi_br[0], roi_bl[0]]
                self.x_min_px = int(min(all_x))
                self.x_max_px = int(max(all_x))
                
                # Y bounds
                self.entry_start_y_px = float(min(roi_tl[1], roi_tr[1]))
                self.exit_end_y_px = float(max(roi_bl[1], roi_br[1]))
                
                # Calculate px_per_cm from ROI height
                roi_height_px = np.linalg.norm(np.array(roi_bl) - np.array(roi_tl))
                self.px_per_cm = roi_height_px / self.roi_height_cm
                
                # Store which direction X increases (for correct mapping)
                # If TL.x > TR.x, X increases from right to left in image
                self.x_direction = 1 if roi_tl[0] < roi_tr[0] else -1
                
                print(f"[DETECTOR] ROI X range: {self.x_min_px} to {self.x_max_px} px")
                print(f"[DETECTOR] X direction: {'normal' if self.x_direction == 1 else 'reversed'}")
                
                # If entry/exit corners not loaded, calculate them
                if self.entry_corners is None:
                    entry_height_px = self.entry_path_cm * self.px_per_cm
                    left_dir = (roi_bl - roi_tl) / np.linalg.norm(roi_bl - roi_tl)
                    right_dir = (roi_br - roi_tr) / np.linalg.norm(roi_br - roi_tr)
                    self.entry_corners = np.array([
                        roi_tl - left_dir * entry_height_px,
                        roi_tr - right_dir * entry_height_px,
                        roi_tr,
                        roi_tl
                    ], dtype=np.float32)
                
                if self.exit_corners is None:
                    exit_height_px = self.exit_path_cm * self.px_per_cm
                    left_dir = (roi_bl - roi_tl) / np.linalg.norm(roi_bl - roi_tl)
                    right_dir = (roi_br - roi_tr) / np.linalg.norm(roi_br - roi_tr)
                    self.exit_corners = np.array([
                        roi_bl,
                        roi_br,
                        roi_br + right_dir * exit_height_px,
                        roi_bl + left_dir * exit_height_px
                    ], dtype=np.float32)
                
                print(f"[DETECTOR] Calibration loaded: ROI {self.roi_width_cm}x{self.roi_height_cm}cm, "
                      f"X=[{self.x_min_px}, {self.x_max_px}], px/cm={self.px_per_cm:.1f}")
            
            # Build ROI binary mask for coverage filtering
            if self.roi_corners is not None:
                self._roi_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
                cv2.fillPoly(self._roi_mask, [self.roi_corners.astype(np.int32)], 1)
                print(f"[DETECTOR] Built ROI mask for coverage filtering")
                
                # Build extended detection mask (ROI + exit zone)
                if EXTENDED_DETECTION_ZONE and self.exit_corners is not None:
                    self._exit_zone_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
                    cv2.fillPoly(self._exit_zone_mask, [self.exit_corners.astype(np.int32)], 1)
                    
                    self._extended_roi_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
                    cv2.fillPoly(self._extended_roi_mask, [self.roi_corners.astype(np.int32)], 1)
                    cv2.fillPoly(self._extended_roi_mask, [self.exit_corners.astype(np.int32)], 1)
                    print(f"[DETECTOR] Built extended detection mask (ROI + exit zone)")
                else:
                    self._exit_zone_mask = None
                    self._extended_roi_mask = None
            
            return True
        except Exception as e:
            print(f"[DETECTOR] Error loading calibration: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_intrinsics(self, intrinsics):
        """Set camera intrinsics for depth calculations."""
        self.intrinsics = intrinsics
    
    def calibrate_floor(self, depth_frame):
        """Simple floor calibration - get average depth of center region."""
        depth_image = np.asanyarray(depth_frame.get_data())
        h, w = depth_image.shape
        center_region = depth_image[h//3:2*h//3, w//3:2*w//3]
        valid = (center_region > 0) & (center_region < 10000)
        if np.sum(valid) > 100:
            self.floor_depth = np.median(center_region[valid]) * 0.001
            print(f"[DETECTOR] Floor calibrated at {self.floor_depth:.3f}m")
            return True
        return False
    
    def pixel_y_to_belt_y_cm(self, pixel_y):
        """Convert pixel Y coordinate to belt Y coordinate in cm."""
        if self.roi_corners is None:
            return 0
        roi_top_y = self.entry_start_y_px
        belt_y_cm = (pixel_y - roi_top_y) / self.px_per_cm
        return belt_y_cm
    
    def pixel_to_belt_cm(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to belt coordinates using homography.
        
        Returns: (belt_x_cm, belt_y_cm)
        belt_x: 0 = left side of belt, 20 = right side
        belt_y: 0 = top (entry), 30 = bottom (exit)
        """
        # Use homography directly (pixel -> belt), NOT homography_inv (belt -> pixel)
        if self.homography is not None:
            pixel_pt = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
            belt_coords = cv2.perspectiveTransform(pixel_pt, self.homography)[0][0]
            # No flip - belt_x maps directly to robot grid
            # belt_x=0 (left) -> robot left positions, belt_x=20 (right) -> robot right positions
            belt_x = belt_coords[0]
            belt_y = belt_coords[1]
            belt_x = max(0, min(self.roi_width_cm, belt_x))
            return belt_x, belt_y
        
        # Fallback to linear
        return self.pixel_x_to_belt_x_cm(pixel_x), self.pixel_y_to_belt_y_cm(pixel_y)

    def pixel_x_to_belt_x_cm(self, pixel_x, pixel_y=None):
        """
        Convert pixel X coordinate to belt X coordinate in cm.
        If pixel_y is provided and homography is available, use perspective transform.
        """
        if pixel_y is not None and self.homography is not None:
            belt_x, _ = self.pixel_to_belt_cm(pixel_x, pixel_y)
            return belt_x
        
        # Fallback: simple linear interpolation
        if self.roi_corners is None:
            return 10.0  # Default to center
        
        x_range = max(1, self.x_max_px - self.x_min_px)
        x_ratio = (pixel_x - self.x_min_px) / x_range
        
        # If X is reversed in image (TL.x > TR.x), flip the ratio
        if hasattr(self, 'x_direction') and self.x_direction == -1:
            x_ratio = 1.0 - x_ratio
        
        x_cm = x_ratio * self.roi_width_cm
        x_cm = max(0, min(self.roi_width_cm, x_cm))
        return x_cm
    
    def belt_cm_to_pixel(self, belt_x_cm, belt_y_cm):
        """
        Convert belt coordinates (cm) back to pixel coordinates.
        Uses homography_inv (belt → pixel).

        Args:
            belt_x_cm: X position on belt in cm (0 = left, 20 = right)
            belt_y_cm: Y position on belt in cm (0 = ROI top)

        Returns:
            (pixel_x, pixel_y) tuple, or None if no calibration.
        """
        if self.homography_inv is not None:
            try:
                belt_pt = np.array([[[float(belt_x_cm), float(belt_y_cm)]]], dtype=np.float32)
                pixel_pt = cv2.perspectiveTransform(belt_pt, self.homography_inv)[0][0]
                return int(pixel_pt[0]), int(pixel_pt[1])
            except Exception:
                pass

        # Fallback: linear inverse
        if self.roi_corners is not None:
            px = self.x_min_px + (belt_x_cm / self.roi_width_cm) * (self.x_max_px - self.x_min_px)
            py = self.entry_start_y_px + belt_y_cm * self.px_per_cm
            return int(px), int(py)

        return None

    def get_height_at_point(self, depth_frame, x, y):
        """Get height above floor at a point.
        
        Returns height in cm, clamped to 0-30cm range for safety.
        Invalid depth readings return 0.
        """
        x, y = int(x), int(y)
        MAX_VALID_HEIGHT_CM = 30.0  # Objects taller than this are likely depth errors
        
        # Try floor depth map first
        if self.floor_depth_map is not None:
            if 0 <= y < self.floor_depth_map.shape[0] and 0 <= x < self.floor_depth_map.shape[1]:
                floor_d = self.floor_depth_map[y, x]
                if floor_d > 0.1:
                    measured_d = depth_frame.get_distance(x, y)
                    if 0.1 < measured_d < 2.0:
                        height_m = floor_d - measured_d
                        height_cm = height_m * 100
                        # Clamp to valid range (negative or huge values are depth errors)
                        return max(0, min(MAX_VALID_HEIGHT_CM, height_cm))
        
        # Fallback to simple floor depth
        if self.floor_depth is not None:
            depth = depth_frame.get_distance(x, y)
            if 0.1 < depth < 2.0:
                height_m = self.floor_depth - depth
                height_cm = height_m * 100
                # Clamp to valid range
                return max(0, min(MAX_VALID_HEIGHT_CM, height_cm))
        
        return 0
    
    def point_in_roi(self, x, y):
        """Check if point is inside ROI polygon."""
        if self.roi_corners is None:
            return True
        return cv2.pointPolygonTest(self.roi_corners.astype(np.int32), (x, y), False) >= 0
    
    def point_in_belt_area(self, x, y):
        """Check if point is inside any belt zone (entry + ROI + exit)."""
        if self.roi_corners is None:
            return True
        
        # Check ROI
        if cv2.pointPolygonTest(self.roi_corners.astype(np.float32), (float(x), float(y)), False) >= 0:
            return True
        
        # Check entry zone
        if self.entry_corners is not None:
            if cv2.pointPolygonTest(self.entry_corners.astype(np.float32), (float(x), float(y)), False) >= 0:
                return True
        
        # Check exit zone
        if self.exit_corners is not None:
            if cv2.pointPolygonTest(self.exit_corners.astype(np.float32), (float(x), float(y)), False) >= 0:
                return True
        
        return False
    
    def apply_edge_detection(self, mask_binary):
        """Apply edge detection to a binary mask."""
        blurred = cv2.GaussianBlur(mask_binary * 255, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges
    
    def measure_object_size_cm(self, mask_binary, depth_frame):
        """Measure object size in centimeters using depth data."""
        if self.intrinsics is None:
            return None, None, None, None
        
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        cx = x + w // 2
        cy = y + h // 2
        
        depth_image = np.asanyarray(depth_frame.get_data())
        
        depths = []
        for dy in range(-5, 6, 2):
            for dx in range(-5, 6, 2):
                px, py = cx + dx, cy + dy
                if 0 <= px < IMAGE_WIDTH and 0 <= py < IMAGE_HEIGHT:
                    if mask_binary[py, px] > 0:
                        d = depth_image[py, px]
                        if 100 < d < 2000:
                            depths.append(d)
        
        if not depths:
            return None, None, None, box
        
        avg_depth_m = np.median(depths) * 0.001
        
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        rect_w, rect_h = rect[1]
        if rect_w < rect_h:
            rect_w, rect_h = rect_h, rect_w
        
        width_m = (rect_w * avg_depth_m) / fx
        height_m = (rect_h * avg_depth_m) / fy
        
        width_cm = width_m * 100
        height_cm = height_m * 100
        area_cm2 = width_cm * height_cm
        
        return width_cm, height_cm, area_cm2, box
    
    def detect(self, color_image, depth_frame):
        """
        Run YOLO detection and return list of detections with belt coordinates.
        
        Includes:
        - ROI mask coverage filtering (reject objects mostly outside ROI)
        - Depth clustering (split merged YOLO masks using depth discontinuities)
        
        Returns:
            List of detection dicts with:
            - centroid: (cx, cy) in pixels
            - class_id: int
            - mask: binary mask
            - min_area_box: rotated rectangle points
            - height_cm: object height
            - angle: orientation angle
            - belt_x_cm, belt_y_cm: belt coordinates
            - depth_mm: average depth in mm
            - zone: 'roi' or 'exit'
            - is_depth_split: bool, True if this sub-detection came from a split mask
        """
        if self.model is None:
            return []
        
        results = self.model(color_image, conf=self.conf_threshold, verbose=False)
        detections = []
        
        if not results or not results[0].masks:
            return detections
        
        # Pre-fetch depth image once for depth clustering
        depth_image_raw = np.asanyarray(depth_frame.get_data())
        
        for i, box in enumerate(results[0].boxes):
            if box.conf < self.conf_threshold:
                continue
            
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Get mask
            raw_mask = results[0].masks.data[i].cpu().numpy()
            mask = cv2.resize(raw_mask, (IMAGE_WIDTH, IMAGE_HEIGHT))
            mask = (mask > 0.5).astype(np.uint8)
            
            if mask.sum() < 100:
                continue
            
            # --- ROI / EXTENDED ZONE MASK COVERAGE FILTERING ---
            # Two-tier filtering:
            #   1. Check if mask meets ROI coverage threshold (strict: 85%)
            #   2. If EXTENDED_DETECTION_ZONE is on and the mask didn't pass
            #      ROI-only, check if it falls within the combined ROI+exit zone
            #      with a lower threshold (objects may be partially out of frame).
            # Each detection is tagged with 'zone': 'roi' or 'exit'.
            detection_zone = 'roi'  # default
            if self._roi_mask is not None:
                total_px = int(mask.sum())
                if total_px == 0:
                    continue
                inside_roi_px = int((mask & (self._roi_mask > 0)).sum())
                roi_coverage = inside_roi_px / total_px
                
                if roi_coverage >= ROI_MIN_MASK_COVERAGE:
                    detection_zone = 'roi'
                elif self._extended_roi_mask is not None:
                    # Object didn't meet strict ROI threshold —
                    # check if it's in the extended zone (ROI + exit)
                    inside_ext_px = int((mask & (self._extended_roi_mask > 0)).sum())
                    ext_coverage = inside_ext_px / total_px
                    if ext_coverage >= EXIT_ZONE_MIN_MASK_COVERAGE:
                        detection_zone = 'exit'
                    else:
                        continue  # Not enough coverage in either zone
                else:
                    continue  # No extended mask, and ROI coverage too low
            else:
                # Fallback: centroid check
                M = cv2.moments(mask)
                if M["m00"] > 0:
                    cx_check = int(M["m10"] / M["m00"])
                    cy_check = int(M["m01"] / M["m00"])
                    if not self.point_in_roi(cx_check, cy_check):
                        continue
            
            # === DEPTH CLUSTERING: attempt to split this mask ===
            if self.use_depth_cluster:
                sub_masks = depth_cluster_mask(mask, depth_image_raw)
            else:
                # No clustering — treat whole mask as one
                valid_d = depth_image_raw[mask > 0]
                valid_d = valid_d[(valid_d > 100) & (valid_d < 2000)]
                med_d = float(np.median(valid_d)) if len(valid_d) > 0 else 0
                sub_masks = [(mask, med_d)]
            
            is_split = len(sub_masks) > 1
            
            # Process each sub-mask as a separate detection
            for sub_idx, (sub_mask, sub_depth_mm) in enumerate(sub_masks):
                # Calculate centroid for this sub-mask
                M_sub = cv2.moments(sub_mask)
                if M_sub["m00"] <= 0:
                    continue
                cx = int(M_sub["m10"] / M_sub["m00"])
                cy = int(M_sub["m01"] / M_sub["m00"])
                
                # Find contours for this sub-mask
                contours, _ = cv2.findContours(
                    sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                box_points = cv2.boxPoints(rect).astype(np.int32)
                
                # Get height above floor
                height_cm = self.get_height_at_point(depth_frame, cx, cy)
                
                # Get object size using depth
                size_result = self.measure_object_size_cm(sub_mask, depth_frame)
                if size_result[0] is not None:
                    width_cm, obj_height_cm, area_cm2, _ = size_result
                else:
                    width_cm, obj_height_cm, area_cm2 = None, None, None
                
                # Get orientation
                angle = get_orientation_pca(sub_mask)
                
                # Get belt coordinates using homography for accuracy
                belt_x_cm, belt_y_cm = self.pixel_to_belt_cm(cx, cy)
                
                # Average depth in this sub-mask
                masked_depths = depth_image_raw[sub_mask > 0]
                valid_depths = masked_depths[(masked_depths > 100) & (masked_depths < 2000)]
                avg_depth_mm = float(np.mean(valid_depths)) if len(valid_depths) > 0 else 0

                # Resolve zone after depth split so each sub-detection gets
                # an accurate zone tag (ROI vs exit).
                sub_detection_zone = detection_zone
                sub_total_px = int(sub_mask.sum())
                if sub_total_px > 0 and self._roi_mask is not None:
                    sub_inside_roi_px = int((sub_mask & (self._roi_mask > 0)).sum())
                    sub_roi_coverage = sub_inside_roi_px / sub_total_px
                    if sub_roi_coverage >= ROI_MIN_MASK_COVERAGE:
                        sub_detection_zone = 'roi'
                    elif self._extended_roi_mask is not None:
                        sub_inside_ext_px = int((sub_mask & (self._extended_roi_mask > 0)).sum())
                        sub_ext_coverage = sub_inside_ext_px / sub_total_px
                        if sub_ext_coverage >= EXIT_ZONE_MIN_MASK_COVERAGE:
                            sub_detection_zone = 'exit'
                
                detections.append({
                    'centroid': (cx, cy),
                    'class_id': class_id,
                    'class_name': CLASS_NAMES.get(class_id, f"Class_{class_id}"),
                    'mask': sub_mask,
                    'contour': largest_contour,
                    'min_area_box': box_points,
                    'height_cm': height_cm,
                    'width_cm': width_cm,
                    'obj_height_cm': obj_height_cm,
                    'area_cm2': area_cm2,
                    'angle': angle,
                    'belt_x_cm': belt_x_cm,
                    'belt_y_cm': belt_y_cm,
                    'depth_mm': avg_depth_mm,
                    'confidence': confidence,
                    'zone': sub_detection_zone,
                    'is_depth_split': is_split,
                    'split_index': sub_idx if is_split else -1,
                })
        
        # === CROSS-CLASS DUPLICATE SUPPRESSION ===
        # Remove detections where YOLO gave the same physical object two
        # different class labels (their masks overlap heavily).
        if self.use_cross_nms and len(detections) > 1:
            before = len(detections)
            detections = cross_class_mask_nms(detections, DUPLICATE_MASK_IOU_THRESHOLD)
            if len(detections) < before:
                suppressed = before - len(detections)
                # Optional: uncomment for debug
                # print(f"[DETECTOR] Cross-class NMS suppressed {suppressed} duplicate(s)")
        
        # === WATERSHED MASK JOINING ===
        # Rejoin split fragments of occluded objects using watershed.
        if self.use_watershed and len(detections) > 1:
            before_join = len(detections)
            detections = watershed_join_masks(
                detections, depth_image_raw, color_image)
            # Recalculate belt coords for any watershed-joined detections
            # whose centroid moved after the merge.
            for det in detections:
                if det.get('is_watershed_joined'):
                    cx, cy = det['centroid']
                    det['belt_x_cm'], det['belt_y_cm'] = self.pixel_to_belt_cm(cx, cy)
                    det['height_cm'] = self.get_height_at_point(depth_frame, cx, cy)
            if len(detections) < before_join:
                joined = before_join - len(detections)
                # Optional: uncomment for debug
                # print(f"[DETECTOR] Watershed joined {joined} fragment pair(s)")
        
        return detections
    
    def draw_detections(self, frame, detections, show_mask=True, show_height=True, mask_alpha=0.4):
        """Draw detections on frame - shows smooth mask overlay with contour edges and directional arrow."""
        overlay = frame.copy()
        
        for det in detections:
            cx, cy = det['centroid']
            class_id = det['class_id']
            color = CLASS_COLORS.get(class_id, DEFAULT_COLOR)
            angle = det.get('angle', 0)
            
            # Draw mask (smooth semi-transparent fill)
            if show_mask and det.get('mask') is not None:
                mask = det['mask']
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask > 0] = color
                # Blend only in masked region for smoother result
                mask_region = mask > 0
                overlay[mask_region] = cv2.addWeighted(
                    overlay[mask_region], 1 - mask_alpha,
                    colored_mask[mask_region], mask_alpha, 0)

                # Draw watershed boundary as transparent white grid
                ws_boundary = det.get('watershed_boundary')
                if ws_boundary is not None and ws_boundary.any():
                    bnd = ws_boundary > 0
                    grid_color = np.array([255, 255, 255], dtype=np.uint8)
                    ws_alpha = WATERSHED_BOUNDARY_ALPHA
                    overlay[bnd] = cv2.addWeighted(
                        overlay[bnd], 1 - ws_alpha,
                        np.full_like(overlay[bnd], grid_color), ws_alpha, 0)
            
            # Draw contour EDGES only (smooth, no flickering like bounding box)
            if det.get('contour') is not None:
                cv2.drawContours(overlay, [det['contour']], -1, (255, 255, 255), 3)  # White outline
                cv2.drawContours(overlay, [det['contour']], -1, color, 2)  # Colored edge
            
            # Draw directional arrow showing object heading
            arrow_length = 35
            angle_rad = np.radians(angle)
            arrow_end_x = int(cx + arrow_length * np.cos(angle_rad))
            arrow_end_y = int(cy + arrow_length * np.sin(angle_rad))
            # Draw arrow with white outline for visibility
            cv2.arrowedLine(overlay, (cx, cy), (arrow_end_x, arrow_end_y), (255, 255, 255), 4, tipLength=0.3)
            cv2.arrowedLine(overlay, (cx, cy), (arrow_end_x, arrow_end_y), (0, 200, 255), 2, tipLength=0.3)  # Orange arrow
            
            # Draw center dot
            cv2.circle(overlay, (cx, cy), 4, (255, 255, 255), -1)
            cv2.circle(overlay, (cx, cy), 3, color, -1)
            
            # Draw label with measurement info
            class_name = CLASS_NAMES.get(class_id, f"C{class_id}")
            label = f"{class_name}"
            if show_height and det.get('height_cm', 0) > 0:
                label += f" H:{det['height_cm']:.1f}cm"
            
            # Add size info if available
            if det.get('width_cm') is not None and det.get('obj_height_cm') is not None:
                label += f" ({det['width_cm']:.1f}x{det['obj_height_cm']:.1f}cm)"
            
            cv2.putText(overlay, label, (cx + 15, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return overlay
    
    def draw_roi(self, frame):
        """Draw ROI on frame."""
        if self.roi_corners is not None:
            cv2.polylines(frame, [self.roi_corners.astype(np.int32)], True, (0, 255, 0), 2)
        return frame
    
    def draw_zones(self, frame, show_roi=True, show_entry=True, show_exit=True):
        """Draw all belt zones on frame."""
        result = frame.copy()
        
        # Draw entry zone (blue)
        if show_entry and self.entry_corners is not None:
            pts = self.entry_corners.astype(np.int32)
            cv2.polylines(result, [pts], True, (255, 150, 0), 2)
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(result, "ENTRY", (center[0]-25, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)
        
        # Draw ROI (green)
        if show_roi and self.roi_corners is not None:
            pts = self.roi_corners.astype(np.int32)
            cv2.polylines(result, [pts], True, (0, 255, 0), 3)
            labels = ['TL', 'TR', 'BR', 'BL']
            for corner, label in zip(pts, labels):
                cv2.circle(result, tuple(corner), 6, (0, 255, 0), -1)
                cv2.putText(result, label, (corner[0]+8, corner[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw exit zone (orange-red)
        if show_exit and self.exit_corners is not None:
            pts = self.exit_corners.astype(np.int32)
            cv2.polylines(result, [pts], True, (0, 100, 255), 2)
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(result, "EXIT", (center[0]-20, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
        
        return result
    
    def process_frame(self, color_image, depth_frame, depth_colormap=None):
        """
        Process a full frame - detect, draw overlays, return visualization.
        
        Returns:
            color_overlay, depth_overlay (or None), detections
        """
        detections = self.detect(color_image, depth_frame)
        
        color_overlay = color_image.copy()
        depth_overlay = depth_colormap.copy() if depth_colormap is not None else None
        
        for det in detections:
            cx, cy = det['centroid']
            class_id = det['class_id']
            color = CLASS_COLORS.get(class_id, DEFAULT_COLOR)
            mask = det.get('mask')
            
            # Draw mask overlay
            if self.show_masks and mask is not None:
                colored_mask = np.zeros_like(color_overlay)
                colored_mask[mask > 0] = color
                mask_region = mask > 0
                color_overlay[mask_region] = cv2.addWeighted(
                    color_overlay[mask_region], 1 - self.mask_alpha,
                    colored_mask[mask_region], self.mask_alpha, 0)
                if depth_overlay is not None:
                    depth_overlay[mask_region] = cv2.addWeighted(
                        depth_overlay[mask_region], 1 - self.mask_alpha,
                        colored_mask[mask_region], self.mask_alpha, 0)
                
                # Draw contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(color_overlay, contours, -1, color, 2)
                if depth_overlay is not None:
                    cv2.drawContours(depth_overlay, contours, -1, color, 2)
            
            # Draw edges
            if self.show_edges and mask is not None:
                edges = self.apply_edge_detection(mask)
                edge_color = (0, 255, 255)  # Cyan
                color_overlay[edges > 0] = edge_color
                if depth_overlay is not None:
                    depth_overlay[edges > 0] = edge_color
                
                # Draw min area box
                if det.get('min_area_box') is not None:
                    cv2.drawContours(color_overlay, [det['min_area_box']], 0, (255, 255, 0), 2)
                    if depth_overlay is not None:
                        cv2.drawContours(depth_overlay, [det['min_area_box']], 0, (255, 255, 0), 2)
            
            # Draw X mark
            if self.show_x_marks:
                x_size = self.x_mark_size
                # White outline
                cv2.line(color_overlay, (cx - x_size, cy - x_size), (cx + x_size, cy + x_size), (255, 255, 255), 4)
                cv2.line(color_overlay, (cx + x_size, cy - x_size), (cx - x_size, cy + x_size), (255, 255, 255), 4)
                # Colored X
                cv2.line(color_overlay, (cx - x_size, cy - x_size), (cx + x_size, cy + x_size), color, 2)
                cv2.line(color_overlay, (cx + x_size, cy - x_size), (cx - x_size, cy + x_size), color, 2)
                
                if depth_overlay is not None:
                    cv2.line(depth_overlay, (cx - x_size, cy - x_size), (cx + x_size, cy + x_size), (255, 255, 255), 4)
                    cv2.line(depth_overlay, (cx + x_size, cy - x_size), (cx - x_size, cy + x_size), (255, 255, 255), 4)
                    cv2.line(depth_overlay, (cx - x_size, cy - x_size), (cx + x_size, cy + x_size), color, 2)
                    cv2.line(depth_overlay, (cx + x_size, cy - x_size), (cx - x_size, cy + x_size), color, 2)
            
            # Draw labels
            if self.show_height:
                class_name = CLASS_NAMES.get(class_id, f"C{class_id}")
                label = f"{class_name}"
                height_cm = det.get('height_cm', 0)
                if height_cm > 0:
                    label += f" H:{height_cm:.1f}cm"
                
                label_y = cy - 25 if self.show_x_marks else cy - 10
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(color_overlay, (cx - 5, label_y - th - 5), (cx + tw + 5, label_y + 5), (0, 0, 0), -1)
                cv2.putText(color_overlay, label, (cx, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if depth_overlay is not None:
                    cv2.rectangle(depth_overlay, (cx - 5, label_y - th - 5), (cx + tw + 5, label_y + 5), (0, 0, 0), -1)
                    cv2.putText(depth_overlay, label, (cx, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return color_overlay, depth_overlay, detections


class ThreadedDetector:
    """Threaded detector for async processing."""
    
    def __init__(self, detector, result_queue):
        self.detector = detector
        self.result_queue = result_queue
        self.input_queue = queue.Queue(maxsize=1)
        self.running = False
        self._thread = None
    
    def start_detection(self):
        """Start the detection thread."""
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def process_frame(self, color_image, depth_frame, timestamp):
        """Queue a frame for processing."""
        if not self.input_queue.full():
            self.input_queue.put((color_image, depth_frame, timestamp))
    
    def _run(self):
        """Main detection loop."""
        while self.running:
            try:
                color_image, depth_frame, timestamp = self.input_queue.get(timeout=0.1)
                detections = self.detector.detect(color_image, depth_frame)
                self.result_queue.put((detections, timestamp))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[DETECTOR] Error: {e}")
    
    def stop(self):
        """Stop the detection thread."""
        self.running = False
