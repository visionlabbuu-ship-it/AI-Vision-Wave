#!/usr/bin/env python3
"""
=============================================================================
CONVEYOR BELT & ROI CALIBRATION TOOL
=============================================================================
Uses a checkerboard pattern to calibrate:
1. Perspective transformation (pixel space → belt space in cm)
2. ROI boundaries (Entry, ROI, Exit zones)
3. Floor plane for height measurement

Checkerboard specs:
- 8×10 squares (7×9 inner corners)
- Each square: 2.5cm × 2.5cm
- Total size: 20cm × 25cm

Output ROI: 20cm × 30cm (matching conveyor belt working area)

Usage:
1. Place checkerboard flat on conveyor belt
2. Align checkerboard edge with desired ROI origin
3. Press 'c' to capture and calibrate
4. Press 's' to save calibration
5. Press 'q' to quit

Author: Pipeline Team
Date: 2026
=============================================================================
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import json
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Checkerboard parameters
CHECKERBOARD_SQUARES_X = 8   # Number of squares in X direction
CHECKERBOARD_SQUARES_Y = 10  # Number of squares in Y direction
SQUARE_SIZE_CM = 2.5         # Size of each square in cm

# Inner corners (one less than squares)
CHECKERBOARD_CORNERS_X = CHECKERBOARD_SQUARES_X - 1  # 7
CHECKERBOARD_CORNERS_Y = CHECKERBOARD_SQUARES_Y - 1  # 9

# Physical checkerboard size
CHECKERBOARD_WIDTH_CM = CHECKERBOARD_SQUARES_X * SQUARE_SIZE_CM   # 20cm
CHECKERBOARD_HEIGHT_CM = CHECKERBOARD_SQUARES_Y * SQUARE_SIZE_CM  # 25cm

# Target ROI size (conveyor belt working area)
ROI_WIDTH_CM = 20.0   # Same as checkerboard width
ROI_HEIGHT_CM = 30.0  # Extended for full belt coverage

# Entry and Exit zones
ENTRY_PATH_CM = 21.5
EXIT_PATH_CM = 21.5

# Camera resolution
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Calibration file
CALIBRATION_FILE = "calibration_data.json"

# =============================================================================
# CALIBRATION CLASS
# =============================================================================

class ConveyorCalibrator:
    """
    Calibrates conveyor belt ROI using checkerboard pattern.
    """
    
    def __init__(self):
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        
        # Calibration results
        self.checkerboard_corners = None  # Detected corners in pixel space
        self.homography = None            # Perspective transform matrix
        self.roi_corners_px = None        # ROI corners in pixels
        self.floor_plane = None           # Floor plane coefficients (a, b, c, d)
        self.corrected_corners = None     # Orientation-corrected corner points
        
        # State
        self.calibration_done = False
        self.flip_vertical = False        # User can toggle to flip ROI orientation
        self.flip_horizontal = False      # User can toggle to mirror left/right
        
        # Preview state (before confirmation)
        self.preview_corners = None       # Live preview corners
        self.floor_calibrated = False
        
    def start_camera(self):
        """Initialize RealSense camera."""
        print("Initializing RealSense camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)
        
        profile = self.pipeline.start(config)
        
        # Get intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Warm up
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        print(f"Camera ready: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        print(f"Intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
        
    def stop_camera(self):
        """Stop camera."""
        if self.pipeline:
            self.pipeline.stop()
            
    def get_frames(self):
        """Get aligned color and depth frames."""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
            
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame, color_frame
    
    def detect_checkerboard(self, image):
        """
        Detect checkerboard corners in image.
        
        Returns:
            corners: Array of corner points or None if not found
            success: True if checkerboard found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        pattern_size = (CHECKERBOARD_CORNERS_X, CHECKERBOARD_CORNERS_Y)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        
        success, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        
        if success:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
        return corners, success
    
    def get_oriented_corners(self, corners):
        """
        Get the 4 ROI corners with user-controlled orientation (flip support).
        
        Returns dict with top_left, top_right, bottom_left, bottom_right
        """
        if corners is None:
            return None
            
        # Reshape corners to grid
        corners_grid = corners.reshape(CHECKERBOARD_CORNERS_Y, CHECKERBOARD_CORNERS_X, 2)
        
        # Get the 4 outer corners
        corner_00 = corners_grid[0, 0]
        corner_0n = corners_grid[0, -1]
        corner_n0 = corners_grid[-1, 0]
        corner_nn = corners_grid[-1, -1]
        
        # Apply vertical flip if enabled
        if self.flip_vertical:
            # Swap top and bottom
            top_left, top_right = corner_n0, corner_nn
            bottom_left, bottom_right = corner_00, corner_0n
        else:
            top_left, top_right = corner_00, corner_0n
            bottom_left, bottom_right = corner_n0, corner_nn
        
        # Apply horizontal flip if enabled
        if self.flip_horizontal:
            top_left, top_right = top_right, top_left
            bottom_left, bottom_right = bottom_right, bottom_left
        
        return {
            'top_left': top_left.copy(),
            'top_right': top_right.copy(),
            'bottom_left': bottom_left.copy(),
            'bottom_right': bottom_right.copy()
        }
    
    def calculate_preview_roi(self, corners):
        """
        Calculate preview ROI corners for live display (before confirmation).
        Uses current flip settings.
        """
        oriented = self.get_oriented_corners(corners)
        if oriented is None:
            return None
            
        top_left = oriented['top_left']
        top_right = oriented['top_right']
        bottom_left = oriented['bottom_left']
        
        # Calculate pixel-per-cm ratios
        x_span_px = np.linalg.norm(top_right - top_left)
        x_span_cm = (CHECKERBOARD_CORNERS_X - 1) * SQUARE_SIZE_CM
        px_per_cm_x = x_span_px / x_span_cm
        
        y_span_px = np.linalg.norm(bottom_left - top_left)
        y_span_cm = (CHECKERBOARD_CORNERS_Y - 1) * SQUARE_SIZE_CM
        px_per_cm_y = y_span_px / y_span_cm
        
        # Direction vectors
        x_dir = (top_right - top_left) / x_span_px
        y_dir = (bottom_left - top_left) / y_span_px
        
        # Extend to full ROI size
        roi_top_left = top_left - x_dir * (SQUARE_SIZE_CM * px_per_cm_x)
        roi_top_left = roi_top_left - y_dir * (SQUARE_SIZE_CM * px_per_cm_y)
        
        roi_top_right = roi_top_left + x_dir * (ROI_WIDTH_CM * px_per_cm_x)
        roi_bottom_left = roi_top_left + y_dir * (ROI_HEIGHT_CM * px_per_cm_y)
        roi_bottom_right = roi_top_right + y_dir * (ROI_HEIGHT_CM * px_per_cm_y)
        
        return np.array([
            roi_top_left,
            roi_top_right,
            roi_bottom_right,
            roi_bottom_left
        ], dtype=np.float32)
    
    def calculate_homography(self, corners):
        """
        Calculate perspective transform from checkerboard corners.
        Uses current flip settings for orientation control.
        
        Args:
            corners: Detected checkerboard corners (7×9 = 63 points)
            
        Returns:
            homography: 3×3 perspective transform matrix
        """
        if corners is None:
            return None
        
        # Get orientation-corrected corners (respects flip settings)
        oriented = self.get_oriented_corners(corners)
        if oriented is None:
            return None
            
        top_left = oriented['top_left']
        top_right = oriented['top_right']
        bottom_left = oriented['bottom_left']
        bottom_right = oriented['bottom_right']
        
        # Store the corrected corners
        self.corrected_corners = oriented
        
        # Source points (pixel coordinates of checkerboard corners)
        src_points = np.array([
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ], dtype=np.float32)
        
        # Store the corrected corners grid for extend_roi_to_full_size
        self.corrected_corners = {
            'top_left': top_left,
            'top_right': top_right,
            'bottom_left': bottom_left,
            'bottom_right': bottom_right
        }
        
        # The inner corners span (CORNERS_X-1) × (CORNERS_Y-1) squares
        # which is 6 × 8 squares = 15cm × 20cm
        inner_width_cm = (CHECKERBOARD_CORNERS_X - 1) * SQUARE_SIZE_CM   # 6 * 2.5 = 15cm
        inner_height_cm = (CHECKERBOARD_CORNERS_Y - 1) * SQUARE_SIZE_CM  # 8 * 2.5 = 20cm
        
        # Destination points in cm (belt coordinate system)
        # Origin at top-left, X increases right, Y increases down (toward exit)
        dst_points = np.array([
            [0, 0],
            [inner_width_cm, 0],
            [inner_width_cm, inner_height_cm],
            [0, inner_height_cm]
        ], dtype=np.float32)
        
        # Calculate homography (pixel → cm)
        homography = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Store ROI corners (extended to full ROI size)
        self.roi_corners_px = src_points
        
        return homography
    
    def extend_roi_to_full_size(self, corners):
        """
        Extend the detected checkerboard area to the full ROI size (20×30 cm).
        
        The checkerboard inner corners span 15×20 cm, but we want 20×30 cm ROI.
        We need to extrapolate the corners.
        """
        if corners is None or self.homography is None:
            return None
        
        # Use the corrected corners (already orientation-fixed)
        if not hasattr(self, 'corrected_corners') or self.corrected_corners is None:
            return None
            
        top_left = self.corrected_corners['top_left']
        top_right = self.corrected_corners['top_right']
        bottom_left = self.corrected_corners['bottom_left']
        bottom_right = self.corrected_corners['bottom_right']
        
        # Calculate pixel-per-cm ratios from the corrected corners
        x_span_px = np.linalg.norm(top_right - top_left)
        x_span_cm = (CHECKERBOARD_CORNERS_X - 1) * SQUARE_SIZE_CM  # 15cm
        px_per_cm_x = x_span_px / x_span_cm
        
        y_span_px = np.linalg.norm(bottom_left - top_left)
        y_span_cm = (CHECKERBOARD_CORNERS_Y - 1) * SQUARE_SIZE_CM  # 20cm
        px_per_cm_y = y_span_px / y_span_cm
        
        # Direction vectors (normalized)
        x_dir = (top_right - top_left) / x_span_px
        y_dir = (bottom_left - top_left) / y_span_px
        
        # Extend to full 20×30 cm ROI
        # Start from top-left inner corner and extend
        roi_top_left = top_left - x_dir * (SQUARE_SIZE_CM * px_per_cm_x)  # Add 1 square left
        roi_top_left = roi_top_left - y_dir * (SQUARE_SIZE_CM * px_per_cm_y)  # Add 1 square up
        
        roi_top_right = roi_top_left + x_dir * (ROI_WIDTH_CM * px_per_cm_x)
        roi_bottom_left = roi_top_left + y_dir * (ROI_HEIGHT_CM * px_per_cm_y)
        roi_bottom_right = roi_top_right + y_dir * (ROI_HEIGHT_CM * px_per_cm_y)
        
        full_roi_corners = np.array([
            roi_top_left,
            roi_top_right,
            roi_bottom_right,
            roi_bottom_left
        ], dtype=np.float32)
        
        # Recalculate homography for full ROI
        dst_points = np.array([
            [0, 0],
            [ROI_WIDTH_CM, 0],
            [ROI_WIDTH_CM, ROI_HEIGHT_CM],
            [0, ROI_HEIGHT_CM]
        ], dtype=np.float32)
        
        full_homography = cv2.getPerspectiveTransform(full_roi_corners, dst_points)
        
        return full_roi_corners, full_homography
    
    def calibrate_floor(self, depth_frame, roi_corners):
        """
        Calculate floor plane from depth data across ENTIRE belt area.
        This includes entry zone, ROI, and exit zone for accurate perspective correction.
        
        Returns:
            plane_coeffs: (a, b, c, d) where ax + by + cz + d = 0
        """
        if roi_corners is None:
            return None
        
        # Calculate entry and exit zone corners for full belt coverage
        entry_corners, exit_corners = self.calculate_entry_exit_zones(roi_corners)
        
        # Create combined mask for entire belt (entry + ROI + exit)
        mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        
        # Fill ROI area
        cv2.fillPoly(mask, [roi_corners.astype(np.int32)], 255)
        
        # Fill entry zone
        if entry_corners is not None:
            cv2.fillPoly(mask, [entry_corners.astype(np.int32)], 255)
        
        # Fill exit zone
        if exit_corners is not None:
            cv2.fillPoly(mask, [exit_corners.astype(np.int32)], 255)
        
        print(f"Floor calibration sampling from full belt area (entry + ROI + exit)")
        
        # Collect 3D points from depth within full belt area
        points_3d = []
        pixel_coords = []  # Store corresponding pixel coordinates
        
        for y in range(0, IMAGE_HEIGHT, 4):  # Subsample for speed
            for x in range(0, IMAGE_WIDTH, 4):
                if mask[y, x] == 0:
                    continue
                    
                depth = depth_frame.get_distance(x, y)
                if depth <= 0 or depth > 2.0:  # Valid depth range
                    continue
                    
                # Deproject to 3D
                point_3d = rs.rs2_deproject_pixel_to_point(
                    self.intrinsics, [x, y], depth)
                points_3d.append(point_3d)
                pixel_coords.append((x, y))
        
        if len(points_3d) < 100:
            print(f"Warning: Only {len(points_3d)} valid depth points")
            return None
            
        points_3d = np.array(points_3d)
        
        # Fit plane using RANSAC
        best_plane = None
        best_inliers = 0
        
        for _ in range(100):  # RANSAC iterations
            # Random sample 3 points
            idx = np.random.choice(len(points_3d), 3, replace=False)
            p1, p2, p3 = points_3d[idx]
            
            # Calculate plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) < 1e-6:
                continue
                
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, p1)
            
            # Count inliers
            distances = np.abs(np.dot(points_3d, normal) + d)
            inliers = np.sum(distances < 0.01)  # 1cm threshold
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (normal[0], normal[1], normal[2], d)
        
        print(f"Floor plane fitted with {best_inliers}/{len(points_3d)} inliers")
        return best_plane
    
    def create_floor_depth_map(self, depth_frame, roi_corners, num_samples=10):
        """
        Create a floor depth reference map by averaging multiple depth frames.
        This map stores the expected floor depth at each pixel position,
        allowing for pixel-by-pixel perspective correction.
        
        The floor depth map is used in detection to calculate TRUE object height:
            true_height = floor_depth_at(x,y) - measured_object_depth
        
        Args:
            depth_frame: Current RealSense depth frame
            roi_corners: ROI corner coordinates
            num_samples: Number of frames to average (default 10)
            
        Returns:
            floor_depth_map: 2D numpy array with expected floor depth at each pixel
        """
        if roi_corners is None or self.intrinsics is None:
            return None
        
        # Calculate entry and exit zone corners for full belt coverage
        entry_corners, exit_corners = self.calculate_entry_exit_zones(roi_corners)
        
        # Create combined mask for entire belt
        belt_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        cv2.fillPoly(belt_mask, [roi_corners.astype(np.int32)], 255)
        if entry_corners is not None:
            cv2.fillPoly(belt_mask, [entry_corners.astype(np.int32)], 255)
        if exit_corners is not None:
            cv2.fillPoly(belt_mask, [exit_corners.astype(np.int32)], 255)
        
        print(f"Creating floor depth map (averaging {num_samples} frames)...")
        
        # Collect depth data from current frame
        depth_data = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001  # Convert to meters
        
        # Create floor depth map
        floor_depth_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        
        # Apply mask and valid depth range
        valid_mask = (belt_mask > 0) & (depth_data > 0.1) & (depth_data < 2.0)
        floor_depth_map[valid_mask] = depth_data[valid_mask]
        
        # Fill holes using interpolation (for pixels with invalid depth)
        # Use median filter to smooth and fill small gaps
        from scipy import ndimage
        
        # First, apply a median filter to reduce noise
        floor_depth_map_filtered = cv2.medianBlur(floor_depth_map, 5)
        
        # For pixels with no valid depth, try to interpolate from neighbors
        invalid_mask = (belt_mask > 0) & (floor_depth_map == 0)
        if np.sum(invalid_mask) > 0:
            # Use morphological closing to fill small holes
            kernel = np.ones((11, 11), np.uint8)
            floor_depth_map_filled = cv2.morphologyEx(floor_depth_map_filtered, cv2.MORPH_CLOSE, kernel)
            floor_depth_map[invalid_mask] = floor_depth_map_filled[invalid_mask]
        
        # Count valid pixels
        valid_count = np.sum((belt_mask > 0) & (floor_depth_map > 0))
        total_count = np.sum(belt_mask > 0)
        coverage = valid_count / total_count * 100 if total_count > 0 else 0
        
        print(f"Floor depth map created: {valid_count}/{total_count} pixels ({coverage:.1f}% coverage)")
        
        # Store for later use
        self.floor_depth_map = floor_depth_map
        self.belt_mask = belt_mask
        
        return floor_depth_map
    
    def calculate_entry_exit_zones(self, roi_corners):
        """
        Calculate entry and exit zone boundaries based on ROI corners.
        
        Entry zone: Above ROI (belt moving toward camera)
        Exit zone: Below ROI (belt moving away from camera)
        """
        if roi_corners is None:
            return None, None
            
        # ROI corners: [top_left, top_right, bottom_right, bottom_left]
        roi_top_left = roi_corners[0]
        roi_top_right = roi_corners[1]
        roi_bottom_right = roi_corners[2]
        roi_bottom_left = roi_corners[3]
        
        # Calculate direction vectors
        left_dir = roi_bottom_left - roi_top_left
        left_dir_norm = left_dir / np.linalg.norm(left_dir)
        right_dir = roi_bottom_right - roi_top_right
        right_dir_norm = right_dir / np.linalg.norm(right_dir)
        
        # Pixels per cm (approximate)
        roi_height_px = np.linalg.norm(left_dir)
        px_per_cm = roi_height_px / ROI_HEIGHT_CM
        
        # Entry zone (extend upward from ROI top)
        entry_extend_px = ENTRY_PATH_CM * px_per_cm
        entry_top_left = roi_top_left - (left_dir_norm * entry_extend_px)
        entry_top_right = roi_top_right - (right_dir_norm * entry_extend_px)
        
        entry_corners = np.array([
            entry_top_left,
            entry_top_right,
            roi_top_right,
            roi_top_left
        ], dtype=np.float32)
        
        # Exit zone (extend downward from ROI bottom)
        exit_extend_px = EXIT_PATH_CM * px_per_cm
        exit_bottom_left = roi_bottom_left + (left_dir_norm * exit_extend_px)
        exit_bottom_right = roi_bottom_right + (right_dir_norm * exit_extend_px)
        
        exit_corners = np.array([
            roi_bottom_left,
            roi_bottom_right,
            exit_bottom_right,
            exit_bottom_left
        ], dtype=np.float32)
        
        return entry_corners, exit_corners
    
    def save_calibration(self, filename=CALIBRATION_FILE):
        """
        Save calibration data to JSON file, including floor depth map.
        """
        if self.homography is None:
            print("Error: No calibration data to save")
            return False
            
        # Calculate entry/exit zones
        entry_corners, exit_corners = self.calculate_entry_exit_zones(self.roi_corners_px)
        
        # Prepare floor depth map data for storage
        floor_depth_map_data = None
        if hasattr(self, 'floor_depth_map') and self.floor_depth_map is not None:
            # Downsample for storage (store every 4th pixel)
            downsampled = self.floor_depth_map[::4, ::4]
            floor_depth_map_data = {
                "data": downsampled.tolist(),
                "original_shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
                "downsample_factor": 4,
                "description": "Floor depth in meters at each pixel (downsampled 4x)"
            }
            print(f"Floor depth map saved (downsampled to {downsampled.shape})")
        
        calibration_data = {
            "timestamp": datetime.now().isoformat(),
            "checkerboard": {
                "squares_x": CHECKERBOARD_SQUARES_X,
                "squares_y": CHECKERBOARD_SQUARES_Y,
                "square_size_cm": SQUARE_SIZE_CM
            },
            "roi": {
                "width_cm": ROI_WIDTH_CM,
                "height_cm": ROI_HEIGHT_CM,
                "corners_px": self.roi_corners_px.tolist() if self.roi_corners_px is not None else None
            },
            "zones": {
                "entry_path_cm": ENTRY_PATH_CM,
                "exit_path_cm": EXIT_PATH_CM,
                "entry_corners_px": entry_corners.tolist() if entry_corners is not None else None,
                "exit_corners_px": exit_corners.tolist() if exit_corners is not None else None
            },
            "transforms": {
                "homography": self.homography.tolist() if self.homography is not None else None,
                "homography_inv": np.linalg.inv(self.homography).tolist() if self.homography is not None else None
            },
            "floor_plane": {
                "coefficients": list(self.floor_plane) if self.floor_plane is not None else None,
                "description": "ax + by + cz + d = 0"
            },
            "floor_depth_map": floor_depth_map_data,
            "camera": {
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "intrinsics": {
                    "fx": self.intrinsics.fx,
                    "fy": self.intrinsics.fy,
                    "ppx": self.intrinsics.ppx,
                    "ppy": self.intrinsics.ppy,
                    "coeffs": list(self.intrinsics.coeffs)
                } if self.intrinsics else None
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
            
        print(f"\n[OK] Calibration saved to {filename}")
        return True
    
    def draw_visualization(self, image, corners, show_extended=True):
        """
        Draw calibration visualization on image.
        """
        result = image.copy()
        
        # Draw detected checkerboard corners
        if corners is not None:
            cv2.drawChessboardCorners(result, 
                                      (CHECKERBOARD_CORNERS_X, CHECKERBOARD_CORNERS_Y), 
                                      corners, True)
        
        # === LIVE PREVIEW: Show ROI before confirmation ===
        # Use preview corners if not yet calibrated, otherwise use confirmed corners
        preview_roi = None
        if corners is not None and not self.calibration_done:
            # Calculate live preview with current flip settings
            preview_roi = self.calculate_preview_roi(corners)
        
        # Determine which ROI to display
        display_roi = self.roi_corners_px if self.calibration_done else preview_roi
        is_preview = not self.calibration_done and preview_roi is not None
        
        # Draw ROI boundaries
        if display_roi is not None:
            roi_int = display_roi.astype(np.int32)
            
            # Different color for preview vs confirmed
            roi_color = (0, 255, 255) if is_preview else (0, 255, 0)  # Yellow for preview, Green for confirmed
            roi_thickness = 2 if is_preview else 3
            cv2.polylines(result, [roi_int], True, roi_color, roi_thickness)
            
            # Label corners with direction info
            labels = ['TL (ENTRY)', 'TR (ENTRY)', 'BR (EXIT)', 'BL (EXIT)']
            for i, (corner, label) in enumerate(zip(roi_int, labels)):
                cv2.circle(result, tuple(corner), 8, roi_color, -1)
                cv2.putText(result, label, (corner[0] + 10, corner[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi_color, 2)
            
            # === DRAW FLOW DIRECTION ARROWS ===
            top_center = ((roi_int[0] + roi_int[1]) / 2).astype(int)
            bottom_center = ((roi_int[2] + roi_int[3]) / 2).astype(int)
            
            flow_dir = bottom_center - top_center
            flow_length = np.linalg.norm(flow_dir)
            if flow_length > 0:
                flow_dir_norm = flow_dir / flow_length
                
                # Draw multiple arrows to show flow
                arrow_color = (0, 200, 255)  # Orange
                for i in range(3):
                    t = 0.25 + i * 0.25
                    arrow_start = top_center + (flow_dir * (t - 0.1)).astype(int)
                    arrow_end = top_center + (flow_dir * (t + 0.05)).astype(int)
                    cv2.arrowedLine(result, tuple(arrow_start), tuple(arrow_end), 
                                   arrow_color, 3, tipLength=0.5)
                
                # Draw "OBJECT FLOW" label
                mid_point = ((top_center + bottom_center) / 2).astype(int)
                right_offset = roi_int[1] - roi_int[0]
                if np.linalg.norm(right_offset) > 0:
                    right_offset_norm = right_offset / np.linalg.norm(right_offset)
                    label_pos = mid_point + (right_offset_norm * 70).astype(int)
                    cv2.putText(result, "OBJECT", (label_pos[0], label_pos[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
                    cv2.putText(result, "FLOW", (label_pos[0], label_pos[1] + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
                    cv2.arrowedLine(result, (label_pos[0] + 20, label_pos[1] + 15), 
                                   (label_pos[0] + 20, label_pos[1] + 45), 
                                   arrow_color, 2, tipLength=0.4)
            
            # Preview/Confirmed label
            if is_preview:
                cv2.putText(result, "PREVIEW - Press 'c' to confirm, 'r' to flip", 
                           (roi_int[0][0], roi_int[0][1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(result, "CONFIRMED", 
                           (roi_int[0][0], roi_int[0][1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw entry/exit zones (only for confirmed calibration or preview)
        if display_roi is not None:
            entry_corners, exit_corners = self.calculate_entry_exit_zones(display_roi)
            
            zone_alpha = 0.1 if is_preview else 0.15
            
            if entry_corners is not None:
                entry_int = entry_corners.astype(np.int32)
                overlay = result.copy()
                cv2.fillPoly(overlay, [entry_int], (255, 255, 0))
                cv2.addWeighted(overlay, zone_alpha, result, 1 - zone_alpha, 0, result)
                cv2.polylines(result, [entry_int], True, (255, 255, 0), 1 if is_preview else 2)
                
                entry_center = np.mean(entry_int, axis=0).astype(int)
                cv2.putText(result, "ENTRY ZONE", (entry_center[0] - 50, entry_center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(result, "(Objects enter here)", (entry_center[0] - 70, entry_center[1] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            if exit_corners is not None:
                exit_int = exit_corners.astype(np.int32)
                overlay = result.copy()
                cv2.fillPoly(overlay, [exit_int], (0, 255, 255))
                cv2.addWeighted(overlay, zone_alpha, result, 1 - zone_alpha, 0, result)
                cv2.polylines(result, [exit_int], True, (0, 255, 255), 1 if is_preview else 2)
                
                exit_center = np.mean(exit_int, axis=0).astype(int)
                cv2.putText(result, "EXIT ZONE", (exit_center[0] - 45, exit_center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(result, "(Objects exit here)", (exit_center[0] - 65, exit_center[1] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw ROI label
            roi_center = np.mean(roi_int, axis=0).astype(int)
            roi_label = "ROI (PICK ZONE)" if not is_preview else "ROI PREVIEW"
            cv2.putText(result, roi_label, (roi_center[0] - 60, roi_center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)
        
        # Status text with flip indicators
        flip_status = ""
        if self.flip_vertical:
            flip_status += " [V-FLIP]"
        if self.flip_horizontal:
            flip_status += " [H-FLIP]"
        
        status_lines = [
            f"Checkerboard: {CHECKERBOARD_SQUARES_X}x{CHECKERBOARD_SQUARES_Y} squares",
            f"ROI Size: {ROI_WIDTH_CM}x{ROI_HEIGHT_CM} cm",
            f"Calibrated: {'YES' if self.calibration_done else 'NO (preview)'}",
            f"Floor: {'YES' if self.floor_plane else 'NO'}",
            f"Orientation:{flip_status if flip_status else ' Default'}",
            "",
            "Controls:",
            "  'c' - Confirm Calibration",
            "  'r' - Flip Vertical (swap entry/exit)",
            "  'h' - Flip Horizontal (mirror L/R)",
            "  'f' - Calibrate Floor",
            "  's' - Save Calibration",
            "  'q' - Quit"
        ]
        
        for i, line in enumerate(status_lines):
            if "YES" in line:
                color = (0, 255, 0)
            elif "FLIP" in line:
                color = (0, 200, 255)
            else:
                color = (255, 255, 255)
            cv2.putText(result, line, (10, 25 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        return result


def main():
    """Main calibration routine."""
    print("=" * 60)
    print("CONVEYOR BELT CALIBRATION TOOL")
    print("=" * 60)
    print(f"Checkerboard: {CHECKERBOARD_SQUARES_X}x{CHECKERBOARD_SQUARES_Y} squares")
    print(f"Square size: {SQUARE_SIZE_CM} cm")
    print(f"Target ROI: {ROI_WIDTH_CM}x{ROI_HEIGHT_CM} cm")
    print("=" * 60)
    
    calibrator = ConveyorCalibrator()
    
    try:
        calibrator.start_camera()
        
        print("\nInstructions:")
        print("1. Place checkerboard flat on conveyor belt")
        print("2. Align top-left corner with desired ROI origin")
        print("3. Press 'c' to capture and calibrate")
        print("4. Press 'f' to calibrate floor plane")
        print("5. Press 'r' to flip vertical (swap entry/exit)")
        print("6. Press 'h' to flip horizontal (mirror left/right)")  
        print("7. Press 's' to save calibration")
        print("8. Press 'q' to quit")
        print("-" * 60)
        
        # Store last detected corners for flip operations
        last_corners = None
        
        while True:
            color_image, depth_frame, _ = calibrator.get_frames()
            if color_image is None:
                continue
            
            # Try to detect checkerboard
            corners, found = calibrator.detect_checkerboard(color_image)
            
            # Store corners for flip operations
            if found:
                last_corners = corners
            
            # Draw visualization
            display = calibrator.draw_visualization(color_image, corners if found else None)
            
            # Show detection status
            if found:
                cv2.putText(display, "CHECKERBOARD DETECTED", (IMAGE_WIDTH - 250, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Searching for checkerboard...", (IMAGE_WIDTH - 250, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow("Conveyor Belt Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('r'):
                # Flip vertical (swap entry/exit direction)
                calibrator.flip_vertical = not calibrator.flip_vertical
                calibrator.calibration_done = False  # Reset confirmation
                print(f"[FLIP] Vertical flip: {'ON' if calibrator.flip_vertical else 'OFF'}")
                
            elif key == ord('h'):
                # Flip horizontal (mirror left/right)
                calibrator.flip_horizontal = not calibrator.flip_horizontal
                calibrator.calibration_done = False  # Reset confirmation
                print(f"[FLIP] Horizontal flip: {'ON' if calibrator.flip_horizontal else 'OFF'}")
                
            elif key == ord('c') and found:
                print("\nConfirming calibration...")
                
                # Calculate homography from checkerboard with current flip settings
                calibrator.homography = calibrator.calculate_homography(corners)
                
                if calibrator.homography is not None:
                    # Extend to full ROI size
                    full_roi, full_homography = calibrator.extend_roi_to_full_size(corners)
                    if full_roi is not None:
                        calibrator.roi_corners_px = full_roi
                        calibrator.homography = full_homography
                    
                    calibrator.calibration_done = True
                    flip_info = ""
                    if calibrator.flip_vertical:
                        flip_info += " [V-FLIP]"
                    if calibrator.flip_horizontal:
                        flip_info += " [H-FLIP]"
                    print(f"[OK] Perspective calibration confirmed!{flip_info}")
                    print(f"  ROI corners (px): {calibrator.roi_corners_px.astype(int).tolist()}")
                else:
                    print("[X] Calibration failed")
                    
            elif key == ord('f'):
                # Use preview ROI if not yet confirmed
                roi_to_use = calibrator.roi_corners_px
                if roi_to_use is None and last_corners is not None:
                    roi_to_use = calibrator.calculate_preview_roi(last_corners)
                    
                if roi_to_use is not None:
                    print("\nCalibrating floor plane (full belt area)...")
                    calibrator.floor_plane = calibrator.calibrate_floor(depth_frame, roi_to_use)
                    if calibrator.floor_plane:
                        print(f"[OK] Floor plane: {calibrator.floor_plane}")
                        
                        # Also create floor depth map for perspective correction
                        print("\nCreating floor depth map for perspective correction...")
                        floor_map = calibrator.create_floor_depth_map(depth_frame, roi_to_use)
                        if floor_map is not None:
                            print("[OK] Floor depth map created successfully")
                        else:
                            print("[!] Floor depth map creation failed")
                else:
                    print("Please detect checkerboard first")
                    
            elif key == ord('s'):
                if calibrator.calibration_done:
                    calibrator.save_calibration()
                else:
                    print("Please confirm calibration first (press 'c')")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        calibrator.stop_camera()
        cv2.destroyAllWindows()
        print("\nCalibration tool closed.")


if __name__ == "__main__":
    main()
