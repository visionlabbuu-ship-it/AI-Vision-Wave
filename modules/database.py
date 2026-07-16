"""
=============================================================================
DATABASE MODULE
=============================================================================
SQLite database for logging detections and sorting data.
CSV logger for detection classification results.
"""

import sqlite3
import csv
import os
from datetime import datetime

from .config import SENSOR_CHANNELS_SPECTRAL


import cv2


class DetectionLogger:
    """
    CSV logger for detection classification results.
    Also saves cropped images of detected objects.
    
    Logs 12 columns:
    - ID
    - Camera_Glass, Camera_Plastic, Camera_Metal, Camera_Paper, Camera_Class (5 columns)
    - Spectrum_Glass, Spectrum_Plastic, Spectrum_Metal, Spectrum_Paper, Spectrum_Class (5 columns)
    - Final_Class (from Bayesian fusion voting)
    """
    
    CLASSES = ["Glass", "Plastic", "Metal", "Paper"]
    
    def __init__(self, log_dir="detection_logs"):
        self.log_dir = log_dir
        self.log_file = None
        self.writer = None
        self.file_handle = None
        self.image_dir = None
        
        # Create subdirectory structure
        self.csv_dir = os.path.join(log_dir, "detections")
        self.img_base_dir = os.path.join(log_dir, "images")
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.img_base_dir, exist_ok=True)
        
        # Initialize log file
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize a new log file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.csv_dir, f"detection_log_{timestamp}.csv")
        
        # Create image directory for this session
        self.image_dir = os.path.join(self.img_base_dir, f"images_{timestamp}")
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Create CSV file with header
        self.file_handle = open(self.log_file, 'w', newline='')
        self.writer = csv.writer(self.file_handle)
        
        # Write header (12 + 18 = 30 columns)
        header = [
            "ID",
            "Camera_Glass", "Camera_Plastic", "Camera_Metal", "Camera_Paper", "Camera_Class",
            "Spectrum_Glass", "Spectrum_Plastic", "Spectrum_Metal", "Spectrum_Paper", "Spectrum_Class",
            "Final_Class"
        ] + SENSOR_CHANNELS_SPECTRAL  # Add 18 spectral channels
        self.writer.writerow(header)
        self.file_handle.flush()
        
        print(f"[LOGGER] Detection log created: {self.log_file}")
        print(f"[LOGGER] Image directory created: {self.image_dir}")
    
    def save_object_image(self, obj_id, image, mask=None):
        """
        Save image of detected object.
        
        Args:
            obj_id: Object ID (used as filename)
            image: Full frame image (BGR)
            mask: Optional object mask for cropping
        """
        if self.image_dir is None or image is None:
            return
        
        try:
            filename = os.path.join(self.image_dir, f"{obj_id}.jpg")
            
            if mask is not None:
                # Crop to bounding box of mask
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    # Add padding
                    pad = 20
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    w = min(image.shape[1] - x, w + 2*pad)
                    h = min(image.shape[0] - y, h + 2*pad)
                    cropped = image[y:y+h, x:x+w]
                    cv2.imwrite(filename, cropped)
                else:
                    cv2.imwrite(filename, image)
            else:
                cv2.imwrite(filename, image)
            
            print(f"[LOGGER] Saved image: {filename}")
        except Exception as e:
            print(f"[LOGGER] Image save error: {e}")
    
    def log_detection(self, obj_id, camera_class, camera_conf, spectrum_class, spectrum_conf, final_class, spectrum_raw=None):
        """
        Log a detection result. Generates 30-column format internally.
        
        Args:
            obj_id: Object ID
            camera_class: Camera/YOLO detected class name
            camera_conf: Camera/YOLO confidence (0-1 or 0-100)
            spectrum_class: Spectrum detected class name (or "N/A")
            spectrum_conf: Spectrum confidence (0-1 or 0-100)
            final_class: Final class from Bayesian fusion voting
            spectrum_raw: Raw 18-channel spectral data (optional)
        """
        if not self.writer:
            return
        
        # Normalize confidence to 0-1 range
        cam_conf = camera_conf / 100.0 if camera_conf > 1 else camera_conf
        spec_conf = spectrum_conf / 100.0 if spectrum_conf > 1 else spectrum_conf
        
        # Generate camera probabilities (detected class gets confidence, others split remaining)
        other_cam = (1.0 - cam_conf) / 3.0 if cam_conf < 1.0 else 0.0
        camera_cols = []
        for cls in self.CLASSES:
            if cls == camera_class:
                camera_cols.append(round(cam_conf, 4))
            else:
                camera_cols.append(round(other_cam, 4))
        
        # Generate spectrum probabilities
        if spectrum_class == "N/A" or spectrum_class is None:
            spectrum_cols = [0.0, 0.0, 0.0, 0.0]
            spectrum_class = "N/A"
        else:
            other_spec = (1.0 - spec_conf) / 3.0 if spec_conf < 1.0 else 0.0
            spectrum_cols = []
            for cls in self.CLASSES:
                if cls == spectrum_class:
                    spectrum_cols.append(round(spec_conf, 4))
                else:
                    spectrum_cols.append(round(other_spec, 4))
        
        # Prepare raw spectral data (18 channels, or zeros if not available)
        if spectrum_raw is not None and len(spectrum_raw) == 18:
            raw_cols = [round(v, 4) for v in spectrum_raw]
        else:
            raw_cols = [0.0] * 18
        
        # Write row: 12 classification columns + 18 spectral raw values = 30 columns
        row = [obj_id] + camera_cols + [camera_class] + spectrum_cols + [spectrum_class] + [final_class] + raw_cols
        self.writer.writerow(row)
        self.file_handle.flush()
        
        print(f"[LOGGER] Logged ID:{obj_id} Cam={camera_class}({cam_conf:.2f}) Spec={spectrum_class} Final={final_class}")
    
    def close(self):
        """Close the log file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            self.writer = None
            print(f"[LOGGER] Log file closed: {self.log_file}")


class DatabaseManager:
    """SQLite database manager for logging detections."""
    
    def __init__(self, db_file="system_data.db"):
        self.db_file = db_file
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self.create_table()
        except sqlite3.Error as e:
            print(f"[DATABASE] Error: {e}")
    
    def create_table(self):
        """Create detections table if not exists."""
        if not self.conn:
            return
        
        try:
            cols = """
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                object_id INTEGER,
                final_class TEXT,
                vision_class TEXT,
                spectrum_class TEXT,
                height_cm REAL,
                belt_x_cm REAL,
                belt_y_cm REAL,
                status TEXT,
                timestamp DATETIME
            """
            
            # Add spectrum channel columns
            for ch in SENSOR_CHANNELS_SPECTRAL:
                cols += f", '{ch}' REAL"
            cols += ", spectrum_confidence REAL"
            
            self.conn.cursor().execute(f"CREATE TABLE IF NOT EXISTS detections ({cols});")
            self._migrate_detection_schema()
            self.conn.commit()
        except Exception as e:
            print(f"[DATABASE] Create table error: {e}")

    def _migrate_detection_schema(self):
        """Backfill columns missing from older SQLite files."""
        cursor = self.conn.cursor()
        rows = cursor.execute("PRAGMA table_info(detections)").fetchall()
        existing = {row[1] for row in rows}
        required_columns = {
            "belt_x_cm": "REAL",
            "belt_y_cm": "REAL",
        }
        for name, col_type in required_columns.items():
            if name not in existing:
                cursor.execute(f"ALTER TABLE detections ADD COLUMN {name} {col_type}")
    
    def log_detection(self, session_id, obj, spectrum_data=None, spec_pred=None, spec_conf=None):
        """
        Log a detection/sorting event.
        
        Args:
            session_id: Session identifier
            obj: Object dict with 'id', 'class_name', 'height_cm', 'status', etc.
            spectrum_data: Raw spectral readings (list of 18 values)
            spec_pred: Spectrum prediction class
            spec_conf: Spectrum prediction confidence
        """
        if not self.conn:
            return
        
        try:
            keys = [
                "session_id", "object_id", "final_class", "vision_class",
                "spectrum_class", "height_cm", "belt_x_cm", "belt_y_cm",
                "status", "timestamp"
            ]
            vals = [
                session_id,
                obj.get('id', 0),
                obj.get('class_name', 'Unknown'),
                obj.get('vision_class_name', obj.get('class_name', 'Unknown')),
                spec_pred if spec_pred else "N/A",
                obj.get('height_cm', 0),
                obj.get('belt_x_cm', 0),
                obj.get('belt_y_cm', 0),
                obj.get('status', 'Unknown'),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
            
            # Add spectrum data columns
            if spectrum_data and len(spectrum_data) == 18:
                keys += [f"'{c}'" for c in SENSOR_CHANNELS_SPECTRAL]
                vals += spectrum_data
            else:
                keys += [f"'{c}'" for c in SENSOR_CHANNELS_SPECTRAL]
                vals += [0.0] * 18
            
            keys += ["spectrum_confidence"]
            vals += [spec_conf if spec_conf else 0.0]
            
            q_marks = ",".join(["?"] * len(keys))
            col_names = ",".join(keys)
            
            self.conn.cursor().execute(
                f"INSERT INTO detections ({col_names}) VALUES ({q_marks})", vals
            )
            self.conn.commit()
        except Exception as e:
            print(f"[DATABASE] Log error: {e}")
    
    def get_session_stats(self, session_id):
        """Get sorting statistics for a session."""
        if not self.conn:
            return {}
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT final_class, COUNT(*) FROM detections WHERE session_id=? GROUP BY final_class",
                (session_id,)
            )
            return dict(cursor.fetchall())
        except:
            return {}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
