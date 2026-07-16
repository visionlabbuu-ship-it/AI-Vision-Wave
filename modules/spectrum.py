"""
=============================================================================
SPECTRUM MODULE
=============================================================================
AS7265X spectral sensor management and ML prediction.
"""

import os
import numpy as np
import warnings
import threading
warnings.filterwarnings("ignore")

from .config import BASE_MODEL_PATH, SENSOR_CHANNELS_SPECTRAL

# Try to import sensor library
AS7265X_AVAILABLE = False
try:
    from as7265x_sparkfun_python import (
        AS7265X, AS7265X_GAIN_64X, AS7265X_LED_CURRENT_LIMIT_12_5MA,
        AS7265x_LED_WHITE, AS7265x_LED_IR, AS7265x_LED_UV
    )
    AS7265X_AVAILABLE = True
except ImportError:
    pass

# Try to import ML libraries
ML_AVAILABLE = False
try:
    import joblib
    import pandas as pd
    ML_AVAILABLE = True
except ImportError:
    pass


class SpectrumManager:
    """
    Manager for AS7265X spectral sensor and ML classification.
    """
    
    def __init__(self, log_func=None):
        self.sensor = None
        self.log = log_func or print
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.catboost_model = None
        self.is_ready = False
        self.hardware_ready = False
        self._sensor_configured = False
        self._read_lock = threading.Lock()

    def ensure_leds_off(self):
        """Force all AS7265x bulbs OFF (WHITE/IR/UV)."""
        if not self.sensor:
            return
        try:
            self.sensor.disableBulb(AS7265x_LED_WHITE)
            self.sensor.disableBulb(AS7265x_LED_IR)
            self.sensor.disableBulb(AS7265x_LED_UV)
        except Exception:
            # Keep this silent — caller may use it as a safety cleanup path.
            pass
    
    def initialize_hardware(self):
        """Initialize the AS7265X sensor."""
        if not AS7265X_AVAILABLE:
            self.log("[SPECTRUM] Sensor library not available")
            return False
        
        try:
            self.sensor = AS7265X(i2c_bus=1)
            if self.sensor.begin():
                self.sensor.disableIndicator()
                self.ensure_leds_off()
                self.hardware_ready = True
                self.log("[SPECTRUM] Sensor connected")
                return True
            else:
                self.log("[SPECTRUM] Sensor not detected")
                return False
        except Exception as e:
            self.log(f"[SPECTRUM] Hardware init error: {e}")
            return False
    
    def load_models(self, model_path=BASE_MODEL_PATH):
        """Load ML models for spectrum classification."""
        if not ML_AVAILABLE:
            self.log("[SPECTRUM] ML libraries not available")
            return False
        
        try:
            # Load CatBoost model
            cb_path = os.path.join(model_path, "CatBoost.joblib")
            if os.path.exists(cb_path):
                self.catboost_model = joblib.load(cb_path)
                self.log("[SPECTRUM] CatBoost model loaded")
            
            # Load scaler and label encoder
            sc_path = os.path.join(model_path, "scaler.joblib")
            le_path = os.path.join(model_path, "label_encoder.joblib")
            
            if os.path.exists(sc_path) and os.path.exists(le_path):
                self.scaler = joblib.load(sc_path)
                self.label_encoder = joblib.load(le_path)
                self.is_ready = True
                self.log("[SPECTRUM] ML assets loaded")
                return True
            
            return False
        except Exception as e:
            self.log(f"[SPECTRUM] Model load error: {e}")
            return False
    
    def disconnect_hardware(self):
        """Disconnect the AS7265X sensor (close I2C bus)."""
        if self.sensor and self.hardware_ready:
            try:
                # Disable all LEDs before closing
                self.ensure_leds_off()
                # Close the I2C bus
                if hasattr(self.sensor, 'bus') and self.sensor.bus:
                    self.sensor.bus.close()
                self.log("[SPECTRUM] Sensor disconnected")
            except Exception as e:
                self.log(f"[SPECTRUM] Disconnect error: {e}")
        self.sensor = None
        self.hardware_ready = False
        self.is_ready = False
        self._sensor_configured = False

    def _configure_sensor_once(self):
        """Set gain / integration / bulb-current once (not every read)."""
        if self._sensor_configured:
            return
        try:
            self.sensor.setGain(AS7265X_GAIN_64X)
            self.sensor.setIntegrationCycles(25)
            self.sensor.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_12_5MA, AS7265x_LED_IR)
            self._sensor_configured = True
            self.log("[SPECTRUM] Sensor configured (gain=64X, cycles=25)")
        except Exception as e:
            self.log(f"[SPECTRUM] Config error: {e}")

    def read_sensor(self):
        """Read spectral data from sensor.
        
        Optimised path:
          1) Configure gain/integration/bulb ONCE (cached).
          2) Force all bulbs OFF, then enable IR for one-shot measurement.
          3) Force all bulbs OFF again (same behaviour as Spectrum_Control.py).
          4) Batch-read all 18 calibrated channels with minimal I2C overhead.
        """
        if not self.hardware_ready or not self.sensor:
            return None
        
        with self._read_lock:
            try:
                self._configure_sensor_once()

                # Match Spectrum_Control.py sequence exactly:
                # always clear WHITE/UV/IR state before taking a reading.
                self.ensure_leds_off()

                self.sensor.enableBulb(AS7265x_LED_IR)
                self.sensor.takeMeasurements()
                self.sensor.disableBulb(AS7265x_LED_IR)

                # Ensure non-IR bulbs remain OFF after every read.
                self.ensure_leds_off()

                # Batch read: returns [A,B,C,D,E,F, G,H,I,J,K,L, R,S,T,U,V,W]
                vals = self.sensor.getAllCalibratedValues()
                A, B, C, D, E, F = vals[0:6]    # UV
                G, H, I, J, K, L = vals[6:12]   # VIS
                R, S, T, U, V, W = vals[12:18]  # NIR

                # Return in original channel order expected by ML model
                return [A, B, C, D, E, F, G, H, R, I, S, J, T, U, V, W, K, L]
            except Exception as e:
                self.log(f"[SPECTRUM] Read error: {e}")
                return None
            finally:
                # Safety net: never leave any bulb ON if an exception occurred mid-read.
                self.ensure_leds_off()
    
    def predict(self, raw_data):
        """
        Predict material class from spectral data.
        
        Returns:
            tuple: (predicted_class, confidence, scaled_data, class_probabilities)
                   class_probabilities is a dict: {"Glass": 0.1, "Plastic": 0.8, ...}
        """
        if not self.is_ready or not raw_data:
            return None, 0.0, None, None
        
        try:
            X_unscaled = np.array(raw_data).reshape(1, -1)
            X_df = pd.DataFrame(X_unscaled, columns=SENSOR_CHANNELS_SPECTRAL)
            X_scaled = self.scaler.transform(X_df)
            
            pred_idx = self.catboost_model.predict(X_scaled).flatten()[0]
            pred_class = self.label_encoder.inverse_transform([int(pred_idx)])[0]
            
            proba = self.catboost_model.predict_proba(X_scaled)[0]  # Get first row
            confidence = np.max(proba) * 100
            
            # Create probability dict for all classes
            all_classes = self.label_encoder.classes_
            class_probs = {cls: float(proba[i]) for i, cls in enumerate(all_classes)}
            
            return pred_class, confidence, X_scaled, class_probs
        except Exception as e:
            self.log(f"[SPECTRUM] Predict error: {e}")
            return "Error", 0.0, None, None
