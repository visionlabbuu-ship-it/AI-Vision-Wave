"""
=============================================================================
CAMERA MODULE
=============================================================================
RealSense camera streaming in a separate thread.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import queue
import time
import os

from .config import IMAGE_WIDTH, IMAGE_HEIGHT, FPS


class CameraStream:
    """Threaded RealSense camera stream with depth alignment."""
    
    def __init__(self, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fps=FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.depth_intrinsics = None
        self.connected = False
        self.align = None
        self._thread = None

    def start_camera(self):
        """Initialize and start the camera."""
        try:
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.pipeline = rs.pipeline()
            profile = self.pipeline.start(self.config)
            self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            self.align = rs.align(rs.stream.color)
            self.connected = True
            self.running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"[CAMERA] Error: {e}")
            return False

    def _run(self):
        """Main camera loop - runs in separate thread."""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                timestamp = time.time()
                
                # Drop old frames if queue is full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put((color_image, depth_frame, timestamp))
            except Exception as e:
                if self.running:
                    print(f"[CAMERA] Frame error: {e}")
                time.sleep(0.01)

    def get_latest(self):
        """Get the latest frame (non-blocking)."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_frame(self, timeout=1.0):
        """Get frame with timeout (blocking)."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def start_recording(self, bag_path):
        """Restart the pipeline with .bag recording enabled.
        Returns True on success."""
        if not self.connected:
            return False
        try:
            self.running = False
            time.sleep(0.1)  # Let thread drain
            self.pipeline.stop()

            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.config.enable_record_to_file(bag_path)
            profile = self.pipeline.start(self.config)
            self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            self.align = rs.align(rs.stream.color)

            self.running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"[CAMERA] start_recording error: {e}")
            return False

    def stop_recording(self):
        """Stop .bag recording by restarting the pipeline without recording.
        Returns True on success."""
        if not self.connected:
            return False
        try:
            self.running = False
            time.sleep(0.1)
            self.pipeline.stop()

            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            profile = self.pipeline.start(self.config)
            self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            self.align = rs.align(rs.stream.color)

            self.running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"[CAMERA] stop_recording error: {e}")
            return False

    def stop(self):
        """Stop the camera stream."""
        self.running = False
        if self.connected:
            try:
                self.pipeline.stop()
            except:
                pass
        self.connected = False


class VideoPlaybackStream:
    """
    Plays a .bag recording with the same interface as CameraStream.
    
    Provides get_latest(), connected, depth_intrinsics, and stop() so it can
    be swapped in wherever CameraStream is used.
    """

    def __init__(self, bag_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, real_time=True):
        self.bag_path = bag_path
        self.width = width
        self.height = height
        self.real_time = real_time  # True = play at original FPS, False = as fast as possible
        self.running = False
        self.connected = False
        self.depth_intrinsics = None
        self.pipeline = rs.pipeline()
        self.align = None
        self._thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.loop = True           # Loop playback by default
        self.loop_count = 0
        self.frame_count = 0

    def start_camera(self):
        """Open the .bag file and start streaming frames (same API as CameraStream)."""
        try:
            config = rs.config()
            config.enable_device_from_file(self.bag_path, repeat_playback=False)
            profile = self.pipeline.start(config)

            # Real-time = original FPS (for production), False = max speed (for test program)
            device = profile.get_device()
            playback = device.as_playback()
            playback.set_real_time(self.real_time)

            self.depth_intrinsics = (
                profile.get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            self.align = rs.align(rs.stream.color)
            self.connected = True
            self.running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            print(f"[VIDEO] Opened: {os.path.basename(self.bag_path)}")
            return True
        except Exception as e:
            print(f"[VIDEO] Error opening {self.bag_path}: {e}")
            return False

    def _run(self):
        """Background thread that feeds frames into the queue."""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                timestamp = time.time()
                self.frame_count += 1

                # Drop old frames if queue is full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.frame_queue.put((color_image, depth_frame, timestamp))

            except RuntimeError:
                # End of file
                if self.loop and self.running:
                    self.loop_count += 1
                    print(f"[VIDEO] Looping playback (loop #{self.loop_count})")
                    try:
                        device = self.pipeline.get_active_profile().get_device()
                        playback = device.as_playback()
                        playback.seek(rs.frame())
                    except Exception as e:
                        print(f"[VIDEO] Loop seek failed: {e}")
                        self.running = False
                else:
                    print("[VIDEO] Playback ended.")
                    self.running = False
            except Exception as e:
                if self.running:
                    print(f"[VIDEO] Frame error: {e}")
                time.sleep(0.01)

    def get_latest(self):
        """Get the latest frame (non-blocking). Same API as CameraStream."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_frame(self, timeout=1.0):
        """Get frame with timeout (blocking)."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the playback stream."""
        self.running = False
        if self.connected:
            try:
                self.pipeline.stop()
            except:
                pass
        self.connected = False
        print(f"[VIDEO] Stopped. Played {self.frame_count} frames, {self.loop_count} loops.")


class _FakeDepthFrame:
    """
    Minimal depth-frame shim for MP4 playback.
    Provides the subset of RealSense depth API used by detector.py:
      - get_data()      -> uint16 depth image in millimeters
      - get_distance(x, y) -> depth at pixel in meters
    """

    def __init__(self, depth_mm):
        self._depth_mm = depth_mm

    def get_data(self):
        return self._depth_mm

    def get_distance(self, x, y):
        h, w = self._depth_mm.shape[:2]
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        return float(self._depth_mm[yi, xi]) * 0.001


class MP4PlaybackStream:
    """
    MP4 playback stream with CameraStream-compatible interface.

    Returns tuples: (color_image_bgr, fake_depth_frame, timestamp)
    so the rest of the pipeline can run unchanged.
    """

    def __init__(self, mp4_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
        self.mp4_path = mp4_path
        self.width = width
        self.height = height
        self.running = False
        self.connected = False
        self.depth_intrinsics = None  # No true depth intrinsics for MP4
        self._thread = None
        # Keep queue small to avoid latency buildup (real-time sync first).
        self.frame_queue = queue.Queue(maxsize=2)
        self.loop = True
        self.loop_count = 0
        self.max_passes = None      # None = unlimited, otherwise total passes to play
        self.current_pass_index = 1
        self.frame_count = 0
        self.cap = None
        self.source_fps = float(FPS)
        self.frame_interval_s = 1.0 / max(1.0, float(FPS))
        self._playback_time_s = 0.0
        self._media_start_s = None

        # Default synthetic depth map: flat 1.0m plane
        self._depth_mm_template = np.full((self.height, self.width), 1000, dtype=np.uint16)

    def set_depth_template_from_meters(self, depth_m_map):
        """
        Set synthetic depth template from a meter-map (HxW float).
        Used to align fake depth with calibrated floor, so baseline height ~= 0cm.
        """
        if depth_m_map is None:
            return
        try:
            d = np.array(depth_m_map, dtype=np.float32)
            if d.shape[:2] != (self.height, self.width):
                d = cv2.resize(d, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            d_mm = np.clip(d * 1000.0, 100, 10000).astype(np.uint16)
            self._depth_mm_template = d_mm
        except Exception as e:
            print(f"[VIDEO] set_depth_template_from_meters error: {e}")

    def start_camera(self):
        """Open MP4 file and start feeding frames."""
        try:
            self.cap = cv2.VideoCapture(self.mp4_path)
            if not self.cap.isOpened():
                print(f"[VIDEO] Error opening MP4: {self.mp4_path}")
                return False

            # Read source FPS from file for real-time pacing
            src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if src_fps <= 1e-3 or src_fps > 240.0:
                src_fps = float(FPS)
            self.source_fps = src_fps
            self.frame_interval_s = 1.0 / max(1e-6, self.source_fps)
            self._playback_time_s = 0.0
            self.current_pass_index = 1

            self.connected = True
            self.running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            print(f"[VIDEO] Opened MP4: {os.path.basename(self.mp4_path)} @ {self.source_fps:.2f} FPS")
            return True
        except Exception as e:
            print(f"[VIDEO] Error opening MP4 {self.mp4_path}: {e}")
            return False

    def _run(self):
        # Pace frames to source media time; drop stale queued frames to stay real-time.
        wall_start_mono = time.monotonic()
        wall_start_epoch = time.time()
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.running = False
                    break

                ok, frame = self.cap.read()
                if not ok or frame is None:
                    can_loop = self.loop and self.running
                    if self.max_passes is not None:
                        can_loop = can_loop and ((self.loop_count + 1) < int(self.max_passes))
                    if can_loop:
                        self.loop_count += 1
                        self.current_pass_index = self.loop_count + 1
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._playback_time_s = 0.0
                        self._media_start_s = None
                        wall_start_mono = time.monotonic()
                        wall_start_epoch = time.time()
                        print(f"[VIDEO] Looping MP4 playback (loop #{self.loop_count})")
                        time.sleep(0.01)
                        continue
                    self.running = False
                    break

                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

                fake_depth = _FakeDepthFrame(self._depth_mm_template)
                # Prefer media timeline when available (phone/VFR videos included).
                media_time_s = None
                pos_msec = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if pos_msec > 0.0:
                    media_time_s = pos_msec / 1000.0
                else:
                    media_time_s = self._playback_time_s
                    self._playback_time_s += self.frame_interval_s

                if self._media_start_s is None:
                    self._media_start_s = media_time_s
                rel_media_s = max(0.0, float(media_time_s) - float(self._media_start_s))
                timestamp = wall_start_epoch + rel_media_s
                self.frame_count += 1

                # Keep the newest frame only when consumer is late.
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self.frame_queue.put_nowait((frame, fake_depth, timestamp))
                except queue.Full:
                    pass

                # Sleep until media-clock target wall time.
                target_mono = wall_start_mono + rel_media_s
                sleep_s = target_mono - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
            except Exception as e:
                if self.running:
                    print(f"[VIDEO] MP4 frame error: {e}")
                time.sleep(0.01)

    def get_latest(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.connected = False
        print(f"[VIDEO] MP4 stopped. Played {self.frame_count} frames, {self.loop_count} loops.")


class MP4LiveDepthPlaybackStream:
    """
    Hybrid playback stream:
    - Color frames from MP4
    - Depth frames from a live RealSense CameraStream

    Keeps the main pipeline running with real depth sensor IO while replaying MP4 color.
    Returns tuples: (color_image_bgr, depth_like_frame, timestamp)
    """

    def __init__(self, mp4_path, live_depth_stream, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
        self.mp4_path = mp4_path
        self.live_depth_stream = live_depth_stream
        self.width = width
        self.height = height
        self.running = False
        self.connected = False
        self.depth_intrinsics = None
        self.frame_queue = queue.Queue(maxsize=2)
        self._thread = None

        self.loop = True
        self.loop_count = 0
        self.max_passes = None
        self.current_pass_index = 1
        self.frame_count = 0

        self._video = MP4PlaybackStream(mp4_path, width=width, height=height)
        self._last_depth_mm = np.full((self.height, self.width), 1000, dtype=np.uint16)

    def start_camera(self):
        """Start MP4 color playback + ensure live depth stream is running."""
        try:
            if self.live_depth_stream is None:
                print("[VIDEO] Live depth stream is None")
                return False

            if not self.live_depth_stream.connected:
                if not self.live_depth_stream.start_camera():
                    print("[VIDEO] Failed to start live depth camera for MP4 replay")
                    return False

            self.depth_intrinsics = self.live_depth_stream.depth_intrinsics

            self._video.loop = self.loop
            self._video.max_passes = self.max_passes
            if not self._video.start_camera():
                return False

            self.running = True
            self.connected = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            print(f"[VIDEO] MP4+LiveDepth started: {os.path.basename(self.mp4_path)}")
            return True
        except Exception as e:
            print(f"[VIDEO] MP4+LiveDepth start error: {e}")
            return False

    def _run(self):
        while self.running:
            try:
                frame = self._video.get_frame(timeout=0.5)
                if frame is None:
                    if not self._video.running:
                        self.running = False
                        break
                    continue

                color_image, _unused_depth, timestamp = frame

                # Pull latest live depth sample and convert to a stable depth-like frame.
                live = self.live_depth_stream.get_latest()
                if live is not None:
                    _live_color, live_depth, _live_ts = live
                    depth_mm = np.asanyarray(live_depth.get_data())
                    if depth_mm.shape[:2] != (self.height, self.width):
                        depth_mm = cv2.resize(depth_mm, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    if depth_mm.dtype != np.uint16:
                        depth_mm = depth_mm.astype(np.uint16)
                    self._last_depth_mm = depth_mm.copy()
                    if self.live_depth_stream.depth_intrinsics is not None:
                        self.depth_intrinsics = self.live_depth_stream.depth_intrinsics

                mixed_depth = _FakeDepthFrame(self._last_depth_mm)

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self.frame_queue.put_nowait((color_image, mixed_depth, timestamp))
                except queue.Full:
                    pass

                self.frame_count = self._video.frame_count
                self.loop_count = self._video.loop_count
                self.current_pass_index = self._video.current_pass_index
            except Exception as e:
                if self.running:
                    print(f"[VIDEO] MP4+LiveDepth frame error: {e}")
                time.sleep(0.01)

        self.connected = False

    def get_latest(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        try:
            self._video.stop()
        except Exception:
            pass
        self.connected = False
        print(f"[VIDEO] MP4+LiveDepth stopped. Played {self.frame_count} frames, {self.loop_count} loops.")
