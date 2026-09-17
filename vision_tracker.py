"""
vision_tracker.py
=================
High-Performance Threaded Computer Vision and Hand Tracking Module.

Key Features:
- ThreadedCamera: Dedicated background worker thread continuously capturing frames
  from cv2.VideoCapture at maximum FPS to eliminate I/O blocking.
- HandTracker:
  * Wraps Google MediaPipe Hands with min_detection_confidence=0.55, min_tracking_confidence=0.50.
  * Raw RGB frame processing paired with mirrored coordinate transformation:
    X_screen = (1.0 - x) * Width.
  * Explicit Handedness label swapping ('Left' <-> 'Right') for intuitive mirrored AR.
  * Pluggable BaseFilter smoothing (OneEuroFilter default, DeadbandFilter, EMAFilter, RawFilter).
  * Instantaneous fingertip velocity tracking (dY/dt) for piano key press velocity gating.
- AsyncHandTracker:
  * Decouples AI inference from UI render loop with independent snapshot lock to eliminate contention.
  * Predictive kinematic dead-reckoning extrapolation for smooth 60+ FPS display.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger("VisionTracker")


@dataclass
class HandData:
    """Structured container for tracked hand landmarks and kinematic properties."""
    handedness: str  # 'Left' or 'Right' (aligned with mirrored display)
    landmarks_norm: np.ndarray  # (21, 3) float32 in [0, 1] screen coordinates
    landmarks_px: np.ndarray    # (21, 3) float32 in pixel coordinates
    fingertip_velocities: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    wrist_velocity: Tuple[float, float] = (0.0, 0.0)
    timestamp: float = 0.0
    # Landmark indices for fingertips: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)


class ThreadedCamera:
    """
    Dedicated background thread for OpenCV VideoCapture.
    Decouples frame acquisition from inference and rendering to maintain a stable 60 FPS.
    """

    def __init__(self, src: int = 0, width: int = 1920, height: int = 1080) -> None:
        self.src = src
        self.target_width = width
        self.target_height = height

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.ret: bool = False
        self.is_running: bool = False
        self.is_simulation: bool = False

        self.camera_fps: float = 0.0
        self._frame_count: int = 0
        self._last_fps_calc: float = time.perf_counter()

        # Frame sequencing & telemetry
        self.frame_id: int = 0
        self.frame_timestamp: float = 0.0
        self._current_frame_id: int = 0
        self._current_timestamp: float = 0.0

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._sim_phase: float = 0.0

    def start(self) -> ThreadedCamera:
        """Initializes camera hardware and launches the worker thread."""
        logger.info("Initializing VideoCapture on source %s...", self.src)
        # Try default native backend first (optimal on Windows MSMF)
        self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened() and hasattr(cv2, 'CAP_DSHOW'):
            # Fallback to DirectShow if native backend fails
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)

        if self.cap.isOpened():
            # Configure high-definition resolution and 60 FPS hardware capture
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Retrieve first warmup frame and detect actual hardware resolution
            self.ret, self.frame = self.cap.read()
            if self.ret and self.frame is not None:
                actual_h, actual_w = self.frame.shape[:2]
                self.target_width = actual_w
                self.target_height = actual_h
                logger.info("Camera successfully opened at crisp %dx%d resolution", actual_w, actual_h)
            else:
                logger.info("Camera opened: %dx%d", self.target_width, self.target_height)
            self.is_simulation = False
        else:
            logger.warning("No camera available on source %s. Falling back to synthetic simulation mode.", self.src)
            self.is_simulation = True
            self.ret = True
            self.frame = self._generate_simulation_frame()

        self.is_running = True
        self._last_fps_calc = time.perf_counter()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="ThreadedCameraWorker")
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        """Background thread target continuously grabbing fresh frames with zero hardware backlog."""
        while self.is_running:
            if not self.is_simulation and self.cap is not None and self.cap.isOpened():
                # Flush any queued driver frames and retrieve strictly the latest frame
                if self.cap.grab():
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None:
                        now = time.perf_counter()
                        self._frame_count += 1
                        elapsed = now - self._last_fps_calc
                        if elapsed >= 0.5:
                            self.camera_fps = self._frame_count / elapsed
                            self._frame_count = 0
                            self._last_fps_calc = now

                        self.frame_id += 1
                        self.frame_timestamp = now
                        with self._lock:
                            self.ret = ret
                            self.frame = frame
                            self._current_frame_id = self.frame_id
                            self._current_timestamp = now
                    else:
                        time.sleep(0.001)
                else:
                    time.sleep(0.001)
            else:
                # Simulation mode frame generation at ~60 FPS
                time.sleep(0.016)
                sim_frame = self._generate_simulation_frame()
                now = time.perf_counter()
                self.frame_id += 1
                self.frame_timestamp = now
                with self._lock:
                    self.ret = True
                    self.frame = sim_frame
                    self._current_frame_id = self.frame_id
                    self._current_timestamp = now

    def _generate_simulation_frame(self) -> np.ndarray:
        """Generates a cyber-grid synthetic frame when no hardware camera is present."""
        w, h = self.target_width, self.target_height
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Subtle dark cyber background
        canvas[:] = (20, 16, 24)

        # Animated scan line
        self._sim_phase = (self._sim_phase + 0.03) % (2.0 * np.pi)
        scan_y = int((np.sin(self._sim_phase) * 0.4 + 0.5) * h)
        cv2.line(canvas, (0, scan_y), (w, scan_y), (80, 50, 100), 1)

        # Grid lines
        for x in range(0, w, 80):
            cv2.line(canvas, (x, 0), (x, h), (35, 28, 42), 1)
        for y in range(0, h, 80):
            cv2.line(canvas, (0, y), (w, y), (35, 28, 42), 1)

        cv2.putText(
            canvas,
            "CAMERA SIMULATION MODE (Hardware Not Detected)",
            (w // 2 - 270, h // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Connect a USB Webcam or test keyboard controls",
            (w // 2 - 210, h // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def open_settings_dialog(self) -> None:
        """Opens native Windows Camera Properties dialog to configure 50/60Hz, exposure, gain, etc."""
        if self.cap is not None and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_SETTINGS, 1.0)
                logger.info("Triggered Windows Camera Properties dialog.")
            except Exception as e:
                logger.warning("Could not open hardware camera settings dialog: %s", e)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Safely returns the most recently captured frame without redundant copying."""
        with self._lock:
            if self.frame is not None:
                return self.ret, self.frame
            return False, None

    def read_sequenced(self) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """Returns (ret, frame, frame_id, timestamp) for sequenced stale-frame skipping."""
        with self._lock:
            if self.frame is not None:
                return self.ret, self.frame, self._current_frame_id, self._current_timestamp
            return False, None, 0, 0.0

    def stop(self) -> None:
        """Gracefully halts the background thread and releases hardware resources."""
        self.is_running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.debug("Error releasing VideoCapture: %s", e)
            finally:
                self.cap = None
        logger.info("ThreadedCamera stopped.")


# =============================================================================
# Abstract Filter Hierarchy & Implementations
# =============================================================================

class BaseFilter(ABC):
    """Abstract base class for hand landmark coordinate smoothing filters."""

    @abstractmethod
    def filter(self, x: np.ndarray, timestamp: float = 0.0) -> np.ndarray:
        """Filters coordinate array x of shape (N, D) given timestamp."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets internal filter history."""
        pass


class RawFilter(BaseFilter):
    """Pure pass-through filter with zero modification (0ms delay baseline)."""

    def filter(self, x: np.ndarray, timestamp: float = 0.0) -> np.ndarray:
        return x.copy()

    def reset(self) -> None:
        pass


class EMAFilter(BaseFilter):
    """
    First-order Exponential Moving Average (EMA) low-pass filter.
    s_t = alpha * x_t + (1 - alpha) * s_{t-1}
    """

    def __init__(self, alpha: float = 0.65) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.prev_val: Optional[np.ndarray] = None

    def filter(self, x: np.ndarray, timestamp: float = 0.0) -> np.ndarray:
        if not np.all(np.isfinite(x)):
            return x.copy()
        if self.prev_val is None or not np.all(np.isfinite(self.prev_val)):
            self.prev_val = x.copy()
            return x.copy()
        filtered = self.alpha * x + (1.0 - self.alpha) * self.prev_val
        self.prev_val = filtered.copy()
        return filtered

    def reset(self) -> None:
        self.prev_val = None


class LowPassFilter:
    """Helper first-order low-pass filter for multi-dimensional numpy arrays."""

    def __init__(self) -> None:
        self.prev_val: Optional[np.ndarray] = None

    def filter(self, val: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
        if self.prev_val is None:
            self.prev_val = val.copy()
            return val.copy()
        filtered = alpha * val + (1.0 - alpha) * self.prev_val
        self.prev_val = filtered.copy()
        return filtered

    def reset(self) -> None:
        self.prev_val = None


class DeadbandFilter(BaseFilter):
    """
    Adaptive Deadband (Noise-Gate) Filter for Real-Time AR Hand Rigging.

    Design Characteristics:
    - Designed for near-zero phase lag during rapid motion (direct 1:1 pass-through above motion threshold).
    - Attenuates coordinate drift / camera sensor noise when stationary below deadband threshold.
    """

    def __init__(
        self,
        deadband_norm: float = 0.0025,       # ~3.2px on 1280x720: cancels sensor noise
        motion_threshold_norm: float = 0.0080, # ~10.0px on 1280x720: full 1:1 raw pass-through
    ) -> None:
        self.deadband = float(deadband_norm)
        self.motion_threshold = float(motion_threshold_norm)
        self.prev_val: Optional[np.ndarray] = None

    def filter(self, x: np.ndarray, timestamp: float = 0.0) -> np.ndarray:
        if not np.all(np.isfinite(x)):
            return x.copy()
        if self.prev_val is None or not np.all(np.isfinite(self.prev_val)):
            self.prev_val = x.copy()
            return x.copy()

        # Euclidean distance per landmark in normalized coords (N, 1)
        delta = np.linalg.norm(x[:, :2] - self.prev_val[:, :2], axis=1, keepdims=True)

        # Dynamic blend via smoothstep Hermite curve
        denom = max(1e-5, self.motion_threshold - self.deadband)
        t = np.clip((delta - self.deadband) / denom, 0.0, 1.0)
        alpha = t * t * (3.0 - 2.0 * t)

        filtered = alpha * x + (1.0 - alpha) * self.prev_val
        self.prev_val = filtered.copy()
        return filtered

    def reset(self) -> None:
        self.prev_val = None


# Backward-compatible alias
ZeroLagDeadbandFilter = DeadbandFilter


class OneEuroFilter(BaseFilter):
    """
    1€ Filter: Adaptive Low-Pass Filter for Human Motion & AR Rig Jitter Elimination.
    Reference: Casiez, Roussel, Vogel (CHI 2012).

    Parameters:
    - min_cutoff: Minimum cutoff frequency in Hz (default: 1.0 Hz). Controls jitter attenuation at rest.
    - beta: Speed coefficient (default: 30.0 for Normalized Device Coordinates [0, 1]).
      Reduces the smoothness/responsiveness trade-off by increasing cutoff frequency proportionally
      to tracking speed during faster motion. Typical rapid motions may drive the adaptive cutoff
      into the tens-of-Hz range, but the implementation has no configured upper cutoff unless max_cutoff is provided.
    - d_cutoff: Cutoff frequency for derivative filtering in Hz (default: 1.0 Hz).
    - max_cutoff: Optional upper bound on cutoff frequency in Hz (default: None, unbounded).
    - dt: Clamped to [1e-4 s, 1.0 s] for numerical stability against timing spikes.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 30.0,
        d_cutoff: float = 1.0,
        max_cutoff: Optional[float] = None,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.max_cutoff = float(max_cutoff) if max_cutoff is not None else None
        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.prev_time: Optional[float] = None

    def _compute_alpha(self, cutoff: np.ndarray, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * np.maximum(1e-4, cutoff))
        return np.clip(1.0 / (1.0 + tau / dt), 0.0, 1.0)

    def filter(self, x: np.ndarray, timestamp: float = 0.0) -> np.ndarray:
        if not np.all(np.isfinite(x)):
            return x.copy()

        if self.prev_time is None:
            self.prev_time = timestamp
            self.dx_filt.filter(np.zeros_like(x), 1.0)
            return self.x_filt.filter(x, 1.0)

        dt = float(np.clip(timestamp - self.prev_time, 1e-4, 1.0))
        self.prev_time = timestamp

        prev_x = self.x_filt.prev_val if self.x_filt.prev_val is not None else x
        dx = (x - prev_x) / dt

        alpha_d = self._compute_alpha(np.full_like(x, self.d_cutoff), dt)
        filtered_dx = self.dx_filt.filter(dx, alpha_d)

        # 2D Euclidean speed magnitude per landmark broadcast across all coordinate dimensions
        speed_2d = np.linalg.norm(filtered_dx[:, :2], axis=1, keepdims=True)
        speed = np.repeat(speed_2d, x.shape[1], axis=1)
        cutoff = self.min_cutoff + self.beta * speed
        if self.max_cutoff is not None:
            cutoff = np.minimum(cutoff, self.max_cutoff)

        alpha = self._compute_alpha(cutoff, dt)
        return self.x_filt.filter(x, alpha)

    def reset(self) -> None:
        self.x_filt.reset()
        self.dx_filt.reset()
        self.prev_time = None


class HandTracker:
    """
    MediaPipe Hands Wrapper with Mirrored Coordinates & Pluggable Filtering.

    Specifications:
    - max_num_hands=2, min_detection_confidence=0.55, min_tracking_confidence=0.50
    - Model complexity 0 (Lite) or 1 (Full).
    - Passes raw RGB to MediaPipe.
    - Flips X coordinate: X_screen = (1.0 - x) * Width.
    - Swaps Handedness label: 'Left' becomes 'Right' and vice versa.
    - Unified BaseFilter hierarchy (OneEuroFilter, DeadbandFilter, EMAFilter, RawFilter).
    - Computes fingertip velocity (dY/dt) for press threshold gating.
    """

    FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.50,
        ema_alpha: float = 0.65,
        filter_mode: str = "one_euro",  # "one_euro", "deadband", "ema", "raw"
        model_complexity: int = 1,      # 1 = Full (Ultra precision), 0 = Lite (Hyper speed)
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.ema_alpha = ema_alpha
        self.filter_mode = filter_mode
        self.model_complexity = model_complexity

        # Unified filter instances per hand label ('Left', 'Right')
        self._filters: Dict[str, BaseFilter] = {}
        self._smoothed_landmarks: Dict[str, np.ndarray] = {}
        self._prev_timestamps: Dict[str, float] = {}
        self._prev_fingertip_px: Dict[str, Dict[int, np.ndarray]] = {}
        self._prev_wrist_px: Dict[str, np.ndarray] = {}

        self.mp_hands = None
        self.hands = None

        if mp is not None and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            self.mp_hands = mp.solutions.hands
            self._init_mp_hands()
        else:
            logger.warning("MediaPipe library not available. Hand tracking will be simulated/disabled.")

    def _init_mp_hands(self) -> None:
        """Initializes or reconfigures MediaPipe Hands instance."""
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception as e:
                logger.debug("Error closing existing MediaPipe hands instance: %s", e)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        logger.info("MediaPipe Hands initialized (model_complexity=%d, filter_mode=%s)", self.model_complexity, self.filter_mode)

    def _create_filter_instance(self) -> BaseFilter:
        """Factory creating the appropriate BaseFilter instance for current filter_mode."""
        mode = self.filter_mode.lower()
        if mode == "raw":
            return RawFilter()
        elif mode == "ema":
            return EMAFilter(alpha=self.ema_alpha)
        elif mode in ("deadband", "zero_lag"):
            return DeadbandFilter()
        else:
            return OneEuroFilter(min_cutoff=1.0, beta=30.0, d_cutoff=1.0)

    def set_filter_mode(self, mode: str) -> None:
        """Dynamically switches filter mode and resets existing filter states."""
        self.filter_mode = mode
        self._filters.clear()
        logger.info("HandTracker filter mode switched to: %s", mode)

    def set_model_complexity(self, complexity: int) -> None:
        """Dynamically switches MediaPipe model complexity (0 = Lite, 1 = Full)."""
        if complexity not in (0, 1) or complexity == self.model_complexity:
            return
        self.model_complexity = complexity
        if self.mp_hands is not None:
            self._init_mp_hands()

    def process(self, raw_bgr_frame: np.ndarray) -> List[HandData]:
        """
        Processes a raw unmirrored BGR frame, runs MediaPipe, mirrors coordinates,
        swaps handedness, applies active filter, and calculates velocities.
        """
        if self.hands is None or raw_bgr_frame is None:
            return []

        h, w, _ = raw_bgr_frame.shape
        current_time = time.perf_counter()

        # Inference resolution scaling: maintain crisp fidelity up to 960px width
        if w > 960:
            infer_w = 960
            infer_h = int(round(960 * (h / w)))
            infer_bgr = cv2.resize(raw_bgr_frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        else:
            infer_bgr = raw_bgr_frame

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        try:
            results = self.hands.process(rgb_frame)
        except Exception as e:
            logger.debug("MediaPipe inference error or graph reset: %s", e)
            return []
        rgb_frame.flags.writeable = True

        tracked_hands: List[HandData] = []
        currently_detected_labels: List[str] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, classification in zip(results.multi_hand_landmarks, results.multi_handedness):
                raw_label = classification.classification[0].label  # 'Left' or 'Right'

                # Spec: Invert Handedness labels so user's physical left hand corresponds to mirrored screen left
                mirrored_label = "Right" if raw_label == "Left" else "Left"
                currently_detected_labels.append(mirrored_label)

                # Extract normalized coordinates with mirrored X: X_screen = (1.0 - x)
                raw_coords = np.empty((21, 3), dtype=np.float32)
                for idx, lm in enumerate(hand_landmarks.landmark):
                    raw_coords[idx, 0] = 1.0 - lm.x  # Mirrored horizontal coordinate
                    raw_coords[idx, 1] = lm.y
                    raw_coords[idx, 2] = lm.z

                # Tracking Filtering via BaseFilter interface
                if mirrored_label not in self._filters:
                    self._filters[mirrored_label] = self._create_filter_instance()
                smoothed_norm = self._filters[mirrored_label].filter(raw_coords, current_time)
                self._smoothed_landmarks[mirrored_label] = smoothed_norm

                # Convert to screen pixel coordinates
                landmarks_px = np.empty_like(smoothed_norm)
                landmarks_px[:, 0] = smoothed_norm[:, 0] * w
                landmarks_px[:, 1] = smoothed_norm[:, 1] * h
                landmarks_px[:, 2] = smoothed_norm[:, 2] * w  # Depth scaled to width

                # Velocity calculation for fingertips and wrist (pixels/second)
                dt = current_time - self._prev_timestamps.get(mirrored_label, current_time)
                dt = max(0.001, dt)
                self._prev_timestamps[mirrored_label] = current_time

                # Wrist velocity calculation
                cur_wrist = landmarks_px[0, :2]
                prev_wrist = self._prev_wrist_px.get(mirrored_label)
                if prev_wrist is not None:
                    vw_x = float((cur_wrist[0] - prev_wrist[0]) / dt)
                    vw_y = float((cur_wrist[1] - prev_wrist[1]) / dt)
                    wrist_vel = (vw_x, vw_y)
                else:
                    wrist_vel = (0.0, 0.0)
                self._prev_wrist_px[mirrored_label] = cur_wrist.copy()

                velocities: Dict[int, Tuple[float, float]] = {}
                prev_tips = self._prev_fingertip_px.get(mirrored_label, {})
                current_tips: Dict[int, np.ndarray] = {}

                for tip_idx in self.FINGERTIP_INDICES:
                    cur_pt = landmarks_px[tip_idx, :2]
                    current_tips[tip_idx] = cur_pt.copy()
                    if tip_idx in prev_tips:
                        delta = cur_pt - prev_tips[tip_idx]
                        vx = float(delta[0] / dt)
                        vy = float(delta[1] / dt)
                        velocities[tip_idx] = (vx, vy)
                    else:
                        velocities[tip_idx] = (0.0, 0.0)

                self._prev_fingertip_px[mirrored_label] = current_tips

                tracked_hands.append(
                    HandData(
                        handedness=mirrored_label,
                        landmarks_norm=smoothed_norm,
                        landmarks_px=landmarks_px,
                        fingertip_velocities=velocities,
                        wrist_velocity=wrist_vel,
                        timestamp=current_time,
                    )
                )

        # Evict stale hands from filter memory if no longer in frame
        for label in list(self._smoothed_landmarks.keys()):
            if label not in currently_detected_labels:
                del self._smoothed_landmarks[label]
                self._filters.pop(label, None)
                self._prev_timestamps.pop(label, None)
                self._prev_fingertip_px.pop(label, None)
                self._prev_wrist_px.pop(label, None)

        return tracked_hands

    def close(self) -> None:
        """Releases MediaPipe Hands pipeline."""
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception as e:
                logger.debug("Error closing MediaPipe hands: %s", e)
            finally:
                self.hands = None


class AsyncHandTracker:
    """
    Ultra-High-Performance Asynchronous Multi-Threaded AI Hand Tracking Engine.

    Architecture:
    - Decouples AI hand tracking from camera capture and display rendering.
    - Runs MediaPipe inference on a dedicated background CPU worker thread.
    - Thread-safe mutex-synchronized double-buffered snapshot with timestamping.
    - Stale-frame skipping: skips re-inferring on unchanged camera frames.
    - Predictive Kinematic Dead-Reckoning to ensure 60-120 FPS render loops.
    """

    def __init__(
        self,
        camera: ThreadedCamera,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.50,
        ema_alpha: float = 0.65,
        filter_mode: str = "one_euro",
        model_complexity: int = 1,
    ) -> None:
        self.camera = camera
        self.tracker = HandTracker(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            ema_alpha=ema_alpha,
            filter_mode=filter_mode,
            model_complexity=model_complexity,
        )
        self._config_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._latest_hands: List[HandData] = []
        self._latest_timestamp: float = time.perf_counter()
        self.is_running: bool = False
        self._thread: Optional[threading.Thread] = None

        # Telemetry
        self.ai_fps: float = 0.0
        self._ai_frame_count: int = 0
        self._last_fps_time: float = time.perf_counter()
        self.inference_latency_ms: float = 0.0
        self.stale_frames_skipped: int = 0

    @property
    def filter_mode(self) -> str:
        with self._config_lock:
            return self.tracker.filter_mode

    @filter_mode.setter
    def filter_mode(self, mode: str) -> None:
        self.set_filter_mode(mode)

    def set_filter_mode(self, mode: str) -> None:
        with self._config_lock:
            self.tracker.set_filter_mode(mode)

    @property
    def model_complexity(self) -> int:
        with self._config_lock:
            return self.tracker.model_complexity

    @model_complexity.setter
    def model_complexity(self, complexity: int) -> None:
        self.set_model_complexity(complexity)

    def set_model_complexity(self, complexity: int) -> None:
        """Dynamically switches MediaPipe model complexity (0 = Lite, 1 = Full)."""
        with self._config_lock:
            self.tracker.set_model_complexity(complexity)

    def process(self, raw_bgr_frame: np.ndarray) -> List[HandData]:
        """Direct synchronous process fallback."""
        with self._config_lock:
            return self.tracker.process(raw_bgr_frame)

    def start(self) -> AsyncHandTracker:
        """Launches the dedicated background AI tracking worker thread."""
        self.is_running = True
        self._last_fps_time = time.perf_counter()
        self._thread = threading.Thread(
            target=self._tracking_loop,
            daemon=True,
            name="AsyncHandTrackerWorker",
        )
        self._thread.start()
        logger.info("AsyncHandTracker worker thread started successfully.")
        return self

    def _tracking_loop(self) -> None:
        """Dedicated worker loop continuously pulling latest camera frame and computing landmarks."""
        last_processed_frame_id: int = -1
        while self.is_running:
            ret, frame, frame_id, frame_ts = self.camera.read_sequenced()
            if not ret or frame is None:
                time.sleep(0.002)
                continue

            # Stale frame skipping: avoid re-running MediaPipe on the identical camera frame
            if frame_id == last_processed_frame_id:
                self.stale_frames_skipped += 1
                time.sleep(0.001)
                continue

            last_processed_frame_id = frame_id
            t_start = time.perf_counter()

            # Heavy inference executed under config lock, completely decoupled from snapshot lock
            with self._config_lock:
                hands = self.tracker.process(frame)
            t_after = time.perf_counter()

            # Publish result to snapshot lock in microseconds
            with self._snapshot_lock:
                self._latest_hands = hands
                self._latest_timestamp = t_after

            # Telemetry tracking
            infer_dur_ms = (t_after - t_start) * 1000.0
            self.inference_latency_ms = 0.85 * self.inference_latency_ms + 0.15 * infer_dur_ms

            # Measure AI inference FPS
            self._ai_frame_count += 1
            dt_fps = t_after - self._last_fps_time
            if dt_fps >= 0.5:
                self.ai_fps = self._ai_frame_count / dt_fps
                self._ai_frame_count = 0
                self._last_fps_time = t_after

            # Yield briefly
            time.sleep(0.001)

    def get_latest_hands(self, current_time: float, extrapolate: bool = True) -> List[HandData]:
        """
        Retrieves the latest tracked hands with optional Predictive Kinematic Dead-Reckoning.
        Extrapolates coordinates forward by dt = (current_time - latest_update_time)
        so hand skeletons glide at 60-120 FPS between AI updates.
        """
        with self._snapshot_lock:
            hands_copy = [
                HandData(
                    handedness=h.handedness,
                    landmarks_norm=h.landmarks_norm.copy(),
                    landmarks_px=h.landmarks_px.copy(),
                    fingertip_velocities=dict(h.fingertip_velocities),
                    wrist_velocity=h.wrist_velocity,
                    timestamp=h.timestamp,
                )
                for h in self._latest_hands
            ]
            update_time = self._latest_timestamp

        if not extrapolate or not hands_copy:
            return hands_copy

        dt = float(np.clip(current_time - update_time, 0.0, 0.045))  # max 45ms forward projection
        if dt <= 0.002:
            return hands_copy

        # Kinematic forward projection for ultra-smooth 60-120 FPS rendering
        for hand in hands_copy:
            vw_x, vw_y = hand.wrist_velocity
            # Extrapolate all landmarks with base wrist movement
            if abs(vw_x) > 4.0 or abs(vw_y) > 4.0:
                hand.landmarks_px[:, 0] += vw_x * dt
                hand.landmarks_px[:, 1] += vw_y * dt

            # Further extrapolate fingertips using relative fingertip velocity
            for tip_idx, (vx, vy) in hand.fingertip_velocities.items():
                if abs(vx) > 5.0 or abs(vy) > 5.0:
                    hand.landmarks_px[tip_idx, 0] += vx * dt * 0.5
                    hand.landmarks_px[tip_idx, 1] += vy * dt * 0.5

        return hands_copy

    def stop(self) -> None:
        """Stops the async worker thread and releases resources."""
        self.is_running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.tracker.close()
        logger.info("AsyncHandTracker stopped.")

    def close(self) -> None:
        self.stop()
