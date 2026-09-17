"""
tests/test_tasks_adapter.py
===========================
Comprehensive unit tests for MediaPipe Tasks HandLandmarker adapter in HandTracker:
- Initialization with real model file (models/hand_landmarker.task)
- Idempotent clean close()
- Model complexity toggling (0 <-> 1) and no-op preservation
- Filter mode toggling and temporal state reset
- Mirrored X-coordinate transformation: (1.0 - raw_x)
- Handedness inversion: ('Left' <-> 'Right') for mirrored AR
- Landmark coordinate invariants: landmarks_px == landmarks_norm * (w, h)
- Kinematic velocity calculations for wrist and fingertips
- Stale hand eviction from tracking buffers
- Multiple resolution scaling (downscaling > 960px, pass-through <= 960px)
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest

from vision_tracker import HandTracker


@dataclass
class MockCategory:
    category_name: str
    score: float = 0.95


@dataclass
class MockNormalizedLandmark:
    x: float
    y: float
    z: float = 0.0


class MockTasksResult:
    def __init__(
        self,
        hand_landmarks: Optional[List[List[MockNormalizedLandmark]]] = None,
        handedness: Optional[List[List[MockCategory]]] = None,
    ) -> None:
        self.hand_landmarks = hand_landmarks or []
        self.handedness = handedness or []


class MockTasksLandmarker:
    """Mock MediaPipe Tasks HandLandmarker for deterministic pipeline testing."""

    def __init__(self) -> None:
        self.closed = False
        self.calls: List[tuple] = []
        self.result = MockTasksResult()

    def detect_for_video(self, image, timestamp_ms: int) -> MockTasksResult:
        self.calls.append((image, timestamp_ms))
        return self.result

    def detect(self, image) -> MockTasksResult:
        self.calls.append((image,))
        return self.result

    def close(self) -> None:
        self.closed = True


def test_real_model_asset_exists():
    """Verifies that models/hand_landmarker.task exists at repo root."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(repo_root, "models", "hand_landmarker.task")
    assert os.path.exists(model_path), f"Missing model file: {model_path}"
    assert os.path.getsize(model_path) > 1_000_000, "Model file appears truncated"


def test_hand_tracker_init_mediapipe_false():
    """Tracker initializes cleanly without instantiating MediaPipe when init_mediapipe=False."""
    tracker = HandTracker(init_mediapipe=False)
    assert tracker.landmarker is None
    assert tracker.hands is None
    tracker.close()  # Must be idempotent


def test_real_hand_tracker_init_and_close():
    """Initializes HandTracker with real model file and verifies clean close."""
    try:
        from mediapipe.tasks.python import vision as mp_vision
        assert hasattr(mp_vision, "HandLandmarker")
    except (ImportError, AttributeError):
        pytest.skip("MediaPipe Tasks vision API not available in current environment")

    tracker = HandTracker(init_mediapipe=True)
    assert tracker.landmarker is not None
    assert tracker.hands is not None
    tracker.close()
    assert tracker.landmarker is None
    assert tracker.hands is None


def test_hand_tracker_model_resolution():
    """Verifies model path resolution logic."""
    tracker = HandTracker(init_mediapipe=False)
    resolved = tracker._resolve_model_path()
    assert os.path.exists(resolved)
    assert resolved.endswith("hand_landmarker.task")


def test_hand_tracker_filter_and_complexity_switching():
    """Verifies filter mode and model complexity switching and temporal state reset."""
    tracker = HandTracker(init_mediapipe=False, model_complexity=1, filter_mode="one_euro")

    # Filter switching
    assert tracker.set_filter_mode("one_euro") is False  # No-op
    assert tracker.set_filter_mode("deadband") is True
    assert tracker.filter_mode == "deadband"

    # Complexity switching
    assert tracker.set_model_complexity(1) is False  # No-op
    assert tracker.set_model_complexity(0) is True
    assert tracker.model_complexity == 0
    assert tracker.set_model_complexity(0) is False  # No-op
    assert tracker.set_model_complexity(1) is True
    assert tracker.model_complexity == 1
    assert tracker.set_model_complexity(5) is False  # Invalid complexity ignored


def test_mock_tasks_process_coordinates_and_handedness():
    """
    Validates that process() correctly parses MediaPipe Tasks output:
    - Inverts handedness: 'Left' -> 'Right', 'Right' -> 'Left'
    - Mirrors X coordinate: 1.0 - raw_x
    - Invariant: landmarks_px == landmarks_norm * (w, h)
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # 1 hand: raw label 'Left' (physically user's right hand in mirror)
    # 21 landmarks placed at raw x=0.30, y=0.40
    landmarks_hand0 = [MockNormalizedLandmark(x=0.30, y=0.40, z=-0.05) for _ in range(21)]
    # Give index tip (8) a distinct coordinate
    landmarks_hand0[8] = MockNormalizedLandmark(x=0.25, y=0.35, z=-0.02)

    mock_lm.result = MockTasksResult(
        hand_landmarks=[landmarks_hand0],
        handedness=[[MockCategory(category_name="Left", score=0.98)]],
    )

    hands = tracker.process(frame, timestamp=1.000)
    assert len(hands) == 1
    hand = hands[0]

    # Handedness inverted from 'Left' to 'Right'
    assert hand.handedness == "Right"

    # Mirrored X coordinate: raw_x=0.30 -> mirrored_x = 1.0 - 0.30 = 0.70
    np.testing.assert_allclose(hand.landmarks_norm[0, 0], 0.70, rtol=1e-5)
    np.testing.assert_allclose(hand.landmarks_norm[0, 1], 0.40, rtol=1e-5)
    # Index tip mirrored: raw_x=0.25 -> mirrored_x = 0.75
    np.testing.assert_allclose(hand.landmarks_norm[8, 0], 0.75, rtol=1e-5)
    np.testing.assert_allclose(hand.landmarks_norm[8, 1], 0.35, rtol=1e-5)

    # Pixel coordinate invariants
    np.testing.assert_allclose(hand.landmarks_px[:, 0], hand.landmarks_norm[:, 0] * w, rtol=1e-5)
    np.testing.assert_allclose(hand.landmarks_px[:, 1], hand.landmarks_norm[:, 1] * h, rtol=1e-5)
    np.testing.assert_allclose(hand.landmarks_px[:, 2], hand.landmarks_norm[:, 2] * w, rtol=1e-5)

    # First observation has zero velocity baseline
    assert hand.wrist_velocity == (0.0, 0.0)
    assert all(v == (0.0, 0.0) for v in hand.fingertip_velocities.values())


def test_mock_tasks_velocity_tracking_and_stale_eviction():
    """
    Validates:
    - Instantaneous velocity computation across successive frames
    - Eviction of stale hand tracking history when hand leaves frame
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Frame 1 at t=1.000: wrist at (0.50, 0.50) -> screen px (500, 500)
    lms_f1 = [MockNormalizedLandmark(x=0.50, y=0.50) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_f1],
        handedness=[[MockCategory(category_name="Right", score=0.99)]],
    )
    res1 = tracker.process(frame, timestamp=1.000)
    assert len(res1) == 1
    assert res1[0].handedness == "Left"  # Inverted
    assert res1[0].wrist_velocity == (0.0, 0.0)

    # Frame 2 at t=1.050 (dt = 50ms): wrist moves to raw_x=0.45 (mirrored_x=0.55, +50px), y=0.52 (+20px)
    lms_f2 = [MockNormalizedLandmark(x=0.45, y=0.52) for _ in range(21)]
    # Tip 8 moves to raw_x=0.40 (mirrored_x=0.60, +100px), y=0.56 (+60px)
    lms_f2[8] = MockNormalizedLandmark(x=0.40, y=0.56)
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_f2],
        handedness=[[MockCategory(category_name="Right", score=0.99)]],
    )
    res2 = tracker.process(frame, timestamp=1.050)
    assert len(res2) == 1
    hand2 = res2[0]

    # Expected wrist velocity: dx = (550 - 500) / 0.050 = 1000 px/s, dy = (520 - 500) / 0.050 = 400 px/s
    np.testing.assert_allclose(hand2.wrist_velocity[0], 1000.0, rtol=1e-3)
    np.testing.assert_allclose(hand2.wrist_velocity[1], 400.0, rtol=1e-3)

    # Expected tip 8 velocity: dx = (600 - 500) / 0.050 = 2000 px/s, dy = (560 - 500) / 0.050 = 1200 px/s
    tip8_v = hand2.fingertip_velocities[8]
    np.testing.assert_allclose(tip8_v[0], 2000.0, rtol=1e-3)
    np.testing.assert_allclose(tip8_v[1], 1200.0, rtol=1e-3)

    # Frame 3: Hand lost (0 detections) -> Eviction from internal buffers
    mock_lm.result = MockTasksResult(hand_landmarks=[], handedness=[])
    res3 = tracker.process(frame, timestamp=1.100)
    assert len(res3) == 0
    assert len(tracker._smoothed_landmarks) == 0
    assert len(tracker._prev_timestamps) == 0
    assert len(tracker._prev_wrist_px) == 0


def test_resolution_downscaling_and_passthrough():
    """
    Verifies that frames with width > 960 are downscaled to 960 for inference,
    while frames with width <= 960 are passed through natively.
    """
    tracker = HandTracker(init_mediapipe=False)
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    # 1. 1920x1080 frame -> downscaled to 960x540
    hd_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    tracker.process(hd_frame, timestamp=1.0)
    assert len(mock_lm.calls) == 1
    passed_img = mock_lm.calls[0][0]
    assert passed_img.width == 960
    assert passed_img.height == 540

    # 2. 640x480 frame -> kept as 640x480
    vga_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.process(vga_frame, timestamp=1.1)
    assert len(mock_lm.calls) == 2
    passed_img2 = mock_lm.calls[1][0]
    assert passed_img2.width == 640
    assert passed_img2.height == 480


def test_clean_idempotent_close():
    """Verifies that close() can be safely called multiple times without exceptions."""
    tracker = HandTracker(init_mediapipe=False)
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    assert not mock_lm.closed
    tracker.close()
    assert mock_lm.closed
    assert tracker.landmarker is None
    assert tracker.hands is None
    # Calling close again must not error
    tracker.close()
