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
    score: Optional[float] = 0.95


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


def test_hand_tracker_filter_and_profile_switching():
    """Verifies filter mode and tracking profile switching and temporal state reset."""
    tracker = HandTracker(init_mediapipe=False, tracking_profile="STABLE", filter_mode="one_euro")

    # Filter switching
    assert tracker.set_filter_mode("one_euro") is False  # No-op
    assert tracker.set_filter_mode("deadband") is True
    assert tracker.filter_mode == "deadband"

    # Profile switching
    assert tracker.set_tracking_profile("STABLE") is False  # No-op
    assert tracker.set_tracking_profile("RESPONSIVE") is True
    assert tracker.tracking_profile == "RESPONSIVE"
    assert tracker.set_tracking_profile("RESPONSIVE") is False  # No-op
    assert tracker.set_tracking_profile("STABLE") is True
    assert tracker.tracking_profile == "STABLE"
    assert tracker.set_tracking_profile("INVALID") is False  # Invalid profile ignored

    # Backward-compatibility: set_model_complexity
    assert tracker.set_model_complexity(1) is False  # Already STABLE (1)
    assert tracker.set_model_complexity(0) is True
    assert tracker.tracking_profile == "RESPONSIVE"
    assert tracker.model_complexity == 0
    assert tracker.set_model_complexity(1) is True
    assert tracker.tracking_profile == "STABLE"
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


def test_model_provenance_and_checksum():
    """
    Verifies that models/hand_landmarker.task matches the exact official
    Google MediaPipe float16 version 1 artifact digest and size.
    """
    import hashlib

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(repo_root, "models", "hand_landmarker.task")
    assert os.path.exists(model_path), f"Model missing at {model_path}"

    file_size = os.path.getsize(model_path)
    assert file_size == 7819105, f"Expected size 7819105, got {file_size}"

    with open(model_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    expected_sha256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
    assert digest == expected_sha256, f"Checksum mismatch: {digest} != {expected_sha256}"


def test_model_resolution_cwd_isolation(tmp_path, monkeypatch):
    """
    Verifies that a fake models/hand_landmarker.task in the current working directory
    is strictly ignored and does NOT shadow the bundled package model.
    """
    fake_models_dir = tmp_path / "models"
    fake_models_dir.mkdir(parents=True)
    fake_task = fake_models_dir / "hand_landmarker.task"
    fake_task.write_bytes(b"FAKE_SHADOWING_MODEL_CONTENT")

    monkeypatch.chdir(tmp_path)
    tracker = HandTracker(init_mediapipe=False)
    resolved = tracker._resolve_model_path()

    assert not resolved.startswith(str(tmp_path)), "CWD model shadowed bundled package model!"
    assert os.path.exists(resolved)
    assert os.path.getsize(resolved) == 7819105


def test_hand_landmarker_options_inspection(monkeypatch):
    """
    Inspects actual HandLandmarkerOptions passed to create_from_options:
    - STABLE (0.65) vs RESPONSIVE (0.50) confidence settings
    - No-op profile assignment does not recreate landmarker
    - Real profile transition recreates it exactly once and resets temporal state
    """
    captured_options = []

    class DummyLandmarker:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    try:
        from mediapipe.tasks.python import vision as mp_vision

        def mock_create(opts):
            captured_options.append(opts)
            return DummyLandmarker()

        monkeypatch.setattr(mp_vision.HandLandmarker, "create_from_options", mock_create)
    except (ImportError, AttributeError):
        pytest.skip("MediaPipe Tasks vision API not available in current environment")

    tracker = HandTracker(init_mediapipe=True, tracking_profile="STABLE")
    assert len(captured_options) == 1
    opts_stable = captured_options[0]
    assert opts_stable.min_hand_detection_confidence == 0.65
    assert opts_stable.min_hand_presence_confidence == 0.65
    assert opts_stable.min_tracking_confidence == 0.65
    assert tracker.last_landmarker_options is opts_stable

    # No-op profile assignment: landmarker not recreated
    assert tracker.set_tracking_profile("STABLE") is False
    assert len(captured_options) == 1
    assert tracker.landmarker_creation_count == 1

    # Real profile switch: landmarker recreated with 0.50 thresholds
    assert tracker.set_tracking_profile("RESPONSIVE") is True
    assert len(captured_options) == 2
    opts_responsive = captured_options[1]
    assert opts_responsive.min_hand_detection_confidence == 0.50
    assert opts_responsive.min_hand_presence_confidence == 0.50
    assert opts_responsive.min_tracking_confidence == 0.50
    assert tracker.landmarker_creation_count == 2
    assert tracker.last_landmarker_options is opts_responsive

    tracker.close()


def test_visual_profile_changes_do_not_rebuild_tracker():
    """
    Verifies that changing visual quality profiles (HIGH -> BALANCED -> LOW)
    in GestureARApp is decoupled from tracking and does NOT recreate HandLandmarker.
    """
    from main import GestureARApp

    app = GestureARApp(start_threads=False, init_mediapipe=False, quality_profile="HIGH")
    try:
        initial_creation_count = app.async_tracker.tracker.landmarker_creation_count
        initial_profile = app.async_tracker.tracking_profile
        assert initial_profile == "STABLE"

        app.set_quality_profile("BALANCED")
        assert app.quality_profile == "BALANCED"
        assert app.async_tracker.tracking_profile == "STABLE"
        assert app.async_tracker.tracker.landmarker_creation_count == initial_creation_count

        app.set_quality_profile("LOW")
        assert app.quality_profile == "LOW"
        assert app.async_tracker.tracking_profile == "STABLE"
        assert app.async_tracker.tracker.landmarker_creation_count == initial_creation_count

        app.set_quality_profile("HIGH")
        assert app.quality_profile == "HIGH"
        assert app.async_tracker.tracking_profile == "STABLE"
        assert app.async_tracker.tracker.landmarker_creation_count == initial_creation_count
    finally:
        app.shutdown()


def test_malformed_landmark_counts_rejection():
    """
    Verifies that hands with fewer or more than 21 landmarks are rejected safely:
    - 20 landmarks: rejected
    - 22 landmarks: rejected
    - 21 landmarks: accepted
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. 20 landmarks: rejected
    lms_20 = [MockNormalizedLandmark(x=0.5, y=0.5) for _ in range(20)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_20],
        handedness=[[MockCategory(category_name="Left")]],
    )
    res_20 = tracker.process(frame, timestamp=1.0)
    assert len(res_20) == 0, f"Expected 0 hands for 20 landmarks, got {len(res_20)}"

    # 2. 22 landmarks: rejected
    lms_22 = [MockNormalizedLandmark(x=0.5, y=0.5) for _ in range(22)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_22],
        handedness=[[MockCategory(category_name="Left")]],
    )
    res_22 = tracker.process(frame, timestamp=1.1)
    assert len(res_22) == 0, f"Expected 0 hands for 22 landmarks, got {len(res_22)}"

    # 3. 21 landmarks: accepted
    lms_21 = [MockNormalizedLandmark(x=0.5, y=0.5) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_21],
        handedness=[[MockCategory(category_name="Left")]],
    )
    res_21 = tracker.process(frame, timestamp=1.2)
    assert len(res_21) == 1, f"Expected 1 hand for 21 landmarks, got {len(res_21)}"
    assert res_21[0].handedness == "Right"  # Mirrored from 'Left'


def test_handedness_category_validation():
    """
    Verifies that only canonical handedness labels ('Left', 'Right') are accepted:
    - 'Left' -> accepted as mirrored 'Right'
    - 'Right' -> accepted as mirrored 'Left'
    - empty string '' -> rejected
    - 'Unknown' -> rejected
    - malformed category object -> rejected
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    lms_21 = [MockNormalizedLandmark(x=0.5, y=0.5) for _ in range(21)]

    # 1. Canonical 'Left' -> mirrored 'Right'
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_21],
        handedness=[[MockCategory(category_name="Left")]],
    )
    res = tracker.process(frame, timestamp=1.0)
    assert len(res) == 1
    assert res[0].handedness == "Right"

    # 2. Canonical 'Right' -> mirrored 'Left'
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_21],
        handedness=[[MockCategory(category_name="Right")]],
    )
    res = tracker.process(frame, timestamp=1.1)
    assert len(res) == 1
    assert res[0].handedness == "Left"

    # 3. Empty string category
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_21],
        handedness=[[MockCategory(category_name="")]],
    )
    res_empty = tracker.process(frame, timestamp=1.2)
    assert len(res_empty) == 0

    # 4. 'Unknown' category
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_21],
        handedness=[[MockCategory(category_name="Unknown")]],
    )
    res_unk = tracker.process(frame, timestamp=1.3)
    assert len(res_unk) == 0

    # 5. Malformed category (empty list)
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_21],
        handedness=[[]],
    )
    res_malformed = tracker.process(frame, timestamp=1.4)
    assert len(res_malformed) == 0


def test_two_hand_simultaneous_tracking_independent_state():
    """
    Verifies that two simultaneous hands:
    - Retain correct mirrored handedness
    - Produce independent filter state instances
    - Produce independent velocities without cross-hand leakage
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Frame 1: Hand 0 is raw 'Left' (mirrored to 'Right') at x=0.20, y=0.50 -> screen px (800, 500)
    #          Hand 1 is raw 'Right' (mirrored to 'Left') at x=0.80, y=0.50 -> screen px (200, 500)
    lms_h0_f1 = [MockNormalizedLandmark(x=0.20, y=0.50) for _ in range(21)]
    lms_h1_f1 = [MockNormalizedLandmark(x=0.80, y=0.50) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_h0_f1, lms_h1_f1],
        handedness=[
            [MockCategory(category_name="Left")],
            [MockCategory(category_name="Right")],
        ],
    )

    hands_f1 = tracker.process(frame, timestamp=1.000)
    assert len(hands_f1) == 2

    by_hand = {h.handedness: h for h in hands_f1}
    assert "Right" in by_hand
    assert "Left" in by_hand

    # Baseline velocities are zero
    assert by_hand["Right"].wrist_velocity == (0.0, 0.0)
    assert by_hand["Left"].wrist_velocity == (0.0, 0.0)

    # Frame 2 at dt = 0.050s:
    # Hand 0 (mirrored Right) moves: raw_x 0.20 -> 0.15 (mirrored_x 0.80 -> 0.85, +50px), y unchanged
    # Hand 1 (mirrored Left) moves: raw_x 0.80 -> 0.83 (mirrored_x 0.20 -> 0.17, -30px), y +20px (0.50 -> 0.52)
    lms_h0_f2 = [MockNormalizedLandmark(x=0.15, y=0.50) for _ in range(21)]
    lms_h1_f2 = [MockNormalizedLandmark(x=0.83, y=0.52) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_h0_f2, lms_h1_f2],
        handedness=[
            [MockCategory(category_name="Left")],
            [MockCategory(category_name="Right")],
        ],
    )

    hands_f2 = tracker.process(frame, timestamp=1.050)
    assert len(hands_f2) == 2
    by_hand2 = {h.handedness: h for h in hands_f2}

    # Expected velocity for Right hand: vx = +50 / 0.05 = +1000 px/s, vy = 0 px/s
    np.testing.assert_allclose(by_hand2["Right"].wrist_velocity[0], 1000.0, rtol=1e-3)
    np.testing.assert_allclose(by_hand2["Right"].wrist_velocity[1], 0.0, atol=1e-3)

    # Expected velocity for Left hand: vx = -30 / 0.05 = -600 px/s, vy = +20 / 0.05 = +400 px/s
    np.testing.assert_allclose(by_hand2["Left"].wrist_velocity[0], -600.0, rtol=1e-3)
    np.testing.assert_allclose(by_hand2["Left"].wrist_velocity[1], 400.0, rtol=1e-3)

    # Assert independent filter instances exist for both
    assert "Right" in tracker._filters
    assert "Left" in tracker._filters
    assert tracker._filters["Right"] is not tracker._filters["Left"]


def test_hand_landmarker_transactional_reconfiguration(monkeypatch):
    """
    Verifies transactional HandLandmarker profile reconfiguration:
    1. STABLE -> RESPONSIVE successful transition creates exactly one replacement.
    2. Old landmarker is closed only after replacement creation succeeds.
    3. Failed replacement creation leaves old landmarker alive and installed.
    4. Failed transition leaves profile unchanged.
    5. Failed transition does not clear temporal state.
    6. Failed AsyncHandTracker transition does not clear _latest_hands.
    7. Successful transition still clears tracking snapshot and resets first-frame velocity baseline.
    8. M-key failure cannot silently leave HUD/profile state inconsistent.
    """
    from vision_tracker import AsyncHandTracker, HandData
    from main import GestureARApp

    call_events = []

    class LifecycleMockLandmarker:
        def __init__(self, name="old"):
            self.name = name
            self.closed = False

        def close(self):
            self.closed = True
            call_events.append(f"closed_{self.name}")

    old_lm = LifecycleMockLandmarker("old_lm")
    tracker = HandTracker(init_mediapipe=False, tracking_profile="STABLE")
    tracker.init_mediapipe = True
    tracker.landmarker = old_lm
    tracker.hands = old_lm

    # Populate dummy temporal state
    tracker._filters["Left"] = tracker._create_filter_instance()
    tracker._prev_timestamps["Left"] = 123.456
    tracker._prev_wrist_px["Left"] = np.array([50.0, 50.0])

    # Case A: Failed replacement creation
    def mock_failed_create(profile):
        call_events.append("attempted_failed_create")
        raise RuntimeError("Simulated MediaPipe HandLandmarker creation failure")

    monkeypatch.setattr(tracker, "_create_landmarker_for_profile", mock_failed_create)

    # 3. Failed replacement creation leaves old landmarker alive and installed
    # 4. Failed transition leaves profile unchanged
    # 5. Failed transition does not clear temporal state
    assert tracker.set_tracking_profile("RESPONSIVE") is False
    assert tracker.tracking_profile == "STABLE"
    assert tracker.landmarker is old_lm
    assert old_lm.closed is False
    assert "closed_old_lm" not in call_events
    assert "Left" in tracker._filters
    assert tracker._prev_timestamps["Left"] == 123.456

    # 6. Failed AsyncHandTracker transition does not clear _latest_hands
    class DummyCamera:
        width = 1280
        height = 720
        def read_sequenced(self):
            return False, None, 0, 0.0

    async_tracker = AsyncHandTracker(camera=DummyCamera(), init_mediapipe=False, tracking_profile="STABLE")
    async_tracker.tracker.init_mediapipe = True
    async_tracker.tracker.landmarker = old_lm
    async_tracker.tracker.hands = old_lm
    dummy_hand = HandData(
        handedness="Right",
        landmarks_norm=np.zeros((21, 3), dtype=np.float32),
        landmarks_px=np.zeros((21, 3), dtype=np.float32),
    )
    async_tracker._latest_hands = [dummy_hand]

    monkeypatch.setattr(async_tracker.tracker, "_create_landmarker_for_profile", mock_failed_create)
    assert async_tracker.set_tracking_profile("RESPONSIVE") is False
    assert async_tracker.tracking_profile == "STABLE"
    assert len(async_tracker._latest_hands) == 1
    assert async_tracker._latest_hands[0] is dummy_hand

    # 8. M-key failure cannot silently leave HUD/profile state inconsistent
    app = GestureARApp(start_threads=False, init_mediapipe=False)
    app.async_tracker.tracker.init_mediapipe = True
    app.async_tracker.tracker.landmarker = old_lm
    app.async_tracker.tracker.hands = old_lm
    app.async_tracker._latest_hands = [dummy_hand]
    monkeypatch.setattr(app.async_tracker.tracker, "_create_landmarker_for_profile", mock_failed_create)

    action = app.handle_key(ord("m"))
    assert action == "TRACKING_PROFILE_FAILED"
    assert app.async_tracker.tracking_profile == "STABLE"
    assert len(app.async_tracker._latest_hands) == 1

    # Case B: Successful transition
    # 1. STABLE -> RESPONSIVE successful transition creates exactly one replacement
    # 2. Old landmarker is closed only after replacement creation succeeds
    # 7. Successful transition still clears tracking snapshot and resets first-frame velocity baseline
    call_events.clear()
    new_lm = LifecycleMockLandmarker("new_lm")

    def mock_success_create(profile):
        call_events.append("created_new_lm")
        return new_lm, None

    monkeypatch.setattr(app.async_tracker.tracker, "_create_landmarker_for_profile", mock_success_create)
    initial_count = app.async_tracker.tracker.landmarker_creation_count

    action_success = app.handle_key(ord("m"))
    assert action_success == "TRACKING_PROFILE"
    assert app.async_tracker.tracker.landmarker_creation_count == initial_count + 1
    assert app.async_tracker.tracking_profile == "RESPONSIVE"
    assert app.async_tracker.tracker.landmarker is new_lm
    assert old_lm.closed is True
    # Verify order: created new first, then closed old
    assert call_events == ["created_new_lm", "closed_old_lm"]
    # Verify snapshot cleared and temporal state reset
    assert len(app.async_tracker._latest_hands) == 0
    assert len(app.async_tracker.tracker._filters) == 0
    assert len(app.async_tracker.tracker._prev_timestamps) == 0


def test_non_finite_landmarks_rejection():
    """
    Verifies rejection of non-finite landmark coordinates (NaN, +Inf, -Inf):
    - finite 21-point hand accepted;
    - one NaN x rejected;
    - one Inf y rejected;
    - one -Inf z rejected;
    - malformed detection does not create _filters;
    - malformed detection does not create _prev_timestamps;
    - subsequent valid hand starts with zero velocity baseline.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Finite 21-point hand accepted
    lms_valid = [MockNormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_valid],
        handedness=[[MockCategory(category_name="Left")]],
    )
    res_valid = tracker.process(frame, timestamp=1.0)
    assert len(res_valid) == 1
    assert "Right" in tracker._filters
    assert "Right" in tracker._prev_timestamps
    assert res_valid[0].wrist_velocity == (0.0, 0.0)

    # Clear state for negative tests
    tracker.reset_temporal_state()
    assert len(tracker._filters) == 0
    assert len(tracker._prev_timestamps) == 0

    # 2. One NaN x rejected
    lms_nan = [MockNormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    lms_nan[4] = MockNormalizedLandmark(x=float("nan"), y=0.5, z=0.0)
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_nan],
        handedness=[[MockCategory(category_name="Left")]],
    )
    res_nan = tracker.process(frame, timestamp=1.1)
    assert len(res_nan) == 0
    assert len(tracker._filters) == 0
    assert len(tracker._prev_timestamps) == 0

    # 3. One Inf y rejected
    lms_inf = [MockNormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    lms_inf[8] = MockNormalizedLandmark(x=0.5, y=float("inf"), z=0.0)
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_inf],
        handedness=[[MockCategory(category_name="Left")]],
    )
    res_inf = tracker.process(frame, timestamp=1.2)
    assert len(res_inf) == 0
    assert len(tracker._filters) == 0
    assert len(tracker._prev_timestamps) == 0

    # 4. One -Inf z rejected
    lms_neginf = [MockNormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    lms_neginf[0] = MockNormalizedLandmark(x=0.5, y=0.5, z=float("-inf"))
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_neginf],
        handedness=[[MockCategory(category_name="Left")]],
    )
    res_neginf = tracker.process(frame, timestamp=1.3)
    assert len(res_neginf) == 0
    assert len(tracker._filters) == 0
    assert len(tracker._prev_timestamps) == 0

    # 5. Subsequent valid hand starts with zero velocity baseline
    lms_subsequent = [MockNormalizedLandmark(x=0.4, y=0.6, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms_subsequent],
        handedness=[[MockCategory(category_name="Left")]],
    )
    res_subsequent = tracker.process(frame, timestamp=1.4)
    assert len(res_subsequent) == 1
    assert res_subsequent[0].wrist_velocity == (0.0, 0.0)
    for tip_idx, vel in res_subsequent[0].fingertip_velocities.items():
        assert vel == (0.0, 0.0)
    assert "Right" in tracker._filters
    assert "Right" in tracker._prev_timestamps


def test_duplicate_raw_left_case_a():
    """
    Case A: Duplicate raw Left detections in a single frame.
    - Resolves to exactly one application 'Right' hand.
    - No false 'Left' hand is emitted.
    - Tracker temporal state registers only 'Right'.
    - Initial baseline velocity is (0.0, 0.0) without spikes.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Candidate 0: raw Left, wrist at raw x=0.8 (mirrored x=200), score 0.90
    cand0_lms = [MockNormalizedLandmark(x=0.8, y=0.5, z=0.0) for _ in range(21)]
    # Candidate 1: raw Left, wrist at raw x=0.2 (mirrored x=800), score 0.70
    cand1_lms = [MockNormalizedLandmark(x=0.2, y=0.5, z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand0_lms, cand1_lms],
        handedness=[[MockCategory(category_name="Left", score=0.90)], [MockCategory(category_name="Left", score=0.70)]],
    )

    hands = tracker.process(frame, timestamp=1.0)
    assert len(hands) == 1
    assert hands[0].handedness == "Right"
    assert "Right" in tracker._filters
    assert "Left" not in tracker._filters
    assert hands[0].wrist_velocity == (0.0, 0.0)
    for tip_vel in hands[0].fingertip_velocities.values():
        assert tip_vel == (0.0, 0.0)
    # Selected candidate is cand0 (higher score 0.90)
    assert pytest.approx(hands[0].landmarks_px[0, 0], abs=1e-3) == 200.0


def test_duplicate_raw_right_case_b():
    """
    Case B: Duplicate raw Right detections in a single frame.
    - Resolves to exactly one application 'Left' hand.
    - No false 'Right' hand is emitted.
    - Tracker temporal state registers only 'Left'.
    - Initial baseline velocity is (0.0, 0.0) without spikes.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Candidate 0: raw Right, wrist at raw x=0.3 (mirrored x=700), score 0.85
    cand0_lms = [MockNormalizedLandmark(x=0.3, y=0.5, z=0.0) for _ in range(21)]
    # Candidate 1: raw Right, wrist at raw x=0.7 (mirrored x=300), score 0.65
    cand1_lms = [MockNormalizedLandmark(x=0.7, y=0.5, z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand0_lms, cand1_lms],
        handedness=[[MockCategory(category_name="Right", score=0.85)], [MockCategory(category_name="Right", score=0.65)]],
    )

    hands = tracker.process(frame, timestamp=1.0)
    assert len(hands) == 1
    assert hands[0].handedness == "Left"
    assert "Left" in tracker._filters
    assert "Right" not in tracker._filters
    assert hands[0].wrist_velocity == (0.0, 0.0)
    for tip_vel in hands[0].fingertip_velocities.values():
        assert tip_vel == (0.0, 0.0)
    # Selected candidate is cand0 (higher score 0.85)
    assert pytest.approx(hands[0].landmarks_px[0, 0], abs=1e-3) == 700.0


def test_duplicate_wrist_continuity_selection_case_c():
    """
    Case C: Wrist continuity selection over confidence.
    - Pre-established wrist near mirrored x=200 (raw x=0.8).
    - Frame 2 has duplicate candidates:
      - Candidate 0: mirrored x=900 (raw x=0.1) with higher score 0.99
      - Candidate 1: mirrored x=210 (raw x=0.79) with lower score 0.50
    - Candidate 1 (closest to 200) MUST be selected due to motion continuity.
    - Velocity corresponds to ~10px displacement, NOT 700px.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Frame 1: Establish wrist at mirrored x=200.0 (raw x=0.8, y=0.5)
    frame1_lms = [MockNormalizedLandmark(x=0.8, y=0.5, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[frame1_lms],
        handedness=[[MockCategory(category_name="Left", score=0.90)]],
    )
    hands1 = tracker.process(frame, timestamp=1.0)
    assert len(hands1) == 1
    assert pytest.approx(hands1[0].landmarks_px[0, 0], abs=1e-3) == 200.0

    # Frame 2 (dt = 0.02s): Duplicate candidates
    # cand0: raw x=0.1 -> mirrored x=900.0, score=0.99 (distance = 700px)
    cand0_lms = [MockNormalizedLandmark(x=0.1, y=0.5, z=0.0) for _ in range(21)]
    # cand1: raw x=0.79 -> mirrored x=210.0, score=0.50 (distance = 10px)
    cand1_lms = [MockNormalizedLandmark(x=0.79, y=0.5, z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand0_lms, cand1_lms],
        handedness=[[MockCategory(category_name="Left", score=0.99)], [MockCategory(category_name="Left", score=0.50)]],
    )

    hands2 = tracker.process(frame, timestamp=1.02)
    assert len(hands2) == 1
    assert hands2[0].handedness == "Right"
    # Candidate 1 should be selected
    assert pytest.approx(hands2[0].landmarks_px[0, 0], abs=1e-3) == 210.0
    # Velocity vx should be ~ 10px / 0.02s = 500 px/s, NOT 700 / 0.02 = 35000 px/s
    assert pytest.approx(hands2[0].wrist_velocity[0], abs=1.0) == 500.0
    assert pytest.approx(hands2[0].wrist_velocity[1], abs=1.0) == 0.0


def test_duplicate_initial_score_selection_case_d():
    """
    Case D: Initial duplicate (no previous state) resolves using classification score.
    - Candidate 0: score 0.60
    - Candidate 1: score 0.95
    - Candidate 1 is chosen.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Candidate 0: raw x=0.5 (mirrored 500), score 0.60
    cand0_lms = [MockNormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    # Candidate 1: raw x=0.8 (mirrored 200), score 0.95
    cand1_lms = [MockNormalizedLandmark(x=0.8, y=0.5, z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand0_lms, cand1_lms],
        handedness=[[MockCategory(category_name="Left", score=0.60)], [MockCategory(category_name="Left", score=0.95)]],
    )

    hands = tracker.process(frame, timestamp=1.0)
    assert len(hands) == 1
    assert hands[0].handedness == "Right"
    assert pytest.approx(hands[0].landmarks_px[0, 0], abs=1e-3) == 200.0


def test_duplicate_score_absent_non_finite_tied_case_e():
    """
    Case E: Score absent (None), non-finite (NaN/Inf), or tied resolves deterministically
    by upstream source order without crashing.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    cand0_lms = [MockNormalizedLandmark(x=0.6, y=0.5, z=0.0) for _ in range(21)]  # mirrored x=400
    cand1_lms = [MockNormalizedLandmark(x=0.3, y=0.5, z=0.0) for _ in range(21)]  # mirrored x=700

    # Sub-case E1: Both scores are None
    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand0_lms, cand1_lms],
        handedness=[[MockCategory(category_name="Left", score=None)], [MockCategory(category_name="Left", score=None)]],
    )
    hands_none = tracker.process(frame, timestamp=1.0)
    assert len(hands_none) == 1
    assert pytest.approx(hands_none[0].landmarks_px[0, 0], abs=1e-3) == 400.0

    tracker.reset_temporal_state()

    # Sub-case E2: Scores are NaN vs Inf
    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand0_lms, cand1_lms],
        handedness=[[MockCategory(category_name="Left", score=float("nan"))], [MockCategory(category_name="Left", score=float("inf"))]],
    )
    hands_nan = tracker.process(frame, timestamp=2.0)
    assert len(hands_nan) == 1
    assert pytest.approx(hands_nan[0].landmarks_px[0, 0], abs=1e-3) == 400.0

    tracker.reset_temporal_state()

    # Sub-case E3: Tied scores (0.85 vs 0.85)
    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand0_lms, cand1_lms],
        handedness=[[MockCategory(category_name="Left", score=0.85)], [MockCategory(category_name="Left", score=0.85)]],
    )
    hands_tied = tracker.process(frame, timestamp=3.0)
    assert len(hands_tied) == 1
    assert pytest.approx(hands_tied[0].landmarks_px[0, 0], abs=1e-3) == 400.0


def test_normal_two_hands_preserved_case_f():
    """
    Case F: Normal two-hand detection (raw Left + raw Right).
    - Preserves both hands as independent application 'Right' and 'Left'.
    - Both filters and temporal tracking states are maintained independently.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Frame 1: raw Left at x=0.8 (mirrored 200), raw Right at x=0.2 (mirrored 800)
    cand_left_lms = [MockNormalizedLandmark(x=0.8, y=0.5, z=0.0) for _ in range(21)]
    cand_right_lms = [MockNormalizedLandmark(x=0.2, y=0.5, z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand_left_lms, cand_right_lms],
        handedness=[[MockCategory(category_name="Left", score=0.95)], [MockCategory(category_name="Right", score=0.92)]],
    )

    hands1 = tracker.process(frame, timestamp=1.0)
    assert len(hands1) == 2
    hand_map1 = {h.handedness: h for h in hands1}
    assert "Right" in hand_map1
    assert "Left" in hand_map1
    assert pytest.approx(hand_map1["Right"].landmarks_px[0, 0], abs=1e-3) == 200.0
    assert pytest.approx(hand_map1["Left"].landmarks_px[0, 0], abs=1e-3) == 800.0
    assert "Right" in tracker._filters
    assert "Left" in tracker._filters

    # Frame 2 (dt = 0.05s): Move Right hand +20px (mirrored 220), Move Left hand -30px (mirrored 770)
    cand_left_lms2 = [MockNormalizedLandmark(x=0.78, y=0.5, z=0.0) for _ in range(21)]  # mirrored 220
    cand_right_lms2 = [MockNormalizedLandmark(x=0.23, y=0.5, z=0.0) for _ in range(21)]  # mirrored 770

    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand_left_lms2, cand_right_lms2],
        handedness=[[MockCategory(category_name="Left", score=0.95)], [MockCategory(category_name="Right", score=0.92)]],
    )

    hands2 = tracker.process(frame, timestamp=1.05)
    assert len(hands2) == 2
    hand_map2 = {h.handedness: h for h in hands2}
    # Right hand: delta_x = 20px, dt = 0.05s -> vx = 400 px/s
    assert pytest.approx(hand_map2["Right"].wrist_velocity[0], abs=1.0) == 400.0
    # Left hand: delta_x = -30px, dt = 0.05s -> vx = -600 px/s
    assert pytest.approx(hand_map2["Left"].wrist_velocity[0], abs=1.0) == -600.0


def test_duplicate_handedness_instrument_safety_case_g():
    """
    Case G: Instrument interaction safety with duplicate detections in a frame.
    - Piano: A duplicate raw Left detection does not trigger a false strike note.
    - Guitar: A duplicate raw Left detection does not trigger a false string strum.
    """
    from instruments import Piano, Guitar

    class MockAudio:
        def __init__(self):
            self.triggered_notes = []
            self.plucked_strings = []

        def play_piano(self, freq: float, velocity: float = 1.0, pan: float = 0.0):
            self.triggered_notes.append((freq, velocity, pan))

        def play_guitar(self, string_idx: int, chord_name: str = "C", velocity: float = 1.0) -> bool:
            self.plucked_strings.append((string_idx, chord_name, velocity))
            return True

    mock_audio = MockAudio()
    # Piano placed on 1280x720 frame
    piano = Piano(bbox=(64, 518, 1216, 691), audio_engine=mock_audio)
    # Guitar placed on 1280x720 frame
    guitar = Guitar(zones={"fretboard": (150, 250, 450, 450), "strum_zone": (550, 450, 850, 650)}, audio_engine=mock_audio)

    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # White key 0 rect: wx1, wy1, wx2, wy2
    wkey = piano.white_keys[0]
    key_center_x = (wkey.rect[0] + wkey.rect[2]) / 2.0
    # In normalised mirrored coords: mirrored_x = key_center_x / w -> raw_x = 1.0 - (key_center_x / w)
    raw_key_x = 1.0 - (key_center_x / w)
    key_hover_y = (wkey.rect[1] + 10.0) / h  # Resting just inside top of key

    # Frame 1 at t=1.0: Hand hovering resting on piano key
    hand_hover = [MockNormalizedLandmark(x=raw_key_x, y=key_hover_y, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[hand_hover],
        handedness=[[MockCategory(category_name="Left", score=0.90)]],
    )
    hands1 = tracker.process(frame, timestamp=1.0)
    piano.update(hands1, frame_shape=(h, w, 3), current_time=1.0)
    guitar.update(hands1, frame_shape=(h, w, 3), current_time=1.0)
    assert len(mock_audio.triggered_notes) == 0
    assert len(mock_audio.plucked_strings) == 0

    # Frame 2 at t=1.05:
    # MediaPipe produces TWO raw Left detections:
    # Candidate 0: Resting in place (raw_key_x, key_hover_y)
    # Candidate 1: Jumped 200px downward (raw_key_x, key_hover_y + 200.0/h)
    cand_stay = [MockNormalizedLandmark(x=raw_key_x, y=key_hover_y, z=0.0) for _ in range(21)]
    cand_jump = [MockNormalizedLandmark(x=raw_key_x, y=key_hover_y + (200.0 / h), z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand_stay, cand_jump],
        handedness=[[MockCategory(category_name="Left", score=0.50)], [MockCategory(category_name="Left", score=0.99)]],
    )

    hands2 = tracker.process(frame, timestamp=1.05)
    assert len(hands2) == 1
    # Candidate 0 must be chosen by wrist continuity
    assert pytest.approx(hands2[0].landmarks_px[0, 1], abs=1.0) == key_hover_y * h
    # Vy must be ~ 0 px/s
    assert abs(hands2[0].fingertip_velocities[8][1]) < 10.0

    piano.update(hands2, frame_shape=(h, w, 3), current_time=1.05)
    guitar.update(hands2, frame_shape=(h, w, 3), current_time=1.05)

    # Assert NO false piano note triggers and NO false guitar strums occurred
    assert len(mock_audio.triggered_notes) == 0, f"Expected 0 piano notes, got {mock_audio.triggered_notes}"
    assert len(mock_audio.plucked_strings) == 0, f"Expected 0 guitar strums, got {mock_audio.plucked_strings}"

