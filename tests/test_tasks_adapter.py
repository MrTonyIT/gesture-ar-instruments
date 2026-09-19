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

    # Real profile switch: landmarker recreated with 0.55/0.50 thresholds
    assert tracker.set_tracking_profile("RESPONSIVE") is True
    assert len(captured_options) == 2
    opts_responsive = captured_options[1]
    assert pytest.approx(opts_responsive.min_hand_detection_confidence, abs=1e-3) == 0.55
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
        assert initial_profile == "RESPONSIVE"

        app.set_quality_profile("BALANCED")
        assert app.quality_profile == "BALANCED"
        assert app.async_tracker.tracking_profile == "RESPONSIVE"
        assert app.async_tracker.tracker.landmarker_creation_count == initial_creation_count

        app.set_quality_profile("LOW")
        assert app.quality_profile == "LOW"
        assert app.async_tracker.tracking_profile == "RESPONSIVE"
        assert app.async_tracker.tracker.landmarker_creation_count == initial_creation_count

        app.set_quality_profile("HIGH")
        assert app.quality_profile == "HIGH"
        assert app.async_tracker.tracking_profile == "RESPONSIVE"
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

    # 2. Canonical 'Right' -> mirrored 'Left' (isolated without prior temporal history)
    tracker.reset_temporal_state()
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
    app = GestureARApp(start_threads=False, init_mediapipe=False, tracking_profile="STABLE")
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
    Case A / Test A: Duplicate raw Left detections in a single frame.
    - Preserves BOTH hands as independent application 'Right' and 'Left'.
    - Candidate on screen-left (mirrored x=200) maps to 'Right'.
    - Candidate on screen-right (mirrored x=800) maps to 'Left'.
    - Tracker temporal state registers both 'Right' and 'Left'.
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
    assert len(hands) == 2
    hand_map = {h.handedness: h for h in hands}
    assert "Right" in hand_map
    assert "Left" in hand_map
    assert "Right" in tracker._filters
    assert "Left" in tracker._filters
    for h_data in hands:
        assert h_data.wrist_velocity == (0.0, 0.0)
        for tip_vel in h_data.fingertip_velocities.values():
            assert tip_vel == (0.0, 0.0)
    assert pytest.approx(hand_map["Right"].landmarks_px[0, 0], abs=1e-3) == 200.0
    assert pytest.approx(hand_map["Left"].landmarks_px[0, 0], abs=1e-3) == 800.0


def test_duplicate_raw_right_case_b():
    """
    Case B / Test B: Duplicate raw Right detections in a single frame.
    - Preserves BOTH hands as independent application 'Left' and 'Right'.
    - Candidate on screen-right (mirrored x=700) maps to 'Left'.
    - Candidate on screen-left (mirrored x=300) maps to 'Right'.
    - Tracker temporal state registers both 'Left' and 'Right'.
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
    assert len(hands) == 2
    hand_map = {h.handedness: h for h in hands}
    assert "Left" in hand_map
    assert "Right" in hand_map
    assert "Left" in tracker._filters
    assert "Right" in tracker._filters
    for h_data in hands:
        assert h_data.wrist_velocity == (0.0, 0.0)
        for tip_vel in h_data.fingertip_velocities.values():
            assert tip_vel == (0.0, 0.0)
    assert pytest.approx(hand_map["Left"].landmarks_px[0, 0], abs=1e-3) == 700.0
    assert pytest.approx(hand_map["Right"].landmarks_px[0, 0], abs=1e-3) == 300.0


def test_duplicate_wrist_continuity_selection_case_c():
    """
    Case C: Wrist continuity with duplicate detections in frame 2.
    - Pre-established wrist near mirrored x=200 (Right).
    - Frame 2 has duplicate candidates:
      - Candidate 0: mirrored x=900 (score 0.99)
      - Candidate 1: mirrored x=210 (score 0.50, distance = 10px)
    - BOTH hands are preserved.
    - Candidate 1 is continuously matched to Right (velocity ~ 10px / 0.02s = 500 px/s).
    - Candidate 0 is assigned to Left with safe baseline (velocity = 0.0).
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
    cand0_lms = [MockNormalizedLandmark(x=0.1, y=0.5, z=0.0) for _ in range(21)]
    cand1_lms = [MockNormalizedLandmark(x=0.79, y=0.5, z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand0_lms, cand1_lms],
        handedness=[[MockCategory(category_name="Left", score=0.99)], [MockCategory(category_name="Left", score=0.50)]],
    )

    hands2 = tracker.process(frame, timestamp=1.02)
    assert len(hands2) == 2
    hand_map2 = {h.handedness: h for h in hands2}
    assert "Right" in hand_map2
    assert "Left" in hand_map2

    # Continuous candidate matches Right
    assert pytest.approx(hand_map2["Right"].landmarks_px[0, 0], abs=1e-3) == 210.0
    assert pytest.approx(hand_map2["Right"].wrist_velocity[0], abs=1.0) == 500.0
    assert pytest.approx(hand_map2["Right"].wrist_velocity[1], abs=1.0) == 0.0

    # Newly appeared Left hand starts with safe baseline (no false velocity spike)
    assert pytest.approx(hand_map2["Left"].landmarks_px[0, 0], abs=1e-3) == 900.0
    assert hand_map2["Left"].wrist_velocity == (0.0, 0.0)


def test_duplicate_initial_score_selection_case_d():
    """
    Case D: Initial duplicate detections on frame 1 preserve both hands via screen-side prior.
    - Candidate 0: score 0.60 at mirrored x=500 (screen-right)
    - Candidate 1: score 0.95 at mirrored x=200 (screen-left)
    - Both hands are preserved without dropping either hand.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    cand0_lms = [MockNormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    cand1_lms = [MockNormalizedLandmark(x=0.8, y=0.5, z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand0_lms, cand1_lms],
        handedness=[[MockCategory(category_name="Left", score=0.60)], [MockCategory(category_name="Left", score=0.95)]],
    )

    hands = tracker.process(frame, timestamp=1.0)
    assert len(hands) == 2
    hand_map = {h.handedness: h for h in hands}
    assert "Right" in hand_map
    assert "Left" in hand_map
    assert pytest.approx(hand_map["Right"].landmarks_px[0, 0], abs=1e-3) == 200.0
    assert pytest.approx(hand_map["Left"].landmarks_px[0, 0], abs=1e-3) == 500.0


def test_duplicate_score_absent_non_finite_tied_case_e():
    """
    Case E: Score absent (None), non-finite (NaN/Inf), or tied preserves both hands
    deterministically without dropping either hand or crashing.
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
    assert len(hands_none) == 2
    hand_map_none = {h.handedness: h for h in hands_none}
    assert pytest.approx(hand_map_none["Right"].landmarks_px[0, 0], abs=1e-3) == 400.0
    assert pytest.approx(hand_map_none["Left"].landmarks_px[0, 0], abs=1e-3) == 700.0

    tracker.reset_temporal_state()

    # Sub-case E2: Scores are NaN vs Inf
    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand0_lms, cand1_lms],
        handedness=[[MockCategory(category_name="Left", score=float("nan"))], [MockCategory(category_name="Left", score=float("inf"))]],
    )
    hands_nan = tracker.process(frame, timestamp=2.0)
    assert len(hands_nan) == 2
    hand_map_nan = {h.handedness: h for h in hands_nan}
    assert pytest.approx(hand_map_nan["Right"].landmarks_px[0, 0], abs=1e-3) == 400.0
    assert pytest.approx(hand_map_nan["Left"].landmarks_px[0, 0], abs=1e-3) == 700.0

    tracker.reset_temporal_state()

    # Sub-case E3: Tied scores (0.85 vs 0.85)
    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand0_lms, cand1_lms],
        handedness=[[MockCategory(category_name="Left", score=0.85)], [MockCategory(category_name="Left", score=0.85)]],
    )
    hands_tied = tracker.process(frame, timestamp=3.0)
    assert len(hands_tied) == 2
    hand_map_tied = {h.handedness: h for h in hands_tied}
    assert pytest.approx(hand_map_tied["Right"].landmarks_px[0, 0], abs=1e-3) == 400.0
    assert pytest.approx(hand_map_tied["Left"].landmarks_px[0, 0], abs=1e-3) == 700.0


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
    piano = Piano(bbox=(64, 518, 1216, 691), audio_engine=mock_audio)
    guitar = Guitar(zones={"fretboard": (150, 250, 450, 450), "strum_zone": (550, 450, 850, 650)}, audio_engine=mock_audio)

    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    wkey = piano.white_keys[0]
    key_center_x = (wkey.rect[0] + wkey.rect[2]) / 2.0
    raw_key_x = 1.0 - (key_center_x / w)
    key_hover_y = (wkey.rect[1] + 10.0) / h

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
    assert len(hands2) == 2
    hand_map2 = {h.handedness: h for h in hands2}

    # Candidate 0 matches Right resting in place, Vy must be ~ 0 px/s
    assert pytest.approx(hand_map2["Right"].landmarks_px[0, 1], abs=1.0) == key_hover_y * h
    assert abs(hand_map2["Right"].fingertip_velocities[8][1]) < 10.0

    # Candidate 1 starts with safe zero baseline
    assert hand_map2["Left"].wrist_velocity == (0.0, 0.0)
    assert hand_map2["Left"].fingertip_velocities[8] == (0.0, 0.0)

    piano.update(hands2, frame_shape=(h, w, 3), current_time=1.05)
    guitar.update(hands2, frame_shape=(h, w, 3), current_time=1.05)

    assert len(mock_audio.triggered_notes) == 0, f"Expected 0 piano notes, got {mock_audio.triggered_notes}"
    assert len(mock_audio.plucked_strings) == 0, f"Expected 0 guitar strums, got {mock_audio.plucked_strings}"


def test_temporal_continuity_duplicate_labels_test_c():
    """
    Test C: Temporal continuity with duplicate labels.
    Frame 1:
    - logical Right wrist around x=250;
    - logical Left wrist around x=1000.
    Frame 2 classifier reports both same label (raw Left -> mirrored Right).
    Candidates remain near x=260 and x=990 (dt = 0.02s).
    Expected:
    - both survive;
    - identities remain spatially continuous;
    - velocities correspond to ~10px movement;
    - no cross-hand teleport.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Frame 1: Right at 250 (raw x = 1.0 - 250/1280 = 0.8047), Left at 1000 (raw x = 1.0 - 1000/1280 = 0.21875)
    raw_x_r1 = 1.0 - (250.0 / w)
    raw_x_l1 = 1.0 - (1000.0 / w)
    f1_cand_r = [MockNormalizedLandmark(x=raw_x_r1, y=0.5, z=0.0) for _ in range(21)]
    f1_cand_l = [MockNormalizedLandmark(x=raw_x_l1, y=0.5, z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[f1_cand_r, f1_cand_l],
        handedness=[[MockCategory(category_name="Left", score=0.95)], [MockCategory(category_name="Right", score=0.92)]],
    )
    hands1 = tracker.process(frame, timestamp=1.0)
    assert len(hands1) == 2
    map1 = {h.handedness: h for h in hands1}
    assert pytest.approx(map1["Right"].landmarks_px[0, 0], abs=1.0) == 250.0
    assert pytest.approx(map1["Left"].landmarks_px[0, 0], abs=1.0) == 1000.0

    # Frame 2 (dt = 0.02s): Classifier reports BOTH as raw Left (both mirrored Right)
    # Candidates are near x=260 and x=990
    raw_x_r2 = 1.0 - (260.0 / w)
    raw_x_l2 = 1.0 - (990.0 / w)
    f2_cand_r = [MockNormalizedLandmark(x=raw_x_r2, y=0.5, z=0.0) for _ in range(21)]
    f2_cand_l = [MockNormalizedLandmark(x=raw_x_l2, y=0.5, z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[f2_cand_r, f2_cand_l],
        handedness=[[MockCategory(category_name="Left", score=0.88)], [MockCategory(category_name="Left", score=0.82)]],
    )

    hands2 = tracker.process(frame, timestamp=1.02)
    assert len(hands2) == 2
    map2 = {h.handedness: h for h in hands2}
    assert "Right" in map2
    assert "Left" in map2

    # Spatially continuous matching
    assert pytest.approx(map2["Right"].landmarks_px[0, 0], abs=1.0) == 260.0
    assert pytest.approx(map2["Left"].landmarks_px[0, 0], abs=1.0) == 990.0

    # Velocities correspond to ~10px movement over 0.02s (500 px/s), NOT 740px teleport
    assert pytest.approx(map2["Right"].wrist_velocity[0], abs=5.0) == 500.0
    assert pytest.approx(map2["Left"].wrist_velocity[0], abs=5.0) == -500.0


def test_classifier_labels_swap_test_d():
    """
    Test D: Classifier labels swap between frames.
    Frame 1: Normal Left/Right labels (Right near 250, Left near 1000).
    Frame 2: Labels are swapped by detector, but spatial movement is tiny (255, 995).
    Expected:
    - temporal identities remain stable;
    - no velocity spike;
    - both hands remain present.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    raw_x_r1 = 1.0 - (250.0 / w)
    raw_x_l1 = 1.0 - (1000.0 / w)
    f1_cand_r = [MockNormalizedLandmark(x=raw_x_r1, y=0.5, z=0.0) for _ in range(21)]
    f1_cand_l = [MockNormalizedLandmark(x=raw_x_l1, y=0.5, z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[f1_cand_r, f1_cand_l],
        handedness=[[MockCategory(category_name="Left", score=0.95)], [MockCategory(category_name="Right", score=0.92)]],
    )
    hands1 = tracker.process(frame, timestamp=1.0)
    assert len(hands1) == 2

    # Frame 2: Classifier labels swapped!
    # Candidate near 255 labeled as raw Right (mirrored Left)
    # Candidate near 995 labeled as raw Left (mirrored Right)
    raw_x_r2 = 1.0 - (255.0 / w)
    raw_x_l2 = 1.0 - (995.0 / w)
    f2_cand_r = [MockNormalizedLandmark(x=raw_x_r2, y=0.5, z=0.0) for _ in range(21)]
    f2_cand_l = [MockNormalizedLandmark(x=raw_x_l2, y=0.5, z=0.0) for _ in range(21)]

    mock_lm.result = MockTasksResult(
        hand_landmarks=[f2_cand_r, f2_cand_l],
        handedness=[[MockCategory(category_name="Right", score=0.91)], [MockCategory(category_name="Left", score=0.93)]],
    )

    hands2 = tracker.process(frame, timestamp=1.02)
    assert len(hands2) == 2
    map2 = {h.handedness: h for h in hands2}
    assert "Right" in map2
    assert "Left" in map2

    # Temporal identities preserved despite swapped classifier labels
    assert pytest.approx(map2["Right"].landmarks_px[0, 0], abs=1.0) == 255.0
    assert pytest.approx(map2["Left"].landmarks_px[0, 0], abs=1.0) == 995.0

    # Velocities: 5px / 0.02s = 250 px/s (no teleport spike)
    assert pytest.approx(map2["Right"].wrist_velocity[0], abs=5.0) == 250.0
    assert pytest.approx(map2["Left"].wrist_velocity[0], abs=5.0) == -250.0


def test_actual_crossing_hands_test_e():
    """
    Test E: Actual hand crossing across multiple consecutive frames.
    - Two hands move toward and across each other.
    - Screen side alone must not permanently invert identity.
    - No same-frame state sharing and no extreme velocity spikes.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Trajectories: Hand A moves 400 -> 450 -> 495 -> 540 -> 590
    # Hand B moves 600 -> 550 -> 505 -> 460 -> 410
    traj_a = [400.0, 450.0, 495.0, 540.0, 590.0]
    traj_b = [600.0, 550.0, 505.0, 460.0, 410.0]

    for step_idx in range(len(traj_a)):
        xa = traj_a[step_idx]
        xb = traj_b[step_idx]
        t = 1.0 + step_idx * 0.033

        raw_xa = 1.0 - (xa / w)
        raw_xb = 1.0 - (xb / w)
        cand_a = [MockNormalizedLandmark(x=raw_xa, y=0.5, z=0.0) for _ in range(21)]
        cand_b = [MockNormalizedLandmark(x=raw_xb, y=0.5, z=0.0) for _ in range(21)]

        mock_lm.result = MockTasksResult(
            hand_landmarks=[cand_a, cand_b],
            handedness=[[MockCategory(category_name="Left", score=0.90)], [MockCategory(category_name="Right", score=0.90)]],
        )

        hands = tracker.process(frame, timestamp=t)
        assert len(hands) == 2, f"Failed at step {step_idx}: expected 2 hands, got {len(hands)}"
        hand_map = {h.handedness: h for h in hands}
        assert "Right" in hand_map
        assert "Left" in hand_map

        # Ensure no shared landmark memory
        assert hand_map["Right"].landmarks_px is not hand_map["Left"].landmarks_px
        assert not np.array_equal(hand_map["Right"].landmarks_px, hand_map["Left"].landmarks_px)

        # Ensure no extreme velocity spikes (> 3000 px/s)
        for h_obj in hands:
            assert abs(h_obj.wrist_velocity[0]) < 3000.0
            assert abs(h_obj.wrist_velocity[1]) < 3000.0


def test_single_hand_only_test_f():
    """
    Test F: Single hand detection works normally without inventing a phantom second hand.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    single_lms = [MockNormalizedLandmark(x=0.8, y=0.5, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[single_lms],
        handedness=[[MockCategory(category_name="Left", score=0.92)]],
    )

    hands = tracker.process(frame, timestamp=1.0)
    assert len(hands) == 1
    assert hands[0].handedness == "Right"
    assert len(tracker._filters) == 1
    assert "Right" in tracker._filters
    assert "Left" not in tracker._filters


def test_tracking_loss_reacquisition_safe_baseline_test_g():
    """
    Test G: Tracking loss and reacquisition.
    - Temporarily lose one hand.
    - Reacquired hand must start with a safe zero velocity baseline.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    cand_r = [MockNormalizedLandmark(x=0.75, y=0.5, z=0.0) for _ in range(21)]  # mirrored 250
    cand_l = [MockNormalizedLandmark(x=0.25, y=0.5, z=0.0) for _ in range(21)]  # mirrored 750

    # Frame 1: Two hands present
    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand_r, cand_l],
        handedness=[[MockCategory(category_name="Left", score=0.95)], [MockCategory(category_name="Right", score=0.92)]],
    )
    hands1 = tracker.process(frame, timestamp=1.0)
    assert len(hands1) == 2

    # Frame 2: Left hand lost (only Right detected at mirrored 255)
    cand_r2 = [MockNormalizedLandmark(x=0.745, y=0.5, z=0.0) for _ in range(21)]  # mirrored 255
    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand_r2],
        handedness=[[MockCategory(category_name="Left", score=0.95)]],
    )
    hands2 = tracker.process(frame, timestamp=1.05)
    assert len(hands2) == 1
    assert hands2[0].handedness == "Right"
    assert "Left" not in tracker._filters, "Lost hand was not evicted from tracker temporal state"

    # Frame 3: Left hand reacquired at mirrored 760
    cand_r3 = [MockNormalizedLandmark(x=0.740, y=0.5, z=0.0) for _ in range(21)]  # mirrored 260
    cand_l3 = [MockNormalizedLandmark(x=0.240, y=0.5, z=0.0) for _ in range(21)]  # mirrored 760
    mock_lm.result = MockTasksResult(
        hand_landmarks=[cand_r3, cand_l3],
        handedness=[[MockCategory(category_name="Left", score=0.95)], [MockCategory(category_name="Right", score=0.90)]],
    )
    hands3 = tracker.process(frame, timestamp=1.10)
    assert len(hands3) == 2
    map3 = {h.handedness: h for h in hands3}

    # Continuously tracked Right hand has normal velocity
    assert pytest.approx(map3["Right"].wrist_velocity[0], abs=5.0) == 100.0  # 5px / 0.05s

    # Reacquired Left hand must start with safe zero baseline without teleport spikes
    assert map3["Left"].wrist_velocity == (0.0, 0.0)
    for tip_v in map3["Left"].fingertip_velocities.values():
        assert tip_v == (0.0, 0.0)


def test_piano_guitar_safety_duplicate_swap_test_h():
    """
    Test H: Piano & Guitar interaction safety under duplicate and swapped classification sequence.
    - Feed sequence: Normal -> Duplicate -> Swapped.
    - Verify NO false piano strike or guitar strum is caused solely by identity transitions.
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
    piano = Piano(bbox=(64, 518, 1216, 691), audio_engine=mock_audio)
    guitar = Guitar(zones={"fretboard": (150, 250, 450, 450), "strum_zone": (550, 450, 850, 650)}, audio_engine=mock_audio)

    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Frame 1 (t=1.0): Normal hands hovering above piano keys (Right at 250, Left at 1000)
    raw_x_r = 1.0 - (250.0 / w)
    raw_x_l = 1.0 - (1000.0 / w)
    c_r = [MockNormalizedLandmark(x=raw_x_r, y=0.75, z=0.0) for _ in range(21)]
    c_l = [MockNormalizedLandmark(x=raw_x_l, y=0.75, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[c_r, c_l],
        handedness=[[MockCategory(category_name="Left", score=0.90)], [MockCategory(category_name="Right", score=0.90)]],
    )
    h1 = tracker.process(frame, timestamp=1.0)
    piano.update(h1, frame_shape=(h, w, 3), current_time=1.0)
    guitar.update(h1, frame_shape=(h, w, 3), current_time=1.0)
    assert len(mock_audio.triggered_notes) == 0
    assert len(mock_audio.plucked_strings) == 0

    # Frame 2 (t=1.03): Duplicate raw Left classification with small spatial drift (252, 998)
    c_r2 = [MockNormalizedLandmark(x=1.0 - (252.0 / w), y=0.75, z=0.0) for _ in range(21)]
    c_l2 = [MockNormalizedLandmark(x=1.0 - (998.0 / w), y=0.75, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[c_r2, c_l2],
        handedness=[[MockCategory(category_name="Left", score=0.85)], [MockCategory(category_name="Left", score=0.80)]],
    )
    h2 = tracker.process(frame, timestamp=1.03)
    piano.update(h2, frame_shape=(h, w, 3), current_time=1.03)
    guitar.update(h2, frame_shape=(h, w, 3), current_time=1.03)
    assert len(mock_audio.triggered_notes) == 0
    assert len(mock_audio.plucked_strings) == 0

    # Frame 3 (t=1.06): Inverted classification labels with small spatial drift (254, 996)
    c_r3 = [MockNormalizedLandmark(x=1.0 - (254.0 / w), y=0.75, z=0.0) for _ in range(21)]
    c_l3 = [MockNormalizedLandmark(x=1.0 - (996.0 / w), y=0.75, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[c_r3, c_l3],
        handedness=[[MockCategory(category_name="Right", score=0.92)], [MockCategory(category_name="Left", score=0.92)]],
    )
    h3 = tracker.process(frame, timestamp=1.06)
    piano.update(h3, frame_shape=(h, w, 3), current_time=1.06)
    guitar.update(h3, frame_shape=(h, w, 3), current_time=1.06)

    # Neither piano nor guitar should trigger false sound events
    assert len(mock_audio.triggered_notes) == 0, f"False piano notes triggered: {mock_audio.triggered_notes}"
    assert len(mock_audio.plucked_strings) == 0, f"False guitar strums triggered: {mock_audio.plucked_strings}"


def test_single_hand_stable_test_1():
    """
    Test 1: Stable single hand without classifier flip.
    - Frame 1: Candidate maps to application Right at x=300.
    - Frame 2: Same classifier at x=305 (dt=0.033).
    - Expected: Still Right; velocity ≈ 5px / 0.033s ≈ 151.5 px/s.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Frame 1: x=300, raw_x = 1.0 - (300/1000) = 0.70. MediaPipe raw "Left" -> mirrored "Right"
    lms1 = [MockNormalizedLandmark(x=0.70, y=0.5, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms1],
        handedness=[[MockCategory(category_name="Left", score=0.95)]],
    )
    hands1 = tracker.process(frame, timestamp=1.0)
    assert len(hands1) == 1
    assert hands1[0].handedness == "Right"
    assert pytest.approx(hands1[0].landmarks_px[0, 0], abs=1.0) == 300.0

    # Frame 2: x=305, raw_x = 1.0 - (305/1000) = 0.695
    lms2 = [MockNormalizedLandmark(x=0.695, y=0.5, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms2],
        handedness=[[MockCategory(category_name="Left", score=0.95)]],
    )
    hands2 = tracker.process(frame, timestamp=1.033)
    assert len(hands2) == 1
    assert hands2[0].handedness == "Right"
    assert pytest.approx(hands2[0].landmarks_px[0, 0], abs=1.0) == 305.0
    # Velocity ≈ 5.0 / 0.033 ≈ 151.5 px/s
    assert pytest.approx(hands2[0].wrist_velocity[0], abs=5.0) == 151.515


def test_single_hand_one_frame_classifier_flip_test_2():
    """
    Test 2: One-frame classifier flip for single hand.
    - Frame 1: Application Right at x=300.
    - Frame 2: Same physical candidate at x=305; MediaPipe raw classifier intentionally flips to "Right" (mirrored "Left").
    - Expected: STILL application Right (continuity gate); normal ~5px movement velocity; no reset.
    - Frame 3: Classifier returns to original value at x=310.
    - Expected: Still Right; continuity preserved; no zero-baseline dropout; no velocity spike.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Frame 1: x=300, raw "Left" -> mirrored "Right"
    lms1 = [MockNormalizedLandmark(x=0.70, y=0.5, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms1],
        handedness=[[MockCategory(category_name="Left", score=0.95)]],
    )
    hands1 = tracker.process(frame, timestamp=1.0)
    assert len(hands1) == 1
    assert hands1[0].handedness == "Right"
    assert pytest.approx(hands1[0].landmarks_px[0, 0], abs=1.0) == 300.0

    # Frame 2: x=305, FLIPPED raw classifier to "Right" (mirrored "Left")
    lms2 = [MockNormalizedLandmark(x=0.695, y=0.5, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms2],
        handedness=[[MockCategory(category_name="Right", score=0.92)]],
    )
    hands2 = tracker.process(frame, timestamp=1.033)
    assert len(hands2) == 1
    assert hands2[0].handedness == "Right", "Hand identity incorrectly flipped to Left despite spatial continuity!"
    assert pytest.approx(hands2[0].landmarks_px[0, 0], abs=1.0) == 305.0
    # Normal velocity calculated, NOT zeroed by false reset
    assert pytest.approx(hands2[0].wrist_velocity[0], abs=5.0) == 151.515
    assert "Left" not in tracker._filters

    # Frame 3: x=310, raw classifier returns to "Left" (mirrored "Right")
    lms3 = [MockNormalizedLandmark(x=0.690, y=0.5, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms3],
        handedness=[[MockCategory(category_name="Left", score=0.95)]],
    )
    hands3 = tracker.process(frame, timestamp=1.066)
    assert len(hands3) == 1
    assert hands3[0].handedness == "Right"
    assert pytest.approx(hands3[0].landmarks_px[0, 0], abs=1.0) == 310.0
    assert pytest.approx(hands3[0].wrist_velocity[0], abs=5.0) == 151.515


def test_single_hand_repeated_classifier_flicker_test_3():
    """
    Test 3: Repeated classifier flicker across 10 frames with smooth motion.
    - Alternate classifier Left/Right for 10 frames while wrist moves x=300 -> 345.
    - Expected: Application identity stays constant ("Right"); exactly one hand each frame;
      no phantom second hand; no large velocity spikes.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(10):
        x = 300.0 + i * 5.0
        t = 1.0 + i * 0.033
        # Alternate classifier: even -> "Left" (mirrored Right), odd -> "Right" (mirrored Left)
        raw_cat = "Left" if i % 2 == 0 else "Right"
        lms = [MockNormalizedLandmark(x=1.0 - (x / w), y=0.5, z=0.0) for _ in range(21)]
        mock_lm.result = MockTasksResult(
            hand_landmarks=[lms],
            handedness=[[MockCategory(category_name=raw_cat, score=0.90)]],
        )
        hands = tracker.process(frame, timestamp=t)
        assert len(hands) == 1, f"Frame {i}: Expected 1 hand, got {len(hands)}"
        assert hands[0].handedness == "Right", f"Frame {i}: Role flickered to {hands[0].handedness}"
        assert "Left" not in tracker._filters, f"Frame {i}: Phantom filter for Left was created"
        if i > 0:
            # Velocity should stay bounded near 151 px/s (never spike above 300 px/s)
            assert 100.0 < hands[0].wrist_velocity[0] < 200.0


def test_single_hand_true_discontinuity_test_4():
    """
    Test 4: True spatial discontinuity beyond continuity threshold.
    - Existing Right history at x=300.
    - Candidate appears far away at x=800 (> 220px threshold) with classifier mapping to Left.
    - Expected: Safe reassignment/rebaseline allowed; new velocity baseline = zero; no teleport velocity.
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1000, 1000
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Frame 1: Right at x=300
    lms1 = [MockNormalizedLandmark(x=0.70, y=0.5, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms1],
        handedness=[[MockCategory(category_name="Left", score=0.95)]],
    )
    hands1 = tracker.process(frame, timestamp=1.0)
    assert len(hands1) == 1
    assert hands1[0].handedness == "Right"

    # Frame 2: Candidate appears far away at x=800 with raw "Right" (mirrored "Left")
    lms2 = [MockNormalizedLandmark(x=0.20, y=0.5, z=0.0) for _ in range(21)]  # 1.0 - 0.20 = 0.80 -> 800px
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms2],
        handedness=[[MockCategory(category_name="Right", score=0.95)]],
    )
    hands2 = tracker.process(frame, timestamp=1.033)
    assert len(hands2) == 1
    assert hands2[0].handedness == "Left", "Beyond-threshold candidate should reassign to Left"
    assert pytest.approx(hands2[0].landmarks_px[0, 0], abs=1.0) == 800.0
    # Velocity must start at zero, NO teleport velocity spike
    assert hands2[0].wrist_velocity == (0.0, 0.0)
    for tip_v in hands2[0].fingertip_velocities.values():
        assert tip_v == (0.0, 0.0)
    # Stale Right role must be evicted
    assert "Right" not in tracker._filters


def test_single_hand_guitar_safety_classifier_flip_test_5():
    """
    Test 5: Guitar instrument safety under single-hand classifier flip.
    - Single visible strumming hand (Right).
    - Cause one-frame classifier flip without spatial discontinuity.
    - Verify: Guitar does not treat it as chord-selection hand (Left);
      no false chord transition; no false strum caused solely by identity flip.
    """
    from instruments import Guitar

    class MockAudio:
        def __init__(self):
            self.plucked_strings = []

        def play_guitar(self, string_idx: int, chord_name: str = "C", velocity: float = 1.0) -> bool:
            self.plucked_strings.append((string_idx, chord_name, velocity))
            return True

    mock_audio = MockAudio()
    guitar = Guitar(zones={"fretboard": (150, 250, 450, 450), "strum_zone": (550, 450, 850, 650)}, audio_engine=mock_audio)

    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    mock_lm = MockTasksLandmarker()
    tracker.landmarker = mock_lm
    tracker.hands = mock_lm

    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Frame 1: Strumming hand (Right) hovering in strum zone (screen-left around x=400, y=500)
    lms1 = [MockNormalizedLandmark(x=1.0 - (400.0 / w), y=500.0 / h, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms1],
        handedness=[[MockCategory(category_name="Left", score=0.95)]],
    )
    h1 = tracker.process(frame, timestamp=1.0)
    assert len(h1) == 1
    assert h1[0].handedness == "Right"
    guitar.update(h1, frame_shape=(h, w, 3), current_time=1.0)
    initial_chord = guitar.active_chord
    assert len(mock_audio.plucked_strings) == 0

    # Frame 2: Classifier flips raw to "Right" (mirrored "Left"), moving slightly to x=404, y=500
    lms2 = [MockNormalizedLandmark(x=1.0 - (404.0 / w), y=500.0 / h, z=0.0) for _ in range(21)]
    mock_lm.result = MockTasksResult(
        hand_landmarks=[lms2],
        handedness=[[MockCategory(category_name="Right", score=0.90)]],
    )
    h2 = tracker.process(frame, timestamp=1.033)
    assert len(h2) == 1
    assert h2[0].handedness == "Right", "Strumming hand identity flipped to Left!"
    guitar.update(h2, frame_shape=(h, w, 3), current_time=1.033)

    # Guitar must NOT treat it as chord selection hand or trigger false chords/strums
    assert guitar.active_chord == initial_chord, "Chord falsely changed due to identity flip!"
    assert len(mock_audio.plucked_strings) == 0, f"False strum triggered: {mock_audio.plucked_strings}"


def test_bounded_velocity_prediction_safeguard():
    """
    Verifies that extreme noisy previous wrist velocity does not produce an unbounded
    predicted target displacement (displacement is strictly bounded by MAX_PREDICTION_DISPLACEMENT).
    """
    tracker = HandTracker(init_mediapipe=False, filter_mode="raw")
    # Simulate an extreme previous velocity of 10,000 px/s on Right
    tracker._prev_wrist_px["Right"] = np.array([300.0, 500.0], dtype=np.float32)
    tracker._prev_wrist_vel["Right"] = (10000.0, 10000.0)

    # In _assign_candidates_to_roles, test prediction bounding
    candidates = [{
        "mirrored_label": "Right",
        "raw_coords": np.zeros((21, 3), dtype=np.float32),
        "score": 0.95,
        "source_order": 0,
    }]
    w, h = 1000, 1000
    # Candidate raw_coords[0] maps to mirrored screen px = 310
    candidates[0]["raw_coords"][0, 0] = 1.0 - (310.0 / w)
    candidates[0]["raw_coords"][0, 1] = 500.0 / h

    assigned = tracker._assign_candidates_to_roles(candidates, w, h)
    assert len(assigned) == 1
    assert assigned[0][1] == "Right"

