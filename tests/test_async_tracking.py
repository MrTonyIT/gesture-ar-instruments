"""
tests/test_async_tracking.py
============================
Unit tests for asynchronous hand tracking timestamp semantics:
- Frame measurement timestamp propagation from ThreadedCamera
- Kinematic dead-reckoning extrapolation age derived from camera capture time
- Forward projection clamping at 60ms
- Strict coordinate consistency invariant between landmarks_px and landmarks_norm
"""

import numpy as np
import pytest

from vision_tracker import AsyncHandTracker, HandData


class DummyCamera:
    """Mock camera providing configured frame dimensions without hardware."""

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self.width = width
        self.height = height
        self.target_width = width
        self.target_height = height

    def read_sequenced(self):
        return False, None, 0, 0.0


@pytest.fixture
def async_tracker():
    cam = DummyCamera(width=1920, height=1080)
    tracker = AsyncHandTracker(
        camera=cam,
        max_num_hands=2,
        model_complexity=0,
        init_mediapipe=False,
    )
    return tracker


def test_extrapolation_age_uses_measurement_timestamp(async_tracker):
    """
    Verifies that kinematic dead-reckoning extrapolation measures age from
    the optical frame capture timestamp (T0), NOT the AI publication timestamp (T1).
    """
    t_capture = 10.000       # T0: camera frame capture
    t_publish = 10.016       # T1: AI inference completed and published
    t_render = 10.040        # T2: UI frame render time

    w, h = 1920, 1080
    initial_px = np.zeros((21, 3), dtype=np.float32)
    initial_px[:, 0] = 500.0
    initial_px[:, 1] = 400.0

    initial_norm = np.zeros((21, 3), dtype=np.float32)
    initial_norm[:, 0] = 500.0 / w
    initial_norm[:, 1] = 400.0 / h

    hand = HandData(
        handedness="Right",
        landmarks_norm=initial_norm.copy(),
        landmarks_px=initial_px.copy(),
        fingertip_velocities={8: (100.0, 50.0)},
        wrist_velocity=(50.0, 20.0),
        timestamp=t_capture,
        inference_timestamp=t_publish,
    )

    with async_tracker._snapshot_lock:
        async_tracker._latest_hands = [hand]
        async_tracker._latest_measurement_ts = t_capture
        async_tracker._latest_publication_ts = t_publish
        async_tracker._last_frame_w = w
        async_tracker._last_frame_h = h

    # Extrapolate to t_render
    extrapolated = async_tracker.get_latest_hands(current_time=t_render, extrapolate=True)
    assert len(extrapolated) == 1
    ext_hand = extrapolated[0]

    # Expected dt = t_render - t_capture = 0.040s (40ms)
    expected_dt = t_render - t_capture  # 0.040
    expected_wrist_dx = 50.0 * expected_dt  # +2.0 px
    expected_wrist_dy = 20.0 * expected_dt  # +0.8 px

    # Wrist is landmark 0
    np.testing.assert_allclose(ext_hand.landmarks_px[0, 0], 500.0 + expected_wrist_dx, rtol=1e-4)
    np.testing.assert_allclose(ext_hand.landmarks_px[0, 1], 400.0 + expected_wrist_dy, rtol=1e-4)

    # Fingertip 8 has wrist_velocity + 0.5 * tip_velocity
    expected_tip8_dx = expected_wrist_dx + 100.0 * expected_dt * 0.5  # 2.0 + 2.0 = 4.0 px
    expected_tip8_dy = expected_wrist_dy + 50.0 * expected_dt * 0.5   # 0.8 + 1.0 = 1.8 px
    np.testing.assert_allclose(ext_hand.landmarks_px[8, 0], 500.0 + expected_tip8_dx, rtol=1e-4)
    np.testing.assert_allclose(ext_hand.landmarks_px[8, 1], 400.0 + expected_tip8_dy, rtol=1e-4)


def test_extrapolation_age_clamping(async_tracker):
    """Verifies that extrapolation dt is safely clamped to 60ms maximum to prevent drift."""
    t_capture = 10.0
    t_render = 10.500  # 500ms elapsed (e.g. temporary hitch)

    w, h = 1920, 1080
    px = np.full((21, 3), 400.0, dtype=np.float32)
    norm = np.zeros((21, 3), dtype=np.float32)
    norm[:, 0] = 400.0 / w
    norm[:, 1] = 400.0 / h

    hand = HandData(
        handedness="Left",
        landmarks_norm=norm,
        landmarks_px=px,
        fingertip_velocities={},
        wrist_velocity=(100.0, 0.0),
        timestamp=t_capture,
        inference_timestamp=t_capture + 0.015,
    )

    with async_tracker._snapshot_lock:
        async_tracker._latest_hands = [hand]
        async_tracker._latest_measurement_ts = t_capture
        async_tracker._last_frame_w = w
        async_tracker._last_frame_h = h

    extrapolated = async_tracker.get_latest_hands(current_time=t_render, extrapolate=True)
    ext_hand = extrapolated[0]

    # Clamped to 0.060s: dx = 100.0 * 0.060 = 6.0 px
    expected_x = 400.0 + 100.0 * 0.060
    np.testing.assert_allclose(ext_hand.landmarks_px[0, 0], expected_x, rtol=1e-4)


def test_coordinate_synchronization_invariant(async_tracker):
    """
    Verifies that after extrapolation, normalized coordinates strictly mirror
    pixel coordinates scaled by frame dimensions:
        landmarks_norm[:, 0] == landmarks_px[:, 0] / frame_w
        landmarks_norm[:, 1] == landmarks_px[:, 1] / frame_h
        landmarks_norm[:, 2] == landmarks_px[:, 2] / frame_w
    """
    w, h = 1280, 720
    np.random.seed(42)
    px = np.random.uniform(100.0, 600.0, (21, 3)).astype(np.float32)
    norm = np.zeros((21, 3), dtype=np.float32)
    norm[:, 0] = px[:, 0] / w
    norm[:, 1] = px[:, 1] / h
    norm[:, 2] = px[:, 2] / w

    hand = HandData(
        handedness="Right",
        landmarks_norm=norm,
        landmarks_px=px,
        fingertip_velocities={4: (80.0, -30.0), 8: (120.0, 90.0)},
        wrist_velocity=(40.0, -25.0),
        timestamp=5.0,
        inference_timestamp=5.015,
    )

    with async_tracker._snapshot_lock:
        async_tracker._latest_hands = [hand]
        async_tracker._latest_measurement_ts = 5.0
        async_tracker._last_frame_w = w
        async_tracker._last_frame_h = h

    extrapolated = async_tracker.get_latest_hands(current_time=5.035, extrapolate=True)
    ext = extrapolated[0]

    np.testing.assert_allclose(ext.landmarks_norm[:, 0], ext.landmarks_px[:, 0] / w, rtol=1e-5)
    np.testing.assert_allclose(ext.landmarks_norm[:, 1], ext.landmarks_px[:, 1] / h, rtol=1e-5)
    np.testing.assert_allclose(ext.landmarks_norm[:, 2], ext.landmarks_px[:, 2] / w, rtol=1e-5)


def test_first_camera_frame_timestamp():
    """
    Verifies that ThreadedCamera.start() produces an initial frame with:
    - a valid monotonic capture timestamp (> 0.0)
    - a valid non-stale frame id (frame_id >= 1)
    - synchronized _current_frame_id and _current_timestamp
    - synchronized public telemetry fields
    - read_sequenced() never returns a valid frame with an artificial timestamp == 0.0
    """
    import time
    from vision_tracker import ThreadedCamera

    # Non-existent device ID 999 triggers synthetic simulation mode deterministically
    cam = ThreadedCamera(src=999, width=640, height=480)
    t_before = time.perf_counter()
    cam.start()
    try:
        ret, frame, frame_id, timestamp = cam.read_sequenced()
        assert ret is True
        assert frame is not None
        assert frame_id >= 1
        assert timestamp >= t_before
        assert cam.frame_id >= 1
        assert cam.frame_timestamp >= t_before
        assert cam._current_frame_id == frame_id
        assert cam._current_timestamp == timestamp
        assert timestamp != 0.0
    finally:
        cam.stop()

