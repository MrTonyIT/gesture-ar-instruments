"""
tests/test_async_tracking.py
============================
Unit tests for asynchronous hand tracking timestamp semantics:
- Frame measurement timestamp propagation from ThreadedCamera
- Kinematic dead-reckoning extrapolation age derived from camera capture time
- Forward projection clamping at 60ms
- Strict coordinate consistency invariant between landmarks_px and landmarks_norm
"""

import threading
import time

import numpy as np
import pytest

from vision_tracker import AsyncHandTracker, HandData, ThreadedCamera


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
        tracking_profile="RESPONSIVE",
        init_mediapipe=False,
    )
    return tracker


def test_extrapolation_age_uses_measurement_timestamp(async_tracker):
    """
    Verifies that kinematic dead-reckoning extrapolation measures age from
    the host acquisition timestamp (T0), NOT the AI publication timestamp (T1).
    """
    t_capture = 10.000       # T0: monotonic host acquisition timestamp
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

    # Fingertip 8 has wrist_velocity + 0.5 * (tip_velocity - wrist_velocity)
    rel_vx = 100.0 - 50.0  # 50.0
    rel_vy = 50.0 - 20.0   # 30.0
    expected_tip8_dx = expected_wrist_dx + rel_vx * expected_dt * 0.5  # 2.0 + 1.0 = 3.0 px
    expected_tip8_dy = expected_wrist_dy + rel_vy * expected_dt * 0.5  # 0.8 + 0.6 = 1.4 px
    np.testing.assert_allclose(ext_hand.landmarks_px[8, 0], 500.0 + expected_tip8_dx, rtol=1e-4)
    np.testing.assert_allclose(ext_hand.landmarks_px[8, 1], 400.0 + expected_tip8_dy, rtol=1e-4)


def test_extrapolation_rigid_translation(async_tracker):
    """
    Verifies that under rigid hand translation (where all fingertip velocities equal
    the wrist velocity), relative velocity is zero, so fingertips experience zero
    excess displacement beyond the global wrist translation (no double-counting).
    """
    t_capture = 1.0
    t_render = 1.040  # dt = 40ms
    w, h = 1920, 1080

    initial_px = np.full((21, 3), 300.0, dtype=np.float32)
    initial_norm = np.zeros((21, 3), dtype=np.float32)
    initial_norm[:, 0] = initial_px[:, 0] / w
    initial_norm[:, 1] = initial_px[:, 1] / h

    # Rigid translation: wrist and all fingertips moving at identical (80.0, -40.0) px/s
    hand = HandData(
        handedness="Right",
        landmarks_norm=initial_norm.copy(),
        landmarks_px=initial_px.copy(),
        fingertip_velocities={4: (80.0, -40.0), 8: (80.0, -40.0), 12: (80.0, -40.0)},
        wrist_velocity=(80.0, -40.0),
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

    dt = 0.040
    expected_dx = 80.0 * dt  # 3.2 px
    expected_dy = -40.0 * dt  # -1.6 px

    # Wrist (0) and fingertips (4, 8, 12) must all translate identically
    for idx in [0, 4, 8, 12]:
        np.testing.assert_allclose(ext_hand.landmarks_px[idx, 0], 300.0 + expected_dx, rtol=1e-4)
        np.testing.assert_allclose(ext_hand.landmarks_px[idx, 1], 300.0 + expected_dy, rtol=1e-4)


def test_extrapolation_articulation(async_tracker):
    """
    Verifies that when the wrist is stationary and a fingertip articulates independently,
    only the articulating fingertip is projected forward by its relative velocity.
    """
    t_capture = 2.0
    t_render = 2.030  # dt = 30ms
    w, h = 1920, 1080

    initial_px = np.full((21, 3), 400.0, dtype=np.float32)
    initial_norm = np.zeros((21, 3), dtype=np.float32)
    initial_norm[:, 0] = initial_px[:, 0] / w
    initial_norm[:, 1] = initial_px[:, 1] / h

    # Stationary wrist, index finger striking downward at (0.0, 120.0) px/s
    hand = HandData(
        handedness="Right",
        landmarks_norm=initial_norm.copy(),
        landmarks_px=initial_px.copy(),
        fingertip_velocities={8: (0.0, 120.0)},
        wrist_velocity=(0.0, 0.0),
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

    dt = 0.030
    # Wrist did not move
    np.testing.assert_allclose(ext_hand.landmarks_px[0, 0], 400.0, rtol=1e-4)
    np.testing.assert_allclose(ext_hand.landmarks_px[0, 1], 400.0, rtol=1e-4)

    # Tip 8 moved by relative velocity * dt * 0.5 = 120.0 * 0.030 * 0.5 = 1.8 px
    expected_tip8_y = 400.0 + 120.0 * dt * 0.5
    np.testing.assert_allclose(ext_hand.landmarks_px[8, 0], 400.0, rtol=1e-4)
    np.testing.assert_allclose(ext_hand.landmarks_px[8, 1], expected_tip8_y, rtol=1e-4)


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
    Uses an injected MockVideoCapture to avoid device queries or platform-specific driver delays.
    """
    import time
    from vision_tracker import ThreadedCamera

    class MockVideoCapture:
        def __init__(self):
            self._opened = True
            self._frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def isOpened(self):
            return self._opened

        def set(self, prop, val):
            return True

        def get(self, prop):
            return 0.0

        def read(self):
            return True, self._frame

        def grab(self):
            return True

        def retrieve(self):
            return True, self._frame

        def release(self):
            self._opened = False

    mock_cap = MockVideoCapture()
    cam = ThreadedCamera(src=0, width=640, height=480, cap=mock_cap)
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


def test_async_hand_tracker_stop_bounded_timeout_when_stuck():
    """
    Verifies that if the AsyncHandTracker background worker thread is blocked/stuck,
    calling stop(timeout=0.1) returns within a bounded window without hanging indefinitely,
    and safely skips tracker.close() to prevent deadlock/race.
    """
    import threading
    import time
    from vision_tracker import AsyncHandTracker

    worker_stuck_event = threading.Event()
    unblock_worker_event = threading.Event()
    close_called = False

    class StuckFakeCamera:
        def read_sequenced(self):
            worker_stuck_event.set()
            # Simulate worker blocked inside driver / external I/O
            unblock_worker_event.wait(timeout=2.0)
            return False, None, 0, 0.0

    class DummyTracker:
        def close(self):
            nonlocal close_called
            close_called = True

    cam = StuckFakeCamera()
    tracker = AsyncHandTracker(camera=cam, init_mediapipe=False)
    tracker.tracker = DummyTracker()
    tracker.start()

    try:
        # Wait until worker begins execution
        assert worker_stuck_event.wait(timeout=2.0)

        # Call stop with short timeout
        t0 = time.perf_counter()
        tracker.stop(timeout=0.1)
        dur = time.perf_counter() - t0

        # Must return in bounded time (around 0.1s, well under 1.0s)
        assert dur < 1.0, f"stop() hung for {dur:.3f}s; must be bounded"
        # Since worker was still alive, close() must have been skipped
        assert not close_called, "tracker.close() must not be called while worker is still alive"
    finally:
        # Clean up worker thread
        unblock_worker_event.set()
        if tracker._thread is not None:
            tracker._thread.join(timeout=1.0)


def test_threaded_camera_stop_race_safety():
    """
    Verifies that if ThreadedCamera's worker thread is still running when stop(timeout=0.1)
    is called, cap.release() is safely skipped to avoid concurrent driver access,
    and can subsequently be cleaned up safely when the thread exits.
    """
    import threading
    import time
    from vision_tracker import ThreadedCamera

    worker_blocked_event = threading.Event()
    unblock_worker_event = threading.Event()
    cap_released = False

    class BlockingCap:
        def __init__(self):
            self._opened = True

        def isOpened(self):
            return self._opened

        def set(self, prop, val):
            return True

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def grab(self):
            worker_blocked_event.set()
            unblock_worker_event.wait(timeout=2.0)
            return True

        def retrieve(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            nonlocal cap_released
            cap_released = True
            self._opened = False

    blocking_cap = BlockingCap()
    cam = ThreadedCamera(src=0, width=640, height=480, cap=blocking_cap)
    cam.start()

    try:
        assert worker_blocked_event.wait(timeout=2.0)
        t0 = time.perf_counter()
        cam.stop(timeout=0.1)
        dur = time.perf_counter() - t0

        assert dur < 1.0, f"ThreadedCamera.stop() hung for {dur:.3f}s"
        # Worker was still alive at join timeout -> release skipped
        assert not cap_released, "cap.release() must not be called while worker thread is alive"
    finally:
        unblock_worker_event.set()
        if cam._thread is not None:
            cam._thread.join(timeout=1.0)
        # Now worker is stopped, calling stop() again should release cap safely
        cam.stop(timeout=1.0)
        assert cap_released, "cap.release() should be called once worker is terminated"



def test_async_hand_tracker_stop_race_safety():
    """
    Deterministic concurrency regression test proving that tracker.close()
    cannot execute while the worker thread is inside an inference section
    holding _config_lock, and that close() executes exactly once upon shutdown.
    """
    import threading
    from vision_tracker import AsyncHandTracker

    inference_entered = threading.Event()
    release_inference = threading.Event()
    close_called = threading.Event()
    close_call_count = 0
    in_inference = False

    class FakeCamera:
        def __init__(self):
            self._frame_id = 0

        def read_sequenced(self):
            self._frame_id += 1
            return True, np.zeros((10, 10, 3), dtype=np.uint8), self._frame_id, 100.0 + self._frame_id * 0.016

    class FakeTracker:
        def __init__(self):
            self.hands = None

        def process(self, frame, timestamp=None):
            nonlocal in_inference
            in_inference = True
            inference_entered.set()
            released = release_inference.wait(timeout=5.0)
            in_inference = False
            assert released, "Timeout waiting for test to release inference"
            return []

        def close(self):
            nonlocal close_call_count
            assert not in_inference, "Fatal race: tracker.close() called while inference is actively executing!"
            close_call_count += 1
            close_called.set()

    fake_cam = FakeCamera()
    fake_tracker = FakeTracker()

    async_tracker = AsyncHandTracker(camera=fake_cam, init_mediapipe=False)
    async_tracker.tracker = fake_tracker

    # Start the worker thread
    async_tracker.start()

    # 1. Wait until worker enters inference section (inside with self._config_lock:)
    assert inference_entered.wait(timeout=5.0), "Worker did not enter inference section"
    assert in_inference, "Worker must be actively in inference"
    assert async_tracker._config_lock.locked(), "_config_lock must be held during inference"

    # 2. Request stop() from another thread while inference holds _config_lock
    stop_thread = threading.Thread(target=async_tracker.stop, name="StopThread")
    stop_thread.start()

    # 3. Verify tracker.close() has NOT executed while inference owns _config_lock
    assert not close_called.is_set(), "tracker.close() executed prematurely while inference was running"
    assert close_call_count == 0

    # 4. Release the fake inference so worker can finish current step
    release_inference.set()

    # 5. Wait for stop_thread to complete (joins worker thread and calls close under _config_lock)
    stop_thread.join(timeout=5.0)
    assert not stop_thread.is_alive(), "stop() failed to complete within timeout"

    # 6. Verify worker thread has exited and close executed exactly once
    assert not async_tracker.is_running
    if async_tracker._thread is not None:
        assert not async_tracker._thread.is_alive(), "Worker thread must have exited"
    assert close_called.is_set(), "tracker.close() must be called on shutdown"
    assert close_call_count == 1, f"Expected close() to execute exactly once, got {close_call_count}"


def test_threaded_camera_backend_detection_and_settings_dialog(monkeypatch):
    """
    Hardware-independent tests proving:
    1. Truthful backend detection via getBackendName() in auto mode (DSHOW -> dshow, MSMF -> msmf).
    2. Fallback to 'auto' when backend detection is unavailable.
    3. Detected DirectShow permits the settings dialog path on Windows.
    4. Detected MSMF rejects the settings dialog path.
    """
    import sys
    import cv2
    from vision_tracker import ThreadedCamera

    class FakeCap:
        def __init__(self, backend_name=None, settings_result=True):
            self._backend_name = backend_name
            self._settings_result = settings_result
            self._opened = True
            self.settings_called = False

        def isOpened(self):
            return self._opened

        def set(self, prop, val):
            if prop == cv2.CAP_PROP_SETTINGS:
                self.settings_called = True
                return self._settings_result
            return True

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def grab(self):
            return True

        def retrieve(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            self._opened = False

        def getBackendName(self):
            if self._backend_name is None:
                raise AttributeError("No backend name available")
            return self._backend_name

    # 1. DirectShow detected in auto mode -> active_backend becomes "dshow" and permits settings dialog
    dshow_cap = FakeCap(backend_name="DSHOW", settings_result=True)
    cam_dshow = ThreadedCamera(src=0, width=640, height=480, backend="auto", cap=dshow_cap)
    cam_dshow.start()
    try:
        assert cam_dshow.active_backend == "dshow"
        # On Windows, settings dialog is permitted and calls CAP_PROP_SETTINGS
        monkeypatch.setattr(sys, "platform", "win32")
        result = cam_dshow.open_settings_dialog()
        assert result is True
        assert dshow_cap.settings_called is True
    finally:
        cam_dshow.stop()

    # 2. MSMF detected in auto mode -> active_backend becomes "msmf" and rejects settings dialog
    msmf_cap = FakeCap(backend_name="MSMF", settings_result=True)
    cam_msmf = ThreadedCamera(src=0, width=640, height=480, backend="auto", cap=msmf_cap)
    cam_msmf.start()
    try:
        assert cam_msmf.active_backend == "msmf"
        monkeypatch.setattr(sys, "platform", "win32")
        result = cam_msmf.open_settings_dialog()
        # Must be rejected because active_backend is msmf, not dshow
        assert result is False
        assert msmf_cap.settings_called is False
    finally:
        cam_msmf.stop()

    # 3. No getBackendName available -> keeps fallback "auto"
    class FakeCapNoBackend:
        def __init__(self):
            self._opened = True

        def isOpened(self):
            return self._opened

        def set(self, prop, val):
            return True

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def grab(self):
            return True

        def retrieve(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            self._opened = False

    fallback_cap = FakeCapNoBackend()
    cam_auto = ThreadedCamera(src=0, width=640, height=480, backend="auto", cap=fallback_cap)
    cam_auto.start()
    try:
        assert cam_auto.active_backend == "auto"
    finally:
        cam_auto.stop()


def test_camera_settings_io_serialization(monkeypatch):
    """
    Deterministic concurrency test with threading.Event proving that while
    VideoCapture grab/retrieve is active, CAP_PROP_SETTINGS cannot execute concurrently,
    and once grab/retrieve finishes, settings executes safely under the lock.
    No time.sleep() guessing.
    """
    import threading
    import sys
    import cv2
    from vision_tracker import ThreadedCamera

    io_started = threading.Event()
    allow_io_finish = threading.Event()
    settings_called_while_in_io = False
    inside_io = False

    class FakeSyncCap:
        def __init__(self):
            self._opened = True
            self.settings_invoked = False

        def isOpened(self):
            return self._opened

        def getBackendName(self):
            return "DSHOW"

        def set(self, prop, val):
            nonlocal inside_io, settings_called_while_in_io
            if prop == cv2.CAP_PROP_SETTINGS:
                if inside_io:
                    settings_called_while_in_io = True
                self.settings_invoked = True
                return True
            return True

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def grab(self):
            nonlocal inside_io
            inside_io = True
            io_started.set()
            # Block inside grab until test allows it to proceed
            allow_io_finish.wait(timeout=5.0)
            return True

        def retrieve(self):
            nonlocal inside_io
            inside_io = False
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            self._opened = False

    monkeypatch.setattr(sys, "platform", "win32")
    fake_cap = FakeSyncCap()
    cam = ThreadedCamera(src=0, width=640, height=480, backend="dshow", cap=fake_cap)
    cam.start()

    try:
        # Wait until capture loop is actively holding _cap_io_lock inside grab()
        assert io_started.wait(timeout=3.0), "Capture loop failed to enter grab()"

        # Concurrently attempt open_settings_dialog in a separate thread
        settings_result = [None]
        def invoke_settings():
            settings_result[0] = cam.open_settings_dialog()

        t_settings = threading.Thread(target=invoke_settings)
        t_settings.start()

        # Brief assertion: prove settings has not been invoked yet while inside_io is True
        assert fake_cap.settings_invoked is False
        assert settings_called_while_in_io is False

        # Release capture loop to complete its retrieve() and unlock _cap_io_lock
        allow_io_finish.set()

        # Settings dialog thread should now acquire lock and finish
        t_settings.join(timeout=3.0)
        assert t_settings.is_alive() is False
        assert settings_result[0] is True
        assert fake_cap.settings_invoked is True
        assert settings_called_while_in_io is False
    finally:
        allow_io_finish.set()
        cam.stop()


def test_temporal_state_reset_and_motion_collision():
    """
    Validates temporal tracking state reset across filter and model hot-swaps:
    - filter switch produces zero baseline velocity on first post-switch observation;
    - model switch produces zero baseline velocity on first post-switch observation;
    - no false Piano note is generated by the switch;
    - no false Guitar strum is generated by the switch;
    - subsequent real movement computes velocity normally.
    """
    from vision_tracker import HandTracker
    from instruments import Piano, Guitar

    tracker = HandTracker(init_mediapipe=False, filter_mode="one_euro", tracking_profile="STABLE")

    # Mock hands result simulation for HandTracker
    class MockClassification:
        def __init__(self, label):
            self.label = label

    class MockClassificationList:
        def __init__(self, label):
            self.classification = [MockClassification(label)]

    class MockLandmark:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class MockHandLandmarks:
        def __init__(self, coords):
            self.landmark = [MockLandmark(c[0], c[1], c[2] if len(c) > 2 else 0.0) for c in coords]

    class MockResults:
        def __init__(self, raw_x, raw_y):
            # 21 landmarks all at (raw_x, raw_y)
            coords = [(raw_x, raw_y, 0.0)] * 21
            self.multi_hand_landmarks = [MockHandLandmarks(coords)]
            self.multi_handedness = [MockClassificationList("Right")]  # Mirrored to Left

    class MockMpHands:
        def __init__(self):
            self.curr_x = 0.5
            self.curr_y = 0.5

        def process(self, rgb):
            return MockResults(self.curr_x, self.curr_y)

        def close(self):
            pass

    mock_mp = MockMpHands()
    tracker.hands = mock_mp

    raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Observation 1: initial baseline
    mock_mp.curr_x, mock_mp.curr_y = 0.5, 0.5
    res1 = tracker.process(raw_frame, timestamp=1.0)
    assert len(res1) == 1
    assert res1[0].wrist_velocity == (0.0, 0.0)
    assert all(v == (0.0, 0.0) for v in res1[0].fingertip_velocities.values())

    # Observation 2: real movement -> velocity is positive
    mock_mp.curr_x, mock_mp.curr_y = 0.55, 0.55
    res2 = tracker.process(raw_frame, timestamp=1.05)
    assert len(res2) == 1
    vw_x, vw_y = res2[0].wrist_velocity
    assert abs(vw_x) > 10.0 or abs(vw_y) > 10.0

    # 1. Filter Switch -> Must produce zero baseline velocity on first observation
    tracker.set_filter_mode("deadband")
    # Coordinates jump significantly, but because state was reset, velocity must be 0
    mock_mp.curr_x, mock_mp.curr_y = 0.8, 0.8
    res_post_filter = tracker.process(raw_frame, timestamp=1.10)
    assert res_post_filter[0].wrist_velocity == (0.0, 0.0)
    assert all(v == (0.0, 0.0) for v in res_post_filter[0].fingertip_velocities.values())

    # Subsequent movement after filter switch computes velocity normally
    mock_mp.curr_x, mock_mp.curr_y = 0.85, 0.85
    res_subsequent = tracker.process(raw_frame, timestamp=1.15)
    assert abs(res_subsequent[0].wrist_velocity[0]) > 10.0

    # 2. Tracking Profile Switch -> Must produce zero baseline velocity on first observation
    tracker.set_tracking_profile("RESPONSIVE")
    mock_mp.curr_x, mock_mp.curr_y = 0.3, 0.3
    res_post_model = tracker.process(raw_frame, timestamp=1.20)
    assert res_post_model[0].wrist_velocity == (0.0, 0.0)
    assert all(v == (0.0, 0.0) for v in res_post_model[0].fingertip_velocities.values())

    # 3. No false Piano note generated by switch
    class MockAudio:
        def __init__(self):
            self.notes = []
            self.strums = []

        def play_note(self, note, freq, vel=1.0):
            self.notes.append((note, freq, vel))
            return True

        def play_guitar(self, string_idx, chord_name="C", velocity=1.0):
            self.strums.append((string_idx, chord_name, velocity))
            return True

    mock_audio = MockAudio()
    piano = Piano(bbox=(100, 500, 1100, 700), audio_engine=mock_audio)
    guitar = Guitar(zones=None, audio_engine=mock_audio)

    # Position finger over piano key with high downward velocity from previous frame
    # When tracking pipeline switches, reset_motion_state is invoked
    piano.reset_motion_state()
    # First post-switch frame has vy = 0.0 (below 60px/s trigger threshold)
    piano.update(res_post_filter, frame_shape=(720, 1280, 3), current_time=1.10)
    assert len(mock_audio.notes) == 0, "False piano note was generated on filter switch!"

    # 4. No false Guitar strum generated by switch
    guitar.reset_motion_state()
    guitar.update(res_post_filter, frame_shape=(720, 1280, 3), current_time=1.10)
    assert len(mock_audio.strums) == 0, "False guitar strum was generated on filter switch!"


def test_noop_config_set_preserves_snapshot_and_state():
    """
    Validates that no-op configuration sets do not clear latest snapshot or reset state,
    while actual pipeline transitions execute clean resets.
    Covers:
    1. model complexity 0 -> 0 preserves _latest_hands and returns False;
    2. current filter mode -> current filter mode preserves _latest_hands and returns False;
    3. actual complexity switch returns True and clears _latest_hands;
    4. actual filter switch returns True and clears _latest_hands;
    5. BALANCED -> LOW in GestureARApp preserves tracking snapshot;
    6. HIGH -> BALANCED performs intentional transition/reset.
    """
    import numpy as np
    from vision_tracker import AsyncHandTracker, HandData, ThreadedCamera
    from main import GestureARApp

    cam = ThreadedCamera(0)
    async_tracker = AsyncHandTracker(camera=cam, init_mediapipe=False)
    dummy_hand = HandData(
        handedness="Right",
        landmarks_norm=np.zeros((21, 3), dtype=np.float32),
        landmarks_px=np.zeros((21, 3), dtype=np.float32),
        timestamp=1.0,
    )

    # Populate snapshot with dummy hand
    with async_tracker._snapshot_lock:
        async_tracker._latest_hands = [dummy_hand]
    assert len(async_tracker._latest_hands) == 1

    # Ensure baseline state: tracking_profile = 'RESPONSIVE', filter_mode = 'one_euro'
    async_tracker.tracker.tracking_profile = "RESPONSIVE"
    async_tracker.tracker.filter_mode = "one_euro"

    # 1. Assigning tracking profile 'RESPONSIVE' while already 'RESPONSIVE' returns False and preserves _latest_hands
    changed_tp = async_tracker.set_tracking_profile("RESPONSIVE")
    assert changed_tp is False, "No-op tracking profile set returned True"
    assert len(async_tracker._latest_hands) == 1, "No-op tracking profile set cleared _latest_hands!"

    # 2. Assigning current filter mode ('one_euro') returns False and preserves _latest_hands
    changed_fm = async_tracker.set_filter_mode("one_euro")
    assert changed_fm is False, "No-op filter set returned True"
    assert len(async_tracker._latest_hands) == 1, "No-op filter set cleared _latest_hands!"

    # Case-insensitive normalization check ('ONE_EURO')
    changed_fm_norm = async_tracker.set_filter_mode("ONE_EURO")
    assert changed_fm_norm is False, "No-op normalized filter set returned True"
    assert len(async_tracker._latest_hands) == 1, "No-op normalized filter set cleared _latest_hands!"

    # 3. Actual tracking profile switch ('RESPONSIVE' -> 'STABLE') returns True and clears _latest_hands
    changed_tp_real = async_tracker.set_tracking_profile("STABLE")
    assert changed_tp_real is True, "Actual tracking profile switch returned False"
    assert len(async_tracker._latest_hands) == 0, "Actual tracking profile switch failed to clear _latest_hands!"

    # Re-populate snapshot
    with async_tracker._snapshot_lock:
        async_tracker._latest_hands = [dummy_hand]
    assert len(async_tracker._latest_hands) == 1

    # 4. Actual filter switch ('one_euro' -> 'deadband') returns True and clears _latest_hands
    changed_fm_real = async_tracker.set_filter_mode("deadband")
    assert changed_fm_real is True, "Actual filter switch returned False"
    assert len(async_tracker._latest_hands) == 0, "Actual filter switch failed to clear _latest_hands!"

    # 5. Visual quality profile transitions (HIGH <-> BALANCED <-> LOW) in GestureARApp
    # are decoupled from tracking and preserve the tracking snapshot.
    app = GestureARApp(start_threads=False, init_mediapipe=False, quality_profile="HIGH")
    assert app.async_tracker.tracking_profile == "RESPONSIVE"

    with app.async_tracker._snapshot_lock:
        app.async_tracker._latest_hands = [dummy_hand]
    assert len(app.async_tracker._latest_hands) == 1

    # Switch HIGH -> BALANCED (visual only, tracking snapshot preserved)
    app.set_quality_profile("BALANCED")
    assert app.quality_profile == "BALANCED"
    assert app.async_tracker.tracking_profile == "RESPONSIVE"
    assert len(app.async_tracker._latest_hands) == 1, "HIGH -> BALANCED caused tracking snapshot dropout!"

    # Switch BALANCED -> LOW (visual only, tracking snapshot preserved)
    app.set_quality_profile("LOW")
    assert app.quality_profile == "LOW"
    assert app.async_tracker.tracking_profile == "RESPONSIVE"
    assert len(app.async_tracker._latest_hands) == 1, "BALANCED -> LOW caused tracking snapshot dropout!"

    # Switch LOW -> HIGH (visual only, tracking snapshot preserved)
    app.set_quality_profile("HIGH")
    assert app.quality_profile == "HIGH"
    assert app.async_tracker.tracking_profile == "RESPONSIVE"
    assert len(app.async_tracker._latest_hands) == 1, "LOW -> HIGH caused tracking snapshot dropout!"

    # 6. Explicit M key or set_tracking_profile DOES perform real tracking transition
    app.handle_key(ord("m"))
    assert app.async_tracker.tracking_profile == "STABLE"
    assert len(app.async_tracker._latest_hands) == 0, "M key tracking profile switch failed to clear tracking snapshot!"


def test_threaded_camera_deferred_cleanup_after_shutdown_timeout():
    """
    Verifies deferred VideoCapture cleanup when stop(timeout) times out:
    1. worker is intentionally blocked;
    2. stop(short_timeout) returns while worker remains blocked;
    3. resource has NOT been released concurrently;
    4. test releases worker operation;
    5. worker exits;
    6. resource is eventually released exactly once;
    7. calling stop(), release(), or close() again remains idempotent.
    """
    class BlockableVideoCapture:
        def __init__(self):
            self.opened = True
            self.release_count = 0
            self.grab_entered = threading.Event()
            self.unblock_grab = threading.Event()

        def isOpened(self):
            return self.opened

        def grab(self):
            self.grab_entered.set()
            self.unblock_grab.wait(timeout=5.0)
            return False

        def retrieve(self):
            return False, None

        def release(self):
            self.release_count += 1
            self.opened = False

        def set(self, prop, val):
            return True

        def read(self):
            return True, np.zeros((100, 100, 3), dtype=np.uint8)

    cap = BlockableVideoCapture()
    cam = ThreadedCamera(cap=cap)
    cam.start()

    # 1. Wait until worker is inside grab() and intentionally blocked
    assert cap.grab_entered.wait(timeout=2.0) is True

    # 2. stop(short_timeout) returns boundedly while worker remains blocked
    cam.stop(timeout=0.01)
    assert cam._thread is not None and cam._thread.is_alive() is True

    # 3. Resource has NOT been released concurrently
    assert cap.release_count == 0

    # 4. Test releases worker operation
    cap.unblock_grab.set()

    # 5. Worker exits
    cam._thread.join(timeout=2.0)
    assert cam._thread.is_alive() is False

    # 6. Resource is eventually released exactly once
    assert cap.release_count == 1

    # 7. Calling stop(), release(), or close() again remains idempotent
    cam.stop()
    cam.release()
    cam.close()
    assert cap.release_count == 1


def test_async_hand_tracker_deferred_cleanup_after_shutdown_timeout():
    """
    Verifies deferred HandLandmarker cleanup when stop(timeout) times out:
    1. worker is intentionally blocked;
    2. stop(short_timeout) returns while worker remains blocked;
    3. resource has NOT been closed concurrently;
    4. test releases worker operation;
    5. worker exits;
    6. resource is eventually closed exactly once;
    7. calling stop() or close() again remains idempotent.
    """
    class BlockableLandmarker:
        def __init__(self):
            self.close_count = 0
            self.process_entered = threading.Event()
            self.unblock_process = threading.Event()

        def process(self, image):
            self.process_entered.set()
            self.unblock_process.wait(timeout=5.0)
            return None

        def close(self):
            self.close_count += 1

    class SequencedMockCamera:
        def __init__(self):
            self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
            self.frame_id = 0

        def read_sequenced(self):
            self.frame_id += 1
            return True, self.frame, self.frame_id, time.perf_counter()

    mock_lm = BlockableLandmarker()
    cam = SequencedMockCamera()
    async_tracker = AsyncHandTracker(camera=cam, init_mediapipe=False)
    async_tracker.tracker.landmarker = mock_lm
    async_tracker.tracker.hands = mock_lm
    async_tracker.start()

    # 1. Wait until worker is inside process() and intentionally blocked
    assert mock_lm.process_entered.wait(timeout=2.0) is True

    # 2. stop(short_timeout) returns boundedly while worker remains blocked
    async_tracker.stop(timeout=0.01)
    assert async_tracker._thread is not None and async_tracker._thread.is_alive() is True

    # 3. Resource has NOT been closed concurrently
    assert mock_lm.close_count == 0

    # 4. Test releases worker operation
    mock_lm.unblock_process.set()

    # 5. Worker exits
    async_tracker._thread.join(timeout=2.0)
    assert async_tracker._thread.is_alive() is False

    # 6. Resource is eventually closed exactly once
    assert mock_lm.close_count == 1

    # 7. Calling stop() or close() again remains idempotent
    async_tracker.stop()
    async_tracker.close()
    assert mock_lm.close_count == 1


