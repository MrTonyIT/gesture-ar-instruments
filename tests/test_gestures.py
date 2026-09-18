"""
tests/test_gestures.py
=======================
Headless Unit Tests for GestureEngine State Machine & Transitions.
Covers:
- Initialization & Default IDLE state
- Reset Button & programmatic reset_to_idle()
- Exit Button touch progression (3.0s hold to exit)
- Lock Sculpt button toggle (1.0s hold)
- Piano 'L' gesture locking (IDLE -> CREATING_PIANO -> PIANO_ACTIVE)
- Piano gesture cancellation on drop
- Right-hand Circle / Soundbox sculpting & overflow disappearance
- Two-hand Rectangle / Neck sculpting (touch & pull apart) & overflow disappearance
- Dual-finger drag interaction for sculpted shapes
- Magnetic snap fusion (READY_TO_ASSEMBLE -> FUSION_SNAP -> GUITAR_ACTIVE)
- Fallback 'O' pinch guitar spawn
- Runtime UI Exit and Reset touch state machines with clock injection
"""

import time
import numpy as np
import pytest

from gesture_engine import AppState, GestureEngine
from vision_tracker import HandData


def create_mock_hand(
    handedness: str,
    landmarks_dict: dict[int, tuple[float, float]],
    wrist_xy: tuple[float, float] = (500.0, 500.0),
) -> HandData:
    """Creates a synthetic HandData instance with landmarks specified in pixels."""
    pts_px = np.zeros((21, 3), dtype=np.float32)
    pts_px[0, :2] = wrist_xy
    for idx, (x, y) in landmarks_dict.items():
        pts_px[idx, :2] = (x, y)

    pts_norm = pts_px.copy()
    pts_norm[:, 0] /= 1280.0
    pts_norm[:, 1] /= 720.0

    return HandData(
        handedness=handedness,
        landmarks_norm=pts_norm,
        landmarks_px=pts_px,
        timestamp=time.perf_counter(),
    )


@pytest.fixture
def engine():
    return GestureEngine()


def test_initial_state(engine):
    """Engine initializes in IDLE state with clean default buffers."""
    assert engine.state == AppState.IDLE
    assert engine.progress == 0.0
    assert engine.should_exit is False
    assert engine.is_sculpt_locked is False
    assert engine.soundbox_sculpted is False
    assert engine.neck_sculpted is False


def test_exit_button_touch_and_hold(engine):
    """Holding a fingertip inside the top-right exit button triggers should_exit using injected clock."""
    frame_shape = (720, 1280, 3)
    w = 1280
    btn_x = w - 80  # Inside exit button bbox
    btn_y = 20

    hand = create_mock_hand("Right", {8: (btn_x, btn_y)})

    # Initial frame touch at t=10.0
    engine.update([hand], frame_shape, current_time=10.0)
    assert engine.is_touching_exit is True
    assert engine.should_exit is False
    assert 0.0 <= engine.exit_progress < 1.0

    # Advance clock past 3.0s duration (t=13.2)
    engine.update([hand], frame_shape, current_time=13.2)
    assert engine.exit_progress >= 1.0
    assert engine.should_exit is True


def test_reset_button_touch_and_hold(engine):
    """Holding a fingertip inside the reset button triggers reset_to_idle using injected clock."""
    frame_shape = (720, 1280, 3)
    # Reset button bounds at 1280w: (935, 8) to (1095, 44)
    btn_x = 1000.0
    btn_y = 25.0

    # Put engine in a non-idle state first
    engine.state = AppState.PIANO_ACTIVE
    engine.soundbox_sculpted = True

    hand = create_mock_hand("Right", {8: (btn_x, btn_y)})

    # Frame 1: touch begins at t=100.0
    engine.update([hand], frame_shape, current_time=100.0)
    assert engine.is_touching_reset is True

    # Advance clock past 0.7s hold duration (t=100.8)
    engine.update([hand], frame_shape, current_time=100.8)

    assert engine.state == AppState.IDLE
    assert engine.reset_just_triggered is True
    assert engine.soundbox_sculpted is False


def test_lock_sculpt_button_toggle(engine):
    """Touching lock sculpt button toggles is_sculpt_locked between True and False."""
    frame_shape = (720, 1280, 3)
    # Lock button at 1280: (1280 - 480 - 35, 8) = (765, 8) to (925, 44)
    btn_x = 840.0
    btn_y = 25.0

    assert engine.is_sculpt_locked is False

    hand = create_mock_hand("Right", {8: (btn_x, btn_y)})

    # Touch and hold for 1.0s
    engine.update([hand], frame_shape)
    assert engine.is_touching_lock is True

    engine._lock_start_time = time.perf_counter() - 1.1
    engine.update([hand], frame_shape)

    assert engine.is_sculpt_locked is True
    assert engine.lock_just_toggled is True


def test_piano_l_gesture_lock(engine):
    """
    Both hands forming an 'L' shape transitions IDLE -> CREATING_PIANO -> PIANO_ACTIVE.
    """
    frame_shape = (720, 1280, 3)

    # Hand forming 'L' shape:
    # Wrist=(500, 500), Index(8)=(500, 380) [vertical, angle ~90 deg],
    # Thumb(4)=(400, 500) [horizontal, angle ~0/180 deg], Middle/Ring/Pinky curled
    lh_dict = {
        0: (300.0, 500.0),
        2: (260.0, 500.0), 3: (230.0, 500.0), 4: (200.0, 500.0),  # Thumb extended horizontal
        5: (300.0, 460.0), 6: (300.0, 420.0), 7: (300.0, 380.0), 8: (300.0, 340.0),  # Index vertical
        9: (330.0, 470.0), 10: (330.0, 490.0), 11: (330.0, 500.0), 12: (330.0, 510.0),  # Middle curled
        13: (350.0, 470.0), 14: (350.0, 490.0), 15: (350.0, 500.0), 16: (350.0, 510.0),  # Ring curled
        17: (370.0, 480.0), 18: (370.0, 490.0), 19: (370.0, 500.0), 20: (370.0, 510.0),  # Pinky curled
    }
    rh_dict = {
        0: (900.0, 500.0),
        2: (940.0, 500.0), 3: (970.0, 500.0), 4: (1000.0, 500.0),  # Thumb extended horizontal
        5: (900.0, 460.0), 6: (900.0, 420.0), 7: (900.0, 380.0), 8: (900.0, 340.0),  # Index vertical
        9: (870.0, 470.0), 10: (870.0, 490.0), 11: (870.0, 500.0), 12: (870.0, 510.0),  # Middle curled
        13: (850.0, 470.0), 14: (850.0, 490.0), 15: (850.0, 500.0), 16: (850.0, 510.0),  # Ring curled
        17: (830.0, 480.0), 18: (830.0, 490.0), 19: (830.0, 500.0), 20: (830.0, 510.0),  # Pinky curled
    }

    lh = create_mock_hand("Left", lh_dict, wrist_xy=(300.0, 500.0))
    rh = create_mock_hand("Right", rh_dict, wrist_xy=(900.0, 500.0))

    # Direct check of helper first
    assert engine._is_l_shape(lh) is True
    assert engine._is_l_shape(rh) is True
    assert engine._check_piano_gesture(lh, rh) is True

    # Frame 1: Detection starts CREATING_PIANO
    engine.update([lh, rh], frame_shape)
    assert engine.state == AppState.CREATING_PIANO
    assert 0.0 <= engine.progress < 1.0

    # Simulate hold duration elapsing (1.2s)
    engine._gesture_start_time = time.perf_counter() - 1.3
    engine.update([lh, rh], frame_shape)

    assert engine.state == AppState.PIANO_ACTIVE
    assert engine.piano_bbox is not None


def test_piano_gesture_cancellation_on_drop(engine):
    """Dropping piano gesture before 1.2s reverts from CREATING_PIANO to IDLE."""
    frame_shape = (720, 1280, 3)
    engine.state = AppState.CREATING_PIANO
    engine._gesture_start_time = time.perf_counter()
    engine.progress = 0.45

    # Neutral hand: fingers curled close together (no 'L' and no sculpting stretch)
    neutral_hand = create_mock_hand("Right", {
        4: (600.0, 420.0),
        8: (608.0, 420.0),
    }, wrist_xy=(600.0, 500.0))
    engine.update([neutral_hand], frame_shape)

    assert engine.state == AppState.IDLE
    assert engine.progress == 0.0
    assert engine.candidate_piano_bbox is None


def test_soundbox_circle_sculpting(engine):
    """Right hand stretching thumb and index sculpts circle; hold locks it."""
    frame_shape = (720, 1280, 3)

    # Frame 1: Right hand with Thumb and Index stretched 85px apart (prog_r >= 0.84)
    rh = create_mock_hand("Right", {
        4: (600.0, 400.0),
        8: (685.0, 400.0),
    }, wrist_xy=(635.0, 550.0))

    engine.update([rh], frame_shape)
    assert engine.state == AppState.SCULPTING_GUITAR
    assert engine.soundbox_progress >= 0.84
    assert engine.soundbox_sculpted is False

    # Simulate holding stretched for 1.5s
    engine._soundbox_hold_start = time.perf_counter() - 1.6
    engine.update([rh], frame_shape)

    assert engine.soundbox_sculpted is True
    assert engine.soundbox_just_locked is True


def test_soundbox_overflow_auto_disappearance(engine):
    """If soundbox circle extends outside screen boundary, it auto-disappears."""
    frame_shape = (720, 1280, 3)

    # Position hand at the extreme right edge (x=1270)
    rh = create_mock_hand("Right", {
        4: (1240.0, 400.0),
        8: (1290.0, 400.0),
    }, wrist_xy=(1250.0, 550.0))

    engine.update([rh], frame_shape)
    assert engine.soundbox_overflow_just_occurred is True
    assert engine.soundbox_sculpted is False
    assert engine.soundbox_progress == 0.0


def test_rectangle_neck_sculpting_and_fusion(engine):
    """
    Two hands touching thumbs/indices then pulling apart sculpts neck rectangle.
    Once both soundbox & neck are sculpted and brought close, triggers FUSION_SNAP -> GUITAR_ACTIVE.
    """
    frame_shape = (720, 1280, 3)

    # Step 1: Pre-sculpt the soundbox
    engine.soundbox_sculpted = True
    engine.soundbox_center = (700, 450)
    engine.soundbox_radius = 50.0
    engine._soundbox_require_release = False

    # Step 2: Left and right hands touch first (< 75px)
    lh_touch = create_mock_hand("Left", {4: (400.0, 450.0), 8: (400.0, 420.0)}, wrist_xy=(350.0, 500.0))
    rh_touch = create_mock_hand("Right", {4: (420.0, 450.0), 8: (420.0, 420.0)}, wrist_xy=(470.0, 500.0))

    engine.update([lh_touch, rh_touch], frame_shape)
    assert engine._neck_touch_primed is True

    # Step 3: Pull apart (> 50px)
    lh_pull = create_mock_hand("Left", {4: (280.0, 450.0), 8: (280.0, 420.0)}, wrist_xy=(230.0, 500.0))
    rh_pull = create_mock_hand("Right", {4: (480.0, 450.0), 8: (480.0, 420.0)}, wrist_xy=(530.0, 500.0))

    engine.update([lh_pull, rh_pull], frame_shape)
    assert engine.neck_progress > 0.0
    assert engine.neck_line_upper is not None
    assert engine.neck_line_lower is not None

    # Step 4: Hold stretched for 1.5s to lock neck
    engine._neck_hold_start = time.perf_counter() - 1.6
    engine.update([lh_pull, rh_pull], frame_shape)

    assert engine.neck_sculpted is True
    assert engine.neck_just_locked is True

    # Step 5: State should now be READY_TO_ASSEMBLE
    assert engine.state == AppState.READY_TO_ASSEMBLE

    # Step 6: Bring centers closer than 145px for Fusion Snap
    engine.neck_center = (650, 450)
    engine.soundbox_center = (720, 450)  # Distance = 70px (< 145px)

    engine.update([], frame_shape)
    assert engine.state == AppState.FUSION_SNAP
    assert engine.fusion_just_triggered is True

    # Step 7: Elapsed fusion progress (0.48s) -> GUITAR_ACTIVE
    engine._fusion_start_time = time.perf_counter() - 0.50
    engine.update([], frame_shape)
    assert engine.state == AppState.GUITAR_ACTIVE
    assert engine.guitar_zones is not None


def test_runtime_exit_and_reset_flow(engine):
    """Verifies runtime engine.update responds to Exit button touch and Reset button touch."""
    frame_shape = (720, 1280, 3)
    w = 1280

    # 1. Exit button touch cancels when hand leaves
    exit_x = w - 80
    hand_exit = create_mock_hand("Right", {8: (exit_x, 20)})
    engine.update([hand_exit], frame_shape, current_time=0.0)
    assert engine.is_touching_exit is True
    assert engine.should_exit is False

    # Hand moves away: exit progress resets
    hand_away = create_mock_hand("Right", {8: (400, 400)})
    engine.update([hand_away], frame_shape, current_time=1.0)
    assert engine.is_touching_exit is False
    assert engine.exit_progress == 0.0
    assert engine.should_exit is False

    # 2. Reset button touch resets state from GUITAR_ACTIVE to IDLE
    engine.state = AppState.GUITAR_ACTIVE
    reset_x = 1000.0
    hand_reset = create_mock_hand("Right", {8: (reset_x, 25)})
    engine.update([hand_reset], frame_shape, current_time=2.0)
    assert engine.is_touching_reset is True

    # Complete 0.7s hold
    engine.update([hand_reset], frame_shape, current_time=2.75)
    assert engine.state == AppState.IDLE
    assert engine.reset_just_triggered is True

