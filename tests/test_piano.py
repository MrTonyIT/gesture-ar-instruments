"""
tests/test_piano.py
===================
Unit Tests for Desk-Docked Virtual Piano Instrument.
Covers:
- Downward velocity gating (suppression of upward/slow touch)
- Black-key precedence over underlying white keys
- Multi-finger polyphonic simultaneous note triggering
- Hover false-positive suppression (no machine-gun retrigger while resting)
- Tracking loss cleanup and recovery
"""

import time
import numpy as np
import pytest

from instruments import Piano
from vision_tracker import HandData


@pytest.fixture
def mock_audio():
    class MockAudioEngine:
        def __init__(self):
            self.triggered_notes = []

        def play_piano(self, freq: float, velocity: float = 1.0, pan: float = 0.0):
            self.triggered_notes.append((freq, velocity, pan))

    return MockAudioEngine()


@pytest.fixture
def piano(mock_audio):
    # Setup piano on 1280x720 frame: X in [64, 1216], Y in [518, 691]
    bbox = (64, 518, 1216, 691)
    p = Piano(bbox=bbox, audio_engine=mock_audio)
    return p


def create_hand(
    tip_positions: dict[int, tuple[float, float]],
    tip_velocities: dict[int, tuple[float, float]],
    handedness: str = "Right",
) -> HandData:
    """Helper creating HandData with specified fingertip positions and velocities."""
    landmarks_norm = np.zeros((21, 3), dtype=np.float32)
    landmarks_px = np.zeros((21, 3), dtype=np.float32)

    for tip_id, (x, y) in tip_positions.items():
        landmarks_px[tip_id, 0] = x
        landmarks_px[tip_id, 1] = y

    return HandData(
        handedness=handedness,
        landmarks_norm=landmarks_norm,
        landmarks_px=landmarks_px,
        fingertip_velocities=tip_velocities,
        wrist_velocity=(0.0, 0.0),
        timestamp=time.perf_counter(),
    )


def test_velocity_gating_slow_vs_strike(piano, mock_audio):
    """
    Fingertip moving slowly downwards (<120 px/s) must be rejected.
    Fingertip striking downwards (>120 px/s) must trigger note.
    """
    # First white key C3:
    wkey = piano.white_keys[0]
    wx1, wy1, wx2, wy2 = wkey.rect
    strike_x = (wx1 + wx2) / 2.0
    strike_y = (wy1 + wy2) / 2.0

    # 1. Slow touch: vy = 30 px/s (below MIN_PIXEL_VELOCITY 60)
    slow_hand = create_hand(
        tip_positions={8: (strike_x, strike_y)},
        tip_velocities={8: (0.0, 30.0)},
    )
    piano.update([slow_hand], frame_shape=(720, 1280, 3), current_time=0.0)
    assert len(mock_audio.triggered_notes) == 0

    # 2. Fast strike: vy = 350 px/s
    strike_hand = create_hand(
        tip_positions={8: (strike_x, strike_y)},
        tip_velocities={8: (0.0, 350.0)},
    )
    piano.update([strike_hand], frame_shape=(720, 1280, 3), current_time=0.02)
    assert len(mock_audio.triggered_notes) == 1
    assert mock_audio.triggered_notes[0][0] == pytest.approx(wkey.freq)


def test_black_key_precedence(piano, mock_audio):
    """
    When fingertip is inside a black key's bounding box,
    the black key must trigger and the white key behind it must NOT trigger.
    """
    bkey = piano.black_keys[0]  # C#3
    bx1, by1, bx2, by2 = bkey.rect
    strike_x = (bx1 + bx2) / 2.0
    strike_y = (by1 + by2) / 2.0

    strike_hand = create_hand(
        tip_positions={8: (strike_x, strike_y)},
        tip_velocities={8: (0.0, 300.0)},
    )
    piano.update([strike_hand], frame_shape=(720, 1280, 3), current_time=0.0)

    assert len(mock_audio.triggered_notes) == 1
    assert mock_audio.triggered_notes[0][0] == pytest.approx(bkey.freq)
    assert bkey.is_active is True


def test_multi_finger_polyphony(piano, mock_audio):
    """Multiple fingers striking simultaneously must trigger distinct notes."""
    # Index (8) on C3, Middle (12) on E3
    wkey_c = piano.white_keys[0]
    wkey_e = piano.white_keys[2]

    hand = create_hand(
        tip_positions={
            8: ((wkey_c.rect[0] + wkey_c.rect[2]) / 2.0, (wkey_c.rect[1] + wkey_c.rect[3]) / 2.0),
            12: ((wkey_e.rect[0] + wkey_e.rect[2]) / 2.0, (wkey_e.rect[1] + wkey_e.rect[3]) / 2.0),
        },
        tip_velocities={
            8: (0.0, 300.0),
            12: (0.0, 320.0),
        },
    )

    piano.update([hand], frame_shape=(720, 1280, 3), current_time=0.0)
    assert len(mock_audio.triggered_notes) == 2
    played_freqs = sorted([n[0] for n in mock_audio.triggered_notes])
    expected_freqs = sorted([wkey_c.freq, wkey_e.freq])
    assert played_freqs == pytest.approx(expected_freqs)


def test_hover_suppression(piano, mock_audio):
    """
    Finger resting inside a key must NOT trigger repeated strikes
    on every frame due to hover or sensor micro-jitter.
    """
    wkey = piano.white_keys[0]
    pos = ((wkey.rect[0] + wkey.rect[2]) / 2.0, (wkey.rect[1] + wkey.rect[3]) / 2.0)

    # First strike at t=0.0
    hand_strike = create_hand(tip_positions={8: pos}, tip_velocities={8: (0.0, 250.0)})
    piano.update([hand_strike], frame_shape=(720, 1280, 3), current_time=0.0)
    assert len(mock_audio.triggered_notes) == 1

    # Finger remains resting inside key for subsequent frames
    # Even if micro-jitter produces positive velocity and debounce expires, hover suppression blocks it
    t_resting = 0.0 + piano.DEBOUNCE_SECONDS + 0.05
    hand_resting = create_hand(tip_positions={8: pos}, tip_velocities={8: (0.0, 150.0)})
    piano.update([hand_resting], frame_shape=(720, 1280, 3), current_time=t_resting)

    assert len(mock_audio.triggered_notes) == 1, "Hover suppression failed: key retriggered while resting"

    # Finger lifts up (vy < -40) at t_lift, then strikes down again after debounce
    t_lift = t_resting + 0.05
    hand_lift = create_hand(tip_positions={8: pos}, tip_velocities={8: (0.0, -80.0)})
    piano.update([hand_lift], frame_shape=(720, 1280, 3), current_time=t_lift)

    t_restrike = t_lift + piano.DEBOUNCE_SECONDS + 0.05
    hand_restrike = create_hand(tip_positions={8: pos}, tip_velocities={8: (0.0, 250.0)})
    piano.update([hand_restrike], frame_shape=(720, 1280, 3), current_time=t_restrike)

    assert len(mock_audio.triggered_notes) == 2, "Second strike after deliberate release should trigger"


def test_tracking_loss_cleanup(piano, mock_audio):
    """When tracking is lost (hands=[]), held key records must clear."""
    wkey = piano.white_keys[0]
    pos = ((wkey.rect[0] + wkey.rect[2]) / 2.0, (wkey.rect[1] + wkey.rect[3]) / 2.0)

    hand = create_hand(tip_positions={8: pos}, tip_velocities={8: (0.0, 250.0)})
    piano.update([hand], frame_shape=(720, 1280, 3))
    assert ("Right", 8) in piano._finger_held_keys

    # Tracking lost
    piano.update([], frame_shape=(720, 1280, 3))
    assert len(piano._finger_held_keys) == 0
