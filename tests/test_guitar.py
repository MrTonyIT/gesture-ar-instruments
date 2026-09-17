"""
tests/test_guitar.py
====================
Unit Tests for Augmented Reality Guitar Instrument.
Covers:
- All 9 chords selection and validation via set_chord
- Direct touch on top chord selector boxes
- Strum line-segment intersection detection
- Strum debounce lockout
- Muted string strumming handling
- Hand tracking loss and cleanup
"""

import time
import numpy as np
import pytest

from instruments import Guitar
from vision_tracker import HandData


class MockAudioEngine:
    def __init__(self):
        self.plucked_strings = []

    def play_guitar(self, string_idx: int, chord_name: str = "C", velocity: float = 1.0) -> bool:
        # Mimic AudioEngine: check if string is muted in chord
        from audio_engine import AudioEngine
        frets = AudioEngine.CHORD_FRETS.get(chord_name, AudioEngine.CHORD_FRETS["C"])
        if frets[string_idx] is None:
            return False
        self.plucked_strings.append((string_idx, chord_name, velocity))
        return True


@pytest.fixture
def mock_audio():
    return MockAudioEngine()


@pytest.fixture
def guitar(mock_audio):
    zones = {
        "fretboard": (150, 250, 450, 450),
        "strum_zone": (550, 450, 850, 650),
    }
    return Guitar(zones=zones, audio_engine=mock_audio)


def create_right_hand(tip_id: int, tip_xy: tuple[float, float]) -> HandData:
    pts = np.zeros((21, 3), dtype=np.float32)
    # Ground wrist stably near body anchor
    pts[0, :2] = [750.0, 480.0]
    pts[tip_id, :2] = [tip_xy[0], tip_xy[1]]
    return HandData(
        handedness="Right",
        landmarks_norm=np.zeros((21, 3)),
        landmarks_px=pts,
        timestamp=time.perf_counter(),
    )


def test_chord_selection_all_9_chords(guitar):
    """Guitar supports switching through all 9 authentic chords."""
    expected_chords = ["C", "G", "D", "A", "E", "Am", "Em", "Dm", "F"]
    for chord in expected_chords:
        assert guitar.set_chord(chord) is True
        assert guitar.active_chord == chord

    # Invalid chord rejected
    assert guitar.set_chord("INVALID_CHORD") is False
    assert guitar.active_chord == expected_chords[-1]


def test_strumming_crossing_detection(guitar, mock_audio):
    """Fingertip crossing a projected string segment triggers play_guitar."""
    shape = (720, 1280, 3)
    # Warmup with stable right hand so anchor settles
    dummy_hand = create_right_hand(8, (700.0, 500.0))
    guitar.update([dummy_hand], frame_shape=shape, current_time=0.0)

    target_string = guitar.strings[2]  # 4th string (D3)
    s_x1, s_y1 = target_string.screen_start
    s_x2, s_y2 = target_string.screen_end
    mid_x = (s_x1 + s_x2) / 2.0
    mid_y = (s_y1 + s_y2) / 2.0

    # Frame 1: Finger above string
    hand1 = create_right_hand(8, (mid_x, mid_y - 15.0))
    guitar.update([hand1], frame_shape=shape, current_time=0.02)
    assert len(mock_audio.plucked_strings) == 0

    # Frame 2: Finger crosses below string
    hand2 = create_right_hand(8, (mid_x, mid_y + 15.0))
    guitar.update([hand2], frame_shape=shape, current_time=0.04)

    assert len(mock_audio.plucked_strings) == 1
    str_idx, chord, vel = mock_audio.plucked_strings[0]
    assert str_idx == 2
    assert chord == guitar.active_chord
    assert 0.25 <= vel <= 1.0


def test_strum_debounce_lockout(guitar, mock_audio):
    """Crossing within 60ms debounce window is locked out."""
    shape = (720, 1280, 3)
    dummy_hand = create_right_hand(8, (700.0, 500.0))
    guitar.update([dummy_hand], frame_shape=shape, current_time=0.0)

    target_string = guitar.strings[3]
    mid_x = (target_string.screen_start[0] + target_string.screen_end[0]) / 2.0
    mid_y = (target_string.screen_start[1] + target_string.screen_end[1]) / 2.0

    # Initial position setup
    h_above = create_right_hand(8, (mid_x, mid_y - 7.0))
    h_below = create_right_hand(8, (mid_x, mid_y + 7.0))

    guitar.update([h_above], frame_shape=shape, current_time=0.02)
    mock_audio.plucked_strings.clear()

    # Initial pluck of single string at t=0.04
    guitar.update([h_below], frame_shape=shape, current_time=0.04)
    assert len(mock_audio.plucked_strings) == 1
    assert mock_audio.plucked_strings[0][0] == 3

    # Immediate reverse crossing within 60ms (t=0.06 < 0.04 + 0.06) is locked out
    guitar.update([h_above], frame_shape=shape, current_time=0.06)
    assert len(mock_audio.plucked_strings) == 1, "Pluck within debounce lockout should not trigger"

    # Reverse crossing after 60ms lockout (t=0.12) triggers
    guitar.update([h_below], frame_shape=shape, current_time=0.12)
    guitar.update([h_above], frame_shape=shape, current_time=0.14)
    assert len(mock_audio.plucked_strings) == 2


def test_muted_string_strumming(guitar, mock_audio):
    """
    Strumming a muted string (e.g. 6th string on C chord)
    does not trigger sound in mock_audio and sets dampened vibration.
    """
    shape = (720, 1280, 3)
    guitar.set_chord("C")
    dummy_hand = create_right_hand(4, (700.0, 500.0))
    guitar.update([dummy_hand], frame_shape=shape, current_time=0.0)

    # 6th string (idx 0) is muted in C chord:
    string_0 = guitar.strings[0]
    mid_x = (string_0.screen_start[0] + string_0.screen_end[0]) / 2.0
    mid_y = (string_0.screen_start[1] + string_0.screen_end[1]) / 2.0

    h_above = create_right_hand(4, (mid_x, mid_y - 7.0))
    h_below = create_right_hand(4, (mid_x, mid_y + 7.0))

    guitar.update([h_above], frame_shape=shape, current_time=0.02)
    mock_audio.plucked_strings.clear()

    guitar.update([h_below], frame_shape=shape, current_time=0.04)

    assert len(mock_audio.plucked_strings) == 0, "Muted string should not produce audio voice"
    # String has subtle dampened vibration amplitude
    assert 0.0 < string_0.vibration_amplitude < 5.0


def test_guitar_direct_touch_chord_selection(guitar):
    """Touching each of the 9 floating chord boxes selects the corresponding chord."""
    shape = (720, 1280, 3)
    # Ensure chord_boxes are projected
    guitar.update([], frame_shape=shape, current_time=0.0)
    assert len(guitar.chord_boxes) == 9

    for chord, (bx1, by1, bx2, by2) in guitar.chord_boxes.items():
        center_x = (bx1 + bx2) / 2.0
        center_y = (by1 + by2) / 2.0

        # Create left hand with index finger touching chord box center
        pts_px = np.zeros((21, 3), dtype=np.float32)
        pts_px[0, :2] = (center_x, center_y + 200.0)  # Wrist
        pts_px[8, :2] = (center_x, center_y)  # Index tip touching box
        pts_norm = pts_px.copy()
        pts_norm[:, 0] /= 1280.0
        pts_norm[:, 1] /= 720.0

        lh = HandData(
            handedness="Left",
            landmarks_norm=pts_norm,
            landmarks_px=pts_px,
            timestamp=0.0,
        )

        guitar.update([lh], frame_shape=shape, current_time=0.1)
        assert guitar.active_chord == chord, f"Direct touch failed for chord {chord}"


def test_guitar_tracking_loss_cleanup(guitar):
    """When tracking is lost, landmarks and previous finger positions are purged cleanly."""
    shape = (720, 1280, 3)
    rh = create_right_hand(8, (600.0, 400.0))
    guitar.update([rh], frame_shape=shape, current_time=0.0)
    assert guitar.rh_landmarks is not None
    assert 8 in guitar._prev_strum_finger_positions

    # Tracking lost
    guitar.update([], frame_shape=shape, current_time=0.05)
    assert guitar.rh_landmarks is None
    assert len(guitar._prev_strum_finger_positions) == 0


def test_guitar_fretted_note_and_frequency():
    """Validates single source of truth get_fretted_note calculation across chords."""
    from instruments import Guitar

    # C major: (None, 3, 2, 0, 1, 0)
    # String 0 (E2): Muted -> None, 'X', True
    f0, note0, muted0 = Guitar.get_fretted_note("C", 0)
    assert f0 is None
    assert note0 == "X"
    assert muted0 is True

    # String 1 (A2) + fret 3 = C3
    f1, note1, muted1 = Guitar.get_fretted_note("C", 1)
    assert muted1 is False
    assert note1 == "C3"
    assert f1 == pytest.approx(110.0 * (2.0 ** (3.0 / 12.0)))

    # String 2 (D3) + fret 2 = E3
    f2, note2, muted2 = Guitar.get_fretted_note("C", 2)
    assert muted2 is False
    assert note2 == "E3"

    # String 4 (B3) + fret 1 = C4
    f4, note4, muted4 = Guitar.get_fretted_note("C", 4)
    assert muted4 is False
    assert note4 == "C4"

    # String 5 (E4) + fret 0 = E4
    f5, note5, muted5 = Guitar.get_fretted_note("C", 5)
    assert muted5 is False
    assert note5 == "E4"

    # G major: (3, 2, 0, 0, 0, 3)
    # String 0 (E2) + fret 3 = G2
    fg0, noteg0, mutedg0 = Guitar.get_fretted_note("G", 0)
    assert mutedg0 is False
    assert noteg0 == "G2"

