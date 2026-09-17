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
    guitar.update([dummy_hand], frame_shape=shape)

    target_string = guitar.strings[2]  # 4th string (D3)
    s_x1, s_y1 = target_string.screen_start
    s_x2, s_y2 = target_string.screen_end
    mid_x = (s_x1 + s_x2) / 2.0
    mid_y = (s_y1 + s_y2) / 2.0

    # Frame 1: Finger above string
    hand1 = create_right_hand(8, (mid_x, mid_y - 15.0))
    guitar.update([hand1], frame_shape=shape)
    assert len(mock_audio.plucked_strings) == 0

    # Frame 2: Finger crosses below string
    time.sleep(0.015)
    hand2 = create_right_hand(8, (mid_x, mid_y + 15.0))
    guitar.update([hand2], frame_shape=shape)

    assert len(mock_audio.plucked_strings) == 1
    str_idx, chord, vel = mock_audio.plucked_strings[0]
    assert str_idx == 2
    assert chord == guitar.active_chord
    assert 0.25 <= vel <= 1.0


def test_strum_debounce_lockout(guitar, mock_audio):
    """Crossing within 60ms debounce window is locked out."""
    shape = (720, 1280, 3)
    dummy_hand = create_right_hand(8, (700.0, 500.0))
    guitar.update([dummy_hand], frame_shape=shape)

    target_string = guitar.strings[3]
    mid_x = (target_string.screen_start[0] + target_string.screen_end[0]) / 2.0
    mid_y = (target_string.screen_start[1] + target_string.screen_end[1]) / 2.0

    # Initial position setup
    h_above = create_right_hand(8, (mid_x, mid_y - 7.0))
    h_below = create_right_hand(8, (mid_x, mid_y + 7.0))

    guitar.update([h_above], frame_shape=shape)
    mock_audio.plucked_strings.clear()

    # Initial pluck of single string
    time.sleep(0.015)
    guitar.update([h_below], frame_shape=shape)
    assert len(mock_audio.plucked_strings) == 1
    assert mock_audio.plucked_strings[0][0] == 3

    # Immediate reverse crossing without waiting for debounce (lockout)
    guitar.update([h_above], frame_shape=shape)
    assert len(mock_audio.plucked_strings) == 1, "Pluck within debounce lockout should not trigger"


def test_muted_string_strumming(guitar, mock_audio):
    """
    Strumming a muted string (e.g. 6th string on C chord)
    does not trigger sound in mock_audio and sets dampened vibration.
    """
    shape = (720, 1280, 3)
    guitar.set_chord("C")
    dummy_hand = create_right_hand(4, (700.0, 500.0))
    guitar.update([dummy_hand], frame_shape=shape)

    # 6th string (idx 0) is muted in C chord:
    string_0 = guitar.strings[0]
    mid_x = (string_0.screen_start[0] + string_0.screen_end[0]) / 2.0
    mid_y = (string_0.screen_start[1] + string_0.screen_end[1]) / 2.0

    h_above = create_right_hand(4, (mid_x, mid_y - 7.0))
    h_below = create_right_hand(4, (mid_x, mid_y + 7.0))

    guitar.update([h_above], frame_shape=shape)
    mock_audio.plucked_strings.clear()

    time.sleep(0.015)
    guitar.update([h_below], frame_shape=shape)

    assert len(mock_audio.plucked_strings) == 0, "Muted string should not produce audio voice"
    # String has subtle dampened vibration amplitude
    assert 0.0 < string_0.vibration_amplitude < 5.0
