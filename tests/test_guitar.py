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



def test_guitar_keyboard_shortcut_f_and_filter_cycling_k():
    """
    Verifies that 'f', 'F', and '9' map to the F chord in application runtime mapping
    and do not collide with filter cycling ('k'/'K').
    """
    from main import FILTER_CYCLE_KEYS, GUITAR_KEY_CHORD_MAP

    assert GUITAR_KEY_CHORD_MAP[ord("f")] == "F"
    assert GUITAR_KEY_CHORD_MAP[ord("F")] == "F"
    assert GUITAR_KEY_CHORD_MAP[ord("9")] == "F"
    assert ord("k") not in GUITAR_KEY_CHORD_MAP
    assert ord("K") not in GUITAR_KEY_CHORD_MAP
    assert ord("k") in FILTER_CYCLE_KEYS
    assert ord("K") in FILTER_CYCLE_KEYS


def test_chord_input_source_arbitration_integration(guitar):
    """
    Deterministic integration tests for event-based chord input-source arbitration.
    Proves:
    1. one-finger gesture selects C;
    2. keyboard F is then selected;
    3. next frame with the SAME one-finger pose leaves chord at F;
    4. changing gesture from one finger to two fingers intentionally switches to G;
    5. keyboard 9 selects and persists as F;
    6. direct-touch selection persists after leaving the box if no new gesture transition occurs;
    7. all 9 keyboard chords still work.
    """
    from typing import Optional
    from main import GUITAR_KEY_CHORD_MAP

    shape = (720, 1280, 3)

    def make_lh(finger_count: int, touch_xy: Optional[tuple[float, float]] = None) -> HandData:
        wrist = np.array([300.0, 400.0], dtype=np.float32)
        pts_px = np.zeros((21, 3), dtype=np.float32)
        pts_px[:, :2] = wrist

        # Hand scale reference: wrist (0) to Middle MCP (9)
        pts_px[9, :2] = wrist + np.array([0.0, -80.0], dtype=np.float32)

        # MCP coordinates for 4 long fingers
        pts_px[6, :2] = wrist + np.array([-30.0, -60.0], dtype=np.float32)
        pts_px[10, :2] = wrist + np.array([0.0, -60.0], dtype=np.float32)
        pts_px[14, :2] = wrist + np.array([30.0, -60.0], dtype=np.float32)
        pts_px[18, :2] = wrist + np.array([60.0, -60.0], dtype=np.float32)

        # Extended vs folded tips: Index (8), Middle (12), Ring (16), Pinky (20)
        tips_mcps = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for i, (tip_id, mcp_id) in enumerate(tips_mcps):
            if i < finger_count:
                pts_px[tip_id, :2] = wrist + np.array([-30.0 + i * 30.0, -120.0], dtype=np.float32)
            else:
                pts_px[tip_id, :2] = wrist + np.array([-30.0 + i * 30.0, -20.0], dtype=np.float32)

        # Thumb (4) folded near base
        pts_px[4, :2] = wrist + np.array([-40.0, -20.0], dtype=np.float32)

        # Optional direct touch position override
        if touch_xy is not None:
            pts_px[8, :2] = [touch_xy[0], touch_xy[1]]

        pts_norm = pts_px.copy()
        pts_norm[:, 0] /= 1280.0
        pts_norm[:, 1] /= 720.0

        return HandData(
            handedness="Left",
            landmarks_norm=pts_norm,
            landmarks_px=pts_px,
            timestamp=0.0,
        )

    # 1. One-finger gesture selects C
    lh_1 = make_lh(1)
    guitar.update([lh_1], frame_shape=shape, current_time=0.10)
    assert guitar.active_chord == "C", f"Expected C on initial 1-finger acquisition, got {guitar.active_chord}"

    # 2. Keyboard F is then selected
    assert guitar.set_chord("F") is True
    assert guitar.active_chord == "F"

    # 3. Next frame with the SAME one-finger pose leaves chord at F
    guitar.update([lh_1], frame_shape=shape, current_time=0.12)
    assert guitar.active_chord == "F", f"Unchanged 1-finger pose overwrote keyboard chord to {guitar.active_chord}"

    # 4. Changing gesture from one finger to two fingers intentionally switches to G
    lh_2 = make_lh(2)
    guitar.update([lh_2], frame_shape=shape, current_time=0.14)
    assert guitar.active_chord == "G", f"Expected transition to G on 2 fingers, got {guitar.active_chord}"

    # 5. Keyboard 9 selects and persists as F
    chord_9 = GUITAR_KEY_CHORD_MAP[ord("9")]
    assert chord_9 == "F"
    assert guitar.set_chord(chord_9) is True
    assert guitar.active_chord == "F"
    # Same 2-finger pose on next frame leaves chord at F
    guitar.update([lh_2], frame_shape=shape, current_time=0.16)
    assert guitar.active_chord == "F", f"Unchanged 2-finger pose overwrote keyboard [9] chord to {guitar.active_chord}"

    # 6. Direct-touch selection persists after leaving the box if no new gesture transition occurs
    bx1, by1, bx2, by2 = guitar.chord_boxes["Am"]
    touch_pos = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
    lh_touch = make_lh(2, touch_xy=touch_pos)
    guitar.update([lh_touch], frame_shape=shape, current_time=0.18)
    assert guitar.active_chord == "Am", f"Direct touch on Am failed: got {guitar.active_chord}"

    # Leaving the box with same 2-finger pose
    guitar.update([lh_2], frame_shape=shape, current_time=0.20)
    assert guitar.active_chord == "Am", f"Chord reverted after leaving touch box to {guitar.active_chord}"

    # 7. All 9 keyboard chords still work
    for key_code in range(ord("1"), ord("9") + 1):
        target_chord = GUITAR_KEY_CHORD_MAP[key_code]
        assert guitar.set_chord(target_chord) is True
        assert guitar.active_chord == target_chord

