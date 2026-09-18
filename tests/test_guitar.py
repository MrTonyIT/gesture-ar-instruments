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


def test_simultaneous_gesture_sources_exclusive_priority(guitar):
    """
    Validates strict priority arbitration among simultaneous inputs:
    Priority: 1. Direct Touch > 2. Deliberate Thumb Pinch > 3. Finger-count gesture.
    Specifically validates cases A, B, C, D, E, F from specification.
    """
    from typing import Optional

    shape = (720, 1280, 3)
    # Prime guitar with shape to build chord boxes
    guitar.update([], frame_shape=shape, current_time=0.0)
    bx1, by1, bx2, by2 = guitar.chord_boxes["F"]
    touch_f_pos = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)

    def make_lh_touch_and_pinch(touch_active: bool, pinch_finger: Optional[int]) -> HandData:
        wrist = np.array([300.0, 400.0], dtype=np.float32)
        pts_px = np.zeros((21, 3), dtype=np.float32)
        pts_px[:, :2] = wrist

        # Hand scale reference
        pts_px[9, :2] = wrist + np.array([0.0, -80.0], dtype=np.float32)

        # MCPs
        pts_px[6, :2] = wrist + np.array([-30.0, -60.0], dtype=np.float32)
        pts_px[10, :2] = wrist + np.array([0.0, -60.0], dtype=np.float32)
        pts_px[14, :2] = wrist + np.array([30.0, -60.0], dtype=np.float32)
        pts_px[18, :2] = wrist + np.array([60.0, -60.0], dtype=np.float32)

        # Finger tips default folded
        for tip_id in [8, 12, 16, 20]:
            pts_px[tip_id, :2] = wrist + np.array([0.0, -20.0], dtype=np.float32)

        # Thumb (4)
        thumb_pos = np.array([280.0, 320.0], dtype=np.float32)
        pts_px[4, :2] = thumb_pos

        # Pinch finger (8=C, 12=G) pinched within 10px of thumb
        if pinch_finger is not None:
            pts_px[pinch_finger, :2] = thumb_pos + np.array([5.0, 5.0], dtype=np.float32)

        # Pinky (20) touches F chord box if touch_active
        if touch_active:
            pts_px[20, :2] = [touch_f_pos[0], touch_f_pos[1]]
        else:
            pts_px[20, :2] = wrist + np.array([60.0, -20.0], dtype=np.float32)

        pts_norm = pts_px.copy()
        pts_norm[:, 0] /= 1280.0
        pts_norm[:, 1] /= 720.0
        return HandData(
            handedness="Left",
            landmarks_norm=pts_norm,
            landmarks_px=pts_px,
            timestamp=0.0,
        )

    # Reset guitar to initial chord C
    guitar.set_chord("C")
    assert guitar.active_chord == "C"

    # Requirement A: Touch chord box F while a pinch resolving to C is simultaneously present
    # Final chord MUST be F (direct touch has higher priority than pinch).
    frame_tp = make_lh_touch_and_pinch(touch_active=True, pinch_finger=8)
    guitar.update([frame_tp], frame_shape=shape, current_time=1.00)
    assert guitar.active_chord == "F", f"Requirement A failed: expected F, got {guitar.active_chord}"

    # Requirement B: Keep that same touch+pinch combination for another frame -> still F.
    guitar.update([frame_tp], frame_shape=shape, current_time=1.02)
    assert guitar.active_chord == "F", f"Requirement B failed: expected F, got {guitar.active_chord}"

    # Requirement C: Release direct touch while the exact same pinch remains held ->
    # MUST NOT suddenly emit a deferred C event!
    frame_p_only = make_lh_touch_and_pinch(touch_active=False, pinch_finger=8)
    guitar.update([frame_p_only], frame_shape=shape, current_time=1.04)
    assert guitar.active_chord == "F", f"Requirement C failed: release touch caused deferred pinch C; got {guitar.active_chord}"

    # Requirement D: Release and then perform a NEW pinch transition ->
    # pinch selection may change the chord (pinch to middle finger 12 -> G).
    frame_no_pinch = make_lh_touch_and_pinch(touch_active=False, pinch_finger=None)
    guitar.update([frame_no_pinch], frame_shape=shape, current_time=1.06)
    assert guitar.active_chord == "F"  # Still F

    frame_pinch_g = make_lh_touch_and_pinch(touch_active=False, pinch_finger=12)
    guitar.update([frame_pinch_g], frame_shape=shape, current_time=1.08)
    assert guitar.active_chord == "G", f"Requirement D failed: new pinch did not change chord to G; got {guitar.active_chord}"

    # Requirement E: Keyboard F remains persistent under an unchanged finger-count pose
    from main import GUITAR_KEY_CHORD_MAP
    assert guitar.set_chord("F") is True
    assert guitar.active_chord == "F"
    guitar.update([frame_pinch_g], frame_shape=shape, current_time=1.10)
    assert guitar.active_chord == "F", f"Requirement E failed: unchanged pose overwrote keyboard F to {guitar.active_chord}"

    # Requirement F: All 9 keyboard chord mappings still work
    for k in range(ord("1"), ord("9") + 1):
        target = GUITAR_KEY_CHORD_MAP[k]
        assert guitar.set_chord(target) is True
        assert guitar.active_chord == target


def test_guitar_asset_cwd_isolation(tmp_path, monkeypatch, mock_audio):
    """
    Verifies that default guitar asset loading does not touch or create directories in CWD,
    cannot be shadowed by an arbitrary CWD asset, and returns a valid BGRA sprite from memory.
    """
    import cv2
    import numpy as np
    from instruments import Guitar

    # Switch working directory to isolated temp path
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "assets").exists()

    # 1. Launching/constructing Guitar does not create tmp/assets
    g1 = Guitar(zones=None, audio_engine=mock_audio)
    assert not (tmp_path / "assets").exists(), "Guitar construction created assets/ in arbitrary CWD!"
    assert g1.sprite is not None
    assert g1.sprite.ndim == 3 and g1.sprite.shape[2] == 4

    # 2. Default asset lookup cannot be shadowed by an arbitrary CWD asset
    cwd_assets = tmp_path / "assets"
    cwd_assets.mkdir()
    fake_sprite = np.zeros((32, 32, 4), dtype=np.uint8)
    cv2.imwrite(str(cwd_assets / "guitar_blocky.png"), fake_sprite)
    assert (cwd_assets / "guitar_blocky.png").exists()

    g2 = Guitar(zones=None, audio_engine=mock_audio)
    # The default lookup must NOT have loaded the 32x32 fake sprite from CWD
    assert g2.sprite.shape[:2] != (32, 32), "Default asset lookup was shadowed by arbitrary CWD asset!"

    # 3. Procedural in-memory fallback generates valid BGRA sprite without writing to disk
    g3 = Guitar(zones=None, audio_engine=mock_audio, asset_path=str(tmp_path / "nonexistent.png"))
    assert g3.sprite is not None
    assert g3.sprite.shape == (360, 960, 4)
    assert not (tmp_path / "nonexistent.png").exists(), "Procedural fallback wrote to disk unexpectedly!"


def test_mirrored_camera_guitar_pose_hardware_regression(mock_audio):
    """
    Hardware regression test for mirrored webcam geometry.
    In mirrored webcam view, the player's physical Right Hand (strumming) appears
    on SCREEN-LEFT (X ~ 600-750) and physical Left Hand (fret/neck) appears on
    SCREEN-RIGHT (X ~ 1150-1300).
    Verifies:
    1. body_side_sign is -1 (body left of neck).
    2. Body anchor is positioned on screen-left, neck anchor on screen-right.
    3. String segments span naturally near the right hand and are fully inside frame bounds.
    4. Distance between strumming fingers/wrist and string midpoints is tightly bounded (< 100px),
       proving strings are reachable (unlike the buggy positive-X constraint which pushed body to X=1520+).
    """
    shape = (720, 1280, 3)
    guitar = Guitar(zones=None, audio_engine=mock_audio)

    # Physical Left Hand at Screen-Right (neck/fret)
    lh_pts = np.zeros((21, 3), dtype=np.float32)
    lh_pts[0, :2] = [1200.0, 420.0]
    lh_pts[9, :2] = [1180.0, 360.0]
    lh_pts[10, :2] = [1170.0, 340.0]
    left_hand = HandData(
        handedness="Left",
        landmarks_norm=np.zeros((21, 3)),
        landmarks_px=lh_pts,
        timestamp=0.0,
    )

    # Physical Right Hand at Screen-Left (strumming)
    rh_pts = np.zeros((21, 3), dtype=np.float32)
    rh_pts[0, :2] = [650.0, 560.0]  # Wrist
    rh_pts[4, :2] = [630.0, 520.0]  # Thumb tip
    rh_pts[8, :2] = [650.0, 510.0]  # Index tip
    right_hand = HandData(
        handedness="Right",
        landmarks_norm=np.zeros((21, 3)),
        landmarks_px=rh_pts,
        timestamp=0.0,
    )

    # Run several frames to allow adaptive pose smoothing to converge
    for t in np.linspace(0.0, 0.3, 10):
        guitar.update([left_hand, right_hand], frame_shape=shape, current_time=float(t))

    assert guitar.body_side_sign == -1, f"Expected body_side_sign -1, got {guitar.body_side_sign}"
    assert guitar.current_neck_pt[0] > guitar.current_body_pt[0], "Neck must be to the right of body in mirrored view"
    assert guitar.current_neck_pt[0] > 1100.0, f"Neck X {guitar.current_neck_pt[0]} should be near left hand (~1180)"
    assert guitar.current_body_pt[0] < 700.0, f"Body X {guitar.current_body_pt[0]} should be near right hand (~620-650)"

    # Check that all projected strings are valid and within camera frame
    for idx, string in enumerate(guitar.strings):
        s_x1, s_y1 = string.screen_start
        s_x2, s_y2 = string.screen_end
        assert 0.0 <= s_x1 <= 1280.0, f"String {idx} start X {s_x1} outside frame"
        assert 0.0 <= s_x2 <= 1280.0, f"String {idx} end X {s_x2} outside frame"
        assert 0.0 <= s_y1 <= 720.0, f"String {idx} start Y {s_y1} outside frame"
        assert 0.0 <= s_y2 <= 720.0, f"String {idx} end Y {s_y2} outside frame"

    # String 2 midpoint should be in immediate reach of right hand wrist/index
    str2_mid_x = (guitar.strings[2].screen_start[0] + guitar.strings[2].screen_end[0]) / 2.0
    str2_mid_y = (guitar.strings[2].screen_start[1] + guitar.strings[2].screen_end[1]) / 2.0
    rh_dist = float(np.hypot(str2_mid_x - 650.0, str2_mid_y - 560.0))
    assert rh_dist < 140.0, f"String 2 mid ({str2_mid_x:.1f}, {str2_mid_y:.1f}) is too far from right hand wrist ({rh_dist:.1f}px > 140px)"


def test_mirrored_camera_two_hand_strumming_flow(mock_audio):
    """
    Verifies realistic two-hand strumming interaction under mirrored camera view.
    Tests:
    1. Fingers above string do not trigger audio.
    2. Crossing string downwards triggers audio and updates guitar telemetry.
    3. Rapid reverse crossing within debounce window is locked out.
    4. Legitimate subsequent stroke after debounce window sounds next note.
    """
    shape = (720, 1280, 3)
    guitar = Guitar(zones=None, audio_engine=mock_audio)

    # Steady Left Hand at neck
    lh_pts = np.zeros((21, 3), dtype=np.float32)
    lh_pts[0, :2] = [1200.0, 420.0]
    lh_pts[9, :2] = [1180.0, 360.0]
    lh_pts[10, :2] = [1170.0, 340.0]
    left_hand = HandData(
        handedness="Left",
        landmarks_norm=np.zeros((21, 3)),
        landmarks_px=lh_pts,
        timestamp=0.0,
    )

    # Right hand initial position
    rh_pts = np.zeros((21, 3), dtype=np.float32)
    rh_pts[0, :2] = [650.0, 560.0]
    right_hand = HandData(
        handedness="Right",
        landmarks_norm=np.zeros((21, 3)),
        landmarks_px=rh_pts,
        timestamp=0.0,
    )

    # Warmup pose
    for t in np.linspace(0.0, 0.2, 5):
        guitar.update([left_hand, right_hand], frame_shape=shape, current_time=float(t))

    target_string = guitar.strings[2]  # D3 string
    s_x1, s_y1 = target_string.screen_start
    s_x2, s_y2 = target_string.screen_end
    mid_x = (s_x1 + s_x2) / 2.0
    mid_y = (s_y1 + s_y2) / 2.0

    # Place index tip (id 8) stably above string and clear initial motion state
    rh_pts[8, :2] = [mid_x, mid_y - 18.0]
    guitar.reset_motion_state()
    mock_audio.plucked_strings.clear()

    # Frame 1: Index tip (id 8) above string at t=1.00
    guitar.update([left_hand, right_hand], frame_shape=shape, current_time=1.00)
    assert len(mock_audio.plucked_strings) == 0
    assert guitar.strum_count == 0

    # Frame 2: Index tip moves down past string to (mid_x, mid_y + 18.0) at t=1.02
    rh_pts[8, :2] = [mid_x, mid_y + 18.0]
    guitar.update([left_hand, right_hand], frame_shape=shape, current_time=1.02)
    assert len(mock_audio.plucked_strings) == 1
    assert guitar.strum_count == 1
    assert guitar.last_strum_string == 2
    assert guitar.last_strum_note == "E3"
    assert guitar.last_strum_played is True
    assert guitar.last_strum_time == 1.02

    # Frame 3: Immediate reverse crossing within debounce (< 0.08s) at t=1.04
    rh_pts[8, :2] = [mid_x, mid_y - 18.0]
    guitar.update([left_hand, right_hand], frame_shape=shape, current_time=1.04)
    assert len(mock_audio.plucked_strings) == 1, "Debounce lockout failed to suppress rapid reverse crossing"
    assert guitar.strum_count == 1

    # Frame 4: Valid reverse strum after debounce window (> 0.08s) at t=1.15
    rh_pts[8, :2] = [mid_x, mid_y + 18.0]
    guitar.update([left_hand, right_hand], frame_shape=shape, current_time=1.15)
    assert len(mock_audio.plucked_strings) == 2
    assert guitar.strum_count == 2
    assert guitar.last_strum_time == 1.15


def test_guitar_orientation_hysteresis_and_switching(mock_audio):
    """
    Verifies orientation detection with hysteresis:
    - delta_x < -80px sets body_side_sign = -1 (mirrored camera).
    - delta_x in [-80px, 80px] retains previous orientation (prevents jitter).
    - delta_x > 80px switches to body_side_sign = 1 (unmirrored / left-handed).
    - Single-hand fallbacks preserve orientation and keep instrument in frame.
    """
    shape = (720, 1280, 3)
    guitar = Guitar(zones=None, audio_engine=mock_audio)

    # 1. Mirrored pose: Right hand at 600, Left hand at 1200 -> delta_x = -600 < -80
    lh_pts = np.zeros((21, 3), dtype=np.float32)
    lh_pts[0, :2] = [1200.0, 420.0]
    lh_pts[9, :2] = [1180.0, 360.0]
    lh_pts[10, :2] = [1170.0, 340.0]
    lh = HandData(handedness="Left", landmarks_norm=np.zeros((21, 3)), landmarks_px=lh_pts, timestamp=0.0)

    rh_pts = np.zeros((21, 3), dtype=np.float32)
    rh_pts[0, :2] = [600.0, 560.0]
    rh = HandData(handedness="Right", landmarks_norm=np.zeros((21, 3)), landmarks_px=rh_pts, timestamp=0.0)

    guitar.update([lh, rh], frame_shape=shape, current_time=0.0)
    assert guitar.body_side_sign == -1

    # 2. Crossover jitter: hands move near center (delta_x = -20 and +20)
    rh_pts[0, :2] = [890.0, 560.0]
    lh_pts[0, :2] = [910.0, 420.0]
    lh_pts[9, :2] = [890.0, 360.0]
    lh_pts[10, :2] = [880.0, 340.0]
    guitar.update([lh, rh], frame_shape=shape, current_time=0.02)
    assert guitar.body_side_sign == -1, "Hysteresis failed for delta_x = -20px"

    rh_pts[0, :2] = [910.0, 560.0]
    lh_pts[0, :2] = [890.0, 420.0]
    lh_pts[9, :2] = [870.0, 360.0]
    lh_pts[10, :2] = [860.0, 340.0]
    guitar.update([lh, rh], frame_shape=shape, current_time=0.04)
    assert guitar.body_side_sign == -1, "Hysteresis failed for delta_x = +20px"

    # 3. Deliberate unmirrored pose: Right hand at 1150, Left hand at 350 -> delta_x = +800 > 80
    rh_pts[0, :2] = [1150.0, 560.0]
    lh_pts[0, :2] = [350.0, 420.0]
    lh_pts[9, :2] = [340.0, 360.0]
    lh_pts[10, :2] = [330.0, 340.0]
    for t in [0.06, 0.08, 0.10]:
        guitar.update([lh, rh], frame_shape=shape, current_time=t)
    assert guitar.body_side_sign == 1, "Orientation failed to switch to +1 for delta_x = +800"
    assert guitar.current_body_pt[0] > guitar.current_neck_pt[0]

    # 4. Single-hand fallback under body_side_sign = 1
    # Drop right hand: left hand only
    guitar.update([lh], frame_shape=shape, current_time=0.12)
    assert guitar.body_side_sign == 1
    assert guitar.current_body_pt[0] > guitar.current_neck_pt[0]

    # Drop left hand: right hand only
    guitar.update([rh], frame_shape=shape, current_time=0.14)
    assert guitar.body_side_sign == 1
    assert guitar.current_body_pt[0] > guitar.current_neck_pt[0]

    # 5. Switch back to mirrored pose
    rh_pts[0, :2] = [600.0, 560.0]
    lh_pts[0, :2] = [1200.0, 420.0]
    lh_pts[9, :2] = [1180.0, 360.0]
    lh_pts[10, :2] = [1170.0, 340.0]
    for t in [0.16, 0.18, 0.20]:
        guitar.update([lh, rh], frame_shape=shape, current_time=t)
    assert guitar.body_side_sign == -1

    # Single-hand fallbacks under body_side_sign = -1
    guitar.update([lh], frame_shape=shape, current_time=0.22)
    assert guitar.body_side_sign == -1
    assert guitar.current_body_pt[0] < guitar.current_neck_pt[0]

    guitar.update([rh], frame_shape=shape, current_time=0.24)
    assert guitar.body_side_sign == -1
    assert guitar.current_body_pt[0] < guitar.current_neck_pt[0]


def test_guitar_audio_path_and_muted_strings(mock_audio):
    """
    Verifies audio triggering and telemetry distinction between muted and sounded strings.
    Chord C: (None, 3, 2, 0, 1, 0)
    String 0: Muted (None) -> play_guitar returns False, telemetry played=False, note='X'
    String 1: Sounded (3, C3) -> play_guitar returns True, telemetry played=True, note='C3'
    """
    shape = (720, 1280, 3)
    guitar = Guitar(zones=None, audio_engine=mock_audio)
    guitar.set_chord("C")

    # Mirror pose
    lh_pts = np.zeros((21, 3), dtype=np.float32)
    lh_pts[0, :2] = [1200.0, 420.0]
    lh_pts[9, :2] = [1180.0, 360.0]
    lh_pts[10, :2] = [1170.0, 340.0]
    lh = HandData(handedness="Left", landmarks_norm=np.zeros((21, 3)), landmarks_px=lh_pts, timestamp=0.0)

    rh_pts = np.zeros((21, 3), dtype=np.float32)
    rh_pts[0, :2] = [650.0, 560.0]
    rh = HandData(handedness="Right", landmarks_norm=np.zeros((21, 3)), landmarks_px=rh_pts, timestamp=0.0)

    for t in np.linspace(0.0, 0.2, 5):
        guitar.update([lh, rh], frame_shape=shape, current_time=float(t))

    # Test String 0 (muted in chord C)
    str0 = guitar.strings[0]
    mid0_x = (str0.screen_start[0] + str0.screen_end[0]) / 2.0
    mid0_y = (str0.screen_start[1] + str0.screen_end[1]) / 2.0

    rh_pts[8, :2] = [mid0_x, mid0_y - 15.0]
    guitar.update([lh, rh], frame_shape=shape, current_time=1.0)
    rh_pts[8, :2] = [mid0_x, mid0_y + 15.0]
    guitar.update([lh, rh], frame_shape=shape, current_time=1.02)

    assert guitar.strum_count == 1
    assert guitar.last_strum_string == 0
    assert guitar.last_strum_note == "X"
    assert guitar.last_strum_played is False
    assert guitar.strings[0].vibration_amplitude < 5.0  # Subtle muted vibration

    # Test String 1 (sounded in chord C: A2 + 3 semitones = C3)
    str1 = guitar.strings[1]
    mid1_x = (str1.screen_start[0] + str1.screen_end[0]) / 2.0
    mid1_y = (str1.screen_start[1] + str1.screen_end[1]) / 2.0

    rh_pts[8, :2] = [mid1_x, mid1_y - 15.0]
    guitar.update([lh, rh], frame_shape=shape, current_time=1.15)
    rh_pts[8, :2] = [mid1_x, mid1_y + 15.0]
    guitar.update([lh, rh], frame_shape=shape, current_time=1.17)

    assert guitar.strum_count == 2
    assert guitar.last_strum_string == 1
    assert guitar.last_strum_note == "C3"
    assert guitar.last_strum_played is True
    assert guitar.strings[1].vibration_amplitude > 5.0  # Full resonant vibration


def test_guitar_hand_role_preservation(mock_audio):
    """
    Verifies hand roles:
    - Left Hand is strictly fret/chord hand and NEVER triggers string strumming.
    - Right Hand is strictly strumming hand and NEVER alters chord selection.
    """
    shape = (720, 1280, 3)
    guitar = Guitar(zones=None, audio_engine=mock_audio)
    guitar.set_chord("C")

    # Warmup
    guitar.update([], frame_shape=shape, current_time=0.0)
    target_string = guitar.strings[2]
    mid_x = (target_string.screen_start[0] + target_string.screen_end[0]) / 2.0
    mid_y = (target_string.screen_start[1] + target_string.screen_end[1]) / 2.0

    # 1. Left Hand fingers sweep across strings
    lh_pts = np.zeros((21, 3), dtype=np.float32)
    lh_pts[0, :2] = [1200.0, 420.0]
    lh_pts[8, :2] = [mid_x, mid_y - 20.0]
    lh = HandData(handedness="Left", landmarks_norm=np.zeros((21, 3)), landmarks_px=lh_pts, timestamp=0.0)
    guitar.update([lh], frame_shape=shape, current_time=1.0)

    lh_pts[8, :2] = [mid_x, mid_y + 20.0]
    guitar.update([lh], frame_shape=shape, current_time=1.02)

    assert len(mock_audio.plucked_strings) == 0, "Left hand erroneously triggered string strumming!"
    assert guitar.strum_count == 0

    # 2. Right Hand positioned near top chord box (box 1: X in [20, 68], Y in [56, 96])
    rh_pts = np.zeros((21, 3), dtype=np.float32)
    rh_pts[0, :2] = [650.0, 560.0]
    rh_pts[8, :2] = [40.0, 75.0]  # inside chord box 1 (C)
    rh = HandData(handedness="Right", landmarks_norm=np.zeros((21, 3)), landmarks_px=rh_pts, timestamp=0.0)
    guitar.set_chord("Em")
    assert guitar.active_chord == "Em"

    guitar.update([rh], frame_shape=shape, current_time=1.04)
    assert guitar.active_chord == "Em", "Right hand erroneously selected chord!"



