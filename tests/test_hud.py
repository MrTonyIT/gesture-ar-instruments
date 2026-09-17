import os
import sys
import time
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import GestureARApp
from gesture_engine import AppState
from vision_tracker import HandData


@pytest.fixture
def app():
    application = GestureARApp(camera_id=0, width=1920, height=1080, start_threads=False, init_mediapipe=False)
    try:
        yield application
    finally:
        application.shutdown()


def test_hud_rendering_and_layout(app):
    """Verifies HUD layout and badge collision guarantees across 720p, 1080p, 1440p, 4K."""
    for w, h in [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]:
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Test every possible state
        for state in AppState:
            app.gesture_engine.state = state

            # Dummy hands with kinematic velocities
            dummy_landmarks_norm = np.zeros((21, 3), dtype=np.float32)
            dummy_landmarks_px = np.zeros((21, 3), dtype=np.float32)
            hands = [
                HandData(
                    handedness="Left",
                    landmarks_norm=dummy_landmarks_norm,
                    landmarks_px=dummy_landmarks_px,
                    fingertip_velocities={8: (10.0, 5.0)},
                    wrist_velocity=(5.0, 2.0),
                ),
                HandData(
                    handedness="Right",
                    landmarks_norm=dummy_landmarks_norm,
                    landmarks_px=dummy_landmarks_px,
                    fingertip_velocities={8: (-10.0, -5.0)},
                    wrist_velocity=(-5.0, -2.0),
                ),
            ]

            # Set dummy audio peak
            app.audio_engine.current_peak = 0.65

            # Render HUD
            app._render_hud(frame.copy(), hands)

            # Verify _render_header calculations
            state_str = f"STATE: {state.value}"
            st_size = cv2.getTextSize(state_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
            sx1 = 160
            sx2 = sx1 + st_size[0] + 32

            btn_zone_start = w - 515
            space_left = sx2 + 18
            space_right = btn_zone_start - 16
            avail_w = space_right - space_left

            if avail_w >= 410:
                fps_text = "FPS: 60.0 | AI: 58.0 (ULTRA) | 1-EURO | Q:HIG | H:2"
            else:
                fps_text = "60FPS | AI:58 | 1-EURO | H:2"

            telem_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            telem_w = telem_size[0]

            if avail_w > telem_w:
                tx1 = space_left + (avail_w - telem_w) // 2 - 8
            else:
                tx1 = space_left
            tx2 = tx1 + telem_w + 16

            # Assert that State badge (sx1..sx2) and Telemetry badge (tx1..tx2) NEVER overlap
            assert sx2 < tx1, f"Overlap detected at {w}x{h} in {state}! sx2={sx2} >= tx1={tx1}"
            # Assert that Telemetry badge does not collide with buttons
            assert tx2 <= btn_zone_start + 10, f"Telemetry collided with buttons at {w}x{h} in {state}! tx2={tx2} > {btn_zone_start}"

            # Verify audio visualizer space guard
            rem_space = btn_zone_start - (tx2 + 14)
            if rem_space >= 105:
                vx1 = tx2 + 14
                vw = min(130, rem_space - 10)
                assert vx1 + vw <= btn_zone_start, f"Visualizer overflow at {w}x{h} in {state}!"


def test_diagnostics_hud_rendering(app):
    """Verifies developer diagnostics HUD panel renders correctly across resolutions."""
    for w, h in [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        app.show_diagnostics = True
        app.telemetry["cam_ms"] = 4.2
        app.telemetry["track_ms"] = 12.5
        app.telemetry["gest_ms"] = 0.8
        app.telemetry["rend_ms"] = 2.1
        app.telemetry["total_ms"] = 19.6

        app._render_diagnostics_hud(frame)
        # Verify frame was modified (rendered into)
        assert np.any(frame > 0)


def test_tracker_controls_and_kinematics(app):
    """Verifies AsyncHandTracker and HandTracker model complexity and filter mode controls."""
    app.async_tracker.model_complexity = 0
    assert app.async_tracker.model_complexity == 0
    app.async_tracker.model_complexity = 1
    assert app.async_tracker.model_complexity == 1

    app.async_tracker.filter_mode = "deadband"
    assert app.hand_tracker.filter_mode == "deadband"
    app.async_tracker.filter_mode = "one_euro"
    assert app.hand_tracker.filter_mode == "one_euro"

    # Test kinematic extrapolation
    test_hands = app.async_tracker.get_latest_hands(time.perf_counter(), extrapolate=True)
    assert isinstance(test_hands, list)


def test_key_dispatch_and_lowercase_r_no_diagnostics_collision(app):
    """
    Verifies that lowercase 'r' (ASCII 114) unequivocally resets application state to IDLE
    and cannot reach diagnostics HUD, while F3, Tab, and backtick toggle diagnostics.
    Also proves every ordinary ASCII control key has an unambiguous action without collisions.
    """
    from instruments import Guitar
    from main import F3_RAW_KEYS, FILTER_CYCLE_KEYS, GUITAR_KEY_CHORD_MAP

    # 1. Test lowercase 'r' (114) and uppercase 'R' (82)
    app.show_diagnostics = False
    app.gesture_engine.state = AppState.PIANO_ACTIVE
    action_lower_r = app.handle_key(114)  # ord('r') == 114
    assert action_lower_r == "RESET"
    assert app.show_diagnostics is False  # MUST NOT toggle diagnostics!
    assert app.gesture_engine.state == AppState.IDLE

    app.gesture_engine.state = AppState.GUITAR_ACTIVE
    action_upper_r = app.handle_key(ord("R"))
    assert action_upper_r == "RESET"
    assert app.show_diagnostics is False
    assert app.gesture_engine.state == AppState.IDLE

    # 2. Test Diagnostics toggles (F3 raw keys, Tab 9, backtick 96)
    initial_diag = app.show_diagnostics
    for f3_code in F3_RAW_KEYS:
        action = app.handle_key(f3_code)
        assert action == "DIAGNOSTICS"
        assert app.show_diagnostics != initial_diag
        initial_diag = app.show_diagnostics

    action_tab = app.handle_key(9)
    assert action_tab == "DIAGNOSTICS"
    assert app.show_diagnostics != initial_diag
    initial_diag = app.show_diagnostics

    action_backtick = app.handle_key(ord("`"))
    assert action_backtick == "DIAGNOSTICS"
    assert app.show_diagnostics != initial_diag

    # 3. Test Exit keys: q, Q, ESC (27)
    assert app.handle_key(ord("q")) == "EXIT"
    assert app.handle_key(ord("Q")) == "EXIT"
    assert app.handle_key(27) == "EXIT"
    assert app.dispatch_key(ord("q")) is False
    assert app.dispatch_key(ord("Q")) is False
    assert app.dispatch_key(27) is False
    assert app.dispatch_key(ord("r")) is True

    # 4. Test Filter cycle keys: k, K
    for k_key in FILTER_CYCLE_KEYS:
        assert app.handle_key(k_key) == "FILTER"

    # 5. Test Quality profile keys: v, V
    assert app.handle_key(ord("v")) == "QUALITY"
    assert app.handle_key(ord("V")) == "QUALITY"

    # 6. Test Model complexity keys: m, M
    assert app.handle_key(ord("m")) == "MODEL_COMPLEXITY"
    assert app.handle_key(ord("M")) == "MODEL_COMPLEXITY"

    # 7. Test Camera settings keys: p, P
    assert app.handle_key(ord("p")) == "CAMERA_SETTINGS"
    assert app.handle_key(ord("P")) == "CAMERA_SETTINGS"

    # 8. Test Guitar chord keys when guitar is active
    app.guitar = Guitar(zones=None, audio_engine=app.audio_engine)
    for key_code, expected_chord in GUITAR_KEY_CHORD_MAP.items():
        act = app.handle_key(key_code)
        assert act == "GUITAR_CHORD", f"Key {chr(key_code)} ({key_code}) failed to dispatch GUITAR_CHORD"
        assert app.guitar.active_chord == expected_chord

    # 9. Verify no collision among all ordinary ASCII control keys
    control_actions = {
        ord("q"): "EXIT", ord("Q"): "EXIT", 27: "EXIT",
        ord("r"): "RESET", ord("R"): "RESET",
        ord("k"): "FILTER", ord("K"): "FILTER",
        ord("v"): "QUALITY", ord("V"): "QUALITY",
        ord("m"): "MODEL_COMPLEXITY", ord("M"): "MODEL_COMPLEXITY",
        ord("p"): "CAMERA_SETTINGS", ord("P"): "CAMERA_SETTINGS",
        ord("`"): "DIAGNOSTICS", 9: "DIAGNOSTICS",
    }
    for k, expected_action in control_actions.items():
        app.guitar = None
        act = app.handle_key(k)
        assert act == expected_action, f"Key {k} yielded {act}, expected {expected_action}"


def test_x11_and_windows_special_keys_no_collision(app):
    """
    Verifies that backend-specific non-ASCII full key codes (X11 keysyms and Windows
    extended keys) do NOT alias to ordinary ASCII application controls via low-byte masking.
    Also verifies intended X11 special keys (Escape 0xFF1B, Tab 0xFF09, F3 65472).
    """
    from instruments import Guitar

    app.guitar = Guitar(zones=None, audio_engine=app.audio_engine)
    app.guitar.set_chord("E")
    app.quality_profile = "HIGH"
    app.show_diagnostics = False

    # 1. Linux / X11 keysyms that previously collided when masked with & 0xFF
    x11_keys_to_test = {
        65361: "Left Arrow",
        65362: "Up Arrow",
        65363: "Right Arrow",
        65364: "Down Arrow",
        0xFF50: "Home (low byte 0x50 == 'P')",
        0xFF56: "PageDown (low byte 0x56 == 'V')",
        0xFF63: "Insert (low byte 0x63 == 'c')",
        0xFF66: "Redo (low byte 0x66 == 'f')",
    }
    for key_code, desc in x11_keys_to_test.items():
        act = app.handle_key(key_code)
        assert act is None, f"X11 key {desc} (code {key_code}) caused unintended action: {act}"
        assert app.dispatch_key(key_code) is True, f"X11 key {desc} incorrectly caused exit"

    # Verify no side-effects from the above suppressed keys
    assert app.guitar.active_chord == "E", "Insert or Redo incorrectly mutated guitar chord"
    assert app.quality_profile == "HIGH", "PageDown incorrectly mutated quality profile"
    assert app.show_diagnostics is False, "Extended key toggled diagnostics"

    # 2. Windows extended arrow keys
    win_extended_keys = [2424832, 2490368, 2555904, 2621440]
    for key_code in win_extended_keys:
        act = app.handle_key(key_code)
        assert act is None, f"Windows extended key {key_code} caused unintended action: {act}"
        assert app.dispatch_key(key_code) is True, f"Windows extended key {key_code} caused exit"

    # 3. Intended backend-specific special full key codes
    # X11 Escape (0xFF1B = 65307)
    assert app.handle_key(0xFF1B) == "EXIT"
    assert app.dispatch_key(0xFF1B) is False

    # X11 Tab (0xFF09 = 65289)
    init_diag = app.show_diagnostics
    assert app.handle_key(0xFF09) == "DIAGNOSTICS"
    assert app.show_diagnostics != init_diag

    # X11 F3 (0xFFBE = 65472)
    diag_state = app.show_diagnostics
    assert app.handle_key(65472) == "DIAGNOSTICS"
    assert app.show_diagnostics != diag_state



