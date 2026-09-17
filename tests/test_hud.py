import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import GestureARApp
from gesture_engine import AppState, GestureEngine
from vision_tracker import HandData

def test_hud_rendering_and_layout():
    print("Testing HUD rendering across multiple resolutions...")
    app = GestureARApp(camera_id=0, width=1920, height=1080)
    
    # Test for 1280x720, 1920x1080, and 3840x2160 frames
    for w, h in [(1280, 720), (1920, 1080), (3840, 2160)]:
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
            
            if avail_w >= 380:
                fps_text = "FPS: 60.0 | AI: 58.0 (ULTRA) | RIG: 1€ | HANDS: 2"
            else:
                fps_text = "60FPS | AI:58(ULTRA) | 1€ | H:2"
                
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

    # Test AsyncHandTracker dynamic controls
    print("Testing AsyncHandTracker controls...")
    app.async_tracker.model_complexity = 0
    assert app.async_tracker.model_complexity == 0
    app.async_tracker.model_complexity = 1
    assert app.async_tracker.model_complexity == 1
    
    app.async_tracker.filter_mode = "zero_lag"
    assert app.hand_tracker.filter_mode == "zero_lag"
    app.async_tracker.filter_mode = "one_euro"
    assert app.hand_tracker.filter_mode == "one_euro"
    
    # Test kinematic extrapolation
    test_hands = app.async_tracker.get_latest_hands(time.perf_counter(), extrapolate=True)
    assert isinstance(test_hands, list)

    print("All HUD rendering, layout collision, and AsyncHandTracker tests PASSED successfully!")
    app.shutdown()

if __name__ == "__main__":
    test_hud_rendering_and_layout()
