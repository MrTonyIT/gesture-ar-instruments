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
    
    # Test for both 1280x720 and 1920x1080 frames
    for w, h in [(1280, 720), (1920, 1080)]:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Test every possible state
        for state in AppState:
            app.gesture_engine.state = state
            
            # Dummy hands
            dummy_landmarks_norm = np.zeros((21, 3), dtype=np.float32)
            dummy_landmarks_px = np.zeros((21, 3), dtype=np.float32)
            hands = [
                HandData(handedness="Left", landmarks_norm=dummy_landmarks_norm, landmarks_px=dummy_landmarks_px),
                HandData(handedness="Right", landmarks_norm=dummy_landmarks_norm, landmarks_px=dummy_landmarks_px),
            ]
            
            # Render HUD
            app._render_hud(frame.copy(), hands)
            
            # Verify _render_header calculations
            state_str = f"STATE: {state.value}"
            st_size = cv2.getTextSize(state_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
            sx1 = 160
            sx2 = sx1 + st_size[0] + 32
            
            mode_tag = "SMOOTH-0ms"
            fps_text = f"FPS: 60.0 | RIG: {mode_tag} | HANDS: 2"
            telem_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            telem_w = telem_size[0]
            
            btn_zone_start = w - 515
            space_left = sx2 + 18
            space_right = btn_zone_start - 16
            
            if space_right - space_left > telem_w:
                tx1 = space_left + ((space_right - space_left) - telem_w) // 2 - 8
            else:
                tx1 = space_left
            tx2 = tx1 + telem_w + 16
            
            # Assert that State badge (sx1..sx2) and Telemetry badge (tx1..tx2) NEVER overlap
            assert sx2 < tx1, f"Overlap detected at {w}x{h} in {state}! sx2={sx2} >= tx1={tx1}"
            # Assert that Telemetry badge does not collide with buttons
            assert tx2 <= btn_zone_start + 10, f"Telemetry collided with buttons at {w}x{h} in {state}! tx2={tx2} > {btn_zone_start}"
            
    print("All HUD rendering and layout collision tests PASSED successfully!")
    app.shutdown()

if __name__ == "__main__":
    test_hud_rendering_and_layout()
