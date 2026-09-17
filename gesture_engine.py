"""
gesture_engine.py
=================
Spatial Gesture Recognition & Ergonomic Instrument State Machine.

Key Updates & Ergonomic Design:
1. Virtual Piano Docking:
   - When both hands form an 'L' shape for 1.2s, locks the Virtual Piano to the
     permanent bottom desk-surface dock (X: 0.05 to 0.95, Y: 0.72 to 0.96).
   - Prevents "Gorilla Arm" fatigue by grounding the instrument where the user's
     arms rest naturally against their desk.

2. Virtual Guitar Placement:
   - When Left Hand forms an 'O' pinch for 1.2s, locks the Virtual Guitar to a
     natural ~25-degree diagonal torso orientation:
       * Left Hand Fretboard: Mid-left chest level (X: 0.15 to 0.40, Y: 0.45 to 0.65).
       * Right Hand Strum Box: Lower-right lap/desk level (X: 0.55 to 0.85, Y: 0.65 to 0.85).

3. Global Reset & Exit Controls:
   - Hold the on-screen [RESET] button for 0.7s (or press 'r') to revert to IDLE.
   - Hold the on-screen [EXIT] button for 3.0s (or press 'q' / ESC) to cleanly shut down.
"""

from __future__ import annotations

import enum
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from vision_tracker import HandData

logger = logging.getLogger("GestureEngine")


class AppState(str, enum.Enum):
    IDLE = "IDLE"
    CREATING_PIANO = "CREATING_PIANO"
    PIANO_ACTIVE = "PIANO_ACTIVE"
    CREATING_GUITAR = "CREATING_GUITAR"
    SCULPTING_GUITAR = "SCULPTING_GUITAR"
    READY_TO_ASSEMBLE = "READY_TO_ASSEMBLE"
    FUSION_SNAP = "FUSION_SNAP"
    GUITAR_ACTIVE = "GUITAR_ACTIVE"


class GestureEngine:
    """
    Evaluates geometric hand poses and executes state transitions for AR instruments.
    Supports VIP Interactive 3D Pinch-to-Sculpt & Assemble Instrument Spawning,
    on-screen touch button reset, and application shutdown.
    """

    HOLD_DURATION = 1.2          # Seconds required to lock an instrument
    PINCH_THRESHOLD_RATIO = 0.045  # Distance / screen_width for 'O' pinch
    EXIT_HOLD_DURATION = 3.0     # Seconds required to hold Exit Button to quit application
    RESET_HOLD_DURATION = 0.70   # 0.7s hold required when touching Reset Button
    SCULPT_HOLD_DURATION = 1.50  # 1.5s hold required to sculpt & lock circle/rectangle
    LOCK_SCULPT_HOLD_DURATION = 1.00  # 1.0s hold required to toggle shape sculpting lock

    # Ergonomic Normalized Bounds
    PIANO_DOCK_NORM = (0.05, 0.72, 0.95, 0.96)  # Bottom 25% desk-surface dock
    GUITAR_FRETBOARD_NORM = (0.15, 0.45, 0.40, 0.65)  # Mid-left chest level
    GUITAR_STRUM_NORM = (0.55, 0.65, 0.85, 0.85)      # Lower-right lap level

    def __init__(self) -> None:
        self.state: AppState = AppState.IDLE

        # Timers & Progress for Instrument Spawning
        self._gesture_start_time: Optional[float] = None
        self.progress: float = 0.0  # 0.0 to 1.0
        self.progress_center: Tuple[int, int] = (0, 0)
        self.progress_label: str = ""

        # Global Exit Button State (Touch & hold top-right button for 3.0s)
        self.should_exit: bool = False
        self._exit_start_time: Optional[float] = None
        self.exit_progress: float = 0.0
        self.exit_center: Tuple[int, int] = (0, 0)
        self.exit_label: str = ""
        self.exit_btn_bbox: Optional[Tuple[int, int, int, int]] = None
        self.is_touching_exit: bool = False

        # Global Top-Right Reset Button State (Touch & hold next to Exit for 0.7s)
        self.reset_btn_bbox: Optional[Tuple[int, int, int, int]] = None
        self.is_touching_reset: bool = False
        self.reset_just_triggered: bool = False
        self._reset_start_time: Optional[float] = None
        self.reset_progress: float = 0.0
        self.reset_center: Tuple[int, int] = (0, 0)
        self.reset_label: str = ""

        # Global Top-Right Lock Sculpt Button State (Touch & hold next to Reset for 1.0s)
        self.is_sculpt_locked: bool = False
        self.lock_btn_bbox: Optional[Tuple[int, int, int, int]] = None
        self.is_touching_lock: bool = False
        self.lock_just_toggled: bool = False
        self._lock_start_time: Optional[float] = None
        self.lock_progress: float = 0.0
        self.lock_center: Tuple[int, int] = (0, 0)
        self.lock_label: str = ""
        self._lock_triggered_in_touch: bool = False

        # Locked Instrument Coordinates (in pixels)
        self.piano_bbox: Optional[Tuple[int, int, int, int]] = None
        self.guitar_zones: Optional[Dict[str, Tuple[int, int, int, int]]] = None

        # Candidate previews for HUD rendering during creation
        self.candidate_piano_bbox: Optional[Tuple[int, int, int, int]] = None
        self.candidate_guitar_zones: Optional[Dict[str, Tuple[int, int, int, int]]] = None

        # 1. Right Hand: Circle / Soundbox
        self.soundbox_sculpted: bool = False
        self.soundbox_progress: float = 0.0
        self.soundbox_hold_progress: float = 0.0
        self.soundbox_radius: float = 0.0
        self.soundbox_center: Tuple[int, int] = (0, 0)
        self.soundbox_anchor: Tuple[int, int] = (0, 0)
        self._soundbox_hold_start: Optional[float] = None
        self.soundbox_just_locked: bool = False
        self._soundbox_require_release: bool = False
        self.is_dragging_soundbox: bool = False
        self._soundbox_drag_offset: Optional[Tuple[float, float]] = None
        self._soundbox_drag_finger_id: Optional[str] = None

        # 2. Left Hand: Rectangle / Fretboard & Neck
        self.neck_sculpted: bool = False
        self.neck_progress: float = 0.0
        self.neck_hold_progress: float = 0.0
        self.neck_width: float = 0.0
        self.neck_height: float = 44.0
        self.neck_center: Tuple[int, int] = (0, 0)
        self.neck_anchor: Tuple[int, int] = (0, 0)
        self._neck_hold_start: Optional[float] = None
        self.neck_just_locked: bool = False
        self._neck_require_release: bool = False
        self._neck_touch_primed: bool = False
        self._neck_touch_center: Tuple[int, int] = (0, 0)
        self.is_dragging_neck: bool = False
        self._neck_drag_offset: Optional[Tuple[float, float]] = None
        self._neck_drag_finger_id: Optional[str] = None
        self.neck_line_upper: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.neck_line_lower: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.neck_overflow_just_occurred: bool = False
        self.soundbox_overflow_just_occurred: bool = False

        # 3. Magnetic Assembly & Fusion
        self.assemble_distance: float = 999.0
        self.magnetic_attraction: float = 0.0
        self._fusion_start_time: Optional[float] = None
        self.fusion_progress: float = 0.0
        self.fusion_center: Tuple[int, int] = (0, 0)
        self.fusion_just_triggered: bool = False

    def reset_to_idle(self) -> None:
        """Flushes active instruments and resets to IDLE state."""
        logger.info("Reset triggered: Returning to IDLE")
        self.state = AppState.IDLE
        self._gesture_start_time = None
        self.progress = 0.0
        self.progress_label = ""
        self._reset_start_time = None
        self.reset_progress = 0.0
        self.reset_label = ""
        self.piano_bbox = None
        self.candidate_piano_bbox = None
        self.guitar_zones = None
        self.candidate_guitar_zones = None

        # Reset Sculpting & Fusion State
        self.soundbox_sculpted = False
        self.soundbox_progress = 0.0
        self.soundbox_hold_progress = 0.0
        self.soundbox_radius = 0.0
        self.soundbox_center = (0, 0)
        self.soundbox_anchor = (0, 0)
        self._soundbox_hold_start = None
        self.soundbox_just_locked = False
        self._soundbox_require_release = False
        self.is_dragging_soundbox = False
        self._soundbox_drag_offset = None
        self._soundbox_drag_finger_id = None

        self.neck_sculpted = False
        self.neck_progress = 0.0
        self.neck_hold_progress = 0.0
        self.neck_width = 0.0
        self.neck_center = (0, 0)
        self.neck_anchor = (0, 0)
        self._neck_hold_start = None
        self.neck_just_locked = False
        self._neck_require_release = False
        self._neck_touch_primed = False
        self._neck_touch_center = (0, 0)
        self.is_dragging_neck = False
        self._neck_drag_offset = None
        self._neck_drag_finger_id = None

        self.assemble_distance = 999.0
        self.magnetic_attraction = 0.0
        self._fusion_start_time = None
        self.fusion_progress = 0.0
        self.fusion_center = (0, 0)
        self.fusion_just_triggered = False
        self.neck_line_upper = None
        self.neck_line_lower = None
        self.neck_overflow_just_occurred = False
        self.soundbox_overflow_just_occurred = False

        # Reset Lock button touch transients (preserves is_sculpt_locked user choice)
        self.is_touching_lock = False
        self._lock_start_time = None
        self.lock_progress = 0.0
        self.lock_label = ""
        self._lock_triggered_in_touch = False
        self.lock_just_toggled = False

    def update(
        self,
        hands: List[HandData],
        frame_shape: Tuple[int, int, int],
        current_time: Optional[float] = None,
    ) -> None:
        """
        Updates gesture evaluation and ergonomic state transitions.

        Args:
            hands: Filtered HandData list.
            frame_shape: (height, width, channels) of the display frame.
            current_time: Optional timestamp for deterministic headless testing.
        """
        h, w, _ = frame_shape
        now = time.perf_counter() if current_time is None else float(current_time)

        # 1. Check Exit Button Touch (Hold fingertip inside top-right button for 3.0s)
        btn_w, btn_h = 160, 36
        exit_btn = (w - btn_w - 15, 8, w - 15, 8 + btn_h)
        self.exit_btn_bbox = exit_btn

        touching_exit = False
        exit_touch_pos = (0, 0)
        for hand in hands:
            # Check all fingertips with small generous padding (+6px)
            for tip_id in [8, 4, 12, 16, 20]:
                tx, ty = float(hand.landmarks_px[tip_id, 0]), float(hand.landmarks_px[tip_id, 1])
                if (exit_btn[0] - 6) <= tx <= (exit_btn[2] + 6) and (exit_btn[1] - 6) <= ty <= (exit_btn[3] + 6):
                    touching_exit = True
                    exit_touch_pos = (int(tx), int(ty))
                    break
            if touching_exit:
                break

        if touching_exit:
            if self._exit_start_time is None:
                self._exit_start_time = now
            elapsed_exit = now - self._exit_start_time
            self.exit_progress = float(np.clip(elapsed_exit / self.EXIT_HOLD_DURATION, 0.0, 1.0))
            self.exit_center = exit_touch_pos
            self.is_touching_exit = True
            rem_sec = max(0.0, self.EXIT_HOLD_DURATION - elapsed_exit)
            self.exit_label = f"EXIT ({rem_sec:.1f}s)"

            if self.exit_progress >= 1.0:
                logger.info("Exit button held for 3.0s! Setting should_exit = True")
                self.should_exit = True
                return
        else:
            self._exit_start_time = None
            self.exit_progress = 0.0
            self.exit_label = ""
            self.is_touching_exit = False

        # Separate hands by handedness
        left_hand: Optional[HandData] = None
        right_hand: Optional[HandData] = None
        for hand in hands:
            if hand.handedness == "Left":
                left_hand = hand
            elif hand.handedness == "Right":
                right_hand = hand

        # 2. Check Top-Right Cyber Reset Button Touch (Hold fingertip for 0.7s to Reset)
        # Positioned right next to Exit button: (w - 345, 8, w - 185, 44)
        btn_w, btn_h = 160, 36
        reset_btn = (w - btn_w * 2 - 25, 8, w - btn_w - 25, 8 + btn_h)  # (w - 345, 8, w - 185, 44)
        self.reset_btn_bbox = reset_btn

        touching_reset = False
        reset_touch_pos = (0, 0)
        for hand in hands:
            for tip_id in [8, 4, 12, 16, 20]:
                tx, ty = float(hand.landmarks_px[tip_id, 0]), float(hand.landmarks_px[tip_id, 1])
                if (reset_btn[0] - 6) <= tx <= (reset_btn[2] + 6) and (reset_btn[1] - 6) <= ty <= (reset_btn[3] + 6):
                    touching_reset = True
                    reset_touch_pos = (int(tx), int(ty))
                    break
            if touching_reset:
                break

        self.reset_just_triggered = False
        if touching_reset:
            if self._reset_start_time is None:
                self._reset_start_time = now
            elapsed_reset = now - self._reset_start_time
            self.reset_progress = float(np.clip(elapsed_reset / self.RESET_HOLD_DURATION, 0.0, 1.0))
            self.reset_center = reset_touch_pos
            self.is_touching_reset = True
            rem_r = max(0.0, self.RESET_HOLD_DURATION - elapsed_reset)
            self.reset_label = f"RESET ({rem_r:.1f}s)"

            if self.reset_progress >= 1.0:
                logger.info("Reset button held for 0.7s! Resetting to IDLE")
                self.reset_to_idle()
                self.reset_just_triggered = True
                self._reset_start_time = None
                self.reset_progress = 0.0
                self.is_touching_reset = False
                return
        else:
            self._reset_start_time = None
            self.reset_progress = 0.0
            self.reset_label = ""
            self.is_touching_reset = False

        # 3. Check Top-Right Cyber Lock Sculpt Button Touch (Hold fingertip for 1.0s to Toggle Lock)
        # Positioned right next to Reset button: (w - 515, 8, w - 355, 44)
        btn_w, btn_h = 160, 36
        lock_btn = (w - btn_w * 3 - 35, 8, w - btn_w * 2 - 35, 8 + btn_h)  # (w - 515, 8, w - 355, 44)
        self.lock_btn_bbox = lock_btn

        touching_lock = False
        lock_touch_pos = (0, 0)
        for hand in hands:
            for tip_id in [8, 4, 12, 16, 20]:
                tx, ty = float(hand.landmarks_px[tip_id, 0]), float(hand.landmarks_px[tip_id, 1])
                if (lock_btn[0] - 6) <= tx <= (lock_btn[2] + 6) and (lock_btn[1] - 6) <= ty <= (lock_btn[3] + 6):
                    touching_lock = True
                    lock_touch_pos = (int(tx), int(ty))
                    break
            if touching_lock:
                break

        self.lock_just_toggled = False
        if touching_lock:
            self.is_touching_lock = True
            self.lock_center = lock_touch_pos
            if not self._lock_triggered_in_touch:
                if self._lock_start_time is None:
                    self._lock_start_time = now
                elapsed_lock = now - self._lock_start_time
                self.lock_progress = float(np.clip(elapsed_lock / self.LOCK_SCULPT_HOLD_DURATION, 0.0, 1.0))
                rem_l = max(0.0, self.LOCK_SCULPT_HOLD_DURATION - elapsed_lock)
                action_str = "UNLOCK" if self.is_sculpt_locked else "LOCK"
                self.lock_label = f"{action_str} ({rem_l:.1f}s)"

                if self.lock_progress >= 1.0:
                    self.is_sculpt_locked = not self.is_sculpt_locked
                    self.lock_just_toggled = True
                    self._lock_triggered_in_touch = True
                    logger.info("Lock Sculpt button toggled! is_sculpt_locked is now %s", self.is_sculpt_locked)
                    if self.is_sculpt_locked:
                        # Clear any in-flight sculpting progress when locking
                        if not self.soundbox_sculpted:
                            self.soundbox_progress = 0.0
                            self.soundbox_hold_progress = 0.0
                            self._soundbox_hold_start = None
                        if not self.neck_sculpted:
                            self._neck_touch_primed = False
                            self.neck_progress = 0.0
                            self.neck_hold_progress = 0.0
                            self._neck_hold_start = None
                        if self.state in (AppState.CREATING_PIANO, AppState.CREATING_GUITAR, AppState.SCULPTING_GUITAR) and not (self.soundbox_sculpted or self.neck_sculpted):
                            self.state = AppState.IDLE
            else:
                self.lock_progress = 1.0
                status_str = "LOCKED" if self.is_sculpt_locked else "UNLOCKED"
                self.lock_label = status_str
        else:
            self._lock_start_time = None
            self.lock_progress = 0.0
            self.lock_label = ""
            self.is_touching_lock = False
            self._lock_triggered_in_touch = False

        # Compute dynamic pixel bounds based on current frame resolution
        dock_piano_px = (
            int(round(self.PIANO_DOCK_NORM[0] * w)),
            int(round(self.PIANO_DOCK_NORM[1] * h)),
            int(round(self.PIANO_DOCK_NORM[2] * w)),
            int(round(self.PIANO_DOCK_NORM[3] * h)),
        )

        diag_guitar_px = {
            "fretboard": (
                int(round(self.GUITAR_FRETBOARD_NORM[0] * w)),
                int(round(self.GUITAR_FRETBOARD_NORM[1] * h)),
                int(round(self.GUITAR_FRETBOARD_NORM[2] * w)),
                int(round(self.GUITAR_FRETBOARD_NORM[3] * h)),
            ),
            "strum_zone": (
                int(round(self.GUITAR_STRUM_NORM[0] * w)),
                int(round(self.GUITAR_STRUM_NORM[1] * h)),
                int(round(self.GUITAR_STRUM_NORM[2] * w)),
                int(round(self.GUITAR_STRUM_NORM[3] * h)),
            ),
        }

        # Reset per-frame one-shot triggers
        self.soundbox_just_locked = False
        self.neck_just_locked = False
        self.fusion_just_triggered = False
        self.neck_overflow_just_occurred = False
        self.soundbox_overflow_just_occurred = False

        # 2. State-Dependent Evaluation: Virtual Piano Spawning
        if self.state in (AppState.IDLE, AppState.CREATING_PIANO):
            # Check for Piano Spawn Gesture: Both hands forming 'L' shape
            piano_detected = self._check_piano_gesture(left_hand, right_hand) if not self.is_sculpt_locked else False
            if piano_detected:
                self.candidate_piano_bbox = dock_piano_px
                if self.state != AppState.CREATING_PIANO:
                    self.state = AppState.CREATING_PIANO
                    self._gesture_start_time = now

                elapsed = now - (self._gesture_start_time or now)
                self.progress = float(np.clip(elapsed / self.HOLD_DURATION, 0.0, 1.0))

                # Position progress bar centered right above the bottom dock
                self.progress_center = (w // 2, int(h * 0.60))
                self.progress_label = f"LOCKING DESK PIANO {int(self.progress * 100)}%"

                if self.progress >= 1.0:
                    self.piano_bbox = dock_piano_px
                    self.state = AppState.PIANO_ACTIVE
                    self.progress = 0.0
                    self.progress_label = ""
                    self._gesture_start_time = None
                    logger.info("Piano locked to ergonomic desk dock: %s", self.piano_bbox)
            else:
                if self.state == AppState.CREATING_PIANO:
                    self.state = AppState.IDLE
                    self._gesture_start_time = None
                    self.progress = 0.0
                    self.progress_label = ""
                    self.candidate_piano_bbox = None

        # 3. State-Dependent Evaluation: VIP Sculpt & Assemble Virtual Guitar
        if self.state not in (AppState.PIANO_ACTIVE, AppState.CREATING_PIANO):
            # Precompute distances between left and right hands if both are present
            min_contact = 999.0
            mid_l = None
            mid_r = None
            if left_hand is not None and right_hand is not None:
                p4_l = left_hand.landmarks_px[4, :2]
                p8_l = left_hand.landmarks_px[8, :2]
                mid_l = (p4_l + p8_l) / 2.0

                p4_r = right_hand.landmarks_px[4, :2]
                p8_r = right_hand.landmarks_px[8, :2]
                mid_r = (p4_r + p8_r) / 2.0

                d_mid = float(np.linalg.norm(mid_l - mid_r))
                d_idx = float(np.linalg.norm(p8_l - p8_r))
                d_thm = float(np.linalg.norm(p4_l - p4_r))
                min_contact = min(d_mid, d_idx, d_thm)

            # 3a. Right Hand Sculpting: Circle / Soundbox (Thùng đàn)
            # Sculpted ONLY by pinching & stretching Thumb (4) and Index (8) of Right Hand.
            # Suppressed when the two hands are touching or actively sculpting the rectangle.
            if right_hand is not None and not self.soundbox_sculpted and not self.is_sculpt_locked:
                two_hands_near = (
                    left_hand is not None and (self._neck_touch_primed or min_contact <= 85.0)
                )
                if not two_hands_near:
                    p4_r = right_hand.landmarks_px[4, :2]
                    p8_r = right_hand.landmarks_px[8, :2]
                    d_r = float(np.linalg.norm(p4_r - p8_r))
                    mid_r_single = (p4_r + p8_r) / 2.0

                    if d_r > 26.0:
                        prog_r = np.clip((d_r - 28.0) / 58.0, 0.0, 1.0)
                        self.soundbox_progress = float(prog_r)
                        self.soundbox_radius = float(np.clip(d_r * 0.75, 22.0, 95.0))
                        self.soundbox_center = (int(mid_r_single[0]), int(mid_r_single[1]))

                        # Screen Overflow Auto-Disappearance check
                        # User requirement: "nếu tràn màn hình thì cho nó tự động biến mât và hình tròn cũng vậy"
                        cx_sb, cy_sb = self.soundbox_center
                        r_sb = self.soundbox_radius
                        if (cx_sb - r_sb < 0) or (cx_sb + r_sb > w) or (cy_sb - r_sb < 0) or (cy_sb + r_sb > h):
                            logger.info("Soundbox circle overflowed screen boundary! Auto-disappeared.")
                            self.soundbox_progress = 0.0
                            self.soundbox_hold_progress = 0.0
                            self._soundbox_hold_start = None
                            self.soundbox_overflow_just_occurred = True
                        elif prog_r >= 0.84:
                            if self._soundbox_hold_start is None:
                                self._soundbox_hold_start = now
                            elapsed_hold_r = now - self._soundbox_hold_start
                            self.soundbox_hold_progress = float(np.clip(elapsed_hold_r / self.SCULPT_HOLD_DURATION, 0.0, 1.0))
                            if elapsed_hold_r >= self.SCULPT_HOLD_DURATION:
                                self.soundbox_sculpted = True
                                self.soundbox_progress = 1.0
                                self.soundbox_hold_progress = 1.0
                                self.soundbox_just_locked = True
                                self._soundbox_require_release = True
                                self._soundbox_hold_start = None
                                logger.info("Soundbox Circle Sculpted & Frozen at %s!", self.soundbox_center)
                        else:
                            self._soundbox_hold_start = None
                            self.soundbox_hold_progress = 0.0
                    else:
                        self.soundbox_progress = 0.0
                        self._soundbox_hold_start = None
                        self.soundbox_hold_progress = 0.0
                else:
                    self.soundbox_progress = 0.0
                    self._soundbox_hold_start = None
                    self.soundbox_hold_progress = 0.0
            elif not self.soundbox_sculpted:
                self.soundbox_progress = 0.0
                self._soundbox_hold_start = None
                self.soundbox_hold_progress = 0.0

            # 3b. Rectangle / Fretboard & Neck (Cần đàn)
            # SPECS: "cái hình chữ nhật là phải do ngón trỏ và ngón cái cảu hai bàn tay chạm vào nhau và kéo ra mới được"
            # MUST be created by Thumb (4) & Index (8) of BOTH hands touching each other first, then pulling apart!
            if not self.neck_sculpted and not self.is_sculpt_locked:
                if left_hand is not None and right_hand is not None and mid_l is not None and mid_r is not None:
                    # 1. Contact Detection: User touches thumbs and index fingers of both hands together
                    if min_contact <= 75.0:
                        self._neck_touch_primed = True
                        self._neck_touch_center = (int((mid_l[0] + mid_r[0]) / 2.0), int((mid_l[1] + mid_r[1]) / 2.0))

                    if self._neck_touch_primed:
                        p8_l = left_hand.landmarks_px[8, :2]
                        p8_r = right_hand.landmarks_px[8, :2]
                        p4_l = left_hand.landmarks_px[4, :2]
                        p4_r = right_hand.landmarks_px[4, :2]

                        # Store 2 dynamic long laser lines between fingers
                        # User requirement: "đoạn ngón tay tro và ngón tay cái 2 bàn tay chạm vào nhau phải có 2 đường thẳng dài khi kéo ra để tạo hình chữ nhật"
                        self.neck_line_upper = ((int(p8_l[0]), int(p8_l[1])), (int(p8_r[0]), int(p8_r[1])))
                        self.neck_line_lower = ((int(p4_l[0]), int(p4_l[1])), (int(p4_r[0]), int(p4_r[1])))

                        # 2. Pulling Apart: Measure distance between the two hands' contact points
                        # User requirement: "đường thẳng dài ko bị cố định" -> no arbitrary ceiling cap
                        d_pull = float(np.linalg.norm(mid_l - mid_r))
                        if d_pull > 50.0:
                            prog_l = float(np.clip((d_pull - 50.0) / 140.0, 0.0, 1.0))
                            self.neck_progress = prog_l
                            self.neck_width = float(max(50.0, d_pull * 1.15))
                            h_diff = abs(((p8_l[1] + p8_r[1]) / 2.0) - ((p4_l[1] + p4_r[1]) / 2.0))
                            self.neck_height = float(np.clip(max(36.0, h_diff * 1.1), 35.0, 95.0))
                            self.neck_center = (int((mid_l[0] + mid_r[0]) / 2.0), int((mid_l[1] + mid_r[1]) / 2.0))

                            # Screen Overflow Auto-Disappearance check
                            # User requirement: "nếu tràn màn hình thì cho nó tự động biến mât và hình tròn cũng vậy"
                            rx1_n = self.neck_center[0] - self.neck_width / 2.0
                            rx2_n = self.neck_center[0] + self.neck_width / 2.0
                            ry1_n = self.neck_center[1] - self.neck_height / 2.0
                            ry2_n = self.neck_center[1] + self.neck_height / 2.0

                            is_overflow_n = (
                                rx1_n < 0 or rx2_n > w or ry1_n < 0 or ry2_n > h
                                or min(p8_l[0], p8_r[0], p4_l[0], p4_r[0]) < 5
                                or max(p8_l[0], p8_r[0], p4_l[0], p4_r[0]) > w - 5
                                or min(p8_l[1], p8_r[1], p4_l[1], p4_r[1]) < 5
                                or max(p8_l[1], p8_r[1], p4_l[1], p4_r[1]) > h - 5
                            )

                            if is_overflow_n and prog_l > 0.05:
                                logger.info("Rectangle overflowed screen boundary! Disappeared.")
                                self.neck_progress = 0.0
                                self.neck_hold_progress = 0.0
                                self._neck_hold_start = None
                                self._neck_touch_primed = False
                                self.neck_line_upper = None
                                self.neck_line_lower = None
                                self.neck_overflow_just_occurred = True
                            elif prog_l >= 0.82:
                                if self._neck_hold_start is None:
                                    self._neck_hold_start = now
                                elapsed_hold_l = now - self._neck_hold_start
                                self.neck_hold_progress = float(np.clip(elapsed_hold_l / self.SCULPT_HOLD_DURATION, 0.0, 1.0))
                                if elapsed_hold_l >= self.SCULPT_HOLD_DURATION:
                                    self.neck_sculpted = True
                                    self.neck_progress = 1.0
                                    self.neck_hold_progress = 1.0
                                    self.neck_just_locked = True
                                    self._neck_require_release = True
                                    self._neck_hold_start = None
                                    self._neck_touch_primed = False
                                    logger.info("Fretboard Neck Rectangle Sculpted by Two Hands Touching & Pulling Apart at %s!", self.neck_center)
                            else:
                                self._neck_hold_start = None
                                self.neck_hold_progress = 0.0
                        else:
                            # Hands touching (< 50px) but not yet pulled apart
                            self.neck_progress = 0.0
                            self._neck_hold_start = None
                            self.neck_hold_progress = 0.0
                            self.neck_center = self._neck_touch_center
                else:
                    # Missing one of the hands before locking: cancel primed touch and progress
                    self._neck_touch_primed = False
                    self.neck_progress = 0.0
                    self._neck_hold_start = None
                    self.neck_hold_progress = 0.0
                    self.neck_line_upper = None
                    self.neck_line_lower = None
            elif not self.neck_sculpted:
                self._neck_touch_primed = False
                self.neck_progress = 0.0
                self._neck_hold_start = None
                self.neck_hold_progress = 0.0
                self.neck_line_upper = None
                self.neck_line_lower = None

            # 3c. Dual-Finger Touch Interaction for Soundbox (Circle)
            # User requirement: "cái hình tròn là ngón cái với ngón trỏ chạm cùng lúc để di chuyển được
            # là ko phải hai ngón chụm lại mà hai ngón chạm điểm nào của hình cũng được miễn là hai ngón chạm vào thì mới di chuyển được"
            TOUCH_MARGIN_SB = 24.0      # Max distance from circle boundary to register finger touch
            RELEASE_MARGIN_SB = 38.0    # Margin to maintain touch while dragging (hysteresis prevents micro-drops)

            if self.soundbox_sculpted:
                # Arming check: user must first disengage/release after sculpting completes
                if self._soundbox_require_release:
                    has_touching_hand = False
                    for hand in hands:
                        p4 = hand.landmarks_px[4, :2]
                        p8 = hand.landmarks_px[8, :2]
                        d4 = float(np.linalg.norm(p4 - self.soundbox_center))
                        d8 = float(np.linalg.norm(p8 - self.soundbox_center))
                        if d4 <= (self.soundbox_radius + TOUCH_MARGIN_SB) and d8 <= (self.soundbox_radius + TOUCH_MARGIN_SB):
                            has_touching_hand = True
                            break
                    if not has_touching_hand:
                        self._soundbox_require_release = False

                if not self._soundbox_require_release:
                    if self.is_dragging_soundbox and self._soundbox_drag_finger_id is not None:
                        matched_hand = None
                        for hand in hands:
                            if hand.handedness == self._soundbox_drag_finger_id:
                                matched_hand = hand
                                break
                        if matched_hand is not None and self._soundbox_drag_offset is not None:
                            p4 = matched_hand.landmarks_px[4, :2]
                            p8 = matched_hand.landmarks_px[8, :2]
                            mid_touch = (p4 + p8) / 2.0
                            new_x = int(mid_touch[0] + self._soundbox_drag_offset[0])
                            new_y = int(mid_touch[1] + self._soundbox_drag_offset[1])
                            cand_center = (
                                max(40, min(w - 40, new_x)),
                                max(60, min(h - 40, new_y))
                            )
                            # Verify BOTH thumb & index are still touching the circle (release margin)
                            d4 = float(np.linalg.norm(p4 - cand_center))
                            d8 = float(np.linalg.norm(p8 - cand_center))
                            if d4 <= (self.soundbox_radius + RELEASE_MARGIN_SB) and d8 <= (self.soundbox_radius + RELEASE_MARGIN_SB):
                                self.soundbox_center = cand_center
                            else:
                                # One or both fingers moved away / released from the circle: freeze immediately!
                                self.is_dragging_soundbox = False
                                self._soundbox_drag_offset = None
                                self._soundbox_drag_finger_id = None
                                logger.info("Soundbox finger released! Strictly frozen at %s", self.soundbox_center)
                        else:
                            # Hand lost: freeze immediately!
                            self.is_dragging_soundbox = False
                            self._soundbox_drag_offset = None
                            self._soundbox_drag_finger_id = None
                            logger.info("Hand lost! Soundbox strictly frozen at %s", self.soundbox_center)
                    else:
                        # Grab check: BOTH thumb (4) and index (8) of a hand must touch ANY point of the circle
                        for hand in hands:
                            p4 = hand.landmarks_px[4, :2]
                            p8 = hand.landmarks_px[8, :2]
                            d4 = float(np.linalg.norm(p4 - self.soundbox_center))
                            d8 = float(np.linalg.norm(p8 - self.soundbox_center))
                            if d4 <= (self.soundbox_radius + TOUCH_MARGIN_SB) and d8 <= (self.soundbox_radius + TOUCH_MARGIN_SB):
                                self.is_dragging_soundbox = True
                                self._soundbox_drag_finger_id = hand.handedness
                                mid_touch = (p4 + p8) / 2.0
                                self._soundbox_drag_offset = (
                                    float(self.soundbox_center[0] - mid_touch[0]),
                                    float(self.soundbox_center[1] - mid_touch[1])
                                )
                                logger.info("Both fingers (%s) touched Soundbox! Dragging initiated at %s.", hand.handedness, mid_touch)
                                break

            # 3d. Dual-Finger Touch Interaction for Neck (Rectangle)
            # User requirement: thumb and index touching anywhere on the rectangle moves it
            TOUCH_MARGIN_NECK = 24.0
            RELEASE_MARGIN_NECK = 38.0

            if self.neck_sculpted:
                rx1_n = self.neck_center[0] - self.neck_width / 2.0
                rx2_n = self.neck_center[0] + self.neck_width / 2.0
                ry1_n = self.neck_center[1] - self.neck_height / 2.0
                ry2_n = self.neck_center[1] + self.neck_height / 2.0

                # Arming check: user must first separate hands after sculpting completes
                if self._neck_require_release:
                    has_touching_neck = False
                    for hand in hands:
                        p4 = hand.landmarks_px[4, :2]
                        p8 = hand.landmarks_px[8, :2]
                        in_4 = (rx1_n - TOUCH_MARGIN_NECK <= p4[0] <= rx2_n + TOUCH_MARGIN_NECK and ry1_n - TOUCH_MARGIN_NECK <= p4[1] <= ry2_n + TOUCH_MARGIN_NECK)
                        in_8 = (rx1_n - TOUCH_MARGIN_NECK <= p8[0] <= rx2_n + TOUCH_MARGIN_NECK and ry1_n - TOUCH_MARGIN_NECK <= p8[1] <= ry2_n + TOUCH_MARGIN_NECK)
                        if in_4 and in_8:
                            has_touching_neck = True
                            break
                    if not has_touching_neck:
                        self._neck_require_release = False

                if not self._neck_require_release:
                    if self.is_dragging_neck and self._neck_drag_finger_id is not None:
                        matched_hand = None
                        for hand in hands:
                            if hand.handedness == self._neck_drag_finger_id:
                                matched_hand = hand
                                break
                        if matched_hand is not None and self._neck_drag_offset is not None:
                            p4 = matched_hand.landmarks_px[4, :2]
                            p8 = matched_hand.landmarks_px[8, :2]
                            mid_touch = (p4 + p8) / 2.0
                            new_x = int(mid_touch[0] + self._neck_drag_offset[0])
                            new_y = int(mid_touch[1] + self._neck_drag_offset[1])
                            cand_center = (
                                max(40, min(w - 40, new_x)),
                                max(60, min(h - 40, new_y))
                            )
                            # Verify BOTH thumb & index are still touching the neck (release margin)
                            crx1 = cand_center[0] - self.neck_width / 2.0 - RELEASE_MARGIN_NECK
                            crx2 = cand_center[0] + self.neck_width / 2.0 + RELEASE_MARGIN_NECK
                            cry1 = cand_center[1] - self.neck_height / 2.0 - RELEASE_MARGIN_NECK
                            cry2 = cand_center[1] + self.neck_height / 2.0 + RELEASE_MARGIN_NECK
                            in_4 = (crx1 <= p4[0] <= crx2 and cry1 <= p4[1] <= cry2)
                            in_8 = (crx1 <= p8[0] <= crx2 and cry1 <= p8[1] <= cry2)
                            if in_4 and in_8:
                                self.neck_center = cand_center
                            else:
                                self.is_dragging_neck = False
                                self._neck_drag_offset = None
                                self._neck_drag_finger_id = None
                                logger.info("Neck finger released! Strictly frozen at %s", self.neck_center)
                        else:
                            self.is_dragging_neck = False
                            self._neck_drag_offset = None
                            self._neck_drag_finger_id = None
                            logger.info("Hand lost! Neck strictly frozen at %s", self.neck_center)
                    else:
                        for hand in hands:
                            # Avoid grabbing both shapes with the exact same hand
                            if self.is_dragging_soundbox and self._soundbox_drag_finger_id == hand.handedness:
                                continue
                            p4 = hand.landmarks_px[4, :2]
                            p8 = hand.landmarks_px[8, :2]
                            in_4 = (rx1_n - TOUCH_MARGIN_NECK <= p4[0] <= rx2_n + TOUCH_MARGIN_NECK and ry1_n - TOUCH_MARGIN_NECK <= p4[1] <= ry2_n + TOUCH_MARGIN_NECK)
                            in_8 = (rx1_n - TOUCH_MARGIN_NECK <= p8[0] <= rx2_n + TOUCH_MARGIN_NECK and ry1_n - TOUCH_MARGIN_NECK <= p8[1] <= ry2_n + TOUCH_MARGIN_NECK)
                            if in_4 and in_8:
                                self.is_dragging_neck = True
                                self._neck_drag_finger_id = hand.handedness
                                mid_touch = (p4 + p8) / 2.0
                                self._neck_drag_offset = (
                                    float(self.neck_center[0] - mid_touch[0]),
                                    float(self.neck_center[1] - mid_touch[1])
                                )
                                logger.info("Both fingers (%s) touched Neck! Dragging initiated at %s.", hand.handedness, mid_touch)
                                break

            # 3e. State Transition & Magnetic Snap Assembly
            if self.state == AppState.FUSION_SNAP:
                elapsed = now - (self._fusion_start_time or now)
                self.fusion_progress = float(np.clip(elapsed / 0.48, 0.0, 1.0))
                if self.fusion_progress >= 1.0:
                    self.guitar_zones = diag_guitar_px
                    self.state = AppState.GUITAR_ACTIVE
                    logger.info("Snap Fusion completed! -> GUITAR_ACTIVE")
            elif self.soundbox_sculpted and self.neck_sculpted:
                if self.state not in (AppState.GUITAR_ACTIVE, AppState.FUSION_SNAP):
                    self.state = AppState.READY_TO_ASSEMBLE
                    # Compute distance between centers of both shapes
                    sc_x, sc_y = self.soundbox_center
                    nc_x, nc_y = self.neck_center
                    d_hands = float(np.sqrt((sc_x - nc_x) ** 2 + (sc_y - nc_y) ** 2))
                    self.assemble_distance = d_hands

                    if d_hands < 350.0:
                        self.magnetic_attraction = float(np.clip((350.0 - d_hands) / 200.0, 0.0, 1.0))
                    else:
                        self.magnetic_attraction = 0.0

                    # Snap threshold: Distance < 145px
                    if d_hands < 145.0:
                        self.state = AppState.FUSION_SNAP
                        self._fusion_start_time = now
                        self.fusion_progress = 0.0
                        self.fusion_center = (int((sc_x + nc_x) / 2), int((sc_y + nc_y) / 2))
                        self.fusion_just_triggered = True
                        logger.info("FUSION SNAP triggered at %s!", self.fusion_center)
            elif (self.soundbox_progress > 0.08 or self.neck_progress > 0.08 or self.soundbox_sculpted or self.neck_sculpted):
                if self.state not in (AppState.GUITAR_ACTIVE, AppState.FUSION_SNAP):
                    self.state = AppState.SCULPTING_GUITAR
            elif self.state in (AppState.IDLE, AppState.CREATING_GUITAR):
                # Fallback: Left Hand 'O' pinch held for 1.2s without stretching
                guitar_detected = self._check_guitar_gesture(left_hand, w) if not self.is_sculpt_locked else False
                if guitar_detected and left_hand is not None:
                    self.candidate_guitar_zones = diag_guitar_px
                    if self.state != AppState.CREATING_GUITAR:
                        self.state = AppState.CREATING_GUITAR
                        self._gesture_start_time = now

                    elapsed = now - (self._gesture_start_time or now)
                    self.progress = float(np.clip(elapsed / self.HOLD_DURATION, 0.0, 1.0))

                    tip4 = left_hand.landmarks_px[4, :2]
                    tip8 = left_hand.landmarks_px[8, :2]
                    cx, cy = int((tip4[0] + tip8[0]) / 2), int((tip4[1] + tip8[1]) / 2)
                    self.progress_center = (cx, cy)
                    self.progress_label = f"LOCKING DIAGONAL GUITAR {int(self.progress * 100)}%"

                    if self.progress >= 1.0:
                        self.guitar_zones = diag_guitar_px
                        self.state = AppState.GUITAR_ACTIVE
                        self.progress = 0.0
                        self.progress_label = ""
                        self._gesture_start_time = None
                        logger.info("Guitar locked via 'O' pinch: %s", self.guitar_zones)
                else:
                    if self.state == AppState.CREATING_GUITAR:
                        self.state = AppState.IDLE
                        self._gesture_start_time = None
                        self.progress = 0.0
                        self.progress_label = ""
                        self.candidate_guitar_zones = None

        # If already active, keep pixel coordinates synchronized to window resolution
        if self.state == AppState.PIANO_ACTIVE:
            self.piano_bbox = dock_piano_px
        elif self.state == AppState.GUITAR_ACTIVE:
            self.guitar_zones = diag_guitar_px


    def _is_l_shape(self, hand: Optional[HandData]) -> bool:
        """
        Detects if a hand is forming an 'L' shape:
        - Thumb and Index finger extended perpendicularly.
        - Middle, Ring, Pinky fingers curled into palm.
        """
        if hand is None:
            return False

        pts = hand.landmarks_px
        wrist = pts[0, :2]

        # 1. Index extended: tip (8) distance to wrist > PIP (6) distance * 1.20
        dist_idx_tip = np.linalg.norm(pts[8, :2] - wrist)
        dist_idx_pip = np.linalg.norm(pts[6, :2] - wrist)
        if dist_idx_tip <= dist_idx_pip * 1.18:
            return False

        # 2. Middle, Ring, Pinky curled: tip distance to wrist < PIP distance * 1.18
        for tip_idx, pip_idx in [(12, 10), (16, 14), (20, 18)]:
            d_tip = np.linalg.norm(pts[tip_idx, :2] - wrist)
            d_pip = np.linalg.norm(pts[pip_idx, :2] - wrist)
            if d_tip > d_pip * 1.18:
                return False

        # 3. Thumb extended: tip (4) distance to MCP (2) > IP (3) to MCP (2) * 1.10
        d_thumb_tip = np.linalg.norm(pts[4, :2] - pts[2, :2])
        d_thumb_ip = np.linalg.norm(pts[3, :2] - pts[2, :2])
        if d_thumb_tip <= d_thumb_ip * 1.08:
            return False

        # 4. Perpendicular angle between thumb vector (2 -> 4) and index vector (5 -> 8)
        vec_thumb = pts[4, :2] - pts[2, :2]
        vec_index = pts[8, :2] - pts[5, :2]

        norm_thumb = np.linalg.norm(vec_thumb)
        norm_index = np.linalg.norm(vec_index)
        if norm_thumb < 1e-4 or norm_index < 1e-4:
            return False

        cos_angle = np.dot(vec_thumb, vec_index) / (norm_thumb * norm_index)
        # Perpendicular implies |cos(theta)| is relatively small (between 45 deg and 135 deg)
        if abs(cos_angle) > 0.70:
            return False

        return True

    def _check_piano_gesture(self, left_hand: Optional[HandData], right_hand: Optional[HandData]) -> bool:
        """Verifies both hands forming 'L' shape simultaneously."""
        if left_hand is None or right_hand is None:
            return False
        return bool(self._is_l_shape(left_hand) and self._is_l_shape(right_hand))

    def _check_guitar_gesture(self, left_hand: Optional[HandData], screen_w: int) -> bool:
        """
        Detects Left Hand 'O' pinch gesture:
        Euclidean distance between Landmark 4 (Thumb tip) and Landmark 8 (Index tip) < threshold.
        """
        if left_hand is None:
            return False

        tip4 = left_hand.landmarks_px[4, :2]
        tip8 = left_hand.landmarks_px[8, :2]
        dist = np.linalg.norm(tip4 - tip8)
        threshold = screen_w * self.PINCH_THRESHOLD_RATIO
        return bool(dist < threshold)

    def _is_index_extended(self, hand: HandData) -> bool:
        """
        Detects if the index finger is extended straight outward (pointing or open hand).
        Returns False if the index finger is curled, folded, or retracted into the palm.
        """
        pts = hand.landmarks_px
        wrist = pts[0, :2]
        mcp = pts[5, :2]
        pip = pts[6, :2]
        tip = pts[8, :2]

        d_tip_mcp = float(np.linalg.norm(tip - mcp))
        d_pip_mcp = float(np.linalg.norm(pip - mcp))
        if d_pip_mcp < 1e-3:
            return False

        d_tip_wrist = float(np.linalg.norm(tip - wrist))
        d_pip_wrist = float(np.linalg.norm(pip - wrist))

        return (d_tip_mcp > d_pip_mcp * 1.25) and (d_tip_wrist > d_pip_wrist * 1.05)
