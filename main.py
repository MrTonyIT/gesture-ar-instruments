"""
main.py
=======
Main Orchestrator & Cyber HUD Visualization for Gesture AR Instruments.

Integrates:
- ThreadedCamera: Non-blocking 60 FPS video capture.
- HandTracker: MediaPipe hand landmarks with mirrored coordinates and EMA smoothing.
- GestureEngine: Spatial gestures, state transitions, progress timers, and reset zones.
- Instruments: Virtual Piano (2 octaves, black-key priority, velocity gating)
               and Virtual Guitar (chord fretboard, 2D line-segment strumming).
- AudioEngine: Non-blocking real-time procedural additive and damped harmonic sound.

Controls:
- Both Hands 'L' Shape: Spawn & Lock Piano (hold 1.2s).
- Left Hand 'O' Pinch: Spawn & Lock Guitar (hold 1.2s).
- Both Wrists to Top 10%: Global Reset to IDLE.
- Press 'q' or ESC: Graceful exit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from audio_engine import AudioEngine
from gesture_engine import AppState, GestureEngine
from instruments import Guitar, Piano
from vision_tracker import HandData, HandTracker, ThreadedCamera

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MainApp")

# MediaPipe Hand Skeleton connections (Pairs of landmark indices)
HAND_CONNECTIONS = [
    # Palm
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17) # Base
]


class GestureARApp:
    """Production AR Instruments Suite Orchestrator."""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.width = width
        self.height = height

        # 1. Initialize Subsystems
        logger.info("Initializing Audio Engine...")
        self.audio_engine = AudioEngine(sample_rate=44100, block_size=256)
        self.audio_engine.start()

        logger.info("Initializing Threaded Camera on source %s...", camera_id)
        self.camera = ThreadedCamera(src=camera_id, width=self.width, height=self.height)
        self.camera.start()

        logger.info("Initializing Hand Tracker...")
        self.hand_tracker = HandTracker(
            max_num_hands=2,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.50,
            ema_alpha=0.65,
            filter_mode="one_euro",
        )

        logger.info("Initializing Gesture Engine...")
        self.gesture_engine = GestureEngine()

        # Active Instruments
        self.piano: Optional[Piano] = None
        self.guitar: Optional[Guitar] = None

        # FPS metrics
        self.prev_frame_time = time.perf_counter()
        self.fps = 0.0

        # Screen overflow auto-disappearance alert
        self.last_overflow_time: float = 0.0
        self.overflow_msg: str = ""

    def run(self) -> None:
        """Main application lifecycle loop."""
        logger.info("Starting Gesture AR Instruments suite. Press 'q' or 'ESC' to exit.")
        window_name = "Interactive Spatial AR Instruments - Cyber HUD"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)

        try:
            while True:
                loop_start = time.perf_counter()

                # 1. Grab raw frame from ThreadedCamera
                ret, raw_bgr_frame = self.camera.read()
                if not ret or raw_bgr_frame is None:
                    time.sleep(0.005)
                    continue

                # 2. Process Hand Tracking on raw frame
                # Coordinates returned are already mirrored and smoothed (X_screen = (1.0 - x) * W)
                hands: List[HandData] = self.hand_tracker.process(raw_bgr_frame)

                # 3. Mirror display frame horizontally for mirror-like AR experience
                display_frame = cv2.flip(raw_bgr_frame, 1)

                # 4. Update Gesture State Machine
                prev_state = self.gesture_engine.state
                self.gesture_engine.update(hands, display_frame.shape)
                cur_state = self.gesture_engine.state

                # 4a. Trigger SFX for Sculpt & Fusion events & Reset & Lock Toggle
                if (
                    self.gesture_engine.soundbox_just_locked
                    or self.gesture_engine.neck_just_locked
                    or self.gesture_engine.reset_just_triggered
                    or self.gesture_engine.lock_just_toggled
                ):
                    self.audio_engine.play_shape_lock()
                if self.gesture_engine.fusion_just_triggered:
                    self.audio_engine.play_fusion_burst()

                # Overflow trigger detection
                if self.gesture_engine.soundbox_overflow_just_occurred:
                    self.last_overflow_time = time.perf_counter()
                    self.overflow_msg = "THUNG DAN TRAN MAN HINH! HINH DA TU DONG BIEN MAT"
                elif self.gesture_engine.neck_overflow_just_occurred:
                    self.last_overflow_time = time.perf_counter()
                    self.overflow_msg = "CAN DAN TRAN MAN HINH! HINH DA TU DONG BIEN MAT"

                # 4b. Check Gesture-based Application Exit (Crossed hands 'X' held for 1.8s)
                if self.gesture_engine.should_exit:
                    logger.info("Gesture Exit confirmed by user: Shutting down application cleanly.")
                    break

                # 5. Handle Instrument Transitions
                if prev_state != cur_state:
                    if cur_state == AppState.PIANO_ACTIVE and self.gesture_engine.piano_bbox is not None:
                        logger.info("Instantiating Virtual Piano...")
                        self.piano = Piano(self.gesture_engine.piano_bbox, self.audio_engine)
                        self.guitar = None
                    elif cur_state == AppState.GUITAR_ACTIVE:
                        logger.info("Instantiating Virtual Guitar...")
                        zones = self.gesture_engine.guitar_zones or {
                            "fretboard": (int(0.15 * self.width), int(0.45 * self.height), int(0.40 * self.width), int(0.65 * self.height)),
                            "strum_zone": (int(0.55 * self.width), int(0.65 * self.height), int(0.85 * self.width), int(0.85 * self.height)),
                        }
                        self.guitar = Guitar(zones, self.audio_engine)
                        self.piano = None
                    elif cur_state in (AppState.IDLE, AppState.SCULPTING_GUITAR, AppState.READY_TO_ASSEMBLE):
                        self.piano = None
                        self.guitar = None

                # 6. Update and Hit-Test Active Instrument
                if cur_state == AppState.PIANO_ACTIVE and self.piano is not None:
                    self.piano.update(hands, display_frame.shape)
                elif cur_state == AppState.GUITAR_ACTIVE and self.guitar is not None:
                    self.guitar.update(hands, display_frame.shape)

                # 7. Render Cyber HUD Overlay
                self._render_hud(display_frame, hands)

                # 8. Frame rate calculation
                now = time.perf_counter()
                dt = now - self.prev_frame_time
                self.prev_frame_time = now
                if dt > 0:
                    current_fps = 1.0 / dt
                    self.fps = 0.9 * self.fps + 0.1 * current_fps

                # Render frame to window
                cv2.imshow(window_name, display_frame)

                # 9. Key Handling: Exit on 'q' or ESC (ASCII 27)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    logger.info("Exit key pressed.")
                    break
                elif key in (ord("r"), ord("R")):
                    # Manual reset shortcut
                    self.gesture_engine.reset_to_idle()
                    self.piano = None
                    self.guitar = None
                elif key in (ord("f"), ord("F")):
                    # Toggle hand tracking filter mode: zero-lag deadband vs pure raw
                    if self.hand_tracker.filter_mode == "zero_lag":
                        self.hand_tracker.filter_mode = "raw"
                        logger.info("Hand tracking switched to PURE RAW (100% direct MediaPipe)")
                    else:
                        self.hand_tracker.filter_mode = "zero_lag"
                        logger.info("Hand tracking switched to ZERO-LAG DEADBAND (0ms delay + jitter-free)")
                elif self.guitar is not None:
                    if key in (ord("1"), ord("c"), ord("C")):
                        self.guitar.active_chord = "C"
                    elif key in (ord("2"), ord("g"), ord("G")):
                        self.guitar.active_chord = "G"
                    elif key in (ord("3"), ord("a"), ord("A")):
                        self.guitar.active_chord = "Am"
                    elif key in (ord("4"), ord("e"), ord("E")):
                        self.guitar.active_chord = "Em"

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt caught.")
        finally:
            self.shutdown()

    def _render_hud(self, frame: np.ndarray, hands: List[HandData]) -> None:
        """Renders cyber HUD elements, instruments, progress rings, and hand skeletons."""
        h, w, _ = frame.shape
        state = self.gesture_engine.state

        # 1. Render Active Instrument or Ergonomic Placement Guides
        if state == AppState.PIANO_ACTIVE and self.piano is not None:
            self.piano.draw(frame)
        elif state == AppState.GUITAR_ACTIVE and self.guitar is not None:
            self.guitar.draw(frame)
        elif state in (AppState.SCULPTING_GUITAR, AppState.READY_TO_ASSEMBLE):
            self._render_sculpting(frame, hands)
        elif state == AppState.FUSION_SNAP:
            self._draw_shockwave(frame, self.gesture_engine.fusion_center, self.gesture_engine.fusion_progress)
        elif state == AppState.CREATING_PIANO and self.gesture_engine.candidate_piano_bbox is not None:
            bx1, by1, bx2, by2 = self.gesture_engine.candidate_piano_bbox
            self._draw_cyber_box(frame, bx1, by1, bx2, by2, (0, 240, 255))
            cv2.putText(frame, "LOCKING TO DESK SURFACE (REST WRISTS)", (bx1 + 14, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 255), 1, cv2.LINE_AA)
        elif state == AppState.CREATING_GUITAR and self.gesture_engine.candidate_guitar_zones is not None:
            fb = self.gesture_engine.candidate_guitar_zones["fretboard"]
            sz = self.gesture_engine.candidate_guitar_zones["strum_zone"]
            self._draw_cyber_box(frame, fb[0], fb[1], fb[2], fb[3], (0, 255, 140))
            self._draw_cyber_box(frame, sz[0], sz[1], sz[2], sz[3], (255, 100, 220))
            cv2.putText(frame, "CHEST CHORDS", (fb[0] + 6, fb[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 140), 1, cv2.LINE_AA)
            cv2.putText(frame, "LAP STRUMMING", (sz[0] + 6, sz[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 100, 220), 1, cv2.LINE_AA)
        elif state == AppState.IDLE:
            # Faint holographic guides for ergonomic desk piano & VIP sculpt guitar
            overlay_guide = frame.copy()
            p_xmin, p_xmax = int(0.05 * w), int(0.95 * w)
            p_ymin, p_ymax = int(0.72 * h), int(0.96 * h)
            cv2.rectangle(overlay_guide, (p_xmin, p_ymin), (p_xmax, p_ymax), (45, 40, 50), 1)
            cv2.putText(overlay_guide, "VUNG PIANO MAT BAN (DAT CO TAY DE CHOI)", (p_xmin + 14, p_ymin + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 85, 100), 1, cv2.LINE_AA)

            # Holographic guidance for sculpting and assembling guitar
            if self.gesture_engine.is_sculpt_locked:
                cv2.putText(overlay_guide, "[🔒 CHE DO KHOA TAO HINH DANG BAT - RIG TAY VAN HOAT DONG]", (int(0.18 * w), int(0.48 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 1, cv2.LINE_AA)
                cv2.putText(overlay_guide, "[AN NUT [L] (1S) O GOC PHAI TREN DE MO LAI TAO HINH]", (int(0.24 * w), int(0.54 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(overlay_guide, "[🖐️ CHAM TRO & CAI 2 TAY KEO RA: CAN DAN]", (int(0.06 * w), int(0.48 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 180), 1, cv2.LINE_AA)
                cv2.putText(overlay_guide, "[🖐️ TAY PHAI KEO GIAN: THUNG TRON]", (int(0.56 * w), int(0.48 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 50, 180), 1, cv2.LINE_AA)
                cv2.putText(overlay_guide, "[⚡ DUA 2 TAY LAI GAN DE GHEP THANH CAY DAN ⚡]", (int(0.30 * w), int(0.54 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay_guide, 0.60, frame, 0.40, 0, frame)

        # 3. Render Circular Progress Bars for Gesture Spawning, Reset, Lock, and Exit
        # 3a. Instrument Spawning Progress (Emerald Ring)
        if self.gesture_engine.progress > 0.0:
            cx, cy = self.gesture_engine.progress_center
            progress = self.gesture_engine.progress
            label = self.gesture_engine.progress_label
            self._draw_progress_ring(frame, cx, cy, progress, label, (0, 255, 180), (40, 40, 50))

        # 3b. Reset Gesture Progress (Gold / Orange Ring)
        if self.gesture_engine.reset_progress > 0.0:
            rcx, rcy = self.gesture_engine.reset_center
            r_prog = self.gesture_engine.reset_progress
            r_lbl = self.gesture_engine.reset_label
            self._draw_progress_ring(frame, rcx, rcy, r_prog, r_lbl, (0, 180, 255), (40, 30, 20))

        # 3c. Lock Sculpt Gesture Progress (Amber / Emerald Ring)
        if self.gesture_engine.lock_progress > 0.0:
            lcx, lcy = self.gesture_engine.lock_center
            l_prog = self.gesture_engine.lock_progress
            l_lbl = self.gesture_engine.lock_label
            ring_col = (0, 255, 140) if self.gesture_engine.is_sculpt_locked else (0, 180, 255)
            self._draw_progress_ring(frame, lcx, lcy, l_prog, l_lbl, ring_col, (30, 25, 20))

        # 3d. Exit Gesture Progress (Crimson / Red Pulsing Warning Ring)
        if self.gesture_engine.exit_progress > 0.0:
            ecx, ecy = self.gesture_engine.exit_center
            e_prog = self.gesture_engine.exit_progress
            e_lbl = self.gesture_engine.exit_label
            self._draw_progress_ring(frame, ecx, ecy, e_prog, e_lbl, (0, 30, 255), (20, 10, 40), is_exit=True)

            # Warning screen border flash when exit progress is high
            if e_prog > 0.4:
                alpha_warn = min(0.35, (e_prog - 0.4) * 0.6)
                warn_overlay = frame.copy()
                cv2.rectangle(warn_overlay, (0, 0), (w, h), (0, 0, 200), 12)
                cv2.addWeighted(warn_overlay, alpha_warn, frame, 1.0 - alpha_warn, 0, frame)

        # 4. Render Mirrored Hand Skeletons & Sleek Cyber Hand Rig
        now_t = time.perf_counter()
        for hand in hands:
            is_left = (hand.handedness == "Left")
            # Cyber palette: Cyan for Left hand, Neon Magenta for Right hand
            primary_color = (0, 240, 255) if is_left else (255, 0, 220)
            glow_color = (0, 140, 180) if is_left else (180, 0, 140)
            pts = hand.landmarks_px.astype(int)

            # 4a. Translucent Cyber Palm Mesh (Adds smooth 3D body & volume to hand rig)
            palm_poly = pts[[0, 1, 5, 9, 13, 17], :2]
            overlay_palm = frame.copy()
            cv2.fillPoly(overlay_palm, [palm_poly], glow_color)
            cv2.addWeighted(overlay_palm, 0.16, frame, 0.84, 0, frame)

            # 4b. Dual-layer cyber bones with natural anatomical tapering
            for idx1, idx2 in HAND_CONNECTIONS:
                pt1 = tuple(pts[idx1, :2])
                pt2 = tuple(pts[idx2, :2])

                # Anatomical tapering: Metacarpals thicker, fingertips sleeker
                if idx1 in (0, 5, 9, 13) and idx2 in (1, 5, 9, 13, 17):
                    th_glow, th_core = 5, 2  # Palm base connections
                elif idx2 in (4, 8, 12, 16, 20):
                    th_glow, th_core = 3, 1  # Distal tips
                else:
                    th_glow, th_core = 4, 2  # Intermediate phalanges

                cv2.line(frame, pt1, pt2, glow_color, th_glow, cv2.LINE_AA)
                cv2.line(frame, pt1, pt2, primary_color, th_core, cv2.LINE_AA)
                cv2.line(frame, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

            # 4c. Render All 21 Joint Nodes (Smooth anti-aliased concentric nodes)
            for i in range(21):
                j_pos = tuple(pts[i, :2])
                if i in (4, 8, 12, 16, 20):
                    # Fingertip Nodes: Prominent reticle with animated breathing pulse
                    pulse_r = int(11 + 2 * np.sin(now_t * 5.0 + i))
                    cv2.circle(frame, j_pos, pulse_r, glow_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, j_pos, 7, primary_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, j_pos, 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, j_pos, 2, primary_color, -1, cv2.LINE_AA)
                elif i == 0:
                    # Wrist Anchor: Glowing marker
                    cv2.circle(frame, j_pos, 10, glow_color, -1, cv2.LINE_AA)
                    cv2.circle(frame, j_pos, 7, primary_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, j_pos, 3, (255, 255, 255), -1, cv2.LINE_AA)
                else:
                    # Intermediate Phalangeal & MCP Joints: Crisp concentric nodes
                    cv2.circle(frame, j_pos, 4, primary_color, -1, cv2.LINE_AA)
                    cv2.circle(frame, j_pos, 2, (255, 255, 255), -1, cv2.LINE_AA)

            # 4d. Handedness Badge at Wrist
            wrist = tuple(pts[0, :2])
            tag = f"[L] LEFT HAND" if is_left else f"[R] RIGHT HAND"
            bx, by = wrist[0] - 40, wrist[1] + 28
            cv2.rectangle(frame, (bx - 4, by - 14), (bx + 105, by + 6), (15, 12, 20), -1)
            cv2.rectangle(frame, (bx - 4, by - 14), (bx + 105, by + 6), primary_color, 1)
            cv2.putText(
                frame,
                tag,
                (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                primary_color,
                1,
                cv2.LINE_AA,
            )

        # 5. Top Cyber Header & Telemetry
        self._render_header(frame, state, len(hands))

    def _render_header(self, frame: np.ndarray, state: AppState, hand_count: int) -> None:
        """Draws top information banner, telemetry status, and instructions."""
        w = frame.shape[1]

        # Darkened top status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (12, 10, 16), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.line(frame, (0, 52), (w, 52), (0, 240, 255), 1)

        # Title
        cv2.putText(
            frame,
            "GESTURE AR INSTRUMENTS",
            (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 240, 255),
            2,
            cv2.LINE_AA,
        )

        # State Badge
        state_colors = {
            AppState.IDLE: (180, 180, 180),
            AppState.CREATING_PIANO: (0, 200, 255),
            AppState.PIANO_ACTIVE: (0, 255, 120),
            AppState.CREATING_GUITAR: (255, 120, 200),
            AppState.SCULPTING_GUITAR: (255, 140, 0),
            AppState.READY_TO_ASSEMBLE: (0, 255, 255),
            AppState.FUSION_SNAP: (0, 240, 255),
            AppState.GUITAR_ACTIVE: (255, 0, 220),
        }
        badge_color = state_colors.get(state, (200, 200, 200))
        cv2.putText(
            frame,
            f"STATE: {state.value}",
            (280, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            badge_color,
            2,
            cv2.LINE_AA,
        )

        # FPS & Hand Tracking Telemetry (Shifted left to accommodate Lock, Reset & Exit buttons)
        mode_tag = "SMOOTH-0ms" if self.hand_tracker.filter_mode == "one_euro" else ("0ms-LAG" if self.hand_tracker.filter_mode == "zero_lag" else "RAW")
        fps_text = f"FPS: {self.fps:.1f} | RIG: {mode_tag} | HANDS: {hand_count}"
        cv2.putText(
            frame,
            fps_text,
            (w - 870, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 240, 255),
            1,
            cv2.LINE_AA,
        )

        # Top Cyber Lock Sculpt Button (Hold fingertip for 1.0s to Toggle Shape Creation Lock)
        lx1, ly1, lx2, ly2 = w - 515, 8, w - 355, 44
        l_prog = self.gesture_engine.lock_progress
        is_touching_lock = self.gesture_engine.is_touching_lock
        is_locked = self.gesture_engine.is_sculpt_locked

        l_sub = frame[ly1:ly2, lx1:lx2]
        if l_sub.size > 0:
            l_overlay = l_sub.copy()
            if is_touching_lock:
                l_overlay[:] = (25, 20, 35) if is_locked else (20, 35, 30)
                fill_w = int((lx2 - lx1) * l_prog)
                if fill_w > 0:
                    fill_col = (0, 255, 140) if is_locked else (0, 160, 255)
                    l_overlay[:, :fill_w] = fill_col
                cv2.addWeighted(l_overlay, 0.85, l_sub, 0.15, 0, l_sub)
                frame[ly1:ly2, lx1:lx2] = l_sub
                border_col = (0, 255, 180) if is_locked else (0, 200, 255)
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), border_col, 2, cv2.LINE_AA)
            else:
                l_overlay[:] = (28, 20, 24) if is_locked else (16, 24, 22)
                cv2.addWeighted(l_overlay, 0.70, l_sub, 0.30, 0, l_sub)
                frame[ly1:ly2, lx1:lx2] = l_sub
                border_col = (0, 160, 255) if is_locked else (0, 180, 140)
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), border_col, 1, cv2.LINE_AA)

        # Text & Countdown inside Lock Sculpt Button
        if is_touching_lock:
            rem_l = max(0.0, 1.0 - (l_prog * 1.0))
            action_txt = "MO" if is_locked else "KHOA"
            l_text = f"{action_txt}: {rem_l:.1f}s ({int(l_prog * 100)}%)"
            cv2.putText(frame, l_text, (lx1 + 8, ly1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (lx2 - 12, ly1 + 12), 4, (0, 255, 200) if is_locked else (0, 180, 255), -1, cv2.LINE_AA)
        else:
            if is_locked:
                cv2.putText(frame, "[L] DA KHOA (1S)", (lx1 + 10, ly1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (lx2 - 12, ly1 + 12), 4, (0, 160, 255), -1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "[L] KHOA HINH (1S)", (lx1 + 8, ly1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 240, 180), 1, cv2.LINE_AA)
                cv2.circle(frame, (lx2 - 12, ly1 + 12), 4, (0, 240, 180), -1, cv2.LINE_AA)

        # Top Cyber Reset Button (Hold fingertip for 0.7s to Reset)
        rx1, ry1, rx2, ry2 = w - 345, 8, w - 185, 44
        r_prog = self.gesture_engine.reset_progress
        is_touching_reset = self.gesture_engine.is_touching_reset

        r_sub = frame[ry1:ry2, rx1:rx2]
        if r_sub.size > 0:
            r_overlay = r_sub.copy()
            if is_touching_reset:
                # Energetic dark cyan / gold fill
                r_overlay[:] = (20, 35, 45)
                fill_w = int((rx2 - rx1) * r_prog)
                if fill_w > 0:
                    r_overlay[:, :fill_w] = (0, 180, 255)  # Glowing Cyber Gold / Orange
                cv2.addWeighted(r_overlay, 0.85, r_sub, 0.15, 0, r_sub)
                frame[ry1:ry2, rx1:rx2] = r_sub
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 220, 255), 2, cv2.LINE_AA)
            else:
                # Resting dark cyan / obsidian badge
                r_overlay[:] = (18, 22, 28)
                cv2.addWeighted(r_overlay, 0.70, r_sub, 0.30, 0, r_sub)
                frame[ry1:ry2, rx1:rx2] = r_sub
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 140, 180), 1, cv2.LINE_AA)

        # Text & Countdown inside Reset Button
        if is_touching_reset:
            rem_r = max(0.0, 0.70 - (r_prog * 0.70))
            r_text = f"RESET: {rem_r:.1f}s ({int(r_prog * 100)}%)"
            cv2.putText(frame, r_text, (rx1 + 10, ry1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (rx2 - 12, ry1 + 12), 4, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "[R] RESET (0.7S)", (rx1 + 14, ry1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        # Top-Right Cyber Exit Button (Hold fingertip for 3.0s to Exit)
        bx1, by1, bx2, by2 = w - 175, 8, w - 15, 44
        e_prog = self.gesture_engine.exit_progress
        is_touching = self.gesture_engine.is_touching_exit

        btn_sub = frame[by1:by2, bx1:bx2]
        if btn_sub.size > 0:
            btn_overlay = btn_sub.copy()
            if is_touching:
                # Energetic dark crimson fill
                btn_overlay[:] = (20, 10, 80)
                # Dynamic fill progress bar from left to right inside the button
                fill_w = int((bx2 - bx1) * e_prog)
                if fill_w > 0:
                    btn_overlay[:, :fill_w] = (0, 35, 210)  # Fiery laser red
                cv2.addWeighted(btn_overlay, 0.85, btn_sub, 0.15, 0, btn_sub)
                frame[by1:by2, bx1:bx2] = btn_sub
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 140, 255), 2, cv2.LINE_AA)
            else:
                # Resting dark crimson / obsidian badge
                btn_overlay[:] = (24, 16, 32)
                cv2.addWeighted(btn_overlay, 0.70, btn_sub, 0.30, 0, btn_sub)
                frame[by1:by2, bx1:bx2] = btn_sub
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60, 40, 160), 1, cv2.LINE_AA)

        # Text & Countdown inside Exit Button
        if is_touching:
            rem_time = max(0.0, 3.0 - (e_prog * 3.0))
            btn_text = f"THOAT: {rem_time:.1f}s ({int(e_prog * 100)}%)"
            cv2.putText(frame, btn_text, (bx1 + 8, by1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (bx2 - 12, by1 + 12), 4, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "[X] THOAT (3S)", (bx1 + 16, by1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 180, 220), 1, cv2.LINE_AA)

        # Context-aware Instruction Banner (Bottom of Screen)
        instruction_map = {
            AppState.IDLE: "CHON DAN: 2 tay 'L' = PIANO | 2 tay keo ra = NAN GUITAR | Nut KHOA (1s) | RESET (0.7s)",
            AppState.CREATING_PIANO: "KHOA PIANO: Giu 2 tay chu 'L' 1.2s...",
            AppState.PIANO_ACTIVE: "PIANO: Go phim tren ban | Cham nut RESET (0.7s) de ve Menu",
            AppState.CREATING_GUITAR: "KHOA GUITAR: Giu tay trai nhup 'O' 1.2s...",
            AppState.SCULPTING_GUITAR: "NAN DAN: Cham tro & cai 2 tay keo ra: CAN DAN (1.5s) | Tay phai: THUNG TRON (1.5s)",
            AppState.READY_TO_ASSEMBLE: "⚡ CHAM DONG THOI NGON CAI & TRO VAO HINH DE KEO! Dua 2 hinh lai gan (<145px) de GHEP DAN! ⚡",
            AppState.FUSION_SNAP: "⚡ SNAP FUSION: DANG KET HOP 2 PHAN THANH CAY DAN 3D HOAN CHINH! ⚡",
            AppState.GUITAR_ACTIVE: "GUITAR: Tay trai gio 1-4 ngon doi hop am | Tay phai dung ngon CAI & TRO danh dan | Nut RESET (0.7s)",
        }
        inst_text = instruction_map.get(state, "")

        # Bottom instruction strip (compact 24px bar docked below piano)
        h = frame.shape[0]
        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (0, h - 24), (w, h), (10, 8, 14), -1)
        cv2.addWeighted(overlay_bottom, 0.75, frame, 0.25, 0, frame)
        cv2.putText(
            frame,
            inst_text,
            (16, h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Nut [KHOA] (1s) | [RESET] (0.7s) | [THOAT] (3s) o tren phai",
            (w - 510, h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_progress_ring(
        self,
        frame: np.ndarray,
        cx: int,
        cy: int,
        progress: float,
        label: str,
        color_active: Tuple[int, int, int],
        color_bg: Tuple[int, int, int],
        is_exit: bool = False,
    ) -> None:
        """Renders an animated glowing circular progress HUD ring with percentage and title."""
        radius = 50
        # Outer dark ring
        cv2.circle(frame, (cx, cy), radius, color_bg, 6, cv2.LINE_AA)
        # Active glowing arc
        angle = int(360.0 * progress)
        if angle > 0:
            cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle, color_active, 6, cv2.LINE_AA)
            cv2.ellipse(frame, (cx, cy), (radius + 4, radius + 4), -90, 0, angle, (255, 255, 255), 1, cv2.LINE_AA)

        # Center text / icon
        if is_exit:
            cv2.putText(frame, "[X]", (cx - 16, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 40, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{int(progress * 100)}%", (cx - 20, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"{int(progress * 100)}%", (cx - 20, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # Description label
        tw = len(label) * 8
        lx = max(10, min(frame.shape[1] - tw - 10, cx - tw // 2))
        ly = min(frame.shape[0] - 20, cy + radius + 24)
        cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_active, 2, cv2.LINE_AA)

    def _draw_cyber_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]) -> None:
        """Draws a futuristic HUD bounding box with corner reticles."""
        # Dashed bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # High-tech corner brackets
        corner_len = 24
        thick = 3
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thick)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thick)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thick)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thick)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thick)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thick)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thick)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thick)

    def _render_sculpting(self, frame: np.ndarray, hands: List[HandData]) -> None:
        """
        Renders the VIP Interactive 3D Spatial Sculpt & Assemble Instrument HUD.
        - Right Hand: Circle / Soundbox (Thùng đàn)
        - Left Hand: Rectangle / Fretboard & Neck (Cần đàn)
        - Both Ready: Cyber lightning arcs and magnetic snapping cues
        """
        ge = self.gesture_engine

        # 1. Right Hand: Circle / Soundbox (Thùng đàn)
        if ge.soundbox_progress > 0.05 or ge.soundbox_sculpted:
            cx, cy = ge.soundbox_center
            r = max(10, int(ge.soundbox_radius))
            is_locked = ge.soundbox_sculpted
            is_dragging = ge.is_dragging_soundbox

            if is_dragging:
                col_primary = (0, 255, 255)  # Bright Gold/Yellow
                col_glow = (0, 180, 220)
            elif is_locked:
                col_primary = (0, 240, 255)  # Frozen Cyan
                col_glow = (0, 150, 200)
            else:
                col_primary = (255, 0, 220)  # Magenta sculpting
                col_glow = (180, 0, 160)

            # Paint-like bounding box & compass crosshairs when actively sculpting
            if not is_locked:
                self._draw_cyber_box(frame, cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8, (140, 40, 160))
                # Compass radial dashes
                for angle_deg in range(0, 360, 45):
                    rad = np.radians(angle_deg)
                    p_in = (int(cx + (r - 6) * np.cos(rad)), int(cy + (r - 6) * np.sin(rad)))
                    p_out = (int(cx + (r + 8) * np.cos(rad)), int(cy + (r + 8) * np.sin(rad)))
                    cv2.line(frame, p_in, p_out, col_primary, 1, cv2.LINE_AA)

            # Outer glow and crisp concentric rings
            cv2.circle(frame, (cx, cy), r, col_glow, 4, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), r, col_primary, 2, cv2.LINE_AA)

            # Center black soundhole rosette (Tâm màu đen - Điểm chạm bắt buộc để kéo)
            r_black = max(16, int(r * 0.38))
            if is_dragging:
                cv2.circle(frame, (cx, cy), r_black, (25, 20, 32), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), r_black, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, cy), r_black, (10, 8, 14), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), r_black, (0, 240, 255), 2, cv2.LINE_AA)
                # Crosshair marker inside black center
                cv2.drawMarker(frame, (cx, cy), (0, 240, 255), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
                cv2.putText(frame, "TAM DEN", (cx - 21, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 240, 255), 1, cv2.LINE_AA)

            # 1.5s Hold Progress Ring when sculpting
            if not is_locked and ge.soundbox_hold_progress > 0.0:
                arc_deg = int(360.0 * ge.soundbox_hold_progress)
                cv2.ellipse(frame, (cx, cy), (r + 12, r + 12), -90, 0, arc_deg, (0, 255, 255), 3, cv2.LINE_AA)

            # Draw visual grab tether from dual-touching fingers to shape center
            if is_dragging and ge._soundbox_drag_finger_id is not None:
                for hand in hands:
                    if hand.handedness == ge._soundbox_drag_finger_id:
                        p4 = hand.landmarks_px[4, :2].astype(int)
                        p8 = hand.landmarks_px[8, :2].astype(int)

                        # Dual-finger touch indicators
                        cv2.circle(frame, (p4[0], p4[1]), 10, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(frame, (p4[0], p4[1]), 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, "CAI", (p4[0] - 12, p4[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

                        cv2.circle(frame, (p8[0], p8[1]), 10, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(frame, (p8[0], p8[1]), 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, "TRO", (p8[0] - 12, p8[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

                        cv2.line(frame, (p4[0], p4[1]), (cx, cy), (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.line(frame, (p8[0], p8[1]), (cx, cy), (0, 255, 255), 2, cv2.LINE_AA)

                        fx, fy = int((p4[0] + p8[0]) / 2), int((p4[1] + p8[1]) / 2)
                        cv2.line(frame, (p4[0], p4[1]), (p8[0], p8[1]), (0, 200, 255), 1, cv2.LINE_AA)
                        cv2.circle(frame, (fx, fy), 6, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, "✌️ 2 NGON CHAM", (fx - 45, fy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
                        break

            # Dynamic Status Badge
            if is_dragging:
                badge = "[ ✌️ DANG CHAM 2 NGON KEO THUNG DAN ]"
            elif is_locked:
                badge = "[ THUNG DAN: DUNG YEN - CHAM DONG THOI NGON CAI & TRO DE KEO ]"
            elif ge.soundbox_hold_progress > 0.0:
                badge = f"NAN THUNG DAN: GIU YEN {int(ge.soundbox_hold_progress * 100)}% (1.5s)"
            else:
                badge = f"NAN THUNG DAN: {int(ge.soundbox_progress * 100)}%"

            text_size = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            bx = max(10, min(frame.shape[1] - text_size[0] - 20, cx - text_size[0] // 2))
            by = min(frame.shape[0] - 25, cy + r + 24)
            cv2.rectangle(frame, (bx - 4, by - 16), (bx + text_size[0] + 8, by + 6), (15, 10, 20), -1)
            cv2.rectangle(frame, (bx - 4, by - 16), (bx + text_size[0] + 8, by + 6), col_primary, 1)
            cv2.putText(frame, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col_primary, 1, cv2.LINE_AA)

        # 2. Both Hands: Rectangle / Fretboard & Neck (Cần đàn)
        if ge._neck_touch_primed and not ge.neck_sculpted and ge.neck_progress <= 0.05:
            # Two hands just touched! Show glowing contact point
            tx, ty = ge._neck_touch_center
            pulse_r = int(18 + 6 * np.sin(time.perf_counter() * 12.0))
            cv2.circle(frame, (tx, ty), pulse_r, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                "✨ DA CHAM 2 TAY! KEO RA DE TAO CAN DAN ✨",
                (max(20, tx - 190), max(30, ty - 26)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if ge.neck_progress > 0.05 or ge.neck_sculpted:
            cx, cy = ge.neck_center
            w_rect = max(20, int(ge.neck_width))
            h_rect = max(15, int(ge.neck_height))
            is_locked = ge.neck_sculpted
            is_dragging = ge.is_dragging_neck

            if is_dragging:
                col_primary = (50, 255, 200)  # Bright Turquoise / Cyan-Green
                col_glow = (0, 200, 140)
            elif is_locked:
                col_primary = (0, 255, 140)   # Frozen Emerald
                col_glow = (0, 160, 90)
            else:
                col_primary = (0, 240, 255)   # Cyan sculpting
                col_glow = (0, 140, 180)

            rx1, ry1 = cx - w_rect // 2, cy - h_rect // 2
            rx2, ry2 = cx + w_rect // 2, cy + h_rect // 2

            # Render 2 long laser guide lines when pulling hands apart
            # User requirement: "đoạn ngón tay tro và ngón tay cái 2 bàn tay chạm vào nhau phải có 2 đường thẳng dài khi kéo ra để tạo hình chữ nhật"
            if not is_locked and ge.neck_line_upper is not None and ge.neck_line_lower is not None:
                u1, u2 = ge.neck_line_upper
                d1, d2 = ge.neck_line_lower

                # Rail 1 (Upper rail - connecting Index fingertips)
                cv2.line(frame, u1, u2, (0, 140, 255), 6, cv2.LINE_AA)
                cv2.line(frame, u1, u2, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, u1, u2, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, u1, 7, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, u2, 7, (0, 255, 255), -1, cv2.LINE_AA)

                # Rail 2 (Lower rail - connecting Thumb fingertips)
                cv2.line(frame, d1, d2, (0, 140, 255), 6, cv2.LINE_AA)
                cv2.line(frame, d1, d2, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, d1, d2, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, d1, 7, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, d2, 7, (0, 255, 255), -1, cv2.LINE_AA)

                # Dynamic frets across the 2 rails
                u1_arr, u2_arr = np.array(u1, dtype=float), np.array(u2, dtype=float)
                d1_arr, d2_arr = np.array(d1, dtype=float), np.array(d2, dtype=float)
                num_f = max(3, int(w_rect / 24))
                u_steps = np.linspace(0.0, 1.0, num_f)
                for step in u_steps:
                    pt_top = tuple((u1_arr + step * (u2_arr - u1_arr)).astype(int))
                    pt_bot = tuple((d1_arr + step * (d2_arr - d1_arr)).astype(int))
                    cv2.line(frame, pt_top, pt_bot, (180, 220, 245), 1, cv2.LINE_AA)

            # Outer glow and border
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), col_glow, 3, cv2.LINE_AA)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), col_primary, 1, cv2.LINE_AA)

            # Internal fret divisions
            num_frets = max(3, int(w_rect / 22))
            fret_xs = np.linspace(rx1 + 8, rx2 - 8, num_frets)
            for fx in fret_xs:
                cv2.line(frame, (int(fx), ry1 + 2), (int(fx), ry2 - 2), (180, 210, 230), 1, cv2.LINE_AA)

            # Headstock notch on the left side
            cv2.rectangle(frame, (rx1 - 18, ry1 - 4), (rx1, ry2 + 4), col_glow, 1, cv2.LINE_AA)

            # 1.5s Hold Progress Bar when sculpting
            if not is_locked and ge.neck_hold_progress > 0.0:
                fill_len = int(w_rect * ge.neck_hold_progress)
                cv2.line(frame, (rx1, ry2 + 5), (rx1 + fill_len, ry2 + 5), (0, 255, 255), 3, cv2.LINE_AA)

            # Draw visual grab tether from dual-touching fingers to neck center
            if is_dragging and ge._neck_drag_finger_id is not None:
                for hand in hands:
                    if hand.handedness == ge._neck_drag_finger_id:
                        p4 = hand.landmarks_px[4, :2].astype(int)
                        p8 = hand.landmarks_px[8, :2].astype(int)

                        cv2.circle(frame, (p4[0], p4[1]), 10, (50, 255, 200), 2, cv2.LINE_AA)
                        cv2.circle(frame, (p4[0], p4[1]), 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, "CAI", (p4[0] - 12, p4[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 255, 200), 1, cv2.LINE_AA)

                        cv2.circle(frame, (p8[0], p8[1]), 10, (50, 255, 200), 2, cv2.LINE_AA)
                        cv2.circle(frame, (p8[0], p8[1]), 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, "TRO", (p8[0] - 12, p8[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 255, 200), 1, cv2.LINE_AA)

                        cv2.line(frame, (p4[0], p4[1]), (cx, cy), (50, 255, 200), 2, cv2.LINE_AA)
                        cv2.line(frame, (p8[0], p8[1]), (cx, cy), (50, 255, 200), 2, cv2.LINE_AA)

                        fx, fy = int((p4[0] + p8[0]) / 2), int((p4[1] + p8[1]) / 2)
                        cv2.line(frame, (p4[0], p4[1]), (p8[0], p8[1]), (100, 255, 180), 1, cv2.LINE_AA)
                        cv2.circle(frame, (fx, fy), 6, (50, 255, 200), -1, cv2.LINE_AA)
                        cv2.putText(frame, "✌️ 2 NGON CHAM", (fx - 45, fy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (50, 255, 200), 1, cv2.LINE_AA)
                        break

            # Dynamic Status Badge
            if is_dragging:
                badge = "[ ✌️ DANG CHAM 2 NGON KEO CAN DAN ]"
            elif is_locked:
                badge = "[ CAN DAN: DUNG YEN - CHAM DONG THOI NGON CAI & TRO DE KEO ]"
            elif ge.neck_hold_progress > 0.0:
                badge = f"NAN CAN DAN: GIU YEN {int(ge.neck_hold_progress * 100)}% (1.5s)"
            else:
                badge = f"NAN CAN DAN (2 TAY KEO RA): {int(ge.neck_progress * 100)}%"

            text_size = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            bx = max(10, min(frame.shape[1] - text_size[0] - 20, cx - text_size[0] // 2))
            by = min(frame.shape[0] - 25, ry2 + 24)
            cv2.rectangle(frame, (bx - 4, by - 16), (bx + text_size[0] + 8, by + 6), (15, 10, 20), -1)
            cv2.rectangle(frame, (bx - 4, by - 16), (bx + text_size[0] + 8, by + 6), col_primary, 1)
            cv2.putText(frame, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col_primary, 1, cv2.LINE_AA)

        # Screen Overflow Disappearance Visual Alert
        if time.perf_counter() - self.last_overflow_time < 2.0:
            banner = f"⚠️ {self.overflow_msg} ⚠️"
            t_sz = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0]
            ox = max(20, (frame.shape[1] - t_sz[0]) // 2)
            oy = int(frame.shape[0] * 0.42)
            cv2.rectangle(frame, (ox - 10, oy - 26), (ox + t_sz[0] + 10, oy + 10), (15, 10, 35), -1)
            cv2.rectangle(frame, (ox - 10, oy - 26), (ox + t_sz[0] + 10, oy + 10), (0, 140, 255), 2)
            cv2.putText(frame, banner, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2, cv2.LINE_AA)

        # 3. Ready to Assemble: Magnetic Attraction Lightning & Center Guidance
        if ge.state == AppState.READY_TO_ASSEMBLE:
            nc = ge.neck_center
            sc = ge.soundbox_center
            mag = ge.magnetic_attraction

            if mag > 0.0:
                self._draw_lightning(frame, nc, sc, mag)

            # Pulsing central guidance text
            pulse = int(190 + 65 * np.sin(time.perf_counter() * 8.0))
            col_pulse = (0, pulse, 255)
            d_px = int(ge.assemble_distance)

            hud_y = 105
            cv2.putText(
                frame,
                f"⚡ DUA 2 HINH LAI GAN (<145px) DE GHEP DAN! (KHOANG CACH: {d_px}px)",
                (frame.shape[1] // 2 - 320, hud_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                col_pulse,
                2,
                cv2.LINE_AA,
            )

    def _draw_lightning(self, frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], intensity: float) -> None:
        """Renders dynamic jagged electric cyber lightning arcs between two points."""
        p1 = np.array(pt1, dtype=np.float32)
        p2 = np.array(pt2, dtype=np.float32)
        diff = p2 - p1
        dist = float(np.linalg.norm(diff))
        if dist < 1.0:
            return

        unit_tan = diff / dist
        unit_norm = np.array([-unit_tan[1], unit_tan[0]], dtype=np.float32)

        # Draw 2 distinct jittered arcs
        for _ in range(2):
            num_segs = max(4, int(dist / 18.0))
            u_steps = np.linspace(0.0, 1.0, num_segs)
            pts = []
            for u in u_steps:
                if u == 0.0 or u == 1.0:
                    pts.append((p1 + u * diff).astype(np.int32))
                else:
                    jitter = float(np.random.uniform(-18.0, 18.0)) * intensity
                    pt = p1 + u * diff + jitter * unit_norm
                    pts.append(pt.astype(np.int32))

            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(frame, [pts_arr], isClosed=False, color=(0, 240, 255), thickness=3, lineType=cv2.LINE_AA)
            cv2.polylines(frame, [pts_arr], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # Scatter a few electric sparks along the connection
        for _ in range(3):
            sp_u = float(np.random.uniform(0.1, 0.9))
            sp_off = float(np.random.uniform(-14.0, 14.0)) * intensity
            sp_pt = p1 + sp_u * diff + sp_off * unit_norm
            cv2.circle(frame, (int(sp_pt[0]), int(sp_pt[1])), 3, (0, 255, 255), -1, cv2.LINE_AA)

    def _draw_shockwave(self, frame: np.ndarray, center: Tuple[int, int], progress: float) -> None:
        """Renders an expanding shockwave pulse and particle burst upon Snap Fusion."""
        cx, cy = center
        h, w, _ = frame.shape
        radius = int(20 + progress * 260)

        # 1. Concentric expanding cyber energy rings
        cv2.circle(frame, (cx, cy), radius, (0, 240, 255), 5, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(6, radius - 16), (255, 0, 220), 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(2, radius - 32), (255, 255, 255), 2, cv2.LINE_AA)

        # 2. Radiating energy particles
        num_sparks = 18
        for angle in np.linspace(0, 2 * np.pi, num_sparks)[:-1]:
            dist_p = radius * 1.12
            px = int(cx + dist_p * np.cos(angle))
            py = int(cy + dist_p * np.sin(angle))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(frame, (px, py), 4, (0, 255, 255), -1, cv2.LINE_AA)

        # 3. Screen flash on initial impact
        if progress < 0.35:
            alpha_flash = float(0.35 * (1.0 - progress / 0.35))
            flash_overlay = frame.copy()
            flash_overlay[:] = (200, 240, 255)
            cv2.addWeighted(flash_overlay, alpha_flash, frame, 1.0 - alpha_flash, 0, frame)

        # 4. Success text
        cv2.putText(
            frame,
            "⚡ SNAP FUSION! GHEP DAN THANH CONG! ⚡",
            (w // 2 - 270, min(h - 80, cy - radius - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def shutdown(self) -> None:
        """Clean resource release for camera, hand tracker, audio stream, and UI."""
        logger.info("Shutting down Gesture AR Instruments suite...")
        self.camera.stop()
        self.hand_tracker.close()
        self.audio_engine.stop()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture AR Instruments - Virtual Piano & Guitar")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Display frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Display frame height (default: 720)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = GestureARApp(
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
    )
    app.run()


if __name__ == "__main__":
    main()
