"""
main.py
=======
Main Orchestrator & Cyber HUD Visualization for Gesture AR Instruments.

Integrates:
- ThreadedCamera: Non-blocking video capture thread with sequence IDs and timestamps.
- AsyncHandTracker: Dedicated worker thread with pluggable BaseFilter (OneEuroFilter default)
  and predictive kinematic dead-reckoning extrapolation targeting 60 FPS interaction.
- GestureEngine: Spatial gestures, state transitions, progress timers, and reset zones.
- Instruments: Virtual Piano (2 octaves, black-key priority, downward velocity gating, hover suppression)
               and Virtual Guitar (authentic 9 chords, muted string suppression, 2D line strumming).
- AudioEngine: Non-blocking procedural additive and plucked string synthesis with soft limiting.

Controls:
- Both Hands 'L' Shape: Spawn & Lock Piano to desk surface (hold 1.2s).
- Thumb & Index Touch + Pull Apart: Sculpt virtual guitar neck rails; sculpt soundbox and snap together.
- Fretboard Hotkeys 1-9 or C, G, D, A, E, F: Select authentic guitar chords.
- F3, Tab, or ` : Toggle Developer Diagnostics HUD.
- V: Cycle Visual Quality Profiles (HIGH, BALANCED, LOW).
- K: Cycle Hand Tracking Filters (1-Euro -> Deadband -> EMA -> Raw).
- Hold [RESET] Button (0.7s) or press 'r': Reset to IDLE.
- Hold [EXIT] Button (3.0s), 'q', or ESC: Graceful exit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import sys
import time
from types import MappingProxyType
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from audio_engine import AudioEngine
from gesture_engine import AppState, GestureEngine
from instruments import Guitar, Piano
from vision_tracker import AsyncHandTracker, HandData, ThreadedCamera

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MainApp")

# Module-level immutable guitar keyboard chord mapping (Key code -> Chord Name)
GUITAR_KEY_CHORD_MAP: MappingProxyType[int, str] = MappingProxyType({
    ord("1"): "C", ord("c"): "C", ord("C"): "C",
    ord("2"): "G", ord("g"): "G", ord("G"): "G",
    ord("3"): "D", ord("d"): "D", ord("D"): "D",
    ord("4"): "A", ord("a"): "A", ord("A"): "A",
    ord("5"): "E", ord("e"): "E", ord("E"): "E",
    ord("6"): "Am",
    ord("7"): "Em",
    ord("8"): "Dm",
    ord("9"): "F", ord("f"): "F", ord("F"): "F",
})

# Filter cycle keyboard shortcuts
FILTER_CYCLE_KEYS: Tuple[int, ...] = (ord("k"), ord("K"))

# Intended backend-specific special full key codes (cv2.waitKeyEx)
ESCAPE_FULL_KEYS: Tuple[int, ...] = (27, 0xFF1B)
TAB_FULL_KEYS: Tuple[int, ...] = (9, 0xFF09)

# Legitimate F3 platform-specific raw key codes from OpenCV waitKeyEx
# Windows: 0x720000 (VK_F3 << 16); Linux/X11: 65472 (0xFFC0)
# NOTE: ASCII 114 is deliberately excluded because 114 == ord("r")
F3_FULL_KEYS: Tuple[int, ...] = (0x720000, 65472)
F3_RAW_KEYS: Tuple[int, ...] = F3_FULL_KEYS

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



@dataclass(frozen=True)
class QualityProfileConfig:
    """Explicit performance and visual quality profile configuration."""

    name: str
    model_complexity: int
    max_particles: int
    enable_glow_effects: bool
    enable_shockwave_flash: bool
    oscilloscope_points: int


QUALITY_PROFILES: Dict[str, QualityProfileConfig] = {
    "HIGH": QualityProfileConfig(
        name="HIGH",
        model_complexity=1,
        max_particles=18,
        enable_glow_effects=True,
        enable_shockwave_flash=True,
        oscilloscope_points=32,
    ),
    "BALANCED": QualityProfileConfig(
        name="BALANCED",
        model_complexity=0,
        max_particles=8,
        enable_glow_effects=True,
        enable_shockwave_flash=False,
        oscilloscope_points=24,
    ),
    "LOW": QualityProfileConfig(
        name="LOW",
        model_complexity=0,
        max_particles=0,
        enable_glow_effects=False,
        enable_shockwave_flash=False,
        oscilloscope_points=12,
    ),
}


class GestureARApp:
    """AR Instruments Suite Orchestrator."""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        audio_engine: Optional[AudioEngine] = None,
        camera: Optional[ThreadedCamera] = None,
        tracker: Optional[AsyncHandTracker] = None,
        start_threads: bool = True,
        quality_profile: str = "HIGH",
        init_mediapipe: bool = True,
        camera_backend: str = "auto",
    ) -> None:
        self.width = width
        self.height = height

        # 1. Initialize Subsystems (supports dependency injection for headless testing)
        if audio_engine is not None:
            self.audio_engine = audio_engine
        else:
            logger.info("Initializing Audio Engine...")
            self.audio_engine = AudioEngine(sample_rate=44100, block_size=256)
            if start_threads:
                self.audio_engine.start()

        if camera is not None:
            self.camera = camera
        else:
            logger.info("Initializing Threaded Camera on source %s (Target: %dx%d, backend: %s)...", camera_id, self.width, self.height, camera_backend)
            self.camera = ThreadedCamera(src=camera_id, width=self.width, height=self.height, backend=camera_backend)
            if start_threads:
                self.camera.start()

        if tracker is not None:
            self.async_tracker = tracker
        else:
            logger.info("Initializing Asynchronous AI Hand Tracker Worker...")
            self.async_tracker = AsyncHandTracker(
                camera=self.camera,
                max_num_hands=2,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.50,
                ema_alpha=0.65,
                filter_mode="one_euro",
                model_complexity=1,
                init_mediapipe=init_mediapipe,
            )
            if start_threads:
                self.async_tracker.start()
        # Maintain hand_tracker alias for seamless compatibility
        self.hand_tracker = self.async_tracker

        logger.info("Initializing Gesture Engine...")
        self.gesture_engine = GestureEngine()

        # Active Instruments
        self.piano: Optional[Piano] = None
        self.guitar: Optional[Guitar] = None

        # Telemetry & Developer Diagnostics HUD
        self.show_diagnostics: bool = False
        self.set_quality_profile(quality_profile)
        self.telemetry = {
            "cam_ms": 0.0,
            "track_ms": 0.0,
            "gest_ms": 0.0,
            "inst_ms": 0.0,
            "rend_ms": 0.0,
            "total_ms": 0.0,
        }

        # FPS metrics
        self.prev_frame_time = time.perf_counter()
        self.fps = 0.0

        # Screen overflow auto-disappearance alert
        self.last_overflow_time: float = 0.0
        self.overflow_msg: str = ""

    def _on_tracking_pipeline_changed(self) -> None:
        """
        Resets active instrument motion-collision state when an application-level
        tracking pipeline switch occurs (filter mode, model complexity, or quality profile).
        Preserves currently selected chord and instrument state.
        """
        if hasattr(self, "piano") and self.piano is not None:
            self.piano.reset_motion_state()
        if hasattr(self, "guitar") and self.guitar is not None:
            self.guitar.reset_motion_state()

    def set_quality_profile(self, profile: str) -> None:
        """Applies explicit quality profile (HIGH, BALANCED, LOW)."""
        key = profile.upper()
        if key not in QUALITY_PROFILES:
            key = "HIGH"
        self.quality_profile = key
        self.quality_config = QUALITY_PROFILES[key]
        if hasattr(self, "async_tracker") and self.async_tracker is not None:
            changed = self.async_tracker.set_model_complexity(self.quality_config.model_complexity)
            if changed:
                self._on_tracking_pipeline_changed()

    def handle_key(self, raw_key: int) -> Optional[str]:
        """
        Deterministic keyboard event handler for GestureARApp without requiring GUI.
        Eliminates arbitrary low-byte masking to prevent extended key collisions.

        Args:
            raw_key: Full integer key code returned by cv2.waitKeyEx (or -1 if no key).

        Returns:
            Dispatched action string (e.g. 'EXIT', 'RESET', 'DIAGNOSTICS', etc.) or None.
        """
        if raw_key < 0:
            return None

        # 1. Backend-specific explicit special key allowlists
        if raw_key in ESCAPE_FULL_KEYS:
            logger.info("Exit key pressed (%s).", raw_key)
            return "EXIT"

        if raw_key in F3_FULL_KEYS or raw_key in TAB_FULL_KEYS:
            self.show_diagnostics = not self.show_diagnostics
            logger.info("Diagnostics HUD toggled: %s", "ON" if self.show_diagnostics else "OFF")
            return "DIAGNOSTICS"

        # Reject any non-ASCII full key codes that are not in explicit special key allowlists
        if raw_key > 0xFF:
            return None

        key = raw_key

        # 2. Exit application (q/Q)
        if key in (ord("q"), ord("Q")):
            logger.info("Exit key pressed (%s).", raw_key)
            return "EXIT"

        # 3. Developer Diagnostics HUD (backtick=`/96)
        if key == ord("`"):
            self.show_diagnostics = not self.show_diagnostics
            logger.info("Diagnostics HUD toggled: %s", "ON" if self.show_diagnostics else "OFF")
            return "DIAGNOSTICS"

        # 4. Manual Reset Shortcut (r/R) -> reset to IDLE
        if key in (ord("r"), ord("R")):
            self.gesture_engine.reset_to_idle()
            self.piano = None
            self.guitar = None
            logger.info("Application state reset to IDLE via [R].")
            return "RESET"

        # 5. Visual Quality Profile (v/V)
        if key in (ord("v"), ord("V")):
            if self.quality_profile == "HIGH":
                self.set_quality_profile("BALANCED")
            elif self.quality_profile == "BALANCED":
                self.set_quality_profile("LOW")
            else:
                self.set_quality_profile("HIGH")
            logger.info("Visual Quality Profile switched to: %s", self.quality_profile)
            return "QUALITY"

        # 6. AI Model Complexity Toggle (m/M): ULTRA (1) vs HYPER-SPEED (0)
        if key in (ord("m"), ord("M")):
            new_mc = 0 if self.async_tracker.model_complexity == 1 else 1
            changed = self.async_tracker.set_model_complexity(new_mc)
            if changed:
                self._on_tracking_pipeline_changed()
            mode_lbl = "ULTRA (Model 1: High Precision)" if new_mc == 1 else "HYPER-SPEED (Model 0: Lowest Latency)"
            logger.info("AI tracking model switched to: %s", mode_lbl)
            return "MODEL_COMPLEXITY"

        # 7. Hand Tracking Filter Mode Cycle (k/K)
        if key in FILTER_CYCLE_KEYS:
            cur_mode = self.hand_tracker.filter_mode.lower()
            if cur_mode == "one_euro":
                target_mode = "deadband"
            elif cur_mode in ("deadband", "zero_lag"):
                target_mode = "ema"
            elif cur_mode == "ema":
                target_mode = "raw"
            else:
                target_mode = "one_euro"
            changed = self.async_tracker.set_filter_mode(target_mode)
            if changed:
                self._on_tracking_pipeline_changed()
            logger.info("Hand tracking switched to %s", target_mode)
            return "FILTER"

        # 8. Hardware Camera Properties Dialog (p/P)
        if key in (ord("p"), ord("P")):
            logger.info("Requesting hardware camera properties dialog [P]...")
            opened = self.camera.open_settings_dialog()
            if not opened:
                logger.info(
                    "Camera settings dialog [P] unavailable. "
                    "On Windows, launch with '--camera-backend dshow' to enable hardware camera controls."
                )
            return "CAMERA_SETTINGS"

        # 9. Guitar Chord Selection (Keys 1-9, C, G, D, A, E, F)
        if self.guitar is not None and key in GUITAR_KEY_CHORD_MAP:
            ch = GUITAR_KEY_CHORD_MAP[key]
            self.guitar.set_chord(ch)
            logger.info("Guitar chord set to %s via keyboard hotkey", ch)
            return "GUITAR_CHORD"

        return None

    def dispatch_key(self, raw_key: int) -> bool:
        """Dispatches keyboard event; returns False if application should terminate, True otherwise."""
        action = self.handle_key(raw_key)
        return action != "EXIT"

    def run(self) -> None:
        """Main application lifecycle loop."""
        logger.info("Starting Gesture AR Instruments suite. Press 'q' or 'ESC' to exit.")
        window_name = "Interactive Spatial AR Instruments - Cyber HUD"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)
        window_configured = False

        try:
            while True:
                loop_start = time.perf_counter()

                # 1. Grab raw frame from ThreadedCamera
                t_cam_start = time.perf_counter()
                ret, raw_bgr_frame = self.camera.read()
                t_cam_end = time.perf_counter()
                if not ret or raw_bgr_frame is None:
                    time.sleep(0.005)
                    continue

                cam_time_ms = (t_cam_end - t_cam_start) * 1000.0
                self.telemetry["cam_ms"] = 0.90 * self.telemetry["cam_ms"] + 0.10 * cam_time_ms

                # Auto-detect camera resolution on first received frame
                if not window_configured:
                    window_configured = True
                    act_h, act_w = raw_bgr_frame.shape[:2]
                    self.width, self.height = act_w, act_h
                    cv2.resizeWindow(window_name, self.width, self.height)
                    logger.info("Window synchronized to native camera resolution: %dx%d", self.width, self.height)

                # 2. Retrieve Latest Hand Tracking with Predictive Kinematic Dead-Reckoning
                t_track_start = time.perf_counter()
                hands: List[HandData] = self.async_tracker.get_latest_hands(loop_start, extrapolate=True)
                t_track_end = time.perf_counter()
                track_time_ms = (t_track_end - t_track_start) * 1000.0
                self.telemetry["track_ms"] = 0.90 * self.telemetry["track_ms"] + 0.10 * track_time_ms

                # 3. Mirror display frame horizontally for mirror-like AR experience
                display_frame = cv2.flip(raw_bgr_frame, 1)

                # 4. Update Gesture State Machine
                t_gest_start = time.perf_counter()
                prev_state = self.gesture_engine.state
                self.gesture_engine.update(hands, display_frame.shape)
                cur_state = self.gesture_engine.state
                t_gest_end = time.perf_counter()
                gest_time_ms = (t_gest_end - t_gest_start) * 1000.0
                self.telemetry["gest_ms"] = 0.90 * self.telemetry["gest_ms"] + 0.10 * gest_time_ms

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
                    self.overflow_msg = "SOUNDBOX EXCEEDED SCREEN BOUNDS! SHAPE AUTO-PURGED"
                elif self.gesture_engine.neck_overflow_just_occurred:
                    self.last_overflow_time = time.perf_counter()
                    self.overflow_msg = "GUITAR NECK EXCEEDED SCREEN BOUNDS! SHAPE AUTO-PURGED"

                # 4b. Check Application Exit ([EXIT] button held for 3.0s)
                if self.gesture_engine.should_exit:
                    logger.info("Application exit confirmed: Shutting down cleanly.")
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
                t_inst_start = time.perf_counter()
                if cur_state == AppState.PIANO_ACTIVE and self.piano is not None:
                    self.piano.update(hands, display_frame.shape)
                elif cur_state == AppState.GUITAR_ACTIVE and self.guitar is not None:
                    self.guitar.update(hands, display_frame.shape)
                t_inst_end = time.perf_counter()
                inst_time_ms = (t_inst_end - t_inst_start) * 1000.0
                self.telemetry["inst_ms"] = 0.90 * self.telemetry["inst_ms"] + 0.10 * inst_time_ms

                # 7. Render Cyber HUD Overlay & Diagnostics
                t_rend_start = time.perf_counter()
                self._render_hud(display_frame, hands)
                if self.show_diagnostics:
                    self._render_diagnostics_hud(display_frame)
                t_rend_end = time.perf_counter()
                rend_time_ms = (t_rend_end - t_rend_start) * 1000.0
                self.telemetry["rend_ms"] = 0.90 * self.telemetry["rend_ms"] + 0.10 * rend_time_ms

                # 8. Render frame to window
                cv2.imshow(window_name, display_frame)

                # 9. Key Handling & OS Event Polling
                raw_key = cv2.waitKeyEx(1)

                # 10. Frame rate calculation & loop telemetry (measures complete pipeline including presentation)
                now = time.perf_counter()
                dt = now - self.prev_frame_time
                self.prev_frame_time = now
                if dt > 0:
                    current_fps = 1.0 / dt
                    self.fps = 0.9 * self.fps + 0.1 * current_fps
                total_frame_ms = (now - loop_start) * 1000.0
                self.telemetry["total_ms"] = 0.90 * self.telemetry["total_ms"] + 0.10 * total_frame_ms

                if not self.dispatch_key(raw_key):
                    break

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
            # Faint holographic guides for ergonomic desk piano & sculpt guitar
            overlay_guide = frame.copy()
            p_xmin, p_xmax = int(0.05 * w), int(0.95 * w)
            p_ymin, p_ymax = int(0.72 * h), int(0.96 * h)
            cv2.rectangle(overlay_guide, (p_xmin, p_ymin), (p_xmax, p_ymax), (45, 40, 50), 1)
            cv2.putText(overlay_guide, "DESK SURFACE PIANO ZONE (REST WRISTS TO PLAY)", (p_xmin + 14, p_ymin + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 85, 100), 1, cv2.LINE_AA)

            # Holographic guidance for sculpting and assembling guitar
            if self.gesture_engine.is_sculpt_locked:
                cv2.putText(overlay_guide, "[LOCKED] SCULPT CREATION LOCKED - HAND TRACKING RIG STILL ACTIVE", (int(0.18 * w), int(0.48 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 1, cv2.LINE_AA)
                cv2.putText(overlay_guide, "[HOLD [L] BUTTON (1s) AT TOP-RIGHT TO UNLOCK CREATION]", (int(0.24 * w), int(0.54 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(overlay_guide, "[PINCH] PINCH THUMB & INDEX (BOTH HANDS) TO SCULPT NECK", (int(0.06 * w), int(0.48 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 180), 1, cv2.LINE_AA)
                cv2.putText(overlay_guide, "[EXPAND] RIGHT HAND PINCH & EXPAND: SOUNDBOX", (int(0.56 * w), int(0.48 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 50, 180), 1, cv2.LINE_AA)
                cv2.putText(overlay_guide, "[FUSION] BRING BOTH PARTS CLOSE (<145px) TO ASSEMBLE GUITAR", (int(0.24 * w), int(0.54 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
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
            if self.quality_config.enable_glow_effects:
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

                if self.quality_config.enable_glow_effects:
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
            tag = "[L] LEFT HAND" if is_left else "[R] RIGHT HAND"
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
        """Draws top futuristic glassmorphic cyber header, telemetry status, and controls."""
        h, w = frame.shape[:2]
        now_t = time.perf_counter()

        # 1. Glassmorphic Top Header Bar Background
        header_h = 52
        top_roi = frame[0:header_h, 0:w]
        if top_roi.size > 0:
            overlay = np.full_like(top_roi, (14, 11, 20), dtype=np.uint8)
            cv2.addWeighted(overlay, 0.75, top_roi, 0.25, 0, top_roi)
            frame[0:header_h, 0:w] = top_roi

        # Neon Cyan Bottom Accent Line with subtle glow
        cv2.line(frame, (0, header_h), (w, header_h), (0, 240, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (0, header_h + 1), (w, header_h + 1), (0, 90, 110), 1, cv2.LINE_AA)

        # 2. Zone 1: Branding (Left)
        cv2.putText(
            frame,
            "GESTURE AR",
            (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (240, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "INSTRUMENTS",
            (16, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # Vertical cyber separator
        cv2.line(frame, (148, 12), (148, 42), (50, 45, 65), 1, cv2.LINE_AA)

        # 3. Zone 2: Dynamic State Capsule Badge (Center-Left)
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
        state_str = f"STATE: {state.value}"
        st_size = cv2.getTextSize(state_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]

        sx1 = 160
        sx2 = sx1 + st_size[0] + 32
        sy1, sy2 = 10, 44

        s_sub = frame[sy1:sy2, sx1:sx2]
        if s_sub.size > 0:
            s_overlay = np.full_like(s_sub, (22, 18, 28), dtype=np.uint8)
            cv2.addWeighted(s_overlay, 0.70, s_sub, 0.30, 0, s_sub)
            frame[sy1:sy2, sx1:sx2] = s_sub
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), badge_color, 1, cv2.LINE_AA)

        # Glowing animated status LED dot
        pulse_r = int(6 + 2 * np.sin(now_t * 6.0))
        cv2.circle(frame, (sx1 + 12, 27), pulse_r, badge_color, 1, cv2.LINE_AA)
        cv2.circle(frame, (sx1 + 12, 27), 4, badge_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            state_str,
            (sx1 + 22, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            badge_color,
            1,
            cv2.LINE_AA,
        )

        # 4. Zone 3: Telemetry Capsule (Center) - Non-Overlapping Spacing
        ai_fps = getattr(self.async_tracker, "ai_fps", 0.0)
        model_tag = "ULTRA" if getattr(self.async_tracker, "model_complexity", 1) == 1 else "HYPER"
        filt_m = self.hand_tracker.filter_mode.lower()
        if filt_m == "one_euro":
            mode_tag = "1-EURO"
        elif filt_m in ("deadband", "zero_lag"):
            mode_tag = "DEADBAND"
        elif filt_m == "ema":
            mode_tag = "EMA"
        else:
            mode_tag = "RAW"
        q_tag = self.quality_profile[:3]

        btn_zone_start = w - 515
        space_left = sx2 + 18
        space_right = btn_zone_start - 16
        avail_w = space_right - space_left

        if avail_w >= 410:
            fps_text = f"FPS: {self.fps:.1f} | AI: {ai_fps:.1f} ({model_tag}) | {mode_tag} | Q:{q_tag} | H:{hand_count}"
        else:
            fps_text = f"{self.fps:.0f}FPS | AI:{ai_fps:.0f} | {mode_tag} | H:{hand_count}"

        telem_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        telem_w = telem_size[0]

        # Position telemetry with graceful fallback if space is tight
        if avail_w > telem_w:
            tx1 = space_left + (avail_w - telem_w) // 2 - 8
        else:
            tx1 = space_left

        tx2 = tx1 + telem_w + 16
        ty1, ty2 = 11, 43

        t_sub = frame[ty1:ty2, tx1:tx2]
        if t_sub.size > 0:
            t_overlay = np.full_like(t_sub, (18, 16, 26), dtype=np.uint8)
            cv2.addWeighted(t_overlay, 0.65, t_sub, 0.35, 0, t_sub)
            frame[ty1:ty2, tx1:tx2] = t_sub
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (65, 55, 85), 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            fps_text,
            (tx1 + 8, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 240, 255),
            1,
            cv2.LINE_AA,
        )

        # 4b. Real-Time Audio Oscilloscope Waveform Capsule
        # Rendered dynamically if space between telemetry and button zone permits (>= 105px)
        rem_space = btn_zone_start - (tx2 + 14)
        if rem_space >= 105:
            vx1 = tx2 + 14
            vw = min(130, rem_space - 10)
            self._draw_audio_oscilloscope(frame, vx1, ty1, vw, ty2 - ty1)

        # 5. Zone 4: Top-Right Glassmorphic Action Buttons
        # 5a. Lock Sculpt Button (Hold fingertip for 1.0s to Toggle Shape Creation Lock)
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

        # Text & Status inside Lock Sculpt Button
        if is_touching_lock:
            rem_l = max(0.0, 1.0 - (l_prog * 1.0))
            action_txt = "UNLOCK" if is_locked else "LOCK"
            l_text = f"{action_txt}: {rem_l:.1f}s ({int(l_prog * 100)}%)"
            cv2.putText(frame, l_text, (lx1 + 8, ly1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (lx2 - 12, ly1 + 12), 4, (0, 255, 200) if is_locked else (0, 180, 255), -1, cv2.LINE_AA)
        else:
            if is_locked:
                cv2.putText(frame, "[L] LOCKED (1s)", (lx1 + 12, ly1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 180, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (lx2 - 12, ly1 + 12), 4, (0, 160, 255), -1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "[L] LOCK SHAPE (1s)", (lx1 + 8, ly1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 180), 1, cv2.LINE_AA)
                cv2.circle(frame, (lx2 - 12, ly1 + 12), 4, (0, 240, 180), -1, cv2.LINE_AA)

        # 5b. Reset Button (Hold fingertip for 0.7s to Reset)
        rx1, ry1, rx2, ry2 = w - 345, 8, w - 185, 44
        r_prog = self.gesture_engine.reset_progress
        is_touching_reset = self.gesture_engine.is_touching_reset

        r_sub = frame[ry1:ry2, rx1:rx2]
        if r_sub.size > 0:
            r_overlay = r_sub.copy()
            if is_touching_reset:
                r_overlay[:] = (20, 35, 45)
                fill_w = int((rx2 - rx1) * r_prog)
                if fill_w > 0:
                    r_overlay[:, :fill_w] = (0, 180, 255)  # Glowing Cyber Gold / Amber
                cv2.addWeighted(r_overlay, 0.85, r_sub, 0.15, 0, r_sub)
                frame[ry1:ry2, rx1:rx2] = r_sub
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 220, 255), 2, cv2.LINE_AA)
            else:
                r_overlay[:] = (18, 22, 28)
                cv2.addWeighted(r_overlay, 0.70, r_sub, 0.30, 0, r_sub)
                frame[ry1:ry2, rx1:rx2] = r_sub
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 140, 180), 1, cv2.LINE_AA)

        # Text & Status inside Reset Button
        if is_touching_reset:
            rem_r = max(0.0, 0.70 - (r_prog * 0.70))
            r_text = f"RESET: {rem_r:.1f}s ({int(r_prog * 100)}%)"
            cv2.putText(frame, r_text, (rx1 + 10, ry1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (rx2 - 12, ry1 + 12), 4, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "[R] RESET (0.7s)", (rx1 + 16, ry1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

        # 5c. Exit Button (Hold fingertip for 3.0s to Exit)
        bx1, by1, bx2, by2 = w - 175, 8, w - 15, 44
        e_prog = self.gesture_engine.exit_progress
        is_touching = self.gesture_engine.is_touching_exit

        btn_sub = frame[by1:by2, bx1:bx2]
        if btn_sub.size > 0:
            btn_overlay = btn_sub.copy()
            if is_touching:
                btn_overlay[:] = (20, 10, 80)
                fill_w = int((bx2 - bx1) * e_prog)
                if fill_w > 0:
                    btn_overlay[:, :fill_w] = (0, 35, 210)  # Laser Red
                cv2.addWeighted(btn_overlay, 0.85, btn_sub, 0.15, 0, btn_sub)
                frame[by1:by2, bx1:bx2] = btn_sub
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 140, 255), 2, cv2.LINE_AA)
            else:
                btn_overlay[:] = (24, 16, 32)
                cv2.addWeighted(btn_overlay, 0.70, btn_sub, 0.30, 0, btn_sub)
                frame[by1:by2, bx1:bx2] = btn_sub
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60, 40, 160), 1, cv2.LINE_AA)

        # Text & Status inside Exit Button
        if is_touching:
            rem_time = max(0.0, 3.0 - (e_prog * 3.0))
            btn_text = f"EXIT: {rem_time:.1f}s ({int(e_prog * 100)}%)"
            cv2.putText(frame, btn_text, (bx1 + 8, by1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (bx2 - 12, by1 + 12), 4, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "[EXIT] HOLD 3s", (bx1 + 18, by1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 180, 220), 1, cv2.LINE_AA)

        # 6. Context-Aware English Instruction Banner (Bottom Dock)
        instruction_map = {
            AppState.IDLE: "SELECT INSTRUMENT: Both hands 'L' = DESK PIANO | Pull apart = SCULPT GUITAR | [L] LOCK (1s)",
            AppState.CREATING_PIANO: "LOCKING PIANO: Hold 'L' pose on surface for 1.2s...",
            AppState.PIANO_ACTIVE: "VIRTUAL PIANO: Tap keys on desk surface | Hold [RESET] (0.7s) to return to menu",
            AppState.CREATING_GUITAR: "LOCKING GUITAR: Hold left-hand 'O' pinch for 1.2s...",
            AppState.SCULPTING_GUITAR: "SCULPT: Both hands pinch & stretch = NECK (1.5s) | Right hand pinch = SOUNDBOX (1.5s)",
            AppState.READY_TO_ASSEMBLE: "ASSEMBLE: Dual-finger touch to drag shapes | Bring within 145px to SNAP FUSE!",
            AppState.FUSION_SNAP: "SNAP FUSION: Merging soundbox and neck into playable 3D Cyber Guitar...",
            AppState.GUITAR_ACTIVE: "GUITAR: Left hand 1-4 fingers switch chord | Right hand pluck with thumb/index | [RESET] (0.7s)",
        }
        inst_text = instruction_map.get(state, "")

        # Bottom instruction strip (compact 24px bar docked below piano)
        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (0, h - 24), (w, h), (10, 8, 14), -1)
        cv2.addWeighted(overlay_bottom, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, h - 24), (w, h - 24), (40, 35, 55), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            inst_text,
            (16, h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Quick: [L] LOCK | [R] RESET | [M] AI | [K] RIG | [V] QUAL | [F3/TAB] DIAG | [P] CAM | [Q/ESC] EXIT",
            (max(10, w - 575), h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    def _render_diagnostics_hud(self, frame: np.ndarray) -> None:
        """Renders developer diagnostics overlay with pipeline timing breakdown."""
        h, w = frame.shape[:2]
        panel_w = 430
        panel_h = 240
        px1 = 20
        py1 = h - panel_h - 36
        px2 = px1 + panel_w
        py2 = py1 + panel_h

        roi = frame[py1:py2, px1:px2]
        if roi.size > 0:
            overlay = np.full_like(roi, (14, 12, 20), dtype=np.uint8)
            cv2.addWeighted(overlay, 0.88, roi, 0.12, 0, roi)
            frame[py1:py2, px1:px2] = roi

        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 240, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (px1, py1), (px2, py1 + 24), (25, 20, 36), -1)
        cv2.line(frame, (px1, py1 + 24), (px2, py1 + 24), (0, 240, 255), 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            "// DIAGNOSTICS & TELEMETRY [F3 / TAB]",
            (px1 + 10, py1 + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 230),
            1,
            cv2.LINE_AA,
        )

        ai_fps = getattr(self.async_tracker, "ai_fps", 0.0)
        ai_lat = getattr(self.async_tracker, "inference_latency_ms", 0.0)
        stale_frames = getattr(self.async_tracker, "stale_frames_skipped", 0)
        voices = self.audio_engine.get_active_voice_count()
        peak = getattr(self.audio_engine, "current_peak", 0.0) * 100.0

        lines = [
            f"Render Loop FPS: {self.fps:.1f} FPS  (Total: {self.telemetry['total_ms']:.1f} ms)",
            f"Camera HW FPS:   {self.camera.camera_fps:.1f} FPS  (Snapshot Access: {self.telemetry['cam_ms']:.1f} ms)",
            f"AI Inference:    {ai_fps:.1f} FPS  (Lat: {ai_lat:.1f} ms, Stale Skip: {stale_frames})",
            f"Active Pipeline: Gesture {self.telemetry['gest_ms']:.1f}ms | Render {self.telemetry['rend_ms']:.1f}ms",
            f"Audio Bus:       {voices} active voices | Peak: {peak:.1f}% (Soft Limiter: Active)",
            f"Engine Config:   Filter={self.hand_tracker.filter_mode.upper()} | Quality={self.quality_profile}",
        ]

        for idx, line in enumerate(lines):
            y_text = py1 + 46 + idx * 22
            cv2.putText(frame, line, (px1 + 12, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 225, 240), 1, cv2.LINE_AA)

        # Proportional Timing Bar
        bar_x = px1 + 12
        bar_y = py2 - 34
        bar_w = panel_w - 24
        bar_h = 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 30, 45), -1)

        t_total = max(1.0, self.telemetry["total_ms"])
        w_cam = int((self.telemetry["cam_ms"] / t_total) * bar_w)
        w_ai = int((self.telemetry["track_ms"] / t_total) * bar_w)
        w_gest = int((self.telemetry["gest_ms"] / t_total) * bar_w)
        w_rend = int((self.telemetry["rend_ms"] / t_total) * bar_w)

        cur_bx = bar_x
        # Cam segment (Cyan)
        if w_cam > 0:
            cv2.rectangle(frame, (cur_bx, bar_y), (min(bar_x + bar_w, cur_bx + w_cam), bar_y + bar_h), (0, 220, 255), -1)
            cur_bx += w_cam
        # Track snapshot segment (Magenta)
        if w_ai > 0:
            cv2.rectangle(frame, (cur_bx, bar_y), (min(bar_x + bar_w, cur_bx + w_ai), bar_y + bar_h), (255, 0, 220), -1)
            cur_bx += w_ai
        # Gesture segment (Yellow)
        if w_gest > 0:
            cv2.rectangle(frame, (cur_bx, bar_y), (min(bar_x + bar_w, cur_bx + w_gest), bar_y + bar_h), (0, 220, 255), -1)
            cur_bx += w_gest
        # Render segment (Green)
        if w_rend > 0:
            cv2.rectangle(frame, (cur_bx, bar_y), (min(bar_x + bar_w, cur_bx + w_rend), bar_y + bar_h), (0, 255, 140), -1)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Cam(Cyan) | Snap(Mag) | Gest(Yel) | Rend(Grn)",
            (bar_x, py2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (160, 180, 200),
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
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thick)
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
                cv2.putText(frame, "CORE", (cx - 15, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 240, 255), 1, cv2.LINE_AA)

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
                        cv2.putText(frame, "THUMB", (p4[0] - 14, p4[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

                        cv2.circle(frame, (p8[0], p8[1]), 10, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(frame, (p8[0], p8[1]), 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, "INDEX", (p8[0] - 14, p8[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

                        cv2.line(frame, (p4[0], p4[1]), (cx, cy), (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.line(frame, (p8[0], p8[1]), (cx, cy), (0, 255, 255), 2, cv2.LINE_AA)

                        fx, fy = int((p4[0] + p8[0]) / 2), int((p4[1] + p8[1]) / 2)
                        cv2.line(frame, (p4[0], p4[1]), (p8[0], p8[1]), (0, 200, 255), 1, cv2.LINE_AA)
                        cv2.circle(frame, (fx, fy), 6, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, "[DUAL-TOUCH]", (fx - 46, fy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
                        break

            # Dynamic Glassmorphic Status Badge
            if is_dragging:
                badge = "[ DUAL-TOUCH: DRAGGING SOUNDBOX ]"
            elif is_locked:
                badge = "[ SOUNDBOX LOCKED: TOUCH THUMB & INDEX TO DRAG ]"
            elif ge.soundbox_hold_progress > 0.0:
                badge = f"SCULPT SOUNDBOX: HOLD STEADY {int(ge.soundbox_hold_progress * 100)}% (1.5s)"
            else:
                badge = f"SCULPT SOUNDBOX: {int(ge.soundbox_progress * 100)}%"

            text_size = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            bx = max(10, min(frame.shape[1] - text_size[0] - 20, cx - text_size[0] // 2))
            by = min(frame.shape[0] - 25, cy + r + 24)

            # Translucent glassmorphic backing for status badge
            sub_b = frame[by - 16:by + 6, bx - 4:bx + text_size[0] + 8]
            if sub_b.size > 0:
                ov_b = np.full_like(sub_b, (16, 12, 24))
                cv2.addWeighted(ov_b, 0.75, sub_b, 0.25, 0, sub_b)
                frame[by - 16:by + 6, bx - 4:bx + text_size[0] + 8] = sub_b
            cv2.rectangle(frame, (bx - 4, by - 16), (bx + text_size[0] + 8, by + 6), col_primary, 1, cv2.LINE_AA)
            cv2.putText(frame, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col_primary, 1, cv2.LINE_AA)

        # 2. Both Hands: Rectangle / Fretboard & Neck (Can dan)
        if ge._neck_touch_primed and not ge.neck_sculpted and ge.neck_progress <= 0.05:
            # Two hands just touched! Show glowing contact point
            tx, ty = ge._neck_touch_center
            pulse_r = int(18 + 6 * np.sin(time.perf_counter() * 12.0))
            cv2.circle(frame, (tx, ty), pulse_r, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                ">> CONTACT DETECTED! PULL APART TO SCULPT NECK <<",
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
                        cv2.putText(frame, "THUMB", (p4[0] - 14, p4[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 255, 200), 1, cv2.LINE_AA)

                        cv2.circle(frame, (p8[0], p8[1]), 10, (50, 255, 200), 2, cv2.LINE_AA)
                        cv2.circle(frame, (p8[0], p8[1]), 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, "INDEX", (p8[0] - 14, p8[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 255, 200), 1, cv2.LINE_AA)

                        cv2.line(frame, (p4[0], p4[1]), (cx, cy), (50, 255, 200), 2, cv2.LINE_AA)
                        cv2.line(frame, (p8[0], p8[1]), (cx, cy), (50, 255, 200), 2, cv2.LINE_AA)

                        fx, fy = int((p4[0] + p8[0]) / 2), int((p4[1] + p8[1]) / 2)
                        cv2.line(frame, (p4[0], p4[1]), (p8[0], p8[1]), (100, 255, 180), 1, cv2.LINE_AA)
                        cv2.circle(frame, (fx, fy), 6, (50, 255, 200), -1, cv2.LINE_AA)
                        cv2.putText(frame, "[DUAL-TOUCH]", (fx - 46, fy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (50, 255, 200), 1, cv2.LINE_AA)
                        break

            # Dynamic Glassmorphic Status Badge
            if is_dragging:
                badge = "[ DUAL-TOUCH: DRAGGING NECK ]"
            elif is_locked:
                badge = "[ NECK LOCKED: TOUCH THUMB & INDEX TO DRAG ]"
            elif ge.neck_hold_progress > 0.0:
                badge = f"SCULPT NECK: HOLD STEADY {int(ge.neck_hold_progress * 100)}% (1.5s)"
            else:
                badge = f"SCULPT NECK (PULL APART): {int(ge.neck_progress * 100)}%"

            text_size = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            bx = max(10, min(frame.shape[1] - text_size[0] - 20, cx - text_size[0] // 2))
            by = min(frame.shape[0] - 25, ry2 + 24)

            # Translucent glassmorphic backing for status badge
            sub_n = frame[by - 16:by + 6, bx - 4:bx + text_size[0] + 8]
            if sub_n.size > 0:
                ov_n = np.full_like(sub_n, (16, 12, 24))
                cv2.addWeighted(ov_n, 0.75, sub_n, 0.25, 0, sub_n)
                frame[by - 16:by + 6, bx - 4:bx + text_size[0] + 8] = sub_n
            cv2.rectangle(frame, (bx - 4, by - 16), (bx + text_size[0] + 8, by + 6), col_primary, 1, cv2.LINE_AA)
            cv2.putText(frame, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col_primary, 1, cv2.LINE_AA)

        # Screen Overflow Disappearance Visual Alert
        if time.perf_counter() - self.last_overflow_time < 2.0:
            banner = f"! OVERFLOW ! {self.overflow_msg} ! OVERFLOW !"
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
                f">> BRING SHAPES CLOSE (<145px) TO ASSEMBLE! (DISTANCE: {d_px}px) <<",
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
        num_sparks = self.quality_config.max_particles
        if num_sparks > 0:
            for angle in np.linspace(0, 2 * np.pi, num_sparks)[:-1]:
                dist_p = radius * 1.12
                px = int(cx + dist_p * np.cos(angle))
                py = int(cy + dist_p * np.sin(angle))
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(frame, (px, py), 4, (0, 255, 255), -1, cv2.LINE_AA)

        # 3. Screen flash on initial impact
        if self.quality_config.enable_shockwave_flash and progress < 0.35:
            alpha_flash = float(0.35 * (1.0 - progress / 0.35))
            flash_overlay = frame.copy()
            flash_overlay[:] = (200, 240, 255)
            cv2.addWeighted(flash_overlay, alpha_flash, frame, 1.0 - alpha_flash, 0, frame)

        # 4. Success text
        cv2.putText(
            frame,
            ">> SNAP FUSION! GUITAR ASSEMBLED SUCCESSFULLY! <<",
            (w // 2 - 300, min(h - 80, cy - radius - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_audio_oscilloscope(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        """Draws a sleek, futuristic audio oscilloscope capsule reacting to sound engine amplitude."""
        if w < 30 or h < 10:
            return

        # 1. Glass background
        roi = frame[y:y + h, x:x + w]
        if roi.size > 0:
            ov = np.full_like(roi, (16, 14, 24), dtype=np.uint8)
            cv2.addWeighted(ov, 0.70, roi, 0.30, 0, roi)
            frame[y:y + h, x:x + w] = roi

        peak = getattr(self.audio_engine, "current_peak", 0.0)
        border_col = (0, 255, 140) if peak > 0.15 else (55, 50, 75)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_col, 1, cv2.LINE_AA)

        # 2. Animated harmonic waveform points
        now_t = time.perf_counter()
        num_pts = getattr(self.quality_config, "oscilloscope_points", 24)
        mid_y = y + h // 2
        pts = []
        for i in range(num_pts):
            u = i / (num_pts - 1)
            px = int(x + 4 + u * (w - 8))
            # Windowing envelope (sin) to taper ends to zero at edges
            envelope = np.sin(np.pi * u)
            # Dual frequency harmonic oscillation modulated by audio peak
            osc = np.sin(u * 12.0 + now_t * 18.0) * 0.7 + np.cos(u * 22.0 - now_t * 24.0) * 0.3
            amp = (peak * 0.85 + 0.06) * (h // 2 - 3) * envelope
            py = int(mid_y + osc * amp)
            pts.append([px, py])

        pts_arr = np.array(pts, dtype=np.int32)
        wave_col = (0, 255, 180) if peak > 0.10 else (0, 180, 220)
        cv2.polylines(frame, [pts_arr], isClosed=False, color=wave_col, thickness=1, lineType=cv2.LINE_AA)

        # Mini spark dot when audio peak is energetic
        if peak > 0.25:
            cv2.circle(frame, (x + w // 2, mid_y), 2, (255, 255, 255), -1, cv2.LINE_AA)

    def shutdown(self) -> None:
        """Clean resource release for camera, hand tracker, audio stream, and UI."""
        logger.info("Shutting down Gesture AR Instruments suite...")
        if hasattr(self, "async_tracker"):
            self.async_tracker.stop()
        self.camera.stop()
        self.audio_engine.stop()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture AR Instruments - Virtual Piano & Guitar")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1920, help="Display frame width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Display frame height (default: 1080)")
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "dshow", "msmf"],
        default="auto",
        help="Camera capture backend (auto, dshow, msmf). Note: dshow and msmf are Windows-only.",
    )
    args = parser.parse_args()
    if sys.platform != "win32" and args.camera_backend in ("dshow", "msmf"):
        parser.error(f"--camera-backend '{args.camera_backend}' is only supported on Windows platforms.")
    return args


def main() -> None:
    args = parse_args()
    app = GestureARApp(
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        camera_backend=args.camera_backend,
    )
    app.run()


if __name__ == "__main__":
    main()
