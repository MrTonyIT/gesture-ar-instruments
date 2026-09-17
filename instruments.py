"""
instruments.py
==============
Ergonomic Spatial AR Instruments Suite: Desk-Surface Piano & Dynamic 3D Low-Poly Guitar.

Key Features & AR Architecture:
1. Virtual Piano (Desk-Surface Docking):
   - Anchored permanently across the bottom 25% of the screen (X: 0.05 to 0.95, Y: 0.72 to 0.96).
   - Designed for physical desk contact: wrists rest on desk, all 5 fingers (Thumb 4,
     Index 8, Middle 12, Ring 16, Pinky 20) tap downward.
   - Black keys priority with downward velocity gating (dY/dt > threshold).
   - Debounce: 120ms lockout per key.

2. Virtual Guitar (Dynamic 3D Low-Poly / Voxel Sprite with Alpha Blending):
   - Asset loading: loads `assets/guitar_blocky.png` with transparent background.
   - Dynamic Procedural Fallback: if asset file is missing, programmatically renders a
     sharp, faceted 3D low-poly gradient polygon guitar body using cv2.fillPoly.
   - Dynamic Spatial Transform & Pose Alignment:
       * Neck anchor -> Left Hand Wrist (Landmark 0) / Palm center.
       * Body / Bridge anchor -> Right Hand area.
       * Dynamic rotation angle theta = arctan2(dy, dx).
       * Dynamic scale factor based on inter-hand distance with EMA smoothing.
       * Vectorized OpenCV warpAffine and alpha blending with strict boundary clipping.
   - 6 Projected Strings:
       * 6 strings projected directly over the soundhole and bridge via the affine matrix.
       * Line-segment intersection detects strumming by Thumb (4), Index (8), or any finger.
       * Transverse standing wave oscillation with dynamic neon glow.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from audio_engine import AudioEngine
from geometry import segments_intersect
from vision_tracker import HandData

logger = logging.getLogger("Instruments")


def render_guitar_overlay(frame: np.ndarray, sprite: np.ndarray, M: np.ndarray) -> None:
    """
    Warps and blends a 4-channel BGRA guitar sprite onto a 3-channel BGR frame
    using fast vectorized OpenCV warpAffine and alpha blending with strict
    boundary clipping guards to eliminate any possibility of crashing.
    Modifies frame in-place.
    """
    if sprite is None or sprite.shape[2] != 4:
        return

    h_bg, w_bg = frame.shape[:2]
    # Warp 4-channel sprite directly to screen dimensions using affine matrix
    warped = cv2.warpAffine(
        sprite,
        M,
        (w_bg, h_bg),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Fast bounded alpha blending: only process the non-zero alpha bounding box
    alpha = warped[:, :, 3]
    mask = alpha > 0
    if not np.any(mask):
        return

    y_indices, x_indices = np.where(mask)
    y1 = max(0, int(y_indices.min()))
    y2 = min(h_bg, int(y_indices.max()) + 1)
    x1 = max(0, int(x_indices.min()))
    x2 = min(w_bg, int(x_indices.max()) + 1)

    if x1 >= x2 or y1 >= y2:
        return

    sub_alpha = (alpha[y1:y2, x1:x2].astype(np.float32) / 255.0)[:, :, np.newaxis]
    sub_fg = warped[y1:y2, x1:x2, :3].astype(np.float32)
    sub_bg = frame[y1:y2, x1:x2].astype(np.float32)

    # Alpha composite in-place
    frame[y1:y2, x1:x2] = (sub_alpha * sub_fg + (1.0 - sub_alpha) * sub_bg).astype(np.uint8)


def create_procedural_blocky_guitar(w: int = 960, h: int = 360) -> np.ndarray:
    """
    VIP Acoustic Guitar Procedural Generator.
    Renders a stunning acoustic dreadnought guitar with rich multi-stage sunburst finish,
    ivory purfling binding, abalone mother-of-pearl rosette, and rosewood belly bridge.
    """
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Palette (BGRA)
    c_espresso = (14, 22, 58, 255)      # Deep dark mahogany rim
    c_caramel = (26, 75, 175, 255)      # Warm caramel mahogany
    c_amber = (50, 140, 240, 255)       # Radiant sunburst amber
    c_honey = (75, 185, 255, 255)       # Golden core
    c_binding = (235, 240, 248, 255)    # Ivory body binding
    c_purfling = (30, 25, 35, 255)      # Black purfling strip
    c_ebony = (24, 20, 28, 255)         # Dark ebony
    c_rosewood = (32, 25, 42, 255)      # Fretboard rosewood
    c_fret = (200, 208, 220, 255)       # Silver-nickel frets
    c_pearl = (245, 248, 255, 255)      # Mother of pearl
    c_gold = (0, 215, 255, 255)         # Warm gold
    c_turquoise = (220, 235, 0, 255)    # Abalone turquoise
    c_guard = (16, 12, 22, 255)         # Celluloid pickguard
    c_hole = (8, 6, 10, 255)            # Deep soundhole void

    # 1. Classic Acoustic Contoured Body (X: 540 to 945)
    body_outer = np.array([
        [545, 158], [585, 90], [640, 68], [700, 78], [735, 112],
        [755, 125], [780, 110], [820, 74], [875, 62], [920, 95],
        [945, 140], [950, 180], [945, 220], [920, 265], [875, 298],
        [820, 286], [780, 250], [755, 235], [735, 248], [700, 282],
        [640, 292], [585, 270], [545, 202]
    ], dtype=np.int32)

    body_mid = np.array([
        [555, 160], [592, 105], [642, 85], [696, 95], [730, 125],
        [752, 136], [776, 124], [814, 92], [866, 82], [908, 110],
        [930, 148], [934, 180], [930, 212], [908, 250], [866, 278],
        [814, 268], [776, 236], [752, 224], [730, 235], [696, 265],
        [642, 275], [592, 255], [555, 200]
    ], dtype=np.int32)

    body_inner = np.array([
        [570, 162], [602, 122], [645, 105], [692, 114], [724, 138],
        [748, 146], [770, 138], [805, 112], [852, 104], [890, 128],
        [910, 156], [914, 180], [910, 204], [890, 232], [852, 256],
        [805, 248], [770, 222], [748, 214], [724, 222], [692, 246],
        [645, 255], [602, 238], [570, 198]
    ], dtype=np.int32)

    body_core = np.array([
        [590, 164], [618, 138], [650, 126], [688, 132], [718, 148],
        [745, 155], [765, 148], [795, 130], [838, 124], [870, 144],
        [885, 165], [888, 180], [885, 195], [870, 216], [838, 236],
        [795, 230], [765, 212], [745, 205], [718, 212], [688, 228],
        [650, 234], [618, 222], [590, 196]
    ], dtype=np.int32)

    # Render Sunburst Gradient Body
    cv2.fillPoly(img, [body_outer], c_espresso)
    cv2.fillPoly(img, [body_mid], c_caramel)
    cv2.fillPoly(img, [body_inner], c_amber)
    cv2.fillPoly(img, [body_core], c_honey)

    # Ivory Multi-Layer Body Binding & Purfling
    cv2.polylines(img, [body_outer], isClosed=True, color=c_binding, thickness=3, lineType=cv2.LINE_AA)
    cv2.polylines(img, [body_outer], isClosed=True, color=c_purfling, thickness=1, lineType=cv2.LINE_AA)
    cv2.polylines(img, [body_mid], isClosed=True, color=(40, 110, 210, 100), thickness=1, lineType=cv2.LINE_AA)

    # Subtle Satin Sheen / Highlight arc along the upper bout
    highlight_arc = np.array([[595, 96], [645, 76], [700, 86], [730, 118]], dtype=np.int32)
    cv2.polylines(img, [highlight_arc], isClosed=False, color=(140, 205, 255, 180), thickness=2, lineType=cv2.LINE_AA)

    # 2. Celluloid Teardrop Pickguard
    guard_poly = np.array([
        [642, 202], [720, 202], [760, 218], [780, 255], [765, 280],
        [720, 290], [675, 275], [638, 240]
    ], dtype=np.int32)
    cv2.fillPoly(img, [guard_poly], c_guard)
    cv2.polylines(img, [guard_poly], isClosed=True, color=c_gold, thickness=2, lineType=cv2.LINE_AA)

    # 3. Abalone Mother-of-Pearl Rosette & Soundhole
    center = (670, 180)
    cv2.circle(img, center, 56, c_hole, -1, cv2.LINE_AA)

    # Concentric Abalone & Gold Rings
    cv2.circle(img, center, 72, c_gold, 2, cv2.LINE_AA)
    cv2.circle(img, center, 68, c_purfling, 1, cv2.LINE_AA)

    num_abalone = 24
    for idx, a in enumerate(np.linspace(0, 2 * np.pi, num_abalone)[:-1]):
        col = c_turquoise if idx % 2 == 0 else c_pearl
        px = int(center[0] + 63 * np.cos(a))
        py = int(center[1] + 63 * np.sin(a))
        cv2.circle(img, (px, py), 3, col, -1, cv2.LINE_AA)

    cv2.circle(img, center, 58, c_purfling, 1, cv2.LINE_AA)
    cv2.circle(img, center, 56, c_gold, 2, cv2.LINE_AA)

    # 4. Sculpted Rosewood Belly Bridge & Bone Saddle
    bridge_poly = np.array([
        [792, 102], [810, 98], [838, 105], [846, 125],
        [838, 180],
        [846, 235], [838, 255], [810, 262], [792, 258], [802, 180]
    ], dtype=np.int32)
    cv2.fillPoly(img, [bridge_poly], c_ebony)
    cv2.polylines(img, [bridge_poly], isClosed=True, color=(80, 65, 55, 255), thickness=2, lineType=cv2.LINE_AA)

    # Bone Compensated Saddle
    cv2.rectangle(img, (806, 110), (812, 250), c_pearl, -1)
    cv2.rectangle(img, (806, 110), (812, 250), (140, 130, 120, 255), 1)

    # 6 Pearloid Bridge Pins with Abalone center dots
    for py in np.linspace(115, 245, 6):
        cv2.circle(img, (826, int(py)), 4, c_pearl, -1, cv2.LINE_AA)
        cv2.circle(img, (826, int(py)), 2, c_turquoise, -1, cv2.LINE_AA)

    # 5. Neck & Fretboard (X: 130 to 560)
    neck_main = np.array([[130, 160], [560, 156], [560, 204], [130, 200]], dtype=np.int32)
    neck_bevel = np.array([[130, 200], [560, 204], [560, 218], [130, 212]], dtype=np.int32)
    cv2.fillPoly(img, [neck_main], c_rosewood)
    cv2.fillPoly(img, [neck_bevel], (18, 14, 22, 255))
    cv2.polylines(img, [neck_main], isClosed=True, color=c_binding, thickness=1, lineType=cv2.LINE_AA)

    # Frets (19 Frets)
    fret_xs = np.linspace(160, 540, 19)
    for fx in fret_xs:
        cv2.line(img, (int(fx), 158), (int(fx), 202), c_fret, 2, cv2.LINE_AA)
        cv2.line(img, (int(fx) + 1, 158), (int(fx) + 1, 202), (100, 105, 115, 255), 1)

    # Mother-of-Pearl Position Marker Dots: 3, 5, 7, 9, 12 (double), 15
    fret_dots = [215, 275, 340, 405, 475]
    for mx in fret_dots:
        cv2.circle(img, (mx, 180), 4, c_pearl, -1, cv2.LINE_AA)
        cv2.circle(img, (mx, 180), 2, c_gold, 1, cv2.LINE_AA)
    # Double dot at 12th fret
    cv2.circle(img, (475, 172), 3, c_pearl, -1, cv2.LINE_AA)
    cv2.circle(img, (475, 188), 3, c_pearl, -1, cv2.LINE_AA)

    # Bone Nut at X=130
    cv2.rectangle(img, (126, 158), (132, 202), c_pearl, -1)
    cv2.rectangle(img, (126, 158), (132, 202), (180, 175, 160, 255), 1)

    # 6. Contoured Classic Headstock (X: 25 to 130)
    head_poly = np.array([
        [130, 158], [80, 142], [42, 145], [26, 160], [26, 200], [42, 215], [80, 218], [130, 202]
    ], dtype=np.int32)
    cv2.fillPoly(img, [head_poly], c_ebony)
    cv2.polylines(img, [head_poly], isClosed=True, color=c_gold, thickness=2, lineType=cv2.LINE_AA)

    # Golden VIP Brand Inlay on Headstock
    cv2.putText(img, "VIP AR", (42, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c_gold, 1, cv2.LINE_AA)

    # 6 Chrome/Gold Tuning Pegs with Pearloid Buttons
    for px in [54, 80, 106]:
        # Upper pegs (Bass strings 6, 5, 4)
        cv2.rectangle(img, (px - 4, 126), (px + 4, 142), (210, 215, 225, 255), -1)
        cv2.circle(img, (px, 122), 7, c_pearl, -1, cv2.LINE_AA)
        cv2.circle(img, (px, 122), 7, c_gold, 1, cv2.LINE_AA)
        # Lower pegs (Treble strings 3, 2, 1)
        cv2.rectangle(img, (px - 4, 218), (px + 4, 234), (210, 215, 225, 255), -1)
        cv2.circle(img, (px, 238), 7, c_pearl, -1, cv2.LINE_AA)
        cv2.circle(img, (px, 238), 7, c_gold, 1, cv2.LINE_AA)

    return img


# =============================================================================
# Piano Key Definitions & Virtual Piano
# =============================================================================

@dataclass
class PianoKey:
    """Represents a single white or black piano key."""
    name: str
    freq: float
    is_black: bool
    rect: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax) in pixels
    last_triggered_time: float = 0.0
    is_active: bool = False
    active_until: float = 0.0


class Piano:
    """
    Ergonomic Desk-Surface Virtual Spatial Piano (2 Octaves: C4 to B5).

    Ergonomic Docking:
    - Permanently docked across the bottom of the display:
        X: 0.05 * width to 0.95 * width
        Y: 0.72 * height to 0.96 * height
    - Supports all 5 fingers (Thumb 4, Index 8, Middle 12, Ring 16, Pinky 20).
    - Black keys priority. Downward velocity gating: dY/dt > threshold.
    - Debounce: 120ms lockout per key.
    """

    NORM_XMIN = 0.05
    NORM_XMAX = 0.95
    NORM_YMIN = 0.72
    NORM_YMAX = 0.96

    DOWNWARD_VELOCITY_NORM = 0.08
    MIN_PIXEL_VELOCITY = 60.0
    DEBOUNCE_SECONDS = 0.120
    ACTIVE_GLOW_DURATION = 0.220

    WHITE_NOTE_NAMES = [
        "C4", "D4", "E4", "F4", "G4", "A4", "B4",
        "C5", "D5", "E5", "F5", "G5", "A5", "B5"
    ]

    WHITE_NOTE_FREQS = [
        261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,
        523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77
    ]

    BLACK_KEY_OFFSETS = [
        (0, "C#4", 277.18),
        (1, "D#4", 311.13),
        (3, "F#4", 369.99),
        (4, "G#4", 415.30),
        (5, "A#4", 466.16),
        (7, "C#5", 554.37),
        (8, "D#5", 622.25),
        (10, "F#5", 739.99),
        (11, "G#5", 830.61),
        (12, "A#5", 932.33),
    ]

    def __init__(self, bbox: Tuple[int, int, int, int], audio_engine: AudioEngine) -> None:
        self.audio_engine = audio_engine
        self.current_shape: Tuple[int, int] = (720, 1280)

        self.xmin, self.ymin, self.xmax, self.ymax = bbox
        self.width = max(10, self.xmax - self.xmin)
        self.height = max(10, self.ymax - self.ymin)

        self.white_keys: List[PianoKey] = []
        self.black_keys: List[PianoKey] = []
        self._finger_held_keys: Dict[Tuple[str, int], str] = {}

        self._build_keyboard(self.xmin, self.ymin, self.xmax, self.ymax)

    def _build_keyboard(self, xmin: int, ymin: int, xmax: int, ymax: int) -> None:
        """Constructs 14 white keys and 10 black keys within the given pixel bounds."""
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        self.width = max(10, self.xmax - self.xmin)
        self.height = max(10, self.ymax - self.ymin)

        self.white_keys.clear()
        self.black_keys.clear()
        self._finger_held_keys.clear()

        num_white = len(self.WHITE_NOTE_NAMES)
        white_key_w = self.width / float(num_white)

        # 1. Construct White Keys
        for i in range(num_white):
            kx1 = int(round(self.xmin + i * white_key_w))
            kx2 = int(round(self.xmin + (i + 1) * white_key_w))
            self.white_keys.append(
                PianoKey(
                    name=self.WHITE_NOTE_NAMES[i],
                    freq=self.WHITE_NOTE_FREQS[i],
                    is_black=False,
                    rect=(kx1, self.ymin, kx2, self.ymax),
                )
            )

        # 2. Construct Black Keys (58% width, 60% height of white keys)
        black_key_w = white_key_w * 0.58
        black_key_h = self.height * 0.60

        for boundary_idx, name, freq in self.BLACK_KEY_OFFSETS:
            center_x = self.xmin + (boundary_idx + 1) * white_key_w
            bx1 = int(round(center_x - black_key_w / 2.0))
            bx2 = int(round(center_x + black_key_w / 2.0))
            by1 = self.ymin
            by2 = int(round(self.ymin + black_key_h))
            self.black_keys.append(
                PianoKey(
                    name=name,
                    freq=freq,
                    is_black=True,
                    rect=(bx1, by1, bx2, by2),
                )
            )

    def sync_resolution(self, frame_shape: Tuple[int, int, int]) -> None:
        """Ensures the desk-docked piano bounds scale cleanly if resolution changes."""
        h, w, _ = frame_shape
        if (h, w) != self.current_shape:
            self.current_shape = (h, w)
            px_xmin = int(round(self.NORM_XMIN * w))
            px_xmax = int(round(self.NORM_XMAX * w))
            px_ymin = int(round(self.NORM_YMIN * h))
            px_ymax = int(round(self.NORM_YMAX * h))
            self._build_keyboard(px_xmin, px_ymin, px_xmax, px_ymax)

    def update(self, hands: List[HandData], frame_shape: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Hit-tests tracked fingertips with downward velocity gating.
        Evaluates Black Keys first to guarantee correct priority.
        Suppresses hover false-positives and cleans up on tracking loss.
        """
        if frame_shape is not None:
            self.sync_resolution(frame_shape)

        now = time.perf_counter()

        for key in self.white_keys + self.black_keys:
            if key.is_active and now > key.active_until:
                key.is_active = False

        if not hands:
            self._finger_held_keys.clear()
            return

        # Target fingertips: All 5 fingers including Thumb (4, 8, 12, 16, 20)
        target_tips = [4, 8, 12, 16, 20]

        for hand in hands:
            for tip_id in target_tips:
                finger_id = (hand.handedness, tip_id)
                tip_pos = hand.landmarks_px[tip_id, :2]
                tx, ty = float(tip_pos[0]), float(tip_pos[1])

                _, vy = hand.fingertip_velocities.get(tip_id, (0.0, 0.0))

                # If finger moves upward or leaves keyboard vertical zone, release held key
                if vy < -40.0 or ty < (self.ymin - 10) or ty > (self.ymax + 30):
                    self._finger_held_keys.pop(finger_id, None)

                if not (self.xmin <= tx <= self.xmax):
                    self._finger_held_keys.pop(finger_id, None)
                    continue

                if ty < self.ymin or ty > (self.ymax + 20):
                    continue

                # Step 1: CHECK BLACK KEYS FIRST
                hit_black_key = False
                for bkey in self.black_keys:
                    bx1, by1, bx2, by2 = bkey.rect
                    if (bx1 - 2) <= tx <= (bx2 + 2) and by1 <= ty <= (by2 + 8):
                        hit_black_key = True
                        self._evaluate_trigger(bkey, vy, now, tx, finger_id)
                        break

                if hit_black_key:
                    continue

                # Step 2: CHECK WHITE KEYS
                for wkey in self.white_keys:
                    wx1, wy1, wx2, wy2 = wkey.rect
                    if wx1 <= tx <= wx2 and wy1 <= ty <= wy2:
                        self._evaluate_trigger(wkey, vy, now, tx, finger_id)
                        break

    def _evaluate_trigger(
        self,
        key: PianoKey,
        vy: float,
        now: float,
        tx: float,
        finger_id: Tuple[str, int],
    ) -> None:
        """Evaluates downward velocity gating, debounce lockout, and hover suppression."""
        if (now - key.last_triggered_time) < self.DEBOUNCE_SECONDS:
            return

        # Hover suppression: prevent restriking the same key while resting inside it
        if self._finger_held_keys.get(finger_id) == key.name:
            return

        norm_vy = vy / max(1.0, float(self.current_shape[0]))
        if vy < self.MIN_PIXEL_VELOCITY or norm_vy < self.DOWNWARD_VELOCITY_NORM:
            return

        norm_velocity = float(np.clip(vy / 600.0, 0.25, 1.0))
        rel_x = (tx - self.xmin) / max(1.0, float(self.width))
        pan = -0.7 + rel_x * 1.4

        self.audio_engine.play_piano(freq=key.freq, velocity=norm_velocity, pan=pan)

        key.last_triggered_time = now
        key.is_active = True
        key.active_until = now + self.ACTIVE_GLOW_DURATION
        self._finger_held_keys[finger_id] = key.name

    def draw(self, frame: np.ndarray) -> None:
        """Renders cyber HUD desk-docked piano with neon feedback."""
        self.sync_resolution(frame.shape)

        overlay_base = frame.copy()
        cv2.rectangle(
            overlay_base,
            (self.xmin - 8, self.ymin - 24),
            (self.xmax + 8, self.ymax + 6),
            (14, 10, 18),
            -1,
        )
        cv2.addWeighted(overlay_base, 0.70, frame, 0.30, 0, frame)

        cv2.putText(
            frame,
            "DESK-SURFACE PIANO: REST WRISTS ON DESK - TAP SURFACE WITH ANY FINGER",
            (self.xmin + 12, self.ymin - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 240, 255),
            1,
            cv2.LINE_AA,
        )

        overlay_white = frame.copy()
        for key in self.white_keys:
            x1, y1, x2, y2 = key.rect
            if key.is_active:
                cv2.rectangle(overlay_white, (x1, y1), (x2, y2), (255, 220, 0), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            else:
                cv2.rectangle(overlay_white, (x1, y1), (x2, y2), (225, 228, 235), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 70), 1)

            lbl_x = x1 + (x2 - x1) // 2 - 10
            lbl_y = y2 - 10
            txt_col = (10, 10, 10) if not key.is_active else (0, 0, 120)
            cv2.putText(overlay_white, key.name, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay_white, 0.78, frame, 0.22, 0, frame)

        overlay_black = frame.copy()
        for key in self.black_keys:
            x1, y1, x2, y2 = key.rect
            if key.is_active:
                cv2.rectangle(overlay_black, (x1, y1), (x2, y2), (255, 0, 220), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 255), 2)
            else:
                cv2.rectangle(overlay_black, (x1, y1), (x2, y2), (20, 18, 24), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 100, 130), 1)

            lbl_x = x1 + (x2 - x1) // 2 - 11
            lbl_y = y2 - 8
            cv2.putText(overlay_black, key.name, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (210, 210, 210), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay_black, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (self.xmin - 2, self.ymin - 2), (self.xmax + 2, self.ymax + 2), (0, 240, 255), 2)


# =============================================================================
# Virtual Guitar with 3D Stylized Low-Poly Sprite & Alpha Blending
# =============================================================================

@dataclass
class ProjectedGuitarString:
    """Represents a guitar string dynamically projected over the 3D guitar graphic."""
    index: int
    note_name: str
    local_start: Tuple[float, float]  # On unrotated sprite
    local_end: Tuple[float, float]    # On unrotated sprite
    screen_start: Tuple[float, float] = (0.0, 0.0)
    screen_end: Tuple[float, float] = (0.0, 0.0)
    last_plucked_time: float = 0.0
    vibration_amplitude: float = 0.0
    vibration_phase: float = 0.0


class Guitar:
    """
    Upgraded Stylized / Low-Poly 3D Guitar with Dynamic Hand Anchoring & Alpha Blending.

    Features:
    - Renders an authentic stylized/blocky guitar overlay using PNG alpha blending.
    - Dynamically anchors:
        * Neck / Fretboard -> Left Hand Wrist / Palm center.
        * Body / Soundhole -> Right Hand lap area.
        * Real-time rotation angle theta and scale based on distance between hands.
    - Projects 6 visible virtual strings over the soundhole and bridge.
    - Multi-finger strumming: detects crossings by Thumb (4), Index (8), and all fingers.
    - Transverse standing wave string vibration animation with neon glow.
    """

    CHORDS = ["C", "G", "D", "A", "E", "Am", "Em", "Dm", "F"]
    STRING_NAMES = ["E2 (6th)", "A2 (5th)", "D3 (4th)", "G3 (3rd)", "B3 (2nd)", "E4 (1st)"]
    STRUM_DEBOUNCE = 0.060  # 60ms lockout per string

    # Sprite reference anchor coordinates (on unrotated 960x360 sprite)
    # Neck anchor aligned right at Fret 1 (150px), directly adjacent to the nut (130px) and headstock (30-130px)
    # Body anchor centered at strumming zone between soundhole (670px) and bridge (815px)
    SPRITE_ANCHOR_NECK = (150.0, 180.0)
    SPRITE_ANCHOR_BODY = (740.0, 180.0)
    SPRITE_REF_SPAN = 590.0  # Reference distance between neck and body anchors

    def __init__(
        self,
        zones: Optional[Dict[str, Tuple[int, int, int, int]]],
        audio_engine: AudioEngine,
        asset_path: str = "assets/guitar_blocky.png",
    ) -> None:
        self.audio_engine = audio_engine
        self.asset_path = asset_path

        # 1. Load or generate blocky guitar sprite
        self.sprite = self._load_or_create_sprite()
        self.sprite_h, self.sprite_w = self.sprite.shape[:2]

        # 2. Dynamic Pose Anchors (EMA smoothed)
        self.current_neck_pt = np.array([280.0, 390.0], dtype=np.float32)
        self.current_body_pt = np.array([820.0, 560.0], dtype=np.float32)
        self.ema_alpha = 0.70  # Snappy, real-time pose tracking without floating delay

        self.active_chord = "C"
        self.active_fret_finger: Optional[int] = 8  # Default to index finger (C)
        self.lh_landmarks: Optional[np.ndarray] = None
        self.lh_finger_count: int = 1
        self.extended_finger_flags: List[bool] = [True, False, False, False]
        self.fret_slide_chord: Optional[str] = "C"
        self.rh_landmarks: Optional[np.ndarray] = None
        self._prev_strum_finger_positions: Dict[int, Tuple[float, float]] = {}
        self._prev_right_timestamp: float = time.perf_counter()
        self._last_vibe_time: float = time.perf_counter()

        # 3. Setup strings in sprite local coordinate system
        self.strings: List[ProjectedGuitarString] = []
        self._setup_local_strings()
        self._project_strings((720, 1280, 3))

        # Fretboard chord boxes (floating near neck)
        self.chord_boxes: Dict[str, Tuple[int, int, int, int]] = {}

        # Dedicated strumming fingers: strictly Thumb (4) and Index (8)
        self.strum_tip_ids: List[int] = [4, 8]

    def _load_or_create_sprite(self) -> np.ndarray:
        """Loads assets/guitar_blocky.png or generates procedural fallback."""
        if os.path.isfile(self.asset_path):
            img = cv2.imread(self.asset_path, cv2.IMREAD_UNCHANGED)
            if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                logger.info("Loaded blocky guitar asset from %s (%s)", self.asset_path, img.shape)
                return img

        logger.warning("Guitar asset not found at %s. Generating procedural low-poly fallback.", self.asset_path)
        img = create_procedural_blocky_guitar(960, 360)
        # Try saving for future fast loads
        try:
            os.makedirs(os.path.dirname(self.asset_path) or ".", exist_ok=True)
            cv2.imwrite(self.asset_path, img)
            logger.info("Saved procedural guitar asset to %s", self.asset_path)
        except Exception as e:
            logger.debug("Could not write asset cache: %s", e)
        return img

    def _setup_local_strings(self) -> None:
        """
        Defines 6 string segments across the soundhole and bridge in local coordinates.
        Widened to 130px span (26.0px separation between strings)
        for ultra-clear, distinct individual string strumming without accidental cross-plucking.
        """
        # On sprite: Soundhole at X=670, Bridge at X=810.
        # Strumming zone spans X from 565 to 820.
        # Strings span Y from 115 to 245 (130px span, 26px separation between strings)
        for i in range(6):
            y_pos = 115.0 + i * 26.0
            self.strings.append(
                ProjectedGuitarString(
                    index=i,
                    note_name=self.STRING_NAMES[i],
                    local_start=(565.0, y_pos),
                    local_end=(820.0, y_pos),
                )
            )

    def _compute_transform(self, frame_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the 2x3 Affine transformation matrix aligning the sprite anchors
        with the current smoothed neck and body anchor points.
        """
        h_bg, w_bg = frame_shape[:2]

        x_l, y_l = float(self.current_neck_pt[0]), float(self.current_neck_pt[1])
        x_r, y_r = float(self.current_body_pt[0]), float(self.current_body_pt[1])

        dx = x_r - x_l
        dy = y_r - y_l
        dist = max(50.0, float(np.sqrt(dx * dx + dy * dy)))

        # Adaptive scale clamping: prevents guitar body from ballooning and obstructing face
        # Capped to 52% of camera frame height or 0.82 max
        max_scale = min(0.82, float((h_bg * 0.52) / max(1.0, float(self.sprite_h))))
        scale = float(np.clip(dist / self.SPRITE_REF_SPAN, 0.50, max_scale))

        raw_theta = float(np.arctan2(dy, dx))
        # Clamp theta to ergonomic playing angle (8 to 28 degrees)
        # Keeps neck tilted comfortably from lap to left hand without angling into face/chin
        theta = float(np.clip(raw_theta, np.radians(8.0), np.radians(28.0)))

        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))

        u1, v1 = self.SPRITE_ANCHOR_NECK

        # Affine matrix: maps (u1, v1) -> (x_l, y_l) and (u2, v2) -> (x_r, y_r)
        M = np.array([
            [scale * cos_t, -scale * sin_t, x_l - scale * (u1 * cos_t - v1 * sin_t)],
            [scale * sin_t,  scale * cos_t, y_l - scale * (u1 * sin_t + v1 * cos_t)]
        ], dtype=np.float32)

        return M, np.array([scale, theta], dtype=np.float32)

    def _project_strings(self, shape: Tuple[int, int, int]) -> None:
        """Projects local string coordinates into screen space using current affine transform."""
        M, _ = self._compute_transform(shape)
        for string in self.strings:
            p_start_homo = np.array([string.local_start[0], string.local_start[1], 1.0], dtype=np.float32)
            p_end_homo = np.array([string.local_end[0], string.local_end[1], 1.0], dtype=np.float32)

            s_start = M @ p_start_homo
            s_end = M @ p_end_homo

            string.screen_start = (float(s_start[0]), float(s_start[1]))
            string.screen_end = (float(s_end[0]), float(s_end[1]))

    def set_chord(self, chord_name: str) -> bool:
        """Explicitly switches active chord (e.g. from hotkeys or external triggers)."""
        if chord_name in self.CHORDS:
            self.active_chord = chord_name
            return True
        return False

    def update(self, hands: List[HandData], frame_shape: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Dynamically anchors guitar pose to tracked hands, updates string projections,
        and tests multi-finger strum collisions.
        """
        now = time.perf_counter()
        shape = frame_shape or (720, 1280, 3)
        h_bg, w_bg = shape[:2]

        left_hand: Optional[HandData] = None
        right_hand: Optional[HandData] = None
        for hand in hands:
            if hand.handedness == "Left":
                left_hand = hand
            elif hand.handedness == "Right":
                right_hand = hand

        # 1. Update Dynamic Anchor Points with Ergonomic Lap Grounding & Finger Grip
        if left_hand is not None and right_hand is not None:
            # Anchor neck directly in finger knuckle cradle (Middle MCP 9 & PIP 10)
            # Neck Y bounded to prevent drifting into chin/face
            raw_neck = 0.55 * left_hand.landmarks_px[9, :2] + 0.45 * left_hand.landmarks_px[10, :2]
            neck_y = max(float(h_bg * 0.40), float(raw_neck[1]))
            target_neck = np.array([float(raw_neck[0]), neck_y], dtype=np.float32)

            # Ground guitar body to lap / lower abdomen zone (Y >= 60% frame height)
            # Strumming hand hovers above the strings/soundhole naturally
            raw_body_x = float(right_hand.landmarks_px[0, 0])
            raw_body_y = float(right_hand.landmarks_px[0, 1])
            body_y = max(float(h_bg * 0.60), raw_body_y + 70.0)
            body_x = max(target_neck[0] + 320.0, raw_body_x + 40.0)
            target_body = np.array([body_x, body_y], dtype=np.float32)

            # Direct Adaptive Pose Tracking for Guitar Anchors
            delta_n = float(np.linalg.norm(target_neck - self.current_neck_pt))
            delta_b = float(np.linalg.norm(target_body - self.current_body_pt))
            a_neck = 1.0 if delta_n > 14.0 else (0.90 if delta_n > 2.5 else 0.0)
            a_body = 1.0 if delta_b > 14.0 else (0.90 if delta_b > 2.5 else 0.0)
            self.current_neck_pt = a_neck * target_neck + (1.0 - a_neck) * self.current_neck_pt
            self.current_body_pt = a_body * target_body + (1.0 - a_body) * self.current_body_pt
        elif left_hand is not None:
            raw_neck = 0.55 * left_hand.landmarks_px[9, :2] + 0.45 * left_hand.landmarks_px[10, :2]
            neck_y = max(float(h_bg * 0.40), float(raw_neck[1]))
            target_neck = np.array([float(raw_neck[0]), neck_y], dtype=np.float32)
            delta_n = float(np.linalg.norm(target_neck - self.current_neck_pt))
            a_neck = 1.0 if delta_n > 14.0 else (0.90 if delta_n > 2.5 else 0.0)
            self.current_neck_pt = a_neck * target_neck + (1.0 - a_neck) * self.current_neck_pt

            # Infer right body position resting down at lap level
            inferred_body = target_neck + np.array([460.0, 140.0], dtype=np.float32)
            inferred_body[1] = max(float(h_bg * 0.60), float(inferred_body[1]))
            self.current_body_pt = 0.90 * inferred_body + 0.10 * self.current_body_pt
        elif right_hand is not None:
            raw_body_x = float(right_hand.landmarks_px[0, 0])
            raw_body_y = float(right_hand.landmarks_px[0, 1])
            body_y = max(float(h_bg * 0.60), raw_body_y + 70.0)
            body_x = max(float(w_bg * 0.55), raw_body_x + 40.0)
            target_body = np.array([body_x, body_y], dtype=np.float32)
            delta_b = float(np.linalg.norm(target_body - self.current_body_pt))
            a_body = 1.0 if delta_b > 14.0 else (0.90 if delta_b > 2.5 else 0.0)
            self.current_body_pt = a_body * target_body + (1.0 - a_body) * self.current_body_pt

            inferred_neck = target_body - np.array([460.0, 140.0], dtype=np.float32)
            inferred_neck[1] = max(float(h_bg * 0.40), float(inferred_neck[1]))
            self.current_neck_pt = 0.90 * inferred_neck + 0.10 * self.current_neck_pt

        # 2. Compute Affine Transform and Project Strings into Screen Coordinates
        self._project_strings(shape)

        # 3. Top-Left Floating Fretboard Chord Selector (9 Chords)
        # Positioned cleanly across top-left: box_w = 48, box_h = 40, gap = 6, start_x = 20, start_y = 56
        box_w, box_h = 48, 40
        start_x = 20
        start_y = 56
        gap = 6
        for idx, chord in enumerate(self.CHORDS):
            bx1 = start_x + idx * (box_w + gap)
            bx2 = bx1 + box_w
            by1 = start_y
            by2 = start_y + box_h
            self.chord_boxes[chord] = (bx1, by1, bx2, by2)

        # 4. Left Hand Chord Selection:
        # METHOD 1 (PRIMARY): Number of Extended Fingers (1=C, 2=G, 3=Am, 4=Em)
        # METHOD 2: Direct Touch on Top 9 Chord Boxes
        # METHOD 3: Fretboard Horizontal Position along the neck (9 zones)
        # METHOD 4: Thumb Pinch
        if left_hand is not None:
            pts = left_hand.landmarks_px
            wrist = pts[0, :2]
            self.lh_landmarks = pts.copy()

            # Measure extension of 4 long fingers: Index (8), Middle (12), Ring (16), Pinky (20)
            ext_index = bool(np.linalg.norm(pts[8, :2] - wrist) > np.linalg.norm(pts[6, :2] - wrist) * 1.15)
            ext_middle = bool(np.linalg.norm(pts[12, :2] - wrist) > np.linalg.norm(pts[10, :2] - wrist) * 1.15)
            ext_ring = bool(np.linalg.norm(pts[16, :2] - wrist) > np.linalg.norm(pts[14, :2] - wrist) * 1.15)
            ext_pinky = bool(np.linalg.norm(pts[20, :2] - wrist) > np.linalg.norm(pts[18, :2] - wrist) * 1.15)

            self.extended_finger_flags = [ext_index, ext_middle, ext_ring, ext_pinky]
            finger_count = sum(self.extended_finger_flags)
            self.lh_finger_count = finger_count

            # Check Fretboard Horizontal Position along the neck (9 zones)
            palm_x = float(pts[9, 0])
            slide_step = 35.0
            slide_base = 200.0
            zone_idx = int(np.clip((palm_x - slide_base) / slide_step, 0, len(self.CHORDS) - 1))
            self.fret_slide_chord = self.CHORDS[zone_idx]

            # Check Thumb Pinch with any finger as alternative
            thumb_tip = pts[4, :2]
            hand_scale = max(40.0, float(np.linalg.norm(pts[0, :2] - pts[9, :2])))
            pinch_thresh = max(36.0, 0.40 * hand_scale)
            pinched_chord: Optional[str] = None
            for tip_id, chord in [(8, "C"), (12, "G"), (16, "D"), (20, "Am")]:
                if np.linalg.norm(thumb_tip - pts[tip_id, :2]) < pinch_thresh:
                    pinched_chord = chord
                    break

            # Check Direct Touch / Reach into Top 9 Chord Boxes
            touched_chord: Optional[str] = None
            lh_tips = pts[[4, 8, 12, 16, 20], :2]
            for chord, (bx1, by1, bx2, by2) in self.chord_boxes.items():
                for tip in lh_tips:
                    if (bx1 - 8) <= tip[0] <= (bx2 + 8) and (by1 - 8) <= tip[1] <= (by2 + 10):
                        touched_chord = chord
                        break
                if touched_chord is not None:
                    break

            # Prioritized Chord Resolution:
            # 1. Direct Touch on top boxes
            if touched_chord is not None:
                self.active_chord = touched_chord
            # 2. Deliberate Thumb Pinch
            elif pinched_chord is not None:
                self.active_chord = pinched_chord
            # 3. Finger Count (1 = C, 2 = G, 3 = Am, 4 = Em)
            elif finger_count == 1:
                self.active_chord = "C"
            elif finger_count == 2:
                self.active_chord = "G"
            elif finger_count == 3:
                self.active_chord = "Am"
            elif finger_count >= 4:
                self.active_chord = "Em"
            # 4. If 0 fingers: keep previous active chord
        else:
            self.lh_landmarks = None
            self.lh_finger_count = 0
            self.extended_finger_flags = [False, False, False, False]

        # 5. Right Hand Strumming: Only Thumb (4) and Index (8)
        if right_hand is not None:
            self.rh_landmarks = right_hand.landmarks_px.copy()
            dt = max(0.001, now - self._prev_right_timestamp)

            for tip_id in self.strum_tip_ids:
                curr_pos = tuple(right_hand.landmarks_px[tip_id, :2].astype(float))

                if tip_id in self._prev_strum_finger_positions:
                    p_prev = self._prev_strum_finger_positions[tip_id]
                    p_curr = curr_pos

                    for string in self.strings:
                        if (now - string.last_plucked_time) < self.STRUM_DEBOUNCE:
                            continue

                        # Test 2D segment intersection against dynamically projected string
                        if segments_intersect(p_prev, p_curr, string.screen_start, string.screen_end):
                            displacement = np.linalg.norm(np.array(p_curr) - np.array(p_prev))
                            velocity = float(displacement / dt)
                            norm_vol = float(np.clip(velocity / 800.0, 0.25, 1.0))

                            played = self.audio_engine.play_guitar(
                                string_idx=string.index,
                                chord_name=self.active_chord,
                                velocity=norm_vol,
                            )

                            string.last_plucked_time = now
                            if played:
                                string.vibration_amplitude = norm_vol * 16.0
                            else:
                                # Muted string: dampened subtle vibration, no voice allocation
                                string.vibration_amplitude = norm_vol * 2.5
                            string.vibration_phase = 0.0

                self._prev_strum_finger_positions[tip_id] = curr_pos

            self._prev_right_timestamp = now
        else:
            self.rh_landmarks = None
            self._prev_strum_finger_positions.clear()

        # 6. Update String Transverse Vibration Dynamics
        dt_vibe = max(0.001, now - self._last_vibe_time)
        self._last_vibe_time = now
        for string in self.strings:
            if string.vibration_amplitude > 0.05:
                string.vibration_amplitude *= np.exp(-6.5 * dt_vibe)
                string.vibration_phase += 26.0 * dt_vibe
            else:
                string.vibration_amplitude = 0.0

    def draw(self, frame: np.ndarray) -> None:
        """Renders the transformed 3D low-poly guitar sprite and projected vibrating strings."""
        M, _ = self._compute_transform(frame.shape)

        # 1. Render 3D Low-Poly Blocky Guitar via Vectorized Alpha Blending
        render_guitar_overlay(frame, self.sprite, M)

        # 2. Render Floating Fretboard Chord Selector (9 Chords HUD)
        # Guide banner placed directly below chord boxes (Y=114)
        cv2.putText(
            frame,
            "LEFT HAND: RAISE 1-4 FINGERS, TOUCH [1]-[9], OR USE KEYS 1-9 / C,G,D,A,E,F",
            (20, 114),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 240, 255),
            1,
            cv2.LINE_AA,
        )

        for idx, chord in enumerate(self.CHORDS):
            bx1, by1, bx2, by2 = self.chord_boxes[chord]
            is_active = (chord == self.active_chord)
            key_num = f"[{idx + 1}]"

            # Fast local ROI blending
            sub_roi = frame[by1:by2, bx1:bx2]
            if sub_roi.size > 0:
                color_bg = np.array([0, 180, 90] if is_active else [28, 20, 36], dtype=np.uint8)
                alpha = 0.65 if is_active else 0.70
                blended = cv2.addWeighted(sub_roi, 1.0 - alpha, np.full_like(sub_roi, color_bg), alpha, 0)
                frame[by1:by2, bx1:bx2] = blended

            border_col = (0, 255, 140) if is_active else (80, 65, 100)
            thickness = 2 if is_active else 1
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_col, thickness)

            # Chord name
            name_col = (255, 255, 255) if is_active else (210, 210, 220)
            t_size = cv2.getTextSize(chord, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1 if not is_active else 2)[0]
            cx = bx1 + (48 - t_size[0]) // 2
            cy = by1 + 18
            cv2.putText(frame, chord, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.44, name_col, 2 if is_active else 1, cv2.LINE_AA)

            # Key label
            key_col = (255, 255, 255) if is_active else (140, 170, 190)
            k_size = cv2.getTextSize(key_num, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)[0]
            kx = bx1 + (48 - k_size[0]) // 2
            ky = by2 - 6
            cv2.putText(frame, key_num, (kx, ky), cv2.FONT_HERSHEY_SIMPLEX, 0.30, key_col, 1, cv2.LINE_AA)
            if is_active:
                cv2.circle(frame, (bx2 - 6, by1 + 6), 3, (0, 255, 255), -1, cv2.LINE_AA)

        # 3. Render 6 Projected Strings with Transverse Standing Wave Oscillation
        for string in self.strings:
            is_muted = AudioEngine.is_string_muted(self.active_chord, string.index)
            p1 = np.array(string.screen_start, dtype=np.float32)
            p2 = np.array(string.screen_end, dtype=np.float32)
            seg_vec = p2 - p1
            seg_len = float(np.linalg.norm(seg_vec))

            if seg_len > 1e-4:
                normal = np.array([-seg_vec[1] / seg_len, seg_vec[0] / seg_len], dtype=np.float32)
            else:
                normal = np.array([0.0, 1.0], dtype=np.float32)

            if string.vibration_amplitude > 0.2:
                # Transverse sinusoidal standing wave
                num_pts = 28
                u_vals = np.linspace(0.0, 1.0, num_pts, dtype=np.float32)
                base_pts = p1[np.newaxis, :] + u_vals[:, np.newaxis] * seg_vec[np.newaxis, :]
                wave = (
                    string.vibration_amplitude
                    * np.sin(np.pi * u_vals)[:, np.newaxis]
                    * np.sin(string.vibration_phase)
                    * normal[np.newaxis, :]
                )
                vibe_pts = (base_pts + wave).astype(np.int32)

                glow_col = (100, 100, 150) if is_muted else (0, 240, 255)
                beam_col = (180, 180, 200) if is_muted else (255, 255, 255)

                cv2.polylines(frame, [vibe_pts], isClosed=False, color=glow_col, thickness=3 if not is_muted else 2, lineType=cv2.LINE_AA)
                cv2.polylines(frame, [vibe_pts], isClosed=False, color=beam_col, thickness=1, lineType=cv2.LINE_AA)

                # High-energy glowing neon particle sparks only for sounded unmuted strings
                if not is_muted and string.vibration_amplitude > 1.0:
                    num_sparks = int(min(14, string.vibration_amplitude * 0.85))
                    rng = np.random.RandomState(int(string.vibration_phase * 15) + string.index * 17)
                    spark_u = rng.uniform(0.12, 0.88, num_sparks)
                    spark_dev = rng.uniform(-string.vibration_amplitude * 1.6, string.vibration_amplitude * 1.6, (num_sparks, 2))
                    for su, off in zip(spark_u, spark_dev):
                        sp_base = p1 + su * seg_vec
                        sp_pos = (int(round(sp_base[0] + off[0])), int(round(sp_base[1] + off[1])))
                        if 0 <= sp_pos[0] < frame.shape[1] and 0 <= sp_pos[1] < frame.shape[0]:
                            cv2.circle(frame, sp_pos, 2, (255, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(frame, sp_pos, 4, (0, 240, 255), 1, cv2.LINE_AA)

                # Note label or muted indicator near bridge
                bx_t = int(round(p2[0])) + 8
                by_t = int(round(p2[1])) + 4
                if is_muted:
                    cv2.putText(frame, "X", (bx_t, by_t), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 220), 1, cv2.LINE_AA)
                else:
                    note_tag = string.note_name.split()[0]
                    cv2.putText(frame, note_tag, (bx_t, by_t), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                # Resting string
                pt1_i = (int(round(p1[0])), int(round(p1[1])))
                pt2_i = (int(round(p2[0])), int(round(p2[1])))
                if is_muted:
                    color = (95, 90, 95)
                    thickness = 1
                else:
                    color = (210, 185, 140) if string.index in (0, 1, 2) else (190, 225, 245)
                    thickness = max(1, 3 - string.index // 2)
                cv2.line(frame, pt1_i, pt2_i, color, thickness, lineType=cv2.LINE_AA)

                # Show 'X' near bridge for muted string even when resting
                if is_muted:
                    bx_t = int(round(p2[0])) + 8
                    by_t = int(round(p2[1])) + 4
                    cv2.putText(frame, "x", (bx_t, by_t), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 160), 1, cv2.LINE_AA)

        # 4. Render AR HUD Feedback directly on Left Hand
        if self.lh_landmarks is not None:
            pts = self.lh_landmarks
            palm_cx = int(round(pts[9, 0]))
            palm_cy = int(round(pts[9, 1]))

            # Glowing reticle on each extended finger
            tip_indices = [8, 12, 16, 20]
            for idx, is_ext in enumerate(self.extended_finger_flags):
                if is_ext:
                    tip_id = tip_indices[idx]
                    tx, ty = int(round(pts[tip_id, 0])), int(round(pts[tip_id, 1]))
                    cv2.circle(frame, (tx, ty), 10, (0, 255, 180), 2, cv2.LINE_AA)
                    cv2.circle(frame, (tx, ty), 4, (255, 255, 255), -1, cv2.LINE_AA)

            # Floating Prominent Chord Badge above the left hand
            bx = max(10, min(frame.shape[1] - 180, palm_cx - 85))
            by = max(30, min(frame.shape[0] - 20, palm_cy - 48))
            badge_text = f"CHORD: [ {self.active_chord} ]"

            # Glassmorphic cyber pill badge
            sub_ch = frame[by - 18:by + 6, bx:bx + 170]
            if sub_ch.size > 0:
                ov_ch = np.full_like(sub_ch, (15, 12, 22))
                cv2.addWeighted(ov_ch, 0.70, sub_ch, 0.30, 0, sub_ch)
                frame[by - 18:by + 6, bx:bx + 170] = sub_ch
            cv2.rectangle(frame, (bx, by - 18), (bx + 170, by + 6), (0, 255, 140), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                badge_text,
                (bx + 8, by - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 255, 180),
                1,
                cv2.LINE_AA,
            )

        # 5. Render Strumming Reticles directly on Right Hand Thumb (4) & Index (8)
        if self.rh_landmarks is not None:
            pts_r = self.rh_landmarks
            for tip_id, color, name in [(4, (255, 0, 220), "THUMB"), (8, (0, 240, 255), "INDEX")]:
                tx = int(round(pts_r[tip_id, 0]))
                ty = int(round(pts_r[tip_id, 1]))
                # Concentric targeting circles on active picking fingers
                cv2.circle(frame, (tx, ty), 12, color, 2, cv2.LINE_AA)
                cv2.circle(frame, (tx, ty), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    name,
                    (tx - 20, ty - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                    cv2.LINE_AA,
                )
