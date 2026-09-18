"""
tests/test_ui_geometry.py
=========================
Unit tests for UI geometry rendering, corner brackets, and bounding boxes.
Verifies _draw_cyber_box produces exactly 8 non-duplicate line segments.
"""

from unittest.mock import patch
import numpy as np

from main import GestureARApp


def test_draw_cyber_box_eight_distinct_corner_lines():
    """
    Verifies that _draw_cyber_box renders exactly 8 distinct line segments
    (2 per corner: 1 horizontal and 1 vertical) with zero duplicate draw calls.
    """
    app = GestureARApp(start_threads=False, init_mediapipe=False)
    try:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        x1, y1, x2, y2 = 100, 150, 400, 350
        color = (0, 255, 200)

        lines_drawn = []

        def mock_line(img, pt1, pt2, col, thick=1, lineType=None):
            # Normalize segment representation for set uniqueness check
            p1, p2 = tuple(pt1), tuple(pt2)
            canonical_seg = (min(p1, p2), max(p1, p2))
            lines_drawn.append((p1, p2, canonical_seg))

        with patch("cv2.line", side_effect=mock_line):
            app._draw_cyber_box(frame, x1, y1, x2, y2, color)

        assert len(lines_drawn) == 8, f"Expected exactly 8 corner line calls, got {len(lines_drawn)}"

        # Verify all 8 lines are unique (no duplicates)
        canonical_segments = [item[2] for item in lines_drawn]
        unique_segments = set(canonical_segments)
        assert len(unique_segments) == 8, f"Detected duplicate corner lines in _draw_cyber_box: {lines_drawn}"

        # Verify 4 horizontal lines and 4 vertical lines
        horizontal_count = 0
        vertical_count = 0
        corner_len = 24
        for p1, p2, _ in lines_drawn:
            if p1[1] == p2[1]:
                horizontal_count += 1
                assert abs(p1[0] - p2[0]) == corner_len
            elif p1[0] == p2[0]:
                vertical_count += 1
                assert abs(p1[1] - p2[1]) == corner_len

        assert horizontal_count == 4, f"Expected 4 horizontal bracket lines, got {horizontal_count}"
        assert vertical_count == 4, f"Expected 4 vertical bracket lines, got {vertical_count}"
    finally:
        app.shutdown()
