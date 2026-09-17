"""
tests/test_geometry.py
======================
Unit tests for 2D geometry functions in geometry.py.
Covers:
- True 2D line segment intersection (segments_intersect)
- Collinear segments (overlapping, disjoint, touching endpoints)
- Degenerate segments (zero-length points)
- Parallel segments
- Orientation cross product (ccw)
- Point on segment (on_segment)
- Point in rectangle (point_in_rect)
- Point in circle (point_in_circle)
- Euclidean distance
"""

import pytest

from geometry import (
    ccw,
    euclidean_distance,
    point_in_circle,
    point_in_rect,
    segments_intersect,
)


def test_standard_intersection():
    """Standard X-crossing of two perpendicular segments."""
    p1 = (0.0, 5.0)
    p2 = (10.0, 5.0)
    q1 = (5.0, 0.0)
    q2 = (5.0, 10.0)
    assert segments_intersect(p1, p2, q1, q2) is True


def test_t_junction_touching():
    """Segment endpoint precisely touches the other segment."""
    p1 = (0.0, 5.0)
    p2 = (10.0, 5.0)
    q1 = (5.0, 5.0)
    q2 = (5.0, 10.0)
    assert segments_intersect(p1, p2, q1, q2) is True


def test_shared_endpoint():
    """Two segments share an exact endpoint."""
    p1 = (0.0, 0.0)
    p2 = (5.0, 5.0)
    q1 = (5.0, 5.0)
    q2 = (10.0, 0.0)
    assert segments_intersect(p1, p2, q1, q2) is True


def test_parallel_disjoint():
    """Two parallel lines that never intersect."""
    p1 = (0.0, 0.0)
    p2 = (10.0, 0.0)
    q1 = (0.0, 5.0)
    q2 = (10.0, 5.0)
    assert segments_intersect(p1, p2, q1, q2) is False


def test_collinear_overlapping():
    """Two collinear segments that overlap."""
    p1 = (0.0, 0.0)
    p2 = (10.0, 0.0)
    q1 = (5.0, 0.0)
    q2 = (15.0, 0.0)
    assert segments_intersect(p1, p2, q1, q2) is True


def test_collinear_disjoint():
    """Two collinear segments that do not overlap."""
    p1 = (0.0, 0.0)
    p2 = (5.0, 0.0)
    q1 = (7.0, 0.0)
    q2 = (12.0, 0.0)
    assert segments_intersect(p1, p2, q1, q2) is False


def test_collinear_touching_at_endpoint():
    """Two collinear segments touching only at one endpoint."""
    p1 = (0.0, 0.0)
    p2 = (5.0, 0.0)
    q1 = (5.0, 0.0)
    q2 = (10.0, 0.0)
    assert segments_intersect(p1, p2, q1, q2) is True


def test_degenerate_zero_length_on_line():
    """Degenerate segment (point) lying on the other segment."""
    p1 = (0.0, 0.0)
    p2 = (10.0, 10.0)
    q1 = (5.0, 5.0)
    q2 = (5.0, 5.0)
    assert segments_intersect(p1, p2, q1, q2) is True


def test_degenerate_zero_length_off_line():
    """Degenerate segment (point) not lying on the other segment."""
    p1 = (0.0, 0.0)
    p2 = (10.0, 0.0)
    q1 = (5.0, 5.0)
    q2 = (5.0, 5.0)
    assert segments_intersect(p1, p2, q1, q2) is False


def test_skew_non_intersecting():
    """Segments that would intersect if extended, but segments do not cross."""
    p1 = (0.0, 0.0)
    p2 = (4.0, 4.0)
    q1 = (5.0, 0.0)
    q2 = (5.0, 2.0)
    assert segments_intersect(p1, p2, q1, q2) is False


def test_ccw_orientations():
    """Verifies counterclockwise, clockwise, and collinear cross product."""
    a = (0.0, 0.0)
    b = (5.0, 0.0)
    c_ccw = (5.0, 5.0)
    c_cw = (5.0, -5.0)
    c_collinear = (10.0, 0.0)

    assert ccw(a, b, c_ccw) > 0
    assert ccw(a, b, c_cw) < 0
    assert abs(ccw(a, b, c_collinear)) < 1e-6


def test_point_in_rect():
    """Tests point-in-rect boundary, interior, and exterior conditions."""
    rect = (10, 20, 100, 80)
    assert point_in_rect((50, 50), rect) is True
    assert point_in_rect((10, 20), rect) is True
    assert point_in_rect((100, 80), rect) is True
    assert point_in_rect((5, 50), rect) is False
    assert point_in_rect((105, 50), rect) is False
    assert point_in_rect((50, 10), rect) is False
    assert point_in_rect((50, 90), rect) is False


def test_point_in_circle():
    """Tests point-in-circle with tolerance margin."""
    center = (100.0, 100.0)
    assert point_in_circle((100.0, 100.0), center, 50.0) is True
    assert point_in_circle((150.0, 100.0), center, 50.0) is True
    assert point_in_circle((160.0, 100.0), center, 50.0) is False
    # With tolerance margin
    assert point_in_circle((160.0, 100.0), center, 50.0, margin=15.0) is True


def test_euclidean_distance():
    """Tests 2D euclidean distance calculation."""
    assert euclidean_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert euclidean_distance((10.0, 10.0), (10.0, 10.0)) == pytest.approx(0.0)
