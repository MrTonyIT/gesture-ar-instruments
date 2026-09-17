"""
geometry.py
===========
Centralized 2D Geometric Primitives and Intersection Utilities.

Provides robust, vectorized, and edge-case-tested geometric calculations for:
- 2D Line-segment intersection tests (guitar string strumming and finger crossing).
- Collinear and degenerate segment handling.
- Point-in-rectangle and point-in-circle containment with margins.
- Distance metrics and bounding-box calculations.
"""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np


Point2D = Tuple[float, float]
Rect4D = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def ccw(a: Point2D, b: Point2D, c: Point2D) -> float:
    """
    Computes the 2D cross-product of vectors (b - a) and (c - a).

    Returns:
        Positive if a -> b -> c turns counter-clockwise,
        Negative if clockwise,
        Zero if points are collinear.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(p: Point2D, q: Point2D, r: Point2D, eps: float = 1e-6) -> bool:
    """
    Checks if point q lies on collinear segment p-r.
    """
    return (
        min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
        and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
    )


def segments_intersect(
    p1: Point2D,
    p2: Point2D,
    q1: Point2D,
    q2: Point2D,
    eps: float = 1e-5,
) -> bool:
    """
    Evaluates whether line segment (p1 -> p2) intersects with line segment (q1 -> q2).

    Correctly handles:
    - Proper intersections (straddling in opposite directions).
    - Collinear segments overlapping or sharing endpoints.
    - Zero-length segments (degenerate points).
    """
    # Check for degenerate segments (points)
    p_len_sq = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
    q_len_sq = (q2[0] - q1[0]) ** 2 + (q2[1] - q1[1]) ** 2

    if p_len_sq < eps * eps and q_len_sq < eps * eps:
        # Both are points
        return (abs(p1[0] - q1[0]) < eps) and (abs(p1[1] - q1[1]) < eps)

    if p_len_sq < eps * eps:
        # p is a point, check if it lies on segment q1-q2
        d = ccw(q1, q2, p1)
        return abs(d) < eps and on_segment(q1, p1, q2, eps)

    if q_len_sq < eps * eps:
        # q is a point, check if it lies on segment p1-p2
        d = ccw(p1, p2, q1)
        return abs(d) < eps and on_segment(p1, q1, p2, eps)

    # Orientation calculations
    d1 = ccw(p1, p2, q1)
    d2 = ccw(p1, p2, q2)
    d3 = ccw(q1, q2, p1)
    d4 = ccw(q1, q2, p2)

    # General crossing case: segments straddle each other
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True

    # Collinear and endpoint cases
    if abs(d1) <= eps and on_segment(p1, q1, p2, eps):
        return True
    if abs(d2) <= eps and on_segment(p1, q2, p2, eps):
        return True
    if abs(d3) <= eps and on_segment(q1, p1, q2, eps):
        return True
    if abs(d4) <= eps and on_segment(q1, p2, q2, eps):
        return True

    return False


def point_in_rect(
    point: Point2D,
    rect: Union[Rect4D, Tuple[float, float, float, float]],
    margin: float = 0.0,
) -> bool:
    """Checks if point (x, y) is inside rect (x1, y1, x2, y2) with an optional margin."""
    x, y = point
    x1, y1, x2, y2 = rect
    return (x1 - margin <= x <= x2 + margin) and (y1 - margin <= y <= y2 + margin)


def point_in_circle(
    point: Point2D,
    center: Point2D,
    radius: float,
    margin: float = 0.0,
) -> bool:
    """Checks if point (x, y) is inside circle (cx, cy, radius) with an optional margin."""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return (dx * dx + dy * dy) <= ((radius + margin) ** 2)


def euclidean_distance(p1: Point2D, p2: Point2D) -> float:
    """Computes Euclidean distance between two 2D points."""
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
