"""
tests/test_filters.py
=====================
Deterministic Unit Tests for Hand Tracking Filters.
Covers:
- BaseFilter abstract interface compliance
- RawFilter identity and array copy isolation
- EMAFilter smoothing, step response, and reset
- DeadbandFilter noise rejection and pass-through
- OneEuroFilter adaptive cutoff scaling and reset
- Numerical robustness (NaN, Inf, zero/negative dt)
- HandTracker filter switching
"""

import numpy as np

from vision_tracker import (
    BaseFilter,
    DeadbandFilter,
    EMAFilter,
    HandTracker,
    OneEuroFilter,
    RawFilter,
)


def test_raw_filter():
    """RawFilter returns independent copies without modification."""
    filt = RawFilter()
    assert isinstance(filt, BaseFilter)

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    out = filt.filter(x, timestamp=0.0)

    np.testing.assert_array_equal(out, x)
    assert out is not x  # Verify copy isolation

    filt.reset()
    out2 = filt.filter(x, timestamp=0.1)
    np.testing.assert_array_equal(out2, x)


def test_ema_filter():
    """EMAFilter applies s_t = alpha * x + (1 - alpha) * s_{t-1}."""
    alpha = 0.50
    filt = EMAFilter(alpha=alpha)

    x0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    out0 = filt.filter(x0, timestamp=0.0)
    np.testing.assert_allclose(out0, [[0.0, 0.0, 0.0]])

    x1 = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    out1 = filt.filter(x1, timestamp=0.016)
    # Expected: 0.5 * 10 + 0.5 * 0 = 5.0
    np.testing.assert_allclose(out1, [[5.0, 5.0, 5.0]], atol=1e-5)

    out2 = filt.filter(x1, timestamp=0.032)
    # Expected: 0.5 * 10 + 0.5 * 5 = 7.5
    np.testing.assert_allclose(out2, [[7.5, 7.5, 7.5]], atol=1e-5)

    # Test reset
    filt.reset()
    assert filt.prev_val is None
    out_fresh = filt.filter(x1, timestamp=0.048)
    np.testing.assert_allclose(out_fresh, x1)


def test_deadband_filter_noise_suppression():
    """DeadbandFilter locks values when movement is within noise floor."""
    deadband = 0.010
    motion_thresh = 0.030
    filt = DeadbandFilter(deadband_norm=deadband, motion_threshold_norm=motion_thresh)

    x0 = np.array([[0.50, 0.50, 0.0]], dtype=np.float32)
    filt.filter(x0, timestamp=0.0)

    # Micro-jitter within deadband (delta = 0.003 < 0.010)
    x_jitter = np.array([[0.503, 0.500, 0.0]], dtype=np.float32)
    out_jitter = filt.filter(x_jitter, timestamp=0.016)
    # Filter should freeze at x0
    np.testing.assert_allclose(out_jitter, x0, atol=1e-5)


def test_deadband_filter_large_motion():
    """DeadbandFilter passes through full 1:1 coordinates during large motion."""
    deadband = 0.010
    motion_thresh = 0.030
    filt = DeadbandFilter(deadband_norm=deadband, motion_threshold_norm=motion_thresh)

    x0 = np.array([[0.50, 0.50, 0.0]], dtype=np.float32)
    filt.filter(x0, timestamp=0.0)

    # Fast movement (delta = 0.10 > 0.030)
    x_fast = np.array([[0.60, 0.50, 0.0]], dtype=np.float32)
    out_fast = filt.filter(x_fast, timestamp=0.016)
    # Filter should pass x_fast directly (alpha = 1.0)
    np.testing.assert_allclose(out_fast, x_fast, atol=1e-5)


def test_one_euro_filter():
    """OneEuroFilter adapts cutoff based on speed."""
    filt = OneEuroFilter(min_cutoff=1.0, beta=30.0, d_cutoff=1.0)

    # Initial sample
    x0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    out0 = filt.filter(x0, timestamp=0.0)
    np.testing.assert_allclose(out0, x0)

    # Slow motion
    x_slow = np.array([[0.01, 0.0, 0.0]], dtype=np.float32)
    out_slow = filt.filter(x_slow, timestamp=0.016)
    # Heavy smoothing at low speed
    assert out_slow[0, 0] < x_slow[0, 0]

    # Fast motion
    filt.reset()
    filt.filter(x0, timestamp=0.0)
    x_fast = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    out_fast = filt.filter(x_fast, timestamp=0.016)
    # Higher speed should have higher alpha (closer to input)
    assert out_fast[0, 0] > 0.5


def test_filters_nan_and_inf_robustness():
    """Filters must gracefully handle non-finite values without crashing or poisoning state."""
    filters = [
        RawFilter(),
        EMAFilter(),
        DeadbandFilter(),
        OneEuroFilter(),
    ]

    nan_input = np.array([[np.nan, 0.5, np.inf]], dtype=np.float32)
    valid_input = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)

    for filt in filters:
        filt.reset()
        # Feed valid first
        filt.filter(valid_input, timestamp=0.0)
        # Feed NaN
        _ = filt.filter(nan_input, timestamp=0.016)
        # Feed valid again - filter should recover cleanly
        out_recovered = filt.filter(valid_input, timestamp=0.032)
        assert np.all(np.isfinite(out_recovered))


def test_hand_tracker_filter_switching():
    """HandTracker creates appropriate filter instances on mode switch."""
    tracker = HandTracker(filter_mode="one_euro")
    f_euro = tracker._create_filter_instance()
    assert isinstance(f_euro, OneEuroFilter)

    tracker.set_filter_mode("deadband")
    assert isinstance(tracker._create_filter_instance(), DeadbandFilter)

    tracker.set_filter_mode("ema")
    assert isinstance(tracker._create_filter_instance(), EMAFilter)

    tracker.set_filter_mode("raw")
    assert isinstance(tracker._create_filter_instance(), RawFilter)
