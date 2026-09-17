"""
tests/test_quality_profiles.py
==============================
Unit tests verifying explicit Quality Profiles (HIGH, BALANCED, LOW):
- Configuration attributes and defaults
- Dynamic switching and tracker model complexity synchronization
- Measurable rendering pipeline differences (particles, glow, flash)
"""

import numpy as np

from main import QUALITY_PROFILES, GestureARApp


def test_quality_profiles_definitions():
    """Verifies that HIGH, BALANCED, and LOW quality profiles are defined with distinct attributes."""
    assert "HIGH" in QUALITY_PROFILES
    assert "BALANCED" in QUALITY_PROFILES
    assert "LOW" in QUALITY_PROFILES

    high = QUALITY_PROFILES["HIGH"]
    balanced = QUALITY_PROFILES["BALANCED"]
    low = QUALITY_PROFILES["LOW"]

    # Model complexity: HIGH is full (1), BALANCED/LOW are lite (0)
    assert high.model_complexity == 1
    assert balanced.model_complexity == 0
    assert low.model_complexity == 0

    # Max particles: monotonically decreasing
    assert high.max_particles == 18
    assert balanced.max_particles == 8
    assert low.max_particles == 0

    # Glow effects
    assert high.enable_glow_effects is True
    assert balanced.enable_glow_effects is True
    assert low.enable_glow_effects is False

    # Screen shockwave flash
    assert high.enable_shockwave_flash is True
    assert balanced.enable_shockwave_flash is False
    assert low.enable_shockwave_flash is False

    # Oscilloscope resolution
    assert high.oscilloscope_points > balanced.oscilloscope_points > low.oscilloscope_points


def test_app_quality_profile_switching():
    """Verifies that set_quality_profile switches config and synchronizes tracker model complexity."""
    app = GestureARApp(start_threads=False, quality_profile="HIGH")
    try:
        assert app.quality_profile == "HIGH"
        assert app.quality_config.max_particles == 18
        assert app.async_tracker.model_complexity == 1

        app.set_quality_profile("BALANCED")
        assert app.quality_profile == "BALANCED"
        assert app.quality_config.max_particles == 8
        assert app.async_tracker.model_complexity == 0

        app.set_quality_profile("LOW")
        assert app.quality_profile == "LOW"
        assert app.quality_config.max_particles == 0
        assert app.quality_config.enable_glow_effects is False
        assert app.async_tracker.model_complexity == 0

        # Invalid profile defaults to HIGH gracefully
        app.set_quality_profile("NON_EXISTENT")
        assert app.quality_profile == "HIGH"
        assert app.async_tracker.model_complexity == 1
    finally:
        app.shutdown()


def test_rendering_effects_under_profiles():
    """Verifies that drawing shockwave and oscilloscope executes cleanly across profiles."""
    app = GestureARApp(start_threads=False, quality_profile="HIGH")
    try:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # In HIGH: shockwave flash and 18 sparks render without exception
        app.set_quality_profile("HIGH")
        app._draw_shockwave(frame.copy(), center=(640, 360), progress=0.2)

        # In BALANCED: flash skipped, 8 sparks
        app.set_quality_profile("BALANCED")
        app._draw_shockwave(frame.copy(), center=(640, 360), progress=0.2)

        # In LOW: zero sparks, zero flash
        app.set_quality_profile("LOW")
        app._draw_shockwave(frame.copy(), center=(640, 360), progress=0.2)

        # Oscilloscope across all profiles
        for prof in ["HIGH", "BALANCED", "LOW"]:
            app.set_quality_profile(prof)
            app._draw_audio_oscilloscope(frame.copy(), x=100, y=100, w=150, h=40)
    finally:
        app.shutdown()
