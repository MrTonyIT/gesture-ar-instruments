"""
tests/test_packaging.py
=======================
Unit tests for Python packaging, setuptools metadata, flat-layout module discovery,
and entry point CLI execution.
"""

import importlib.metadata
import subprocess
import sys
from pathlib import Path


def test_pyproject_flat_layout_modules_declared():
    """Verifies that pyproject.toml explicitly lists all top-level modules to avoid setuptools errors."""
    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml must exist at repo root"

    text = pyproject_path.read_text(encoding="utf-8")
    assert "[tool.setuptools]" in text
    assert "py-modules" in text

    expected_modules = [
        "main",
        "vision_tracker",
        "gesture_engine",
        "instruments",
        "audio_engine",
        "geometry",
        "benchmark_filters",
    ]
    for mod in expected_modules:
        assert f'"{mod}"' in text or f"'{mod}'" in text, f"Module {mod} missing in pyproject.toml py-modules"


def test_pyproject_dependencies():
    """Verifies pyproject.toml dependencies: unified opencv-contrib-python, no unused scipy."""
    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"

    text = pyproject_path.read_text(encoding="utf-8").lower()

    assert "opencv-contrib-python" in text, "Must depend on opencv-contrib-python (single unified OpenCV wheel)"
    assert "opencv-python>=" not in text and "opencv-python==" not in text, "Must not dual-declare opencv-python"
    assert "scipy" not in text, "Unused scipy dependency must not be present"
    assert "mediapipe" in text, "mediapipe dependency required"
    assert "sounddevice" in text, "sounddevice dependency required"
    assert "numpy" in text, "numpy dependency required"


def test_cli_entry_point_help():
    """Verifies main module executes --help cleanly without opening camera or audio hardware."""
    res = subprocess.run(
        [sys.executable, "-m", "main", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res.returncode == 0, f"--help failed with code {res.returncode}: {res.stderr}"
    assert "Gesture AR Instruments" in res.stdout
    assert "--camera-id" in res.stdout
    assert "--width" in res.stdout
    assert "--height" in res.stdout


def test_installed_package_metadata():
    """Verifies installed wheel/package metadata when installed in the current environment."""
    try:
        dist = importlib.metadata.distribution("gesture-ar-instruments")
    except importlib.metadata.PackageNotFoundError:
        try:
            dist = importlib.metadata.distribution("gesture_ar_instruments")
        except importlib.metadata.PackageNotFoundError:
            dist = None

    if dist is not None:
        assert dist.version == "1.0.0"
        entry_points = [ep.name for ep in dist.entry_points if ep.group == "console_scripts"]
        assert "gesture-ar" in entry_points, "Entry point gesture-ar must be registered"
