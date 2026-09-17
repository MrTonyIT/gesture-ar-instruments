# 🎸 Gesture AR Instruments

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/MediaPipe-Hands-007ACC?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe" />
  <img src="https://img.shields.io/badge/Audio-Procedural%20Synth%20(32%20Voices)-00E676?style=for-the-badge&logo=speaker&logoColor=white" alt="Audio Engine" />
  <img src="https://img.shields.io/badge/Tests-73%20Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <strong>Research-Grade Spatial Augmented Reality Musical Instruments (Air Guitar & Tabletop Piano) powered by MediaPipe Computer Vision, 1€ Adaptive Tracking Filters, Procedural Harmonic Synthesis, and Real-Time Telemetry HUD.</strong>
</p>

---

## 🌟 Overview

**Gesture AR Instruments** transforms standard RGB webcams into low-latency augmented reality musical instruments. Without requiring specialized hardware, wearable gloves, or depth sensors, users can sculpt virtual instruments in 3D mid-air, manipulate them with natural dual-finger touch, snap guitar components together via magnetic plasma physics, strum authentic 9-chord progressions, or dock a 24-key piano onto a physical desk surface.

A 73-test automated suite covers tracking filters, geometry, synthesis, instrument interactions, packaging, and major state-machine paths.

---

## 🚀 Key Systems & Features

### 🎸 1. Authentic 9-Chord AR Guitar
- **9 Supported Chords**: Full open and barre guitar voicings for `C`, `G`, `D`, `A`, `E`, `Am`, `Em`, `Dm`, and `F`.
- **Physical Muted String Suppression**: Low strings that are muted in authentic chord voicings (e.g. 6th string on `C` and `Am`, 5th and 6th on `D` and `Dm`) produce zero sound, allocate **0 synthesis voices**, and display an intuitive `"X"` indicator near the bridge with dampened vibration.
- **Top Fretboard Selector HUD**: 9 interactive chord boxes `[1]` through `[9]` placed along the top edge. Chords can be switched via:
  1. **Direct Touch**: Reaching into the chord box with any fingertip.
  2. **Left-Hand Extended Fingers**: 1 finger = `C`, 2 fingers = `G`, 3 fingers = `Am`, 4 fingers = `Em`.
  3. **Keyboard Hotkeys**: Numbers `1`–`9` or letter keys `C`, `G`, `D`, `A`, `E`, `F`.
- **Dual-Finger Sculpt & Snap**: Touch thumb and index fingertips of both hands together, pull apart to project dynamic laser rails, sculpt the soundbox circle, and bring within 145px to trigger magnetic plasma assembly.

### 🎹 2. Ergonomic Desk-Surface Piano
- **Desk-Docked Architecture**: Grounded to the bottom 25% of the frame ($Y \in [0.72, 0.96]$) so wrists rest comfortably on the tabletop, preventing "gorilla arm" fatigue.
- **24 Keys (2 Octaves)**: 14 white natural keys and 10 black accidentals.
- **Downward Velocity Gating ($v_y \ge 60\text{ px/s}, \bar{v}_y \ge 0.08$)**: Differentiates deliberate downward key strikes from resting or lateral finger motion, eliminating ghost notes.
- **Geometric Collision Precedence**: Evaluates black keys before white keys with boundary padding to ensure sharp accidentals are never missed.
- **Hover Suppression**: Tracks held fingers per key to prevent unintended re-strikes while fingers hover or rest on the surface.

### 📊 3. Developer Diagnostics HUD & Quality Profiles
- **Real-Time Telemetry HUD (`F3`, `Tab`, or `` ` ``)**: Displays high-resolution timing bars and millisecond metrics for:
  - `Cam I/O`: Camera capture thread latency.
  - `Track Snapshot`: Latency of hand tracking queue ingestion.
  - `AI Inference`: Raw MediaPipe neural network latency and stale frame skip counter.
  - `Gesture Machine`: State evaluation and touch collision processing time.
  - `Render Loop`: Vectorized alpha blending and HUD drawing time.
  - `Audio Bus`: Active concurrent voice count and peak amplitude with soft-limiter indicator.
- **Visual Quality Profiles (`V` key)**: Cycle between `HIGH` (Model Complexity 1, 60 FPS target), `BALANCED` (Complexity 0), and `LOW` for maximum frame rate on low-power hardware.

### 🎛️ 4. Pluggable Tracking Filter Hierarchy
Choose between four filtering strategies in `vision_tracker.py`:
- **`one_euro` (Default)**: Casiez et al. (CHI 2012) adaptive cutoff filter ($f_{c,\min} = 1.0\text{ Hz}, \beta = 30.0$). Operates in Normalized Device Coordinates (NDC) to attenuate stationary jitter ($1.77\text{ px}$ vs $5.90\text{ px}$ raw at 1080p) while reducing the smoothness/responsiveness trade-off by increasing cutoff during faster motion (sub-frame $7.8\text{ ms}$ phase lag measured in synthetic benchmark).
- **`deadband`**: Dual-threshold velocity-adaptive deadband ($\delta_{\text{static}} = 0.0025, \delta_{\text{dynamic}} = 0.0080$) with instantaneous step response ($0.0\text{ ms}$ delay relative to step frame).
- **`ema`**: First-order exponential moving average ($\alpha = 0.65$).
- **`raw`**: Unfiltered camera pass-through for latency baseline comparisons.

---

## 🔬 Empirical Filter Benchmark Results

All figures below are directly computed from synthetic deterministic test trajectories generated by `benchmark_filters.py` and saved to `benchmarks/filter_benchmark_results.csv`:

| Trajectory | Filter | RMSE (NDC) | MAE (NDC) | Stat. Jitter (1080p) | Resid. Noise (1080p) | Phase Lag (ms) | Detected Lag (Frames) | Settling Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stationary Noisy (3s)** | RawFilter | 0.00307 | 0.00243 | 5.90 px | 6.17 px | N/A | 0 | N/A |
| | EMAFilter ($\alpha=0.65$) | 0.00208 | 0.00166 | 3.98 px | 3.39 px | N/A | 0 | N/A |
| | DeadbandFilter | 0.00271 | 0.00208 | 5.16 px | 4.97 px | N/A | 0 | N/A |
| | **OneEuroFilter** | **0.00094** | **0.00074** | **1.77 px** | **0.84 px** | N/A | 0 | N/A |
| **Constant Velocity (3s)**| RawFilter | 0.00197 | 0.00160 | N/A | 3.90 px | N/A | 0 | N/A |
| | EMAFilter | 0.00222 | 0.00185 | N/A | 2.19 px | N/A | 0 | N/A |
| | DeadbandFilter | 0.00269 | 0.00219 | N/A | 4.66 px | N/A | 0 | N/A |
| | OneEuroFilter | 0.00304 | 0.00278 | N/A | 1.83 px | N/A | 0 | N/A |
| **Sinusoidal 1.5Hz (4s)** | RawFilter | 0.00202 | 0.00163 | N/A | 3.92 px | 0.1 ms | 0 (< 16.7 ms) | N/A |
| | EMAFilter | 0.01204 | 0.01075 | N/A | 3.49 px | 9.0 ms | 1 (16.7 ms) | N/A |
| | **DeadbandFilter** | **0.00223** | **0.00180** | N/A | 4.20 px | **0.1 ms** | 0 (< 16.7 ms) | N/A |
| | OneEuroFilter | 0.01246 | 0.00933 | N/A | 6.69 px | 7.8 ms | 0 (< 16.7 ms) | N/A |
| **Step Response (2s)** | **RawFilter** | **0.00107** | **0.00083** | N/A | 1.95 px | N/A | 0 | **0.0 ms** |
| | EMAFilter | 0.01035 | 0.00192 | N/A | 16.03 px | N/A | 0 | 50.0 ms |
| | **DeadbandFilter** | **0.00087** | **0.00079** | N/A | **0.16 px** | N/A | 0 | **0.0 ms** |
| | OneEuroFilter | 0.00440 | 0.00085 | N/A | 7.84 px | N/A | 0 | 33.3 ms |
| **Rapid Strum 4.0Hz (3s)** | RawFilter | 0.00204 | 0.00164 | N/A | 2.17 px | 0.0 ms | 0 (< 16.7 ms) | N/A |
| | EMAFilter | 0.02188 | 0.01968 | N/A | 7.18 px | 8.1 ms | 0 (< 16.7 ms) | N/A |
| | DeadbandFilter | 0.00217 | 0.00172 | N/A | 2.37 px | 0.0 ms | 0 (< 16.7 ms) | N/A |
| | OneEuroFilter | 0.02640 | 0.02014 | N/A | 13.10 px | 9.3 ms | 1 (16.7 ms) | N/A |

> [!NOTE]
> Phase lag is evaluated via continuous Fourier harmonic analysis. Discrete cross-correlation at 60 FPS has an integer quantization resolution of $\pm 8.3\text{ ms}$; a measurement of 0 detected frames bounds delay to $< 16.7\text{ ms}$ (sub-frame). Settling latency is measured relative to the discrete step event frame ($\Delta t = 16.7\text{ ms}$ resolution). Detailed derivations and analysis are documented in [`docs/RESEARCH.md`](docs/RESEARCH.md).

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Hardware Layer
        CAM[Threaded Camera Capture\nIndependent Producer Thread]
        AUDIO_OUT[Sound Card Output\nCallback Block: 256 Frames (5.8 ms)]
    end

    subgraph Tracking & Processing
        CAM --> WORKER[AsyncHandTracker Worker\nMediaPipe Hands Worker Thread]
        WORKER --> FILTER[Pluggable BaseFilter\n1€ / Deadband / EMA / Raw]
        FILTER --> KINEMATICS[Kinematic Dead-Reckoning\nPredictive Extrapolation]
    end

    subgraph Logic & State Machine
        KINEMATICS --> GE[Gesture Engine\nState Machine & Collision Geometry]
        GE --> PIANO[Desk Piano Engine\nVelocity Gating & Black Key Precedence]
        GE --> GUITAR[Virtual 3D Guitar\n9 Chords & Segment Strumming]
    end

    subgraph Synthesis & Display
        PIANO --> SYNTH[Procedural Audio Engine\nMax 32 Active Voices + Soft Limiter]
        GUITAR --> SYNTH
        SYNTH --> AUDIO_OUT
        GE --> HUD[Cyber HUD & Diagnostics Telemetry]
        HUD --> DISPLAY[Window Display\n60 FPS Vectorized Overlay]
    end
```

---

## 🛠️ Installation & Quick Start

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/MrTonyIT/gesture-ar-instruments.git
cd gesture-ar-instruments

# Create virtual environment
python -m venv .venv

# Activate on Windows:
.\.venv\Scripts\Activate.ps1
# Or on Linux / macOS:
source .venv/bin/activate

# Install core runtime dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development and test tooling
pip install -r requirements-dev.txt

# (Recommended for CI / Research Reproduction) Install with CI-verified direct dependency constraints:
# pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt

# Or install as an editable package with CLI entry point:
pip install .
```

> [!NOTE]
> Standalone distributions installed via `pip install .` automatically synthesize procedural low-poly guitar graphics via `create_procedural_blocky_guitar()` when run outside the repository root where `assets/` is not co-located.

### 2. Launch the Application
```bash
# Run via console entry point:
gesture-ar

# Or run via Python module:
python main.py

# Specify camera device and window size
python main.py --camera-id 0 --width 1920 --height 1080
```

### 3. Running the Test Suite & Benchmarks
```bash
# Run 69 headless unit tests
python -m pytest -q

# Run tracking filter benchmarks and generate CSV results
python benchmark_filters.py

# Check code formatting and linting
python -m ruff check .
```

---

## ⌨️ Controls & Gestures Guide

| Action | Control / Gesture | Target Instrument | Description |
| :--- | :--- | :---: | :--- |
| **Spawn Piano** | Both hands 'L' shape (1.2s) | Tabletop Piano | Locks 2-octave piano to desk surface |
| **Play Piano** | Tap desk surface downward ($v_y > 60\text{ px/s}$) | Tabletop Piano | Realistic touch strikes with hover suppression |
| **Sculpt Guitar** | Touch thumbs/indices & pull apart | Air Guitar | Stretches dynamic laser rails for guitar neck |
| **Sculpt Soundbox**| Right hand pinch & expand | Air Guitar | Sculpt circular body (1.5s hold to freeze) |
| **Snap Fusion** | Bring neck & soundbox $< 145\text{px}$ | Air Guitar | Merges components into playable 3D guitar |
| **Select Chord** | Keys `1`–`9` or `C, G, D, A, E, F` | Air Guitar | Instant authentic 9-chord switching |
| **Fingertip Chord**| Raise 1–4 fingers on Left Hand | Air Guitar | 1=`C`, 2=`G`, 3=`Am`, 4=`Em` |
| **Touch Chord** | Reach into top chord boxes `[1]`–`[9]` | Air Guitar | Direct AR touch selection |
| **Strum Guitar** | Thumb (4) or Index (8) downward stroke | Air Guitar | Strums 6 projected strings with acoustic resonance |
| **Diagnostics** | `F3`, `Tab`, or `` ` `` | Telemetry HUD | Toggles developer diagnostics panel |
| **Quality Mode** | `V` key | Application | Cycles `HIGH` $\to$ `BALANCED` $\to$ `LOW` |
| **Reset State** | Hold `[RESET]` button (0.7s) | Application | Resets state machine to `IDLE` |
| **Exit** | Hold `[EXIT]` button (3.0s) or `ESC` | Application | Gracefully releases all hardware resources |

---

## 📂 Project Structure

```
gesture-ar-instruments/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated multi-version CI (Python 3.10, 3.11)
├── assets/
│   └── guitar_blocky.png        # Transparent 3D low-poly guitar sprite asset
├── benchmarks/
│   └── filter_benchmark_results.csv # Empirical benchmark dataset (Jitter, Lag, Error)
├── docs/
│   ├── RESEARCH.md              # Research paper: filter math & empirical benchmark data
│   └── USER_STUDY_PROTOCOL.md   # Formal 24-participant usability evaluation methodology
├── tests/
│   ├── test_async_tracking.py   # Asynchronous timestamping & kinematic extrapolation
│   ├── test_audio.py            # Synthesis engine, voicings, polyphony caps
│   ├── test_filters.py          # 1€, Deadband, EMA, Raw filter algorithms
│   ├── test_geometry.py         # Line-segment intersection & point-in-rect primitives
│   ├── test_gestures.py         # State machine transitions, timeouts, sculpt & fusion
│   ├── test_guitar.py           # 9-chord voicings, muted strings, strum debounce
│   ├── test_hud.py              # Multi-resolution HUD layout (720p - 4K) & diagnostics
│   ├── test_packaging.py        # PEP 517 build, entry points, metadata checks
│   ├── test_piano.py            # Velocity gating, black key precedence, hover suppression
│   ├── test_quality_profiles.py # Explicit HIGH, BALANCED, LOW profile switching
│   └── test_ui_geometry.py      # Corner reticle non-duplicate line validation
├── audio_engine.py              # Low-latency procedural synth with 32-voice cap & soft limiter
├── benchmark_filters.py         # Empirical synthetic trajectory benchmark harness
├── geometry.py                  # Robust 2D computational geometry primitives
├── gesture_engine.py            # Spatial state machine, sculpting & touch logic
├── instruments.py               # Tabletop Piano & Virtual Guitar physics & hit detection
├── main.py                      # Application orchestrator, ThreadedCamera & Diagnostics HUD
├── vision_tracker.py            # MediaPipe Hands with pluggable BaseFilter hierarchy
├── pyproject.toml               # PEP 517/621 project configuration & tool configs
├── constraints.txt              # CI-verified direct dependency constraints for reproduction
├── requirements.txt             # Core runtime dependencies (flexible declarations)
├── requirements-dev.txt         # Development & test tooling (pytest, ruff)
├── LICENSE                      # MIT Open Source License
└── README.md                    # Project documentation
```

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
