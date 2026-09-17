# Empirical Research & Systems Engineering: Gesture AR Instruments

**Author:** MrTonyIT & Antigravity Advanced Systems  
**Date:** September 2026  
**Repository Branch:** `research-upgrade-v2`  
**Evaluation Protocol:** Synthetic Deterministic Trajectory Benchmarks (`benchmark_filters.py`) & Headless Pytest Suite (`tests/`)

---

## 1. Executive Abstract

Spatial augmented reality (AR) musical instruments face a classical human-computer interaction (HCI) trade-off: **high-frequency tremor attenuation versus dynamic phase lag**. When tracking bare hands through monocular RGB webcams targeting 60 FPS, landmark jitter degrades virtual string plucking and piano key strikes. Traditional low-pass filtering eliminates tremor but introduces phase lag, causing users to perceive latency and miss rhythmic downbeats.

This research paper documents the engineering design of `gesture-ar-instruments`. We present:
1. **A formal evaluation of four kinematic tracking filter architectures** (Raw Pass-Through, Exponential Moving Average, Adaptive Deadband, and 1€ Filter).
2. **Deterministic empirical benchmarks** across five motion profiles recorded in `benchmarks/filter_benchmark_results.csv`.
3. **An authentic 9-chord AR guitar system** ($C, G, D, A, E, Am, Em, Dm, F$) with physical muted-string suppression ($0$ voice allocations).
4. **An ergonomic desk-docked piano** with downward velocity gating ($v_y \ge 60\text{ px/s}, \bar{v}_y \ge 0.08$), black-key geometric precedence, and hover suppression.
5. **A thread-safe procedural audio synthesis engine** with decoupled DSP rendering and an invariant maximum of 32 concurrent voices.

---

## 2. Kinematic Filter Mathematics & Architectures

Tracking filters inherit from `BaseFilter(ABC)` in `vision_tracker.py`. Coordinates are normalized such that $x, y \in [0, 1]$ (mirrored horizontal and vertical image axes), while $z \in \mathbb{R}$ represents relative depth from the wrist origin (scaled roughly to image width, negative toward the camera, and not bounded to $[0, 1]$).

### 2.1 Raw Pass-Through (`RawFilter`)
Serves as the empirical baseline (algorithmic pass-through baseline with no smoothing-state delay). Output is identical to input:
$$\hat{\mathbf{x}}_k = \mathbf{x}_k$$

### 2.2 Exponential Moving Average (`EMAFilter`)
Applies first-order infinite impulse response (IIR) smoothing with a fixed smoothing factor $\alpha \in (0, 1]$:
$$\hat{\mathbf{x}}_k = \alpha \mathbf{x}_k + (1 - \alpha) \hat{\mathbf{x}}_{k-1}$$
While computationally trivial, a static $\alpha = 0.65$ trades off high-frequency noise rejection against phase lag during rapid motion.

### 2.3 Velocity-Adaptive Deadband (`DeadbandFilter`)
Constructed to minimize phase lag during motion while freezing static micro-tremor:
$$\Delta = \|\mathbf{x}_k - \hat{\mathbf{x}}_{k-1}\|_2$$
$$\hat{\mathbf{x}}_k = \begin{cases} 
\mathbf{x}_k & \text{if } \Delta > \delta_{\text{dynamic}} \\
\hat{\mathbf{x}}_{k-1} + \alpha_{\text{glide}} (\mathbf{x}_k - \hat{\mathbf{x}}_{k-1}) & \text{if } \delta_{\text{static}} < \Delta \le \delta_{\text{dynamic}} \\
\hat{\mathbf{x}}_{k-1} & \text{if } \Delta \le \delta_{\text{static}}
\end{cases}$$
Default engine parameters:
- $\delta_{\text{static}} = 0.0025\text{ NDC}$ ($\approx 3.2\text{ px}$ on 720p)
- $\delta_{\text{dynamic}} = 0.0080\text{ NDC}$ ($\approx 10.2\text{ px}$ on 720p)
- Transition smoothing: Cubic Hermite curve $\alpha_{\text{glide}} = 3t^2 - 2t^3$ where $t = \frac{\Delta - \delta_{\text{static}}}{\delta_{\text{dynamic}} - \delta_{\text{static}}}$.

### 2.4 One Euro Filter (`OneEuroFilter`)
Based on Casiez, Roussel, and Vogel (CHI 2012), the 1€ filter dynamically adapts its cutoff frequency $f_c$ based on the rate of change (velocity $\dot{\mathbf{x}}$):
$$T_e = \text{clip}(t_k - t_{k-1}, 10^{-4}\text{ s}, 1.0\text{ s})$$
$$\dot{\mathbf{x}}_k = \frac{\mathbf{x}_k - \hat{\mathbf{x}}_{k-1}}{T_e}$$
$$\hat{\dot{\mathbf{x}}}_k = \text{EMA}_{\alpha_d}(\dot{\mathbf{x}}_k, \hat{\dot{\mathbf{x}}}_{k-1}) \quad \text{with } \alpha_d = \frac{1}{1 + \frac{1}{2\pi f_{c, d} T_e}}$$
$$f_c = f_{c, \min} + \beta \|\hat{\dot{\mathbf{x}}}_k\|_2$$
$$\alpha = \frac{1}{1 + \frac{1}{2\pi f_c T_e}}$$
$$\hat{\mathbf{x}}_k = \alpha \mathbf{x}_k + (1 - \alpha) \hat{\mathbf{x}}_{k-1}$$

**Parameter Configuration & Coordinate System Scale:**
- $f_{c, \min} = 1.0\text{ Hz}$ (smoothness at rest)
- $\beta = 30.0$ (speed coefficient calibrated for Normalized Device Coordinates $[0, 1]$)
- $f_{c, d} = 1.0\text{ Hz}$ (derivative cutoff)
- $f_{c, \max} = \text{None}$ (unbounded scaling)
- $T_e$ clamp: $[10^{-4}\text{ s}, 1.0\text{ s}]$ preventing numerical instability during irregular camera frame arrivals.

> [!NOTE]
> **Why $\beta = 30.0$ in NDC vs $\beta = 0.007$ in pixel space:**  
> The 1€ filter speed coefficient $\beta$ multiplies tracking velocity. In pixel coordinates, fingertip speed often reaches $500\text{--}1500\text{ px/s}$, where $\beta \approx 0.007$ scales the cutoff by $0.007 \times 500 \approx 3.5\text{ Hz}$. In Normalized Device Coordinates $[0, 1]$, fingertip speed is typically $0.1\text{--}2.0\text{ s}^{-1}$. A coefficient of $\beta = 30.0$ reduces the smoothness/responsiveness trade-off by increasing the cutoff frequency proportionally during faster motion. Typical simulated rapid motions may drive the adaptive cutoff into the tens-of-Hz range, but the implementation has no explicit maximum cutoff unless $f_{c, \max}$ is configured.

---

## 3. Empirical Benchmark Methodology & Results

The benchmark suite (`benchmark_filters.py`) feeds five deterministic trajectories at 60 FPS into each filter. All trajectory metadata is explicitly tracked for reproducibility:

1. **Stationary Noisy:** Duration $3.0\text{ s}$, $N = 180$ samples, nominal $60.0\text{ FPS}$, Gaussian noise $\sigma = 0.003\text{ NDC}$, seed $= 42$.
2. **Constant Velocity:** Duration $3.0\text{ s}$, $N = 180$ samples, nominal $60.0\text{ FPS}$, velocity $v = 0.20\text{ NDC/s}$, Gaussian noise $\sigma = 0.002\text{ NDC}$, seed $= 43$.
3. **Sinusoidal 1.5 Hz:** Duration $4.0\text{ s}$, $N = 240$ samples, nominal $60.0\text{ FPS}$, amplitude $A = 0.20\text{ NDC}$, Gaussian noise $\sigma = 0.002\text{ NDC}$, seed $= 44$.
4. **Step Discontinuity:** Duration $2.0\text{ s}$, $N = 120$ samples, nominal $60.0\text{ FPS}$, step at $t = 0.5\text{ s}$, magnitude $= 0.30\text{ NDC}$, Gaussian noise $\sigma = 0.001\text{ NDC}$, seed $= 45$.
5. **Rapid Strum 4.0 Hz:** Duration $3.0\text{ s}$, $N = 180$ samples, nominal $60.0\text{ FPS}$, frequency $f = 4.0\text{ Hz}$, amplitude $A = 0.15\text{ NDC}$, Gaussian noise $\sigma = 0.002\text{ NDC}$, seed $= 46$.

### 3.1 Empirical Results Table

Directly generated by `benchmark_filters.py` and stored in `benchmarks/filter_benchmark_results.csv`:

| Trajectory | Filter | RMSE (NDC) | MAE (NDC) | Stat. Jitter (1080p) | Error-Δ RMS (1080p) | Phase Lag (ms) | Detected Lag (Frames) | Settling Time (ms) |
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

### 3.2 Scientific Analysis of Empirical Trade-offs
1. **Tremor Attenuation:** In stationary tracking, `OneEuroFilter` reduces coordinate jitter from $5.90\text{ px}$ to **$1.77\text{ px}$ RMS** at 1080p, while residual error-delta drops from $6.17\text{ px}$ to **$0.84\text{ px}$** (a **$7.3\times$ reduction in residual error-delta RMS**).
2. **Phase Lag Estimator Resolution:**  
   At a sampling rate of 60 Hz ($\Delta t = 16.67\text{ ms}$), whole-sample cross-correlation has an integer resolution of $\pm 8.33\text{ ms}$. A result of `0 detected frames` proves only that lag is within the sub-frame window ($< 16.7\text{ ms}$).  
   Continuous Fourier harmonic phase analysis provides an estimated sub-sample phase delay:
   - For $1.5\text{ Hz}$ motion: `RawFilter` and `DeadbandFilter` have $0.1\text{ ms}$ lag, `OneEuroFilter` has $7.8\text{ ms}$ lag, and `EMAFilter` has $9.0\text{ ms}$ lag (triggering a 1-frame whole-sample delay).
3. **Step Settling Dynamics:** Discrete frame latency is computed relative to the step occurrence frame (where discrete resolution is bounded by $\Delta t = 16.67\text{ ms}$ at 60 FPS). `RawFilter` and `DeadbandFilter` settle instantaneously at sample 0 ($0.0\text{ ms}$ delay relative to the step event frame). `OneEuroFilter` settles in 2 frames ($33.3\text{ ms}$) and `EMAFilter` settles in 3 frames ($50.0\text{ ms}$).

---

## 4. Audio Engine & Polyphony Architecture

The synthesizer in `audio_engine.py` operates on a dedicated stream buffer ($44,100\text{ Hz}$, block size $256$ frames $\approx 5.8\text{ ms}$ callback block duration; actual end-to-end output latency is hardware- and driver-dependent).

### 4.1 Authentic 9-Chord Guitars & Muted String Logic
Standard open-position guitar chords mute specific strings:
```python
CHORD_FRETS = {
    "C":  [None, 3, 2, 0, 1, 0],     # Low E muted
    "G":  [3, 2, 0, 0, 0, 3],        # All 6 sounded
    "D":  [None, None, 0, 2, 3, 2],  # Low E & A muted
    "A":  [None, 0, 2, 2, 2, 0],     # Low E muted
    "E":  [0, 2, 2, 1, 0, 0],        # All 6 sounded
    "Am": [None, 0, 2, 2, 1, 0],     # Low E muted
    "Em": [0, 2, 2, 0, 0, 0],        # All 6 sounded
    "Dm": [None, None, 0, 2, 3, 1],  # Low E & A muted
    "F":  [1, 3, 3, 2, 1, 1],        # Barre F chord (all sounded)
}
```
When a strum crosses a string whose voicing fret is `None`:
1. `AudioEngine.play_guitar()` immediately returns `False`.
2. **Zero synthesis voices** are added to the active audio pipeline.
3. The visual renderer displays an `"X"` indicator near the bridge and damps string amplitude to $\le 2.5\text{ px}$ without glowing neon sparks.

### 4.2 Decoupled Audio Callback & Lock-Minimized Audio Mixing
To prevent priority inversions on the PortAudio callback thread:
1. Active voices are snapshotted under mutex in a short critical section.
2. Waveform rendering (`voice.render(frames)`) and additive mixing execute **outside the lock**. Piano voices synthesize with a percussive Attack-Decay (AD) exponential envelope, while plucked strings utilize exponential decay.
3. Only finished voice removal re-acquires the lock briefly. Zero logging or I/O calls occur on the realtime audio thread.
4. Hardware buffer status flags (`status.output_underflow`, `status.output_overflow`) are recorded into telemetry counters.
5. Mixed samples pass through `np.tanh` soft-limiting to eliminate digital clipping.

---

## 5. Desk-Surface Piano Kinematics

In `instruments.py`, the virtual piano is grounded to the lower desk area ($Y \in [0.72, 0.96]$) so the user's wrists rest on the physical desk, designed to reduce sustained arm elevation fatigue.

### 5.1 Downward Velocity Gating
Resting fingers inside a key rect must not continuously re-trigger notes:
1. **Debounce lockout:** $\Delta t > 0.120\text{ s}$ ($120\text{ ms}$).
2. **Hover suppression:** Finger ID `(handedness, tip_id)` must not already be recorded in `_finger_held_keys`.
3. **Downward velocity threshold:**
   $$v_y \ge 60.0\text{ px/s} \quad \text{and} \quad \frac{v_y}{H} \ge 0.08\text{ s}^{-1}$$
Fingers moving upward ($v_y < -40.0\text{ px/s}$) or exiting the key boundary release the held key lock.

### 5.2 Geometric Collision Precedence
Black accidentals overlap white natural keys. If evaluated after white keys, black keys can never be struck cleanly. The engine evaluates:
1. **Pass 1:** Black key bounding boxes (with $\pm 2\text{ px}$ horizontal padding). If hit, evaluation halts (`continue`).
2. **Pass 2:** White key bounding boxes.

---

## 6. Hardware Benchmark — Not Yet Measured

> [!IMPORTANT]
> **Measurement Policy:**  
> The repository includes real-time telemetry metrics logged dynamically by `main.py` (`cam_ms`, `track_ms`, `gest_ms`, `rend_ms`, `total_ms`, `inference_latency_ms`, `stale_frames_skipped`). However, no committed multi-machine hardware benchmark dataset currently exists in the repository. Unverified hardware latency claims (e.g. CPU models, camera driver latency) are deliberately omitted until an automated hardware profiling harness is run and committed.
