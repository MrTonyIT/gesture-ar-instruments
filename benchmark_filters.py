"""
benchmark_filters.py
====================
Deterministic Empirical Benchmark Suite for Hand Tracking Filters.

Evaluates:
- RawFilter (Baseline direct pass-through)
- EMAFilter (Exponential Moving Average, alpha=0.65)
- DeadbandFilter (Adaptive Zero-Lag Noise Gate)
- OneEuroFilter (Casiez et al. 2012 Adaptive Low-Pass)

Trajectories Evaluated:
1. Stationary Noisy: Static position with realistic camera sensor noise.
2. Constant Velocity: Linear translation across frame with sensor noise.
3. Sinusoidal Oscillation: 1.5 Hz cyclic gesture movement.
4. Step Discontinuity: Sudden 0.3-unit coordinate jump to measure settling time.
5. High-Speed Strum: 4.0 Hz rapid back-and-forth strumming gesture.

Metrics Computed:
- RMS Jitter (normalized units and px at 1080p)
- Mean Absolute Error (MAE)
- Maximum Tracking Error (Max Error)
- Phase Lag (ms via cross-correlation peak)
- Settling Time (ms to within 2% band on step input)
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import numpy as np

from vision_tracker import BaseFilter, DeadbandFilter, EMAFilter, OneEuroFilter, RawFilter


def generate_stationary_noisy(
    duration: float = 3.0,
    fps: float = 60.0,
    noise_std: float = 0.003,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a stationary hand position with Gaussian sensor noise."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fps)
    timestamps = np.linspace(0.0, duration, n_samples)

    ground_truth = np.full((n_samples, 21, 3), 0.5, dtype=np.float32)
    noise = rng.normal(0.0, noise_std, ground_truth.shape).astype(np.float32)
    noisy_input = ground_truth + noise
    return timestamps, ground_truth, noisy_input


def generate_constant_velocity(
    duration: float = 3.0,
    fps: float = 60.0,
    v: float = 0.20,  # 0.20 screen width per second
    noise_std: float = 0.002,
    seed: int = 43,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates linear constant velocity motion across the frame."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fps)
    timestamps = np.linspace(0.0, duration, n_samples)

    ground_truth = np.zeros((n_samples, 21, 3), dtype=np.float32)
    start_pos = 0.20
    for i, t in enumerate(timestamps):
        x = start_pos + v * t
        ground_truth[i, :, 0] = x
        ground_truth[i, :, 1] = 0.50
        ground_truth[i, :, 2] = 0.00

    noise = rng.normal(0.0, noise_std, ground_truth.shape).astype(np.float32)
    noisy_input = ground_truth + noise
    return timestamps, ground_truth, noisy_input


def generate_sinusoidal(
    duration: float = 4.0,
    fps: float = 60.0,
    freq: float = 1.5,  # 1.5 Hz
    amplitude: float = 0.20,
    noise_std: float = 0.002,
    seed: int = 44,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates sinusoidal oscillatory gesture movement."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fps)
    timestamps = np.linspace(0.0, duration, n_samples)

    ground_truth = np.zeros((n_samples, 21, 3), dtype=np.float32)
    for i, t in enumerate(timestamps):
        x = 0.50 + amplitude * np.sin(2.0 * np.pi * freq * t)
        ground_truth[i, :, 0] = x
        ground_truth[i, :, 1] = 0.50
        ground_truth[i, :, 2] = 0.00

    noise = rng.normal(0.0, noise_std, ground_truth.shape).astype(np.float32)
    noisy_input = ground_truth + noise
    return timestamps, ground_truth, noisy_input


def generate_step_discontinuity(
    duration: float = 2.0,
    fps: float = 60.0,
    step_time: float = 0.5,
    step_magnitude: float = 0.30,
    noise_std: float = 0.001,
    seed: int = 45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates an abrupt step displacement for transient settling time analysis."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fps)
    timestamps = np.linspace(0.0, duration, n_samples)

    ground_truth = np.zeros((n_samples, 21, 3), dtype=np.float32)
    for i, t in enumerate(timestamps):
        val = 0.30 if t < step_time else (0.30 + step_magnitude)
        ground_truth[i, :, 0] = val
        ground_truth[i, :, 1] = 0.50
        ground_truth[i, :, 2] = 0.00

    noise = rng.normal(0.0, noise_std, ground_truth.shape).astype(np.float32)
    noisy_input = ground_truth + noise
    return timestamps, ground_truth, noisy_input


def generate_rapid_strum(
    duration: float = 3.0,
    fps: float = 60.0,
    freq: float = 4.0,  # 4.0 Hz rapid strumming
    amplitude: float = 0.15,
    noise_std: float = 0.002,
    seed: int = 46,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates fast cyclic strumming trajectory."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fps)
    timestamps = np.linspace(0.0, duration, n_samples)

    ground_truth = np.zeros((n_samples, 21, 3), dtype=np.float32)
    for i, t in enumerate(timestamps):
        y = 0.60 + amplitude * np.sin(2.0 * np.pi * freq * t)
        ground_truth[i, :, 0] = 0.70
        ground_truth[i, :, 1] = y
        ground_truth[i, :, 2] = 0.00

    noise = rng.normal(0.0, noise_std, ground_truth.shape).astype(np.float32)
    noisy_input = ground_truth + noise
    return timestamps, ground_truth, noisy_input


def run_filter_on_trajectory(
    filter_instance: BaseFilter,
    timestamps: np.ndarray,
    noisy_input: np.ndarray,
) -> np.ndarray:
    """Executes filter sequentially across trajectory timestamps."""
    filter_instance.reset()
    n_samples = len(timestamps)
    output = np.empty_like(noisy_input)

    for i in range(n_samples):
        output[i] = filter_instance.filter(noisy_input[i], float(timestamps[i]))

    return output


def compute_metrics(
    timestamps: np.ndarray,
    ground_truth: np.ndarray,
    filtered_output: np.ndarray,
    is_step: bool = False,
    step_time: float = 0.5,
    step_mag: float = 0.30,
) -> Dict[str, float]:
    """
    Computes rigorous empirical metrics comparing filtered output against ground truth.
    All evaluations focus on the X-coordinate of landmark 8 (Index fingertip).
    """
    dt = float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0.01667

    gt_x = ground_truth[:, 8, 0]
    filt_x = filtered_output[:, 8, 0]

    errors = filt_x - gt_x
    mae = float(np.mean(np.abs(errors)))
    max_err = float(np.max(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Jitter: high-frequency variation (standard deviation of consecutive differences)
    diffs = np.diff(filt_x)
    jitter_rms = float(np.sqrt(np.mean(diffs ** 2)))

    # Cross-correlation phase lag estimation
    s1 = filt_x - np.mean(filt_x)
    s2 = gt_x - np.mean(gt_x)
    std1 = np.std(s1)
    std2 = np.std(s2)

    if std1 > 1e-6 and std2 > 1e-6:
        corr = np.correlate(s1, s2, mode="full")
        lags = np.arange(-len(s1) + 1, len(s1))
        best_lag_idx = int(np.argmax(corr))
        lag_samples = lags[best_lag_idx]
        lag_ms = float(max(0.0, lag_samples * dt * 1000.0))
    else:
        lag_ms = 0.0

    settling_time_ms = 0.0
    if is_step:
        post_step_mask = timestamps >= step_time
        post_ts = timestamps[post_step_mask]
        post_filt = filt_x[post_step_mask]
        target = gt_x[-1]
        threshold = 0.02 * step_mag

        settled = np.abs(post_filt - target) <= threshold
        settled_idx = None
        for k in range(len(settled)):
            if np.all(settled[k:]):
                settled_idx = k
                break

        if settled_idx is not None:
            settling_time_ms = float((post_ts[settled_idx] - step_time) * 1000.0)
        else:
            settling_time_ms = float((post_ts[-1] - step_time) * 1000.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "max_err": max_err,
        "jitter_rms": jitter_rms,
        "jitter_px_1080p": jitter_rms * 1920.0,
        "lag_ms": lag_ms,
        "settling_time_ms": settling_time_ms,
    }


def run_all_benchmarks(output_csv: str = "benchmarks/filter_benchmark_results.csv") -> List[Dict[str, str | float]]:
    """Runs all 4 filters through all 5 trajectories and writes results."""
    trajectories = [
        ("Stationary Noisy (3s)", generate_stationary_noisy(), False, 0.0, 0.0),
        ("Constant Velocity (3s)", generate_constant_velocity(), False, 0.0, 0.0),
        ("Sinusoidal 1.5Hz (4s)", generate_sinusoidal(), False, 0.0, 0.0),
        ("Step Response (2s)", generate_step_discontinuity(), True, 0.5, 0.30),
        ("Rapid Strum 4.0Hz (3s)", generate_rapid_strum(), False, 0.0, 0.0),
    ]

    filters: List[Tuple[str, BaseFilter]] = [
        ("RawFilter (Pass-Through)", RawFilter()),
        ("EMAFilter (alpha=0.65)", EMAFilter(alpha=0.65)),
        ("DeadbandFilter (Adaptive)", DeadbandFilter(deadband_norm=0.0025, motion_threshold_norm=0.0080)),
        ("OneEuroFilter (Casiez 2012)", OneEuroFilter(min_cutoff=1.0, beta=30.0, d_cutoff=1.0)),
    ]

    results_table: List[Dict[str, str | float]] = []

    print("\n" + "=" * 88)
    print("EMPIRICAL FILTER BENCHMARK SUITE (GESTURE AR INSTRUMENTS)")
    print("=" * 88)

    for traj_name, (ts, gt, noisy), is_step, step_t, step_mag in trajectories:
        print(f"\n--- Scenario: {traj_name} ---")
        header = f"{'Filter Name':<30} | {'RMSE':<10} | {'MAE':<10} | {'Jitter (px)':<12} | {'Lag (ms)':<10} | {'Settling (ms)':<12}"
        print(header)
        print("-" * len(header))

        for filt_name, filt_inst in filters:
            filtered = run_filter_on_trajectory(filt_inst, ts, noisy)
            metrics = compute_metrics(ts, gt, filtered, is_step=is_step, step_time=step_t, step_mag=step_mag)

            settle_str = f"{metrics['settling_time_ms']:.1f}" if is_step else "N/A"
            row_str = (
                f"{filt_name:<30} | "
                f"{metrics['rmse']:.6f}   | "
                f"{metrics['mae']:.6f}   | "
                f"{metrics['jitter_px_1080p']:<12.3f} | "
                f"{metrics['lag_ms']:<10.1f} | "
                f"{settle_str:<12}"
            )
            print(row_str)

            record: Dict[str, str | float] = {
                "trajectory": traj_name,
                "filter": filt_name,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "max_err": metrics["max_err"],
                "jitter_rms": metrics["jitter_rms"],
                "jitter_px_1080p": metrics["jitter_px_1080p"],
                "lag_ms": metrics["lag_ms"],
                "settling_time_ms": metrics["settling_time_ms"] if is_step else -1.0,
            }
            results_table.append(record)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "trajectory",
        "filter",
        "rmse",
        "mae",
        "max_err",
        "jitter_rms",
        "jitter_px_1080p",
        "lag_ms",
        "settling_time_ms",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)

    print(f"\n[INFO] Comprehensive benchmark results saved to: {output_csv}\n")
    return results_table


if __name__ == "__main__":
    run_all_benchmarks()
