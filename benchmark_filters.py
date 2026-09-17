"""
benchmark_filters.py
====================
Deterministic Empirical Benchmark Suite for Hand Tracking Filters.

Evaluates:
- RawFilter (Baseline direct pass-through)
- EMAFilter (Exponential Moving Average, alpha=0.65)
- DeadbandFilter (Adaptive Noise Gate: deadband=0.0025, motion_thresh=0.0080)
- OneEuroFilter (Casiez et al. 2012: min_cutoff=1.0, beta=30.0, d_cutoff=1.0)

Scientific Metric Definitions:
- Position RMSE: Root-mean-square position error relative to ground-truth.
- MAE: Mean absolute tracking error.
- Max Error: Maximum single-sample absolute tracking error.
- Stationary Jitter RMS: Standard deviation of static coordinate positions at 1080p.
- Residual Noise RMS: Standard deviation of consecutive error differences with true motion subtracted.
- Phase Lag (ms): Continuous phase shift measured via Fourier harmonic analysis at the motion frequency.
- Detected Lag (frames): Whole-sample lag from discrete cross-correlation (resolution: +/- 8.33 ms at 60 Hz).
- Settling Time (ms): Discrete frame latency required to permanently enter and stay within +/-2% of step magnitude (resolution: 1 frame = 16.7 ms at 60 Hz). Instantaneous response settles at sample 0 (0.0 ms delay).
- Overshoot (%): Maximum transient excursion beyond target step level.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import numpy as np

from vision_tracker import BaseFilter, DeadbandFilter, EMAFilter, OneEuroFilter, RawFilter

REFERENCE_WIDTH = 1920.0
REFERENCE_HEIGHT = 1080.0


def generate_stationary_noisy(
    duration: float = 3.0,
    fps: float = 60.0,
    noise_std: float = 0.003,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float | int]]:
    """Generates a stationary hand position with Gaussian sensor noise."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fps)
    timestamps = np.arange(n_samples, dtype=np.float64) / fps

    ground_truth = np.full((n_samples, 21, 3), 0.5, dtype=np.float32)
    noise = rng.normal(0.0, noise_std, ground_truth.shape).astype(np.float32)
    noisy_input = ground_truth + noise
    meta = {
        "duration_s": duration,
        "sample_count": n_samples,
        "nominal_fps": fps,
        "noise_std": noise_std,
        "seed": seed,
    }
    return timestamps, ground_truth, noisy_input, meta


def generate_constant_velocity(
    duration: float = 3.0,
    fps: float = 60.0,
    v: float = 0.20,  # 0.20 screen width per second
    noise_std: float = 0.002,
    seed: int = 43,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float | int]]:
    """Generates linear constant velocity motion across the frame."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fps)
    timestamps = np.arange(n_samples, dtype=np.float64) / fps

    ground_truth = np.zeros((n_samples, 21, 3), dtype=np.float32)
    start_pos = 0.20
    for i, t in enumerate(timestamps):
        x = start_pos + v * t
        ground_truth[i, :, 0] = x
        ground_truth[i, :, 1] = 0.50
        ground_truth[i, :, 2] = 0.00

    noise = rng.normal(0.0, noise_std, ground_truth.shape).astype(np.float32)
    noisy_input = ground_truth + noise
    meta = {
        "duration_s": duration,
        "sample_count": n_samples,
        "nominal_fps": fps,
        "noise_std": noise_std,
        "seed": seed,
    }
    return timestamps, ground_truth, noisy_input, meta


def generate_sinusoidal(
    duration: float = 4.0,
    fps: float = 60.0,
    freq: float = 1.5,  # 1.5 Hz
    amplitude: float = 0.20,
    noise_std: float = 0.002,
    seed: int = 44,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float | int]]:
    """Generates sinusoidal oscillatory gesture movement."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fps)
    timestamps = np.arange(n_samples, dtype=np.float64) / fps

    ground_truth = np.zeros((n_samples, 21, 3), dtype=np.float32)
    for i, t in enumerate(timestamps):
        x = 0.50 + amplitude * np.sin(2.0 * np.pi * freq * t)
        ground_truth[i, :, 0] = x
        ground_truth[i, :, 1] = 0.50
        ground_truth[i, :, 2] = 0.00

    noise = rng.normal(0.0, noise_std, ground_truth.shape).astype(np.float32)
    noisy_input = ground_truth + noise
    meta = {
        "duration_s": duration,
        "sample_count": n_samples,
        "nominal_fps": fps,
        "noise_std": noise_std,
        "seed": seed,
    }
    return timestamps, ground_truth, noisy_input, meta


def generate_step_discontinuity(
    duration: float = 2.0,
    fps: float = 60.0,
    step_time: float = 0.5,
    step_magnitude: float = 0.30,
    noise_std: float = 0.001,
    seed: int = 45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float | int]]:
    """Generates an abrupt step displacement for transient settling time analysis."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fps)
    timestamps = np.arange(n_samples, dtype=np.float64) / fps

    ground_truth = np.zeros((n_samples, 21, 3), dtype=np.float32)
    for i, t in enumerate(timestamps):
        val = 0.30 if t < step_time else (0.30 + step_magnitude)
        ground_truth[i, :, 0] = val
        ground_truth[i, :, 1] = 0.50
        ground_truth[i, :, 2] = 0.00

    noise = rng.normal(0.0, noise_std, ground_truth.shape).astype(np.float32)
    noisy_input = ground_truth + noise
    meta = {
        "duration_s": duration,
        "sample_count": n_samples,
        "nominal_fps": fps,
        "noise_std": noise_std,
        "seed": seed,
    }
    return timestamps, ground_truth, noisy_input, meta


def generate_rapid_strum(
    duration: float = 3.0,
    fps: float = 60.0,
    freq: float = 4.0,  # 4.0 Hz rapid strumming
    amplitude: float = 0.15,
    noise_std: float = 0.002,
    seed: int = 46,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float | int]]:
    """Generates fast cyclic strumming trajectory."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fps)
    timestamps = np.arange(n_samples, dtype=np.float64) / fps

    ground_truth = np.zeros((n_samples, 21, 3), dtype=np.float32)
    for i, t in enumerate(timestamps):
        y = 0.60 + amplitude * np.sin(2.0 * np.pi * freq * t)
        ground_truth[i, :, 0] = 0.70
        ground_truth[i, :, 1] = y
        ground_truth[i, :, 2] = 0.00

    noise = rng.normal(0.0, noise_std, ground_truth.shape).astype(np.float32)
    noisy_input = ground_truth + noise
    meta = {
        "duration_s": duration,
        "sample_count": n_samples,
        "nominal_fps": fps,
        "noise_std": noise_std,
        "seed": seed,
    }
    return timestamps, ground_truth, noisy_input, meta


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
    eval_axis: int = 0,
    is_step: bool = False,
    step_time: float = 0.5,
    step_mag: float = 0.30,
    is_periodic: bool = False,
    osc_freq: float = 1.5,
    is_stationary: bool = False,
) -> Dict[str, float | int]:
    """
    Computes empirical metrics comparing filtered output against ground truth.
    Evaluates Landmark 8 (Index fingertip) on specified axis (0=X, 1=Y).
    """
    gt = ground_truth[:, 8, eval_axis]
    filt = filtered_output[:, 8, eval_axis]

    errors = filt - gt
    mae = float(np.mean(np.abs(errors)))
    max_err = float(np.max(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Select axis-appropriate reference pixel scale (X=1920, Y=1080)
    pixel_scale = REFERENCE_HEIGHT if eval_axis == 1 else REFERENCE_WIDTH

    # Residual noise RMS: high-frequency error variation with true motion subtracted
    # Delta e_i = e_i - e_{i-1}. Standard error std = RMS(Delta e) / sqrt(2)
    diff_errors = np.diff(errors)
    residual_noise_rms = float(np.sqrt(np.mean(diff_errors ** 2)) / np.sqrt(2.0))
    residual_noise_px = float(residual_noise_rms * pixel_scale)

    # Stationary jitter: only defined for static hand tracking
    if is_stationary:
        stationary_jitter_rms = float(np.std(filt))
        stationary_jitter_px = float(stationary_jitter_rms * pixel_scale)
    else:
        stationary_jitter_rms = -1.0
        stationary_jitter_px = -1.0

    # Phase lag estimation
    detected_lag_frames = 0
    phase_lag_ms = -1.0

    if is_periodic and osc_freq > 0.0:
        s1 = filt - np.mean(filt)
        s2 = gt - np.mean(gt)
        std1 = float(np.std(s1))
        std2 = float(np.std(s2))

        if std1 > 1e-5 and std2 > 1e-5:
            # 1. Discrete cross-correlation (gives integer frames)
            corr = np.correlate(s1, s2, mode="full")
            lags = np.arange(-len(s1) + 1, len(s1))
            best_lag_idx = int(np.argmax(corr))
            detected_lag_frames = int(lags[best_lag_idx])

            # 2. Continuous Fourier phase analysis at target frequency
            sin_basis = np.sin(2.0 * np.pi * osc_freq * timestamps)
            cos_basis = np.cos(2.0 * np.pi * osc_freq * timestamps)

            # Discard initial cycle transient for steady-state phase estimation
            warmup_mask = timestamps >= (1.0 / osc_freq)
            w_filt = filt[warmup_mask]
            w_gt = gt[warmup_mask]
            w_sin = sin_basis[warmup_mask]
            w_cos = cos_basis[warmup_mask]

            phi_gt = float(np.arctan2(np.sum(w_gt * w_cos), np.sum(w_gt * w_sin)))
            phi_filt = float(np.arctan2(np.sum(w_filt * w_cos), np.sum(w_filt * w_sin)))

            d_phi = (phi_gt - phi_filt) % (2.0 * np.pi)
            if d_phi > np.pi:
                d_phi -= 2.0 * np.pi
            phase_lag_ms = float(max(0.0, (d_phi / (2.0 * np.pi * osc_freq)) * 1000.0))

    settling_time_ms = -1.0
    overshoot_pct = 0.0
    if is_step:
        post_step_mask = timestamps >= step_time
        post_ts = timestamps[post_step_mask]
        post_filt = filt[post_step_mask]
        target = gt[-1]
        threshold = 0.02 * step_mag  # 2% band of step magnitude

        settled = np.abs(post_filt - target) <= threshold
        settled_idx = None
        for k in range(len(settled)):
            if np.all(settled[k:]):
                settled_idx = k
                break

        if settled_idx is not None:
            # Sample-aware settling latency relative to step occurrence frame (post_ts[0])
            # Resolution is 1 frame (16.7 ms at 60 Hz); sample 0 indicates instantaneous response
            settling_time_ms = float((post_ts[settled_idx] - post_ts[0]) * 1000.0)
        else:
            settling_time_ms = float((post_ts[-1] - post_ts[0]) * 1000.0)

        max_val = float(np.max(post_filt))
        if max_val > target:
            overshoot_pct = float(((max_val - target) / step_mag) * 100.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "max_err": max_err,
        "stationary_jitter_px": stationary_jitter_px,
        "residual_noise_px": residual_noise_px,
        "phase_lag_ms": phase_lag_ms,
        "detected_lag_frames": detected_lag_frames,
        "settling_time_ms": settling_time_ms,
        "overshoot_pct": overshoot_pct,
    }


def run_all_benchmarks(output_csv: str = "benchmarks/filter_benchmark_results.csv") -> List[Dict[str, str | float | int]]:
    """Runs all 4 filters through all 5 trajectories and writes results."""
    # (name, generator_data, eval_axis, is_step, step_t, step_mag, is_periodic, osc_freq, is_stationary)
    trajectories = [
        ("Stationary Noisy (3s)", generate_stationary_noisy(), 0, False, 0.0, 0.0, False, 0.0, True),
        ("Constant Velocity (3s)", generate_constant_velocity(), 0, False, 0.0, 0.0, False, 0.0, False),
        ("Sinusoidal 1.5Hz (4s)", generate_sinusoidal(), 0, False, 0.0, 0.0, True, 1.5, False),
        ("Step Response (2s)", generate_step_discontinuity(), 0, True, 0.5, 0.30, False, 0.0, False),
        ("Rapid Strum 4.0Hz (3s)", generate_rapid_strum(), 1, False, 0.0, 0.0, True, 4.0, False),
    ]

    filters: List[Tuple[str, BaseFilter]] = [
        ("RawFilter (Pass-Through)", RawFilter()),
        ("EMAFilter (alpha=0.65)", EMAFilter(alpha=0.65)),
        ("DeadbandFilter (Adaptive)", DeadbandFilter(deadband_norm=0.0025, motion_threshold_norm=0.0080)),
        ("OneEuroFilter (Casiez 2012)", OneEuroFilter(min_cutoff=1.0, beta=30.0, d_cutoff=1.0)),
    ]

    results_table: List[Dict[str, str | float | int]] = []

    print("\n" + "=" * 106)
    print("EMPIRICAL FILTER BENCHMARK SUITE (GESTURE AR INSTRUMENTS)")
    print("=" * 106)

    for traj_name, (ts, gt, noisy, meta), axis, is_step, step_t, step_mag, is_periodic, osc_f, is_stat in trajectories:
        print(f"\n--- Scenario: {traj_name} (Duration: {meta['duration_s']}s, N={meta['sample_count']}, sigma={meta['noise_std']}) ---")
        header = f"{'Filter Name':<28} | {'RMSE':<8} | {'MAE':<8} | {'Stat.Jitter':<11} | {'Resid.Noise':<11} | {'Lag (ms)':<9} | {'Settling (ms)'}"
        print(header)
        print("-" * len(header))

        for filt_name, filt_inst in filters:
            filtered = run_filter_on_trajectory(filt_inst, ts, noisy)
            metrics = compute_metrics(
                ts, gt, filtered,
                eval_axis=axis,
                is_step=is_step,
                step_time=step_t,
                step_mag=step_mag,
                is_periodic=is_periodic,
                osc_freq=osc_f,
                is_stationary=is_stat,
            )

            stat_str = f"{metrics['stationary_jitter_px']:.2f} px" if is_stat else "N/A"
            resid_str = f"{metrics['residual_noise_px']:.2f} px"
            lag_str = f"{metrics['phase_lag_ms']:.1f}" if is_periodic else "N/A"
            settle_str = f"{metrics['settling_time_ms']:.1f}" if is_step else "N/A"

            row_str = (
                f"{filt_name:<28} | "
                f"{metrics['rmse']:.5f}  | "
                f"{metrics['mae']:.5f}  | "
                f"{stat_str:<11} | "
                f"{resid_str:<11} | "
                f"{lag_str:<9} | "
                f"{settle_str}"
            )
            print(row_str)

            record: Dict[str, str | float | int] = {
                "trajectory": traj_name,
                "filter": filt_name,
                "duration_s": meta["duration_s"],
                "sample_count": meta["sample_count"],
                "nominal_fps": meta["nominal_fps"],
                "noise_std": meta["noise_std"],
                "seed": meta["seed"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "max_err": metrics["max_err"],
                "stationary_jitter_px_1080p": metrics["stationary_jitter_px"],
                "residual_noise_px_1080p": metrics["residual_noise_px"],
                "phase_lag_ms": metrics["phase_lag_ms"],
                "detected_lag_frames": metrics["detected_lag_frames"],
                "settling_time_ms": metrics["settling_time_ms"],
                "overshoot_pct": metrics["overshoot_pct"],
            }
            results_table.append(record)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "trajectory",
        "filter",
        "duration_s",
        "sample_count",
        "nominal_fps",
        "noise_std",
        "seed",
        "rmse",
        "mae",
        "max_err",
        "stationary_jitter_px_1080p",
        "residual_noise_px_1080p",
        "phase_lag_ms",
        "detected_lag_frames",
        "settling_time_ms",
        "overshoot_pct",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)

    print(f"\n[INFO] Comprehensive benchmark results saved to: {output_csv}\n")
    return results_table


if __name__ == "__main__":
    run_all_benchmarks()
