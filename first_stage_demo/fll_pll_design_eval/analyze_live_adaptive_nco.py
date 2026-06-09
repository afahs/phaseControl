#!/usr/bin/env python3
"""Analyze live adaptive-NCO PLL logs from cw_vcxo_adaptive_nco_discipline.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TWO_PI = 2.0 * math.pi


def parse_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(str(value).strip())
    except ValueError:
        return float("nan")


def wrap_pm_pi(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + math.pi) % TWO_PI - math.pi


def load_csv(path: Path) -> dict[str, np.ndarray | list[str]]:
    numeric_cols = [
        "t_rel_s",
        "residual_phase_rad",
        "residual_unwrapped_phase_rad",
        "residual_freq_hz",
        "nco_freq_correction_hz",
        "nco_phase_correction_rad",
        "nco_unwrapped_phase_correction_rad",
        "phase_error_rad",
        "phase_window_rad",
        "phase_window_confidence",
        "target_nco_freq_hz",
        "hardware_frequency_error_hz",
        "filtered_nco_freq_hz",
        "control_cmd",
        "requested_cmd",
        "dac_code_cmd",
        "amp",
        "coherence",
        "phase_integral_hz",
    ]
    cols: dict[str, list[float] | list[str]] = {key: [] for key in numeric_cols}
    cols["phase_mode"] = []
    cols["control_stage"] = []
    cols["status"] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in numeric_cols:
                cols[key].append(parse_float(row.get(key)))  # type: ignore[union-attr]
            for key in ("phase_mode", "control_stage", "status"):
                cols[key].append(str(row.get(key, "")).strip())  # type: ignore[union-attr]
    out: dict[str, np.ndarray | list[str]] = {}
    for key, values in cols.items():
        if key in numeric_cols:
            out[key] = np.asarray(values, dtype=np.float64)  # type: ignore[arg-type]
        else:
            out[key] = list(values)  # type: ignore[arg-type]
    return out


def slope_hz(t: np.ndarray, phase: np.ndarray, mask: np.ndarray) -> float:
    fit = mask & np.isfinite(t) & np.isfinite(phase)
    if np.count_nonzero(fit) < 3:
        return float("nan")
    t0 = float(t[fit][0])
    coeff = np.polyfit(t[fit] - t0, phase[fit], 1)
    return float(coeff[0] / TWO_PI)


def detrended_rms_ns(t: np.ndarray, phase: np.ndarray, mask: np.ndarray, rf_hz: float) -> float:
    fit = mask & np.isfinite(t) & np.isfinite(phase)
    if np.count_nonzero(fit) < 3:
        return float("nan")
    t0 = float(t[fit][0])
    coeff = np.polyfit(t[fit] - t0, phase[fit], 1)
    residual_phase = phase[fit] - (coeff[0] * (t[fit] - t0) + coeff[1])
    residual_ns = residual_phase / (TWO_PI * rf_hz) * 1.0e9
    return float(np.sqrt(np.mean(residual_ns * residual_ns)))


def pct_abs(x: np.ndarray, pct: float) -> float:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(np.abs(finite), pct))


def longest_dwell_s(t: np.ndarray, ok: np.ndarray) -> float:
    best = 0.0
    start: float | None = None
    prev = float("nan")
    for ti, good in zip(t, ok):
        if good and start is None:
            start = float(ti)
        elif not good and start is not None:
            best = max(best, float(prev) - start)
            start = None
        prev = float(ti)
    if start is not None and np.isfinite(prev):
        best = max(best, prev - start)
    return float(best)


def summarize(data: dict[str, np.ndarray | list[str]], rf_hz: float, phase_band_rad: float) -> dict[str, object]:
    t = data["t_rel_s"]  # type: ignore[assignment]
    phase = data["residual_unwrapped_phase_rad"]  # type: ignore[assignment]
    nco_phase = data["nco_unwrapped_phase_correction_rad"]  # type: ignore[assignment]
    reconstructed_phase = phase + nco_phase  # type: ignore[operator]
    wrapped = wrap_pm_pi(phase)  # type: ignore[arg-type]
    reconstructed_wrapped = wrap_pm_pi(reconstructed_phase)
    coherence = data["coherence"]  # type: ignore[assignment]
    amp = data["amp"]  # type: ignore[assignment]
    residual_freq = data["residual_freq_hz"]  # type: ignore[assignment]
    nco_freq = data["nco_freq_correction_hz"]  # type: ignore[assignment]
    cmd = data["control_cmd"]  # type: ignore[assignment]
    phase_error = data["phase_error_rad"]  # type: ignore[assignment]
    valid = np.isfinite(t) & np.isfinite(phase) & np.isfinite(coherence) & np.isfinite(amp) & (coherence >= 0.2) & (amp >= 0.02)  # type: ignore[operator]
    phase_band = valid & np.isfinite(wrapped) & (np.abs(wrapped) <= phase_band_rad)
    phase_modes = Counter(data["phase_mode"])  # type: ignore[arg-type]
    control_stages = Counter(data["control_stage"])  # type: ignore[arg-type]
    duration = float(t[-1] - t[0]) if len(t) else float("nan")  # type: ignore[index]
    return {
        "rows": int(len(t)),
        "duration_s": duration,
        "valid_fraction": float(np.mean(valid)) if len(valid) else float("nan"),
        "phase_slope_hz": slope_hz(t, phase, valid),  # type: ignore[arg-type]
        "reconstructed_phase_slope_hz": slope_hz(t, reconstructed_phase, valid),  # type: ignore[arg-type]
        "timing_resid_rms_ns": detrended_rms_ns(t, phase, valid, rf_hz),  # type: ignore[arg-type]
        "wrapped_phase_rms_rad": float(np.sqrt(np.nanmean(wrapped[valid] ** 2))) if np.any(valid) else float("nan"),
        "wrapped_phase_p95_abs_rad": pct_abs(wrapped[valid], 95.0),
        "reconstructed_wrapped_phase_rms_rad": float(np.sqrt(np.nanmean(reconstructed_wrapped[valid] ** 2))) if np.any(valid) else float("nan"),
        "reconstructed_wrapped_phase_p95_abs_rad": pct_abs(reconstructed_wrapped[valid], 95.0),
        "phase_error_p95_abs_rad": pct_abs(phase_error[valid], 95.0),  # type: ignore[index]
        "phase_band_rad": float(phase_band_rad),
        "phase_band_fraction": float(np.mean(phase_band)) if len(phase_band) else float("nan"),
        "best_phase_band_dwell_s": longest_dwell_s(t, phase_band),  # type: ignore[arg-type]
        "mean_residual_freq_hz": float(np.nanmean(residual_freq[valid])) if np.any(valid) else float("nan"),  # type: ignore[index]
        "p95_abs_residual_freq_hz": pct_abs(residual_freq[valid], 95.0),  # type: ignore[index]
        "mean_nco_freq_correction_hz": float(np.nanmean(nco_freq[valid])) if np.any(valid) else float("nan"),  # type: ignore[index]
        "p95_abs_nco_freq_correction_hz": pct_abs(nco_freq[valid], 95.0),  # type: ignore[index]
        "mean_coherence": float(np.nanmean(coherence[valid])) if np.any(valid) else float("nan"),  # type: ignore[index]
        "p05_coherence": float(np.nanpercentile(coherence[valid], 5.0)) if np.any(valid) else float("nan"),  # type: ignore[index]
        "mean_amp": float(np.nanmean(amp[valid])) if np.any(valid) else float("nan"),  # type: ignore[index]
        "control_cmd_min": float(np.nanmin(cmd)) if np.any(np.isfinite(cmd)) else float("nan"),  # type: ignore[arg-type]
        "control_cmd_max": float(np.nanmax(cmd)) if np.any(np.isfinite(cmd)) else float("nan"),  # type: ignore[arg-type]
        "control_cmd_final": float(cmd[np.isfinite(cmd)][-1]) if np.any(np.isfinite(cmd)) else float("nan"),  # type: ignore[index,arg-type]
        "phase_mode_counts": dict(phase_modes.most_common()),
        "control_stage_counts": dict(control_stages.most_common()),
    }


def plot(data: dict[str, np.ndarray | list[str]], out_dir: Path, prefix: str, rf_hz: float, zoom_start_s: float, zoom_duration_s: float) -> None:
    t_abs = data["t_rel_s"]  # type: ignore[assignment]
    t = t_abs - t_abs[0]  # type: ignore[index,operator]
    phase = data["residual_unwrapped_phase_rad"]  # type: ignore[assignment]
    nco_phase = data["nco_unwrapped_phase_correction_rad"]  # type: ignore[assignment]
    reconstructed_phase = phase + nco_phase  # type: ignore[operator]
    wrapped = wrap_pm_pi(phase)  # type: ignore[arg-type]
    reconstructed_wrapped = wrap_pm_pi(reconstructed_phase)
    nco_freq = data["nco_freq_correction_hz"]  # type: ignore[assignment]
    cmd = data["control_cmd"]  # type: ignore[assignment]
    coherence = data["coherence"]  # type: ignore[assignment]
    timing_ns = phase / (TWO_PI * rf_hz) * 1.0e9  # type: ignore[operator]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].plot(t, wrapped, lw=0.75, color="#d62728")
    axes[1].plot(t, timing_ns, lw=0.75, color="#444444")
    axes[2].plot(t, nco_freq, lw=0.75, color="#1f77b4")
    axes[3].plot(t, cmd, lw=0.75, color="#2ca02c", label="control cmd")
    ax2 = axes[3].twinx()
    ax2.plot(t, coherence, lw=0.5, color="#9467bd", alpha=0.5, label="coherence")
    axes[0].set_ylabel("wrapped phase (rad)")
    axes[1].set_ylabel("raw timing (ns)")
    axes[2].set_ylabel("NCO freq corr (Hz)")
    axes[3].set_ylabel("DAC cmd")
    ax2.set_ylabel("coherence")
    axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Live Adaptive-NCO PLL")
    fig.savefig(out_dir / f"{prefix}_overview.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(t, wrapped, lw=0.75, color="#d62728")
    ax.axhline(0.0, color="#222222", lw=0.8, alpha=0.6)
    ax.axhline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.axhline(-1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.set_title("Live PLL Wrapped Phase, Full Record")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wrapped phase (rad)")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / f"{prefix}_wrapped_phase_full.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(t, reconstructed_wrapped, lw=0.75, color="#ff7f0e")
    ax.axhline(0.0, color="#222222", lw=0.8, alpha=0.6)
    ax.axhline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.axhline(-1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.set_title("Live PLL Reconstructed Physical Wrapped Phase, Full Record")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wrapped phase (rad)")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / f"{prefix}_reconstructed_wrapped_phase_full.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(t, reconstructed_wrapped, lw=0.75, color="#ff7f0e")
    axes[0].set_title("Reconstructed Physical Wrapped Phase")
    axes[1].plot(t, wrapped, lw=0.75, color="#d62728")
    axes[1].set_title("Corrected Residual Wrapped Phase")
    for ax in axes:
        ax.axhline(0.0, color="#222222", lw=0.8, alpha=0.6)
        ax.axhline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
        ax.axhline(-1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
        ax.set_ylabel("phase (rad)")
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel("time (s)")
    fig.suptitle("Live PLL Phase: Physical Observable vs Corrected Residual")
    fig.savefig(out_dir / f"{prefix}_physical_vs_residual_wrapped_phase_full.png", dpi=160)
    plt.close(fig)

    start = min(max(float(zoom_start_s), 0.0), max(float(t[-1] - zoom_duration_s), 0.0))
    stop = start + float(zoom_duration_s)
    mask = (t >= start) & (t <= stop) & np.isfinite(wrapped)
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(t[mask] - start, wrapped[mask], marker=".", ms=2.5, lw=0.8, color="#d62728")
    ax.axhline(0.0, color="#222222", lw=0.8, alpha=0.6)
    ax.axhline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.axhline(-1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.set_title(f"Live PLL Wrapped Phase, {zoom_duration_s:.6g} s Zoom Starting at {start:.6g} s")
    ax.set_xlabel("time within zoom (s)")
    ax.set_ylabel("wrapped phase (rad)")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / f"{prefix}_wrapped_phase_one_sixth_second.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(t[mask] - start, reconstructed_wrapped[mask], marker=".", ms=2.5, lw=0.8, color="#ff7f0e")
    ax.axhline(0.0, color="#222222", lw=0.8, alpha=0.6)
    ax.axhline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.axhline(-1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.set_title(f"Live PLL Reconstructed Physical Wrapped Phase, {zoom_duration_s:.6g} s Zoom Starting at {start:.6g} s")
    ax.set_xlabel("time within zoom (s)")
    ax.set_ylabel("wrapped phase (rad)")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / f"{prefix}_reconstructed_wrapped_phase_one_sixth_second.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(t[mask] - start, reconstructed_wrapped[mask], marker=".", ms=2.5, lw=0.8, color="#ff7f0e")
    axes[0].set_title("Reconstructed Physical Wrapped Phase")
    axes[1].plot(t[mask] - start, wrapped[mask], marker=".", ms=2.5, lw=0.8, color="#d62728")
    axes[1].set_title("Corrected Residual Wrapped Phase")
    for ax in axes:
        ax.axhline(0.0, color="#222222", lw=0.8, alpha=0.6)
        ax.axhline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
        ax.axhline(-1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
        ax.set_ylabel("phase (rad)")
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel("time within zoom (s)")
    fig.suptitle(f"Live PLL Phase, {zoom_duration_s:.6g} s Zoom Starting at {start:.6g} s")
    fig.savefig(out_dir / f"{prefix}_physical_vs_residual_wrapped_phase_one_sixth_second.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rf-hz", type=float, default=50.0e6)
    parser.add_argument("--phase-band-rad", type=float, default=1.0)
    parser.add_argument("--zoom-start-s", type=float, default=30.0)
    parser.add_argument("--zoom-duration-s", type=float, default=1.0 / 6.0)
    args = parser.parse_args()

    out_dir = args.out_dir or args.csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.csv_path.stem
    data = load_csv(args.csv_path)
    metrics = summarize(data, float(args.rf_hz), float(args.phase_band_rad))
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    plot(data, out_dir, prefix, float(args.rf_hz), float(args.zoom_start_s), float(args.zoom_duration_s))
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
