#!/usr/bin/env python3
"""Plot phase-centric open-vs-closed comparisons for the dual-reference controller."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_summary(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return data


def to_float_series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        try:
            vals.append(float(raw))
        except Exception:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float64)


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.asarray([], dtype=bool)
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if window <= 1:
        return np.copy(a)
    for i in range(a.size):
        seg = a[max(0, i - window + 1) : i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.mean(seg))
    return out


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = finite_mask(x, y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    x0 = xv - np.mean(xv)
    denom = float(np.sum(x0 * x0))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x0 * (yv - np.mean(yv))) / denom)


def centered_series(a: np.ndarray) -> np.ndarray:
    mask = np.isfinite(a)
    if not np.any(mask):
        return np.copy(a)
    out = np.copy(a)
    out[mask] = a[mask] - a[np.flatnonzero(mask)[0]]
    return out


def hist_range(*arrays: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size], axis=0)
    if vals.size == 0:
        return (-math.pi, math.pi)
    lo = float(np.nanpercentile(vals, 1.0))
    hi = float(np.nanpercentile(vals, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return (-math.pi, math.pi)
    pad = 0.1 * (hi - lo)
    return (lo - pad, hi + pad)


def plot_phase_dashboard(
    *,
    open_rows: list[dict[str, str]],
    closed_rows: list[dict[str, str]],
    open_summary: dict[str, Any],
    closed_summary: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    open_t = to_float_series(open_rows, "elapsed_s")
    closed_t = to_float_series(closed_rows, "elapsed_s")

    open_obs_unwrapped = centered_series(to_float_series(open_rows, "observer_phase_unwrapped_rad"))
    closed_obs_unwrapped = centered_series(to_float_series(closed_rows, "observer_phase_unwrapped_rad"))
    open_obs_wrapped = to_float_series(open_rows, "observer_phase_wrapped_rad")
    closed_obs_wrapped = to_float_series(closed_rows, "observer_phase_wrapped_rad")
    open_ctrl_err = to_float_series(open_rows, "control_phase_err_filt_rad")
    closed_ctrl_err = to_float_series(closed_rows, "control_phase_err_filt_rad")
    open_ctrl_err_raw = to_float_series(open_rows, "control_phase_err_raw_rad")
    closed_ctrl_err_raw = to_float_series(closed_rows, "control_phase_err_raw_rad")
    open_ctrl_coh = to_float_series(open_rows, "control_coherence")
    closed_ctrl_coh = to_float_series(closed_rows, "control_coherence")
    open_timing = centered_series(to_float_series(open_rows, "observer_timing_proxy_ns"))
    closed_timing = centered_series(to_float_series(closed_rows, "observer_timing_proxy_ns"))

    open_obs_slope_hz = float(open_summary.get("observer_phase_drift_slope_hz", float("nan")))
    closed_obs_slope_hz = float(closed_summary.get("observer_phase_drift_slope_hz", float("nan")))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(open_t, open_obs_unwrapped, color="#b8402a", linewidth=1.8, label="Open loop")
    ax.plot(closed_t, closed_obs_unwrapped, color="#006d77", linewidth=1.8, label="Closed loop")
    ax.set_title("Observer Unwrapped Phase")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Phase change [rad]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    txt = (
        f"open slope: {open_obs_slope_hz:.2f} Hz\n"
        f"closed slope: {closed_obs_slope_hz:.2f} Hz"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9)

    ax = axes[0, 1]
    ax.plot(open_t, open_timing, color="#b8402a", linewidth=1.8, label="Open loop")
    ax.plot(closed_t, closed_timing, color="#006d77", linewidth=1.8, label="Closed loop")
    ax.set_title("Observer Timing Proxy")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Timing change [ns]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[0, 2]
    ax.plot(open_t, open_obs_wrapped, color="#b8402a", linewidth=1.0, alpha=0.85, label="Open loop")
    ax.plot(closed_t, closed_obs_wrapped, color="#006d77", linewidth=1.0, alpha=0.85, label="Closed loop")
    ax.axhline(math.pi, color="#999999", linewidth=0.8, alpha=0.5)
    ax.axhline(-math.pi, color="#999999", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-math.pi * 1.1, math.pi * 1.1)
    ax.set_title("Observer Wrapped Phase")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Wrapped phase [rad]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(open_t, open_ctrl_err_raw, color="#f4a261", linewidth=0.9, alpha=0.5)
    ax.plot(closed_t, closed_ctrl_err_raw, color="#83c5be", linewidth=0.9, alpha=0.5)
    ax.plot(open_t, open_ctrl_err, color="#b8402a", linewidth=1.8, label="Open loop filtered")
    ax.plot(closed_t, closed_ctrl_err, color="#006d77", linewidth=1.8, label="Closed loop filtered")
    ax.axhline(0.0, color="#999999", linewidth=0.8, alpha=0.6)
    ax.set_title("Control Phase Error")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Phase error [rad]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 1]
    hmin, hmax = hist_range(open_ctrl_err, closed_ctrl_err)
    bins = np.linspace(hmin, hmax, 70)
    ax.hist(open_ctrl_err[np.isfinite(open_ctrl_err)], bins=bins, density=True, alpha=0.45, color="#b8402a", label="Open loop")
    ax.hist(closed_ctrl_err[np.isfinite(closed_ctrl_err)], bins=bins, density=True, alpha=0.45, color="#006d77", label="Closed loop")
    ax.axvline(0.0, color="#999999", linewidth=0.8, alpha=0.6)
    ax.set_title("Filtered Control Phase Error Histogram")
    ax.set_xlabel("Phase error [rad]")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 2]
    ax.plot(open_t, rolling_mean(open_ctrl_coh, 31), color="#b8402a", linewidth=1.8, label="Open loop")
    ax.plot(closed_t, rolling_mean(closed_ctrl_coh, 31), color="#006d77", linewidth=1.8, label="Closed loop")
    ax.axhline(0.025, color="#999999", linewidth=0.8, alpha=0.6, linestyle="--")
    ax.set_title("Control Coherence")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Coherence")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.suptitle(
        "Dual-Reference PI Controller Phase Comparison\n"
        "Observer traces judge timing drift; control traces show what the loop is seeing.",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    improvement_pct = float("nan")
    open_drift = float(open_summary.get("observer_timing_drift_samples_per_s", float("nan")))
    closed_drift = float(closed_summary.get("observer_timing_drift_samples_per_s", float("nan")))
    if np.isfinite(open_drift) and np.isfinite(closed_drift) and abs(open_drift) > 1e-12:
        improvement_pct = float(100.0 * (abs(open_drift) - abs(closed_drift)) / abs(open_drift))

    return {
        "open_observer_phase_slope_hz": open_obs_slope_hz,
        "closed_observer_phase_slope_hz": closed_obs_slope_hz,
        "open_timing_drift_samples_per_s": open_drift,
        "closed_timing_drift_samples_per_s": closed_drift,
        "timing_drift_improvement_pct": improvement_pct,
        "open_mean_abs_control_phase_err_rad": float(open_summary.get("mean_abs_control_phase_err_rad", float("nan"))),
        "closed_mean_abs_control_phase_err_rad": float(closed_summary.get("mean_abs_control_phase_err_rad", float("nan"))),
        "open_median_control_coherence": float(open_summary.get("median_control_coherence", float("nan"))),
        "closed_median_control_coherence": float(closed_summary.get("median_control_coherence", float("nan"))),
        "closed_control_reference_walk_hz": float(closed_summary.get("control_reference_walk_hz", float("nan"))),
        "dashboard_path": str(output_path.resolve()),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot open-vs-closed phase comparison for the dual-reference PI controller.")
    p.add_argument("--open-csv", type=Path, required=True)
    p.add_argument("--open-summary", type=Path, required=True)
    p.add_argument("--closed-csv", type=Path, required=True)
    p.add_argument("--closed-summary", type=Path, required=True)
    p.add_argument("--dashboard-out", type=Path, required=True)
    p.add_argument("--metrics-out", type=Path, default=None)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    metrics = plot_phase_dashboard(
        open_rows=load_csv_rows(args.open_csv),
        closed_rows=load_csv_rows(args.closed_csv),
        open_summary=load_summary(args.open_summary),
        closed_summary=load_summary(args.closed_summary),
        output_path=args.dashboard_out,
    )
    if args.metrics_out is not None:
        write_json(args.metrics_out, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
