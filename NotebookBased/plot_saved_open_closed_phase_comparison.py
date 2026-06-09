#!/usr/bin/env python3
"""Plot phase-centric open-vs-closed comparisons for saved NotebookBased runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_summary(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected dict summary in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def to_float_series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        try:
            vals.append(float(row.get(key, "nan")))
        except Exception:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float64)


def first_finite_series(rows: list[dict[str, str]], *keys: str) -> np.ndarray:
    for key in keys:
        arr = to_float_series(rows, key)
        if np.any(np.isfinite(arr)):
            return arr
    return np.full(len(rows), np.nan, dtype=np.float64)


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


def centered_series(a: np.ndarray) -> np.ndarray:
    mask = np.isfinite(a)
    if not np.any(mask):
        return np.copy(a)
    out = np.copy(a)
    out[mask] = a[mask] - a[np.flatnonzero(mask)[0]]
    return out


def hist_range(*arrays: np.ndarray) -> tuple[float, float]:
    finite_chunks = [a[np.isfinite(a)] for a in arrays if a.size]
    if not finite_chunks:
        return (-math.pi, math.pi)
    vals = np.concatenate(finite_chunks, axis=0)
    if vals.size == 0:
        return (-math.pi, math.pi)
    lo = float(np.nanpercentile(vals, 1.0))
    hi = float(np.nanpercentile(vals, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return (-math.pi, math.pi)
    pad = 0.1 * (hi - lo)
    return (lo - pad, hi + pad)


def infer_paths(run_dir: Path, nsamp: int) -> dict[str, Path]:
    return {
        "monitor_csv": run_dir / "monitor" / f"nsamp_{nsamp}" / f"monitor_nsamp{nsamp}.csv",
        "monitor_summary": run_dir / "monitor" / f"nsamp_{nsamp}" / f"monitor_nsamp{nsamp}_summary.json",
        "pid_csv": run_dir / "pid" / f"nsamp_{nsamp}" / f"pid_nsamp{nsamp}.csv",
        "pid_summary": run_dir / "pid" / f"nsamp_{nsamp}" / f"pid_nsamp{nsamp}_summary.json",
    }


def timing_drift_samples_per_s(summary: dict[str, Any]) -> float:
    direct = float(summary.get("primary_timing_drift_samples_per_s", float("nan")))
    if np.isfinite(direct):
        return direct
    drift_ns_per_s = float(summary.get("primary_timing_drift_ns_per_s", float("nan")))
    sample_rate_hz = float(summary.get("sample_rate_hz", float("nan")))
    if not np.isfinite(drift_ns_per_s) or sample_rate_hz <= 0.0:
        return float("nan")
    return float(drift_ns_per_s * sample_rate_hz / 1e9)


def plot_phase_dashboard(
    *,
    open_rows: list[dict[str, str]],
    closed_rows: list[dict[str, str]],
    open_summary: dict[str, Any],
    closed_summary: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    open_t = first_finite_series(open_rows, "elapsed_s")
    closed_t = first_finite_series(closed_rows, "elapsed_s")

    open_obs_unwrapped = centered_series(
        first_finite_series(open_rows, "observer_phase_unwrapped_rad", "phase_unwrapped_rad")
    )
    closed_obs_unwrapped = centered_series(
        first_finite_series(closed_rows, "observer_phase_unwrapped_rad", "phase_unwrapped_rad")
    )
    open_obs_wrapped = first_finite_series(open_rows, "observer_phase_wrapped_rad", "phase_wrapped_rad")
    closed_obs_wrapped = first_finite_series(closed_rows, "observer_phase_wrapped_rad", "phase_wrapped_rad")
    open_ctrl_err = first_finite_series(open_rows, "phase_err_filt_rad", "control_phase_err_filt_rad")
    closed_ctrl_err = first_finite_series(closed_rows, "phase_err_filt_rad", "control_phase_err_filt_rad")
    open_ctrl_err_raw = first_finite_series(open_rows, "phase_err_raw_rad", "control_phase_err_raw_rad")
    closed_ctrl_err_raw = first_finite_series(closed_rows, "phase_err_raw_rad", "control_phase_err_raw_rad")
    open_ctrl_coh = first_finite_series(open_rows, "control_coherence", "coherence", "observer_coherence")
    closed_ctrl_coh = first_finite_series(closed_rows, "control_coherence", "coherence", "observer_coherence")
    open_timing = centered_series(first_finite_series(open_rows, "observer_tau_err_ns"))
    closed_timing = centered_series(first_finite_series(closed_rows, "observer_tau_err_ns"))

    open_obs_slope_hz = float(
        open_summary.get("observer_phase_drift_slope_full_hz", open_summary.get("phase_drift_slope_full_hz", float("nan")))
    )
    closed_obs_slope_hz = float(
        closed_summary.get(
            "observer_phase_drift_slope_full_hz",
            closed_summary.get("phase_drift_slope_full_hz", float("nan")),
        )
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(open_t, open_obs_unwrapped, color="#b8402a", linewidth=1.8, label="Open loop")
    ax.plot(closed_t, closed_obs_unwrapped, color="#006d77", linewidth=1.8, label="Closed loop")
    ax.set_title("Observer Unwrapped Phase")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Phase change [rad]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.98,
        f"open slope: {open_obs_slope_hz:.2f} Hz\nclosed slope: {closed_obs_slope_hz:.2f} Hz",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

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
        "NotebookBased Controller Phase Comparison\n"
        "Observer traces judge drift; control traces show whether pull-in actually approaches a phase-stable regime.",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    open_drift = timing_drift_samples_per_s(open_summary)
    closed_drift = timing_drift_samples_per_s(closed_summary)
    if np.isfinite(open_drift) and np.isfinite(closed_drift) and abs(open_drift) > 1e-12:
        timing_improvement_pct = float(100.0 * (abs(open_drift) - abs(closed_drift)) / abs(open_drift))
    else:
        timing_improvement_pct = float("nan")

    open_phase_slope = float(open_summary.get("phase_drift_slope_full_hz", float("nan")))
    closed_phase_slope = float(closed_summary.get("phase_drift_slope_full_hz", float("nan")))
    if np.isfinite(open_phase_slope) and np.isfinite(closed_phase_slope) and abs(open_phase_slope) > 1e-12:
        phase_improvement_pct = float(100.0 * (abs(open_phase_slope) - abs(closed_phase_slope)) / abs(open_phase_slope))
    else:
        phase_improvement_pct = float("nan")

    return {
        "open_observer_phase_slope_hz": open_obs_slope_hz,
        "closed_observer_phase_slope_hz": closed_obs_slope_hz,
        "open_timing_drift_samples_per_s": open_drift,
        "closed_timing_drift_samples_per_s": closed_drift,
        "timing_drift_improvement_pct": timing_improvement_pct,
        "phase_drift_improvement_pct": phase_improvement_pct,
        "open_mean_abs_phase_err_rad": float(open_summary.get("mean_abs_phase_err_rad", float("nan"))),
        "closed_mean_abs_phase_err_rad": float(closed_summary.get("mean_abs_phase_err_rad", float("nan"))),
        "open_median_control_coherence": float(open_summary.get("median_coherence", float("nan"))),
        "closed_median_control_coherence": float(closed_summary.get("median_coherence", float("nan"))),
        "closed_phase_mode_fraction": float(closed_summary.get("phase_mode_fraction", float("nan"))),
        "closed_phase_updates_applied": int(closed_summary.get("phase_updates_applied", 0)),
        "closed_stage_switches": int(closed_summary.get("stage_switches", 0)),
        "dashboard_path": str(output_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot saved open-vs-closed phase comparison for NotebookBased runs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--nsamp", type=int, default=32768)
    parser.add_argument("--output-name", type=str, default="open_closed_phase_dashboard.png")
    parser.add_argument("--metrics-name", type=str, default="phase_dashboard_metrics.json")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    paths = infer_paths(run_dir, int(args.nsamp))
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {key}: {path}")

    comparison_dir = run_dir / "comparison"
    dashboard_path = comparison_dir / str(args.output_name)
    metrics_path = comparison_dir / str(args.metrics_name)

    metrics = plot_phase_dashboard(
        open_rows=load_csv_rows(paths["monitor_csv"]),
        closed_rows=load_csv_rows(paths["pid_csv"]),
        open_summary=load_summary(paths["monitor_summary"]),
        closed_summary=load_summary(paths["pid_summary"]),
        output_path=dashboard_path,
    )
    write_json(metrics_path, metrics)

    print(f"Wrote dashboard: {dashboard_path}")
    print(f"Wrote metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
