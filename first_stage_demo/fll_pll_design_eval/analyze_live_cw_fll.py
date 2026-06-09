#!/usr/bin/env python3
"""Analyze live cw_vcxo_discipline.py frequency-only/FLL logs."""

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
    numeric = [
        "t_rel_s",
        "phase_rad",
        "unwrapped_phase_rad",
        "f_off_hz",
        "amp",
        "coherence",
        "control_cmd",
        "requested_cmd",
        "dac_code_cmd",
        "dac_volts",
    ]
    text = ["control_stage", "status", "gate_reason"]
    cols: dict[str, list[float] | list[str]] = {key: [] for key in numeric + text}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in numeric:
                cols[key].append(parse_float(row.get(key)))  # type: ignore[union-attr]
            for key in text:
                cols[key].append(str(row.get(key, "")).strip())  # type: ignore[union-attr]
    out: dict[str, np.ndarray | list[str]] = {}
    for key in numeric:
        out[key] = np.asarray(cols[key], dtype=np.float64)  # type: ignore[arg-type]
    for key in text:
        out[key] = list(cols[key])  # type: ignore[arg-type]
    return out


def slope_hz(t: np.ndarray, phase: np.ndarray, mask: np.ndarray) -> float:
    fit = mask & np.isfinite(t) & np.isfinite(phase)
    if np.count_nonzero(fit) < 3:
        return float("nan")
    t0 = float(t[fit][0])
    coeff = np.polyfit(t[fit] - t0, phase[fit], 1)
    return float(coeff[0] / TWO_PI)


def timing_rms_ns(t: np.ndarray, phase: np.ndarray, mask: np.ndarray, rf_hz: float) -> float:
    fit = mask & np.isfinite(t) & np.isfinite(phase)
    if np.count_nonzero(fit) < 3:
        return float("nan")
    t0 = float(t[fit][0])
    coeff = np.polyfit(t[fit] - t0, phase[fit], 1)
    resid = phase[fit] - (coeff[0] * (t[fit] - t0) + coeff[1])
    ns = resid / (TWO_PI * rf_hz) * 1.0e9
    return float(np.sqrt(np.mean(ns * ns)))


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
    phase = data["unwrapped_phase_rad"]  # type: ignore[assignment]
    wrapped = wrap_pm_pi(phase)  # type: ignore[arg-type]
    f_off = data["f_off_hz"]  # type: ignore[assignment]
    amp = data["amp"]  # type: ignore[assignment]
    coherence = data["coherence"]  # type: ignore[assignment]
    cmd = data["control_cmd"]  # type: ignore[assignment]
    valid = np.isfinite(t) & np.isfinite(phase) & np.isfinite(amp) & np.isfinite(coherence) & (amp >= 0.02) & (coherence >= 0.2)  # type: ignore[operator]
    band = valid & np.isfinite(wrapped) & (np.abs(wrapped) <= phase_band_rad)
    return {
        "rows": int(len(t)),
        "duration_s": float(t[-1] - t[0]) if len(t) else float("nan"),  # type: ignore[index]
        "valid_fraction": float(np.mean(valid)) if len(valid) else float("nan"),
        "phase_slope_hz": slope_hz(t, phase, valid),  # type: ignore[arg-type]
        "timing_resid_rms_ns": timing_rms_ns(t, phase, valid, rf_hz),  # type: ignore[arg-type]
        "wrapped_phase_rms_rad": float(np.sqrt(np.nanmean(wrapped[valid] ** 2))) if np.any(valid) else float("nan"),
        "wrapped_phase_p95_abs_rad": pct_abs(wrapped[valid], 95.0),
        "phase_band_rad": float(phase_band_rad),
        "phase_band_fraction": float(np.mean(band)) if len(band) else float("nan"),
        "best_phase_band_dwell_s": longest_dwell_s(t, band),  # type: ignore[arg-type]
        "mean_f_off_hz": float(np.nanmean(f_off[valid])) if np.any(valid) else float("nan"),  # type: ignore[index]
        "median_f_off_hz": float(np.nanmedian(f_off[valid])) if np.any(valid) else float("nan"),  # type: ignore[index]
        "mean_abs_f_off_hz": float(np.nanmean(np.abs(f_off[valid]))) if np.any(valid) else float("nan"),  # type: ignore[index]
        "p95_abs_f_off_hz": pct_abs(f_off[valid], 95.0),  # type: ignore[index]
        "p99_abs_f_off_hz": pct_abs(f_off[valid], 99.0),  # type: ignore[index]
        "mean_coherence": float(np.nanmean(coherence[valid])) if np.any(valid) else float("nan"),  # type: ignore[index]
        "p05_coherence": float(np.nanpercentile(coherence[valid], 5.0)) if np.any(valid) else float("nan"),  # type: ignore[index]
        "mean_amp": float(np.nanmean(amp[valid])) if np.any(valid) else float("nan"),  # type: ignore[index]
        "control_cmd_min": float(np.nanmin(cmd)) if np.any(np.isfinite(cmd)) else float("nan"),  # type: ignore[arg-type]
        "control_cmd_max": float(np.nanmax(cmd)) if np.any(np.isfinite(cmd)) else float("nan"),  # type: ignore[arg-type]
        "control_cmd_final": float(cmd[np.isfinite(cmd)][-1]) if np.any(np.isfinite(cmd)) else float("nan"),  # type: ignore[index,arg-type]
        "control_stage_counts": dict(Counter(data["control_stage"]).most_common()),  # type: ignore[arg-type]
        "status_counts": dict(Counter(data["status"]).most_common()),  # type: ignore[arg-type]
    }


def plot(data: dict[str, np.ndarray | list[str]], out_dir: Path, prefix: str, rf_hz: float, zoom_start_s: float, zoom_duration_s: float) -> None:
    t_abs = data["t_rel_s"]  # type: ignore[assignment]
    t = t_abs - t_abs[0]  # type: ignore[index,operator]
    phase = data["unwrapped_phase_rad"]  # type: ignore[assignment]
    wrapped = wrap_pm_pi(phase)  # type: ignore[arg-type]
    timing_ns = phase / (TWO_PI * rf_hz) * 1.0e9  # type: ignore[operator]
    f_off = data["f_off_hz"]  # type: ignore[assignment]
    cmd = data["control_cmd"]  # type: ignore[assignment]
    coherence = data["coherence"]  # type: ignore[assignment]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].plot(t, wrapped, color="#1f77b4", lw=0.75)
    axes[1].plot(t, timing_ns, color="#444444", lw=0.75)
    axes[2].plot(t, f_off, color="#d62728", lw=0.75)
    axes[3].plot(t, cmd, color="#2ca02c", lw=0.75)
    ax2 = axes[3].twinx()
    ax2.plot(t, coherence, color="#9467bd", lw=0.5, alpha=0.5)
    axes[0].set_ylabel("wrapped phase (rad)")
    axes[1].set_ylabel("raw timing (ns)")
    axes[2].set_ylabel("f_off (Hz)")
    axes[3].set_ylabel("DAC cmd")
    ax2.set_ylabel("coherence")
    axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Live Frequency-Only FLL")
    fig.savefig(out_dir / f"{prefix}_overview.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(t, wrapped, color="#1f77b4", lw=0.75)
    ax.axhline(0.0, color="#222222", lw=0.8, alpha=0.6)
    ax.axhline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.axhline(-1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.set_title("Live FLL Wrapped Phase, Full Record")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wrapped phase (rad)")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / f"{prefix}_wrapped_phase_full.png", dpi=160)
    plt.close(fig)

    start = min(max(float(zoom_start_s), 0.0), max(float(t[-1] - zoom_duration_s), 0.0))
    stop = start + float(zoom_duration_s)
    mask = (t >= start) & (t <= stop) & np.isfinite(wrapped)
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(t[mask] - start, wrapped[mask], marker=".", ms=2.5, lw=0.8, color="#1f77b4")
    ax.axhline(0.0, color="#222222", lw=0.8, alpha=0.6)
    ax.axhline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.axhline(-1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.set_title(f"Live FLL Wrapped Phase, {zoom_duration_s:.6g} s Zoom Starting at {start:.6g} s")
    ax.set_xlabel("time within zoom (s)")
    ax.set_ylabel("wrapped phase (rad)")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / f"{prefix}_wrapped_phase_one_sixth_second.png", dpi=160)
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
