#!/usr/bin/env python3
"""Replay-design and evaluate software FLL/PLL controllers on CW phase logs.

The input is an open-loop or pseudo-open-loop CSV from ``cw_vcxo_discipline.py``.
The controllers are intentionally software/NCO style controllers: they correct
the measured phase trace immediately in replay, which tests the architecture
recommended in the handoff separately from the slow DAC/VCXO actuator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TWO_PI = 2.0 * math.pi
DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1]
    / "logs"
    / "pseudo_open_loop_after_30s_freq_centering_detected_tone_50mhz_5min_20260514.csv"
)


def finite_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def wrap_pm_pi(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + math.pi) % TWO_PI - math.pi


def unwrap_from_wrapped(wrapped: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(wrapped, dtype=np.float64))


@dataclass(frozen=True)
class Trace:
    t_s: np.ndarray
    phase_rad: np.ndarray
    f_off_hz: np.ndarray
    amp: np.ndarray
    coherence: np.ndarray


@dataclass(frozen=True)
class LoopResult:
    label: str
    residual_wrapped_phase_rad: np.ndarray
    residual_unwrapped_phase_rad: np.ndarray
    correction_phase_rad: np.ndarray
    correction_freq_hz: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class LoopConfig:
    label: str
    kind: str
    bandwidth_hz: float
    damping: float
    max_freq_hz: float
    fll_alpha: float = 0.0


def load_trace(csv_path: Path, max_rows: int | None = None) -> Trace:
    columns = {
        "t_rel_s": [],
        "unwrapped_phase_rad": [],
        "f_off_hz": [],
        "amp": [],
        "coherence": [],
    }
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            for key in columns:
                columns[key].append(finite_float(row.get(key)))

    t = np.asarray(columns["t_rel_s"], dtype=np.float64)
    phase = np.asarray(columns["unwrapped_phase_rad"], dtype=np.float64)
    f_off = np.asarray(columns["f_off_hz"], dtype=np.float64)
    amp = np.asarray(columns["amp"], dtype=np.float64)
    coherence = np.asarray(columns["coherence"], dtype=np.float64)

    finite = np.isfinite(t) & np.isfinite(phase)
    if np.count_nonzero(finite) < 4:
        raise RuntimeError(f"Not enough finite phase rows in {csv_path}")
    return Trace(
        t_s=t[finite],
        phase_rad=phase[finite],
        f_off_hz=f_off[finite],
        amp=amp[finite],
        coherence=coherence[finite],
    )


def median_dt_s(t_s: np.ndarray) -> float:
    dt = np.diff(t_s)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        raise RuntimeError("Trace has no positive sample intervals.")
    return float(np.median(dt))


def evaluate_open_loop(trace: Trace, amp_min: float, coherence_min: float) -> LoopResult:
    valid = (
        np.isfinite(trace.phase_rad)
        & np.isfinite(trace.amp)
        & np.isfinite(trace.coherence)
        & (trace.amp >= amp_min)
        & (trace.coherence >= coherence_min)
    )
    wrapped = wrap_pm_pi(trace.phase_rad)
    return LoopResult(
        label="open_loop",
        residual_wrapped_phase_rad=np.asarray(wrapped, dtype=np.float64),
        residual_unwrapped_phase_rad=np.asarray(trace.phase_rad, dtype=np.float64),
        correction_phase_rad=np.zeros_like(trace.phase_rad),
        correction_freq_hz=np.zeros_like(trace.phase_rad),
        valid=valid,
    )


def run_fll(
    trace: Trace,
    cfg: LoopConfig,
    amp_min: float,
    coherence_min: float,
) -> LoopResult:
    n = trace.t_s.size
    residual_wrapped = np.full(n, np.nan, dtype=np.float64)
    residual_unwrapped = np.full(n, np.nan, dtype=np.float64)
    correction_phase = np.zeros(n, dtype=np.float64)
    correction_freq = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    alpha = float(cfg.fll_alpha)
    if alpha <= 0.0:
        dt = median_dt_s(trace.t_s)
        alpha = 1.0 - math.exp(-TWO_PI * max(cfg.bandwidth_hz, 1.0e-6) * dt)
    alpha = min(max(alpha, 0.0), 1.0)

    phase_corr = 0.0
    freq_cmd = 0.0
    prev_t = float(trace.t_s[0])
    prev_wrapped: float | None = None
    unwrapped_resid = 0.0

    for i in range(n):
        dt = max(float(trace.t_s[i] - prev_t), 0.0) if i > 0 else 0.0
        phase_corr += TWO_PI * freq_cmd * dt
        raw_resid = float(trace.phase_rad[i] - phase_corr)
        wrapped = float(wrap_pm_pi(raw_resid))
        good = bool(
            np.isfinite(wrapped)
            and np.isfinite(trace.f_off_hz[i])
            and np.isfinite(trace.amp[i])
            and np.isfinite(trace.coherence[i])
            and trace.amp[i] >= amp_min
            and trace.coherence[i] >= coherence_min
        )
        if prev_wrapped is None:
            unwrapped_resid = wrapped
        else:
            unwrapped_resid += float(wrap_pm_pi(wrapped - prev_wrapped))
        prev_wrapped = wrapped
        if good:
            freq_cmd = (1.0 - alpha) * freq_cmd + alpha * float(trace.f_off_hz[i])
            freq_cmd = min(max(freq_cmd, -cfg.max_freq_hz), cfg.max_freq_hz)
        residual_wrapped[i] = wrapped
        residual_unwrapped[i] = unwrapped_resid
        correction_phase[i] = phase_corr
        correction_freq[i] = freq_cmd
        valid[i] = good
        prev_t = float(trace.t_s[i])

    return LoopResult(cfg.label, residual_wrapped, residual_unwrapped, correction_phase, correction_freq, valid)


def run_pll(
    trace: Trace,
    cfg: LoopConfig,
    amp_min: float,
    coherence_min: float,
) -> LoopResult:
    n = trace.t_s.size
    residual_wrapped = np.full(n, np.nan, dtype=np.float64)
    residual_unwrapped = np.full(n, np.nan, dtype=np.float64)
    correction_phase = np.zeros(n, dtype=np.float64)
    correction_freq = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    # Continuous type-II PLL gains mapped to a variable-dt digital NCO.
    omega_n = TWO_PI * max(float(cfg.bandwidth_hz), 1.0e-6)
    kp_hz_per_rad = (2.0 * float(cfg.damping) * omega_n) / TWO_PI
    ki_hz_per_rad_s = (omega_n * omega_n) / TWO_PI

    phase_corr = 0.0
    freq_integrator = 0.0
    prev_t = float(trace.t_s[0])
    prev_wrapped: float | None = None
    unwrapped_resid = 0.0

    for i in range(n):
        dt = max(float(trace.t_s[i] - prev_t), 0.0) if i > 0 else 0.0
        phase_corr += TWO_PI * freq_integrator * dt
        raw_resid = float(trace.phase_rad[i] - phase_corr)
        wrapped = float(wrap_pm_pi(raw_resid))
        good = bool(
            np.isfinite(wrapped)
            and np.isfinite(trace.amp[i])
            and np.isfinite(trace.coherence[i])
            and trace.amp[i] >= amp_min
            and trace.coherence[i] >= coherence_min
        )
        if prev_wrapped is None:
            unwrapped_resid = wrapped
        else:
            unwrapped_resid += float(wrap_pm_pi(wrapped - prev_wrapped))
        prev_wrapped = wrapped
        if good:
            # Positive residual means the input phase is ahead; advance the NCO.
            freq_integrator += ki_hz_per_rad_s * wrapped * dt
            freq_integrator = min(max(freq_integrator, -cfg.max_freq_hz), cfg.max_freq_hz)
            freq_cmd = freq_integrator + kp_hz_per_rad * wrapped
            freq_cmd = min(max(freq_cmd, -cfg.max_freq_hz), cfg.max_freq_hz)
            phase_corr += TWO_PI * freq_cmd * dt
            correction_freq[i] = freq_cmd
        else:
            correction_freq[i] = freq_integrator
        correction_phase[i] = phase_corr
        residual_wrapped[i] = wrapped
        residual_unwrapped[i] = unwrapped_resid
        valid[i] = good
        prev_t = float(trace.t_s[i])

    # Recompute residual after the last intra-sample correction for consistency.
    residual_wrapped = np.asarray(wrap_pm_pi(trace.phase_rad - correction_phase), dtype=np.float64)
    residual_unwrapped = unwrap_from_wrapped(residual_wrapped)
    return LoopResult(cfg.label, residual_wrapped, residual_unwrapped, correction_phase, correction_freq, valid)


def robust_polyfit_slope_hz(t_s: np.ndarray, phase_rad: np.ndarray, valid: np.ndarray) -> float:
    mask = valid & np.isfinite(t_s) & np.isfinite(phase_rad)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    coeff = np.polyfit(t_s[mask] - t_s[mask][0], phase_rad[mask], 1)
    return float(coeff[0] / TWO_PI)


def robust_polyfit_phase_line(t_s: np.ndarray, phase_rad: np.ndarray, valid: np.ndarray) -> np.ndarray:
    line = np.full_like(phase_rad, np.nan, dtype=np.float64)
    mask = valid & np.isfinite(t_s) & np.isfinite(phase_rad)
    if np.count_nonzero(mask) < 3:
        return line
    t0 = float(t_s[mask][0])
    coeff = np.polyfit(t_s[mask] - t0, phase_rad[mask], 1)
    line[:] = coeff[0] * (t_s - t0) + coeff[1]
    return line


def residual_frequency_hz(t_s: np.ndarray, phase_rad: np.ndarray) -> np.ndarray:
    out = np.full_like(phase_rad, np.nan, dtype=np.float64)
    dt = np.diff(t_s)
    dp = np.diff(phase_rad)
    good = np.isfinite(dt) & (dt > 0.0) & np.isfinite(dp)
    out[1:][good] = dp[good] / (TWO_PI * dt[good])
    return out


def percentile_abs(values: np.ndarray, pct: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(np.abs(finite), pct))


def longest_true_dwell_s(t_s: np.ndarray, mask: np.ndarray) -> float:
    best = 0.0
    start: float | None = None
    prev = float("nan")
    for t, ok in zip(t_s, mask):
        if ok and start is None:
            start = float(t)
        elif not ok and start is not None:
            best = max(best, float(prev) - start)
            start = None
        prev = float(t)
    if start is not None and np.isfinite(prev):
        best = max(best, prev - start)
    return float(best)


def top_psd_peaks(t_s: np.ndarray, x: np.ndarray, min_hz: float, max_hz: float, count: int = 8) -> list[dict[str, float]]:
    mask = np.isfinite(t_s) & np.isfinite(x)
    if np.count_nonzero(mask) < 16:
        return []
    t = t_s[mask]
    y = x[mask]
    dt = median_dt_s(t)
    y = y - np.mean(y)
    window = np.hanning(y.size)
    spec = np.fft.rfft(y * window)
    freqs = np.fft.rfftfreq(y.size, d=dt)
    psd = (np.abs(spec) ** 2) / max(float(np.sum(window * window)), 1.0)
    band = (freqs >= min_hz) & (freqs <= max_hz) & np.isfinite(psd)
    idx = np.flatnonzero(band)
    if idx.size == 0:
        return []
    order = idx[np.argsort(psd[idx])[::-1]]
    selected: list[int] = []
    min_sep_hz = max((max_hz - min_hz) / 200.0, 0.01)
    for candidate in order:
        hz = float(freqs[candidate])
        if all(abs(hz - float(freqs[old])) >= min_sep_hz for old in selected):
            selected.append(int(candidate))
        if len(selected) >= count:
            break
    return [
        {
            "hz": float(freqs[i]),
            "period_s": float(1.0 / freqs[i]) if freqs[i] > 0.0 else float("inf"),
            "psd": float(psd[i]),
        }
        for i in selected
    ]


def summarize(trace: Trace, result: LoopResult, rf_hz: float, phase_band_rad: float) -> dict[str, object]:
    valid = result.valid & np.isfinite(result.residual_unwrapped_phase_rad)
    resid_freq = residual_frequency_hz(trace.t_s, result.residual_unwrapped_phase_rad)
    timing_ns = result.residual_unwrapped_phase_rad / (TWO_PI * rf_hz) * 1.0e9
    slope_hz = robust_polyfit_slope_hz(trace.t_s, result.residual_unwrapped_phase_rad, valid)
    detrended = np.full_like(timing_ns, np.nan)
    if np.count_nonzero(valid) >= 3 and np.isfinite(slope_hz):
        phase_line = robust_polyfit_phase_line(trace.t_s, result.residual_unwrapped_phase_rad, valid)
        detrended_phase = result.residual_unwrapped_phase_rad - phase_line
        detrended = detrended_phase / (TWO_PI * rf_hz) * 1.0e9

    valid_timing = timing_ns[valid & np.isfinite(timing_ns)]
    detrended_valid = detrended[valid & np.isfinite(detrended)]
    resid_freq_valid = resid_freq[valid & np.isfinite(resid_freq)]
    phase_good = (
        valid
        & np.isfinite(result.residual_wrapped_phase_rad)
        & (np.abs(result.residual_wrapped_phase_rad) <= phase_band_rad)
    )
    duration = float(trace.t_s[-1] - trace.t_s[0])

    minute_segments = []
    for start in np.arange(0.0, duration, 60.0):
        stop = min(start + 60.0, duration + 1.0e-9)
        seg = (trace.t_s >= trace.t_s[0] + start) & (trace.t_s < trace.t_s[0] + stop) & valid
        if np.count_nonzero(seg) < 3:
            continue
        minute_segments.append(
            {
                "start_s": float(start),
                "phase_slope_hz": robust_polyfit_slope_hz(trace.t_s, result.residual_unwrapped_phase_rad, seg),
                "median_abs_residual_freq_hz": percentile_abs(resid_freq[seg], 50.0),
                "timing_resid_rms_ns": float(np.sqrt(np.mean(detrended[seg] ** 2))),
                "phase_band_fraction": float(np.mean(phase_good[seg])),
            }
        )

    return {
        "label": result.label,
        "rows": int(trace.t_s.size),
        "valid_fraction": float(np.mean(valid)),
        "duration_s": duration,
        "phase_slope_hz": slope_hz,
        "mean_abs_residual_freq_hz": float(np.mean(np.abs(resid_freq_valid))) if resid_freq_valid.size else float("nan"),
        "median_abs_residual_freq_hz": percentile_abs(resid_freq_valid, 50.0),
        "p95_abs_residual_freq_hz": percentile_abs(resid_freq_valid, 95.0),
        "p99_abs_residual_freq_hz": percentile_abs(resid_freq_valid, 99.0),
        "mean_coherence": float(np.nanmean(trace.coherence[valid])) if np.any(valid) else float("nan"),
        "p05_coherence": float(np.nanpercentile(trace.coherence[valid], 5.0)) if np.any(valid) else float("nan"),
        "raw_timing_pp_ns": float(np.nanmax(valid_timing) - np.nanmin(valid_timing)) if valid_timing.size else float("nan"),
        "timing_resid_rms_ns": float(np.sqrt(np.mean(detrended_valid * detrended_valid))) if detrended_valid.size else float("nan"),
        "timing_resid_p95_abs_ns": percentile_abs(detrended_valid, 95.0),
        "wrapped_phase_rms_rad": float(np.sqrt(np.nanmean(result.residual_wrapped_phase_rad[valid] ** 2))) if np.any(valid) else float("nan"),
        "wrapped_phase_p95_abs_rad": percentile_abs(result.residual_wrapped_phase_rad[valid], 95.0),
        "phase_band_rad": float(phase_band_rad),
        "phase_band_fraction": float(np.mean(phase_good)) if phase_good.size else float("nan"),
        "best_phase_band_dwell_s": longest_true_dwell_s(trace.t_s, phase_good),
        "minute_segments": minute_segments,
        "top_wrapped_phase_psd_peaks_0p02_to_20hz": top_psd_peaks(
            trace.t_s, result.residual_wrapped_phase_rad, 0.02, 20.0
        ),
        "top_raw_timing_psd_peaks_0p02_to_20hz": top_psd_peaks(trace.t_s, timing_ns, 0.02, 20.0),
    }


def plot_results(trace: Trace, results: Iterable[LoopResult], metrics: dict[str, dict[str, object]], out_dir: Path, rf_hz: float) -> None:
    results = list(results)
    colors = {
        "open_loop": "#333333",
        "fll_slow_frequency_centering": "#1f77b4",
        "pll_fast_phase_lock": "#d62728",
    }
    t = trace.t_s - trace.t_s[0]

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    for result in results:
        c = colors.get(result.label, None)
        timing_ns = result.residual_unwrapped_phase_rad / (TWO_PI * rf_hz) * 1.0e9
        resid_freq = residual_frequency_hz(trace.t_s, result.residual_unwrapped_phase_rad)
        axes[0].plot(t, result.residual_wrapped_phase_rad, lw=0.8, alpha=0.85, label=result.label, color=c)
        axes[1].plot(t, timing_ns, lw=0.8, alpha=0.85, label=result.label, color=c)
        axes[2].plot(t, resid_freq, lw=0.7, alpha=0.75, label=result.label, color=c)
        if result.label != "open_loop":
            axes[3].plot(t, result.correction_freq_hz, lw=0.8, alpha=0.9, label=result.label, color=c)
    axes[0].set_ylabel("wrapped phase (rad)")
    axes[1].set_ylabel("raw timing (ns)")
    axes[2].set_ylabel("residual freq (Hz)")
    axes[3].set_ylabel("NCO freq cmd (Hz)")
    axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("FLL vs PLL Replay on Corrected-Tone Pseudo-Open-Loop Phase")
    fig.savefig(out_dir / "fll_pll_replay_timeseries.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    labels = [r.label for r in results]
    timing_rms = [float(metrics[label]["timing_resid_rms_ns"]) for label in labels]
    p95_phase = [float(metrics[label]["wrapped_phase_p95_abs_rad"]) for label in labels]
    axes[0].bar(labels, timing_rms, color=[colors.get(x, "#777777") for x in labels])
    axes[0].set_ylabel("detrended timing RMS (ns)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(labels, p95_phase, color=[colors.get(x, "#777777") for x in labels])
    axes[1].set_ylabel("p95 |wrapped phase| (rad)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Primary Stability Metrics")
    fig.savefig(out_dir / "fll_pll_metric_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for result in results:
        timing_ns = result.residual_unwrapped_phase_rad / (TWO_PI * rf_hz) * 1.0e9
        peaks = top_psd_peaks(trace.t_s, timing_ns, 0.02, 20.0, count=120)
        if not peaks:
            continue
        ax.semilogy(
            [p["hz"] for p in peaks],
            [p["psd"] for p in peaks],
            marker=".",
            linestyle="none",
            label=result.label,
            color=colors.get(result.label, None),
            alpha=0.8,
        )
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("relative PSD peak power")
    ax.set_title("Residual Timing PSD Peaks, 0.02-20 Hz")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "fll_pll_psd_peak_comparison.png", dpi=160)
    plt.close(fig)


def plot_wrapped_phase_detail(
    trace: Trace,
    result: LoopResult,
    out_dir: Path,
    zoom_start_s: float,
    zoom_duration_s: float,
    prefix: str,
    title_label: str,
    color: str,
) -> None:
    t = trace.t_s - trace.t_s[0]
    phase = result.residual_wrapped_phase_rad

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(t, phase, color=color, lw=0.75)
    ax.axhline(0.0, color="#222222", lw=0.8, alpha=0.6)
    ax.axhline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.axhline(-1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.set_title(f"{title_label} Wrapped Phase, Full Record")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wrapped phase (rad)")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / f"{prefix}_wrapped_phase_full.png", dpi=160)
    plt.close(fig)

    start = float(zoom_start_s)
    stop = start + float(zoom_duration_s)
    mask = (t >= start) & (t <= stop) & np.isfinite(phase)
    if np.count_nonzero(mask) < 2:
        # Fall back to the beginning of the record if the requested zoom is not present.
        start = 0.0
        stop = float(zoom_duration_s)
        mask = (t >= start) & (t <= stop) & np.isfinite(phase)

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(t[mask] - start, phase[mask], marker=".", ms=2.5, color=color, lw=0.8)
    ax.axhline(0.0, color="#222222", lw=0.8, alpha=0.6)
    ax.axhline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.axhline(-1.0, color="#666666", lw=0.8, ls="--", alpha=0.45)
    ax.set_title(f"{title_label} Wrapped Phase, {zoom_duration_s:.6g} s Zoom Starting at {start:.6g} s")
    ax.set_xlabel("time within zoom (s)")
    ax.set_ylabel("wrapped phase (rad)")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / f"{prefix}_wrapped_phase_one_sixth_second.png", dpi=160)
    plt.close(fig)


def write_report(out_dir: Path, input_csv: Path, configs: list[LoopConfig], metrics: dict[str, dict[str, object]]) -> None:
    def fmt(label: str, key: str, suffix: str = "") -> str:
        value = metrics[label].get(key, float("nan"))
        if isinstance(value, float):
            return f"{value:.6g}{suffix}"
        return str(value)

    labels = list(metrics)
    lines = [
        "# FLL/PLL Design Evaluation",
        "",
        f"Input CSV: `{input_csv}`",
        "",
        "This replay treats the measured high-coherence phase trace as the input to a software NCO correction path.",
        "It evaluates architecture behavior only; it does not claim the slow DAC/VCXO path can realize the PLL bandwidth.",
        "",
        "## Controller Designs",
        "",
    ]
    for cfg in configs:
        lines.append(
            f"- `{cfg.label}`: {cfg.kind}, bandwidth={cfg.bandwidth_hz:g} Hz, "
            f"damping={cfg.damping:g}, max_freq={cfg.max_freq_hz:g} Hz"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| case | phase slope (Hz) | timing RMS (ns) | p95 |phase| (rad) | p95 |freq| (Hz) | phase-band fraction | best dwell (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in labels:
        lines.append(
            f"| `{label}` | {fmt(label, 'phase_slope_hz')} | "
            f"{fmt(label, 'timing_resid_rms_ns')} | {fmt(label, 'wrapped_phase_p95_abs_rad')} | "
            f"{fmt(label, 'p95_abs_residual_freq_hz')} | {fmt(label, 'phase_band_fraction')} | "
            f"{fmt(label, 'best_phase_band_dwell_s')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The FLL design is a slow frequency-centering loop. It reduces long-term slope and frequency error but intentionally does not force the instantaneous phase to zero.",
            "- The PLL design is a fast software phase loop. Its useful metric is wrapped phase/timing stability after correction, not just fitted frequency slope.",
            "- Coherence and amplitude are used only as measurement gates; the pseudo-open-loop input already has high mean coherence in the handoff.",
            "",
            "Generated plots:",
            "",
            "- `fll_pll_replay_timeseries.png`",
            "- `fll_pll_metric_comparison.png`",
            "- `fll_pll_psd_peak_comparison.png`",
            "- `fll_wrapped_phase_full.png`",
            "- `fll_wrapped_phase_one_sixth_second.png`",
            "- `pll_wrapped_phase_full.png`",
            "- `pll_wrapped_phase_one_sixth_second.png`",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--rf-hz", type=float, default=50.0e6)
    parser.add_argument("--amp-min", type=float, default=0.02)
    parser.add_argument("--coherence-min", type=float, default=0.2)
    parser.add_argument("--phase-band-rad", type=float, default=1.0)
    parser.add_argument("--zoom-start-s", type=float, default=260.0)
    parser.add_argument("--zoom-duration-s", type=float, default=1.0 / 6.0)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = load_trace(args.input_csv, max_rows=args.max_rows)

    fll_cfg = LoopConfig(
        label="fll_slow_frequency_centering",
        kind="first-order FLL from measured f_off_hz",
        bandwidth_hz=0.05,
        damping=1.0,
        max_freq_hz=250.0,
    )
    pll_cfg = LoopConfig(
        label="pll_fast_phase_lock",
        kind="type-II software PLL/NCO",
        bandwidth_hz=50.0,
        damping=0.707,
        max_freq_hz=250.0,
    )

    results = [
        evaluate_open_loop(trace, args.amp_min, args.coherence_min),
        run_fll(trace, fll_cfg, args.amp_min, args.coherence_min),
        run_pll(trace, pll_cfg, args.amp_min, args.coherence_min),
    ]
    metrics = {
        result.label: summarize(trace, result, rf_hz=float(args.rf_hz), phase_band_rad=float(args.phase_band_rad))
        for result in results
    }
    payload = {
        "input_csv": str(args.input_csv),
        "rf_hz": float(args.rf_hz),
        "amp_min": float(args.amp_min),
        "coherence_min": float(args.coherence_min),
        "phase_band_rad": float(args.phase_band_rad),
        "configs": [asdict(fll_cfg), asdict(pll_cfg)],
        "metrics": metrics,
    }
    (out_dir / "fll_pll_metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    with (out_dir / "fll_pll_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "label",
            "phase_slope_hz",
            "timing_resid_rms_ns",
            "timing_resid_p95_abs_ns",
            "wrapped_phase_p95_abs_rad",
            "p95_abs_residual_freq_hz",
            "phase_band_fraction",
            "best_phase_band_dwell_s",
            "mean_coherence",
            "p05_coherence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, row in metrics.items():
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    plot_results(trace, results, metrics, out_dir, rf_hz=float(args.rf_hz))
    plot_wrapped_phase_detail(
        trace,
        results[1],
        out_dir,
        float(args.zoom_start_s),
        float(args.zoom_duration_s),
        prefix="fll",
        title_label="FLL",
        color="#1f77b4",
    )
    plot_wrapped_phase_detail(
        trace,
        results[2],
        out_dir,
        float(args.zoom_start_s),
        float(args.zoom_duration_s),
        prefix="pll",
        title_label="PLL",
        color="#d62728",
    )
    write_report(out_dir, args.input_csv, [fll_cfg, pll_cfg], metrics)

    for label, row in metrics.items():
        print(
            f"{label}: slope={row['phase_slope_hz']:.6g} Hz, "
            f"timing_rms={row['timing_resid_rms_ns']:.6g} ns, "
            f"p95_phase={row['wrapped_phase_p95_abs_rad']:.6g} rad, "
            f"phase_band_fraction={row['phase_band_fraction']:.6g}"
        )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
