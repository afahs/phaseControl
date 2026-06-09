#!/usr/bin/env python3
"""Compare measured phase disturbance spectrum against actuator bandwidth limits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TWO_PI = 2.0 * math.pi
DEFAULT_TRACE = (
    Path(__file__).resolve().parents[1]
    / "logs"
    / "pseudo_open_loop_after_30s_freq_centering_detected_tone_50mhz_5min_20260514.csv"
)
DEFAULT_LATENCY_SUMMARY = (
    Path(__file__).resolve().parents[1]
    / "logs"
    / "actuator_latency_50mhz_step5_20260427.txt"
)


def parse_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(str(value).strip())
    except ValueError:
        return float("nan")


def wrap_pm_pi(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + math.pi) % TWO_PI - math.pi


def load_phase_trace(path: Path, amp_min: float, coherence_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t: list[float] = []
    unwrapped: list[float] = []
    wrapped: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ti = parse_float(row.get("t_rel_s"))
            ph = parse_float(row.get("unwrapped_phase_rad"))
            ph_wrapped = parse_float(row.get("phase_rad"))
            amp = parse_float(row.get("amp"))
            coh = parse_float(row.get("coherence"))
            if not (
                np.isfinite(ti)
                and np.isfinite(ph)
                and np.isfinite(amp)
                and np.isfinite(coh)
                and amp >= amp_min
                and coh >= coherence_min
            ):
                continue
            t.append(ti)
            unwrapped.append(ph)
            wrapped.append(ph_wrapped if np.isfinite(ph_wrapped) else float(wrap_pm_pi(ph)))
    if len(t) < 16:
        raise RuntimeError(f"Not enough valid phase samples in {path}")
    return (
        np.asarray(t, dtype=np.float64),
        np.asarray(unwrapped, dtype=np.float64),
        np.asarray(wrapped, dtype=np.float64),
    )


def median_dt(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        raise RuntimeError("No positive time intervals.")
    return float(np.median(dt))


def detrend_linear(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(t) & np.isfinite(x)
    if np.count_nonzero(mask) < 3:
        return x - np.nanmean(x)
    t0 = float(t[mask][0])
    coeff = np.polyfit(t[mask] - t0, x[mask], 1)
    return x - (coeff[0] * (t - t0) + coeff[1])


def welch_psd(x: np.ndarray, dt_s: float, segment_len: int = 16384, overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 32:
        raise RuntimeError("Not enough samples for PSD.")
    segment_len = min(int(segment_len), x.size)
    segment_len = max(32, segment_len)
    step = max(1, int(segment_len * (1.0 - overlap)))
    window = np.hanning(segment_len)
    win_power = float(np.sum(window * window))
    scale = 1.0 / (1.0 / dt_s * win_power)
    spectra = []
    for start in range(0, x.size - segment_len + 1, step):
        seg = x[start : start + segment_len]
        seg = seg - float(np.mean(seg))
        spec = np.fft.rfft(seg * window)
        psd = (np.abs(spec) ** 2) * scale
        if psd.size > 2:
            psd[1:-1] *= 2.0
        spectra.append(psd)
    if not spectra:
        seg = x - float(np.mean(x))
        window = np.hanning(seg.size)
        win_power = float(np.sum(window * window))
        spec = np.fft.rfft(seg * window)
        psd = (np.abs(spec) ** 2) / (1.0 / dt_s * win_power)
        if psd.size > 2:
            psd[1:-1] *= 2.0
        spectra = [psd]
        segment_len = seg.size
    freqs = np.fft.rfftfreq(segment_len, dt_s)
    return freqs, np.mean(np.vstack(spectra), axis=0)


def cumulative_fraction(freqs: np.ndarray, psd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(freqs) & np.isfinite(psd) & (freqs >= 0.0)
    f = freqs[valid]
    p = psd[valid]
    if f.size < 2:
        return f, np.full_like(f, np.nan)
    df = np.diff(f, prepend=f[0])
    power = np.maximum(p, 0.0) * df
    total = float(np.sum(power[1:]))
    if total <= 0.0:
        return f, np.full_like(f, np.nan)
    return f, np.cumsum(power) / total


def fraction_below(freqs: np.ndarray, psd: np.ndarray, bandwidth_hz: float) -> float:
    f, c = cumulative_fraction(freqs, psd)
    good = np.isfinite(f) & np.isfinite(c)
    if np.count_nonzero(good) == 0:
        return float("nan")
    return float(np.interp(float(bandwidth_hz), f[good], c[good], left=0.0, right=1.0))


def parse_latency_summary(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            out[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return out


def delay_bandwidths(delay_s: float) -> dict[str, float]:
    return {
        "45deg_phase_lag_hz": 1.0 / (8.0 * delay_s),
        "90deg_phase_lag_hz": 1.0 / (4.0 * delay_s),
        "180deg_phase_lag_hz": 1.0 / (2.0 * delay_s),
        "360deg_phase_lag_hz": 1.0 / delay_s,
    }


def write_report(out_dir: Path, payload: dict[str, object]) -> None:
    rows = payload["candidate_bandwidths"]  # type: ignore[assignment]
    lines = [
        "# Correctable Spectrum Comparison",
        "",
        f"Input trace: `{payload['input_trace']}`",
        "",
        "The disturbance PSD is computed from the high-coherence corrected-tone pseudo-open-loop phase trace.",
        "The actuator limits are represented as delay-limited feedback bandwidths; this does not replace a full sine-sweep transfer function, but it tests the structural claim directly against the measured disturbance spectrum.",
        "",
        "## Candidate Bandwidth Fractions",
        "",
        "| limit | bandwidth (Hz) | detrended unwrapped phase power below | wrapped phase power below |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:  # type: ignore[union-attr]
        lines.append(
            f"| {row['label']} | {row['bandwidth_hz']:.4g} | "
            f"{100.0 * row['detrended_unwrapped_fraction_below']:.2f}% | "
            f"{100.0 * row['wrapped_fraction_below']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Generated Plots",
            "",
            "- `disturbance_psd_vs_actuator_limits.png`",
            "- `cumulative_phase_power_vs_bandwidth.png`",
            "- `delay_phase_lag_limits.png`",
            "",
            "Interpretation: if most disturbance power is above the usable delay-limited bandwidth, a DAC/VCXO feedback loop cannot remove it as a real-time phase actuator even when the phase detector can measure it. A software NCO can still correct higher-frequency phase after sampling because it has a much faster digital actuation path.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(
    out_dir: Path,
    freqs_unwrapped: np.ndarray,
    psd_unwrapped: np.ndarray,
    freqs_wrapped: np.ndarray,
    psd_wrapped: np.ndarray,
    candidate_rows: list[dict[str, float | str]],
    delays_s: list[float],
) -> None:
    limit_rows = [row for row in candidate_rows if "45deg" in str(row["label"]) or "90deg" in str(row["label"])]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    mask_u = freqs_unwrapped > 0.0
    mask_w = freqs_wrapped > 0.0
    ax.loglog(freqs_unwrapped[mask_u], psd_unwrapped[mask_u], lw=1.0, label="detrended unwrapped phase PSD")
    ax.loglog(freqs_wrapped[mask_w], psd_wrapped[mask_w], lw=0.9, alpha=0.8, label="wrapped phase PSD")
    for row in limit_rows:
        label = str(row["label"])
        bw = float(row["bandwidth_hz"])
        if bw <= 0.0:
            continue
        ax.axvline(bw, lw=0.9, ls="--", alpha=0.65)
        ax.text(bw, ax.get_ylim()[1] * 0.5, label, rotation=90, va="center", ha="right", fontsize=8)
    ax.set_xlim(0.01, min(250.0, float(np.nanmax(freqs_unwrapped))))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("phase PSD (rad^2/Hz)")
    ax.set_title("Measured Phase Disturbance Spectrum vs Delay-Limited Actuator Bandwidth")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_dir / "disturbance_psd_vs_actuator_limits.png", dpi=160)
    plt.close(fig)

    f_u, c_u = cumulative_fraction(freqs_unwrapped, psd_unwrapped)
    f_w, c_w = cumulative_fraction(freqs_wrapped, psd_wrapped)
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.semilogx(f_u[1:], 100.0 * c_u[1:], lw=1.4, label="detrended unwrapped phase")
    ax.semilogx(f_w[1:], 100.0 * c_w[1:], lw=1.2, label="wrapped phase")
    for row in limit_rows:
        bw = float(row["bandwidth_hz"])
        ax.axvline(bw, lw=0.9, ls="--", alpha=0.65)
    ax.set_xlim(0.01, min(250.0, float(np.nanmax(freqs_unwrapped))))
    ax.set_ylim(0.0, 100.5)
    ax.set_xlabel("candidate closed-loop bandwidth (Hz)")
    ax.set_ylabel("phase variance below bandwidth (%)")
    ax.set_title("How Much Measured Phase Power Is Below a Correctable Bandwidth?")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_dir / "cumulative_phase_power_vs_bandwidth.png", dpi=160)
    plt.close(fig)

    f = np.logspace(-2, math.log10(50.0), 1000)
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for delay_s in delays_s:
        lag_deg = 360.0 * f * delay_s
        ax.semilogx(f, lag_deg, lw=1.2, label=f"delay={1000.0 * delay_s:.0f} ms")
    for deg in (45.0, 90.0, 180.0, 360.0):
        ax.axhline(deg, color="#666666", lw=0.8, ls="--", alpha=0.45)
        ax.text(f[-1], deg, f" {deg:.0f} deg", va="center", fontsize=8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("pure-delay phase lag (degrees)")
    ax.set_title("Actuator Delay Phase Lag")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left")
    fig.savefig(out_dir / "delay_phase_lag_limits.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--latency-summary", type=Path, default=DEFAULT_LATENCY_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "spectrum_correctability")
    parser.add_argument("--amp-min", type=float, default=0.02)
    parser.add_argument("--coherence-min", type=float, default=0.2)
    parser.add_argument("--segment-len", type=int, default=16384)
    parser.add_argument("--delay-ms", type=float, default=150.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t, unwrapped, wrapped = load_phase_trace(args.input_trace, args.amp_min, args.coherence_min)
    dt = median_dt(t)
    detrended = detrend_linear(t, unwrapped)
    wrapped_centered = np.asarray(wrap_pm_pi(wrapped), dtype=np.float64)
    freqs_u, psd_u = welch_psd(detrended, dt, segment_len=int(args.segment_len))
    freqs_w, psd_w = welch_psd(wrapped_centered, dt, segment_len=int(args.segment_len))

    latency = parse_latency_summary(args.latency_summary)
    delays_s = [
        float(latency.get("latency_median_s", 0.03196)),
        0.07585,
        float(args.delay_ms) / 1000.0,
    ]
    labels = [
        "measured median latency",
        "measured slow latency",
        f"assumed {args.delay_ms:g} ms effective latency",
    ]
    candidate_rows: list[dict[str, float | str]] = []
    for label, delay_s in zip(labels, delays_s):
        for lag_label, bw in delay_bandwidths(delay_s).items():
            candidate_rows.append(
                {
                    "label": f"{label}, {lag_label.replace('_', ' ')}",
                    "delay_s": float(delay_s),
                    "bandwidth_hz": float(bw),
                    "detrended_unwrapped_fraction_below": fraction_below(freqs_u, psd_u, bw),
                    "wrapped_fraction_below": fraction_below(freqs_w, psd_w, bw),
                }
            )
    for label, bw in (
        ("50 ms update Nyquist", 10.0),
        ("2 ms software-NCO Nyquist", 250.0),
    ):
        candidate_rows.append(
            {
                "label": label,
                "delay_s": float("nan"),
                "bandwidth_hz": float(bw),
                "detrended_unwrapped_fraction_below": fraction_below(freqs_u, psd_u, bw),
                "wrapped_fraction_below": fraction_below(freqs_w, psd_w, bw),
            }
        )

    payload: dict[str, object] = {
        "input_trace": str(args.input_trace),
        "latency_summary": str(args.latency_summary),
        "rows": int(t.size),
        "duration_s": float(t[-1] - t[0]),
        "dt_s": float(dt),
        "sample_rate_hz": float(1.0 / dt),
        "candidate_bandwidths": candidate_rows,
        "plots": [
            str(args.out_dir / "disturbance_psd_vs_actuator_limits.png"),
            str(args.out_dir / "cumulative_phase_power_vs_bandwidth.png"),
            str(args.out_dir / "delay_phase_lag_limits.png"),
        ],
    }
    (args.out_dir / "correctable_spectrum_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    plot_outputs(args.out_dir, freqs_u, psd_u, freqs_w, psd_w, candidate_rows, delays_s)
    write_report(args.out_dir, payload)
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
