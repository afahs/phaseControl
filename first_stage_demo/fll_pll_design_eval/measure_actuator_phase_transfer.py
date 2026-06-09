#!/usr/bin/env python3
"""Measure DAC/VCXO actuator-to-phase transfer with sinusoidal command injection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIRST_STAGE_DIR = Path(__file__).resolve().parents[1]
if str(FIRST_STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(FIRST_STAGE_DIR))

import cw_vcxo_discipline as base


TWO_PI = 2.0 * math.pi


def parse_freqs(text: str) -> list[float]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0.0:
            raise ValueError("Sweep frequencies must be positive.")
        out.append(value)
    if not out:
        raise ValueError("At least one sweep frequency is required.")
    return out


def wrap_pm_pi(x: float) -> float:
    return float((x + math.pi) % TWO_PI - math.pi)


def fit_sinusoid(t: np.ndarray, y: np.ndarray, freq_hz: float) -> dict[str, float]:
    mask = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(mask) < 8:
        return {
            "amplitude": float("nan"),
            "phase_rad": float("nan"),
            "offset": float("nan"),
            "rms_residual": float("nan"),
            "r2": float("nan"),
        }
    tt = t[mask]
    yy = y[mask]
    w = TWO_PI * float(freq_hz)
    design = np.column_stack([np.cos(w * tt), np.sin(w * tt), np.ones_like(tt)])
    coeff, *_ = np.linalg.lstsq(design, yy, rcond=None)
    a, b, c = [float(v) for v in coeff]
    fit = design @ coeff
    resid = yy - fit
    amp = float(math.hypot(a, b))
    phase = float(math.atan2(-b, a))
    ss_res = float(np.sum(resid * resid))
    centered = yy - float(np.mean(yy))
    ss_tot = float(np.sum(centered * centered))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return {
        "amplitude": amp,
        "phase_rad": phase,
        "offset": c,
        "rms_residual": float(math.sqrt(ss_res / max(1, yy.size))),
        "r2": r2,
    }


def linear_detrend(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=np.float64).copy()
    mask = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return out - np.nanmean(out)
    t0 = float(t[mask][0])
    coeff = np.polyfit(t[mask] - t0, y[mask], 1)
    out[mask] = y[mask] - (coeff[0] * (t[mask] - t0) + coeff[1])
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", dest="sample_rate_hz", type=float, default=1_024_000.0)
    parser.add_argument("--center-freq", dest="center_freq_hz", type=float, required=True)
    parser.add_argument("--tone-rf", dest="tone_rf_hz", type=float, default=None)
    parser.add_argument("--tone-bb", "--expected-offset-hz", dest="tone_bb_hz", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--gain", default="0")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--ugradio-queue-blocks", type=int, default=128)
    parser.add_argument("--warmup-blocks", type=int, default=16)
    parser.add_argument("--sdr-backend", default="ugradio")

    parser.add_argument("--use-mcp4725", action="store_true")
    parser.add_argument("--cmd-min", "--dac-min", dest="command_min", type=float, default=0.0)
    parser.add_argument("--cmd-max", "--dac-max", dest="command_max", type=float, default=4095.0)
    parser.add_argument("--cmd-center", "--dac-center", dest="command_center", type=float, default=2050.0)
    parser.add_argument("--command-amplitude", "--dac-amplitude", dest="command_amplitude", type=float, default=3.0)
    parser.add_argument("--clock-apr-ppm", type=int, default=120)
    parser.add_argument("--skip-clock-init", action="store_true")
    parser.add_argument("--mcp4725-dither", action="store_true")
    parser.add_argument("--control-settle-s", type=float, default=0.0)

    parser.add_argument("--freqs-hz", default="0.05,0.1,0.2,0.5,1,2,5")
    parser.add_argument("--cycles", type=float, default=3.0)
    parser.add_argument("--min-dwell-s", type=float, default=6.0)
    parser.add_argument("--settle-cycles", type=float, default=0.5)
    parser.add_argument("--command-update-interval-s", type=float, default=0.02)
    parser.add_argument("--amp-threshold", type=float, default=0.02)
    parser.add_argument("--coherence-threshold", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "actuator_phase_transfer")
    parser.add_argument("--label", default="live_actuator_phase_transfer_20260520")
    return parser


def resolve_args() -> argparse.Namespace:
    args = build_parser().parse_args()
    args.mode = "closed_loop"
    args.control_architecture = "actuator_phase_transfer"
    args.use_pwm = False
    args.command_start = float(args.command_center)
    args.closed_loop_command_window = None
    args.mcp4725_dither = bool(args.mcp4725_dither)
    base.resolve_command_defaults(args)
    if not args.use_mcp4725:
        raise ValueError("This live transfer test currently expects --use-mcp4725.")
    if args.command_amplitude <= 0.0:
        raise ValueError("--command-amplitude must be positive.")
    if args.command_update_interval_s <= 0.0:
        raise ValueError("--command-update-interval-s must be positive.")
    return args


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.label}.csv"
    json_path = args.out_dir / f"{args.label}_transfer.json"
    summary_path = args.out_dir / f"{args.label}_summary.csv"
    return csv_path, json_path, summary_path


def run_live(args: argparse.Namespace, csv_path: Path) -> None:
    freqs = parse_freqs(args.freqs_hz)
    tone_ref = base.resolve_tone_reference(args)
    sdr = base.choose_sdr_backend(args)
    actuator = base.build_actuator(args)
    detector = base.TonePhaseDetector(args.sample_rate_hz, args.block_size, tone_ref.tone_bb_hz)
    stop = base.GracefulStop()
    stop.install()
    block_dt_s = float(args.block_size) / float(args.sample_rate_hz)
    update_blocks = max(1, int(round(float(args.command_update_interval_s) / block_dt_s)))
    fieldnames = [
        "wall_rel_s",
        "t_rel_s",
        "block_index",
        "sweep_index",
        "sweep_freq_hz",
        "sweep_local_t_s",
        "command_requested",
        "command_applied",
        "phase_rad",
        "unwrapped_phase_rad",
        "dphi_rad",
        "f_off_hz",
        "amp",
        "coherence",
        "measurement_valid",
        "write_elapsed_s",
    ]
    try:
        actuator.open()
        applied = actuator.write_command(float(args.command_center), settle_s=float(args.control_settle_s))
        print(f"Actuator opened at command {applied:.3f}")
        sdr.warmup(args.block_size, args.warmup_blocks)
        wall_start = time.monotonic()
        with csv_path.open("w", newline="", encoding="utf-8", buffering=1024 * 1024) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            block_index = 0
            for sweep_index, freq_hz in enumerate(freqs):
                dwell_s = max(float(args.min_dwell_s), float(args.cycles) / float(freq_hz))
                blocks = int(math.ceil(dwell_s / block_dt_s))
                print(
                    f"sweep {sweep_index}: f={freq_hz:g} Hz, dwell={blocks * block_dt_s:.3f}s, "
                    f"amp={args.command_amplitude:g} codes"
                )
                for local_block in range(blocks):
                    if stop.stop_requested:
                        break
                    local_t = (local_block + 0.5) * block_dt_s
                    requested = float(args.command_center) + float(args.command_amplitude) * math.sin(
                        TWO_PI * float(freq_hz) * local_t
                    )
                    requested = base.clamp(requested, args.command_min, args.command_max)
                    write_elapsed = 0.0
                    if local_block % update_blocks == 0:
                        t0 = time.monotonic()
                        applied = actuator.write_command(requested, settle_s=float(args.control_settle_s))
                        write_elapsed = time.monotonic() - t0
                    iq = sdr.read_block(args.block_size)
                    wall_now = time.monotonic()
                    meas = detector.process_block(iq)
                    valid = bool(
                        np.isfinite(meas.phase_rad)
                        and meas.amp >= float(args.amp_threshold)
                        and meas.coherence >= float(args.coherence_threshold)
                    )
                    writer.writerow(
                        {
                            "wall_rel_s": wall_now - wall_start,
                            "t_rel_s": (block_index + 0.5) * block_dt_s,
                            "block_index": block_index,
                            "sweep_index": sweep_index,
                            "sweep_freq_hz": freq_hz,
                            "sweep_local_t_s": local_t,
                            "command_requested": requested,
                            "command_applied": applied,
                            "phase_rad": meas.phase_rad,
                            "unwrapped_phase_rad": meas.unwrapped_phase_rad,
                            "dphi_rad": meas.dphi_rad,
                            "f_off_hz": meas.f_off_hz,
                            "amp": meas.amp,
                            "coherence": meas.coherence,
                            "measurement_valid": 1 if valid else 0,
                            "write_elapsed_s": write_elapsed,
                        }
                    )
                    block_index += 1
                f.flush()
                if stop.stop_requested:
                    break
    finally:
        try:
            actuator.write_command(float(args.command_center), settle_s=0.0)
        except Exception:
            pass
        actuator.close()
        stop.restore()


def load_rows(csv_path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                cols.setdefault(key, []).append(float(value) if value not in (None, "") else float("nan"))
    return {key: np.asarray(values, dtype=np.float64) for key, values in cols.items()}


def analyze(csv_path: Path, json_path: Path, summary_path: Path, args: argparse.Namespace) -> dict[str, object]:
    data = load_rows(csv_path)
    results = []
    for sweep_index in sorted(set(int(x) for x in data["sweep_index"][np.isfinite(data["sweep_index"])])):
        mask = data["sweep_index"] == sweep_index
        if np.count_nonzero(mask) < 16:
            continue
        freq_hz = float(np.nanmedian(data["sweep_freq_hz"][mask]))
        local_t = data["sweep_local_t_s"][mask]
        settle_s = float(args.settle_cycles) / freq_hz
        fit_mask = mask.copy()
        fit_mask[mask] = local_t >= settle_s
        valid = fit_mask & (data["measurement_valid"] > 0.5)
        t = data["sweep_local_t_s"][valid]
        cmd = data["command_applied"][valid]
        phase = data["unwrapped_phase_rad"][valid]
        freq = data["f_off_hz"][valid]
        phase_dt = linear_detrend(t, phase)
        freq_centered = freq - np.nanmean(freq)
        cmd_fit = fit_sinusoid(t, cmd, freq_hz)
        phase_fit = fit_sinusoid(t, phase_dt, freq_hz)
        freq_fit = fit_sinusoid(t, freq_centered, freq_hz)
        phase_lag = wrap_pm_pi(phase_fit["phase_rad"] - cmd_fit["phase_rad"])
        freq_lag = wrap_pm_pi(freq_fit["phase_rad"] - cmd_fit["phase_rad"])
        # For this setup increasing DAC command lowers measured baseband frequency.
        # Therefore command->frequency has an ideal 180 deg sign inversion, and
        # command->phase has that inversion plus one integration, giving +90 deg.
        phase_extra_lag = wrap_pm_pi(math.radians(90.0) - phase_lag)
        freq_extra_lag = wrap_pm_pi(math.radians(180.0) - freq_lag)
        cmd_amp = cmd_fit["amplitude"]
        row = {
            "sweep_index": sweep_index,
            "freq_hz": freq_hz,
            "fit_rows": int(np.count_nonzero(valid)),
            "command_amp_codes": cmd_amp,
            "phase_amp_rad": phase_fit["amplitude"],
            "phase_gain_rad_per_code": phase_fit["amplitude"] / cmd_amp if cmd_amp > 0.0 else float("nan"),
            "phase_lag_deg": math.degrees(phase_lag),
            "phase_extra_lag_from_ideal_deg": math.degrees(phase_extra_lag),
            "phase_extra_delay_from_ideal_s": phase_extra_lag / (TWO_PI * freq_hz),
            "phase_fit_r2": phase_fit["r2"],
            "freq_amp_hz": freq_fit["amplitude"],
            "freq_gain_hz_per_code": freq_fit["amplitude"] / cmd_amp if cmd_amp > 0.0 else float("nan"),
            "freq_lag_deg": math.degrees(freq_lag),
            "freq_extra_lag_from_ideal_deg": math.degrees(freq_extra_lag),
            "freq_extra_delay_from_ideal_s": freq_extra_lag / (TWO_PI * freq_hz),
            "freq_fit_r2": freq_fit["r2"],
            "mean_coherence": float(np.nanmean(data["coherence"][valid])) if np.any(valid) else float("nan"),
            "p05_coherence": float(np.nanpercentile(data["coherence"][valid], 5.0)) if np.any(valid) else float("nan"),
        }
        results.append(row)

    payload = {
        "csv_path": str(csv_path),
        "results": results,
        "config": {
            "freqs_hz": parse_freqs(args.freqs_hz),
            "command_center": float(args.command_center),
            "command_amplitude": float(args.command_amplitude),
            "command_update_interval_s": float(args.command_update_interval_s),
            "settle_cycles": float(args.settle_cycles),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys()) if results else ["freq_hz"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    return payload


def plot_results(csv_path: Path, payload: dict[str, object], args: argparse.Namespace) -> None:
    out_dir = csv_path.parent
    results = payload["results"]  # type: ignore[assignment]
    if not results:
        return
    freq = np.asarray([float(r["freq_hz"]) for r in results], dtype=np.float64)  # type: ignore[index]
    phase_gain = np.asarray([float(r["phase_gain_rad_per_code"]) for r in results], dtype=np.float64)  # type: ignore[index]
    phase_lag = np.asarray([float(r["phase_lag_deg"]) for r in results], dtype=np.float64)  # type: ignore[index]
    phase_r2 = np.asarray([float(r["phase_fit_r2"]) for r in results], dtype=np.float64)  # type: ignore[index]
    freq_gain = np.asarray([float(r["freq_gain_hz_per_code"]) for r in results], dtype=np.float64)  # type: ignore[index]
    freq_lag = np.asarray([float(r["freq_lag_deg"]) for r in results], dtype=np.float64)  # type: ignore[index]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].loglog(freq, phase_gain, marker="o")
    axes[0, 0].set_ylabel("phase gain (rad/code)")
    axes[0, 1].semilogx(freq, phase_lag, marker="o")
    axes[0, 1].set_ylabel("phase lag (deg)")
    axes[1, 0].loglog(freq, freq_gain, marker="o", color="#d62728")
    axes[1, 0].set_ylabel("frequency gain (Hz/code)")
    axes[1, 1].semilogx(freq, freq_lag, marker="o", color="#d62728")
    axes[1, 1].set_ylabel("frequency lag (deg)")
    for ax in axes.ravel():
        ax.set_xlabel("command sine frequency (Hz)")
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle("Measured Actuator-to-Phase/Frequency Transfer")
    fig.savefig(out_dir / f"{args.label}_bode.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.semilogx(freq, phase_r2, marker="o")
    ax.set_xlabel("command sine frequency (Hz)")
    ax.set_ylabel("phase sine-fit R^2")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("Phase Response Fit Quality")
    fig.savefig(out_dir / f"{args.label}_fit_quality.png", dpi=160)
    plt.close(fig)

    data = load_rows(csv_path)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(data["wall_rel_s"], data["command_applied"], lw=0.8)
    axes[1].plot(data["wall_rel_s"], data["unwrapped_phase_rad"], lw=0.7)
    axes[2].plot(data["wall_rel_s"], data["f_off_hz"], lw=0.7)
    axes[0].set_ylabel("DAC command")
    axes[1].set_ylabel("unwrapped phase (rad)")
    axes[2].set_ylabel("f_off (Hz)")
    axes[2].set_xlabel("wall time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Actuator Sine Sweep Time Series")
    fig.savefig(out_dir / f"{args.label}_timeseries.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = resolve_args()
    csv_path, json_path, summary_path = output_paths(args)
    if not csv_path.exists():
        run_live(args, csv_path)
    else:
        print(f"Using existing CSV: {csv_path}")
    payload = analyze(csv_path, json_path, summary_path, args)
    plot_results(csv_path, payload, args)
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
