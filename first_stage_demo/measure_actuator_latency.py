#!/usr/bin/env python3
"""Measure VCXO actuator latency from DAC steps and SDR phase/frequency response."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import cw_vcxo_discipline as base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", dest="sample_rate_hz", type=float, default=1_024_000.0)
    parser.add_argument("--center-freq", dest="center_freq_hz", type=float, required=True)
    parser.add_argument("--tone-rf", dest="tone_rf_hz", type=float, default=None)
    parser.add_argument("--tone-bb", "--expected-offset-hz", dest="tone_bb_hz", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--gain", default="0")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--ugradio-queue-blocks", type=int, default=128)
    parser.add_argument("--warmup-blocks", type=int, default=16)
    parser.add_argument("--sdr-backend", default="ugradio")

    parser.add_argument("--use-mcp4725", action="store_true")
    parser.add_argument("--use-pwm", action="store_true")
    parser.add_argument("--cmd-min", "--dac-min", dest="command_min", type=float, default=None)
    parser.add_argument("--cmd-max", "--dac-max", dest="command_max", type=float, default=None)
    parser.add_argument("--cmd-start", "--dac-start", dest="command_start", type=float, default=2050.0)
    parser.add_argument("--clock-apr-ppm", type=int, default=120)
    parser.add_argument("--skip-clock-init", action="store_true")
    parser.add_argument("--mcp4725-dither", action="store_true")
    parser.add_argument("--pwm-pin", type=int, default=18)
    parser.add_argument("--pwm-frequency-hz", type=float, default=2000.0)
    parser.add_argument("--pwm-vdd", type=float, default=3.3)
    parser.add_argument("--control-settle-s", type=float, default=0.0)
    parser.add_argument("--vcxo-hz-per-command", type=float, default=-4.5475)

    parser.add_argument("--step-delta", type=float, default=5.0)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--pre-step-s", type=float, default=8.0)
    parser.add_argument("--post-step-s", type=float, default=24.0)
    parser.add_argument("--analysis-smooth-s", type=float, default=0.08)
    parser.add_argument("--analysis-pre-margin-s", type=float, default=1.0)
    parser.add_argument("--analysis-final-s", type=float, default=4.0)
    parser.add_argument("--response-fraction", type=float, default=0.2)
    parser.add_argument("--amp-threshold", type=float, default=0.02)
    parser.add_argument("--coherence-threshold", type=float, default=0.2)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-summary", default=None)
    return parser


def resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.mode = "closed_loop"
    args.control_architecture = "actuator_latency"
    args.invert_control = False
    args.closed_loop_command_window = None
    args.pull_in_kp = 0.0
    base.resolve_command_defaults(args)
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")
    if args.cycles <= 0:
        raise ValueError("--cycles must be positive.")
    if args.pre_step_s <= 0.0 or args.post_step_s <= 0.0:
        raise ValueError("--pre-step-s and --post-step-s must be positive.")
    return args


def output_path(args: argparse.Namespace, suffix: str) -> str:
    configured = args.out_csv if suffix == ".csv" else args.out_summary
    if configured:
        path = configured
    else:
        path = os.path.join(
            "first_stage_demo",
            "logs",
            f"{base.timestamp_for_filename()}_actuator_latency{suffix}",
        )
    base.ensure_parent_dir(path)
    return os.path.abspath(path)


def smooth_boxcar(x: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples <= 1:
        return x.astype(float, copy=True)
    kernel = np.ones(samples, dtype=float) / float(samples)
    finite = np.isfinite(x)
    values = np.where(finite, x, 0.0)
    weights = np.convolve(finite.astype(float), kernel, mode="same")
    smoothed = np.convolve(values, kernel, mode="same")
    out = np.full_like(x, np.nan, dtype=float)
    good = weights > 0.5
    out[good] = smoothed[good] / weights[good]
    return out


def estimate_step_latencies(
    rows: list[dict[str, float]],
    events: list[dict[str, float]],
    block_dt_s: float,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    t = np.array([row["wall_rel_s"] for row in rows], dtype=float)
    f = np.array([row["f_off_hz"] for row in rows], dtype=float)
    valid = np.array([row["measurement_valid"] for row in rows], dtype=bool)
    f = np.where(valid, f, np.nan)
    smooth_blocks = max(1, int(round(float(args.analysis_smooth_s) / block_dt_s)))
    fs = smooth_boxcar(f, smooth_blocks)
    estimates: list[dict[str, float]] = []
    for event in events:
        t0 = float(event["step_wall_rel_s"])
        direction = float(event["to_command"] - event["from_command"])
        pre_mask = (
            (t >= t0 - float(args.pre_step_s) + float(args.analysis_pre_margin_s))
            & (t < t0 - float(args.analysis_pre_margin_s))
            & np.isfinite(fs)
        )
        final_mask = (
            (t >= t0 + float(args.post_step_s) - float(args.analysis_final_s))
            & (t < t0 + float(args.post_step_s))
            & np.isfinite(fs)
        )
        pre = float(np.median(fs[pre_mask])) if np.any(pre_mask) else float("nan")
        final = float(np.median(fs[final_mask])) if np.any(final_mask) else float("nan")
        observed_delta = final - pre
        threshold = pre + float(args.response_fraction) * observed_delta
        after_mask = (t >= t0) & (t < t0 + float(args.post_step_s)) & np.isfinite(fs)
        after_idx = np.flatnonzero(after_mask)
        latency = float("nan")
        if np.isfinite(observed_delta) and abs(observed_delta) > 1.0e-9 and after_idx.size:
            if observed_delta >= 0.0:
                crossed = after_idx[fs[after_idx] >= threshold]
            else:
                crossed = after_idx[fs[after_idx] <= threshold]
            if crossed.size:
                latency = float(t[int(crossed[0])] - t0)
        estimates.append(
            {
                **event,
                "direction_codes": direction,
                "pre_f_off_hz": pre,
                "final_f_off_hz": final,
                "observed_delta_hz": observed_delta,
                "expected_delta_hz": direction * float(args.vcxo_hz_per_command),
                "response_fraction": float(args.response_fraction),
                "latency_s": latency,
            }
        )
    return estimates


def write_plot(csv_path: str, summary: list[dict[str, float]]) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    t: list[float] = []
    f: list[float] = []
    cmd: list[float] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                t.append(float(row["wall_rel_s"]))
                f.append(float(row["f_off_hz"]))
                cmd.append(float(row["command"]))
            except Exception:
                pass
    if not t:
        return None
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    axes[0].plot(t, f, lw=0.7)
    axes[0].set_ylabel("f_off_hz")
    axes[1].plot(t, cmd, lw=0.9)
    axes[1].set_ylabel("DAC code")
    axes[1].set_xlabel("wall_rel_s")
    for est in summary:
        t0 = est["step_wall_rel_s"]
        for ax in axes:
            ax.axvline(t0, color="k", alpha=0.35, lw=0.8)
            if np.isfinite(est["latency_s"]):
                ax.axvline(t0 + est["latency_s"], color="tab:red", alpha=0.65, lw=0.8)
    fig.tight_layout()
    png_path = str(Path(csv_path).with_suffix(".png"))
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return png_path


def run(args: argparse.Namespace) -> tuple[str, str]:
    tone_ref = base.resolve_tone_reference(args)
    csv_path = output_path(args, ".csv")
    summary_path = output_path(args, ".txt")
    sdr = base.choose_sdr_backend(args)
    actuator = base.build_actuator(args)
    detector = base.TonePhaseDetector(
        sample_rate_hz=args.sample_rate_hz,
        block_size=args.block_size,
        tone_bb_hz=tone_ref.tone_bb_hz,
    )
    stop = base.GracefulStop()
    stop.install()

    block_dt_s = float(args.block_size) / float(args.sample_rate_hz)
    commands = [float(args.command_start), float(args.command_start) + float(args.step_delta)]
    fieldnames = [
        "wall_rel_s",
        "t_rel_s",
        "block_index",
        "command",
        "step_index",
        "step_state",
        "phase_rad",
        "unwrapped_phase_rad",
        "dphi_rad",
        "f_off_hz",
        "amp",
        "coherence",
        "snr_db",
        "power",
        "measurement_valid",
        "write_elapsed_s",
    ]
    rows_for_analysis: list[dict[str, float]] = []
    events: list[dict[str, float]] = []

    try:
        actuator.open()
        applied = actuator.write_command(commands[0], settle_s=args.control_settle_s)
        sdr.warmup(args.block_size, args.warmup_blocks)

        total_s = float(args.pre_step_s) + float(args.cycles) * (
            float(args.pre_step_s) + float(args.post_step_s)
        )
        total_blocks = int(math.ceil(total_s / block_dt_s))
        first_step_block = int(round(float(args.pre_step_s) / block_dt_s))
        cycle_blocks = int(round((float(args.pre_step_s) + float(args.post_step_s)) / block_dt_s))
        post_blocks = int(round(float(args.post_step_s) / block_dt_s))
        active_command = float(applied)
        write_elapsed_s = 0.0
        wall_start = time.monotonic()
        next_status = wall_start
        step_index = -1
        step_end_block = -1

        with open(csv_path, "w", newline="", encoding="utf-8", buffering=1024 * 1024) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            print(f"Actuator latency CSV : {csv_path}")
            print(f"Summary              : {summary_path}")
            print(f"Tone BB reference    : {tone_ref.tone_bb_hz:.3f} Hz")
            print(f"Block duration       : {block_dt_s:.6f} s")
            print(f"Commands             : {commands[0]:.3f} <-> {commands[1]:.3f}")
            for block_index in range(total_blocks):
                if stop.stop_requested:
                    break

                if block_index >= first_step_block and (
                    (block_index - first_step_block) % cycle_blocks == 0
                ):
                    from_command = active_command
                    to_command = commands[(len(events) + 1) % 2]
                    write_start = time.monotonic()
                    active_command = actuator.write_command(to_command, settle_s=args.control_settle_s)
                    write_elapsed_s = time.monotonic() - write_start
                    step_index += 1
                    step_end_block = block_index + post_blocks
                    events.append(
                        {
                            "step_index": float(step_index),
                            "from_command": float(from_command),
                            "to_command": float(active_command),
                            "step_wall_rel_s": float(time.monotonic() - wall_start),
                            "write_elapsed_s": float(write_elapsed_s),
                        }
                    )

                iq = sdr.read_block(args.block_size)
                wall_now = time.monotonic()
                wall_rel_s = wall_now - wall_start
                t_rel_s = (block_index + 0.5) * block_dt_s
                meas = detector.process_block(iq)
                valid = bool(
                    meas.valid_phase_step
                    and meas.amp >= float(args.amp_threshold)
                    and meas.coherence >= float(args.coherence_threshold)
                    and np.isfinite(meas.f_off_hz)
                )
                step_state = "post_step" if block_index < step_end_block else "pre_step"
                row = {
                    "wall_rel_s": wall_rel_s,
                    "t_rel_s": t_rel_s,
                    "block_index": block_index,
                    "command": active_command,
                    "step_index": step_index,
                    "step_state": step_state,
                    "phase_rad": meas.phase_rad,
                    "unwrapped_phase_rad": meas.unwrapped_phase_rad,
                    "dphi_rad": meas.dphi_rad,
                    "f_off_hz": meas.f_off_hz,
                    "amp": meas.amp,
                    "coherence": meas.coherence,
                    "snr_db": meas.snr_db,
                    "power": meas.power,
                    "measurement_valid": int(valid),
                    "write_elapsed_s": write_elapsed_s,
                }
                writer.writerow(row)
                rows_for_analysis.append(
                    {
                        "wall_rel_s": float(wall_rel_s),
                        "f_off_hz": float(meas.f_off_hz),
                        "measurement_valid": float(valid),
                    }
                )
                write_elapsed_s = 0.0
                if wall_now >= next_status:
                    print(
                        f"t={wall_rel_s:7.2f}s cmd={active_command:8.3f} "
                        f"f_off={base.format_float(meas.f_off_hz, 9, 3)} Hz "
                        f"amp={meas.amp:.4f} coh={meas.coherence:.3f}",
                        flush=True,
                    )
                    next_status = wall_now + 10.0
    finally:
        try:
            actuator.close()
        finally:
            sdr.close()

    estimates = estimate_step_latencies(rows_for_analysis, events, block_dt_s, args)
    finite_latencies = [e["latency_s"] for e in estimates if np.isfinite(e["latency_s"])]
    plot_path = write_plot(csv_path, estimates)
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"csv_path={csv_path}\n")
        if plot_path:
            fh.write(f"plot_path={plot_path}\n")
        fh.write(f"block_dt_s={block_dt_s:.9f}\n")
        fh.write(f"smooth_s={float(args.analysis_smooth_s):.6f}\n")
        fh.write(f"response_fraction={float(args.response_fraction):.3f}\n")
        if finite_latencies:
            fh.write(f"latency_median_s={float(np.median(finite_latencies)):.6f}\n")
            fh.write(f"latency_min_s={float(np.min(finite_latencies)):.6f}\n")
            fh.write(f"latency_max_s={float(np.max(finite_latencies)):.6f}\n")
        else:
            fh.write("latency_median_s=nan\n")
        for estimate in estimates:
            fh.write(
                "step "
                f"{int(estimate['step_index'])}: "
                f"{estimate['from_command']:.3f}->{estimate['to_command']:.3f}, "
                f"write_elapsed_s={estimate['write_elapsed_s']:.6f}, "
                f"pre_f_off_hz={estimate['pre_f_off_hz']:.6f}, "
                f"final_f_off_hz={estimate['final_f_off_hz']:.6f}, "
                f"observed_delta_hz={estimate['observed_delta_hz']:.6f}, "
                f"expected_delta_hz={estimate['expected_delta_hz']:.6f}, "
                f"latency_s={estimate['latency_s']:.6f}\n"
            )

    print(f"Saved CSV to {csv_path}")
    if plot_path:
        print(f"Saved plot to {plot_path}")
    print(f"Saved summary to {summary_path}")
    if finite_latencies:
        print(
            f"Estimated actuator latency: median={float(np.median(finite_latencies)):.4f}s "
            f"range=[{float(np.min(finite_latencies)):.4f}, {float(np.max(finite_latencies)):.4f}]s"
        )
    else:
        print("Estimated actuator latency: unavailable")
    return csv_path, summary_path


def main(argv: list[str] | None = None) -> int:
    try:
        args = resolve_args(argv)
        run(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
