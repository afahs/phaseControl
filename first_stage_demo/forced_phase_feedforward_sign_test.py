#!/usr/bin/env python3
"""Force positive/negative phase feed-forward offsets and log measured phase."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import cw_vcxo_discipline as demo


def parse_sequence(text: str) -> list[tuple[float, float]]:
    sequence: list[tuple[float, float]] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"sequence part {part!r} must be formatted as target_hz:duration_s"
            )
        target_text, duration_text = part.split(":", 1)
        target_hz = float(target_text)
        duration_s = float(duration_text)
        if duration_s <= 0.0:
            raise argparse.ArgumentTypeError("segment durations must be positive")
        sequence.append((target_hz, duration_s))
    if not sequence:
        raise argparse.ArgumentTypeError("sequence must contain at least one segment")
    return sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=float, default=1_024_000.0)
    parser.add_argument("--center-freq", type=float, default=50_120_000.0)
    parser.add_argument("--tone-bb", type=float, default=-135_750.0)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--gain", default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--ugradio-queue-blocks", type=int, default=64)
    parser.add_argument("--cmd-start", type=float, default=2050.0)
    parser.add_argument("--cmd-min", type=float, default=0.0)
    parser.add_argument("--cmd-max", type=float, default=4095.0)
    parser.add_argument("--vcxo-hz-per-command", type=float, default=-4.5475)
    parser.add_argument("--clock-apr-ppm", type=int, default=120)
    parser.add_argument("--skip-clock-init", action="store_true")
    parser.add_argument("--no-mcp4725-dither", dest="mcp4725_dither", action="store_false")
    parser.set_defaults(mcp4725_dither=True)
    parser.add_argument("--control-settle-s", type=float, default=0.0)
    parser.add_argument("--warmup-blocks", type=int, default=8)
    parser.add_argument(
        "--sequence",
        type=parse_sequence,
        default=parse_sequence("0:6,0.5:10,-0.5:10,0:6"),
        help="Comma-separated target_hz:duration_s segments.",
    )
    parser.add_argument(
        "--out-bin",
        type=Path,
        default=Path("first_stage_demo/logs/forced_phase_ff_sign_test_20260603.bin"),
    )
    return parser.parse_args()


def current_segment(sequence: list[tuple[float, float]], t_rel_s: float) -> tuple[int, float, float]:
    elapsed = 0.0
    for index, (target_hz, duration_s) in enumerate(sequence):
        next_elapsed = elapsed + duration_s
        if t_rel_s < next_elapsed:
            return index, target_hz, elapsed
        elapsed = next_elapsed
    index = len(sequence) - 1
    target_hz, _duration_s = sequence[index]
    return index, target_hz, elapsed


def main() -> int:
    args = parse_args()
    total_duration_s = sum(duration_s for _target_hz, duration_s in args.sequence)
    detector = demo.TonePhaseDetector(args.sample_rate, args.block_size, args.tone_bb)
    block_dt_s = detector.block_dt_s
    total_blocks = int(math.ceil(total_duration_s / block_dt_s))
    freq_unambiguous_hz = 0.5 / block_dt_s
    tone_ref = demo.ToneReference(tone_bb_hz=args.tone_bb, tone_rf_hz=None, source="forced_phase_ff")

    sdr = demo.UgradioSdrSource(
        sample_rate_hz=args.sample_rate,
        center_freq_hz=args.center_freq,
        gain=args.gain,
        device_index=args.device_index,
        queue_blocks=args.ugradio_queue_blocks,
    )
    actuator = demo.Mcp4725Actuator(
        command_min=args.cmd_min,
        command_max=args.cmd_max,
        clock_apr_ppm=args.clock_apr_ppm,
        skip_clock_init=args.skip_clock_init,
        enable_dither=args.mcp4725_dither,
    )
    args.out_bin.parent.mkdir(parents=True, exist_ok=True)

    class RowArgs:
        pass

    row_args = RowArgs()
    row_args.mode = "forced_phase_ff_sign_test"
    row_args.sample_rate_hz = float(args.sample_rate)
    row_args.block_size = int(args.block_size)
    row_args.center_freq_hz = float(args.center_freq)
    row_args.phase_hold_gain_hz_per_rad = 0.0
    row_args.control_update_interval_s = block_dt_s
    row_args.control_average_interval_s = block_dt_s
    row_args.command_min = float(args.cmd_min)
    row_args.command_max = float(args.cmd_max)
    row_args.command_start = float(args.cmd_start)
    row_args.invert_control = False
    row_args.pll_frequency_proportional_gain = 0.0
    row_args.pll_frequency_integral_gain = 0.0
    row_args.acquisition_frequency_proportional_gain = 0.0
    row_args.acquisition_frequency_integral_gain = 0.0
    row_args.acquisition_frequency_deadband_hz = 0.0
    row_args.acquisition_command_slew_rate_per_s = 0.0
    row_args.frequency_error_deadband_hz = 0.0
    row_args.command_slew_rate_per_s = 0.0
    row_args.base_frequency_integral_gain = 0.0
    row_args.base_frequency_deadband_hz = 0.0
    row_args.base_command_slew_rate_per_s = 0.0
    row_args.base_min_coherence = 0.0
    row_args.base_max_phase_residual_rad = math.nan
    row_args.amp_threshold = 0.0
    row_args.coherence_threshold = 0.0
    row_args.vcxo_hz_per_command = float(args.vcxo_hz_per_command)

    try:
        actuator.open()
        applied_command = actuator.write_command(args.cmd_start, settle_s=args.control_settle_s)
        sdr.warmup(args.block_size, args.warmup_blocks)
        wall_start = time.monotonic()
        with args.out_bin.open("wb") as f:
            writer = demo.BinaryRowWriter(f, demo.csv_fieldnames())
            for block_index in range(total_blocks):
                t_rel_s = (block_index + 0.5) * block_dt_s
                segment_index, target_hz, segment_start_s = current_segment(args.sequence, t_rel_s)
                requested_offset = target_hz / float(args.vcxo_hz_per_command)
                requested_command = demo.clamp(
                    float(args.cmd_start) + requested_offset,
                    float(args.cmd_min),
                    float(args.cmd_max),
                )
                applied_command = actuator.write_command(
                    requested_command,
                    settle_s=args.control_settle_s,
                )
                iq = sdr.read_block(args.block_size)
                meas = detector.process_block(iq)
                control = demo.ControlUpdate(
                    applied_command=float(applied_command),
                    requested_command=float(requested_command),
                    p_term=0.0,
                    i_term=0.0,
                    error_hz=float(target_hz),
                )
                demo.write_csv_row(
                    writer=writer,
                    args=row_args,
                    tone_ref=tone_ref,
                    sdr_backend=sdr.backend_name,
                    actuator=actuator,
                    block_dt_s=block_dt_s,
                    freq_unambiguous_hz=freq_unambiguous_hz,
                    block_index=block_index,
                    t_rel_s=t_rel_s,
                    wall_rel_s=time.monotonic() - wall_start,
                    meas=meas,
                    update_allowed=True,
                    gate_reason="forced_phase_ff",
                    control=control,
                    applied_command=applied_command,
                    status=f"forced_phase_ff_{target_hz:+.3f}hz",
                    test_step_index=segment_index,
                    test_step_cmd=requested_command,
                    test_valid_for_fit=True,
                    extra_fields={
                        "control_target_f_off_hz": float(target_hz),
                        "control_base_cmd": float(args.cmd_start),
                        "phase_hold_active": 1,
                        "phase_control_mode": "forced",
                        "phase_hold_freq_setpoint_hz": float(target_hz),
                        "phase_hold_cmd_offset": float(requested_offset),
                        "phase_hold_raw_freq_setpoint_hz": float(target_hz),
                        "phase_control_max_freq_hz": abs(float(target_hz)),
                        "phase_hold_max_freq_hz": abs(float(target_hz)),
                        "control_stage": "forced",
                        "servo_profile": "forced",
                        "base_error_raw_hz": float(target_hz),
                        "base_error_filt_hz": float(target_hz),
                        "effective_frequency_error_hz": float(target_hz),
                        "unlimited_requested_cmd": float(requested_command),
                        "test_step_cmd": float(requested_command),
                    },
                )
                if block_index % max(1, int(round(1.0 / block_dt_s))) == 0:
                    f.flush()
        print(f"wrote {args.out_bin}")
    finally:
        try:
            actuator.write_command(args.cmd_start, settle_s=args.control_settle_s)
        except Exception:
            pass
        actuator.close()
        sdr.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
