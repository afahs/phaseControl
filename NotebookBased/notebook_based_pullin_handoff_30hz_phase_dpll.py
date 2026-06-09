#!/usr/bin/env python3
"""NotebookBased DPLL with pull-in and a simple 30 Hz residual handoff to phase mode."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import notebook_based_hybrid_pullin_notebook_phase_dpll as base


VDD_DAC_DEFAULT = base.VDD_DAC_DEFAULT
wrap_to_pi = base.wrap_to_pi
code_to_volts = base.code_to_volts
open_sdr = base.open_sdr
cvh = base.cvh

REFERENCE_SAMPLE_RATE_HZ = 2.4e6
REFERENCE_NSAMP = 16384
REFERENCE_LOOP_PERIOD_S = float(REFERENCE_NSAMP) / float(REFERENCE_SAMPLE_RATE_HZ)


def build_arg_parser():
    parser = base.build_arg_parser()
    parser.description = "Hybrid DPLL with pull-in, then phase mode after 30 Hz residual dwell."
    parser.set_defaults(
        single_stage_phase_only=False,
        control_measurement_source="adaptive",
        phase_reference_mode="first_valid",
        phase_enter_condition="residual_and_quality",
        phase_exit_condition="residual_and_phase",
        phase_exit_min_err_rad=1.0,
        phase_min_hold_updates=64,
        phase_min_hold_s=64.0 * REFERENCE_LOOP_PERIOD_S,
        phase_detector_chunks=4,
        phase_fit_rms_max_rad=0.8,
        phase_fit_max_abs_rad=1.6,
        phase_innovation_max_rad=1.8,
        phase_innovation_hard_max_rad=0.0,
        phase_innovation_soft_blend=0.0,
        phase_innovation_soft_fit_rms_max_rad=0.0,
        phase_predict_through_holds=True,
        phase_observer_recenter_on_entry=True,
        phase_observer_recenter_max_step_hz=30.0,
        phase_observer_track_alpha=0.05,
        phase_observer_track_min_quality=0.90,
        phase_observer_track_max_step_hz=5.0,
        ignore_blocks_after_dac_write=1,
        phase_hold_kp_hz_per_rad=8.0,
        phase_hold_ki_hz_per_rad_s=3.0,
        phase_hold_kd_hz_per_hz=0.20,
        phase_accepted_raw_enter_window_updates=12,
        phase_accepted_raw_enter_min_frac=0.75,
        phase_accepted_raw_exit_window_updates=12,
        phase_accepted_raw_exit_min_frac=0.50,
        phase_reference_ramp_accepted_updates=24,
        phase_reference_ramp_min_scale=0.25,
        pullin_switch_hz=30.0,
        pullin_exit_hz=100.0,
        pullin_enter_dwell_updates=3,
        pullin_enter_dwell_s=3.0 * REFERENCE_LOOP_PERIOD_S,
        pullin_exit_dwell_updates=16,
        pullin_exit_dwell_s=16.0 * REFERENCE_LOOP_PERIOD_S,
    )
    return parser


async def run_one(args, nsamp: int, csv_path: Path):
    base.open_sdr = open_sdr
    base.cvh = cvh
    return await base.run_one(args, nsamp, csv_path)


def write_summary_csv(summaries, path: Path) -> None:
    base.write_summary_csv(summaries, path)


def print_run_summary(summary) -> None:
    base.print_run_summary(summary)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.single_stage_phase_only = False

    try:
        nsamp_values = base.parse_nsamp_values(args.nsamp_values, int(args.nsamp))
    except Exception as exc:
        print(f"Invalid --nsamp-values: {exc}", file=sys.stderr)
        return 2

    if float(args.sample_rate) <= 0.0:
        print("--sample-rate must be > 0.", file=sys.stderr)
        return 2
    if int(args.blocks_per_update) != 1:
        print("--blocks-per-update must be 1 for this implementation.", file=sys.stderr)
        return 2
    if float(args.pullin_exit_hz) < float(args.pullin_switch_hz):
        print("--pullin-exit-hz should be >= --pullin-switch-hz.", file=sys.stderr)
        return 2
    if int(args.log_flush_rows) <= 0:
        print("--log-flush-rows must be > 0.", file=sys.stderr)
        return 2
    if float(args.log_flush_interval_s) < 0.0:
        print("--log-flush-interval-s must be >= 0.", file=sys.stderr)
        return 2

    summaries = []
    for idx, nsamp in enumerate(nsamp_values):
        suffix = f"_nsamp{int(nsamp)}" if len(nsamp_values) > 1 else ""
        csv_path = base.path_with_suffix(Path(args.csv_out), suffix)

        if not args.quiet:
            loop_period = float(nsamp) / float(args.sample_rate)
            print(
                f"Starting run {idx + 1}/{len(nsamp_values)} | mode={args.mode} | nsamp={nsamp} | "
                f"loop_period_nom={loop_period:.6f}s | csv={csv_path}",
                flush=True,
            )
            if loop_period > 1.0:
                print(
                    "Warning: nominal loop period exceeds 1 second; consider smaller nsamp or higher sample rate.",
                    flush=True,
                )

        run_summary = asyncio.run(run_one(args, int(nsamp), csv_path))
        summaries.append(run_summary)

        if not args.quiet:
            print_run_summary(run_summary)

    base.ensure_parent_dir(Path(args.summary_out))
    with Path(args.summary_out).open("w") as fh:
        if len(summaries) == 1:
            json.dump(summaries[0], fh, indent=2, sort_keys=True)
        else:
            json.dump(summaries, fh, indent=2, sort_keys=True)

    summary_csv_path = args.summary_csv
    if summary_csv_path is None:
        summary_csv_path = Path(args.summary_out).with_suffix(".csv")
    write_summary_csv(summaries, Path(summary_csv_path))

    if not args.quiet:
        print(f"Summary JSON written to: {Path(args.summary_out)}")
        print(f"Summary CSV written to:  {Path(summary_csv_path)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
