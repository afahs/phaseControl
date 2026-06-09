#!/usr/bin/env python3
"""True single-stage phase-mode entrypoint for the NotebookBased live DPLL."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import notebook_based_hybrid_pullin_notebook_phase_dpll as base


VDD_DAC_DEFAULT = base.VDD_DAC_DEFAULT
wrap_to_pi = base.wrap_to_pi
code_to_volts = base.code_to_volts
open_sdr = base.open_sdr
cvh = base.cvh


def build_arg_parser():
    parser = base.build_arg_parser()
    parser.description = "True single-stage phase-only live DPLL."
    parser.set_defaults(single_stage_phase_only=True)
    return parser


async def run_one(args, nsamp: int, csv_path: Path) -> dict[str, Any]:
    base.open_sdr = open_sdr
    base.cvh = cvh
    return await base.run_one(args, nsamp, csv_path)


def write_summary_csv(summaries: list[dict[str, Any]], path: Path) -> None:
    base.write_summary_csv(summaries, path)


def print_run_summary(summary: dict[str, Any]) -> None:
    base.print_run_summary(summary)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.single_stage_phase_only = True

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

    summaries: list[dict[str, Any]] = []
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
