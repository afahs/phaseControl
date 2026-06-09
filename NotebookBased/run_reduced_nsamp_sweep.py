#!/usr/bin/env python3
"""Run organized monitor/PID sweeps for the notebook-based DPLL."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import notebook_based_pid_dpll as dpll


BASE_DIR = Path(__file__).resolve().parent
DPLL_SCRIPT = BASE_DIR / "notebook_based_pid_dpll.py"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def timestamp_for_filename() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def parse_nsamp_values(text: str) -> list[int]:
    values: list[int] = []
    for token in str(text).split(","):
        item = token.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("All nsamp values must be positive.")
        values.append(value)
    if not values:
        raise ValueError("At least one nsamp value is required.")
    return values


def parse_modes(text: str) -> list[str]:
    allowed = {"monitor", "pid"}
    values: list[str] = []
    for token in str(text).split(","):
        item = token.strip().lower()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unsupported mode {item!r}. Allowed: monitor,pid")
        if item not in values:
            values.append(item)
    if not values:
        raise ValueError("At least one mode is required.")
    return values


def load_summary(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a single-summary JSON object in {path}.")
    return data


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def reconstruct_summary_from_csv(
    *,
    csv_path: Path,
    mode: str,
    nsamp: int,
    sample_rate_hz: float,
    center_freq_hz: float,
    drift_window_s: float,
    summary_json_path: Path,
    summary_csv_path: Path,
) -> dict[str, Any]:
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"Cannot reconstruct summary because {csv_path} has no data rows.")

    tone_offset_hz = float(rows[-1].get("tone_offset_hz", "nan"))
    summary = dpll.compute_summary(
        rows=rows,
        run_mode=str(mode),
        nsamp=int(nsamp),
        sample_rate_hz=float(sample_rate_hz),
        center_freq_hz=float(center_freq_hz),
        tone_offset_hz=float(tone_offset_hz),
        drift_window_s=float(drift_window_s),
    )
    summary["csv_path"] = str(csv_path)
    summary["stage_switches"] = int(sum(int(r.get("stage_switch_event", 0) or 0) for r in rows))
    summary["pullin_updates_applied"] = int(
        sum(1 for r in rows if str(r.get("action", "")).endswith("_apply") and r.get("stage") == "pull_in")
    )
    summary["phase_updates_applied"] = int(
        sum(1 for r in rows if str(r.get("action", "")).endswith("_apply") and r.get("stage") == "phase_lock")
    )
    summary["dac_code_start"] = int(rows[0].get("dac_code_cmd", 0) or 0)
    summary["dac_code_end"] = int(rows[-1].get("dac_code_cmd", 0) or 0)
    summary["note"] = "summary reconstructed from CSV after timeout"

    ensure_parent_dir(summary_json_path)
    with summary_json_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    write_summary_csv(summary, summary_csv_path)
    return summary


def timing_drift_samples_per_s(summary: dict[str, Any]) -> float:
    direct = float(summary.get("primary_timing_drift_samples_per_s", "nan"))
    if direct == direct:
        return direct
    drift_ns_per_s = float(summary.get("primary_timing_drift_ns_per_s", "nan"))
    sample_rate_hz = float(summary.get("sample_rate_hz", "nan"))
    if sample_rate_hz <= 0.0:
        return float("nan")
    sample_period_ns = 1e9 / sample_rate_hz
    return drift_ns_per_s / sample_period_ns


def build_command(
    args: argparse.Namespace,
    *,
    mode: str,
    nsamp: int,
    csv_path: Path,
    summary_json_path: Path,
    summary_csv_path: Path,
    observer_fixed_offset_hz: float | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(DPLL_SCRIPT),
        "--mode",
        str(mode),
        "--device-index",
        str(args.device_index),
        "--center-freq",
        str(args.center_freq),
        "--sample-rate",
        str(args.sample_rate),
        "--gain",
        str(args.gain),
        "--settle-seconds",
        str(args.settle_seconds),
        "--nsamp",
        str(nsamp),
        "--blocks-per-update",
        "1",
        "--async-queue-blocks",
        str(args.async_queue_blocks),
        "--run-seconds",
        str(args.run_seconds),
        "--tone-estimate-blocks",
        str(args.tone_estimate_blocks),
        "--dc-exclusion-hz",
        str(args.dc_exclusion_hz),
        "--drift-window-s",
        str(args.drift_window_s),
        "--control-measurement-source",
        str(args.control_measurement_source),
        "--csv-out",
        str(csv_path),
        "--summary-out",
        str(summary_json_path),
        "--summary-csv",
        str(summary_csv_path),
        "--notes",
        str(args.notes),
    ]

    if args.target_offset_hz is not None:
        cmd.extend(["--target-offset-hz", str(args.target_offset_hz)])
    if observer_fixed_offset_hz is not None:
        cmd.extend(["--observer-fixed-offset-hz", str(observer_fixed_offset_hz)])

    if mode == "pid":
        cmd.extend(
            [
                "--coh-min",
                str(args.pid_coh_min),
                "--pullin-quality-min",
                str(args.pid_pullin_quality_min),
                "--pullin-filter-alpha",
                str(args.pid_pullin_filter_alpha),
                "--tone-track-alpha",
                str(args.pid_tone_track_alpha),
                "--tone-track-min-coh",
                str(args.pid_tone_track_min_coh),
                "--tone-track-max-step-hz",
                str(args.pid_tone_track_max_step_hz),
                "--control-sign",
                str(args.control_sign),
                "--kv-hz-per-code",
                str(args.kv_hz_per_code),
                "--Kf-pullin-hz-per-hz",
                str(args.Kf_pullin_hz_per_hz),
                "--Kp-hz-per-rad",
                str(args.Kp_hz_per_rad),
                "--Ki-hz-per-rad-s",
                str(args.Ki_hz_per_rad_s),
                "--Kd-hz-s-per-rad",
                str(args.Kd_hz_s_per_rad),
                "--max-u-hz",
                str(args.max_u_hz),
                "--max-dac-step-per-update",
                str(args.max_dac_step_per_update),
                "--pullin-switch-hz",
                str(args.pullin_switch_hz),
                "--pullin-exit-hz",
                str(args.pullin_exit_hz),
                "--pullin-enter-dwell-updates",
                str(args.pullin_enter_dwell_updates),
                "--pullin-exit-dwell-updates",
                str(args.pullin_exit_dwell_updates),
                "--pullin-target-hz",
                str(args.pullin_target_hz),
                "--apr-ppm",
                str(args.apr_ppm),
                "--fine-settle-s",
                str(args.fine_settle_s),
            ]
        )
        if args.skip_clock_init:
            cmd.append("--skip-clock-init")
        if args.apply_start_dac:
            cmd.append("--apply-start-dac")
        else:
            cmd.append("--no-apply-start-dac")

    return cmd


def summarize_comparison(
    nsamp: int,
    monitor_summary: dict[str, Any],
    pid_summary: dict[str, Any],
) -> dict[str, Any]:
    monitor_timing = float(monitor_summary.get("primary_timing_drift_ns_per_s", "nan"))
    pid_timing = float(pid_summary.get("primary_timing_drift_ns_per_s", "nan"))
    monitor_phase = float(monitor_summary.get("phase_drift_slope_full_hz", "nan"))
    pid_phase = float(pid_summary.get("phase_drift_slope_full_hz", "nan"))

    if abs(monitor_timing) > 1e-12:
        timing_improvement_pct = 100.0 * (abs(monitor_timing) - abs(pid_timing)) / abs(monitor_timing)
    else:
        timing_improvement_pct = float("nan")

    if abs(monitor_phase) > 1e-12:
        phase_improvement_pct = 100.0 * (abs(monitor_phase) - abs(pid_phase)) / abs(monitor_phase)
    else:
        phase_improvement_pct = float("nan")

    return {
        "nsamp": int(nsamp),
        "loop_period_nom_s": float(nsamp) / float(monitor_summary["sample_rate_hz"]),
        "monitor_updates": int(monitor_summary.get("updates", 0)),
        "pid_updates": int(pid_summary.get("updates", 0)),
        "monitor_phase_drift_slope_full_hz": monitor_phase,
        "pid_phase_drift_slope_full_hz": pid_phase,
        "monitor_timing_drift_ns_per_s": monitor_timing,
        "pid_timing_drift_ns_per_s": pid_timing,
        "monitor_timing_drift_samples_per_s": timing_drift_samples_per_s(monitor_summary),
        "pid_timing_drift_samples_per_s": timing_drift_samples_per_s(pid_summary),
        "phase_improvement_pct": float(phase_improvement_pct),
        "timing_improvement_pct": float(timing_improvement_pct),
        "monitor_median_coherence": float(monitor_summary.get("median_coherence", "nan")),
        "pid_median_coherence": float(pid_summary.get("median_coherence", "nan")),
        "monitor_mean_abs_freq_err_hz": float(monitor_summary.get("mean_abs_freq_err_hz", "nan")),
        "pid_mean_abs_freq_err_hz": float(pid_summary.get("mean_abs_freq_err_hz", "nan")),
        "monitor_quality_fraction": float(monitor_summary.get("quality_fraction", "nan")),
        "pid_quality_fraction": float(pid_summary.get("quality_fraction", "nan")),
        "monitor_quality_updates": int(monitor_summary.get("quality_updates", 0)),
        "pid_quality_updates": int(pid_summary.get("quality_updates", 0)),
        "pid_phase_updates_applied": int(pid_summary.get("phase_updates_applied", 0)),
        "pid_pullin_updates_applied": int(pid_summary.get("pullin_updates_applied", 0)),
        "pid_stage_switches": int(pid_summary.get("stage_switches", 0)),
    }


def write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    ensure_parent_dir(path)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run organized reduced-nsamp monitor/PID sweeps for the notebook DPLL."
    )

    p.add_argument("--label", type=str, default="reduced_nsamp")
    p.add_argument("--artifact-root", type=Path, default=BASE_DIR / "runs")
    p.add_argument("--nsamp-values", type=str, default="8192,16384,32768")
    p.add_argument("--modes", type=str, default="monitor,pid")

    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--center-freq", type=float, default=100e6)
    p.add_argument("--sample-rate", type=float, default=2.4e6)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--settle-seconds", type=float, default=0.25)
    p.add_argument("--run-seconds", type=float, default=15.0)
    p.add_argument(
        "--command-timeout-margin-s",
        type=float,
        default=8.0,
        help="Extra wall time allowed beyond run-seconds before the outer runner kills a wedged capture.",
    )
    p.add_argument("--async-queue-blocks", type=int, default=4)
    p.add_argument("--tone-estimate-blocks", type=int, default=8)
    p.add_argument("--dc-exclusion-hz", type=float, default=5e3)
    p.add_argument("--target-offset-hz", type=float, default=None)
    p.add_argument(
        "--observer-fixed-offset-hz",
        type=float,
        default=None,
        help="Optional fixed observer reference. If omitted, each run freezes its own bootstrap tone.",
    )
    p.add_argument(
        "--reuse-monitor-observer-for-pid",
        action="store_true",
        help="Reuse the matching monitor tone estimate as the PID run's fixed observer reference.",
    )
    p.add_argument("--drift-window-s", type=float, default=8.0)
    p.add_argument(
        "--control-measurement-source",
        choices=["adaptive", "observer"],
        default="observer",
        help="Which notebook DPLL measurement stream drives the live control loop.",
    )
    p.add_argument("--notes", type=str, default="reduced-nsamp sweep")

    p.add_argument("--pid-coh-min", type=float, default=0.02)
    p.add_argument("--pid-pullin-quality-min", type=float, default=0.10)
    p.add_argument("--pid-pullin-filter-alpha", type=float, default=0.80)
    p.add_argument("--pid-tone-track-alpha", type=float, default=0.0)
    p.add_argument("--pid-tone-track-min-coh", type=float, default=0.01)
    p.add_argument("--pid-tone-track-max-step-hz", type=float, default=2000.0)
    p.add_argument("--control-sign", type=float, default=1.0)
    p.add_argument("--kv-hz-per-code", type=float, default=4.243)
    p.add_argument("--Kf-pullin-hz-per-hz", type=float, default=0.07)
    p.add_argument("--Kp-hz-per-rad", type=float, default=40.0)
    p.add_argument("--Ki-hz-per-rad-s", type=float, default=4.0)
    p.add_argument("--Kd-hz-s-per-rad", type=float, default=0.0)
    p.add_argument("--max-u-hz", type=float, default=2000.0)
    p.add_argument("--max-dac-step-per-update", type=int, default=20)
    p.add_argument("--pullin-switch-hz", type=float, default=100.0)
    p.add_argument("--pullin-exit-hz", type=float, default=300.0)
    p.add_argument("--pullin-enter-dwell-updates", type=int, default=3)
    p.add_argument("--pullin-exit-dwell-updates", type=int, default=2)
    p.add_argument("--pullin-target-hz", type=float, default=0.0)
    p.add_argument("--apr-ppm", type=int, default=120)
    p.add_argument("--fine-settle-s", type=float, default=0.0)
    p.add_argument("--skip-clock-init", action="store_true")
    p.add_argument("--apply-start-dac", dest="apply_start_dac", action="store_true", default=True)
    p.add_argument("--no-apply-start-dac", dest="apply_start_dac", action="store_false")

    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        nsamp_values = parse_nsamp_values(args.nsamp_values)
        modes = parse_modes(args.modes)
    except Exception as exc:
        print(f"Invalid arguments: {exc}", file=sys.stderr)
        return 2

    run_id = f"{timestamp_for_filename()}_{args.label}"
    session_dir = Path(args.artifact_root) / run_id
    summary_index: list[dict[str, Any]] = []
    monitor_summaries: dict[int, dict[str, Any]] = {}
    pid_summaries: dict[int, dict[str, Any]] = {}
    observer_offsets_by_nsamp: dict[int, float | None] = {}

    for mode in modes:
        for nsamp in nsamp_values:
            mode_dir = session_dir / mode / f"nsamp_{nsamp}"
            csv_path = mode_dir / f"{mode}_nsamp{nsamp}.csv"
            summary_json_path = mode_dir / f"{mode}_nsamp{nsamp}_summary.json"
            summary_csv_path = mode_dir / f"{mode}_nsamp{nsamp}_summary.csv"
            ensure_parent_dir(csv_path)

            observer_fixed_offset_hz = args.observer_fixed_offset_hz
            if (
                observer_fixed_offset_hz is None
                and bool(args.reuse_monitor_observer_for_pid)
                and mode == "pid"
            ):
                observer_fixed_offset_hz = observer_offsets_by_nsamp.get(int(nsamp))

            cmd = build_command(
                args,
                mode=mode,
                nsamp=nsamp,
                csv_path=csv_path,
                summary_json_path=summary_json_path,
                summary_csv_path=summary_csv_path,
                observer_fixed_offset_hz=observer_fixed_offset_hz,
            )

            print(
                f"Running mode={mode} nsamp={nsamp} "
                f"(loop_period={nsamp / args.sample_rate:.6f}s) -> {mode_dir}",
                flush=True,
            )
            timeout_s = max(float(args.run_seconds) + float(args.command_timeout_margin_s), 1.0)
            wrapped_cmd = ["timeout", "--kill-after=3s", f"{timeout_s}s", *cmd]
            result = subprocess.run(wrapped_cmd, check=False, cwd=BASE_DIR.parent)

            if summary_json_path.exists():
                summary = load_summary(summary_json_path)
            elif csv_path.exists() and csv_path.stat().st_size > 0 and result.returncode == 124:
                summary = reconstruct_summary_from_csv(
                    csv_path=csv_path,
                    mode=mode,
                    nsamp=int(nsamp),
                    sample_rate_hz=float(args.sample_rate),
                    center_freq_hz=float(args.center_freq),
                    drift_window_s=float(args.drift_window_s),
                    summary_json_path=summary_json_path,
                    summary_csv_path=summary_csv_path,
                )
            else:
                raise subprocess.CalledProcessError(result.returncode, wrapped_cmd)

            summary_index.append(
                {
                    "mode": str(mode),
                    "nsamp": int(nsamp),
                    "returncode": int(result.returncode),
                    "csv_path": str(csv_path),
                    "summary_json_path": str(summary_json_path),
                    "summary_csv_path": str(summary_csv_path),
                }
            )
            if mode == "monitor":
                monitor_summaries[int(nsamp)] = summary
                observer_offsets_by_nsamp[int(nsamp)] = float(summary.get("tone_offset_hz", "nan"))
            else:
                pid_summaries[int(nsamp)] = summary

    comparisons: list[dict[str, Any]] = []
    if "monitor" in modes and "pid" in modes:
        for nsamp in nsamp_values:
            if nsamp not in monitor_summaries or nsamp not in pid_summaries:
                continue
            comparisons.append(
                summarize_comparison(
                    nsamp,
                    monitor_summaries[nsamp],
                    pid_summaries[nsamp],
                )
            )

    index_path = session_dir / "session_index.json"
    ensure_parent_dir(index_path)
    with index_path.open("w") as fh:
        json.dump(summary_index, fh, indent=2, sort_keys=True)

    if comparisons:
        comparison_dir = session_dir / "comparison"
        comparison_json = comparison_dir / "nsamp_comparison.json"
        comparison_csv = comparison_dir / "nsamp_comparison.csv"
        ensure_parent_dir(comparison_json)
        with comparison_json.open("w") as fh:
            json.dump(comparisons, fh, indent=2, sort_keys=True)
        write_comparison_csv(comparisons, comparison_csv)

        print("\nComparison summary:")
        for row in comparisons:
            print(
                f"  nsamp={row['nsamp']}: timing {row['monitor_timing_drift_samples_per_s']:.3f}"
                f" -> {row['pid_timing_drift_samples_per_s']:.3f} samples/s, "
                f"phase drift |Hz| {abs(row['monitor_phase_drift_slope_full_hz']):.2f}"
                f" -> {abs(row['pid_phase_drift_slope_full_hz']):.2f}, "
                f"timing improvement {row['timing_improvement_pct']:.1f}%, "
                f"quality {row['monitor_quality_fraction']:.2f} -> {row['pid_quality_fraction']:.2f}"
            )

    print(f"\nArtifacts written under: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
