#!/usr/bin/env python3
"""Run an integral-gain sweep for the dual-reference live controller."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dpllNotebook.notebook_style_dpll_integral_dualref as controller
import dpllNotebook.plot_integral_dualref_phase_comparison as phase_plot


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def build_controller_args(
    *,
    mode: str,
    nsamp: int,
    max_updates: int,
    csv_path: Path,
    summary_path: Path,
    target_offset_hz: float,
    observer_fixed_offset_hz: float,
    phase_frac_gain_codes: float,
    integral_gain_codes_per_rad_s: float,
    control_sign: float,
    control_track_alpha: float,
    quiet: bool,
) -> list[str]:
    args = [
        "--mode",
        str(mode),
        "--nsamp",
        str(nsamp),
        "--max-updates",
        str(max_updates),
        "--target-offset-hz",
        str(target_offset_hz),
        "--observer-fixed-offset-hz",
        str(observer_fixed_offset_hz),
        "--phase-frac-gain-codes",
        str(phase_frac_gain_codes),
        "--integral-gain-codes-per-rad-s",
        str(integral_gain_codes_per_rad_s),
        "--control-sign",
        str(control_sign),
        "--control-track-alpha",
        str(control_track_alpha),
        "--csv-out",
        str(csv_path),
        "--summary-out",
        str(summary_path),
    ]
    if quiet:
        args.append("--quiet")
    return args


def parse_controller_args(argv: list[str]) -> argparse.Namespace:
    parser = controller.build_arg_parser()
    return parser.parse_args(argv)


def run_capture(
    *,
    python_exe: str,
    controller_script: Path,
    mode: str,
    nsamp: int,
    max_updates: int,
    csv_path: Path,
    summary_path: Path,
    target_offset_hz: float,
    observer_fixed_offset_hz: float,
    phase_frac_gain_codes: float,
    integral_gain_codes_per_rad_s: float,
    control_sign: float,
    control_track_alpha: float,
    quiet: bool,
) -> dict[str, Any]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    argv = build_controller_args(
        mode=mode,
        nsamp=nsamp,
        max_updates=max_updates,
        csv_path=csv_path,
        summary_path=summary_path,
        target_offset_hz=target_offset_hz,
        observer_fixed_offset_hz=observer_fixed_offset_hz,
        phase_frac_gain_codes=phase_frac_gain_codes,
        integral_gain_codes_per_rad_s=integral_gain_codes_per_rad_s,
        control_sign=control_sign,
        control_track_alpha=control_track_alpha,
        quiet=quiet,
    )
    cmd = [python_exe, str(controller_script)] + argv
    print(f"\nRunning {mode}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "MPLCONFIGDIR": "/tmp/mplconfig"},
    )

    header_target = int(max_updates) + 1
    last_rows = -1
    last_report_s = 0.0

    while True:
        row_count = 0
        last_elapsed_s = float("nan")
        if csv_path.exists():
            with csv_path.open() as fh:
                row_count = sum(1 for _ in fh)
            if row_count > 1:
                rows = load_rows(csv_path)
                if rows:
                    try:
                        last_elapsed_s = float(rows[-1].get("elapsed_s", "nan"))
                    except Exception:
                        last_elapsed_s = float("nan")

        now_s = time.time()
        if row_count != last_rows and (now_s - last_report_s >= 5.0 or row_count >= header_target):
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "rows_including_header": row_count,
                        "last_elapsed_s": last_elapsed_s,
                        "proc_alive": proc.poll() is None,
                    }
                ),
                flush=True,
            )
            last_rows = row_count
            last_report_s = now_s

        if row_count >= header_target:
            break
        if proc.poll() is not None:
            break
        time.sleep(2.0)

    if proc.poll() is None:
        print("Target rows reached; interrupting child for teardown bypass.", flush=True)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)

    child_output = ""
    if proc.stdout is not None:
        child_output = proc.stdout.read()
    if child_output.strip():
        print("child_output_start", flush=True)
        print(child_output.strip(), flush=True)
        print("child_output_end", flush=True)
    print(f"child_returncode {proc.returncode}", flush=True)

    rows = load_rows(csv_path) if csv_path.exists() else []
    args = parse_controller_args(argv)
    summary = controller.compute_summary(rows=rows, args=args, observer_tone_offset_hz=observer_fixed_offset_hz)
    summary["csv_path"] = str(csv_path.resolve())
    summary["summary_path"] = str(summary_path.resolve())
    summary["child_returncode"] = int(proc.returncode)
    write_json(summary_path, summary)
    return {
        "rows": rows,
        "summary": summary,
        "child_output": child_output,
    }


def improvement_pct(open_drift: float, closed_drift: float) -> float:
    if abs(float(open_drift)) <= 1e-12:
        return float("nan")
    return float(100.0 * (abs(float(open_drift)) - abs(float(closed_drift))) / abs(float(open_drift)))


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for item in results:
        summary = item["summary"]
        scored.append(
            {
                "integral_gain_codes_per_rad_s": float(item["integral_gain_codes_per_rad_s"]),
                "observer_timing_drift_samples_per_s": float(summary.get("observer_timing_drift_samples_per_s", float("nan"))),
                "observer_timing_drift_ns_per_s": float(summary.get("observer_timing_drift_ns_per_s", float("nan"))),
                "observer_phase_drift_slope_hz": float(summary.get("observer_phase_drift_slope_hz", float("nan"))),
                "mean_abs_control_phase_err_rad": float(summary.get("mean_abs_control_phase_err_rad", float("nan"))),
                "median_control_coherence": float(summary.get("median_control_coherence", float("nan"))),
                "dac_update_fraction": float(summary.get("dac_update_fraction", float("nan"))),
                "dac_updates_applied": int(summary.get("dac_updates_applied", 0)),
                "max_abs_observer_timing_proxy_ns": float(summary.get("max_abs_observer_timing_proxy_ns", float("nan"))),
                "csv_path": str(summary.get("csv_path", "")),
                "summary_path": str(summary.get("summary_path", "")),
            }
        )
    scored.sort(key=lambda item: abs(float(item["observer_timing_drift_samples_per_s"])))
    return scored


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep integral gain for the dual-reference controller.")
    parser.add_argument(
        "--python-exe",
        type=str,
        default="/home/rrrodriguez13/sdr/.env/bin/python",
        help="Python interpreter with SDR and plotting dependencies.",
    )
    parser.add_argument(
        "--controller-script",
        type=Path,
        default=ROOT / "dpllNotebook/notebook_style_dpll_integral_dualref.py",
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--nsamp", type=int, default=32768)
    parser.add_argument("--max-updates", type=int, default=4395)
    parser.add_argument("--target-offset-hz", type=float, default=-31347.65625)
    parser.add_argument("--observer-fixed-offset-hz", type=float, default=-31347.65625)
    parser.add_argument("--phase-frac-gain-codes", type=float, default=6000000.0)
    parser.add_argument("--control-sign", type=float, default=-1.0)
    parser.add_argument("--control-track-alpha", type=float, default=0.0)
    parser.add_argument("--integral-gains", type=float, nargs="+", default=[0.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    parser.add_argument("--quiet", action="store_true", default=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_root = args.run_root
    if run_root is None:
        run_root = ROOT / "dpllNotebook" / "runs" / time.strftime("%Y%m%d_%H%M%S_integral_gain_sweep")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "monitor").mkdir(parents=True, exist_ok=True)
    (run_root / "comparison").mkdir(parents=True, exist_ok=True)

    open_csv = run_root / "monitor" / "open_loop.csv"
    open_summary = run_root / "monitor" / "open_loop_summary.json"
    open_result = run_capture(
        python_exe=str(args.python_exe),
        controller_script=Path(args.controller_script),
        mode="monitor",
        nsamp=int(args.nsamp),
        max_updates=int(args.max_updates),
        csv_path=open_csv,
        summary_path=open_summary,
        target_offset_hz=float(args.target_offset_hz),
        observer_fixed_offset_hz=float(args.observer_fixed_offset_hz),
        phase_frac_gain_codes=float(args.phase_frac_gain_codes),
        integral_gain_codes_per_rad_s=0.0,
        control_sign=float(args.control_sign),
        control_track_alpha=float(args.control_track_alpha),
        quiet=bool(args.quiet),
    )

    results: list[dict[str, Any]] = []
    for gain in args.integral_gains:
        gain_dir = run_root / "pid" / f"igain_{str(gain).replace('.', 'p')}"
        gain_dir.mkdir(parents=True, exist_ok=True)
        closed_csv = gain_dir / "closed_loop.csv"
        closed_summary = gain_dir / "closed_loop_summary.json"
        closed_result = run_capture(
            python_exe=str(args.python_exe),
            controller_script=Path(args.controller_script),
            mode="pid",
            nsamp=int(args.nsamp),
            max_updates=int(args.max_updates),
            csv_path=closed_csv,
            summary_path=closed_summary,
            target_offset_hz=float(args.target_offset_hz),
            observer_fixed_offset_hz=float(args.observer_fixed_offset_hz),
            phase_frac_gain_codes=float(args.phase_frac_gain_codes),
            integral_gain_codes_per_rad_s=float(gain),
            control_sign=float(args.control_sign),
            control_track_alpha=float(args.control_track_alpha),
            quiet=bool(args.quiet),
        )

        dash_path = run_root / "comparison" / f"phase_dashboard_igain_{str(gain).replace('.', 'p')}.png"
        metrics = phase_plot.plot_phase_dashboard(
            open_rows=open_result["rows"],
            closed_rows=closed_result["rows"],
            open_summary=open_result["summary"],
            closed_summary=closed_result["summary"],
            output_path=dash_path,
        )
        metrics_path = run_root / "comparison" / f"phase_metrics_igain_{str(gain).replace('.', 'p')}.json"
        write_json(metrics_path, metrics)

        enriched = {
            "integral_gain_codes_per_rad_s": float(gain),
            "summary": closed_result["summary"],
            "phase_metrics": metrics,
            "phase_metrics_path": str(metrics_path.resolve()),
            "phase_dashboard_path": str(dash_path.resolve()),
            "improvement_vs_open_pct": improvement_pct(
                float(open_result["summary"].get("observer_timing_drift_samples_per_s", float("nan"))),
                float(closed_result["summary"].get("observer_timing_drift_samples_per_s", float("nan"))),
            ),
        }
        results.append(enriched)

    ranked = rank_results(results)
    sweep_summary = {
        "run_root": str(run_root.resolve()),
        "config": {
            "nsamp": int(args.nsamp),
            "max_updates": int(args.max_updates),
            "target_offset_hz": float(args.target_offset_hz),
            "observer_fixed_offset_hz": float(args.observer_fixed_offset_hz),
            "phase_frac_gain_codes": float(args.phase_frac_gain_codes),
            "control_sign": float(args.control_sign),
            "control_track_alpha": float(args.control_track_alpha),
            "integral_gains": [float(x) for x in args.integral_gains],
        },
        "open_summary": open_result["summary"],
        "results_ranked_by_abs_drift": ranked,
    }
    write_json(run_root / "comparison" / "integral_gain_sweep_summary.json", sweep_summary)
    print("\nSweep ranking:")
    print(json.dumps(ranked, indent=2, sort_keys=True))
    print(f"\nSweep summary: {run_root / 'comparison' / 'integral_gain_sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
