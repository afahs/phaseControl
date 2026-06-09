#!/usr/bin/env python3
"""Sweep phase-threshold and gain settings using the synthetic open/closed harness."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_saved_phase_run as phase_eval
import run_simulated_open_closed as sim


BASE_DIR = Path(__file__).resolve().parent


def timestamp_for_filename() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def parse_float_list(text: str | None, fallback: float) -> list[float]:
    if text is None:
        return [float(fallback)]
    values: list[float] = []
    for token in str(text).split(","):
        item = token.strip()
        if not item:
            continue
        values.append(float(item))
    return values or [float(fallback)]


def parse_int_list(text: str | None, fallback: int) -> list[int]:
    if text is None:
        return [int(fallback)]
    values: list[int] = []
    for token in str(text).split(","):
        item = token.strip()
        if not item:
            continue
        values.append(int(item))
    return values or [int(fallback)]


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def finite_max(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if finite.size else float("nan")


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def bool_fraction(values: list[bool]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def write_json(path: Path, data: Any) -> None:
    sim.ensure_parent_dir(path)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    sim.ensure_parent_dir(path)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(
    *,
    sweep_dir: Path,
    aggregate_rows: list[dict[str, Any]],
    top_k: int,
) -> str:
    lines = [
        "# Phase Tuning Sweep",
        "",
        f"- Sweep directory: `{sweep_dir}`",
        "- Ranking order: finite primary-phase metric coverage, mean steady-state median |primary phase error|, mean primary phase-band fraction, mean best good dwell, sham gap.",
        "",
        "## Top Trials",
        "",
    ]
    for idx, row in enumerate(aggregate_rows[:top_k], start=1):
        params = row["parameter_summary"]
        lines.extend(
            [
                f"{idx}. `{row['trial_id']}`",
                f"   - Parameters: `{params}`",
                f"   - Finite phase metric fraction: `{row['finite_selected_phase_metric_fraction']}`",
                f"   - Mean steady-state median |primary phase error|: `{row['mean_selected_steady_state_median_abs_phase_err_primary_rad']}` rad",
                f"   - Mean full-run primary phase error proxy: `{row['mean_selected_full_run_primary_err_proxy_rad']}` rad",
                f"   - Mean steady-state primary phase-band fraction: `{row['mean_selected_steady_state_primary_phase_band_fraction']}`",
                f"   - Mean full-run primary phase-band fraction: `{row['mean_selected_full_run_primary_phase_band_fraction']}`",
                f"   - Mean best good dwell: `{row['mean_selected_best_good_dwell_s']}` s",
                f"   - Mean phase-mode fraction: `{row['mean_pid_phase_mode_fraction']}`",
                f"   - Mean sham phase-error gap: `{row['mean_selected_sham_phase_err_gap_rad']}` rad",
            ]
        )
    return "\n".join(lines) + "\n"


def evaluate_single_mode(
    *,
    rows: list[dict[str, Any]],
    sample_rate_hz: float,
    phase_band_rad: float,
    min_lock_dwell_s: float,
    max_bad_gap_s: float,
    coh_min: float | None,
    require_phase_lock_stage: bool,
) -> phase_eval.SettledMetrics:
    return phase_eval.detect_settled_metrics(
        rows=rows,
        sample_rate_hz=float(sample_rate_hz),
        phase_band_rad=float(phase_band_rad),
        min_lock_dwell_s=float(min_lock_dwell_s),
        max_bad_gap_s=float(max_bad_gap_s),
        require_quality_ok=True,
        coh_min=float(coh_min) if coh_min is not None else None,
        require_phase_lock_stage=bool(require_phase_lock_stage),
    )


def choose_eval_output(
    stage_output: dict[str, Any],
    phase_only_output: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    stage_metric = float_or_nan(
        stage_output.get("pid_phase_metrics", {}).get(
            "steady_state_median_abs_phase_err_primary_rad",
        )
    )
    if np.isfinite(stage_metric):
        return "stage_required", stage_output
    return "phase_only", phase_only_output


def sham_metrics_for_require_stage(
    *,
    session_dir: Path,
    nsamp: int,
    phase_band_rad: float,
    min_lock_dwell_s: float,
    max_bad_gap_s: float,
    coh_min: float | None,
    require_phase_lock_stage: bool,
) -> tuple[phase_eval.SettledMetrics, phase_eval.SettledMetrics]:
    monitor_rows = phase_eval.load_csv_rows(sim.mode_artifact_paths(session_dir, "monitor", int(nsamp))["csv"])
    sham_rows = phase_eval.load_csv_rows(sim.mode_artifact_paths(session_dir, "sham", int(nsamp))["csv"])
    sample_rate_hz = float(
        phase_eval.load_json(sim.mode_artifact_paths(session_dir, "sham", int(nsamp))["summary_json"]).get(
            "sample_rate_hz",
            2.4e6,
        )
    )
    return (
        evaluate_single_mode(
            rows=monitor_rows,
            sample_rate_hz=sample_rate_hz,
            phase_band_rad=float(phase_band_rad),
            min_lock_dwell_s=float(min_lock_dwell_s),
            max_bad_gap_s=float(max_bad_gap_s),
            coh_min=coh_min,
            require_phase_lock_stage=bool(require_phase_lock_stage),
        ),
        evaluate_single_mode(
            rows=sham_rows,
            sample_rate_hz=sample_rate_hz,
            phase_band_rad=float(phase_band_rad),
            min_lock_dwell_s=float(min_lock_dwell_s),
            max_bad_gap_s=float(max_bad_gap_s),
            coh_min=coh_min,
            require_phase_lock_stage=bool(require_phase_lock_stage),
        ),
    )


def build_trial_grid(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    sweep_axes: list[tuple[str, list[Any]]] = [
        ("Kf_pullin_hz_per_hz", parse_float_list(args.Kf_pullin_hz_per_hz_values, args.Kf_pullin_hz_per_hz)),
        ("pullin_switch_hz", parse_float_list(args.pullin_switch_hz_values, args.pullin_switch_hz)),
        ("pullin_enter_dwell_updates", parse_int_list(args.pullin_enter_dwell_updates_values, args.pullin_enter_dwell_updates)),
        ("phase_hold_enter_err_rad", parse_float_list(args.phase_hold_enter_err_rad_values, args.phase_hold_enter_err_rad)),
        ("phase_hold_exit_err_rad", parse_float_list(args.phase_hold_exit_err_rad_values, args.phase_hold_exit_err_rad)),
        ("phase_accepted_raw_enter_min_frac", parse_float_list(args.phase_accepted_raw_enter_min_frac_values, args.phase_accepted_raw_enter_min_frac)),
        ("phase_accepted_raw_exit_min_frac", parse_float_list(args.phase_accepted_raw_exit_min_frac_values, args.phase_accepted_raw_exit_min_frac)),
        ("phase_hold_kp_hz_per_rad", parse_float_list(args.phase_hold_kp_values, args.phase_hold_kp_hz_per_rad)),
        ("phase_hold_ki_hz_per_rad_s", parse_float_list(args.phase_hold_ki_values, args.phase_hold_ki_hz_per_rad_s)),
        ("phase_hold_kd_hz_per_hz", parse_float_list(args.phase_hold_kd_values, args.phase_hold_kd_hz_per_hz)),
        ("phase_hold_max_u_hz", parse_float_list(args.phase_hold_max_u_values, args.phase_hold_max_u_hz)),
        ("phase_reference_ramp_min_scale", parse_float_list(args.phase_reference_ramp_min_scale_values, args.phase_reference_ramp_min_scale)),
        ("phase_reference_ramp_accepted_updates", parse_int_list(args.phase_reference_ramp_accepted_updates_values, args.phase_reference_ramp_accepted_updates)),
        ("phase_accepted_raw_enter_window_updates", parse_int_list(args.phase_accepted_raw_enter_window_updates_values, args.phase_accepted_raw_enter_window_updates)),
        ("phase_accepted_raw_exit_window_updates", parse_int_list(args.phase_accepted_raw_exit_window_updates_values, args.phase_accepted_raw_exit_window_updates)),
    ]

    names = [name for name, _ in sweep_axes]
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*(vals for _, vals in sweep_axes)):
        combos.append({name: value for name, value in zip(names, values)})
    return names, combos


def parameter_summary(param_names: list[str], params: dict[str, Any]) -> str:
    parts = [f"{name}={params[name]}" for name in param_names]
    return ", ".join(parts)


def aggregate_results(
    *,
    trial_rows: list[dict[str, Any]],
    param_names: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in trial_rows:
        grouped.setdefault(str(row["trial_id"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for trial_id, rows in grouped.items():
        params = {name: rows[0][name] for name in param_names}
        phase_vals = [float_or_nan(r["selected_steady_state_median_abs_phase_err_primary_rad"]) for r in rows]
        finite_phase_metric_fraction = float(
            np.mean(np.isfinite(np.asarray(phase_vals, dtype=np.float64)))
        )
        out = {
            "trial_id": str(trial_id),
            **params,
            "parameter_summary": parameter_summary(param_names, params),
            "seed_count": int(len(rows)),
            "finite_selected_phase_metric_fraction": float(finite_phase_metric_fraction),
            "selected_eval_name": ",".join(sorted({str(r["selected_eval_name"]) for r in rows})),
            "selected_primary_phase_metric": ",".join(sorted({str(r["selected_primary_phase_metric"]) for r in rows})),
            "mean_selected_steady_state_median_abs_phase_err_primary_rad": finite_mean(
                [float_or_nan(r["selected_steady_state_median_abs_phase_err_primary_rad"]) for r in rows]
            ),
            "mean_selected_full_run_primary_err_proxy_rad": finite_mean(
                [float_or_nan(r["selected_full_run_primary_err_proxy_rad"]) for r in rows]
            ),
            "max_selected_steady_state_median_abs_phase_err_primary_rad": finite_max(
                [float_or_nan(r["selected_steady_state_median_abs_phase_err_primary_rad"]) for r in rows]
            ),
            "mean_selected_steady_state_primary_phase_band_fraction": finite_mean(
                [float_or_nan(r["selected_steady_state_primary_phase_band_fraction"]) for r in rows]
            ),
            "mean_selected_full_run_primary_phase_band_fraction": finite_mean(
                [float_or_nan(r["selected_full_run_primary_phase_band_fraction"]) for r in rows]
            ),
            "mean_selected_best_good_dwell_s": finite_mean(
                [float_or_nan(r["selected_best_good_dwell_s"]) for r in rows]
            ),
            "mean_pid_stage_switches": finite_mean(
                [float_or_nan(r["pid_stage_switches"]) for r in rows]
            ),
            "mean_pid_phase_mode_fraction": finite_mean(
                [float_or_nan(r["pid_phase_mode_fraction"]) for r in rows]
            ),
            "selected_locked_fraction": bool_fraction(
                [bool(int(r["selected_locked"])) for r in rows]
            ),
            "stage_locked_fraction": bool_fraction(
                [bool(int(r["stage_locked"])) for r in rows]
            ),
            "phase_only_locked_fraction": bool_fraction(
                [bool(int(r["phase_only_locked"])) for r in rows]
            ),
            "mean_selected_primary_phase_err_improvement_pct": finite_mean(
                [float_or_nan(r["selected_primary_phase_err_improvement_pct"]) for r in rows]
            ),
            "mean_selected_primary_phase_band_improvement_abs": finite_mean(
                [float_or_nan(r["selected_primary_phase_band_improvement_abs"]) for r in rows]
            ),
            "mean_selected_sham_phase_err_gap_rad": finite_mean(
                [float_or_nan(r["selected_sham_phase_err_gap_rad"]) for r in rows]
            ),
            "mean_selected_sham_phase_band_gap": finite_mean(
                [float_or_nan(r["selected_sham_phase_band_gap"]) for r in rows]
            ),
        }
        aggregate_rows.append(out)

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        err = float_or_nan(row["mean_selected_steady_state_median_abs_phase_err_primary_rad"])
        err_fallback = float_or_nan(row["mean_selected_full_run_primary_err_proxy_rad"])
        band = float_or_nan(row["mean_selected_steady_state_primary_phase_band_fraction"])
        band_fallback = float_or_nan(row["mean_selected_full_run_primary_phase_band_fraction"])
        dwell = float_or_nan(row["mean_selected_best_good_dwell_s"])
        sham_gap = float_or_nan(row["mean_selected_sham_phase_err_gap_rad"])
        improvement = float_or_nan(row["mean_selected_primary_phase_err_improvement_pct"])
        phase_mode_fraction = float_or_nan(row["mean_pid_phase_mode_fraction"])
        effective_err = err if np.isfinite(err) else err_fallback
        effective_band = band if np.isfinite(band) else band_fallback
        return (
            -float_or_nan(row["finite_selected_phase_metric_fraction"]),
            effective_err if np.isfinite(effective_err) else float("inf"),
            -(effective_band if np.isfinite(effective_band) else -1.0),
            -(dwell if np.isfinite(dwell) else -1.0),
            -(phase_mode_fraction if np.isfinite(phase_mode_fraction) else -1.0),
            -(sham_gap if np.isfinite(sham_gap) else -1e9),
            -(improvement if np.isfinite(improvement) else -1e9),
        )

    aggregate_rows.sort(key=sort_key)
    return aggregate_rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep synthetic phase thresholds and gains, ranking trials by primary phase error.",
        parents=[sim.build_arg_parser()],
        conflict_handler="resolve",
    )
    p.set_defaults(label="phase_tuning_sweep", skip_postprocess=True)
    p.add_argument("--seed-values", type=str, default=None, help="Comma-separated simulation seeds. Defaults to --sim-seed.")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-trials", type=int, default=None)
    p.add_argument("--Kf-pullin-hz-per-hz-values", type=str, default=None)
    p.add_argument("--pullin-switch-hz-values", type=str, default=None)
    p.add_argument("--pullin-enter-dwell-updates-values", type=str, default=None)
    p.add_argument("--phase-hold-enter-err-rad-values", type=str, default=None)
    p.add_argument("--phase-hold-exit-err-rad-values", type=str, default=None)
    p.add_argument("--phase-accepted-raw-enter-min-frac-values", type=str, default=None)
    p.add_argument("--phase-accepted-raw-exit-min-frac-values", type=str, default=None)
    p.add_argument("--phase-hold-kp-values", type=str, default=None)
    p.add_argument("--phase-hold-ki-values", type=str, default=None)
    p.add_argument("--phase-hold-kd-values", type=str, default=None)
    p.add_argument("--phase-hold-max-u-values", type=str, default=None)
    p.add_argument("--phase-reference-ramp-min-scale-values", type=str, default=None)
    p.add_argument("--phase-reference-ramp-accepted-updates-values", type=str, default=None)
    p.add_argument("--phase-accepted-raw-enter-window-updates-values", type=str, default=None)
    p.add_argument("--phase-accepted-raw-exit-window-updates-values", type=str, default=None)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    param_names, trial_param_sets = build_trial_grid(args)
    if args.max_trials is not None:
        trial_param_sets = trial_param_sets[: max(0, int(args.max_trials))]
    seed_values = parse_int_list(args.seed_values, int(args.sim_seed))

    sweep_id = f"{timestamp_for_filename()}_{args.label}"
    sweep_dir = Path(args.artifact_root) / sweep_id
    sim.ensure_parent_dir(sweep_dir / "results" / "placeholder.txt")

    write_json(
        sweep_dir / "sweep_config.json",
        {
            "base_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "param_names": param_names,
            "trial_param_sets": trial_param_sets,
            "seed_values": seed_values,
        },
    )

    per_seed_rows: list[dict[str, Any]] = []
    session_index: list[dict[str, Any]] = []

    for trial_idx, params in enumerate(trial_param_sets, start=1):
        for seed in seed_values:
            trial_id = f"trial_{trial_idx:03d}"
            session_dir = sweep_dir / "trials" / f"{trial_id}_seed{int(seed)}"
            print(
                f"Running {trial_id} seed={seed} | {parameter_summary(param_names, params)}",
                flush=True,
            )
            trial_args = argparse.Namespace(**vars(args))
            for name, value in params.items():
                setattr(trial_args, name, value)
            trial_args.sim_seed = int(seed)
            trial_args.notes = f"{args.notes} | {parameter_summary(param_names, params)} | seed={seed}"

            try:
                sim_cfg = sim.simulation_config_for_mode(trial_args, mode="monitor")
                write_json(session_dir / "simulation_config.json", asdict(sim_cfg))
                write_json(
                    session_dir / "trial_parameters.json",
                    {
                        "trial_id": trial_id,
                        "seed": int(seed),
                        "parameters": params,
                    },
                )

                monitor_summary, _monitor_truth = sim.run_mode_once(
                    trial_args,
                    session_dir=session_dir,
                    mode="monitor",
                    artifact_mode="monitor",
                    observer_fixed_offset_hz=None,
                )
                pid_observer_offset = float(monitor_summary.get("tone_offset_hz", float("nan")))
                pid_summary, _pid_truth = sim.run_mode_once(
                    trial_args,
                    session_dir=session_dir,
                    mode="pid",
                    artifact_mode="pid",
                    observer_fixed_offset_hz=pid_observer_offset if np.isfinite(pid_observer_offset) else None,
                )
                sham_summary = None
                if not bool(trial_args.skip_sham):
                    sham_summary, _sham_truth = sim.run_mode_once(
                        trial_args,
                        session_dir=session_dir,
                        mode="pid",
                        artifact_mode="sham",
                        observer_fixed_offset_hz=pid_observer_offset if np.isfinite(pid_observer_offset) else None,
                        sham_control_mode="random_sign",
                    )

                stage_eval = sim.run_phase_evaluation(
                    run_dir=session_dir,
                    nsamp=int(trial_args.nsamp),
                    phase_band_rad=float(trial_args.phase_eval_phase_band_rad),
                    min_lock_dwell_s=float(trial_args.phase_eval_min_lock_dwell_s),
                    max_bad_gap_s=float(trial_args.phase_eval_max_bad_gap_s),
                    coh_min=float(trial_args.phase_eval_coh_min) if trial_args.phase_eval_coh_min is not None else None,
                    require_quality_ok=True,
                    require_phase_lock_stage=True,
                    json_name="phase_lock_evaluation.json",
                    report_name="phase_lock_evaluation.md",
                )
                phase_only_eval = sim.run_phase_evaluation(
                    run_dir=session_dir,
                    nsamp=int(trial_args.nsamp),
                    phase_band_rad=float(trial_args.phase_eval_phase_band_rad),
                    min_lock_dwell_s=float(trial_args.phase_eval_min_lock_dwell_s),
                    max_bad_gap_s=float(trial_args.phase_eval_max_bad_gap_s),
                    coh_min=float(trial_args.phase_eval_coh_min) if trial_args.phase_eval_coh_min is not None else None,
                    require_quality_ok=True,
                    require_phase_lock_stage=False,
                    json_name="phase_lock_evaluation_phase_only.json",
                    report_name="phase_lock_evaluation_phase_only.md",
                )
                selected_eval_name, selected_eval = choose_eval_output(stage_eval, phase_only_eval)

                selected_pid = selected_eval["pid_phase_metrics"]
                selected_comparison = selected_eval["comparison"]
                selected_primary_metric = str(selected_pid.get("primary_phase_metric_name"))
                if selected_primary_metric == "accepted_raw":
                    full_run_primary_err_proxy_rad = float_or_nan(
                        pid_summary.get("mean_abs_phase_err_accepted_raw_rad")
                    )
                else:
                    full_run_primary_err_proxy_rad = float_or_nan(
                        pid_summary.get("mean_abs_phase_err_rad")
                    )
                sham_phase_err_gap = float("nan")
                sham_phase_band_gap = float("nan")
                if sham_summary is not None:
                    if selected_eval_name == "stage_required":
                        _sham_monitor_metrics, sham_metrics = sham_metrics_for_require_stage(
                            session_dir=session_dir,
                            nsamp=int(trial_args.nsamp),
                            phase_band_rad=float(trial_args.phase_eval_phase_band_rad),
                            min_lock_dwell_s=float(trial_args.phase_eval_min_lock_dwell_s),
                            max_bad_gap_s=float(trial_args.phase_eval_max_bad_gap_s),
                            coh_min=float(trial_args.phase_eval_coh_min) if trial_args.phase_eval_coh_min is not None else None,
                            require_phase_lock_stage=True,
                        )
                    else:
                        _sham_monitor_metrics, sham_metrics = sham_metrics_for_require_stage(
                            session_dir=session_dir,
                            nsamp=int(trial_args.nsamp),
                            phase_band_rad=float(trial_args.phase_eval_phase_band_rad),
                            min_lock_dwell_s=float(trial_args.phase_eval_min_lock_dwell_s),
                            max_bad_gap_s=float(trial_args.phase_eval_max_bad_gap_s),
                            coh_min=float(trial_args.phase_eval_coh_min) if trial_args.phase_eval_coh_min is not None else None,
                            require_phase_lock_stage=False,
                        )
                    pid_err = float_or_nan(selected_pid.get("steady_state_median_abs_phase_err_primary_rad"))
                    sham_err = float_or_nan(sham_metrics.steady_state_median_abs_phase_err_primary_rad)
                    if np.isfinite(pid_err) and np.isfinite(sham_err):
                        sham_phase_err_gap = float(sham_err - pid_err)
                    pid_band = float_or_nan(selected_pid.get("steady_state_primary_phase_band_fraction"))
                    sham_band = float_or_nan(sham_metrics.steady_state_primary_phase_band_fraction)
                    if np.isfinite(pid_band) and np.isfinite(sham_band):
                        sham_phase_band_gap = float(pid_band - sham_band)

                row = {
                    "trial_id": str(trial_id),
                    "seed": int(seed),
                    "run_dir": str(session_dir),
                    "selected_eval_name": str(selected_eval_name),
                    "selected_primary_phase_metric": str(selected_primary_metric),
                    "selected_locked": int(bool(selected_pid.get("locked"))),
                    "stage_locked": int(bool(stage_eval["pid_phase_metrics"].get("locked"))),
                    "phase_only_locked": int(bool(phase_only_eval["pid_phase_metrics"].get("locked"))),
                    "selected_steady_state_median_abs_phase_err_primary_rad": float_or_nan(
                        selected_pid.get("steady_state_median_abs_phase_err_primary_rad")
                    ),
                    "selected_full_run_primary_err_proxy_rad": float_or_nan(
                        full_run_primary_err_proxy_rad
                    ),
                    "selected_steady_state_primary_phase_band_fraction": float_or_nan(
                        selected_pid.get("steady_state_primary_phase_band_fraction")
                    ),
                    "selected_full_run_primary_phase_band_fraction": float_or_nan(
                        selected_pid.get("full_run_primary_phase_band_fraction")
                    ),
                    "selected_best_good_dwell_s": float_or_nan(
                        selected_pid.get("best_good_dwell_s")
                    ),
                    "selected_primary_phase_err_improvement_pct": float_or_nan(
                        selected_comparison.get("primary_phase_err_improvement_pct")
                    ),
                    "selected_primary_phase_band_improvement_abs": float_or_nan(
                        selected_comparison.get("primary_phase_band_improvement_abs")
                    ),
                    "selected_sham_phase_err_gap_rad": float_or_nan(sham_phase_err_gap),
                    "selected_sham_phase_band_gap": float_or_nan(sham_phase_band_gap),
                    "stage_required_steady_state_median_abs_phase_err_primary_rad": float_or_nan(
                        stage_eval["pid_phase_metrics"].get("steady_state_median_abs_phase_err_primary_rad")
                    ),
                    "phase_only_steady_state_median_abs_phase_err_primary_rad": float_or_nan(
                        phase_only_eval["pid_phase_metrics"].get("steady_state_median_abs_phase_err_primary_rad")
                    ),
                    "pid_stage_switches": int(pid_summary.get("stage_switches", 0)),
                    "pid_phase_mode_fraction": float_or_nan(pid_summary.get("phase_mode_fraction")),
                }
                row.update(params)
                per_seed_rows.append(row)
                session_index.append(
                    {
                        "trial_id": str(trial_id),
                        "seed": int(seed),
                        "run_dir": str(session_dir),
                    }
                )
            except Exception as exc:
                error_row = {
                    "trial_id": str(trial_id),
                    "seed": int(seed),
                    "run_dir": str(session_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                error_row.update(params)
                per_seed_rows.append(error_row)
                print(f"  failed: {error_row['error']}", flush=True)

    aggregate_rows = aggregate_results(trial_rows=[r for r in per_seed_rows if "error" not in r], param_names=param_names)
    results_dir = sweep_dir / "results"
    write_json(results_dir / "session_index.json", session_index)
    write_json(results_dir / "trial_results.json", per_seed_rows)
    write_csv_rows(results_dir / "trial_results.csv", per_seed_rows)
    write_json(results_dir / "aggregate_results.json", aggregate_rows)
    write_csv_rows(results_dir / "aggregate_results.csv", aggregate_rows)
    (results_dir / "aggregate_results.md").write_text(
        render_markdown(
            sweep_dir=sweep_dir,
            aggregate_rows=aggregate_rows,
            top_k=max(1, int(args.top_k)),
        )
    )

    print(f"\nSweep artifacts written under: {sweep_dir}")
    if aggregate_rows:
        best = aggregate_rows[0]
        print(
            "Best trial:\n"
            f"  {best['trial_id']} | {best['parameter_summary']}\n"
            f"  mean steady-state median |primary phase error| = "
            f"{best['mean_selected_steady_state_median_abs_phase_err_primary_rad']:.6f} rad\n"
            f"  mean primary phase-band fraction = {best['mean_selected_steady_state_primary_phase_band_fraction']:.6f}\n"
            f"  mean best good dwell = {best['mean_selected_best_good_dwell_s']:.6f} s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
