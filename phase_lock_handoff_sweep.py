from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
NOTEBOOK_BASED = ROOT / "NotebookBased"


@dataclass(frozen=True)
class LockEvalConfig:
    phase_band_rad: float = 1.0
    min_lock_dwell_s: float = 2.0
    max_bad_gap_s: float = 0.05
    coh_min: float = 0.025
    require_quality_ok: bool = True
    require_phase_lock_stage: bool = True


@dataclass(frozen=True)
class StudyConfig:
    controller_module: str = "notebook_based_hybrid_pullin_notebook_phase_dpll"
    label: str = "phase_lock_handoff_sweep"
    artifact_root: Path = NOTEBOOK_BASED / "runs"
    nsamp: int = 32768
    run_seconds: float = 6.0
    sample_rate: float = 2.4e6
    base_offset_hz: float = -30688.4765625
    phase_grid_rad: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50)
    residual_grid_hz: tuple[float, ...] = (0.0, 10.0, 25.0, 50.0, 75.0, 100.0)
    residual_sweep_phase_rad: float = 0.10
    control_measurement_source: str = "observer"
    phase_enter_condition: str = "full"
    phase_exit_condition: str = "full"
    phase_min_hold_updates: int = 0
    pullin_switch_hz: float = 100.0
    pullin_exit_hz: float = 300.0
    pullin_enter_dwell_updates: int = 3
    noise_std: float = 0.0
    phase_noise_std_rad: float = 0.0
    eval_cfg: LockEvalConfig = field(default_factory=LockEvalConfig)


def timestamp_for_filename() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any] | list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, text: str) -> None:
    ensure_parent_dir(path)
    path.write_text(text)


def float_list_to_cli(values: tuple[float, ...]) -> str:
    return ",".join(f"{float(v):g}" for v in values)


def parse_csv_float_list(raw: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(raw).split(",")]
    return tuple(float(part) for part in parts if part)


def slug_value(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    text = text.replace("-", "m").replace(".", "p")
    return text or "0"


def load_notebookbased_modules(controller_module: str) -> tuple[Any, Any]:
    if str(NOTEBOOK_BASED) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_BASED))

    import run_simulated_open_closed as harness  # type: ignore
    import evaluate_saved_phase_run as phase_eval  # type: ignore

    harness.controller = importlib.import_module(str(controller_module))
    wrapped_marker = "_phase_handoff_bg_disabled"
    if not getattr(harness.build_controller_namespace, wrapped_marker, False):
        original = harness.build_controller_namespace

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            ns = original(*args, **kwargs)
            ns.log_background_process = False
            return ns

        setattr(wrapped, wrapped_marker, True)
        harness.build_controller_namespace = wrapped
    return harness, phase_eval


def build_base_args(harness: Any, cfg: StudyConfig) -> argparse.Namespace:
    parser = harness.build_arg_parser()
    args = parser.parse_args(
        [
            "--controller-module",
            str(cfg.controller_module),
            "--label",
            str(cfg.label),
            "--nsamp",
            str(cfg.nsamp),
            "--run-seconds",
            str(cfg.run_seconds),
            "--sample-rate",
            str(cfg.sample_rate),
            "--control-measurement-source",
            str(cfg.control_measurement_source),
            "--sim-step-hz",
            "0",
            "--sim-step-at-s",
            "0",
            "--sim-noise-std",
            str(cfg.noise_std),
            "--sim-phase-noise-std-rad",
            str(cfg.phase_noise_std_rad),
            "--sim-random-walk-hz-per-sqrt-s",
            "0",
            "--sim-drift-rate-hz-per-s",
            "0",
            "--skip-postprocess",
        ]
    )
    args.target_offset_hz = float(cfg.base_offset_hz)
    args.phase_enter_condition = str(cfg.phase_enter_condition)
    args.phase_exit_condition = str(cfg.phase_exit_condition)
    args.phase_min_hold_updates = int(cfg.phase_min_hold_updates)
    args.pullin_switch_hz = float(cfg.pullin_switch_hz)
    args.pullin_exit_hz = float(cfg.pullin_exit_hz)
    args.pullin_enter_dwell_updates = int(cfg.pullin_enter_dwell_updates)
    return args


def run_pid_case(
    *,
    harness: Any,
    args: argparse.Namespace,
    case_dir: Path,
    observer_fixed_offset_hz: float,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    pid_dir = case_dir / "pid" / f"nsamp_{int(args.nsamp)}"
    csv_path = pid_dir / f"pid_nsamp{int(args.nsamp)}.csv"
    truth_csv_path = pid_dir / f"pid_nsamp{int(args.nsamp)}_truth.csv"
    summary_json_path = pid_dir / f"pid_nsamp{int(args.nsamp)}_summary.json"
    summary_csv_path = pid_dir / f"pid_nsamp{int(args.nsamp)}_summary.csv"

    sim_cfg = harness.simulation_config_for_mode(args, mode="pid")
    ns = harness.build_controller_namespace(
        args,
        mode="pid",
        csv_path=csv_path,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        observer_fixed_offset_hz=float(observer_fixed_offset_hz),
    )

    with harness.patched_controller_hardware(sim_cfg) as hw:
        summary = asyncio.run(harness.controller.run_one(ns, int(args.nsamp), csv_path))
        truth_rows = [] if hw.model is None else list(hw.model.history)

    harness.write_summary(summary, summary_json_path, summary_csv_path)
    harness.write_csv_rows(truth_csv_path, truth_rows)
    return csv_path, summary_json_path, summary, truth_rows


def nanmedian_abs(arr: np.ndarray) -> float:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float("nan")
    return float(np.nanmedian(np.abs(arr[finite])))


def last_finite_abs(arr: np.ndarray) -> float:
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if finite_idx.size == 0:
        return float("nan")
    return float(abs(arr[int(finite_idx[-1])]))


def evaluate_case(
    *,
    phase_eval: Any,
    csv_path: Path,
    summary: dict[str, Any],
    cfg: StudyConfig,
) -> dict[str, Any]:
    rows = phase_eval.load_csv_rows(csv_path)
    metrics = phase_eval.detect_settled_metrics(
        rows=rows,
        sample_rate_hz=float(summary.get("sample_rate_hz", cfg.sample_rate)),
        phase_band_rad=float(cfg.eval_cfg.phase_band_rad),
        min_lock_dwell_s=float(cfg.eval_cfg.min_lock_dwell_s),
        max_bad_gap_s=float(cfg.eval_cfg.max_bad_gap_s),
        require_quality_ok=bool(cfg.eval_cfg.require_quality_ok),
        coh_min=float(cfg.eval_cfg.coh_min),
        require_phase_lock_stage=bool(cfg.eval_cfg.require_phase_lock_stage),
    )

    phase_err_filt = phase_eval.first_finite_series(rows, "phase_err_filt_rad", "control_phase_err_filt_rad")
    pullin_error_lpf = phase_eval.first_finite_series(rows, "pullin_error_lpf_hz")
    stage_mask = phase_eval.to_phase_stage_mask(rows)

    entered_phase_lock = bool(np.isfinite(metrics.first_phase_lock_stage_s))
    return {
        "entered_phase_lock": entered_phase_lock,
        "locked": bool(metrics.locked),
        "lock_reason": str(metrics.lock_reason),
        "first_phase_lock_stage_s": float(metrics.first_phase_lock_stage_s),
        "best_good_dwell_s": float(metrics.best_good_dwell_s),
        "steady_state_mean_abs_phase_err_filt_rad": float(metrics.steady_state_mean_abs_phase_err_filt_rad),
        "steady_state_median_abs_phase_err_filt_rad": float(metrics.steady_state_median_abs_phase_err_filt_rad),
        "full_run_phase_stage_fraction": float(metrics.full_run_phase_stage_fraction),
        "full_run_phase_band_fraction": float(metrics.full_run_phase_band_fraction),
        "summary_mean_abs_phase_err_rad": float(summary.get("mean_abs_phase_err_rad", float("nan"))),
        "summary_median_coherence": float(summary.get("median_coherence", float("nan"))),
        "summary_quality_fraction": float(summary.get("quality_fraction", float("nan"))),
        "median_abs_phase_err_filt_rad": nanmedian_abs(np.asarray(phase_err_filt, dtype=np.float64)),
        "median_abs_pullin_error_lpf_hz": nanmedian_abs(np.asarray(pullin_error_lpf, dtype=np.float64)),
        "final_abs_pullin_error_lpf_hz": last_finite_abs(np.asarray(pullin_error_lpf, dtype=np.float64)),
        "phase_lock_updates": int(np.count_nonzero(np.asarray(stage_mask, dtype=bool))),
    }


def run_case(
    *,
    harness: Any,
    phase_eval: Any,
    cfg: StudyConfig,
    case_dir: Path,
    sweep_name: str,
    phase_rad: float,
    residual_hz: float,
) -> dict[str, Any]:
    args = build_base_args(harness, cfg)
    args.notes = (
        f"phase handoff study | sweep={sweep_name} | phase_rad={float(phase_rad):.6f} | "
        f"residual_hz={float(residual_hz):.6f}"
    )
    args.sim_initial_tone_offset_hz = float(cfg.base_offset_hz) + float(residual_hz)
    args.sim_initial_phase_rad = float(phase_rad)

    csv_path, summary_json_path, summary, truth_rows = run_pid_case(
        harness=harness,
        args=args,
        case_dir=case_dir,
        observer_fixed_offset_hz=float(cfg.base_offset_hz),
    )
    eval_result = evaluate_case(
        phase_eval=phase_eval,
        csv_path=csv_path,
        summary=summary,
        cfg=cfg,
    )

    result = {
        "case_label": case_dir.name,
        "sweep_name": str(sweep_name),
        "phase_rad": float(phase_rad),
        "residual_hz": float(residual_hz),
        "case_dir": str(case_dir),
        "csv_path": str(csv_path),
        "summary_json_path": str(summary_json_path),
        "truth_rows": int(len(truth_rows)),
        "run_seconds": float(cfg.run_seconds),
        "observer_fixed_offset_hz": float(cfg.base_offset_hz),
        "initial_tone_offset_hz": float(args.sim_initial_tone_offset_hz),
        **eval_result,
    }
    write_json(case_dir / "case_result.json", result)
    return result


def max_success_value(rows: list[dict[str, Any]], value_key: str, success_key: str) -> float:
    values = [float(row[value_key]) for row in rows if bool(row[success_key])]
    return float(max(values)) if values else float("nan")


def min_failure_value(rows: list[dict[str, Any]], value_key: str, success_key: str) -> float:
    values = [float(row[value_key]) for row in rows if not bool(row[success_key])]
    return float(min(values)) if values else float("nan")


def success_values(rows: list[dict[str, Any]], value_key: str, success_key: str) -> list[float]:
    return [float(row[value_key]) for row in rows if bool(row[success_key])]


def contiguous_success_limit(rows: list[dict[str, Any]], value_key: str, success_key: str) -> float:
    if not rows:
        return float("nan")
    sorted_rows = sorted(rows, key=lambda row: float(row[value_key]))
    limit = float("nan")
    for row in sorted_rows:
        value = float(row[value_key])
        if bool(row[success_key]):
            limit = value
            continue
        break
    return limit


def derive_threshold_summary(
    *,
    cfg: StudyConfig,
    phase_results: list[dict[str, Any]],
    residual_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "phase_sweep_zero_residual_hz": {
            "entered_success_values_rad": success_values(phase_results, "phase_rad", "entered_phase_lock"),
            "locked_success_values_rad": success_values(phase_results, "phase_rad", "locked"),
            "contiguous_enter_limit_rad": contiguous_success_limit(phase_results, "phase_rad", "entered_phase_lock"),
            "contiguous_lock_limit_rad": contiguous_success_limit(phase_results, "phase_rad", "locked"),
            "max_phase_rad_entered": max_success_value(phase_results, "phase_rad", "entered_phase_lock"),
            "max_phase_rad_locked": max_success_value(phase_results, "phase_rad", "locked"),
            "min_phase_rad_failed_to_enter": min_failure_value(phase_results, "phase_rad", "entered_phase_lock"),
            "min_phase_rad_failed_to_lock": min_failure_value(phase_results, "phase_rad", "locked"),
        },
        f"residual_sweep_phase_{float(cfg.residual_sweep_phase_rad):g}_rad": {
            "entered_success_values_hz": success_values(residual_results, "residual_hz", "entered_phase_lock"),
            "locked_success_values_hz": success_values(residual_results, "residual_hz", "locked"),
            "contiguous_enter_limit_hz": contiguous_success_limit(residual_results, "residual_hz", "entered_phase_lock"),
            "contiguous_lock_limit_hz": contiguous_success_limit(residual_results, "residual_hz", "locked"),
            "max_residual_hz_entered": max_success_value(residual_results, "residual_hz", "entered_phase_lock"),
            "max_residual_hz_locked": max_success_value(residual_results, "residual_hz", "locked"),
            "min_residual_hz_failed_to_enter": min_failure_value(residual_results, "residual_hz", "entered_phase_lock"),
            "min_residual_hz_failed_to_lock": min_failure_value(residual_results, "residual_hz", "locked"),
        },
    }


def build_report_text(
    *,
    cfg: StudyConfig,
    study_dir: Path,
    phase_results: list[dict[str, Any]],
    residual_results: list[dict[str, Any]],
    threshold_summary: dict[str, Any],
) -> str:
    phase_thresholds = threshold_summary["phase_sweep_zero_residual_hz"]
    residual_key = f"residual_sweep_phase_{float(cfg.residual_sweep_phase_rad):g}_rad"
    residual_thresholds = threshold_summary[residual_key]

    lines = [
        "# Phase Lock Handoff Study",
        "",
        f"Study directory: `{study_dir}`",
        "",
        "This study assumes pull-in has already done the heavy lifting.",
        "The observer is pinned to the desired tone offset, then the current hybrid controller is asked to phase-lock from controlled residual phase and residual frequency errors.",
        "",
        "## What This Measures",
        "",
        "- `entered_phase_lock`: the controller handed off from pull-in into `phase_lock` at least once",
        "- `locked`: the saved PID trace met the settled-window lock definition",
        f"- Lock definition: `|phase_err_filt| <= {cfg.eval_cfg.phase_band_rad}` rad for at least `{cfg.eval_cfg.min_lock_dwell_s}` s, with coherence >= `{cfg.eval_cfg.coh_min}` and `phase_lock` stage required",
        "",
        "## Key Thresholds",
        "",
        f"- Phase sweep is monotonic in this run: sustained lock holds through `{phase_thresholds['contiguous_lock_limit_rad']}` rad and fails by `{phase_thresholds['min_phase_rad_failed_to_lock']}` rad",
        f"- Residual sweep is not monotonic in this run: handoff succeeds at `{residual_thresholds['entered_success_values_hz']}` Hz but only sustains lock at `{residual_thresholds['locked_success_values_hz']}` Hz",
        f"- The guaranteed contiguous residual range from zero is `<= {residual_thresholds['contiguous_enter_limit_hz']}` Hz for handoff and `<= {residual_thresholds['contiguous_lock_limit_hz']}` Hz for sustained lock",
        "",
        "## Phase Sweep",
        "",
        "| initial phase (rad) | entered | locked | best dwell (s) | mean |phase err| (rad) | median coh |",
        "| ---: | :---: | :---: | ---: | ---: | ---: |",
    ]
    for row in phase_results:
        lines.append(
            f"| {row['phase_rad']:.3f} | {bool(row['entered_phase_lock'])} | {bool(row['locked'])} | "
            f"{float(row['best_good_dwell_s']):.3f} | {float(row['summary_mean_abs_phase_err_rad']):.3f} | "
            f"{float(row['summary_median_coherence']):.3f} |"
        )

    lines.extend(
        [
            "",
            f"## Residual Sweep at {cfg.residual_sweep_phase_rad} rad",
            "",
            "| residual error (Hz) | entered | locked | best dwell (s) | mean |phase err| (rad) | median coh |",
            "| ---: | :---: | :---: | ---: | ---: | ---: |",
        ]
    )
    for row in residual_results:
        lines.append(
            f"| {row['residual_hz']:.1f} | {bool(row['entered_phase_lock'])} | {bool(row['locked'])} | "
            f"{float(row['best_good_dwell_s']):.3f} | {float(row['summary_mean_abs_phase_err_rad']):.3f} | "
            f"{float(row['summary_median_coherence']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If a case enters `phase_lock` but does not count as `locked`, then pull-in was good enough for the handoff gate but not good enough for the phase loop to stay captured.",
            "That gap is the practical answer to how clean the pull-in handoff needs to be for this controller.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_results(
    *,
    cfg: StudyConfig,
    phase_results: list[dict[str, Any]],
    residual_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    phase_x = np.asarray([float(row["phase_rad"]) for row in phase_results], dtype=np.float64)
    phase_enter = np.asarray([1.0 if bool(row["entered_phase_lock"]) else 0.0 for row in phase_results], dtype=np.float64)
    phase_lock = np.asarray([1.0 if bool(row["locked"]) else 0.0 for row in phase_results], dtype=np.float64)
    phase_dwell = np.asarray([float(row["best_good_dwell_s"]) for row in phase_results], dtype=np.float64)

    resid_x = np.asarray([float(row["residual_hz"]) for row in residual_results], dtype=np.float64)
    resid_enter = np.asarray([1.0 if bool(row["entered_phase_lock"]) else 0.0 for row in residual_results], dtype=np.float64)
    resid_lock = np.asarray([1.0 if bool(row["locked"]) else 0.0 for row in residual_results], dtype=np.float64)
    resid_dwell = np.asarray([float(row["best_good_dwell_s"]) for row in residual_results], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(phase_x, phase_enter, marker="o", linewidth=2.0, label="entered phase_lock")
    ax.plot(phase_x, phase_lock, marker="s", linewidth=2.0, label="sustained lock")
    ax.set_title("Phase Sweep, Residual = 0 Hz")
    ax.set_xlabel("Initial phase error (rad)")
    ax.set_ylabel("Success")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    ax = axes[1, 0]
    ax.plot(phase_x, phase_dwell, marker="o", linewidth=2.0, color="#b35c00")
    ax.axhline(float(cfg.eval_cfg.min_lock_dwell_s), color="#444444", linestyle="--", linewidth=1.5)
    ax.set_title("Good Dwell vs Initial Phase")
    ax.set_xlabel("Initial phase error (rad)")
    ax.set_ylabel("Best good dwell (s)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(resid_x, resid_enter, marker="o", linewidth=2.0, label="entered phase_lock")
    ax.plot(resid_x, resid_lock, marker="s", linewidth=2.0, label="sustained lock")
    ax.set_title(f"Residual Sweep, Phase = {cfg.residual_sweep_phase_rad:g} rad")
    ax.set_xlabel("Residual frequency error after pull-in (Hz)")
    ax.set_ylabel("Success")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    ax = axes[1, 1]
    ax.plot(resid_x, resid_dwell, marker="o", linewidth=2.0, color="#0f6c73")
    ax.axhline(float(cfg.eval_cfg.min_lock_dwell_s), color="#444444", linestyle="--", linewidth=1.5)
    ax.set_title("Good Dwell vs Residual Frequency Error")
    ax.set_xlabel("Residual frequency error after pull-in (Hz)")
    ax.set_ylabel("Best good dwell (s)")
    ax.grid(True, alpha=0.3)

    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_study(cfg: StudyConfig) -> dict[str, Any]:
    harness, phase_eval = load_notebookbased_modules(cfg.controller_module)
    study_dir = cfg.artifact_root / f"{timestamp_for_filename()}_{cfg.label}"
    study_dir.mkdir(parents=True, exist_ok=True)

    phase_results: list[dict[str, Any]] = []
    for phase_rad in cfg.phase_grid_rad:
        case_dir = study_dir / "phase_sweep" / f"phase_{slug_value(phase_rad)}_resid_{slug_value(0.0)}"
        phase_results.append(
            run_case(
                harness=harness,
                phase_eval=phase_eval,
                cfg=cfg,
                case_dir=case_dir,
                sweep_name="phase_sweep",
                phase_rad=float(phase_rad),
                residual_hz=0.0,
            )
        )

    residual_results: list[dict[str, Any]] = []
    for residual_hz in cfg.residual_grid_hz:
        case_dir = study_dir / "residual_sweep" / (
            f"phase_{slug_value(cfg.residual_sweep_phase_rad)}_resid_{slug_value(residual_hz)}"
        )
        residual_results.append(
            run_case(
                harness=harness,
                phase_eval=phase_eval,
                cfg=cfg,
                case_dir=case_dir,
                sweep_name="residual_sweep",
                phase_rad=float(cfg.residual_sweep_phase_rad),
                residual_hz=float(residual_hz),
            )
        )

    phase_results.sort(key=lambda row: float(row["phase_rad"]))
    residual_results.sort(key=lambda row: float(row["residual_hz"]))
    threshold_summary = derive_threshold_summary(
        cfg=cfg,
        phase_results=phase_results,
        residual_results=residual_results,
    )

    output = {
        "study_dir": str(study_dir),
        "config": {
            **asdict(cfg),
            "artifact_root": str(cfg.artifact_root),
        },
        "phase_results": phase_results,
        "residual_results": residual_results,
        "threshold_summary": threshold_summary,
    }
    write_json(study_dir / "phase_lock_handoff_summary.json", output)
    write_csv_rows(study_dir / "phase_sweep_results.csv", phase_results)
    write_csv_rows(study_dir / "residual_sweep_results.csv", residual_results)
    write_markdown(
        study_dir / "phase_lock_handoff_report.md",
        build_report_text(
            cfg=cfg,
            study_dir=study_dir,
            phase_results=phase_results,
            residual_results=residual_results,
            threshold_summary=threshold_summary,
        ),
    )
    plot_results(
        cfg=cfg,
        phase_results=phase_results,
        residual_results=residual_results,
        output_path=study_dir / "phase_lock_handoff_summary.png",
    )
    return output


def format_summary(output: dict[str, Any]) -> str:
    cfg_dict = output["config"]
    threshold_summary = output["threshold_summary"]
    phase_thresholds = threshold_summary["phase_sweep_zero_residual_hz"]
    residual_key = f"residual_sweep_phase_{float(cfg_dict['residual_sweep_phase_rad']):g}_rad"
    residual_thresholds = threshold_summary[residual_key]
    return (
        "Phase-lock handoff study\n"
        f"study_dir: {output['study_dir']}\n"
        f"phase sweep @ residual 0 Hz: sustained lock through {phase_thresholds['contiguous_lock_limit_rad']} rad, "
        f"fails by {phase_thresholds['min_phase_rad_failed_to_lock']} rad\n"
        f"residual sweep @ phase {cfg_dict['residual_sweep_phase_rad']} rad: handoff successes at "
        f"{residual_thresholds['entered_success_values_hz']} Hz, lock successes at {residual_thresholds['locked_success_values_hz']} Hz"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantify hybrid controller phase-lock handoff and sustain boundaries under idealized pull-in."
    )
    parser.add_argument("--label", type=str, default="phase_lock_handoff_sweep")
    parser.add_argument("--artifact-root", type=Path, default=NOTEBOOK_BASED / "runs")
    parser.add_argument("--controller-module", type=str, default="notebook_based_hybrid_pullin_notebook_phase_dpll")
    parser.add_argument("--nsamp", type=int, default=32768)
    parser.add_argument("--run-seconds", type=float, default=6.0)
    parser.add_argument("--sample-rate", type=float, default=2.4e6)
    parser.add_argument("--base-offset-hz", type=float, default=-30688.4765625)
    parser.add_argument(
        "--phase-grid-rad",
        type=str,
        default=float_list_to_cli(StudyConfig().phase_grid_rad),
        help="Comma-separated initial phase errors for the zero-residual sweep.",
    )
    parser.add_argument(
        "--residual-grid-hz",
        type=str,
        default=float_list_to_cli(StudyConfig().residual_grid_hz),
        help="Comma-separated residual frequency errors for the fixed-phase sweep.",
    )
    parser.add_argument("--residual-sweep-phase-rad", type=float, default=0.10)
    parser.add_argument(
        "--phase-enter-condition",
        choices=["full", "pullin_only"],
        default="full",
    )
    parser.add_argument(
        "--phase-exit-condition",
        choices=["full", "residual_only"],
        default="full",
    )
    parser.add_argument("--phase-min-hold-updates", type=int, default=0)
    parser.add_argument("--pullin-switch-hz", type=float, default=100.0)
    parser.add_argument("--pullin-exit-hz", type=float, default=300.0)
    parser.add_argument("--pullin-enter-dwell-updates", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--phase-noise-std-rad", type=float, default=0.0)
    parser.add_argument("--phase-band-rad", type=float, default=1.0)
    parser.add_argument("--min-lock-dwell-s", type=float, default=2.0)
    parser.add_argument("--max-bad-gap-s", type=float, default=0.05)
    parser.add_argument("--coh-min", type=float, default=0.025)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = StudyConfig(
        controller_module=str(args.controller_module),
        label=str(args.label),
        artifact_root=Path(args.artifact_root).resolve(),
        nsamp=int(args.nsamp),
        run_seconds=float(args.run_seconds),
        sample_rate=float(args.sample_rate),
        base_offset_hz=float(args.base_offset_hz),
        phase_grid_rad=parse_csv_float_list(args.phase_grid_rad),
        residual_grid_hz=parse_csv_float_list(args.residual_grid_hz),
        residual_sweep_phase_rad=float(args.residual_sweep_phase_rad),
        phase_enter_condition=str(args.phase_enter_condition),
        phase_exit_condition=str(args.phase_exit_condition),
        phase_min_hold_updates=int(args.phase_min_hold_updates),
        pullin_switch_hz=float(args.pullin_switch_hz),
        pullin_exit_hz=float(args.pullin_exit_hz),
        pullin_enter_dwell_updates=int(args.pullin_enter_dwell_updates),
        noise_std=float(args.noise_std),
        phase_noise_std_rad=float(args.phase_noise_std_rad),
        eval_cfg=LockEvalConfig(
            phase_band_rad=float(args.phase_band_rad),
            min_lock_dwell_s=float(args.min_lock_dwell_s),
            max_bad_gap_s=float(args.max_bad_gap_s),
            coh_min=float(args.coh_min),
        ),
    )
    output = run_study(cfg)
    print(json.dumps(output["threshold_summary"], indent=2, sort_keys=True))
    print()
    print(format_summary(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
