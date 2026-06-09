#!/usr/bin/env python3
"""Run NotebookBased open/closed comparisons against a deterministic synthetic stream."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import importlib
import json
import math
import os
import time
import types
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_saved_phase_run as phase_eval


controller = importlib.import_module("notebook_based_pullin_handoff_30hz_phase_dpll")


BASE_DIR = Path(__file__).resolve().parent


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def timestamp_for_filename() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


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


@dataclass
class SyntheticStreamConfig:
    sample_rate_hz: float
    initial_tone_offset_hz: float
    path_phase_offset_rad: float
    step_hz: float
    step_at_s: float
    drift_rate_hz_per_s: float
    random_walk_hz_per_sqrt_s: float
    receiver_drift_rate_hz_per_s: float
    receiver_random_walk_hz_per_sqrt_s: float
    amplitude: float
    noise_std: float
    phase_noise_std_rad: float
    iq_scale: float
    plant_hz_per_code: float
    plant_tau_s: float
    dac_center_code: int
    realtime: bool
    seed: int


class SyntheticLoopModel:
    def __init__(self, config: SyntheticStreamConfig):
        self.config = config
        self.rng = np.random.default_rng(int(config.seed))
        self.sample_index = 0
        self.phase_rad = float(config.path_phase_offset_rad)
        self.source_random_walk_hz = 0.0
        self.receiver_random_walk_hz = 0.0
        self.control_correction_hz = 0.0
        self.dac_code_cmd = int(config.dac_center_code)
        self.history: list[dict[str, Any]] = []

    def set_dac_code(self, code: int) -> int:
        self.dac_code_cmd = int(code)
        return int(self.dac_code_cmd)

    def dac_code_to_volts(self, code: int) -> float:
        return controller.code_to_volts(int(code), controller.VDD_DAC_DEFAULT)

    def target_correction_hz(self) -> float:
        delta_codes = float(int(self.dac_code_cmd) - int(self.config.dac_center_code))
        return -float(self.config.plant_hz_per_code) * delta_codes

    def source_offset_hz(self, t_s: float) -> float:
        step_term = float(self.config.step_hz) if float(t_s) >= float(self.config.step_at_s) else 0.0
        return (
            float(self.config.initial_tone_offset_hz)
            + step_term
            + (float(self.config.drift_rate_hz_per_s) * float(t_s))
            + float(self.source_random_walk_hz)
        )

    def receiver_free_offset_hz(self, t_s: float) -> float:
        return (
            (float(self.config.receiver_drift_rate_hz_per_s) * float(t_s))
            + float(self.receiver_random_walk_hz)
        )

    def true_offset_hz(self, t_s: float) -> float:
        return (
            self.source_offset_hz(t_s)
            - self.receiver_free_offset_hz(t_s)
            + float(self.control_correction_hz)
        )

    def generate_iq_bytes(self, nsamp: int) -> bytes:
        fs = float(self.config.sample_rate_hz)
        n = int(nsamp)
        t0_s = float(self.sample_index) / fs
        dt_s = float(n) / fs

        alpha = 1.0
        if float(self.config.plant_tau_s) > 0.0:
            alpha = float(1.0 - math.exp(-dt_s / float(self.config.plant_tau_s)))
        self.control_correction_hz += alpha * (self.target_correction_hz() - self.control_correction_hz)

        if float(self.config.random_walk_hz_per_sqrt_s) > 0.0:
            self.source_random_walk_hz += float(
                self.rng.normal(scale=float(self.config.random_walk_hz_per_sqrt_s) * math.sqrt(dt_s))
            )
        if float(self.config.receiver_random_walk_hz_per_sqrt_s) > 0.0:
            self.receiver_random_walk_hz += float(
                self.rng.normal(scale=float(self.config.receiver_random_walk_hz_per_sqrt_s) * math.sqrt(dt_s))
            )

        sample_idx = np.arange(n, dtype=np.float64)
        sample_t_s = t0_s + (sample_idx / fs)
        step_term = float(self.config.step_hz) * (sample_t_s >= float(self.config.step_at_s))
        source_offset_hz = (
            float(self.config.initial_tone_offset_hz)
            + step_term
            + (float(self.config.drift_rate_hz_per_s) * sample_t_s)
            + float(self.source_random_walk_hz)
        ).astype(np.float64)
        receiver_offset_hz = (
            (float(self.config.receiver_drift_rate_hz_per_s) * sample_t_s)
            + float(self.receiver_random_walk_hz)
            + float(self.control_correction_hz)
        ).astype(np.float64)
        tone_freq_hz = (source_offset_hz - receiver_offset_hz).astype(np.float64)

        phase = self.phase_rad + np.cumsum((2.0 * math.pi * tone_freq_hz) / fs)
        self.phase_rad = controller.wrap_to_pi(float(phase[-1]))
        self.sample_index += n

        sig = float(self.config.amplitude) * np.exp(1j * phase)
        if float(self.config.phase_noise_std_rad) > 0.0:
            sig *= np.exp(
                1j * self.rng.normal(scale=float(self.config.phase_noise_std_rad), size=n)
            )
        if float(self.config.noise_std) > 0.0:
            noise = float(self.config.noise_std) * (
                self.rng.normal(size=n) + 1j * self.rng.normal(size=n)
            ) / math.sqrt(2.0)
            sig = sig + noise

        scale = float(self.config.iq_scale)
        i = np.clip(np.round(np.real(sig) * scale), -127, 127).astype(np.int16)
        q = np.clip(np.round(np.imag(sig) * scale), -127, 127).astype(np.int16)
        raw = np.empty(2 * n, dtype=np.uint8)
        raw[0::2] = np.asarray(i + 128, dtype=np.uint8)
        raw[1::2] = np.asarray(q + 128, dtype=np.uint8)

        mid_t_s = t0_s + (0.5 * dt_s)
        self.history.append(
            {
                "sample_start": int(self.sample_index - n),
                "elapsed_s": float(mid_t_s),
                "path_phase_offset_rad": float(self.config.path_phase_offset_rad),
                "source_offset_hz": float(self.source_offset_hz(mid_t_s)),
                "receiver_free_offset_hz": float(self.receiver_free_offset_hz(mid_t_s)),
                "true_offset_hz": float(self.true_offset_hz(mid_t_s)),
                "control_correction_hz": float(self.control_correction_hz),
                "target_correction_hz": float(self.target_correction_hz()),
                "source_random_walk_hz": float(self.source_random_walk_hz),
                "receiver_random_walk_hz": float(self.receiver_random_walk_hz),
                "dac_code_cmd": int(self.dac_code_cmd),
                "step_active": int(mid_t_s >= float(self.config.step_at_s)),
            }
        )
        return raw.tobytes()


class SyntheticSDR:
    def __init__(self, model: SyntheticLoopModel, realtime: bool):
        self.model = model
        self.realtime = bool(realtime)
        self.stopped = False

    async def stream(self, num_samples_or_bytes: int, format: str = "bytes"):
        if format != "bytes":
            raise ValueError("SyntheticSDR only supports format='bytes'.")
        if int(num_samples_or_bytes) % 2 != 0:
            raise ValueError("Expected even byte count for interleaved I/Q samples.")
        nsamp = int(num_samples_or_bytes) // 2
        while not self.stopped:
            yield self.model.generate_iq_bytes(nsamp)
            if self.realtime:
                await asyncio.sleep(float(nsamp) / float(self.model.config.sample_rate_hz))
            else:
                await asyncio.sleep(0)

    async def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.stopped = True


class SyntheticClockVCXO:
    def __init__(self, model: SyntheticLoopModel):
        self.model = model
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def init_clock_vcxo_28p8mhz(self, apr_ppm: int = 120) -> None:
        _ = int(apr_ppm)

    def set_dac_code(self, code: int, settle_s: float = 0.0) -> int:
        _ = float(settle_s)
        return int(self.model.set_dac_code(int(code)))

    def dac_code_to_volts(self, code: int) -> float:
        return float(self.model.dac_code_to_volts(int(code)))


@dataclass
class PatchedHardwareState:
    model: SyntheticLoopModel | None = None


@contextlib.contextmanager
def patched_controller_hardware(config: SyntheticStreamConfig):
    state = PatchedHardwareState(model=None)
    old_open_sdr = controller.open_sdr
    old_cvh = controller.cvh

    def open_sdr(_args: argparse.Namespace) -> SyntheticSDR:
        state.model = SyntheticLoopModel(config)
        return SyntheticSDR(state.model, realtime=bool(config.realtime))

    def make_clock() -> SyntheticClockVCXO:
        if state.model is None:
            raise RuntimeError("Synthetic clock requested before synthetic SDR initialization.")
        return SyntheticClockVCXO(state.model)

    controller.open_sdr = open_sdr
    controller.cvh = types.SimpleNamespace(ClockVCXO=make_clock)
    try:
        yield state
    finally:
        controller.open_sdr = old_open_sdr
        controller.cvh = old_cvh


def write_summary(summary: dict[str, Any], summary_json_path: Path, summary_csv_path: Path) -> None:
    write_json(summary_json_path, summary)
    controller.write_summary_csv([summary], summary_csv_path)


def mode_artifact_paths(session_dir: Path, mode: str, nsamp: int) -> dict[str, Path]:
    mode_dir = session_dir / mode / f"nsamp_{int(nsamp)}"
    return {
        "mode_dir": mode_dir,
        "csv": mode_dir / f"{mode}_nsamp{int(nsamp)}.csv",
        "truth_csv": mode_dir / f"{mode}_nsamp{int(nsamp)}_truth.csv",
        "summary_json": mode_dir / f"{mode}_nsamp{int(nsamp)}_summary.json",
        "summary_csv": mode_dir / f"{mode}_nsamp{int(nsamp)}_summary.csv",
    }


def summarize_comparison(
    *,
    nsamp: int,
    monitor_summary: dict[str, Any],
    pid_summary: dict[str, Any],
) -> dict[str, Any]:
    monitor_timing = float(monitor_summary.get("primary_timing_drift_ns_per_s", float("nan")))
    pid_timing = float(pid_summary.get("primary_timing_drift_ns_per_s", float("nan")))
    monitor_phase = float(monitor_summary.get("phase_drift_slope_full_hz", float("nan")))
    pid_phase = float(pid_summary.get("phase_drift_slope_full_hz", float("nan")))

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
        "monitor_timing_drift_samples_per_s": float(monitor_summary.get("primary_timing_drift_samples_per_s", float("nan"))),
        "pid_timing_drift_samples_per_s": float(pid_summary.get("primary_timing_drift_samples_per_s", float("nan"))),
        "phase_improvement_pct": float(phase_improvement_pct),
        "timing_improvement_pct": float(timing_improvement_pct),
        "monitor_median_coherence": float(monitor_summary.get("median_coherence", float("nan"))),
        "pid_median_coherence": float(pid_summary.get("median_coherence", float("nan"))),
        "monitor_mean_abs_freq_err_hz": float(monitor_summary.get("mean_abs_freq_err_hz", float("nan"))),
        "pid_mean_abs_freq_err_hz": float(pid_summary.get("mean_abs_freq_err_hz", float("nan"))),
        "monitor_quality_fraction": float(monitor_summary.get("quality_fraction", float("nan"))),
        "pid_quality_fraction": float(pid_summary.get("quality_fraction", float("nan"))),
        "monitor_quality_updates": int(monitor_summary.get("quality_updates", 0)),
        "pid_quality_updates": int(pid_summary.get("quality_updates", 0)),
        "monitor_mean_abs_phase_err_accepted_raw_rad": float(monitor_summary.get("mean_abs_phase_err_accepted_raw_rad", float("nan"))),
        "pid_mean_abs_phase_err_accepted_raw_rad": float(pid_summary.get("mean_abs_phase_err_accepted_raw_rad", float("nan"))),
        "pid_phase_updates_applied": int(pid_summary.get("phase_updates_applied", 0)),
        "pid_pullin_updates_applied": int(pid_summary.get("pullin_updates_applied", 0)),
        "pid_stage_switches": int(pid_summary.get("stage_switches", 0)),
    }


def build_controller_namespace(
    args: argparse.Namespace,
    *,
    mode: str,
    csv_path: Path,
    summary_json_path: Path,
    summary_csv_path: Path,
    observer_fixed_offset_hz: float | None,
    sham_control_mode: str = "off",
) -> argparse.Namespace:
    parser = controller.build_arg_parser()
    desired_updates = max(
        1,
        int(math.ceil(float(args.run_seconds) * float(args.sample_rate) / float(args.nsamp))),
    )
    controller_run_seconds = (
        float(args.run_seconds) if bool(args.sim_realtime) else max(float(args.run_seconds) * 100.0, 1e6)
    )
    argv = [
        "--mode",
        str(mode),
        "--center-freq",
        str(args.center_freq),
        "--sample-rate",
        str(args.sample_rate),
        "--gain",
        str(args.gain),
        "--settle-seconds",
        str(args.settle_seconds),
        "--nsamp",
        str(args.nsamp),
        "--blocks-per-update",
        "1",
        "--async-queue-blocks",
        str(args.async_queue_blocks),
        "--run-seconds",
        str(controller_run_seconds),
        "--max-updates",
        str(desired_updates),
        "--tone-estimate-blocks",
        str(args.tone_estimate_blocks),
        "--dc-exclusion-hz",
        str(args.dc_exclusion_hz),
        "--control-measurement-source",
        str(args.control_measurement_source),
        "--phase-reference-mode",
        str(args.phase_reference_mode),
        "--phase-enter-condition",
        str(getattr(args, "phase_enter_condition", "full")),
        "--phase-exit-condition",
        str(getattr(args, "phase_exit_condition", "full")),
        "--phase-filter-alpha",
        str(args.phase_filter_alpha),
        "--phase-enter-max-err-rad",
        str(args.phase_enter_max_err_rad),
        "--phase-exit-min-err-rad",
        str(getattr(args, "phase_exit_min_err_rad", 1.0)),
        "--phase-frac-gain-codes",
        str(args.phase_frac_gain_codes),
        "--phase-hold-frac-gain-codes",
        str(args.phase_hold_frac_gain_codes),
        "--phase-center-integral-gain-codes-per-rad-s",
        str(args.phase_center_integral_gain_codes_per_rad_s),
        "--phase-center-freq-gain-codes-per-hz-s",
        str(args.phase_center_freq_gain_codes_per_hz_s),
        "--phase-hold-center-freq-gain-codes-per-hz-s",
        str(args.phase_hold_center_freq_gain_codes_per_hz_s),
        "--phase-center-freq-filter-alpha",
        str(args.phase_center_freq_filter_alpha),
        "--phase-center-freq-max-hz",
        str(args.phase_center_freq_max_hz),
        "--phase-center-integral-limit-codes",
        str(args.phase_center_integral_limit_codes),
        "--phase-hold-enter-err-rad",
        str(args.phase_hold_enter_err_rad),
        "--phase-hold-exit-err-rad",
        str(args.phase_hold_exit_err_rad),
        "--phase-hold-enter-dwell-updates",
        str(args.phase_hold_enter_dwell_updates),
        "--phase-hold-exit-dwell-updates",
        str(args.phase_hold_exit_dwell_updates),
        "--phase-accepted-raw-enter-window-updates",
        str(getattr(args, "phase_accepted_raw_enter_window_updates", 12)),
        "--phase-accepted-raw-enter-min-frac",
        str(getattr(args, "phase_accepted_raw_enter_min_frac", 0.75)),
        "--phase-accepted-raw-exit-window-updates",
        str(getattr(args, "phase_accepted_raw_exit_window_updates", 12)),
        "--phase-accepted-raw-exit-min-frac",
        str(getattr(args, "phase_accepted_raw_exit_min_frac", 0.50)),
        "--phase-reference-ramp-accepted-updates",
        str(getattr(args, "phase_reference_ramp_accepted_updates", 24)),
        "--phase-reference-ramp-min-scale",
        str(getattr(args, "phase_reference_ramp_min_scale", 0.25)),
        "--phase-hold-integral-max-err-rad",
        str(args.phase_hold_integral_max_err_rad),
        "--phase-hold-detector-limit-rad",
        str(args.phase_hold_detector_limit_rad),
        "--phase-hold-deadband-rad",
        str(args.phase_hold_deadband_rad),
        "--phase-hold-kp-hz-per-rad",
        str(args.phase_hold_kp_hz_per_rad),
        "--phase-hold-ki-hz-per-rad-s",
        str(args.phase_hold_ki_hz_per_rad_s),
        "--phase-hold-kd-hz-per-hz",
        str(args.phase_hold_kd_hz_per_hz),
        "--phase-hold-rate-filter-alpha",
        str(args.phase_hold_rate_filter_alpha),
        "--phase-hold-rate-max-hz",
        str(args.phase_hold_rate_max_hz),
        "--phase-hold-integral-limit-hz",
        str(args.phase_hold_integral_limit_hz),
        "--phase-hold-max-u-hz",
        str(args.phase_hold_max_u_hz),
        "--max-phase-dac-step-per-update",
        str(args.max_phase_dac_step_per_update),
        "--max-phase-hold-dac-step-per-update",
        str(args.max_phase_hold_dac_step_per_update),
        "--phase-min-hold-updates",
        str(getattr(args, "phase_min_hold_updates", 0)),
        "--drift-window-s",
        str(args.drift_window_s),
        "--slip-max-cycles-per-update",
        str(args.slip_max_cycles_per_update),
        "--sham-control-mode",
        str(sham_control_mode),
        "--csv-out",
        str(csv_path),
        "--summary-out",
        str(summary_json_path),
        "--summary-csv",
        str(summary_csv_path),
        "--notes",
        str(args.notes),
        "--dac-start",
        str(args.sim_dac_center_code),
        "--quiet",
    ]

    if bool(args.phase_predict_through_holds):
        argv.append("--phase-predict-through-holds")

    if observer_fixed_offset_hz is not None:
        argv.extend(["--observer-fixed-offset-hz", str(observer_fixed_offset_hz)])

    if mode == "pid":
        argv.extend(
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
                "--integral-limit-hz",
                str(args.integral_limit_hz),
                "--apr-ppm",
                str(args.apr_ppm),
                "--fine-settle-s",
                str(args.fine_settle_s),
            ]
        )
        if args.apply_start_dac:
            argv.append("--apply-start-dac")
        else:
            argv.append("--no-apply-start-dac")
        if args.skip_clock_init:
            argv.append("--skip-clock-init")

    return parser.parse_args(argv)


def simulation_config_for_mode(args: argparse.Namespace, *, mode: str) -> SyntheticStreamConfig:
    seed_offset = 0 if mode == "monitor" else int(args.pid_seed_offset)
    return SyntheticStreamConfig(
        sample_rate_hz=float(args.sample_rate),
        initial_tone_offset_hz=float(args.sim_initial_tone_offset_hz),
        path_phase_offset_rad=float(
            args.sim_path_phase_offset_rad
            if args.sim_path_phase_offset_rad is not None
            else args.sim_initial_phase_rad
        ),
        step_hz=float(args.sim_step_hz),
        step_at_s=float(args.sim_step_at_s),
        drift_rate_hz_per_s=float(args.sim_drift_rate_hz_per_s),
        random_walk_hz_per_sqrt_s=float(args.sim_random_walk_hz_per_sqrt_s),
        receiver_drift_rate_hz_per_s=float(args.sim_receiver_drift_rate_hz_per_s),
        receiver_random_walk_hz_per_sqrt_s=float(args.sim_receiver_random_walk_hz_per_sqrt_s),
        amplitude=float(args.sim_amplitude),
        noise_std=float(args.sim_noise_std),
        phase_noise_std_rad=float(args.sim_phase_noise_std_rad),
        iq_scale=float(args.sim_iq_scale),
        plant_hz_per_code=float(args.sim_plant_hz_per_code),
        plant_tau_s=float(args.sim_plant_tau_s),
        dac_center_code=int(args.sim_dac_center_code),
        realtime=bool(args.sim_realtime),
        seed=int(args.sim_seed) + seed_offset,
    )


def run_mode_once(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    mode: str,
    artifact_mode: str | None,
    observer_fixed_offset_hz: float | None,
    sham_control_mode: str = "off",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact_mode = str(artifact_mode) if artifact_mode is not None else str(mode)
    paths = mode_artifact_paths(session_dir, artifact_mode, int(args.nsamp))
    sim_cfg = simulation_config_for_mode(args, mode=mode)
    ns = build_controller_namespace(
        args,
        mode=mode,
        csv_path=paths["csv"],
        summary_json_path=paths["summary_json"],
        summary_csv_path=paths["summary_csv"],
        observer_fixed_offset_hz=observer_fixed_offset_hz,
        sham_control_mode=str(sham_control_mode),
    )

    with patched_controller_hardware(sim_cfg) as hw:
        summary = asyncio.run(controller.run_one(ns, int(args.nsamp), paths["csv"]))
        truth_rows = [] if hw.model is None else list(hw.model.history)

    write_summary(summary, paths["summary_json"], paths["summary_csv"])
    write_csv_rows(paths["truth_csv"], truth_rows)
    return summary, truth_rows


def run_phase_evaluation(
    *,
    run_dir: Path,
    nsamp: int,
    phase_band_rad: float,
    min_lock_dwell_s: float,
    max_bad_gap_s: float,
    coh_min: float | None,
    require_quality_ok: bool,
    require_phase_lock_stage: bool,
    json_name: str,
    report_name: str,
) -> dict[str, Any]:
    paths = phase_eval.infer_paths(run_dir, int(nsamp))
    monitor_rows = phase_eval.load_csv_rows(paths["monitor_csv"])
    pid_rows = phase_eval.load_csv_rows(paths["pid_csv"])
    monitor_summary = phase_eval.load_json(paths["monitor_summary"])
    pid_summary = phase_eval.load_json(paths["pid_summary"])
    sample_rate_hz = float(pid_summary.get("sample_rate_hz", monitor_summary.get("sample_rate_hz", 2.4e6)))

    monitor_metrics = phase_eval.detect_settled_metrics(
        rows=monitor_rows,
        sample_rate_hz=sample_rate_hz,
        phase_band_rad=float(phase_band_rad),
        min_lock_dwell_s=float(min_lock_dwell_s),
        max_bad_gap_s=float(max_bad_gap_s),
        require_quality_ok=bool(require_quality_ok),
        coh_min=float(coh_min) if coh_min is not None else None,
        require_phase_lock_stage=bool(require_phase_lock_stage),
    )
    pid_metrics = phase_eval.detect_settled_metrics(
        rows=pid_rows,
        sample_rate_hz=sample_rate_hz,
        phase_band_rad=float(phase_band_rad),
        min_lock_dwell_s=float(min_lock_dwell_s),
        max_bad_gap_s=float(max_bad_gap_s),
        require_quality_ok=bool(require_quality_ok),
        coh_min=float(coh_min) if coh_min is not None else None,
        require_phase_lock_stage=bool(require_phase_lock_stage),
    )
    comparison = {
        "primary_phase_metric": str(pid_metrics.primary_phase_metric_name),
        "primary_phase_err_improvement_pct": phase_eval.improvement_pct(
            float(monitor_metrics.steady_state_mean_abs_phase_err_primary_rad),
            float(pid_metrics.steady_state_mean_abs_phase_err_primary_rad),
        ),
        "primary_phase_band_improvement_abs": (
            float(
                pid_metrics.steady_state_primary_phase_band_fraction
                - monitor_metrics.steady_state_primary_phase_band_fraction
            )
            if (
                np.isfinite(float(pid_metrics.steady_state_primary_phase_band_fraction))
                and np.isfinite(float(monitor_metrics.steady_state_primary_phase_band_fraction))
            )
            else float("nan")
        ),
        "timing_drift_improvement_pct": phase_eval.improvement_pct(
            float(monitor_summary.get("primary_timing_drift_samples_per_s", float("nan"))),
            float(pid_summary.get("primary_timing_drift_samples_per_s", float("nan"))),
        ),
        "phase_drift_improvement_pct": phase_eval.improvement_pct(
            float(monitor_summary.get("phase_drift_slope_full_hz", float("nan"))),
            float(pid_summary.get("phase_drift_slope_full_hz", float("nan"))),
        ),
        "mean_abs_phase_err_improvement_pct": phase_eval.improvement_pct(
            float(monitor_summary.get("mean_abs_phase_err_rad", float("nan"))),
            float(pid_summary.get("mean_abs_phase_err_rad", float("nan"))),
        ),
        "accepted_raw_phase_band_improvement_abs": (
            float(pid_metrics.steady_state_accepted_raw_phase_band_fraction - monitor_metrics.steady_state_accepted_raw_phase_band_fraction)
            if (
                np.isfinite(float(pid_metrics.steady_state_accepted_raw_phase_band_fraction))
                and np.isfinite(float(monitor_metrics.steady_state_accepted_raw_phase_band_fraction))
            )
            else float("nan")
        ),
    }
    output = {
        "run_dir": str(run_dir),
        "nsamp": int(nsamp),
        "config": {
            "phase_band_rad": float(phase_band_rad),
            "min_lock_dwell_s": float(min_lock_dwell_s),
            "max_bad_gap_s": float(max_bad_gap_s),
            "coh_min": None if coh_min is None else float(coh_min),
            "require_quality_ok": bool(require_quality_ok),
            "require_phase_lock_stage": bool(require_phase_lock_stage),
        },
        "monitor_summary": monitor_summary,
        "pid_summary": pid_summary,
        "monitor_phase_metrics": asdict(monitor_metrics),
        "pid_phase_metrics": asdict(pid_metrics),
        "comparison": comparison,
    }
    comparison_dir = run_dir / "comparison"
    json_path = comparison_dir / str(json_name)
    md_path = comparison_dir / str(report_name)
    phase_eval.write_json(json_path, output)
    phase_eval.write_markdown(
        md_path,
        phase_eval.render_markdown(
            run_dir=run_dir,
            nsamp=int(nsamp),
            output=output,
        ),
    )
    return output


def run_postprocess(args: argparse.Namespace, *, session_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import plot_saved_open_closed_comparison as timing_plot
    import plot_saved_open_closed_phase_comparison as phase_plot

    paths = phase_plot.infer_paths(session_dir, int(args.nsamp))
    monitor_rows = phase_plot.load_csv_rows(paths["monitor_csv"])
    pid_rows = phase_plot.load_csv_rows(paths["pid_csv"])
    monitor_summary = phase_plot.load_summary(paths["monitor_summary"])
    pid_summary = phase_plot.load_summary(paths["pid_summary"])
    comparison_dir = session_dir / "comparison"

    timing_metrics = timing_plot.plot_dashboard(
        output_path=comparison_dir / "open_closed_timing_dashboard.png",
        run_label=session_dir.name,
        monitor_rows=monitor_rows,
        pid_rows=pid_rows,
        monitor_summary=monitor_summary,
        pid_summary=pid_summary,
    )
    timing_plot.write_json(comparison_dir / "timing_proxy_metrics.json", timing_metrics)
    timing_plot.write_markdown(
        comparison_dir / "timing_proxy_report.md",
        session_dir,
        timing_metrics["monitor"],
        timing_metrics["closed_loop"],
        timing_metrics.get("comparison"),
    )

    phase_metrics = phase_plot.plot_phase_dashboard(
        open_rows=monitor_rows,
        closed_rows=pid_rows,
        open_summary=monitor_summary,
        closed_summary=pid_summary,
        output_path=comparison_dir / "open_closed_phase_dashboard.png",
    )
    phase_plot.write_json(comparison_dir / "phase_dashboard_metrics.json", phase_metrics)

    run_phase_evaluation(
        run_dir=session_dir,
        nsamp=int(args.nsamp),
        phase_band_rad=float(args.phase_eval_phase_band_rad),
        min_lock_dwell_s=float(args.phase_eval_min_lock_dwell_s),
        max_bad_gap_s=float(args.phase_eval_max_bad_gap_s),
        coh_min=float(args.phase_eval_coh_min) if args.phase_eval_coh_min is not None else None,
        require_quality_ok=True,
        require_phase_lock_stage=True,
        json_name="phase_lock_evaluation.json",
        report_name="phase_lock_evaluation.md",
    )
    run_phase_evaluation(
        run_dir=session_dir,
        nsamp=int(args.nsamp),
        phase_band_rad=float(args.phase_eval_phase_band_rad),
        min_lock_dwell_s=float(args.phase_eval_min_lock_dwell_s),
        max_bad_gap_s=float(args.phase_eval_max_bad_gap_s),
        coh_min=float(args.phase_eval_coh_min) if args.phase_eval_coh_min is not None else None,
        require_quality_ok=True,
        require_phase_lock_stage=False,
        json_name="phase_lock_evaluation_phase_only.json",
        report_name="phase_lock_evaluation_phase_only.md",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a synthetic open-vs-closed comparison for NotebookBased DPLL.")
    p.add_argument(
        "--controller-module",
        type=str,
        default="notebook_based_pullin_handoff_30hz_phase_dpll",
        help="NotebookBased controller module to exercise inside the synthetic harness.",
    )
    p.add_argument("--label", type=str, default="simulated_open_closed")
    p.add_argument("--artifact-root", type=Path, default=BASE_DIR / "runs")
    p.add_argument("--nsamp", type=int, default=8192)
    p.add_argument("--run-seconds", type=float, default=20.0)
    p.add_argument("--sample-rate", type=float, default=2.4e6)
    p.add_argument("--center-freq", type=float, default=100e6)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--settle-seconds", type=float, default=0.0)
    p.add_argument("--skip-sham", action="store_true", help="Skip the sham-control synthetic run.")
    p.add_argument("--async-queue-blocks", type=int, default=2)
    p.add_argument("--tone-estimate-blocks", type=int, default=8)
    p.add_argument("--dc-exclusion-hz", type=float, default=5e3)
    p.add_argument("--control-measurement-source", choices=["adaptive", "observer"], default="adaptive")
    p.add_argument("--phase-reference-mode", choices=["first_valid", "absolute"], default="first_valid")
    p.add_argument("--phase-enter-condition", choices=["full", "pullin_only", "residual_and_quality"], default="residual_and_quality")
    p.add_argument("--phase-exit-condition", choices=["full", "residual_only", "residual_and_phase"], default="residual_and_phase")
    p.add_argument("--phase-predict-through-holds", dest="phase_predict_through_holds", action="store_true", default=True)
    p.add_argument("--no-phase-predict-through-holds", dest="phase_predict_through_holds", action="store_false")
    p.add_argument("--phase-filter-alpha", type=float, default=0.85)
    p.add_argument("--phase-enter-max-err-rad", type=float, default=0.5)
    p.add_argument("--phase-exit-min-err-rad", type=float, default=1.0)
    p.add_argument("--phase-frac-gain-codes", type=float, default=409500.0)
    p.add_argument("--phase-hold-frac-gain-codes", type=float, default=102375.0)
    p.add_argument("--phase-center-integral-gain-codes-per-rad-s", type=float, default=12.0)
    p.add_argument("--phase-center-freq-gain-codes-per-hz-s", type=float, default=0.35)
    p.add_argument("--phase-hold-center-freq-gain-codes-per-hz-s", type=float, default=0.08)
    p.add_argument("--phase-center-freq-filter-alpha", type=float, default=0.80)
    p.add_argument("--phase-center-freq-max-hz", type=float, default=25.0)
    p.add_argument("--phase-center-integral-limit-codes", type=float, default=64.0)
    p.add_argument("--phase-hold-enter-err-rad", type=float, default=0.20)
    p.add_argument("--phase-hold-exit-err-rad", type=float, default=0.35)
    p.add_argument("--phase-hold-enter-dwell-updates", type=int, default=3)
    p.add_argument("--phase-hold-exit-dwell-updates", type=int, default=2)
    p.add_argument("--phase-accepted-raw-enter-window-updates", type=int, default=12)
    p.add_argument("--phase-accepted-raw-enter-min-frac", type=float, default=0.75)
    p.add_argument("--phase-accepted-raw-exit-window-updates", type=int, default=12)
    p.add_argument("--phase-accepted-raw-exit-min-frac", type=float, default=0.50)
    p.add_argument("--phase-reference-ramp-accepted-updates", type=int, default=24)
    p.add_argument("--phase-reference-ramp-min-scale", type=float, default=0.25)
    p.add_argument("--phase-min-hold-updates", type=int, default=64)
    p.add_argument("--phase-hold-integral-max-err-rad", type=float, default=1.5)
    p.add_argument("--phase-hold-detector-limit-rad", type=float, default=1.2)
    p.add_argument("--phase-hold-deadband-rad", type=float, default=0.08)
    p.add_argument("--phase-hold-kp-hz-per-rad", type=float, default=8.0)
    p.add_argument("--phase-hold-ki-hz-per-rad-s", type=float, default=3.0)
    p.add_argument("--phase-hold-kd-hz-per-hz", type=float, default=0.20)
    p.add_argument("--phase-hold-rate-filter-alpha", type=float, default=0.75)
    p.add_argument("--phase-hold-rate-max-hz", type=float, default=12.0)
    p.add_argument("--phase-hold-integral-limit-hz", type=float, default=40.0)
    p.add_argument("--phase-hold-max-u-hz", type=float, default=20.0)
    p.add_argument("--max-phase-dac-step-per-update", type=int, default=2)
    p.add_argument("--max-phase-hold-dac-step-per-update", type=int, default=1)
    p.add_argument("--drift-window-s", type=float, default=8.0)
    p.add_argument("--notes", type=str, default="synthetic NotebookBased comparison")

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
    p.add_argument("--integral-limit-hz", type=float, default=200.0)
    p.add_argument("--max-u-hz", type=float, default=2000.0)
    p.add_argument("--max-dac-step-per-update", type=int, default=20)
    p.add_argument("--pullin-switch-hz", type=float, default=30.0)
    p.add_argument("--pullin-exit-hz", type=float, default=100.0)
    p.add_argument("--pullin-enter-dwell-updates", type=int, default=3)
    p.add_argument("--pullin-exit-dwell-updates", type=int, default=16)
    p.add_argument("--pullin-target-hz", type=float, default=0.0)
    p.add_argument("--skip-clock-init", action="store_true")
    p.add_argument("--apply-start-dac", dest="apply_start_dac", action="store_true", default=True)
    p.add_argument("--no-apply-start-dac", dest="apply_start_dac", action="store_false")
    p.add_argument("--apr-ppm", type=int, default=120)
    p.add_argument("--fine-settle-s", type=float, default=0.0)
    p.add_argument("--slip-max-cycles-per-update", type=int, default=32)

    p.add_argument("--sim-initial-tone-offset-hz", type=float, default=-30688.4765625)
    p.add_argument("--sim-initial-phase-rad", type=float, default=0.0)
    p.add_argument("--sim-path-phase-offset-rad", type=float, default=None)
    p.add_argument("--sim-step-hz", type=float, default=1500.0)
    p.add_argument("--sim-step-at-s", type=float, default=0.5)
    p.add_argument("--sim-drift-rate-hz-per-s", type=float, default=0.0)
    p.add_argument("--sim-random-walk-hz-per-sqrt-s", type=float, default=0.0)
    p.add_argument("--sim-receiver-drift-rate-hz-per-s", type=float, default=0.0)
    p.add_argument("--sim-receiver-random-walk-hz-per-sqrt-s", type=float, default=0.0)
    p.add_argument("--sim-amplitude", type=float, default=0.85)
    p.add_argument("--sim-noise-std", type=float, default=0.04)
    p.add_argument("--sim-phase-noise-std-rad", type=float, default=0.0)
    p.add_argument("--sim-iq-scale", type=float, default=90.0)
    p.add_argument("--sim-plant-hz-per-code", type=float, default=4.243)
    p.add_argument("--sim-plant-tau-s", type=float, default=0.05)
    p.add_argument("--sim-dac-center-code", type=int, default=2029)
    p.add_argument("--sim-seed", type=int, default=12345)
    p.add_argument("--pid-seed-offset", type=int, default=0)
    p.add_argument("--sim-realtime", action="store_true")

    p.add_argument("--phase-eval-phase-band-rad", type=float, default=1.0)
    p.add_argument("--phase-eval-min-lock-dwell-s", type=float, default=2.0)
    p.add_argument("--phase-eval-max-bad-gap-s", type=float, default=0.05)
    p.add_argument("--phase-eval-coh-min", type=float, default=0.025)
    p.add_argument("--skip-postprocess", action="store_true")
    return p


def main() -> int:
    global controller
    args = build_arg_parser().parse_args()
    controller = importlib.import_module(str(args.controller_module))
    run_id = f"{timestamp_for_filename()}_{args.label}"
    session_dir = Path(args.artifact_root) / run_id
    ensure_parent_dir(session_dir / "session_index.json")

    sim_config = simulation_config_for_mode(args, mode="monitor")
    write_json(session_dir / "simulation_config.json", asdict(sim_config))

    monitor_summary, monitor_truth = run_mode_once(
        args,
        session_dir=session_dir,
        mode="monitor",
        artifact_mode="monitor",
        observer_fixed_offset_hz=None,
    )
    pid_observer_offset = float(monitor_summary.get("tone_offset_hz", float("nan")))
    pid_summary, pid_truth = run_mode_once(
        args,
        session_dir=session_dir,
        mode="pid",
        artifact_mode="pid",
        observer_fixed_offset_hz=pid_observer_offset if np.isfinite(pid_observer_offset) else None,
    )
    sham_summary: dict[str, Any] | None = None
    sham_truth: list[dict[str, Any]] = []
    if not bool(args.skip_sham):
        sham_summary, sham_truth = run_mode_once(
            args,
            session_dir=session_dir,
            mode="pid",
            artifact_mode="sham",
            observer_fixed_offset_hz=pid_observer_offset if np.isfinite(pid_observer_offset) else None,
            sham_control_mode="random_sign",
        )

    session_index = [
        {
            "mode": "monitor",
            "nsamp": int(args.nsamp),
            "csv_path": str(mode_artifact_paths(session_dir, "monitor", int(args.nsamp))["csv"].resolve()),
            "summary_json_path": str(mode_artifact_paths(session_dir, "monitor", int(args.nsamp))["summary_json"].resolve()),
            "truth_csv_path": str(mode_artifact_paths(session_dir, "monitor", int(args.nsamp))["truth_csv"].resolve()),
        },
        {
            "mode": "pid",
            "nsamp": int(args.nsamp),
            "csv_path": str(mode_artifact_paths(session_dir, "pid", int(args.nsamp))["csv"].resolve()),
            "summary_json_path": str(mode_artifact_paths(session_dir, "pid", int(args.nsamp))["summary_json"].resolve()),
            "truth_csv_path": str(mode_artifact_paths(session_dir, "pid", int(args.nsamp))["truth_csv"].resolve()),
        },
    ]
    if sham_summary is not None:
        session_index.append(
            {
                "mode": "sham",
                "nsamp": int(args.nsamp),
                "csv_path": str(mode_artifact_paths(session_dir, "sham", int(args.nsamp))["csv"].resolve()),
                "summary_json_path": str(mode_artifact_paths(session_dir, "sham", int(args.nsamp))["summary_json"].resolve()),
                "truth_csv_path": str(mode_artifact_paths(session_dir, "sham", int(args.nsamp))["truth_csv"].resolve()),
            }
        )
    write_json(session_dir / "session_index.json", session_index)

    comparison = summarize_comparison(
        nsamp=int(args.nsamp),
        monitor_summary=monitor_summary,
        pid_summary=pid_summary,
    )
    comparison_dir = session_dir / "comparison"
    write_json(comparison_dir / "nsamp_comparison.json", [comparison])
    write_csv_rows(comparison_dir / "nsamp_comparison.csv", [comparison])

    truth_summary = {
        "monitor_truth_rows": len(monitor_truth),
        "pid_truth_rows": len(pid_truth),
        "sham_truth_rows": len(sham_truth),
        "monitor_final_true_offset_hz": float(monitor_truth[-1]["true_offset_hz"]) if monitor_truth else float("nan"),
        "pid_final_true_offset_hz": float(pid_truth[-1]["true_offset_hz"]) if pid_truth else float("nan"),
        "sham_final_true_offset_hz": float(sham_truth[-1]["true_offset_hz"]) if sham_truth else float("nan"),
        "monitor_final_dac_code": int(monitor_truth[-1]["dac_code_cmd"]) if monitor_truth else int(args.sim_dac_center_code),
        "pid_final_dac_code": int(pid_truth[-1]["dac_code_cmd"]) if pid_truth else int(args.sim_dac_center_code),
        "sham_final_dac_code": int(sham_truth[-1]["dac_code_cmd"]) if sham_truth else int(args.sim_dac_center_code),
    }
    write_json(comparison_dir / "synthetic_truth_summary.json", truth_summary)

    if sham_summary is not None:
        monitor_rows = phase_eval.load_csv_rows(mode_artifact_paths(session_dir, "monitor", int(args.nsamp))["csv"])
        sham_rows = phase_eval.load_csv_rows(mode_artifact_paths(session_dir, "sham", int(args.nsamp))["csv"])
        sample_rate_hz = float(
            sham_summary.get("sample_rate_hz", monitor_summary.get("sample_rate_hz", args.sample_rate))
        )
        sham_monitor_metrics = phase_eval.detect_settled_metrics(
            rows=monitor_rows,
            sample_rate_hz=sample_rate_hz,
            phase_band_rad=float(args.phase_eval_phase_band_rad),
            min_lock_dwell_s=float(args.phase_eval_min_lock_dwell_s),
            max_bad_gap_s=float(args.phase_eval_max_bad_gap_s),
            require_quality_ok=True,
            coh_min=float(args.phase_eval_coh_min) if args.phase_eval_coh_min is not None else None,
            require_phase_lock_stage=True,
        )
        sham_metrics = phase_eval.detect_settled_metrics(
            rows=sham_rows,
            sample_rate_hz=sample_rate_hz,
            phase_band_rad=float(args.phase_eval_phase_band_rad),
            min_lock_dwell_s=float(args.phase_eval_min_lock_dwell_s),
            max_bad_gap_s=float(args.phase_eval_max_bad_gap_s),
            require_quality_ok=True,
            coh_min=float(args.phase_eval_coh_min) if args.phase_eval_coh_min is not None else None,
            require_phase_lock_stage=True,
        )
        sham_comparison = {
            "timing_drift_improvement_pct": phase_eval.improvement_pct(
                float(monitor_summary.get("primary_timing_drift_samples_per_s", float("nan"))),
                float(sham_summary.get("primary_timing_drift_samples_per_s", float("nan"))),
            ),
            "phase_drift_improvement_pct": phase_eval.improvement_pct(
                float(monitor_summary.get("phase_drift_slope_full_hz", float("nan"))),
                float(sham_summary.get("phase_drift_slope_full_hz", float("nan"))),
            ),
            "mean_abs_phase_err_improvement_pct": phase_eval.improvement_pct(
                float(monitor_summary.get("mean_abs_phase_err_rad", float("nan"))),
                float(sham_summary.get("mean_abs_phase_err_rad", float("nan"))),
            ),
            "accepted_raw_phase_band_improvement_abs": (
                float(
                    sham_metrics.steady_state_accepted_raw_phase_band_fraction
                    - sham_monitor_metrics.steady_state_accepted_raw_phase_band_fraction
                )
                if (
                    np.isfinite(float(sham_metrics.steady_state_accepted_raw_phase_band_fraction))
                    and np.isfinite(float(sham_monitor_metrics.steady_state_accepted_raw_phase_band_fraction))
                )
                else float("nan")
            ),
        }
        write_json(
            comparison_dir / "phase_lock_evaluation_sham.json",
            {
                "run_dir": str(session_dir),
                "nsamp": int(args.nsamp),
                "monitor_summary": monitor_summary,
                "sham_summary": sham_summary,
                "monitor_phase_metrics": asdict(sham_monitor_metrics),
                "sham_phase_metrics": asdict(sham_metrics),
                "comparison": sham_comparison,
            },
        )

    if not args.skip_postprocess:
        run_postprocess(args, session_dir=session_dir)

    print(f"Artifacts written under: {session_dir}")
    print(
        "Comparison summary:\n"
        f"  nsamp={comparison['nsamp']}: timing {comparison['monitor_timing_drift_samples_per_s']:.3f}"
        f" -> {comparison['pid_timing_drift_samples_per_s']:.3f} samples/s, "
        f"phase drift |Hz| {abs(comparison['monitor_phase_drift_slope_full_hz']):.2f}"
        f" -> {abs(comparison['pid_phase_drift_slope_full_hz']):.2f}, "
        f"timing improvement {comparison['timing_improvement_pct']:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
