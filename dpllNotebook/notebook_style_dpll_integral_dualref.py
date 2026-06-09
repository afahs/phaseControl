#!/usr/bin/env python3
"""Notebook-style live DPLL with integral action and dual phase references."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import dpllNotebook.notebook_style_dpll as base


@dataclass
class ControllerState:
    control_phase_err_lpf: base.CircularLpfState = field(default_factory=base.CircularLpfState)
    phase_err_int_rad_s: float = 0.0
    dac_code_cmd: int = 0
    dac_code_rb: int | None = None
    dac_volts: float = float("nan")
    observer_timing_proxy_ref_phase_rad: float | None = None
    control_updates_applied: int = 0
    held_updates: int = 0
    low_quality_holds: int = 0


@dataclass
class ControlDecision:
    phase_fraction_cycles_per_sample: float
    p_codes: float
    i_codes: float
    phase_err_int_rad_s: float
    dac_target_float: float
    dac_target_code: int
    dac_step_req_codes: int
    dac_code_proposed: int


CSV_COLUMNS = [
    "unix_time_s",
    "elapsed_s",
    "update_idx",
    "mode",
    "action",
    "sample_start",
    "nsamp",
    "sample_rate_hz",
    "center_freq_hz",
    "control_tone_offset_hz",
    "control_tone_track_step_hz",
    "observer_tone_offset_hz",
    "control_rf_reference_hz",
    "observer_rf_reference_hz",
    "control_mix_real",
    "control_mix_imag",
    "control_mix_abs",
    "observer_mix_real",
    "observer_mix_imag",
    "observer_mix_abs",
    "control_phase_wrapped_rad",
    "control_phase_unwrapped_rad",
    "observer_phase_wrapped_rad",
    "observer_phase_unwrapped_rad",
    "phase_setpoint_rad",
    "control_phase_err_raw_rad",
    "control_phase_err_filt_rad",
    "observer_phase_err_raw_rad",
    "control_phase_fraction_cycles_per_sample",
    "control_coherence",
    "observer_coherence",
    "control_pullin_quality",
    "observer_pullin_quality",
    "control_quality_ok",
    "observer_quality_ok",
    "control_amp_rms",
    "observer_amp_rms",
    "control_freq_err_intrablock_hz",
    "control_freq_err_update_hz",
    "observer_freq_err_intrablock_hz",
    "observer_freq_err_update_hz",
    "observer_timing_proxy_ns",
    "control_slip_cycles",
    "observer_slip_cycles",
    "control_dphi_wrapped_rad",
    "control_dphi_used_rad",
    "control_dphi_expected_rad",
    "observer_dphi_wrapped_rad",
    "observer_dphi_used_rad",
    "observer_dphi_expected_rad",
    "phase_err_int_rad_s",
    "u_p_codes",
    "u_i_codes",
    "dac_center_code",
    "dac_target_float",
    "dac_target_code",
    "dac_code_cmd",
    "dac_code_rb",
    "dac_code_proposed",
    "dac_volts",
    "dac_step_req_codes",
    "dac_step_applied_codes",
    "phase_frac_gain_codes",
    "integral_gain_codes_per_rad_s",
    "integral_limit_codes",
    "max_dac_step_per_update",
    "notes",
]


class CsvLogger:
    def __init__(self, path: Path):
        self.path = path
        base.ensure_parent_dir(path)
        self.fh = path.open("w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()
        self.fh.flush()

    def write(self, row: dict[str, Any]) -> None:
        out = {k: row.get(k, "") for k in CSV_COLUMNS}
        self.writer.writerow(out)
        self.fh.flush()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass


def phase_err_to_dac_target(
    *,
    phase_err_filt_rad: float,
    phase_err_int_rad_s: float,
    nsamp: int,
    dac_center: int,
    phase_frac_gain_codes: float,
    integral_gain_codes_per_rad_s: float,
    control_sign: float,
    code_min: int,
    code_max: int,
) -> tuple[float, float, float, float, int]:
    phase_fraction = float(float(phase_err_filt_rad) / (2.0 * math.pi * float(nsamp)))
    p_codes = float(control_sign) * float(phase_frac_gain_codes) * float(phase_fraction)
    i_codes = float(control_sign) * float(integral_gain_codes_per_rad_s) * float(phase_err_int_rad_s)
    dac_target_float = float(dac_center) + float(p_codes) + float(i_codes)
    dac_target_code = base.clamp_int(dac_target_float, int(code_min), int(code_max))
    return float(phase_fraction), float(p_codes), float(i_codes), float(dac_target_float), int(dac_target_code)


def compute_control_decision(
    *,
    current_code: int,
    phase_err_filt_rad: float,
    current_integral_rad_s: float,
    dt_update_s: float,
    nsamp: int,
    dac_center: int,
    phase_frac_gain_codes: float,
    integral_gain_codes_per_rad_s: float,
    integral_limit_codes: float,
    control_sign: float,
    code_min: int,
    code_max: int,
    max_dac_step_per_update: int,
    quality_ok: bool,
) -> ControlDecision:
    integral_state = float(current_integral_rad_s)

    if quality_ok and abs(float(integral_gain_codes_per_rad_s)) > 1e-18:
        if np.isfinite(dt_update_s) and float(dt_update_s) > 0.0:
            candidate = float(current_integral_rad_s) + (float(phase_err_filt_rad) * float(dt_update_s))
        else:
            candidate = float(current_integral_rad_s)

        if abs(float(integral_limit_codes)) > 0.0:
            integral_limit_rad_s = abs(float(integral_limit_codes) / float(integral_gain_codes_per_rad_s))
            candidate = base.clamp(candidate, -integral_limit_rad_s, integral_limit_rad_s)

        (
            _,
            _,
            current_i_codes,
            _,
            _,
        ) = phase_err_to_dac_target(
            phase_err_filt_rad=float(phase_err_filt_rad),
            phase_err_int_rad_s=float(current_integral_rad_s),
            nsamp=int(nsamp),
            dac_center=int(dac_center),
            phase_frac_gain_codes=float(phase_frac_gain_codes),
            integral_gain_codes_per_rad_s=float(integral_gain_codes_per_rad_s),
            control_sign=float(control_sign),
            code_min=int(code_min),
            code_max=int(code_max),
        )
        (
            _,
            _,
            candidate_i_codes,
            candidate_target_float,
            _,
        ) = phase_err_to_dac_target(
            phase_err_filt_rad=float(phase_err_filt_rad),
            phase_err_int_rad_s=float(candidate),
            nsamp=int(nsamp),
            dac_center=int(dac_center),
            phase_frac_gain_codes=float(phase_frac_gain_codes),
            integral_gain_codes_per_rad_s=float(integral_gain_codes_per_rad_s),
            control_sign=float(control_sign),
            code_min=int(code_min),
            code_max=int(code_max),
        )
        pushes_more_positive = candidate_i_codes > (current_i_codes + 1e-12)
        pushes_more_negative = candidate_i_codes < (current_i_codes - 1e-12)
        if not (
            (candidate_target_float > float(code_max) and pushes_more_positive)
            or (candidate_target_float < float(code_min) and pushes_more_negative)
        ):
            integral_state = float(candidate)

    (
        phase_fraction,
        p_codes,
        i_codes,
        dac_target_float,
        dac_target_code,
    ) = phase_err_to_dac_target(
        phase_err_filt_rad=float(phase_err_filt_rad),
        phase_err_int_rad_s=float(integral_state),
        nsamp=int(nsamp),
        dac_center=int(dac_center),
        phase_frac_gain_codes=float(phase_frac_gain_codes),
        integral_gain_codes_per_rad_s=float(integral_gain_codes_per_rad_s),
        control_sign=float(control_sign),
        code_min=int(code_min),
        code_max=int(code_max),
    )

    if not quality_ok:
        return ControlDecision(
            phase_fraction_cycles_per_sample=float(phase_fraction),
            p_codes=float(p_codes),
            i_codes=float(i_codes),
            phase_err_int_rad_s=float(current_integral_rad_s),
            dac_target_float=float(dac_target_float),
            dac_target_code=int(dac_target_code),
            dac_step_req_codes=0,
            dac_code_proposed=int(current_code),
        )

    step_to_target = int(dac_target_code) - int(current_code)
    dac_step_req = base.clamp_int(
        step_to_target,
        -int(max_dac_step_per_update),
        int(max_dac_step_per_update),
    )
    dac_code_proposed = base.clamp_int(
        int(current_code) + int(dac_step_req),
        int(code_min),
        int(code_max),
    )
    return ControlDecision(
        phase_fraction_cycles_per_sample=float(phase_fraction),
        p_codes=float(p_codes),
        i_codes=float(i_codes),
        phase_err_int_rad_s=float(integral_state),
        dac_target_float=float(dac_target_float),
        dac_target_code=int(dac_target_code),
        dac_step_req_codes=int(dac_step_req),
        dac_code_proposed=int(dac_code_proposed),
    )


def run_controller_self_checks() -> None:
    phase_fraction, p_codes, i_codes, dac_target_float, dac_target_code = phase_err_to_dac_target(
        phase_err_filt_rad=0.0,
        phase_err_int_rad_s=0.0,
        nsamp=256,
        dac_center=2048,
        phase_frac_gain_codes=1000.0,
        integral_gain_codes_per_rad_s=10.0,
        control_sign=1.0,
        code_min=0,
        code_max=4095,
    )
    if phase_fraction != 0.0 or p_codes != 0.0 or i_codes != 0.0 or dac_target_float != 2048.0 or dac_target_code != 2048:
        raise AssertionError("Zero phase and zero integral must map to dac_center.")

    _, _, i_codes_pos, _, target_pos = phase_err_to_dac_target(
        phase_err_filt_rad=0.0,
        phase_err_int_rad_s=2.0,
        nsamp=256,
        dac_center=2000,
        phase_frac_gain_codes=0.0,
        integral_gain_codes_per_rad_s=10.0,
        control_sign=1.0,
        code_min=0,
        code_max=4095,
    )
    if i_codes_pos <= 0.0 or target_pos <= 2000:
        raise AssertionError("Positive integral state must increase the target for positive control sign.")

    hold_decision = compute_control_decision(
        current_code=1500,
        phase_err_filt_rad=1.0,
        current_integral_rad_s=2.0,
        dt_update_s=0.1,
        nsamp=1,
        dac_center=2048,
        phase_frac_gain_codes=1e6,
        integral_gain_codes_per_rad_s=10.0,
        integral_limit_codes=200.0,
        control_sign=1.0,
        code_min=0,
        code_max=4095,
        max_dac_step_per_update=8,
        quality_ok=False,
    )
    if hold_decision.dac_code_proposed != 1500 or hold_decision.dac_step_req_codes != 0 or hold_decision.phase_err_int_rad_s != 2.0:
        raise AssertionError("Low-quality updates must hold both DAC command and integral state.")

    limited_decision = compute_control_decision(
        current_code=1000,
        phase_err_filt_rad=10.0,
        current_integral_rad_s=0.0,
        dt_update_s=0.1,
        nsamp=1,
        dac_center=2048,
        phase_frac_gain_codes=1e6,
        integral_gain_codes_per_rad_s=10.0,
        integral_limit_codes=200.0,
        control_sign=1.0,
        code_min=0,
        code_max=4095,
        max_dac_step_per_update=7,
        quality_ok=True,
    )
    if abs(int(limited_decision.dac_step_req_codes)) > 7:
        raise AssertionError("DAC rate limiting must respect max_dac_step_per_update.")


def compute_summary(
    *,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    observer_tone_offset_hz: float,
) -> dict[str, Any]:
    if not rows:
        return {
            "mode": str(args.mode),
            "nsamp": int(args.nsamp),
            "sample_rate_hz": float(args.sample_rate),
            "center_freq_hz": float(args.center_freq),
            "loop_period_nom_s": float(args.nsamp) / float(args.sample_rate),
            "observer_tone_offset_hz": float(observer_tone_offset_hz),
            "updates": 0,
            "elapsed_s": 0.0,
            "median_control_coherence": float("nan"),
            "median_observer_coherence": float("nan"),
            "median_control_pullin_quality": float("nan"),
            "median_observer_pullin_quality": float("nan"),
            "mean_abs_control_phase_err_rad": float("nan"),
            "median_abs_control_phase_err_rad": float("nan"),
            "mean_abs_control_phase_err_filt_rad": float("nan"),
            "median_abs_control_phase_err_filt_rad": float("nan"),
            "mean_abs_observer_phase_err_rad": float("nan"),
            "median_abs_observer_phase_err_rad": float("nan"),
            "control_phase_drift_slope_hz": float("nan"),
            "observer_phase_drift_slope_hz": float("nan"),
            "observer_timing_drift_ns_per_s": float("nan"),
            "observer_timing_drift_samples_per_s": float("nan"),
            "max_abs_observer_timing_proxy_ns": float("nan"),
            "p95_abs_observer_timing_proxy_ns": float("nan"),
            "quality_fraction": float("nan"),
            "dac_updates_applied": 0,
            "dac_updates_held": 0,
            "dac_update_fraction": float("nan"),
            "low_quality_holds": 0,
            "control_reference_walk_hz": float("nan"),
        }

    def _arr(key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, float("nan"))) for r in rows], dtype=np.float64)

    t = _arr("elapsed_s")
    control_phu = _arr("control_phase_unwrapped_rad")
    observer_phu = _arr("observer_phase_unwrapped_rad")
    control_pe = _arr("control_phase_err_raw_rad")
    control_pef = _arr("control_phase_err_filt_rad")
    observer_pe = _arr("observer_phase_err_raw_rad")
    control_coh = _arr("control_coherence")
    observer_coh = _arr("observer_coherence")
    control_pq = _arr("control_pullin_quality")
    observer_pq = _arr("observer_pullin_quality")
    observer_timing = _arr("observer_timing_proxy_ns")
    control_ref = _arr("control_tone_offset_hz")
    quality = np.asarray([bool(int(float(r.get("control_quality_ok", 0)))) for r in rows], dtype=bool)

    def _row_bool(row: dict[str, Any], key: str) -> bool:
        raw = row.get(key, False)
        if isinstance(raw, bool):
            return bool(raw)
        if raw is None:
            return False
        if isinstance(raw, (int, float, np.integer, np.floating)):
            return bool(raw)
        s = str(raw).strip().lower()
        if s in {"", "0", "false", "no", "n"}:
            return False
        if s in {"1", "true", "yes", "y"}:
            return True
        try:
            return bool(int(float(s)))
        except Exception:
            return bool(s)

    applied = np.asarray(
        [
            _row_bool(r, "dac_applied")
            or (abs(float(r.get("dac_step_applied_codes", 0.0))) > 0.0)
            for r in rows
        ],
        dtype=bool,
    )

    control_phase_slope = base.linear_slope(t, control_phu)
    observer_phase_slope = base.linear_slope(t, observer_phu)
    control_phase_slope_hz = (
        float(control_phase_slope / (2.0 * math.pi)) if np.isfinite(control_phase_slope) else float("nan")
    )
    observer_phase_slope_hz = (
        float(observer_phase_slope / (2.0 * math.pi)) if np.isfinite(observer_phase_slope) else float("nan")
    )

    valid_timing = np.isfinite(t) & np.isfinite(observer_timing)
    observer_timing_drift_ns_per_s = (
        base.linear_slope(t[valid_timing], observer_timing[valid_timing])
        if np.count_nonzero(valid_timing) >= 2
        else float("nan")
    )
    observer_timing_drift_samples_per_s = (
        float(observer_timing_drift_ns_per_s * float(args.sample_rate) / 1e9)
        if np.isfinite(observer_timing_drift_ns_per_s)
        else float("nan")
    )
    abs_observer_timing = np.abs(observer_timing[np.isfinite(observer_timing)])
    elapsed_s = float(max(0.0, float(t[-1] - t[0]))) if t.size else 0.0
    valid_control_ref = control_ref[np.isfinite(control_ref)]
    control_reference_walk_hz = (
        float(valid_control_ref[-1] - valid_control_ref[0]) if valid_control_ref.size >= 2 else float("nan")
    )

    return {
        "mode": str(args.mode),
        "nsamp": int(args.nsamp),
        "sample_rate_hz": float(args.sample_rate),
        "center_freq_hz": float(args.center_freq),
        "loop_period_nom_s": float(args.nsamp) / float(args.sample_rate),
        "observer_tone_offset_hz": float(observer_tone_offset_hz),
        "updates": int(len(rows)),
        "elapsed_s": float(elapsed_s),
        "median_control_coherence": float(np.nanmedian(control_coh)) if control_coh.size else float("nan"),
        "median_observer_coherence": float(np.nanmedian(observer_coh)) if observer_coh.size else float("nan"),
        "median_control_pullin_quality": float(np.nanmedian(control_pq)) if control_pq.size else float("nan"),
        "median_observer_pullin_quality": float(np.nanmedian(observer_pq)) if observer_pq.size else float("nan"),
        "mean_abs_control_phase_err_rad": float(np.nanmean(np.abs(control_pe))) if control_pe.size else float("nan"),
        "median_abs_control_phase_err_rad": float(np.nanmedian(np.abs(control_pe))) if control_pe.size else float("nan"),
        "mean_abs_control_phase_err_filt_rad": float(np.nanmean(np.abs(control_pef))) if control_pef.size else float("nan"),
        "median_abs_control_phase_err_filt_rad": float(np.nanmedian(np.abs(control_pef))) if control_pef.size else float("nan"),
        "mean_abs_observer_phase_err_rad": float(np.nanmean(np.abs(observer_pe))) if observer_pe.size else float("nan"),
        "median_abs_observer_phase_err_rad": float(np.nanmedian(np.abs(observer_pe))) if observer_pe.size else float("nan"),
        "control_phase_drift_slope_hz": float(control_phase_slope_hz),
        "observer_phase_drift_slope_hz": float(observer_phase_slope_hz),
        "observer_timing_drift_ns_per_s": float(observer_timing_drift_ns_per_s),
        "observer_timing_drift_samples_per_s": float(observer_timing_drift_samples_per_s),
        "max_abs_observer_timing_proxy_ns": float(np.nanmax(abs_observer_timing)) if abs_observer_timing.size else float("nan"),
        "p95_abs_observer_timing_proxy_ns": float(np.nanpercentile(abs_observer_timing, 95.0)) if abs_observer_timing.size else float("nan"),
        "quality_fraction": float(np.mean(quality)) if quality.size else float("nan"),
        "dac_updates_applied": int(np.count_nonzero(applied)),
        "dac_updates_held": int(len(rows) - np.count_nonzero(applied)),
        "dac_update_fraction": float(np.count_nonzero(applied) / len(rows)),
        "low_quality_holds": int(np.count_nonzero(~quality)),
        "control_reference_walk_hz": float(control_reference_walk_hz),
    }


def print_run_summary(summary: dict[str, Any]) -> None:
    print("\nRun summary:")
    print(f"  mode: {summary.get('mode')}")
    print(f"  nsamp: {summary.get('nsamp')}")
    print(f"  observer_tone_offset_hz: {summary.get('observer_tone_offset_hz')}")
    print(f"  updates: {summary.get('updates')}")
    print(f"  median_control_coherence: {summary.get('median_control_coherence')}")
    print(f"  median_observer_coherence: {summary.get('median_observer_coherence')}")
    print(f"  mean_abs_control_phase_err_rad: {summary.get('mean_abs_control_phase_err_rad')}")
    print(f"  observer_timing_drift_ns_per_s: {summary.get('observer_timing_drift_ns_per_s')}")
    print(f"  observer_timing_drift_samples_per_s: {summary.get('observer_timing_drift_samples_per_s')}")
    print(f"  control_reference_walk_hz: {summary.get('control_reference_walk_hz')}")
    print(f"  dac_updates_applied: {summary.get('dac_updates_applied')}")
    print(f"  dac_update_fraction: {summary.get('dac_update_fraction')}")
    print(f"  csv_path: {summary.get('csv_path')}")


async def run_one(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.nsamp) <= 0:
        raise ValueError("nsamp must be positive.")
    if args.mode == "pid" and base.cvh is None:
        raise RuntimeError("clock_vcxo_helper.py import failed, but mode=pid requires DAC control access.")

    code_min = base.pct_to_code(float(args.dac_min_pct))
    code_max = base.pct_to_code(float(args.dac_max_pct))
    if code_min >= code_max:
        raise ValueError("DAC range invalid: ensure dac_min_pct < dac_max_pct.")

    if args.dac_center is None:
        dac_center = int(round((code_min + code_max) / 2.0))
    else:
        dac_center = base.clamp_int(int(args.dac_center), code_min, code_max)

    nominal_loop_period_s = float(args.nsamp) / float(args.sample_rate)
    queue_get_timeout_s = max(1.0, nominal_loop_period_s)

    logger = CsvLogger(Path(args.csv_out))
    device = None
    producer_task: asyncio.Task[None] | None = None
    clock_obj = None

    rows: list[dict[str, Any]] = []
    control_tone_offset_hz: float | None = None if args.target_offset_hz is None else float(args.target_offset_hz)
    observer_tone_offset_hz: float | None = (
        None if args.observer_fixed_offset_hz is None else float(args.observer_fixed_offset_hz)
    )
    state = ControllerState(
        dac_code_cmd=int(dac_center),
        dac_code_rb=None,
        dac_volts=base.code_to_volts(int(dac_center), float(args.vdd_dac)),
    )
    control_unwrapper = base.SlipAwareUnwrapper()
    observer_unwrapper = base.SlipAwareUnwrapper()

    update_idx = 0
    bootstrap_blocks: list[base.QueuedBlock] = []
    wall_start_s = float(time.time())

    try:
        device = base.open_sdr(args)
        if float(args.settle_seconds) > 0.0:
            await asyncio.sleep(float(args.settle_seconds))

        block_queue: asyncio.Queue[base.QueuedBlock | base.StreamSentinel] = asyncio.Queue(
            maxsize=base.default_async_queue_blocks()
        )
        producer_task = asyncio.create_task(base.stream_iq_blocks(device, int(args.nsamp), block_queue))

        if args.mode == "pid":
            clock_obj = base.cvh.ClockVCXO()
            clock_obj.open()
            if not args.skip_clock_init:
                clock_obj.init_clock_vcxo_28p8mhz(apr_ppm=int(args.apr_ppm))
            if args.apply_start_dac:
                rb0 = int(clock_obj.set_dac_code(int(dac_center), settle_s=float(args.fine_settle_s)))
                state.dac_code_cmd = int(rb0)
                state.dac_code_rb = int(rb0)
                state.dac_volts = float(clock_obj.dac_code_to_volts(int(rb0)))

        if observer_tone_offset_hz is None and control_tone_offset_hz is not None:
            observer_tone_offset_hz = float(control_tone_offset_hz)

        stop_requested = False
        while not stop_requested:
            if args.run_seconds is not None and (float(time.time()) - wall_start_s) >= float(args.run_seconds):
                break
            if args.max_updates is not None and update_idx >= int(args.max_updates):
                break

            try:
                item = await asyncio.wait_for(block_queue.get(), timeout=queue_get_timeout_s)
            except asyncio.TimeoutError:
                if producer_task is not None and producer_task.done():
                    exc = producer_task.exception()
                    if exc is not None:
                        raise RuntimeError("Async SDR stream task failed.") from exc
                    break
                continue

            if isinstance(item, base.StreamSentinel):
                if item.error is not None:
                    raise RuntimeError("Async SDR stream terminated unexpectedly.") from item.error
                break

            if control_tone_offset_hz is None:
                bootstrap_blocks.append(item)
                if len(bootstrap_blocks) < max(1, int(args.tone_estimate_blocks)):
                    continue
                stacked = np.stack(
                    [
                        np.asarray(b.iq_block, dtype=np.complex64).reshape(-1)
                        - np.mean(np.asarray(b.iq_block, dtype=np.complex64).reshape(-1))
                        for b in bootstrap_blocks
                    ],
                    axis=0,
                )
                control_tone_offset_hz = base.estimate_tone_offset_hz(
                    stacked,
                    sample_rate_hz=float(args.sample_rate),
                    dc_exclusion_hz=float(args.dc_exclusion_hz),
                )
                if observer_tone_offset_hz is None:
                    observer_tone_offset_hz = float(control_tone_offset_hz)

            work_items: list[base.QueuedBlock]
            if bootstrap_blocks:
                work_items = list(bootstrap_blocks)
                bootstrap_blocks.clear()
            else:
                work_items = [item]

            for blk in work_items:
                control_ref_hz = float(control_tone_offset_hz)
                observer_ref_hz = float(
                    observer_tone_offset_hz if observer_tone_offset_hz is not None else control_tone_offset_hz
                )
                control_meas = base.measure_block(
                    queued_block=blk,
                    sample_rate_hz=float(args.sample_rate),
                    tone_offset_hz=float(control_ref_hz),
                    update_idx=int(update_idx),
                    unwrapper=control_unwrapper,
                )
                observer_meas = base.measure_block(
                    queued_block=blk,
                    sample_rate_hz=float(args.sample_rate),
                    tone_offset_hz=float(observer_ref_hz),
                    update_idx=int(update_idx),
                    unwrapper=observer_unwrapper,
                )
                if args.run_seconds is not None and control_meas.elapsed_s > float(args.run_seconds):
                    stop_requested = True
                    break

                control_rf_hz = float(args.center_freq) + float(control_ref_hz)
                observer_rf_hz = float(args.center_freq) + float(observer_ref_hz)
                if (
                    state.observer_timing_proxy_ref_phase_rad is None
                    and np.isfinite(observer_rf_hz)
                    and abs(observer_rf_hz) > 1e-6
                ):
                    state.observer_timing_proxy_ref_phase_rad = float(observer_meas.unwrapped_phase_rad)
                observer_timing_proxy_ns = base.phase_to_timing_proxy_ns(
                    unwrapped_phase_rad=float(observer_meas.unwrapped_phase_rad),
                    ref_phase_rad=state.observer_timing_proxy_ref_phase_rad,
                    rf_hz=float(observer_rf_hz),
                )

                control_phase_err_raw = base.wrap_to_pi(
                    float(control_meas.wrapped_phase_rad) - float(args.phase_setpoint_rad)
                )
                control_phase_err_filt = base.circular_phase_lpf_update(
                    float(control_phase_err_raw),
                    float(args.phase_filter_alpha),
                    state.control_phase_err_lpf,
                )
                observer_phase_err_raw = base.wrap_to_pi(
                    float(observer_meas.wrapped_phase_rad) - float(args.phase_setpoint_rad)
                )

                control_coherence_ok = bool(
                    np.isfinite(control_meas.coherence) and (float(control_meas.coherence) >= float(args.coh_min))
                )
                control_pullin_quality_ok = bool(
                    np.isfinite(control_meas.pullin_quality)
                    and (float(control_meas.pullin_quality) >= float(args.pullin_quality_min))
                )
                observer_quality_ok = bool(
                    np.isfinite(observer_meas.coherence)
                    and (float(observer_meas.coherence) >= float(args.coh_min))
                    and np.isfinite(observer_meas.pullin_quality)
                    and (float(observer_meas.pullin_quality) >= float(args.pullin_quality_min))
                )
                control_quality_ok = bool(control_coherence_ok and control_pullin_quality_ok)

                dt_update_s = float(control_meas.dt_update_s)
                if dt_update_s <= 0.0 or (not np.isfinite(dt_update_s)):
                    dt_update_s = nominal_loop_period_s

                decision = compute_control_decision(
                    current_code=int(state.dac_code_cmd),
                    phase_err_filt_rad=float(control_phase_err_filt),
                    current_integral_rad_s=float(state.phase_err_int_rad_s),
                    dt_update_s=float(dt_update_s),
                    nsamp=int(args.nsamp),
                    dac_center=int(dac_center),
                    phase_frac_gain_codes=float(args.phase_frac_gain_codes),
                    integral_gain_codes_per_rad_s=float(args.integral_gain_codes_per_rad_s),
                    integral_limit_codes=float(args.integral_limit_codes),
                    control_sign=float(args.control_sign),
                    code_min=int(code_min),
                    code_max=int(code_max),
                    max_dac_step_per_update=int(args.max_dac_step_per_update),
                    quality_ok=bool(control_quality_ok),
                )
                if control_quality_ok:
                    state.phase_err_int_rad_s = float(decision.phase_err_int_rad_s)

                control_tone_track_step_hz = 0.0
                if (
                    float(args.control_track_alpha) > 0.0
                    and np.isfinite(control_meas.coherence)
                    and (float(control_meas.coherence) >= float(args.control_track_min_coh))
                    and np.isfinite(control_meas.freq_err_intrablock_hz)
                ):
                    control_tone_track_step_hz = base.clamp(
                        float(args.control_track_alpha) * float(control_meas.freq_err_intrablock_hz),
                        -float(args.control_track_max_step_hz),
                        float(args.control_track_max_step_hz),
                    )
                    control_tone_offset_hz = float(control_ref_hz + control_tone_track_step_hz)
                else:
                    control_tone_offset_hz = float(control_ref_hz)

                action = "pi_hold"
                if not control_coherence_ok:
                    action = "hold_low_control_coherence"
                elif not control_pullin_quality_ok:
                    action = "hold_low_control_pullin_quality"

                dac_step_applied = 0
                dac_applied = False
                if args.mode == "pid":
                    if control_quality_ok and decision.dac_code_proposed != int(state.dac_code_cmd):
                        if clock_obj is not None:
                            rb = int(
                                clock_obj.set_dac_code(
                                    int(decision.dac_code_proposed),
                                    settle_s=float(args.fine_settle_s),
                                )
                            )
                        else:
                            rb = int(decision.dac_code_proposed)
                        dac_step_applied = int(rb - int(state.dac_code_cmd))
                        state.dac_code_cmd = int(rb)
                        state.dac_code_rb = int(rb)
                        if clock_obj is not None:
                            state.dac_volts = float(clock_obj.dac_code_to_volts(int(rb)))
                        else:
                            state.dac_volts = base.code_to_volts(int(rb), float(args.vdd_dac))
                        state.control_updates_applied += 1
                        dac_applied = True
                        action = "pi_apply"
                    else:
                        state.held_updates += 1
                        if not control_quality_ok:
                            state.low_quality_holds += 1
                else:
                    state.held_updates += 1
                    if control_quality_ok:
                        action = "monitor_pi"
                    else:
                        state.low_quality_holds += 1

                row: dict[str, Any] = {
                    "unix_time_s": float(control_meas.unix_time_s),
                    "elapsed_s": float(control_meas.elapsed_s),
                    "update_idx": int(update_idx),
                    "mode": str(args.mode),
                    "action": str(action),
                    "sample_start": int(control_meas.sample_start),
                    "nsamp": int(args.nsamp),
                    "sample_rate_hz": float(args.sample_rate),
                    "center_freq_hz": float(args.center_freq),
                    "control_tone_offset_hz": float(control_ref_hz),
                    "control_tone_track_step_hz": float(control_tone_track_step_hz),
                    "observer_tone_offset_hz": float(observer_ref_hz),
                    "control_rf_reference_hz": float(control_rf_hz),
                    "observer_rf_reference_hz": float(observer_rf_hz),
                    "control_mix_real": float(control_meas.mix_real),
                    "control_mix_imag": float(control_meas.mix_imag),
                    "control_mix_abs": float(control_meas.mix_abs),
                    "observer_mix_real": float(observer_meas.mix_real),
                    "observer_mix_imag": float(observer_meas.mix_imag),
                    "observer_mix_abs": float(observer_meas.mix_abs),
                    "control_phase_wrapped_rad": float(control_meas.wrapped_phase_rad),
                    "control_phase_unwrapped_rad": float(control_meas.unwrapped_phase_rad),
                    "observer_phase_wrapped_rad": float(observer_meas.wrapped_phase_rad),
                    "observer_phase_unwrapped_rad": float(observer_meas.unwrapped_phase_rad),
                    "phase_setpoint_rad": float(args.phase_setpoint_rad),
                    "control_phase_err_raw_rad": float(control_phase_err_raw),
                    "control_phase_err_filt_rad": float(control_phase_err_filt),
                    "observer_phase_err_raw_rad": float(observer_phase_err_raw),
                    "control_phase_fraction_cycles_per_sample": float(decision.phase_fraction_cycles_per_sample),
                    "control_coherence": float(control_meas.coherence),
                    "observer_coherence": float(observer_meas.coherence),
                    "control_pullin_quality": float(control_meas.pullin_quality),
                    "observer_pullin_quality": float(observer_meas.pullin_quality),
                    "control_quality_ok": int(control_quality_ok),
                    "observer_quality_ok": int(observer_quality_ok),
                    "control_amp_rms": float(control_meas.amp_rms),
                    "observer_amp_rms": float(observer_meas.amp_rms),
                    "control_freq_err_intrablock_hz": float(control_meas.freq_err_intrablock_hz),
                    "control_freq_err_update_hz": float(control_meas.freq_err_update_hz),
                    "observer_freq_err_intrablock_hz": float(observer_meas.freq_err_intrablock_hz),
                    "observer_freq_err_update_hz": float(observer_meas.freq_err_update_hz),
                    "observer_timing_proxy_ns": float(observer_timing_proxy_ns),
                    "control_slip_cycles": int(control_meas.slip_cycles),
                    "observer_slip_cycles": int(observer_meas.slip_cycles),
                    "control_dphi_wrapped_rad": float(control_meas.dphi_wrapped_rad),
                    "control_dphi_used_rad": float(control_meas.dphi_used_rad),
                    "control_dphi_expected_rad": float(control_meas.dphi_expected_rad),
                    "observer_dphi_wrapped_rad": float(observer_meas.dphi_wrapped_rad),
                    "observer_dphi_used_rad": float(observer_meas.dphi_used_rad),
                    "observer_dphi_expected_rad": float(observer_meas.dphi_expected_rad),
                    "phase_err_int_rad_s": float(state.phase_err_int_rad_s),
                    "u_p_codes": float(decision.p_codes),
                    "u_i_codes": float(decision.i_codes),
                    "dac_center_code": int(dac_center),
                    "dac_target_float": float(decision.dac_target_float),
                    "dac_target_code": int(decision.dac_target_code),
                    "dac_code_cmd": int(state.dac_code_cmd),
                    "dac_code_rb": "" if state.dac_code_rb is None else int(state.dac_code_rb),
                    "dac_code_proposed": int(decision.dac_code_proposed),
                    "dac_volts": float(state.dac_volts),
                    "dac_step_req_codes": int(decision.dac_step_req_codes),
                    "dac_step_applied_codes": int(dac_step_applied),
                    "phase_frac_gain_codes": float(args.phase_frac_gain_codes),
                    "integral_gain_codes_per_rad_s": float(args.integral_gain_codes_per_rad_s),
                    "integral_limit_codes": float(args.integral_limit_codes),
                    "max_dac_step_per_update": int(args.max_dac_step_per_update),
                    "notes": str(args.notes),
                    "dac_applied": bool(dac_applied),
                }
                rows.append(row)
                logger.write(row)
                update_idx += 1
                if args.max_updates is not None and update_idx >= int(args.max_updates):
                    stop_requested = True
                    break
    finally:
        if producer_task is not None:
            try:
                if not producer_task.done():
                    producer_task.cancel()
                    try:
                        await asyncio.wait_for(producer_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        pass
                else:
                    await producer_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if clock_obj is not None:
            try:
                clock_obj.close()
            except Exception:
                pass
        logger.close()

    final_observer_ref_hz = (
        float(observer_tone_offset_hz)
        if observer_tone_offset_hz is not None
        else float(control_tone_offset_hz if control_tone_offset_hz is not None else float("nan"))
    )
    summary = compute_summary(
        rows=rows,
        args=args,
        observer_tone_offset_hz=float(final_observer_ref_hz),
    )
    summary["csv_path"] = str(Path(args.csv_out).resolve())
    summary["summary_path"] = str(Path(args.summary_out).resolve())
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Notebook-style live DPLL with integral action and dual phase references."
    )

    p.add_argument("--mode", choices=["monitor", "pid"], default="pid")
    p.add_argument("--notes", type=str, default="")

    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--center-freq", type=float, default=100e6)
    p.add_argument("--sample-rate", type=float, default=2.4e6)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--settle-seconds", type=float, default=0.25)
    p.add_argument("--nsamp", type=int, default=32768)
    p.add_argument("--run-seconds", type=float, default=120.0)
    p.add_argument("--max-updates", type=int, default=None)

    p.add_argument("--target-offset-hz", type=float, default=None)
    p.add_argument(
        "--observer-fixed-offset-hz",
        type=float,
        default=None,
        help="Fixed observer-reference tone offset (Hz). If omitted, uses the initial control reference.",
    )
    p.add_argument("--tone-estimate-blocks", type=int, default=8)
    p.add_argument("--dc-exclusion-hz", type=float, default=5e3)
    p.add_argument(
        "--control-track-alpha",
        type=float,
        default=0.02,
        help="Slow tracking gain for the control reference. Observer reference stays fixed.",
    )
    p.add_argument(
        "--control-track-min-coh",
        type=float,
        default=0.05,
        help="Minimum control coherence required before adapting the control reference.",
    )
    p.add_argument(
        "--control-track-max-step-hz",
        type=float,
        default=1000.0,
        help="Per-update clamp for control-reference tracking.",
    )

    p.add_argument("--phase-setpoint-rad", type=float, default=0.0)
    p.add_argument("--phase-filter-alpha", type=float, default=0.85)
    p.add_argument("--coh-min", type=float, default=0.025)
    p.add_argument("--pullin-quality-min", type=float, default=0.10)
    p.add_argument(
        "--phase-frac-gain-codes",
        type=float,
        default=8000000.0,
        help="Proportional gain from phase-fraction (cycles/sample) to DAC codes.",
    )
    p.add_argument(
        "--integral-gain-codes-per-rad-s",
        type=float,
        default=20.0,
        help="Integral gain from accumulated phase error (rad*s) to DAC codes.",
    )
    p.add_argument(
        "--integral-limit-codes",
        type=float,
        default=256.0,
        help="Clamp for the integral contribution expressed in DAC codes.",
    )
    p.add_argument("--control-sign", type=float, default=1.0)

    p.add_argument("--dac-center", type=int, default=None)
    p.add_argument("--dac-min-pct", type=float, default=10.0)
    p.add_argument("--dac-max-pct", type=float, default=90.0)
    p.add_argument("--max-dac-step-per-update", type=int, default=32)
    p.add_argument("--vdd-dac", type=float, default=base.VDD_DAC_DEFAULT)
    p.add_argument("--apr-ppm", type=int, default=120)
    p.add_argument("--skip-clock-init", action="store_true")
    p.add_argument(
        "--apply-start-dac",
        dest="apply_start_dac",
        action="store_true",
        default=True,
        help="Apply dac-center at run start in pid mode.",
    )
    p.add_argument(
        "--no-apply-start-dac",
        dest="apply_start_dac",
        action="store_false",
        help="Do not write dac-center at run start in pid mode.",
    )
    p.add_argument("--fine-settle-s", type=float, default=0.0)

    base_dir = Path(__file__).resolve().parent
    p.add_argument(
        "--csv-out",
        type=Path,
        default=base_dir / "notebook_style_dpll_integral_dualref.csv",
        help="CSV output path.",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=base_dir / "notebook_style_dpll_integral_dualref_summary.json",
        help="Summary JSON output path.",
    )
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> int:
    run_controller_self_checks()
    parser = build_arg_parser()
    args = parser.parse_args()

    summary = asyncio.run(run_one(args))
    summary_path = Path(args.summary_out)
    base.ensure_parent_dir(summary_path)
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    if not args.quiet:
        print_run_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
