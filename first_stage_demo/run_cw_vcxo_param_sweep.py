#!/usr/bin/env python3
"""Run a focused hardware parameter sweep for cw_vcxo_discipline.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "first_stage_demo"
BINARY_MAGIC = b"CWVXCOBIN1\n"


@dataclass(frozen=True)
class SweepCase:
    name: str
    phase_gain: float
    phase_max_hz: float
    phase_alpha: float
    phase_error_alpha: float
    tracking_deadband_hz: float
    base_filter_alpha: float
    base_lock_threshold_hz: float
    base_lock_dwell_updates: int
    startup_min_coherence: float
    startup_min_valid_blocks: int
    startup_gain: float = 0.65
    slew_gain_hz_per_rad: float | None = None
    slew_max_freq_hz: float | None = None
    slew_enter_error_rad: float = 0.75
    slew_exit_error_rad: float = 0.25
    recovery_enter_dwell_updates: int = 1
    recovery_exit_dwell_updates: int = 1
    recovery_min_dwell_updates: int = 0
    phase_integral_gain_hz_per_rad_s: float = 0.0
    phase_integral_max_hz: float = 0.0


def default_cases() -> list[SweepCase]:
    return [
        SweepCase("balanced", 3.0, 8.0, 0.35, 1.0, 5.0, 0.98, 8.0, 250, 0.90, 40),
        SweepCase("lower_phase_gain", 2.0, 6.0, 0.35, 1.0, 5.0, 0.98, 8.0, 250, 0.90, 40),
        SweepCase("higher_phase_gain", 4.0, 8.0, 0.35, 1.0, 5.0, 0.98, 8.0, 250, 0.90, 40),
        SweepCase("faster_base_gate", 3.0, 8.0, 0.35, 1.0, 5.0, 0.95, 10.0, 150, 0.90, 40),
        SweepCase("stricter_base_gate", 3.0, 8.0, 0.35, 1.0, 5.0, 0.98, 6.0, 250, 0.90, 40),
        SweepCase("lower_startup_gain", 3.0, 8.0, 0.35, 1.0, 5.0, 0.98, 8.0, 250, 0.90, 40, 0.50),
    ]


def phase_tune_cases() -> list[SweepCase]:
    return [
        SweepCase("baseline_exitdwell", 3.0, 8.0, 0.35, 1.0, 5.0, 0.98, 6.0, 250, 0.90, 40),
        SweepCase("phase_err_alpha_0p8", 3.0, 8.0, 0.35, 0.80, 5.0, 0.98, 6.0, 250, 0.90, 40),
        SweepCase("phase_err_alpha_0p5", 3.0, 8.0, 0.35, 0.50, 5.0, 0.98, 6.0, 250, 0.90, 40),
        SweepCase("phase_err_alpha_0p3", 3.0, 8.0, 0.35, 0.30, 5.0, 0.98, 6.0, 250, 0.90, 40),
        SweepCase("phase_err_alpha_0p5_gain4", 4.0, 10.0, 0.35, 0.50, 5.0, 0.98, 6.0, 250, 0.90, 40),
        SweepCase("phase_err_alpha_0p3_gain4", 4.0, 10.0, 0.35, 0.30, 5.0, 0.98, 6.0, 250, 0.90, 40),
    ]


def phase_ff_grid_cases() -> list[SweepCase]:
    cases: list[SweepCase] = []
    for phase_gain in (0.5, 1.0, 1.5, 2.0, 3.0):
        for phase_max_hz in (2.0, 4.0, 6.0, 8.0):
            gain_label = str(phase_gain).replace(".", "p")
            max_label = str(phase_max_hz).replace(".", "p")
            cases.append(
                SweepCase(
                    f"ff_gain{gain_label}_max{max_label}",
                    phase_gain,
                    phase_max_hz,
                    0.35,
                    1.0,
                    5.0,
                    0.98,
                    6.0,
                    250,
                    0.90,
                    40,
                )
            )
    return cases


def phase_smoothing_alpha_cases() -> list[SweepCase]:
    cases: list[SweepCase] = []
    for phase_max_hz in (6.0, 8.0):
        for phase_alpha in (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
            max_label = str(phase_max_hz).replace(".", "p")
            alpha_label = str(phase_alpha).replace(".", "p")
            cases.append(
                SweepCase(
                    f"alpha{alpha_label}_gain1p5_max{max_label}",
                    1.5,
                    phase_max_hz,
                    phase_alpha,
                    1.0,
                    5.0,
                    0.98,
                    6.0,
                    250,
                    0.90,
                    40,
                )
            )
    return cases


def two_mode_cases() -> list[SweepCase]:
    return [
        SweepCase(
            "baseline_current",
            0.35,
            1.5,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=1.5,
            slew_max_freq_hz=8.0,
            slew_enter_error_rad=1.25,
            slew_exit_error_rad=0.35,
        ),
        SweepCase(
            "soft_recover",
            0.35,
            1.0,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.75,
            slew_max_freq_hz=3.0,
            slew_enter_error_rad=2.0,
            slew_exit_error_rad=0.60,
        ),
        SweepCase(
            "softer_recover",
            0.25,
            0.75,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.50,
            slew_max_freq_hz=2.0,
            slew_enter_error_rad=2.0,
            slew_exit_error_rad=0.60,
        ),
        SweepCase(
            "higher_enter",
            0.35,
            1.0,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.75,
            slew_max_freq_hz=3.0,
            slew_enter_error_rad=2.50,
            slew_exit_error_rad=0.75,
        ),
        SweepCase(
            "hold_dominant",
            0.50,
            1.5,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.75,
            slew_max_freq_hz=3.0,
            slew_enter_error_rad=2.50,
            slew_exit_error_rad=0.75,
        ),
        SweepCase(
            "low_hold",
            0.20,
            0.75,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.50,
            slew_max_freq_hz=2.0,
            slew_enter_error_rad=2.50,
            slew_exit_error_rad=0.75,
        ),
        SweepCase(
            "mid_recover",
            0.35,
            1.5,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=1.0,
            slew_max_freq_hz=4.0,
            slew_enter_error_rad=2.0,
            slew_exit_error_rad=0.60,
        ),
        SweepCase(
            "soft_alpha0p2",
            0.35,
            1.0,
            0.20,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.75,
            slew_max_freq_hz=3.0,
            slew_enter_error_rad=2.0,
            slew_exit_error_rad=0.60,
        ),
    ]


def two_mode_focused_cases() -> list[SweepCase]:
    return [
        SweepCase(
            "hold_dom_dwell",
            0.50,
            1.5,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.75,
            slew_max_freq_hz=3.0,
            slew_enter_error_rad=2.50,
            slew_exit_error_rad=0.75,
            recovery_enter_dwell_updates=4,
            recovery_exit_dwell_updates=8,
            recovery_min_dwell_updates=25,
        ),
        SweepCase(
            "hold_dom_stronger_dwell",
            0.50,
            1.5,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.75,
            slew_max_freq_hz=3.0,
            slew_enter_error_rad=2.50,
            slew_exit_error_rad=0.75,
            recovery_enter_dwell_updates=8,
            recovery_exit_dwell_updates=12,
            recovery_min_dwell_updates=50,
        ),
        SweepCase(
            "low_hold_dwell",
            0.20,
            0.75,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.50,
            slew_max_freq_hz=2.0,
            slew_enter_error_rad=2.50,
            slew_exit_error_rad=0.75,
            recovery_enter_dwell_updates=4,
            recovery_exit_dwell_updates=8,
            recovery_min_dwell_updates=25,
        ),
        SweepCase(
            "higher_enter_dwell",
            0.35,
            1.0,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.75,
            slew_max_freq_hz=3.0,
            slew_enter_error_rad=3.00,
            slew_exit_error_rad=1.00,
            recovery_enter_dwell_updates=4,
            recovery_exit_dwell_updates=8,
            recovery_min_dwell_updates=25,
        ),
        SweepCase(
            "hold_dom_low_recover_clamp",
            0.50,
            1.5,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.60,
            slew_max_freq_hz=2.0,
            slew_enter_error_rad=2.50,
            slew_exit_error_rad=0.75,
            recovery_enter_dwell_updates=4,
            recovery_exit_dwell_updates=8,
            recovery_min_dwell_updates=25,
        ),
        SweepCase(
            "hold_dom_fast_alpha_dwell",
            0.50,
            1.5,
            0.50,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.75,
            slew_max_freq_hz=3.0,
            slew_enter_error_rad=2.50,
            slew_exit_error_rad=0.75,
            recovery_enter_dwell_updates=4,
            recovery_exit_dwell_updates=8,
            recovery_min_dwell_updates=25,
        ),
    ]


def two_mode_bias_cases() -> list[SweepCase]:
    return [
        SweepCase(
            "hold_dom_bias_slow",
            0.50,
            1.5,
            0.50,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.75,
            slew_max_freq_hz=3.0,
            slew_enter_error_rad=3.00,
            slew_exit_error_rad=1.00,
            recovery_enter_dwell_updates=8,
            recovery_exit_dwell_updates=12,
            recovery_min_dwell_updates=50,
            phase_integral_gain_hz_per_rad_s=0.02,
            phase_integral_max_hz=0.75,
        ),
        SweepCase(
            "hold_dom_bias_medium",
            0.50,
            1.5,
            0.50,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.75,
            slew_max_freq_hz=3.0,
            slew_enter_error_rad=3.00,
            slew_exit_error_rad=1.00,
            recovery_enter_dwell_updates=8,
            recovery_exit_dwell_updates=12,
            recovery_min_dwell_updates=50,
            phase_integral_gain_hz_per_rad_s=0.05,
            phase_integral_max_hz=1.00,
        ),
        SweepCase(
            "low_hold_bias_slow",
            0.20,
            0.75,
            0.35,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.50,
            slew_max_freq_hz=2.0,
            slew_enter_error_rad=3.00,
            slew_exit_error_rad=1.00,
            recovery_enter_dwell_updates=8,
            recovery_exit_dwell_updates=12,
            recovery_min_dwell_updates=50,
            phase_integral_gain_hz_per_rad_s=0.02,
            phase_integral_max_hz=0.75,
        ),
        SweepCase(
            "rare_recover_bias",
            0.50,
            1.5,
            0.50,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.60,
            slew_max_freq_hz=2.0,
            slew_enter_error_rad=3.10,
            slew_exit_error_rad=1.00,
            recovery_enter_dwell_updates=12,
            recovery_exit_dwell_updates=16,
            recovery_min_dwell_updates=75,
            phase_integral_gain_hz_per_rad_s=0.03,
            phase_integral_max_hz=0.75,
        ),
    ]


def two_mode_authority_cases() -> list[SweepCase]:
    return [
        SweepCase(
            "hold_authority_med_assist",
            0.60,
            2.0,
            0.55,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.25,
            slew_max_freq_hz=0.75,
            slew_enter_error_rad=3.10,
            slew_exit_error_rad=1.00,
            recovery_enter_dwell_updates=12,
            recovery_exit_dwell_updates=16,
            recovery_min_dwell_updates=75,
            phase_integral_gain_hz_per_rad_s=0.04,
            phase_integral_max_hz=1.00,
        ),
        SweepCase(
            "hold_authority_high_assist",
            0.75,
            2.5,
            0.55,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.35,
            slew_max_freq_hz=1.00,
            slew_enter_error_rad=3.10,
            slew_exit_error_rad=1.00,
            recovery_enter_dwell_updates=12,
            recovery_exit_dwell_updates=16,
            recovery_min_dwell_updates=75,
            phase_integral_gain_hz_per_rad_s=0.04,
            phase_integral_max_hz=1.00,
        ),
        SweepCase(
            "bias_authority_med",
            0.60,
            2.0,
            0.55,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.25,
            slew_max_freq_hz=0.75,
            slew_enter_error_rad=3.10,
            slew_exit_error_rad=1.00,
            recovery_enter_dwell_updates=12,
            recovery_exit_dwell_updates=16,
            recovery_min_dwell_updates=75,
            phase_integral_gain_hz_per_rad_s=0.08,
            phase_integral_max_hz=1.25,
        ),
        SweepCase(
            "rare_recover_more_hold",
            0.70,
            2.5,
            0.60,
            1.0,
            5.0,
            0.98,
            6.0,
            100,
            0.90,
            40,
            slew_gain_hz_per_rad=0.20,
            slew_max_freq_hz=0.50,
            slew_enter_error_rad=3.12,
            slew_exit_error_rad=1.25,
            recovery_enter_dwell_updates=16,
            recovery_exit_dwell_updates=20,
            recovery_min_dwell_updates=100,
            phase_integral_gain_hz_per_rad_s=0.06,
            phase_integral_max_hz=1.25,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-seconds", type=float, default=90.0)
    parser.add_argument("--zoom-start-s", type=float, default=75.0)
    parser.add_argument("--zoom-duration-s", type=float, default=1.0 / 6.0)
    parser.add_argument("--timeout-margin-s", type=float, default=55.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--label", default=time.strftime("cw_vcxo_sweep_%Y%m%d_%H%M%S"))
    parser.add_argument("--logs-dir", type=Path, default=DEMO / "logs")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit the default case list. 0 runs all cases.")
    parser.add_argument(
        "--case-set",
        choices=(
            "default",
            "phase_tune",
            "phase_ff_grid",
            "phase_smoothing_alpha",
            "two_mode",
            "two_mode_focused",
            "two_mode_bias",
            "two_mode_authority",
        ),
        default="default",
        help="Named focused case set to run.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def read_binary_metrics(path: Path, zoom_start_s: float, zoom_duration_s: float) -> dict[str, Any]:
    fields = [
        "t_rel_s",
        "phase_rad",
        "unwrapped_phase_rad",
        "f_off_hz",
        "control_base_cmd",
        "dac_code_cmd",
        "coherence",
        "phase_hold_active",
        "base_phase_hold_allowed",
        "base_lock_counter",
        "phase_hold_exit_counter",
        "phase_error_rad",
        "phase_error_physical_rad",
        "phase_error_cycle_rad",
        "phase_error_branch_residual_rad",
        "phase_error_physical_residual_rad",
        "phase_cycle_branch_reject_count",
        "phase_cycle_branch_reject_detected",
        "phase_cycle_slip_candidate_detected",
        "phase_cycle_slip_dwell_counter",
        "phase_cycle_slip_count",
        "phase_cycle_slip_detected",
        "phase_hold_freq_setpoint_hz",
        "phase_hold_cmd_offset",
        "phase_integral_hz",
        "base_error_filt_hz",
    ]
    data: dict[str, list[float]] = {name: [] for name in fields}
    string_counts: dict[str, dict[str, int]] = {
        "status": {},
        "phase_control_mode": {},
        "phase_cycle_error_source": {},
        "gate_reason": {},
    }
    truncated = False
    rows = 0
    phase_mode_transitions = 0
    last_phase_mode: str | None = None
    with path.open("rb") as f:
        if f.readline() != BINARY_MAGIC:
            raise ValueError(f"{path} is not a CW VXCO binary log")
        header_len_raw = f.read(4)
        if len(header_len_raw) != 4:
            raise ValueError(f"{path} has a truncated header")
        header_len = struct.unpack("<I", header_len_raw)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        numeric_fields = list(header["numeric_fields"])
        string_fields = list(header["string_fields"])
        numeric_struct = struct.Struct("<" + ("d" * len(numeric_fields)))
        field_index = {name: numeric_fields.index(name) for name in fields if name in numeric_fields}
        string_index = {
            name: string_fields.index(name)
            for name in string_counts
            if name in string_fields
        }
        while True:
            record_len_raw = f.read(4)
            if not record_len_raw:
                break
            if len(record_len_raw) != 4:
                truncated = True
                break
            record_len = struct.unpack("<I", record_len_raw)[0]
            payload = f.read(record_len)
            if len(payload) != record_len:
                truncated = True
                break
            values = numeric_struct.unpack_from(payload, 0)
            for name in fields:
                idx = field_index.get(name)
                data[name].append(values[idx] if idx is not None else math.nan)

            offset = numeric_struct.size
            strings: list[str] = []
            for _ in string_fields:
                if offset + 2 > len(payload):
                    break
                size = struct.unpack_from("<H", payload, offset)[0]
                offset += 2
                strings.append(payload[offset : offset + size].decode("utf-8", "replace"))
                offset += size
            for name, idx in string_index.items():
                if idx < len(strings):
                    counts = string_counts[name]
                    value = strings[idx]
                    counts[value] = counts.get(value, 0) + 1
                    if name == "phase_control_mode":
                        if last_phase_mode is not None and value != last_phase_mode:
                            phase_mode_transitions += 1
                        last_phase_mode = value
            rows += 1

    arr = {name: np.asarray(values, dtype=np.float64) for name, values in data.items()}
    t = arr["t_rel_s"]
    wrapped = arr["phase_rad"]
    unwrapped = arr["unwrapped_phase_rad"]

    def fit(mask: np.ndarray) -> tuple[float, float, int]:
        finite = mask & np.isfinite(t) & np.isfinite(unwrapped)
        if int(np.count_nonzero(finite)) < 3:
            return float("nan"), float("nan"), int(np.count_nonzero(finite))
        slope_rad_s, _ = np.polyfit(t[finite], unwrapped[finite], 1)
        span = float(np.nanmax(unwrapped[finite]) - np.nanmin(unwrapped[finite]))
        return float(slope_rad_s / (2.0 * math.pi)), span, int(np.count_nonzero(finite))

    t_min = float(np.nanmin(t)) if t.size else float("nan")
    t_max = float(np.nanmax(t)) if t.size else float("nan")
    duration_s = float(t_max - t_min) if t.size else float("nan")
    late_start = float(np.nanmin(t) + 0.5 * duration_s) if np.isfinite(duration_s) else float("nan")
    full_slope_hz, full_span_rad, _ = fit(np.ones_like(t, dtype=bool))
    late_slope_hz, late_span_rad, late_rows = fit(t >= late_start)
    zoom_start_actual_s = float(zoom_start_s)
    if np.isfinite(t_max) and zoom_start_actual_s < 0.0:
        zoom_start_actual_s = max(t_min, t_max + zoom_start_actual_s)
    zoom_mask = (t >= zoom_start_actual_s) & (t < zoom_start_actual_s + float(zoom_duration_s))
    zoom_slope_hz, zoom_span_rad, zoom_rows = fit(zoom_mask)
    zoom_wraps = 0
    zoom_wrapped = wrapped[zoom_mask & np.isfinite(wrapped)]
    if zoom_wrapped.size > 1:
        zoom_wraps = int(np.sum(np.abs(np.diff(zoom_wrapped)) > math.pi))
    wrapped_abs = np.abs(wrapped)
    wrapped_finite = np.isfinite(wrapped_abs)
    full_wrapped_abs_median_rad = (
        float(np.nanmedian(wrapped_abs[wrapped_finite])) if np.any(wrapped_finite) else float("nan")
    )
    first_half_wrapped = wrapped_abs[(t < late_start) & wrapped_finite]
    late_wrapped = wrapped_abs[(t >= late_start) & wrapped_finite]
    first_half_wrapped_abs_median_rad = (
        float(np.nanmedian(first_half_wrapped)) if first_half_wrapped.size else float("nan")
    )
    late_wrapped_abs_median_rad = (
        float(np.nanmedian(late_wrapped)) if late_wrapped.size else float("nan")
    )
    wrapped_abs_median_delta_rad = (
        float(late_wrapped_abs_median_rad - first_half_wrapped_abs_median_rad)
        if np.isfinite(late_wrapped_abs_median_rad) and np.isfinite(first_half_wrapped_abs_median_rad)
        else float("nan")
    )
    zoom_wrapped_abs_median_rad = (
        float(np.nanmedian(np.abs(zoom_wrapped))) if zoom_wrapped.size else float("nan")
    )

    def finite_stat(name: str, fn: Any) -> float:
        values = arr[name]
        finite = values[np.isfinite(values)]
        return float(fn(finite)) if finite.size else float("nan")

    phase_active = arr["phase_hold_active"]
    phase_active_finite = phase_active[np.isfinite(phase_active)]
    base_allowed = arr["base_phase_hold_allowed"]
    base_allowed_finite = base_allowed[np.isfinite(base_allowed)]
    phase_error = arr["phase_error_rad"]
    cycle_error = arr["phase_error_cycle_rad"]
    branch_residual = arr["phase_error_branch_residual_rad"]
    active_error_mask = (phase_active > 0.5) & np.isfinite(phase_error)
    zoom_active_error_mask = (
        (t >= zoom_start_actual_s)
        & (t < zoom_start_actual_s + float(zoom_duration_s))
        & (phase_active > 0.5)
        & np.isfinite(phase_error)
    )
    active_phase_error = phase_error[active_error_mask]
    zoom_active_phase_error = phase_error[zoom_active_error_mask]
    active_median_abs_phase_error_rad = (
        float(np.nanmedian(np.abs(active_phase_error)))
        if active_phase_error.size
        else float("nan")
    )
    active_rms_phase_error_rad = (
        float(np.sqrt(np.nanmean(active_phase_error * active_phase_error)))
        if active_phase_error.size
        else float("nan")
    )
    active_frac_abs_phase_error_lt_0p5 = (
        float(np.mean(np.abs(active_phase_error) < 0.5))
        if active_phase_error.size
        else float("nan")
    )
    zoom_median_abs_phase_error_rad = (
        float(np.nanmedian(np.abs(zoom_active_phase_error)))
        if zoom_active_phase_error.size
        else float("nan")
    )
    zoom_cycle_mask = zoom_mask & np.isfinite(cycle_error)
    zoom_cycle_error = cycle_error[zoom_cycle_mask]
    zoom_branch_residual = branch_residual[zoom_mask & np.isfinite(branch_residual)]
    phase_cycle_slip_count = arr["phase_cycle_slip_count"]
    finite_slips = phase_cycle_slip_count[np.isfinite(phase_cycle_slip_count)]
    cycle_slip_final = float(finite_slips[-1]) if finite_slips.size else float("nan")
    phase_cycle_branch_reject_count = arr["phase_cycle_branch_reject_count"]
    finite_branch_rejects = phase_cycle_branch_reject_count[
        np.isfinite(phase_cycle_branch_reject_count)
    ]
    cycle_branch_reject_final = (
        float(finite_branch_rejects[-1]) if finite_branch_rejects.size else float("nan")
    )
    cycle_branch_reject_rows = float(
        np.count_nonzero(np.nan_to_num(arr["phase_cycle_branch_reject_detected"]) > 0.5)
    )
    cycle_slip_candidate_rows = float(
        np.count_nonzero(np.nan_to_num(arr["phase_cycle_slip_candidate_detected"]) > 0.5)
    )
    zoom_dac = arr["dac_code_cmd"][zoom_mask & np.isfinite(arr["dac_code_cmd"])]
    zoom_dac_span = float(np.ptp(zoom_dac)) if zoom_dac.size else float("nan")
    zoom_ff = arr["phase_hold_cmd_offset"][zoom_mask & np.isfinite(arr["phase_hold_cmd_offset"])]
    zoom_ff_span = float(np.ptp(zoom_ff)) if zoom_ff.size else float("nan")
    zoom_cycle_span = (
        float(np.nanmax(zoom_cycle_error) - np.nanmin(zoom_cycle_error))
        if zoom_cycle_error.size
        else float("nan")
    )
    zoom_cycle_rms = (
        float(np.sqrt(np.nanmean(zoom_cycle_error * zoom_cycle_error)))
        if zoom_cycle_error.size
        else float("nan")
    )
    zoom_branch_rms = (
        float(np.sqrt(np.nanmean(zoom_branch_residual * zoom_branch_residual)))
        if zoom_branch_residual.size
        else float("nan")
    )
    phase_mode_counts = string_counts["phase_control_mode"]
    rows_for_mode = max(1, rows)
    phase_active_frac = float(np.mean(phase_active_finite)) if phase_active_finite.size else float("nan")
    phase_mode_idle_frac = float(phase_mode_counts.get("idle", 0) / rows_for_mode)
    phase_mode_hold_frac = float(phase_mode_counts.get("hold", 0) / rows_for_mode)
    phase_mode_recover_frac = float(phase_mode_counts.get("recover", 0) / rows_for_mode)
    transition_rate_hz = (
        float(phase_mode_transitions) / duration_s
        if np.isfinite(duration_s) and duration_s > 0.0
        else float("nan")
    )

    def finite_or(value: float, fallback: float) -> float:
        return float(value) if np.isfinite(value) else float(fallback)

    score = (
        0.35 * finite_or(zoom_span_rad, 20.0)
        + 0.20 * finite_or(zoom_cycle_rms, 10.0)
        + 0.10 * finite_or(zoom_branch_rms, 5.0)
        + 0.35 * abs(finite_or(late_slope_hz, 10.0))
        + 0.10 * abs(finite_or(zoom_slope_hz, 10.0))
        + 0.20 * finite_or(late_wrapped_abs_median_rad, math.pi)
        + 0.30 * max(0.0, finite_or(wrapped_abs_median_delta_rad, math.pi))
        + 0.75 * float(zoom_wraps)
        + 2.0 * max(0.0, finite_or(cycle_slip_final, 10.0))
        + 2.5 * phase_mode_recover_frac
        + 0.35 * finite_or(transition_rate_hz, 20.0)
        + 0.05 * finite_or(zoom_dac_span, 10.0)
        + (1.0 if phase_active_frac < 0.8 else 0.0)
        + (10.0 if zoom_rows < max(10, int(0.5 * float(zoom_duration_s) / 0.002)) else 0.0)
    )
    return {
        "rows": rows,
        "truncated": truncated,
        "duration_s": duration_s,
        "full_slope_hz": full_slope_hz,
        "full_span_rad": full_span_rad,
        "late_slope_hz": late_slope_hz,
        "late_span_rad": late_span_rad,
        "late_rows": late_rows,
        "zoom_start_s": zoom_start_actual_s,
        "requested_zoom_start_s": float(zoom_start_s),
        "zoom_slope_hz": zoom_slope_hz,
        "zoom_span_rad": zoom_span_rad,
        "zoom_rows": zoom_rows,
        "zoom_wraps": zoom_wraps,
        "full_wrapped_abs_median_rad": full_wrapped_abs_median_rad,
        "first_half_wrapped_abs_median_rad": first_half_wrapped_abs_median_rad,
        "late_wrapped_abs_median_rad": late_wrapped_abs_median_rad,
        "wrapped_abs_median_delta_rad": wrapped_abs_median_delta_rad,
        "zoom_wrapped_abs_median_rad": zoom_wrapped_abs_median_rad,
        "zoom_cycle_span_rad": zoom_cycle_span,
        "zoom_cycle_rms_rad": zoom_cycle_rms,
        "zoom_branch_rms_rad": zoom_branch_rms,
        "cycle_slip_count_final": cycle_slip_final,
        "cycle_branch_reject_count_final": cycle_branch_reject_final,
        "cycle_branch_reject_rows": cycle_branch_reject_rows,
        "cycle_slip_candidate_rows": cycle_slip_candidate_rows,
        "zoom_dac_span": zoom_dac_span,
        "zoom_phase_ff_cmd_span": zoom_ff_span,
        "phase_integral_min_hz": finite_stat("phase_integral_hz", np.min),
        "phase_integral_max_hz_observed": finite_stat("phase_integral_hz", np.max),
        "phase_integral_span_hz": finite_stat("phase_integral_hz", np.ptp),
        "phase_hold_active_frac": phase_active_frac,
        "phase_mode_idle_frac": phase_mode_idle_frac,
        "phase_mode_hold_frac": phase_mode_hold_frac,
        "phase_mode_recover_frac": phase_mode_recover_frac,
        "phase_mode_transitions": int(phase_mode_transitions),
        "phase_mode_transition_rate_hz": transition_rate_hz,
        "base_phase_hold_allowed_frac": float(np.mean(base_allowed_finite)) if base_allowed_finite.size else float("nan"),
        "base_lock_counter_max": finite_stat("base_lock_counter", np.max),
        "phase_hold_exit_counter_max": finite_stat("phase_hold_exit_counter", np.max),
        "active_median_abs_phase_error_rad": active_median_abs_phase_error_rad,
        "active_rms_phase_error_rad": active_rms_phase_error_rad,
        "active_frac_abs_phase_error_lt_0p5": active_frac_abs_phase_error_lt_0p5,
        "zoom_median_abs_phase_error_rad": zoom_median_abs_phase_error_rad,
        "zoom_active_rows": int(np.count_nonzero(zoom_active_error_mask)),
        "f_off_mean_hz": finite_stat("f_off_hz", np.mean),
        "f_off_std_hz": finite_stat("f_off_hz", np.std),
        "coherence_mean": finite_stat("coherence", np.mean),
        "coherence_min": finite_stat("coherence", np.min),
        "base_cmd_min": finite_stat("control_base_cmd", np.min),
        "base_cmd_max": finite_stat("control_base_cmd", np.max),
        "base_cmd_span": finite_stat("control_base_cmd", np.ptp),
        "dac_min": finite_stat("dac_code_cmd", np.min),
        "dac_max": finite_stat("dac_code_cmd", np.max),
        "score": float(score),
        "status_counts": string_counts["status"],
        "phase_control_mode_counts": string_counts["phase_control_mode"],
        "phase_cycle_error_source_counts": string_counts["phase_cycle_error_source"],
        "gate_reason_counts": string_counts["gate_reason"],
    }


def extract_startup_summary(stdout_path: Path) -> dict[str, Any]:
    text = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
    lines = [line for line in text.splitlines() if line.startswith("Initial frequency step")]
    applied = any("cmd " in line for line in lines)
    skipped = any("skipped" in line for line in lines)
    settled = any("settled" in line for line in lines)
    final_cmd = float("nan")
    for line in lines:
        matches = re.findall(r"cmd\s+([0-9.]+)->([0-9.]+)", line)
        if matches:
            final_cmd = safe_float(matches[-1][1])
    return {
        "startup_step_applied": applied,
        "startup_step_skipped": skipped,
        "startup_step_settled": settled,
        "startup_final_cmd": final_cmd,
        "startup_lines": " | ".join(lines),
    }


def base_command(case: SweepCase, args: argparse.Namespace, out_bin: Path) -> list[str]:
    command = [
        str(args.python),
        str(DEMO / "cw_vcxo_discipline.py"),
        "--mode",
        "closed_loop",
        "--sample-rate",
        "1024000",
        "--center-freq",
        "50120000",
        "--tone-bb",
        "-135750",
        "--block-size",
        "2048",
        "--duration-s",
        f"{float(args.run_seconds):g}",
        "--gain",
        "auto",
        "--cmd-start",
        "2050",
        "--vcxo-hz-per-command",
        "-4.5475",
        "--use-mcp4725",
        "--mcp4725-dither",
        "--pll-frequency-proportional-gain",
        "0.01",
        "--pll-frequency-integral-gain",
        "0.0",
        "--acquisition-frequency-proportional-gain",
        "0.02",
        "--acquisition-frequency-integral-gain",
        "0.0",
        "--acquisition-frequency-deadband-hz",
        "0.0",
        "--acquisition-command-slew-rate-per-s",
        "200",
        "--frequency-error-deadband-hz",
        f"{case.tracking_deadband_hz:g}",
        "--command-slew-rate-per-s",
        "50",
        "--base-frequency-integral-gain",
        "0.04",
        "--base-frequency-deadband-hz",
        "2.0",
        "--base-command-slew-rate-per-s",
        "0.40",
        "--base-error-filter-alpha",
        f"{case.base_filter_alpha:g}",
        "--base-min-coherence",
        "0.90",
        "--base-max-phase-residual-rad",
        "1.75",
        "--base-max-f-off-fraction",
        "0.50",
        "--phase-hold-gain-hz-per-rad",
        f"{case.phase_gain:g}",
        "--phase-hold-arm-threshold-hz",
        "100",
        "--phase-hold-arm-dwell-updates",
        "3",
        "--phase-hold-exit-dwell-updates",
        "25",
        "--phase-hold-base-lock-dwell-updates",
        str(int(case.base_lock_dwell_updates)),
        "--phase-hold-base-lock-threshold-hz",
        f"{case.base_lock_threshold_hz:g}",
        "--phase-hold-max-freq-hz",
        f"{case.phase_max_hz:g}",
        "--phase-hold-smoothing-alpha",
        f"{case.phase_alpha:g}",
        "--phase-error-filter-alpha",
        f"{case.phase_error_alpha:g}",
        "--phase-error-min-confidence",
        "0.0",
        "--phase-error-max-residual-rad",
        "3.05",
        "--phase-cycle-tracking",
        "--phase-cycle-error-source",
        "physical",
        "--phase-target-capture",
        "first_valid",
        "--phase-cycle-max-branch-error-rad",
        "3.05",
        "--phase-integral-gain-hz-per-rad-s",
        f"{case.phase_integral_gain_hz_per_rad_s:g}",
        "--phase-integral-max-hz",
        f"{case.phase_integral_max_hz:g}",
        "--pull-in",
        "--pull-in-switch-hz",
        "30",
        "--pull-in-exit-hz",
        "100",
        "--pull-in-enter-dwell-updates",
        "3",
        "--pull-in-exit-dwell-updates",
        "0",
        "--control-update-interval-s",
        "0.002",
        "--control-average-interval-s",
        "0.002",
        "--control-min-valid-blocks",
        "1",
        "--control-max-f-off-fraction",
        "0.90",
        "--control-max-phase-residual-rad",
        "3.05",
        "--control-min-fit-r2",
        "0.0",
        "--initial-frequency-step-blocks",
        "64",
        "--initial-frequency-step-min-valid-blocks",
        str(int(case.startup_min_valid_blocks)),
        "--initial-frequency-step-min-coherence",
        f"{case.startup_min_coherence:g}",
        "--initial-frequency-step-gain",
        f"{case.startup_gain:g}",
        "--initial-frequency-step-passes",
        "3",
        "--initial-frequency-step-refine-gain",
        "0.35",
        "--initial-frequency-step-deadband-hz",
        "1.0",
        "--profile-loop",
        "--log-flush-interval-s",
        "1.0",
        "--out-bin",
        str(out_bin),
        "--status-interval-s",
        "10.0",
    ]
    if case.slew_gain_hz_per_rad is not None:
        command.extend(["--phase-slew-gain-hz-per-rad", f"{case.slew_gain_hz_per_rad:g}"])
    if case.slew_max_freq_hz is not None:
        command.extend(["--phase-slew-max-freq-hz", f"{case.slew_max_freq_hz:g}"])
    command.extend(
        [
            "--phase-slew-enter-error-rad",
            f"{case.slew_enter_error_rad:g}",
            "--phase-slew-exit-error-rad",
            f"{case.slew_exit_error_rad:g}",
            "--phase-recovery-enter-dwell-updates",
            str(int(case.recovery_enter_dwell_updates)),
            "--phase-recovery-exit-dwell-updates",
            str(int(case.recovery_exit_dwell_updates)),
            "--phase-recovery-min-dwell-updates",
            str(int(case.recovery_min_dwell_updates)),
        ]
    )
    return command


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = [
        key
        for key in rows[0].keys()
        if not isinstance(rows[0].get(key), (dict, list))
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)


def main() -> int:
    args = parse_args()
    if args.case_set == "phase_tune":
        cases = phase_tune_cases()
    elif args.case_set == "phase_ff_grid":
        cases = phase_ff_grid_cases()
    elif args.case_set == "phase_smoothing_alpha":
        cases = phase_smoothing_alpha_cases()
    elif args.case_set == "two_mode":
        cases = two_mode_cases()
    elif args.case_set == "two_mode_focused":
        cases = two_mode_focused_cases()
    elif args.case_set == "two_mode_bias":
        cases = two_mode_bias_cases()
    elif args.case_set == "two_mode_authority":
        cases = two_mode_authority_cases()
    else:
        cases = default_cases()
    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]

    sweep_dir = Path(args.logs_dir) / str(args.label)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        prefix = f"{index:02d}_{case.name}"
        out_bin = sweep_dir / f"{prefix}.bin"
        stdout_path = sweep_dir / f"{prefix}.stdout.log"
        stderr_path = sweep_dir / f"{prefix}.stderr.log"
        command = base_command(case, args, out_bin)
        print(f"[{index}/{len(cases)}] running {case.name}", flush=True)
        start = time.monotonic()
        timeout_s = float(args.run_seconds) + float(args.timeout_margin_s)
        run_command = ["timeout", "-k", "3", "-s", "TERM", f"{timeout_s:g}", *command]
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            result = subprocess.run(
                run_command,
                stdout=stdout,
                stderr=stderr,
                cwd=ROOT,
                check=False,
            )
        elapsed_s = time.monotonic() - start
        metrics = read_binary_metrics(out_bin, float(args.zoom_start_s), float(args.zoom_duration_s))
        startup = extract_startup_summary(stdout_path)
        row = {
            "case": case.name,
            "returncode": result.returncode,
            "elapsed_s": elapsed_s,
            "bin_path": str(out_bin),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            **case.__dict__,
            **startup,
            **metrics,
        }
        rows.append(row)
        print(
            f"  score={row['score']:.3g} late={row['late_slope_hz']:.3g} Hz "
            f"zoom_span={row['zoom_span_rad']:.3g} rad wraps={row['zoom_wraps']} "
            f"slips={row['cycle_slip_count_final']:.0f} "
            f"branch_rejects={row['cycle_branch_reject_count_final']:.0f} "
            f"candidates={row['cycle_slip_candidate_rows']:.0f} "
            f"hold={row['phase_hold_active_frac']:.2f} "
            f"mode_hold={row['phase_mode_hold_frac']:.2f} recover={row['phase_mode_recover_frac']:.2f} "
            f"mode_trans={row['phase_mode_transitions']} "
            f"wrap_med_late={row['late_wrapped_abs_median_rad']:.3g} "
            f"wrap_med_delta={row['wrapped_abs_median_delta_rad']:.3g} "
            f"int_span={row['phase_integral_span_hz']:.3g} "
            f"dac_span={row['zoom_dac_span']:.3g} startup_applied={row['startup_step_applied']}",
            flush=True,
        )

    def score_key(row: dict[str, Any]) -> tuple[int, float]:
        score = safe_float(row.get("score"))
        return (0, score) if np.isfinite(score) else (1, float("inf"))

    ranked = sorted(rows, key=score_key)
    csv_path = sweep_dir / "summary.csv"
    json_path = sweep_dir / "summary.json"
    write_csv(csv_path, ranked)
    json_path.write_text(json.dumps(ranked, indent=2, sort_keys=True))
    print(f"summary_csv={csv_path}")
    print(f"summary_json={json_path}")
    if ranked:
        print(f"best_case={ranked[0]['case']} score={ranked[0]['score']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
