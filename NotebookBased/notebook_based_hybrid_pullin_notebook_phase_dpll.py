#!/usr/bin/env python3
"""Hybrid NotebookBased DPLL with pull-in plus notebook-style near-lock phase mode.

Design goals from user requirements:
- Live async SDR streaming (plot_sdr_phase style queueing)
- Staged control: frequency pull-in first, then notebook-style phase trim
- DAC-only actuation through clock_vcxo_helper.ClockVCXO
- 0-rad phase setpoint
- Monitor mode with identical metrics but no actuation
- CSV logging + summary metrics emphasizing phase-drift reduction
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ugradio import sdr as ugradio_sdr

    _UGRADIO_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    ugradio_sdr = None
    _UGRADIO_IMPORT_ERROR = exc

try:
    import clock_vcxo_helper as cvh
except Exception:
    cvh = None


DAC_FS_CODE = 4095
VDD_DAC_DEFAULT = 3.3


def wrap_to_pi(x: float) -> float:
    y = (float(x) + math.pi) % (2.0 * math.pi) - math.pi
    if y <= -math.pi:
        y += 2.0 * math.pi
    return float(y)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(x))))


def clamp_int(x: int | float, lo: int, hi: int) -> int:
    return int(max(int(lo), min(int(hi), int(round(float(x))))))


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def pct_to_code(pct: float) -> int:
    return clamp_int((float(pct) / 100.0) * float(DAC_FS_CODE), 0, DAC_FS_CODE)


def code_to_volts(code: int, vdd_dac: float) -> float:
    return (float(clamp_int(code, 0, DAC_FS_CODE)) / float(DAC_FS_CODE)) * float(vdd_dac)


@dataclass
class QueuedBlock:
    unix_time_s: float
    sample_start: int
    iq_block: np.ndarray


@dataclass
class StreamSentinel:
    error: BaseException | None = None


@dataclass
class CircularLpfState:
    initialized: bool = False
    z_re: float = 1.0
    z_im: float = 0.0


def circular_phase_lpf_update(phase_rad: float, alpha: float, state: CircularLpfState) -> float:
    a = clamp(alpha, 0.0, 0.999999)
    xr = math.cos(float(phase_rad))
    xi = math.sin(float(phase_rad))
    if not state.initialized:
        state.z_re = xr
        state.z_im = xi
        state.initialized = True
    else:
        state.z_re = (a * state.z_re) + ((1.0 - a) * xr)
        state.z_im = (a * state.z_im) + ((1.0 - a) * xi)
    return float(math.atan2(state.z_im, state.z_re))


def circular_phase_lpf_force(phase_rad: float, state: CircularLpfState) -> float:
    state.z_re = math.cos(float(phase_rad))
    state.z_im = math.sin(float(phase_rad))
    state.initialized = True
    return float(math.atan2(state.z_im, state.z_re))


def scalar_lpf_update(value: float, alpha: float, prev: float | None) -> float:
    if not np.isfinite(value):
        return float(prev) if prev is not None and np.isfinite(prev) else float("nan")
    if prev is None or (not np.isfinite(prev)):
        return float(value)
    a = clamp(alpha, 0.0, 0.999999)
    return float((a * float(prev)) + ((1.0 - a) * float(value)))


def phase_err_to_notebook_dac_target(
    *,
    phase_err_filt_rad: float,
    nsamp: int,
    phase_center_code: int,
    phase_center_trim_codes: float,
    phase_frac_gain_codes: float,
    control_sign: float,
    code_min: int,
    code_max: int,
) -> tuple[float, float, int]:
    phase_fraction = float(float(phase_err_filt_rad) / (2.0 * math.pi * float(nsamp)))
    dac_target_float = float(phase_center_code) + float(phase_center_trim_codes) + (
        float(control_sign) * float(phase_frac_gain_codes) * float(phase_fraction)
    )
    dac_target_code = clamp_int(dac_target_float, int(code_min), int(code_max))
    return float(phase_fraction), float(dac_target_float), int(dac_target_code)


@dataclass
class SlipAwareUnwrapper:
    prev_wrapped_phase_rad: float | None = None
    prev_time_s: float | None = None
    unwrapped_phase_rad: float = 0.0
    slip_trigger_rad: float = math.pi
    slip_max_cycles_per_update: int = 8

    def reset(self) -> None:
        self.prev_wrapped_phase_rad = None
        self.prev_time_s = None
        self.unwrapped_phase_rad = 0.0

    def update(
        self,
        wrapped_phase_rad: float,
        t_now_s: float,
        fallback_freq_hz: float,
    ) -> tuple[float, float, float, int, float, float, float]:
        phase = float(wrapped_phase_rad)
        t_now = float(t_now_s)
        fb = float(fallback_freq_hz) if np.isfinite(fallback_freq_hz) else 0.0

        if self.prev_wrapped_phase_rad is None or self.prev_time_s is None:
            self.prev_wrapped_phase_rad = phase
            self.prev_time_s = t_now
            self.unwrapped_phase_rad = phase
            return float(self.unwrapped_phase_rad), float(fb), 0.0, 0, 0.0, 0.0, 0.0

        dt = float(t_now - self.prev_time_s)
        if dt <= 0.0 or (not np.isfinite(dt)):
            dt = 0.0

        dphi_wrapped = wrap_to_pi(phase - self.prev_wrapped_phase_rad)
        dphi_used = float(dphi_wrapped)
        expected_dphi = 0.0
        slip_cycles = 0

        if dt > 0.0:
            expected_dphi = 2.0 * math.pi * fb * dt
            err = expected_dphi - dphi_wrapped
            if abs(err) >= float(self.slip_trigger_rad):
                k = int(round(err / (2.0 * math.pi)))
                k = clamp_int(
                    k,
                    -int(self.slip_max_cycles_per_update),
                    int(self.slip_max_cycles_per_update),
                )
                if k != 0:
                    candidate = float(dphi_wrapped + (2.0 * math.pi * float(k)))
                    if abs(candidate - expected_dphi) < abs(dphi_wrapped - expected_dphi):
                        dphi_used = candidate
                        slip_cycles = int(k)

        if dt > 0.0:
            freq_update_hz = float(dphi_used / (2.0 * math.pi * dt))
        else:
            freq_update_hz = float(fb)

        self.unwrapped_phase_rad += dphi_used
        self.prev_wrapped_phase_rad = phase
        self.prev_time_s = t_now

        return (
            float(self.unwrapped_phase_rad),
            float(freq_update_hz),
            float(dt),
            int(slip_cycles),
            float(dphi_wrapped),
            float(dphi_used),
            float(expected_dphi),
        )


@dataclass
class Measurement:
    update_idx: int
    unix_time_s: float
    elapsed_s: float
    sample_start: int
    tone_offset_hz: float
    wrapped_phase_rad: float
    unwrapped_phase_rad: float
    coherence: float
    pullin_quality: float
    amp_rms: float
    freq_err_intrablock_hz: float
    freq_err_update_hz: float
    phase_fit_slope_hz: float
    phase_fit_rms_rad: float
    phase_fit_max_abs_rad: float
    phase_fit_chunks_used: int
    dt_update_s: float
    slip_cycles: int
    dphi_wrapped_rad: float
    dphi_used_rad: float
    dphi_expected_rad: float


@dataclass
class PreparedBlock:
    unix_time_s: float
    sample_start: int
    elapsed_s: float
    iq_centered: np.ndarray
    sample_index: np.ndarray
    abs_sum: float
    amp_rms: float


@dataclass
class DemodStats:
    tone_offset_hz: float
    wrapped_phase_rad: float
    coherence: float
    pullin_quality: float
    freq_err_intrablock_hz: float
    phase_fit_slope_hz: float
    phase_fit_rms_rad: float
    phase_fit_max_abs_rad: float
    phase_fit_chunks_used: int


@dataclass
class ControllerState:
    in_phase_mode: bool = False
    phase_int_rad_s: float = 0.0
    prev_phase_err_filt_rad: float | None = None
    phase_reference_rad: float | None = None
    phase_err_lpf: CircularLpfState = field(default_factory=CircularLpfState)
    pullin_err_lpf_hz: float | None = None
    pullin_lock_counter: int = 0
    pullin_unlock_counter: int = 0
    pullin_lock_dwell_s: float = 0.0
    pullin_unlock_dwell_s: float = 0.0
    phase_center_code: int | None = None
    phase_center_trim_codes: float = 0.0
    phase_substage: str = "acquire"
    phase_hold_lock_counter: int = 0
    phase_hold_unlock_counter: int = 0
    phase_freq_err_lpf_hz: float | None = None
    phase_hold_rate_lpf_hz: float | None = None
    phase_hold_int_hz: float = 0.0
    phase_step_sigma_codes: float = 0.0
    phase_mode_updates: int = 0
    phase_mode_elapsed_s: float = 0.0
    phase_reference_age_s: float = 0.0
    phase_reference_accepted_updates: int = 0
    phase_accepted_raw_abs_window: deque[float] = field(default_factory=deque)

    dac_code_cmd: int = 0
    dac_code_rb: int | None = None
    dac_volts: float = float("nan")

    stage_switches: int = 0
    pullin_updates_applied: int = 0
    phase_updates_applied: int = 0
    post_dac_ignore_updates: int = 0


def reset_phase_error_state(state: ControllerState) -> None:
    state.phase_int_rad_s = 0.0
    state.prev_phase_err_filt_rad = None
    state.phase_err_lpf = CircularLpfState()
    state.phase_hold_int_hz = 0.0
    state.phase_hold_rate_lpf_hz = None


def clear_phase_reference(state: ControllerState) -> None:
    state.phase_reference_rad = None
    state.phase_reference_age_s = 0.0
    state.phase_reference_accepted_updates = 0
    state.phase_accepted_raw_abs_window.clear()
    state.phase_substage = "acquire"
    state.phase_hold_lock_counter = 0
    state.phase_hold_unlock_counter = 0
    reset_phase_error_state(state)


def set_phase_reference(state: ControllerState, phase_reference_rad: float) -> None:
    state.phase_reference_rad = wrap_to_pi(float(phase_reference_rad))
    state.phase_reference_age_s = 0.0
    state.phase_reference_accepted_updates = 0
    state.phase_accepted_raw_abs_window.clear()
    state.phase_substage = "acquire"
    state.phase_hold_lock_counter = 0
    state.phase_hold_unlock_counter = 0
    reset_phase_error_state(state)


def current_phase_reference(
    state: ControllerState,
    phase_reference_mode: str,
) -> tuple[float, bool]:
    if str(phase_reference_mode) == "absolute":
        return 0.0, True
    if state.phase_reference_rad is None or (not np.isfinite(float(state.phase_reference_rad))):
        return float("nan"), False
    return float(state.phase_reference_rad), True


def append_accepted_raw_phase_sample(
    state: ControllerState,
    args: argparse.Namespace,
    *,
    abs_phase_err_rad: float,
    dt_s: float,
) -> None:
    if not np.isfinite(float(abs_phase_err_rad)):
        return
    state.phase_reference_age_s = max(
        0.0,
        float(state.phase_reference_age_s) + max(0.0, float(dt_s)),
    )
    state.phase_reference_accepted_updates += 1
    window_capacity = max(
        0,
        int(args.phase_accepted_raw_enter_window_updates),
        int(args.phase_accepted_raw_exit_window_updates),
    )
    if window_capacity <= 0:
        return
    state.phase_accepted_raw_abs_window.append(float(abs_phase_err_rad))
    while len(state.phase_accepted_raw_abs_window) > window_capacity:
        state.phase_accepted_raw_abs_window.popleft()


def accepted_raw_window_fraction(
    state: ControllerState,
    *,
    threshold_rad: float,
    window_updates: int,
) -> tuple[float, bool]:
    window_updates = int(window_updates)
    threshold_rad = float(threshold_rad)
    if window_updates <= 0 or threshold_rad <= 0.0:
        return float("nan"), False
    values = list(state.phase_accepted_raw_abs_window)
    if not values:
        return float("nan"), False
    recent = values[-window_updates:]
    if not recent:
        return float("nan"), False
    fraction = float(np.mean(np.asarray(recent, dtype=np.float64) <= threshold_rad))
    ready = len(recent) >= window_updates
    return float(fraction), bool(ready)


def phase_reference_gain_scale(
    state: ControllerState,
    args: argparse.Namespace,
    *,
    phase_reference_valid: bool,
) -> float:
    if not bool(phase_reference_valid):
        return 0.0
    ramp_updates = int(args.phase_reference_ramp_accepted_updates)
    if ramp_updates <= 0:
        return 1.0
    min_scale = clamp(float(args.phase_reference_ramp_min_scale), 0.0, 1.0)
    frac = clamp(
        float(state.phase_reference_accepted_updates) / float(ramp_updates),
        0.0,
        1.0,
    )
    return float(min_scale + ((1.0 - min_scale) * frac))


def engage_phase_mode(
    state: ControllerState,
    args: argparse.Namespace,
    *,
    pullin_err_hz: float,
) -> None:
    state.in_phase_mode = True
    clear_phase_reference(state)
    state.phase_center_code = int(state.dac_code_cmd)
    state.phase_substage = "acquire"
    state.phase_hold_lock_counter = 0
    state.phase_hold_unlock_counter = 0
    state.phase_freq_err_lpf_hz = (
        float(pullin_err_hz) if np.isfinite(pullin_err_hz) else None
    )
    state.phase_hold_rate_lpf_hz = 0.0
    state.phase_hold_int_hz = 0.0
    state.phase_step_sigma_codes = 0.0
    state.phase_mode_updates = 0
    state.phase_mode_elapsed_s = 0.0
    if abs(float(args.kv_hz_per_code)) > 1e-12 and np.isfinite(pullin_err_hz):
        state.phase_center_trim_codes = clamp(
            float(args.control_sign) * float(pullin_err_hz) / float(args.kv_hz_per_code),
            -float(args.phase_center_integral_limit_codes),
            float(args.phase_center_integral_limit_codes),
        )
    else:
        state.phase_center_trim_codes = 0.0
    state.pullin_lock_counter = 0
    state.pullin_unlock_counter = 0
    state.pullin_lock_dwell_s = 0.0
    state.pullin_unlock_dwell_s = 0.0


@dataclass
class RuntimeStats:
    producer_blocks_dropped_queue_full: int = 0
    consumer_blocks_skipped_latest_only: int = 0
    bootstrap_blocks_discarded: int = 0
    shared_demod_updates: int = 0
    log_enqueue_total_s: float = 0.0
    log_enqueue_max_s: float = 0.0
    log_enqueue_events: int = 0


CSV_COLUMNS = [
    "unix_time_s",
    "elapsed_s",
    "update_idx",
    "mode",
    "stage",
    "action",
    "stage_switch_event",
    "sample_start",
    "tone_offset_hz",
    "tone_track_step_hz",
    "observer_tone_offset_hz",
    "observer_track_step_hz",
    "tracked_tone_offset_hz",
    "pullin_reference_offset_hz",
    "phase_wrapped_rad",
    "phase_unwrapped_rad",
    "observer_phase_wrapped_rad",
    "observer_phase_unwrapped_rad",
    "control_measurement_source",
    "control_phase_wrapped_rad",
    "control_phase_unwrapped_rad",
    "phase_reference_mode",
    "phase_reference_rad",
    "phase_reference_valid",
    "phase_reference_age_s",
    "phase_reference_accepted_updates",
    "phase_reference_ramp_scale",
    "phase_setpoint_rad",
    "phase_err_raw_rad",
    "phase_err_relative_raw_rad",
    "phase_err_accepted_raw_rad",
    "phase_accepted_raw_window_count",
    "phase_accepted_raw_enter_fraction",
    "phase_accepted_raw_exit_fraction",
    "phase_accepted_raw_enter_ok",
    "phase_accepted_raw_exit_ok",
    "phase_err_control_rad",
    "phase_err_filt_rad",
    "coherence",
    "pullin_quality",
    "observer_coherence",
    "observer_pullin_quality",
    "control_coherence",
    "control_pullin_quality",
    "amp_rms",
    "freq_err_intrablock_hz",
    "freq_err_update_hz",
    "phase_fit_slope_hz",
    "phase_fit_rms_rad",
    "phase_fit_max_abs_rad",
    "phase_fit_chunks_used",
    "observer_freq_err_intrablock_hz",
    "observer_freq_err_update_hz",
    "observer_phase_fit_slope_hz",
    "observer_phase_fit_rms_rad",
    "observer_phase_fit_max_abs_rad",
    "observer_phase_fit_chunks_used",
    "control_freq_err_intrablock_hz",
    "control_freq_err_update_hz",
    "control_phase_fit_slope_hz",
    "control_phase_fit_rms_rad",
    "control_phase_fit_max_abs_rad",
    "control_phase_fit_chunks_used",
    "observer_tau_err_ns",
    "pullin_target_hz",
    "pullin_error_hz",
    "pullin_error_lpf_hz",
    "phase_measure_pred_rad",
    "phase_predict_rate_hz",
    "phase_measure_innovation_rad",
    "phase_detector_quality_ok",
    "phase_innovation_ok",
    "phase_innovation_softened",
    "phase_measurement_rejected",
    "phase_state_predicted",
    "slip_cycles",
    "dphi_wrapped_rad",
    "dphi_used_rad",
    "dphi_expected_rad",
    "u_hz_raw",
    "u_hz_limited",
    "sham_control_mode",
    "dac_code_cmd",
    "dac_code_rb",
    "dac_code_proposed",
    "dac_code_apply_target",
    "dac_volts",
    "dac_step_req_codes",
    "dac_step_apply_target_codes",
    "dac_step_applied_codes",
    "phase_int_rad_s",
    "phase_center_code",
    "phase_center_trim_codes",
    "phase_substage",
    "phase_freq_err_lpf_hz",
    "phase_hold_rate_lpf_hz",
    "phase_hold_int_hz",
    "phase_step_sigma_codes",
    "phase_target_float",
    "phase_target_code",
    "phase_fraction_cycles_per_sample",
    "phase_err_for_entry_ok",
    "post_dac_ignore_active",
    "post_dac_ignore_updates_remaining",
    "queue_skipped_blocks_latest",
    "wall_block_age_s",
    "timing_queue_wait_s",
    "timing_measure_s",
    "timing_control_s",
    "timing_dac_s",
    "timing_total_s",
    "loop_period_s_nom",
    "nsamp",
    "sample_rate_hz",
    "kv_hz_per_code",
    "notes",
]


def csv_logger_worker(
    path_str: str,
    flush_every_rows: int,
    flush_interval_s: float,
    queue: Any,
) -> None:
    path = Path(path_str)
    ensure_parent_dir(path)
    rows_since_flush = 0
    last_flush_s = time.monotonic()

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        fh.flush()

        while True:
            item = queue.get()
            if item is None:
                break
            writer.writerow(item)
            rows_since_flush += 1

            now_s = time.monotonic()
            if (
                rows_since_flush >= max(1, int(flush_every_rows))
                or (now_s - last_flush_s) >= max(0.0, float(flush_interval_s))
            ):
                fh.flush()
                rows_since_flush = 0
                last_flush_s = now_s

        fh.flush()


class CsvLogger:
    def __init__(
        self,
        path: Path,
        *,
        flush_every_rows: int,
        flush_interval_s: float,
        use_background_process: bool,
    ):
        self.path = path
        self.fh = None
        self.writer = None
        self.ctx = None
        self.queue = None
        self.proc = None
        self.use_background_process = bool(use_background_process)
        self.logger_mode = "inline"

        if self.use_background_process:
            try:
                self.ctx = mp.get_context("spawn")
                self.queue = self.ctx.Queue()
                self.proc = self.ctx.Process(
                    target=csv_logger_worker,
                    args=(
                        str(path),
                        int(flush_every_rows),
                        float(flush_interval_s),
                        self.queue,
                    ),
                    daemon=True,
                )
                self.proc.start()
                self.logger_mode = "background_process"
                return
            except Exception:
                self.proc = None
                self.queue = None
                self.ctx = None

        ensure_parent_dir(path)
        self.fh = path.open("w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()
        self.fh.flush()

    def write(self, row: dict[str, Any]) -> None:
        out = {k: row.get(k, "") for k in CSV_COLUMNS}
        if self.proc is not None and self.queue is not None:
            self.queue.put(out)
            return
        assert self.writer is not None
        assert self.fh is not None
        self.writer.writerow(out)

    def close(self) -> None:
        if self.proc is not None and self.queue is not None:
            try:
                self.queue.put(None)
            except Exception:
                pass
            try:
                self.proc.join(timeout=5.0)
            except Exception:
                pass
            if self.proc.is_alive():
                try:
                    self.proc.terminate()
                except Exception:
                    pass
                try:
                    self.proc.join(timeout=1.0)
                except Exception:
                    pass
            try:
                self.queue.close()
            except Exception:
                pass
            try:
                self.queue.join_thread()
            except Exception:
                pass
            self.proc = None
            self.queue = None
        try:
            if self.fh is not None:
                self.fh.flush()
                self.fh.close()
        except Exception:
            pass


def parse_nsamp_values(nsamp_values: str | None, default_nsamp: int) -> list[int]:
    if nsamp_values is None:
        return [int(default_nsamp)]
    vals: list[int] = []
    for token in str(nsamp_values).split(","):
        t = token.strip()
        if not t:
            continue
        v = int(t)
        if v <= 0:
            raise ValueError("All nsamp values must be positive.")
        vals.append(v)
    if not vals:
        vals.append(int(default_nsamp))
    return vals


def path_with_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.with_name(f"{path.name}{suffix}")


def bytes_to_iq_block(raw_bytes: bytes, nsamples: int) -> np.ndarray:
    expected = 2 * int(nsamples)
    if len(raw_bytes) != expected:
        raise RuntimeError(
            f"Expected {expected} bytes for one IQ block, received {len(raw_bytes)}."
        )
    raw = (np.frombuffer(raw_bytes, dtype=np.uint8) - 128).view(np.int8)
    raw = raw.reshape(int(nsamples), 2).astype(np.float32, copy=False)
    return (raw[:, 0] + 1j * raw[:, 1]).astype(np.complex64)


def open_sdr(args: argparse.Namespace) -> Any:
    if ugradio_sdr is None:
        raise RuntimeError(f"ugradio.sdr import failed: {_UGRADIO_IMPORT_ERROR}")
    return ugradio_sdr.SDR(
        device_index=int(args.device_index),
        direct=False,
        center_freq=float(args.center_freq),
        sample_rate=float(args.sample_rate),
        gain=float(args.gain),
    )


def default_async_queue_blocks() -> int:
    return 2


def signal_stream_end(
    queue: asyncio.Queue[QueuedBlock | StreamSentinel],
    error: BaseException | None = None,
) -> None:
    sentinel = StreamSentinel(error=error)
    while True:
        try:
            queue.put_nowait(sentinel)
            return
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return


async def stream_iq_blocks(
    device: Any,
    stream_nsamples: int,
    emit_nsamples: int,
    queue: asyncio.Queue[QueuedBlock | StreamSentinel],
    runtime_stats: RuntimeStats,
) -> None:
    stream_nsamples = int(stream_nsamples)
    emit_nsamples = int(emit_nsamples)
    if stream_nsamples <= 0 or emit_nsamples <= 0:
        raise ValueError("stream_nsamples and emit_nsamples must be positive.")
    if stream_nsamples % emit_nsamples != 0:
        raise ValueError("stream_nsamples must be an integer multiple of emit_nsamples.")

    sample_start = 0
    try:
        async for raw in device.stream(
            num_samples_or_bytes=2 * int(stream_nsamples),
            format="bytes",
        ):
            iq_block = bytes_to_iq_block(raw, stream_nsamples)
            captured_unix_time_s = float(time.time())
            for start in range(0, int(iq_block.size), int(emit_nsamples)):
                stop = start + int(emit_nsamples)
                block = QueuedBlock(
                    unix_time_s=captured_unix_time_s,
                    sample_start=int(sample_start),
                    iq_block=iq_block[start:stop],
                )
                while True:
                    try:
                        queue.put_nowait(block)
                        break
                    except asyncio.QueueFull:
                        try:
                            dropped = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            await asyncio.sleep(0)
                            continue
                        if isinstance(dropped, QueuedBlock):
                            runtime_stats.producer_blocks_dropped_queue_full += 1
                sample_start += int(emit_nsamples)
    except asyncio.CancelledError:
        signal_stream_end(queue)
        raise
    except Exception as exc:
        signal_stream_end(queue, error=exc)
        raise
    else:
        signal_stream_end(queue)
    finally:
        try:
            await device.stop()
        except Exception:
            pass


def estimate_tone_offset_hz(
    iq_blocks: np.ndarray,
    sample_rate_hz: float,
    dc_exclusion_hz: float,
) -> float:
    spectra = np.fft.fft(iq_blocks, axis=1)
    power = np.mean(np.abs(spectra) ** 2, axis=0)
    freqs = np.fft.fftfreq(iq_blocks.shape[1], d=1.0 / float(sample_rate_hz))

    mask = np.ones(freqs.shape, dtype=bool)
    if float(dc_exclusion_hz) > 0.0:
        mask &= np.abs(freqs) >= float(dc_exclusion_hz)
    if not np.any(mask):
        mask[:] = True

    idx_candidates = np.flatnonzero(mask)
    peak_idx = int(idx_candidates[int(np.argmax(power[mask]))])
    return float(freqs[peak_idx])


def phase_increment_stats(z: np.ndarray, sample_rate_hz: float) -> tuple[float, float]:
    x = np.asarray(z, dtype=np.complex64).reshape(-1)
    if x.size < 2:
        return float("nan"), float("nan")
    prod = x[1:] * np.conjugate(x[:-1])
    s = np.sum(prod, dtype=np.complex128)
    if not (np.isfinite(s.real) and np.isfinite(s.imag)):
        return float("nan"), float("nan")
    denom = float(np.sum(np.abs(prod), dtype=np.float64))
    if denom <= 0.0 or (not np.isfinite(denom)):
        quality = float("nan")
    else:
        quality = float(np.abs(s) / (denom + 1e-12))
    freq_hz = float((float(sample_rate_hz) / (2.0 * math.pi)) * np.angle(s))
    return float(freq_hz), float(quality)


def block_sum_phase_stats(z: np.ndarray, abs_sum: float) -> tuple[float, float]:
    s = np.sum(np.asarray(z, dtype=np.complex64).reshape(-1), dtype=np.complex128)
    if not (np.isfinite(s.real) and np.isfinite(s.imag)):
        return float("nan"), float("nan")
    wrapped_phase = float(np.angle(s))
    denom = float(abs_sum)
    if denom <= 0.0 or (not np.isfinite(denom)):
        coherence = float("nan")
    else:
        coherence = float(np.abs(s) / (denom + 1e-12))
    return float(wrapped_phase), float(coherence)


def chunk_fit_desloped_phase_stats(
    z: np.ndarray,
    *,
    abs_sum: float,
    sample_rate_hz: float,
    chunks: int,
) -> tuple[float, float, float, float, float, int]:
    x = np.asarray(z, dtype=np.complex64).reshape(-1)
    nchunks = max(int(chunks), 1)
    if nchunks <= 1 or x.size < max(2 * nchunks, 8):
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0

    edges = np.linspace(0, x.size, nchunks + 1, dtype=np.int64)
    block_center = 0.5 * float(x.size - 1)
    chunk_times_s: list[float] = []
    chunk_phases_rad: list[float] = []
    chunk_weights: list[float] = []

    for start, stop in zip(edges[:-1], edges[1:]):
        if int(stop) - int(start) < 2:
            continue
        chunk = x[int(start):int(stop)]
        chunk_abs_sum = float(np.sum(np.abs(chunk), dtype=np.float64))
        phase_chunk, coherence_chunk = block_sum_phase_stats(chunk, chunk_abs_sum)
        if not (np.isfinite(phase_chunk) and np.isfinite(coherence_chunk)):
            continue
        chunk_center = 0.5 * float(int(start) + int(stop) - 1)
        chunk_time_s = float((chunk_center - block_center) / float(sample_rate_hz))
        chunk_weight = float(max(coherence_chunk * chunk_abs_sum, 1e-9))
        chunk_times_s.append(chunk_time_s)
        chunk_phases_rad.append(float(phase_chunk))
        chunk_weights.append(chunk_weight)

    if len(chunk_phases_rad) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0

    t = np.asarray(chunk_times_s, dtype=np.float64)
    y = np.unwrap(np.asarray(chunk_phases_rad, dtype=np.float64))
    w = np.asarray(chunk_weights, dtype=np.float64)
    finite = np.isfinite(t) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(finite) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0

    t = t[finite]
    y = y[finite]
    w = w[finite]
    sw = float(np.sum(w))
    st = float(np.sum(w * t))
    stt = float(np.sum(w * t * t))
    sy = float(np.sum(w * y))
    sty = float(np.sum(w * t * y))
    det = float((sw * stt) - (st * st))
    if not np.isfinite(det) or abs(det) <= 1e-24:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0

    slope_rad_per_s = float(((sw * sty) - (st * sy)) / det)
    intercept_rad = float((sy - (slope_rad_per_s * st)) / sw)
    y_fit = intercept_rad + (slope_rad_per_s * t)
    residual_rad = y - y_fit
    fit_rms_rad = float(np.sqrt(np.sum(w * residual_rad * residual_rad) / sw))
    fit_max_abs_rad = float(np.max(np.abs(residual_rad)))
    centered_index = np.arange(x.size, dtype=np.float64) - block_center
    desloper = np.exp(
        -1j * float(slope_rad_per_s) * centered_index / float(sample_rate_hz)
    ).astype(np.complex64)
    _, coherence_desloped = block_sum_phase_stats(x * desloper, float(abs_sum))
    return (
        float(wrap_to_pi(intercept_rad)),
        float(coherence_desloped),
        float(slope_rad_per_s / (2.0 * math.pi)),
        float(fit_rms_rad),
        float(fit_max_abs_rad),
        int(t.size),
    )


def prepare_block(
    queued_block: QueuedBlock,
    sample_rate_hz: float,
) -> PreparedBlock:
    iq = np.asarray(queued_block.iq_block, dtype=np.complex64).reshape(-1)
    iq = iq - np.mean(iq)

    sample_index = float(queued_block.sample_start) + np.arange(iq.size, dtype=np.float64)
    elapsed_s = float(
        (float(queued_block.sample_start) + (0.5 * float(iq.size))) / float(sample_rate_hz)
    )
    return PreparedBlock(
        unix_time_s=float(queued_block.unix_time_s),
        sample_start=int(queued_block.sample_start),
        elapsed_s=float(elapsed_s),
        iq_centered=iq,
        sample_index=sample_index,
        abs_sum=float(np.sum(np.abs(iq), dtype=np.float64)),
        amp_rms=float(np.sqrt(np.mean(np.abs(iq) ** 2) + 1e-30)),
    )


def demodulate_prepared_block(
    prepared_block: PreparedBlock,
    sample_rate_hz: float,
    tone_offset_hz: float,
    phase_detector_chunks: int = 1,
) -> DemodStats:
    rotator = np.exp(
        -2j * np.pi * float(tone_offset_hz) * prepared_block.sample_index / float(sample_rate_hz)
    ).astype(np.complex64)
    demod = prepared_block.iq_centered * rotator

    wrapped_phase, coherence = block_sum_phase_stats(demod, float(prepared_block.abs_sum))
    freq_intrablock, pullin_quality = phase_increment_stats(demod, sample_rate_hz)
    phase_fit_slope_hz = float("nan")
    phase_fit_rms_rad = float("nan")
    phase_fit_max_abs_rad = float("nan")
    phase_fit_chunks_used = 0
    if demod.size >= 2 and np.isfinite(freq_intrablock):
        centered_index = np.arange(demod.size, dtype=np.float64) - (0.5 * float(demod.size - 1))
        desloper = np.exp(
            -2j * np.pi * float(freq_intrablock) * centered_index / float(sample_rate_hz)
        ).astype(np.complex64)
        wrapped_phase_desloped, coherence_desloped = block_sum_phase_stats(
            demod * desloper,
            float(prepared_block.abs_sum),
        )
        if (
            np.isfinite(wrapped_phase_desloped)
            and np.isfinite(coherence_desloped)
            and (
                (not np.isfinite(coherence))
                or (coherence_desloped >= coherence)
            )
        ):
            wrapped_phase = float(wrapped_phase_desloped)
            coherence = float(coherence_desloped)
    if int(phase_detector_chunks) > 1:
        (
            wrapped_phase_chunkfit,
            coherence_chunkfit,
            phase_fit_slope_hz,
            phase_fit_rms_rad,
            phase_fit_max_abs_rad,
            phase_fit_chunks_used,
        ) = chunk_fit_desloped_phase_stats(
            demod,
            abs_sum=float(prepared_block.abs_sum),
            sample_rate_hz=float(sample_rate_hz),
            chunks=int(phase_detector_chunks),
        )
        if np.isfinite(wrapped_phase_chunkfit):
            wrapped_phase = float(wrapped_phase_chunkfit)
        if np.isfinite(coherence_chunkfit):
            coherence = float(coherence_chunkfit)
        if np.isfinite(phase_fit_slope_hz):
            freq_intrablock = float(phase_fit_slope_hz)
    return DemodStats(
        tone_offset_hz=float(tone_offset_hz),
        wrapped_phase_rad=float(wrapped_phase),
        coherence=float(coherence),
        pullin_quality=float(pullin_quality),
        freq_err_intrablock_hz=float(freq_intrablock),
        phase_fit_slope_hz=float(phase_fit_slope_hz),
        phase_fit_rms_rad=float(phase_fit_rms_rad),
        phase_fit_max_abs_rad=float(phase_fit_max_abs_rad),
        phase_fit_chunks_used=int(phase_fit_chunks_used),
    )


def build_measurement(
    prepared_block: PreparedBlock,
    demod_stats: DemodStats,
    update_idx: int,
    unwrapper: SlipAwareUnwrapper,
) -> Measurement:
    (
        unwrapped_phase,
        freq_update_hz,
        dt_update_s,
        slip_cycles,
        dphi_wrapped,
        dphi_used,
        dphi_expected,
    ) = unwrapper.update(
        wrapped_phase_rad=float(demod_stats.wrapped_phase_rad),
        t_now_s=float(prepared_block.elapsed_s),
        fallback_freq_hz=float(demod_stats.freq_err_intrablock_hz),
    )

    return Measurement(
        update_idx=int(update_idx),
        unix_time_s=float(prepared_block.unix_time_s),
        elapsed_s=float(prepared_block.elapsed_s),
        sample_start=int(prepared_block.sample_start),
        tone_offset_hz=float(demod_stats.tone_offset_hz),
        wrapped_phase_rad=float(demod_stats.wrapped_phase_rad),
        unwrapped_phase_rad=float(unwrapped_phase),
        coherence=float(demod_stats.coherence),
        pullin_quality=float(demod_stats.pullin_quality),
        amp_rms=float(prepared_block.amp_rms),
        freq_err_intrablock_hz=float(demod_stats.freq_err_intrablock_hz),
        freq_err_update_hz=float(freq_update_hz),
        phase_fit_slope_hz=float(demod_stats.phase_fit_slope_hz),
        phase_fit_rms_rad=float(demod_stats.phase_fit_rms_rad),
        phase_fit_max_abs_rad=float(demod_stats.phase_fit_max_abs_rad),
        phase_fit_chunks_used=int(demod_stats.phase_fit_chunks_used),
        dt_update_s=float(dt_update_s),
        slip_cycles=int(slip_cycles),
        dphi_wrapped_rad=float(dphi_wrapped),
        dphi_used_rad=float(dphi_used),
        dphi_expected_rad=float(dphi_expected),
    )


def phase_fit_available(meas: Measurement) -> bool:
    return bool(
        int(meas.phase_fit_chunks_used) >= 2
        and np.isfinite(meas.phase_fit_rms_rad)
        and np.isfinite(meas.phase_fit_max_abs_rad)
    )


def phase_detector_quality_ok(meas: Measurement, args: argparse.Namespace) -> bool:
    if phase_fit_available(meas):
        rms_ok = (
            float(args.phase_fit_rms_max_rad) <= 0.0
            or (
                np.isfinite(meas.phase_fit_rms_rad)
                and float(meas.phase_fit_rms_rad) <= float(args.phase_fit_rms_max_rad)
            )
        )
        max_ok = (
            float(args.phase_fit_max_abs_rad) <= 0.0
            or (
                np.isfinite(meas.phase_fit_max_abs_rad)
                and float(meas.phase_fit_max_abs_rad) <= float(args.phase_fit_max_abs_rad)
            )
        )
        return bool(rms_ok and max_ok)
    return bool(
        np.isfinite(meas.coherence)
        and (float(meas.coherence) >= float(args.coh_min))
    )


def drain_to_latest_block(
    queue: asyncio.Queue[QueuedBlock | StreamSentinel],
    first_item: QueuedBlock | StreamSentinel,
) -> tuple[QueuedBlock | None, StreamSentinel | None, int]:
    latest_block = first_item if isinstance(first_item, QueuedBlock) else None
    end_item = first_item if isinstance(first_item, StreamSentinel) else None
    skipped_blocks = 0

    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        if isinstance(item, StreamSentinel):
            if end_item is None or (end_item.error is None and item.error is not None):
                end_item = item
            continue

        if latest_block is not None:
            skipped_blocks += 1
        latest_block = item

    return latest_block, end_item, skipped_blocks


def linear_slope(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    if xa.size < 2 or ya.size != xa.size:
        return float("nan")
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        return float("nan")
    x0 = xa - np.mean(xa)
    denom = float(np.sum(x0 * x0))
    if denom <= 0.0:
        return float("nan")
    num = float(np.sum(x0 * (ya - np.mean(ya))))
    return float(num / denom)


def compute_summary(
    *,
    rows: list[dict[str, Any]],
    run_mode: str,
    nsamp: int,
    sample_rate_hz: float,
    center_freq_hz: float,
    tone_offset_hz: float,
    drift_window_s: float,
) -> dict[str, Any]:
    updates = len(rows)
    if updates == 0:
        return {
            "mode": run_mode,
            "nsamp": int(nsamp),
            "sample_rate_hz": float(sample_rate_hz),
            "center_freq_hz": float(center_freq_hz),
            "loop_period_nom_s": float(nsamp) / float(sample_rate_hz),
            "tone_offset_hz": float(tone_offset_hz),
            "updates": 0,
            "median_coherence": float("nan"),
            "phase_drift_slope_full_hz": float("nan"),
            "phase_drift_slope_start_hz": float("nan"),
            "phase_drift_slope_end_hz": float("nan"),
            "phase_drift_reduction_hz_abs": float("nan"),
            "phase_drift_reduction_pct": float("nan"),
            "mean_abs_phase_err_rad": float("nan"),
            "mean_abs_phase_err_accepted_raw_rad": float("nan"),
            "median_abs_phase_err_accepted_raw_rad": float("nan"),
            "mean_abs_freq_err_hz": float("nan"),
            "phase_mode_fraction": float("nan"),
            "phase_reference_valid_fraction": float("nan"),
            "accepted_raw_updates": 0,
            "observer_median_coherence": float("nan"),
            "observer_mean_abs_freq_err_hz": float("nan"),
            "observer_phase_drift_slope_full_hz": float("nan"),
            "observer_timing_drift_slope_ns_per_s_full": float("nan"),
            "observer_timing_drift_slope_ns_per_s_start": float("nan"),
            "observer_timing_drift_slope_ns_per_s_end": float("nan"),
            "observer_timing_drift_reduction_ns_per_s_abs": float("nan"),
            "observer_timing_drift_reduction_pct": float("nan"),
            "observer_timing_drift_slope_ns_per_s_quality": float("nan"),
            "observer_phase_drift_slope_quality_hz": float("nan"),
            "observer_mean_abs_freq_err_hz_quality": float("nan"),
            "quality_updates": 0,
            "quality_fraction": float("nan"),
            "quality_elapsed_s": float("nan"),
            "primary_timing_drift_ns_per_s": float("nan"),
            "primary_timing_drift_samples_per_s": float("nan"),
            "mean_wall_block_age_s": float("nan"),
            "median_wall_block_age_s": float("nan"),
            "max_wall_block_age_s": float("nan"),
            "mean_timing_queue_wait_s": float("nan"),
            "mean_timing_measure_s": float("nan"),
            "mean_timing_control_s": float("nan"),
            "mean_timing_dac_s": float("nan"),
            "mean_timing_total_s": float("nan"),
            "max_timing_total_s": float("nan"),
            "consumer_blocks_skipped_latest_only": 0,
        }

    def _fget(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key, "nan"))
        except Exception:
            return float("nan")

    t = np.asarray([float(r["elapsed_s"]) for r in rows], dtype=np.float64)
    phu = np.asarray([float(r["phase_unwrapped_rad"]) for r in rows], dtype=np.float64)
    coh = np.asarray([float(r["coherence"]) for r in rows], dtype=np.float64)
    pe = np.asarray([float(r["phase_err_raw_rad"]) for r in rows], dtype=np.float64)
    pe_accepted_raw = np.asarray([_fget(r, "phase_err_accepted_raw_rad") for r in rows], dtype=np.float64)
    ferr = np.asarray([float(r["freq_err_intrablock_hz"]) for r in rows], dtype=np.float64)
    stage_vals = [str(r["stage"]) for r in rows]
    action_vals = [str(r.get("action", "")) for r in rows]
    phase_reference_valid = np.asarray([_fget(r, "phase_reference_valid") for r in rows], dtype=np.float64)

    obs_phu = np.asarray([_fget(r, "observer_phase_unwrapped_rad") for r in rows], dtype=np.float64)
    obs_coh = np.asarray([_fget(r, "observer_coherence") for r in rows], dtype=np.float64)
    obs_ferr = np.asarray([_fget(r, "observer_freq_err_intrablock_hz") for r in rows], dtype=np.float64)
    obs_toff = np.asarray([_fget(r, "observer_tone_offset_hz") for r in rows], dtype=np.float64)
    wall_age = np.asarray([_fget(r, "wall_block_age_s") for r in rows], dtype=np.float64)
    t_wait = np.asarray([_fget(r, "timing_queue_wait_s") for r in rows], dtype=np.float64)
    t_meas = np.asarray([_fget(r, "timing_measure_s") for r in rows], dtype=np.float64)
    t_ctrl = np.asarray([_fget(r, "timing_control_s") for r in rows], dtype=np.float64)
    t_dac = np.asarray([_fget(r, "timing_dac_s") for r in rows], dtype=np.float64)
    t_total = np.asarray([_fget(r, "timing_total_s") for r in rows], dtype=np.float64)
    skipped_latest = np.asarray([_fget(r, "queue_skipped_blocks_latest") for r in rows], dtype=np.float64)

    if not np.any(np.isfinite(obs_phu)):
        obs_phu = np.copy(phu)
    if not np.any(np.isfinite(obs_coh)):
        obs_coh = np.copy(coh)
    if not np.any(np.isfinite(obs_ferr)):
        obs_ferr = np.copy(ferr)
    if not np.any(np.isfinite(obs_toff)):
        obs_toff = np.full_like(t, float(tone_offset_hz), dtype=np.float64)

    slope_full = linear_slope(t, phu)
    slope_full_hz = float(slope_full / (2.0 * math.pi)) if np.isfinite(slope_full) else float("nan")

    obs_slope_full = linear_slope(t, obs_phu)
    obs_slope_full_hz = (
        float(obs_slope_full / (2.0 * math.pi)) if np.isfinite(obs_slope_full) else float("nan")
    )

    t0 = float(t[0])
    t1 = float(t[-1])
    span = max(0.0, t1 - t0)
    w = min(float(drift_window_s), max(span / 2.0, 0.0))

    if w > 0.0:
        start_mask = t <= (t0 + w)
        end_mask = t >= (t1 - w)
    else:
        start_mask = np.zeros_like(t, dtype=bool)
        end_mask = np.zeros_like(t, dtype=bool)

    slope_start = linear_slope(t[start_mask], phu[start_mask]) if np.any(start_mask) else float("nan")
    slope_end = linear_slope(t[end_mask], phu[end_mask]) if np.any(end_mask) else float("nan")

    slope_start_hz = float(slope_start / (2.0 * math.pi)) if np.isfinite(slope_start) else float("nan")
    slope_end_hz = float(slope_end / (2.0 * math.pi)) if np.isfinite(slope_end) else float("nan")

    if np.isfinite(slope_start_hz) and np.isfinite(slope_end_hz):
        reduction_abs = float(abs(slope_start_hz) - abs(slope_end_hz))
        if abs(slope_start_hz) > 1e-12:
            reduction_pct = float(100.0 * reduction_abs / abs(slope_start_hz))
        else:
            reduction_pct = float("nan")
    else:
        reduction_abs = float("nan")
        reduction_pct = float("nan")

    phase_mode_frac = (
        float(sum(1 for s in stage_vals if s == "phase_lock")) / float(len(stage_vals))
        if stage_vals
        else float("nan")
    )
    quality_mask = np.asarray(
        [
            (not str(a).startswith("hold_low_")) and (str(a) != "hold_invalid_kv")
            for a in action_vals
        ],
        dtype=bool,
    )
    phase_reference_valid_mask = np.isfinite(phase_reference_valid) & (phase_reference_valid >= 0.5)
    quality_updates = int(np.count_nonzero(quality_mask))
    accepted_raw_updates = int(np.count_nonzero(np.isfinite(pe_accepted_raw)))
    quality_fraction = float(quality_updates / updates) if updates > 0 else float("nan")
    if quality_updates >= 2:
        quality_elapsed_s = float(t[quality_mask][-1] - t[quality_mask][0])
    elif quality_updates == 1:
        quality_elapsed_s = 0.0
    else:
        quality_elapsed_s = float("nan")

    obs_rf_hz = float(center_freq_hz) + obs_toff
    obs_tau_ns = np.full(obs_phu.shape, np.nan, dtype=np.float64)
    valid_obs = (
        np.isfinite(obs_phu)
        & np.isfinite(obs_rf_hz)
        & (np.abs(obs_rf_hz) > 1e-6)
        & np.isfinite(t)
    )
    if np.any(valid_obs):
        first_idx = int(np.flatnonzero(valid_obs)[0])
        obs_tau_ns[valid_obs] = (
            (obs_phu[valid_obs] - obs_phu[first_idx]) / (2.0 * math.pi * obs_rf_hz[valid_obs])
        ) * 1e9

    obs_tau_slope_full = linear_slope(t[valid_obs], obs_tau_ns[valid_obs]) if np.any(valid_obs) else float("nan")

    obs_tau_slope_start = (
        linear_slope(t[start_mask & valid_obs], obs_tau_ns[start_mask & valid_obs])
        if np.any(start_mask & valid_obs)
        else float("nan")
    )
    obs_tau_slope_end = (
        linear_slope(t[end_mask & valid_obs], obs_tau_ns[end_mask & valid_obs])
        if np.any(end_mask & valid_obs)
        else float("nan")
    )

    if np.isfinite(obs_tau_slope_start) and np.isfinite(obs_tau_slope_end):
        obs_tau_reduction_abs = float(abs(obs_tau_slope_start) - abs(obs_tau_slope_end))
        if abs(obs_tau_slope_start) > 1e-12:
            obs_tau_reduction_pct = float(100.0 * obs_tau_reduction_abs / abs(obs_tau_slope_start))
        else:
            obs_tau_reduction_pct = float("nan")
    else:
        obs_tau_reduction_abs = float("nan")
        obs_tau_reduction_pct = float("nan")

    valid_obs_quality = valid_obs & quality_mask
    obs_slope_quality = (
        linear_slope(t[valid_obs_quality], obs_phu[valid_obs_quality])
        if np.count_nonzero(valid_obs_quality) >= 2
        else float("nan")
    )
    obs_slope_quality_hz = (
        float(obs_slope_quality / (2.0 * math.pi)) if np.isfinite(obs_slope_quality) else float("nan")
    )
    obs_tau_slope_quality = (
        linear_slope(t[valid_obs_quality], obs_tau_ns[valid_obs_quality])
        if np.count_nonzero(valid_obs_quality) >= 2
        else float("nan")
    )
    obs_ferr_quality = obs_ferr[quality_mask & np.isfinite(obs_ferr)]
    obs_mean_abs_freq_err_quality = (
        float(np.nanmean(np.abs(obs_ferr_quality))) if obs_ferr_quality.size else float("nan")
    )

    primary_tau_ns_per_s = (
        float(obs_tau_slope_quality) if np.isfinite(obs_tau_slope_quality) else float(obs_tau_slope_full)
    )
    if np.isfinite(primary_tau_ns_per_s):
        primary_tau_samples_per_s = float(primary_tau_ns_per_s * float(sample_rate_hz) / 1e9)
    else:
        primary_tau_samples_per_s = float("nan")

    finite_wall_age = wall_age[np.isfinite(wall_age)]
    finite_t_wait = t_wait[np.isfinite(t_wait)]
    finite_t_meas = t_meas[np.isfinite(t_meas)]
    finite_t_ctrl = t_ctrl[np.isfinite(t_ctrl)]
    finite_t_dac = t_dac[np.isfinite(t_dac)]
    finite_t_total = t_total[np.isfinite(t_total)]
    finite_skipped_latest = skipped_latest[np.isfinite(skipped_latest)]

    return {
        "mode": str(run_mode),
        "nsamp": int(nsamp),
        "sample_rate_hz": float(sample_rate_hz),
        "center_freq_hz": float(center_freq_hz),
        "loop_period_nom_s": float(nsamp) / float(sample_rate_hz),
        "tone_offset_hz": float(tone_offset_hz),
        "updates": int(updates),
        "elapsed_s": float(span),
        "median_coherence": float(np.nanmedian(coh)) if coh.size else float("nan"),
        "mean_abs_phase_err_rad": float(np.nanmean(np.abs(pe))) if pe.size else float("nan"),
        "mean_abs_phase_err_accepted_raw_rad": float(np.nanmean(np.abs(pe_accepted_raw))) if accepted_raw_updates > 0 else float("nan"),
        "median_abs_phase_err_accepted_raw_rad": float(np.nanmedian(np.abs(pe_accepted_raw))) if accepted_raw_updates > 0 else float("nan"),
        "mean_abs_freq_err_hz": float(np.nanmean(np.abs(ferr))) if ferr.size else float("nan"),
        "phase_mode_fraction": float(phase_mode_frac),
        "phase_reference_valid_fraction": float(np.mean(phase_reference_valid_mask)) if phase_reference_valid_mask.size else float("nan"),
        "accepted_raw_updates": int(accepted_raw_updates),
        "phase_drift_slope_full_hz": float(slope_full_hz),
        "phase_drift_slope_start_hz": float(slope_start_hz),
        "phase_drift_slope_end_hz": float(slope_end_hz),
        "phase_drift_reduction_hz_abs": float(reduction_abs),
        "phase_drift_reduction_pct": float(reduction_pct),
        "observer_median_coherence": float(np.nanmedian(obs_coh)) if obs_coh.size else float("nan"),
        "observer_mean_abs_freq_err_hz": float(np.nanmean(np.abs(obs_ferr))) if obs_ferr.size else float("nan"),
        "observer_phase_drift_slope_full_hz": float(obs_slope_full_hz),
        "observer_timing_drift_slope_ns_per_s_full": float(obs_tau_slope_full),
        "observer_timing_drift_slope_ns_per_s_start": float(obs_tau_slope_start),
        "observer_timing_drift_slope_ns_per_s_end": float(obs_tau_slope_end),
        "observer_timing_drift_reduction_ns_per_s_abs": float(obs_tau_reduction_abs),
        "observer_timing_drift_reduction_pct": float(obs_tau_reduction_pct),
        "observer_timing_drift_slope_ns_per_s_quality": float(obs_tau_slope_quality),
        "observer_phase_drift_slope_quality_hz": float(obs_slope_quality_hz),
        "observer_mean_abs_freq_err_hz_quality": float(obs_mean_abs_freq_err_quality),
        "quality_updates": int(quality_updates),
        "quality_fraction": float(quality_fraction),
        "quality_elapsed_s": float(quality_elapsed_s),
        "primary_timing_drift_ns_per_s": float(primary_tau_ns_per_s),
        "primary_timing_drift_samples_per_s": float(primary_tau_samples_per_s),
        "mean_wall_block_age_s": float(np.nanmean(finite_wall_age)) if finite_wall_age.size else float("nan"),
        "median_wall_block_age_s": float(np.nanmedian(finite_wall_age)) if finite_wall_age.size else float("nan"),
        "max_wall_block_age_s": float(np.nanmax(finite_wall_age)) if finite_wall_age.size else float("nan"),
        "mean_timing_queue_wait_s": float(np.nanmean(finite_t_wait)) if finite_t_wait.size else float("nan"),
        "mean_timing_measure_s": float(np.nanmean(finite_t_meas)) if finite_t_meas.size else float("nan"),
        "mean_timing_control_s": float(np.nanmean(finite_t_ctrl)) if finite_t_ctrl.size else float("nan"),
        "mean_timing_dac_s": float(np.nanmean(finite_t_dac)) if finite_t_dac.size else float("nan"),
        "mean_timing_total_s": float(np.nanmean(finite_t_total)) if finite_t_total.size else float("nan"),
        "max_timing_total_s": float(np.nanmax(finite_t_total)) if finite_t_total.size else float("nan"),
        "consumer_blocks_skipped_latest_only": int(np.nansum(finite_skipped_latest)) if finite_skipped_latest.size else 0,
    }


def to_float(x: float | str | None) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


async def run_one(args: argparse.Namespace, nsamp: int, csv_path: Path) -> dict[str, Any]:
    if args.blocks_per_update != 1:
        raise ValueError("This implementation currently requires --blocks-per-update 1.")

    if nsamp <= 0:
        raise ValueError("nsamp must be positive.")

    stream_nsamp = int(args.stream_nsamp) if args.stream_nsamp is not None else int(nsamp)
    if stream_nsamp <= 0:
        raise ValueError("stream_nsamp must be positive.")
    if stream_nsamp < int(nsamp):
        raise ValueError("stream_nsamp must be >= nsamp in this implementation.")
    if stream_nsamp % int(nsamp) != 0:
        raise ValueError("stream_nsamp must be an integer multiple of nsamp.")

    if args.mode == "pid" and cvh is None:
        raise RuntimeError(
            "clock_vcxo_helper.py import failed, but mode=pid requires DAC control access."
        )

    code_min = pct_to_code(float(args.dac_min_pct))
    code_max = pct_to_code(float(args.dac_max_pct))
    if code_min >= code_max:
        raise ValueError("DAC range invalid: ensure dac_min_pct < dac_max_pct.")
    sham_control_mode = str(args.sham_control_mode)
    sham_rng = (
        np.random.default_rng(int(args.sham_rng_seed))
        if (args.mode == "pid" and sham_control_mode != "off")
        else None
    )

    if args.dac_start is None:
        dac_start = int(round((code_min + code_max) / 2.0))
    else:
        dac_start = clamp_int(int(args.dac_start), code_min, code_max)

    nominal_loop_period_s = float(nsamp) / float(args.sample_rate)
    pullin_enter_dwell_s_target = (
        float(args.pullin_enter_dwell_s)
        if args.pullin_enter_dwell_s is not None
        else float(args.pullin_enter_dwell_updates) * nominal_loop_period_s
    )
    pullin_exit_dwell_s_target = (
        float(args.pullin_exit_dwell_s)
        if args.pullin_exit_dwell_s is not None
        else float(args.pullin_exit_dwell_updates) * nominal_loop_period_s
    )
    phase_min_hold_s_target = (
        float(args.phase_min_hold_s)
        if args.phase_min_hold_s is not None
        else float(args.phase_min_hold_updates) * nominal_loop_period_s
    )
    queue_get_timeout_s = max(float(args.queue_get_timeout_s), nominal_loop_period_s)

    runtime_stats = RuntimeStats()
    logger = CsvLogger(
        csv_path,
        flush_every_rows=int(args.log_flush_rows),
        flush_interval_s=float(args.log_flush_interval_s),
        use_background_process=bool(args.log_background_process),
    )
    device = None
    producer_task: asyncio.Task[None] | None = None
    clock_obj = None

    rows: list[dict[str, Any]] = []
    tone_offset_hz: float | None = None
    observer_tone_offset_hz: float | None = (
        None if args.observer_fixed_offset_hz is None else float(args.observer_fixed_offset_hz)
    )

    state = ControllerState(
        in_phase_mode=False,
        phase_int_rad_s=0.0,
        prev_phase_err_filt_rad=None,
        dac_code_cmd=int(dac_start),
        dac_code_rb=None,
        dac_volts=code_to_volts(int(dac_start), float(args.vdd_dac)),
    )

    unwrapper = SlipAwareUnwrapper(
        slip_trigger_rad=float(args.slip_trigger_rad),
        slip_max_cycles_per_update=int(args.slip_max_cycles_per_update),
    )
    observer_unwrapper = SlipAwareUnwrapper(
        slip_trigger_rad=float(args.slip_trigger_rad),
        slip_max_cycles_per_update=int(args.slip_max_cycles_per_update),
    )

    update_idx = 0
    bootstrap_blocks: list[PreparedBlock] = []
    wall_start_s = float(time.time())

    try:
        device = open_sdr(args)
        if float(args.settle_seconds) > 0.0:
            await asyncio.sleep(float(args.settle_seconds))

        queue_depth = args.async_queue_blocks
        if queue_depth is None:
            queue_depth = default_async_queue_blocks()
        queue_depth = max(int(queue_depth), 1)

        block_queue: asyncio.Queue[QueuedBlock | StreamSentinel] = asyncio.Queue(maxsize=queue_depth)
        producer_task = asyncio.create_task(
            stream_iq_blocks(device, stream_nsamp, nsamp, block_queue, runtime_stats)
        )

        if args.mode == "pid":
            clock_obj = cvh.ClockVCXO()
            clock_obj.open()
            if not args.skip_clock_init:
                clock_obj.init_clock_vcxo_28p8mhz(apr_ppm=int(args.apr_ppm))
            if args.apply_start_dac:
                rb0 = int(clock_obj.set_dac_code(int(dac_start), settle_s=float(args.fine_settle_s)))
                state.dac_code_cmd = int(rb0)
                state.dac_code_rb = int(rb0)
                state.dac_volts = float(clock_obj.dac_code_to_volts(int(rb0)))

        if bool(args.single_stage_phase_only):
            engage_phase_mode(state, args, pullin_err_hz=0.0)

        if args.target_offset_hz is not None:
            tone_offset_hz = float(args.target_offset_hz)
        if observer_tone_offset_hz is None and tone_offset_hz is not None:
            observer_tone_offset_hz = float(tone_offset_hz)

        stop_after_current: StreamSentinel | None = None

        while True:
            if args.run_seconds is not None and (
                float(time.time()) - float(wall_start_s)
            ) >= float(args.run_seconds):
                break
            if args.max_updates is not None and update_idx >= int(args.max_updates):
                break
            if stop_after_current is not None:
                if stop_after_current.error is not None:
                    raise RuntimeError("Async SDR stream terminated unexpectedly.") from stop_after_current.error
                break

            t_wait_start = time.perf_counter()
            try:
                item = await asyncio.wait_for(block_queue.get(), timeout=queue_get_timeout_s)
            except asyncio.TimeoutError:
                if producer_task is not None and producer_task.done():
                    exc = producer_task.exception()
                    if exc is not None:
                        raise RuntimeError("Async SDR stream task failed.") from exc
                    break
                continue
            t_wait_s = float(time.perf_counter() - t_wait_start)

            blk, end_item, skipped_blocks = drain_to_latest_block(block_queue, item)
            runtime_stats.consumer_blocks_skipped_latest_only += int(skipped_blocks)
            if blk is None:
                if end_item is not None and end_item.error is not None:
                    raise RuntimeError("Async SDR stream terminated unexpectedly.") from end_item.error
                break

            stop_after_current = end_item
            wall_block_age_s = max(0.0, float(time.time()) - float(blk.unix_time_s))

            t_measure_start = time.perf_counter()
            prepared_block = prepare_block(blk, float(args.sample_rate))

            if tone_offset_hz is None:
                bootstrap_blocks.append(prepared_block)
                target = max(1, int(args.tone_estimate_blocks))
                if len(bootstrap_blocks) < target:
                    continue
                stacked = np.stack(
                    [np.asarray(b.iq_centered, dtype=np.complex64).reshape(-1) for b in bootstrap_blocks],
                    axis=0,
                )
                tone_offset_hz = estimate_tone_offset_hz(
                    stacked,
                    sample_rate_hz=float(args.sample_rate),
                    dc_exclusion_hz=float(args.dc_exclusion_hz),
                )
                if observer_tone_offset_hz is None:
                    observer_tone_offset_hz = float(tone_offset_hz)
                runtime_stats.bootstrap_blocks_discarded += len(bootstrap_blocks)
                bootstrap_blocks.clear()
                continue

            obs_ref_hz = float(
                observer_tone_offset_hz if observer_tone_offset_hz is not None else tone_offset_hz
            )
            if abs(float(tone_offset_hz) - float(obs_ref_hz)) <= 1e-12:
                shared_demod = demodulate_prepared_block(
                    prepared_block,
                    sample_rate_hz=float(args.sample_rate),
                    tone_offset_hz=float(tone_offset_hz),
                    phase_detector_chunks=int(args.phase_detector_chunks),
                )
                meas = build_measurement(
                    prepared_block,
                    shared_demod,
                    update_idx=int(update_idx),
                    unwrapper=unwrapper,
                )
                observer_meas = build_measurement(
                    prepared_block,
                    shared_demod,
                    update_idx=int(update_idx),
                    unwrapper=observer_unwrapper,
                )
                runtime_stats.shared_demod_updates += 1
            else:
                meas = build_measurement(
                    prepared_block,
                    demodulate_prepared_block(
                        prepared_block,
                        sample_rate_hz=float(args.sample_rate),
                        tone_offset_hz=float(tone_offset_hz),
                        phase_detector_chunks=int(args.phase_detector_chunks),
                    ),
                    update_idx=int(update_idx),
                    unwrapper=unwrapper,
                )
                observer_meas = build_measurement(
                    prepared_block,
                    demodulate_prepared_block(
                        prepared_block,
                        sample_rate_hz=float(args.sample_rate),
                        tone_offset_hz=float(obs_ref_hz),
                        phase_detector_chunks=int(args.phase_detector_chunks),
                    ),
                    update_idx=int(update_idx),
                    unwrapper=observer_unwrapper,
                )
            if args.control_measurement_source == "observer":
                control_meas = observer_meas
            else:
                control_meas = meas
            t_measure_s = float(time.perf_counter() - t_measure_start)

            # Pull-in should prefer the update-to-update phase slope because it reflects
            # the inter-block drift that the DAC can actually correct. The intrablock
            # estimate is still useful as a fallback, but it is much noisier on live data.
            tracked_freq_residual_hz = float(meas.freq_err_update_hz)
            if not np.isfinite(tracked_freq_residual_hz):
                tracked_freq_residual_hz = float(meas.freq_err_intrablock_hz)
            if not np.isfinite(tracked_freq_residual_hz):
                tracked_freq_residual_hz = 0.0
            tracked_tone_offset_hz = float(meas.tone_offset_hz + tracked_freq_residual_hz)
            pullin_reference_offset_hz = float(obs_ref_hz)

            if args.run_seconds is not None and control_meas.elapsed_s > float(args.run_seconds):
                break

            t_control_start = time.perf_counter()
            ignore_control_update = bool(state.post_dac_ignore_updates > 0)
            if ignore_control_update:
                state.post_dac_ignore_updates = max(0, int(state.post_dac_ignore_updates) - 1)
            control_dt_s = float(control_meas.dt_update_s)
            if control_dt_s <= 0.0 or (not np.isfinite(control_dt_s)):
                control_dt_s = nominal_loop_period_s
            if state.in_phase_mode and np.isfinite(control_meas.phase_fit_slope_hz):
                phase_freq_err_hz = float(control_meas.phase_fit_slope_hz)
            else:
                phase_freq_err_hz = float(control_meas.freq_err_update_hz)
                if not np.isfinite(phase_freq_err_hz):
                    phase_freq_err_hz = float(control_meas.freq_err_intrablock_hz)
            if not np.isfinite(phase_freq_err_hz):
                phase_freq_err_hz = 0.0
            phase_reference_mode = str(args.phase_reference_mode)
            phase_reference_rad, phase_reference_valid = current_phase_reference(
                state,
                phase_reference_mode,
            )
            phase_err_absolute_raw = wrap_to_pi(
                float(control_meas.wrapped_phase_rad) - float(args.phase_setpoint_rad)
            )
            phase_err_relative_raw = (
                wrap_to_pi(
                    float(control_meas.wrapped_phase_rad)
                    - float(phase_reference_rad)
                    - float(args.phase_setpoint_rad)
                )
                if phase_reference_valid
                else float("nan")
            )
            phase_err_raw = float(
                phase_err_relative_raw if phase_reference_valid else phase_err_absolute_raw
            )
            phase_err_control_rad = float(phase_err_raw)
            phase_err_update_rad = float(phase_err_raw)
            phase_detector_ok = phase_detector_quality_ok(control_meas, args)
            phase_measure_pred_rad = float("nan")
            phase_predict_rate_hz = float("nan")
            phase_measure_innovation_rad = float("nan")
            phase_innovation_ok = True
            phase_innovation_softened = False
            phase_state_predicted = False
            phase_predict_through_holds = bool(args.phase_predict_through_holds)
            phase_innovation_soft_limit_rad = float(args.phase_innovation_max_rad)
            phase_innovation_hard_limit_rad = float(args.phase_innovation_hard_max_rad)
            phase_innovation_soft_blend = clamp(float(args.phase_innovation_soft_blend), 0.0, 1.0)
            phase_innovation_soft_fit_rms_max_rad = float(args.phase_innovation_soft_fit_rms_max_rad)
            if (
                phase_innovation_soft_limit_rad > 0.0
                and phase_innovation_hard_limit_rad <= 0.0
            ):
                phase_innovation_hard_limit_rad = 2.0 * float(phase_innovation_soft_limit_rad)
            if (
                phase_innovation_soft_limit_rad > 0.0
                and phase_innovation_hard_limit_rad > 0.0
                and phase_innovation_hard_limit_rad < phase_innovation_soft_limit_rad
            ):
                phase_innovation_hard_limit_rad = float(phase_innovation_soft_limit_rad)
            if (
                state.in_phase_mode
                and phase_reference_valid
                and state.prev_phase_err_filt_rad is not None
                and np.isfinite(float(state.prev_phase_err_filt_rad))
            ):
                predicted_rate_hz = float(phase_freq_err_hz)
                if (
                    state.phase_hold_rate_lpf_hz is not None
                    and np.isfinite(float(state.phase_hold_rate_lpf_hz))
                ):
                    predicted_rate_hz = float(state.phase_hold_rate_lpf_hz)
                elif (
                    state.phase_freq_err_lpf_hz is not None
                    and np.isfinite(float(state.phase_freq_err_lpf_hz))
                ):
                    predicted_rate_hz = float(state.phase_freq_err_lpf_hz)
                phase_predict_rate_hz = float(predicted_rate_hz)
                phase_measure_pred_rad = wrap_to_pi(
                    float(state.prev_phase_err_filt_rad)
                    + (2.0 * math.pi * float(predicted_rate_hz) * float(control_dt_s))
                )
                if (not ignore_control_update) and phase_innovation_soft_limit_rad > 0.0:
                    phase_measure_innovation_rad = wrap_to_pi(
                        float(phase_err_raw) - float(phase_measure_pred_rad)
                    )
                    abs_phase_measure_innovation_rad = abs(float(phase_measure_innovation_rad))
                    soft_fit_ok = True
                    if phase_innovation_soft_fit_rms_max_rad > 0.0:
                        soft_fit_ok = bool(
                            np.isfinite(float(control_meas.phase_fit_rms_rad))
                            and float(control_meas.phase_fit_rms_rad)
                            <= phase_innovation_soft_fit_rms_max_rad
                        )
                    if (
                        phase_innovation_hard_limit_rad > 0.0
                        and abs_phase_measure_innovation_rad > phase_innovation_hard_limit_rad
                    ):
                        phase_innovation_ok = False
                    elif (
                        abs_phase_measure_innovation_rad > phase_innovation_soft_limit_rad
                        and soft_fit_ok
                        and phase_innovation_soft_blend > 0.0
                    ):
                        phase_innovation_softened = True
                        innovation_used_rad = clamp(
                            float(phase_measure_innovation_rad),
                            -float(phase_innovation_soft_limit_rad),
                            float(phase_innovation_soft_limit_rad),
                        )
                        phase_err_update_rad = wrap_to_pi(
                            float(phase_measure_pred_rad) + float(innovation_used_rad)
                        )
                        phase_err_control_rad = wrap_to_pi(
                            float(phase_measure_pred_rad)
                            + (float(phase_innovation_soft_blend) * float(innovation_used_rad))
                        )
                        phase_err_update_rad = float(phase_err_control_rad)
                    elif abs_phase_measure_innovation_rad > phase_innovation_soft_limit_rad:
                        phase_innovation_ok = False
            phase_measurement_rejected = False
            if ignore_control_update:
                if phase_predict_through_holds and np.isfinite(float(phase_measure_pred_rad)):
                    phase_err_filt = circular_phase_lpf_force(
                        float(phase_measure_pred_rad),
                        state.phase_err_lpf,
                    )
                    phase_err_control_rad = float(phase_err_filt)
                    phase_state_predicted = True
                elif (
                    state.prev_phase_err_filt_rad is not None
                    and np.isfinite(float(state.prev_phase_err_filt_rad))
                ):
                    phase_err_filt = float(state.prev_phase_err_filt_rad)
                    phase_err_control_rad = float(phase_err_filt)
                else:
                    phase_err_filt = float(phase_err_raw)
                    phase_err_control_rad = float(phase_err_filt)
            elif state.in_phase_mode and ((not phase_detector_ok) or (not phase_innovation_ok)):
                phase_measurement_rejected = True
                if phase_predict_through_holds and np.isfinite(float(phase_measure_pred_rad)):
                    phase_err_filt = circular_phase_lpf_force(
                        float(phase_measure_pred_rad),
                        state.phase_err_lpf,
                    )
                    phase_err_control_rad = float(phase_err_filt)
                    phase_state_predicted = True
                elif (
                    state.prev_phase_err_filt_rad is not None
                    and np.isfinite(float(state.prev_phase_err_filt_rad))
                ):
                    phase_err_filt = float(state.prev_phase_err_filt_rad)
                    phase_err_control_rad = float(phase_err_filt)
                else:
                    phase_err_filt = float(phase_err_raw)
                    phase_err_control_rad = float(phase_err_filt)
            else:
                phase_err_filt = circular_phase_lpf_update(
                    phase_err_update_rad,
                    float(args.phase_filter_alpha),
                    state.phase_err_lpf,
                )
            phase_hold_err_abs = (
                abs(float(phase_err_control_rad))
                if np.isfinite(float(phase_err_control_rad))
                else abs(float(phase_err_filt))
            )

            tone_track_step_hz = 0.0
            # Optional tone-reference tracking to keep demod near DC while clock changes.
            # This uses pull-in quality rather than raw coherence because tone tracking
            # should remain available while the residual tone is still outside the
            # narrow raw-coherence region of the current block length.
            if (
                (not ignore_control_update)
                and (not state.in_phase_mode)
                and float(args.tone_track_alpha) > 0.0
                and np.isfinite(meas.freq_err_intrablock_hz)
                and (meas.pullin_quality >= float(args.tone_track_min_coh))
            ):
                tone_track_step_hz = clamp(
                    float(args.tone_track_alpha) * float(meas.freq_err_intrablock_hz),
                    -float(args.tone_track_max_step_hz),
                    float(args.tone_track_max_step_hz),
                )
                tone_offset_hz = float(tone_offset_hz + tone_track_step_hz)

            observer_rf_hz = float(args.center_freq) + float(obs_ref_hz)
            if np.isfinite(observer_rf_hz) and abs(observer_rf_hz) > 1e-6:
                observer_tau_err_ns = float(
                    observer_meas.unwrapped_phase_rad / (2.0 * math.pi * observer_rf_hz) * 1e9
                )
            else:
                observer_tau_err_ns = float("nan")

            pullin_err_raw_hz = float(
                (tracked_tone_offset_hz - pullin_reference_offset_hz) - float(args.pullin_target_hz)
            )
            if (not ignore_control_update) and np.isfinite(pullin_err_raw_hz):
                state.pullin_err_lpf_hz = scalar_lpf_update(
                    float(pullin_err_raw_hz),
                    float(args.pullin_filter_alpha),
                    state.pullin_err_lpf_hz,
                )
            pullin_err_hz = float(
                state.pullin_err_lpf_hz
                if state.pullin_err_lpf_hz is not None and np.isfinite(state.pullin_err_lpf_hz)
                else pullin_err_raw_hz
            )

            stage_switch_event = 0
            phase_coh_for_control = float(control_meas.coherence)
            pullin_quality_for_control = float(control_meas.pullin_quality)
            phase_coh_ok = bool(
                np.isfinite(phase_coh_for_control)
                and (phase_coh_for_control >= float(args.coh_min))
            )
            pullin_quality_ok = bool(
                np.isfinite(pullin_quality_for_control)
                and (pullin_quality_for_control >= float(args.pullin_quality_min))
            )
            phase_err_for_entry_ok = (
                (float(args.phase_enter_max_err_rad) <= 0.0)
                or (
                    np.isfinite(phase_err_filt)
                    and (abs(float(phase_err_filt)) <= float(args.phase_enter_max_err_rad))
                )
            )
            if str(args.phase_enter_condition) == "pullin_only":
                enter_ok = bool(
                    np.isfinite(pullin_err_hz)
                    and (abs(pullin_err_hz) <= float(args.pullin_switch_hz))
                )
            elif str(args.phase_enter_condition) == "residual_and_quality":
                enter_ok = bool(
                    pullin_quality_ok
                    and phase_detector_ok
                    and np.isfinite(pullin_err_hz)
                    and (abs(pullin_err_hz) <= float(args.pullin_switch_hz))
                )
            else:
                enter_ok = (
                    pullin_quality_ok
                    and phase_detector_ok
                    and (abs(pullin_err_hz) <= float(args.pullin_switch_hz))
                    and phase_err_for_entry_ok
                )
            if str(args.phase_exit_condition) == "residual_only":
                exit_ok = bool(
                    np.isfinite(pullin_err_hz)
                    and (abs(pullin_err_hz) >= float(args.pullin_exit_hz))
                )
            elif str(args.phase_exit_condition) == "residual_and_phase":
                phase_err_for_exit_ok = (
                    (float(args.phase_exit_min_err_rad) <= 0.0)
                    or (
                        np.isfinite(phase_err_filt)
                        and (abs(float(phase_err_filt)) >= float(args.phase_exit_min_err_rad))
                    )
                )
                exit_ok = bool(
                    np.isfinite(pullin_err_hz)
                    and (abs(pullin_err_hz) >= float(args.pullin_exit_hz))
                    and (
                        phase_err_for_exit_ok
                        or (not phase_detector_ok)
                        or (not pullin_quality_ok)
                    )
                )
            else:
                exit_ok = (
                    (not phase_detector_ok)
                    or (not pullin_quality_ok)
                    or (abs(pullin_err_hz) >= float(args.pullin_exit_hz))
                )
            if ignore_control_update:
                pass
            elif bool(args.single_stage_phase_only):
                state.in_phase_mode = True
                state.pullin_lock_counter = 0
                state.pullin_unlock_counter = 0
                state.pullin_lock_dwell_s = 0.0
                state.pullin_unlock_dwell_s = 0.0
                state.phase_mode_updates += 1
                state.phase_mode_elapsed_s += float(control_dt_s)
            elif not state.in_phase_mode:
                if enter_ok:
                    state.pullin_lock_counter += 1
                    state.pullin_lock_dwell_s += float(control_dt_s)
                else:
                    state.pullin_lock_counter = 0
                    state.pullin_lock_dwell_s = 0.0
                state.pullin_unlock_counter = 0
                state.pullin_unlock_dwell_s = 0.0
                state.phase_mode_updates = 0
                state.phase_mode_elapsed_s = 0.0
                if state.pullin_lock_dwell_s >= float(pullin_enter_dwell_s_target):
                    engage_phase_mode(state, args, pullin_err_hz=float(pullin_err_hz))
                    stage_switch_event = 1
                    state.stage_switches += 1
            else:
                state.pullin_lock_counter = 0
                state.pullin_lock_dwell_s = 0.0
                state.phase_mode_updates += 1
                state.phase_mode_elapsed_s += float(control_dt_s)
                if (
                    state.phase_mode_elapsed_s >= float(phase_min_hold_s_target)
                    and exit_ok
                ):
                    state.pullin_unlock_counter += 1
                    state.pullin_unlock_dwell_s += float(control_dt_s)
                else:
                    state.pullin_unlock_counter = 0
                    state.pullin_unlock_dwell_s = 0.0
                if state.pullin_unlock_dwell_s >= float(pullin_exit_dwell_s_target):
                    state.in_phase_mode = False
                    clear_phase_reference(state)
                    state.phase_center_code = None
                    state.phase_center_trim_codes = 0.0
                    state.phase_substage = "acquire"
                    state.phase_hold_lock_counter = 0
                    state.phase_hold_unlock_counter = 0
                    state.phase_freq_err_lpf_hz = None
                    state.phase_step_sigma_codes = 0.0
                    state.phase_mode_updates = 0
                    state.phase_mode_elapsed_s = 0.0
                    state.pullin_unlock_counter = 0
                    state.pullin_unlock_dwell_s = 0.0
                    stage_switch_event = 1
                    state.stage_switches += 1

            stage = "phase_lock" if state.in_phase_mode else "pull_in"
            observer_track_step_hz = 0.0
            observer_recentered_on_entry = False
            observer_target_offset_hz = float(
                tracked_tone_offset_hz - float(args.pullin_target_hz)
            )
            observer_ref_can_move = bool(
                args.observer_fixed_offset_hz is None
                and observer_tone_offset_hz is not None
                and np.isfinite(observer_target_offset_hz)
            )
            if (
                (not ignore_control_update)
                and observer_ref_can_move
                and state.in_phase_mode
                and stage_switch_event == 1
                and bool(args.phase_observer_recenter_on_entry)
            ):
                observer_track_step_hz = clamp(
                    float(observer_target_offset_hz - float(observer_tone_offset_hz)),
                    -float(args.phase_observer_recenter_max_step_hz),
                    float(args.phase_observer_recenter_max_step_hz),
                )
                if abs(float(observer_track_step_hz)) > 1e-12:
                    observer_tone_offset_hz = float(observer_tone_offset_hz + observer_track_step_hz)
                    observer_recentered_on_entry = True
                    observer_unwrapper.reset()
                    if args.control_measurement_source == "observer":
                        clear_phase_reference(state)
            elif (
                (not ignore_control_update)
                and observer_ref_can_move
                and state.in_phase_mode
                and float(args.phase_observer_track_alpha) > 0.0
                and np.isfinite(meas.pullin_quality)
                and (float(meas.pullin_quality) >= float(args.phase_observer_track_min_quality))
            ):
                observer_track_step_hz = clamp(
                    float(args.phase_observer_track_alpha)
                    * float(observer_target_offset_hz - float(observer_tone_offset_hz)),
                    -float(args.phase_observer_track_max_step_hz),
                    float(args.phase_observer_track_max_step_hz),
                )
                observer_tone_offset_hz = float(observer_tone_offset_hz + observer_track_step_hz)

            if (
                phase_reference_mode == "first_valid"
                and state.in_phase_mode
                and (not phase_reference_valid)
                and (not ignore_control_update)
                and (
                    (args.control_measurement_source != "observer")
                    or (not observer_recentered_on_entry)
                )
                and phase_detector_ok
                and pullin_quality_ok
            ):
                set_phase_reference(state, float(control_meas.wrapped_phase_rad))
                phase_reference_rad, phase_reference_valid = current_phase_reference(
                    state,
                    phase_reference_mode,
                )
                phase_err_relative_raw = wrap_to_pi(
                    float(control_meas.wrapped_phase_rad)
                    - float(phase_reference_rad)
                    - float(args.phase_setpoint_rad)
                )
                phase_err_raw = float(phase_err_relative_raw)
                phase_err_control_rad = float(phase_err_raw)
                phase_err_update_rad = float(phase_err_raw)
                phase_measure_pred_rad = float("nan")
                phase_predict_rate_hz = float("nan")
                phase_measure_innovation_rad = float("nan")
                phase_innovation_ok = True
                phase_innovation_softened = False
                phase_measurement_rejected = False
                phase_state_predicted = False
                phase_err_filt = circular_phase_lpf_update(
                    phase_err_update_rad,
                    float(args.phase_filter_alpha),
                    state.phase_err_lpf,
                )
                phase_hold_err_abs = abs(float(phase_err_control_rad))
            phase_reference_rad, phase_reference_valid = current_phase_reference(
                state,
                phase_reference_mode,
            )
            phase_err_relative_raw = (
                wrap_to_pi(
                    float(control_meas.wrapped_phase_rad)
                    - float(phase_reference_rad)
                    - float(args.phase_setpoint_rad)
                )
                if phase_reference_valid
                else float("nan")
            )
            if phase_reference_valid:
                phase_err_raw = float(phase_err_relative_raw)
            else:
                phase_err_raw = float(phase_err_absolute_raw)
                if (not phase_state_predicted) or (not state.in_phase_mode):
                    phase_err_control_rad = float(phase_err_raw)

            phase_err_accepted_raw_rad = (
                float(phase_err_relative_raw)
                if (
                    state.in_phase_mode
                    and phase_reference_valid
                    and (not ignore_control_update)
                    and (
                        (args.control_measurement_source != "observer")
                        or (not observer_recentered_on_entry)
                    )
                    and phase_detector_ok
                    and pullin_quality_ok
                    and phase_innovation_ok
                    and (not phase_measurement_rejected)
                )
                else float("nan")
            )
            if np.isfinite(float(phase_err_accepted_raw_rad)):
                append_accepted_raw_phase_sample(
                    state,
                    args,
                    abs_phase_err_rad=abs(float(phase_err_accepted_raw_rad)),
                    dt_s=float(control_dt_s),
                )

            phase_accepted_raw_enter_fraction = float("nan")
            phase_accepted_raw_exit_fraction = float("nan")
            phase_accepted_raw_enter_ok = False
            phase_accepted_raw_exit_ok = False
            if state.in_phase_mode and phase_reference_valid:
                if int(args.phase_accepted_raw_enter_window_updates) > 0:
                    (
                        phase_accepted_raw_enter_fraction,
                        phase_accepted_raw_enter_ready,
                    ) = accepted_raw_window_fraction(
                        state,
                        threshold_rad=float(args.phase_hold_enter_err_rad),
                        window_updates=int(args.phase_accepted_raw_enter_window_updates),
                    )
                    phase_accepted_raw_enter_ok = bool(
                        phase_accepted_raw_enter_ready
                        and np.isfinite(float(phase_accepted_raw_enter_fraction))
                        and (
                            float(phase_accepted_raw_enter_fraction)
                            >= float(args.phase_accepted_raw_enter_min_frac)
                        )
                    )
                else:
                    phase_accepted_raw_enter_ok = bool(
                        np.isfinite(float(phase_hold_err_abs))
                        and (
                            float(phase_hold_err_abs)
                            <= float(args.phase_hold_enter_err_rad)
                        )
                    )

                if int(args.phase_accepted_raw_exit_window_updates) > 0:
                    (
                        phase_accepted_raw_exit_fraction,
                        phase_accepted_raw_exit_ready,
                    ) = accepted_raw_window_fraction(
                        state,
                        threshold_rad=float(args.phase_hold_exit_err_rad),
                        window_updates=int(args.phase_accepted_raw_exit_window_updates),
                    )
                    phase_accepted_raw_exit_ok = bool(
                        phase_accepted_raw_exit_ready
                        and np.isfinite(float(phase_accepted_raw_exit_fraction))
                        and (
                            float(phase_accepted_raw_exit_fraction)
                            >= float(args.phase_accepted_raw_exit_min_frac)
                        )
                    )
                else:
                    phase_accepted_raw_exit_ok = bool(
                        np.isfinite(float(phase_hold_err_abs))
                        and (
                            float(phase_hold_err_abs)
                            <= float(args.phase_hold_exit_err_rad)
                        )
                    )

            if state.in_phase_mode:
                if not phase_reference_valid:
                    state.phase_substage = "acquire"
                    state.phase_hold_lock_counter = 0
                    state.phase_hold_unlock_counter = 0
                    state.phase_hold_int_hz = 0.0
                elif state.phase_substage == "hold":
                    state.phase_hold_lock_counter = 0
                    if phase_accepted_raw_exit_ok:
                        state.phase_hold_unlock_counter = 0
                    else:
                        state.phase_hold_unlock_counter += 1
                    if (
                        state.phase_hold_unlock_counter
                        >= int(args.phase_hold_exit_dwell_updates)
                    ):
                        state.phase_substage = "acquire"
                        state.phase_hold_lock_counter = 0
                        state.phase_hold_unlock_counter = 0
                        state.phase_hold_int_hz = 0.0
                else:
                    state.phase_hold_unlock_counter = 0
                    if phase_accepted_raw_enter_ok:
                        state.phase_hold_lock_counter += 1
                    else:
                        state.phase_hold_lock_counter = 0
                    if (
                        state.phase_hold_lock_counter
                        >= int(args.phase_hold_enter_dwell_updates)
                    ):
                        state.phase_substage = "hold"
                        state.phase_hold_lock_counter = 0
                        state.phase_hold_unlock_counter = 0

            phase_reference_ramp_scale = (
                phase_reference_gain_scale(
                    state,
                    args,
                    phase_reference_valid=phase_reference_valid,
                )
                if state.in_phase_mode
                else 0.0
            )
            phase_substage = str(state.phase_substage) if state.in_phase_mode else "pull_in"

            action = "hold"
            u_hz_raw = 0.0
            u_hz_limited = 0.0
            dac_step_req = 0
            dac_step_apply_target = 0
            dac_step_applied = 0
            dac_code_proposed = int(state.dac_code_cmd)
            dac_code_apply_target = int(state.dac_code_cmd)
            phase_target_float = float(state.dac_code_cmd)
            phase_target_code = int(state.dac_code_cmd)
            phase_fraction_cycles_per_sample = 0.0
            phase_freq_err_lpf_hz = float("nan")
            phase_hold_rate_lpf_hz = float("nan")
            phase_reference_pending = bool(
                state.in_phase_mode
                and phase_reference_mode == "first_valid"
                and (not phase_reference_valid)
            )
            phase_accepted_raw_window_count = int(len(state.phase_accepted_raw_abs_window))

            if bool(args.single_stage_phase_only):
                control_quality_ok = bool(pullin_quality_ok and (not phase_reference_pending))
            else:
                control_quality_ok = (
                    pullin_quality_ok
                    if not state.in_phase_mode
                    else (
                        (not phase_reference_pending)
                        and phase_detector_ok
                        and pullin_quality_ok
                        and phase_innovation_ok
                    )
                )
            if ignore_control_update:
                action = "post_dac_predict" if phase_state_predicted else "post_dac_ignore"
            elif observer_recentered_on_entry and args.control_measurement_source == "observer":
                action = "phase_recenter_hold"
            elif phase_reference_pending:
                action = "phase_reference_hold"
            elif phase_measurement_rejected and state.in_phase_mode:
                if not phase_detector_ok:
                    action = "hold_bad_phase_fit_predict" if phase_state_predicted else "hold_bad_phase_fit"
                else:
                    action = (
                        "hold_bad_phase_innovation_predict"
                        if phase_state_predicted
                        else "hold_bad_phase_innovation"
                    )
            elif not control_quality_ok:
                if not pullin_quality_ok:
                    action = "hold_low_pullin_quality"
                else:
                    action = "hold_bad_phase_measure"
            else:
                if not state.in_phase_mode:
                    u_hz_raw = float(args.Kf_pullin_hz_per_hz) * pullin_err_hz
                    action = "pull_in"
                else:
                    if state.phase_center_code is None:
                        state.phase_center_code = int(state.dac_code_cmd)
                    dt = float(control_meas.dt_update_s)
                    if dt <= 0.0 or (not np.isfinite(dt)):
                        dt = nominal_loop_period_s
                    residual_freq_hz = float(phase_freq_err_hz)
                    if not np.isfinite(residual_freq_hz):
                        residual_freq_hz = float(pullin_err_hz)
                    if np.isfinite(residual_freq_hz):
                        residual_freq_hz = clamp(
                            float(residual_freq_hz),
                            -float(args.phase_center_freq_max_hz),
                            float(args.phase_center_freq_max_hz),
                        )
                        state.phase_freq_err_lpf_hz = scalar_lpf_update(
                            residual_freq_hz,
                            float(args.phase_center_freq_filter_alpha),
                            state.phase_freq_err_lpf_hz,
                        )
                    phase_freq_err_lpf_hz = float(
                        state.phase_freq_err_lpf_hz
                        if state.phase_freq_err_lpf_hz is not None
                        and np.isfinite(state.phase_freq_err_lpf_hz)
                        else residual_freq_hz
                    )
                    phase_err_for_pd = (
                        clamp(
                            float(phase_err_control_rad),
                            -float(args.phase_hold_detector_limit_rad),
                            float(args.phase_hold_detector_limit_rad),
                        )
                        if np.isfinite(phase_err_control_rad)
                        else 0.0
                    )
                    if abs(float(phase_err_for_pd)) <= float(args.phase_hold_deadband_rad):
                        phase_err_for_pd = 0.0
                    else:
                        phase_err_for_pd = float(
                            math.copysign(
                                abs(float(phase_err_for_pd)) - float(args.phase_hold_deadband_rad),
                                float(phase_err_for_pd),
                            )
                        )
                    phase_err_pd = float(math.sin(float(phase_err_for_pd)))
                    hold_rate_raw_hz = float(phase_freq_err_hz)
                    if not np.isfinite(hold_rate_raw_hz):
                        hold_rate_raw_hz = 0.0
                    hold_rate_raw_hz = clamp(
                        float(hold_rate_raw_hz),
                        -float(args.phase_hold_rate_max_hz),
                        float(args.phase_hold_rate_max_hz),
                    )
                    state.phase_hold_rate_lpf_hz = scalar_lpf_update(
                        float(hold_rate_raw_hz),
                        float(args.phase_hold_rate_filter_alpha),
                        state.phase_hold_rate_lpf_hz,
                    )
                    phase_hold_rate_hz = float(
                        state.phase_hold_rate_lpf_hz
                        if state.phase_hold_rate_lpf_hz is not None
                        and np.isfinite(state.phase_hold_rate_lpf_hz)
                        else hold_rate_raw_hz
                    )
                    phase_hold_rate_lpf_hz = float(phase_hold_rate_hz)

                    hold_kp_hz_per_rad = (
                        float(args.phase_hold_kp_hz_per_rad)
                        * float(phase_reference_ramp_scale)
                    )
                    hold_ki_hz_per_rad_s = (
                        float(args.phase_hold_ki_hz_per_rad_s)
                        * float(phase_reference_ramp_scale)
                        if state.phase_substage == "hold"
                        else 0.0
                    )
                    hold_kd_hz_per_hz = (
                        float(args.phase_hold_kd_hz_per_hz)
                        * float(phase_reference_ramp_scale)
                    )
                    phase_hold_max_u_hz = (
                        float(args.phase_hold_max_u_hz)
                        * float(phase_reference_ramp_scale)
                    )
                    max_phase_step_base = int(args.max_phase_hold_dac_step_per_update)
                    if max_phase_step_base > 0 and float(phase_reference_ramp_scale) > 0.0:
                        max_phase_step = max(
                            1,
                            int(round(float(max_phase_step_base) * float(phase_reference_ramp_scale))),
                        )
                    else:
                        max_phase_step = 0
                    phase_center_trim_codes = float(state.phase_center_trim_codes)
                    phase_hold_center_freq_gain = (
                        float(args.phase_hold_center_freq_gain_codes_per_hz_s)
                        * float(phase_reference_ramp_scale)
                    )
                    if (
                        state.phase_substage != "hold"
                        or (not np.isfinite(dt))
                        or float(phase_hold_err_abs) > float(args.phase_hold_integral_max_err_rad)
                        or hold_ki_hz_per_rad_s <= 0.0
                    ):
                        state.phase_hold_int_hz = 0.0
                    else:
                        state.phase_hold_int_hz = clamp(
                            float(state.phase_hold_int_hz)
                            + (
                                float(hold_ki_hz_per_rad_s)
                                * float(phase_err_pd)
                                * float(dt)
                            ),
                            -float(args.phase_hold_integral_limit_hz),
                            float(args.phase_hold_integral_limit_hz),
                        )

                    u_hz_raw = (
                        (float(hold_kp_hz_per_rad) * float(phase_err_pd))
                        + float(state.phase_hold_int_hz)
                        - (float(hold_kd_hz_per_hz) * float(phase_hold_rate_hz))
                    )
                    u_hz_limited = clamp(
                        float(u_hz_raw),
                        -float(phase_hold_max_u_hz),
                        float(phase_hold_max_u_hz),
                    )
                    if (
                        abs(float(phase_hold_center_freq_gain)) > 1e-18
                        and np.isfinite(dt)
                        and np.isfinite(phase_hold_rate_hz)
                    ):
                        phase_center_trim_codes += (
                            float(args.control_sign)
                            * float(phase_hold_center_freq_gain)
                            * float(phase_hold_rate_hz)
                            * float(dt)
                        )
                        phase_center_trim_codes = clamp(
                            phase_center_trim_codes,
                            -float(args.phase_center_integral_limit_codes),
                            float(args.phase_center_integral_limit_codes),
                        )
                    state.phase_center_trim_codes = float(phase_center_trim_codes)
                    if abs(float(args.kv_hz_per_code)) <= 1e-12:
                        action = "hold_invalid_kv"
                        u_hz_raw = 0.0
                        u_hz_limited = 0.0
                        phase_target_float = float(state.dac_code_cmd)
                        phase_target_code = int(state.dac_code_cmd)
                    else:
                        phase_target_float = (
                            float(state.phase_center_code)
                            + float(phase_center_trim_codes)
                            + (
                                float(args.control_sign)
                                * float(u_hz_limited)
                                / float(args.kv_hz_per_code)
                            )
                        )
                        phase_target_code = clamp_int(
                            float(phase_target_float),
                            int(code_min),
                            int(code_max),
                        )
                    phase_fraction_cycles_per_sample = float("nan")
                    state.phase_int_rad_s = 0.0
                    phase_step_float = clamp(
                        float(phase_target_float) - float(state.dac_code_cmd),
                        -float(max_phase_step),
                        float(max_phase_step),
                    )
                    sigma_input = float(phase_step_float + float(state.phase_step_sigma_codes))
                    dac_step_req = clamp_int(
                        sigma_input,
                        -int(max_phase_step),
                        int(max_phase_step),
                    )
                    dac_code_proposed = clamp_int(
                        int(state.dac_code_cmd) + int(dac_step_req),
                        int(code_min),
                        int(code_max),
                    )
                    if (
                        (dac_code_proposed == int(code_max) and dac_step_req > 0)
                        or (dac_code_proposed == int(code_min) and dac_step_req < 0)
                    ):
                        state.phase_step_sigma_codes = 0.0
                    else:
                        state.phase_step_sigma_codes = clamp(
                            float(sigma_input) - float(dac_step_req),
                            -0.5,
                            0.5,
                        )
                    action = f"phase_{state.phase_substage}"

                if not state.in_phase_mode:
                    u_hz_limited = clamp(float(u_hz_raw), -float(args.max_u_hz), float(args.max_u_hz))
                    if abs(float(args.kv_hz_per_code)) <= 1e-12:
                        action = "hold_invalid_kv"
                        u_hz_limited = 0.0
                    else:
                        delta_codes = (
                            float(args.control_sign)
                            * float(u_hz_limited)
                            / float(args.kv_hz_per_code)
                        )
                        delta_codes = clamp(
                            delta_codes,
                            -float(args.max_dac_step_per_update),
                            float(args.max_dac_step_per_update),
                        )
                        dac_step_req = int(round(delta_codes))
                        dac_code_proposed = clamp_int(
                            int(state.dac_code_cmd) + int(dac_step_req),
                            int(code_min),
                            int(code_max),
                        )

                    saturated = (
                        dac_code_proposed == int(code_min)
                        or dac_code_proposed == int(code_max)
                    )
                    pushes_outward = (
                        (dac_code_proposed == int(code_max) and dac_step_req > 0)
                        or (dac_code_proposed == int(code_min) and dac_step_req < 0)
                    )
                    clipped_u = abs(float(u_hz_raw)) > (abs(float(args.max_u_hz)) + 1e-12)
                else:
                    saturated = False
                    pushes_outward = False
                    clipped_u = False

            t_control_s = float(time.perf_counter() - t_control_start)
            t_dac_s = 0.0
            dac_step_apply_target = int(dac_step_req)
            dac_code_apply_target = int(dac_code_proposed)

            if args.mode == "pid":
                if sham_rng is not None and int(dac_step_req) != 0:
                    step_mag = abs(int(dac_step_req))
                    step_sign = -1 if int(sham_rng.integers(0, 2)) == 0 else 1
                    dac_step_apply_target = int(step_sign * step_mag)
                    dac_code_apply_target = clamp_int(
                        int(state.dac_code_cmd) + int(dac_step_apply_target),
                        int(code_min),
                        int(code_max),
                    )
                    if dac_code_apply_target == int(state.dac_code_cmd):
                        alt_step = int(-dac_step_apply_target)
                        alt_code = clamp_int(
                            int(state.dac_code_cmd) + int(alt_step),
                            int(code_min),
                            int(code_max),
                        )
                        if alt_code != int(state.dac_code_cmd):
                            dac_step_apply_target = int(alt_step)
                            dac_code_apply_target = int(alt_code)
                if dac_code_apply_target != int(state.dac_code_cmd):
                    t_dac_start = time.perf_counter()
                    if clock_obj is not None:
                        rb = int(
                            clock_obj.set_dac_code(
                                int(dac_code_apply_target),
                                settle_s=float(args.fine_settle_s),
                            )
                        )
                    else:
                        rb = int(dac_code_apply_target)

                    dac_step_applied = int(rb - int(state.dac_code_cmd))
                    state.dac_code_cmd = int(rb)
                    state.dac_code_rb = int(rb)
                    if clock_obj is not None:
                        state.dac_volts = float(clock_obj.dac_code_to_volts(int(rb)))
                    else:
                        state.dac_volts = code_to_volts(int(rb), float(args.vdd_dac))
                    t_dac_s = float(time.perf_counter() - t_dac_start)

                    if state.in_phase_mode:
                        state.phase_updates_applied += 1
                    else:
                        state.pullin_updates_applied += 1
                    state.post_dac_ignore_updates = max(
                        0,
                        int(args.ignore_blocks_after_dac_write),
                    )

                    if sham_rng is not None:
                        action = f"{action}_sham_apply"
                    else:
                        action = f"{action}_apply"
                else:
                    if sham_rng is not None and int(dac_step_req) != 0:
                        action = f"{action}_sham_hold"
                    else:
                        action = f"{action}_hold"
            else:
                action = f"monitor_{stage}"

            if (
                ((not ignore_control_update) or phase_state_predicted)
                and (
                    (not state.in_phase_mode)
                    or phase_reference_mode != "first_valid"
                    or phase_reference_valid
                )
            ):
                state.prev_phase_err_filt_rad = float(phase_err_filt)
            t_total_s = float(t_measure_s + t_control_s + t_dac_s)

            row: dict[str, Any] = {
                "unix_time_s": f"{control_meas.unix_time_s:.9f}",
                "elapsed_s": f"{control_meas.elapsed_s:.9f}",
                "update_idx": int(update_idx),
                "mode": str(args.mode),
                "stage": stage,
                "action": action,
                "stage_switch_event": int(stage_switch_event),
                "sample_start": int(control_meas.sample_start),
                "tone_offset_hz": f"{float(tone_offset_hz):.9f}",
                "tone_track_step_hz": f"{float(tone_track_step_hz):.9f}",
                "observer_tone_offset_hz": f"{float(obs_ref_hz):.9f}",
                "observer_track_step_hz": f"{float(observer_track_step_hz):.9f}",
                "tracked_tone_offset_hz": f"{float(tracked_tone_offset_hz):.9f}",
                "pullin_reference_offset_hz": f"{float(pullin_reference_offset_hz):.9f}",
                "phase_wrapped_rad": f"{control_meas.wrapped_phase_rad:.9f}",
                "phase_unwrapped_rad": f"{control_meas.unwrapped_phase_rad:.9f}",
                "observer_phase_wrapped_rad": f"{observer_meas.wrapped_phase_rad:.9f}",
                "observer_phase_unwrapped_rad": f"{observer_meas.unwrapped_phase_rad:.9f}",
                "control_measurement_source": str(args.control_measurement_source),
                "control_phase_wrapped_rad": f"{control_meas.wrapped_phase_rad:.9f}",
                "control_phase_unwrapped_rad": f"{control_meas.unwrapped_phase_rad:.9f}",
                "phase_reference_mode": str(phase_reference_mode),
                "phase_reference_rad": f"{float(phase_reference_rad):.9f}",
                "phase_reference_valid": int(phase_reference_valid),
                "phase_reference_age_s": f"{float(state.phase_reference_age_s):.9f}",
                "phase_reference_accepted_updates": int(state.phase_reference_accepted_updates),
                "phase_reference_ramp_scale": f"{float(phase_reference_ramp_scale):.9f}",
                "phase_setpoint_rad": f"{float(args.phase_setpoint_rad):.9f}",
                "phase_err_raw_rad": f"{phase_err_raw:.9f}",
                "phase_err_relative_raw_rad": f"{float(phase_err_relative_raw):.9f}",
                "phase_err_accepted_raw_rad": f"{float(phase_err_accepted_raw_rad):.9f}",
                "phase_accepted_raw_window_count": int(phase_accepted_raw_window_count),
                "phase_accepted_raw_enter_fraction": f"{float(phase_accepted_raw_enter_fraction):.9f}",
                "phase_accepted_raw_exit_fraction": f"{float(phase_accepted_raw_exit_fraction):.9f}",
                "phase_accepted_raw_enter_ok": int(phase_accepted_raw_enter_ok),
                "phase_accepted_raw_exit_ok": int(phase_accepted_raw_exit_ok),
                "phase_err_control_rad": f"{phase_err_control_rad:.9f}",
                "phase_err_filt_rad": f"{phase_err_filt:.9f}",
                "coherence": f"{control_meas.coherence:.9f}",
                "pullin_quality": f"{control_meas.pullin_quality:.9f}",
                "observer_coherence": f"{observer_meas.coherence:.9f}",
                "observer_pullin_quality": f"{observer_meas.pullin_quality:.9f}",
                "control_coherence": f"{control_meas.coherence:.9f}",
                "control_pullin_quality": f"{control_meas.pullin_quality:.9f}",
                "amp_rms": f"{control_meas.amp_rms:.9f}",
                "freq_err_intrablock_hz": f"{control_meas.freq_err_intrablock_hz:.9f}",
                "freq_err_update_hz": f"{control_meas.freq_err_update_hz:.9f}",
                "phase_fit_slope_hz": f"{control_meas.phase_fit_slope_hz:.9f}",
                "phase_fit_rms_rad": f"{control_meas.phase_fit_rms_rad:.9f}",
                "phase_fit_max_abs_rad": f"{control_meas.phase_fit_max_abs_rad:.9f}",
                "phase_fit_chunks_used": int(control_meas.phase_fit_chunks_used),
                "observer_freq_err_intrablock_hz": f"{observer_meas.freq_err_intrablock_hz:.9f}",
                "observer_freq_err_update_hz": f"{observer_meas.freq_err_update_hz:.9f}",
                "observer_phase_fit_slope_hz": f"{observer_meas.phase_fit_slope_hz:.9f}",
                "observer_phase_fit_rms_rad": f"{observer_meas.phase_fit_rms_rad:.9f}",
                "observer_phase_fit_max_abs_rad": f"{observer_meas.phase_fit_max_abs_rad:.9f}",
                "observer_phase_fit_chunks_used": int(observer_meas.phase_fit_chunks_used),
                "control_freq_err_intrablock_hz": f"{phase_freq_err_hz:.9f}",
                "control_freq_err_update_hz": f"{control_meas.freq_err_update_hz:.9f}",
                "control_phase_fit_slope_hz": f"{control_meas.phase_fit_slope_hz:.9f}",
                "control_phase_fit_rms_rad": f"{control_meas.phase_fit_rms_rad:.9f}",
                "control_phase_fit_max_abs_rad": f"{control_meas.phase_fit_max_abs_rad:.9f}",
                "control_phase_fit_chunks_used": int(control_meas.phase_fit_chunks_used),
                "observer_tau_err_ns": f"{observer_tau_err_ns:.9f}",
                "pullin_target_hz": f"{float(args.pullin_target_hz):.9f}",
                "pullin_error_hz": f"{pullin_err_raw_hz:.9f}",
                "pullin_error_lpf_hz": f"{pullin_err_hz:.9f}",
                "phase_measure_pred_rad": f"{float(phase_measure_pred_rad):.9f}",
                "phase_predict_rate_hz": f"{float(phase_predict_rate_hz):.9f}",
                "phase_measure_innovation_rad": f"{float(phase_measure_innovation_rad):.9f}",
                "phase_detector_quality_ok": int(phase_detector_ok),
                "phase_innovation_ok": int(phase_innovation_ok),
                "phase_innovation_softened": int(phase_innovation_softened),
                "phase_measurement_rejected": int(phase_measurement_rejected),
                "phase_state_predicted": int(phase_state_predicted),
                "slip_cycles": int(control_meas.slip_cycles),
                "dphi_wrapped_rad": f"{control_meas.dphi_wrapped_rad:.9f}",
                "dphi_used_rad": f"{control_meas.dphi_used_rad:.9f}",
                "dphi_expected_rad": f"{control_meas.dphi_expected_rad:.9f}",
                "u_hz_raw": f"{float(u_hz_raw):.9f}",
                "u_hz_limited": f"{float(u_hz_limited):.9f}",
                "sham_control_mode": str(sham_control_mode),
                "dac_code_cmd": int(state.dac_code_cmd),
                "dac_code_rb": "" if state.dac_code_rb is None else int(state.dac_code_rb),
                "dac_code_proposed": int(dac_code_proposed),
                "dac_code_apply_target": int(dac_code_apply_target),
                "dac_volts": f"{to_float(state.dac_volts):.9f}",
                "dac_step_req_codes": int(dac_step_req),
                "dac_step_apply_target_codes": int(dac_step_apply_target),
                "dac_step_applied_codes": int(dac_step_applied),
                "phase_int_rad_s": f"{float(state.phase_int_rad_s):.9f}",
                "phase_center_code": "" if state.phase_center_code is None else int(state.phase_center_code),
                "phase_center_trim_codes": f"{float(state.phase_center_trim_codes):.9f}",
                "phase_substage": str(phase_substage),
                "phase_freq_err_lpf_hz": f"{float(phase_freq_err_lpf_hz):.9f}",
                "phase_hold_rate_lpf_hz": f"{float(phase_hold_rate_lpf_hz):.9f}",
                "phase_hold_int_hz": f"{float(state.phase_hold_int_hz):.9f}",
                "phase_step_sigma_codes": f"{float(state.phase_step_sigma_codes):.9f}",
                "phase_target_float": f"{float(phase_target_float):.9f}",
                "phase_target_code": int(phase_target_code),
                "phase_fraction_cycles_per_sample": f"{float(phase_fraction_cycles_per_sample):.12f}",
                "phase_err_for_entry_ok": int(phase_err_for_entry_ok),
                "post_dac_ignore_active": int(ignore_control_update),
                "post_dac_ignore_updates_remaining": int(state.post_dac_ignore_updates),
                "queue_skipped_blocks_latest": int(skipped_blocks),
                "wall_block_age_s": f"{float(wall_block_age_s):.9f}",
                "timing_queue_wait_s": f"{float(t_wait_s):.9f}",
                "timing_measure_s": f"{float(t_measure_s):.9f}",
                "timing_control_s": f"{float(t_control_s):.9f}",
                "timing_dac_s": f"{float(t_dac_s):.9f}",
                "timing_total_s": f"{float(t_total_s):.9f}",
                "loop_period_s_nom": f"{nominal_loop_period_s:.9f}",
                "nsamp": int(nsamp),
                "sample_rate_hz": f"{float(args.sample_rate):.3f}",
                "kv_hz_per_code": f"{float(args.kv_hz_per_code):.9f}",
                "notes": str(args.notes),
            }

            t_log_start = time.perf_counter()
            logger.write(row)
            t_log_s = float(time.perf_counter() - t_log_start)
            runtime_stats.log_enqueue_total_s += t_log_s
            runtime_stats.log_enqueue_max_s = max(runtime_stats.log_enqueue_max_s, t_log_s)
            runtime_stats.log_enqueue_events += 1
            rows.append(row)

            update_idx += 1
            if args.max_updates is not None and update_idx >= int(args.max_updates):
                break
            if args.run_seconds is not None and control_meas.elapsed_s > float(args.run_seconds):
                break

            if args.run_seconds is not None and (
                float(time.time()) - float(wall_start_s)
            ) >= float(args.run_seconds):
                break
            if rows and args.run_seconds is not None:
                if float(rows[-1]["elapsed_s"]) > float(args.run_seconds):
                    break

        if tone_offset_hz is None and bootstrap_blocks:
            stacked = np.stack(
                [np.asarray(b.iq_centered, dtype=np.complex64).reshape(-1) for b in bootstrap_blocks],
                axis=0,
            )
            tone_offset_hz = estimate_tone_offset_hz(
                stacked,
                sample_rate_hz=float(args.sample_rate),
                dc_exclusion_hz=float(args.dc_exclusion_hz),
            )

        if tone_offset_hz is None:
            tone_offset_hz = 0.0

        summary = compute_summary(
            rows=rows,
            run_mode=str(args.mode),
            nsamp=int(nsamp),
            sample_rate_hz=float(args.sample_rate),
            center_freq_hz=float(args.center_freq),
            tone_offset_hz=float(tone_offset_hz),
            drift_window_s=float(args.drift_window_s),
        )
        summary["csv_path"] = str(csv_path)
        summary["stream_nsamp"] = int(stream_nsamp)
        summary["stage_switches"] = int(state.stage_switches)
        summary["pullin_updates_applied"] = int(state.pullin_updates_applied)
        summary["phase_updates_applied"] = int(state.phase_updates_applied)
        summary["dac_min_code"] = int(code_min)
        summary["dac_max_code"] = int(code_max)
        summary["dac_start_code"] = int(dac_start)
        summary["producer_blocks_dropped_queue_full"] = int(runtime_stats.producer_blocks_dropped_queue_full)
        summary["consumer_blocks_skipped_latest_only_runtime"] = int(runtime_stats.consumer_blocks_skipped_latest_only)
        summary["bootstrap_blocks_discarded"] = int(runtime_stats.bootstrap_blocks_discarded)
        summary["shared_demod_updates"] = int(runtime_stats.shared_demod_updates)
        summary["logger_mode"] = str(logger.logger_mode)
        summary["sham_control_mode"] = str(sham_control_mode)
        summary["mean_log_enqueue_s"] = (
            float(runtime_stats.log_enqueue_total_s / runtime_stats.log_enqueue_events)
            if runtime_stats.log_enqueue_events > 0
            else float("nan")
        )
        summary["max_log_enqueue_s"] = float(runtime_stats.log_enqueue_max_s)
        wall_elapsed_s = max(0.0, float(time.time()) - float(wall_start_s))
        summary["wall_elapsed_s"] = float(wall_elapsed_s)
        summary["sample_elapsed_per_wall_s"] = (
            float(summary["elapsed_s"]) / float(wall_elapsed_s)
            if wall_elapsed_s > 0.0 and np.isfinite(float(summary["elapsed_s"]))
            else float("nan")
        )

        return summary

    finally:
        if producer_task is not None:
            if not producer_task.done():
                producer_task.cancel()
            try:
                await asyncio.wait_for(producer_task, timeout=2.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                pass
            except BaseException:
                pass
        logger.close()
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
        if clock_obj is not None:
            try:
                clock_obj.close()
            except Exception:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hybrid live DPLL with staged pull-in and notebook-style phase-lock control."
    )

    p.add_argument("--mode", choices=["pid", "monitor"], default="pid")
    p.add_argument("--notes", type=str, default="")

    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--center-freq", type=float, default=100e6)
    p.add_argument("--sample-rate", type=float, default=2.4e6)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--settle-seconds", type=float, default=0.25)

    p.add_argument("--nsamp", type=int, default=8192)
    p.add_argument(
        "--stream-nsamp",
        type=int,
        default=None,
        help="Optional SDR capture block size. If set larger than nsamp, each captured block is split into nsamp-sized control updates.",
    )
    p.add_argument(
        "--nsamp-values",
        type=str,
        default=None,
        help="Comma-separated nsamp sweep values (e.g. 16384,32768,65536).",
    )
    p.add_argument("--blocks-per-update", type=int, default=1)
    p.add_argument("--async-queue-blocks", type=int, default=None)
    p.add_argument(
        "--queue-get-timeout-s",
        type=float,
        default=1.0,
        help="Timeout while waiting for the next streamed block so wall-time exit still works.",
    )
    p.add_argument(
        "--log-background-process",
        dest="log_background_process",
        action="store_true",
        default=True,
        help="Write CSV rows through a background process instead of blocking the control loop on file I/O.",
    )
    p.add_argument(
        "--no-log-background-process",
        dest="log_background_process",
        action="store_false",
        help="Write CSV rows inline in the control loop.",
    )
    p.add_argument(
        "--log-flush-rows",
        type=int,
        default=128,
        help="Flush the background CSV writer after this many rows.",
    )
    p.add_argument(
        "--log-flush-interval-s",
        type=float,
        default=0.5,
        help="Flush the background CSV writer after this many wall-clock seconds.",
    )

    p.add_argument("--run-seconds", type=float, default=120.0)
    p.add_argument("--max-updates", type=int, default=None)

    p.add_argument("--target-offset-hz", type=float, default=None)
    p.add_argument(
        "--observer-fixed-offset-hz",
        type=float,
        default=None,
        help="Fixed observer-reference tone offset (Hz). If omitted, uses initial tone offset.",
    )
    p.add_argument(
        "--control-measurement-source",
        choices=["adaptive", "observer"],
        default="observer",
        help="Which measurement stream drives pull-in and phase control.",
    )
    p.add_argument(
        "--single-stage-phase-only",
        action="store_true",
        help="Start in phase_lock hold mode immediately and never return to pull_in.",
    )
    p.add_argument("--tone-estimate-blocks", type=int, default=8)
    p.add_argument("--dc-exclusion-hz", type=float, default=5e3)
    p.add_argument(
        "--phase-detector-chunks",
        type=int,
        default=1,
        help="Split each demodulated block into this many chunks for chunk-fit phase estimation (1 disables chunking).",
    )
    p.add_argument(
        "--phase-fit-rms-max-rad",
        type=float,
        default=0.8,
        help="Maximum chunk-fit RMS phase residual allowed before phase mode treats the measurement as unreliable. <=0 disables the RMS gate.",
    )
    p.add_argument(
        "--phase-fit-max-abs-rad",
        type=float,
        default=1.6,
        help="Maximum absolute chunk-fit residual allowed before phase mode treats the measurement as unreliable. <=0 disables the max-residual gate.",
    )
    p.add_argument(
        "--phase-innovation-max-rad",
        type=float,
        default=1.2,
        help="Soft innovation limit in radians. Moderate overshoots are clipped to this band before updating phase mode. <=0 disables innovation gating.",
    )
    p.add_argument(
        "--phase-innovation-hard-max-rad",
        type=float,
        default=0.0,
        help="Hard innovation reject limit in radians. If <=0, phase mode uses 2x --phase-innovation-max-rad.",
    )
    p.add_argument(
        "--phase-innovation-soft-blend",
        type=float,
        default=1.0,
        help="Blend weight for softened innovation updates, from predicted phase (0) toward the clipped innovation measurement (1).",
    )
    p.add_argument(
        "--phase-innovation-soft-fit-rms-max-rad",
        type=float,
        default=0.0,
        help="Only allow soft innovation updates when the chunk-fit RMS is at or below this threshold. <=0 disables the extra soft-update fit gate.",
    )
    p.add_argument(
        "--phase-predict-through-holds",
        action="store_true",
        help="When phase measurements are ignored or rejected, advance the filtered phase state using the predicted phase instead of freezing it.",
    )
    p.add_argument(
        "--tone-track-alpha",
        type=float,
        default=0.0,
        help="Adaptive tone-reference update gain (0 disables tracking).",
    )
    p.add_argument(
        "--tone-track-min-coh",
        type=float,
        default=0.0,
        help="Minimum coherence required to apply tone-reference tracking.",
    )
    p.add_argument(
        "--tone-track-max-step-hz",
        type=float,
        default=2000.0,
        help="Per-update clamp for adaptive tone-reference step.",
    )
    p.add_argument(
        "--phase-observer-recenter-on-entry",
        action="store_true",
        help="When phase mode engages, recenter the observer reference onto the tracked tone estimate.",
    )
    p.add_argument(
        "--phase-observer-recenter-max-step-hz",
        type=float,
        default=30.0,
        help="Maximum observer-reference jump applied on the update where phase mode first engages.",
    )
    p.add_argument(
        "--phase-observer-track-alpha",
        type=float,
        default=0.0,
        help="Low-pass gain for observer-reference tracking while phase mode is active (0 disables tracking).",
    )
    p.add_argument(
        "--phase-observer-track-min-quality",
        type=float,
        default=0.90,
        help="Minimum adaptive pull-in quality required before phase-mode observer tracking is allowed.",
    )
    p.add_argument(
        "--phase-observer-track-max-step-hz",
        type=float,
        default=5.0,
        help="Maximum observer-reference adjustment per update while phase mode is active.",
    )

    p.add_argument(
        "--phase-setpoint-rad",
        type=float,
        default=0.0,
        help="Desired phase offset in radians. With --phase-reference-mode=first_valid, this is relative to the captured phase reference.",
    )
    p.add_argument(
        "--phase-reference-mode",
        choices=["first_valid", "absolute"],
        default="absolute",
        help="Phase reference origin: capture the first valid phase sample after phase entry, or use absolute zero as the reference.",
    )
    p.add_argument("--phase-filter-alpha", type=float, default=0.85)
    p.add_argument(
        "--phase-enter-condition",
        choices=["full", "pullin_only", "residual_and_quality"],
        default="full",
        help="Phase-lock handoff rule: use the original full gate set, only the filtered pull-in residual, or residual plus measurement quality without an absolute phase gate.",
    )
    p.add_argument(
        "--phase-exit-condition",
        choices=["full", "residual_only", "residual_and_phase"],
        default="full",
        help="Phase-lock exit rule: use the original full gate set or only large filtered pull-in residual.",
    )
    p.add_argument(
        "--phase-exit-min-err-rad",
        type=float,
        default=1.0,
        help="For residual_and_phase exit mode, require at least this much filtered phase error along with the residual threshold. <=0 disables the phase-error part.",
    )
    p.add_argument(
        "--phase-min-hold-updates",
        type=int,
        default=0,
        help="Minimum number of control updates to remain in phase_lock before exit checks can pull the loop back to pull_in.",
    )
    p.add_argument(
        "--phase-min-hold-s",
        type=float,
        default=None,
        help="Minimum wall-clock time to remain in phase_lock before exit checks can pull the loop back to pull_in. Overrides --phase-min-hold-updates when set.",
    )
    p.add_argument("--coh-min", type=float, default=0.02)
    p.add_argument(
        "--pullin-quality-min",
        type=float,
        default=0.10,
        help="Minimum pull-in quality required to trust intrablock frequency measurements.",
    )
    p.add_argument(
        "--pullin-filter-alpha",
        type=float,
        default=0.80,
        help="Low-pass filter alpha for the pull-in frequency error used by the staged controller.",
    )

    p.add_argument("--pullin-target-hz", type=float, default=0.0)
    p.add_argument("--pullin-switch-hz", type=float, default=100.0)
    p.add_argument("--pullin-exit-hz", type=float, default=300.0)
    p.add_argument(
        "--pullin-enter-dwell-updates",
        type=int,
        default=3,
        help="Consecutive in-threshold updates required before entering phase_lock.",
    )
    p.add_argument(
        "--pullin-enter-dwell-s",
        type=float,
        default=None,
        help="Wall-clock dwell required before entering phase_lock. Overrides --pullin-enter-dwell-updates when set.",
    )
    p.add_argument(
        "--pullin-exit-dwell-updates",
        type=int,
        default=2,
        help="Consecutive out-of-threshold updates required before exiting phase_lock.",
    )
    p.add_argument(
        "--pullin-exit-dwell-s",
        type=float,
        default=None,
        help="Wall-clock dwell required before exiting phase_lock. Overrides --pullin-exit-dwell-updates when set.",
    )
    p.add_argument(
        "--ignore-blocks-after-dac-write",
        type=int,
        default=0,
        help="After each DAC write, ignore this many subsequent measurement blocks for filtering and control updates.",
    )
    p.add_argument("--Kf-pullin-hz-per-hz", type=float, default=0.07)
    p.add_argument(
        "--phase-enter-max-err-rad",
        type=float,
        default=0.5,
        help="Maximum filtered phase error allowed when entering phase_lock. <=0 disables the gate.",
    )
    p.add_argument(
        "--phase-frac-gain-codes",
        type=float,
        default=409500.0,
        help="Notebook-style phase gain from phase fraction to DAC codes while in phase_lock acquire mode.",
    )
    p.add_argument(
        "--phase-hold-frac-gain-codes",
        type=float,
        default=102375.0,
        help="Legacy notebook-style hold gain. Retained for compatibility; PI/D hold mode does not use it.",
    )
    p.add_argument(
        "--phase-center-integral-gain-codes-per-rad-s",
        type=float,
        default=12.0,
        help="Legacy center-trim integral gain. Retained for compatibility; PI/D hold mode does not use it.",
    )
    p.add_argument(
        "--phase-center-freq-gain-codes-per-hz-s",
        type=float,
        default=0.35,
        help="Residual-frequency trim gain used in phase_lock acquire mode.",
    )
    p.add_argument(
        "--phase-hold-center-freq-gain-codes-per-hz-s",
        type=float,
        default=0.08,
        help="Residual-frequency trim gain used in phase_lock hold mode.",
    )
    p.add_argument(
        "--phase-center-freq-filter-alpha",
        type=float,
        default=0.80,
        help="Low-pass filter alpha for the residual frequency estimate used to move the phase-mode center.",
    )
    p.add_argument(
        "--phase-center-freq-max-hz",
        type=float,
        default=25.0,
        help="Absolute clamp applied to the residual frequency estimate before it can move the phase-mode center.",
    )
    p.add_argument(
        "--phase-center-integral-limit-codes",
        type=float,
        default=64.0,
        help="Clamp for the integral trim contribution to the notebook-style phase center.",
    )
    p.add_argument(
        "--phase-hold-enter-err-rad",
        type=float,
        default=0.20,
        help="Filtered phase-error threshold required before phase_lock transitions from acquire to hold.",
    )
    p.add_argument(
        "--phase-hold-exit-err-rad",
        type=float,
        default=0.35,
        help="Filtered phase-error threshold that kicks phase_lock from hold back to acquire.",
    )
    p.add_argument(
        "--phase-hold-enter-dwell-updates",
        type=int,
        default=3,
        help="Consecutive in-threshold updates required before phase_lock hold mode engages.",
    )
    p.add_argument(
        "--phase-hold-exit-dwell-updates",
        type=int,
        default=2,
        help="Consecutive out-of-threshold updates required before phase_lock hold mode returns to acquire.",
    )
    p.add_argument(
        "--phase-accepted-raw-enter-window-updates",
        type=int,
        default=12,
        help="Number of accepted-raw phase samples used for hold entry. <=0 disables the accepted-raw window gate.",
    )
    p.add_argument(
        "--phase-accepted-raw-enter-min-frac",
        type=float,
        default=0.75,
        help="Minimum in-band fraction required over the accepted-raw entry window to enter hold.",
    )
    p.add_argument(
        "--phase-accepted-raw-exit-window-updates",
        type=int,
        default=12,
        help="Number of accepted-raw phase samples used for hold continue/exit.",
    )
    p.add_argument(
        "--phase-accepted-raw-exit-min-frac",
        type=float,
        default=0.50,
        help="Minimum in-band fraction required over the accepted-raw exit window to remain in hold.",
    )
    p.add_argument(
        "--phase-reference-ramp-accepted-updates",
        type=int,
        default=24,
        help="Ramp phase-hold authority from --phase-reference-ramp-min-scale to full over this many accepted post-reference samples. <=0 disables ramping.",
    )
    p.add_argument(
        "--phase-reference-ramp-min-scale",
        type=float,
        default=0.25,
        help="Starting fraction of hold authority immediately after phase reference capture.",
    )
    p.add_argument(
        "--phase-hold-integral-max-err-rad",
        type=float,
        default=1.5,
        help="Only integrate phase error in hold mode while the raw phase error stays within this band.",
    )
    p.add_argument(
        "--phase-hold-detector-limit-rad",
        type=float,
        default=1.2,
        help="Clip the raw phase error to this band before the hold-mode sine phase detector.",
    )
    p.add_argument(
        "--phase-hold-deadband-rad",
        type=float,
        default=0.08,
        help="Zero-phase deadband applied before the hold-mode phase detector.",
    )
    p.add_argument(
        "--phase-hold-kp-hz-per-rad",
        type=float,
        default=8.0,
        help="Proportional hold-mode gain from phase-detector output to frequency command.",
    )
    p.add_argument(
        "--phase-hold-ki-hz-per-rad-s",
        type=float,
        default=3.0,
        help="Integral hold-mode gain from phase-detector output to frequency command.",
    )
    p.add_argument(
        "--phase-hold-kd-hz-per-hz",
        type=float,
        default=0.0,
        help="Derivative/damping hold-mode gain applied to residual phase-rate estimate.",
    )
    p.add_argument(
        "--phase-hold-rate-filter-alpha",
        type=float,
        default=0.75,
        help="Low-pass filter alpha for the hold-mode phase-rate estimate.",
    )
    p.add_argument(
        "--phase-hold-rate-max-hz",
        type=float,
        default=12.0,
        help="Clamp applied to the raw hold-mode phase-rate estimate before filtering.",
    )
    p.add_argument(
        "--phase-hold-integral-limit-hz",
        type=float,
        default=40.0,
        help="Clamp for the hold-mode integral state, in equivalent frequency-command units.",
    )
    p.add_argument(
        "--phase-hold-max-u-hz",
        type=float,
        default=20.0,
        help="Clamp for the hold-mode frequency command before conversion to DAC offset.",
    )
    p.add_argument(
        "--max-phase-dac-step-per-update",
        type=int,
        default=2,
        help="Per-update DAC clamp used while in phase_lock acquire mode.",
    )
    p.add_argument(
        "--max-phase-hold-dac-step-per-update",
        type=int,
        default=1,
        help="Per-update DAC clamp used while in phase_lock hold mode.",
    )

    p.add_argument("--Kp-hz-per-rad", type=float, default=40.0)
    p.add_argument("--Ki-hz-per-rad-s", type=float, default=4.0)
    p.add_argument("--Kd-hz-s-per-rad", type=float, default=0.0)
    p.add_argument("--integral-limit-hz", type=float, default=200.0)
    p.add_argument("--max-u-hz", type=float, default=2000.0)

    p.add_argument("--control-sign", type=float, default=1.0)
    p.add_argument("--kv-hz-per-code", type=float, default=4.243)
    p.add_argument("--max-dac-step-per-update", type=int, default=100)
    p.add_argument("--dac-min-pct", type=float, default=10.0)
    p.add_argument("--dac-max-pct", type=float, default=90.0)
    p.add_argument("--dac-start", type=int, default=None)
    p.add_argument("--vdd-dac", type=float, default=VDD_DAC_DEFAULT)

    p.add_argument("--apr-ppm", type=int, default=120)
    p.add_argument("--skip-clock-init", action="store_true")
    p.add_argument(
        "--apply-start-dac",
        dest="apply_start_dac",
        action="store_true",
        default=True,
        help="Apply --dac-start at run start (PID mode).",
    )
    p.add_argument(
        "--no-apply-start-dac",
        dest="apply_start_dac",
        action="store_false",
        help="Do not write a start DAC value at run start (PID mode).",
    )
    p.add_argument("--fine-settle-s", type=float, default=0.0)
    p.add_argument(
        "--sham-control-mode",
        choices=["off", "random_sign"],
        default="off",
        help="Optional causality-break test: keep the detector/controller live but replace the applied DAC step with a sham command.",
    )
    p.add_argument(
        "--sham-rng-seed",
        type=int,
        default=0,
        help="Seed used for sham-control randomization.",
    )

    p.add_argument("--slip-trigger-rad", type=float, default=math.pi)
    p.add_argument("--slip-max-cycles-per-update", type=int, default=8)

    p.add_argument("--drift-window-s", type=float, default=30.0)

    base_dir = Path(__file__).resolve().parent
    p.add_argument(
        "--csv-out",
        type=Path,
        default=base_dir / "notebook_based_hybrid_pullin_notebook_phase_log.csv",
        help="CSV output path. Sweep mode appends _nsampXXXX.",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=base_dir / "notebook_based_hybrid_pullin_notebook_phase_summary.json",
        help="Summary JSON path. Sweep mode writes a JSON list.",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional summary CSV path for one run or sweep.",
    )

    p.add_argument("--quiet", action="store_true")
    return p


def write_summary_csv(summaries: list[dict[str, Any]], path: Path) -> None:
    ensure_parent_dir(path)
    if not summaries:
        return

    keys = sorted({k for s in summaries for k in s.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)


def print_run_summary(summary: dict[str, Any]) -> None:
    print("\nRun summary:")
    print(f"  mode: {summary.get('mode')}")
    print(f"  nsamp: {summary.get('nsamp')}")
    if summary.get("stream_nsamp") is not None:
        print(f"  stream_nsamp: {summary.get('stream_nsamp')}")
    print(f"  sample_rate_hz: {summary.get('sample_rate_hz')}")
    print(f"  loop_period_nom_s: {summary.get('loop_period_nom_s')}")
    print(f"  tone_offset_hz: {summary.get('tone_offset_hz')}")
    print(f"  updates: {summary.get('updates')}")
    print(f"  median_coherence: {summary.get('median_coherence')}")
    print(f"  phase_drift_slope_start_hz: {summary.get('phase_drift_slope_start_hz')}")
    print(f"  phase_drift_slope_end_hz: {summary.get('phase_drift_slope_end_hz')}")
    print(f"  phase_drift_reduction_hz_abs: {summary.get('phase_drift_reduction_hz_abs')}")
    print(f"  phase_drift_reduction_pct: {summary.get('phase_drift_reduction_pct')}")
    print(f"  observer_timing_drift_ns_per_s: {summary.get('primary_timing_drift_ns_per_s')}")
    print(f"  observer_timing_drift_samples_per_s: {summary.get('primary_timing_drift_samples_per_s')}")
    print(
        "  observer_timing_drift_start/end_ns_per_s: "
        f"{summary.get('observer_timing_drift_slope_ns_per_s_start')} -> "
        f"{summary.get('observer_timing_drift_slope_ns_per_s_end')}"
    )
    print(f"  observer_timing_drift_reduction_pct: {summary.get('observer_timing_drift_reduction_pct')}")
    print(f"  quality_fraction: {summary.get('quality_fraction')}")
    print(f"  csv_path: {summary.get('csv_path')}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        nsamp_values = parse_nsamp_values(args.nsamp_values, int(args.nsamp))
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
        suffix = ""
        if len(nsamp_values) > 1:
            suffix = f"_nsamp{int(nsamp)}"

        csv_path = path_with_suffix(Path(args.csv_out), suffix)

        if not args.quiet:
            loop_period = float(nsamp) / float(args.sample_rate)
            stream_nsamp = int(args.stream_nsamp) if args.stream_nsamp is not None else int(nsamp)
            print(
                f"Starting run {idx + 1}/{len(nsamp_values)} | mode={args.mode} | nsamp={nsamp} | "
                f"stream_nsamp={stream_nsamp} | loop_period_nom={loop_period:.6f}s | csv={csv_path}",
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

    ensure_parent_dir(Path(args.summary_out))
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
