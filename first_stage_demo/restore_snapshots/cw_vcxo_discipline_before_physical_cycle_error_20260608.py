#!/usr/bin/env python3
"""
First-stage CW tone discipline experiment for an SDR reference VCXO.

Hardware assumptions
- A signal generator provides a single CW tone into the SDR RF input by cable.
- The SDR reference is steered through either:
  1. MCP4725 DAC + VCXO path via ``clock_vcxo_helper.py`` (preferred), or
  2. a Raspberry Pi PWM output followed by an analog low-pass filter.
- The Raspberry Pi runs this script and has access to the SDR IQ stream.

Processing model
- The SDR is tuned near the CW tone.
- Each IQ block is mixed by the expected tone baseband frequency.
- The mixed samples are averaged into one complex phasor per block.
- Residual frequency error is estimated from block-to-block phase change:
      dphi_k = angle(z_k * conj(z_{k-1}))
      df_hat = dphi_k / (2*pi*T_block)
- Closed-loop runs use a single physical PLL controller: a phase detector,
  windowed frequency estimate, phase discipline state machine, and calibrated
  VCXO actuator model.

Recommended workflow
1. Run ``open_loop`` first and confirm the tone is strong and phase drifts slowly.
2. Run ``vcxo_test`` to measure monotonic frequency shift versus control command.
3. Use the measured sensitivity as ``--vcxo-hz-per-command`` and tune the PLL
   frequency gains with a nonzero averaging window before enabling phase hold.

Examples
- Open-loop:
    python3 first_stage_demo/cw_vcxo_discipline.py \\
        --mode open_loop --sample-rate 240000 --center-freq 100000000 \\
        --tone-rf 100010000 --block-size 16384 --duration-s 60

- Closed-loop with MCP4725:
    python3 first_stage_demo/cw_vcxo_discipline.py \\
        --mode closed_loop --sample-rate 240000 --center-freq 100000000 \\
        --tone-rf 100010000 --block-size 16384 --duration-s 180 \\
        --use-mcp4725 --dac-start 2048 --vcxo-hz-per-command -4.55 \\
        --pll-frequency-proportional-gain 0.01 --pll-frequency-integral-gain 0.0 \\
        --control-update-interval-s 0.002 --control-average-interval-s 0.05

- VCXO sweep:
    python3 first_stage_demo/cw_vcxo_discipline.py \\
        --mode vcxo_test --sample-rate 240000 --center-freq 100000000 \\
        --tone-rf 100010000 --block-size 16384 --use-mcp4725 \\
        --vcxo-test-start 1700 --vcxo-test-stop 2400 --vcxo-test-step 100

What success should look like
- Open-loop: non-zero mean frequency offset and steadily drifting unwrapped phase.
- VCXO test: inferred frequency offset changes monotonically with command.
- Closed-loop: mean residual frequency offset moves toward zero and long-term phase
  drift is reduced relative to open-loop.

Notes
- MCP4725 control supports first-order dithering between adjacent DAC codes so the
  average control value can realize fractional-code commands from the slow loop.
- CSV logs include explicit pull-in/fine-lock stage labels and handoff events so
  startup acquisition can be reviewed after each run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import queue
import signal
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def clamp_int(x: float, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(round(float(x)))))


def wrap_to_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def timestamp_for_filename() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def float_or_nan(value: Optional[float]) -> float:
    if value is None:
        return float("nan")
    return float(value)


def format_float(value: float, width: int = 8, precision: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return f"{'nan':>{width}}"
    return f"{value:>{width}.{precision}f}"


def coherence_to_snr_db(coherence: float) -> float:
    coherence = clamp(float(coherence), 0.0, 0.999999999999)
    sig_frac = coherence * coherence
    noise_frac = max(1.0e-12, 1.0 - sig_frac)
    snr_lin = sig_frac / noise_frac
    return 10.0 * math.log10(max(snr_lin, 1.0e-12))


def bytes_to_iq_block(raw_bytes: bytes, nsamples: int) -> np.ndarray:
    expected = 2 * int(nsamples)
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    if raw.size != expected:
        raise RuntimeError(
            f"Expected {expected} SDR bytes for one IQ block, received {raw.size}."
        )
    raw = raw.astype(np.float32, copy=False).reshape(int(nsamples), 2)
    raw -= 128.0
    iq = (raw[:, 0] + 1j * raw[:, 1]) / 128.0
    return iq.astype(np.complex64, copy=False)


@dataclass
class ToneReference:
    tone_bb_hz: float
    tone_rf_hz: Optional[float]
    source: str


@dataclass
class BlockMeasurement:
    phase_rad: float
    unwrapped_phase_rad: float
    dphi_rad: float
    f_off_hz: float
    amp: float
    coherence: float
    snr_db: float
    power: float
    z_re: float
    z_im: float
    z_abs: float
    valid_phase_step: bool


@dataclass
class ControlUpdate:
    applied_command: float
    requested_command: float
    p_term: float
    i_term: float
    error_hz: float
    base_delta: float = 0.0
    base_update_allowed: bool = False


@dataclass
class ControlMeasurementSummary:
    t_rel_s: float
    phase_rad: float
    f_off_hz: float
    amp: float
    coherence: float
    fit_r2: float
    valid_count: int
    window_span_s: float


@dataclass
class ClockEstimate:
    phase_rad: float
    unwrapped_phase_rad: float
    frequency_hz: float
    drift_hz_per_s: float
    phase_residual_rad: float
    confidence: float
    valid: bool
    cycle_slip: bool


@dataclass
class PhaseDisciplineCommand:
    stage: str
    switch_event: str
    phase_mode: str
    target_phase_rad: float
    phase_error_rad: float
    phase_error_raw_rad: float
    phase_error_filtered_rad: float
    phase_error_update_allowed: bool
    phase_error_cycle_rad: float
    phase_error_predicted_rad: float
    phase_error_branch_index: float
    phase_error_branch_residual_rad: float
    phase_cycle_update_allowed: bool
    phase_cycle_slip_detected: bool
    phase_cycle_slip_count: int
    target_f_off_hz: float
    raw_target_f_off_hz: float
    frequency_error_hz: float
    phase_gain_hz_per_rad: float
    max_target_f_off_hz: float
    phase_active: bool
    just_armed: bool


@dataclass
class ControlStageState:
    stage: str
    switch_event: str
    enter_counter: int
    exit_counter: int
    stage_switches: int


@dataclass
class RecaptureState:
    active: bool
    just_started: bool
    just_completed: bool
    reason: str
    bad_block_counter: int
    ambiguity_counter: int
    capture_block_count: int
    capture_count: int
    search_center_hz: float
    estimated_tone_bb_hz: float
    estimated_error_hz: float
    estimated_peak_power_db: float
    estimated_peak_prominence_db: float
    estimate_valid: bool


@dataclass
class SweepStepSummary:
    step_index: int
    command: float
    mean_f_off_hz: float
    std_f_off_hz: float
    mean_amp: float
    mean_coherence: float
    valid_count: int


@dataclass
class PrecisionSequenceStep:
    command: float
    role: str
    pair_index: int


@dataclass
class PrecisionDwellResult:
    step_index: int
    command: float
    role: str
    pair_index: int
    wall_mid_rel_s: float
    sample_mid_rel_s: float
    coarse_bb_hz: float
    refined_bb_hz: float
    fine_residual_hz: float
    coarse_peak_power_db: float
    mean_amp: float
    mean_coherence: float
    fit_r2: float
    valid_block_count: int
    dwell_block_count: int
    valid: bool


@dataclass
class StreamSentinel:
    error: Optional[BaseException] = None


@dataclass
class CoarseToneEstimate:
    tone_hz: float
    peak_power_db: float
    peak_prominence_db: float
    second_peak_power_db: float
    noise_floor_db: float
    valid: bool


class GracefulStop:
    def __init__(self) -> None:
        self.stop_requested = False
        self._old_handlers: Dict[int, Any] = {}

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
            except Exception:
                pass

    def restore(self) -> None:
        for sig, handler in self._old_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass

    def _handle(self, signum, frame) -> None:
        self.stop_requested = True


class LoopProfiler:
    def __init__(self, enabled: bool, block_dt_s: float) -> None:
        self.enabled = bool(enabled)
        self.block_dt_s = max(float(block_dt_s), 1.0e-12)
        self._segments: Dict[str, List[float]] = {
            "read": [],
            "detect": [],
            "control": [],
            "actuator": [],
            "log": [],
            "total": [],
        }

    def observe(
        self,
        read_s: float,
        detect_s: float,
        control_s: float,
        actuator_s: float,
        log_s: float,
        total_s: float,
    ) -> None:
        if not self.enabled:
            return
        self._segments["read"].append(float(read_s))
        self._segments["detect"].append(float(detect_s))
        self._segments["control"].append(float(control_s))
        self._segments["actuator"].append(float(actuator_s))
        self._segments["log"].append(float(log_s))
        self._segments["total"].append(float(total_s))

    def report(self, label: str = "Loop timing") -> None:
        if not self.enabled or not self._segments["total"]:
            return
        total = np.asarray(self._segments["total"], dtype=np.float64)
        read = np.asarray(self._segments["read"], dtype=np.float64)
        compute = np.maximum(0.0, total - read)
        overruns = int(np.count_nonzero(total > self.block_dt_s))
        compute_overruns = int(np.count_nonzero(compute > self.block_dt_s))
        print(
            f"{label}: blocks={total.size}, block_dt={self.block_dt_s * 1.0e3:.3f} ms, "
            f"wall_overruns={overruns} ({100.0 * overruns / max(1, total.size):.2f}%), "
            f"compute_overruns={compute_overruns} ({100.0 * compute_overruns / max(1, total.size):.2f}%)"
        )
        for name in ("read", "detect", "control", "actuator", "log", "total"):
            values = np.asarray(self._segments[name], dtype=np.float64)
            print(
                f"  {name:8s} mean={np.mean(values) * 1.0e3:8.3f} ms "
                f"p95={np.percentile(values, 95.0) * 1.0e3:8.3f} ms "
                f"max={np.max(values) * 1.0e3:8.3f} ms"
            )
        print(
            f"  {'compute':8s} mean={np.mean(compute) * 1.0e3:8.3f} ms "
            f"p95={np.percentile(compute, 95.0) * 1.0e3:8.3f} ms "
            f"max={np.max(compute) * 1.0e3:8.3f} ms"
        )


class ToneTracker:
    def __init__(self, sample_rate_hz: float, block_size: int, tone_bb_hz: float) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.block_size = int(block_size)
        self.tone_bb_hz = float(tone_bb_hz)
        self.block_dt_s = self.block_size / self.sample_rate_hz
        self._phase_inc_rad = 2.0 * math.pi * self.tone_bb_hz / self.sample_rate_hz
        self._dc_subtract = abs(self.tone_bb_hz) >= (2.0 / max(self.block_dt_s, 1.0e-12))
        self._sample_offsets = np.arange(self.block_size, dtype=np.float64)
        self._base_rotator = np.exp(-1j * self._phase_inc_rad * self._sample_offsets)
        self._base_rotator_sum = np.sum(self._base_rotator, dtype=np.complex128)
        self._block_phase_rad = 0.0
        self._block_phase_step_rad = math.remainder(
            -self._phase_inc_rad * self.block_size,
            2.0 * math.pi,
        )
        self._prev_z: Optional[complex] = None
        self._unwrapped_phase_rad = 0.0

    def process_block(self, iq: np.ndarray) -> BlockMeasurement:
        if iq.shape[0] != self.block_size:
            raise ValueError(
                f"Expected block size {self.block_size}, got {iq.shape[0]} samples."
            )

        iq_values = iq.astype(np.complex128, copy=False)
        mean_iq = 0.0j
        if self._dc_subtract:
            mean_iq = complex(np.mean(iq_values, dtype=np.complex128))

        block_rotator = complex(
            math.cos(self._block_phase_rad),
            math.sin(self._block_phase_rad),
        )
        z = np.dot(iq_values, self._base_rotator)
        if self._dc_subtract:
            z -= mean_iq * self._base_rotator_sum
        z *= block_rotator
        self._block_phase_rad = math.remainder(
            self._block_phase_rad + self._block_phase_step_rad,
            2.0 * math.pi,
        )

        amp = float(abs(z) / max(1, self.block_size))
        power = float(np.vdot(iq_values, iq_values).real / max(1, self.block_size))
        if self._dc_subtract:
            power = max(0.0, power - float(abs(mean_iq) ** 2))
        coherence = clamp(amp / max(math.sqrt(power), 1.0e-12), 0.0, 1.0)
        snr_db = coherence_to_snr_db(coherence)
        phase_rad = float(np.angle(z))

        if self._prev_z is None:
            dphi_rad = float("nan")
            f_off_hz = float("nan")
            self._unwrapped_phase_rad = phase_rad
            valid_phase_step = False
        else:
            dphi_rad = float(np.angle(z * np.conj(self._prev_z)))
            f_off_hz = dphi_rad / (2.0 * math.pi * self.block_dt_s)
            self._unwrapped_phase_rad += dphi_rad
            valid_phase_step = True

        self._prev_z = z
        return BlockMeasurement(
            phase_rad=phase_rad,
            unwrapped_phase_rad=float(self._unwrapped_phase_rad),
            dphi_rad=dphi_rad,
            f_off_hz=f_off_hz,
            amp=amp,
            coherence=coherence,
            snr_db=snr_db,
            power=power,
            z_re=float(np.real(z)),
            z_im=float(np.imag(z)),
            z_abs=amp,
            valid_phase_step=valid_phase_step,
        )


class FrequencyRecaptureManager:
    def __init__(
        self,
        enabled: bool,
        bad_block_threshold: int,
        ambiguity_hz: float,
        ambiguity_block_threshold: int,
        release_hz: float,
        search_center_hz: float,
        trigger_on_ambiguity: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.bad_block_threshold = max(1, int(bad_block_threshold))
        self.ambiguity_hz = max(0.0, float(ambiguity_hz))
        self.ambiguity_block_threshold = max(1, int(ambiguity_block_threshold))
        self.release_hz = max(0.0, float(release_hz))
        self.trigger_on_ambiguity = bool(trigger_on_ambiguity)
        self.active = False
        self.reason = ""
        self.bad_block_counter = 0
        self.ambiguity_counter = 0
        self.capture_block_count = 0
        self.capture_count = 0
        self.search_center_hz = float(search_center_hz)
        self.estimated_tone_bb_hz = float("nan")
        self.estimated_error_hz = float("nan")
        self.estimated_peak_power_db = float("nan")
        self.estimated_peak_prominence_db = float("nan")
        self.estimate_valid = False

    def snapshot(
        self,
        just_started: bool = False,
        just_completed: bool = False,
    ) -> RecaptureState:
        return RecaptureState(
            active=bool(self.active),
            just_started=bool(just_started),
            just_completed=bool(just_completed),
            reason=str(self.reason),
            bad_block_counter=int(self.bad_block_counter),
            ambiguity_counter=int(self.ambiguity_counter),
            capture_block_count=int(self.capture_block_count),
            capture_count=int(self.capture_count),
            search_center_hz=float(self.search_center_hz),
            estimated_tone_bb_hz=float(self.estimated_tone_bb_hz),
            estimated_error_hz=float(self.estimated_error_hz),
            estimated_peak_power_db=float(self.estimated_peak_power_db),
            estimated_peak_prominence_db=float(self.estimated_peak_prominence_db),
            estimate_valid=bool(self.estimate_valid),
        )

    def update_search_center(self, search_center_hz: float) -> RecaptureState:
        self.search_center_hz = float(search_center_hz)
        return self.snapshot()

    def _reset_counters(self) -> None:
        self.bad_block_counter = 0
        self.ambiguity_counter = 0
        self.capture_block_count = 0

    def start(self, reason: str) -> RecaptureState:
        if not self.enabled:
            return self.snapshot()
        self.active = True
        self.reason = str(reason)
        self.capture_count += 1
        self.estimated_tone_bb_hz = float("nan")
        self.estimated_error_hz = float("nan")
        self.estimated_peak_power_db = float("nan")
        self.estimated_peak_prominence_db = float("nan")
        self.estimate_valid = False
        self._reset_counters()
        return self.snapshot(just_started=True)

    def observe_block(self, gate_reason: str, meas: BlockMeasurement) -> RecaptureState:
        if not self.enabled:
            return self.snapshot()
        if self.active:
            return self.snapshot()

        bad_gate = str(gate_reason) in ("amp_below_threshold", "coherence_below_threshold")
        ambiguous = (
            meas.valid_phase_step
            and np.isfinite(meas.f_off_hz)
            and abs(float(meas.f_off_hz)) >= self.ambiguity_hz
        )

        if bad_gate:
            self.bad_block_counter += 1
        else:
            self.bad_block_counter = 0

        if ambiguous:
            self.ambiguity_counter += 1
        else:
            self.ambiguity_counter = 0

        if self.bad_block_counter >= self.bad_block_threshold:
            return self.start(f"gate_{gate_reason}")
        if self.trigger_on_ambiguity and self.ambiguity_counter >= self.ambiguity_block_threshold:
            return self.start("f_off_near_ambiguity")
        return self.snapshot()

    def note_capture_block(self) -> RecaptureState:
        if self.enabled and self.active:
            self.capture_block_count += 1
        return self.snapshot()

    def record_estimate(
        self,
        tone_bb_hz: float,
        tone_bb_ref_hz: float,
        peak_power_db: float,
        peak_prominence_db: float,
        estimate_valid: bool,
    ) -> RecaptureState:
        self.estimated_tone_bb_hz = float(tone_bb_hz)
        self.estimated_error_hz = float(tone_bb_hz) - float(tone_bb_ref_hz)
        self.estimated_peak_power_db = float(peak_power_db)
        self.estimated_peak_prominence_db = float(peak_prominence_db)
        self.estimate_valid = bool(estimate_valid)
        self.capture_block_count = 0
        just_completed = False
        if self.estimate_valid:
            self.search_center_hz = float(tone_bb_hz)
        if self.estimate_valid and abs(float(self.estimated_error_hz)) <= self.release_hz:
            self.active = False
            self.reason = ""
            self.bad_block_counter = 0
            self.ambiguity_counter = 0
            just_completed = True
        return self.snapshot(just_completed=just_completed)


class TonePhaseDetector:
    """Block phasor detector for the known CW tone."""

    def __init__(self, sample_rate_hz: float, block_size: int, tone_bb_hz: float) -> None:
        self.tracker = ToneTracker(sample_rate_hz, block_size, tone_bb_hz)

    @property
    def block_dt_s(self) -> float:
        return self.tracker.block_dt_s

    def process_block(self, iq: np.ndarray) -> BlockMeasurement:
        return self.tracker.process_block(iq)


class AlphaBetaClockStateEstimator:
    """Estimate phase and frequency as the state of the controlled clock."""

    def __init__(
        self,
        block_dt_s: float,
        alpha: float,
        beta: float,
        gamma: float,
        cycle_slip_threshold_rad: float,
    ) -> None:
        self.block_dt_s = max(float(block_dt_s), 1.0e-9)
        self.alpha = clamp(float(alpha), 0.0, 1.0)
        self.beta = max(0.0, float(beta))
        self.gamma = max(0.0, float(gamma))
        self.cycle_slip_threshold_rad = clamp(
            float(cycle_slip_threshold_rad),
            0.1,
            math.pi,
        )
        self.initialized = False
        self.phase_rad = 0.0
        self.unwrapped_phase_rad = 0.0
        self.frequency_hz = 0.0
        self.drift_hz_per_s = 0.0
        self.last_confidence = 0.0

    def reset(
        self,
        phase_rad: Optional[float] = None,
        frequency_hz: float = 0.0,
    ) -> None:
        if phase_rad is None or not np.isfinite(phase_rad):
            self.initialized = False
            self.phase_rad = 0.0
            self.unwrapped_phase_rad = 0.0
        else:
            self.initialized = True
            self.phase_rad = wrap_to_pi(float(phase_rad))
            self.unwrapped_phase_rad = float(self.phase_rad)
        self.frequency_hz = float(frequency_hz) if np.isfinite(frequency_hz) else 0.0
        self.drift_hz_per_s = 0.0
        self.last_confidence = 0.0

    def update(
        self,
        meas: BlockMeasurement,
        measurement_valid: bool,
    ) -> ClockEstimate:
        confidence = clamp(float(meas.coherence) * float(meas.amp) / 0.10, 0.0, 1.0)
        if not self.initialized:
            self.phase_rad = wrap_to_pi(meas.phase_rad)
            self.unwrapped_phase_rad = float(self.phase_rad)
            self.frequency_hz = float(meas.f_off_hz) if np.isfinite(meas.f_off_hz) else 0.0
            self.drift_hz_per_s = 0.0
            self.initialized = True
            self.last_confidence = confidence
            return ClockEstimate(
                phase_rad=float(self.phase_rad),
                unwrapped_phase_rad=float(self.unwrapped_phase_rad),
                frequency_hz=float(self.frequency_hz),
                drift_hz_per_s=float(self.drift_hz_per_s),
                phase_residual_rad=0.0,
                confidence=float(confidence),
                valid=bool(measurement_valid),
                cycle_slip=False,
            )

        dt_s = self.block_dt_s
        predicted_unwrapped = self.unwrapped_phase_rad + (
            2.0 * math.pi * self.frequency_hz * dt_s
        )
        predicted_phase = wrap_to_pi(predicted_unwrapped)
        residual_rad = wrap_to_pi(float(meas.phase_rad) - predicted_phase)
        cycle_slip = bool(abs(residual_rad) >= self.cycle_slip_threshold_rad)

        if measurement_valid and not cycle_slip:
            weighted_alpha = self.alpha * confidence
            weighted_beta = self.beta * confidence
            weighted_gamma = self.gamma * confidence
            self.unwrapped_phase_rad = predicted_unwrapped + (weighted_alpha * residual_rad)
            self.phase_rad = wrap_to_pi(self.unwrapped_phase_rad)
            if np.isfinite(meas.f_off_hz):
                self.frequency_hz += weighted_beta * (
                    float(meas.f_off_hz) - self.frequency_hz
                )
            else:
                self.frequency_hz += (
                    weighted_beta * residual_rad / (2.0 * math.pi * dt_s)
                )
            self.drift_hz_per_s += (
                weighted_gamma * residual_rad / (2.0 * math.pi * dt_s * dt_s)
            )
        else:
            self.unwrapped_phase_rad = predicted_unwrapped
            self.phase_rad = predicted_phase

        self.last_confidence = confidence
        return ClockEstimate(
            phase_rad=float(self.phase_rad),
            unwrapped_phase_rad=float(self.unwrapped_phase_rad),
            frequency_hz=float(self.frequency_hz),
            drift_hz_per_s=float(self.drift_hz_per_s),
            phase_residual_rad=float(residual_rad),
            confidence=float(confidence),
            valid=bool(measurement_valid and not cycle_slip),
            cycle_slip=cycle_slip,
        )


class VcxoActuatorModel:
    """Physical conversion between desired frequency correction and command."""

    def __init__(
        self,
        hz_per_command: float,
        command_min: float,
        command_max: float,
        initial_command: float,
        proportional_gain: float,
        integral_gain: float,
        frequency_deadband_hz: float,
        command_slew_rate_per_s: float,
    ) -> None:
        if not np.isfinite(hz_per_command) or abs(float(hz_per_command)) < 1.0e-12:
            raise ValueError("hz_per_command must be finite and non-zero.")
        self.hz_per_command = float(hz_per_command)
        self.command_min = float(command_min)
        self.command_max = float(command_max)
        self.command = clamp(float(initial_command), self.command_min, self.command_max)
        self.base_command = float(self.command)
        self.proportional_gain = max(0.0, float(proportional_gain))
        self.integral_gain = max(0.0, float(integral_gain))
        self.frequency_deadband_hz = max(0.0, float(frequency_deadband_hz))
        self.command_slew_rate_per_s = max(0.0, float(command_slew_rate_per_s))
        self.integral_command = 0.0
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_effective_error_hz = 0.0
        self.last_unlimited_command = float(self.command)
        self.last_slew_limited = False
        self.last_base_delta = 0.0
        self.last_base_update_allowed = False
        self.last_profile = "tracking"

    def seed_from_command(self, command: float) -> None:
        self.command = clamp(float(command), self.command_min, self.command_max)
        self.base_command = float(self.command)
        self.integral_command = 0.0
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_effective_error_hz = 0.0
        self.last_unlimited_command = float(self.command)
        self.last_slew_limited = False
        self.last_base_delta = 0.0
        self.last_base_update_allowed = False
        self.last_profile = "tracking"

    def adapt_base_command(
        self,
        frequency_error_hz: float,
        dt_s: float,
        gain_per_s: float,
        deadband_hz: float,
        slew_rate_per_s: float,
    ) -> float:
        self.last_base_delta = 0.0
        self.last_base_update_allowed = False
        gain = max(0.0, float(gain_per_s))
        if gain <= 0.0:
            return 0.0
        error_hz = float(frequency_error_hz)
        if not np.isfinite(error_hz):
            return 0.0
        deadband = max(0.0, float(deadband_hz))
        effective_error_hz = 0.0 if abs(error_hz) <= deadband else error_hz
        if effective_error_hz == 0.0:
            return 0.0

        dt = max(float(dt_s), 1.0e-6)
        requested_delta = -gain * effective_error_hz * dt / self.hz_per_command
        slew_rate = max(0.0, float(slew_rate_per_s))
        if slew_rate > 0.0:
            max_delta = slew_rate * dt
            requested_delta = clamp(requested_delta, -max_delta, max_delta)

        updated_base = clamp(
            self.base_command + requested_delta,
            self.command_min,
            self.command_max,
        )
        applied_delta = float(updated_base - self.base_command)
        self.base_command = float(updated_base)
        self.last_base_delta = applied_delta
        self.last_base_update_allowed = applied_delta != 0.0
        return applied_delta

    def update(
        self,
        frequency_error_hz: float,
        dt_s: float,
        proportional_gain: Optional[float] = None,
        integral_gain: Optional[float] = None,
        frequency_deadband_hz: Optional[float] = None,
        command_slew_rate_per_s: Optional[float] = None,
        feedforward_command_offset: float = 0.0,
        profile: str = "tracking",
    ) -> ControlUpdate:
        dt_s = max(float(dt_s), 1.0e-6)
        error_hz = float(frequency_error_hz)
        ff_cmd = float(feedforward_command_offset)
        if not np.isfinite(ff_cmd):
            ff_cmd = 0.0
        p_gain = self.proportional_gain if proportional_gain is None else max(0.0, float(proportional_gain))
        i_gain = self.integral_gain if integral_gain is None else max(0.0, float(integral_gain))
        deadband_hz = self.frequency_deadband_hz if frequency_deadband_hz is None else max(0.0, float(frequency_deadband_hz))
        slew_rate = self.command_slew_rate_per_s if command_slew_rate_per_s is None else max(0.0, float(command_slew_rate_per_s))
        effective_error_hz = 0.0 if abs(error_hz) <= deadband_hz else error_hz
        p_cmd = -p_gain * effective_error_hz / self.hz_per_command
        i_candidate = self.integral_command - (
            i_gain * effective_error_hz * dt_s / self.hz_per_command
        )
        requested_unlimited = self.base_command + ff_cmd + p_cmd + i_candidate

        requested = clamp(requested_unlimited, self.command_min, self.command_max)
        range_limited = requested != requested_unlimited
        slew_limited = False
        if slew_rate > 0.0:
            max_delta = slew_rate * dt_s
            delta = requested - self.command
            limited_delta = clamp(delta, -max_delta, max_delta)
            slew_limited = limited_delta != delta
            requested = self.command + limited_delta

        if not range_limited and not slew_limited:
            self.integral_command = i_candidate

        self.command = float(requested)
        self.last_p_term = float(p_cmd)
        self.last_i_term = float(self.integral_command)
        self.last_effective_error_hz = float(effective_error_hz)
        self.last_unlimited_command = float(requested_unlimited)
        self.last_slew_limited = bool(slew_limited)
        self.last_profile = str(profile)
        return ControlUpdate(
            applied_command=float(requested),
            requested_command=float(requested),
            p_term=float(p_cmd),
            i_term=float(self.integral_command),
            error_hz=float(error_hz),
            base_delta=float(self.last_base_delta),
            base_update_allowed=bool(self.last_base_update_allowed),
        )


class PhaseDisciplineController:
    """Mode manager and phase-first controller for one VCXO actuator."""

    def __init__(
        self,
        phase_target_rad: Optional[float],
        target_is_explicit: bool,
        arm_threshold_hz: float,
        arm_dwell_updates: int,
        hold_exit_dwell_updates: int,
        pull_in_enabled: bool,
        switch_hz: float,
        exit_hz: float,
        enter_dwell_updates: int,
        exit_dwell_updates: int,
        hold_gain_hz_per_rad: float,
        hold_max_freq_hz: float,
        slew_gain_hz_per_rad: float,
        slew_max_freq_hz: float,
        slew_enter_error_rad: float,
        slew_exit_error_rad: float,
        recovery_enter_dwell_updates: int,
        recovery_exit_dwell_updates: int,
        recovery_min_dwell_updates: int,
        smoothing_alpha: float,
        target_slew_rate_rad_per_s: float,
        phase_integral_gain_hz_per_rad_s: float,
        phase_integral_max_hz: float,
        phase_error_filter_alpha: float,
        phase_error_min_confidence: float,
        phase_error_max_residual_rad: float,
        phase_cycle_tracking_enabled: bool,
        phase_cycle_max_branch_error_rad: float,
        update_interval_s: float,
    ) -> None:
        self.final_phase_target_rad = (
            wrap_to_pi(float(phase_target_rad)) if phase_target_rad is not None else None
        )
        self.phase_target_rad: Optional[float] = None
        self.target_is_explicit = bool(target_is_explicit)
        self.arm_threshold_hz = max(0.0, float(arm_threshold_hz))
        self.arm_dwell_updates = max(1, int(arm_dwell_updates))
        self.hold_exit_dwell_updates = max(0, int(hold_exit_dwell_updates))
        self.pull_in_enabled = bool(pull_in_enabled)
        self.switch_hz = max(0.0, float(switch_hz))
        self.exit_hz = max(self.switch_hz, float(exit_hz))
        self.enter_dwell_updates = max(1, int(enter_dwell_updates))
        self.exit_dwell_updates = max(0, int(exit_dwell_updates))
        self.hold_gain_hz_per_rad = max(0.0, float(hold_gain_hz_per_rad))
        self.hold_max_freq_hz = max(0.0, float(hold_max_freq_hz))
        self.slew_gain_hz_per_rad = max(0.0, float(slew_gain_hz_per_rad))
        self.slew_max_freq_hz = max(0.0, float(slew_max_freq_hz))
        self.slew_enter_error_rad = max(0.0, float(slew_enter_error_rad))
        self.slew_exit_error_rad = max(0.0, float(slew_exit_error_rad))
        self.recovery_enter_dwell_updates = max(1, int(recovery_enter_dwell_updates))
        self.recovery_exit_dwell_updates = max(1, int(recovery_exit_dwell_updates))
        self.recovery_min_dwell_updates = max(0, int(recovery_min_dwell_updates))
        self.smoothing_alpha = clamp(float(smoothing_alpha), 0.0, 1.0)
        self.target_slew_rate_rad_per_s = max(0.0, float(target_slew_rate_rad_per_s))
        self.phase_integral_gain_hz_per_rad_s = max(0.0, float(phase_integral_gain_hz_per_rad_s))
        self.phase_integral_max_hz = max(0.0, float(phase_integral_max_hz))
        self.phase_error_filter_alpha = clamp(float(phase_error_filter_alpha), 0.0, 1.0)
        self.phase_error_min_confidence = clamp(float(phase_error_min_confidence), 0.0, 1.0)
        self.phase_error_max_residual_rad = max(0.0, float(phase_error_max_residual_rad))
        self.phase_cycle_tracking_enabled = bool(phase_cycle_tracking_enabled)
        self.phase_cycle_max_branch_error_rad = max(0.0, float(phase_cycle_max_branch_error_rad))
        self.update_interval_s = max(1.0e-9, float(update_interval_s))
        self.stage = "pull_in" if self.pull_in_enabled else "fine_lock"
        self.phase_mode = "idle"
        self.enter_counter = 0
        self.exit_counter = 0
        self.arm_counter = 0
        self.phase_hold_exit_counter = 0
        self.recovery_enter_counter = 0
        self.recovery_enter_sign = 0
        self.recovery_exit_counter = 0
        self.recovery_mode_counter = 0
        self.stage_switches = 0
        self.filtered_target_f_off_hz = 0.0
        self.filter_initialized = False
        self.filtered_phase_error_rad = 0.0
        self.phase_error_filter_initialized = False
        self.last_phase_error_update_allowed = False
        self.phase_cycle_error_rad = 0.0
        self.phase_cycle_initialized = False
        self.phase_cycle_slip_count = 0
        self.last_phase_error_predicted_rad = float("nan")
        self.last_phase_error_branch_index = 0.0
        self.last_phase_error_branch_residual_rad = float("nan")
        self.last_phase_cycle_update_allowed = False
        self.last_phase_cycle_slip_detected = False
        self.phase_integral_hz = 0.0

    def _reset_phase_error_observer(self) -> None:
        self.filtered_phase_error_rad = 0.0
        self.phase_error_filter_initialized = False
        self.last_phase_error_update_allowed = False
        self.phase_cycle_error_rad = 0.0
        self.phase_cycle_initialized = False
        self.phase_cycle_slip_count = 0
        self.last_phase_error_predicted_rad = float("nan")
        self.last_phase_error_branch_index = 0.0
        self.last_phase_error_branch_residual_rad = float("nan")
        self.last_phase_cycle_update_allowed = False
        self.last_phase_cycle_slip_detected = False

    def _reset_recovery_dwell(self) -> None:
        self.recovery_enter_counter = 0
        self.recovery_enter_sign = 0
        self.recovery_exit_counter = 0
        self.recovery_mode_counter = 0

    @property
    def phase_enabled(self) -> bool:
        return (
            self.hold_gain_hz_per_rad > 0.0
            or self.slew_gain_hz_per_rad > 0.0
            or self.phase_integral_gain_hz_per_rad_s > 0.0
        )

    def force_pull_in(self) -> ControlStageState:
        if not self.pull_in_enabled:
            return self.stage_snapshot()
        if self.stage != "pull_in":
            self.stage_switches += 1
        self.stage = "pull_in"
        self.phase_mode = "idle"
        self.enter_counter = 0
        self.exit_counter = 0
        self.arm_counter = 0
        self.phase_hold_exit_counter = 0
        self.phase_target_rad = None
        self.filtered_target_f_off_hz = 0.0
        self.filter_initialized = False
        self._reset_phase_error_observer()
        self._reset_recovery_dwell()
        self.phase_integral_hz = 0.0
        return self.stage_snapshot("recapture_pull_in")

    def stage_snapshot(self, event: str = "") -> ControlStageState:
        return ControlStageState(
            stage=str(self.stage),
            switch_event=str(event),
            enter_counter=int(self.enter_counter),
            exit_counter=int(self.exit_counter),
            stage_switches=int(self.stage_switches),
        )

    def update(
        self,
        estimate: ClockEstimate,
        update_valid: bool,
        phase_hold_allowed: bool = True,
    ) -> PhaseDisciplineCommand:
        switch_event = ""
        if not update_valid or not np.isfinite(estimate.frequency_hz):
            return self._command(estimate, switch_event, phase_hold_allowed=phase_hold_allowed)

        abs_freq_hz = abs(float(estimate.frequency_hz))
        if self.pull_in_enabled and self.stage == "pull_in":
            self.exit_counter = 0
            if abs_freq_hz <= self.switch_hz:
                self.enter_counter += 1
            else:
                self.enter_counter = 0
            if self.enter_counter >= self.enter_dwell_updates:
                self.stage = "fine_lock"
                self.enter_counter = 0
                self.exit_counter = 0
                self.stage_switches += 1
                switch_event = "enter_fine_lock"
        else:
            self.enter_counter = 0
            if self.pull_in_enabled and self.exit_dwell_updates > 0 and abs_freq_hz >= self.exit_hz:
                self.exit_counter += 1
            else:
                self.exit_counter = 0
            if self.exit_dwell_updates > 0 and self.exit_counter >= self.exit_dwell_updates:
                self.stage = "pull_in"
                self.phase_mode = "idle"
                self.arm_counter = 0
                self.phase_hold_exit_counter = 0
                self.phase_target_rad = None
                self.filtered_target_f_off_hz = 0.0
                self.filter_initialized = False
                self._reset_phase_error_observer()
                self._reset_recovery_dwell()
                self.phase_integral_hz = 0.0
                self.enter_counter = 0
                self.exit_counter = 0
                self.stage_switches += 1
                switch_event = "pushback_pull_in"

        return self._command(estimate, switch_event, phase_hold_allowed=phase_hold_allowed)

    def _command(
        self,
        estimate: ClockEstimate,
        switch_event: str,
        phase_hold_allowed: bool = True,
    ) -> PhaseDisciplineCommand:
        initial_phase_handoff_blocked = (
            not bool(phase_hold_allowed)
            and self.phase_target_rad is None
            and self.phase_mode == "idle"
        )
        if not self.phase_enabled or initial_phase_handoff_blocked:
            self.phase_mode = "idle"
            self.arm_counter = 0
            self.phase_hold_exit_counter = 0
            self.filtered_target_f_off_hz = 0.0
            self.filter_initialized = False
            self._reset_phase_error_observer()
            self._reset_recovery_dwell()
            self.phase_integral_hz = 0.0
            return PhaseDisciplineCommand(
                stage=str(self.stage),
                switch_event=str(switch_event),
                phase_mode="idle",
                target_phase_rad=float_or_nan(self.phase_target_rad),
                phase_error_rad=float("nan"),
                phase_error_raw_rad=float("nan"),
                phase_error_filtered_rad=float("nan"),
                phase_error_update_allowed=False,
                phase_error_cycle_rad=float("nan"),
                phase_error_predicted_rad=float("nan"),
                phase_error_branch_index=float("nan"),
                phase_error_branch_residual_rad=float("nan"),
                phase_cycle_update_allowed=False,
                phase_cycle_slip_detected=False,
                phase_cycle_slip_count=int(self.phase_cycle_slip_count),
                target_f_off_hz=0.0,
                raw_target_f_off_hz=0.0,
                frequency_error_hz=float(estimate.frequency_hz),
                phase_gain_hz_per_rad=0.0,
                max_target_f_off_hz=0.0,
                phase_active=False,
                just_armed=False,
            )

        just_armed = False
        # An explicit phase target does not need a separate "capture target" dwell.
        # Once the frequency estimate is inside the handoff band, keep a low-bandwidth
        # phase term active even if noisy measurements temporarily push the stage back
        # to pull-in. This lets the loop reduce phase instead of only nulling slope.
        phase_engage_threshold_hz = self.arm_threshold_hz
        if self.target_is_explicit:
            phase_engage_threshold_hz = max(
                self.arm_threshold_hz,
                self.switch_hz,
                self.exit_hz,
            )
        required_arm_updates = 1 if self.target_is_explicit else self.arm_dwell_updates
        phase_already_acquired = (
            self.phase_target_rad is not None
            and self.phase_mode != "idle"
        )
        if abs(float(estimate.frequency_hz)) <= phase_engage_threshold_hz:
            self.arm_counter += 1
            self.phase_hold_exit_counter = 0
        else:
            if phase_already_acquired and self.hold_exit_dwell_updates > 0:
                self.phase_hold_exit_counter += 1
                if self.phase_hold_exit_counter < self.hold_exit_dwell_updates:
                    self.arm_counter = max(self.arm_counter, required_arm_updates)
                else:
                    self.arm_counter = 0
                    self.phase_hold_exit_counter = 0
                    if not self.target_is_explicit:
                        self.phase_target_rad = None
                    self.phase_mode = "idle"
                    self.filtered_target_f_off_hz = 0.0
                    self.filter_initialized = False
                    self._reset_phase_error_observer()
                    self._reset_recovery_dwell()
                    self.phase_integral_hz = 0.0
            else:
                self.arm_counter = 0
                self.phase_hold_exit_counter = 0
                if not self.target_is_explicit:
                    self.phase_target_rad = None
                self.phase_mode = "idle"
                self.filtered_target_f_off_hz = 0.0
                self.filter_initialized = False
                self._reset_phase_error_observer()
                self._reset_recovery_dwell()
                self.phase_integral_hz = 0.0

        if self.arm_counter < required_arm_updates:
            return PhaseDisciplineCommand(
                stage=str(self.stage),
                switch_event=str(switch_event),
                phase_mode="idle",
                target_phase_rad=float_or_nan(self.phase_target_rad),
                phase_error_rad=float("nan"),
                phase_error_raw_rad=float("nan"),
                phase_error_filtered_rad=float("nan"),
                phase_error_update_allowed=False,
                phase_error_cycle_rad=float("nan"),
                phase_error_predicted_rad=float("nan"),
                phase_error_branch_index=float("nan"),
                phase_error_branch_residual_rad=float("nan"),
                phase_cycle_update_allowed=False,
                phase_cycle_slip_detected=False,
                phase_cycle_slip_count=int(self.phase_cycle_slip_count),
                target_f_off_hz=0.0,
                raw_target_f_off_hz=0.0,
                frequency_error_hz=float(estimate.frequency_hz),
                phase_gain_hz_per_rad=0.0,
                max_target_f_off_hz=0.0,
                phase_active=False,
                just_armed=False,
            )

        if self.phase_target_rad is None:
            self.phase_target_rad = wrap_to_pi(float(estimate.phase_rad))
            just_armed = True
        elif self.phase_mode == "idle":
            just_armed = True

        if self.target_is_explicit and self.final_phase_target_rad is not None:
            target_delta_rad = wrap_to_pi(float(self.final_phase_target_rad) - float(self.phase_target_rad))
            max_step_rad = self.target_slew_rate_rad_per_s * self.update_interval_s
            if max_step_rad > 0.0:
                target_step_rad = clamp(target_delta_rad, -max_step_rad, max_step_rad)
                self.phase_target_rad = wrap_to_pi(float(self.phase_target_rad) + target_step_rad)
            if abs(target_delta_rad) > max(1.0e-6, max_step_rad):
                self.phase_mode = "ramp"
            else:
                self.phase_target_rad = float(self.final_phase_target_rad)
                self.phase_mode = "hold"

        phase_error_raw_rad = wrap_to_pi(float(estimate.phase_rad) - float(self.phase_target_rad))
        phase_error_update_allowed = bool(
            np.isfinite(phase_error_raw_rad)
            and np.isfinite(estimate.confidence)
            and float(estimate.confidence) >= self.phase_error_min_confidence
            and (
                not np.isfinite(estimate.phase_residual_rad)
                or abs(float(estimate.phase_residual_rad)) <= self.phase_error_max_residual_rad
            )
            and not bool(estimate.cycle_slip)
        )
        if phase_error_update_allowed or not self.phase_error_filter_initialized:
            if not self.phase_error_filter_initialized:
                self.filtered_phase_error_rad = float(phase_error_raw_rad)
                self.phase_error_filter_initialized = True
            else:
                self.filtered_phase_error_rad = wrap_to_pi(
                    float(self.filtered_phase_error_rad)
                    + self.phase_error_filter_alpha
                    * wrap_to_pi(
                        float(phase_error_raw_rad) - float(self.filtered_phase_error_rad)
                    )
                )
        self.last_phase_error_update_allowed = bool(phase_error_update_allowed)
        predicted_phase_error_rad = (
            float(self.phase_cycle_error_rad)
            + 2.0
            * math.pi
            * float(estimate.frequency_hz)
            * float(self.update_interval_s)
        )
        if not self.phase_cycle_initialized or not self.phase_cycle_tracking_enabled:
            predicted_phase_error_rad = float(self.filtered_phase_error_rad)

        if np.isfinite(predicted_phase_error_rad) and np.isfinite(phase_error_raw_rad):
            branch_index = float(
                round((predicted_phase_error_rad - float(phase_error_raw_rad)) / (2.0 * math.pi))
            )
            branch_phase_error_rad = float(phase_error_raw_rad) + branch_index * 2.0 * math.pi
            branch_residual_rad = branch_phase_error_rad - predicted_phase_error_rad
        else:
            branch_index = float("nan")
            branch_phase_error_rad = float("nan")
            branch_residual_rad = float("nan")

        phase_cycle_slip_detected = bool(
            phase_error_update_allowed
            and np.isfinite(branch_residual_rad)
            and abs(float(branch_residual_rad)) > self.phase_cycle_max_branch_error_rad
        )
        phase_cycle_update_allowed = bool(
            self.phase_cycle_tracking_enabled
            and phase_error_update_allowed
            and np.isfinite(branch_phase_error_rad)
            and not phase_cycle_slip_detected
        )
        if not self.phase_cycle_tracking_enabled:
            self.phase_cycle_error_rad = float(self.filtered_phase_error_rad)
            self.phase_cycle_initialized = True
        elif not self.phase_cycle_initialized:
            self.phase_cycle_error_rad = (
                float(branch_phase_error_rad)
                if np.isfinite(branch_phase_error_rad)
                else float(self.filtered_phase_error_rad)
            )
            self.phase_cycle_initialized = True
        elif phase_cycle_update_allowed:
            self.phase_cycle_error_rad = float(predicted_phase_error_rad) + self.phase_error_filter_alpha * float(branch_residual_rad)
        else:
            self.phase_cycle_error_rad = float(predicted_phase_error_rad)
            if phase_cycle_slip_detected:
                self.phase_cycle_slip_count += 1

        self.last_phase_error_predicted_rad = float(predicted_phase_error_rad)
        self.last_phase_error_branch_index = float(branch_index)
        self.last_phase_error_branch_residual_rad = float(branch_residual_rad)
        self.last_phase_cycle_update_allowed = bool(phase_cycle_update_allowed)
        self.last_phase_cycle_slip_detected = bool(phase_cycle_slip_detected)
        phase_error_rad = (
            float(self.phase_cycle_error_rad)
            if self.phase_cycle_tracking_enabled
            else float(self.filtered_phase_error_rad)
        )
        if self.phase_mode == "idle":
            self.phase_mode = "hold"
            self._reset_recovery_dwell()

        abs_phase_error_rad = abs(float(phase_error_rad)) if np.isfinite(phase_error_rad) else float("inf")
        recovery_enabled = bool(
            self.slew_gain_hz_per_rad > 0.0
            and self.slew_max_freq_hz > 0.0
            and self.slew_enter_error_rad > 0.0
        )
        if recovery_enabled:
            if self.phase_mode == "recover":
                self.recovery_mode_counter += 1
                self.recovery_enter_counter = 0
                self.recovery_enter_sign = 0
                if (
                    abs_phase_error_rad <= self.slew_exit_error_rad
                    and self.recovery_mode_counter >= self.recovery_min_dwell_updates
                ):
                    self.recovery_exit_counter += 1
                else:
                    self.recovery_exit_counter = 0
                if self.recovery_exit_counter >= self.recovery_exit_dwell_updates:
                    self.phase_mode = "hold"
                    self._reset_recovery_dwell()
            elif abs_phase_error_rad >= self.slew_enter_error_rad:
                entry_sign = 1 if float(phase_error_rad) > 0.0 else -1
                entry_update_allowed = bool(
                    phase_error_update_allowed
                    and not phase_cycle_slip_detected
                    and (
                        not self.phase_cycle_tracking_enabled
                        or phase_cycle_update_allowed
                    )
                )
                if entry_update_allowed and (
                    self.recovery_enter_sign == 0
                    or self.recovery_enter_sign == entry_sign
                ):
                    self.recovery_enter_counter += 1
                    self.recovery_enter_sign = int(entry_sign)
                    self.recovery_exit_counter = 0
                    self.recovery_mode_counter = 0
                    if self.recovery_enter_counter >= self.recovery_enter_dwell_updates:
                        self.phase_mode = "recover"
                        self.recovery_enter_counter = 0
                        self.recovery_enter_sign = 0
                        self.recovery_exit_counter = 0
                        self.recovery_mode_counter = 1
                elif entry_update_allowed:
                    self.recovery_enter_counter = 1
                    self.recovery_enter_sign = int(entry_sign)
                    self.recovery_exit_counter = 0
                    self.recovery_mode_counter = 0
                else:
                    self._reset_recovery_dwell()
            else:
                self._reset_recovery_dwell()
        else:
            self._reset_recovery_dwell()

        recovery_active = self.phase_mode == "recover"
        gain = self.hold_gain_hz_per_rad
        max_freq = self.hold_max_freq_hz
        if recovery_active:
            gain += self.slew_gain_hz_per_rad
            max_freq += self.slew_max_freq_hz

        recovery_bias_error_limit_rad = max(1.5, 0.5 * float(self.slew_enter_error_rad), float(self.slew_exit_error_rad))
        phase_integral_update_allowed = bool(
            self.phase_integral_gain_hz_per_rad_s > 0.0
            and self.phase_integral_max_hz > 0.0
            and phase_error_update_allowed
            and not phase_cycle_slip_detected
            and (
                self.phase_mode == "hold"
                or (
                    recovery_active
                    and abs_phase_error_rad <= recovery_bias_error_limit_rad
                )
            )
        )
        if phase_integral_update_allowed:
            self.phase_integral_hz -= (
                self.phase_integral_gain_hz_per_rad_s
                * float(phase_error_rad)
                * self.update_interval_s
            )
            self.phase_integral_hz = clamp(
                self.phase_integral_hz,
                -self.phase_integral_max_hz,
                self.phase_integral_max_hz,
            )

        hold_target_f_off_hz = clamp(
            -float(self.hold_gain_hz_per_rad) * float(phase_error_rad) + self.phase_integral_hz,
            -float(self.hold_max_freq_hz),
            float(self.hold_max_freq_hz),
        )
        recovery_target_f_off_hz = 0.0
        if recovery_active:
            recovery_target_f_off_hz = clamp(
                -float(self.slew_gain_hz_per_rad) * float(phase_error_rad),
                -float(self.slew_max_freq_hz),
                float(self.slew_max_freq_hz),
            )
        raw_target_f_off_hz = hold_target_f_off_hz + recovery_target_f_off_hz
        limited_target_f_off_hz = clamp(raw_target_f_off_hz, -max_freq, max_freq)
        if not self.filter_initialized:
            self.filtered_target_f_off_hz = float(limited_target_f_off_hz)
            self.filter_initialized = True
        else:
            self.filtered_target_f_off_hz += self.smoothing_alpha * (
                float(limited_target_f_off_hz) - self.filtered_target_f_off_hz
            )
        self.filtered_target_f_off_hz = clamp(
            self.filtered_target_f_off_hz,
            -float(max_freq),
            float(max_freq),
        )

        frequency_error_hz = float(estimate.frequency_hz) - float(self.filtered_target_f_off_hz)
        return PhaseDisciplineCommand(
            stage=str(self.stage),
            switch_event=str(switch_event),
            phase_mode=str(self.phase_mode),
            target_phase_rad=float(self.phase_target_rad),
            phase_error_rad=float(phase_error_rad),
            phase_error_raw_rad=float(phase_error_raw_rad),
            phase_error_filtered_rad=float(self.filtered_phase_error_rad),
            phase_error_update_allowed=bool(phase_error_update_allowed),
            phase_error_cycle_rad=float(self.phase_cycle_error_rad),
            phase_error_predicted_rad=float(predicted_phase_error_rad),
            phase_error_branch_index=float(branch_index),
            phase_error_branch_residual_rad=float(branch_residual_rad),
            phase_cycle_update_allowed=bool(phase_cycle_update_allowed),
            phase_cycle_slip_detected=bool(phase_cycle_slip_detected),
            phase_cycle_slip_count=int(self.phase_cycle_slip_count),
            target_f_off_hz=float(self.filtered_target_f_off_hz),
            raw_target_f_off_hz=float(raw_target_f_off_hz),
            frequency_error_hz=float(frequency_error_hz),
            phase_gain_hz_per_rad=float(gain),
            max_target_f_off_hz=float(max_freq),
            phase_active=True,
            just_armed=bool(just_armed),
        )


class SDRBase:
    backend_name = "unknown"

    def warmup(self, block_size: int, blocks: int) -> None:
        for _ in range(int(max(0, blocks))):
            self.read_block(block_size)

    def read_block(self, block_size: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class UgradioSdrSource(SDRBase):
    backend_name = "ugradio"

    def __init__(
        self,
        sample_rate_hz: float,
        center_freq_hz: float,
        gain: str,
        device_index: int,
        queue_blocks: int,
    ) -> None:
        try:
            from ugradio import sdr as ugradio_sdr
        except Exception as exc:
            raise RuntimeError(
                "Could not import ugradio.sdr. Install ugradio in the active environment."
            ) from exc

        gain_value = float(gain) if str(gain).lower() != "auto" else 0.0
        self._sdr = ugradio_sdr.SDR(
            device_index=int(device_index),
            direct=False,
            center_freq=float(center_freq_hz),
            sample_rate=float(sample_rate_hz),
            gain=float(gain_value),
        )
        self._queue_blocks = max(4, int(queue_blocks))
        self._block_queue: "queue.Queue[np.ndarray | StreamSentinel]" = queue.Queue(
            maxsize=self._queue_blocks
        )
        self._stream_block_size: Optional[int] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream_task: Optional[asyncio.Task[Any]] = None
        self._thread_ready = threading.Event()
        self._stop_requested = threading.Event()
        self._read_timeout_s = 5.0

    def _clear_queue(self) -> None:
        while True:
            try:
                self._block_queue.get_nowait()
            except queue.Empty:
                return

    def _signal_stream_end(self, error: Optional[BaseException] = None) -> None:
        sentinel = StreamSentinel(error=error)
        while True:
            try:
                self._block_queue.put_nowait(sentinel)
                return
            except queue.Full:
                try:
                    self._block_queue.get_nowait()
                except queue.Empty:
                    return

    async def _stream_iq_blocks(self, block_size: int) -> None:
        try:
            async for raw in self._sdr.stream(
                num_samples_or_bytes=2 * int(block_size),
                format="bytes",
            ):
                iq_block = bytes_to_iq_block(raw, block_size)
                while not self._stop_requested.is_set():
                    try:
                        self._block_queue.put_nowait(iq_block)
                        break
                    except queue.Full:
                        await asyncio.sleep(0.001)
                if self._stop_requested.is_set():
                    break
        except asyncio.CancelledError:
            self._signal_stream_end()
            raise
        except Exception as exc:
            self._signal_stream_end(error=exc)
            raise
        else:
            self._signal_stream_end()
        finally:
            try:
                await self._sdr.stop()
            except Exception:
                pass

    def _stream_thread_main(self, block_size: int) -> None:
        loop = asyncio.new_event_loop()
        self._stream_loop = loop
        asyncio.set_event_loop(loop)
        self._thread_ready.set()
        task = loop.create_task(self._stream_iq_blocks(block_size))
        self._stream_task = task
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            pending = [pending_task for pending_task in asyncio.all_tasks(loop) if not pending_task.done()]
            for pending_task in pending:
                pending_task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._stream_task = None
            self._stream_loop = None
            loop.close()

    def _stop_stream(self, swallow_errors: bool) -> None:
        self._stop_requested.set()
        if self._stream_loop is not None and self._stream_task is not None:
            def _cancel_task() -> None:
                if self._stream_task is not None and not self._stream_task.done():
                    self._stream_task.cancel()

            try:
                self._stream_loop.call_soon_threadsafe(_cancel_task)
            except Exception:
                pass
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=5.0)
            thread_alive = self._stream_thread.is_alive()
            self._stream_thread = None
            if thread_alive and not swallow_errors:
                raise RuntimeError("Timed out stopping ugradio async stream thread.")
        self._stream_block_size = None

    def _ensure_stream(self, block_size: int) -> None:
        requested = int(block_size)
        thread_alive = self._stream_thread is not None and self._stream_thread.is_alive()
        if thread_alive and self._stream_block_size == requested:
            return
        self._stop_stream(swallow_errors=True)
        self._clear_queue()
        self._stop_requested.clear()
        self._thread_ready.clear()
        self._stream_block_size = requested
        self._stream_thread = threading.Thread(
            target=self._stream_thread_main,
            args=(requested,),
            daemon=True,
            name="ugradio-stream-reader",
        )
        self._stream_thread.start()
        if not self._thread_ready.wait(timeout=2.0):
            raise RuntimeError("Timed out starting ugradio async stream thread.")

    def read_block(self, block_size: int) -> np.ndarray:
        self._ensure_stream(block_size)
        try:
            item = self._block_queue.get(timeout=self._read_timeout_s)
        except queue.Empty as exc:
            thread_alive = self._stream_thread is not None and self._stream_thread.is_alive()
            if not thread_alive:
                raise RuntimeError("ugradio async stream stopped producing IQ blocks.") from exc
            raise RuntimeError(
                f"Timed out waiting for an IQ block from ugradio async stream after {self._read_timeout_s:.1f} s."
            ) from exc

        if isinstance(item, StreamSentinel):
            self._stop_stream(swallow_errors=True)
            if item.error is not None:
                raise RuntimeError("ugradio async stream terminated unexpectedly.") from item.error
            raise RuntimeError("ugradio async stream ended.")
        return item

    def close(self) -> None:
        self._stop_stream(swallow_errors=True)
        self._clear_queue()
        try:
            self._sdr.close()
        except Exception:
            pass


class ActuatorBase:
    actuator_name = "none"
    command_label = "control_cmd"

    def open(self) -> None:
        pass

    def write_command(self, command: float, settle_s: float = 0.0) -> float:
        return float(command)

    def command_to_volts(self, command: float) -> Optional[float]:
        return None

    def csv_status(self) -> Dict[str, float]:
        return {
            "actuator_dither_enabled": 0.0,
            "actuator_dither_error": float("nan"),
            "actuator_write_skipped": float("nan"),
        }

    def close(self) -> None:
        pass


class NullActuator(ActuatorBase):
    actuator_name = "none"
    command_label = "none"


class Mcp4725Actuator(ActuatorBase):
    actuator_name = "mcp4725"
    command_label = "dac_code"

    def __init__(
        self,
        command_min: float,
        command_max: float,
        clock_apr_ppm: int,
        skip_clock_init: bool,
        enable_dither: bool,
    ) -> None:
        self.command_min = int(command_min)
        self.command_max = int(command_max)
        self.clock_apr_ppm = int(clock_apr_ppm)
        self.skip_clock_init = bool(skip_clock_init)
        self.enable_dither = bool(enable_dither)
        self._clock = None
        self._dither_error = 0.0
        self._last_code: Optional[int] = None
        self._last_write_skipped = False

    def _quantize_code(self, command: float) -> int:
        limited = clamp(float(command), self.command_min, self.command_max)
        if not self.enable_dither:
            self._dither_error = 0.0
            return clamp_int(limited, self.command_min, self.command_max)

        effective = limited + self._dither_error
        code = clamp_int(effective, self.command_min, self.command_max)

        if (code <= self.command_min and effective <= self.command_min) or (
            code >= self.command_max and effective >= self.command_max
        ):
            self._dither_error = 0.0
        else:
            self._dither_error = clamp(effective - float(code), -0.5, 0.5)
        return code

    def open(self) -> None:
        try:
            import clock_vcxo_helper as cvh
        except Exception as exc:
            repo_root = str(REPO_ROOT)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            try:
                import clock_vcxo_helper as cvh
            except Exception as retry_exc:
                raise RuntimeError(
                    "Unable to import clock_vcxo_helper.py for MCP4725 control."
                ) from retry_exc

        self._clock = cvh.ClockVCXO()
        self._clock.open()
        if not self.skip_clock_init:
            self._clock.init_clock_vcxo_28p8mhz(apr_ppm=self.clock_apr_ppm)

    def write_command(self, command: float, settle_s: float = 0.0) -> float:
        if self._clock is None:
            raise RuntimeError("MCP4725 actuator is not open.")
        code = self._quantize_code(command)
        if self._last_code == code:
            self._last_write_skipped = True
            return float(code)
        self._last_write_skipped = False
        readback = self._clock.set_dac_code(code, settle_s=float(settle_s))
        self._last_code = int(round(float(readback)))
        return float(readback)

    def command_to_volts(self, command: float) -> Optional[float]:
        if self._clock is None:
            return None
        return float(self._clock.dac_code_to_volts(int(round(float(command)))))

    def csv_status(self) -> Dict[str, float]:
        return {
            "actuator_dither_enabled": float(self.enable_dither),
            "actuator_dither_error": float(self._dither_error),
            "actuator_write_skipped": float(1.0 if self._last_write_skipped else 0.0),
        }

    def close(self) -> None:
        if self._clock is not None:
            try:
                self._clock.close()
            except Exception:
                pass
            self._clock = None


class PwmActuator(ActuatorBase):
    actuator_name = "pwm"
    command_label = "pwm_duty"

    def __init__(
        self,
        gpio_pin: int,
        pwm_frequency_hz: float,
        pwm_vdd: float,
        command_min: float,
        command_max: float,
    ) -> None:
        self.gpio_pin = int(gpio_pin)
        self.pwm_frequency_hz = float(pwm_frequency_hz)
        self.pwm_vdd = float(pwm_vdd)
        self.command_min = float(command_min)
        self.command_max = float(command_max)
        self._device = None

    def open(self) -> None:
        if self.command_min < 0.0 or self.command_max > 1.0:
            raise ValueError("PWM command range must stay within [0.0, 1.0].")
        try:
            from gpiozero import PWMOutputDevice
        except Exception as exc:
            raise RuntimeError(
                "Unable to import gpiozero for PWM control."
            ) from exc

        self._device = PWMOutputDevice(
            pin=self.gpio_pin,
            frequency=self.pwm_frequency_hz,
            initial_value=0.0,
        )

    def write_command(self, command: float, settle_s: float = 0.0) -> float:
        if self._device is None:
            raise RuntimeError("PWM actuator is not open.")
        duty = clamp(float(command), self.command_min, self.command_max)
        self._device.value = duty
        if settle_s > 0.0:
            time.sleep(float(settle_s))
        return float(duty)

    def command_to_volts(self, command: float) -> Optional[float]:
        duty = clamp(float(command), self.command_min, self.command_max)
        return float(self.pwm_vdd * duty)

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.off()
                self._device.close()
            except Exception:
                pass
            self._device = None


def resolve_tone_reference(args: argparse.Namespace) -> ToneReference:
    if args.tone_bb_hz is not None:
        return ToneReference(
            tone_bb_hz=float(args.tone_bb_hz),
            tone_rf_hz=float(args.tone_rf_hz) if args.tone_rf_hz is not None else None,
            source="tone_bb_or_expected_offset_hz",
        )

    if args.tone_rf_hz is None:
        raise ValueError(
            "Specify either --tone-bb/--expected-offset-hz or provide --tone-rf with --center-freq."
        )

    tone_rf_hz = float(args.tone_rf_hz)
    tone_bb_hz = tone_rf_hz - float(args.center_freq_hz)
    return ToneReference(
        tone_bb_hz=float(tone_bb_hz),
        tone_rf_hz=tone_rf_hz,
        source="tone_rf_minus_center_freq",
    )


def choose_sdr_backend(args: argparse.Namespace) -> SDRBase:
    backend = str(args.sdr_backend).lower()
    if backend != "ugradio":
        raise ValueError("Only the ugradio SDR backend is supported in first_stage_demo.")
    return UgradioSdrSource(
        sample_rate_hz=args.sample_rate_hz,
        center_freq_hz=args.center_freq_hz,
        gain=args.gain,
        device_index=args.device_index,
        queue_blocks=args.ugradio_queue_blocks,
    )


def resolve_command_defaults(args: argparse.Namespace) -> None:
    if args.use_mcp4725 and args.use_pwm:
        raise ValueError("Choose only one actuator path: --use-mcp4725 or --use-pwm.")

    actuator_name = infer_requested_actuator_name(args)
    if actuator_name == "pwm":
        if args.command_min is None:
            args.command_min = 0.0
        if args.command_max is None:
            args.command_max = 1.0
        if args.command_start is None:
            args.command_start = 0.5
    else:
        if args.command_min is None:
            args.command_min = 0.0
        if args.command_max is None:
            args.command_max = 4095.0
        if args.command_start is None:
            args.command_start = 2048.0

    if float(args.command_max) <= float(args.command_min):
        raise ValueError("Command range is invalid: max must be greater than min.")
    if not (float(args.command_min) <= float(args.command_start) <= float(args.command_max)):
        raise ValueError("Command start must lie inside the configured command range.")


def apply_closed_loop_command_window(args: argparse.Namespace) -> None:
    if str(args.mode) != "closed_loop" or args.closed_loop_command_window is None:
        return

    window = float(args.closed_loop_command_window)
    if not np.isfinite(window) or window <= 0.0:
        raise ValueError("--closed-loop-command-window must be positive when provided.")

    start = float(args.command_start)
    args.command_min = max(float(args.command_min), start - window)
    args.command_max = min(float(args.command_max), start + window)
    if float(args.command_max) <= float(args.command_min):
        raise ValueError(
            "--closed-loop-command-window leaves no valid command range after applying --cmd-min/--cmd-max."
        )
    if not (float(args.command_min) <= start <= float(args.command_max)):
        raise ValueError("Command start must lie inside the bounded closed-loop command range.")


def resolve_phase_hold_defaults(args: argparse.Namespace) -> None:
    if args.phase_slew_gain_hz_per_rad is None:
        args.phase_slew_gain_hz_per_rad = 0.0
    if args.phase_slew_max_freq_hz is None:
        args.phase_slew_max_freq_hz = max(
            float(args.phase_hold_max_freq_hz),
            0.0,
        )


def infer_requested_actuator_name(args: argparse.Namespace) -> str:
    if args.use_pwm:
        return "pwm"
    if args.use_mcp4725:
        return "mcp4725"
    return "mcp4725"


def build_actuator(args: argparse.Namespace) -> ActuatorBase:
    if args.mode == "open_loop":
        return NullActuator()

    actuator_name = infer_requested_actuator_name(args)
    if actuator_name == "pwm":
        return PwmActuator(
            gpio_pin=args.pwm_pin,
            pwm_frequency_hz=args.pwm_frequency_hz,
            pwm_vdd=args.pwm_vdd,
            command_min=args.command_min,
            command_max=args.command_max,
        )
    return Mcp4725Actuator(
        command_min=args.command_min,
        command_max=args.command_max,
        clock_apr_ppm=args.clock_apr_ppm,
        skip_clock_init=args.skip_clock_init,
        enable_dither=args.mcp4725_dither,
    )


def build_output_csv_path(args: argparse.Namespace) -> str:
    if args.out_csv:
        path = args.out_csv
    else:
        path = os.path.join(
            "first_stage_demo",
            "logs",
            f"{timestamp_for_filename()}_{args.mode}.csv",
        )
    ensure_parent_dir(path)
    return os.path.abspath(path)


def build_output_log_path(args: argparse.Namespace) -> tuple[str, str]:
    if getattr(args, "out_bin", None):
        path = str(args.out_bin)
        ensure_parent_dir(path)
        return os.path.abspath(path), "binary"
    return build_output_csv_path(args), "csv"


def csv_fieldnames() -> List[str]:
    return [
        "wall_rel_s",
        "t_rel_s",
        "block_index",
        "mode",
        "sdr_backend",
        "actuator",
        "sample_rate_hz",
        "block_size",
        "block_dt_s",
        "freq_unambiguous_hz",
        "center_freq_hz",
        "tone_rf_hz",
        "tone_bb_ref_hz",
        "tone_reference_source",
        "phase_rad",
        "unwrapped_phase_rad",
        "dphi_rad",
        "f_off_hz",
        "amp",
        "coherence",
        "snr_db",
        "power",
        "z_re",
        "z_im",
        "z_abs",
        "valid_phase_step",
        "update_allowed",
        "gate_reason",
        "phase_hold_enabled",
        "phase_hold_active",
        "phase_hold_arm_counter",
        "phase_hold_exit_counter",
        "phase_hold_arm_threshold_hz",
        "phase_hold_arm_dwell_updates",
        "phase_hold_exit_dwell_updates",
        "phase_control_mode",
        "phase_target_rad",
        "phase_error_rad",
        "phase_error_raw_rad",
        "phase_error_filtered_rad",
        "phase_error_update_allowed",
        "phase_error_filter_alpha",
        "phase_error_min_confidence",
        "phase_error_max_residual_rad",
        "phase_cycle_tracking_enabled",
        "phase_error_cycle_rad",
        "phase_error_predicted_rad",
        "phase_error_branch_index",
        "phase_error_branch_residual_rad",
        "phase_cycle_update_allowed",
        "phase_cycle_slip_detected",
        "phase_cycle_slip_count",
        "phase_cycle_max_branch_error_rad",
        "phase_hold_gain_hz_per_rad",
        "phase_control_gain_hz_per_rad",
        "phase_hold_raw_freq_setpoint_hz",
        "phase_hold_freq_setpoint_hz",
        "phase_hold_cmd_offset",
        "phase_hold_max_freq_hz",
        "phase_control_max_freq_hz",
        "phase_hold_smoothing_alpha",
        "phase_integral_hz",
        "phase_slew_gain_hz_per_rad",
        "phase_slew_max_freq_hz",
        "phase_slew_enter_error_rad",
        "phase_slew_exit_error_rad",
        "phase_recovery_enter_dwell_updates",
        "phase_recovery_exit_dwell_updates",
        "phase_recovery_min_dwell_updates",
        "phase_recovery_enter_counter",
        "phase_recovery_enter_sign",
        "phase_recovery_exit_counter",
        "phase_recovery_mode_counter",
        "control_update_interval_s",
        "control_average_interval_s",
        "control_update_interval_blocks",
        "control_average_interval_blocks",
        "control_window_valid_blocks",
        "control_window_span_s",
        "control_phase_fit_r2",
        "control_meas_phase_rad",
        "control_meas_f_off_hz",
        "control_target_f_off_hz",
        "control_error_hz",
        "control_base_cmd",
        "clock_est_phase_rad",
        "clock_est_frequency_hz",
        "clock_est_drift_hz_per_s",
        "clock_est_phase_residual_rad",
        "clock_est_confidence",
        "clock_est_cycle_slip",
        "vcxo_hz_per_command",
        "control_stage",
        "control_stage_switch_event",
        "control_stage_switches",
        "pull_in_enter_counter",
        "pull_in_exit_counter",
        "pull_in_enabled",
        "pull_in_switch_hz",
        "pull_in_exit_hz",
        "pull_in_enter_dwell_updates",
        "pull_in_exit_dwell_updates",
        "recapture_enabled",
        "recapture_active",
        "recapture_reason",
        "recapture_bad_block_counter",
        "recapture_ambiguity_counter",
        "recapture_capture_block_count",
        "recapture_count",
        "recapture_ambiguity_hz",
        "recapture_search_span_hz",
        "recapture_search_center_hz",
        "recapture_release_hz",
        "recapture_estimated_tone_bb_hz",
        "recapture_estimated_error_hz",
        "recapture_estimated_peak_power_db",
        "recapture_estimated_peak_prominence_db",
        "recapture_estimate_valid",
        "control_p_term",
        "control_i_term",
        "control_cmd",
        "requested_cmd",
        "dac_code_cmd",
        "pwm_cmd",
        "dac_volts",
        "actuator_dither_enabled",
        "actuator_dither_error",
        "actuator_write_skipped",
        "command_min",
        "command_max",
        "command_start",
        "invert_control",
        "pll_frequency_proportional_gain",
        "pll_frequency_integral_gain",
        "acquisition_frequency_proportional_gain",
        "acquisition_frequency_integral_gain",
        "acquisition_frequency_deadband_hz",
        "acquisition_command_slew_rate_per_s",
        "frequency_error_deadband_hz",
        "command_slew_rate_per_s",
        "base_frequency_integral_gain",
        "base_frequency_deadband_hz",
        "base_command_slew_rate_per_s",
        "base_update_allowed",
        "base_cmd_delta",
        "base_error_raw_hz",
        "base_error_filt_hz",
        "base_lock_counter",
        "base_phase_hold_allowed",
        "base_min_coherence",
        "base_max_phase_residual_rad",
        "base_max_f_off_hz",
        "effective_frequency_error_hz",
        "unlimited_requested_cmd",
        "command_slew_limited",
        "servo_profile",
        "amp_threshold",
        "coherence_threshold",
        "status",
        "test_step_index",
        "test_step_cmd",
        "test_valid_for_fit",
        "precision_role",
        "precision_pair_index",
        "precision_valid_dwell",
        "precision_coarse_bb_hz",
        "precision_refined_bb_hz",
        "precision_fine_residual_hz",
        "precision_coarse_peak_power_db",
        "precision_fit_r2",
        "precision_wall_mid_rel_s",
        "precision_sample_mid_rel_s",
        "precision_center_cmd",
        "precision_delta_cmd",
        "precision_valid_block_count",
        "precision_dwell_block_count",
    ]


class BinaryRowWriter:
    """Streaming binary row writer for the same logical fields as the CSV log."""

    MAGIC = b"CWVXCOBIN1\n"
    STRING_FIELDS = {
        "mode",
            "sdr_backend",
        "actuator",
        "tone_reference_source",
        "gate_reason",
        "phase_control_mode",
        "control_stage",
        "control_stage_switch_event",
        "recapture_reason",
        "servo_profile",
        "status",
        "precision_role",
    }

    def __init__(self, file_obj, fieldnames: List[str]) -> None:
        self.file_obj = file_obj
        self.fieldnames = list(fieldnames)
        self.string_fields = [f for f in self.fieldnames if f in self.STRING_FIELDS]
        self.numeric_fields = [f for f in self.fieldnames if f not in self.STRING_FIELDS]
        self._numeric_struct = struct.Struct("<" + ("d" * len(self.numeric_fields)))
        self._record_len_struct = struct.Struct("<I")
        self._string_len_struct = struct.Struct("<H")
        self._string_cache: Dict[str, bytes] = {}
        header = {
            "format": "cw_vcxo_binary_log",
            "version": 1,
            "fieldnames": self.fieldnames,
            "numeric_fields": self.numeric_fields,
            "string_fields": self.string_fields,
            "numeric_dtype": "float64_le",
            "record_layout": "uint32 record_len, float64 numeric_fields, repeated uint16 byte_len + utf8 string",
        }
        header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.file_obj.write(self.MAGIC)
        self.file_obj.write(struct.pack("<I", len(header_bytes)))
        self.file_obj.write(header_bytes)

    def writerow(self, row: Dict[str, Any]) -> None:
        numeric_values = []
        row_get = row.get
        for field in self.numeric_fields:
            value = row_get(field, float("nan"))
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                numeric_values.append(float("nan"))

        parts = [self._numeric_struct.pack(*numeric_values)]
        for field in self.string_fields:
            raw = row_get(field, "")
            text = "" if raw is None else str(raw)
            encoded = self._string_cache.get(text)
            if encoded is None:
                encoded = text.encode("utf-8")
                if len(encoded) > 65535:
                    encoded = encoded[:65535]
                self._string_cache[text] = encoded
            if len(encoded) > 65535:
                encoded = encoded[:65535]
            parts.append(self._string_len_struct.pack(len(encoded)))
            parts.append(encoded)

        payload = b"".join(parts)
        self.file_obj.write(self._record_len_struct.pack(len(payload)))
        self.file_obj.write(payload)


def write_csv_row(
    writer: Any,
    args: argparse.Namespace,
    tone_ref: ToneReference,
    sdr_backend: str,
    actuator: ActuatorBase,
    block_dt_s: float,
    freq_unambiguous_hz: float,
    block_index: int,
    t_rel_s: float,
    wall_rel_s: float,
    meas: BlockMeasurement,
    update_allowed: bool,
    gate_reason: str,
    control: Optional[ControlUpdate],
    applied_command: Optional[float],
    status: str,
    test_step_index: Optional[int] = None,
    test_step_cmd: Optional[float] = None,
    test_valid_for_fit: Optional[bool] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    command_value = float_or_nan(applied_command)
    row = {
        "wall_rel_s": wall_rel_s,
        "t_rel_s": t_rel_s,
        "block_index": block_index,
        "mode": args.mode,
        "sdr_backend": sdr_backend,
        "actuator": actuator.actuator_name,
        "sample_rate_hz": args.sample_rate_hz,
        "block_size": args.block_size,
        "block_dt_s": block_dt_s,
        "freq_unambiguous_hz": freq_unambiguous_hz,
        "center_freq_hz": args.center_freq_hz,
        "tone_rf_hz": float_or_nan(tone_ref.tone_rf_hz),
        "tone_bb_ref_hz": tone_ref.tone_bb_hz,
        "tone_reference_source": tone_ref.source,
        "phase_rad": meas.phase_rad,
        "unwrapped_phase_rad": meas.unwrapped_phase_rad,
        "dphi_rad": meas.dphi_rad,
        "f_off_hz": meas.f_off_hz,
        "amp": meas.amp,
        "coherence": meas.coherence,
        "snr_db": meas.snr_db,
        "power": meas.power,
        "z_re": meas.z_re,
        "z_im": meas.z_im,
        "z_abs": meas.z_abs,
        "valid_phase_step": int(bool(meas.valid_phase_step)),
        "update_allowed": int(bool(update_allowed)),
        "gate_reason": gate_reason,
        "phase_hold_enabled": int(bool(float(args.phase_hold_gain_hz_per_rad) > 0.0)),
        "phase_hold_active": 0,
        "phase_hold_arm_counter": float("nan"),
        "phase_hold_exit_counter": float("nan"),
        "phase_hold_arm_threshold_hz": float("nan"),
        "phase_hold_arm_dwell_updates": float("nan"),
        "phase_hold_exit_dwell_updates": float("nan"),
        "phase_control_mode": "",
        "phase_target_rad": float("nan"),
        "phase_error_rad": float("nan"),
        "phase_error_raw_rad": float("nan"),
        "phase_error_filtered_rad": float("nan"),
        "phase_error_update_allowed": float("nan"),
        "phase_error_filter_alpha": float("nan"),
        "phase_error_min_confidence": float("nan"),
        "phase_error_max_residual_rad": float("nan"),
        "phase_cycle_tracking_enabled": float("nan"),
        "phase_error_cycle_rad": float("nan"),
        "phase_error_predicted_rad": float("nan"),
        "phase_error_branch_index": float("nan"),
        "phase_error_branch_residual_rad": float("nan"),
        "phase_cycle_update_allowed": float("nan"),
        "phase_cycle_slip_detected": float("nan"),
        "phase_cycle_slip_count": float("nan"),
        "phase_cycle_max_branch_error_rad": float("nan"),
        "phase_hold_gain_hz_per_rad": float(args.phase_hold_gain_hz_per_rad),
        "phase_control_gain_hz_per_rad": float("nan"),
        "phase_hold_raw_freq_setpoint_hz": float("nan"),
        "phase_hold_freq_setpoint_hz": float("nan"),
        "phase_hold_cmd_offset": float("nan"),
        "phase_hold_max_freq_hz": float("nan"),
        "phase_control_max_freq_hz": float("nan"),
        "phase_hold_smoothing_alpha": float("nan"),
        "phase_integral_hz": float("nan"),
        "phase_slew_gain_hz_per_rad": float("nan"),
        "phase_slew_max_freq_hz": float("nan"),
        "phase_slew_enter_error_rad": float("nan"),
        "phase_slew_exit_error_rad": float("nan"),
        "phase_recovery_enter_dwell_updates": float("nan"),
        "phase_recovery_exit_dwell_updates": float("nan"),
        "phase_recovery_min_dwell_updates": float("nan"),
        "phase_recovery_enter_counter": float("nan"),
        "phase_recovery_enter_sign": float("nan"),
        "phase_recovery_exit_counter": float("nan"),
        "phase_recovery_mode_counter": float("nan"),
        "control_update_interval_s": float(args.control_update_interval_s),
        "control_average_interval_s": float(args.control_average_interval_s),
        "control_update_interval_blocks": float("nan"),
        "control_average_interval_blocks": float("nan"),
        "control_window_valid_blocks": float("nan"),
        "control_window_span_s": float("nan"),
        "control_phase_fit_r2": float("nan"),
        "control_meas_phase_rad": float("nan"),
        "control_meas_f_off_hz": float("nan"),
        "control_target_f_off_hz": float("nan"),
        "control_error_hz": float_or_nan(control.error_hz if control else None),
        "control_base_cmd": float("nan"),
        "clock_est_phase_rad": float("nan"),
        "clock_est_frequency_hz": float("nan"),
        "clock_est_drift_hz_per_s": float("nan"),
        "clock_est_phase_residual_rad": float("nan"),
        "clock_est_confidence": float("nan"),
        "clock_est_cycle_slip": float("nan"),
        "vcxo_hz_per_command": float_or_nan(getattr(args, "vcxo_hz_per_command", None)),
        "control_stage": "",
        "control_stage_switch_event": "",
        "control_stage_switches": float("nan"),
        "pull_in_enter_counter": float("nan"),
        "pull_in_exit_counter": float("nan"),
        "pull_in_enabled": float("nan"),
        "pull_in_switch_hz": float("nan"),
        "pull_in_exit_hz": float("nan"),
        "pull_in_enter_dwell_updates": float("nan"),
        "pull_in_exit_dwell_updates": float("nan"),
        "recapture_enabled": float("nan"),
        "recapture_active": float("nan"),
        "recapture_reason": "",
        "recapture_bad_block_counter": float("nan"),
        "recapture_ambiguity_counter": float("nan"),
        "recapture_capture_block_count": float("nan"),
        "recapture_count": float("nan"),
        "recapture_ambiguity_hz": float("nan"),
        "recapture_search_span_hz": float("nan"),
        "recapture_search_center_hz": float("nan"),
        "recapture_release_hz": float("nan"),
        "recapture_estimated_tone_bb_hz": float("nan"),
        "recapture_estimated_error_hz": float("nan"),
        "recapture_estimated_peak_power_db": float("nan"),
        "recapture_estimated_peak_prominence_db": float("nan"),
        "recapture_estimate_valid": float("nan"),
        "control_p_term": float_or_nan(control.p_term if control else None),
        "control_i_term": float_or_nan(control.i_term if control else None),
        "control_cmd": command_value,
        "requested_cmd": float_or_nan(control.requested_command if control else None),
        "dac_code_cmd": command_value if actuator.actuator_name == "mcp4725" else float("nan"),
        "pwm_cmd": command_value if actuator.actuator_name == "pwm" else float("nan"),
        "dac_volts": float_or_nan(actuator.command_to_volts(command_value) if np.isfinite(command_value) else None),
        "actuator_dither_enabled": 0.0,
        "actuator_dither_error": float("nan"),
        "actuator_write_skipped": float("nan"),
        "command_min": args.command_min,
        "command_max": args.command_max,
        "command_start": args.command_start,
        "invert_control": int(bool(args.invert_control)),
        "pll_frequency_proportional_gain": args.pll_frequency_proportional_gain,
        "pll_frequency_integral_gain": args.pll_frequency_integral_gain,
        "acquisition_frequency_proportional_gain": args.acquisition_frequency_proportional_gain,
        "acquisition_frequency_integral_gain": args.acquisition_frequency_integral_gain,
        "acquisition_frequency_deadband_hz": args.acquisition_frequency_deadband_hz,
        "acquisition_command_slew_rate_per_s": args.acquisition_command_slew_rate_per_s,
        "frequency_error_deadband_hz": args.frequency_error_deadband_hz,
        "command_slew_rate_per_s": args.command_slew_rate_per_s,
        "base_frequency_integral_gain": args.base_frequency_integral_gain,
        "base_frequency_deadband_hz": args.base_frequency_deadband_hz,
        "base_command_slew_rate_per_s": args.base_command_slew_rate_per_s,
        "base_update_allowed": float_or_nan(control.base_update_allowed if control else None),
        "base_cmd_delta": float_or_nan(control.base_delta if control else None),
        "base_error_raw_hz": float("nan"),
        "base_error_filt_hz": float("nan"),
        "base_lock_counter": float("nan"),
        "base_phase_hold_allowed": float("nan"),
        "base_min_coherence": args.base_min_coherence,
        "base_max_phase_residual_rad": args.base_max_phase_residual_rad,
        "base_max_f_off_hz": float("nan"),
        "effective_frequency_error_hz": float("nan"),
        "unlimited_requested_cmd": float("nan"),
        "command_slew_limited": float("nan"),
        "servo_profile": "",
        "amp_threshold": args.amp_threshold,
        "coherence_threshold": args.coherence_threshold,
        "status": status,
        "test_step_index": float_or_nan(test_step_index),
        "test_step_cmd": float_or_nan(test_step_cmd),
        "test_valid_for_fit": float_or_nan(1.0 if test_valid_for_fit else 0.0 if test_valid_for_fit is not None else None),
        "precision_role": "",
        "precision_pair_index": float("nan"),
        "precision_valid_dwell": float("nan"),
        "precision_coarse_bb_hz": float("nan"),
        "precision_refined_bb_hz": float("nan"),
        "precision_fine_residual_hz": float("nan"),
        "precision_coarse_peak_power_db": float("nan"),
        "precision_fit_r2": float("nan"),
        "precision_wall_mid_rel_s": float("nan"),
        "precision_sample_mid_rel_s": float("nan"),
        "precision_center_cmd": float("nan"),
        "precision_delta_cmd": float("nan"),
        "precision_valid_block_count": float("nan"),
        "precision_dwell_block_count": float("nan"),
    }
    row.update(actuator.csv_status())
    if extra_fields:
        row.update(extra_fields)
    writer.writerow(row)


def should_allow_update(
    meas: BlockMeasurement,
    args: argparse.Namespace,
    max_abs_f_off_hz: Optional[float] = None,
) -> (bool, str):
    if not meas.valid_phase_step or not np.isfinite(meas.f_off_hz):
        return False, "no_prev_phasor"
    if max_abs_f_off_hz is not None and abs(float(meas.f_off_hz)) >= float(max_abs_f_off_hz):
        return False, "f_off_near_ambiguity"
    if meas.amp < float(args.amp_threshold):
        return False, "amp_below_threshold"
    if meas.coherence < float(args.coherence_threshold):
        return False, "coherence_below_threshold"
    return True, "update_ok"


def compute_phase_hold_terms(
    phase_rad: float,
    phase_target_rad: Optional[float],
    phase_hold_gain_hz_per_rad: float,
) -> tuple[float, float]:
    if phase_target_rad is None or float(phase_hold_gain_hz_per_rad) <= 0.0:
        return float("nan"), 0.0
    phase_error_rad = wrap_to_pi(float(phase_rad) - float(phase_target_rad))
    freq_setpoint_hz = -float(phase_hold_gain_hz_per_rad) * phase_error_rad
    return float(phase_error_rad), float(freq_setpoint_hz)


def summarize_control_window(
    samples: Iterable[tuple[float, BlockMeasurement]],
) -> Optional[ControlMeasurementSummary]:
    items = list(samples)
    if len(items) < 2:
        return None

    times_s = np.asarray([item[0] for item in items], dtype=np.float64)
    phases_rad = np.asarray([item[1].unwrapped_phase_rad for item in items], dtype=np.float64)
    amps = np.asarray([item[1].amp for item in items], dtype=np.float64)
    coherences = np.asarray([item[1].coherence for item in items], dtype=np.float64)
    weights = np.asarray(
        [max(item[1].amp * max(item[1].coherence, 1.0e-3), 1.0e-6) for item in items],
        dtype=np.float64,
    )

    t_centered = times_s - float(np.mean(times_s))
    X = np.column_stack(
        [
            np.ones(times_s.size, dtype=np.float64),
            t_centered,
        ]
    )
    beta, _, _, _, fit_r2, rank = weighted_least_squares(X, phases_rad, weights)
    if rank < 2:
        return None

    t_last_centered = float(t_centered[-1])
    phase_last_rad = float(beta[0] + (beta[1] * t_last_centered))
    f_off_hz = float(beta[1] / (2.0 * math.pi))
    return ControlMeasurementSummary(
        t_rel_s=float(times_s[-1]),
        phase_rad=wrap_to_pi(phase_last_rad),
        f_off_hz=f_off_hz,
        amp=float(np.mean(amps)),
        coherence=float(np.mean(coherences)),
        fit_r2=float(fit_r2),
        valid_count=int(times_s.size),
        window_span_s=float(max(0.0, times_s[-1] - times_s[0])),
    )


def current_block_control_summary(
    t_rel_s: float,
    block_dt_s: float,
    meas: BlockMeasurement,
    estimate: ClockEstimate,
) -> ControlMeasurementSummary:
    f_off_hz = float(meas.f_off_hz)
    if not np.isfinite(f_off_hz):
        f_off_hz = float(estimate.frequency_hz)
    phase_rad = float(estimate.phase_rad)
    if not np.isfinite(phase_rad):
        phase_rad = float(meas.phase_rad)
    return ControlMeasurementSummary(
        t_rel_s=float(t_rel_s),
        phase_rad=wrap_to_pi(phase_rad),
        f_off_hz=float(f_off_hz),
        amp=float(meas.amp),
        coherence=float(meas.coherence),
        fit_r2=float("nan"),
        valid_count=1,
        window_span_s=float(block_dt_s),
    )


def status_line(
    t_rel_s: float,
    meas: BlockMeasurement,
    actuator: ActuatorBase,
    command: Optional[float],
    status: str,
) -> str:
    volts = actuator.command_to_volts(command) if command is not None and np.isfinite(command) else None
    return (
        f"t={format_float(t_rel_s, width=7, precision=1)}s | "
        f"f_off={format_float(meas.f_off_hz, width=9, precision=3)} Hz | "
        f"phase={format_float(meas.phase_rad, width=8, precision=3)} rad | "
        f"amp={format_float(meas.amp, width=7, precision=4)} | "
        f"coh={format_float(meas.coherence, width=6, precision=3)} | "
        f"act={status:<18} | "
        f"{actuator.command_label}={format_float(float_or_nan(command), width=8, precision=3)} | "
        f"V={format_float(float_or_nan(volts), width=6, precision=3)}"
    )


def infer_vcxo_hz_per_command(args: argparse.Namespace) -> float:
    if args.vcxo_hz_per_command is not None:
        return float(args.vcxo_hz_per_command)
    magnitude = 20.0 if infer_requested_actuator_name(args) == "mcp4725" else 1.0
    return float(-magnitude if args.invert_control else magnitude)


def blank_block_measurement() -> BlockMeasurement:
    return BlockMeasurement(
        phase_rad=float("nan"),
        unwrapped_phase_rad=float("nan"),
        dphi_rad=float("nan"),
        f_off_hz=float("nan"),
        amp=float("nan"),
        coherence=float("nan"),
        snr_db=float("nan"),
        power=float("nan"),
        z_re=float("nan"),
        z_im=float("nan"),
        z_abs=float("nan"),
        valid_phase_step=False,
    )


def maybe_rescan_startup_tone(
    args: argparse.Namespace,
    tone_ref: ToneReference,
    sdr: SDRBase,
) -> ToneReference:
    if not bool(args.startup_tone_check_enabled):
        return tone_ref

    check_blocks = max(2, int(args.startup_check_blocks))
    blocks: List[np.ndarray] = []
    tracker = ToneTracker(args.sample_rate_hz, args.block_size, tone_ref.tone_bb_hz)
    measurements: List[BlockMeasurement] = []

    for _ in range(max(0, int(args.warmup_blocks))):
        sdr.read_block(args.block_size)

    for _ in range(check_blocks):
        iq = sdr.read_block(args.block_size)
        blocks.append(iq)
        measurements.append(tracker.process_block(iq))

    coherences = np.asarray([m.coherence for m in measurements], dtype=np.float64)
    amps = np.asarray([m.amp for m in measurements], dtype=np.float64)
    mean_coherence = float(np.nanmean(coherences)) if coherences.size else float("nan")
    mean_amp = float(np.nanmean(amps)) if amps.size else float("nan")
    print(
        f"Startup tone check   : tone_bb={tone_ref.tone_bb_hz:.6f} Hz, "
        f"mean_coh={mean_coherence:.3f}, mean_amp={mean_amp:.4f}"
    )

    if (
        np.isfinite(mean_coherence)
        and mean_coherence >= float(args.startup_min_coherence)
        and np.isfinite(mean_amp)
        and mean_amp >= float(args.amp_threshold)
    ):
        return tone_ref

    coarse_samples = first_n_samples_from_blocks(
        blocks,
        min(
            int(args.startup_scan_fft_samples),
            max(1, check_blocks * int(args.block_size)),
        ),
    )
    estimate = estimate_coarse_tone_hz_from_samples(
        coarse_samples,
        sample_rate_hz=args.sample_rate_hz,
        search_center_hz=tone_ref.tone_bb_hz,
        search_span_hz=args.startup_scan_span_hz,
    )
    estimate_valid = bool(
        estimate.valid
        and np.isfinite(estimate.peak_prominence_db)
        and estimate.peak_prominence_db >= float(args.startup_min_prominence_db)
    )
    print(
        f"Startup tone scan    : bb={estimate.tone_hz:.6f} Hz, "
        f"prom={estimate.peak_prominence_db:.3f} dB, valid={int(estimate_valid)}"
    )
    if not estimate_valid:
        if bool(args.startup_require_tone):
            raise RuntimeError(
                "Startup tone check failed and coarse scan did not find a trusted tone."
            )
        return tone_ref

    corrected = ToneReference(
        tone_bb_hz=float(estimate.tone_hz),
        tone_rf_hz=float(args.center_freq_hz + estimate.tone_hz),
        source="startup_coarse_scan",
    )
    print(
        f"Startup tone update  : tone_bb {tone_ref.tone_bb_hz:.6f} -> "
        f"{corrected.tone_bb_hz:.6f} Hz"
    )
    return corrected


def apply_initial_frequency_step(
    args: argparse.Namespace,
    tone_ref: ToneReference,
    sdr: SDRBase,
    actuator: ActuatorBase,
    current_command: float,
    max_abs_f_off_hz: float,
) -> float:
    if (
        not bool(args.initial_frequency_step_enabled)
        or args.mode != "closed_loop"
        or actuator.actuator_name == "none"
    ):
        return float(current_command)

    def measure_median_f_off() -> tuple[float, int]:
        tracker = ToneTracker(args.sample_rate_hz, args.block_size, tone_ref.tone_bb_hz)
        valid_f_offsets: List[float] = []
        blocks = max(2, int(args.initial_frequency_step_blocks))
        for _ in range(blocks):
            meas = tracker.process_block(sdr.read_block(args.block_size))
            if (
                meas.valid_phase_step
                and np.isfinite(meas.f_off_hz)
                and abs(float(meas.f_off_hz)) < float(max_abs_f_off_hz)
                and float(meas.amp) >= float(args.amp_threshold)
                and float(meas.coherence) >= float(args.initial_frequency_step_min_coherence)
            ):
                valid_f_offsets.append(float(meas.f_off_hz))
        if not valid_f_offsets:
            return float("nan"), 0
        return float(np.median(np.asarray(valid_f_offsets, dtype=np.float64))), len(valid_f_offsets)

    applied_command = float(current_command)
    max_valid = max(1, int(args.initial_frequency_step_min_valid_blocks))
    passes = max(1, int(args.initial_frequency_step_passes))
    for step_index in range(passes):
        f_error_hz, valid_count = measure_median_f_off()
        if valid_count < max_valid or not np.isfinite(f_error_hz):
            print(
                "Initial frequency step: skipped "
                f"(pass={step_index + 1}, valid_blocks={valid_count}/{max_valid})"
            )
            break
        if abs(f_error_hz) <= float(args.initial_frequency_step_deadband_hz):
            print(
                "Initial frequency step: settled "
                f"(pass={step_index + 1}, median_f_off={f_error_hz:.3f} Hz)"
            )
            break

        gain = float(args.initial_frequency_step_gain)
        if step_index > 0:
            gain *= float(args.initial_frequency_step_refine_gain)
        command_delta = -gain * f_error_hz / float(args.vcxo_hz_per_command)
        max_delta = float(args.initial_frequency_step_max_command_delta)
        if max_delta > 0.0:
            command_delta = clamp(command_delta, -max_delta, max_delta)
        requested_command = clamp(
            applied_command + command_delta,
            float(args.command_min),
            float(args.command_max),
        )
        previous_command = applied_command
        applied_command = actuator.write_command(requested_command, settle_s=args.control_settle_s)
        print(
            "Initial frequency step: "
            f"pass={step_index + 1}/{passes}, "
            f"median_f_off={f_error_hz:.3f} Hz, "
            f"valid_blocks={valid_count}/{max(2, int(args.initial_frequency_step_blocks))}, "
            f"gain={gain:.3f}, "
            f"cmd {previous_command:.3f}->{float(applied_command):.3f}"
        )

        discard_blocks = max(0, int(args.initial_frequency_step_discard_blocks))
        for _ in range(discard_blocks):
            sdr.read_block(args.block_size)
    return float(applied_command)


def run_phase_discipline_loop(
    args: argparse.Namespace,
    tone_ref: ToneReference,
    sdr: SDRBase,
    actuator: ActuatorBase,
    writer: csv.DictWriter,
    csv_file,
    stop: GracefulStop,
) -> None:
    tone_ref = maybe_rescan_startup_tone(args, tone_ref, sdr)
    detector = TonePhaseDetector(args.sample_rate_hz, args.block_size, tone_ref.tone_bb_hz)
    block_dt_s = detector.block_dt_s
    profiler = LoopProfiler(args.profile_loop, block_dt_s)
    profile_loop = bool(args.profile_loop)
    freq_unambiguous_hz = 0.5 / block_dt_s
    control_update_every_blocks = max(
        1, int(round(float(args.control_update_interval_s) / max(block_dt_s, 1.0e-9)))
    )
    requested_control_average_blocks = max(
        1, int(round(float(args.control_average_interval_s) / max(block_dt_s, 1.0e-9)))
    )
    control_average_blocks = min(control_update_every_blocks, requested_control_average_blocks)
    control_min_valid_blocks = min(
        max(1, int(args.control_min_valid_blocks)),
        control_average_blocks,
    )
    control_window_max_age_s = max(control_average_blocks * block_dt_s, block_dt_s) * 1.5
    recapture_ambiguity_hz = float(args.recapture_ambiguity_fraction) * float(freq_unambiguous_hz)
    control_max_abs_f_off_hz = float(args.control_max_f_off_fraction) * float(freq_unambiguous_hz)
    base_max_abs_f_off_hz = float(args.base_max_f_off_fraction) * float(freq_unambiguous_hz)
    estimator = AlphaBetaClockStateEstimator(
        block_dt_s=block_dt_s,
        alpha=args.pll_estimator_alpha,
        beta=args.pll_estimator_beta,
        gamma=args.pll_estimator_gamma,
        cycle_slip_threshold_rad=args.pll_cycle_slip_threshold_rad,
    )
    servo = VcxoActuatorModel(
        hz_per_command=args.vcxo_hz_per_command,
        command_min=args.command_min,
        command_max=args.command_max,
        initial_command=args.command_start,
        proportional_gain=args.pll_frequency_proportional_gain,
        integral_gain=args.pll_frequency_integral_gain,
        frequency_deadband_hz=args.frequency_error_deadband_hz,
        command_slew_rate_per_s=args.command_slew_rate_per_s,
    )
    phase_controller = PhaseDisciplineController(
        phase_target_rad=args.phase_target_rad,
        target_is_explicit=args.phase_target_rad is not None,
        arm_threshold_hz=args.phase_hold_arm_threshold_hz,
        arm_dwell_updates=args.phase_hold_arm_dwell_updates,
        hold_exit_dwell_updates=args.phase_hold_exit_dwell_updates,
        pull_in_enabled=args.pull_in_enabled,
        switch_hz=args.pull_in_switch_hz,
        exit_hz=args.pull_in_exit_hz,
        enter_dwell_updates=args.pull_in_enter_dwell_updates,
        exit_dwell_updates=args.pull_in_exit_dwell_updates,
        hold_gain_hz_per_rad=args.phase_hold_gain_hz_per_rad,
        hold_max_freq_hz=args.phase_hold_max_freq_hz,
        slew_gain_hz_per_rad=args.phase_slew_gain_hz_per_rad,
        slew_max_freq_hz=args.phase_slew_max_freq_hz,
        slew_enter_error_rad=args.phase_slew_enter_error_rad,
        slew_exit_error_rad=args.phase_slew_exit_error_rad,
        recovery_enter_dwell_updates=args.phase_recovery_enter_dwell_updates,
        recovery_exit_dwell_updates=args.phase_recovery_exit_dwell_updates,
        recovery_min_dwell_updates=args.phase_recovery_min_dwell_updates,
        smoothing_alpha=args.phase_hold_smoothing_alpha,
        target_slew_rate_rad_per_s=args.phase_target_slew_rate_rad_per_s,
        phase_integral_gain_hz_per_rad_s=args.phase_integral_gain_hz_per_rad_s,
        phase_integral_max_hz=args.phase_integral_max_hz,
        phase_error_filter_alpha=args.phase_error_filter_alpha,
        phase_error_min_confidence=args.phase_error_min_confidence,
        phase_error_max_residual_rad=args.phase_error_max_residual_rad,
        phase_cycle_tracking_enabled=args.phase_cycle_tracking_enabled,
        phase_cycle_max_branch_error_rad=args.phase_cycle_max_branch_error_rad,
        update_interval_s=args.control_update_interval_s,
    )
    recapture_manager = FrequencyRecaptureManager(
        enabled=bool(args.recapture_enabled),
        bad_block_threshold=int(args.recapture_invalid_blocks),
        ambiguity_hz=recapture_ambiguity_hz,
        ambiguity_block_threshold=int(args.recapture_ambiguity_blocks),
        release_hz=float(args.recapture_release_hz),
        search_center_hz=float(tone_ref.tone_bb_hz),
        trigger_on_ambiguity=bool(args.recapture_on_ambiguity),
    )
    recapture_blocks: deque[np.ndarray] = deque(maxlen=max(1, int(args.recapture_dwell_blocks)))
    control_window: deque[tuple[float, BlockMeasurement]] = deque(maxlen=control_average_blocks)
    applied_command = actuator.write_command(args.command_start, settle_s=args.control_settle_s)
    applied_command = apply_initial_frequency_step(
        args=args,
        tone_ref=tone_ref,
        sdr=sdr,
        actuator=actuator,
        current_command=applied_command,
        max_abs_f_off_hz=control_max_abs_f_off_hz,
    )
    detector = TonePhaseDetector(args.sample_rate_hz, args.block_size, tone_ref.tone_bb_hz)
    servo.seed_from_command(applied_command)
    last_control_update_t_rel_s: Optional[float] = None
    initial_stage = "pull_in" if args.pull_in_enabled else "fine_lock"
    last_phase_command = PhaseDisciplineCommand(
        stage=initial_stage,
        switch_event="",
        phase_mode="idle",
        target_phase_rad=float_or_nan(args.phase_target_rad),
        phase_error_rad=float("nan"),
        phase_error_raw_rad=float("nan"),
        phase_error_filtered_rad=float("nan"),
        phase_error_update_allowed=False,
        phase_error_cycle_rad=float("nan"),
        phase_error_predicted_rad=float("nan"),
        phase_error_branch_index=float("nan"),
        phase_error_branch_residual_rad=float("nan"),
        phase_cycle_update_allowed=False,
        phase_cycle_slip_detected=False,
        phase_cycle_slip_count=0,
        target_f_off_hz=0.0,
        raw_target_f_off_hz=0.0,
        frequency_error_hz=float("nan"),
        phase_gain_hz_per_rad=0.0,
        max_target_f_off_hz=0.0,
        phase_active=False,
        just_armed=False,
    )
    last_estimate = ClockEstimate(
        phase_rad=float("nan"),
        unwrapped_phase_rad=float("nan"),
        frequency_hz=float("nan"),
        drift_hz_per_s=float("nan"),
        phase_residual_rad=float("nan"),
        confidence=float("nan"),
        valid=False,
        cycle_slip=False,
    )
    recapture_state = recapture_manager.snapshot()
    base_error_filt_hz: Optional[float] = None
    base_centered_counter = 0
    last_base_error_raw_hz = float("nan")
    last_base_error_filt_hz = float("nan")
    last_base_phase_hold_allowed = True

    if not bool(args.startup_tone_check_enabled):
        sdr.warmup(args.block_size, args.warmup_blocks)
    wall_start = time.monotonic()
    next_status_wall = wall_start
    flush_every = max(1, int(round(float(args.log_flush_interval_s) / block_dt_s)))
    total_blocks = int(math.ceil(args.duration_s / block_dt_s))

    for block_index in range(total_blocks):
        if stop.stop_requested:
            print("Stop requested, ending run.")
            break
        loop_t0 = time.perf_counter() if profile_loop else 0.0
        iq = sdr.read_block(args.block_size)
        read_done = time.perf_counter() if profile_loop else 0.0
        wall_now = time.monotonic()
        wall_rel_s = wall_now - wall_start
        t_rel_s = (block_index + 0.5) * block_dt_s
        detect_t0 = time.perf_counter() if profile_loop else 0.0
        meas = detector.process_block(iq)
        detect_done = time.perf_counter() if profile_loop else 0.0
        control_t0 = detect_done
        actuator_write_s = 0.0
        update_allowed, gate_reason = should_allow_update(
            meas,
            args,
            max_abs_f_off_hz=control_max_abs_f_off_hz,
        )
        estimate = estimator.update(meas, update_allowed)
        last_estimate = estimate
        if estimate.cycle_slip and update_allowed:
            update_allowed = False
            gate_reason = "cycle_slip"
        if (
            update_allowed
            and np.isfinite(estimate.phase_residual_rad)
            and abs(float(estimate.phase_residual_rad)) > float(args.control_max_phase_residual_rad)
        ):
            update_allowed = False
            gate_reason = "phase_residual_too_large"

        stage_state = phase_controller.stage_snapshot()
        control_update = ControlUpdate(
            applied_command=float(applied_command),
            requested_command=float(applied_command),
            p_term=float("nan"),
            i_term=float(servo.last_i_term),
            error_hz=float("nan"),
        )
        status = "observe"
        control_summary = None
        recapture_state = recapture_manager.snapshot()

        observed_recapture = recapture_manager.observe_block(gate_reason, meas)
        if observed_recapture.just_started:
            recapture_blocks.clear()
            control_window.clear()
            phase_controller.force_pull_in()
            estimator.reset(phase_rad=meas.phase_rad, frequency_hz=0.0)
            last_control_update_t_rel_s = None
        recapture_state = observed_recapture

        if recapture_state.active:
            recapture_blocks.append(iq)
            recapture_state = recapture_manager.note_capture_block()
            update_allowed = False
            gate_reason = "recapture_active"
            last_phase_command = PhaseDisciplineCommand(
                stage=phase_controller.stage_snapshot().stage,
                switch_event="",
                phase_mode="idle",
                target_phase_rad=float_or_nan(phase_controller.phase_target_rad),
                phase_error_rad=float("nan"),
                phase_error_raw_rad=float("nan"),
                phase_error_filtered_rad=float("nan"),
                phase_error_update_allowed=False,
                phase_error_cycle_rad=float("nan"),
                phase_error_predicted_rad=float("nan"),
                phase_error_branch_index=float("nan"),
                phase_error_branch_residual_rad=float("nan"),
                phase_cycle_update_allowed=False,
                phase_cycle_slip_detected=False,
                phase_cycle_slip_count=int(phase_controller.phase_cycle_slip_count),
                target_f_off_hz=0.0,
                raw_target_f_off_hz=0.0,
                frequency_error_hz=float("nan"),
                phase_gain_hz_per_rad=0.0,
                max_target_f_off_hz=0.0,
                phase_active=False,
                just_armed=False,
            )
            stage_state = phase_controller.stage_snapshot()
            if len(recapture_blocks) >= int(args.recapture_dwell_blocks):
                capture_block_count = len(recapture_blocks)
                capture_span_s = float(capture_block_count) * block_dt_s
                coarse_samples = first_n_samples_from_blocks(
                    list(recapture_blocks),
                    capture_block_count * int(args.block_size),
                )
                coarse_estimate = estimate_coarse_tone_hz_from_samples(
                    coarse_samples,
                    sample_rate_hz=args.sample_rate_hz,
                    search_center_hz=recapture_manager.search_center_hz,
                    search_span_hz=args.recapture_search_span_hz,
                )
                edge_margin_hz = 0.9 * float(args.recapture_search_span_hz)
                within_search_window = abs(
                    float(coarse_estimate.tone_hz) - float(recapture_manager.search_center_hz)
                ) <= edge_margin_hz
                estimate_valid = bool(
                    coarse_estimate.valid
                    and np.isfinite(coarse_estimate.peak_prominence_db)
                    and float(coarse_estimate.peak_prominence_db)
                    >= float(args.recapture_min_peak_prominence_db)
                    and within_search_window
                )
                recapture_state = recapture_manager.record_estimate(
                    tone_bb_hz=coarse_estimate.tone_hz,
                    tone_bb_ref_hz=tone_ref.tone_bb_hz,
                    peak_power_db=coarse_estimate.peak_power_db,
                    peak_prominence_db=coarse_estimate.peak_prominence_db,
                    estimate_valid=estimate_valid,
                )
                recapture_blocks.clear()
                coarse_error_hz = float(recapture_state.estimated_error_hz)
                if estimate_valid:
                    control_update = servo.update(
                        coarse_error_hz,
                        dt_s=max(block_dt_s, capture_span_s),
                        proportional_gain=args.acquisition_frequency_proportional_gain,
                        integral_gain=args.acquisition_frequency_integral_gain,
                        frequency_deadband_hz=args.acquisition_frequency_deadband_hz,
                        command_slew_rate_per_s=args.acquisition_command_slew_rate_per_s,
                        profile="acquisition",
                    )
                    actuator_t0 = time.perf_counter() if profile_loop else 0.0
                    applied_command = actuator.write_command(
                        control_update.requested_command,
                        settle_s=args.control_settle_s,
                    )
                    if profile_loop:
                        actuator_write_s += time.perf_counter() - actuator_t0
                    servo.seed_from_command(applied_command)
                    estimator.reset(phase_rad=meas.phase_rad, frequency_hz=coarse_error_hz)
                    control_update.applied_command = float(applied_command)
                if not estimate_valid:
                    status = "recapture_reject_peak"
                elif recapture_state.just_completed:
                    status = "recapture_complete"
                else:
                    status = "recapture_update"
            else:
                status = "recapture_collecting"
        else:
            while control_window and (t_rel_s - control_window[0][0]) > control_window_max_age_s:
                control_window.popleft()
            if update_allowed:
                control_window.append((t_rel_s, meas))

            should_update_now = ((block_index + 1) % control_update_every_blocks) == 0
            if should_update_now and update_allowed and len(control_window) >= control_min_valid_blocks:
                if control_average_blocks <= 1:
                    control_summary = current_block_control_summary(
                        t_rel_s=t_rel_s,
                        block_dt_s=block_dt_s,
                        meas=meas,
                        estimate=estimate,
                    )
                else:
                    control_summary = summarize_control_window(control_window)
                if control_summary is None:
                    status = f"{stage_state.stage}_hold_control_fit_failed"
                elif (
                    np.isfinite(control_summary.fit_r2)
                    and control_summary.fit_r2 < float(args.control_min_fit_r2)
                ):
                    status = f"{stage_state.stage}_hold_control_fit_r2"
                    update_allowed = False
                    gate_reason = "control_fit_r2_below_threshold"
                elif (
                    not np.isfinite(control_summary.f_off_hz)
                    or abs(float(control_summary.f_off_hz)) >= control_max_abs_f_off_hz
                ):
                    status = f"{stage_state.stage}_hold_control_f_off_near_ambiguity"
                    update_allowed = False
                    gate_reason = "control_f_off_near_ambiguity"
                else:
                    control_estimate = ClockEstimate(
                        phase_rad=float(control_summary.phase_rad),
                        unwrapped_phase_rad=float(estimate.unwrapped_phase_rad),
                        frequency_hz=float(control_summary.f_off_hz),
                        drift_hz_per_s=float(estimate.drift_hz_per_s),
                        phase_residual_rad=float(estimate.phase_residual_rad),
                        confidence=float(clamp(control_summary.coherence * control_summary.amp / 0.10, 0.0, 1.0)),
                        valid=True,
                        cycle_slip=False,
                    )
                    phase_hold_allowed_by_base = bool(
                        int(base_centered_counter)
                        >= int(args.phase_hold_base_lock_dwell_updates)
                    )
                    phase_command = phase_controller.update(
                        control_estimate,
                        update_valid=True,
                        phase_hold_allowed=phase_hold_allowed_by_base,
                    )
                    last_phase_command = phase_command
                    last_base_phase_hold_allowed = bool(phase_hold_allowed_by_base)
                    last_base_error_raw_hz = (
                        float(phase_command.frequency_error_hz)
                        if phase_command.phase_active
                        else float(control_summary.f_off_hz)
                    )
                    if np.isfinite(last_base_error_raw_hz):
                        alpha = clamp(float(args.base_error_filter_alpha), 0.0, 1.0)
                        if base_error_filt_hz is None or not np.isfinite(base_error_filt_hz):
                            base_error_filt_hz = last_base_error_raw_hz
                        else:
                            base_error_filt_hz = (
                                alpha * float(base_error_filt_hz)
                                + ((1.0 - alpha) * last_base_error_raw_hz)
                            )
                    last_base_error_filt_hz = (
                        float(base_error_filt_hz)
                        if base_error_filt_hz is not None and np.isfinite(base_error_filt_hz)
                        else float("nan")
                    )
                    if (
                        np.isfinite(last_base_error_filt_hz)
                        and abs(last_base_error_filt_hz)
                        <= float(args.phase_hold_base_lock_threshold_hz)
                    ):
                        base_centered_counter += 1
                    else:
                        base_centered_counter = 0
                    stage_state = phase_controller.stage_snapshot(phase_command.switch_event)
                    if phase_command.switch_event == "enter_fine_lock":
                        servo.seed_from_command(applied_command)
                    if last_control_update_t_rel_s is None:
                        control_dt_s = control_update_every_blocks * block_dt_s
                    else:
                        control_dt_s = max(block_dt_s, control_summary.t_rel_s - last_control_update_t_rel_s)
                    if phase_command.stage == "pull_in":
                        servo_profile = "acquisition"
                        p_gain = args.acquisition_frequency_proportional_gain
                        i_gain = args.acquisition_frequency_integral_gain
                        deadband_hz = args.acquisition_frequency_deadband_hz
                        slew_rate = args.acquisition_command_slew_rate_per_s
                    else:
                        servo_profile = "tracking"
                        p_gain = args.pll_frequency_proportional_gain
                        i_gain = args.pll_frequency_integral_gain
                        deadband_hz = args.frequency_error_deadband_hz
                        slew_rate = args.command_slew_rate_per_s
                    servo.last_base_delta = 0.0
                    servo.last_base_update_allowed = False
                    base_update_ok = bool(
                        phase_command.stage == "fine_lock"
                        and update_allowed
                        and np.isfinite(control_summary.f_off_hz)
                        and abs(float(control_summary.f_off_hz)) <= base_max_abs_f_off_hz
                        and np.isfinite(control_summary.coherence)
                        and float(control_summary.coherence) >= float(args.base_min_coherence)
                        and np.isfinite(estimate.phase_residual_rad)
                        and abs(float(estimate.phase_residual_rad))
                        <= float(args.base_max_phase_residual_rad)
                    )
                    base_frequency_error_hz = (
                        last_base_error_filt_hz
                        if np.isfinite(last_base_error_filt_hz)
                        else float(control_summary.f_off_hz)
                    )
                    if base_update_ok:
                        servo.adapt_base_command(
                            base_frequency_error_hz,
                            dt_s=control_dt_s,
                            gain_per_s=args.base_frequency_integral_gain,
                            deadband_hz=args.base_frequency_deadband_hz,
                            slew_rate_per_s=args.base_command_slew_rate_per_s,
                        )
                    control_update = servo.update(
                        phase_command.frequency_error_hz,
                        dt_s=control_dt_s,
                        proportional_gain=p_gain,
                        integral_gain=i_gain,
                        frequency_deadband_hz=deadband_hz,
                        command_slew_rate_per_s=slew_rate,
                        feedforward_command_offset=(
                            float(phase_command.target_f_off_hz) / float(args.vcxo_hz_per_command)
                            if phase_command.phase_active and np.isfinite(phase_command.target_f_off_hz)
                            else 0.0
                        ),
                        profile=servo_profile,
                    )
                    actuator_t0 = time.perf_counter() if profile_loop else 0.0
                    applied_command = actuator.write_command(
                        control_update.requested_command,
                        settle_s=args.control_settle_s,
                    )
                    if profile_loop:
                        actuator_write_s += time.perf_counter() - actuator_t0
                    control_update.applied_command = float(applied_command)
                    last_control_update_t_rel_s = float(control_summary.t_rel_s)
                    recapture_state = recapture_manager.update_search_center(
                        tone_ref.tone_bb_hz + float(control_summary.f_off_hz)
                    )
                    if phase_command.switch_event:
                        status = phase_command.switch_event
                    elif phase_command.just_armed:
                        status = "fine_lock_phase_hold_armed"
                    elif phase_command.stage == "fine_lock" and phase_command.phase_active:
                        status = f"fine_lock_{phase_command.phase_mode}_update"
                    else:
                        status = f"{phase_command.stage}_update"
            else:
                if not update_allowed:
                    status = f"{stage_state.stage}_hold_{gate_reason}"
                elif not should_update_now:
                    status = f"{stage_state.stage}_hold_between_updates"
                else:
                    status = f"{stage_state.stage}_hold_insufficient_window"

        control_done = time.perf_counter() if profile_loop else 0.0
        if control_summary is None:
            control_summary = ControlMeasurementSummary(
                t_rel_s=float(t_rel_s),
                phase_rad=float(estimate.phase_rad),
                f_off_hz=float(estimate.frequency_hz),
                amp=float(meas.amp),
                coherence=float(meas.coherence),
                fit_r2=float("nan"),
                valid_count=1 if estimate.valid else 0,
                window_span_s=block_dt_s,
            )
        phase_cmd = last_phase_command
        log_t0 = time.perf_counter() if profile_loop else 0.0
        write_csv_row(
            writer=writer,
            args=args,
            tone_ref=tone_ref,
            sdr_backend=sdr.backend_name,
            actuator=actuator,
            block_dt_s=block_dt_s,
            freq_unambiguous_hz=freq_unambiguous_hz,
            block_index=block_index,
            t_rel_s=t_rel_s,
            wall_rel_s=wall_rel_s,
            meas=meas,
            update_allowed=update_allowed,
            gate_reason=gate_reason,
            control=control_update,
            applied_command=applied_command,
            status=status,
            extra_fields={
                "control_update_interval_blocks": float(control_update_every_blocks),
                "control_average_interval_blocks": float(control_average_blocks),
                "control_window_valid_blocks": float(control_summary.valid_count),
                "control_window_span_s": float(control_summary.window_span_s),
                "control_phase_fit_r2": float(control_summary.fit_r2),
                "control_meas_phase_rad": float(control_summary.phase_rad),
                "control_meas_f_off_hz": float(control_summary.f_off_hz),
                "control_target_f_off_hz": float(phase_cmd.target_f_off_hz),
                "control_base_cmd": float(servo.base_command),
                "clock_est_phase_rad": float(estimate.phase_rad),
                "clock_est_frequency_hz": float(estimate.frequency_hz),
                "clock_est_drift_hz_per_s": float(estimate.drift_hz_per_s),
                "clock_est_phase_residual_rad": float(estimate.phase_residual_rad),
                "clock_est_confidence": float(estimate.confidence),
                "clock_est_cycle_slip": float(1.0 if estimate.cycle_slip else 0.0),
                "control_stage": stage_state.stage,
                "control_stage_switch_event": stage_state.switch_event,
                "control_stage_switches": float(stage_state.stage_switches),
                "pull_in_enter_counter": float(stage_state.enter_counter),
                "pull_in_exit_counter": float(stage_state.exit_counter),
                "pull_in_enabled": float(bool(args.pull_in_enabled)),
                "pull_in_switch_hz": float(args.pull_in_switch_hz),
                "pull_in_exit_hz": float(args.pull_in_exit_hz),
                "pull_in_enter_dwell_updates": float(args.pull_in_enter_dwell_updates),
                "pull_in_exit_dwell_updates": float(args.pull_in_exit_dwell_updates),
                "phase_hold_active": int(bool(phase_cmd.phase_active)),
                "phase_hold_arm_counter": float(phase_controller.arm_counter),
                "phase_hold_exit_counter": float(phase_controller.phase_hold_exit_counter),
                "phase_hold_arm_threshold_hz": float(args.phase_hold_arm_threshold_hz),
                "phase_hold_arm_dwell_updates": float(args.phase_hold_arm_dwell_updates),
                "phase_hold_exit_dwell_updates": float(args.phase_hold_exit_dwell_updates),
                "phase_control_mode": phase_cmd.phase_mode,
                "phase_target_rad": float_or_nan(phase_cmd.target_phase_rad),
                "phase_error_rad": float_or_nan(phase_cmd.phase_error_rad),
                "phase_error_raw_rad": float_or_nan(phase_cmd.phase_error_raw_rad),
                "phase_error_filtered_rad": float_or_nan(phase_cmd.phase_error_filtered_rad),
                "phase_error_update_allowed": float(1.0 if phase_cmd.phase_error_update_allowed else 0.0),
                "phase_error_filter_alpha": float(args.phase_error_filter_alpha),
                "phase_error_min_confidence": float(args.phase_error_min_confidence),
                "phase_error_max_residual_rad": float(args.phase_error_max_residual_rad),
                "phase_cycle_tracking_enabled": float(1.0 if args.phase_cycle_tracking_enabled else 0.0),
                "phase_error_cycle_rad": float_or_nan(phase_cmd.phase_error_cycle_rad),
                "phase_error_predicted_rad": float_or_nan(phase_cmd.phase_error_predicted_rad),
                "phase_error_branch_index": float_or_nan(phase_cmd.phase_error_branch_index),
                "phase_error_branch_residual_rad": float_or_nan(phase_cmd.phase_error_branch_residual_rad),
                "phase_cycle_update_allowed": float(1.0 if phase_cmd.phase_cycle_update_allowed else 0.0),
                "phase_cycle_slip_detected": float(1.0 if phase_cmd.phase_cycle_slip_detected else 0.0),
                "phase_cycle_slip_count": float(phase_cmd.phase_cycle_slip_count),
                "phase_cycle_max_branch_error_rad": float(args.phase_cycle_max_branch_error_rad),
                "phase_hold_raw_freq_setpoint_hz": float(phase_cmd.raw_target_f_off_hz),
                "phase_control_gain_hz_per_rad": float(phase_cmd.phase_gain_hz_per_rad),
                "phase_hold_freq_setpoint_hz": float(phase_cmd.target_f_off_hz),
                "phase_hold_cmd_offset": (
                    float(phase_cmd.target_f_off_hz) / float(args.vcxo_hz_per_command)
                    if phase_cmd.phase_active and np.isfinite(phase_cmd.target_f_off_hz)
                    else 0.0
                ),
                "phase_hold_max_freq_hz": float(args.phase_hold_max_freq_hz),
                "phase_control_max_freq_hz": float(phase_cmd.max_target_f_off_hz),
                "phase_hold_smoothing_alpha": float(args.phase_hold_smoothing_alpha),
                "phase_integral_hz": float(phase_controller.phase_integral_hz),
                "phase_slew_gain_hz_per_rad": float(args.phase_slew_gain_hz_per_rad),
                "phase_slew_max_freq_hz": float(args.phase_slew_max_freq_hz),
                "phase_slew_enter_error_rad": float(args.phase_slew_enter_error_rad),
                "phase_slew_exit_error_rad": float(args.phase_slew_exit_error_rad),
                "phase_recovery_enter_dwell_updates": float(args.phase_recovery_enter_dwell_updates),
                "phase_recovery_exit_dwell_updates": float(args.phase_recovery_exit_dwell_updates),
                "phase_recovery_min_dwell_updates": float(args.phase_recovery_min_dwell_updates),
                "phase_recovery_enter_counter": float(phase_controller.recovery_enter_counter),
                "phase_recovery_enter_sign": float(phase_controller.recovery_enter_sign),
                "phase_recovery_exit_counter": float(phase_controller.recovery_exit_counter),
                "phase_recovery_mode_counter": float(phase_controller.recovery_mode_counter),
                "recapture_enabled": float(bool(args.recapture_enabled)),
                "recapture_active": int(bool(recapture_state.active)),
                "recapture_reason": recapture_state.reason,
                "recapture_bad_block_counter": float(recapture_state.bad_block_counter),
                "recapture_ambiguity_counter": float(recapture_state.ambiguity_counter),
                "recapture_capture_block_count": float(recapture_state.capture_block_count),
                "recapture_count": float(recapture_state.capture_count),
                "recapture_ambiguity_hz": float(recapture_ambiguity_hz),
                "recapture_search_span_hz": float(args.recapture_search_span_hz),
                "recapture_search_center_hz": float(recapture_state.search_center_hz),
                "recapture_release_hz": float(args.recapture_release_hz),
                "recapture_estimated_tone_bb_hz": float_or_nan(recapture_state.estimated_tone_bb_hz),
                "recapture_estimated_error_hz": float_or_nan(recapture_state.estimated_error_hz),
                "recapture_estimated_peak_power_db": float_or_nan(recapture_state.estimated_peak_power_db),
                "recapture_estimated_peak_prominence_db": float_or_nan(recapture_state.estimated_peak_prominence_db),
                "recapture_estimate_valid": float(1.0 if recapture_state.estimate_valid else 0.0),
                "base_max_f_off_hz": float(base_max_abs_f_off_hz),
                "base_error_raw_hz": float(last_base_error_raw_hz),
                "base_error_filt_hz": float(last_base_error_filt_hz),
                "base_lock_counter": float(base_centered_counter),
                "base_phase_hold_allowed": float(1.0 if last_base_phase_hold_allowed else 0.0),
                "effective_frequency_error_hz": float(servo.last_effective_error_hz),
                "unlimited_requested_cmd": float(servo.last_unlimited_command),
                "command_slew_limited": float(1.0 if servo.last_slew_limited else 0.0),
                "servo_profile": servo.last_profile,
            },
        )

        if (block_index + 1) % flush_every == 0:
            csv_file.flush()
        if wall_now >= next_status_wall:
            print(status_line(t_rel_s, meas, actuator, applied_command, status))
            csv_file.flush()
            next_status_wall = wall_now + args.status_interval_s
        if profile_loop:
            log_done = time.perf_counter()
            profiler.observe(
                read_s=read_done - loop_t0,
                detect_s=detect_done - detect_t0,
                control_s=control_done - control_t0,
                actuator_s=actuator_write_s,
                log_s=log_done - log_t0,
                total_s=log_done - loop_t0,
            )

    profiler.report("Closed-loop timing")


def build_vcxo_sweep_values(args: argparse.Namespace) -> List[float]:
    start = float(args.vcxo_test_start if args.vcxo_test_start is not None else args.command_min)
    stop = float(args.vcxo_test_stop if args.vcxo_test_stop is not None else args.command_max)

    if args.vcxo_test_step is not None:
        step = float(args.vcxo_test_step)
    else:
        step = 0.05 if infer_requested_actuator_name(args) == "pwm" else 100.0

    if step == 0.0:
        raise ValueError("VCXO sweep step must be non-zero.")
    if stop > start and step < 0.0:
        step = abs(step)
    if stop < start and step > 0.0:
        step = -abs(step)

    values: List[float] = []
    value = start
    if step > 0:
        while value <= stop + (0.5 * abs(step)):
            values.append(float(value))
            value += step
    else:
        while value >= stop - (0.5 * abs(step)):
            values.append(float(value))
            value += step

    if args.vcxo_test_bidirectional and len(values) > 1:
        values.extend(list(reversed(values[:-1])))

    return values


def summarize_step(
    step_index: int,
    command: float,
    valid_freqs: Iterable[float],
    valid_amps: Iterable[float],
    valid_coherences: Iterable[float],
) -> Optional[SweepStepSummary]:
    freq_arr = np.asarray(list(valid_freqs), dtype=np.float64)
    if freq_arr.size == 0:
        return None
    amp_arr = np.asarray(list(valid_amps), dtype=np.float64)
    coh_arr = np.asarray(list(valid_coherences), dtype=np.float64)
    return SweepStepSummary(
        step_index=int(step_index),
        command=float(command),
        mean_f_off_hz=float(np.mean(freq_arr)),
        std_f_off_hz=float(np.std(freq_arr)),
        mean_amp=float(np.mean(amp_arr)) if amp_arr.size else float("nan"),
        mean_coherence=float(np.mean(coh_arr)) if coh_arr.size else float("nan"),
        valid_count=int(freq_arr.size),
    )


def weighted_least_squares(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D and match X rows.")

    if weights is None:
        sqrt_w = np.ones(X.shape[0], dtype=np.float64)
        w = np.ones(X.shape[0], dtype=np.float64)
    else:
        w = np.maximum(np.asarray(weights, dtype=np.float64), 1.0e-12)
        if w.shape[0] != X.shape[0]:
            raise ValueError("weights must match X rows.")
        sqrt_w = np.sqrt(w)

    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    beta, _, rank, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    y_hat = X @ beta
    rss = float(np.sum(w * ((y - y_hat) ** 2)))
    y_mean = float(np.sum(w * y) / max(np.sum(w), 1.0e-12))
    tss = float(np.sum(w * ((y - y_mean) ** 2)))
    r2 = float("nan") if tss <= 0.0 else float(1.0 - (rss / tss))

    cov = None
    if X.shape[0] > X.shape[1] and rank == X.shape[1]:
        sigma2 = rss / float(X.shape[0] - X.shape[1])
        xtwx = Xw.T @ Xw
        try:
            cov = sigma2 * np.linalg.inv(xtwx)
        except np.linalg.LinAlgError:
            cov = None

    return beta, y_hat, cov, rss, r2, rank


def first_n_samples_from_blocks(blocks: List[np.ndarray], n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        return np.empty(0, dtype=np.complex64)

    pieces: List[np.ndarray] = []
    collected = 0
    for blk in blocks:
        need = int(n_samples - collected)
        if need <= 0:
            break
        if blk.shape[0] <= need:
            pieces.append(blk)
            collected += blk.shape[0]
        else:
            pieces.append(blk[:need])
            collected += need
            break

    if not pieces:
        return np.empty(0, dtype=np.complex64)
    return np.concatenate(pieces).astype(np.complex64, copy=False)


def quadratic_interpolated_peak_hz(freq_hz: np.ndarray, log_power: np.ndarray, idx: int) -> float:
    idx = int(idx)
    if idx <= 0 or idx >= (freq_hz.size - 1):
        return float(freq_hz[idx])
    alpha = float(log_power[idx - 1])
    beta = float(log_power[idx])
    gamma = float(log_power[idx + 1])
    denom = alpha - (2.0 * beta) + gamma
    if abs(denom) < 1.0e-12:
        delta = 0.0
    else:
        delta = 0.5 * (alpha - gamma) / denom
        delta = clamp(delta, -1.0, 1.0)
    bin_hz = float(freq_hz[1] - freq_hz[0]) if freq_hz.size >= 2 else 0.0
    return float(freq_hz[idx] + (delta * bin_hz))


def estimate_coarse_tone_hz_from_samples(
    iq: np.ndarray,
    sample_rate_hz: float,
    search_center_hz: float,
    search_span_hz: float,
) -> CoarseToneEstimate:
    iq = np.asarray(iq, dtype=np.complex128)
    if iq.size < 16:
        return CoarseToneEstimate(
            tone_hz=float(search_center_hz),
            peak_power_db=float("nan"),
            peak_prominence_db=float("nan"),
            second_peak_power_db=float("nan"),
            noise_floor_db=float("nan"),
            valid=False,
        )

    window = np.hanning(iq.size).astype(np.float64)
    spec = np.fft.fftshift(np.fft.fft(iq * window))
    power = np.abs(spec) ** 2
    freq_hz = np.fft.fftshift(np.fft.fftfreq(iq.size, d=1.0 / float(sample_rate_hz)))

    mask = np.abs(freq_hz - float(search_center_hz)) <= float(search_span_hz)
    if not np.any(mask):
        mask = np.ones_like(freq_hz, dtype=bool)

    mask_indices = np.flatnonzero(mask)
    peak_sub_idx = int(np.argmax(power[mask]))
    peak_idx = int(mask_indices[peak_sub_idx])
    log_power = np.log(np.maximum(power, 1.0e-30))
    peak_hz = quadratic_interpolated_peak_hz(freq_hz, log_power, peak_idx)
    peak_power_db = 10.0 * math.log10(max(float(power[peak_idx]), 1.0e-30))
    masked_power = np.asarray(power[mask], dtype=np.float64)
    noise_floor_lin = float(np.median(masked_power)) if masked_power.size else float("nan")
    noise_floor_db = (
        10.0 * math.log10(max(noise_floor_lin, 1.0e-30))
        if np.isfinite(noise_floor_lin)
        else float("nan")
    )

    guard_bins = 2
    peak_mask = np.ones(masked_power.size, dtype=bool)
    lo = max(0, peak_sub_idx - guard_bins)
    hi = min(masked_power.size, peak_sub_idx + guard_bins + 1)
    peak_mask[lo:hi] = False
    if np.any(peak_mask):
        second_peak_lin = float(np.max(masked_power[peak_mask]))
        second_peak_db = 10.0 * math.log10(max(second_peak_lin, 1.0e-30))
    else:
        second_peak_db = float("nan")

    comparison_floor_db = noise_floor_db
    if np.isfinite(second_peak_db):
        comparison_floor_db = max(comparison_floor_db, second_peak_db) if np.isfinite(comparison_floor_db) else second_peak_db
    peak_prominence_db = (
        float(peak_power_db - comparison_floor_db)
        if np.isfinite(peak_power_db) and np.isfinite(comparison_floor_db)
        else float("nan")
    )
    return CoarseToneEstimate(
        tone_hz=float(peak_hz),
        peak_power_db=float(peak_power_db),
        peak_prominence_db=float(peak_prominence_db),
        second_peak_power_db=float(second_peak_db),
        noise_floor_db=float(noise_floor_db),
        valid=bool(np.isfinite(peak_prominence_db)),
    )


def build_precision_sequence(args: argparse.Namespace) -> List[PrecisionSequenceStep]:
    center_cmd = float(
        args.vcxo_precision_center
        if args.vcxo_precision_center is not None
        else args.command_start
    )
    if args.vcxo_precision_delta is not None:
        delta_cmd = abs(float(args.vcxo_precision_delta))
    else:
        delta_cmd = 0.01 if infer_requested_actuator_name(args) == "pwm" else 10.0
    if delta_cmd <= 0.0:
        raise ValueError("--vcxo-precision-delta must be positive.")

    low_cmd = center_cmd - delta_cmd
    high_cmd = center_cmd + delta_cmd
    if low_cmd < float(args.command_min) or high_cmd > float(args.command_max):
        raise ValueError("Precision test center +/- delta must stay inside command range.")

    cycles = int(args.vcxo_precision_cycles)
    if cycles <= 0:
        raise ValueError("--vcxo-precision-cycles must be positive.")

    seq: List[PrecisionSequenceStep] = [PrecisionSequenceStep(center_cmd, "center", -1)]
    for pair_index in range(cycles):
        if pair_index % 2 == 0:
            seq.append(PrecisionSequenceStep(high_cmd, "plus", pair_index))
            seq.append(PrecisionSequenceStep(low_cmd, "minus", pair_index))
        else:
            seq.append(PrecisionSequenceStep(low_cmd, "minus", pair_index))
            seq.append(PrecisionSequenceStep(high_cmd, "plus", pair_index))
    seq.append(PrecisionSequenceStep(center_cmd, "center", cycles))
    return seq


def analyze_precision_dwell(
    raw_blocks: List[np.ndarray],
    sample_rate_hz: float,
    block_size: int,
    search_center_hz: float,
    search_span_hz: float,
    fft_samples: int,
    settle_blocks: int,
    amp_threshold: float,
    coherence_threshold: float,
    min_fit_blocks: int,
    step_index: int,
    command: float,
    role: str,
    pair_index: int,
    wall_mid_rel_s: float,
    sample_mid_rel_s: float,
):
    if not raw_blocks:
        raise ValueError("raw_blocks must not be empty.")

    use_blocks = raw_blocks[settle_blocks:] if settle_blocks < len(raw_blocks) else []
    coarse_samples = first_n_samples_from_blocks(use_blocks, max(int(fft_samples), int(block_size)))
    if coarse_samples.size < int(block_size):
        coarse_bb_hz = float(search_center_hz)
        coarse_peak_power_db = float("nan")
    else:
        coarse_estimate = estimate_coarse_tone_hz_from_samples(
            coarse_samples,
            sample_rate_hz=sample_rate_hz,
            search_center_hz=search_center_hz,
            search_span_hz=search_span_hz,
        )
        coarse_bb_hz = float(coarse_estimate.tone_hz)
        coarse_peak_power_db = float(coarse_estimate.peak_power_db)

    tracker = ToneTracker(sample_rate_hz, block_size, coarse_bb_hz)
    meas_list: List[BlockMeasurement] = []
    fit_times_s: List[float] = []
    fit_phases_rad: List[float] = []
    fit_weights: List[float] = []
    amps_after_settle: List[float] = []
    coh_after_settle: List[float] = []

    for local_idx, blk in enumerate(raw_blocks):
        meas = tracker.process_block(blk)
        meas_list.append(meas)
        if local_idx >= settle_blocks:
            amps_after_settle.append(meas.amp)
            coh_after_settle.append(meas.coherence)
            if (
                meas.valid_phase_step
                and np.isfinite(meas.f_off_hz)
                and meas.amp >= float(amp_threshold)
                and meas.coherence >= float(coherence_threshold)
            ):
                fit_times_s.append((local_idx + 0.5) * float(block_size) / float(sample_rate_hz))
                fit_phases_rad.append(meas.unwrapped_phase_rad)
                fit_weights.append(max(meas.amp * max(meas.coherence, 1.0e-3), 1.0e-6))

    refined_bb_hz = float(coarse_bb_hz)
    fine_residual_hz = float("nan")
    fit_r2 = float("nan")
    valid = False
    valid_block_count = int(len(fit_times_s))

    if valid_block_count >= int(min_fit_blocks):
        X = np.column_stack(
            [
                np.ones(valid_block_count, dtype=np.float64),
                np.asarray(fit_times_s, dtype=np.float64),
            ]
        )
        y = np.asarray(fit_phases_rad, dtype=np.float64)
        w = np.asarray(fit_weights, dtype=np.float64)
        beta, _, _, _, fit_r2, rank = weighted_least_squares(X, y, w)
        if rank == 2:
            slope_rad_per_s = float(beta[1])
            fine_residual_hz = slope_rad_per_s / (2.0 * math.pi)
            refined_bb_hz = float(coarse_bb_hz + fine_residual_hz)
            valid = np.isfinite(refined_bb_hz)

    mean_amp = float(np.mean(amps_after_settle)) if amps_after_settle else float("nan")
    mean_coherence = float(np.mean(coh_after_settle)) if coh_after_settle else float("nan")

    result = PrecisionDwellResult(
        step_index=int(step_index),
        command=float(command),
        role=role,
        pair_index=int(pair_index),
        wall_mid_rel_s=float(wall_mid_rel_s),
        sample_mid_rel_s=float(sample_mid_rel_s),
        coarse_bb_hz=float(coarse_bb_hz),
        refined_bb_hz=float(refined_bb_hz),
        fine_residual_hz=float(fine_residual_hz),
        coarse_peak_power_db=float(coarse_peak_power_db),
        mean_amp=float(mean_amp),
        mean_coherence=float(mean_coherence),
        fit_r2=float(fit_r2),
        valid_block_count=int(valid_block_count),
        dwell_block_count=int(len(raw_blocks)),
        valid=bool(valid),
    )
    return result, meas_list


def report_precision_sensitivity(
    results: List[PrecisionDwellResult],
    center_cmd: float,
    out_csv: str,
) -> None:
    summary_path = os.path.splitext(out_csv)[0] + "_precision_summary.txt"
    valid_results = [r for r in results if r.valid]
    if len(valid_results) < 4:
        message = "Not enough valid precision dwells to estimate Hz/code."
        print(message)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(message + "\n")
        return

    t_wall = np.asarray([r.wall_mid_rel_s for r in valid_results], dtype=np.float64)
    t_centered = t_wall - float(np.mean(t_wall))
    cmd_offset = np.asarray([r.command - center_cmd for r in valid_results], dtype=np.float64)
    freq_meas_hz = np.asarray([r.refined_bb_hz for r in valid_results], dtype=np.float64)
    weights = np.asarray(
        [
            max(r.mean_coherence, 1.0e-3) * math.sqrt(max(r.valid_block_count, 1))
            for r in valid_results
        ],
        dtype=np.float64,
    )

    X = np.column_stack(
        [
            np.ones_like(t_centered),
            t_centered,
            cmd_offset,
        ]
    )
    beta, _, cov, rss, r2, rank = weighted_least_squares(X, freq_meas_hz, weights)
    if rank < 3:
        message = "Precision regression was rank-deficient; unable to estimate Hz/code."
        print(message)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(message + "\n")
        return

    intercept_hz = float(beta[0])
    drift_hz_per_s = float(beta[1])
    slope_hz_per_command = float(beta[2])
    slope_stderr = float(math.sqrt(cov[2, 2])) if cov is not None and cov[2, 2] >= 0.0 else float("nan")
    invert_recommended = slope_hz_per_command < 0.0

    pairwise_slopes: List[float] = []
    by_pair: Dict[int, Dict[str, PrecisionDwellResult]] = {}
    for res in valid_results:
        if res.pair_index < 0:
            continue
        by_pair.setdefault(int(res.pair_index), {})[res.role] = res
    for pair_index, pair in sorted(by_pair.items()):
        plus = pair.get("plus")
        minus = pair.get("minus")
        if plus is None or minus is None:
            continue
        delta_cmd = float(plus.command - minus.command)
        if abs(delta_cmd) < 1.0e-12:
            continue
        drift_corrected_df = (
            float(plus.refined_bb_hz - minus.refined_bb_hz)
            - (drift_hz_per_s * float(plus.wall_mid_rel_s - minus.wall_mid_rel_s))
        )
        pairwise_slopes.append(drift_corrected_df / delta_cmd)

    pairwise_median = float(np.median(pairwise_slopes)) if pairwise_slopes else float("nan")
    pairwise_std = float(np.std(pairwise_slopes)) if pairwise_slopes else float("nan")

    print("")
    print("VCXO precision characterization summary")
    print(f"  estimated_sensitivity_hz_per_command = {slope_hz_per_command:.6f}")
    print(f"  slope_stderr_hz_per_command          = {slope_stderr:.6f}")
    print(f"  drift_hz_per_s                      = {drift_hz_per_s:.6f}")
    print(f"  intercept_bb_hz_at_center           = {intercept_hz:.6f}")
    print(f"  regression_r2                       = {r2:.6f}")
    print(f"  pairwise_median_hz_per_command      = {pairwise_median:.6f}")
    print(f"  pairwise_std_hz_per_command         = {pairwise_std:.6f}")
    print(f"  valid_dwells                        = {len(valid_results)}")
    print(f"  recommended_invert_control          = {int(invert_recommended)}")
    if invert_recommended:
        print("  interpretation: increasing command reduces measured baseband frequency")
    else:
        print("  interpretation: increasing command increases measured baseband frequency")
    print("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("VCXO precision characterization summary\n")
        f.write(f"estimated_sensitivity_hz_per_command={slope_hz_per_command:.12f}\n")
        f.write(f"slope_stderr_hz_per_command={slope_stderr:.12f}\n")
        f.write(f"drift_hz_per_s={drift_hz_per_s:.12f}\n")
        f.write(f"intercept_bb_hz_at_center={intercept_hz:.12f}\n")
        f.write(f"regression_r2={r2:.12f}\n")
        f.write(f"pairwise_median_hz_per_command={pairwise_median:.12f}\n")
        f.write(f"pairwise_std_hz_per_command={pairwise_std:.12f}\n")
        f.write(f"valid_dwells={len(valid_results)}\n")
        f.write(f"recommended_invert_control={int(invert_recommended)}\n")
        f.write("valid_dwell_table:\n")
        for res in valid_results:
            f.write(
                "  "
                f"step={res.step_index}, role={res.role}, pair={res.pair_index}, "
                f"cmd={res.command:.3f}, wall_mid_s={res.wall_mid_rel_s:.3f}, "
                f"coarse_bb_hz={res.coarse_bb_hz:.6f}, refined_bb_hz={res.refined_bb_hz:.6f}, "
                f"fine_residual_hz={res.fine_residual_hz:.6f}, amp={res.mean_amp:.6f}, "
                f"coh={res.mean_coherence:.6f}, r2={res.fit_r2:.6f}, "
                f"valid_blocks={res.valid_block_count}\n"
            )

    print(f"Saved precision summary to {summary_path}")


def maybe_plot_csv(csv_path: str) -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        from analyze_cw_vcxo_log import create_plots
    except Exception as exc:
        print(f"Plot generation skipped: unable to import analysis helper ({exc}).", file=sys.stderr)
        return

    png_path = os.path.splitext(csv_path)[0] + "_plots.png"
    try:
        create_plots(csv_path, png_path)
        print(f"Saved plots to {png_path}")
    except Exception as exc:
        print(f"Plot generation failed: {exc}", file=sys.stderr)


def run_open_or_closed_loop(
    args: argparse.Namespace,
    tone_ref: ToneReference,
    sdr: SDRBase,
    actuator: ActuatorBase,
    writer: csv.DictWriter,
    csv_file,
    stop: GracefulStop,
) -> None:
    if args.mode == "closed_loop":
        run_phase_discipline_loop(args, tone_ref, sdr, actuator, writer, csv_file, stop)
        return

    tracker = ToneTracker(args.sample_rate_hz, args.block_size, tone_ref.tone_bb_hz)
    block_dt_s = tracker.block_dt_s
    profiler = LoopProfiler(args.profile_loop, block_dt_s)
    profile_loop = bool(args.profile_loop)
    freq_unambiguous_hz = 0.5 / block_dt_s
    sdr.warmup(args.block_size, args.warmup_blocks)

    wall_start = time.monotonic()
    next_status_wall = wall_start
    flush_every = max(1, int(round(float(args.log_flush_interval_s) / block_dt_s)))
    total_blocks = int(math.ceil(args.duration_s / block_dt_s))

    for block_index in range(total_blocks):
        if stop.stop_requested:
            print("Stop requested, ending run.")
            break
        loop_t0 = time.perf_counter() if profile_loop else 0.0
        iq = sdr.read_block(args.block_size)
        read_done = time.perf_counter() if profile_loop else 0.0
        wall_now = time.monotonic()
        wall_rel_s = wall_now - wall_start
        t_rel_s = (block_index + 0.5) * block_dt_s
        detect_t0 = time.perf_counter() if profile_loop else 0.0
        meas = tracker.process_block(iq)
        detect_done = time.perf_counter() if profile_loop else 0.0
        control_done = detect_done
        log_t0 = time.perf_counter() if profile_loop else 0.0
        write_csv_row(
            writer=writer,
            args=args,
            tone_ref=tone_ref,
            sdr_backend=sdr.backend_name,
            actuator=actuator,
            block_dt_s=block_dt_s,
            freq_unambiguous_hz=freq_unambiguous_hz,
            block_index=block_index,
            t_rel_s=t_rel_s,
            wall_rel_s=wall_rel_s,
            meas=meas,
            update_allowed=False,
            gate_reason="open_loop",
            control=None,
            applied_command=None,
            status="open_loop_log",
        )
        if (block_index + 1) % flush_every == 0:
            csv_file.flush()
        if wall_now >= next_status_wall:
            print(status_line(t_rel_s, meas, actuator, None, "open_loop_log"))
            csv_file.flush()
            next_status_wall = wall_now + args.status_interval_s
        if profile_loop:
            log_done = time.perf_counter()
            profiler.observe(
                read_s=read_done - loop_t0,
                detect_s=detect_done - detect_t0,
                control_s=control_done - detect_done,
                actuator_s=0.0,
                log_s=log_done - log_t0,
                total_s=log_done - loop_t0,
            )

    profiler.report("Open-loop timing")


def run_vcxo_test(
    args: argparse.Namespace,
    tone_ref: ToneReference,
    sdr: SDRBase,
    actuator: ActuatorBase,
    writer: csv.DictWriter,
    csv_file,
    stop: GracefulStop,
) -> None:
    tracker = ToneTracker(args.sample_rate_hz, args.block_size, tone_ref.tone_bb_hz)
    block_dt_s = tracker.block_dt_s
    freq_unambiguous_hz = 0.5 / block_dt_s
    sweep_values = build_vcxo_sweep_values(args)
    dwell_s = float(args.vcxo_test_dwell_s)
    settle_s = float(args.vcxo_test_settle_s)
    total_required_s = dwell_s * len(sweep_values)

    summaries: List[SweepStepSummary] = []
    step_index = 0
    current_command = actuator.write_command(sweep_values[step_index], settle_s=args.control_settle_s)
    sdr.warmup(args.block_size, args.warmup_blocks)

    wall_start = time.monotonic()
    step_started_wall = wall_start
    next_status_wall = wall_start
    flush_every = max(1, int(round(float(args.log_flush_interval_s) / block_dt_s)))
    max_duration_s = float(args.duration_s) if args.duration_s and args.duration_s > 0 else None

    valid_freqs: List[float] = []
    valid_amps: List[float] = []
    valid_coherences: List[float] = []
    block_index = 0

    while step_index < len(sweep_values):
        if stop.stop_requested:
            print("Stop requested, ending VCXO test.")
            break
        wall_now = time.monotonic()
        if max_duration_s is not None and (wall_now - wall_start) >= max_duration_s:
            print("VCXO test stopped early because duration limit was reached.")
            break

        iq = sdr.read_block(args.block_size)
        wall_now = time.monotonic()
        wall_rel_s = wall_now - wall_start
        step_rel_s = wall_now - step_started_wall
        t_rel_s = (block_index + 0.5) * block_dt_s
        meas = tracker.process_block(iq)
        gate_ok, gate_reason = should_allow_update(meas, args)
        valid_for_fit = gate_ok and (step_rel_s >= settle_s)
        status = "vcxo_test_measure"

        if valid_for_fit:
            valid_freqs.append(meas.f_off_hz)
            valid_amps.append(meas.amp)
            valid_coherences.append(meas.coherence)
        elif step_rel_s < settle_s:
            status = "vcxo_test_settle"
        else:
            status = f"vcxo_test_hold_{gate_reason}"

        control_update = ControlUpdate(
            applied_command=float(current_command),
            requested_command=float(current_command),
            p_term=float("nan"),
            i_term=float("nan"),
            error_hz=float_or_nan(meas.f_off_hz if np.isfinite(meas.f_off_hz) else None),
        )

        write_csv_row(
            writer=writer,
            args=args,
            tone_ref=tone_ref,
            sdr_backend=sdr.backend_name,
            actuator=actuator,
            block_dt_s=block_dt_s,
            freq_unambiguous_hz=freq_unambiguous_hz,
            block_index=block_index,
            t_rel_s=t_rel_s,
            wall_rel_s=wall_rel_s,
            meas=meas,
            update_allowed=False,
            gate_reason=gate_reason,
            control=control_update,
            applied_command=current_command,
            status=status,
            test_step_index=step_index,
            test_step_cmd=current_command,
            test_valid_for_fit=valid_for_fit,
        )

        if (block_index + 1) % flush_every == 0:
            csv_file.flush()

        if wall_now >= next_status_wall:
            print(status_line(t_rel_s, meas, actuator, current_command, status))
            csv_file.flush()
            next_status_wall = wall_now + args.status_interval_s

        if step_rel_s >= dwell_s:
            summary = summarize_step(
                step_index=step_index,
                command=current_command,
                valid_freqs=valid_freqs,
                valid_amps=valid_amps,
                valid_coherences=valid_coherences,
            )
            if summary is not None:
                summaries.append(summary)
                print(
                    f"vcxo_step={summary.step_index:2d} | "
                    f"cmd={summary.command:8.3f} | "
                    f"mean_f_off={summary.mean_f_off_hz:9.4f} Hz | "
                    f"std={summary.std_f_off_hz:8.4f} | "
                    f"amp={summary.mean_amp:7.4f} | "
                    f"coh={summary.mean_coherence:6.3f} | "
                    f"n={summary.valid_count}"
                )
            else:
                print(
                    f"vcxo_step={step_index:2d} | cmd={current_command:8.3f} | "
                    "no valid samples after settle window"
                )

            step_index += 1
            if step_index >= len(sweep_values):
                break
            valid_freqs = []
            valid_amps = []
            valid_coherences = []
            current_command = actuator.write_command(
                sweep_values[step_index],
                settle_s=args.control_settle_s,
            )
            step_started_wall = time.monotonic()

        block_index += 1

    csv_file.flush()
    report_vcxo_sensitivity(summaries)


def run_vcxo_precision_test(
    args: argparse.Namespace,
    tone_ref: ToneReference,
    sdr: SDRBase,
    actuator: ActuatorBase,
    writer: csv.DictWriter,
    csv_file,
    stop: GracefulStop,
    out_csv: str,
) -> None:
    block_dt_s = float(args.block_size) / float(args.sample_rate_hz)
    freq_unambiguous_hz = 0.5 / block_dt_s
    dwell_s = float(args.vcxo_precision_dwell_s)
    settle_s = float(args.vcxo_precision_settle_s)
    settle_blocks = max(0, int(round(settle_s / block_dt_s)))
    blocks_per_dwell = max(1, int(round(dwell_s / block_dt_s)))
    sequence = build_precision_sequence(args)
    center_cmd = float(
        args.vcxo_precision_center
        if args.vcxo_precision_center is not None
        else args.command_start
    )
    delta_cmd = abs(
        float(args.vcxo_precision_delta)
        if args.vcxo_precision_delta is not None
        else (0.01 if infer_requested_actuator_name(args) == "pwm" else 10.0)
    )
    max_duration_s = float(args.duration_s) if args.duration_s and args.duration_s > 0 else None

    sdr.warmup(args.block_size, args.warmup_blocks)
    search_center_hz = float(tone_ref.tone_bb_hz)

    wall_start = time.monotonic()
    total_block_index = 0
    results: List[PrecisionDwellResult] = []

    for step_index, step in enumerate(sequence):
        if stop.stop_requested:
            print("Stop requested, ending VCXO precision test.")
            break
        if max_duration_s is not None and (time.monotonic() - wall_start) >= max_duration_s:
            print("VCXO precision test stopped early because duration limit was reached.")
            break

        current_command = actuator.write_command(step.command, settle_s=args.control_settle_s)
        dwell_wall_start = time.monotonic()
        raw_blocks: List[np.ndarray] = []
        block_wall_rels: List[float] = []
        block_sample_rels: List[float] = []

        for _ in range(blocks_per_dwell):
            if stop.stop_requested:
                break
            iq = sdr.read_block(args.block_size)
            wall_now = time.monotonic()
            raw_blocks.append(iq)
            block_wall_rels.append(wall_now - wall_start)
            block_sample_rels.append((total_block_index + 0.5) * block_dt_s)
            total_block_index += 1

        if not raw_blocks:
            break

        wall_mid_rel_s = 0.5 * (
            (dwell_wall_start - wall_start) + block_wall_rels[-1]
        )
        sample_mid_rel_s = 0.5 * (block_sample_rels[0] + block_sample_rels[-1])

        dwell_result, meas_list = analyze_precision_dwell(
            raw_blocks=raw_blocks,
            sample_rate_hz=args.sample_rate_hz,
            block_size=args.block_size,
            search_center_hz=search_center_hz,
            search_span_hz=args.vcxo_precision_search_span_hz,
            fft_samples=args.vcxo_precision_fft_samples,
            settle_blocks=settle_blocks,
            amp_threshold=args.amp_threshold,
            coherence_threshold=args.coherence_threshold,
            min_fit_blocks=args.vcxo_precision_min_fit_blocks,
            step_index=step_index,
            command=current_command,
            role=step.role,
            pair_index=step.pair_index,
            wall_mid_rel_s=wall_mid_rel_s,
            sample_mid_rel_s=sample_mid_rel_s,
        )
        results.append(dwell_result)
        if dwell_result.valid and np.isfinite(dwell_result.refined_bb_hz):
            search_center_hz = float(dwell_result.refined_bb_hz)

        for local_idx, meas in enumerate(meas_list):
            gate_ok, gate_reason = should_allow_update(meas, args)
            used_for_fit = (
                local_idx >= settle_blocks
                and gate_ok
                and dwell_result.valid
            )
            if local_idx < settle_blocks:
                status = "precision_settle"
                gate_reason_local = "precision_settle"
            elif used_for_fit:
                status = "precision_fit_block"
                gate_reason_local = "precision_fit_block"
            elif gate_ok:
                status = "precision_valid_block_not_used"
                gate_reason_local = "precision_not_used"
            else:
                status = f"precision_hold_{gate_reason}"
                gate_reason_local = gate_reason

            control_update = ControlUpdate(
                applied_command=float(current_command),
                requested_command=float(current_command),
                p_term=float("nan"),
                i_term=float("nan"),
                error_hz=float_or_nan(dwell_result.refined_bb_hz if dwell_result.valid else None),
            )
            write_csv_row(
                writer=writer,
                args=args,
                tone_ref=tone_ref,
                sdr_backend=sdr.backend_name,
                actuator=actuator,
                block_dt_s=block_dt_s,
                freq_unambiguous_hz=freq_unambiguous_hz,
                block_index=int(step_index * blocks_per_dwell + local_idx),
                t_rel_s=float(block_sample_rels[local_idx]),
                wall_rel_s=float(block_wall_rels[local_idx]),
                meas=meas,
                update_allowed=False,
                gate_reason=gate_reason_local,
                control=control_update,
                applied_command=current_command,
                status=status,
                test_step_index=step_index,
                test_step_cmd=current_command,
                test_valid_for_fit=used_for_fit,
                extra_fields={
                    "precision_role": step.role,
                    "precision_pair_index": float(step.pair_index),
                    "precision_valid_dwell": 1.0 if dwell_result.valid else 0.0,
                    "precision_coarse_bb_hz": dwell_result.coarse_bb_hz,
                    "precision_refined_bb_hz": dwell_result.refined_bb_hz,
                    "precision_fine_residual_hz": dwell_result.fine_residual_hz,
                    "precision_coarse_peak_power_db": dwell_result.coarse_peak_power_db,
                    "precision_fit_r2": dwell_result.fit_r2,
                    "precision_wall_mid_rel_s": dwell_result.wall_mid_rel_s,
                    "precision_sample_mid_rel_s": dwell_result.sample_mid_rel_s,
                    "precision_center_cmd": center_cmd,
                    "precision_delta_cmd": delta_cmd,
                    "precision_valid_block_count": float(dwell_result.valid_block_count),
                    "precision_dwell_block_count": float(dwell_result.dwell_block_count),
                },
            )

        csv_file.flush()
        validity = "valid" if dwell_result.valid else "invalid"
        print(
            f"precision_step={step_index:2d} | role={step.role:<6} | "
            f"cmd={current_command:8.3f} | coarse={dwell_result.coarse_bb_hz:11.3f} Hz | "
            f"refined={dwell_result.refined_bb_hz:11.3f} Hz | "
            f"fine={dwell_result.fine_residual_hz:8.3f} Hz | "
            f"amp={dwell_result.mean_amp:7.4f} | coh={dwell_result.mean_coherence:6.3f} | "
            f"r2={dwell_result.fit_r2:6.3f} | n={dwell_result.valid_block_count:4d} | {validity}"
        )

    csv_file.flush()
    report_precision_sensitivity(results, center_cmd=center_cmd, out_csv=out_csv)


def report_vcxo_sensitivity(summaries: List[SweepStepSummary]) -> None:
    if len(summaries) < 2:
        print("Not enough valid VCXO sweep points to estimate sensitivity.")
        return

    cmd = np.asarray([s.command for s in summaries], dtype=np.float64)
    freq = np.asarray([s.mean_f_off_hz for s in summaries], dtype=np.float64)
    slope, intercept = np.polyfit(cmd, freq, 1)
    invert_recommended = slope < 0.0
    print("")
    print("VCXO characterization summary")
    print(f"  estimated_sensitivity_hz_per_command = {slope:.6f}")
    print(f"  intercept_hz                         = {intercept:.6f}")
    print(f"  recommended_invert_control          = {int(invert_recommended)}")
    if invert_recommended:
        print("  interpretation: increasing command reduces measured frequency offset")
    else:
        print("  interpretation: increasing command increases measured frequency offset")
    print("")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="First-stage CW tone VCXO discipline prototype.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("open_loop", "closed_loop", "vcxo_test", "vcxo_precision_test"),
        help="Experiment mode.",
    )
    parser.add_argument(
        "--sdr-backend",
        default="ugradio",
        choices=("ugradio",),
        help="SDR backend to use.",
    )
    parser.add_argument("--device-index", type=int, default=0, help="SDR device index.")
    parser.add_argument(
        "--sample-rate",
        "--sample-rate-hz",
        dest="sample_rate_hz",
        type=float,
        required=True,
        help="SDR sample rate in samples/second.",
    )
    parser.add_argument(
        "--center-freq",
        "--center-freq-hz",
        dest="center_freq_hz",
        type=float,
        required=True,
        help="SDR RF center frequency in Hz.",
    )
    parser.add_argument(
        "--tone-rf",
        "--tone-rf-hz",
        dest="tone_rf_hz",
        type=float,
        default=None,
        help="Expected CW tone RF frequency in Hz.",
    )
    parser.add_argument(
        "--tone-bb",
        "--expected-offset-hz",
        dest="tone_bb_hz",
        type=float,
        default=None,
        help="Expected CW tone baseband frequency after tuning, in Hz.",
    )
    parser.add_argument("--block-size", type=int, default=16384, help="Samples per block.")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Run duration in seconds. If omitted in vcxo_test or vcxo_precision_test, it is derived from the step schedule.",
    )
    parser.add_argument(
        "--gain",
        default="auto",
        help='SDR gain value or "auto".',
    )
    parser.add_argument(
        "--warmup-blocks",
        type=int,
        default=4,
        help="Discard this many blocks after opening the SDR.",
    )
    parser.add_argument(
        "--amp-threshold",
        type=float,
        default=0.02,
        help="Minimum phasor amplitude required for control or VCXO fit accumulation.",
    )
    parser.add_argument(
        "--coherence-threshold",
        type=float,
        default=0.0,
        help="Minimum coherence metric required for control or VCXO fit accumulation.",
    )
    parser.add_argument(
        "--cmd-min",
        "--dac-min",
        dest="command_min",
        type=float,
        default=None,
        help="Minimum allowed control command. For PWM this is duty fraction.",
    )
    parser.add_argument(
        "--cmd-max",
        "--dac-max",
        dest="command_max",
        type=float,
        default=None,
        help="Maximum allowed control command. For PWM this is duty fraction.",
    )
    parser.add_argument(
        "--cmd-start",
        "--dac-start",
        dest="command_start",
        type=float,
        default=None,
        help="Starting control command, also used as the controller center.",
    )
    parser.add_argument(
        "--closed-loop-command-window",
        "--closed-loop-dac-window",
        dest="closed_loop_command_window",
        type=float,
        default=None,
        help="When running closed loop, clamp the actuator command to command_start +/- this window after applying the configured command min/max.",
    )
    parser.add_argument(
        "--invert-control",
        action="store_true",
        help="Invert control polarity if increasing command reduces frequency offset.",
    )
    parser.add_argument(
        "--vcxo-hz-per-command",
        type=float,
        default=None,
        help="Measured VCXO sensitivity in Hz per command unit. Negative if increasing command lowers measured baseband frequency.",
    )
    parser.add_argument(
        "--pll-estimator-alpha",
        type=float,
        default=0.15,
        help="Alpha-beta estimator phase correction gain.",
    )
    parser.add_argument(
        "--pll-estimator-beta",
        type=float,
        default=0.25,
        help="Alpha-beta estimator frequency correction gain.",
    )
    parser.add_argument(
        "--pll-estimator-gamma",
        type=float,
        default=0.0,
        help="Optional alpha-beta-gamma drift correction gain.",
    )
    parser.add_argument(
        "--pll-cycle-slip-threshold-rad",
        type=float,
        default=3.05,
        help="Reject estimator phase residuals above this magnitude as likely cycle slips.",
    )
    parser.add_argument(
        "--pll-frequency-proportional-gain",
        type=float,
        default=0.01,
        help="Physical frequency-servo proportional gain applied to measured frequency error.",
    )
    parser.add_argument(
        "--pll-frequency-integral-gain",
        type=float,
        default=0.0,
        help="Physical frequency-servo integral gain in 1/s.",
    )
    parser.add_argument(
        "--acquisition-frequency-proportional-gain",
        type=float,
        default=0.02,
        help="Frequency-servo proportional gain used only during pull-in and coarse recapture.",
    )
    parser.add_argument(
        "--acquisition-frequency-integral-gain",
        type=float,
        default=0.0,
        help="Frequency-servo integral gain used only during pull-in and coarse recapture.",
    )
    parser.add_argument(
        "--acquisition-frequency-deadband-hz",
        type=float,
        default=0.0,
        help="Frequency-error deadband used during pull-in and coarse recapture.",
    )
    parser.add_argument(
        "--acquisition-command-slew-rate-per-s",
        type=float,
        default=200.0,
        help="Maximum actuator command movement per second during pull-in and coarse recapture. Use 0 to disable.",
    )
    parser.add_argument(
        "--initial-frequency-step",
        dest="initial_frequency_step_enabled",
        action="store_true",
        default=True,
        help="Before starting closed-loop logging, apply a direct one-shot DAC correction from the measured residual frequency.",
    )
    parser.add_argument(
        "--no-initial-frequency-step",
        dest="initial_frequency_step_enabled",
        action="store_false",
        help="Disable the one-shot startup frequency-centering correction.",
    )
    parser.add_argument(
        "--initial-frequency-step-blocks",
        type=int,
        default=64,
        help="Blocks used to estimate the median residual frequency for the startup one-shot correction.",
    )
    parser.add_argument(
        "--initial-frequency-step-min-valid-blocks",
        type=int,
        default=48,
        help="Minimum valid residual-frequency blocks required before applying the startup one-shot correction.",
    )
    parser.add_argument(
        "--initial-frequency-step-min-coherence",
        type=float,
        default=0.95,
        help="Minimum coherence for blocks used by the startup one-shot frequency correction.",
    )
    parser.add_argument(
        "--initial-frequency-step-gain",
        type=float,
        default=0.65,
        help="Fraction of the direct -f_off/vcxo_hz_per_command correction applied at startup.",
    )
    parser.add_argument(
        "--initial-frequency-step-passes",
        type=int,
        default=3,
        help="Maximum startup frequency-step measurement/correction passes before logging begins.",
    )
    parser.add_argument(
        "--initial-frequency-step-refine-gain",
        type=float,
        default=0.35,
        help="Additional gain multiplier used on startup frequency-step passes after the first.",
    )
    parser.add_argument(
        "--initial-frequency-step-max-command-delta",
        type=float,
        default=25.0,
        help="Maximum absolute command jump allowed for the startup one-shot correction. Use 0 to disable this clamp.",
    )
    parser.add_argument(
        "--initial-frequency-step-deadband-hz",
        type=float,
        default=1.0,
        help="Skip the startup one-shot correction when median |f_off| is already within this threshold.",
    )
    parser.add_argument(
        "--initial-frequency-step-discard-blocks",
        type=int,
        default=4,
        help="Blocks to read and discard after the startup one-shot correction before beginning the logged phase loop.",
    )
    parser.add_argument(
        "--frequency-error-deadband-hz",
        type=float,
        default=5.0,
        help="Do not move the actuator for windowed frequency errors within this magnitude.",
    )
    parser.add_argument(
        "--command-slew-rate-per-s",
        type=float,
        default=50.0,
        help="Maximum actuator command movement per second. Use 0 to disable slew limiting.",
    )
    parser.add_argument(
        "--base-frequency-integral-gain",
        type=float,
        default=0.0,
        help="Slow high-confidence integrator gain that moves the actuator base command to remove steady residual frequency.",
    )
    parser.add_argument(
        "--base-frequency-deadband-hz",
        type=float,
        default=2.0,
        help="Residual-frequency deadband for slow base-command adaptation.",
    )
    parser.add_argument(
        "--base-command-slew-rate-per-s",
        type=float,
        default=0.25,
        help="Maximum slow base-command movement per second. Use 0 to disable slew limiting.",
    )
    parser.add_argument(
        "--base-error-filter-alpha",
        type=float,
        default=0.98,
        help="IIR alpha for the slow base-command frequency error. Higher values damp noisy per-block estimates more.",
    )
    parser.add_argument(
        "--base-min-coherence",
        type=float,
        default=0.90,
        help="Minimum control-summary coherence required before slow base-command adaptation.",
    )
    parser.add_argument(
        "--base-max-phase-residual-rad",
        type=float,
        default=1.75,
        help="Maximum estimator wrapped phase residual allowed for slow base-command adaptation.",
    )
    parser.add_argument(
        "--base-max-f-off-fraction",
        type=float,
        default=0.50,
        help="Maximum |f_off| as a fraction of the ambiguity limit allowed for slow base-command adaptation.",
    )
    parser.add_argument(
        "--phase-hold-gain-hz-per-rad",
        type=float,
        default=0.0,
        help="Outer-loop phase hold gain. A positive value converts wrapped phase error into a desired residual frequency so the loop holds phase near a target instead of only nulling average phase slope.",
    )
    parser.add_argument(
        "--phase-target-rad",
        type=float,
        default=None,
        help="Wrapped phase target in radians for phase hold. If omitted and phase hold is enabled, the first valid measured phase becomes the target.",
    )
    parser.add_argument(
        "--phase-hold-arm-threshold-hz",
        type=float,
        default=25.0,
        help="Residual-frequency threshold that must be maintained in fine lock before the phase-hold term is allowed to engage.",
    )
    parser.add_argument(
        "--phase-hold-arm-dwell-updates",
        type=int,
        default=12,
        help="Consecutive fine-lock control updates below --phase-hold-arm-threshold-hz before phase hold engages.",
    )
    parser.add_argument(
        "--phase-hold-exit-dwell-updates",
        type=int,
        default=25,
        help="Consecutive phase-hold updates above --phase-hold-arm-threshold-hz required before dropping an acquired wrapped phase target.",
    )
    parser.add_argument(
        "--phase-hold-base-lock-dwell-updates",
        type=int,
        default=250,
        help="Consecutive control updates with filtered base frequency error inside --phase-hold-base-lock-threshold-hz before wrapped phase hold may arm.",
    )
    parser.add_argument(
        "--phase-hold-base-lock-threshold-hz",
        type=float,
        default=8.0,
        help="Filtered base frequency-error threshold used only for deciding when wrapped phase hold may arm.",
    )
    parser.add_argument(
        "--phase-hold-max-freq-hz",
        type=float,
        default=5.0,
        help="Clamp applied to the phase-hold frequency setpoint after converting phase error into Hz.",
    )
    parser.add_argument(
        "--phase-hold-smoothing-alpha",
        type=float,
        default=0.2,
        help="One-pole smoothing factor for the phase-hold frequency setpoint. 1.0 applies the raw clamped setpoint directly.",
    )
    parser.add_argument(
        "--phase-error-filter-alpha",
        type=float,
        default=1.0,
        help="Circular one-pole update factor for the phase-error observer. 1.0 uses the current accepted phase error directly.",
    )
    parser.add_argument(
        "--phase-error-min-confidence",
        type=float,
        default=0.0,
        help="Minimum phase-estimator confidence required to update the phase-error observer.",
    )
    parser.add_argument(
        "--phase-error-max-residual-rad",
        type=float,
        default=3.05,
        help="Maximum estimator phase residual allowed to update the phase-error observer.",
    )
    parser.add_argument(
        "--phase-cycle-tracking",
        dest="phase_cycle_tracking_enabled",
        action="store_true",
        default=True,
        help="Use prediction plus nearest-branch selection to maintain a cycle-aware phase-error state.",
    )
    parser.add_argument(
        "--no-phase-cycle-tracking",
        dest="phase_cycle_tracking_enabled",
        action="store_false",
        help="Disable cycle-aware phase tracking and control from the wrapped filtered phase error.",
    )
    parser.add_argument(
        "--phase-cycle-max-branch-error-rad",
        type=float,
        default=2.50,
        help="Maximum residual between predicted phase error and nearest wrapped-measurement branch before flagging a cycle slip and holding prediction.",
    )
    parser.add_argument(
        "--phase-target-slew-rate-rad-per-s",
        type=float,
        default=0.02,
        help="For explicit phase targets, first acquire the current phase and then slew the target toward the requested phase at this rate. Use 0 to hold the acquired phase.",
    )
    parser.add_argument(
        "--phase-integral-gain-hz-per-rad-s",
        type=float,
        default=0.0,
        help="Optional slow phase integral gain that contributes to the residual-frequency target.",
    )
    parser.add_argument(
        "--phase-integral-max-hz",
        type=float,
        default=1.0,
        help="Clamp for the slow phase-integral contribution in Hz.",
    )
    parser.add_argument(
        "--phase-slew-gain-hz-per-rad",
        type=float,
        default=None,
        help="Additional recovery-assist phase gain used when cycle-aware phase error is above --phase-slew-enter-error-rad. If omitted, no recovery assist is added.",
    )
    parser.add_argument(
        "--phase-slew-max-freq-hz",
        type=float,
        default=None,
        help="Clamp for the additional recovery-assist frequency setpoint. If omitted, defaults to --phase-hold-max-freq-hz.",
    )
    parser.add_argument(
        "--phase-slew-enter-error-rad",
        type=float,
        default=0.75,
        help="Cycle-aware phase-error magnitude above which recovery mode engages.",
    )
    parser.add_argument(
        "--phase-slew-exit-error-rad",
        type=float,
        default=0.25,
        help="Once in recovery mode, return to steady hold when cycle-aware phase error drops below this magnitude.",
    )
    parser.add_argument(
        "--phase-recovery-enter-dwell-updates",
        type=int,
        default=1,
        help="Consecutive updates above --phase-slew-enter-error-rad required before entering recovery mode.",
    )
    parser.add_argument(
        "--phase-recovery-exit-dwell-updates",
        type=int,
        default=1,
        help="Consecutive recovery updates below --phase-slew-exit-error-rad required before returning to hold mode.",
    )
    parser.add_argument(
        "--phase-recovery-min-dwell-updates",
        type=int,
        default=0,
        help="Minimum updates to remain in recovery mode before exit dwell can complete.",
    )
    parser.add_argument(
        "--pull-in",
        dest="pull_in_enabled",
        action="store_true",
        default=True,
        help="Enable a coarse startup pull-in stage before handing off to the fine closed-loop controller.",
    )
    parser.add_argument(
        "--no-pull-in",
        dest="pull_in_enabled",
        action="store_false",
        help="Disable the coarse startup pull-in stage and start directly in the fine controller.",
    )
    parser.add_argument(
        "--pull-in-switch-hz",
        type=float,
        default=30.0,
        help="Residual frequency threshold required before handoff from pull-in to fine lock.",
    )
    parser.add_argument(
        "--pull-in-exit-hz",
        type=float,
        default=100.0,
        help="Residual frequency threshold that pushes the loop back into pull-in after handoff.",
    )
    parser.add_argument(
        "--pull-in-enter-dwell-updates",
        type=int,
        default=3,
        help="Consecutive valid control updates below --pull-in-switch-hz before handoff to fine lock.",
    )
    parser.add_argument(
        "--pull-in-exit-dwell-updates",
        type=int,
        default=2,
        help="Consecutive fine-lock updates above --pull-in-exit-hz before pushback into pull-in. Set to 0 to disable pushback.",
    )
    parser.add_argument(
        "--control-update-interval-s",
        type=float,
        default=0.05,
        help="Nominal interval between actuator updates. The controller uses windowed estimates and only writes the actuator at this cadence.",
    )
    parser.add_argument(
        "--control-average-interval-s",
        type=float,
        default=0.05,
        help="Nominal averaging window for the control measurement. Wrapped phase and residual frequency are estimated from this window instead of one raw block.",
    )
    parser.add_argument(
        "--control-min-valid-blocks",
        type=int,
        default=8,
        help="Minimum number of gated-valid blocks required in the control window before the loop updates.",
    )
    parser.add_argument(
        "--control-max-f-off-fraction",
        type=float,
        default=0.70,
        help="Reject control updates when |f_off| exceeds this fraction of the block-to-block ambiguity limit.",
    )
    parser.add_argument(
        "--control-max-phase-residual-rad",
        type=float,
        default=2.5,
        help="Reject control updates when the estimator wrapped phase residual exceeds this magnitude.",
    )
    parser.add_argument(
        "--control-min-fit-r2",
        type=float,
        default=0.20,
        help="Minimum R^2 accepted for multi-block control-window phase fits.",
    )
    parser.add_argument(
        "--startup-tone-check",
        dest="startup_tone_check_enabled",
        action="store_true",
        default=True,
        help="Before closed-loop actuator control, verify tone coherence and run a local coarse scan if the supplied tone bin is poor.",
    )
    parser.add_argument(
        "--no-startup-tone-check",
        dest="startup_tone_check_enabled",
        action="store_false",
        help="Disable the pre-control tone coherence check and startup coarse scan.",
    )
    parser.add_argument(
        "--startup-check-blocks",
        type=int,
        default=32,
        help="Blocks used to verify startup tone coherence before closed-loop control.",
    )
    parser.add_argument(
        "--startup-min-coherence",
        type=float,
        default=0.80,
        help="Mean startup coherence below this value triggers a coarse scan before closed-loop control.",
    )
    parser.add_argument(
        "--startup-scan-span-hz",
        type=float,
        default=1000.0,
        help="Half-span around the supplied tone baseband used for the startup coarse scan.",
    )
    parser.add_argument(
        "--startup-scan-fft-samples",
        type=int,
        default=65536,
        help="Maximum samples from the startup check window used by the coarse scan.",
    )
    parser.add_argument(
        "--startup-min-prominence-db",
        type=float,
        default=8.0,
        help="Minimum peak prominence for accepting a startup coarse-scan tone update.",
    )
    parser.add_argument(
        "--startup-require-tone",
        action="store_true",
        default=True,
        help="Abort closed-loop startup if coherence is poor and the coarse scan cannot find a trusted tone.",
    )
    parser.add_argument(
        "--no-startup-require-tone",
        dest="startup_require_tone",
        action="store_false",
        help="Continue with the supplied tone reference if startup tone scan fails.",
    )
    parser.add_argument(
        "--recapture",
        dest="recapture_enabled",
        action="store_true",
        default=True,
        help="Enable coarse frequency recapture when the block estimator is repeatedly gated or near ambiguity.",
    )
    parser.add_argument(
        "--no-recapture",
        dest="recapture_enabled",
        action="store_false",
        help="Disable coarse FFT-based recapture and rely only on the raw block estimator.",
    )
    parser.add_argument(
        "--recapture-invalid-blocks",
        type=int,
        default=20,
        help="Consecutive low-quality blocks required before recapture starts.",
    )
    parser.add_argument(
        "--recapture-ambiguity-fraction",
        type=float,
        default=0.70,
        help="Fraction of the block-estimator ambiguity limit treated as unsafe for normal control updates.",
    )
    parser.add_argument(
        "--recapture-ambiguity-blocks",
        type=int,
        default=4,
        help="Consecutive near-ambiguity blocks required before recapture starts.",
    )
    parser.add_argument(
        "--recapture-dwell-blocks",
        type=int,
        default=32,
        help="Blocks to accumulate before each coarse FFT-based recapture estimate.",
    )
    parser.add_argument(
        "--recapture-search-span-hz",
        type=float,
        default=750.0,
        help="FFT search half-span around the last trusted tone estimate during recapture.",
    )
    parser.add_argument(
        "--recapture-min-peak-prominence-db",
        type=float,
        default=8.0,
        help="Minimum coarse-FFT peak prominence required before a recapture estimate is trusted.",
    )
    parser.add_argument(
        "--recapture-release-hz",
        type=float,
        default=100.0,
        help="Residual-frequency magnitude below which recapture returns control to the normal block estimator.",
    )
    parser.add_argument(
        "--recapture-on-ambiguity",
        action="store_true",
        default=False,
        help="Allow near-block-ambiguity frequency estimates to start coarse recapture. By default recapture starts only after sustained low-quality signal gates.",
    )
    parser.add_argument("--out-csv", default=None, help="Output CSV path.")
    parser.add_argument(
        "--out-bin",
        default=None,
        help="Output binary log path. Uses the same logical fields as CSV but writes streaming binary records.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots after the run.")
    parser.add_argument(
        "--status-interval-s",
        type=float,
        default=1.0,
        help="Console status print interval.",
    )
    parser.add_argument(
        "--log-flush-interval-s",
        type=float,
        default=1.0,
        help="Flush the output log at this interval. Larger values reduce fast-loop disk stalls.",
    )
    parser.add_argument(
        "--profile-loop",
        action="store_true",
        help="Time read, detector, control, actuator, logging, and total loop segments and print a summary.",
    )
    parser.add_argument(
        "--control-settle-s",
        type=float,
        default=0.0,
        help="Optional wait after each actuator command write.",
    )
    parser.add_argument("--use-mcp4725", action="store_true", help="Use the MCP4725 DAC path.")
    parser.add_argument("--use-pwm", action="store_true", help="Use Raspberry Pi PWM instead of MCP4725.")
    parser.add_argument(
        "--mcp4725-dither",
        dest="mcp4725_dither",
        action="store_true",
        default=True,
        help="Enable first-order DAC-code dithering so fractional requested codes are realized in time average.",
    )
    parser.add_argument(
        "--no-mcp4725-dither",
        dest="mcp4725_dither",
        action="store_false",
        help="Disable MCP4725 DAC-code dithering and round requested commands to the nearest code.",
    )
    parser.add_argument("--pwm-pin", type=int, default=18, help="GPIO pin for PWM output.")
    parser.add_argument(
        "--pwm-frequency-hz",
        type=float,
        default=2000.0,
        help="PWM carrier frequency in Hz.",
    )
    parser.add_argument(
        "--pwm-vdd",
        type=float,
        default=3.3,
        help="Equivalent filtered PWM high level in volts.",
    )
    parser.add_argument(
        "--ugradio-queue-blocks",
        "--ugradio-batch-blocks",
        dest="ugradio_queue_blocks",
        type=int,
        default=64,
        help="Buffered IQ blocks for the persistent ugradio async stream queue. Older flag name is kept as a compatibility alias.",
    )
    parser.add_argument(
        "--clock-apr-ppm",
        type=int,
        default=120,
        help="APR pull range passed to clock_vcxo_helper init.",
    )
    parser.add_argument(
        "--skip-clock-init",
        action="store_true",
        help="Do not call init_clock_vcxo_28p8mhz() before using the DAC.",
    )
    parser.add_argument(
        "--vcxo-test-start",
        type=float,
        default=None,
        help="First sweep command for vcxo_test.",
    )
    parser.add_argument(
        "--vcxo-test-stop",
        type=float,
        default=None,
        help="Last sweep command for vcxo_test.",
    )
    parser.add_argument(
        "--vcxo-test-step",
        type=float,
        default=None,
        help="Sweep step size for vcxo_test.",
    )
    parser.add_argument(
        "--vcxo-test-dwell-s",
        type=float,
        default=5.0,
        help="Time to hold each sweep point in vcxo_test.",
    )
    parser.add_argument(
        "--vcxo-test-settle-s",
        type=float,
        default=1.0,
        help="Ignore the first part of each dwell when estimating sensitivity.",
    )
    parser.add_argument(
        "--vcxo-test-bidirectional",
        action="store_true",
        help="Sweep up and then back down in vcxo_test.",
    )
    parser.add_argument(
        "--vcxo-precision-center",
        type=float,
        default=None,
        help="Center command for vcxo_precision_test. Defaults to --cmd-start/--dac-start.",
    )
    parser.add_argument(
        "--vcxo-precision-delta",
        type=float,
        default=None,
        help="Half-step command size for vcxo_precision_test.",
    )
    parser.add_argument(
        "--vcxo-precision-cycles",
        type=int,
        default=6,
        help="Number of alternating plus/minus cycles for vcxo_precision_test.",
    )
    parser.add_argument(
        "--vcxo-precision-dwell-s",
        type=float,
        default=2.0,
        help="Time per dwell for vcxo_precision_test.",
    )
    parser.add_argument(
        "--vcxo-precision-settle-s",
        type=float,
        default=0.5,
        help="Ignore this much of each dwell before fitting phase slope in vcxo_precision_test.",
    )
    parser.add_argument(
        "--vcxo-precision-search-span-hz",
        type=float,
        default=5000.0,
        help="FFT coarse-search half-span around the current expected tone in vcxo_precision_test.",
    )
    parser.add_argument(
        "--vcxo-precision-fft-samples",
        type=int,
        default=65536,
        help="Number of post-settle samples used for coarse FFT estimation in vcxo_precision_test.",
    )
    parser.add_argument(
        "--vcxo-precision-min-fit-blocks",
        type=int,
        default=64,
        help="Minimum gated blocks required to fit a dwell frequency in vcxo_precision_test.",
    )

    args = parser.parse_args(argv)

    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")
    if args.sample_rate_hz <= 0.0:
        raise ValueError("--sample-rate must be positive.")
    if args.mode not in ("vcxo_test", "vcxo_precision_test") and (args.duration_s is None or args.duration_s <= 0.0):
        raise ValueError("--duration-s must be positive for open_loop and closed_loop.")
    if args.mode in ("vcxo_test", "vcxo_precision_test") and args.duration_s is not None and args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive when provided.")
    if args.vcxo_precision_dwell_s <= 0.0:
        raise ValueError("--vcxo-precision-dwell-s must be positive.")
    if args.vcxo_precision_settle_s < 0.0:
        raise ValueError("--vcxo-precision-settle-s must be non-negative.")
    if args.vcxo_precision_search_span_hz <= 0.0:
        raise ValueError("--vcxo-precision-search-span-hz must be positive.")
    if args.vcxo_precision_fft_samples <= 0:
        raise ValueError("--vcxo-precision-fft-samples must be positive.")
    if args.vcxo_precision_min_fit_blocks <= 1:
        raise ValueError("--vcxo-precision-min-fit-blocks must be at least 2.")
    if args.pull_in_switch_hz < 0.0:
        raise ValueError("--pull-in-switch-hz must be non-negative.")
    if args.pull_in_exit_hz < args.pull_in_switch_hz:
        raise ValueError("--pull-in-exit-hz must be >= --pull-in-switch-hz.")
    if args.pull_in_enter_dwell_updates <= 0:
        raise ValueError("--pull-in-enter-dwell-updates must be at least 1.")
    if args.pull_in_exit_dwell_updates < 0:
        raise ValueError("--pull-in-exit-dwell-updates must be non-negative.")
    if args.phase_hold_arm_threshold_hz < 0.0:
        raise ValueError("--phase-hold-arm-threshold-hz must be non-negative.")
    if args.phase_hold_arm_dwell_updates <= 0:
        raise ValueError("--phase-hold-arm-dwell-updates must be at least 1.")
    if args.phase_hold_exit_dwell_updates < 0:
        raise ValueError("--phase-hold-exit-dwell-updates must be non-negative.")
    if args.phase_hold_base_lock_dwell_updates < 0:
        raise ValueError("--phase-hold-base-lock-dwell-updates must be non-negative.")
    if args.phase_hold_base_lock_threshold_hz < 0.0:
        raise ValueError("--phase-hold-base-lock-threshold-hz must be non-negative.")
    if args.phase_hold_max_freq_hz <= 0.0:
        raise ValueError("--phase-hold-max-freq-hz must be positive.")
    if not (0.0 < args.phase_hold_smoothing_alpha <= 1.0):
        raise ValueError("--phase-hold-smoothing-alpha must lie in (0, 1].")
    if not (0.0 < args.phase_error_filter_alpha <= 1.0):
        raise ValueError("--phase-error-filter-alpha must lie in (0, 1].")
    if not (0.0 <= args.phase_error_min_confidence <= 1.0):
        raise ValueError("--phase-error-min-confidence must lie in [0, 1].")
    if args.phase_error_max_residual_rad < 0.0:
        raise ValueError("--phase-error-max-residual-rad must be non-negative.")
    if args.phase_cycle_max_branch_error_rad < 0.0:
        raise ValueError("--phase-cycle-max-branch-error-rad must be non-negative.")
    if args.phase_target_slew_rate_rad_per_s < 0.0:
        raise ValueError("--phase-target-slew-rate-rad-per-s must be non-negative.")
    if args.phase_integral_gain_hz_per_rad_s < 0.0:
        raise ValueError("--phase-integral-gain-hz-per-rad-s must be non-negative.")
    if args.phase_integral_max_hz < 0.0:
        raise ValueError("--phase-integral-max-hz must be non-negative.")
    if args.phase_slew_gain_hz_per_rad is not None and args.phase_slew_gain_hz_per_rad < 0.0:
        raise ValueError("--phase-slew-gain-hz-per-rad must be non-negative.")
    if args.phase_slew_max_freq_hz is not None and args.phase_slew_max_freq_hz <= 0.0:
        raise ValueError("--phase-slew-max-freq-hz must be positive.")
    if args.phase_slew_enter_error_rad < 0.0:
        raise ValueError("--phase-slew-enter-error-rad must be non-negative.")
    if args.phase_slew_exit_error_rad < 0.0:
        raise ValueError("--phase-slew-exit-error-rad must be non-negative.")
    if args.phase_slew_exit_error_rad > args.phase_slew_enter_error_rad:
        raise ValueError("--phase-slew-exit-error-rad must be <= --phase-slew-enter-error-rad.")
    if args.phase_recovery_enter_dwell_updates < 1:
        raise ValueError("--phase-recovery-enter-dwell-updates must be >= 1.")
    if args.phase_recovery_exit_dwell_updates < 1:
        raise ValueError("--phase-recovery-exit-dwell-updates must be >= 1.")
    if args.phase_recovery_min_dwell_updates < 0:
        raise ValueError("--phase-recovery-min-dwell-updates must be non-negative.")
    if args.control_update_interval_s <= 0.0:
        raise ValueError("--control-update-interval-s must be positive.")
    if args.control_average_interval_s <= 0.0:
        raise ValueError("--control-average-interval-s must be positive.")
    if args.control_min_valid_blocks <= 0:
        raise ValueError("--control-min-valid-blocks must be positive.")
    if not (0.0 < args.control_max_f_off_fraction <= 1.0):
        raise ValueError("--control-max-f-off-fraction must lie in (0, 1].")
    if args.control_max_phase_residual_rad <= 0.0:
        raise ValueError("--control-max-phase-residual-rad must be positive.")
    if not (0.0 <= args.control_min_fit_r2 <= 1.0):
        raise ValueError("--control-min-fit-r2 must lie in [0, 1].")
    if args.log_flush_interval_s <= 0.0:
        raise ValueError("--log-flush-interval-s must be positive.")
    if args.base_frequency_integral_gain < 0.0:
        raise ValueError("--base-frequency-integral-gain must be non-negative.")
    if args.base_frequency_deadband_hz < 0.0:
        raise ValueError("--base-frequency-deadband-hz must be non-negative.")
    if args.base_command_slew_rate_per_s < 0.0:
        raise ValueError("--base-command-slew-rate-per-s must be non-negative.")
    if not (0.0 <= args.base_error_filter_alpha <= 1.0):
        raise ValueError("--base-error-filter-alpha must lie in [0, 1].")
    if not (0.0 <= args.base_min_coherence <= 1.0):
        raise ValueError("--base-min-coherence must lie in [0, 1].")
    if args.base_max_phase_residual_rad <= 0.0:
        raise ValueError("--base-max-phase-residual-rad must be positive.")
    if not (0.0 < args.base_max_f_off_fraction <= 1.0):
        raise ValueError("--base-max-f-off-fraction must lie in (0, 1].")
    if args.startup_check_blocks <= 0:
        raise ValueError("--startup-check-blocks must be positive.")
    if args.startup_min_coherence < 0.0 or args.startup_min_coherence > 1.0:
        raise ValueError("--startup-min-coherence must lie in [0, 1].")
    if args.startup_scan_span_hz <= 0.0:
        raise ValueError("--startup-scan-span-hz must be positive.")
    if args.startup_scan_fft_samples <= 0:
        raise ValueError("--startup-scan-fft-samples must be positive.")
    if args.startup_min_prominence_db < 0.0:
        raise ValueError("--startup-min-prominence-db must be non-negative.")
    if args.vcxo_hz_per_command is not None and abs(float(args.vcxo_hz_per_command)) < 1.0e-12:
        raise ValueError("--vcxo-hz-per-command must be non-zero when provided.")
    if not (0.0 <= args.pll_estimator_alpha <= 1.0):
        raise ValueError("--pll-estimator-alpha must lie in [0, 1].")
    if args.pll_estimator_beta < 0.0:
        raise ValueError("--pll-estimator-beta must be non-negative.")
    if args.pll_estimator_gamma < 0.0:
        raise ValueError("--pll-estimator-gamma must be non-negative.")
    if not (0.0 < args.pll_cycle_slip_threshold_rad <= math.pi):
        raise ValueError("--pll-cycle-slip-threshold-rad must lie in (0, pi].")
    if args.pll_frequency_proportional_gain < 0.0:
        raise ValueError("--pll-frequency-proportional-gain must be non-negative.")
    if args.pll_frequency_integral_gain < 0.0:
        raise ValueError("--pll-frequency-integral-gain must be non-negative.")
    if args.acquisition_frequency_proportional_gain < 0.0:
        raise ValueError("--acquisition-frequency-proportional-gain must be non-negative.")
    if args.acquisition_frequency_integral_gain < 0.0:
        raise ValueError("--acquisition-frequency-integral-gain must be non-negative.")
    if args.acquisition_frequency_deadband_hz < 0.0:
        raise ValueError("--acquisition-frequency-deadband-hz must be non-negative.")
    if args.acquisition_command_slew_rate_per_s < 0.0:
        raise ValueError("--acquisition-command-slew-rate-per-s must be non-negative.")
    if args.initial_frequency_step_blocks < 2:
        raise ValueError("--initial-frequency-step-blocks must be at least 2.")
    if args.initial_frequency_step_min_valid_blocks < 1:
        raise ValueError("--initial-frequency-step-min-valid-blocks must be at least 1.")
    if args.initial_frequency_step_min_valid_blocks > args.initial_frequency_step_blocks - 1:
        raise ValueError("--initial-frequency-step-min-valid-blocks must be <= --initial-frequency-step-blocks - 1.")
    if not (0.0 <= args.initial_frequency_step_min_coherence <= 1.0):
        raise ValueError("--initial-frequency-step-min-coherence must lie in [0, 1].")
    if args.initial_frequency_step_gain < 0.0:
        raise ValueError("--initial-frequency-step-gain must be non-negative.")
    if args.initial_frequency_step_passes < 1:
        raise ValueError("--initial-frequency-step-passes must be at least 1.")
    if args.initial_frequency_step_refine_gain < 0.0:
        raise ValueError("--initial-frequency-step-refine-gain must be non-negative.")
    if args.initial_frequency_step_max_command_delta < 0.0:
        raise ValueError("--initial-frequency-step-max-command-delta must be non-negative.")
    if args.initial_frequency_step_deadband_hz < 0.0:
        raise ValueError("--initial-frequency-step-deadband-hz must be non-negative.")
    if args.initial_frequency_step_discard_blocks < 0:
        raise ValueError("--initial-frequency-step-discard-blocks must be non-negative.")
    if args.frequency_error_deadband_hz < 0.0:
        raise ValueError("--frequency-error-deadband-hz must be non-negative.")
    if args.command_slew_rate_per_s < 0.0:
        raise ValueError("--command-slew-rate-per-s must be non-negative.")
    if args.recapture_invalid_blocks <= 0:
        raise ValueError("--recapture-invalid-blocks must be at least 1.")
    if not (0.0 < args.recapture_ambiguity_fraction < 1.0):
        raise ValueError("--recapture-ambiguity-fraction must lie in (0, 1).")
    if args.recapture_ambiguity_blocks <= 0:
        raise ValueError("--recapture-ambiguity-blocks must be at least 1.")
    if args.recapture_dwell_blocks <= 0:
        raise ValueError("--recapture-dwell-blocks must be at least 1.")
    if args.recapture_search_span_hz <= 0.0:
        raise ValueError("--recapture-search-span-hz must be positive.")
    if args.recapture_min_peak_prominence_db < 0.0:
        raise ValueError("--recapture-min-peak-prominence-db must be non-negative.")
    if args.recapture_release_hz < 0.0:
        raise ValueError("--recapture-release-hz must be non-negative.")

    resolve_command_defaults(args)
    apply_closed_loop_command_window(args)
    resolve_phase_hold_defaults(args)
    args.vcxo_hz_per_command = infer_vcxo_hz_per_command(args)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        tone_ref = resolve_tone_reference(args)
        out_log, log_format = build_output_log_path(args)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    stop = GracefulStop()
    stop.install()

    sdr = None
    actuator: ActuatorBase = NullActuator()
    log_file = None

    try:
        sdr = choose_sdr_backend(args)
        actuator = build_actuator(args)
        actuator.open()

        if log_format == "binary":
            log_file = open(out_log, "wb", buffering=1024 * 1024)
            writer = BinaryRowWriter(log_file, csv_fieldnames())
        else:
            log_file = open(out_log, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(log_file, fieldnames=csv_fieldnames())
            writer.writeheader()

        block_dt_s = args.block_size / args.sample_rate_hz
        freq_unambiguous_hz = 0.5 / block_dt_s
        recapture_ambiguity_hz = float(args.recapture_ambiguity_fraction) * float(
            freq_unambiguous_hz
        )

        print(f"Mode                 : {args.mode}")
        print(f"SDR backend          : {sdr.backend_name}")
        print(f"Actuator             : {actuator.actuator_name}")
        print(f"VCXO sensitivity     : {args.vcxo_hz_per_command:.6f} Hz/command")
        print(f"Tone reference source: {tone_ref.source}")
        print(f"Tone BB reference    : {tone_ref.tone_bb_hz:.6f} Hz")
        print(f"Block duration       : {block_dt_s:.6f} s")
        print(f"Unambiguous |f_off|  : < {freq_unambiguous_hz:.3f} Hz per block")
        print(f"Log format           : {log_format}")
        print(f"Log path             : {out_log}")
        if args.mode == "closed_loop":
            print(
                f"Frequency servo      : p_gain={args.pll_frequency_proportional_gain:.6f}, "
                f"i_gain={args.pll_frequency_integral_gain:.6f} 1/s, "
                f"deadband=+/-{args.frequency_error_deadband_hz:.3f} Hz, "
                f"slew={args.command_slew_rate_per_s:.3f} command/s, "
                f"start={args.command_start:.3f}, range=[{args.command_min:.3f}, {args.command_max:.3f}], "
                f"invert={int(args.invert_control)}"
            )
            print(
                f"Acquisition servo    : p_gain={args.acquisition_frequency_proportional_gain:.6f}, "
                f"i_gain={args.acquisition_frequency_integral_gain:.6f} 1/s, "
                f"deadband=+/-{args.acquisition_frequency_deadband_hz:.3f} Hz, "
                f"slew={args.acquisition_command_slew_rate_per_s:.3f} command/s"
            )
            print(
                f"Initial freq step    : enabled={int(args.initial_frequency_step_enabled)}, "
                f"blocks={int(args.initial_frequency_step_blocks)}, "
                f"min_valid={int(args.initial_frequency_step_min_valid_blocks)}, "
                f"gain={args.initial_frequency_step_gain:.3f}, "
                f"passes={int(args.initial_frequency_step_passes)}, "
                f"refine_gain={args.initial_frequency_step_refine_gain:.3f}, "
                f"max_delta={args.initial_frequency_step_max_command_delta:.3f} command"
            )
            print(
                f"Phase hold           : gain={args.phase_hold_gain_hz_per_rad:.6f} Hz/rad, "
                f"arm={int(args.phase_hold_arm_dwell_updates)} updates < {args.phase_hold_arm_threshold_hz:.3f} Hz, "
                f"exit={int(args.phase_hold_exit_dwell_updates)} updates, "
                f"base_lock={int(args.phase_hold_base_lock_dwell_updates)} updates < {args.phase_hold_base_lock_threshold_hz:.3f} Hz, "
                f"clamp=+/-{args.phase_hold_max_freq_hz:.3f} Hz, "
                f"alpha={args.phase_hold_smoothing_alpha:.3f}, "
                f"phase_err_alpha={args.phase_error_filter_alpha:.3f}, "
                f"cycle_tracking={int(args.phase_cycle_tracking_enabled)}, "
                f"cycle_branch={args.phase_cycle_max_branch_error_rad:.3f} rad, "
                f"error=wrapped"
            )
            print(
                f"Phase recovery       : gain={args.phase_slew_gain_hz_per_rad:.6f} Hz/rad, "
                f"clamp=+/-{args.phase_slew_max_freq_hz:.3f} Hz, "
                f"enter={args.phase_slew_enter_error_rad:.3f} rad, "
                f"exit={args.phase_slew_exit_error_rad:.3f} rad, "
                f"enter_dwell={args.phase_recovery_enter_dwell_updates}, "
                f"exit_dwell={args.phase_recovery_exit_dwell_updates}, "
                f"min_dwell={args.phase_recovery_min_dwell_updates}"
            )
            print(
                f"Pull-in stage        : enabled={int(args.pull_in_enabled)}, "
                f"switch={args.pull_in_switch_hz:.3f} Hz, exit={args.pull_in_exit_hz:.3f} Hz, "
                f"enter={int(args.pull_in_enter_dwell_updates)}, exit={int(args.pull_in_exit_dwell_updates)}"
            )
            print(
                f"Recapture            : enabled={int(args.recapture_enabled)}, "
                f"bad_blocks={int(args.recapture_invalid_blocks)}, "
                f"ambiguity={recapture_ambiguity_hz:.3f} Hz, "
                f"dwell={int(args.recapture_dwell_blocks)} blocks, "
                f"span=+/-{args.recapture_search_span_hz:.3f} Hz, "
                f"prominence>={args.recapture_min_peak_prominence_db:.3f} dB, "
                f"release={args.recapture_release_hz:.3f} Hz, "
                f"ambiguity_trigger={int(args.recapture_on_ambiguity)}"
            )
        print("")

        if args.mode == "vcxo_test":
            run_vcxo_test(args, tone_ref, sdr, actuator, writer, log_file, stop)
        elif args.mode == "vcxo_precision_test":
            run_vcxo_precision_test(args, tone_ref, sdr, actuator, writer, log_file, stop, out_log)
        else:
            run_open_or_closed_loop(args, tone_ref, sdr, actuator, writer, log_file, stop)

        log_file.flush()
        print("")
        print(f"Saved {log_format} log to {out_log}")
        if args.plot and log_format == "csv":
            maybe_plot_csv(out_log)
        elif args.plot:
            print("Plotting is only implemented for CSV logs; skipping --plot for binary log.")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as exc:
        print(f"\nRun failed: {exc}", file=sys.stderr)
        return 1
    finally:
        stop.restore()
        if log_file is not None:
            try:
                log_file.flush()
                log_file.close()
            except Exception:
                pass
        try:
            actuator.close()
        except Exception:
            pass
        if sdr is not None:
            try:
                sdr.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
