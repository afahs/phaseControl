#!/usr/bin/env python3
"""Adaptive-NCO CW VCXO discipline experiment.

This is a separate phase-focused controller from ``cw_vcxo_discipline.py``.
The inner software NCO follows the measured CW tone so the block phasor stays
near DC. The hardware loop then steers the VCXO/DAC to drive the NCO frequency
correction and slowly-slewed NCO phase correction toward zero.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import cw_vcxo_discipline as base


@dataclass
class NcoMeasurement:
    physical_phase_rad: float
    physical_unwrapped_phase_rad: float
    residual_phase_rad: float
    residual_unwrapped_phase_rad: float
    residual_dphi_rad: float
    residual_freq_hz: float
    correction_phase_rad: float
    correction_unwrapped_phase_rad: float
    correction_freq_hz: float
    tracked_tone_bb_hz: float
    amp: float
    coherence: float
    snr_db: float
    power: float
    z_re: float
    z_im: float
    z_abs: float
    valid: bool


@dataclass
class NcoControlDecision:
    phase_error_rad: float
    target_nco_freq_hz: float
    hardware_frequency_error_hz: float
    filtered_nco_freq_hz: float
    mode: str
    just_armed: bool
    servo_allowed: bool
    acquisition_valid_count: int
    capture_count: int
    phase_integral_hz: float


@dataclass
class PhaseWindowMeasurement:
    phase_rad: float
    confidence: float
    blocks: int
    valid: bool
    step_rms_rad: float = float("nan")
    max_step_rad: float = float("nan")


class AdaptiveNcoToneTracker:
    """Software PLL that keeps the tone phasor centered near DC."""

    def __init__(
        self,
        sample_rate_hz: float,
        block_size: int,
        tone_bb_hz: float,
        alpha: float,
        beta: float,
        freq_limit_hz: float,
    ) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.block_size = int(block_size)
        self.tone_bb_hz = float(tone_bb_hz)
        self.block_dt_s = self.block_size / self.sample_rate_hz
        self.alpha = base.clamp(float(alpha), 0.0, 1.0)
        self.beta = max(0.0, float(beta))
        self.freq_limit_hz = max(1.0e-9, float(freq_limit_hz))
        self._sample_offsets = np.arange(self.block_size, dtype=np.float64)
        self._nominal_phase_rad = 0.0
        self._correction_phase_rad = 0.0
        self._correction_unwrapped_phase_rad = 0.0
        self._correction_freq_hz = 0.0
        self._prev_residual_phase_rad: Optional[float] = None
        self._residual_unwrapped_phase_rad = 0.0

    def process_block(
        self,
        iq: np.ndarray,
        update_nco: bool,
        amp_threshold: float,
        coherence_threshold: float,
    ) -> NcoMeasurement:
        if iq.shape[0] != self.block_size:
            raise ValueError(f"Expected {self.block_size} samples, got {iq.shape[0]}.")

        iq_centered = iq.astype(np.complex128, copy=False)
        iq_centered = iq_centered - np.mean(iq_centered, dtype=np.complex128)
        tracked_freq_hz = self.tone_bb_hz + self._correction_freq_hz
        phase = (
            self._nominal_phase_rad
            + self._correction_phase_rad
            + (2.0 * math.pi * tracked_freq_hz / self.sample_rate_hz) * self._sample_offsets
        )
        mixed = iq_centered * np.exp(-1j * phase)
        z = np.sum(mixed, dtype=np.complex128)
        amp = float(abs(z) / max(1, self.block_size))
        power = float(np.mean(np.abs(iq_centered) ** 2))
        coherence = base.clamp(amp / max(math.sqrt(power), 1.0e-12), 0.0, 1.0)
        residual_phase_rad = float(np.angle(z))
        applied_correction_phase_rad = float(self._correction_phase_rad)
        applied_correction_unwrapped_phase_rad = float(self._correction_unwrapped_phase_rad)

        if self._prev_residual_phase_rad is None:
            residual_dphi_rad = float("nan")
            residual_freq_hz = float("nan")
            self._residual_unwrapped_phase_rad = residual_phase_rad
        else:
            residual_dphi_rad = base.wrap_to_pi(
                residual_phase_rad - self._prev_residual_phase_rad
            )
            residual_freq_hz = residual_dphi_rad / (2.0 * math.pi * self.block_dt_s)
            self._residual_unwrapped_phase_rad += residual_dphi_rad
        self._prev_residual_phase_rad = residual_phase_rad

        measurement_valid = bool(
            np.isfinite(residual_phase_rad)
            and amp >= float(amp_threshold)
            and coherence >= float(coherence_threshold)
        )

        phase_correction_rad = self.alpha * residual_phase_rad if update_nco and measurement_valid else 0.0
        freq_correction_hz = (
            self.beta * residual_phase_rad / (2.0 * math.pi * self.block_dt_s)
            if update_nco and measurement_valid
            else 0.0
        )

        self._nominal_phase_rad = base.wrap_to_pi(
            self._nominal_phase_rad + 2.0 * math.pi * self.tone_bb_hz * self.block_dt_s
        )
        self._correction_unwrapped_phase_rad += (
            2.0 * math.pi * self._correction_freq_hz * self.block_dt_s
            + phase_correction_rad
        )
        self._correction_phase_rad = base.wrap_to_pi(self._correction_unwrapped_phase_rad)
        self._correction_freq_hz = base.clamp(
            self._correction_freq_hz + freq_correction_hz,
            -self.freq_limit_hz,
            self.freq_limit_hz,
        )

        return NcoMeasurement(
            physical_phase_rad=base.wrap_to_pi(applied_correction_phase_rad + residual_phase_rad),
            physical_unwrapped_phase_rad=float(
                applied_correction_unwrapped_phase_rad + self._residual_unwrapped_phase_rad
            ),
            residual_phase_rad=residual_phase_rad,
            residual_unwrapped_phase_rad=float(self._residual_unwrapped_phase_rad),
            residual_dphi_rad=float(residual_dphi_rad),
            residual_freq_hz=float(residual_freq_hz),
            correction_phase_rad=float(self._correction_phase_rad),
            correction_unwrapped_phase_rad=float(self._correction_unwrapped_phase_rad),
            correction_freq_hz=float(self._correction_freq_hz),
            tracked_tone_bb_hz=float(self.tone_bb_hz + self._correction_freq_hz),
            amp=amp,
            coherence=coherence,
            snr_db=base.coherence_to_snr_db(coherence),
            power=power,
            z_re=float(np.real(z)),
            z_im=float(np.imag(z)),
            z_abs=float(abs(z)),
            valid=measurement_valid,
        )


class ShortWindowPhaseDetector:
    """Short-window circular phase detector for the fast phase loop."""

    def __init__(self, window_blocks: int, min_blocks: int) -> None:
        self.window_blocks = max(1, int(window_blocks))
        self.min_blocks = max(1, min(int(min_blocks), self.window_blocks))
        self._phasors: deque[complex] = deque(maxlen=self.window_blocks)
        self._weights: deque[float] = deque(maxlen=self.window_blocks)
        self._phases: deque[float] = deque(maxlen=self.window_blocks)

    def observe(self, meas: NcoMeasurement, valid: bool) -> None:
        if not valid or not np.isfinite(meas.residual_phase_rad):
            return

        phase_rad = base.wrap_to_pi(
            float(meas.correction_phase_rad) + float(meas.residual_phase_rad)
        )
        weight = max(float(meas.amp) * max(float(meas.coherence), 1.0e-3), 1.0e-9)
        self._phasors.append(complex(math.cos(phase_rad), math.sin(phase_rad)) * weight)
        self._weights.append(weight)
        self._phases.append(phase_rad)

    def snapshot(self, confidence_threshold: float) -> PhaseWindowMeasurement:
        blocks = len(self._phasors)
        if blocks < self.min_blocks:
            return PhaseWindowMeasurement(
                phase_rad=float("nan"),
                confidence=0.0,
                blocks=blocks,
                valid=False,
                step_rms_rad=float("nan"),
                max_step_rad=float("nan"),
            )

        total_weight = float(sum(self._weights))
        if total_weight <= 0.0:
            return PhaseWindowMeasurement(
                phase_rad=float("nan"),
                confidence=0.0,
                blocks=blocks,
                valid=False,
                step_rms_rad=float("nan"),
                max_step_rad=float("nan"),
            )

        z = sum(self._phasors)
        confidence = base.clamp(abs(z) / total_weight, 0.0, 1.0)
        phase_rad = float(math.atan2(z.imag, z.real))
        phase_steps = [
            abs(base.wrap_to_pi(float(b) - float(a)))
            for a, b in zip(self._phases, list(self._phases)[1:])
        ]
        if phase_steps:
            step_rms_rad = float(math.sqrt(sum(step * step for step in phase_steps) / len(phase_steps)))
            max_step_rad = float(max(phase_steps))
        else:
            step_rms_rad = float("nan")
            max_step_rad = float("nan")
        return PhaseWindowMeasurement(
            phase_rad=phase_rad,
            confidence=float(confidence),
            blocks=blocks,
            valid=bool(confidence >= float(confidence_threshold)),
            step_rms_rad=step_rms_rad,
            max_step_rad=max_step_rad,
        )


class PhasePriorityNcoController:
    """Outer hardware objective: drive NCO correction frequency and phase to zero."""

    def __init__(
        self,
        final_phase_target_rad: Optional[float],
        capture_frequency_hz: float,
        capture_dwell_updates: int,
        acquisition_dwell_updates: int,
        acquisition_frequency_smoothing_alpha: float,
        acquisition_frequency_deadband_hz: float,
        acquisition_max_frequency_error_hz: float,
        phase_gain_hz_per_rad: float,
        phase_max_freq_hz: float,
        phase_integral_gain_hz_per_rad_s: float,
        phase_integral_max_hz: float,
        phase_integral_min_confidence: float = 0.85,
        phase_integral_max_step_rms_rad: float = 0.75,
        phase_integral_arm_updates: int = 20,
        target_slew_rate_rad_per_s: float = 0.02,
        update_interval_s: float = 0.01,
    ) -> None:
        self.final_phase_target_rad = (
            base.wrap_to_pi(final_phase_target_rad) if final_phase_target_rad is not None else None
        )
        self.capture_frequency_hz = max(0.0, float(capture_frequency_hz))
        self.capture_dwell_updates = max(1, int(capture_dwell_updates))
        self.acquisition_dwell_updates = max(1, int(acquisition_dwell_updates))
        self.acquisition_frequency_smoothing_alpha = base.clamp(
            float(acquisition_frequency_smoothing_alpha), 0.0, 1.0
        )
        self.acquisition_frequency_deadband_hz = max(0.0, float(acquisition_frequency_deadband_hz))
        self.acquisition_max_frequency_error_hz = max(0.0, float(acquisition_max_frequency_error_hz))
        self.phase_gain_hz_per_rad = max(0.0, float(phase_gain_hz_per_rad))
        self.phase_max_freq_hz = max(0.0, float(phase_max_freq_hz))
        self.phase_integral_gain_hz_per_rad_s = max(0.0, float(phase_integral_gain_hz_per_rad_s))
        self.phase_integral_max_hz = max(0.0, float(phase_integral_max_hz))
        self.phase_integral_min_confidence = base.clamp(
            float(phase_integral_min_confidence), 0.0, 1.0
        )
        self.phase_integral_max_step_rms_rad = max(0.0, float(phase_integral_max_step_rms_rad))
        self.phase_integral_arm_updates = max(1, int(phase_integral_arm_updates))
        self.target_slew_rate_rad_per_s = max(0.0, float(target_slew_rate_rad_per_s))
        self.update_interval_s = max(1.0e-9, float(update_interval_s))
        self.phase_target_rad: Optional[float] = None
        self.phase_integral_hz = 0.0
        self.filtered_nco_freq_hz: Optional[float] = None
        self.acquisition_valid_counter = 0
        self.capture_counter = 0
        self.active = False
        self.mode = "acquire"
        self.integral_good_counter = 0

    def update(
        self,
        meas: NcoMeasurement,
        update_allowed: bool,
        phase_meas: Optional[PhaseWindowMeasurement],
        dt_s: Optional[float] = None,
    ) -> NcoControlDecision:
        update_dt_s = self.update_interval_s if dt_s is None else max(float(dt_s), 1.0e-9)
        if not meas.valid or not np.isfinite(meas.correction_freq_hz):
            self.acquisition_valid_counter = 0
            self.capture_counter = 0
            self.active = False
            self.integral_good_counter = 0
            self.mode = "hold_invalid"
            return NcoControlDecision(
                phase_error_rad=float("nan"),
                target_nco_freq_hz=0.0,
                hardware_frequency_error_hz=float("nan"),
                filtered_nco_freq_hz=base.float_or_nan(self.filtered_nco_freq_hz),
                mode=self.mode,
                just_armed=False,
                servo_allowed=False,
                acquisition_valid_count=self.acquisition_valid_counter,
                capture_count=self.capture_counter,
                phase_integral_hz=self.phase_integral_hz,
            )

        self.acquisition_valid_counter += 1
        raw_freq_hz = float(meas.correction_freq_hz)
        if self.filtered_nco_freq_hz is None or not np.isfinite(self.filtered_nco_freq_hz):
            self.filtered_nco_freq_hz = raw_freq_hz
        else:
            alpha = self.acquisition_frequency_smoothing_alpha
            self.filtered_nco_freq_hz = ((1.0 - alpha) * self.filtered_nco_freq_hz) + (
                alpha * raw_freq_hz
            )

        filtered_freq_hz = float(self.filtered_nco_freq_hz)
        stable_for_servo = bool(update_allowed) and (
            self.acquisition_valid_counter >= self.acquisition_dwell_updates
        )

        if stable_for_servo and abs(filtered_freq_hz) <= self.capture_frequency_hz:
            self.capture_counter += 1
        else:
            self.capture_counter = 0
            self.active = False
            self.integral_good_counter = 0
            self.mode = "acquire"

        if self.capture_counter < self.capture_dwell_updates:
            acquisition_error_hz = 0.0
            servo_allowed = False
            if stable_for_servo and abs(filtered_freq_hz) > self.acquisition_frequency_deadband_hz:
                acquisition_error_hz = base.clamp(
                    filtered_freq_hz,
                    -self.acquisition_max_frequency_error_hz,
                    self.acquisition_max_frequency_error_hz,
                )
                servo_allowed = True
            return NcoControlDecision(
                phase_error_rad=float("nan"),
                target_nco_freq_hz=0.0,
                hardware_frequency_error_hz=float(acquisition_error_hz),
                filtered_nco_freq_hz=filtered_freq_hz,
                mode=self.mode,
                just_armed=False,
                servo_allowed=servo_allowed,
                acquisition_valid_count=self.acquisition_valid_counter,
                capture_count=self.capture_counter,
                phase_integral_hz=self.phase_integral_hz,
            )

        just_armed = False
        if self.phase_target_rad is None:
            initial_phase = (
                float(phase_meas.phase_rad)
                if phase_meas is not None and phase_meas.valid
                else float(meas.correction_phase_rad)
            )
            self.phase_target_rad = initial_phase
            self.phase_integral_hz = 0.0
            self.integral_good_counter = 0
            just_armed = True
        self.active = True

        if self.final_phase_target_rad is not None:
            delta = base.wrap_to_pi(float(self.final_phase_target_rad) - float(self.phase_target_rad))
            max_step = self.target_slew_rate_rad_per_s * self.update_interval_s
            if max_step > 0.0:
                self.phase_target_rad = base.wrap_to_pi(
                    float(self.phase_target_rad) + base.clamp(delta, -max_step, max_step)
                )
            if abs(delta) > max(1.0e-6, max_step):
                self.mode = "ramp"
            else:
                self.phase_target_rad = float(self.final_phase_target_rad)
                self.mode = "hold"
        else:
            self.mode = "hold"

        if phase_meas is None or not phase_meas.valid:
            self.mode = f"{self.mode}_no_phase"
            return NcoControlDecision(
                phase_error_rad=float("nan"),
                target_nco_freq_hz=0.0,
                hardware_frequency_error_hz=filtered_freq_hz,
                filtered_nco_freq_hz=filtered_freq_hz,
                mode=self.mode,
                just_armed=just_armed,
                servo_allowed=bool(update_allowed),
                acquisition_valid_count=self.acquisition_valid_counter,
                capture_count=self.capture_counter,
                phase_integral_hz=self.phase_integral_hz,
            )

        phase_error_rad = base.wrap_to_pi(float(phase_meas.phase_rad) - float(self.phase_target_rad))
        integral_quality_ok = bool(
            phase_meas.confidence >= self.phase_integral_min_confidence
            and np.isfinite(phase_meas.step_rms_rad)
            and phase_meas.step_rms_rad <= self.phase_integral_max_step_rms_rad
        )
        if integral_quality_ok:
            self.integral_good_counter += 1
        else:
            self.integral_good_counter = 0

        integral_enabled = bool(
            self.phase_integral_gain_hz_per_rad_s > 0.0
            and self.phase_integral_max_hz > 0.0
            and self.integral_good_counter >= self.phase_integral_arm_updates
        )
        if integral_enabled:
            self.phase_integral_hz = base.clamp(
                self.phase_integral_hz
                - (self.phase_integral_gain_hz_per_rad_s * phase_error_rad * update_dt_s),
                -self.phase_integral_max_hz,
                self.phase_integral_max_hz,
            )
        else:
            self.phase_integral_hz *= 0.98
            if abs(self.phase_integral_hz) < 1.0e-6:
                self.phase_integral_hz = 0.0

        target_nco_freq_hz = base.clamp(
            (-self.phase_gain_hz_per_rad * phase_error_rad) + self.phase_integral_hz,
            -self.phase_max_freq_hz,
            self.phase_max_freq_hz,
        )
        hardware_frequency_error_hz = filtered_freq_hz - target_nco_freq_hz
        return NcoControlDecision(
            phase_error_rad=float(phase_error_rad),
            target_nco_freq_hz=float(target_nco_freq_hz),
            hardware_frequency_error_hz=float(hardware_frequency_error_hz),
            filtered_nco_freq_hz=filtered_freq_hz,
            mode=self.mode,
            just_armed=just_armed,
            servo_allowed=bool(update_allowed),
            acquisition_valid_count=self.acquisition_valid_counter,
            capture_count=self.capture_counter,
            phase_integral_hz=self.phase_integral_hz,
        )


def limit_control_step(
    control: base.ControlUpdate,
    previous_command: float,
    max_step: float,
) -> base.ControlUpdate:
    if (
        not np.isfinite(previous_command)
        or not np.isfinite(max_step)
        or float(max_step) <= 0.0
    ):
        return control

    limited_requested = base.clamp(
        float(control.requested_command),
        float(previous_command) - float(max_step),
        float(previous_command) + float(max_step),
    )
    return base.ControlUpdate(
        applied_command=float(limited_requested),
        requested_command=float(limited_requested),
        p_term=float(control.p_term),
        i_term=float(control.i_term),
        error_hz=float(control.error_hz),
    )


def command_offset_for_frequency_offset(freq_offset_hz: float, hz_per_command: float) -> float:
    if not np.isfinite(freq_offset_hz) or abs(float(hz_per_command)) < 1.0e-12:
        return 0.0
    return float(freq_offset_hz) / float(hz_per_command)


class BufferedCsvLogger:
    def __init__(
        self,
        writer: csv.DictWriter,
        csv_file,
        flush_rows: int,
        flush_interval_s: float,
    ) -> None:
        self.writer = writer
        self.csv_file = csv_file
        self.flush_rows = max(1, int(flush_rows))
        self.flush_interval_s = max(0.0, float(flush_interval_s))
        self.rows: list[dict] = []
        self.last_flush = time.monotonic()

    def writerow(self, row: dict, wall_now: float) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.flush_rows or (
            self.flush_interval_s > 0.0
            and (float(wall_now) - self.last_flush) >= self.flush_interval_s
        ):
            self.flush()

    def flush(self) -> None:
        if self.rows:
            self.writer.writerows(self.rows)
            self.rows.clear()
        self.csv_file.flush()
        self.last_flush = time.monotonic()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive-NCO phase-priority CW VCXO discipline."
    )
    parser.add_argument("--mode", choices=["open_loop", "closed_loop"], default="closed_loop")
    parser.add_argument("--sample-rate", dest="sample_rate_hz", type=float, default=1024000.0)
    parser.add_argument("--center-freq", dest="center_freq_hz", type=float, required=True)
    parser.add_argument("--tone-rf", dest="tone_rf_hz", type=float, default=None)
    parser.add_argument("--tone-bb", "--expected-offset-hz", dest="tone_bb_hz", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--gain", default="0")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--ugradio-queue-blocks", "--ugradio-batch-blocks", dest="ugradio_queue_blocks", type=int, default=64)
    parser.add_argument("--warmup-blocks", type=int, default=8)
    parser.add_argument("--sdr-backend", default="ugradio")

    parser.add_argument("--use-mcp4725", action="store_true")
    parser.add_argument("--use-pwm", action="store_true")
    parser.add_argument("--cmd-min", "--dac-min", dest="command_min", type=float, default=None)
    parser.add_argument("--cmd-max", "--dac-max", dest="command_max", type=float, default=None)
    parser.add_argument("--cmd-start", "--dac-start", dest="command_start", type=float, default=None)
    parser.add_argument("--closed-loop-command-window", "--closed-loop-dac-window", dest="closed_loop_command_window", type=float, default=None)
    parser.add_argument("--clock-apr-ppm", type=int, default=120)
    parser.add_argument("--skip-clock-init", action="store_true")
    parser.add_argument("--mcp4725-dither", action="store_true")
    parser.add_argument("--pwm-pin", type=int, default=18)
    parser.add_argument("--pwm-frequency-hz", type=float, default=2000.0)
    parser.add_argument("--pwm-vdd", type=float, default=3.3)
    parser.add_argument("--control-settle-s", type=float, default=0.0)

    parser.add_argument("--vcxo-hz-per-command", type=float, required=True)
    parser.add_argument("--pll-frequency-proportional-gain", type=float, default=0.25)
    parser.add_argument("--pll-frequency-integral-gain", type=float, default=0.01)
    parser.add_argument("--acquisition-frequency-proportional-gain", type=float, default=0.03)
    parser.add_argument("--acquisition-frequency-integral-gain", type=float, default=0.0)
    parser.add_argument("--max-command-step", type=float, default=1.0)
    parser.add_argument("--max-center-command-step", type=float, default=0.5)
    parser.add_argument("--phase-command-gain", type=float, default=1.0)
    parser.add_argument("--control-update-interval-s", type=float, default=0.05)
    parser.add_argument("--fast-phase-update-interval-s", type=float, default=0.003)

    parser.add_argument("--nco-alpha", type=float, default=0.25)
    parser.add_argument("--nco-beta", type=float, default=0.05)
    parser.add_argument("--nco-freq-limit-hz", type=float, default=500.0)
    parser.add_argument("--acquisition-frequency-smoothing-alpha", type=float, default=0.08)
    parser.add_argument("--acquisition-frequency-deadband-hz", type=float, default=2.0)
    parser.add_argument("--acquisition-max-frequency-error-hz", type=float, default=40.0)
    parser.add_argument("--acquisition-dwell-updates", type=int, default=10)
    parser.add_argument("--capture-frequency-hz", type=float, default=20.0)
    parser.add_argument("--capture-dwell-updates", type=int, default=12)
    parser.add_argument("--phase-target-rad", type=float, default=0.0)
    parser.add_argument("--phase-target-slew-rate-rad-per-s", type=float, default=0.02)
    parser.add_argument("--phase-gain-hz-per-rad", type=float, default=0.25)
    parser.add_argument("--phase-max-freq-hz", type=float, default=0.5)
    parser.add_argument("--phase-window-s", type=float, default=0.003)
    parser.add_argument("--phase-window-min-blocks", type=int, default=2)
    parser.add_argument("--phase-window-confidence-threshold", type=float, default=0.45)
    parser.add_argument("--phase-step-rms-gate-rad", type=float, default=1.25)
    parser.add_argument("--phase-integral-gain-hz-per-rad-s", type=float, default=0.0)
    parser.add_argument("--phase-integral-max-hz", type=float, default=0.5)
    parser.add_argument("--phase-integral-min-confidence", type=float, default=0.85)
    parser.add_argument("--phase-integral-max-step-rms-rad", type=float, default=0.75)
    parser.add_argument("--phase-integral-arm-updates", type=int, default=20)

    parser.add_argument("--amp-threshold", type=float, default=0.02)
    parser.add_argument("--coherence-threshold", type=float, default=0.2)
    parser.add_argument("--status-interval-s", type=float, default=30.0)
    parser.add_argument("--log-every-blocks", type=int, default=1)
    parser.add_argument("--csv-flush-rows", type=int, default=512)
    parser.add_argument("--csv-flush-interval-s", type=float, default=1.0)
    parser.add_argument("--out-csv", default=None)
    return parser


def resolve_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.control_architecture = "adaptive_nco"
    args.invert_control = False
    args.pull_in_kp = 0.05
    base.resolve_command_defaults(args)
    base.apply_closed_loop_command_window(args)
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive.")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")
    if abs(float(args.vcxo_hz_per_command)) < 1.0e-12:
        raise ValueError("--vcxo-hz-per-command must be non-zero.")
    if args.max_command_step is not None and float(args.max_command_step) < 0.0:
        raise ValueError("--max-command-step must be non-negative.")
    if args.max_center_command_step is not None and float(args.max_center_command_step) < 0.0:
        raise ValueError("--max-center-command-step must be non-negative.")
    if float(args.phase_command_gain) < 0.0:
        raise ValueError("--phase-command-gain must be non-negative.")
    if float(args.fast_phase_update_interval_s) <= 0.0:
        raise ValueError("--fast-phase-update-interval-s must be positive.")
    if float(args.phase_window_s) <= 0.0:
        raise ValueError("--phase-window-s must be positive.")
    if float(args.phase_step_rms_gate_rad) < 0.0:
        raise ValueError("--phase-step-rms-gate-rad must be non-negative.")
    if not 0.0 <= float(args.phase_integral_min_confidence) <= 1.0:
        raise ValueError("--phase-integral-min-confidence must be in [0, 1].")
    if float(args.phase_integral_max_step_rms_rad) < 0.0:
        raise ValueError("--phase-integral-max-step-rms-rad must be non-negative.")
    if int(args.phase_integral_arm_updates) <= 0:
        raise ValueError("--phase-integral-arm-updates must be positive.")
    if int(args.log_every_blocks) <= 0:
        raise ValueError("--log-every-blocks must be positive.")
    return args


def output_csv_path(args: argparse.Namespace) -> str:
    if args.out_csv:
        path = args.out_csv
    else:
        path = os.path.join(
            "first_stage_demo",
            "logs",
            f"{base.timestamp_for_filename()}_adaptive_nco_{args.mode}.csv",
        )
    base.ensure_parent_dir(path)
    return os.path.abspath(path)


def fieldnames() -> list[str]:
    return [
        "wall_rel_s",
        "t_rel_s",
        "block_index",
        "mode",
        "control_architecture",
        "sdr_backend",
        "actuator",
        "sample_rate_hz",
        "block_size",
        "block_dt_s",
        "center_freq_hz",
        "tone_rf_hz",
        "tone_bb_ref_hz",
        "phase_rad",
        "unwrapped_phase_rad",
        "residual_phase_rad",
        "residual_unwrapped_phase_rad",
        "residual_dphi_rad",
        "residual_freq_hz",
        "nco_phase_correction_rad",
        "nco_unwrapped_phase_correction_rad",
        "nco_freq_correction_hz",
        "tracked_tone_bb_hz",
        "amp",
        "coherence",
        "snr_db",
        "power",
        "z_re",
        "z_im",
        "z_abs",
        "measurement_valid",
        "update_allowed",
        "gate_reason",
        "phase_active",
        "phase_mode",
        "phase_target_rad",
        "phase_error_rad",
        "phase_window_rad",
        "phase_window_confidence",
        "phase_window_blocks",
        "phase_window_step_rms_rad",
        "phase_window_max_step_rad",
        "phase_integral_hz",
        "target_nco_freq_hz",
        "hardware_frequency_error_hz",
        "filtered_nco_freq_hz",
        "acquisition_valid_count",
        "capture_count",
        "servo_allowed",
        "control_stage",
        "control_p_term",
        "control_i_term",
        "center_cmd",
        "phase_cmd_offset",
        "phase_frequency_offset_hz",
        "control_cmd",
        "requested_cmd",
        "fast_phase_update",
        "loop_wall_dt_s",
        "actuator_write_elapsed_s",
        "dac_code_cmd",
        "pwm_cmd",
        "dac_volts",
        "command_min",
        "command_max",
        "command_start",
        "vcxo_hz_per_command",
        "status",
    ]


def run(args: argparse.Namespace) -> str:
    tone_ref = base.resolve_tone_reference(args)
    out_csv = output_csv_path(args)
    sdr = base.choose_sdr_backend(args)
    actuator = base.build_actuator(args)
    stop = base.GracefulStop()
    stop.install()

    block_dt_s = float(args.block_size) / float(args.sample_rate_hz)
    phase_update_blocks = max(1, int(round(float(args.fast_phase_update_interval_s) / block_dt_s)))
    slow_update_blocks = max(1, int(round(float(args.control_update_interval_s) / block_dt_s)))
    phase_window_blocks = max(1, int(round(float(args.phase_window_s) / block_dt_s)))
    tracker = AdaptiveNcoToneTracker(
        sample_rate_hz=args.sample_rate_hz,
        block_size=args.block_size,
        tone_bb_hz=tone_ref.tone_bb_hz,
        alpha=args.nco_alpha,
        beta=args.nco_beta,
        freq_limit_hz=args.nco_freq_limit_hz,
    )
    phase_detector = ShortWindowPhaseDetector(
        window_blocks=phase_window_blocks,
        min_blocks=args.phase_window_min_blocks,
    )
    controller = PhasePriorityNcoController(
        final_phase_target_rad=args.phase_target_rad,
        capture_frequency_hz=args.capture_frequency_hz,
        capture_dwell_updates=args.capture_dwell_updates,
        acquisition_dwell_updates=args.acquisition_dwell_updates,
        acquisition_frequency_smoothing_alpha=args.acquisition_frequency_smoothing_alpha,
        acquisition_frequency_deadband_hz=args.acquisition_frequency_deadband_hz,
        acquisition_max_frequency_error_hz=args.acquisition_max_frequency_error_hz,
        phase_gain_hz_per_rad=args.phase_gain_hz_per_rad,
        phase_max_freq_hz=args.phase_max_freq_hz,
        phase_integral_gain_hz_per_rad_s=args.phase_integral_gain_hz_per_rad_s,
        phase_integral_max_hz=args.phase_integral_max_hz,
        phase_integral_min_confidence=args.phase_integral_min_confidence,
        phase_integral_max_step_rms_rad=args.phase_integral_max_step_rms_rad,
        phase_integral_arm_updates=args.phase_integral_arm_updates,
        target_slew_rate_rad_per_s=args.phase_target_slew_rate_rad_per_s,
        update_interval_s=phase_update_blocks * block_dt_s,
    )
    acquisition_servo = base.VcxoActuatorModel(
        hz_per_command=args.vcxo_hz_per_command,
        command_min=args.command_min,
        command_max=args.command_max,
        initial_command=args.command_start,
        proportional_gain=args.acquisition_frequency_proportional_gain,
        integral_gain=args.acquisition_frequency_integral_gain,
    )
    center_servo = base.VcxoActuatorModel(
        hz_per_command=args.vcxo_hz_per_command,
        command_min=args.command_min,
        command_max=args.command_max,
        initial_command=args.command_start,
        proportional_gain=args.pll_frequency_proportional_gain,
        integral_gain=args.pll_frequency_integral_gain,
    )

    csv_file = None
    csv_logger = None
    try:
        actuator.open()
        applied_command = (
            actuator.write_command(args.command_start, settle_s=args.control_settle_s)
            if args.mode == "closed_loop"
            else float("nan")
        )
        if args.mode == "closed_loop":
            acquisition_servo.seed_from_command(applied_command)
            center_servo.seed_from_command(applied_command)

        csv_file = open(out_csv, "w", newline="", encoding="utf-8", buffering=1024 * 1024)
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames())
        writer.writeheader()
        csv_logger = BufferedCsvLogger(
            writer=writer,
            csv_file=csv_file,
            flush_rows=args.csv_flush_rows,
            flush_interval_s=args.csv_flush_interval_s,
        )

        print(f"Mode                 : {args.mode}")
        print("Control architecture : adaptive_nco")
        print(f"SDR backend          : {sdr.backend_name}")
        print(f"Actuator             : {actuator.actuator_name}")
        print(f"Command range        : [{args.command_min:.3f}, {args.command_max:.3f}], start={args.command_start:.3f}")
        print(f"VCXO sensitivity     : {args.vcxo_hz_per_command:.6f} Hz/command")
        print(f"Tone BB reference    : {tone_ref.tone_bb_hz:.6f} Hz")
        print(f"NCO gains            : alpha={args.nco_alpha:.4f}, beta={args.nco_beta:.4f}, limit=+/-{args.nco_freq_limit_hz:.2f} Hz")
        print(f"Acquisition servo    : kp={args.acquisition_frequency_proportional_gain:.4f}, ki={args.acquisition_frequency_integral_gain:.4f}, filtered alpha={args.acquisition_frequency_smoothing_alpha:.4f}, dwell={args.acquisition_dwell_updates}, max error={args.acquisition_max_frequency_error_hz:.2f} Hz")
        print(f"Center servo         : kp={args.pll_frequency_proportional_gain:.4f}, ki={args.pll_frequency_integral_gain:.4f}, update={slow_update_blocks * block_dt_s:.4f}s, max center step={args.max_center_command_step:.3f}")
        print(f"Fast phase loop      : update={phase_update_blocks * block_dt_s:.4f}s, window={phase_window_blocks * block_dt_s:.4f}s, confidence>={args.phase_window_confidence_threshold:.3f}")
        print(f"Phase quality gate   : step_rms<={args.phase_step_rms_gate_rad:.3f} rad, integral confidence>={args.phase_integral_min_confidence:.3f}, integral step_rms<={args.phase_integral_max_step_rms_rad:.3f} rad for {args.phase_integral_arm_updates} updates")
        print(f"Phase objective      : target={args.phase_target_rad:.3f} rad, ramp={args.phase_target_slew_rate_rad_per_s:.4f} rad/s, gain={args.phase_gain_hz_per_rad:.4f} Hz/rad, clamp=+/-{args.phase_max_freq_hz:.3f} Hz, integral={args.phase_integral_gain_hz_per_rad_s:.4f} Hz/rad/s, command gain={args.phase_command_gain:.3f}")
        print(f"CSV logging          : every {args.log_every_blocks} block(s), flush rows={args.csv_flush_rows}, flush interval={args.csv_flush_interval_s:.3f}s")
        print(f"CSV log              : {out_csv}\n")

        sdr.warmup(args.block_size, args.warmup_blocks)
        wall_start = time.monotonic()
        next_status = wall_start
        last_control = base.ControlUpdate(
            applied_command=float(applied_command),
            requested_command=float(applied_command),
            p_term=float("nan"),
            i_term=0.0,
            error_hz=float("nan"),
        )
        center_command = float(applied_command)
        phase_command_offset = 0.0
        phase_frequency_offset_hz = 0.0
        last_phase_error = float("nan")
        last_phase_window = PhaseWindowMeasurement(
            phase_rad=float("nan"),
            confidence=0.0,
            blocks=0,
            valid=False,
            step_rms_rad=float("nan"),
            max_step_rad=float("nan"),
        )
        last_phase_integral_hz = 0.0
        last_target_nco_freq = 0.0
        last_hardware_error = float("nan")
        last_filtered_nco_freq = float("nan")
        last_acquisition_valid_count = 0
        last_capture_count = 0
        last_servo_allowed = False
        last_control_stage = "none"
        last_phase_mode = "idle"
        last_loop_wall = wall_start
        last_loop_dt_s = float("nan")
        last_actuator_write_elapsed_s = 0.0
        total_blocks = int(math.ceil(float(args.duration_s) / block_dt_s))

        for block_index in range(total_blocks):
            if stop.stop_requested:
                print("Stop requested, ending run.")
                break
            iq = sdr.read_block(args.block_size)
            wall_now = time.monotonic()
            last_loop_dt_s = wall_now - last_loop_wall
            last_loop_wall = wall_now
            wall_rel_s = wall_now - wall_start
            t_rel_s = (block_index + 0.5) * block_dt_s
            meas = tracker.process_block(
                iq,
                update_nco=True,
                amp_threshold=args.amp_threshold,
                coherence_threshold=args.coherence_threshold,
            )
            update_allowed = bool(meas.valid)
            gate_reason = "update_ok"
            if meas.amp < float(args.amp_threshold):
                update_allowed = False
                gate_reason = "amp_below_threshold"
            elif meas.coherence < float(args.coherence_threshold):
                update_allowed = False
                gate_reason = "coherence_below_threshold"

            phase_detector.observe(meas, update_allowed)
            fast_phase_update = bool((block_index + 1) % phase_update_blocks == 0)
            slow_center_update = bool((block_index + 1) % slow_update_blocks == 0)
            status = "observe"
            if args.mode == "closed_loop" and fast_phase_update:
                phase_window = phase_detector.snapshot(args.phase_window_confidence_threshold)
                if (
                    phase_window.valid
                    and np.isfinite(phase_window.step_rms_rad)
                    and phase_window.step_rms_rad > float(args.phase_step_rms_gate_rad)
                ):
                    phase_window = PhaseWindowMeasurement(
                        phase_rad=phase_window.phase_rad,
                        confidence=phase_window.confidence,
                        blocks=phase_window.blocks,
                        valid=False,
                        step_rms_rad=phase_window.step_rms_rad,
                        max_step_rad=phase_window.max_step_rad,
                    )
                decision = controller.update(
                    meas,
                    update_allowed=update_allowed,
                    phase_meas=phase_window,
                    dt_s=phase_update_blocks * block_dt_s,
                )
                last_phase_error = decision.phase_error_rad
                last_phase_window = phase_window
                last_phase_integral_hz = decision.phase_integral_hz
                last_target_nco_freq = decision.target_nco_freq_hz
                last_hardware_error = decision.hardware_frequency_error_hz
                last_filtered_nco_freq = decision.filtered_nco_freq_hz
                last_phase_mode = decision.mode
                last_acquisition_valid_count = decision.acquisition_valid_count
                last_capture_count = decision.capture_count
                last_servo_allowed = decision.servo_allowed
                last_control_stage = "phase" if controller.active else "acquisition"
                if decision.just_armed:
                    center_servo.seed_from_command(center_command)

                center_control = None
                if (
                    decision.servo_allowed
                    and slow_center_update
                    and np.isfinite(decision.filtered_nco_freq_hz)
                ):
                    if controller.active:
                        center_error_hz = base.clamp(
                            decision.filtered_nco_freq_hz,
                            -float(args.acquisition_max_frequency_error_hz),
                            float(args.acquisition_max_frequency_error_hz),
                        )
                        center_control = center_servo.update(
                            center_error_hz,
                            dt_s=slow_update_blocks * block_dt_s,
                        )
                        center_control = limit_control_step(
                            center_control,
                            previous_command=center_command,
                            max_step=float(args.max_center_command_step),
                        )
                        center_command = center_control.requested_command
                        center_servo.command = center_command
                    elif np.isfinite(decision.hardware_frequency_error_hz):
                        center_control = acquisition_servo.update(
                            decision.hardware_frequency_error_hz,
                            dt_s=slow_update_blocks * block_dt_s,
                        )
                        center_control = limit_control_step(
                            center_control,
                            previous_command=center_command,
                            max_step=float(args.max_center_command_step),
                        )
                        center_command = center_control.requested_command
                        acquisition_servo.command = center_command

                if controller.active and np.isfinite(decision.target_nco_freq_hz):
                    phase_frequency_offset_hz = (
                        float(args.phase_command_gain) * float(decision.target_nco_freq_hz)
                    )
                    phase_command_offset = command_offset_for_frequency_offset(
                        phase_frequency_offset_hz,
                        args.vcxo_hz_per_command,
                    )
                else:
                    phase_frequency_offset_hz = 0.0
                    phase_command_offset = 0.0

                requested_command = base.clamp(
                    center_command + phase_command_offset,
                    args.command_min,
                    args.command_max,
                )
                direct_control = base.ControlUpdate(
                    applied_command=requested_command,
                    requested_command=requested_command,
                    p_term=center_control.p_term if center_control is not None else 0.0,
                    i_term=center_control.i_term if center_control is not None else 0.0,
                    error_hz=decision.hardware_frequency_error_hz,
                )
                direct_control = limit_control_step(
                    direct_control,
                    previous_command=float(applied_command),
                    max_step=float(args.max_command_step),
                )

                if decision.servo_allowed and np.isfinite(direct_control.requested_command):
                    write_start = time.monotonic()
                    applied_command = actuator.write_command(
                        direct_control.requested_command,
                        settle_s=args.control_settle_s,
                    )
                    last_actuator_write_elapsed_s = time.monotonic() - write_start
                    direct_control.applied_command = float(applied_command)
                    last_control = direct_control
                    if decision.just_armed:
                        status = "phase_armed"
                    else:
                        suffix = "center_update" if center_control is not None else "phase_update"
                        status = f"{decision.mode}_{last_control_stage}_{suffix}"
                else:
                    status = f"{decision.mode}_{gate_reason}"
            elif args.mode == "closed_loop":
                status = f"{last_phase_mode}_hold_between_updates"
            else:
                status = "open_loop_log"

            if block_index % int(args.log_every_blocks) == 0 or fast_phase_update:
                dac_volts = actuator.command_to_volts(applied_command) if args.mode == "closed_loop" else None
                row = {
                    "wall_rel_s": wall_rel_s,
                    "t_rel_s": t_rel_s,
                    "block_index": block_index,
                    "mode": args.mode,
                    "control_architecture": "adaptive_nco",
                    "sdr_backend": sdr.backend_name,
                    "actuator": actuator.actuator_name,
                    "sample_rate_hz": args.sample_rate_hz,
                    "block_size": args.block_size,
                    "block_dt_s": block_dt_s,
                    "center_freq_hz": args.center_freq_hz,
                    "tone_rf_hz": base.float_or_nan(tone_ref.tone_rf_hz),
                    "tone_bb_ref_hz": tone_ref.tone_bb_hz,
                    "phase_rad": meas.physical_phase_rad,
                    "unwrapped_phase_rad": meas.physical_unwrapped_phase_rad,
                    "residual_phase_rad": meas.residual_phase_rad,
                    "residual_unwrapped_phase_rad": meas.residual_unwrapped_phase_rad,
                    "residual_dphi_rad": meas.residual_dphi_rad,
                    "residual_freq_hz": meas.residual_freq_hz,
                    "nco_phase_correction_rad": meas.correction_phase_rad,
                    "nco_unwrapped_phase_correction_rad": meas.correction_unwrapped_phase_rad,
                    "nco_freq_correction_hz": meas.correction_freq_hz,
                    "tracked_tone_bb_hz": meas.tracked_tone_bb_hz,
                    "amp": meas.amp,
                    "coherence": meas.coherence,
                    "snr_db": meas.snr_db,
                    "power": meas.power,
                    "z_re": meas.z_re,
                    "z_im": meas.z_im,
                    "z_abs": meas.z_abs,
                    "measurement_valid": int(meas.valid),
                    "update_allowed": int(update_allowed),
                    "gate_reason": gate_reason,
                    "phase_active": int(controller.active),
                    "phase_mode": last_phase_mode,
                    "phase_target_rad": base.float_or_nan(controller.phase_target_rad),
                    "phase_error_rad": last_phase_error,
                    "phase_window_rad": last_phase_window.phase_rad,
                    "phase_window_confidence": last_phase_window.confidence,
                    "phase_window_blocks": last_phase_window.blocks,
                    "phase_window_step_rms_rad": last_phase_window.step_rms_rad,
                    "phase_window_max_step_rad": last_phase_window.max_step_rad,
                    "phase_integral_hz": last_phase_integral_hz,
                    "target_nco_freq_hz": last_target_nco_freq,
                    "hardware_frequency_error_hz": last_hardware_error,
                    "filtered_nco_freq_hz": last_filtered_nco_freq,
                    "acquisition_valid_count": last_acquisition_valid_count,
                    "capture_count": last_capture_count,
                    "servo_allowed": int(last_servo_allowed),
                    "control_stage": last_control_stage,
                    "control_p_term": last_control.p_term,
                    "control_i_term": last_control.i_term,
                    "center_cmd": center_command,
                    "phase_cmd_offset": phase_command_offset,
                    "phase_frequency_offset_hz": phase_frequency_offset_hz,
                    "control_cmd": applied_command if args.mode == "closed_loop" else float("nan"),
                    "requested_cmd": last_control.requested_command,
                    "fast_phase_update": int(fast_phase_update),
                    "loop_wall_dt_s": last_loop_dt_s,
                    "actuator_write_elapsed_s": last_actuator_write_elapsed_s,
                    "dac_code_cmd": applied_command if actuator.command_label == "dac_code" else float("nan"),
                    "pwm_cmd": applied_command if actuator.command_label == "pwm_duty" else float("nan"),
                    "dac_volts": base.float_or_nan(dac_volts),
                    "command_min": args.command_min,
                    "command_max": args.command_max,
                    "command_start": args.command_start,
                    "vcxo_hz_per_command": args.vcxo_hz_per_command,
                    "status": status,
                }
                csv_logger.writerow(row, wall_now=wall_now)

            if wall_now >= next_status:
                print(
                    f"t={t_rel_s:7.1f}s | nco_f={meas.correction_freq_hz:9.3f} Hz | "
                    f"filt={base.format_float(last_filtered_nco_freq, width=8, precision=3)} Hz | "
                    f"nco_phase={meas.correction_phase_rad:8.3f} rad | "
                    f"phase_err={base.format_float(last_phase_error, width=8, precision=3)} rad | "
                    f"win_conf={last_phase_window.confidence:5.3f} | "
                    f"amp={meas.amp:7.4f} | coh={meas.coherence:6.3f} | "
                    f"act={status:<24} | cmd={base.format_float(applied_command, width=8, precision=3)}"
                )
                next_status = wall_now + float(args.status_interval_s)

        csv_logger.flush()
        print(f"\nSaved CSV log to {out_csv}")
        return out_csv
    finally:
        if csv_logger is not None:
            try:
                csv_logger.flush()
            except Exception:
                pass
        if csv_file is not None:
            csv_file.close()
        try:
            actuator.close()
        finally:
            sdr.close()


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = resolve_args(argv)
        run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Run failed: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
