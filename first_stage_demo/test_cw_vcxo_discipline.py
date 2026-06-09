#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import cw_vcxo_adaptive_nco_discipline as adaptive
import cw_vcxo_discipline as demo


class FakeActuator(demo.ActuatorBase):
    actuator_name = "fake"
    command_label = "fake_cmd"

    def __init__(self, initial_command: float = 0.0) -> None:
        self.command = float(initial_command)

    def write_command(self, command: float, settle_s: float = 0.0) -> float:
        self.command = float(command)
        return float(self.command)


class FakeSdr(demo.SDRBase):
    backend_name = "synthetic"

    def __init__(
        self,
        sample_rate_hz: float,
        block_size: int,
        actuator: FakeActuator,
        command_start: float,
        initial_offset_hz: float,
        hz_per_command: float,
    ) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.block_size = int(block_size)
        self.actuator = actuator
        self.command_start = float(command_start)
        self.initial_offset_hz = float(initial_offset_hz)
        self.hz_per_command = float(hz_per_command)
        self.phase_rad = 0.0

    def read_block(self, block_size: int) -> np.ndarray:
        if int(block_size) != self.block_size:
            raise ValueError("Unexpected block size for synthetic SDR.")

        residual_hz = self.initial_offset_hz + (
            self.hz_per_command * (self.actuator.command - self.command_start)
        )
        n = np.arange(self.block_size, dtype=np.float64)
        phase = self.phase_rad + (
            2.0 * math.pi * residual_hz * n / self.sample_rate_hz
        )
        iq = np.exp(1j * phase).astype(np.complex64)
        self.phase_rad = float(
            self.phase_rad
            + (2.0 * math.pi * residual_hz * self.block_size / self.sample_rate_hz)
        )
        return iq


def nco_measurement(freq_hz: float, valid: bool = True) -> adaptive.NcoMeasurement:
    return adaptive.NcoMeasurement(
        physical_phase_rad=0.3,
        physical_unwrapped_phase_rad=0.3,
        residual_phase_rad=0.1,
        residual_unwrapped_phase_rad=0.1,
        residual_dphi_rad=0.0,
        residual_freq_hz=0.0,
        correction_phase_rad=0.2,
        correction_unwrapped_phase_rad=0.2,
        correction_freq_hz=float(freq_hz),
        tracked_tone_bb_hz=float(freq_hz),
        amp=0.1,
        coherence=0.9,
        snr_db=0.0,
        power=0.0,
        z_re=0.0,
        z_im=0.0,
        z_abs=0.0,
        valid=valid,
    )


def clock_estimate(phase_rad: float, frequency_hz: float = 0.0) -> demo.ClockEstimate:
    return demo.ClockEstimate(
        phase_rad=float(phase_rad),
        unwrapped_phase_rad=float(phase_rad),
        frequency_hz=float(frequency_hz),
        drift_hz_per_s=0.0,
        phase_residual_rad=0.0,
        confidence=1.0,
        valid=True,
        cycle_slip=False,
    )


class AdaptiveNcoControllerTests(unittest.TestCase):
    def test_frequency_offset_to_command_uses_measured_slope_sign(self) -> None:
        self.assertAlmostEqual(
            adaptive.command_offset_for_frequency_offset(1.0, -4.0),
            -0.25,
        )
        self.assertAlmostEqual(
            adaptive.command_offset_for_frequency_offset(-1.0, -4.0),
            0.25,
        )

    def test_acquisition_waits_for_valid_dwell_before_servo(self) -> None:
        controller = adaptive.PhasePriorityNcoController(
            final_phase_target_rad=0.0,
            capture_frequency_hz=20.0,
            capture_dwell_updates=3,
            acquisition_dwell_updates=3,
            acquisition_frequency_smoothing_alpha=0.25,
            acquisition_frequency_deadband_hz=2.0,
            acquisition_max_frequency_error_hz=40.0,
            phase_gain_hz_per_rad=0.25,
            phase_max_freq_hz=0.5,
            phase_integral_gain_hz_per_rad_s=0.0,
            phase_integral_max_hz=0.5,
            target_slew_rate_rad_per_s=0.02,
            update_interval_s=0.05,
        )

        first = controller.update(nco_measurement(100.0), update_allowed=True, phase_meas=None)
        second = controller.update(nco_measurement(100.0), update_allowed=True, phase_meas=None)
        third = controller.update(nco_measurement(100.0), update_allowed=True, phase_meas=None)

        self.assertFalse(first.servo_allowed)
        self.assertFalse(second.servo_allowed)
        self.assertTrue(third.servo_allowed)
        self.assertEqual(third.mode, "acquire")
        self.assertAlmostEqual(third.hardware_frequency_error_hz, 40.0)
        self.assertFalse(controller.active)

    def test_phase_arms_after_filtered_capture_dwell(self) -> None:
        controller = adaptive.PhasePriorityNcoController(
            final_phase_target_rad=0.0,
            capture_frequency_hz=20.0,
            capture_dwell_updates=3,
            acquisition_dwell_updates=1,
            acquisition_frequency_smoothing_alpha=1.0,
            acquisition_frequency_deadband_hz=2.0,
            acquisition_max_frequency_error_hz=40.0,
            phase_gain_hz_per_rad=0.25,
            phase_max_freq_hz=0.5,
            phase_integral_gain_hz_per_rad_s=0.0,
            phase_integral_max_hz=0.5,
            target_slew_rate_rad_per_s=0.02,
            update_interval_s=0.05,
        )

        phase_meas = adaptive.PhaseWindowMeasurement(
            phase_rad=0.2,
            confidence=1.0,
            blocks=5,
            valid=True,
        )
        decisions = [
            controller.update(nco_measurement(1.0), update_allowed=True, phase_meas=phase_meas)
            for _ in range(3)
        ]

        self.assertFalse(decisions[0].just_armed)
        self.assertFalse(decisions[1].just_armed)
        self.assertTrue(decisions[2].just_armed)
        self.assertTrue(controller.active)
        self.assertEqual(decisions[2].mode, "ramp")


class PullInClosedLoopTests(unittest.TestCase):
    def test_phase_controller_uses_recovery_then_hold_mode(self) -> None:
        controller = demo.PhaseDisciplineController(
            phase_target_rad=None,
            target_is_explicit=False,
            arm_threshold_hz=100.0,
            arm_dwell_updates=1,
            hold_exit_dwell_updates=25,
            pull_in_enabled=False,
            switch_hz=30.0,
            exit_hz=100.0,
            enter_dwell_updates=1,
            exit_dwell_updates=0,
            hold_gain_hz_per_rad=0.25,
            hold_max_freq_hz=1.0,
            slew_gain_hz_per_rad=2.0,
            slew_max_freq_hz=8.0,
            slew_enter_error_rad=0.75,
            slew_exit_error_rad=0.25,
            recovery_enter_dwell_updates=1,
            recovery_exit_dwell_updates=1,
            recovery_min_dwell_updates=0,
            smoothing_alpha=1.0,
            target_slew_rate_rad_per_s=0.0,
            phase_integral_gain_hz_per_rad_s=0.0,
            phase_integral_max_hz=0.0,
            phase_error_filter_alpha=1.0,
            phase_error_min_confidence=0.0,
            phase_error_max_residual_rad=3.05,
            phase_cycle_tracking_enabled=True,
            phase_cycle_max_branch_error_rad=2.5,
            update_interval_s=0.002,
        )

        armed = controller.update(clock_estimate(0.0), update_valid=True)
        self.assertEqual(armed.phase_mode, "hold")
        self.assertAlmostEqual(armed.phase_gain_hz_per_rad, 0.25)
        self.assertAlmostEqual(armed.max_target_f_off_hz, 1.0)

        recovery = controller.update(clock_estimate(1.0), update_valid=True)
        self.assertEqual(recovery.phase_mode, "recover")
        self.assertAlmostEqual(recovery.phase_gain_hz_per_rad, 2.25)
        self.assertAlmostEqual(recovery.max_target_f_off_hz, 9.0)
        self.assertAlmostEqual(recovery.target_f_off_hz, -2.25)

        hold = controller.update(clock_estimate(0.1), update_valid=True)
        self.assertEqual(hold.phase_mode, "hold")
        self.assertAlmostEqual(hold.phase_gain_hz_per_rad, 0.25)
        self.assertAlmostEqual(hold.max_target_f_off_hz, 1.0)
        self.assertAlmostEqual(hold.target_f_off_hz, -0.025)

    def test_phase_controller_reclamps_filtered_target_on_hold_mode(self) -> None:
        controller = demo.PhaseDisciplineController(
            phase_target_rad=None,
            target_is_explicit=False,
            arm_threshold_hz=100.0,
            arm_dwell_updates=1,
            hold_exit_dwell_updates=25,
            pull_in_enabled=False,
            switch_hz=30.0,
            exit_hz=100.0,
            enter_dwell_updates=1,
            exit_dwell_updates=0,
            hold_gain_hz_per_rad=0.25,
            hold_max_freq_hz=1.0,
            slew_gain_hz_per_rad=2.0,
            slew_max_freq_hz=8.0,
            slew_enter_error_rad=0.75,
            slew_exit_error_rad=0.25,
            recovery_enter_dwell_updates=1,
            recovery_exit_dwell_updates=1,
            recovery_min_dwell_updates=0,
            smoothing_alpha=0.2,
            target_slew_rate_rad_per_s=0.0,
            phase_integral_gain_hz_per_rad_s=0.0,
            phase_integral_max_hz=0.0,
            phase_error_filter_alpha=1.0,
            phase_error_min_confidence=0.0,
            phase_error_max_residual_rad=3.05,
            phase_cycle_tracking_enabled=True,
            phase_cycle_max_branch_error_rad=2.5,
            update_interval_s=0.002,
        )

        controller.update(clock_estimate(0.0), update_valid=True)
        recovery = controller.update(clock_estimate(5.0), update_valid=True)
        self.assertEqual(recovery.phase_mode, "recover")
        self.assertLess(abs(recovery.target_f_off_hz), recovery.max_target_f_off_hz)

        hold = controller.update(clock_estimate(0.1), update_valid=True)
        self.assertEqual(hold.phase_mode, "hold")
        self.assertLessEqual(abs(hold.target_f_off_hz), hold.max_target_f_off_hz)

    def test_phase_controller_recovery_dwell_delays_mode_changes(self) -> None:
        controller = demo.PhaseDisciplineController(
            phase_target_rad=None,
            target_is_explicit=False,
            arm_threshold_hz=100.0,
            arm_dwell_updates=1,
            hold_exit_dwell_updates=25,
            pull_in_enabled=False,
            switch_hz=30.0,
            exit_hz=100.0,
            enter_dwell_updates=1,
            exit_dwell_updates=0,
            hold_gain_hz_per_rad=0.25,
            hold_max_freq_hz=1.0,
            slew_gain_hz_per_rad=2.0,
            slew_max_freq_hz=8.0,
            slew_enter_error_rad=0.75,
            slew_exit_error_rad=0.25,
            recovery_enter_dwell_updates=3,
            recovery_exit_dwell_updates=2,
            recovery_min_dwell_updates=4,
            smoothing_alpha=1.0,
            target_slew_rate_rad_per_s=0.0,
            phase_integral_gain_hz_per_rad_s=0.0,
            phase_integral_max_hz=0.0,
            phase_error_filter_alpha=1.0,
            phase_error_min_confidence=0.0,
            phase_error_max_residual_rad=3.05,
            phase_cycle_tracking_enabled=True,
            phase_cycle_max_branch_error_rad=2.5,
            update_interval_s=0.002,
        )

        controller.update(clock_estimate(0.0), update_valid=True)
        first_high = controller.update(clock_estimate(1.0), update_valid=True)
        second_high = controller.update(clock_estimate(1.0), update_valid=True)
        third_high = controller.update(clock_estimate(1.0), update_valid=True)
        self.assertEqual(first_high.phase_mode, "hold")
        self.assertEqual(second_high.phase_mode, "hold")
        self.assertEqual(third_high.phase_mode, "recover")

        # The first low-error samples do not exit because min dwell has not elapsed
        # and exit dwell requires two accepted low-error updates.
        low_1 = controller.update(clock_estimate(0.1), update_valid=True)
        low_2 = controller.update(clock_estimate(0.1), update_valid=True)
        low_3 = controller.update(clock_estimate(0.1), update_valid=True)
        low_4 = controller.update(clock_estimate(0.1), update_valid=True)
        self.assertEqual(low_1.phase_mode, "recover")
        self.assertEqual(low_2.phase_mode, "recover")
        self.assertEqual(low_3.phase_mode, "recover")
        self.assertEqual(low_4.phase_mode, "hold")

    def test_phase_controller_recovery_entry_requires_stable_error_sign(self) -> None:
        controller = demo.PhaseDisciplineController(
            phase_target_rad=None,
            target_is_explicit=False,
            arm_threshold_hz=100.0,
            arm_dwell_updates=1,
            hold_exit_dwell_updates=25,
            pull_in_enabled=False,
            switch_hz=30.0,
            exit_hz=100.0,
            enter_dwell_updates=1,
            exit_dwell_updates=0,
            hold_gain_hz_per_rad=0.25,
            hold_max_freq_hz=1.0,
            slew_gain_hz_per_rad=2.0,
            slew_max_freq_hz=8.0,
            slew_enter_error_rad=0.75,
            slew_exit_error_rad=0.25,
            recovery_enter_dwell_updates=3,
            recovery_exit_dwell_updates=1,
            recovery_min_dwell_updates=0,
            smoothing_alpha=1.0,
            target_slew_rate_rad_per_s=0.0,
            phase_integral_gain_hz_per_rad_s=0.0,
            phase_integral_max_hz=0.0,
            phase_error_filter_alpha=1.0,
            phase_error_min_confidence=0.0,
            phase_error_max_residual_rad=3.05,
            phase_cycle_tracking_enabled=True,
            phase_cycle_max_branch_error_rad=2.5,
            update_interval_s=0.002,
        )

        controller.update(clock_estimate(0.0), update_valid=True)
        positive = controller.update(clock_estimate(1.0), update_valid=True)
        sign_flip = controller.update(clock_estimate(-1.0), update_valid=True)
        second_negative = controller.update(clock_estimate(-1.0), update_valid=True)
        third_negative = controller.update(clock_estimate(-1.0), update_valid=True)

        self.assertEqual(positive.phase_mode, "hold")
        self.assertEqual(sign_flip.phase_mode, "hold")
        self.assertEqual(second_negative.phase_mode, "hold")
        self.assertEqual(third_negative.phase_mode, "recover")

    def test_phase_integral_updates_from_valid_active_modes(self) -> None:
        controller = demo.PhaseDisciplineController(
            phase_target_rad=None,
            target_is_explicit=False,
            arm_threshold_hz=100.0,
            arm_dwell_updates=1,
            hold_exit_dwell_updates=25,
            pull_in_enabled=False,
            switch_hz=30.0,
            exit_hz=100.0,
            enter_dwell_updates=1,
            exit_dwell_updates=0,
            hold_gain_hz_per_rad=0.25,
            hold_max_freq_hz=2.0,
            slew_gain_hz_per_rad=2.0,
            slew_max_freq_hz=8.0,
            slew_enter_error_rad=0.75,
            slew_exit_error_rad=0.25,
            recovery_enter_dwell_updates=1,
            recovery_exit_dwell_updates=1,
            recovery_min_dwell_updates=0,
            smoothing_alpha=1.0,
            target_slew_rate_rad_per_s=0.0,
            phase_integral_gain_hz_per_rad_s=1.0,
            phase_integral_max_hz=1.0,
            phase_error_filter_alpha=1.0,
            phase_error_min_confidence=0.0,
            phase_error_max_residual_rad=3.05,
            phase_cycle_tracking_enabled=True,
            phase_cycle_max_branch_error_rad=2.5,
            update_interval_s=0.1,
        )

        controller.update(clock_estimate(0.0), update_valid=True)
        hold_1 = controller.update(clock_estimate(0.2), update_valid=True)
        self.assertEqual(hold_1.phase_mode, "hold")
        self.assertAlmostEqual(controller.phase_integral_hz, -0.02)

        recovery = controller.update(clock_estimate(1.0), update_valid=True)
        self.assertEqual(recovery.phase_mode, "recover")
        self.assertAlmostEqual(controller.phase_integral_hz, -0.12)

        hold_2 = controller.update(clock_estimate(0.1), update_valid=True)
        self.assertEqual(hold_2.phase_mode, "hold")
        self.assertAlmostEqual(controller.phase_integral_hz, -0.13)

    def test_handoff_manager_enters_and_pushes_back(self) -> None:
        manager = demo.PullInHandoffManager(
            enabled=True,
            switch_hz=5.0,
            exit_hz=10.0,
            enter_dwell_updates=3,
            exit_dwell_updates=2,
        )

        self.assertEqual(manager.snapshot().stage, "pull_in")
        self.assertEqual(manager.observe_update(4.0, True).stage, "pull_in")
        self.assertEqual(manager.observe_update(3.0, True).stage, "pull_in")

        entered = manager.observe_update(2.0, True)
        self.assertEqual(entered.stage, "fine_lock")
        self.assertEqual(entered.switch_event, "enter_fine_lock")

        self.assertEqual(manager.observe_update(11.0, True).stage, "fine_lock")
        pushed_back = manager.observe_update(12.0, True)
        self.assertEqual(pushed_back.stage, "pull_in")
        self.assertEqual(pushed_back.switch_event, "pushback_pull_in")

    def test_synthetic_closed_loop_hands_off_and_reduces_residual(self) -> None:
        args = demo.parse_args(
            [
                "--mode",
                "closed_loop",
                "--sample-rate",
                "4096",
                "--center-freq",
                "100000000",
                "--tone-bb",
                "0",
                "--block-size",
                "32",
                "--duration-s",
                "2",
                "--cmd-min",
                "-20",
                "--cmd-max",
                "20",
                "--cmd-start",
                "0",
                "--kp",
                "0.0",
                "--ki",
                "0.02",
                "--pull-in-kp",
                "0.05",
                "--pull-in-ki",
                "0.0",
                "--pull-in-switch-hz",
                "5.0",
                "--pull-in-exit-hz",
                "12.0",
                "--pull-in-enter-dwell-updates",
                "3",
                "--pull-in-exit-dwell-updates",
                "2",
                "--recapture-ambiguity-fraction",
                "0.99",
                "--control-update-interval-s",
                "0.0078125",
                "--control-average-interval-s",
                "0.015625",
                "--control-min-valid-blocks",
                "2",
                "--amp-threshold",
                "0.01",
                "--coherence-threshold",
                "0.0",
                "--warmup-blocks",
                "0",
                "--sdr-backend",
                "ugradio",
            ]
        )
        tone_ref = demo.resolve_tone_reference(args)
        actuator = FakeActuator(initial_command=args.command_start)
        sdr = FakeSdr(
            sample_rate_hz=args.sample_rate_hz,
            block_size=args.block_size,
            actuator=actuator,
            command_start=args.command_start,
            initial_offset_hz=60.0,
            hz_per_command=20.0,
        )
        stop = demo.GracefulStop()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "synthetic_closed_loop.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=demo.csv_fieldnames())
                writer.writeheader()
                demo.run_open_or_closed_loop(
                    args=args,
                    tone_ref=tone_ref,
                    sdr=sdr,
                    actuator=actuator,
                    writer=writer,
                    csv_file=csv_file,
                    stop=stop,
                )

            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))

        stages = [row["control_stage"] for row in rows if row["control_stage"]]
        self.assertIn("pull_in", stages)
        self.assertIn("fine_lock", stages)
        self.assertTrue(
            any(row["control_stage_switch_event"] == "enter_fine_lock" for row in rows),
            "Expected a fine-lock handoff event in the synthetic run.",
        )

        finite_f = []
        for row in rows:
            try:
                finite_f.append(float(row["f_off_hz"]))
            except Exception:
                continue
        finite_f = [value for value in finite_f if math.isfinite(value)]
        self.assertGreater(len(finite_f), 10)
        self.assertLess(abs(finite_f[-1]), 5.0)
        self.assertAlmostEqual(actuator.command, -3.0, delta=1.0)

    def test_synthetic_closed_loop_recaptures_large_frequency_error(self) -> None:
        args = demo.parse_args(
            [
                "--mode",
                "closed_loop",
                "--sample-rate",
                "4096",
                "--center-freq",
                "100000000",
                "--tone-bb",
                "0",
                "--block-size",
                "32",
                "--duration-s",
                "2.5",
                "--cmd-min",
                "-30",
                "--cmd-max",
                "30",
                "--cmd-start",
                "0",
                "--kp",
                "0.0",
                "--ki",
                "0.01",
                "--pull-in-kp",
                "0.05",
                "--pull-in-ki",
                "0.0",
                "--pull-in-switch-hz",
                "5.0",
                "--pull-in-exit-hz",
                "12.0",
                "--pull-in-enter-dwell-updates",
                "3",
                "--pull-in-exit-dwell-updates",
                "2",
                "--recapture-ambiguity-fraction",
                "0.99",
                "--amp-threshold",
                "0.05",
                "--coherence-threshold",
                "0.0",
                "--recapture-invalid-blocks",
                "2",
                "--recapture-dwell-blocks",
                "8",
                "--recapture-search-span-hz",
                "600",
                "--recapture-release-hz",
                "40",
                "--warmup-blocks",
                "0",
                "--sdr-backend",
                "ugradio",
            ]
        )
        tone_ref = demo.resolve_tone_reference(args)
        actuator = FakeActuator(initial_command=args.command_start)
        sdr = FakeSdr(
            sample_rate_hz=args.sample_rate_hz,
            block_size=args.block_size,
            actuator=actuator,
            command_start=args.command_start,
            initial_offset_hz=400.0,
            hz_per_command=20.0,
        )
        stop = demo.GracefulStop()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "synthetic_recapture.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=demo.csv_fieldnames())
                writer.writeheader()
                demo.run_open_or_closed_loop(
                    args=args,
                    tone_ref=tone_ref,
                    sdr=sdr,
                    actuator=actuator,
                    writer=writer,
                    csv_file=csv_file,
                    stop=stop,
                )

            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))

        recapture_statuses = [row["status"] for row in rows if row["status"].startswith("recapture")]
        self.assertTrue(recapture_statuses, "Expected recapture status rows in the synthetic run.")
        self.assertIn("recapture_update", recapture_statuses)
        self.assertIn("recapture_complete", recapture_statuses)

        estimated_errors = []
        for row in rows:
            try:
                estimated_error = float(row["recapture_estimated_error_hz"])
            except Exception:
                continue
            if math.isfinite(estimated_error):
                estimated_errors.append(estimated_error)

        self.assertTrue(estimated_errors)
        self.assertGreater(max(abs(value) for value in estimated_errors), 300.0)
        self.assertLess(min(abs(value) for value in estimated_errors), 40.0)

        estimate_valid_values = []
        search_centers = []
        for row in rows:
            try:
                estimate_valid = float(row["recapture_estimate_valid"])
                search_center = float(row["recapture_search_center_hz"])
            except Exception:
                continue
            if math.isfinite(estimate_valid):
                estimate_valid_values.append(estimate_valid)
            if math.isfinite(search_center):
                search_centers.append(search_center)

        self.assertIn(1.0, estimate_valid_values)
        self.assertTrue(search_centers)
        self.assertLess(max(abs(value) for value in search_centers), 500.0)

        finite_f = []
        for row in rows:
            try:
                value = float(row["f_off_hz"])
            except Exception:
                continue
            if math.isfinite(value):
                finite_f.append(value)

        self.assertGreater(len(finite_f), 10)
        self.assertLess(abs(finite_f[-1]), 10.0)

    def test_synthetic_precision_test_logs_frequency_limit(self) -> None:
        args = demo.parse_args(
            [
                "--mode",
                "vcxo_precision_test",
                "--sample-rate",
                "4096",
                "--center-freq",
                "100000000",
                "--tone-bb",
                "0",
                "--block-size",
                "32",
                "--cmd-min",
                "-5",
                "--cmd-max",
                "5",
                "--cmd-start",
                "0",
                "--vcxo-precision-center",
                "0",
                "--vcxo-precision-delta",
                "1",
                "--vcxo-precision-cycles",
                "1",
                "--vcxo-precision-dwell-s",
                "0.125",
                "--vcxo-precision-settle-s",
                "0.0",
                "--vcxo-precision-search-span-hz",
                "100",
                "--vcxo-precision-fft-samples",
                "128",
                "--vcxo-precision-min-fit-blocks",
                "4",
                "--amp-threshold",
                "0.001",
                "--coherence-threshold",
                "0.0",
                "--warmup-blocks",
                "0",
            ]
        )
        tone_ref = demo.resolve_tone_reference(args)
        actuator = FakeActuator(initial_command=args.command_start)
        sdr = FakeSdr(
            sample_rate_hz=args.sample_rate_hz,
            block_size=args.block_size,
            actuator=actuator,
            command_start=args.command_start,
            initial_offset_hz=0.0,
            hz_per_command=8.0,
        )
        stop = demo.GracefulStop()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "synthetic_precision.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=demo.csv_fieldnames())
                writer.writeheader()
                demo.run_vcxo_precision_test(
                    args=args,
                    tone_ref=tone_ref,
                    sdr=sdr,
                    actuator=actuator,
                    writer=writer,
                    csv_file=csv_file,
                    stop=stop,
                    out_csv=str(csv_path),
                )

            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))

            summary_path = Path(str(csv_path).replace(".csv", "_precision_summary.txt"))
            summary_exists = summary_path.exists()

        self.assertGreater(len(rows), 0)
        freq_limits = [
            float(row["freq_unambiguous_hz"])
            for row in rows
            if math.isfinite(float(row["freq_unambiguous_hz"]))
        ]
        self.assertTrue(freq_limits)
        self.assertAlmostEqual(freq_limits[0], 64.0, places=6)
        self.assertTrue(summary_exists)
        self.assertIn("plus", {row["precision_role"] for row in rows})
        self.assertIn("minus", {row["precision_role"] for row in rows})

    def test_phase_hold_manager_arms_clamps_and_smooths(self) -> None:
        manager = demo.PhaseHoldManager(
            gain_hz_per_rad=10.0,
            slew_gain_hz_per_rad=10.0,
            phase_target_rad=0.0,
            target_is_explicit=True,
            arm_threshold_hz=25.0,
            arm_dwell_updates=2,
            max_freq_hz=0.5,
            smoothing_alpha=0.25,
            slew_max_freq_hz=0.5,
            slew_enter_error_rad=0.75,
            slew_exit_error_rad=0.25,
        )

        first = manager.observe_fine_lock_update(phase_rad=1.0, f_off_hz=20.0)
        self.assertFalse(first.active)
        self.assertEqual(first.arm_counter, 1)

        second = manager.observe_fine_lock_update(phase_rad=1.0, f_off_hz=15.0)
        self.assertTrue(second.active)
        self.assertTrue(second.just_armed)
        self.assertAlmostEqual(second.raw_freq_setpoint_hz, -10.0, places=6)
        self.assertAlmostEqual(second.filtered_freq_setpoint_hz, -0.5, places=6)

        third = manager.observe_fine_lock_update(phase_rad=-1.0, f_off_hz=10.0)
        self.assertTrue(third.active)
        self.assertFalse(third.just_armed)
        self.assertAlmostEqual(third.raw_freq_setpoint_hz, 10.0, places=6)
        self.assertAlmostEqual(third.filtered_freq_setpoint_hz, -0.25, places=6)

        manager.reset_for_pull_in()
        self.assertFalse(manager.active)
        self.assertIsNotNone(manager.phase_target_rad)

    def test_phase_hold_manager_switches_between_slew_and_hold(self) -> None:
        manager = demo.PhaseHoldManager(
            gain_hz_per_rad=1.0,
            slew_gain_hz_per_rad=4.0,
            phase_target_rad=0.0,
            target_is_explicit=True,
            arm_threshold_hz=25.0,
            arm_dwell_updates=1,
            max_freq_hz=0.25,
            smoothing_alpha=1.0,
            slew_max_freq_hz=1.0,
            slew_enter_error_rad=0.75,
            slew_exit_error_rad=0.25,
        )

        first = manager.observe_fine_lock_update(phase_rad=1.0, f_off_hz=0.0)
        self.assertTrue(first.active)
        self.assertEqual(first.control_mode, "slew")
        self.assertAlmostEqual(first.raw_freq_setpoint_hz, -4.0, places=6)
        self.assertAlmostEqual(first.filtered_freq_setpoint_hz, -1.0, places=6)

        second = manager.observe_fine_lock_update(phase_rad=0.1, f_off_hz=0.0)
        self.assertTrue(second.active)
        self.assertEqual(second.control_mode, "hold")
        self.assertAlmostEqual(second.raw_freq_setpoint_hz, -0.1, places=6)
        self.assertAlmostEqual(second.filtered_freq_setpoint_hz, -0.1, places=6)

    def test_phase_hold_manager_preserves_implicit_target_across_pull_in_reset(self) -> None:
        manager = demo.PhaseHoldManager(
            gain_hz_per_rad=2.0,
            slew_gain_hz_per_rad=2.0,
            phase_target_rad=None,
            target_is_explicit=False,
            arm_threshold_hz=25.0,
            arm_dwell_updates=1,
            max_freq_hz=1.0,
            smoothing_alpha=1.0,
            slew_max_freq_hz=1.0,
            slew_enter_error_rad=0.75,
            slew_exit_error_rad=0.25,
        )

        armed = manager.observe_fine_lock_update(phase_rad=1.25, f_off_hz=0.0)
        self.assertTrue(armed.active)
        self.assertTrue(armed.just_armed)
        self.assertAlmostEqual(manager.phase_target_rad, 1.25, places=6)

        manager.reset_for_pull_in()
        self.assertFalse(manager.active)
        self.assertAlmostEqual(manager.phase_target_rad, 1.25, places=6)

    def test_synthetic_closed_loop_phase_hold_engages_after_fine_lock_dwell(self) -> None:
        args = demo.parse_args(
            [
                "--mode",
                "closed_loop",
                "--sample-rate",
                "4096",
                "--center-freq",
                "100000000",
                "--tone-bb",
                "0",
                "--block-size",
                "32",
                "--duration-s",
                "3",
                "--cmd-min",
                "-20",
                "--cmd-max",
                "20",
                "--cmd-start",
                "0",
                "--kp",
                "0.0",
                "--ki",
                "0.02",
                "--pull-in-kp",
                "0.05",
                "--pull-in-ki",
                "0.0",
                "--pull-in-switch-hz",
                "5.0",
                "--pull-in-exit-hz",
                "12.0",
                "--pull-in-enter-dwell-updates",
                "3",
                "--pull-in-exit-dwell-updates",
                "2",
                "--recapture-ambiguity-fraction",
                "0.99",
                "--phase-hold-gain-hz-per-rad",
                "10.0",
                "--phase-target-rad",
                "1.0",
                "--phase-hold-arm-threshold-hz",
                "25.0",
                "--phase-hold-arm-dwell-updates",
                "12",
                "--phase-hold-max-freq-hz",
                "0.25",
                "--phase-hold-smoothing-alpha",
                "1.0",
                "--phase-hold-cmd-per-hz",
                "2.0",
                "--control-update-interval-s",
                "0.0078125",
                "--control-average-interval-s",
                "0.015625",
                "--control-min-valid-blocks",
                "2",
                "--amp-threshold",
                "0.01",
                "--coherence-threshold",
                "0.0",
                "--warmup-blocks",
                "0",
                "--sdr-backend",
                "ugradio",
            ]
        )
        tone_ref = demo.resolve_tone_reference(args)
        actuator = FakeActuator(initial_command=args.command_start)
        sdr = FakeSdr(
            sample_rate_hz=args.sample_rate_hz,
            block_size=args.block_size,
            actuator=actuator,
            command_start=args.command_start,
            initial_offset_hz=60.0,
            hz_per_command=20.0,
        )
        stop = demo.GracefulStop()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "synthetic_phase_hold.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=demo.csv_fieldnames())
                writer.writeheader()
                demo.run_open_or_closed_loop(
                    args=args,
                    tone_ref=tone_ref,
                    sdr=sdr,
                    actuator=actuator,
                    writer=writer,
                    csv_file=csv_file,
                    stop=stop,
                )

            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))

        fine_lock_updates = []
        first_active_index = None
        active_rows = []
        for row in rows:
            if row["control_stage"] != "fine_lock":
                continue
            meas_f = row.get("control_meas_f_off_hz", "")
            try:
                meas_f_value = float(meas_f)
            except Exception:
                meas_f_value = float("nan")
            if math.isfinite(meas_f_value):
                fine_lock_updates.append(row)
            if row.get("phase_hold_active") == "1":
                active_rows.append(row)
                if first_active_index is None:
                    first_active_index = len(fine_lock_updates)

        self.assertTrue(active_rows, "Expected phase hold to engage in the synthetic run.")
        self.assertIsNotNone(first_active_index)
        self.assertLess(
            first_active_index,
            int(args.phase_hold_arm_dwell_updates),
            "Explicit phase targets should engage as soon as the frequency estimate is captured.",
        )
        self.assertTrue(
            any("update" in row.get("status", "") for row in active_rows),
            "Expected explicit phase control to produce active update rows.",
        )

        applied_setpoints = []
        for row in active_rows:
            try:
                setpoint = float(row["phase_hold_freq_setpoint_hz"])
            except Exception:
                continue
            if math.isfinite(setpoint):
                applied_setpoints.append(setpoint)

        self.assertTrue(applied_setpoints)
        self.assertLessEqual(
            max(abs(value) for value in applied_setpoints),
            float(max(args.phase_hold_max_freq_hz, args.phase_slew_max_freq_hz)) + 1.0e-9,
        )

        nonzero_phase_targets = []
        checked_unified_rows = 0
        for row in active_rows:
            try:
                requested_cmd = float(row["requested_cmd"])
                base_cmd = float(row["control_base_cmd"])
                phase_cmd_offset = float(row["phase_hold_cmd_offset"])
                control_error_hz = float(row["control_error_hz"])
                control_meas_f_off_hz = float(row["control_meas_f_off_hz"])
                control_target_f_off_hz = float(row["control_target_f_off_hz"])
                phase_freq_setpoint_hz = float(row["phase_hold_freq_setpoint_hz"])
            except Exception:
                continue
            if not all(
                math.isfinite(value)
                for value in (
                    requested_cmd,
                    base_cmd,
                    phase_cmd_offset,
                    control_error_hz,
                    control_meas_f_off_hz,
                    control_target_f_off_hz,
                    phase_freq_setpoint_hz,
                )
            ):
                continue
            self.assertAlmostEqual(
                requested_cmd,
                base_cmd,
                places=6,
            )
            self.assertAlmostEqual(
                phase_cmd_offset,
                0.0,
                places=6,
            )
            self.assertAlmostEqual(
                control_target_f_off_hz,
                phase_freq_setpoint_hz,
                places=6,
            )
            self.assertAlmostEqual(
                control_error_hz,
                control_meas_f_off_hz - control_target_f_off_hz,
                places=6,
            )
            nonzero_phase_targets.append(abs(control_target_f_off_hz))
            checked_unified_rows += 1

        self.assertGreater(checked_unified_rows, 0)
        self.assertGreater(max(nonzero_phase_targets), 0.01)
        self.assertLessEqual(
            max(nonzero_phase_targets),
            float(max(args.phase_hold_max_freq_hz, args.phase_slew_max_freq_hz)) + 1.0e-9,
        )


if __name__ == "__main__":
    unittest.main()
