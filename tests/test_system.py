from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np

from beacon_clock.acquisition import BufferPool, Block
from beacon_clock.config import Config, Calibration
from beacon_clock.control import Controller, State, phase_margin
from beacon_clock.estimator import Measurement, PilotEstimator
from beacon_clock.simulation import SimulatedRig
from beacon_clock.runtime import run
from beacon_clock.telemetry import Writer, read_run, VALID, DTYPE
from beacon_clock.analysis import clean_segments, analyze
from beacon_clock.characterize import characterize
from beacon_clock.cli import replay
from beacon_clock.diagnostics import spectrum


def calibration(c, gain=-.08):
    return Calibration(gain, 2048, .016, c.center_hz, c.pilot_hz,
                       c.expected_offset_hz, c.sample_rate, c.apr_ppm, True)


class AcquisitionTests(unittest.TestCase):
    def test_callback_owns_memory_and_marks_overflow(self):
        pool = BufferPool(8, 2)
        raw = bytearray(range(16))
        pool.callback(raw)
        first = pool.ready.get_nowait()
        raw[0] = 99
        self.assertEqual(first.raw[0], 0)
        pool.callback(raw)
        pool.callback(raw)
        pool.callback(raw)
        newest = pool.ready.get_nowait()
        self.assertEqual(pool.dropped_blocks, 2)
        self.assertEqual(newest.sample_index, 24)
        self.assertTrue(newest.lost)
        self.assertEqual(newest.segment, 1)
        pool.release(first)
        pool.release(newest)
        self.assertEqual(pool.free.qsize(), 4)

    def test_wrong_usb_size_is_visible(self):
        pool = BufferPool(8, 2)
        pool.callback(bytes(7))
        self.assertIsInstance(pool.error, ValueError)


class EstimatorTests(unittest.TestCase):
    def test_continuous_phase_survives_fast_frequency_changes(self):
        c = Config()
        estimator = PilotEstimator(c)
        phase_at_start = 0.
        for k in range(30):
            f = 60. if k % 2 == 0 else -60.
            index = k*c.block_samples
            t = (index+np.arange(c.block_samples))/c.sample_rate
            angle = 2*np.pi*c.expected_offset_hz*t + phase_at_start + 2*np.pi*f*(t-t[0])
            z = .3*np.exp(1j*angle)
            raw = np.empty(2*c.block_samples, dtype=np.uint8)
            raw[::2] = np.rint(127.5+128*z.real).astype(np.uint8)
            raw[1::2] = np.rint(127.5+128*z.imag).astype(np.uint8)
            m = estimator.process(Block(raw, sample_index=index))
            self.assertTrue(m.valid)
            self.assertFalse(m.slip)
            expected = phase_at_start+2*np.pi*f*(m.t-t[0])
            self.assertAlmostEqual(m.phase, expected, delta=.03)
            phase_at_start += 2*np.pi*f*c.block_s

    def test_spectrum_wraparound_and_independent_phase_windows(self):
        # A search across DC includes the final FFT bin; neighbor access must wrap.
        c = replace(Config(), expected_offset_hz=0.)
        with tempfile.TemporaryDirectory() as root:
            result = spectrum(c, SimulatedRig(c, offset=-12., drift=0),
                              Path(root)/"spectrum.json", seconds=.5)
        self.assertAlmostEqual(result["peaks"][0]["residual_hz"], -12., delta=1.)
        self.assertEqual(result["phase_windows"]["large_change_pairs"], 0)
        self.assertLess(result["phase_windows"]["wrapped_change_abs_p99_rad"], .05)

    def test_frequency_phase_and_mixer_reconstruction(self):
        c = Config()
        for f in (-1200., -7.3, 0., 413.2):
            rig = SimulatedRig(c, offset=f, drift=0)
            estimator = PilotEstimator(c)
            measurements = [estimator.process(rig.get()) for _ in range(30)]
            self.assertTrue(all(m.valid for m in measurements))
            self.assertAlmostEqual(np.mean([m.frequency for m in measurements]), f, delta=.3)
            slope = np.polyfit([m.t for m in measurements], [m.phase for m in measurements], 1)[0]/(2*np.pi)
            self.assertAlmostEqual(slope, f, delta=.1)

    def test_missing_pilot_and_gap(self):
        rig = SimulatedRig(Config(), drift=0)
        estimator = PilotEstimator(Config())
        self.assertTrue(estimator.process(rig.get()).valid)
        rig.dropout = True
        self.assertFalse(estimator.process(rig.get()).valid)
        rig.dropout = False
        rig.index += 8192*30
        m = estimator.process(rig.get())
        self.assertTrue(m.valid)
        self.assertFalse(m.slip)

    def test_clipping_rejected(self):
        block = Block(np.tile(np.array([0, 255], dtype=np.uint8), 8192))
        self.assertFalse(PilotEstimator(Config()).process(block).valid)


class ControlTests(unittest.TestCase):
    def test_both_polarities_quantized_delayed_plant(self):
        c = Config()
        for gain in (-.08, .08):
            control = Controller(c, calibration(c, gain), 2048)
            phase = 0.
            history = [2048, 2048]
            errors = []
            for k in range(3000):
                freq = 6 + gain*(history.pop(0)-2048)
                phase += 2*np.pi*freq*.1
                m = Measurement(k*.1, phase, freq, .3, 40, .99, 0, True, 0)
                code = control.update(m)
                history.append(code)
                if k > 2500:
                    errors.append(freq)
            self.assertEqual(control.state, State.TRACK)
            self.assertLess(abs(np.mean(errors)), .02)
            self.assertLess(np.std(errors), .1)

    def test_holdover_fault_and_saturation(self):
        c = Config()
        ctrl = Controller(c, calibration(c), 2048)
        bad = Measurement(0, 0, 1e6, .3, 40, 1, 0, True, 0)
        code = ctrl.update(bad)
        self.assertLessEqual(abs(code-2048), c.max_step_codes)
        self.assertEqual(ctrl.update(bad, .2), code)
        self.assertEqual(ctrl.state, State.HOLDOVER)
        ctrl.fault()
        self.assertEqual(ctrl.update(bad), code)
        self.assertEqual(ctrl.state, State.FAULT)

    def test_bad_margin_rejected(self):
        c = replace(Config(), bandwidth_hz=.3)
        cal = replace(calibration(c), delay_s=1.)
        self.assertLess(phase_margin(c, cal.delay_s), 45)
        with self.assertRaises(ValueError):
            Controller(c, cal, 2048)

    def test_bandwidth_candidates_against_physical_model(self):
        for bandwidth in (.03, .1, .3):
            c = replace(Config(), bandwidth_hz=bandwidth)
            cal = calibration(c)
            if phase_margin(c, cal.delay_s+c.max_age_s) < 45:
                with self.assertRaises(ValueError):
                    Controller(c, cal, 2048)
                continue
            ctrl = Controller(c, cal, 2048)
            phase = 0.
            delayed = 2048
            tail = []
            for k in range(4000):
                freq = 8 - .08*(delayed-2048) + .0002*k*.1
                phase += 2*np.pi*freq*.1
                delayed = ctrl.update(Measurement(k*.1,phase,freq,.3,40,1,0,True,0))
                if k > 3500:
                    tail.append(freq)
            self.assertEqual(ctrl.state, State.TRACK)
            self.assertLess(abs(np.mean(tail)), .02)
            self.assertLess(np.std(tail), .1)


def values(i):
    return (i*8192, i*.1, i*.01, .1, 1, .3, 40., .99, 0., .001, .001, 0., 2048, 1, VALID)


class TelemetryTests(unittest.TestCase):
    def test_allan_keeps_distinct_averaging_times_separate(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"run"
            c = Config()
            writer = Writer(path, c, {})
            drift = 1e-10
            for i in range(80):
                row = list(values(i))
                row[1] = i*.1 if i < 40 else 10+(i-40)*.2
                row[2] = 2*np.pi*c.pilot_hz*.5*drift*row[1]**2
                row[4] = 1 if i < 40 else 2
                writer.append(tuple(row))
                if i % 10 == 9:
                    time.sleep(.01)
            writer.close({})
            result = analyze(path)
            for p in result["allan"]:
                self.assertAlmostEqual(p["deviation"]/(drift*p["tau_s"]/np.sqrt(2)), 1., places=6)

    def test_closed_stdout_does_not_stop_binary_writer(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"run"
            with patch("builtins.print", side_effect=BrokenPipeError("disconnected")):
                writer = Writer(path, Config(), {}, status=True)
                for i in range(10):
                    writer.append(values(i))
                writer.close({})
            metadata, rows = read_run(path)
            self.assertEqual(len(rows), 10)
            self.assertIsNone(metadata["telemetry"]["error"])
            del rows

    def test_cap_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"run"
            c = replace(Config(), max_output_bytes=65536+DTYPE.itemsize*5)
            writer = Writer(path, c, {})
            for i in range(10):
                writer.append(values(i))
            writer.close({})
            meta, rows = read_run(path)
            self.assertEqual(len(rows), 5)
            self.assertTrue(meta["telemetry"]["capped"])
            self.assertLessEqual(sum(p.stat().st_size for p in path.iterdir()), c.max_output_bytes)
            self.assertAlmostEqual(rows[2]["phase"], .02)
            del rows

    def test_stalled_writer_is_bounded_and_nonblocking(self):
        with tempfile.TemporaryDirectory() as root:
            writer = Writer(Path(root)/"run", replace(Config(), writer_batches=1), {}, stall_s=.15)
            start = time.monotonic()
            for i in range(1000):
                writer.append(values(i))
            self.assertLess(time.monotonic()-start, .1)
            self.assertGreater(writer.dropped, 900)
            writer.close({})

    def test_analysis_does_not_bridge_dropped_records(self):
        rows = np.empty(8, dtype=DTYPE)
        for i in range(8):
            rows[i] = (i, *values(i), np.nan, np.nan, np.nan, 0)
        rows[4:]["record"] += 2
        chunks = clean_segments(rows)
        self.assertEqual([len(x) for x in chunks], [4,4])

    def test_allan_deviation_matches_known_frequency_ramp(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"run"
            c = Config()
            writer = Writer(path, c, {})
            fractional_drift = 1e-10
            for i in range(60):
                row = list(values(i))
                row[2] = 2*np.pi*c.pilot_hz*.5*fractional_drift*row[1]**2
                writer.append(tuple(row))
                if i % 10 == 9:
                    time.sleep(.01)
            writer.close({})
            result = analyze(path)
            self.assertGreater(len(result["allan"]), 2)
            for point in result["allan"]:
                expected = fractional_drift*point["tau_s"]/np.sqrt(2)
                self.assertAlmostEqual(point["deviation"]/expected, 1., places=6)


class IntegrationTests(unittest.TestCase):
    def test_rejected_stale_command_reacquires_from_actual_dac(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        class Clock:
            base = offset = 0.
            calls = 0
            injected = False
            def read(self):
                self.calls += 1
                if self.calls == 2 and not self.injected:
                    self.offset += .2  # latency between estimate and first write
                    self.injected = True
                return self.base+self.offset+.001
        clock = Clock()
        class Rig(SimulatedRig):
            def get(self, timeout=1.):
                clock.base = self.index/self.c.sample_rate
                clock.calls = 0
                block = super().get(timeout)
                block.arrival = clock.base+clock.offset
                return block
            def write(self, code):
                if abs(code-self.code) > self.c.max_step_codes:
                    raise OSError("proposal diverged from actual DAC")
                super().write(code)
        fake_time = SimpleNamespace(monotonic=clock.read, perf_counter=time.perf_counter,
                                    process_time=time.process_time)
        c = Config()
        rig = Rig(c, offset=80)
        with tempfile.TemporaryDirectory() as root:
            with patch("beacon_clock.runtime.time", fake_time):
                summary = run(c, rig, Path(root)/"run", 1., rig, calibration(c), "discipline")
        self.assertTrue(clock.injected)
        self.assertEqual(summary["stale_blocks"], 1)
        self.assertIsNone(summary["failure"])
        self.assertGreater(summary["dac_writes"], 0)

    def test_cli_simulation_uses_supplied_measured_plant(self):
        from contextlib import redirect_stdout
        from dataclasses import asdict
        from io import StringIO
        from beacon_clock.cli import main
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"run"
            cp = Path(root)/"calibration.json"
            cal = replace(calibration(Config(), gain=-3.17), delay_s=.104)
            cp.write_text(json.dumps(asdict(cal)))
            with redirect_stdout(StringIO()):
                result = main(["discipline", "--simulate", "--duration", "1",
                               "--calibration", str(cp), "--output", str(path)])
            self.assertEqual(result, 0)
            meta, rows = read_run(path)
            self.assertEqual(meta["summary"]["source_settings"]["actual_hz_per_code"], cal.hz_per_code)
            self.assertEqual(meta["summary"]["source_settings"]["delay_s"], cal.delay_s)
            del rows

    def test_i2c_failure_faults_and_stops_updates(self):
        class BrokenDac:
            def read(self):
                return 2048
            def write(self, code):
                raise OSError("injected I2C error")
        with tempfile.TemporaryDirectory() as root:
            c = Config()
            summary = run(c, SimulatedRig(c), Path(root)/"run", 2, BrokenDac(),
                          calibration(c), "discipline")
            self.assertIn("injected I2C error", summary["failure"])
            self.assertEqual(summary["final_state"], int(State.FAULT))
            self.assertEqual(summary["dac_writes"], 0)

    def test_disabled_telemetry_keeps_summary(self):
        with tempfile.TemporaryDirectory() as root:
            c = replace(Config(), telemetry_enabled=False)
            path = Path(root)/"run"
            summary = run(c, SimulatedRig(c), path, 1)
            metadata, rows = read_run(path)
            self.assertEqual(len(rows), 0)
            self.assertFalse(metadata["telemetry"]["enabled"])
            self.assertGreater(summary["blocks"], 100)

    def test_discipline_and_deterministic_replay(self):
        with tempfile.TemporaryDirectory() as root:
            c = Config()
            rig = SimulatedRig(c)
            path = Path(root)/"run"
            cal = calibration(c)
            summary = run(c, rig, path, 30, rig, cal, "discipline")
            self.assertIsNone(summary["failure"])
            self.assertGreater(summary["dac_writes"], 0)
            meta, rows = read_run(path)
            self.assertLess(abs(np.mean(rows[-50:]["frequency"])), .2)
            del rows
            from dataclasses import asdict
            cp = Path(root)/"calibration.json"
            cp.write_text(json.dumps(asdict(cal)))
            a = replay(path, cp, Path(root)/"a.json")
            b = replay(path, cp, Path(root)/"b.json")
            self.assertEqual(a, b)

    def test_characterization_recovers_slope_and_restores_code(self):
        with tempfile.TemporaryDirectory() as root:
            c = Config()
            rig = SimulatedRig(c, drift=.02)
            result = characterize(c, rig, rig, Path(root)/"run", cycles=3, dwell=2.)
            self.assertTrue(result["calibration"]["valid"])
            self.assertAlmostEqual(result["calibration"]["hz_per_code"], rig.gain, delta=.02)
            self.assertEqual(rig.code, 2048)


if __name__ == "__main__":
    unittest.main()
