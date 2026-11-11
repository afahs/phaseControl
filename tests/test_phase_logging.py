import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from beacon_clock.config import Config, Calibration
from beacon_clock.control import Controller, State, LockLoss
from beacon_clock.estimator import Measurement
from beacon_clock.telemetry import DTYPE_V1, read_run
from beacon_clock.simulation import SimulatedRig
from beacon_clock.runtime import run


class PhaseLoggingTests(unittest.TestCase):
    def test_loss_reason_is_a_transition_event(self):
        c = Config()
        cal = Calibration(-.08, 2048, .016, c.center_hz, c.pilot_hz,
                          c.expected_offset_hz, c.sample_rate, c.apr_ppm, True)
        controller = Controller(c, cal, 2048)
        controller.state = State.TRACK
        controller.target = 10.
        controller.locked = True
        m = Measurement(.1, 12., 2., .3, 40., .99, 0., True, 1)
        controller.update(m)
        self.assertEqual(controller.pending_loss, LockLoss.PHASE | LockLoss.FREQUENCY)
        self.assertEqual(controller.loss_count, 1)
        controller.hold(LockLoss.GAP)
        self.assertEqual(controller.loss_count, 1)
        controller.locked = True
        controller.hold(LockLoss.GAP)
        self.assertEqual(controller.loss_counts['gap'], 1)

    def test_legacy_binary_remains_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'run.json').write_text(json.dumps({'version': 1}))
            rows = np.zeros(2, dtype=DTYPE_V1)
            rows['phase'] = [1., 2.]
            (path / 'telemetry.bin').write_bytes(b'BCLK001\n' + rows.tobytes())
            _, loaded = read_run(path)
            np.testing.assert_array_equal(loaded['phase'], [1., 2.])
            del loaded

    def test_live_entry_point_records_target_and_error(self):
        c = Config(telemetry_hz=125)
        cal = Calibration(-.08, 2048, .016, c.center_hz, c.pilot_hz,
                          c.expected_offset_hz, c.sample_rate, c.apr_ppm, True)
        with tempfile.TemporaryDirectory() as folder:
            rig = SimulatedRig(c)
            path = Path(folder) / 'run'
            result = run(c, rig, path, 15, rig, cal, 'discipline')
            self.assertIsNone(result['failure'])
            meta, rows = read_run(path)
            self.assertEqual(meta['version'], 2)
            tracking = rows['state'] == State.TRACK
            self.assertTrue(tracking.any())
            np.testing.assert_allclose(rows['phase_error'][tracking],
                                       rows['phase'][tracking] - rows['phase_target'][tracking])
            self.assertTrue(np.isnan(rows['phase_target'][~tracking]).all())
            del rows
