from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from beacon_clock import grid_calibrate as grid
from beacon_clock.config import Config, Calibration


class FakeDac:
    def __init__(self, config):
        self.code=3137
    def read(self):
        return self.code
    def write(self, code):
        self.code=code
    def inspect_clock(self):
        return {'compatible':True}
    def close(self):
        pass


class GridResumeTests(unittest.TestCase):
    def exercise(self, reject):
        c=Config(apr_ppm=30,telemetry_hz=125)
        def calibration(center):
            return asdict(Calibration(-.8,center,.104,c.center_hz,c.pilot_hz,
                                      c.expected_offset_hz,c.sample_rate,c.apr_ppm,True))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            centers=sorted(set(range(500,3601,100))|{3137})
            (root/'config.json').write_text(json.dumps(asdict(c)))
            accepted=[dict(center=n,calibration=calibration(n)) for n in centers[:-1]]
            (root/'grid-progress.json').write_text(json.dumps(accepted))
            (root/'code-3600').mkdir()  # Preserve a previously rejected recording.
            attempts=[]
            def characterize(config,source,dac,path,cycles,step,dwell,metadata_extra):
                attempts.append((path.name,cycles,dwell))
                path.mkdir()
                cal=calibration(dac.code)
                (path/'calibration.json').write_text(json.dumps(cal))
                return dict(calibration=cal)
            with patch.object(grid,'Dac',FakeDac), patch.object(grid,'RtlSource'), \
                 patch.object(grid,'characterize',side_effect=characterize), \
                 patch.object(grid,'cycle_gains',side_effect=ValueError('noisy') if reject else None,
                              return_value=dict(cycle_std_hz_per_code=.01)), \
                 patch.object(grid,'run',return_value={'failure':None}) as run, \
                 patch.object(grid.time,'sleep'), patch('builtins.print'), \
                 patch('sys.argv',['grid','--output',str(root),'--resume']):
                if reject:
                    with self.assertRaises(ValueError):
                        grid.main()
                    run.assert_not_called()
                    self.assertEqual(len(json.loads((root/'grid-progress.json').read_text())),32)
                else:
                    grid.main()
                    self.assertEqual(len(json.loads((root/'calibration-map.json').read_text())['gain_table']),33)
                    self.assertEqual(run.call_count,1)
                    self.assertEqual(run.call_args.kwargs['mode'],'discipline')
                self.assertEqual(attempts[0],('code-3600-retry-1',6,3.))
                self.assertTrue((root/'code-3600').is_dir())

    def test_resume_preserves_points_and_uses_longer_retry(self):
        self.exercise(False)

    def test_failed_retries_do_not_enable_feedback(self):
        self.exercise(True)
