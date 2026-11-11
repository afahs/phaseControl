from dataclasses import replace
import unittest
from beacon_clock.config import Config, Calibration
from beacon_clock.control import Controller, State
from beacon_clock.estimator import Measurement


class GainMapTests(unittest.TestCase):
    def setUp(self):
        self.c = Config(dac_min=500,dac_max=3600)
        self.cal = Calibration(-.8,2000,.016,self.c.center_hz,self.c.pilot_hz,
                               self.c.expected_offset_hz,self.c.sample_rate,self.c.apr_ppm,True,
                               gain_table=((500,-.5),(2000,-.8),(3600,-1.12)))

    def test_linear_gain_integral_inverse_both_signs(self):
        # Independent analytic plant: g(c) = -.4 - .0002*c.
        def f(c):
            return -.4*c-.0001*c*c
        for sign in (1,-1):
            cal = replace(self.cal,hz_per_code=self.cal.hz_per_code*sign,
                          gain_table=tuple((c,g*sign) for c,g in self.cal.gain_table))
            cal.validate(self.c)
            for start in (500,777,2000,3137,3600):
                for end in (500,1200,2000,3200,3600):
                    self.assertAlmostEqual(cal.shift_code(start,sign*(f(end)-f(start))),end,places=8)
            self.assertAlmostEqual(cal.gain_at(3137),sign*(-.4-.0002*3137))

    def test_rejects_incomplete_or_nonmonotonic_map(self):
        for table in (((600,-.5),(3600,-1.)),((500,-.5),(3500,-1.)),
                      ((500,-.5),(500,-.6),(3600,-1.)),((500,-.5),(3600,.5)),
                      ((500,-.5),(3600,float('nan')))):
            with self.assertRaises(ValueError):
                replace(self.cal,gain_table=table).validate(self.c)

    def test_scalar_equivalence_and_controller_limits(self):
        flat = replace(self.cal,gain_table=((500,-.8),(3600,-.8)))
        scalar = replace(flat,gain_table=())
        for start in (500,2000,3600):
            for delta in (-5000,-.3,0,.3,5000):
                self.assertAlmostEqual(flat.shift_code(start,delta),scalar.shift_code(start,delta),places=8)
        controller = Controller(self.c,self.cal,2000)
        for k in range(100):
            before = controller.code
            m = Measurement((k+1)/10,0.,10000.,.3,40.,.99,0.,True,0)
            controller.update(m)
            self.assertLessEqual(abs(controller.code-before),self.c.max_step_codes)
            self.assertTrue(500<=controller.code<=3600)

    def test_nonlinear_plant_acquires_and_tracks(self):
        # Plant has a known equilibrium and code-dependent gain; no RF DSP needed.
        controller = Controller(self.c,self.cal,2000)
        phase = 0.
        def plant(code):
            return -.4*code-.0001*code*code
        for k in range(3000):
            frequency = plant(controller.code)-plant(2100)
            phase += 2*3.141592653589793*frequency*.1
            controller.update(Measurement((k+1)/10,phase,frequency,.3,40.,.99,0.,True,0))
        self.assertEqual(controller.state,State.TRACK)
        self.assertTrue(controller.locked)
        self.assertEqual(controller.code,2100)
