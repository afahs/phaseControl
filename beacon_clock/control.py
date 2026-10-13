from enum import IntEnum
import math


class State(IntEnum):
    ACQUIRE = 0
    TRACK = 1
    HOLDOVER = 2
    FAULT = 3
    MONITOR = 4


def phase_margin(config, delay_s):
    """PI + integrator model with pure delay and half-update ZOH delay.

    bandwidth_hz denotes the second-order natural frequency, not ENBW.
    """
    wn = 2 * math.pi * config.bandwidth_hz
    z = config.damping
    wc = wn * math.sqrt(2 * z*z + math.sqrt(4 * z**4 + 1))
    return math.degrees(math.atan2(2 * z * wn * wc, wn*wn)
                        - wc * (delay_s + .5 / config.control_hz))


class Controller:
    def __init__(self, config, calibration, initial_code):
        calibration.validate(config)
        if phase_margin(config, calibration.delay_s + config.max_age_s) < 45:
            raise ValueError("modeled phase margin below 45 degrees; reduce bandwidth")
        self.c, self.cal = config, calibration
        self.code = int(initial_code)
        self.state = State.ACQUIRE
        self.bias = float(self.code)
        self.integral = self.target = 0.
        self.last_t = None
        self.good = self.lock_good = 0
        self.locked = False
        self.error = 0.
        wn = 2 * math.pi * config.bandwidth_hz
        self.kp = 2 * config.damping * wn / (2 * math.pi)
        self.ki = wn*wn / (2 * math.pi)

    def hold(self):
        if self.state != State.FAULT:
            self.state = State.HOLDOVER
        self.good = self.lock_good = 0
        self.locked = False
        self.last_t = None

    def fault(self):
        self.hold()
        self.state = State.FAULT

    def update(self, m, age=0.):
        if self.state == State.FAULT:
            return self.code
        if not m.valid or age > self.c.max_age_s or age < 0:
            self.hold()
            return self.code
        if self.state == State.HOLDOVER:
            self.state = State.ACQUIRE
            self.integral = 0.
        dt = 1 / self.c.control_hz if self.last_t is None else m.t - self.last_t
        self.last_t = m.t
        if dt <= 0 or dt > 2 / self.c.control_hz:
            self.hold()
            return self.code
        c, gain = self.c, self.cal.hz_per_code
        tolerance = max(.5, abs(gain) * .75)
        if self.state == State.ACQUIRE:
            # Keep a floating accumulator; actual output is still integer/no dithering.
            self.bias = self.code - .75 * m.frequency / gain
            desired = self.bias
            self.good = self.good + 1 if abs(m.frequency) < tolerance else 0
            if self.good >= 10:
                self.state = State.TRACK
                self.target, self.bias, self.integral = m.phase, float(self.code), 0.
                desired = self.bias
        else:
            self.error = m.phase - self.target  # deliberately NOT wrapped
            candidate_integral = self.integral + self.ki * self.error * dt
            desired = self.bias - (self.kp * self.error + candidate_integral) / gain
            low, high = max(c.dac_min, self.code - c.max_step_codes), min(c.dac_max, self.code + c.max_step_codes)
            old_desired = self.bias - (self.kp * self.error + self.integral) / gain
            if low <= desired <= high or abs(desired - min(high, max(low, desired))) < abs(old_desired - min(high, max(low, old_desired))):
                self.integral = candidate_integral
            self.lock_good = self.lock_good + 1 if abs(self.error) < .5 and abs(m.frequency) < tolerance else 0
            if self.lock_good >= 10:
                self.locked = True
            if abs(self.error) > 1.0 or abs(m.frequency) > 2 * tolerance:
                self.locked = False
            if abs(self.error) > 4 * math.pi:
                self.hold()
                return self.code
        desired = max(c.dac_min, min(c.dac_max, desired))
        desired = max(self.code - c.max_step_codes, min(self.code + c.max_step_codes, desired))
        self.code = int(round(desired))
        return self.code
