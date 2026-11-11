from enum import IntEnum, IntFlag
import math


class State(IntEnum):
    ACQUIRE = 0
    TRACK = 1
    HOLDOVER = 2
    FAULT = 3
    MONITOR = 4


class LockLoss(IntFlag):
    PHASE = 1
    FREQUENCY = 2
    INVALID = 4
    STALE = 8
    GAP = 16
    UPDATE_INTERVAL = 32
    PHASE_RANGE = 64
    FAULT = 128
    STREAM_TIMEOUT = 256



def phase_margin(config, delay_s):
    """PI + integrator, physical/transport delay, window and ZOH delays.

    bandwidth_hz denotes the second-order natural frequency, not ENBW.
    """
    wn = 2 * math.pi * config.bandwidth_hz
    z = config.damping
    wc = wn * math.sqrt(2 * z*z + math.sqrt(4 * z**4 + 1))
    return math.degrees(math.atan2(2 * z * wn * wc, wn*wn)
                        - wc * (delay_s + 1 / config.control_hz))


class Controller:
    def __init__(self, config, calibration, initial_code):
        calibration.validate(config)
        if not config.dac_min <= initial_code <= config.dac_max:
            raise ValueError("initial DAC code outside operating limits")
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
        self.target = math.nan
        self.pending_loss = 0
        self.loss_count = 0
        self.loss_counts = {reason.name.lower(): 0 for reason in LockLoss}
        wn = 2 * math.pi * config.bandwidth_hz
        self.kp = 2 * config.damping * wn / (2 * math.pi)
        self.ki = wn*wn / (2 * math.pi)

    def lose_lock(self, reason):
        if self.locked:
            self.pending_loss |= int(reason)
            self.loss_count += 1
            for flag in LockLoss:
                self.loss_counts[flag.name.lower()] += int(bool(reason & flag))
        self.locked = False

    def hold(self, reason=LockLoss.INVALID):
        self.lose_lock(reason)
        if self.state != State.FAULT:
            self.state = State.HOLDOVER
        self.good = self.lock_good = 0
        self.locked = False
        self.last_t = None

    def fault(self):
        self.hold(LockLoss.FAULT)
        self.state = State.FAULT

    def update(self, m, age=0.):
        if self.state == State.FAULT:
            return self.code
        if not m.valid or age > self.c.max_age_s or age < 0:
            self.hold(LockLoss.INVALID if not m.valid else LockLoss.STALE)
            return self.code
        if self.state == State.HOLDOVER:
            self.state = State.ACQUIRE
            self.integral = 0.
        dt = 1 / self.c.control_hz if self.last_t is None else m.t - self.last_t
        self.last_t = m.t
        if dt <= 0 or dt > 2 / self.c.control_hz:
            self.hold(LockLoss.UPDATE_INTERVAL)
            return self.code
        c, gain = self.c, self.cal.gain_at(self.code)
        tolerance = max(.5, abs(gain) * .75)
        if self.state == State.ACQUIRE:
            # Keep a floating accumulator; actual output is still integer/no dithering.
            self.bias = self.cal.shift_code(self.code, -.75 * m.frequency)
            desired = self.bias
            self.good = self.good + 1 if abs(m.frequency) < tolerance else 0
            if self.good >= 10:
                self.state = State.TRACK
                self.target, self.bias, self.integral = m.phase, float(self.code), 0.
                self.error = 0.
                desired = self.bias
        else:
            self.error = m.phase - self.target  # deliberately NOT wrapped
            candidate_integral = self.integral + self.ki * self.error * dt
            desired = self.cal.shift_code(self.bias, -(self.kp * self.error + candidate_integral))
            low, high = max(c.dac_min, self.code - c.max_step_codes), min(c.dac_max, self.code + c.max_step_codes)
            old_desired = self.cal.shift_code(self.bias, -(self.kp * self.error + self.integral))
            if low <= desired <= high or abs(desired - min(high, max(low, desired))) < abs(old_desired - min(high, max(low, old_desired))):
                self.integral = candidate_integral
            self.lock_good = self.lock_good + 1 if abs(self.error) < .5 and abs(m.frequency) < tolerance else 0
            if self.lock_good >= 10:
                self.locked = True
            if abs(self.error) > 1.0 or abs(m.frequency) > 2 * tolerance:
                reason = (LockLoss.PHASE if abs(self.error) > 1.0 else 0)
                reason |= LockLoss.FREQUENCY if abs(m.frequency) > 2 * tolerance else 0
                reason |= LockLoss.PHASE_RANGE if abs(self.error) > 4 * math.pi else 0
                self.lose_lock(reason)
            if abs(self.error) > 4 * math.pi:
                self.hold(LockLoss.PHASE_RANGE)
                return self.code
        desired = max(c.dac_min, min(c.dac_max, desired))
        desired = max(self.code - c.max_step_codes, min(self.code + c.max_step_codes, desired))
        self.code = int(round(desired))
        return self.code
