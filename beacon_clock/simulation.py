from collections import deque
import math
import time
import numpy as np
from .acquisition import Block


class SimulatedRig:
    """Quantized actuator + delayed frequency response + continuous RF phase."""
    def __init__(self, config, seed=7, gain=-.08, offset=8., drift=.002, delay_s=.016):
        self.c = config
        self.rng = np.random.default_rng(seed)
        self.gain, self.offset, self.drift = gain, offset, drift
        self.code = 2048
        self.phase = 0.
        self.index = 0
        self.history = deque([self.code] * max(1, math.ceil(delay_s / config.block_s)))
        self.delay_s = delay_s
        self.dropout = False
        self.block = Block(np.empty(config.block_samples * 2, dtype=np.uint8))
        self.n = np.arange(config.block_samples) / config.sample_rate
        self.settings = dict(simulated=True, seed=seed, actual_hz_per_code=gain,
                             offset_hz=offset, drift_hz_per_s=drift, delay_s=delay_s)
        self.dropped_blocks = self.high_water = 0

    def start(self):
        pass

    def read(self):
        return self.code

    def write(self, code):
        if not self.c.dac_min <= code <= self.c.dac_max:
            raise ValueError("simulated DAC range violation")
        self.code = int(code)

    def get(self, timeout=1.):
        applied = self.history.popleft()
        self.history.append(self.code)
        t = self.index / self.c.sample_rate
        f = self.offset + self.gain*(applied-2048) + self.drift*t
        angles = self.phase + 2*np.pi*(self.c.expected_offset_hz+f)*self.n
        z = (.0 if self.dropout else .3) * np.exp(1j*angles)
        z += .015 * (self.rng.normal(size=len(z)) + 1j*self.rng.normal(size=len(z)))
        self.phase = math.remainder(self.phase + 2*math.pi*(self.c.expected_offset_hz+f)*self.c.block_s, 2*math.pi)
        self.block.raw[::2] = np.clip(np.rint(127.5+128*z.real), 0, 255).astype(np.uint8)
        self.block.raw[1::2] = np.clip(np.rint(127.5+128*z.imag), 0, 255).astype(np.uint8)
        self.block.sequence = self.index // self.c.block_samples
        self.block.sample_index = self.index
        self.block.arrival = time.monotonic()
        self.index += self.c.block_samples
        return self.block

    def release(self, block):
        pass

    def close(self):
        pass
