from dataclasses import dataclass
import math
import numpy as np


@dataclass(slots=True)
class Measurement:
    t: float
    phase: float
    frequency: float
    amplitude: float
    snr_db: float
    coherence: float
    clipping: float
    valid: bool
    segment: int
    slip: bool = False


class PilotEstimator:
    """Fixed coarse mixer + FIR; reconstruct phase against the NOMINAL pilot.

    Coarse frequency is latched per continuity segment, not fed to the actuator.
    The phase of that mixer is explicitly restored to the output measurement.
    """
    def __init__(self, config):
        self.c = config
        n = config.block_samples
        self.n = np.arange(n, dtype=np.float64)
        self.window = np.hanning(n)
        self.bins = np.fft.fftfreq(n, 1 / config.sample_rate)
        self.search = np.flatnonzero(abs(self.bins - config.expected_offset_hz) <= config.search_hz)
        taps = 129
        h = 2 * 8000 / config.sample_rate * np.sinc(
            2 * 8000 / config.sample_rate * (np.arange(taps) - (taps - 1) / 2))
        h *= np.hamming(taps)
        self.h = (h / h.sum()).astype(np.float32)
        self.tail = np.zeros(taps - 1, dtype=np.complex64)
        self.iq = np.empty(n, dtype=np.complex64)
        self.mixed = np.empty(n, dtype=np.complex64)
        self.joined = np.empty(n + taps - 1, dtype=np.complex64)
        self.reset()

    def reset(self):
        self.coarse = None
        self.previous = None
        self.segment = -1
        self.next_index = None
        self.tail.fill(0)

    def process(self, block):
        c = self.c
        discontinuity = (block.segment != self.segment or block.lost
                         or (self.next_index is not None and block.sample_index != self.next_index))
        if discontinuity:
            self.reset()
        self.segment = block.segment
        self.next_index = block.sample_index + c.block_samples
        raw = block.raw
        np.subtract(raw[::2], 127.5, out=self.iq.real, casting="unsafe")
        np.subtract(raw[1::2], 127.5, out=self.iq.imag, casting="unsafe")
        self.iq /= 128
        clipping = float(np.count_nonzero((raw <= 1) | (raw >= 254)) / raw.size)
        first = self.coarse is None
        if first:
            spectrum = abs(np.fft.fft(self.iq * self.window))
            p = int(self.search[np.argmax(spectrum[self.search])])
            a, b, d = np.log(np.maximum(spectrum[[(p - 1) % len(raw[::2]), p,
                                                 (p + 1) % len(raw[::2])]], 1e-20))
            denom = a - 2 * b + d
            fraction = float(np.clip(.5 * (a - d) / denom, -.5, .5)) if denom else 0.
            self.coarse = self.bins[p] + fraction * c.sample_rate / c.block_samples - c.expected_offset_hz
            self.mix_hz = c.expected_offset_hz + self.coarse
            self.template = np.exp(-2j * np.pi * self.mix_hz * self.n / c.sample_rate).astype(np.complex64)
        start_phase = math.remainder(2 * math.pi * self.mix_hz * block.sample_index / c.sample_rate, 2 * math.pi)
        np.multiply(self.iq, self.template, out=self.mixed)
        self.mixed *= complex(math.cos(start_phase), -math.sin(start_phase))
        self.joined[:len(self.tail)] = self.tail
        self.joined[len(self.tail):] = self.mixed
        self.tail[:] = self.mixed[-len(self.tail):]
        start = 128 if first else 0
        positions = np.arange(start, c.block_samples, 32)
        # Compute only retained FIR outputs (polyphase decimation), not all 8192.
        # The sliding view owns no memory and remains valid for this worker call.
        windows = np.lib.stride_tricks.sliding_window_view(self.joined, len(self.h))
        z = np.einsum("ij,j->i", windows[start::32], self.h[::-1], optimize=False)
        times = (block.sample_index + positions - (len(self.h) - 1) / 2) / c.sample_rate
        t = float(times.mean())
        x = times - t
        angles = np.unwrap(np.angle(z))
        phase_mix = float(angles.mean())
        slope = float(np.dot(x, angles - phase_mix) / np.dot(x, x))
        derotated = z * np.exp(-1j * (phase_mix + slope * x))
        amplitude = float(abs(derotated.mean()))
        noise = float(np.mean(abs(derotated - derotated.mean()) ** 2))
        snr = float(10 * np.log10(max(amplitude ** 2, 1e-20) / max(noise, 1e-20)))
        coherence = float(amplitude / max(float(np.mean(abs(z))), 1e-20))
        frequency = self.coarse + slope / (2 * np.pi)
        phase = math.remainder(phase_mix + 2 * math.pi * self.coarse * t, 2 * math.pi)
        slip = False
        if self.previous:
            old = self.previous
            predicted = old.phase + 2 * math.pi * old.frequency * (t - old.t)
            phase += 2 * math.pi * round((predicted - phase) / (2 * math.pi))
            slip = abs(phase - predicted) > math.pi / 2
        valid = (amplitude >= c.min_amplitude and snr >= c.min_snr_db
                 and coherence >= c.min_coherence and clipping <= c.max_clip_fraction
                 and abs(frequency) < c.search_hz and not slip)
        m = Measurement(t, phase, frequency, amplitude, snr, coherence, clipping,
                        valid, block.segment, slip)
        if valid:
            self.previous = m
        else:
            self.reset()
        return m
