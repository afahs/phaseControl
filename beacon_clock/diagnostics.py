"""Bounded receive-only spectral diagnostic. Raw IQ stays in memory."""
import json
import math
from pathlib import Path
import numpy as np


def spectrum(config, source, output, seconds=2.):
    if not 0.25 <= seconds <= 4:
        raise ValueError("spectrum capture must be between 0.25 and 4 seconds")
    if Path(output).exists():
        raise FileExistsError(output)
    blocks = math.ceil(seconds/config.block_s)
    data = np.empty(blocks*config.block_samples, dtype=np.complex64)
    clipped = 0
    source.start()
    try:
        expected = None
        for i in range(blocks+4):
            block = source.get()
            try:
                if expected is not None and (block.sample_index != expected or block.lost):
                    raise RuntimeError("diagnostic capture has a gap; retry")
                expected = block.sample_index+config.block_samples
                if i < 4:
                    continue
                dest = data[(i-4)*config.block_samples:(i-3)*config.block_samples]
                np.subtract(block.raw[::2], 127.5, out=dest.real, casting="unsafe")
                np.subtract(block.raw[1::2], 127.5, out=dest.imag, casting="unsafe")
                clipped += int(np.count_nonzero((block.raw <= 1) | (block.raw >= 254)))
            finally:
                source.release(block)
    finally:
        source.close()
    data /= 128
    amplitude = abs(np.fft.fft(data*np.hanning(len(data))))
    frequencies = np.fft.fftfreq(len(data), 1/config.sample_rate)
    indexes = np.flatnonzero(abs(frequencies-config.expected_offset_hz) < config.search_hz)
    local = indexes[(amplitude[indexes] > amplitude[indexes-1]) &
                    (amplitude[indexes] >= amplitude[indexes+1])]
    ordered = local[np.argsort(amplitude[local])[::-1]]
    peaks = []
    peak = float(np.max(amplitude[indexes]))
    for j in ordered:
        f = float(frequencies[j]-config.expected_offset_hz)
        if all(abs(f-p["residual_hz"]) > 2 for p in peaks):
            peaks.append(dict(residual_hz=f, relative_db=float(20*np.log10(amplitude[j]/peak))))
        if len(peaks) == 12:
            break
    result = dict(resolution_hz=config.sample_rate/len(data), seconds=len(data)/config.sample_rate,
                  clipping_fraction=clipped/(2*len(data)), peaks=peaks,
                  source_settings=source.settings, raw_iq_saved=False,
                  caveat="FFT peaks alone do not distinguish phase modulation from separate emitters")
    with Path(output).open("x") as f:
        json.dump(result, f, indent=2)
    return result
