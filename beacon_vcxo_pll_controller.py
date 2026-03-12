#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beacon-referenced VCXO PLL controller for RTL-SDR + Si5351/MCP4725 clock helper.

Key design points
- ACQUIRE uses FFT-centered snapshot phase (robust, no inter-update phase continuity assumed)
- TRACK uses persistent NCO frequency + phase (classical PLL-compatible phase reference)
- OPEN LOOP is raw snapshot monitoring by default; optional persistent reference via --open_loop_persistent_ref
- ADJUSTED LOOP keeps persistent-NCO monitoring
- QUALITY_SELF uses fixed-tone continuous-rotator phase (plot_sdr_phase style)
- QUALITY_BEACON uses beacon-referenced persistent-NCO monitoring without actuation
- ACQUIRE supports one-shot FFT coarse pull-in followed by DAC pretrim; DAC PI otherwise runs only in TRACK
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ugradio import sdr as ugradio_sdr
    _UGRADIO_IMPORT_ERROR = None
except Exception as exc:
    ugradio_sdr = None
    _UGRADIO_IMPORT_ERROR = exc

try:
    import clock_vcxo_helper as cvh
except Exception:
    cvh = None


MODE_OPEN = "OPEN"
MODE_ADJUSTED = "ADJUSTED"
MODE_QUALITY_SELF = "QUALITY_SELF"
MODE_QUALITY_BEACON = "QUALITY_BEACON"
MODE_ACQUIRE = "ACQUIRE"
MODE_TRACK = "TRACK"
MODE_HOLDOVER = "HOLDOVER"


def wrap_to_pi(x):
    y = (float(x) + math.pi) % (2.0 * math.pi) - math.pi
    if y <= -math.pi:
        y += 2.0 * math.pi
    return float(y)


def clamp(x, lo, hi):
    x = float(x)
    lo = float(lo)
    hi = float(hi)
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def clamp_int(x, lo, hi):
    x = int(x)
    lo = int(lo)
    hi = int(hi)
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def now_unix():
    return time.time()


def timestamp_for_filename():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ensure_parent_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d)


def circular_mean(phases, weights=None):
    phases = np.asarray(phases, dtype=np.float64)
    if phases.size == 0:
        return 0.0
    ej = np.exp(1j * phases)
    if weights is None:
        z = np.mean(ej)
    else:
        w = np.asarray(weights, dtype=np.float64)
        sw = float(np.sum(w))
        if sw <= 0:
            z = np.mean(ej)
        else:
            z = np.sum(w * ej) / sw
    return float(np.angle(z))


def circular_resultant_length(phases, weights=None):
    phases = np.asarray(phases, dtype=np.float64)
    if phases.size == 0:
        return 0.0
    ej = np.exp(1j * phases)
    if weights is None:
        z = np.mean(ej)
    else:
        w = np.asarray(weights, dtype=np.float64)
        sw = float(np.sum(w))
        if sw <= 0:
            z = np.mean(ej)
        else:
            z = np.sum(w * ej) / sw
    return float(np.abs(z))


@dataclass
class CircularLpfState:
    initialized: bool = False
    z_re: float = 1.0
    z_im: float = 0.0


def circular_phase_lpf_update(phase_rad, alpha, state):
    a = clamp(alpha, 0.0, 0.999999)
    xr = math.cos(float(phase_rad))
    xi = math.sin(float(phase_rad))
    if not state.initialized:
        state.z_re = xr
        state.z_im = xi
        state.initialized = True
    else:
        state.z_re = a * state.z_re + (1.0 - a) * xr
        state.z_im = a * state.z_im + (1.0 - a) * xi
    return float(math.atan2(state.z_im, state.z_re))


def expected_baseband_freq_hz(beacon_hz, center_hz):
    return float(beacon_hz) - float(center_hz)


# ==========================
# Clock helper wrappers
# ==========================

def clock_create():
    if cvh is None:
        raise RuntimeError("clock_vcxo_helper.py not available")
    return cvh.ClockVCXO()


def clock_open(obj):
    obj.open()


def clock_close(obj):
    try:
        obj.close()
    except Exception:
        pass


def clock_init_28p8(obj, apr_ppm=120):
    obj.init_clock_vcxo_28p8mhz(apr_ppm=int(apr_ppm))


def clock_set_coarse_hz(obj, f_hz, settle_s=0.02):
    return float(obj.set_clock0_freq_hz(float(f_hz), settle_s=float(settle_s)))


def clock_set_dac_code(obj, code, settle_s=0.0):
    return int(obj.set_dac_code(int(code), settle_s=float(settle_s)))


def clock_dac_code_to_volts(obj, code):
    return float(obj.dac_code_to_volts(int(code)))


# ==========================
# SDR wrappers
# ==========================

def _iq_int8_blocks_to_complex64(iq_blocks):
    x = np.asarray(iq_blocks, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError("Expected ugradio IQ blocks with shape (nblocks, nsamp, 2).")
    return (x[..., 0] + 1j * x[..., 1]).astype(np.complex64)


def sdr_open(sample_rate, center_freq, gain):
    if ugradio_sdr is None:
        raise RuntimeError("ugradio.sdr import failed: {}".format(_UGRADIO_IMPORT_ERROR))
    return ugradio_sdr.SDR(
        device_index=0,
        direct=False,
        center_freq=float(center_freq),
        sample_rate=float(sample_rate),
        gain=float(gain),
    )


def sdr_close(sdr):
    if sdr is None:
        return
    try:
        sdr.close()
    except Exception:
        pass


def sdr_read_block(sdr, nsamp):
    data = sdr.capture_data(nsamples=int(nsamp), nblocks=1)
    return _iq_int8_blocks_to_complex64(data)[0]


def read_blocks(sdr, nsamp_block, blocks_per_update):
    data = sdr.capture_data(nsamples=int(nsamp_block), nblocks=int(blocks_per_update))
    blocks = _iq_int8_blocks_to_complex64(data)
    return [blocks[i] for i in range(blocks.shape[0])]


# ==========================
# Measurement
# ==========================

@dataclass
class MeasureConfig:
    fs: float
    beacon_hz: float
    center_hz: float
    fft_search_span_hz: float


@dataclass
class PhaseHistory:
    prev_phase_rad: Optional[float] = None
    prev_time_s: Optional[float] = None
    unwrapped_phase_rad: float = 0.0
    slip_aware_enabled: bool = True
    slip_trigger_rad: float = math.pi
    slip_max_cycles_per_update: int = 8
    slip_events: int = 0
    slip_cycles_total: int = 0
    last_slip_cycles: int = 0
    last_dphi_wrapped_rad: float = 0.0
    last_dphi_used_rad: float = 0.0
    last_expected_dphi_rad: float = 0.0

    def update(self, phase_rad, t_now_s, fallback_freq_hz):
        phase_rad = float(phase_rad)
        t_now_s = float(t_now_s)
        if self.prev_phase_rad is None or self.prev_time_s is None:
            self.prev_phase_rad = phase_rad
            self.prev_time_s = t_now_s
            self.unwrapped_phase_rad = phase_rad
            self.last_slip_cycles = 0
            self.last_dphi_wrapped_rad = 0.0
            self.last_dphi_used_rad = 0.0
            self.last_expected_dphi_rad = 0.0
            return float(fallback_freq_hz), 0.0
        dphi_wrapped = wrap_to_pi(phase_rad - self.prev_phase_rad)
        dt = t_now_s - self.prev_time_s
        dphi_used = float(dphi_wrapped)
        expected_dphi = 0.0
        slip_cycles = 0
        if dt > 0:
            expected_dphi = 2.0 * math.pi * float(fallback_freq_hz) * dt
            if self.slip_aware_enabled:
                err = expected_dphi - dphi_wrapped
                if abs(err) >= float(self.slip_trigger_rad):
                    k = int(round(err / (2.0 * math.pi)))
                    k = clamp_int(
                        k,
                        -int(self.slip_max_cycles_per_update),
                        int(self.slip_max_cycles_per_update),
                    )
                    if k != 0:
                        cand = float(dphi_wrapped + (2.0 * math.pi * float(k)))
                        if abs(cand - expected_dphi) < abs(dphi_wrapped - expected_dphi):
                            dphi_used = cand
                            slip_cycles = int(k)

        if dt > 0:
            f_upd = dphi_used / (2.0 * math.pi * dt)
        else:
            f_upd = float(fallback_freq_hz)
            dt = 0.0
        self.unwrapped_phase_rad += dphi_used
        self.prev_phase_rad = phase_rad
        self.prev_time_s = t_now_s
        self.last_slip_cycles = int(slip_cycles)
        self.last_dphi_wrapped_rad = float(dphi_wrapped)
        self.last_dphi_used_rad = float(dphi_used)
        self.last_expected_dphi_rad = float(expected_dphi)
        if slip_cycles != 0:
            self.slip_events += 1
            self.slip_cycles_total += abs(int(slip_cycles))
        return float(f_upd), float(dt)

    def reset(self):
        self.prev_phase_rad = None
        self.prev_time_s = None
        self.unwrapped_phase_rad = 0.0
        self.last_slip_cycles = 0
        self.last_dphi_wrapped_rad = 0.0
        self.last_dphi_used_rad = 0.0
        self.last_expected_dphi_rad = 0.0


@dataclass
class CorrelatorBlockResult:
    phase_rad: float
    coh: float
    freq_err_hz: float
    phase_end_rad: float
    z: complex


@dataclass
class UpdateMeasurement:
    phase_meas_rad: float
    phase_unwrapped_rad: float
    freq_err_block_hz: float
    freq_err_update_hz: float
    coh: float
    snr_db: float
    peak_freq_bb_hz: float
    peak_freq_abs_hz: float
    tone_power: float
    noise_power: float
    z_re: float
    z_im: float
    z_abs: float
    f_ref_corr_bb_hz: float
    meas_path: str
    dt_phase_update_s: float


@dataclass
class AcquireAggregate:
    phase_rad: float
    coh: float
    snr_db: float
    peak_bb_hz: float
    peak_abs_hz: float
    tone_power: float
    noise_power: float
    corr_freq_resid_hz: float
    z: complex


@dataclass
class TrackAggregate:
    phase_rad: float
    coh: float
    snr_db: float
    peak_bb_hz: float
    peak_abs_hz: float
    tone_power: float
    noise_power: float
    corr_freq_err_hz: float
    z: complex
    nco_phase_end_rad: float


def _fft_peak_near_expected(iq, fs, f_expected_bb_hz, span_hz):
    n = int(len(iq))
    if n <= 0:
        return 0.0, 0.0, 1e-12
    x = np.asarray(iq, dtype=np.complex64)
    w = np.hanning(n).astype(np.float32)
    xw = x * w
    X = np.fft.fftshift(np.fft.fft(xw))
    P = (np.abs(X) ** 2).astype(np.float64)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / float(fs)))

    if span_hz is None or float(span_hz) <= 0:
        idxs = np.arange(n)
    else:
        mask = (freqs >= (f_expected_bb_hz - float(span_hz))) & (freqs <= (f_expected_bb_hz + float(span_hz)))
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            idxs = np.arange(n)

    k_local = int(np.argmax(P[idxs]))
    k_peak = int(idxs[k_local])
    peak_freq = float(freqs[k_peak])
    peak_pow = float(P[k_peak])

    guard = max(3, n // 2048)
    noise_mask = np.zeros(n, dtype=np.bool_)
    noise_mask[idxs] = True
    lo = max(0, k_peak - guard)
    hi = min(n, k_peak + guard + 1)
    noise_mask[lo:hi] = False
    if np.any(noise_mask):
        noise_pow = float(np.median(P[noise_mask]))
    else:
        noise_pow = 1e-12
    noise_pow = max(noise_pow, 1e-12)
    return peak_freq, peak_pow, noise_pow


def correlate_phase_freq(iq, fs, f_ref_bb_hz, phase_ref0_rad):
    n = int(len(iq))
    if n < 8:
        return CorrelatorBlockResult(0.0, 0.0, 0.0, float(phase_ref0_rad), 0.0 + 0.0j)
    n_idx = np.arange(n, dtype=np.float64)
    dtheta = 2.0 * math.pi * float(f_ref_bb_hz) / float(fs)
    theta = float(phase_ref0_rad) + dtheta * n_idx
    ref_conj = np.exp(-1j * theta)
    x = np.asarray(iq, dtype=np.complex64)
    z = np.sum(x * ref_conj, dtype=np.complex128)
    phase = float(math.atan2(float(np.imag(z)), float(np.real(z))))
    denom = float(np.sum(np.abs(x))) + 1e-12
    coh = float(np.abs(z) / denom)

    nh = n // 2
    z1 = np.sum(x[:nh] * ref_conj[:nh], dtype=np.complex128)
    z2 = np.sum(x[nh:] * ref_conj[nh:], dtype=np.complex128)
    ph1 = math.atan2(float(np.imag(z1)), float(np.real(z1)))
    ph2 = math.atan2(float(np.imag(z2)), float(np.real(z2)))
    dph = wrap_to_pi(ph2 - ph1)
    dt = float(nh) / float(fs)
    freq_err = dph / (2.0 * math.pi * dt) if dt > 0 else 0.0

    phase_end = wrap_to_pi(float(phase_ref0_rad) + dtheta * float(n))
    return CorrelatorBlockResult(phase, coh, float(freq_err), float(phase_end), complex(z))


def _median_or(x, default=0.0):
    arr = np.asarray(x, dtype=np.float64)
    return float(np.median(arr)) if arr.size else float(default)


def measure_acquire_snapshot(iq_blocks, mcfg):
    if not iq_blocks:
        return AcquireAggregate(0.0, 0.0, -999.0, 0.0, mcfg.center_hz, 0.0, 1.0, 0.0, 0.0 + 0.0j)

    f_expected = expected_baseband_freq_hz(mcfg.beacon_hz, mcfg.center_hz)
    phases = []
    cohs = []
    snrs = []
    peaks_bb = []
    peaks_abs = []
    tone_ps = []
    noise_ps = []
    f_resids = []
    z_list = []

    for iqb in iq_blocks:
        peak_bb, tone_p, noise_p = _fft_peak_near_expected(iqb, mcfg.fs, f_expected, mcfg.fft_search_span_hz)
        snr_db = 10.0 * math.log10(max(tone_p / max(noise_p, 1e-12), 1e-12))
        corr = correlate_phase_freq(iqb, mcfg.fs, peak_bb, 0.0)
        phases.append(corr.phase_rad)
        cohs.append(corr.coh)
        snrs.append(snr_db)
        peaks_bb.append(peak_bb)
        peaks_abs.append(mcfg.center_hz + peak_bb)
        tone_ps.append(tone_p)
        noise_ps.append(noise_p)
        f_resids.append(corr.freq_err_hz)
        z_list.append(corr.z)

    w = np.maximum(np.asarray(cohs, dtype=np.float64), 1e-9)
    phase = circular_mean(np.asarray(phases, dtype=np.float64), weights=w)
    z_stack = np.asarray(z_list, dtype=np.complex128)
    z_sum = np.sum(z_stack) if z_stack.size else 0.0 + 0.0j
    return AcquireAggregate(
        phase_rad=float(phase),
        coh=_median_or(cohs, 0.0),
        snr_db=_median_or(snrs, -999.0),
        peak_bb_hz=_median_or(peaks_bb, 0.0),
        peak_abs_hz=_median_or(peaks_abs, mcfg.center_hz),
        tone_power=_median_or(tone_ps, 0.0),
        noise_power=_median_or(noise_ps, 1.0),
        corr_freq_resid_hz=_median_or(f_resids, 0.0),
        z=complex(z_sum),
    )


def measure_track_persistent_nco(iq_blocks, mcfg, nco_freq_bb_hz, nco_phase_start_rad):
    if not iq_blocks:
        return TrackAggregate(0.0, 0.0, -999.0, 0.0, mcfg.center_hz, 0.0, 1.0, 0.0, 0.0 + 0.0j, float(nco_phase_start_rad))

    f_expected = expected_baseband_freq_hz(mcfg.beacon_hz, mcfg.center_hz)
    cohs = []
    snrs = []
    peaks_bb = []
    peaks_abs = []
    tone_ps = []
    noise_ps = []
    ferrs = []
    z_sum = 0.0 + 0.0j
    phase_cursor = float(nco_phase_start_rad)

    for iqb in iq_blocks:
        peak_bb, tone_p, noise_p = _fft_peak_near_expected(iqb, mcfg.fs, f_expected, mcfg.fft_search_span_hz)
        snr_db = 10.0 * math.log10(max(tone_p / max(noise_p, 1e-12), 1e-12))
        corr = correlate_phase_freq(iqb, mcfg.fs, float(nco_freq_bb_hz), phase_cursor)
        phase_cursor = float(corr.phase_end_rad)
        z_sum += corr.z
        cohs.append(corr.coh)
        snrs.append(snr_db)
        peaks_bb.append(peak_bb)
        peaks_abs.append(mcfg.center_hz + peak_bb)
        tone_ps.append(tone_p)
        noise_ps.append(noise_p)
        ferrs.append(corr.freq_err_hz)

    phase = float(np.angle(z_sum)) if abs(z_sum) > 0 else 0.0
    return TrackAggregate(
        phase_rad=phase,
        coh=_median_or(cohs, 0.0),
        snr_db=_median_or(snrs, -999.0),
        peak_bb_hz=_median_or(peaks_bb, 0.0),
        peak_abs_hz=_median_or(peaks_abs, mcfg.center_hz),
        tone_power=_median_or(tone_ps, 0.0),
        noise_power=_median_or(noise_ps, 1.0),
        corr_freq_err_hz=_median_or(ferrs, 0.0),
        z=complex(z_sum),
        nco_phase_end_rad=float(phase_cursor),
    )


def build_update_measurement_from_acquire(iq_blocks, mcfg, phase_hist, t_now_s):
    agg = measure_acquire_snapshot(iq_blocks, mcfg)
    fft_freq_err_hz = float(agg.peak_abs_hz - mcfg.beacon_hz)
    freq_err_block = float(fft_freq_err_hz + agg.corr_freq_resid_hz)
    freq_err_upd, dt_phase = phase_hist.update(agg.phase_rad, t_now_s, freq_err_block)
    return UpdateMeasurement(
        phase_meas_rad=float(agg.phase_rad),
        phase_unwrapped_rad=float(phase_hist.unwrapped_phase_rad),
        freq_err_block_hz=float(freq_err_block),
        freq_err_update_hz=float(freq_err_upd),
        coh=float(agg.coh),
        snr_db=float(agg.snr_db),
        peak_freq_bb_hz=float(agg.peak_bb_hz),
        peak_freq_abs_hz=float(agg.peak_abs_hz),
        tone_power=float(agg.tone_power),
        noise_power=float(agg.noise_power),
        z_re=float(np.real(agg.z)),
        z_im=float(np.imag(agg.z)),
        z_abs=float(abs(agg.z)),
        f_ref_corr_bb_hz=float(agg.peak_bb_hz),
        meas_path="acquire_snapshot",
        dt_phase_update_s=float(dt_phase),
    )


def build_update_measurement_from_track(iq_blocks, mcfg, phase_hist, t_now_s, nco_freq_bb_hz, nco_phase_start_rad):
    agg = measure_track_persistent_nco(iq_blocks, mcfg, nco_freq_bb_hz, nco_phase_start_rad)
    freq_err_block = float((agg.peak_abs_hz - mcfg.beacon_hz) + agg.corr_freq_err_hz)
    freq_err_upd, dt_phase = phase_hist.update(agg.phase_rad, t_now_s, freq_err_block)
    meas = UpdateMeasurement(
        phase_meas_rad=float(agg.phase_rad),
        phase_unwrapped_rad=float(phase_hist.unwrapped_phase_rad),
        freq_err_block_hz=float(freq_err_block),
        freq_err_update_hz=float(freq_err_upd),
        coh=float(agg.coh),
        snr_db=float(agg.snr_db),
        peak_freq_bb_hz=float(agg.peak_bb_hz),
        peak_freq_abs_hz=float(agg.peak_abs_hz),
        tone_power=float(agg.tone_power),
        noise_power=float(agg.noise_power),
        z_re=float(np.real(agg.z)),
        z_im=float(np.imag(agg.z)),
        z_abs=float(abs(agg.z)),
        f_ref_corr_bb_hz=float(nco_freq_bb_hz),
        meas_path="persistent_nco",
        dt_phase_update_s=float(dt_phase),
    )
    return meas, float(agg.nco_phase_end_rad)


def estimate_dominant_tone_offset_hz(iq_blocks, sample_rate_hz, dc_exclusion_hz=5e3):
    if not iq_blocks:
        return 0.0
    x = np.stack(
        [
            np.asarray(iqb, dtype=np.complex64).reshape(-1)
            - np.mean(np.asarray(iqb, dtype=np.complex64).reshape(-1))
            for iqb in iq_blocks
        ],
        axis=0,
    )
    spectra = np.fft.fft(x, axis=1)
    power = np.mean(np.abs(spectra) ** 2, axis=0)
    freqs = np.fft.fftfreq(x.shape[1], d=1.0 / float(sample_rate_hz))

    mask = np.ones(freqs.shape, dtype=bool)
    if dc_exclusion_hz is not None and float(dc_exclusion_hz) > 0:
        mask &= np.abs(freqs) >= float(dc_exclusion_hz)
    if not np.any(mask):
        mask[:] = True

    idx_candidates = np.flatnonzero(mask)
    peak_idx = int(idx_candidates[int(np.argmax(power[mask]))])
    return float(freqs[peak_idx])


def correlate_phase_fixed_tone(iq, fs, tone_offset_hz, sample_start):
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if x.size < 8:
        return 0.0, 0.0, 0.0 + 0.0j
    x = x - np.mean(x)
    n = float(sample_start) + np.arange(x.size, dtype=np.float64)
    rotator = np.exp(
        -2j * np.pi * float(tone_offset_hz) * n / float(fs)
    ).astype(np.complex64)
    z = np.sum(x * rotator, dtype=np.complex128)
    phase = float(np.angle(z))
    coh = float(np.abs(z) / (np.sum(np.abs(x)) + 1e-12))
    return phase, coh, complex(z)


def build_update_measurement_from_quality_self(
    iq_blocks,
    mcfg,
    phase_hist,
    t_now_s,
    tone_offset_hz,
    sample_start,
):
    if not iq_blocks:
        meas = UpdateMeasurement(
            phase_meas_rad=0.0,
            phase_unwrapped_rad=float(phase_hist.unwrapped_phase_rad),
            freq_err_block_hz=0.0,
            freq_err_update_hz=0.0,
            coh=0.0,
            snr_db=-999.0,
            peak_freq_bb_hz=0.0,
            peak_freq_abs_hz=float(mcfg.center_hz),
            tone_power=0.0,
            noise_power=1.0,
            z_re=0.0,
            z_im=0.0,
            z_abs=0.0,
            f_ref_corr_bb_hz=float(tone_offset_hz),
            meas_path="quality_self",
            dt_phase_update_s=0.0,
        )
        return meas, int(sample_start)

    iq = np.concatenate(
        [np.asarray(iqb, dtype=np.complex64).reshape(-1) for iqb in iq_blocks]
    )
    phase, coh, z = correlate_phase_fixed_tone(
        iq, mcfg.fs, tone_offset_hz, sample_start
    )
    peak_bb, tone_p, noise_p = _fft_peak_near_expected(
        iq, mcfg.fs, float(tone_offset_hz), span_hz=None
    )
    snr_db = 10.0 * math.log10(max(tone_p / max(noise_p, 1e-12), 1e-12))
    freq_err_block = float(peak_bb - float(tone_offset_hz))
    freq_err_upd, dt_phase = phase_hist.update(phase, t_now_s, freq_err_block)

    meas = UpdateMeasurement(
        phase_meas_rad=float(phase),
        phase_unwrapped_rad=float(phase_hist.unwrapped_phase_rad),
        freq_err_block_hz=float(freq_err_block),
        freq_err_update_hz=float(freq_err_upd),
        coh=float(coh),
        snr_db=float(snr_db),
        peak_freq_bb_hz=float(peak_bb),
        peak_freq_abs_hz=float(mcfg.center_hz + peak_bb),
        tone_power=float(tone_p),
        noise_power=float(noise_p),
        z_re=float(np.real(z)),
        z_im=float(np.imag(z)),
        z_abs=float(abs(z)),
        f_ref_corr_bb_hz=float(tone_offset_hz),
        meas_path="quality_self",
        dt_phase_update_s=float(dt_phase),
    )
    return meas, int(sample_start + iq.size)


# ==========================
# Control structures
# ==========================

@dataclass
class ControlParams:
    sample_rate: float = 2.4e6
    center_freq_hz: float = 99.5e6
    beacon_freq_hz: float = 100.0e6
    gain: float = 1.0

    nsamp_block: int = 65536
    blocks_per_update: int = 1

    apr_ppm: int = 120
    clock0_nom_hz: float = 28.8e6
    clock0_cmd_hz_min: float = 28.75e6
    clock0_cmd_hz_max: float = 28.85e6
    coarse_settle_s: float = 0.02
    fine_settle_s: float = 0.0

    dac_code_min: int = 1024
    dac_code_max: int = 3686
    dac_code_start: int = 2355
    max_dac_step_per_update: int = 100
    kv_hz_per_code: float = 4.243

    coarse_enable: bool = True
    fft_pullin_enable: bool = True
    fft_pullin_once_only: bool = True
    max_coarse_step_hz_per_update: float = 5000.0
    Kf_coarse_clock_hz_per_hz_err: float = 0.5
    f_switch_hz: float = 50.0
    freq_err_drop_hz: float = 300.0

    Kp_phi_code_per_rad: float = 20.0
    Ki_phi_code_per_rad_per_update: float = 1.0
    I_term_min: float = -500.0
    I_term_max: float = 500.0

    coh_min_acq: float = 0.05
    coh_min_trk: float = 0.02
    coh_min_ctrl: float = 0.10
    snr_min_acq_db: float = 8.0
    snr_min_trk_db: float = 10.0

    lock_count_required: int = 3
    unlock_count_required: int = 3
    holdover_decay_alpha: float = 0.995

    fft_search_span_hz: float = 200e3
    track_nco_alpha: float = 0.8
    track_use_fft_freq_assist: bool = True
    phase_filt_alpha: float = 0.85

    open_loop: bool = False
    open_loop_persistent_ref: bool = False
    adjusted_loop: bool = False
    quality_self: bool = False
    quality_beacon: bool = False
    verbose: bool = True
    leave_dac_unchanged_on_exit: bool = True

    phase_setpoint_rad: float = 0.0
    calibrate_phase_setpoint: bool = False
    setpoint_cal_samples: int = 30
    quality_self_tone_hz: Optional[float] = None
    quality_self_dc_exclusion_hz: float = 5e3

    run_seconds: Optional[float] = None
    max_updates: Optional[int] = None


@dataclass
class SetpointCalState:
    enabled: bool = False
    target_samples: int = 30
    done: bool = False
    count: int = 0
    z_sum: complex = 0.0 + 0.0j

    def push(self, phase_rad):
        if self.done or (not self.enabled):
            return False, None
        self.z_sum += complex(math.cos(float(phase_rad)), math.sin(float(phase_rad)))
        self.count += 1
        if self.count >= max(1, int(self.target_samples)):
            self.done = True
            if abs(self.z_sum) > 0:
                return True, float(np.angle(self.z_sum))
            return True, 0.0
        return False, None


@dataclass
class ControlState:
    mode: str
    action: str = "init"
    clock0_cmd_hz: float = 28.8e6
    dac_code_cmd: int = 2355
    dac_code_rb: Optional[int] = None
    dac_volts: Optional[float] = None
    I_term: float = 0.0
    lock_counter: int = 0
    unlock_counter: int = 0
    fft_pullin_done: bool = False
    fft_dac_pretrim_done: bool = False
    quality_sample_start: int = 0
    t_start_s: float = 0.0
    update_index: int = 0

    nco_initialized: bool = False
    nco_freq_bb_hz: float = 0.0
    nco_phase_rad: float = 0.0
    df_nco_peak_hz: float = 0.0

    phase_setpoint_rad: float = 0.0
    phase_ctrl_err_raw_rad: float = 0.0
    phase_ctrl_err_filt_rad: float = 0.0
    phase_filt: CircularLpfState = None
    slip_last_cycles: int = 0
    slip_events_total: int = 0
    slip_cycles_total: int = 0
    phase_dphi_wrapped_rad: float = 0.0
    phase_dphi_used_rad: float = 0.0
    phase_dphi_expected_rad: float = 0.0

    t_prev_meas_s: Optional[float] = None
    last_error_msg: str = ""
    setpoint_cal_event: int = 0

    def __post_init__(self):
        if self.phase_filt is None:
            self.phase_filt = CircularLpfState()


@dataclass
class PiStepResult:
    changed: bool
    step_req_codes: int
    step_applied_codes: int
    new_code_cmd: int


class GracefulStop(object):
    def __init__(self):
        self.stop = False
        self.count = 0
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        self.count += 1
        self.stop = True
        if self.count > 1:
            raise KeyboardInterrupt()


def coarse_update_from_freq_error(f_err_hz, params, state, clock_obj):
    if clock_obj is None or (not params.coarse_enable):
        return False
    raw_step = float(params.Kf_coarse_clock_hz_per_hz_err) * float(f_err_hz)
    step = clamp(raw_step, -params.max_coarse_step_hz_per_update, params.max_coarse_step_hz_per_update)
    new_cmd = clamp(state.clock0_cmd_hz + step, params.clock0_cmd_hz_min, params.clock0_cmd_hz_max)
    if abs(new_cmd - state.clock0_cmd_hz) < 1e-9:
        return False
    rb = clock_set_coarse_hz(clock_obj, new_cmd, settle_s=params.coarse_settle_s)
    state.clock0_cmd_hz = float(rb)
    return True


def dac_feedforward_from_freq_error(f_err_hz, params, state, clock_obj):
    kv = float(params.kv_hz_per_code)
    if abs(kv) < 1e-12:
        return PiStepResult(
            changed=False,
            step_req_codes=0,
            step_applied_codes=0,
            new_code_cmd=int(state.dac_code_cmd),
        )

    raw_step_hz = float(params.Kf_coarse_clock_hz_per_hz_err) * float(f_err_hz)
    raw_step_codes = raw_step_hz / kv
    u_codes = clamp(
        raw_step_codes,
        -params.max_dac_step_per_update,
        params.max_dac_step_per_update,
    )
    step_req = int(round(u_codes))
    new_code = clamp_int(
        state.dac_code_cmd + step_req,
        params.dac_code_min,
        params.dac_code_max,
    )
    step_applied = int(new_code - state.dac_code_cmd)
    changed = (new_code != state.dac_code_cmd)

    if changed and clock_obj is not None:
        rb = clock_set_dac_code(clock_obj, new_code, settle_s=params.fine_settle_s)
        state.dac_code_cmd = int(new_code)
        state.dac_code_rb = int(rb)
        state.dac_volts = clock_dac_code_to_volts(clock_obj, state.dac_code_cmd)
    elif changed:
        state.dac_code_cmd = int(new_code)

    return PiStepResult(
        changed=changed,
        step_req_codes=step_req,
        step_applied_codes=step_applied,
        new_code_cmd=int(state.dac_code_cmd),
    )


def holdover_update(params, state):
    state.I_term *= float(params.holdover_decay_alpha)


def pi_update_dac(phase_err_rad, params, state, clock_obj, allow_integrate):
    phase_err = float(phase_err_rad)
    p_term = float(params.Kp_phi_code_per_rad) * phase_err
    i_candidate = state.I_term
    if allow_integrate:
        i_candidate = clamp(
            state.I_term + float(params.Ki_phi_code_per_rad_per_update) * phase_err,
            params.I_term_min,
            params.I_term_max,
        )

    u_codes = p_term + i_candidate
    u_codes = clamp(u_codes, -params.max_dac_step_per_update, params.max_dac_step_per_update)
    step_req = int(round(u_codes))
    new_code = clamp_int(state.dac_code_cmd + step_req, params.dac_code_min, params.dac_code_max)
    step_applied = int(new_code - state.dac_code_cmd)

    saturated = (new_code == params.dac_code_min or new_code == params.dac_code_max)
    pushes_outward = ((new_code == params.dac_code_max and step_req > 0) or
                      (new_code == params.dac_code_min and step_req < 0))
    if allow_integrate and not (saturated and pushes_outward):
        state.I_term = float(i_candidate)

    changed = (new_code != state.dac_code_cmd)
    if changed and clock_obj is not None:
        rb = clock_set_dac_code(clock_obj, new_code, settle_s=params.fine_settle_s)
        state.dac_code_cmd = int(new_code)
        state.dac_code_rb = int(rb)
        state.dac_volts = clock_dac_code_to_volts(clock_obj, state.dac_code_cmd)
    elif changed:
        state.dac_code_cmd = int(new_code)

    return PiStepResult(changed=changed, step_req_codes=step_req, step_applied_codes=step_applied, new_code_cmd=int(state.dac_code_cmd))


CSV_COLUMNS = [
    "unix_time",
    "elapsed_s",
    "update_idx",
    "mode",
    "action",
    "open_loop",
    "meas_path",
    "dt_update_s",
    "phase_err_rad",
    "phase_err_unwrapped_rad",
    "phase_ctrl_err_rad",
    "phase_ctrl_err_raw_rad",
    "phase_ctrl_err_filt_rad",
    "phase_setpoint_rad",
    "setpoint_cal_event",
    "slip_event",
    "slip_cycles",
    "slip_events_total",
    "slip_cycles_total",
    "phase_dphi_wrapped_rad",
    "phase_dphi_used_rad",
    "phase_dphi_expected_rad",
    "freq_err_block_hz",
    "freq_err_update_hz",
    "coh",
    "snr_db",
    "meas_quality_ok",
    "peak_freq_abs_hz",
    "peak_freq_baseband_hz",
    "f_ref_corr_bb_hz",
    "nco_freq_bb_hz",
    "df_nco_peak_hz",
    "nco_phase_rad",
    "z_re",
    "z_im",
    "z_abs",
    "tone_power",
    "noise_power",
    "beacon_freq_hz",
    "center_freq_hz",
    "clock0_cmd_hz",
    "clock0_offset_hz",
    "dac_code_cmd",
    "dac_code_rb",
    "dac_volts",
    "dac_step_req_codes",
    "dac_step_applied_codes",
    "I_term",
    "lock_counter",
    "unlock_counter",
    "nsamp_block",
    "blocks_per_update",
    "sample_rate",
    "kv_hz_per_code",
    "notes",
]


class CsvLogger(object):
    def __init__(self, path):
        self.path = path
        ensure_parent_dir(path)
        self.fh = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()
        self.fh.flush()

    def write(self, row):
        out = {}
        for k in CSV_COLUMNS:
            out[k] = row.get(k, "")
        self.writer.writerow(out)
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


def print_status(params, state, meas):
    if not params.verbose:
        return
    print(
        "t={:8.2f}s | {:>8s} | {:<18s} | ph={:+7.3f} | phi_ctl={:+7.3f} | f_upd={:+8.2f} Hz | "
        "coh={:5.3f} | snr={:5.1f} dB | peak={:9.1f} Hz bb | nco={:9.1f} Hz bb | slip={:+2d}/{} | dac={} | I={:+7.2f}".format(
            now_unix() - state.t_start_s,
            state.mode,
            state.action,
            meas.phase_meas_rad,
            state.phase_ctrl_err_filt_rad,
            meas.freq_err_update_hz,
            meas.coh,
            meas.snr_db,
            meas.peak_freq_bb_hz,
            state.nco_freq_bb_hz if state.nco_initialized else float("nan"),
            state.slip_last_cycles,
            state.slip_events_total,
            state.dac_code_cmd,
            state.I_term,
        ),
        flush=True,
    )


def sync_state_with_phase_history(state, phase_hist):
    state.slip_last_cycles = int(phase_hist.last_slip_cycles)
    state.slip_events_total = int(phase_hist.slip_events)
    state.slip_cycles_total = int(phase_hist.slip_cycles_total)
    state.phase_dphi_wrapped_rad = float(phase_hist.last_dphi_wrapped_rad)
    state.phase_dphi_used_rad = float(phase_hist.last_dphi_used_rad)
    state.phase_dphi_expected_rad = float(phase_hist.last_expected_dphi_rad)


def make_log_path(path_arg):
    if path_arg.endswith(os.sep) or os.path.isdir(path_arg):
        return os.path.join(path_arg, "pll_log_{}.csv".format(timestamp_for_filename()))
    root, ext = os.path.splitext(path_arg)
    if ext.lower() == ".csv":
        return path_arg
    return "{}_{}.csv".format(path_arg, timestamp_for_filename())


def build_arg_parser():
    p = argparse.ArgumentParser(description="Beacon VCXO PLL controller (RTL-SDR + Si5351/MCP4725)")
    p.add_argument(
        "--mode",
        choices=[
            "open_loop",
            "adjusted_loop",
            "quality_self",
            "quality_beacon",
            "closed_loop",
        ],
        default="closed_loop",
    )
    p.add_argument("--log_csv", required=True, type=str)
    p.add_argument("--notes", default="", type=str)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run_seconds", type=float, default=None)
    p.add_argument("--max_updates", type=int, default=None)
    p.add_argument("--run_self_tests", action="store_true", help="Run deterministic unit/synthetic checks and exit")

    p.add_argument("--sample_rate", type=float, default=2.4e6)
    p.add_argument("--center_freq", type=float, default=99.5e6)
    p.add_argument("--beacon_freq", type=float, default=100e6)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--nsamp_block", type=int, default=65536)
    p.add_argument("--blocks_per_update", type=int, default=1)

    p.add_argument("--apr_ppm", type=int, default=120)
    p.add_argument("--clock0_nom_hz", type=float, default=28.8e6)
    p.add_argument("--clock0_min_hz", type=float, default=28.75e6)
    p.add_argument("--clock0_max_hz", type=float, default=28.85e6)
    p.add_argument("--coarse_settle_s", type=float, default=0.02)
    p.add_argument("--fine_settle_s", type=float, default=0.0)

    p.add_argument("--dac_min", type=int, default=1024)
    p.add_argument("--dac_max", type=int, default=3686)
    p.add_argument("--dac_start", type=int, default=None)
    p.add_argument("--max_dac_step", type=int, default=100)
    p.add_argument("--kv_hz_per_code", type=float, default=4.243)

    p.add_argument("--disable_coarse", action="store_true")
    p.add_argument("--disable_fft_pullin", action="store_true")
    p.add_argument("--fft_pullin_once_only", action="store_true", default=False)
    p.add_argument("--max_coarse_step_hz", type=float, default=5000.0)
    p.add_argument("--Kf_coarse", type=float, default=0.5)
    p.add_argument("--f_switch_hz", type=float, default=50.0)
    p.add_argument("--freq_err_drop_hz", type=float, default=300.0)

    p.add_argument("--Kp_phi", type=float, default=20.0)
    p.add_argument("--Ki_phi", type=float, default=1.0)
    p.add_argument("--I_min", type=float, default=-500.0)
    p.add_argument("--I_max", type=float, default=500.0)

    p.add_argument("--coh_min_acq", type=float, default=0.05)
    p.add_argument("--coh_min_trk", type=float, default=0.02)
    p.add_argument("--coh_min_ctrl", type=float, default=0.10)
    p.add_argument("--snr_min_acq_db", type=float, default=8.0)
    p.add_argument("--snr_min_trk_db", type=float, default=10.0)
    p.add_argument("--lock_count_required", type=int, default=3)
    p.add_argument("--unlock_count_required", type=int, default=3)
    p.add_argument("--holdover_decay_alpha", type=float, default=0.995)

    p.add_argument("--fft_search_span_hz", type=float, default=200e3)
    p.add_argument("--track_nco_alpha", type=float, default=0.8)
    p.add_argument("--disable_track_fft_assist", action="store_true")
    p.add_argument("--phase_filt_alpha", type=float, default=0.85)
    p.add_argument(
        "--open_loop_persistent_ref",
        action="store_true",
        help="In open_loop mode, seed a beacon-referenced NCO once and keep a persistent phase reference (no actuation).",
    )

    p.add_argument("--phase_setpoint_rad", type=float, default=0.0)
    p.add_argument("--calibrate_phase_setpoint", action="store_true")
    p.add_argument("--setpoint_cal_samples", type=int, default=30)
    p.add_argument(
        "--quality_self_tone_hz",
        type=float,
        default=None,
        help="Fixed baseband tone offset for quality_self mode. If omitted, estimate from startup data.",
    )
    p.add_argument(
        "--quality_self_dc_exclusion_hz",
        type=float,
        default=5e3,
        help="Exclude tones this close to DC when auto-estimating quality_self tone.",
    )
    return p


def params_from_args(args):
    dac_start = args.dac_start
    if dac_start is None:
        dac_start = int(round((int(args.dac_min) + int(args.dac_max)) / 2.0))

    return ControlParams(
        sample_rate=float(args.sample_rate),
        center_freq_hz=float(args.center_freq),
        beacon_freq_hz=float(args.beacon_freq),
        gain=float(args.gain),
        nsamp_block=int(args.nsamp_block),
        blocks_per_update=int(args.blocks_per_update),
        apr_ppm=int(args.apr_ppm),
        clock0_nom_hz=float(args.clock0_nom_hz),
        clock0_cmd_hz_min=float(args.clock0_min_hz),
        clock0_cmd_hz_max=float(args.clock0_max_hz),
        coarse_settle_s=float(args.coarse_settle_s),
        fine_settle_s=float(args.fine_settle_s),
        dac_code_min=int(args.dac_min),
        dac_code_max=int(args.dac_max),
        dac_code_start=int(dac_start),
        max_dac_step_per_update=int(args.max_dac_step),
        kv_hz_per_code=float(args.kv_hz_per_code),
        coarse_enable=(not bool(args.disable_coarse)),
        fft_pullin_enable=(not bool(args.disable_fft_pullin)),
        fft_pullin_once_only=True,
        max_coarse_step_hz_per_update=float(args.max_coarse_step_hz),
        Kf_coarse_clock_hz_per_hz_err=float(args.Kf_coarse),
        f_switch_hz=float(args.f_switch_hz),
        freq_err_drop_hz=float(args.freq_err_drop_hz),
        Kp_phi_code_per_rad=float(args.Kp_phi),
        Ki_phi_code_per_rad_per_update=float(args.Ki_phi),
        I_term_min=float(args.I_min),
        I_term_max=float(args.I_max),
        coh_min_acq=float(args.coh_min_acq),
        coh_min_trk=float(args.coh_min_trk),
        coh_min_ctrl=float(args.coh_min_ctrl),
        snr_min_acq_db=float(args.snr_min_acq_db),
        snr_min_trk_db=float(args.snr_min_trk_db),
        lock_count_required=int(args.lock_count_required),
        unlock_count_required=int(args.unlock_count_required),
        holdover_decay_alpha=float(args.holdover_decay_alpha),
        fft_search_span_hz=float(args.fft_search_span_hz),
        track_nco_alpha=float(args.track_nco_alpha),
        track_use_fft_freq_assist=(not bool(args.disable_track_fft_assist)),
        phase_filt_alpha=float(args.phase_filt_alpha),
        open_loop=(args.mode == "open_loop"),
        open_loop_persistent_ref=bool(args.open_loop_persistent_ref),
        adjusted_loop=(args.mode == "adjusted_loop"),
        quality_self=(args.mode == "quality_self"),
        quality_beacon=(args.mode == "quality_beacon"),
        verbose=(not bool(args.quiet)),
        phase_setpoint_rad=float(args.phase_setpoint_rad),
        calibrate_phase_setpoint=bool(args.calibrate_phase_setpoint),
        setpoint_cal_samples=int(args.setpoint_cal_samples),
        quality_self_tone_hz=(
            None if args.quality_self_tone_hz is None else float(args.quality_self_tone_hz)
        ),
        quality_self_dc_exclusion_hz=float(args.quality_self_dc_exclusion_hz),
        run_seconds=(None if args.run_seconds is None else float(args.run_seconds)),
        max_updates=(None if args.max_updates is None else int(args.max_updates)),
    )


def _phase_quality_ok_for_ctrl(params, meas):
    return (meas.coh >= float(params.coh_min_ctrl)) and (meas.snr_db >= float(params.snr_min_trk_db))


def _phase_quality_ok_for_track(params, meas):
    return (meas.coh >= float(params.coh_min_trk)) and (meas.snr_db >= float(params.snr_min_trk_db))


def _snr_ok_for_acq(params, meas):
    return meas.snr_db >= float(params.snr_min_acq_db)


def _track_stability_inconsistent(params, state, meas):
    df_peak = abs(float(state.df_nco_peak_hz))
    return (abs(meas.freq_err_update_hz) > float(params.freq_err_drop_hz)) or (df_peak > max(2.0 * float(params.freq_err_drop_hz), 200.0))


def log_row(logger, params, state, meas, args_notes, dt_update_s, dac_step_result, meas_quality_ok):
    logger.write({
        "unix_time": now_unix(),
        "elapsed_s": now_unix() - state.t_start_s,
        "update_idx": state.update_index,
        "mode": state.mode,
        "action": state.action,
        "open_loop": int(params.open_loop),
        "meas_path": meas.meas_path,
        "dt_update_s": "" if dt_update_s is None else float(dt_update_s),
        "phase_err_rad": meas.phase_meas_rad,
        "phase_err_unwrapped_rad": meas.phase_unwrapped_rad,
        "phase_ctrl_err_rad": state.phase_ctrl_err_raw_rad,
        "phase_ctrl_err_raw_rad": state.phase_ctrl_err_raw_rad,
        "phase_ctrl_err_filt_rad": state.phase_ctrl_err_filt_rad,
        "phase_setpoint_rad": state.phase_setpoint_rad,
        "setpoint_cal_event": state.setpoint_cal_event,
        "slip_event": int(state.slip_last_cycles != 0),
        "slip_cycles": state.slip_last_cycles,
        "slip_events_total": state.slip_events_total,
        "slip_cycles_total": state.slip_cycles_total,
        "phase_dphi_wrapped_rad": state.phase_dphi_wrapped_rad,
        "phase_dphi_used_rad": state.phase_dphi_used_rad,
        "phase_dphi_expected_rad": state.phase_dphi_expected_rad,
        "freq_err_block_hz": meas.freq_err_block_hz,
        "freq_err_update_hz": meas.freq_err_update_hz,
        "coh": meas.coh,
        "snr_db": meas.snr_db,
        "meas_quality_ok": int(bool(meas_quality_ok)),
        "peak_freq_abs_hz": meas.peak_freq_abs_hz,
        "peak_freq_baseband_hz": meas.peak_freq_bb_hz,
        "f_ref_corr_bb_hz": meas.f_ref_corr_bb_hz,
        "nco_freq_bb_hz": state.nco_freq_bb_hz if state.nco_initialized else "",
        "df_nco_peak_hz": state.df_nco_peak_hz if state.nco_initialized else "",
        "nco_phase_rad": state.nco_phase_rad if state.nco_initialized else "",
        "z_re": meas.z_re,
        "z_im": meas.z_im,
        "z_abs": meas.z_abs,
        "tone_power": meas.tone_power,
        "noise_power": meas.noise_power,
        "beacon_freq_hz": params.beacon_freq_hz,
        "center_freq_hz": params.center_freq_hz,
        "clock0_cmd_hz": state.clock0_cmd_hz,
        "clock0_offset_hz": state.clock0_cmd_hz - params.clock0_nom_hz,
        "dac_code_cmd": state.dac_code_cmd,
        "dac_code_rb": "" if state.dac_code_rb is None else state.dac_code_rb,
        "dac_volts": "" if state.dac_volts is None else state.dac_volts,
        "dac_step_req_codes": "" if dac_step_result is None else dac_step_result.step_req_codes,
        "dac_step_applied_codes": "" if dac_step_result is None else dac_step_result.step_applied_codes,
        "I_term": state.I_term,
        "lock_counter": state.lock_counter,
        "unlock_counter": state.unlock_counter,
        "nsamp_block": params.nsamp_block,
        "blocks_per_update": params.blocks_per_update,
        "sample_rate": params.sample_rate,
        "kv_hz_per_code": params.kv_hz_per_code,
        "notes": args_notes,
    })


# ==========================
# Self tests (deterministic, no hardware)
# ==========================

def _synth_cw(fs, f_hz, n, phase0=0.0, amp=1.0, noise_std=0.0, seed=0):
    n_idx = np.arange(int(n), dtype=np.float64)
    x = float(amp) * np.exp(1j * (float(phase0) + 2.0 * math.pi * float(f_hz) * n_idx / float(fs)))
    if noise_std > 0:
        rs = np.random.RandomState(int(seed))
        noise = noise_std * (rs.randn(int(n)) + 1j * rs.randn(int(n)))
        x = x + noise
    return np.asarray(x, dtype=np.complex64)


def run_self_tests():
    # wrap_to_pi
    assert abs(wrap_to_pi(0.0) - 0.0) < 1e-12
    assert abs(wrap_to_pi(math.pi) - math.pi) < 1e-12
    assert abs(wrap_to_pi(-math.pi) - math.pi) < 1e-12
    assert abs(wrap_to_pi(3.0 * math.pi / 2.0) + math.pi / 2.0) < 1e-12

    # circular mean / LPF
    ph = np.array([math.radians(179.0), math.radians(-179.0)], dtype=np.float64)
    mu = circular_mean(ph)
    assert abs(abs(mu) - math.pi) < math.radians(2.0)
    lpf = CircularLpfState()
    y1 = circular_phase_lpf_update(math.radians(170.0), 0.9, lpf)
    y2 = circular_phase_lpf_update(math.radians(-170.0), 0.9, lpf)
    assert abs(abs(y1) - math.radians(170.0)) < math.radians(1.0)
    assert abs(abs(y2) - math.pi) < math.radians(20.0)

    # Slip-aware unwrap should recover multi-cycle motion when fallback frequency is known.
    hist = PhaseHistory()
    f_slip = 25.0
    dt_slip = 0.2
    _ = hist.update(0.0, 0.0, f_slip)
    wrapped_next = wrap_to_pi(2.0 * math.pi * f_slip * dt_slip)
    _ = hist.update(wrapped_next, dt_slip, f_slip)
    assert hist.last_slip_cycles == 5
    assert hist.slip_events == 1
    assert hist.slip_cycles_total == 5
    assert abs(hist.last_dphi_used_rad - (2.0 * math.pi * f_slip * dt_slip)) < 1e-9

    # NCO phase propagation across blocks
    fs = 2.4e6
    f0 = 500e3
    n = 262144
    ph0 = 1.234
    corr = correlate_phase_freq(_synth_cw(fs, f0, n, phase0=0.2, noise_std=0.0), fs, f0, ph0)
    expected_end = wrap_to_pi(ph0 + 2.0 * math.pi * f0 * (n / fs))
    assert abs(wrap_to_pi(corr.phase_end_rad - expected_end)) < 1e-9

    # PI clamp / saturation
    params = ControlParams()
    params.dac_code_min = 1024
    params.dac_code_max = 1030
    params.dac_code_start = 1029
    params.max_dac_step_per_update = 100
    params.Kp_phi_code_per_rad = 1000.0
    params.Ki_phi_code_per_rad_per_update = 100.0
    st = ControlState(mode=MODE_TRACK, dac_code_cmd=1029)
    out = pi_update_dac(1.0, params, st, clock_obj=None, allow_integrate=True)
    assert out.new_code_cmd == 1030
    assert abs(out.step_applied_codes) <= params.max_dac_step_per_update

    # Feed-forward DAC step from FFT frequency error
    params = ControlParams()
    params.dac_code_min = 1024
    params.dac_code_max = 1030
    params.max_dac_step_per_update = 2
    params.kv_hz_per_code = 2.0
    params.Kf_coarse_clock_hz_per_hz_err = 1.0
    st = ControlState(mode=MODE_ACQUIRE, dac_code_cmd=1025)
    out = dac_feedforward_from_freq_error(10.0, params, st, clock_obj=None)
    assert out.new_code_cmd == 1027
    assert out.step_applied_codes == 2

    # quality_self tone estimate + fixed-tone phase measurement
    true_bb_q = 500e3 + 117.0
    q_blocks = [
        _synth_cw(fs, true_bb_q, n, phase0=0.3, noise_std=0.0, seed=11),
        _synth_cw(
            fs,
            true_bb_q,
            n,
            phase0=0.3 + 2.0 * math.pi * true_bb_q * (n / fs),
            noise_std=0.0,
            seed=12,
        ),
    ]
    est_bb = estimate_dominant_tone_offset_hz(q_blocks, fs, dc_exclusion_hz=1e3)
    assert abs(est_bb - true_bb_q) <= (2.0 * fs / n)
    q_hist = PhaseHistory()
    q1, next_start = build_update_measurement_from_quality_self(
        [q_blocks[0]],
        MeasureConfig(fs=fs, beacon_hz=100e6, center_hz=99.5e6, fft_search_span_hz=50e3),
        q_hist,
        1.0,
        tone_offset_hz=true_bb_q,
        sample_start=0,
    )
    q2, next_start = build_update_measurement_from_quality_self(
        [q_blocks[1]],
        MeasureConfig(fs=fs, beacon_hz=100e6, center_hz=99.5e6, fft_search_span_hz=50e3),
        q_hist,
        1.1,
        tone_offset_hz=true_bb_q,
        sample_start=next_start,
    )
    assert next_start == 2 * n
    assert abs(wrap_to_pi(q2.phase_meas_rad - q1.phase_meas_rad)) < 0.2

    # Synthetic coherence + persistent-NCO phase stability
    mcfg = MeasureConfig(fs=fs, beacon_hz=100e6, center_hz=99.5e6, fft_search_span_hz=50e3)
    true_bb = 500e3 + 123.0
    blocks = [_synth_cw(fs, true_bb, n, phase0=0.4, noise_std=0.02, seed=1),
              _synth_cw(fs, true_bb, n, phase0=0.4 + 2.0 * math.pi * true_bb * (n / fs), noise_std=0.02, seed=2)]
    agg_acq = measure_acquire_snapshot(blocks, mcfg)
    assert agg_acq.coh > 0.2
    hist = PhaseHistory()
    m1, nco_end = build_update_measurement_from_track([blocks[0]], mcfg, hist, 1.0, true_bb, 0.0)
    m2, _ = build_update_measurement_from_track([blocks[1]], mcfg, hist, 1.1, true_bb, nco_end)
    # Same synthetic tone, persistent NCO should keep phase near constant modulo noise.
    assert abs(wrap_to_pi(m2.phase_meas_rad - m1.phase_meas_rad)) < 0.2

    # Frequency mismatch reduces coherence
    good = correlate_phase_freq(_synth_cw(fs, true_bb, n, phase0=0.0, noise_std=0.01, seed=3), fs, true_bb, 0.0)
    bad = correlate_phase_freq(_synth_cw(fs, true_bb, n, phase0=0.0, noise_std=0.01, seed=3), fs, true_bb + 3000.0, 0.0)
    assert good.coh > bad.coh

    print("Self-tests passed.")
    return 0


# ==========================
# Main loop
# ==========================

def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.run_self_tests:
        return run_self_tests()

    params = params_from_args(args)
    log_csv = make_log_path(args.log_csv)
    logger = CsvLogger(log_csv)
    print("Logging to: {}".format(log_csv), flush=True)

    mcfg = MeasureConfig(
        fs=params.sample_rate,
        beacon_hz=params.beacon_freq_hz,
        center_hz=params.center_freq_hz,
        fft_search_span_hz=params.fft_search_span_hz,
    )

    state = ControlState(
        mode=(
            MODE_OPEN
            if params.open_loop
            else (
                MODE_ADJUSTED
                if params.adjusted_loop
                else (
                    MODE_QUALITY_SELF
                    if params.quality_self
                    else (
                        MODE_QUALITY_BEACON
                        if params.quality_beacon
                        else MODE_ACQUIRE
                    )
                )
            )
        ),
        clock0_cmd_hz=float(params.clock0_nom_hz),
        dac_code_cmd=int(params.dac_code_start),
        t_start_s=now_unix(),
        phase_setpoint_rad=float(params.phase_setpoint_rad),
    )
    phase_hist = PhaseHistory()
    cal_state = SetpointCalState(
        enabled=bool(params.calibrate_phase_setpoint),
        target_samples=int(params.setpoint_cal_samples),
        done=(not bool(params.calibrate_phase_setpoint)),
    )

    stopper = GracefulStop()
    sdr = None
    clock_obj = None

    try:
        sdr = sdr_open(params.sample_rate, params.center_freq_hz, params.gain)

        if not (
            params.open_loop
            or params.adjusted_loop
            or params.quality_self
            or params.quality_beacon
        ):
            if cvh is None:
                raise RuntimeError("closed_loop selected but clock_vcxo_helper.py import failed")
            clock_obj = clock_create()
            clock_open(clock_obj)
            clock_init_28p8(clock_obj, apr_ppm=params.apr_ppm)
            state.clock0_cmd_hz = clock_set_coarse_hz(clock_obj, params.clock0_nom_hz, settle_s=params.coarse_settle_s)
            state.dac_code_rb = clock_set_dac_code(clock_obj, state.dac_code_cmd, settle_s=params.fine_settle_s)
            state.dac_volts = clock_dac_code_to_volts(clock_obj, state.dac_code_cmd)
        else:
            state.clock0_cmd_hz = float(params.clock0_nom_hz)
            state.dac_volts = float("nan")

        t_block = float(params.nsamp_block) / float(params.sample_rate)
        t_upd_nom = t_block * float(params.blocks_per_update)
        if params.verbose:
            print(
                "Mode={} | Fs={:.1f} | block={} ({:.3f}s) | blocks/update={} (~{:.3f}s) | center={:.6f} MHz | beacon={:.6f} MHz".format(
                    state.mode, params.sample_rate, params.nsamp_block, t_block,
                    params.blocks_per_update, t_upd_nom,
                    params.center_freq_hz / 1e6, params.beacon_freq_hz / 1e6,
                ),
                flush=True,
            )

        while not stopper.stop:
            if params.run_seconds is not None and (now_unix() - state.t_start_s) >= float(params.run_seconds):
                break
            if params.max_updates is not None and state.update_index >= int(params.max_updates):
                break

            iq_blocks = read_blocks(sdr, params.nsamp_block, params.blocks_per_update)
            t_meas = now_unix()
            state.setpoint_cal_event = 0
            dac_step_result = None
            quality_seeded = False

            if params.quality_self:
                if not state.nco_initialized:
                    if params.quality_self_tone_hz is not None:
                        state.nco_freq_bb_hz = float(params.quality_self_tone_hz)
                    else:
                        state.nco_freq_bb_hz = estimate_dominant_tone_offset_hz(
                            iq_blocks,
                            sample_rate_hz=params.sample_rate,
                            dc_exclusion_hz=params.quality_self_dc_exclusion_hz,
                        )
                    state.nco_initialized = True
                    state.quality_sample_start = 0
                    phase_hist.reset()
                    quality_seeded = True

                meas, next_sample_start = build_update_measurement_from_quality_self(
                    iq_blocks,
                    mcfg,
                    phase_hist,
                    t_meas,
                    tone_offset_hz=state.nco_freq_bb_hz,
                    sample_start=state.quality_sample_start,
                )
                state.quality_sample_start = int(next_sample_start)

            elif params.quality_beacon:
                if state.nco_initialized:
                    meas, nco_end = build_update_measurement_from_track(
                        iq_blocks,
                        mcfg,
                        phase_hist,
                        t_meas,
                        state.nco_freq_bb_hz,
                        state.nco_phase_rad,
                    )
                    state.nco_phase_rad = float(nco_end)
                else:
                    meas = build_update_measurement_from_acquire(
                        iq_blocks, mcfg, phase_hist, t_meas
                    )

            else:
                use_persistent = False
                if params.adjusted_loop:
                    use_persistent = state.nco_initialized
                elif params.open_loop and params.open_loop_persistent_ref:
                    use_persistent = state.nco_initialized
                elif not params.open_loop:
                    use_persistent = (state.mode == MODE_TRACK and state.nco_initialized)

                if use_persistent:
                    meas, nco_end = build_update_measurement_from_track(
                        iq_blocks, mcfg, phase_hist, t_meas,
                        state.nco_freq_bb_hz, state.nco_phase_rad,
                    )
                    state.nco_phase_rad = float(nco_end)
                else:
                    meas = build_update_measurement_from_acquire(iq_blocks, mcfg, phase_hist, t_meas)

            if state.nco_initialized:
                state.df_nco_peak_hz = float(state.nco_freq_bb_hz - meas.peak_freq_bb_hz)
            else:
                state.df_nco_peak_hz = 0.0
            sync_state_with_phase_history(state, phase_hist)

            state.phase_ctrl_err_raw_rad = wrap_to_pi(meas.phase_meas_rad - state.phase_setpoint_rad)
            state.phase_ctrl_err_filt_rad = state.phase_ctrl_err_raw_rad

            if params.quality_self:
                state.mode = MODE_QUALITY_SELF
                state.action = "qself_seed_tone" if quality_seeded else "qself_monitor"
                state.phase_ctrl_err_raw_rad = float(meas.phase_meas_rad)
                state.phase_ctrl_err_filt_rad = float(meas.phase_meas_rad)

            elif params.quality_beacon:
                state.mode = MODE_QUALITY_BEACON
                if not state.nco_initialized:
                    if _snr_ok_for_acq(params, meas):
                        state.nco_freq_bb_hz = float(meas.peak_freq_bb_hz)
                        state.nco_phase_rad = 0.0
                        state.nco_initialized = True
                        state.df_nco_peak_hz = 0.0
                        phase_hist.reset()
                        state.action = "qbeacon_seed_nco"
                    else:
                        state.action = "qbeacon_wait_seed"
                else:
                    state.action = "qbeacon_monitor"
                state.phase_ctrl_err_raw_rad = float(meas.phase_meas_rad)
                state.phase_ctrl_err_filt_rad = float(meas.phase_meas_rad)

            elif params.open_loop:
                state.mode = MODE_OPEN
                if params.open_loop_persistent_ref:
                    if not state.nco_initialized:
                        if _snr_ok_for_acq(params, meas):
                            state.nco_freq_bb_hz = float(meas.peak_freq_bb_hz)
                            state.nco_phase_rad = 0.0
                            state.nco_initialized = True
                            state.df_nco_peak_hz = 0.0
                            phase_hist.reset()
                            state.action = "open_seed_nco"
                        else:
                            state.action = "open_wait_seed"
                    else:
                        state.action = "open_persistent_monitor"
                else:
                    state.action = "open_stream"
                    state.nco_initialized = False

            elif params.adjusted_loop:
                state.mode = MODE_ADJUSTED
                if not state.nco_initialized and _snr_ok_for_acq(params, meas):
                    state.nco_freq_bb_hz = float(meas.peak_freq_bb_hz)
                    state.nco_phase_rad = 0.0
                    state.nco_initialized = True
                    phase_hist.reset()
                    state.action = "adjust_seed_nco"
                else:
                    state.action = "adjust_monitor"
                    if state.nco_initialized and params.track_use_fft_freq_assist:
                        a = clamp(params.track_nco_alpha, 0.0, 0.999999)
                        state.nco_freq_bb_hz = a * state.nco_freq_bb_hz + (1.0 - a) * float(meas.peak_freq_bb_hz)
                        state.df_nco_peak_hz = float(state.nco_freq_bb_hz - meas.peak_freq_bb_hz)

                if state.nco_initialized:
                    state.phase_ctrl_err_raw_rad = wrap_to_pi(meas.phase_meas_rad - state.phase_setpoint_rad)
                    state.phase_ctrl_err_filt_rad = circular_phase_lpf_update(
                        state.phase_ctrl_err_raw_rad, params.phase_filt_alpha, state.phase_filt
                    )

                if state.nco_initialized and _phase_quality_ok_for_ctrl(params, meas) and (not cal_state.done):
                    done_now, setpoint = cal_state.push(meas.phase_meas_rad)
                    if done_now:
                        state.phase_setpoint_rad = float(setpoint)
                        state.phase_filt = CircularLpfState()
                        state.setpoint_cal_event = 1
                        state.action = "adjust_setpoint_cal"

            else:
                if state.mode == MODE_ACQUIRE:
                    snr_ok = _snr_ok_for_acq(params, meas)
                    if not snr_ok:
                        state.lock_counter = 0
                        state.unlock_counter += 1
                        state.action = "acq_hold_low_snr"
                    else:
                        state.unlock_counter = 0
                        fft_freq_err = float(meas.peak_freq_abs_hz - params.beacon_freq_hz)
                        if params.fft_pullin_enable and (not state.fft_pullin_done):
                            changed = coarse_update_from_freq_error(
                                fft_freq_err, params, state, clock_obj
                            )
                            state.action = "acq_coarse_fft" if changed else "acq_coarse_fft_hold"
                            state.fft_pullin_done = True
                        elif params.fft_pullin_enable and (not state.fft_dac_pretrim_done):
                            dac_step_result = dac_feedforward_from_freq_error(
                                fft_freq_err, params, state, clock_obj
                            )
                            state.action = "acq_dac_fft" if dac_step_result.changed else "acq_dac_fft_hold"
                            state.fft_dac_pretrim_done = True
                        else:
                            changed = coarse_update_from_freq_error(meas.freq_err_block_hz, params, state, clock_obj)
                            state.action = "acq_coarse_fll" if changed else "acq_coarse_fll_hold"

                        if (abs(meas.freq_err_block_hz) < params.f_switch_hz and
                                meas.coh >= params.coh_min_trk and
                                meas.snr_db >= params.snr_min_trk_db):
                            state.lock_counter += 1
                        else:
                            state.lock_counter = 0

                        if state.lock_counter >= params.lock_count_required:
                            state.mode = MODE_TRACK
                            state.lock_counter = 0
                            state.unlock_counter = 0
                            state.I_term = 0.0
                            state.nco_freq_bb_hz = float(meas.peak_freq_bb_hz)
                            state.nco_phase_rad = 0.0
                            state.nco_initialized = True
                            state.df_nco_peak_hz = 0.0
                            state.phase_filt = CircularLpfState()
                            phase_hist.reset()
                            state.action = "switch_to_track"

                elif state.mode == MODE_TRACK:
                    if params.track_use_fft_freq_assist and state.nco_initialized:
                        a = clamp(params.track_nco_alpha, 0.0, 0.999999)
                        state.nco_freq_bb_hz = a * state.nco_freq_bb_hz + (1.0 - a) * float(meas.peak_freq_bb_hz)
                        state.df_nco_peak_hz = float(state.nco_freq_bb_hz - meas.peak_freq_bb_hz)

                    stay_ok = _phase_quality_ok_for_track(params, meas)
                    ctrl_ok = _phase_quality_ok_for_ctrl(params, meas)
                    state.phase_ctrl_err_raw_rad = wrap_to_pi(meas.phase_meas_rad - state.phase_setpoint_rad)
                    state.phase_ctrl_err_filt_rad = circular_phase_lpf_update(
                        state.phase_ctrl_err_raw_rad, params.phase_filt_alpha, state.phase_filt
                    )

                    if ctrl_ok and (not cal_state.done):
                        done_now, setpoint = cal_state.push(meas.phase_meas_rad)
                        if done_now:
                            state.phase_setpoint_rad = float(setpoint)
                            state.phase_filt = CircularLpfState()
                            state.I_term = 0.0
                            state.phase_ctrl_err_raw_rad = wrap_to_pi(meas.phase_meas_rad - state.phase_setpoint_rad)
                            state.phase_ctrl_err_filt_rad = state.phase_ctrl_err_raw_rad
                            state.setpoint_cal_event = 1
                            state.action = "track_setpoint_cal"

                    if not stay_ok:
                        state.unlock_counter += 1
                        holdover_update(params, state)
                        state.action = "trk_hold_lowq"
                        if state.unlock_counter >= params.unlock_count_required:
                            state.mode = MODE_HOLDOVER
                            state.action = "to_holdover"
                    elif _track_stability_inconsistent(params, state, meas):
                        # Reacquire only when frequency consistency collapses; do not trigger solely on phase ~ pi.
                        state.mode = MODE_ACQUIRE
                        state.lock_counter = 0
                        state.unlock_counter = 0
                        state.nco_initialized = False
                        state.phase_filt = CircularLpfState()
                        phase_hist.reset()
                        state.action = "trk_reacquire_freq"
                    else:
                        state.unlock_counter = 0
                        if ctrl_ok:
                            dac_step_result = pi_update_dac(
                                state.phase_ctrl_err_filt_rad, params, state, clock_obj,
                                allow_integrate=True,
                            )
                            state.action = "dac_fine" if dac_step_result.changed else "dac_hold"
                        else:
                            # freeze integrator and actuator when phase quality is below control gate
                            _ = pi_update_dac(
                                state.phase_ctrl_err_filt_rad, params, state, clock_obj=None,
                                allow_integrate=False,
                            )
                            state.action = "trk_hold_ctrlq"

                elif state.mode == MODE_HOLDOVER:
                    holdover_update(params, state)
                    state.action = "holdover"
                    if _snr_ok_for_acq(params, meas) and (meas.coh >= params.coh_min_acq):
                        state.mode = MODE_ACQUIRE
                        state.unlock_counter = 0
                        state.lock_counter = 0
                        state.nco_initialized = False
                        phase_hist.reset()
                        state.action = "recover_to_acq"
                else:
                    state.mode = MODE_ACQUIRE
                    state.action = "reset_mode"

            if (
                not (
                    params.open_loop
                    or params.adjusted_loop
                    or params.quality_self
                    or params.quality_beacon
                )
            ) and clock_obj is not None and state.dac_volts is None:
                try:
                    state.dac_volts = clock_dac_code_to_volts(clock_obj, state.dac_code_cmd)
                except Exception:
                    state.dac_volts = float("nan")

            dt_update_s = None
            if state.t_prev_meas_s is not None:
                dt_update_s = t_meas - state.t_prev_meas_s
            state.t_prev_meas_s = float(t_meas)

            state.update_index += 1
            meas_quality_ok = _phase_quality_ok_for_ctrl(params, meas)
            print_status(params, state, meas)
            log_row(logger, params, state, meas, args.notes, dt_update_s, dac_step_result, meas_quality_ok)

    except KeyboardInterrupt:
        print("\nInterrupted. Closing SDR/I2C handles (DAC left unchanged).", flush=True)
    except Exception as exc:
        print("\nERROR: {}".format(exc), file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        try:
            logger.close()
        except Exception:
            pass
        sdr_close(sdr)
        if clock_obj is not None:
            clock_close(clock_obj)

    return 0


if __name__ == "__main__":
    sys.exit(main())
