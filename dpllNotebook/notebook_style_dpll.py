#!/usr/bin/env python3
"""Notebook-style live DPLL with fixed tone reference and absolute DAC targeting."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ugradio import sdr as ugradio_sdr

    _UGRADIO_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    ugradio_sdr = None
    _UGRADIO_IMPORT_ERROR = exc

try:
    import clock_vcxo_helper as cvh
except Exception:
    cvh = None


DAC_FS_CODE = 4095
VDD_DAC_DEFAULT = 3.3


def wrap_to_pi(x: float) -> float:
    y = (float(x) + math.pi) % (2.0 * math.pi) - math.pi
    if y <= -math.pi:
        y += 2.0 * math.pi
    return float(y)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(x))))


def clamp_int(x: int | float, lo: int, hi: int) -> int:
    return int(max(int(lo), min(int(hi), int(round(float(x))))))


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def pct_to_code(pct: float) -> int:
    return clamp_int((float(pct) / 100.0) * float(DAC_FS_CODE), 0, DAC_FS_CODE)


def code_to_volts(code: int, vdd_dac: float = VDD_DAC_DEFAULT) -> float:
    return (float(clamp_int(code, 0, DAC_FS_CODE)) / float(DAC_FS_CODE)) * float(vdd_dac)


@dataclass
class QueuedBlock:
    unix_time_s: float
    sample_start: int
    iq_block: np.ndarray


@dataclass
class StreamSentinel:
    error: BaseException | None = None


@dataclass
class CircularLpfState:
    initialized: bool = False
    z_re: float = 1.0
    z_im: float = 0.0


def circular_phase_lpf_update(phase_rad: float, alpha: float, state: CircularLpfState) -> float:
    a = clamp(alpha, 0.0, 0.999999)
    xr = math.cos(float(phase_rad))
    xi = math.sin(float(phase_rad))
    if not state.initialized:
        state.z_re = xr
        state.z_im = xi
        state.initialized = True
    else:
        state.z_re = (a * state.z_re) + ((1.0 - a) * xr)
        state.z_im = (a * state.z_im) + ((1.0 - a) * xi)
    return float(math.atan2(state.z_im, state.z_re))


@dataclass
class SlipAwareUnwrapper:
    prev_wrapped_phase_rad: float | None = None
    prev_time_s: float | None = None
    unwrapped_phase_rad: float = 0.0
    slip_trigger_rad: float = math.pi
    slip_max_cycles_per_update: int = 8

    def update(
        self,
        wrapped_phase_rad: float,
        t_now_s: float,
        fallback_freq_hz: float,
    ) -> tuple[float, float, float, int, float, float, float]:
        phase = float(wrapped_phase_rad)
        t_now = float(t_now_s)
        fb = float(fallback_freq_hz) if np.isfinite(fallback_freq_hz) else 0.0

        if self.prev_wrapped_phase_rad is None or self.prev_time_s is None:
            self.prev_wrapped_phase_rad = phase
            self.prev_time_s = t_now
            self.unwrapped_phase_rad = phase
            return float(self.unwrapped_phase_rad), float(fb), 0.0, 0, 0.0, 0.0, 0.0

        dt = float(t_now - self.prev_time_s)
        if dt <= 0.0 or (not np.isfinite(dt)):
            dt = 0.0

        dphi_wrapped = wrap_to_pi(phase - self.prev_wrapped_phase_rad)
        dphi_used = float(dphi_wrapped)
        expected_dphi = 0.0
        slip_cycles = 0

        if dt > 0.0:
            expected_dphi = 2.0 * math.pi * fb * dt
            err = expected_dphi - dphi_wrapped
            if abs(err) >= float(self.slip_trigger_rad):
                k = int(round(err / (2.0 * math.pi)))
                k = clamp_int(k, -int(self.slip_max_cycles_per_update), int(self.slip_max_cycles_per_update))
                if k != 0:
                    candidate = float(dphi_wrapped + (2.0 * math.pi * float(k)))
                    if abs(candidate - expected_dphi) < abs(dphi_wrapped - expected_dphi):
                        dphi_used = candidate
                        slip_cycles = int(k)

        if dt > 0.0:
            freq_update_hz = float(dphi_used / (2.0 * math.pi * dt))
        else:
            freq_update_hz = float(fb)

        self.unwrapped_phase_rad += dphi_used
        self.prev_wrapped_phase_rad = phase
        self.prev_time_s = t_now
        return (
            float(self.unwrapped_phase_rad),
            float(freq_update_hz),
            float(dt),
            int(slip_cycles),
            float(dphi_wrapped),
            float(dphi_used),
            float(expected_dphi),
        )


@dataclass
class Measurement:
    update_idx: int
    unix_time_s: float
    elapsed_s: float
    sample_start: int
    tone_offset_hz: float
    mix_real: float
    mix_imag: float
    mix_abs: float
    wrapped_phase_rad: float
    unwrapped_phase_rad: float
    coherence: float
    pullin_quality: float
    amp_rms: float
    freq_err_intrablock_hz: float
    freq_err_update_hz: float
    dt_update_s: float
    slip_cycles: int
    dphi_wrapped_rad: float
    dphi_used_rad: float
    dphi_expected_rad: float


@dataclass
class ControllerState:
    phase_err_lpf: CircularLpfState = field(default_factory=CircularLpfState)
    dac_code_cmd: int = 0
    dac_code_rb: int | None = None
    dac_volts: float = float("nan")
    timing_proxy_ref_phase_rad: float | None = None
    applied_updates: int = 0
    held_updates: int = 0
    low_quality_holds: int = 0


@dataclass
class ControlDecision:
    phase_fraction_cycles_per_sample: float
    dac_target_float: float
    dac_target_code: int
    dac_step_req_codes: int
    dac_code_proposed: int


CSV_COLUMNS = [
    "unix_time_s",
    "elapsed_s",
    "update_idx",
    "mode",
    "action",
    "sample_start",
    "nsamp",
    "sample_rate_hz",
    "center_freq_hz",
    "tone_offset_hz",
    "rf_reference_hz",
    "mix_real",
    "mix_imag",
    "mix_abs",
    "phase_wrapped_rad",
    "phase_unwrapped_rad",
    "phase_setpoint_rad",
    "phase_err_raw_rad",
    "phase_err_filt_rad",
    "phase_fraction_cycles_per_sample",
    "coherence",
    "pullin_quality",
    "coherence_ok",
    "pullin_quality_ok",
    "control_quality_ok",
    "amp_rms",
    "freq_err_intrablock_hz",
    "freq_err_update_hz",
    "timing_proxy_ns",
    "slip_cycles",
    "dphi_wrapped_rad",
    "dphi_used_rad",
    "dphi_expected_rad",
    "dac_center_code",
    "dac_target_float",
    "dac_target_code",
    "dac_code_cmd",
    "dac_code_rb",
    "dac_code_proposed",
    "dac_volts",
    "dac_step_req_codes",
    "dac_step_applied_codes",
    "phase_frac_gain_codes",
    "max_dac_step_per_update",
    "notes",
]


class CsvLogger:
    def __init__(self, path: Path):
        self.path = path
        ensure_parent_dir(path)
        self.fh = path.open("w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()
        self.fh.flush()

    def write(self, row: dict[str, Any]) -> None:
        out = {k: row.get(k, "") for k in CSV_COLUMNS}
        self.writer.writerow(out)
        self.fh.flush()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass


def bytes_to_iq_block(raw_bytes: bytes, nsamples: int) -> np.ndarray:
    expected = 2 * int(nsamples)
    if len(raw_bytes) != expected:
        raise RuntimeError(f"Expected {expected} bytes for one IQ block, received {len(raw_bytes)}.")
    raw = (np.frombuffer(raw_bytes, dtype=np.uint8) - 128).view(np.int8)
    raw = raw.reshape(int(nsamples), 2).astype(np.float32, copy=False)
    return (raw[:, 0] + 1j * raw[:, 1]).astype(np.complex64)


def open_sdr(args: argparse.Namespace) -> Any:
    if ugradio_sdr is None:
        raise RuntimeError(f"ugradio.sdr import failed: {_UGRADIO_IMPORT_ERROR}")
    return ugradio_sdr.SDR(
        device_index=int(args.device_index),
        direct=False,
        center_freq=float(args.center_freq),
        sample_rate=float(args.sample_rate),
        gain=float(args.gain),
    )


def default_async_queue_blocks() -> int:
    return 8


def signal_stream_end(
    queue: asyncio.Queue[QueuedBlock | StreamSentinel],
    error: BaseException | None = None,
) -> None:
    sentinel = StreamSentinel(error=error)
    while True:
        try:
            queue.put_nowait(sentinel)
            return
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return


async def stream_iq_blocks(
    device: Any,
    nsamples: int,
    queue: asyncio.Queue[QueuedBlock | StreamSentinel],
) -> None:
    sample_start = 0
    try:
        async for raw in device.stream(
            num_samples_or_bytes=2 * int(nsamples),
            format="bytes",
        ):
            iq_block = bytes_to_iq_block(raw, nsamples)
            await queue.put(
                QueuedBlock(
                    unix_time_s=float(time.time()),
                    sample_start=int(sample_start),
                    iq_block=iq_block,
                )
            )
            sample_start += int(iq_block.size)
    except asyncio.CancelledError:
        signal_stream_end(queue)
        raise
    except Exception as exc:
        signal_stream_end(queue, error=exc)
        raise
    else:
        signal_stream_end(queue)
    finally:
        try:
            await device.stop()
        except Exception:
            pass


def estimate_tone_offset_hz(
    iq_blocks: np.ndarray,
    sample_rate_hz: float,
    dc_exclusion_hz: float,
) -> float:
    spectra = np.fft.fft(iq_blocks, axis=1)
    power = np.mean(np.abs(spectra) ** 2, axis=0)
    freqs = np.fft.fftfreq(iq_blocks.shape[1], d=1.0 / float(sample_rate_hz))

    mask = np.ones(freqs.shape, dtype=bool)
    if float(dc_exclusion_hz) > 0.0:
        mask &= np.abs(freqs) >= float(dc_exclusion_hz)
    if not np.any(mask):
        mask[:] = True

    idx_candidates = np.flatnonzero(mask)
    peak_idx = int(idx_candidates[int(np.argmax(power[mask]))])
    return float(freqs[peak_idx])


def phase_increment_stats(z: np.ndarray, sample_rate_hz: float) -> tuple[float, float]:
    x = np.asarray(z, dtype=np.complex64).reshape(-1)
    if x.size < 2:
        return float("nan"), float("nan")
    prod = x[1:] * np.conjugate(x[:-1])
    s = np.sum(prod, dtype=np.complex128)
    if not (np.isfinite(s.real) and np.isfinite(s.imag)):
        return float("nan"), float("nan")
    denom = float(np.sum(np.abs(prod), dtype=np.float64))
    if denom <= 0.0 or (not np.isfinite(denom)):
        quality = float("nan")
    else:
        quality = float(np.abs(s) / (denom + 1e-12))
    freq_hz = float((float(sample_rate_hz) / (2.0 * math.pi)) * np.angle(s))
    return float(freq_hz), float(quality)


def measure_block(
    queued_block: QueuedBlock,
    sample_rate_hz: float,
    tone_offset_hz: float,
    update_idx: int,
    unwrapper: SlipAwareUnwrapper,
) -> Measurement:
    iq = np.asarray(queued_block.iq_block, dtype=np.complex64).reshape(-1)
    iq = iq - np.mean(iq)

    n = float(queued_block.sample_start) + np.arange(iq.size, dtype=np.float64)
    rotator = np.exp(-2j * np.pi * float(tone_offset_hz) * n / float(sample_rate_hz)).astype(np.complex64)
    demod = iq * rotator

    mix = np.mean(demod, dtype=np.complex128)
    wrapped_phase = float(np.angle(mix))
    coherence = float(np.abs(mix) / (np.mean(np.abs(iq)) + 1e-12))
    amp_rms = float(np.sqrt(np.mean(np.abs(demod) ** 2) + 1e-30))
    freq_intrablock, pullin_quality = phase_increment_stats(demod, sample_rate_hz)

    elapsed_s = float((float(queued_block.sample_start) + (0.5 * float(iq.size))) / float(sample_rate_hz))
    (
        unwrapped_phase,
        freq_update_hz,
        dt_update_s,
        slip_cycles,
        dphi_wrapped,
        dphi_used,
        dphi_expected,
    ) = unwrapper.update(
        wrapped_phase_rad=wrapped_phase,
        t_now_s=elapsed_s,
        fallback_freq_hz=freq_intrablock,
    )

    return Measurement(
        update_idx=int(update_idx),
        unix_time_s=float(queued_block.unix_time_s),
        elapsed_s=float(elapsed_s),
        sample_start=int(queued_block.sample_start),
        tone_offset_hz=float(tone_offset_hz),
        mix_real=float(mix.real),
        mix_imag=float(mix.imag),
        mix_abs=float(abs(mix)),
        wrapped_phase_rad=float(wrapped_phase),
        unwrapped_phase_rad=float(unwrapped_phase),
        coherence=float(coherence),
        pullin_quality=float(pullin_quality),
        amp_rms=float(amp_rms),
        freq_err_intrablock_hz=float(freq_intrablock),
        freq_err_update_hz=float(freq_update_hz),
        dt_update_s=float(dt_update_s),
        slip_cycles=int(slip_cycles),
        dphi_wrapped_rad=float(dphi_wrapped),
        dphi_used_rad=float(dphi_used),
        dphi_expected_rad=float(dphi_expected),
    )


def linear_slope(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    if xa.size < 2 or ya.size != xa.size:
        return float("nan")
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        return float("nan")
    x0 = xa - np.mean(xa)
    denom = float(np.sum(x0 * x0))
    if denom <= 0.0:
        return float("nan")
    num = float(np.sum(x0 * (ya - np.mean(ya))))
    return float(num / denom)


def phase_to_timing_proxy_ns(
    unwrapped_phase_rad: float,
    ref_phase_rad: float | None,
    rf_hz: float,
) -> float:
    if ref_phase_rad is None:
        return float("nan")
    if not (np.isfinite(unwrapped_phase_rad) and np.isfinite(ref_phase_rad) and np.isfinite(rf_hz)):
        return float("nan")
    if abs(float(rf_hz)) <= 1e-6:
        return float("nan")
    return float(((float(unwrapped_phase_rad) - float(ref_phase_rad)) / (2.0 * math.pi * float(rf_hz))) * 1e9)


def phase_err_to_dac_target(
    *,
    phase_err_filt_rad: float,
    nsamp: int,
    dac_center: int,
    phase_frac_gain_codes: float,
    control_sign: float,
    code_min: int,
    code_max: int,
) -> tuple[float, float, int]:
    phase_fraction = float(float(phase_err_filt_rad) / (2.0 * math.pi * float(nsamp)))
    dac_target_float = float(dac_center) + (
        float(control_sign) * float(phase_frac_gain_codes) * float(phase_fraction)
    )
    dac_target_code = clamp_int(dac_target_float, int(code_min), int(code_max))
    return float(phase_fraction), float(dac_target_float), int(dac_target_code)


def compute_control_decision(
    *,
    current_code: int,
    phase_err_filt_rad: float,
    nsamp: int,
    dac_center: int,
    phase_frac_gain_codes: float,
    control_sign: float,
    code_min: int,
    code_max: int,
    max_dac_step_per_update: int,
    quality_ok: bool,
) -> ControlDecision:
    phase_fraction, dac_target_float, dac_target_code = phase_err_to_dac_target(
        phase_err_filt_rad=float(phase_err_filt_rad),
        nsamp=int(nsamp),
        dac_center=int(dac_center),
        phase_frac_gain_codes=float(phase_frac_gain_codes),
        control_sign=float(control_sign),
        code_min=int(code_min),
        code_max=int(code_max),
    )

    if not quality_ok:
        return ControlDecision(
            phase_fraction_cycles_per_sample=float(phase_fraction),
            dac_target_float=float(dac_target_float),
            dac_target_code=int(dac_target_code),
            dac_step_req_codes=0,
            dac_code_proposed=int(current_code),
        )

    step_to_target = int(dac_target_code) - int(current_code)
    dac_step_req = clamp_int(
        step_to_target,
        -int(max_dac_step_per_update),
        int(max_dac_step_per_update),
    )
    dac_code_proposed = clamp_int(
        int(current_code) + int(dac_step_req),
        int(code_min),
        int(code_max),
    )
    return ControlDecision(
        phase_fraction_cycles_per_sample=float(phase_fraction),
        dac_target_float=float(dac_target_float),
        dac_target_code=int(dac_target_code),
        dac_step_req_codes=int(dac_step_req),
        dac_code_proposed=int(dac_code_proposed),
    )


def run_controller_self_checks() -> None:
    phase_fraction, dac_target_float, dac_target_code = phase_err_to_dac_target(
        phase_err_filt_rad=0.0,
        nsamp=256,
        dac_center=2048,
        phase_frac_gain_codes=1000.0,
        control_sign=1.0,
        code_min=0,
        code_max=4095,
    )
    if phase_fraction != 0.0 or dac_target_float != 2048.0 or dac_target_code != 2048:
        raise AssertionError("Zero phase error must map to dac_center.")

    _, pos_float, _ = phase_err_to_dac_target(
        phase_err_filt_rad=0.5,
        nsamp=128,
        dac_center=2000,
        phase_frac_gain_codes=2000.0,
        control_sign=1.0,
        code_min=0,
        code_max=4095,
    )
    _, neg_float, _ = phase_err_to_dac_target(
        phase_err_filt_rad=-0.5,
        nsamp=128,
        dac_center=2000,
        phase_frac_gain_codes=2000.0,
        control_sign=1.0,
        code_min=0,
        code_max=4095,
    )
    if abs((pos_float - 2000.0) + (neg_float - 2000.0)) > 1e-9:
        raise AssertionError("Positive and negative phase errors must map symmetrically about dac_center.")

    _, _, clamped_high = phase_err_to_dac_target(
        phase_err_filt_rad=10.0,
        nsamp=1,
        dac_center=2048,
        phase_frac_gain_codes=1e6,
        control_sign=1.0,
        code_min=100,
        code_max=3000,
    )
    _, _, clamped_low = phase_err_to_dac_target(
        phase_err_filt_rad=-10.0,
        nsamp=1,
        dac_center=2048,
        phase_frac_gain_codes=1e6,
        control_sign=1.0,
        code_min=100,
        code_max=3000,
    )
    if clamped_high != 3000 or clamped_low != 100:
        raise AssertionError("DAC targets must clamp to configured bounds.")

    hold_decision = compute_control_decision(
        current_code=1500,
        phase_err_filt_rad=1.0,
        nsamp=1,
        dac_center=2048,
        phase_frac_gain_codes=1e6,
        control_sign=1.0,
        code_min=0,
        code_max=4095,
        max_dac_step_per_update=8,
        quality_ok=False,
    )
    if hold_decision.dac_code_proposed != 1500 or hold_decision.dac_step_req_codes != 0:
        raise AssertionError("Low-quality updates must hold the previous DAC command.")

    limited_decision = compute_control_decision(
        current_code=1000,
        phase_err_filt_rad=10.0,
        nsamp=1,
        dac_center=2048,
        phase_frac_gain_codes=1e6,
        control_sign=1.0,
        code_min=0,
        code_max=4095,
        max_dac_step_per_update=7,
        quality_ok=True,
    )
    if abs(int(limited_decision.dac_step_req_codes)) > 7:
        raise AssertionError("DAC rate limiting must respect max_dac_step_per_update.")


def compute_summary(
    *,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    tone_offset_hz: float,
    dac_center: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "mode": str(args.mode),
            "nsamp": int(args.nsamp),
            "sample_rate_hz": float(args.sample_rate),
            "center_freq_hz": float(args.center_freq),
            "loop_period_nom_s": float(args.nsamp) / float(args.sample_rate),
            "tone_offset_hz": float(tone_offset_hz),
            "updates": 0,
            "elapsed_s": 0.0,
            "median_coherence": float("nan"),
            "median_pullin_quality": float("nan"),
            "mean_abs_phase_err_rad": float("nan"),
            "median_abs_phase_err_rad": float("nan"),
            "mean_abs_phase_err_filt_rad": float("nan"),
            "median_abs_phase_err_filt_rad": float("nan"),
            "dac_center_code": int(dac_center),
            "dac_updates_applied": 0,
            "dac_updates_held": 0,
            "dac_update_fraction": float("nan"),
            "low_quality_holds": 0,
            "timing_proxy_drift_ns_per_s": float("nan"),
            "timing_proxy_drift_samples_per_s": float("nan"),
            "max_abs_timing_proxy_ns": float("nan"),
            "p95_abs_timing_proxy_ns": float("nan"),
        }

    def _arr(key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, float("nan"))) for r in rows], dtype=np.float64)

    t = _arr("elapsed_s")
    coh = _arr("coherence")
    pq = _arr("pullin_quality")
    pe = _arr("phase_err_raw_rad")
    pe_f = _arr("phase_err_filt_rad")
    timing = _arr("timing_proxy_ns")
    applied = np.asarray([bool(r.get("dac_applied", False)) for r in rows], dtype=bool)
    low_quality = np.asarray([not bool(r.get("control_quality_ok", False)) for r in rows], dtype=bool)

    finite_timing = np.isfinite(t) & np.isfinite(timing)
    timing_proxy_drift_ns_per_s = (
        linear_slope(t[finite_timing], timing[finite_timing]) if np.count_nonzero(finite_timing) >= 2 else float("nan")
    )
    timing_proxy_drift_samples_per_s = (
        float(timing_proxy_drift_ns_per_s * float(args.sample_rate) / 1e9)
        if np.isfinite(timing_proxy_drift_ns_per_s)
        else float("nan")
    )

    abs_timing = np.abs(timing[np.isfinite(timing)])
    elapsed_s = float(max(0.0, float(t[-1] - t[0]))) if t.size else 0.0

    return {
        "mode": str(args.mode),
        "nsamp": int(args.nsamp),
        "sample_rate_hz": float(args.sample_rate),
        "center_freq_hz": float(args.center_freq),
        "loop_period_nom_s": float(args.nsamp) / float(args.sample_rate),
        "tone_offset_hz": float(tone_offset_hz),
        "updates": int(len(rows)),
        "elapsed_s": float(elapsed_s),
        "median_coherence": float(np.nanmedian(coh)) if coh.size else float("nan"),
        "median_pullin_quality": float(np.nanmedian(pq)) if pq.size else float("nan"),
        "mean_abs_phase_err_rad": float(np.nanmean(np.abs(pe))) if pe.size else float("nan"),
        "median_abs_phase_err_rad": float(np.nanmedian(np.abs(pe))) if pe.size else float("nan"),
        "mean_abs_phase_err_filt_rad": float(np.nanmean(np.abs(pe_f))) if pe_f.size else float("nan"),
        "median_abs_phase_err_filt_rad": float(np.nanmedian(np.abs(pe_f))) if pe_f.size else float("nan"),
        "dac_center_code": int(dac_center),
        "dac_updates_applied": int(np.count_nonzero(applied)),
        "dac_updates_held": int(len(rows) - np.count_nonzero(applied)),
        "dac_update_fraction": float(np.count_nonzero(applied) / len(rows)),
        "low_quality_holds": int(np.count_nonzero(low_quality)),
        "timing_proxy_drift_ns_per_s": float(timing_proxy_drift_ns_per_s),
        "timing_proxy_drift_samples_per_s": float(timing_proxy_drift_samples_per_s),
        "max_abs_timing_proxy_ns": float(np.nanmax(abs_timing)) if abs_timing.size else float("nan"),
        "p95_abs_timing_proxy_ns": float(np.nanpercentile(abs_timing, 95.0)) if abs_timing.size else float("nan"),
    }


def print_run_summary(summary: dict[str, Any]) -> None:
    print("\nRun summary:")
    print(f"  mode: {summary.get('mode')}")
    print(f"  nsamp: {summary.get('nsamp')}")
    print(f"  sample_rate_hz: {summary.get('sample_rate_hz')}")
    print(f"  tone_offset_hz: {summary.get('tone_offset_hz')}")
    print(f"  updates: {summary.get('updates')}")
    print(f"  median_coherence: {summary.get('median_coherence')}")
    print(f"  median_pullin_quality: {summary.get('median_pullin_quality')}")
    print(f"  mean_abs_phase_err_rad: {summary.get('mean_abs_phase_err_rad')}")
    print(f"  dac_updates_applied: {summary.get('dac_updates_applied')}")
    print(f"  dac_update_fraction: {summary.get('dac_update_fraction')}")
    print(f"  timing_proxy_drift_ns_per_s: {summary.get('timing_proxy_drift_ns_per_s')}")
    print(f"  timing_proxy_drift_samples_per_s: {summary.get('timing_proxy_drift_samples_per_s')}")
    print(f"  csv_path: {summary.get('csv_path')}")


async def run_one(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.nsamp) <= 0:
        raise ValueError("nsamp must be positive.")
    if args.mode == "pid" and cvh is None:
        raise RuntimeError("clock_vcxo_helper.py import failed, but mode=pid requires DAC control access.")

    code_min = pct_to_code(float(args.dac_min_pct))
    code_max = pct_to_code(float(args.dac_max_pct))
    if code_min >= code_max:
        raise ValueError("DAC range invalid: ensure dac_min_pct < dac_max_pct.")

    if args.dac_center is None:
        dac_center = int(round((code_min + code_max) / 2.0))
    else:
        dac_center = clamp_int(int(args.dac_center), code_min, code_max)

    nominal_loop_period_s = float(args.nsamp) / float(args.sample_rate)
    queue_get_timeout_s = max(1.0, nominal_loop_period_s)

    logger = CsvLogger(Path(args.csv_out))
    device = None
    producer_task: asyncio.Task[None] | None = None
    clock_obj = None

    rows: list[dict[str, Any]] = []
    tone_offset_hz: float | None = None if args.target_offset_hz is None else float(args.target_offset_hz)
    state = ControllerState(
        dac_code_cmd=int(dac_center),
        dac_code_rb=None,
        dac_volts=code_to_volts(int(dac_center)),
    )
    unwrapper = SlipAwareUnwrapper()

    update_idx = 0
    bootstrap_blocks: list[QueuedBlock] = []
    wall_start_s = float(time.time())

    try:
        device = open_sdr(args)
        if float(args.settle_seconds) > 0.0:
            await asyncio.sleep(float(args.settle_seconds))

        block_queue: asyncio.Queue[QueuedBlock | StreamSentinel] = asyncio.Queue(
            maxsize=default_async_queue_blocks()
        )
        producer_task = asyncio.create_task(stream_iq_blocks(device, int(args.nsamp), block_queue))

        if args.mode == "pid":
            clock_obj = cvh.ClockVCXO()
            clock_obj.open()
            if not args.skip_clock_init:
                clock_obj.init_clock_vcxo_28p8mhz(apr_ppm=int(args.apr_ppm))
            if args.apply_start_dac:
                rb0 = int(clock_obj.set_dac_code(int(dac_center), settle_s=float(args.fine_settle_s)))
                state.dac_code_cmd = int(rb0)
                state.dac_code_rb = int(rb0)
                state.dac_volts = float(clock_obj.dac_code_to_volts(int(rb0)))

        stop_requested = False
        while not stop_requested:
            if args.run_seconds is not None and (float(time.time()) - wall_start_s) >= float(args.run_seconds):
                break
            if args.max_updates is not None and update_idx >= int(args.max_updates):
                break

            try:
                item = await asyncio.wait_for(block_queue.get(), timeout=queue_get_timeout_s)
            except asyncio.TimeoutError:
                if producer_task is not None and producer_task.done():
                    exc = producer_task.exception()
                    if exc is not None:
                        raise RuntimeError("Async SDR stream task failed.") from exc
                    break
                continue

            if isinstance(item, StreamSentinel):
                if item.error is not None:
                    raise RuntimeError("Async SDR stream terminated unexpectedly.") from item.error
                break

            if tone_offset_hz is None:
                bootstrap_blocks.append(item)
                if len(bootstrap_blocks) < max(1, int(args.tone_estimate_blocks)):
                    continue
                stacked = np.stack(
                    [
                        np.asarray(b.iq_block, dtype=np.complex64).reshape(-1)
                        - np.mean(np.asarray(b.iq_block, dtype=np.complex64).reshape(-1))
                        for b in bootstrap_blocks
                    ],
                    axis=0,
                )
                tone_offset_hz = estimate_tone_offset_hz(
                    stacked,
                    sample_rate_hz=float(args.sample_rate),
                    dc_exclusion_hz=float(args.dc_exclusion_hz),
                )

            work_items: list[QueuedBlock]
            if bootstrap_blocks:
                work_items = list(bootstrap_blocks)
                bootstrap_blocks.clear()
            else:
                work_items = [item]

            for blk in work_items:
                meas = measure_block(
                    queued_block=blk,
                    sample_rate_hz=float(args.sample_rate),
                    tone_offset_hz=float(tone_offset_hz),
                    update_idx=int(update_idx),
                    unwrapper=unwrapper,
                )
                if args.run_seconds is not None and meas.elapsed_s > float(args.run_seconds):
                    stop_requested = True
                    break

                rf_reference_hz = float(args.center_freq) + float(tone_offset_hz)
                if state.timing_proxy_ref_phase_rad is None and np.isfinite(rf_reference_hz) and abs(rf_reference_hz) > 1e-6:
                    state.timing_proxy_ref_phase_rad = float(meas.unwrapped_phase_rad)
                timing_proxy_ns = phase_to_timing_proxy_ns(
                    unwrapped_phase_rad=float(meas.unwrapped_phase_rad),
                    ref_phase_rad=state.timing_proxy_ref_phase_rad,
                    rf_hz=float(rf_reference_hz),
                )

                phase_err_raw = wrap_to_pi(float(meas.wrapped_phase_rad) - float(args.phase_setpoint_rad))
                phase_err_filt = circular_phase_lpf_update(
                    float(phase_err_raw),
                    float(args.phase_filter_alpha),
                    state.phase_err_lpf,
                )

                coherence_ok = bool(
                    np.isfinite(meas.coherence) and (float(meas.coherence) >= float(args.coh_min))
                )
                pullin_quality_ok = bool(
                    np.isfinite(meas.pullin_quality)
                    and (float(meas.pullin_quality) >= float(args.pullin_quality_min))
                )
                control_quality_ok = bool(coherence_ok and pullin_quality_ok)

                decision = compute_control_decision(
                    current_code=int(state.dac_code_cmd),
                    phase_err_filt_rad=float(phase_err_filt),
                    nsamp=int(args.nsamp),
                    dac_center=int(dac_center),
                    phase_frac_gain_codes=float(args.phase_frac_gain_codes),
                    control_sign=float(args.control_sign),
                    code_min=int(code_min),
                    code_max=int(code_max),
                    max_dac_step_per_update=int(args.max_dac_step_per_update),
                    quality_ok=bool(control_quality_ok),
                )

                action = "notebook_hold"
                if not coherence_ok:
                    action = "hold_low_coherence"
                elif not pullin_quality_ok:
                    action = "hold_low_pullin_quality"

                dac_step_applied = 0
                dac_applied = False
                if args.mode == "pid":
                    if control_quality_ok and decision.dac_code_proposed != int(state.dac_code_cmd):
                        if clock_obj is not None:
                            rb = int(
                                clock_obj.set_dac_code(
                                    int(decision.dac_code_proposed),
                                    settle_s=float(args.fine_settle_s),
                                )
                            )
                        else:
                            rb = int(decision.dac_code_proposed)
                        dac_step_applied = int(rb - int(state.dac_code_cmd))
                        state.dac_code_cmd = int(rb)
                        state.dac_code_rb = int(rb)
                        if clock_obj is not None:
                            state.dac_volts = float(clock_obj.dac_code_to_volts(int(rb)))
                        else:
                            state.dac_volts = code_to_volts(int(rb))
                        state.applied_updates += 1
                        dac_applied = True
                        action = "notebook_apply"
                    else:
                        state.held_updates += 1
                        if not control_quality_ok:
                            state.low_quality_holds += 1
                else:
                    state.held_updates += 1
                    if control_quality_ok:
                        action = "monitor_notebook"
                    else:
                        state.low_quality_holds += 1

                row: dict[str, Any] = {
                    "unix_time_s": float(meas.unix_time_s),
                    "elapsed_s": float(meas.elapsed_s),
                    "update_idx": int(update_idx),
                    "mode": str(args.mode),
                    "action": str(action),
                    "sample_start": int(meas.sample_start),
                    "nsamp": int(args.nsamp),
                    "sample_rate_hz": float(args.sample_rate),
                    "center_freq_hz": float(args.center_freq),
                    "tone_offset_hz": float(tone_offset_hz),
                    "rf_reference_hz": float(rf_reference_hz),
                    "mix_real": float(meas.mix_real),
                    "mix_imag": float(meas.mix_imag),
                    "mix_abs": float(meas.mix_abs),
                    "phase_wrapped_rad": float(meas.wrapped_phase_rad),
                    "phase_unwrapped_rad": float(meas.unwrapped_phase_rad),
                    "phase_setpoint_rad": float(args.phase_setpoint_rad),
                    "phase_err_raw_rad": float(phase_err_raw),
                    "phase_err_filt_rad": float(phase_err_filt),
                    "phase_fraction_cycles_per_sample": float(decision.phase_fraction_cycles_per_sample),
                    "coherence": float(meas.coherence),
                    "pullin_quality": float(meas.pullin_quality),
                    "coherence_ok": int(coherence_ok),
                    "pullin_quality_ok": int(pullin_quality_ok),
                    "control_quality_ok": int(control_quality_ok),
                    "amp_rms": float(meas.amp_rms),
                    "freq_err_intrablock_hz": float(meas.freq_err_intrablock_hz),
                    "freq_err_update_hz": float(meas.freq_err_update_hz),
                    "timing_proxy_ns": float(timing_proxy_ns),
                    "slip_cycles": int(meas.slip_cycles),
                    "dphi_wrapped_rad": float(meas.dphi_wrapped_rad),
                    "dphi_used_rad": float(meas.dphi_used_rad),
                    "dphi_expected_rad": float(meas.dphi_expected_rad),
                    "dac_center_code": int(dac_center),
                    "dac_target_float": float(decision.dac_target_float),
                    "dac_target_code": int(decision.dac_target_code),
                    "dac_code_cmd": int(state.dac_code_cmd),
                    "dac_code_rb": "" if state.dac_code_rb is None else int(state.dac_code_rb),
                    "dac_code_proposed": int(decision.dac_code_proposed),
                    "dac_volts": float(state.dac_volts),
                    "dac_step_req_codes": int(decision.dac_step_req_codes),
                    "dac_step_applied_codes": int(dac_step_applied),
                    "phase_frac_gain_codes": float(args.phase_frac_gain_codes),
                    "max_dac_step_per_update": int(args.max_dac_step_per_update),
                    "notes": str(args.notes),
                    "dac_applied": bool(dac_applied),
                }
                rows.append(row)
                logger.write(row)
                update_idx += 1
                if args.max_updates is not None and update_idx >= int(args.max_updates):
                    stop_requested = True
                    break
    finally:
        if producer_task is not None:
            try:
                if not producer_task.done():
                    producer_task.cancel()
                await producer_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if clock_obj is not None:
            try:
                clock_obj.close()
            except Exception:
                pass
        logger.close()

    final_tone_offset_hz = float(tone_offset_hz) if tone_offset_hz is not None else float("nan")
    summary = compute_summary(
        rows=rows,
        args=args,
        tone_offset_hz=float(final_tone_offset_hz),
        dac_center=int(dac_center),
    )
    summary["csv_path"] = str(Path(args.csv_out).resolve())
    summary["summary_path"] = str(Path(args.summary_out).resolve())
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Notebook-style live DPLL with fixed tone reference and absolute DAC targeting."
    )

    p.add_argument("--mode", choices=["monitor", "pid"], default="pid")
    p.add_argument("--notes", type=str, default="")

    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--center-freq", type=float, default=100e6)
    p.add_argument("--sample-rate", type=float, default=2.4e6)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--settle-seconds", type=float, default=0.25)
    p.add_argument("--nsamp", type=int, default=32768)
    p.add_argument("--run-seconds", type=float, default=120.0)
    p.add_argument("--max-updates", type=int, default=None)

    p.add_argument("--target-offset-hz", type=float, default=None)
    p.add_argument("--tone-estimate-blocks", type=int, default=8)
    p.add_argument("--dc-exclusion-hz", type=float, default=5e3)

    p.add_argument("--phase-setpoint-rad", type=float, default=0.0)
    p.add_argument("--phase-filter-alpha", type=float, default=0.85)
    p.add_argument("--coh-min", type=float, default=0.025)
    p.add_argument("--pullin-quality-min", type=float, default=0.10)
    p.add_argument(
        "--phase-frac-gain-codes",
        type=float,
        default=409500.0,
        help="Notebook-style gain from phase-fraction (cycles/sample) to DAC codes.",
    )
    p.add_argument("--control-sign", type=float, default=1.0)

    p.add_argument("--dac-center", type=int, default=None)
    p.add_argument("--dac-min-pct", type=float, default=10.0)
    p.add_argument("--dac-max-pct", type=float, default=90.0)
    p.add_argument("--max-dac-step-per-update", type=int, default=32)
    p.add_argument("--apr-ppm", type=int, default=120)
    p.add_argument("--skip-clock-init", action="store_true")
    p.add_argument(
        "--apply-start-dac",
        dest="apply_start_dac",
        action="store_true",
        default=True,
        help="Apply dac-center at run start in pid mode.",
    )
    p.add_argument(
        "--no-apply-start-dac",
        dest="apply_start_dac",
        action="store_false",
        help="Do not write dac-center at run start in pid mode.",
    )
    p.add_argument("--fine-settle-s", type=float, default=0.0)

    base_dir = Path(__file__).resolve().parent
    p.add_argument(
        "--csv-out",
        type=Path,
        default=base_dir / "notebook_style_dpll.csv",
        help="CSV output path.",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=base_dir / "notebook_style_dpll_summary.json",
        help="Summary JSON output path.",
    )
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> int:
    run_controller_self_checks()
    parser = build_arg_parser()
    args = parser.parse_args()

    summary = asyncio.run(run_one(args))
    summary_path = Path(args.summary_out)
    ensure_parent_dir(summary_path)
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    if not args.quiet:
        print_run_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
