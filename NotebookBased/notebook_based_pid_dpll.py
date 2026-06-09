#!/usr/bin/env python3
"""Notebook-based live DPLL with staged pull-in + DAC-only PID control.

Design goals from user requirements:
- Live async SDR streaming (plot_sdr_phase style queueing)
- Staged control: frequency pull-in first, then phase PID
- DAC-only actuation through clock_vcxo_helper.ClockVCXO
- 0-rad phase setpoint
- Monitor mode with identical metrics but no actuation
- CSV logging + summary metrics emphasizing phase-drift reduction
"""

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
from typing import Any, Iterable, Optional

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


def code_to_volts(code: int, vdd_dac: float) -> float:
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


def scalar_lpf_update(value: float, alpha: float, prev: float | None) -> float:
    if not np.isfinite(value):
        return float(prev) if prev is not None and np.isfinite(prev) else float("nan")
    if prev is None or (not np.isfinite(prev)):
        return float(value)
    a = clamp(alpha, 0.0, 0.999999)
    return float((a * float(prev)) + ((1.0 - a) * float(value)))


@dataclass
class SlipAwareUnwrapper:
    prev_wrapped_phase_rad: float | None = None
    prev_time_s: float | None = None
    unwrapped_phase_rad: float = 0.0
    slip_trigger_rad: float = math.pi
    slip_max_cycles_per_update: int = 8

    def reset(self) -> None:
        self.prev_wrapped_phase_rad = None
        self.prev_time_s = None
        self.unwrapped_phase_rad = 0.0

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
                k = clamp_int(
                    k,
                    -int(self.slip_max_cycles_per_update),
                    int(self.slip_max_cycles_per_update),
                )
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
    in_phase_mode: bool = False
    phase_int_rad_s: float = 0.0
    prev_phase_err_filt_rad: float | None = None
    phase_err_lpf: CircularLpfState = field(default_factory=CircularLpfState)
    pullin_err_lpf_hz: float | None = None
    pullin_lock_counter: int = 0
    pullin_unlock_counter: int = 0

    dac_code_cmd: int = 0
    dac_code_rb: int | None = None
    dac_volts: float = float("nan")

    stage_switches: int = 0
    pullin_updates_applied: int = 0
    phase_updates_applied: int = 0


CSV_COLUMNS = [
    "unix_time_s",
    "elapsed_s",
    "update_idx",
    "mode",
    "stage",
    "action",
    "stage_switch_event",
    "sample_start",
    "tone_offset_hz",
    "tone_track_step_hz",
    "observer_tone_offset_hz",
    "tracked_tone_offset_hz",
    "pullin_reference_offset_hz",
    "phase_wrapped_rad",
    "phase_unwrapped_rad",
    "observer_phase_wrapped_rad",
    "observer_phase_unwrapped_rad",
    "control_measurement_source",
    "control_phase_wrapped_rad",
    "control_phase_unwrapped_rad",
    "phase_setpoint_rad",
    "phase_err_raw_rad",
    "phase_err_filt_rad",
    "coherence",
    "pullin_quality",
    "observer_coherence",
    "observer_pullin_quality",
    "control_coherence",
    "control_pullin_quality",
    "amp_rms",
    "freq_err_intrablock_hz",
    "freq_err_update_hz",
    "observer_freq_err_intrablock_hz",
    "observer_freq_err_update_hz",
    "control_freq_err_intrablock_hz",
    "control_freq_err_update_hz",
    "observer_tau_err_ns",
    "pullin_target_hz",
    "pullin_error_hz",
    "pullin_error_lpf_hz",
    "slip_cycles",
    "dphi_wrapped_rad",
    "dphi_used_rad",
    "dphi_expected_rad",
    "u_hz_raw",
    "u_hz_limited",
    "dac_code_cmd",
    "dac_code_rb",
    "dac_code_proposed",
    "dac_volts",
    "dac_step_req_codes",
    "dac_step_applied_codes",
    "phase_int_rad_s",
    "loop_period_s_nom",
    "nsamp",
    "sample_rate_hz",
    "kv_hz_per_code",
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


def parse_nsamp_values(nsamp_values: str | None, default_nsamp: int) -> list[int]:
    if nsamp_values is None:
        return [int(default_nsamp)]
    vals: list[int] = []
    for token in str(nsamp_values).split(","):
        t = token.strip()
        if not t:
            continue
        v = int(t)
        if v <= 0:
            raise ValueError("All nsamp values must be positive.")
        vals.append(v)
    if not vals:
        vals.append(int(default_nsamp))
    return vals


def path_with_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.with_name(f"{path.name}{suffix}")


def bytes_to_iq_block(raw_bytes: bytes, nsamples: int) -> np.ndarray:
    expected = 2 * int(nsamples)
    if len(raw_bytes) != expected:
        raise RuntimeError(
            f"Expected {expected} bytes for one IQ block, received {len(raw_bytes)}."
        )
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
    rotator = np.exp(
        -2j * np.pi * float(tone_offset_hz) * n / float(sample_rate_hz)
    ).astype(np.complex64)
    demod = iq * rotator

    z = np.sum(demod, dtype=np.complex128)
    wrapped_phase = float(np.angle(z))
    coherence = float(np.abs(z) / (np.sum(np.abs(iq)) + 1e-12))
    amp_rms = float(np.sqrt(np.mean(np.abs(demod) ** 2) + 1e-30))
    freq_intrablock, pullin_quality = phase_increment_stats(demod, sample_rate_hz)

    elapsed_s = float(
        (float(queued_block.sample_start) + (0.5 * float(iq.size))) / float(sample_rate_hz)
    )

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


def compute_summary(
    *,
    rows: list[dict[str, Any]],
    run_mode: str,
    nsamp: int,
    sample_rate_hz: float,
    center_freq_hz: float,
    tone_offset_hz: float,
    drift_window_s: float,
) -> dict[str, Any]:
    updates = len(rows)
    if updates == 0:
        return {
            "mode": run_mode,
            "nsamp": int(nsamp),
            "sample_rate_hz": float(sample_rate_hz),
            "center_freq_hz": float(center_freq_hz),
            "loop_period_nom_s": float(nsamp) / float(sample_rate_hz),
            "tone_offset_hz": float(tone_offset_hz),
            "updates": 0,
            "median_coherence": float("nan"),
            "phase_drift_slope_full_hz": float("nan"),
            "phase_drift_slope_start_hz": float("nan"),
            "phase_drift_slope_end_hz": float("nan"),
            "phase_drift_reduction_hz_abs": float("nan"),
            "phase_drift_reduction_pct": float("nan"),
            "mean_abs_phase_err_rad": float("nan"),
            "mean_abs_freq_err_hz": float("nan"),
            "phase_mode_fraction": float("nan"),
            "observer_median_coherence": float("nan"),
            "observer_mean_abs_freq_err_hz": float("nan"),
            "observer_phase_drift_slope_full_hz": float("nan"),
            "observer_timing_drift_slope_ns_per_s_full": float("nan"),
            "observer_timing_drift_slope_ns_per_s_start": float("nan"),
            "observer_timing_drift_slope_ns_per_s_end": float("nan"),
            "observer_timing_drift_reduction_ns_per_s_abs": float("nan"),
            "observer_timing_drift_reduction_pct": float("nan"),
            "observer_timing_drift_slope_ns_per_s_quality": float("nan"),
            "observer_phase_drift_slope_quality_hz": float("nan"),
            "observer_mean_abs_freq_err_hz_quality": float("nan"),
            "quality_updates": 0,
            "quality_fraction": float("nan"),
            "quality_elapsed_s": float("nan"),
            "primary_timing_drift_ns_per_s": float("nan"),
            "primary_timing_drift_samples_per_s": float("nan"),
        }

    def _fget(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key, "nan"))
        except Exception:
            return float("nan")

    t = np.asarray([float(r["elapsed_s"]) for r in rows], dtype=np.float64)
    phu = np.asarray([float(r["phase_unwrapped_rad"]) for r in rows], dtype=np.float64)
    coh = np.asarray([float(r["coherence"]) for r in rows], dtype=np.float64)
    pe = np.asarray([float(r["phase_err_raw_rad"]) for r in rows], dtype=np.float64)
    ferr = np.asarray([float(r["freq_err_intrablock_hz"]) for r in rows], dtype=np.float64)
    stage_vals = [str(r["stage"]) for r in rows]
    action_vals = [str(r.get("action", "")) for r in rows]

    obs_phu = np.asarray([_fget(r, "observer_phase_unwrapped_rad") for r in rows], dtype=np.float64)
    obs_coh = np.asarray([_fget(r, "observer_coherence") for r in rows], dtype=np.float64)
    obs_ferr = np.asarray([_fget(r, "observer_freq_err_intrablock_hz") for r in rows], dtype=np.float64)
    obs_toff = np.asarray([_fget(r, "observer_tone_offset_hz") for r in rows], dtype=np.float64)

    if not np.any(np.isfinite(obs_phu)):
        obs_phu = np.copy(phu)
    if not np.any(np.isfinite(obs_coh)):
        obs_coh = np.copy(coh)
    if not np.any(np.isfinite(obs_ferr)):
        obs_ferr = np.copy(ferr)
    if not np.any(np.isfinite(obs_toff)):
        obs_toff = np.full_like(t, float(tone_offset_hz), dtype=np.float64)

    slope_full = linear_slope(t, phu)
    slope_full_hz = float(slope_full / (2.0 * math.pi)) if np.isfinite(slope_full) else float("nan")

    obs_slope_full = linear_slope(t, obs_phu)
    obs_slope_full_hz = (
        float(obs_slope_full / (2.0 * math.pi)) if np.isfinite(obs_slope_full) else float("nan")
    )

    t0 = float(t[0])
    t1 = float(t[-1])
    span = max(0.0, t1 - t0)
    w = min(float(drift_window_s), max(span / 2.0, 0.0))

    if w > 0.0:
        start_mask = t <= (t0 + w)
        end_mask = t >= (t1 - w)
    else:
        start_mask = np.zeros_like(t, dtype=bool)
        end_mask = np.zeros_like(t, dtype=bool)

    slope_start = linear_slope(t[start_mask], phu[start_mask]) if np.any(start_mask) else float("nan")
    slope_end = linear_slope(t[end_mask], phu[end_mask]) if np.any(end_mask) else float("nan")

    slope_start_hz = float(slope_start / (2.0 * math.pi)) if np.isfinite(slope_start) else float("nan")
    slope_end_hz = float(slope_end / (2.0 * math.pi)) if np.isfinite(slope_end) else float("nan")

    if np.isfinite(slope_start_hz) and np.isfinite(slope_end_hz):
        reduction_abs = float(abs(slope_start_hz) - abs(slope_end_hz))
        if abs(slope_start_hz) > 1e-12:
            reduction_pct = float(100.0 * reduction_abs / abs(slope_start_hz))
        else:
            reduction_pct = float("nan")
    else:
        reduction_abs = float("nan")
        reduction_pct = float("nan")

    phase_mode_frac = (
        float(sum(1 for s in stage_vals if s == "phase_lock")) / float(len(stage_vals))
        if stage_vals
        else float("nan")
    )
    quality_mask = np.asarray(
        [
            (not str(a).startswith("hold_low_")) and (str(a) != "hold_invalid_kv")
            for a in action_vals
        ],
        dtype=bool,
    )
    quality_updates = int(np.count_nonzero(quality_mask))
    quality_fraction = float(quality_updates / updates) if updates > 0 else float("nan")
    if quality_updates >= 2:
        quality_elapsed_s = float(t[quality_mask][-1] - t[quality_mask][0])
    elif quality_updates == 1:
        quality_elapsed_s = 0.0
    else:
        quality_elapsed_s = float("nan")

    obs_rf_hz = float(center_freq_hz) + obs_toff
    obs_tau_ns = np.full(obs_phu.shape, np.nan, dtype=np.float64)
    valid_obs = (
        np.isfinite(obs_phu)
        & np.isfinite(obs_rf_hz)
        & (np.abs(obs_rf_hz) > 1e-6)
        & np.isfinite(t)
    )
    if np.any(valid_obs):
        first_idx = int(np.flatnonzero(valid_obs)[0])
        obs_tau_ns[valid_obs] = (
            (obs_phu[valid_obs] - obs_phu[first_idx]) / (2.0 * math.pi * obs_rf_hz[valid_obs])
        ) * 1e9

    obs_tau_slope_full = linear_slope(t[valid_obs], obs_tau_ns[valid_obs]) if np.any(valid_obs) else float("nan")

    obs_tau_slope_start = (
        linear_slope(t[start_mask & valid_obs], obs_tau_ns[start_mask & valid_obs])
        if np.any(start_mask & valid_obs)
        else float("nan")
    )
    obs_tau_slope_end = (
        linear_slope(t[end_mask & valid_obs], obs_tau_ns[end_mask & valid_obs])
        if np.any(end_mask & valid_obs)
        else float("nan")
    )

    if np.isfinite(obs_tau_slope_start) and np.isfinite(obs_tau_slope_end):
        obs_tau_reduction_abs = float(abs(obs_tau_slope_start) - abs(obs_tau_slope_end))
        if abs(obs_tau_slope_start) > 1e-12:
            obs_tau_reduction_pct = float(100.0 * obs_tau_reduction_abs / abs(obs_tau_slope_start))
        else:
            obs_tau_reduction_pct = float("nan")
    else:
        obs_tau_reduction_abs = float("nan")
        obs_tau_reduction_pct = float("nan")

    valid_obs_quality = valid_obs & quality_mask
    obs_slope_quality = (
        linear_slope(t[valid_obs_quality], obs_phu[valid_obs_quality])
        if np.count_nonzero(valid_obs_quality) >= 2
        else float("nan")
    )
    obs_slope_quality_hz = (
        float(obs_slope_quality / (2.0 * math.pi)) if np.isfinite(obs_slope_quality) else float("nan")
    )
    obs_tau_slope_quality = (
        linear_slope(t[valid_obs_quality], obs_tau_ns[valid_obs_quality])
        if np.count_nonzero(valid_obs_quality) >= 2
        else float("nan")
    )
    obs_ferr_quality = obs_ferr[quality_mask & np.isfinite(obs_ferr)]
    obs_mean_abs_freq_err_quality = (
        float(np.nanmean(np.abs(obs_ferr_quality))) if obs_ferr_quality.size else float("nan")
    )

    primary_tau_ns_per_s = (
        float(obs_tau_slope_quality) if np.isfinite(obs_tau_slope_quality) else float(obs_tau_slope_full)
    )
    if np.isfinite(primary_tau_ns_per_s):
        primary_tau_samples_per_s = float(primary_tau_ns_per_s * float(sample_rate_hz) / 1e9)
    else:
        primary_tau_samples_per_s = float("nan")

    return {
        "mode": str(run_mode),
        "nsamp": int(nsamp),
        "sample_rate_hz": float(sample_rate_hz),
        "center_freq_hz": float(center_freq_hz),
        "loop_period_nom_s": float(nsamp) / float(sample_rate_hz),
        "tone_offset_hz": float(tone_offset_hz),
        "updates": int(updates),
        "elapsed_s": float(span),
        "median_coherence": float(np.nanmedian(coh)) if coh.size else float("nan"),
        "mean_abs_phase_err_rad": float(np.nanmean(np.abs(pe))) if pe.size else float("nan"),
        "mean_abs_freq_err_hz": float(np.nanmean(np.abs(ferr))) if ferr.size else float("nan"),
        "phase_mode_fraction": float(phase_mode_frac),
        "phase_drift_slope_full_hz": float(slope_full_hz),
        "phase_drift_slope_start_hz": float(slope_start_hz),
        "phase_drift_slope_end_hz": float(slope_end_hz),
        "phase_drift_reduction_hz_abs": float(reduction_abs),
        "phase_drift_reduction_pct": float(reduction_pct),
        "observer_median_coherence": float(np.nanmedian(obs_coh)) if obs_coh.size else float("nan"),
        "observer_mean_abs_freq_err_hz": float(np.nanmean(np.abs(obs_ferr))) if obs_ferr.size else float("nan"),
        "observer_phase_drift_slope_full_hz": float(obs_slope_full_hz),
        "observer_timing_drift_slope_ns_per_s_full": float(obs_tau_slope_full),
        "observer_timing_drift_slope_ns_per_s_start": float(obs_tau_slope_start),
        "observer_timing_drift_slope_ns_per_s_end": float(obs_tau_slope_end),
        "observer_timing_drift_reduction_ns_per_s_abs": float(obs_tau_reduction_abs),
        "observer_timing_drift_reduction_pct": float(obs_tau_reduction_pct),
        "observer_timing_drift_slope_ns_per_s_quality": float(obs_tau_slope_quality),
        "observer_phase_drift_slope_quality_hz": float(obs_slope_quality_hz),
        "observer_mean_abs_freq_err_hz_quality": float(obs_mean_abs_freq_err_quality),
        "quality_updates": int(quality_updates),
        "quality_fraction": float(quality_fraction),
        "quality_elapsed_s": float(quality_elapsed_s),
        "primary_timing_drift_ns_per_s": float(primary_tau_ns_per_s),
        "primary_timing_drift_samples_per_s": float(primary_tau_samples_per_s),
    }


def to_float(x: float | str | None) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


async def run_one(args: argparse.Namespace, nsamp: int, csv_path: Path) -> dict[str, Any]:
    if args.blocks_per_update != 1:
        raise ValueError("This implementation currently requires --blocks-per-update 1.")

    if nsamp <= 0:
        raise ValueError("nsamp must be positive.")

    if args.mode == "pid" and cvh is None:
        raise RuntimeError(
            "clock_vcxo_helper.py import failed, but mode=pid requires DAC control access."
        )

    code_min = pct_to_code(float(args.dac_min_pct))
    code_max = pct_to_code(float(args.dac_max_pct))
    if code_min >= code_max:
        raise ValueError("DAC range invalid: ensure dac_min_pct < dac_max_pct.")

    if args.dac_start is None:
        dac_start = int(round((code_min + code_max) / 2.0))
    else:
        dac_start = clamp_int(int(args.dac_start), code_min, code_max)

    nominal_loop_period_s = float(nsamp) / float(args.sample_rate)
    queue_get_timeout_s = max(float(args.queue_get_timeout_s), nominal_loop_period_s)

    logger = CsvLogger(csv_path)
    device = None
    producer_task: asyncio.Task[None] | None = None
    clock_obj = None

    rows: list[dict[str, Any]] = []
    tone_offset_hz: float | None = None
    observer_tone_offset_hz: float | None = (
        None if args.observer_fixed_offset_hz is None else float(args.observer_fixed_offset_hz)
    )

    state = ControllerState(
        in_phase_mode=False,
        phase_int_rad_s=0.0,
        prev_phase_err_filt_rad=None,
        dac_code_cmd=int(dac_start),
        dac_code_rb=None,
        dac_volts=code_to_volts(int(dac_start), float(args.vdd_dac)),
    )

    unwrapper = SlipAwareUnwrapper(
        slip_trigger_rad=float(args.slip_trigger_rad),
        slip_max_cycles_per_update=int(args.slip_max_cycles_per_update),
    )
    observer_unwrapper = SlipAwareUnwrapper(
        slip_trigger_rad=float(args.slip_trigger_rad),
        slip_max_cycles_per_update=int(args.slip_max_cycles_per_update),
    )

    update_idx = 0
    bootstrap_blocks: list[QueuedBlock] = []
    wall_start_s = float(time.time())

    try:
        device = open_sdr(args)
        if float(args.settle_seconds) > 0.0:
            await asyncio.sleep(float(args.settle_seconds))

        queue_depth = args.async_queue_blocks
        if queue_depth is None:
            queue_depth = default_async_queue_blocks()
        queue_depth = max(int(queue_depth), 4)

        block_queue: asyncio.Queue[QueuedBlock | StreamSentinel] = asyncio.Queue(maxsize=queue_depth)
        producer_task = asyncio.create_task(stream_iq_blocks(device, nsamp, block_queue))

        if args.mode == "pid":
            clock_obj = cvh.ClockVCXO()
            clock_obj.open()
            if not args.skip_clock_init:
                clock_obj.init_clock_vcxo_28p8mhz(apr_ppm=int(args.apr_ppm))
            if args.apply_start_dac:
                rb0 = int(clock_obj.set_dac_code(int(dac_start), settle_s=float(args.fine_settle_s)))
                state.dac_code_cmd = int(rb0)
                state.dac_code_rb = int(rb0)
                state.dac_volts = float(clock_obj.dac_code_to_volts(int(rb0)))

        if args.target_offset_hz is not None:
            tone_offset_hz = float(args.target_offset_hz)
        if observer_tone_offset_hz is None and tone_offset_hz is not None:
            observer_tone_offset_hz = float(tone_offset_hz)

        while True:
            if args.run_seconds is not None and (
                float(time.time()) - float(wall_start_s)
            ) >= float(args.run_seconds):
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
                target = max(1, int(args.tone_estimate_blocks))
                if len(bootstrap_blocks) < target:
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
                if observer_tone_offset_hz is None:
                    observer_tone_offset_hz = float(tone_offset_hz)

            # Process bootstrap backlog first, then the current block.
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
                obs_ref_hz = float(
                    observer_tone_offset_hz if observer_tone_offset_hz is not None else tone_offset_hz
                )
                observer_meas = measure_block(
                    queued_block=blk,
                    sample_rate_hz=float(args.sample_rate),
                    tone_offset_hz=obs_ref_hz,
                    update_idx=int(update_idx),
                    unwrapper=observer_unwrapper,
                )
                if args.control_measurement_source == "observer":
                    control_meas = observer_meas
                else:
                    control_meas = meas

                # Pull-in should prefer the update-to-update phase slope because it reflects
                # the inter-block drift that the DAC can actually correct. The intrablock
                # estimate is still useful as a fallback, but it is much noisier on live data.
                tracked_freq_residual_hz = float(meas.freq_err_update_hz)
                if not np.isfinite(tracked_freq_residual_hz):
                    tracked_freq_residual_hz = float(meas.freq_err_intrablock_hz)
                if not np.isfinite(tracked_freq_residual_hz):
                    tracked_freq_residual_hz = 0.0
                tracked_tone_offset_hz = float(meas.tone_offset_hz + tracked_freq_residual_hz)
                pullin_reference_offset_hz = float(obs_ref_hz)

                if args.run_seconds is not None and control_meas.elapsed_s > float(args.run_seconds):
                    break

                phase_err_raw = wrap_to_pi(
                    float(control_meas.wrapped_phase_rad) - float(args.phase_setpoint_rad)
                )
                phase_err_filt = circular_phase_lpf_update(
                    phase_err_raw,
                    float(args.phase_filter_alpha),
                    state.phase_err_lpf,
                )

                tone_track_step_hz = 0.0
                # Optional tone-reference tracking to keep demod near DC while clock changes.
                # This uses pull-in quality rather than raw coherence because tone tracking
                # should remain available while the residual tone is still outside the
                # narrow raw-coherence region of the current block length.
                if (
                    (not state.in_phase_mode)
                    and float(args.tone_track_alpha) > 0.0
                    and np.isfinite(meas.freq_err_intrablock_hz)
                    and (meas.pullin_quality >= float(args.tone_track_min_coh))
                ):
                    tone_track_step_hz = clamp(
                        float(args.tone_track_alpha) * float(meas.freq_err_intrablock_hz),
                        -float(args.tone_track_max_step_hz),
                        float(args.tone_track_max_step_hz),
                    )
                    tone_offset_hz = float(tone_offset_hz + tone_track_step_hz)

                observer_rf_hz = float(args.center_freq) + float(obs_ref_hz)
                if np.isfinite(observer_rf_hz) and abs(observer_rf_hz) > 1e-6:
                    observer_tau_err_ns = float(
                        observer_meas.unwrapped_phase_rad / (2.0 * math.pi * observer_rf_hz) * 1e9
                    )
                else:
                    observer_tau_err_ns = float("nan")

                phase_freq_err_hz = float(control_meas.freq_err_update_hz)
                if not np.isfinite(phase_freq_err_hz):
                    phase_freq_err_hz = float(control_meas.freq_err_intrablock_hz)
                if not np.isfinite(phase_freq_err_hz):
                    phase_freq_err_hz = 0.0

                pullin_err_raw_hz = float(
                    (tracked_tone_offset_hz - pullin_reference_offset_hz) - float(args.pullin_target_hz)
                )
                if np.isfinite(pullin_err_raw_hz):
                    state.pullin_err_lpf_hz = scalar_lpf_update(
                        float(pullin_err_raw_hz),
                        float(args.pullin_filter_alpha),
                        state.pullin_err_lpf_hz,
                    )
                pullin_err_hz = float(
                    state.pullin_err_lpf_hz
                    if state.pullin_err_lpf_hz is not None and np.isfinite(state.pullin_err_lpf_hz)
                    else pullin_err_raw_hz
                )

                stage_switch_event = 0
                phase_coh_for_control = float(control_meas.coherence)
                pullin_quality_for_control = float(control_meas.pullin_quality)
                phase_coh_ok = bool(
                    np.isfinite(phase_coh_for_control)
                    and (phase_coh_for_control >= float(args.coh_min))
                )
                pullin_quality_ok = bool(
                    np.isfinite(pullin_quality_for_control)
                    and (pullin_quality_for_control >= float(args.pullin_quality_min))
                )
                enter_ok = (
                    pullin_quality_ok
                    and phase_coh_ok
                    and (abs(pullin_err_hz) <= float(args.pullin_switch_hz))
                )
                exit_ok = (
                    (not phase_coh_ok)
                    or (not pullin_quality_ok)
                    or (abs(pullin_err_hz) >= float(args.pullin_exit_hz))
                )
                if not state.in_phase_mode:
                    if enter_ok:
                        state.pullin_lock_counter += 1
                    else:
                        state.pullin_lock_counter = 0
                    state.pullin_unlock_counter = 0
                    if state.pullin_lock_counter >= int(args.pullin_enter_dwell_updates):
                        state.in_phase_mode = True
                        state.phase_int_rad_s = 0.0
                        state.prev_phase_err_filt_rad = None
                        state.pullin_lock_counter = 0
                        stage_switch_event = 1
                        state.stage_switches += 1
                else:
                    state.pullin_lock_counter = 0
                    if exit_ok:
                        state.pullin_unlock_counter += 1
                    else:
                        state.pullin_unlock_counter = 0
                    if state.pullin_unlock_counter >= int(args.pullin_exit_dwell_updates):
                        state.in_phase_mode = False
                        state.phase_int_rad_s = 0.0
                        state.prev_phase_err_filt_rad = None
                        state.pullin_unlock_counter = 0
                        stage_switch_event = 1
                        state.stage_switches += 1

                stage = "phase_lock" if state.in_phase_mode else "pull_in"

                action = "hold"
                u_hz_raw = 0.0
                u_hz_limited = 0.0
                dac_step_req = 0
                dac_step_applied = 0
                dac_code_proposed = int(state.dac_code_cmd)

                control_quality_ok = pullin_quality_ok if not state.in_phase_mode else (phase_coh_ok and pullin_quality_ok)
                if not control_quality_ok:
                    action = "hold_low_pullin_quality" if not pullin_quality_ok else "hold_low_phase_coherence"
                else:
                    if not state.in_phase_mode:
                        u_hz_raw = float(args.Kf_pullin_hz_per_hz) * pullin_err_hz
                        action = "pull_in"
                    else:
                        dt = float(control_meas.dt_update_s)
                        if dt <= 0.0 or (not np.isfinite(dt)):
                            dt = nominal_loop_period_s

                        p_hz = float(args.Kp_hz_per_rad) * float(phase_err_filt)

                        i_candidate = float(state.phase_int_rad_s)
                        if abs(float(args.Ki_hz_per_rad_s)) > 1e-18 and np.isfinite(dt):
                            i_candidate += float(phase_err_filt) * dt
                            i_limit_rad_s = abs(float(args.integral_limit_hz) / float(args.Ki_hz_per_rad_s))
                            i_candidate = clamp(i_candidate, -i_limit_rad_s, i_limit_rad_s)

                        if state.prev_phase_err_filt_rad is None or dt <= 0.0:
                            d_err = 0.0
                        else:
                            d_err = wrap_to_pi(float(phase_err_filt) - float(state.prev_phase_err_filt_rad)) / dt

                        d_hz = float(args.Kd_hz_s_per_rad) * float(d_err)
                        i_hz = float(args.Ki_hz_per_rad_s) * float(i_candidate)
                        u_hz_raw = float(p_hz + i_hz + d_hz)
                        action = "phase_pid"

                    u_hz_limited = clamp(float(u_hz_raw), -float(args.max_u_hz), float(args.max_u_hz))

                    if abs(float(args.kv_hz_per_code)) <= 1e-12:
                        action = "hold_invalid_kv"
                        u_hz_limited = 0.0
                    else:
                        delta_codes = (
                            float(args.control_sign)
                            * float(u_hz_limited)
                            / float(args.kv_hz_per_code)
                        )
                        delta_codes = clamp(
                            delta_codes,
                            -float(args.max_dac_step_per_update),
                            float(args.max_dac_step_per_update),
                        )
                        dac_step_req = int(round(delta_codes))
                        dac_code_proposed = clamp_int(
                            int(state.dac_code_cmd) + int(dac_step_req),
                            int(code_min),
                            int(code_max),
                        )

                        saturated = (
                            dac_code_proposed == int(code_min)
                            or dac_code_proposed == int(code_max)
                        )
                        pushes_outward = (
                            (dac_code_proposed == int(code_max) and dac_step_req > 0)
                            or (dac_code_proposed == int(code_min) and dac_step_req < 0)
                        )
                        clipped_u = abs(float(u_hz_raw)) > (abs(float(args.max_u_hz)) + 1e-12)

                        if state.in_phase_mode and action == "phase_pid":
                            dt = float(control_meas.dt_update_s)
                            if dt <= 0.0 or (not np.isfinite(dt)):
                                dt = nominal_loop_period_s
                            i_candidate2 = float(state.phase_int_rad_s)
                            if abs(float(args.Ki_hz_per_rad_s)) > 1e-18 and np.isfinite(dt):
                                i_candidate2 += float(phase_err_filt) * dt
                                i_limit_rad_s = abs(float(args.integral_limit_hz) / float(args.Ki_hz_per_rad_s))
                                i_candidate2 = clamp(i_candidate2, -i_limit_rad_s, i_limit_rad_s)
                            if not ((saturated and pushes_outward) or clipped_u):
                                state.phase_int_rad_s = float(i_candidate2)

                        if args.mode == "pid":
                            if dac_code_proposed != int(state.dac_code_cmd):
                                if clock_obj is not None:
                                    rb = int(
                                        clock_obj.set_dac_code(
                                            int(dac_code_proposed),
                                            settle_s=float(args.fine_settle_s),
                                        )
                                    )
                                else:
                                    rb = int(dac_code_proposed)

                                dac_step_applied = int(rb - int(state.dac_code_cmd))
                                state.dac_code_cmd = int(rb)
                                state.dac_code_rb = int(rb)
                                if clock_obj is not None:
                                    state.dac_volts = float(clock_obj.dac_code_to_volts(int(rb)))
                                else:
                                    state.dac_volts = code_to_volts(int(rb), float(args.vdd_dac))

                                if state.in_phase_mode:
                                    state.phase_updates_applied += 1
                                else:
                                    state.pullin_updates_applied += 1

                                action = f"{action}_apply"
                            else:
                                action = f"{action}_hold"
                        else:
                            action = f"monitor_{stage}"

                state.prev_phase_err_filt_rad = float(phase_err_filt)

                row: dict[str, Any] = {
                    "unix_time_s": f"{control_meas.unix_time_s:.9f}",
                    "elapsed_s": f"{control_meas.elapsed_s:.9f}",
                    "update_idx": int(update_idx),
                    "mode": str(args.mode),
                    "stage": stage,
                    "action": action,
                    "stage_switch_event": int(stage_switch_event),
                    "sample_start": int(control_meas.sample_start),
                    "tone_offset_hz": f"{float(tone_offset_hz):.9f}",
                    "tone_track_step_hz": f"{float(tone_track_step_hz):.9f}",
                    "observer_tone_offset_hz": f"{float(obs_ref_hz):.9f}",
                    "tracked_tone_offset_hz": f"{float(tracked_tone_offset_hz):.9f}",
                    "pullin_reference_offset_hz": f"{float(pullin_reference_offset_hz):.9f}",
                    "phase_wrapped_rad": f"{control_meas.wrapped_phase_rad:.9f}",
                    "phase_unwrapped_rad": f"{control_meas.unwrapped_phase_rad:.9f}",
                    "observer_phase_wrapped_rad": f"{observer_meas.wrapped_phase_rad:.9f}",
                    "observer_phase_unwrapped_rad": f"{observer_meas.unwrapped_phase_rad:.9f}",
                    "control_measurement_source": str(args.control_measurement_source),
                    "control_phase_wrapped_rad": f"{control_meas.wrapped_phase_rad:.9f}",
                    "control_phase_unwrapped_rad": f"{control_meas.unwrapped_phase_rad:.9f}",
                    "phase_setpoint_rad": f"{float(args.phase_setpoint_rad):.9f}",
                    "phase_err_raw_rad": f"{phase_err_raw:.9f}",
                    "phase_err_filt_rad": f"{phase_err_filt:.9f}",
                    "coherence": f"{control_meas.coherence:.9f}",
                    "pullin_quality": f"{control_meas.pullin_quality:.9f}",
                    "observer_coherence": f"{observer_meas.coherence:.9f}",
                    "observer_pullin_quality": f"{observer_meas.pullin_quality:.9f}",
                    "control_coherence": f"{control_meas.coherence:.9f}",
                    "control_pullin_quality": f"{control_meas.pullin_quality:.9f}",
                    "amp_rms": f"{control_meas.amp_rms:.9f}",
                    "freq_err_intrablock_hz": f"{control_meas.freq_err_intrablock_hz:.9f}",
                    "freq_err_update_hz": f"{control_meas.freq_err_update_hz:.9f}",
                    "observer_freq_err_intrablock_hz": f"{observer_meas.freq_err_intrablock_hz:.9f}",
                    "observer_freq_err_update_hz": f"{observer_meas.freq_err_update_hz:.9f}",
                    "control_freq_err_intrablock_hz": f"{phase_freq_err_hz:.9f}",
                    "control_freq_err_update_hz": f"{control_meas.freq_err_update_hz:.9f}",
                    "observer_tau_err_ns": f"{observer_tau_err_ns:.9f}",
                    "pullin_target_hz": f"{float(args.pullin_target_hz):.9f}",
                    "pullin_error_hz": f"{pullin_err_raw_hz:.9f}",
                    "pullin_error_lpf_hz": f"{pullin_err_hz:.9f}",
                    "slip_cycles": int(control_meas.slip_cycles),
                    "dphi_wrapped_rad": f"{control_meas.dphi_wrapped_rad:.9f}",
                    "dphi_used_rad": f"{control_meas.dphi_used_rad:.9f}",
                    "dphi_expected_rad": f"{control_meas.dphi_expected_rad:.9f}",
                    "u_hz_raw": f"{float(u_hz_raw):.9f}",
                    "u_hz_limited": f"{float(u_hz_limited):.9f}",
                    "dac_code_cmd": int(state.dac_code_cmd),
                    "dac_code_rb": "" if state.dac_code_rb is None else int(state.dac_code_rb),
                    "dac_code_proposed": int(dac_code_proposed),
                    "dac_volts": f"{to_float(state.dac_volts):.9f}",
                    "dac_step_req_codes": int(dac_step_req),
                    "dac_step_applied_codes": int(dac_step_applied),
                    "phase_int_rad_s": f"{float(state.phase_int_rad_s):.9f}",
                    "loop_period_s_nom": f"{nominal_loop_period_s:.9f}",
                    "nsamp": int(nsamp),
                    "sample_rate_hz": f"{float(args.sample_rate):.3f}",
                    "kv_hz_per_code": f"{float(args.kv_hz_per_code):.9f}",
                    "notes": str(args.notes),
                }

                logger.write(row)
                rows.append(row)

                update_idx += 1
                if args.max_updates is not None and update_idx >= int(args.max_updates):
                    break
                if args.run_seconds is not None and control_meas.elapsed_s > float(args.run_seconds):
                    break

            if args.max_updates is not None and update_idx >= int(args.max_updates):
                break
            if args.run_seconds is not None and (
                float(time.time()) - float(wall_start_s)
            ) >= float(args.run_seconds):
                break
            if rows and args.run_seconds is not None:
                if float(rows[-1]["elapsed_s"]) > float(args.run_seconds):
                    break

        if tone_offset_hz is None and bootstrap_blocks:
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

        if tone_offset_hz is None:
            tone_offset_hz = 0.0

        summary = compute_summary(
            rows=rows,
            run_mode=str(args.mode),
            nsamp=int(nsamp),
            sample_rate_hz=float(args.sample_rate),
            center_freq_hz=float(args.center_freq),
            tone_offset_hz=float(tone_offset_hz),
            drift_window_s=float(args.drift_window_s),
        )
        summary["csv_path"] = str(csv_path)
        summary["stage_switches"] = int(state.stage_switches)
        summary["pullin_updates_applied"] = int(state.pullin_updates_applied)
        summary["phase_updates_applied"] = int(state.phase_updates_applied)
        summary["dac_min_code"] = int(code_min)
        summary["dac_max_code"] = int(code_max)
        summary["dac_start_code"] = int(dac_start)

        return summary

    finally:
        if producer_task is not None:
            if not producer_task.done():
                producer_task.cancel()
            try:
                await asyncio.wait_for(producer_task, timeout=2.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                pass
            except BaseException:
                pass
        logger.close()
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
        if clock_obj is not None:
            try:
                clock_obj.close()
            except Exception:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Notebook-based live DPLL with async SDR queue and DAC-only staged PID control."
    )

    p.add_argument("--mode", choices=["pid", "monitor"], default="pid")
    p.add_argument("--notes", type=str, default="")

    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--center-freq", type=float, default=100e6)
    p.add_argument("--sample-rate", type=float, default=2.4e6)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--settle-seconds", type=float, default=0.25)

    p.add_argument("--nsamp", type=int, default=65536)
    p.add_argument(
        "--nsamp-values",
        type=str,
        default=None,
        help="Comma-separated nsamp sweep values (e.g. 16384,32768,65536).",
    )
    p.add_argument("--blocks-per-update", type=int, default=1)
    p.add_argument("--async-queue-blocks", type=int, default=None)
    p.add_argument(
        "--queue-get-timeout-s",
        type=float,
        default=1.0,
        help="Timeout while waiting for the next streamed block so wall-time exit still works.",
    )

    p.add_argument("--run-seconds", type=float, default=120.0)
    p.add_argument("--max-updates", type=int, default=None)

    p.add_argument("--target-offset-hz", type=float, default=None)
    p.add_argument(
        "--observer-fixed-offset-hz",
        type=float,
        default=None,
        help="Fixed observer-reference tone offset (Hz). If omitted, uses initial tone offset.",
    )
    p.add_argument(
        "--control-measurement-source",
        choices=["adaptive", "observer"],
        default="observer",
        help="Which measurement stream drives pull-in and phase control.",
    )
    p.add_argument("--tone-estimate-blocks", type=int, default=8)
    p.add_argument("--dc-exclusion-hz", type=float, default=5e3)
    p.add_argument(
        "--tone-track-alpha",
        type=float,
        default=0.0,
        help="Adaptive tone-reference update gain (0 disables tracking).",
    )
    p.add_argument(
        "--tone-track-min-coh",
        type=float,
        default=0.0,
        help="Minimum coherence required to apply tone-reference tracking.",
    )
    p.add_argument(
        "--tone-track-max-step-hz",
        type=float,
        default=2000.0,
        help="Per-update clamp for adaptive tone-reference step.",
    )

    p.add_argument("--phase-setpoint-rad", type=float, default=0.0)
    p.add_argument("--phase-filter-alpha", type=float, default=0.85)
    p.add_argument("--coh-min", type=float, default=0.025)
    p.add_argument(
        "--pullin-quality-min",
        type=float,
        default=0.10,
        help="Minimum pull-in quality required to trust intrablock frequency measurements.",
    )
    p.add_argument(
        "--pullin-filter-alpha",
        type=float,
        default=0.80,
        help="Low-pass filter alpha for the pull-in frequency error used by the staged controller.",
    )

    p.add_argument("--pullin-target-hz", type=float, default=0.0)
    p.add_argument("--pullin-switch-hz", type=float, default=100.0)
    p.add_argument("--pullin-exit-hz", type=float, default=300.0)
    p.add_argument(
        "--pullin-enter-dwell-updates",
        type=int,
        default=4,
        help="Consecutive in-threshold updates required before entering phase_lock.",
    )
    p.add_argument(
        "--pullin-exit-dwell-updates",
        type=int,
        default=2,
        help="Consecutive out-of-threshold updates required before exiting phase_lock.",
    )
    p.add_argument("--Kf-pullin-hz-per-hz", type=float, default=0.07)

    p.add_argument("--Kp-hz-per-rad", type=float, default=40.0)
    p.add_argument("--Ki-hz-per-rad-s", type=float, default=4.0)
    p.add_argument("--Kd-hz-s-per-rad", type=float, default=0.0)
    p.add_argument("--integral-limit-hz", type=float, default=200.0)
    p.add_argument("--max-u-hz", type=float, default=2000.0)

    p.add_argument("--control-sign", type=float, default=1.0)
    p.add_argument("--kv-hz-per-code", type=float, default=4.243)
    p.add_argument("--max-dac-step-per-update", type=int, default=100)
    p.add_argument("--dac-min-pct", type=float, default=10.0)
    p.add_argument("--dac-max-pct", type=float, default=90.0)
    p.add_argument("--dac-start", type=int, default=None)
    p.add_argument("--vdd-dac", type=float, default=VDD_DAC_DEFAULT)

    p.add_argument("--apr-ppm", type=int, default=120)
    p.add_argument("--skip-clock-init", action="store_true")
    p.add_argument(
        "--apply-start-dac",
        dest="apply_start_dac",
        action="store_true",
        default=True,
        help="Apply --dac-start at run start (PID mode).",
    )
    p.add_argument(
        "--no-apply-start-dac",
        dest="apply_start_dac",
        action="store_false",
        help="Do not write a start DAC value at run start (PID mode).",
    )
    p.add_argument("--fine-settle-s", type=float, default=0.0)

    p.add_argument("--slip-trigger-rad", type=float, default=math.pi)
    p.add_argument("--slip-max-cycles-per-update", type=int, default=8)

    p.add_argument("--drift-window-s", type=float, default=30.0)

    base_dir = Path(__file__).resolve().parent
    p.add_argument(
        "--csv-out",
        type=Path,
        default=base_dir / "notebook_based_pid_log.csv",
        help="CSV output path. Sweep mode appends _nsampXXXX.",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=base_dir / "notebook_based_pid_summary.json",
        help="Summary JSON path. Sweep mode writes a JSON list.",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional summary CSV path for one run or sweep.",
    )

    p.add_argument("--quiet", action="store_true")
    return p


def write_summary_csv(summaries: list[dict[str, Any]], path: Path) -> None:
    ensure_parent_dir(path)
    if not summaries:
        return

    keys = sorted({k for s in summaries for k in s.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)


def print_run_summary(summary: dict[str, Any]) -> None:
    print("\nRun summary:")
    print(f"  mode: {summary.get('mode')}")
    print(f"  nsamp: {summary.get('nsamp')}")
    print(f"  sample_rate_hz: {summary.get('sample_rate_hz')}")
    print(f"  loop_period_nom_s: {summary.get('loop_period_nom_s')}")
    print(f"  tone_offset_hz: {summary.get('tone_offset_hz')}")
    print(f"  updates: {summary.get('updates')}")
    print(f"  median_coherence: {summary.get('median_coherence')}")
    print(f"  phase_drift_slope_start_hz: {summary.get('phase_drift_slope_start_hz')}")
    print(f"  phase_drift_slope_end_hz: {summary.get('phase_drift_slope_end_hz')}")
    print(f"  phase_drift_reduction_hz_abs: {summary.get('phase_drift_reduction_hz_abs')}")
    print(f"  phase_drift_reduction_pct: {summary.get('phase_drift_reduction_pct')}")
    print(f"  observer_timing_drift_ns_per_s: {summary.get('primary_timing_drift_ns_per_s')}")
    print(f"  observer_timing_drift_samples_per_s: {summary.get('primary_timing_drift_samples_per_s')}")
    print(
        "  observer_timing_drift_start/end_ns_per_s: "
        f"{summary.get('observer_timing_drift_slope_ns_per_s_start')} -> "
        f"{summary.get('observer_timing_drift_slope_ns_per_s_end')}"
    )
    print(f"  observer_timing_drift_reduction_pct: {summary.get('observer_timing_drift_reduction_pct')}")
    print(f"  quality_fraction: {summary.get('quality_fraction')}")
    print(f"  csv_path: {summary.get('csv_path')}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        nsamp_values = parse_nsamp_values(args.nsamp_values, int(args.nsamp))
    except Exception as exc:
        print(f"Invalid --nsamp-values: {exc}", file=sys.stderr)
        return 2

    if float(args.sample_rate) <= 0.0:
        print("--sample-rate must be > 0.", file=sys.stderr)
        return 2
    if int(args.blocks_per_update) != 1:
        print("--blocks-per-update must be 1 for this implementation.", file=sys.stderr)
        return 2
    if float(args.pullin_exit_hz) < float(args.pullin_switch_hz):
        print("--pullin-exit-hz should be >= --pullin-switch-hz.", file=sys.stderr)
        return 2

    summaries: list[dict[str, Any]] = []

    for idx, nsamp in enumerate(nsamp_values):
        suffix = ""
        if len(nsamp_values) > 1:
            suffix = f"_nsamp{int(nsamp)}"

        csv_path = path_with_suffix(Path(args.csv_out), suffix)

        if not args.quiet:
            loop_period = float(nsamp) / float(args.sample_rate)
            print(
                f"Starting run {idx + 1}/{len(nsamp_values)} | mode={args.mode} | nsamp={nsamp} | "
                f"loop_period_nom={loop_period:.6f}s | csv={csv_path}",
                flush=True,
            )
            if loop_period > 1.0:
                print(
                    "Warning: nominal loop period exceeds 1 second; consider smaller nsamp or higher sample rate.",
                    flush=True,
                )

        run_summary = asyncio.run(run_one(args, int(nsamp), csv_path))
        summaries.append(run_summary)

        if not args.quiet:
            print_run_summary(run_summary)

    ensure_parent_dir(Path(args.summary_out))
    with Path(args.summary_out).open("w") as fh:
        if len(summaries) == 1:
            json.dump(summaries[0], fh, indent=2, sort_keys=True)
        else:
            json.dump(summaries, fh, indent=2, sort_keys=True)

    summary_csv_path = args.summary_csv
    if summary_csv_path is None:
        summary_csv_path = Path(args.summary_out).with_suffix(".csv")
    write_summary_csv(summaries, Path(summary_csv_path))

    if not args.quiet:
        print(f"Summary JSON written to: {Path(args.summary_out)}")
        print(f"Summary CSV written to:  {Path(summary_csv_path)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
