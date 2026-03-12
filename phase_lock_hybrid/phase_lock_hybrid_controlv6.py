#!/usr/bin/env python3
"""
phase_lock_hybrid_control.py

Phase-increment estimator for f_off + hybrid control:
- Coarse corrections via Si5351 I2C frequency set (Clock0) when |u_hz| is large
- Fine corrections via VC pin (MCP4725 DAC) when |u_hz| is small

Also supports --open_loop to hold DAC steady and not apply corrections.

Estimator:
    f_off = (fs / 2π) * angle( sum z[n] * conj(z[n-1]) )
where z = iq * exp(-j 2π f_ref t)

CSV columns:
t_rel_s, f_off_hz, f_off_drift_hz, y_frac, y_drift_frac, x_int_s,
amp_rms, coh_db, snr_fft_db,
actuator, u_hz,
clock_f_cmd_hz, clock_f_rb_hz,
dac_code_cmd, dac_code_rb, dac_volts, dac_code_f,
Kv_hz_per_code, f_ref_hz, mode
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from rtlsdr import RtlSdr

from clock_vcxo_helper import ClockVCXO


# -----------------------------
# SDR helpers
# -----------------------------
def safe_set_freq_correction(sdr: RtlSdr, ppm_val: float) -> None:
    """
    In your environment, sdr.freq_correction = float(ppm) errors (ctypes wrong type).
    Set int ppm and guard.
    """
    try:
        sdr.freq_correction = int(round(float(ppm_val)))
    except Exception as e:
        print(f"Warning: could not set freq_correction ppm={ppm_val!r}: {e}", flush=True)


def make_model_phasor(f_ref: float, f_sample: float, nsamp: int) -> np.ndarray:
    n = np.arange(nsamp, dtype=np.float64)
    ts = n / f_sample
    return np.exp(-2j * np.pi * f_ref * ts).astype(np.complex64)


# -----------------------------
# Estimator + diagnostics
# -----------------------------
def f_off_and_coherence(demod: np.ndarray, f_sample: float, trim: int = 0) -> Tuple[float, float, float]:
    """
    Returns:
      f_off_hz : phase-increment estimate
      coh_db   : coherence consistent with estimator (20log10(|S| / sum|z|^2))
      amp_rms  : sqrt(mean(|z|^2))
    """
    z = np.asarray(demod, np.complex64).reshape(-1)
    n0 = z.size
    trim = int(max(0, trim))
    if n0 < 256 or 2 * trim >= n0 - 128:
        return float("nan"), float("nan"), float("nan")
    z = z[trim:-trim]
    if z.size < 256:
        return float("nan"), float("nan"), float("nan")

    s = np.sum(z[1:] * np.conjugate(z[:-1]))
    p = np.sum(np.abs(z) ** 2) + 1e-30

    if not (np.isfinite(s.real) and np.isfinite(s.imag)):
        return float("nan"), float("nan"), float("nan")

    f_off = float((f_sample / (2.0 * np.pi)) * np.angle(s))
    coh = float(np.abs(s) / p)
    coh_db = float(20.0 * np.log10(coh + 1e-30))
    amp_rms = float(np.sqrt(np.mean(np.abs(z) ** 2) + 1e-30))
    return f_off, coh_db, amp_rms




def phase_err_and_coherence(demod: np.ndarray, trim: int = 0) -> Tuple[float, float, float, float]:
    """
    Phase discriminator for a CW tone after demodulating to (near) DC.

    Returns:
      phi_wrap : wrapped phase of mean(demod) in radians (-pi..pi]
      coh_db   : 20log10(|mean(z)| / mean(|z|))  (a practical coherence proxy)
      amp_rms  : sqrt(mean(|z|^2))
      mag_mean : |mean(z)| (useful for diagnostics)
    """
    z = np.asarray(demod, np.complex64).reshape(-1)
    n0 = z.size
    trim = int(max(0, trim))
    if (2 * trim) >= n0:
        trim = 0
    if trim > 0:
        z = z[trim:-trim]

    # Robust stats
    z_mean = np.mean(z)
    mag_mean = float(np.abs(z_mean))
    amp_rms = float(np.sqrt(np.mean(np.abs(z) ** 2)))

    denom = float(np.mean(np.abs(z))) + 1e-12
    coh = mag_mean / denom
    coh_db = float(20.0 * np.log10(coh + 1e-15))

    phi_wrap = float(np.angle(z_mean))
    return phi_wrap, coh_db, amp_rms, mag_mean

def fft_snr_db_near_dc(
    demod: np.ndarray,
    f_sample: float,
    trim: int = 0,
    search_hz: float = 200_000.0,
    exclude_bins: int = 8,
) -> float:
    z = np.asarray(demod, np.complex64).reshape(-1)
    n0 = z.size
    trim = int(max(0, trim))
    if n0 < 2048 or 2 * trim >= n0 - 1024:
        return float("nan")
    z = z[trim:-trim]
    n = z.size
    if n < 2048:
        return float("nan")

    w = np.hanning(n).astype(np.float32)
    Z = np.fft.fft(z * w)
    mag = np.abs(Z).astype(np.float64)
    freqs = np.fft.fftfreq(n, d=1.0 / f_sample)

    mask = np.abs(freqs) <= float(abs(search_hz))
    idx = np.flatnonzero(mask)
    if idx.size < 64:
        return float("nan")
    m = mag[idx]
    m = m[np.isfinite(m)]
    if m.size < 64:
        return float("nan")

    k_peak_local = int(np.argmax(m))
    pk = float(m[k_peak_local])

    lo = max(0, k_peak_local - int(exclude_bins))
    hi = min(m.size, k_peak_local + int(exclude_bins) + 1)
    m_noise = np.concatenate([m[:lo], m[hi:]]) if (lo > 0 or hi < m.size) else m
    if m_noise.size < 32:
        return float("nan")
    med = float(np.median(m_noise) + 1e-30)
    return float(20.0 * np.log10(pk / med + 1e-30))

def peak_freq_hz_near_dc(
    x: np.ndarray,
    f_sample: float,
    trim: int = 0,
    search_hz: float = 200_000.0,
    exclude_bins: int = 8,
) -> float:
    """
    Return the frequency (Hz) of the strongest FFT bin within +/-search_hz of DC.
    Excludes a small band around DC to avoid DC spur dominance.
    """
    z = np.asarray(x, np.complex64).reshape(-1)
    n0 = z.size
    trim = int(max(0, trim))
    if (2 * trim) >= n0:
        trim = 0
    if trim > 0:
        z = z[trim:-trim]

    n = z.size
    if n < 16:
        return float("nan")

    # FFT
    X = np.fft.fft(z)
    freqs = np.fft.fftfreq(n, d=1.0 / float(f_sample))
    P = np.abs(X) ** 2

    # Window around DC
    mask = np.abs(freqs) <= float(search_hz)

    # Exclude bins around DC (both sides)
    df = float(f_sample) / float(n)
    dc_excl_hz = float(exclude_bins) * df
    mask &= (np.abs(freqs) >= dc_excl_hz)

    idx = np.where(mask)[0]
    if idx.size == 0:
        return float("nan")

    k = idx[np.argmax(P[idx])]
    return float(freqs[k])

# -----------------------------
# Control helpers
# -----------------------------
def clamp_code(code: float, code_min: int, code_max: int) -> float:
    return float(max(code_min, min(code_max, float(code))))


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # SDR / signal
    ap.add_argument("--duration_s", type=float, default=30 * 60)
    ap.add_argument("--f_sample", type=float, default=2.4e6)
    ap.add_argument("--nsamp", type=int, default=262144)
    ap.add_argument("--t_update", type=float, default=0.5)
    ap.add_argument("--f_center", type=float, required=True)
    ap.add_argument("--f_tone_rf", type=float, required=True)
    ap.add_argument("--gain", type=str, default="1")
    ap.add_argument("--ppm", type=float, default=120)

    # Clock / DAC
    ap.add_argument("--init_clock", action="store_true")
    ap.add_argument("--apr_ppm", type=int, default=120)
    ap.add_argument("--start_code", type=int, default=2048)
    ap.add_argument("--settle_s", type=float, default=0.0)

    # Coarse clock setpoint
    ap.add_argument("--f_out_hz", type=float, default=28.8e6,
                    help="Nominal Si5351 output frequency (Clock0) for coarse control.")
    ap.add_argument("--coarse_thresh_hz", type=float, default=10_000.0,
                    help="If |u_hz| >= this, apply coarse I2C frequency step instead of DAC VC.")
    ap.add_argument("--max_coarse_step_hz", type=float, default=50_000.0,
                    help="Limit magnitude of each coarse I2C step (Hz).")
    ap.add_argument("--max_coarse_offset_hz", type=float, default=20_000.0,
                    help="Absolute clamp: limit total coarse deviation from nominal f_out_hz (Hz).")
    ap.add_argument("--coarse_settle_s", type=float, default=0.05,
                    help="Optional delay after I2C frequency update (seconds).")
    ap.add_argument("--reset_i_on_coarse", action="store_true",
                    help="Reset PI integrator when a coarse step is applied (anti-windup).")

    # Calibration: DAC VC gain
    ap.add_argument("--cal_code_lo", type=int, default=410)
    ap.add_argument("--cal_code_hi", type=int, default=3686)
    ap.add_argument("--cal_delta_f_hz", type=float, default=17000.0)

    # PI gains (on drift error in Hz)
    ap.add_argument("--Kp_hz", type=float, default=0.4)
    ap.add_argument("--Ki_hz", type=float, default=0.05)

    # Phase-loop gains (DPLL): output u_hz in Hz from phase error (radians)
    # u_hz = Kp_phi * phi_err + Ki_phi * ∫phi_err dt
    ap.add_argument("--Kp_phi", type=float, default=0.8,
                    help="Proportional gain for phase loop (Hz per rad).")
    ap.add_argument("--Ki_phi", type=float, default=0.05,
                    help="Integral gain for phase loop (Hz per (rad*s)).")
    ap.add_argument("--unwrap_alpha", type=float, default=0.2,
                    help="EWMA coefficient for residual frequency estimate used in phase unwrapping (0..1).")
    ap.add_argument("--min_coh_db", type=float, default=-35.0,
                    help="Hold actuator if coherence (dB) falls below this threshold.")

    # Pull-in (frequency) stage
    ap.add_argument("--phase_switch_hz", type=float, default=50.0,
                    help="If |f_pull_hz - f_off_target_hz| > this, run frequency pull-in; else run phase lock.")
    ap.add_argument("--Kf_pullin", type=float, default=0.3,
                    help="Pull-in gain: u_hz = -Kf_pullin * (f_pull_hz - f_off_target_hz).")
    ap.add_argument("--pullin_use_peak", action="store_true",
                    help="Use FFT peak frequency near DC (demod peak) as f_pull_hz instead of phase-increment f_off_hz.")
    ap.add_argument("--pullin_min_snr_db", type=float, default=20.0,
                    help="Hold during pull-in if SNRfft (dB) is below this.")
    # Behavior / safety
    ap.add_argument("--open_loop", action="store_true",
                    help="If set, hold DAC steady and do not apply any corrections.")
    ap.add_argument("--code_min", type=int, default=0)
    ap.add_argument("--code_max", type=int, default=4095)
    ap.add_argument("--max_step_codes", type=float, default=200.0)
    ap.add_argument("--warmup_blocks", type=int, default=1)

    # Estimator
    ap.add_argument("--trim", type=int, default=2048)
    ap.add_argument("--f_off_target_hz", type = float, default = 0.0, help = "control setpoint for absolute frequency offset (Hz)")

    # FFT diag
    ap.add_argument("--fft_search_hz", type=float, default=200_000.0)
    ap.add_argument("--fft_exclude_bins", type=int, default=8)

    # Output
    ap.add_argument("--out_csv", type=str, default="phase_lock_hybrid_log.csv")

    args = ap.parse_args()

    f_sample = float(args.f_sample)
    nsamp = int(args.nsamp)
    t_block = nsamp / f_sample
    t_update = float(args.t_update)
    if t_update < t_block:
        raise ValueError(f"t_update must be >= t_block. Got t_update={t_update:.3f}, t_block={t_block:.3f}")

    f_center = float(args.f_center)
    f_tone_rf = float(args.f_tone_rf)
    f_ref = f_tone_rf - f_center

    # DAC calibration
    if int(args.cal_code_hi) == int(args.cal_code_lo):
        raise ValueError("cal_code_hi must differ from cal_code_lo.")
    Kv_hz_per_code = float(args.cal_delta_f_hz) / float(int(args.cal_code_hi) - int(args.cal_code_lo))

    model = make_model_phasor(f_ref=f_ref, f_sample=f_sample, nsamp=nsamp)

    prev_t_rel: Optional[float] = None

    # Drift-from-start reference (frequency; used for logging only)
    f0_ref: Optional[float] = None

    # Phase-loop reference and state
    phi_ref: Optional[float] = None          # unwrapped phase reference (radians), set once
    phi_unwrap_prev: Optional[float] = None  # previous unwrapped phase (radians)
    f_unwrap_est_hz: float = 0.0             # residual frequency estimate for unwrapping (Hz)
    int_tau_s: float = 0.0                   # integral of time error (s^2)

    x_int = 0.0

    # DAC state
    dac_code_i = int(max(args.code_min, min(args.code_max, int(args.start_code))))
    dac_code_f = float(dac_code_i)
    last_written_code = dac_code_i

    # Coarse clock state
    clock_f_cmd = float(args.f_out_hz)
    clock_f_rb = float("nan")

    clk = ClockVCXO()
    try:
        if args.init_clock:
            clk.init_clock_vcxo_28p8mhz(apr_ppm=int(args.apr_ppm))

        # Set initial coarse clock frequency (best-effort; requires helper support)
        try:
            clock_f_rb = float(clk.set_clock0_freq_hz(clock_f_cmd))
        except Exception:
            clock_f_rb = float("nan")

        # Set initial DAC
        last_written_code = clk.set_dac_code(dac_code_i, settle_s=float(args.settle_s))
        dac_code_f = float(last_written_code)

        sdr = RtlSdr()
        try:
            sdr.sample_rate = f_sample
            sdr.center_freq = f_center
            safe_set_freq_correction(sdr, float(args.ppm))
            if args.gain.lower() == "auto":
                sdr.gain = "auto"
            else:
                sdr.gain = float(args.gain)

            _ = sdr.read_samples(min(nsamp, 262144))

            t0 = time.time()
            t_end = t0 + float(args.duration_s)
            out_csv = Path(args.out_csv)

            with out_csv.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "t_rel_s",
                    "f_off_hz",
                    "f_off_drift_hz",
                    "y_frac",
                    "y_drift_frac",
                    "x_int_s",
                    "amp_rms",
                    "coh_db",
                    "phi_wrap_rad",
                    "phi_unwrap_rad",
                    "phi_err_rad",
                    "tau_err_s",
                    "f_unwrap_est_hz",
                    "f_meas_unwrap_hz",
                    "mag_mean",
                    "snr_fft_db",
                    "actuator",
                    "u_hz",
                    "clock_f_cmd_hz",
                    "clock_f_rb_hz",
                    "dac_code_cmd",
                    "dac_code_rb",
                    "dac_volts",
                    "dac_code_f",
                    "Kv_hz_per_code",
                    "f_ref_hz",
                    "mode",
                ])

                mode_str = "open_loop" if args.open_loop else "closed_loop"
                print(f"Mode={mode_str} | start_code={last_written_code} | Kv≈{Kv_hz_per_code:.3f} Hz/code")
                print(f"Coarse: f_out_hz={clock_f_cmd:.3f}, thresh={float(args.coarse_thresh_hz):.1f} Hz")
                print(f"Logging to: {out_csv.resolve()}")
                print("Press Ctrl+C to stop.\n", flush=True)

                block_idx = 0
                next_start = time.time()

                while True:
                    now = time.time()
                    if now >= t_end:
                        break
                    if now < next_start:
                        time.sleep(min(0.2, next_start - now))

                    t_capture_start = time.time()
                    iq = sdr.read_samples(nsamp).astype(np.complex64)
                    t_rel = t_capture_start - t0

                    demod = iq * model
                    f_peak_demod_hz = peak_freq_hz_near_dc(
                        demod,
                        f_sample=f_sample,
                        trim=int(args.trim),
                        search_hz=float(args.fft_search_hz),
                        exclude_bins=0,
                    )

                    # print occasionally so you don't spam
                    if (block_idx % 5) == 0:
                        print(f"demod peak: {f_peak_demod_hz: .1f} Hz (relative to DC)", flush=True)

                    if block_idx < int(args.warmup_blocks):
                        block_idx += 1
                        prev_t_rel = t_rel
                        next_start = t_capture_start + t_update
                        continue

                    f_off_hz, coh_db, amp_rms = f_off_and_coherence(demod, f_sample=f_sample, trim=int(args.trim))

                    demod_phi = demod

                    # Choose a derotation frequency consistent with the selected pull-in metric.
                    # When --pullin_use_peak is enabled, the FFT peak near DC is the most reliable
                    # estimate while far from lock; using it here prevents a persistent ~kHz residual
                    # in demod_phi due to estimator bias between FFT and phase-increment methods.
                    f_derot_hz = float(f_off_hz)
                    if args.pullin_use_peak and np.isfinite(f_peak_demod_hz):
                        f_derot_hz = float(f_peak_demod_hz)

                    # Pass 1: remove coarse residual before mean-phase/coherence
                    if np.isfinite(f_derot_hz):
                        n = np.arange(nsamp, dtype=np.float32)
                        rot1 = np.exp(-2j * np.pi * (float(f_derot_hz) / float(f_sample)) * n).astype(np.complex64)
                        demod_phi = demod_phi * rot1

                    # Pass 2: estimate remaining residual on the partially-corrected signal
                    # IMPORTANT: in pullin_use_peak mode, use the SAME peak-based estimator here.
                    if args.pullin_use_peak:
                        f_res2_hz = peak_freq_hz_near_dc(demod_phi, f_sample=f_sample, trim=int(args.trim),
                                                        search_hz=float(args.fft_search_hz))
                    else:
                        f_res2_hz, _, _ = f_off_and_coherence(demod_phi, f_sample=f_sample, trim=int(args.trim))

                    if np.isfinite(f_res2_hz) and (abs(float(f_res2_hz)) > 1e-3):
                        rot2 = np.exp(-2j * np.pi * (float(f_res2_hz) / float(f_sample)) * n).astype(np.complex64)
                        demod_phi = demod_phi * rot2
                    f_peak_demod_phi_hz = peak_freq_hz_near_dc(demod_phi,f_sample=f_sample, trim = int(args.trim), search_hz = float(args.fft_search_hz),exclude_bins = int(args.fft_exclude_bins))
                    if(block_idx %5 == 0):
                        print(f"demod_phi peak:{f_peak_demod_phi_hz: .1f}Hz (relative to DC)", flush = True)
                    # Coherence / amplitude diagnostics: use the derotated signal (near DC) so coherence
                    # reflects SNR rather than residual frequency.
                    _phi_wrap_diag, coh_db_phi, amp_rms_phi, mag_mean = phase_err_and_coherence(demod_phi, trim=int(args.trim))
                    coh_db = coh_db_phi

                    # Phase/time error measurement: IMPORTANT
                    # Use the *model-only* demodulated signal (no residual-frequency derotation) so that
                    # clock drift appears as a phase ramp across blocks.
                    phi_wrap, _coh_raw, _amp_raw, _mag_raw = phase_err_and_coherence(demod, trim=int(args.trim))

                    
                    # Phase unwrapping + derived timing error (relative to initial reference)
                    phi_unwrap = float("nan")
                    phi_err = float("nan")
                    tau_err_s = float("nan")
                    f_meas_unwrap_hz = float("nan")

                    if np.isfinite(phi_wrap):
                        if phi_ref is None:
                            # Define the phase reference at the first good block after warmup.
                            phi_ref = float(phi_wrap)
                            phi_unwrap_prev = float(phi_wrap)
                            f_unwrap_est_hz = 0.0
                            int_tau_s = 0.0
                        elif (phi_unwrap_prev is not None) and (prev_t_rel is not None):
                            dt_u = max(float(t_rel - prev_t_rel), 1e-9)
                            alpha = clamp(float(args.unwrap_alpha), 0.0, 1.0)

                            # Predict phase using the residual frequency estimate, then unwrap to nearest branch.
                            phi_pred = float(phi_unwrap_prev) + (2.0 * np.pi * float(f_unwrap_est_hz) * dt_u)
                            k = int(round((phi_pred - float(phi_wrap)) / (2.0 * np.pi)))
                            phi_unwrap = float(phi_wrap) + (2.0 * np.pi * float(k))

                            phi_err = phi_unwrap - float(phi_ref)
                            tau_err_s = phi_err / (2.0 * np.pi * float(f_tone_rf))

                            # Update residual frequency estimate used for next unwrapping step
                            f_meas_unwrap_hz = (phi_unwrap - float(phi_unwrap_prev)) / (2.0 * np.pi * dt_u)
                            f_unwrap_est_hz = (1.0 - alpha) * float(f_unwrap_est_hz) + alpha * float(f_meas_unwrap_hz)

                            phi_unwrap_prev = float(phi_unwrap)


                    # Drift-from-start: set f0_ref once (first finite f_off after warmup),
                    # then define drift as f_off - f0_ref.
                    if np.isfinite(f_off_hz) and (f0_ref is None):
                        f0_ref = float(f_off_hz)

                    f0_val = float(f0_ref) if (f0_ref is not None) else float("nan")
                    f_off_drift = (float(f_off_hz) - float(f0_ref)) if (np.isfinite(f_off_hz) and (f0_ref is not None)) else float("nan")
                    y = (f_off_hz / f_tone_rf) if np.isfinite(f_off_hz) else float("nan")
                    y_drift = (f_off_drift / f_tone_rf) if np.isfinite(f_off_drift) else float("nan")

                    if prev_t_rel is not None and np.isfinite(y_drift):
                        dt_x = max(float(t_rel - prev_t_rel), 1e-9)
                        x_int += float(y_drift) * dt_x

                    snr_fft_db = fft_snr_db_near_dc(
                        demod,
                        f_sample=f_sample,
                        trim=int(args.trim),
                        search_hz=float(args.fft_search_hz),
                        exclude_bins=int(args.fft_exclude_bins),
                    )

                    actuator = "hold"
                    u_hz = float("nan")

                    dac_code_cmd = last_written_code
                    dac_code_rb = last_written_code

                    # Two-stage control:
                    #   1) Pull-in on frequency when far from DC (robust even with low coherence)
                    #   2) Phase lock when close enough and coherent
                    f_pull_hz = float(f_off_hz)
                    pullin_ok = np.isfinite(f_pull_hz) and np.isfinite(snr_fft_db) and (float(snr_fft_db) >= float(args.pullin_min_snr_db))
                    if (not args.open_loop) and pullin_ok:
                        # Stage A: frequency pull-in if still far (based on f_off_hz)
                        if abs(f_pull_hz - float(args.f_off_target_hz)) > float(args.phase_switch_hz):
                            u_hz = float(args.Kf_pullin) * (f_pull_hz - float(args.f_off_target_hz))

                    # Stage B: phase lock only when truly close AND coherent
                        elif np.isfinite(phi_err) and (float(coh_db) >= float(args.min_coh_db)):
                            dt_u = max(float(t_rel - (prev_t_rel if prev_t_rel is not None else t_rel)), 1e-9)
                            # PI on time error (tau_err_s) so the loop naturally drives frequency fast/slow to align sampling time.
                            # Map existing Kp_phi/Ki_phi (rad-domain) into equivalent tau-domain gains each update:
                            #   phi = tau * 2π f_tone  =>  u = Kp_phi*phi + Ki_phi*∫phi dt = (Kp_phi*2πf)*tau + (Ki_phi*2πf)*∫tau dt
                            int_tau_s += float(tau_err_s) * dt_u
                            Kp_tau = float(args.Kp_phi) * (2.0 * np.pi * float(f_tone_rf))
                            Ki_tau = float(args.Ki_phi) * (2.0 * np.pi * float(f_tone_rf))
                            u_hz = (Kp_tau * float(tau_err_s)) + (Ki_tau * float(int_tau_s))

                    if np.isfinite(u_hz):
                        # Hybrid split: big corrections -> coarse I2C, small -> DAC VC
                                                if abs(u_hz) >= float(args.coarse_thresh_hz):
                                                    actuator = "i2c_coarse"

                                                    step_hz = clamp(-u_hz, -float(args.max_coarse_step_hz), float(args.max_coarse_step_hz))

                                                    # Proposed new command
                                                    clock_f_cmd_new = clock_f_cmd + step_hz
                                                    # Absolute clamp around nominal to prevent "walking away"
                                                    f_nom = float(args.f_out_hz)
                                                    f_lo = f_nom - float(args.max_coarse_offset_hz)
                                                    f_hi = f_nom + float(args.max_coarse_offset_hz)
                                                    clock_f_cmd_new = clamp(clock_f_cmd_new, f_lo, f_hi)
                                                    clock_f_cmd = clock_f_cmd_new

                                                    # Apply coarse update (requires helper support)
                                                    clock_f_rb = float(clk.set_clock0_freq_hz(clock_f_cmd))
                                                    if float(args.coarse_settle_s) > 0:
                                                        time.sleep(float(args.coarse_settle_s))

                                                    # Optional anti-windup
                                                    if args.reset_i_on_coarse:
                                                        int_tau_s = 0.0

                                                    # Keep DAC steady during coarse moves
                                                    dac_code_cmd = last_written_code
                                                    dac_code_rb = last_written_code

                                                else:
                                                    actuator = "dac_fine"

                                                    delta_code = u_hz / Kv_hz_per_code
                                                    delta_code = clamp(delta_code, -float(args.max_step_codes), float(args.max_step_codes))

                                                    dac_code_f = clamp_code(dac_code_f + float(delta_code), args.code_min, args.code_max)
                                                    dac_code_cmd = int(round(dac_code_f))

                                                    if dac_code_cmd != last_written_code:
                                                        dac_code_rb = clk.set_dac_code(dac_code_cmd, settle_s=float(args.settle_s))
                                                        last_written_code = dac_code_rb
                                                        dac_code_f = float(dac_code_rb)
                                                    else:
                                                        dac_code_rb = last_written_code

                    dac_volts = clk.dac_code_to_volts(dac_code_rb)

                    w.writerow([
                        t_rel,
                        f_off_hz,
                        f_off_drift,
                        y,
                        y_drift,
                        x_int,
                        amp_rms,
                        coh_db,
                        phi_wrap,
                        phi_unwrap,
                        phi_err,
                        tau_err_s,
                        f_unwrap_est_hz,
                        f_meas_unwrap_hz,
                        mag_mean,
                        snr_fft_db,
                        actuator,
                        u_hz,
                        clock_f_cmd,
                        clock_f_rb,
                        dac_code_cmd,
                        dac_code_rb,
                        dac_volts,
                        dac_code_f,
                        Kv_hz_per_code,
                        f_ref,
                        mode_str,
                    ])
                    f.flush()

                    tag = "RUN"
                    f0_txt = f"{f0_val: .3f}" if np.isfinite(f0_val) else "   nan"
                    drift_txt = f"{f_off_drift: .3f}" if np.isfinite(f_off_drift) else "   nan"
                    if(args.pullin_use_peak):
                        drift_txt = " " 
                    u_txt = f"{u_hz: .1f}" if np.isfinite(u_hz) else " nan"
                    print(
                        f"t={t_rel:7.1f}s | {tag} f0={f0_txt} Hz | f_off={f_off_hz: .3f} Hz | drift={drift_txt} Hz | "
                        f"act={actuator:9s} u={u_txt:>8s} Hz | "
                        f"clk={clock_f_cmd/1e6:0.6f} MHz | code={dac_code_rb:4d} ({dac_volts:0.3f} V) | "
                        f"coh={coh_db:6.1f} dB | SNRfft~{snr_fft_db:6.1f} dB",
                        flush=True,
                    )

                    prev_t_rel = t_rel
                    block_idx += 1
                    next_start = t_capture_start + t_update

            print(f"\nDone. Wrote: {out_csv.resolve()}")
            print(f"t_block={t_block:.6f}s, t_update={t_update:.3f}s, f_ref={f_ref:.3f} Hz, Kv≈{Kv_hz_per_code:.3f} Hz/code")

        finally:
            sdr.close()

    finally:
        clk.close()


if __name__ == "__main__":
    main()

