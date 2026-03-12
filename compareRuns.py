#!/usr/bin/env python3
"""
Compare open-loop vs closed-loop clock discipline runs.

Usage:
  python compare_runs.py --open csvTests/open.csv --closed csvTests/closed.csv --outdir results

Notes:
- Uses y_frac (fractional frequency error) for Allan deviation.
- Uses f_meas_unwrap_hz (if present) as primary frequency-error series; falls back to f_off_hz.
- Uses tau_err_s as primary "time error" series; falls back to phi_err_rad/(2π f_ref_hz) if needed.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def robust_float_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s

def safe_median_dt(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return np.nan
    return float(np.median(dt))

def detrend_linear(t: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Remove best-fit line x ≈ a*t + b. Return residual, slope a, intercept b.
    """
    m = np.isfinite(t) & np.isfinite(x)
    if m.sum() < 2:
        return x * np.nan, np.nan, np.nan
    tt = t[m]
    xx = x[m]
    A = np.vstack([tt, np.ones_like(tt)]).T
    a, b = np.linalg.lstsq(A, xx, rcond=None)[0]
    resid = x.copy()
    resid[m] = xx - (a * tt + b)
    return resid, float(a), float(b)

def rms(x: np.ndarray) -> float:
    m = np.isfinite(x)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean(x[m] ** 2)))

def mad_std(x: np.ndarray) -> float:
    """Robust sigma estimate: 1.4826 * MAD."""
    m = np.isfinite(x)
    if m.sum() == 0:
        return np.nan
    med = np.median(x[m])
    mad = np.median(np.abs(x[m] - med))
    return float(1.4826 * mad)

def wrap_to_pi(rad: np.ndarray) -> np.ndarray:
    return (rad + np.pi) % (2 * np.pi) - np.pi


# -----------------------------
# Allan deviation (simple, overlapping)
# -----------------------------
def allan_deviation(y: np.ndarray, tau0: float, taus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlapping Allan deviation for fractional frequency y sampled at tau0.

    Returns:
      taus_valid, adev
    """
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    y = y[m]
    if len(y) < 10 or not np.isfinite(tau0) or tau0 <= 0:
        return np.array([]), np.array([])

    # Remove mean (helps numerics)
    y = y - np.mean(y)

    adev = []
    taus_ok = []
    for tau in taus:
        n = int(round(tau / tau0))
        if n < 1:
            continue
        # overlapping Allan variance uses second difference of averaged y
        # Compute running average of length n
        if 2 * n >= len(y):
            continue
        y_avg = np.convolve(y, np.ones(n) / n, mode="valid")
        # Second difference
        d2 = y_avg[2*n:] - 2 * y_avg[n:-n] + y_avg[:-2*n]
        avar = 0.5 * np.mean(d2 ** 2)
        adev.append(np.sqrt(avar))
        taus_ok.append(n * tau0)

    return np.array(taus_ok), np.array(adev)


# -----------------------------
# Loading and selecting signals
# -----------------------------
@dataclass
class Signals:
    name: str
    t: np.ndarray
    f_err_hz: np.ndarray
    y_frac: np.ndarray
    tau_err_s: np.ndarray
    phi_err_rad: np.ndarray
    coh_db: np.ndarray
    snr_fft_db: np.ndarray
    u_hz: np.ndarray
    dac_code: np.ndarray
    mode: Optional[np.ndarray]
    f_ref_hz: np.ndarray

def choose_freq_error(df: pd.DataFrame) -> pd.Series:
    # Prefer unwrapped measured frequency if present
    if "f_meas_unwrap_hz" in df.columns:
        return robust_float_series(df["f_meas_unwrap_hz"])
    # Next best: estimated unwrapped
    if "f_unwrap_est_hz" in df.columns:
        return robust_float_series(df["f_unwrap_est_hz"])
    # Fallback: f_off_hz (may be biased in your earlier versions, but still useful)
    return robust_float_series(df["f_off_hz"])

def choose_tau_err(df: pd.DataFrame) -> pd.Series:
    if "tau_err_s" in df.columns:
        return robust_float_series(df["tau_err_s"])
    # Fallback: tau ≈ phi_err/(2π f_ref)
    phi = robust_float_series(df["phi_err_rad"]) if "phi_err_rad" in df.columns else pd.Series(np.nan, index=df.index)
    fref = robust_float_series(df["f_ref_hz"]) if "f_ref_hz" in df.columns else pd.Series(np.nan, index=df.index)
    tau = phi / (2 * np.pi * fref)
    return tau

def load_signals(path: str, name: str) -> Signals:
    df = pd.read_csv(path)
    # Standardize expected columns
    t = robust_float_series(df["t_rel_s"]).to_numpy()
    f_err = choose_freq_error(df).to_numpy()
    y_frac = robust_float_series(df["y_frac"]).to_numpy() if "y_frac" in df.columns else (f_err / 1e8)
    tau_err = choose_tau_err(df).to_numpy()
    phi_err = robust_float_series(df["phi_err_rad"]).to_numpy() if "phi_err_rad" in df.columns else np.full_like(t, np.nan)
    coh = robust_float_series(df["coh_db"]).to_numpy() if "coh_db" in df.columns else np.full_like(t, np.nan)
    snr = robust_float_series(df["snr_fft_db"]).to_numpy() if "snr_fft_db" in df.columns else np.full_like(t, np.nan)
    u = robust_float_series(df["u_hz"]).to_numpy() if "u_hz" in df.columns else np.full_like(t, np.nan)
    dac = robust_float_series(df["dac_code_rb"]).to_numpy() if "dac_code_rb" in df.columns else (
        robust_float_series(df["dac_code_cmd"]).to_numpy() if "dac_code_cmd" in df.columns else np.full_like(t, np.nan)
    )
    mode = df["mode"].astype(str).to_numpy() if "mode" in df.columns else None
    f_ref = robust_float_series(df["f_ref_hz"]).to_numpy() if "f_ref_hz" in df.columns else np.full_like(t, np.nan)

    return Signals(
        name=name, t=t, f_err_hz=f_err, y_frac=y_frac, tau_err_s=tau_err,
        phi_err_rad=phi_err, coh_db=coh, snr_fft_db=snr, u_hz=u, dac_code=dac,
        mode=mode, f_ref_hz=f_ref
    )


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(sig: Signals) -> Dict[str, float]:
    t = sig.t
    f = sig.f_err_hz
    tau = sig.tau_err_s
    coh = sig.coh_db
    u = sig.u_hz
    dac = sig.dac_code

    dt = safe_median_dt(t)

    # Frequency stats
    f_resid, f_slope_hz_per_s, _ = detrend_linear(t, f)
    f_mean = float(np.nanmean(f))
    f_std = float(np.nanstd(f))
    f_rms = rms(f)
    f_std_robust = mad_std(f)
    f_resid_rms = rms(f_resid)

    # Time error stats
    tau_resid, tau_slope_s_per_s, _ = detrend_linear(t, tau)
    tau_mean = float(np.nanmean(tau))
    tau_std = float(np.nanstd(tau))
    tau_rms = rms(tau)
    tau_resid_rms = rms(tau_resid)

    # Coherence / SNR summaries
    coh_med = float(np.nanmedian(coh))
    snr_med = float(np.nanmedian(sig.snr_fft_db))

    # Actuator activity (closed-loop should have some, open-loop almost none)
    u_rms = rms(u)
    # DAC code step activity
    dd = np.diff(dac)
    dd = dd[np.isfinite(dd)]
    dac_step_rms = float(np.sqrt(np.mean(dd**2))) if len(dd) else np.nan
    dac_span = float(np.nanmax(dac) - np.nanmin(dac)) if np.isfinite(dac).any() else np.nan

    return {
        "duration_s": float(np.nanmax(t) - np.nanmin(t)) if np.isfinite(t).any() else np.nan,
        "median_dt_s": dt,
        "f_mean_hz": f_mean,
        "f_std_hz": f_std,
        "f_std_robust_hz": f_std_robust,
        "f_rms_hz": f_rms,
        "f_drift_slope_hz_per_s": f_slope_hz_per_s,
        "f_resid_rms_hz": f_resid_rms,
        "tau_mean_s": tau_mean,
        "tau_std_s": tau_std,
        "tau_rms_s": tau_rms,
        "tau_drift_slope_s_per_s": tau_slope_s_per_s,
        "tau_resid_rms_s": tau_resid_rms,
        "coh_db_median": coh_med,
        "snr_fft_db_median": snr_med,
        "u_rms_hz": u_rms,
        "dac_step_rms_codes": dac_step_rms,
        "dac_span_codes": dac_span,
    }


# -----------------------------
# Plotting
# -----------------------------
def plot_timeseries(ax, t, x, label, ylabel, title):
    ax.plot(t, x, label=label, linewidth=1)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", dest="open_path", required=True, help="Open-loop CSV")
    ap.add_argument("--closed", dest="closed_path", required=True, help="Closed-loop CSV")
    ap.add_argument("--outdir", default="results", help="Output directory for plots/metrics")
    ap.add_argument("--f_tone_hz", type=float, default=100e6, help="RF tone frequency for labeling")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    open_sig = load_signals(args.open_path, "open")
    closed_sig = load_signals(args.closed_path, "closed")

    # Metrics
    m_open = compute_metrics(open_sig)
    m_closed = compute_metrics(closed_sig)

    metrics_df = pd.DataFrame([m_open, m_closed], index=["open", "closed"]).T
    metrics_path = os.path.join(args.outdir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path)

    # Allan deviation using y_frac
    dt_open = safe_median_dt(open_sig.t)
    dt_closed = safe_median_dt(closed_sig.t)
    # Choose a common tau0 for plotting (use each run’s own dt for computation)
    # Taus: log-spaced between ~1 sample and ~1/10 total duration
    def make_taus(sig: Signals):
        dt = safe_median_dt(sig.t)
        dur = float(np.nanmax(sig.t) - np.nanmin(sig.t))
        if not np.isfinite(dt) or not np.isfinite(dur) or dur <= 0:
            return np.array([])
        tau_min = dt
        tau_max = max(dt * 2, dur / 10)
        # 20 points logspace
        return np.unique(np.round(np.logspace(np.log10(tau_min), np.log10(tau_max), 20) / dt).astype(int)) * dt

    taus_open = make_taus(open_sig)
    taus_closed = make_taus(closed_sig)
    to, adev_o = allan_deviation(open_sig.y_frac, dt_open, taus_open) if len(taus_open) else (np.array([]), np.array([]))
    tc, adev_c = allan_deviation(closed_sig.y_frac, dt_closed, taus_closed) if len(taus_closed) else (np.array([]), np.array([]))

    # -----------------------------
    # Plots
    # -----------------------------
    # 1) Frequency error vs time (raw)
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_timeseries(ax, open_sig.t, open_sig.f_err_hz, "open", "Hz", "Frequency error vs time")
    plot_timeseries(ax, closed_sig.t, closed_sig.f_err_hz, "closed", "Hz", "Frequency error vs time")
    ax.legend()
    save_fig(os.path.join(args.outdir, "freq_error_vs_time.png"))

    # 2) Frequency drift removed (residual after linear detrend)
    open_f_resid, open_slope, _ = detrend_linear(open_sig.t, open_sig.f_err_hz)
    closed_f_resid, closed_slope, _ = detrend_linear(closed_sig.t, closed_sig.f_err_hz)
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_timeseries(ax, open_sig.t, open_f_resid, f"open (slope={open_slope:.3g} Hz/s)", "Hz", "Frequency residual (linear detrend)")
    plot_timeseries(ax, closed_sig.t, closed_f_resid, f"closed (slope={closed_slope:.3g} Hz/s)", "Hz", "Frequency residual (linear detrend)")
    ax.legend()
    save_fig(os.path.join(args.outdir, "freq_residual_detrended.png"))

    # 3) Time error (tau_err_s) vs time
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_timeseries(ax, open_sig.t, open_sig.tau_err_s, "open", "s", "Time error (tau_err_s) vs time")
    plot_timeseries(ax, closed_sig.t, closed_sig.tau_err_s, "closed", "s", "Time error (tau_err_s) vs time")
    ax.legend()
    save_fig(os.path.join(args.outdir, "time_error_tau_vs_time.png"))

    # 4) Coherence and SNR vs time (closed-loop is usually what you care about)
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_timeseries(ax, open_sig.t, open_sig.coh_db, "open", "dB", "Coherence vs time")
    plot_timeseries(ax, closed_sig.t, closed_sig.coh_db, "closed", "dB", "Coherence vs time")
    ax.legend()
    save_fig(os.path.join(args.outdir, "coherence_vs_time.png"))

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_timeseries(ax, open_sig.t, open_sig.snr_fft_db, "open", "dB", "FFT SNR vs time")
    plot_timeseries(ax, closed_sig.t, closed_sig.snr_fft_db, "closed", "dB", "FFT SNR vs time")
    ax.legend()
    save_fig(os.path.join(args.outdir, "snr_fft_vs_time.png"))

    # 5) Actuator effort (u_hz) and DAC code vs time
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_timeseries(ax, open_sig.t, open_sig.u_hz, "open", "Hz", "Actuator command u_hz vs time")
    plot_timeseries(ax, closed_sig.t, closed_sig.u_hz, "closed", "Hz", "Actuator command u_hz vs time")
    ax.legend()
    save_fig(os.path.join(args.outdir, "u_hz_vs_time.png"))

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_timeseries(ax, open_sig.t, open_sig.dac_code, "open", "codes", "DAC code vs time")
    plot_timeseries(ax, closed_sig.t, closed_sig.dac_code, "closed", "codes", "DAC code vs time")
    ax.legend()
    save_fig(os.path.join(args.outdir, "dac_code_vs_time.png"))

    # 6) Allan deviation plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if len(to):
        ax.loglog(to, adev_o, marker="o", label="open")
    if len(tc):
        ax.loglog(tc, adev_c, marker="o", label="closed")
    ax.set_xlabel("Averaging time τ (s)")
    ax.set_ylabel("Allan deviation σy(τ)")
    ax.set_title("Allan deviation (from y_frac)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    save_fig(os.path.join(args.outdir, "allan_deviation.png"))

    # Write a readable text summary too
    summary_txt = os.path.join(args.outdir, "metrics_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("=== Summary metrics (open vs closed) ===\n\n")
        f.write(metrics_df.to_string(float_format=lambda v: f"{v: .6g}") + "\n\n")
        f.write("Interpretation tips:\n")
        f.write("- f_drift_slope_hz_per_s: closer to 0 is better (less drift).\n")
        f.write("- f_resid_rms_hz: lower is better (less short-term noise after removing drift).\n")
        f.write("- tau_resid_rms_s: lower is better (time/phase error stability).\n")
        f.write("- Allan deviation: closed should beat open at larger τ if drift is suppressed.\n")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote plots to: {args.outdir}/")
    print(f"Wrote summary: {summary_txt}")


if __name__ == "__main__":
    main()

