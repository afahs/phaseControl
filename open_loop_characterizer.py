#!/usr/bin/env python3
"""
analyze_open_loop_pll_csv.py

Open-loop characterization for beacon_vcxo_pll_controller CSV logs.

Outputs:
- Summary metrics (frequency / phase / timing)
- Candidate phase setpoint (circular mean over quality-gated interval)
- Plots (optional)

Usage:
    python3 analyze_open_loop_pll_csv.py path/to/log.csv --beacon-freq 100e6 --plot
"""

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Helpers
# ---------------------------

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def robust_std(x: np.ndarray) -> float:
    """Robust std estimate via MAD."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def linear_fit(t: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit y = a + b t.
    Returns (a, b).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if t.size < 2:
        return np.nan, np.nan
    # Center time to reduce conditioning issues
    t0 = t[0]
    tc = t - t0
    b, a0 = np.polyfit(tc, y, 1)  # y = b*tc + a0
    a = a0 - b * t0
    return a, b


def detrend_linear(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Return residual y - (a + b t), plus (a, b).
    """
    a, b = linear_fit(t, y)
    if not np.isfinite(a) or not np.isfinite(b):
        return np.full_like(np.asarray(y, dtype=float), np.nan), a, b
    resid = y - (a + b * t)
    return resid, a, b


def circular_mean(phi_rad: np.ndarray) -> float:
    phi_rad = np.asarray(phi_rad, dtype=float)
    phi_rad = phi_rad[np.isfinite(phi_rad)]
    if phi_rad.size == 0:
        return np.nan
    z = np.exp(1j * phi_rad)
    return float(np.angle(np.mean(z)))


def circular_resultant_length(phi_rad: np.ndarray) -> float:
    phi_rad = np.asarray(phi_rad, dtype=float)
    phi_rad = phi_rad[np.isfinite(phi_rad)]
    if phi_rad.size == 0:
        return np.nan
    z = np.exp(1j * phi_rad)
    return float(np.abs(np.mean(z)))


def choose_col(df: pd.DataFrame, preferred: str, fallback: Optional[str] = None):
    if preferred in df.columns:
        return preferred
    if fallback is not None and fallback in df.columns:
        return fallback
    return None


@dataclass
class Summary:
    n_rows: int
    n_used: int
    duration_s: float
    dt_median_s: float
    dt_mean_s: float

    f_mean_hz: float
    f_std_hz: float
    f_std_robust_hz: float
    f_drift_slope_hz_per_s: float
    f_resid_rms_hz: float
    f_resid_robust_hz: float

    phi_circ_mean_rad: float
    phi_resultant_len: float
    phi_unwrap_slope_rad_per_s: float
    phi_unwrap_resid_rms_rad: float
    phi_unwrap_resid_robust_rad: float

    tau_jitter_rms_s: float
    tau_jitter_robust_s: float


# ---------------------------
# Core analysis
# ---------------------------

def analyze(
    df: pd.DataFrame,
    beacon_freq_hz: float,
    coh_min: Optional[float],
    snr_min_db: Optional[float],
    t_start_s: Optional[float],
    t_stop_s: Optional[float],
    require_open_loop: bool,
) -> Tuple[pd.DataFrame, Summary]:
    # Basic numeric coercion
    for c in [
        "elapsed_s", "unix_time", "coh", "snr_db",
        "phase_err_rad", "phase_err_unwrapped_rad",
        "freq_err_update_hz", "freq_err_block_hz",
        "peak_freq_baseband_hz", "peak_freq_abs_hz",
        "dt_update_s", "meas_quality_ok",
        "open_loop",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Time column
    tcol = choose_col(df, "elapsed_s", "unix_time")
    if tcol is None:
        raise ValueError("CSV must contain elapsed_s or unix_time")

    t = df[tcol].to_numpy(dtype=float)

    # Sort by time and deduplicate exact repeats if any
    order = np.argsort(t)
    df = df.iloc[order].reset_index(drop=True)
    t = df[tcol].to_numpy(dtype=float)

    # Build dt_update_s if missing
    if "dt_update_s" not in df.columns:
        dt = np.full(len(df), np.nan)
        if len(df) > 1:
            dt[1:] = np.diff(t)
        df["dt_update_s"] = dt
    else:
        # Fill missing values from elapsed diffs
        dt_calc = np.full(len(df), np.nan)
        if len(df) > 1:
            dt_calc[1:] = np.diff(t)
        m_missing = ~np.isfinite(df["dt_update_s"].to_numpy(dtype=float))
        arr = df["dt_update_s"].to_numpy(dtype=float)
        arr[m_missing] = dt_calc[m_missing]
        df["dt_update_s"] = arr

    # Base mask: finite time + key observables
    phase_col = choose_col(df, "phase_err_unwrapped_rad", "phase_err_rad")
    if phase_col is None:
        raise ValueError("CSV must contain phase_err_unwrapped_rad or phase_err_rad")

    freq_col = choose_col(df, "freq_err_update_hz", "freq_err_block_hz")
    if freq_col is None:
        raise ValueError("CSV must contain freq_err_update_hz or freq_err_block_hz")

    mask = np.isfinite(df[tcol].to_numpy(dtype=float))
    mask &= np.isfinite(df[phase_col].to_numpy(dtype=float))
    mask &= np.isfinite(df[freq_col].to_numpy(dtype=float))

    # Time window mask
    if t_start_s is not None:
        mask &= (df[tcol].to_numpy(dtype=float) >= float(t_start_s))
    if t_stop_s is not None:
        mask &= (df[tcol].to_numpy(dtype=float) <= float(t_stop_s))

    # Open-loop mask
    if require_open_loop and "open_loop" in df.columns:
        mask &= (df["open_loop"].to_numpy(dtype=float) > 0.5)

    # Quality mask
    if "meas_quality_ok" in df.columns:
        # If present, use it as a baseline quality indicator but allow stricter overrides below
        mq = df["meas_quality_ok"].to_numpy(dtype=float)
        mask &= np.isfinite(mq)
        mask &= (mq > 0.5)

    if coh_min is not None and "coh" in df.columns:
        coh = df["coh"].to_numpy(dtype=float)
        mask &= np.isfinite(coh)
        mask &= (coh >= float(coh_min))

    if snr_min_db is not None and "snr_db" in df.columns:
        snr = df["snr_db"].to_numpy(dtype=float)
        mask &= np.isfinite(snr)
        mask &= (snr >= float(snr_min_db))

    used = df.loc[mask].copy().reset_index(drop=True)
    if len(used) < 3:
        raise ValueError(
            "Too few usable samples after filtering. "
            "Try relaxing --coh-min / --snr-min or widening the time window."
        )

    tt = used[tcol].to_numpy(dtype=float)
    # Re-zero time for readable slopes/intercepts
    tt0 = tt[0]
    used["t_rel_s"] = tt - tt0
    t_rel = used["t_rel_s"].to_numpy(dtype=float)

    f = used[freq_col].to_numpy(dtype=float)
    phi_wrap = used["phase_err_rad"].to_numpy(dtype=float) if "phase_err_rad" in used.columns else wrap_to_pi(used[phase_col].to_numpy(dtype=float))

    # Prefer logged unwrapped phase if present and valid, else unwrap wrapped phase
    if "phase_err_unwrapped_rad" in used.columns and np.isfinite(used["phase_err_unwrapped_rad"].to_numpy(dtype=float)).sum() >= 3:
        phi_unwrap = used["phase_err_unwrapped_rad"].to_numpy(dtype=float)
    else:
        phi_unwrap = np.unwrap(phi_wrap)

    # Frequency stats
    f_resid, f_a, f_slope = detrend_linear(t_rel, f)
    f_mean = float(np.nanmean(f))
    f_std = float(np.nanstd(f, ddof=1)) if len(f) >= 2 else np.nan
    f_std_rob = robust_std(f)
    f_resid_rms = float(np.sqrt(np.nanmean(f_resid**2)))
    f_resid_rob = robust_std(f_resid)

    # Phase stats (unwrap trend + residual)
    phi_resid, phi_a, phi_slope = detrend_linear(t_rel, phi_unwrap)
    phi_resid_rms = float(np.sqrt(np.nanmean(phi_resid**2)))
    phi_resid_rob = robust_std(phi_resid)

    # Circular stats (candidate setpoint over selected interval)
    phi_cmean = circular_mean(phi_wrap)
    phi_R = circular_resultant_length(phi_wrap)

    # Timing jitter estimate at beacon frequency from detrended phase residual
    tau_rms = float(phi_resid_rms / (2.0 * math.pi * beacon_freq_hz))
    tau_rob = float(phi_resid_rob / (2.0 * math.pi * beacon_freq_hz))

    dt_arr = used["dt_update_s"].to_numpy(dtype=float)
    dt_mean = float(np.nanmean(dt_arr))
    dt_median = float(np.nanmedian(dt_arr))
    duration_s = float(t_rel[-1] - t_rel[0]) if len(t_rel) > 1 else 0.0

    summary = Summary(
        n_rows=len(df),
        n_used=len(used),
        duration_s=duration_s,
        dt_median_s=dt_median,
        dt_mean_s=dt_mean,

        f_mean_hz=f_mean,
        f_std_hz=f_std,
        f_std_robust_hz=f_std_rob,
        f_drift_slope_hz_per_s=float(f_slope),
        f_resid_rms_hz=f_resid_rms,
        f_resid_robust_hz=f_resid_rob,

        phi_circ_mean_rad=float(phi_cmean),
        phi_resultant_len=float(phi_R),
        phi_unwrap_slope_rad_per_s=float(phi_slope),
        phi_unwrap_resid_rms_rad=float(phi_resid_rms),
        phi_unwrap_resid_robust_rad=float(phi_resid_rob),

        tau_jitter_rms_s=float(tau_rms),
        tau_jitter_robust_s=float(tau_rob),
    )

    # Attach derived columns for plotting/export
    used["freq_used_hz"] = f
    used["freq_detrended_hz"] = f_resid
    used["phase_wrap_used_rad"] = phi_wrap
    used["phase_unwrap_used_rad"] = phi_unwrap
    used["phase_detrended_rad"] = phi_resid

    # Derived equivalent frequency from phase slope (consistency check)
    used.attrs["phase_slope_eq_hz"] = float(phi_slope / (2.0 * math.pi))

    return used, summary


def print_summary(summary: Summary):
    print("\n=== Open-loop characterization summary ===")
    print(f"Rows total / used: {summary.n_rows} / {summary.n_used}")
    print(f"Duration used:      {summary.duration_s:.3f} s")
    print(f"Update dt mean:     {summary.dt_mean_s:.6f} s")
    print(f"Update dt median:   {summary.dt_median_s:.6f} s")

    print("\n-- Frequency error (selected column) --")
    print(f"Mean:               {summary.f_mean_hz:+.6f} Hz")
    print(f"Std:                {summary.f_std_hz:.6f} Hz")
    print(f"Robust std (MAD):   {summary.f_std_robust_hz:.6f} Hz")
    print(f"Drift slope:        {summary.f_drift_slope_hz_per_s:+.6e} Hz/s")
    print(f"Detrended RMS:      {summary.f_resid_rms_hz:.6f} Hz")
    print(f"Detrended robust:   {summary.f_resid_robust_hz:.6f} Hz")

    print("\n-- Phase error --")
    print(f"Circular mean:      {summary.phi_circ_mean_rad:+.6f} rad  (candidate setpoint)")
    print(f"Resultant length R: {summary.phi_resultant_len:.6f}  (1=highly concentrated)")
    print(f"Unwrap slope:       {summary.phi_unwrap_slope_rad_per_s:+.6e} rad/s")
    print(f"Detrended RMS:      {summary.phi_unwrap_resid_rms_rad:.6f} rad")
    print(f"Detrended robust:   {summary.phi_unwrap_resid_robust_rad:.6f} rad")

    print("\n-- Timing jitter estimate @ beacon --")
    print(f"RMS:                {summary.tau_jitter_rms_s:.3e} s")
    print(f"Robust:             {summary.tau_jitter_robust_s:.3e} s")


def make_plots(used: pd.DataFrame, beacon_freq_hz: float, title_prefix: str = ""):
    t = used["t_rel_s"].to_numpy(dtype=float)

    # Plot 1: Frequency error
    plt.figure(figsize=(10, 4))
    plt.plot(t, used["freq_used_hz"].to_numpy(dtype=float), label="freq err")
    if "freq_detrended_hz" in used.columns:
        plt.plot(t, used["freq_detrended_hz"].to_numpy(dtype=float), label="detrended")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency error (Hz)")
    plt.title((title_prefix + "Frequency error vs time").strip())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Plot 2: Wrapped and unwrapped phase
    plt.figure(figsize=(10, 4))
    plt.plot(t, used["phase_wrap_used_rad"].to_numpy(dtype=float), ".", markersize=3, label="wrapped")
    plt.plot(t, used["phase_unwrap_used_rad"].to_numpy(dtype=float), label="unwrapped")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase error (rad)")
    plt.title((title_prefix + "Phase error vs time").strip())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Plot 3: Detrended phase (timing-relevant jitter proxy)
    plt.figure(figsize=(10, 4))
    phase_det = used["phase_detrended_rad"].to_numpy(dtype=float)
    plt.plot(t, phase_det)
    plt.xlabel("Time (s)")
    plt.ylabel("Detrended phase (rad)")
    plt.title((title_prefix + "Detrended phase residual").strip())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 4: Coherence / SNR if present
    if "coh" in used.columns or "snr_db" in used.columns:
        if "coh" in used.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(t, used["coh"].to_numpy(dtype=float))
            plt.xlabel("Time (s)")
            plt.ylabel("Coherence")
            plt.title((title_prefix + "Coherence vs time").strip())
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

        if "snr_db" in used.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(t, used["snr_db"].to_numpy(dtype=float))
            plt.xlabel("Time (s)")
            plt.ylabel("SNR (dB)")
            plt.title((title_prefix + "SNR vs time").strip())
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

    plt.show()


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=str)
    ap.add_argument("--beacon-freq", type=float, default=100e6,
                    help="Beacon frequency in Hz for timing jitter conversion (default: 100e6)")
    ap.add_argument("--coh-min", type=float, default=None,
                    help="Optional analysis-time coherence threshold")
    ap.add_argument("--snr-min", type=float, default=None,
                    help="Optional analysis-time SNR threshold (dB)")
    ap.add_argument("--t-start", type=float, default=None,
                    help="Optional start time in elapsed_s (or unix_time if elapsed_s absent)")
    ap.add_argument("--t-stop", type=float, default=None,
                    help="Optional stop time in elapsed_s (or unix_time if elapsed_s absent)")
    ap.add_argument("--include-closed-loop", action="store_true",
                    help="Do not require open_loop==1 in the CSV")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--export-filtered-csv", type=str, default=None,
                    help="Optional path to save filtered/derived dataset")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)

    used, summary = analyze(
        df=df,
        beacon_freq_hz=args.beacon_freq,
        coh_min=args.coh_min,
        snr_min_db=args.snr_min,
        t_start_s=args.t_start,
        t_stop_s=args.t_stop,
        require_open_loop=(not args.include_closed_loop),
    )

    print_summary(summary)

    phase_eq_hz = used.attrs.get("phase_slope_eq_hz", np.nan)
    if np.isfinite(phase_eq_hz):
        print(f"\nPhase-slope equivalent freq offset: {phase_eq_hz:+.6f} Hz")

    if args.export_filtered_csv:
        used.to_csv(args.export_filtered_csv, index=False)
        print(f"\nWrote filtered dataset: {args.export_filtered_csv}")

    if args.plot:
        make_plots(used, beacon_freq_hz=args.beacon_freq, title_prefix="Open-loop | ")


if __name__ == "__main__":
    main()
