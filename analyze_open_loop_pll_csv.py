#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze beacon VCXO PLL CSV logs with robust drift segmentation."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Analyze PLL controller CSV logs")
    p.add_argument("csv_path", type=str)
    p.add_argument(
        "--mode_filter",
        default="open_loop",
        choices=[
            "open_loop",
            "adjusted_loop",
            "quality_self",
            "quality_beacon",
            "closed_loop",
            "all",
        ],
                   help="Row pre-filter")
    p.add_argument("--t_start", type=float, default=None, help="Elapsed time lower bound (s)")
    p.add_argument("--t_end", type=float, default=None, help="Elapsed time upper bound (s)")
    p.add_argument("--coh_min", type=float, default=0.10)
    p.add_argument("--snr_min_db", type=float, default=10.0)
    p.add_argument("--window_s", type=float, default=60.0, help="Window length for windowed circular R")
    p.add_argument("--slope_window_s", type=float, default=300.0, help="Window length for piecewise drift slopes")
    p.add_argument("--min_window_points", type=int, default=8, help="Min points per slope window")
    p.add_argument("--min_stable_dwell_s", type=float, default=30.0,
                   help="Min contiguous stable duration for stable-interval drift stats")
    p.add_argument("--stable_mode", default="all", choices=["all", "open", "track"],
                   help="Stable-mask mode gating: all/OPEN-only/TRACK-only")
    p.add_argument("--ignore_meas_quality_ok", action="store_true",
                   help="Do not require logged meas_quality_ok in stable masks")
    p.add_argument("--include_all_rows_for_drift", action="store_true",
                   help="Use all filtered rows for the single global drift fit")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def _to_float(x, default=np.nan):
    if x is None:
        return float(default)
    s = str(x).strip()
    if s == "":
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def _to_int(x, default=0):
    if x is None:
        return int(default)
    s = str(x).strip()
    if s == "":
        return int(default)
    try:
        return int(float(s))
    except Exception:
        return int(default)


def circular_stats(phases, weights=None):
    phases = np.asarray(phases, dtype=np.float64)
    if phases.size == 0:
        return 0.0, 0.0
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
    return float(np.angle(z)), float(np.abs(z))


def finite_mask(*xs):
    if not xs:
        return np.array([], dtype=bool)
    m = np.ones(len(xs[0]), dtype=bool)
    for x in xs:
        m &= np.isfinite(x)
    return m


def read_csv_rows(path):
    rows = []
    with open(path, "r", newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            rows.append(r)
    return rows


def filter_rows(rows, args):
    out = []
    for r in rows:
        elapsed = _to_float(r.get("elapsed_s"))
        if args.t_start is not None and (not np.isfinite(elapsed) or elapsed < args.t_start):
            continue
        if args.t_end is not None and (not np.isfinite(elapsed) or elapsed > args.t_end):
            continue

        if args.mode_filter != "all":
            open_loop_val = _to_int(r.get("open_loop"), default=-1)
            mode_str = str(r.get("mode", "")).strip().upper()
            if args.mode_filter == "open_loop":
                if not (open_loop_val == 1 or mode_str == "OPEN"):
                    continue
            elif args.mode_filter == "adjusted_loop":
                if mode_str != "ADJUSTED":
                    continue
            elif args.mode_filter == "quality_self":
                if mode_str != "QUALITY_SELF":
                    continue
            elif args.mode_filter == "quality_beacon":
                if mode_str != "QUALITY_BEACON":
                    continue
            elif args.mode_filter == "closed_loop":
                if open_loop_val == 1 or mode_str in (
                    "OPEN",
                    "ADJUSTED",
                    "QUALITY_SELF",
                    "QUALITY_BEACON",
                ):
                    continue
        out.append(r)
    return out


def build_arrays(rows):
    keys = [
        "elapsed_s", "dt_update_s", "freq_err_update_hz", "freq_err_block_hz",
        "peak_freq_baseband_hz", "coh", "snr_db", "phase_err_rad",
        "phase_ctrl_err_raw_rad", "phase_ctrl_err_filt_rad", "phase_setpoint_rad",
        "meas_quality_ok", "z_abs", "nco_freq_bb_hz", "df_nco_peak_hz",
    ]
    arr = {k: [] for k in keys}
    arr["mode"] = []
    arr["action"] = []
    for r in rows:
        for k in keys:
            if k == "meas_quality_ok":
                arr[k].append(_to_int(r.get(k), default=0))
            else:
                arr[k].append(_to_float(r.get(k)))
        arr["mode"].append(str(r.get("mode", "")).strip().upper())
        arr["action"].append(str(r.get("action", "")))
    for k in keys:
        dtype = np.int64 if k == "meas_quality_ok" else np.float64
        arr[k] = np.asarray(arr[k], dtype=dtype)
    arr["mode"] = np.asarray(arr["mode"], dtype=object)
    arr["action"] = np.asarray(arr["action"], dtype=object)
    return arr


def cadence_stats(dt):
    m = np.isfinite(dt) & (dt > 0)
    if not np.any(m):
        return None
    x = dt[m]
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def drift_stats(t, f):
    m = finite_mask(t, f)
    if np.count_nonzero(m) < 3:
        return None
    tt = t[m]
    ff = f[m]
    x = tt - float(tt[0])
    coeff = np.polyfit(x, ff, 1)
    slope_hz_s = float(coeff[0])
    intercept = float(coeff[1])
    fit = slope_hz_s * x + intercept
    resid = ff - fit
    return {
        "n": int(ff.size),
        "mean": float(np.mean(ff)),
        "std": float(np.std(ff)),
        "slope_hz_s": slope_hz_s,
        "detrended_rms": float(np.sqrt(np.mean(resid ** 2))),
        "resid": resid,
        "t": tt,
        "f": ff,
    }


def windowed_resultant(t, phase, window_s):
    out = []
    m = finite_mask(t, phase)
    if np.count_nonzero(m) < 2:
        return out
    tt = t[m]
    ph = phase[m]
    w = float(window_s)
    if w <= 0:
        mu, R = circular_stats(ph)
        out.append((float(np.mean(tt)), float(R), float(mu), int(ph.size)))
        return out
    left = float(tt[0])
    stop = float(tt[-1])
    while left <= stop:
        right = left + w
        mk = (tt >= left) & (tt < right)
        if np.any(mk):
            mu, R = circular_stats(ph[mk])
            out.append((float(np.mean(tt[mk])), float(R), float(mu), int(np.count_nonzero(mk))))
        left += w
    return out


def contiguous_true_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    segs = []
    n = mask.size
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        segs.append((i, j))
        i = j
    return segs


def stable_segments(t, stable_mask, min_dwell_s):
    segs = []
    for i0, i1 in contiguous_true_segments(stable_mask):
        tt = t[i0:i1]
        if tt.size < 2:
            continue
        dwell = float(tt[-1] - tt[0])
        if dwell >= float(min_dwell_s):
            segs.append((i0, i1, dwell))
    return segs


def piecewise_window_slopes(t, f, mask, window_s, min_points):
    out = []
    m = finite_mask(t, f) & mask
    if np.count_nonzero(m) < max(3, min_points):
        return out
    tt = t[m]
    ff = f[m]
    w = float(window_s)
    if w <= 0:
        d = drift_stats(tt, ff)
        if d is not None and d["n"] >= min_points:
            out.append({
                "center_s": float(np.mean(tt)), "start_s": float(tt[0]), "end_s": float(tt[-1]),
                "n": int(d["n"]), "slope_hz_s": float(d["slope_hz_s"]), "std_hz": float(d["std"]),
                "detrended_rms_hz": float(d["detrended_rms"])})
        return out

    left = float(tt[0])
    stop = float(tt[-1])
    while left <= stop:
        right = left + w
        mk = (tt >= left) & (tt < right)
        if np.count_nonzero(mk) >= int(min_points):
            d = drift_stats(tt[mk], ff[mk])
            if d is not None:
                out.append({
                    "center_s": float(np.mean(tt[mk])),
                    "start_s": float(np.min(tt[mk])),
                    "end_s": float(np.max(tt[mk])),
                    "n": int(d["n"]),
                    "slope_hz_s": float(d["slope_hz_s"]),
                    "std_hz": float(d["std"]),
                    "detrended_rms_hz": float(d["detrended_rms"]),
                })
        left += w
    return out


def summarize_piecewise(slopes):
    if not slopes:
        return None
    s = np.asarray([x["slope_hz_s"] for x in slopes], dtype=np.float64)
    a = np.abs(s)
    return {
        "n_windows": int(s.size),
        "slope_median": float(np.median(s)),
        "slope_iqr": float(np.percentile(s, 75) - np.percentile(s, 25)),
        "slope_abs_p90": float(np.percentile(a, 90)),
        "slope_abs_max": float(np.max(a)),
    }


def estimate_timing_jitter_from_phase(t, phase, mask):
    m = finite_mask(t, phase) & mask
    if np.count_nonzero(m) < 10:
        return None
    tt = t[m]
    ph = phase[m]
    ph_unw = np.unwrap(ph)
    x = tt - float(tt[0])
    coeff = np.polyfit(x, ph_unw, 1)
    fit = coeff[0] * x + coeff[1]
    resid = ph_unw - fit
    return {
        "n": int(resid.size),
        "phase_resid_rms_rad": float(np.sqrt(np.mean(resid ** 2))),
        "phase_resid": resid,
        "t": tt,
    }


def mode_mask(mode_arr, stable_mode):
    if stable_mode == "all":
        return np.ones(mode_arr.size, dtype=bool)
    if stable_mode == "open":
        return mode_arr == "OPEN"
    if stable_mode == "track":
        return mode_arr == "TRACK"
    return np.ones(mode_arr.size, dtype=bool)


def setpoint_candidate(t, phase, stable_mask, window_s):
    win = windowed_resultant(t[stable_mask], phase[stable_mask], window_s)
    if not win:
        return None
    best = None
    for c_t, R, mu, n in win:
        if n < 5:
            continue
        if best is None or R > best[1]:
            best = (c_t, R, mu, n)
    if best is None:
        return None
    return {
        "center_s": float(best[0]),
        "R": float(best[1]),
        "phase_setpoint_rad": float(best[2]),
        "n": int(best[3]),
    }


def maybe_plot(arr, stable_mask, win_R, drift_global, piecewise_global, piecewise_stable, args):
    if not args.plot:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Plot requested but matplotlib is unavailable: {}".format(exc))
        return

    t = arr["elapsed_s"]
    f = arr["freq_err_update_hz"]
    ph = arr["phase_ctrl_err_filt_rad"]
    if not np.any(np.isfinite(ph)):
        ph = arr["phase_err_rad"]

    fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

    m = finite_mask(t, arr["coh"])
    axs[0].plot(t[m], arr["coh"][m], '.', ms=2, label='coh')
    axs[0].axhline(args.coh_min, color='r', ls='--', lw=1)
    axs[0].set_ylabel('coh')
    axs[0].grid(True, alpha=0.25)

    m = finite_mask(t, f)
    axs[1].plot(t[m], f[m], '.', ms=2, label='freq_err_update_hz')
    if drift_global is not None:
        tt = drift_global["t"]
        ff = drift_global["f"]
        x = tt - float(tt[0])
        coeff = np.polyfit(x, ff, 1)
        fit = coeff[0] * x + coeff[1]
        axs[1].plot(tt, fit, '-', lw=1.2, label='global fit')
    axs[1].set_ylabel('Hz')
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(loc='best')

    if piecewise_global:
        c = np.asarray([x["center_s"] for x in piecewise_global], dtype=np.float64)
        s = np.asarray([x["slope_hz_s"] for x in piecewise_global], dtype=np.float64)
        axs[2].plot(c, s, '-o', ms=3, lw=1.0, label='window slopes (global mask)')
    if piecewise_stable:
        c2 = np.asarray([x["center_s"] for x in piecewise_stable], dtype=np.float64)
        s2 = np.asarray([x["slope_hz_s"] for x in piecewise_stable], dtype=np.float64)
        axs[2].plot(c2, s2, '-o', ms=3, lw=1.0, label='window slopes (stable mask)')
    axs[2].axhline(0.0, color='k', lw=0.8)
    axs[2].set_ylabel('slope Hz/s')
    axs[2].grid(True, alpha=0.25)
    axs[2].legend(loc='best')

    if piecewise_stable:
        s2 = np.asarray([x["slope_hz_s"] for x in piecewise_stable], dtype=np.float64)
        axs[3].hist(s2, bins=30, alpha=0.8)
    axs[3].set_ylabel('slope hist')
    axs[3].grid(True, alpha=0.25)

    m = finite_mask(t, ph)
    axs[4].plot(t[m], ph[m], '.', ms=2, label='phase')
    stable_m = stable_mask & m
    if np.any(stable_m):
        axs[4].plot(t[stable_m], ph[stable_m], '.', ms=2, label='stable subset')
    axs[4].set_ylabel('rad')
    axs[4].grid(True, alpha=0.25)
    axs[4].legend(loc='best')

    if win_R:
        tw = np.asarray([w[0] for w in win_R], dtype=np.float64)
        rw = np.asarray([w[1] for w in win_R], dtype=np.float64)
        axs[5].plot(tw, rw, '-o', ms=3)
    axs[5].set_ylabel('windowed R')
    axs[5].set_xlabel('elapsed_s')
    axs[5].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def mode_level_stats(arr, stable_base_mask):
    out = {}
    for mode in ["OPEN", "ACQUIRE", "TRACK", "HOLDOVER"]:
        m = stable_base_mask & (arr["mode"] == mode)
        d = drift_stats(arr["elapsed_s"][m], arr["freq_err_update_hz"][m]) if np.any(m) else None
        if d is None:
            continue
        out[mode] = d
    return out


def print_drift_block(label, d):
    if d is None:
        print("{}: insufficient data".format(label))
        return
    print("{}: mean={:.6f} Hz std={:.6f} Hz slope={:.6e} Hz/s detrended_rms={:.6f} Hz (n={})".format(
        label, d["mean"], d["std"], d["slope_hz_s"], d["detrended_rms"], d["n"]))


def print_piecewise_block(label, pw):
    summ = summarize_piecewise(pw)
    if summ is None:
        print("{}: insufficient windows".format(label))
        return
    print("{}: windows={} slope_median={:+.6e} Hz/s slope_IQR={:.6e} Hz/s |slope|_p90={:.6e} Hz/s |slope|_max={:.6e} Hz/s".format(
        label,
        summ["n_windows"],
        summ["slope_median"],
        summ["slope_iqr"],
        summ["slope_abs_p90"],
        summ["slope_abs_max"],
    ))


def main():
    args = parse_args()
    rows = read_csv_rows(args.csv_path)
    if not rows:
        print("No rows in CSV")
        return 1
    rows_f = filter_rows(rows, args)
    if not rows_f:
        print("No rows after filtering")
        return 1

    arr = build_arrays(rows_f)
    t = arr["elapsed_s"]
    f = arr["freq_err_update_hz"]

    cadence = cadence_stats(arr["dt_update_s"])

    base = finite_mask(t, f, arr["coh"], arr["snr_db"])
    quality = base & (arr["coh"] >= float(args.coh_min)) & (arr["snr_db"] >= float(args.snr_min_db))

    if (not args.ignore_meas_quality_ok) and np.any(np.isfinite(arr["meas_quality_ok"].astype(np.float64))):
        quality = quality & (arr["meas_quality_ok"] > 0)

    mode_gate = mode_mask(arr["mode"], args.stable_mode)
    stable = quality & mode_gate

    phase = arr["phase_ctrl_err_filt_rad"].copy()
    if not np.any(np.isfinite(phase)):
        phase = arr["phase_err_rad"].copy()

    # Global drift fit (single line), optionally all rows
    drift_mask = base.copy()
    if not args.include_all_rows_for_drift:
        drift_mask = quality.copy()
    drift_global = drift_stats(t[drift_mask], f[drift_mask]) if np.any(drift_mask) else None

    # Mode-separated fits on quality rows
    mode_stats = mode_level_stats(arr, quality)

    # Piecewise slopes
    pw_global = piecewise_window_slopes(t, f, quality, args.slope_window_s, args.min_window_points)
    pw_stable = piecewise_window_slopes(t, f, stable, args.slope_window_s, args.min_window_points)

    # Stable segments and stable-only global drift
    segs = stable_segments(t, stable, args.min_stable_dwell_s)
    stable_points_mask = np.zeros_like(stable)
    for i0, i1, _ in segs:
        stable_points_mask[i0:i1] = True
    drift_stable = drift_stats(t[stable_points_mask], f[stable_points_mask]) if np.any(stable_points_mask) else None

    # Circular phase concentration
    phase_mask = finite_mask(t, phase)
    mu_all, R_all = circular_stats(phase[phase_mask]) if np.any(phase_mask) else (0.0, 0.0)
    mu_q, R_q = circular_stats(phase[quality]) if np.any(quality) else (0.0, 0.0)
    mu_st, R_st = circular_stats(phase[stable_points_mask]) if np.any(stable_points_mask) else (0.0, 0.0)

    win_R = windowed_resultant(t[stable_points_mask], phase[stable_points_mask], args.window_s) if np.any(stable_points_mask) else []
    setpoint = setpoint_candidate(t, phase, stable_points_mask, args.window_s)

    jitter = estimate_timing_jitter_from_phase(t, phase, stable_points_mask)
    beacon_hz = _to_float(rows_f[0].get("beacon_freq_hz"), default=np.nan)
    timing_jitter_s = np.nan
    if jitter is not None and np.isfinite(beacon_hz) and beacon_hz > 0:
        timing_jitter_s = jitter["phase_resid_rms_rad"] / (2.0 * math.pi * beacon_hz)

    print("CSV: {}".format(args.csv_path))
    print("Rows: {} total, {} after filter".format(len(rows), len(rows_f)))
    if cadence is not None:
        print("Cadence dt_update_s: n={n} mean={mean:.6f}s median={median:.6f}s std={std:.6f}s min={min:.6f}s max={max:.6f}s".format(**cadence))
    else:
        print("Cadence dt_update_s: insufficient data")

    qn = int(np.count_nonzero(quality))
    sn = int(np.count_nonzero(stable))
    seg_count = len(segs)
    seg_dwell_total = float(np.sum([x[2] for x in segs])) if segs else 0.0
    print("Quality rows: {} / {} ({:.1f}%) with coh>={:.3f}, snr>={:.1f} dB{}".format(
        qn, len(rows_f), 100.0 * qn / max(1, len(rows_f)), args.coh_min, args.snr_min_db,
        " + meas_quality_ok" if not args.ignore_meas_quality_ok else ""))
    print("Stable mask rows: {} (mode gate='{}'); stable segments: {} (total dwell {:.1f} s, min_dwell {:.1f} s)".format(
        sn, args.stable_mode, seg_count, seg_dwell_total, args.min_stable_dwell_s))

    print_drift_block("Global drift fit", drift_global)
    print_drift_block("Stable-only drift fit", drift_stable)

    if mode_stats:
        print("Mode-separated drift fits on quality rows:")
        for mode in ["OPEN", "ACQUIRE", "TRACK", "HOLDOVER"]:
            d = mode_stats.get(mode)
            if d is None:
                continue
            print("  {}: mean={:.6f} Hz std={:.6f} Hz slope={:.6e} Hz/s detrended_rms={:.6f} Hz (n={})".format(
                mode, d["mean"], d["std"], d["slope_hz_s"], d["detrended_rms"], d["n"]))

    print_piecewise_block("Piecewise slopes (quality mask)", pw_global)
    print_piecewise_block("Piecewise slopes (stable mask)", pw_stable)

    print("Phase concentration (all filtered): mean={:+.6f} rad R={:.6f}".format(mu_all, R_all))
    print("Phase concentration (quality): mean={:+.6f} rad R={:.6f}".format(mu_q, R_q))
    print("Phase concentration (stable segments): mean={:+.6f} rad R={:.6f}".format(mu_st, R_st))
    if R_st < 0.2 and sn >= 10:
        print("WARNING: low stable-segment R. This can reflect residual drift/wander even with consistent reference.")

    if win_R:
        Rs = np.asarray([w[1] for w in win_R], dtype=np.float64)
        print("Windowed R ({} s on stable segments): windows={} min={:.6f} median={:.6f} max={:.6f}".format(
            args.window_s, len(win_R), float(np.min(Rs)), float(np.median(Rs)), float(np.max(Rs))))
    else:
        print("Windowed R: insufficient stable data")

    if setpoint is not None:
        print("Setpoint candidate (stable segments): phase={:+.6f} rad at t={:.2f}s (R={:.6f}, n={})".format(
            setpoint["phase_setpoint_rad"], setpoint["center_s"], setpoint["R"], setpoint["n"]))
    else:
        print("Setpoint candidate: insufficient stable data")

    if jitter is not None:
        print("Timing jitter (stable segments, detrended phase): phase_rms={:.6e} rad (n={})".format(
            jitter["phase_resid_rms_rad"], jitter["n"]))
        if np.isfinite(timing_jitter_s):
            print("Equivalent timing jitter at beacon {:.3f} MHz: {:.3e} s ({:.3f} ps)".format(
                beacon_hz / 1e6, timing_jitter_s, timing_jitter_s * 1e12))
    else:
        print("Timing jitter estimate: insufficient stable data")

    maybe_plot(arr, stable_points_mask, win_R, drift_global, pw_global, pw_stable, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
