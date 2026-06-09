#!/usr/bin/env python3
"""Plot wrapped/raw-unwrapped/detrended phase PSD from a cw_vcxo CSV log."""

from __future__ import annotations

import argparse
import csv
import os
import json
import math
from pathlib import Path

import matplotlib

#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_phase_columns(csv_path: Path) -> dict[str, np.ndarray]:
    cols = ("t_rel_s", "phase_rad", "unwrapped_phase_rad")
    data: dict[str, list[float]] = {c: [] for c in cols}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = []
            for col in cols:
                try:
                    values.append(float(row[col]))
                except (KeyError, ValueError):
                    values.append(math.nan)
            if all(math.isfinite(v) for v in values):
                for col, value in zip(cols, values):
                    data[col].append(value)

    return {col: np.asarray(values, dtype=float) for col, values in data.items()}


def periodogram(x: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Hann-window periodogram."""
    x = np.asarray(x, dtype=float)
    #x = x - np.nanmean(x)
    x = np.exp(1j*x)
    win = np.hanning(len(x))
    spec = np.fft.fft(x * win)
    freq = np.fft.fftfreq(len(x), d=1.0 / fs_hz)
    psd = np.abs(spec) ** 2 / (fs_hz * np.sum(win**2))
    if len(psd) > 2:
        psd[1:-1] *= 2.0
    return freq, psd


def log_bin_median(
    freq: np.ndarray,
    psd: np.ndarray,
    bins_per_decade: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (freq > 0) & np.isfinite(freq) & np.isfinite(psd) & (psd > 0)
    freq = freq[mask]
    psd = psd[mask]
    edges = np.logspace(
        np.log10(freq.min()),
        np.log10(freq.max()),
        int(np.ceil(np.log10(freq.max() / freq.min()) * bins_per_decade)) + 1,
    )

    bin_freq = []
    bin_psd = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freq >= lo) & (freq < hi)
        if np.any(m):
            bin_freq.append(math.sqrt(lo * hi))
            bin_psd.append(float(np.median(psd[m])))
    return np.asarray(bin_freq), np.asarray(bin_psd)


def strongest_bins(
    freq: np.ndarray,
    psd: np.ndarray,
    fmin_hz: float,
    fmax_hz: float,
    count: int = 8,
) -> list[dict[str, float]]:
    mask = (freq >= fmin_hz) & (freq <= fmax_hz) & np.isfinite(psd)
    ff = freq[mask]
    pp = psd[mask]
    if len(pp) == 0:
        return []
    order = np.argsort(pp)[-count:][::-1]
    return [{"freq_hz": float(ff[i]), "psd": float(pp[i])} for i in order]


def plot_phase_psd(
    csv_path: Path,
    out_png: Path,
    out_json: Path | None,
    title: str,
) -> None:
    data = read_phase_columns(csv_path)
    t = data["t_rel_s"] - data["t_rel_s"][0]
    phase = data["phase_rad"]
    unwrapped = data["unwrapped_phase_rad"]

    fs_hz = 1.0 / float(np.median(np.diff(t)))
    slope_rad_s, intercept = np.polyfit(t, unwrapped, 1)
    slope_hz = slope_rad_s / (2.0 * np.pi)
    detrended = unwrapped - (slope_rad_s * t + intercept)

    series = [
        ("Wrapped phase PSD", phase, "tab:blue"),
        ("Raw unwrapped phase PSD", unwrapped, "tab:orange"),
        ("Linear-detrended unwrapped phase PSD", detrended, "tab:green"),
    ]

    red_reference_hz = [1.0 / 0.6, 1.0 / 0.3, 1.0 / 0.15]
    black_reference_hz = abs(slope_hz)
    metrics: dict[str, object] = {
        "csv": str(csv_path),
        "rows": int(len(t)),
        "fs_hz": fs_hz,
        "linear_phase_slope_hz": slope_hz,
        "series": {},
    }

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True, constrained_layout=True)
    for ax, (label, values, color) in zip(axes, series):
        freq, psd = periodogram(values, fs_hz)
        smooth_f, smooth_psd = log_bin_median(freq, psd)
        mask = freq > 0

        ax.loglog(freq[mask], psd[mask], color=color, alpha=0.12, lw=0.35, label="full-record periodogram")
        ax.loglog(smooth_f, smooth_psd, color=color, lw=1.8, label="log-bin median")

        for ref_hz in red_reference_hz:
            ax.axvline(ref_hz, color="red", ls="--", lw=0.9, alpha=0.65)
        ax.axvline(
            black_reference_hz,
            color="black",
            lw=1.4,
            alpha=0.75,
            label=f"|phase slope| {black_reference_hz:.2f} Hz",
        )

        ax.set_title(label)
        ax.set_ylabel("rad^2/Hz")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(loc="upper right")
        metrics["series"][label] = strongest_bins(freq, psd, 0.01, fs_hz / 2.0)

    axes[-1].set_xlabel("Fourier frequency (Hz)")
    fig.suptitle(title, fontsize=18)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    if out_json is not None:
        out_json.write_text(json.dumps(metrics, indent=2))

def plot_phase_psd_simple(
    csv_path: Path,
    out_png: Path,
    out_json: Path | None,
    title: str,
) -> None:
    filename = 'tmp.npz'
    if not os.path.exists(filename):
        data = read_phase_columns(csv_path)
        np.savez(filename, **data)
    else:
        data = dict(np.load(filename))

    t = data["t_rel_s"] - data["t_rel_s"][0]
    phase = data["phase_rad"]
    dt =  float(np.median(np.diff(t)))
    df = 1.0/dt
    
    fig, axes = plt.subplots(1, 1, figsize=(16, 14), sharex=True, constrained_layout=True)
    
    x = np.exp(1j*phase)
    x -= np.mean(x)
    #fmax = 500.
    fmax = 75
    osc = np.exp(-2 * np.pi * 1j * t * fmax)
    coeff = np.mean(x * osc)
    x_detrend = x - coeff * osc.conj()
    x.shape = x_detrend.shape = (-1 , 1000)
    win = np.hanning(x.shape[-1])
    win.shape = (1,-1)
    spec = np.fft.fft(x * win)
    freq = np.fft.fftfreq(x.shape[-1], d=dt)
    #psd = np.mean(np.abs(spec) ** 2,axis = 0)
    psd = np.abs(spec) ** 2
        
    spec_detrend = np.fft.fft(x_detrend * win)
    #psd_detrend = np.mean(np.abs(spec_detrend) ** 2,axis = 0)
    psd_detrend = np.abs(spec_detrend) ** 2
    axes.semilogy(freq,psd[:3].T)
    #axes.semilogy(freq,psd_detrend[:3].T,alpha=0.5)
    #axes[0].plot(np.real(x[0]))
    #axes[0].plot(np.real(x_detrend[0]))
    #axes[0].plot(np.real(coeff * osc.conj()[:1000]))
    #axes[1].plot(np.imag(x[0]))
    #axes[1].plot(np.imag(x_detrend[0]))
    #axes[1].plot(np.imag(coeff * osc.conj()[:1000]))
    

    axes.set_xlabel("Fourier frequency (Hz)")
    #axes[1].set_xlabel("Fourier frequency (Hz)")
    fig.suptitle(title, fontsize=18)
    #out_png.parent.mkdir(parents=True, exist_ok=True)
    #fig.savefig(out_png, dpi=160)
    #plt.close(fig)
    plt.show()

    
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("first_stage_demo/logs/pseudo_open_loop_one_freq_correction_code2050_50mhz_5min_20260513.csv"),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("first_stage_demo/logs/cleaned/pseudo_open_loop_code2050_50mhz_wrapped_unwrapped_detrended_phase_psd_simple.png"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("first_stage_demo/logs/cleaned/pseudo_open_loop_code2050_50mhz_wrapped_unwrapped_detrended_phase_psd_metrics_20260513.json"),
    )
    parser.add_argument(
        "--title",
        default="Pseudo-open-loop phase PSD comparison: wrapped vs raw unwrapped vs detrended unwrapped",
    )
    args = parser.parse_args()

    plot_phase_psd_simple(args.csv, args.out_png, args.out_json, args.title)


if __name__ == "__main__":
    main()
