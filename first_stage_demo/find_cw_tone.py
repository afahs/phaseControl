#!/usr/bin/env python3
"""Locate a CW tone near an RF frequency using the same ugradio SDR path."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

import cw_vcxo_discipline as base


def db10(x: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(x, 1.0e-24))


def scan_centers(target_hz: float, span_hz: float, sample_rate_hz: float, overlap: float) -> list[float]:
    usable_step = sample_rate_hz * max(0.1, min(0.95, 1.0 - float(overlap)))
    start = target_hz - 0.5 * span_hz
    stop = target_hz + 0.5 * span_hz
    if span_hz <= sample_rate_hz:
        return [target_hz]

    centers: list[float] = []
    c = start + 0.5 * sample_rate_hz
    while c < stop - 0.5 * sample_rate_hz:
        centers.append(c)
        c += usable_step
    centers.append(stop - 0.5 * sample_rate_hz)
    return sorted(set(round(x, 3) for x in centers))


def scan_one_center(args: argparse.Namespace, center_hz: float) -> dict[str, float]:
    sdr = base.UgradioSdrSource(
        sample_rate_hz=args.sample_rate_hz,
        center_freq_hz=center_hz,
        gain=args.gain,
        device_index=args.device_index,
        queue_blocks=args.ugradio_queue_blocks,
    )
    try:
        for _ in range(max(0, args.warmup_blocks)):
            sdr.read_block(args.block_size)

        window = np.hanning(args.block_size).astype(np.float64)
        window_power = float(np.sum(window * window))
        psd_accum = np.zeros(args.block_size, dtype=np.float64)
        rms_accum = 0.0
        for _ in range(args.blocks):
            iq = sdr.read_block(args.block_size).astype(np.complex128, copy=False)
            iq = iq - np.mean(iq, dtype=np.complex128)
            rms_accum += float(np.mean(np.abs(iq) ** 2))
            spec = np.fft.fftshift(np.fft.fft(iq * window))
            psd_accum += (np.abs(spec) ** 2) / max(window_power, 1.0e-12)

        psd = psd_accum / max(1, args.blocks)
        freqs = np.fft.fftshift(np.fft.fftfreq(args.block_size, d=1.0 / args.sample_rate_hz))

        mask = np.ones_like(psd, dtype=bool)
        mask &= np.abs(freqs) >= args.dc_exclusion_hz
        edge_limit = 0.5 * args.sample_rate_hz - args.edge_exclusion_hz
        mask &= np.abs(freqs) <= edge_limit
        if not np.any(mask):
            raise RuntimeError("No FFT bins remain after DC/edge exclusions.")

        masked_indices = np.flatnonzero(mask)
        peak_index = int(masked_indices[int(np.argmax(psd[mask]))])
        noise_floor = float(np.median(psd[mask]))
        peak_power = float(psd[peak_index])
        peak_bb_hz = float(freqs[peak_index])
        return {
            "center_hz": float(center_hz),
            "peak_rf_hz": float(center_hz + peak_bb_hz),
            "peak_bb_hz": peak_bb_hz,
            "peak_power_db": float(db10(peak_power)),
            "noise_floor_db": float(db10(noise_floor)),
            "prominence_db": float(db10(peak_power / max(noise_floor, 1.0e-24))),
            "mean_power_db": float(db10(rms_accum / max(1, args.blocks))),
        }
    finally:
        sdr.close()


def write_csv(path: str, rows: list[dict[str, float]]) -> None:
    base.ensure_parent_dir(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-rf", type=float, default=50_000_000.0)
    parser.add_argument("--span-hz", type=float, default=2_000_000.0)
    parser.add_argument("--sample-rate", dest="sample_rate_hz", type=float, default=1_024_000.0)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--blocks", type=int, default=48)
    parser.add_argument("--warmup-blocks", type=int, default=8)
    parser.add_argument("--gain", default="0")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--ugradio-queue-blocks", type=int, default=32)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--dc-exclusion-hz", type=float, default=5_000.0)
    parser.add_argument("--edge-exclusion-hz", type=float, default=40_000.0)
    parser.add_argument("--min-prominence-db", type=float, default=12.0)
    parser.add_argument("--out-csv", default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    centers = scan_centers(args.target_rf, args.span_hz, args.sample_rate_hz, args.overlap)
    print(
        f"Scanning {len(centers)} center(s) around {args.target_rf:.3f} Hz "
        f"over {args.span_hz:.3f} Hz span",
        flush=True,
    )

    rows: list[dict[str, float]] = []
    for index, center_hz in enumerate(centers, start=1):
        row = scan_one_center(args, center_hz)
        rows.append(row)
        print(
            f"[{index}/{len(centers)}] center={row['center_hz']:.3f} Hz "
            f"peak_rf={row['peak_rf_hz']:.3f} Hz bb={row['peak_bb_hz']:.3f} Hz "
            f"prom={row['prominence_db']:.1f} dB",
            flush=True,
        )

    best = max(rows, key=lambda r: r["prominence_db"])
    if args.out_csv is not None:
        write_csv(args.out_csv, rows)

    print(
        "BEST "
        f"peak_rf_hz={best['peak_rf_hz']:.6f} "
        f"center_hz={best['center_hz']:.6f} "
        f"peak_bb_hz={best['peak_bb_hz']:.6f} "
        f"prominence_db={best['prominence_db']:.3f} "
        f"peak_power_db={best['peak_power_db']:.3f}",
        flush=True,
    )
    if best["prominence_db"] < args.min_prominence_db:
        print(
            f"WARNING: best peak prominence is below {args.min_prominence_db:.1f} dB; "
            "the detected RF may not be the intended CW tone.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
