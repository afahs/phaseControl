#!/usr/bin/env python3
"""Capture raw IQ once and replay through two phase-analysis loops.

Pipelines:
1) signalTest.plot_sdr_phase loop (dominant-tone rotator + unwrap)
2) beacon open-loop analysis path (acquire snapshot per update)
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import beacon_vcxo_pll_controller as base
from signalTest import plot_sdr_phase as phase_loop


@dataclass
class ReplaySeries:
    elapsed_s: np.ndarray
    wrapped_phase_rad: np.ndarray
    unwrapped_phase_rad: np.ndarray
    coherence: np.ndarray
    freq_err_hz: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture SDR IQ once and compare phase behavior between plot_sdr_phase and open-loop analysis."
    )
    parser.add_argument("--sample_rate", type=float, default=2.4e6)
    parser.add_argument("--center_freq", type=float, default=99.5e6)
    parser.add_argument("--beacon_freq", type=float, default=100e6)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--nsamp", type=int, default=262144)
    parser.add_argument("--capture_seconds", type=float, default=15.0)
    parser.add_argument("--nblocks", type=int, default=None)
    parser.add_argument("--settle_seconds", type=float, default=0.25)
    parser.add_argument("--target_offset_hz", type=float, default=None)
    parser.add_argument("--tone_estimate_blocks", type=int, default=8)
    parser.add_argument("--dc_exclusion_hz", type=float, default=5e3)
    parser.add_argument("--fft_search_span_hz", type=float, default=200e3)
    parser.add_argument(
        "--iq_npz",
        type=Path,
        default=None,
        help="Optional previously captured IQ npz. If omitted, capture from SDR now.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help="Output prefix. Defaults to beaconLogs/iq_replay_compare_<timestamp>.",
    )
    return parser.parse_args()


def timestamp_for_filename() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def infer_nblocks(args: argparse.Namespace) -> int:
    if args.nblocks is not None:
        return max(1, int(args.nblocks))
    if args.capture_seconds <= 0:
        return 1
    block_s = float(args.nsamp) / float(args.sample_rate)
    return max(1, int(math.ceil(float(args.capture_seconds) / block_s)))


def to_complex_iq(raw_iq: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw_iq)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(
            f"Expected raw IQ shape (nblocks, nsamp, 2), got {arr.shape!r}"
        )
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        x = arr.astype(np.float32) - 128.0
    else:
        x = arr.astype(np.float32)
    return (x[..., 0] + 1j * x[..., 1]).astype(np.complex64)


def capture_raw_iq(args: argparse.Namespace, nblocks: int) -> np.ndarray:
    sdr = None
    try:
        sdr = base.sdr_open(args.sample_rate, args.center_freq, args.gain)
        if args.settle_seconds > 0:
            time.sleep(float(args.settle_seconds))
        raw = sdr.capture_data(nsamples=int(args.nsamp), nblocks=int(nblocks))
        raw_iq = np.asarray(raw)
        if raw_iq.ndim != 3 or raw_iq.shape[-1] != 2:
            raise RuntimeError(
                f"capture_data returned unexpected shape {raw_iq.shape!r}"
            )
        return raw_iq
    finally:
        base.sdr_close(sdr)


def save_iq_npz(path: Path, raw_iq: np.ndarray, args: argparse.Namespace) -> None:
    ensure_parent_dir(path)
    np.savez_compressed(
        path,
        raw_iq=raw_iq,
        sample_rate=float(args.sample_rate),
        center_freq=float(args.center_freq),
        beacon_freq=float(args.beacon_freq),
        gain=float(args.gain),
        nsamp=int(args.nsamp),
    )


def load_iq_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "raw_iq" not in data:
            raise RuntimeError(f"{path} does not contain 'raw_iq'")
        return np.asarray(data["raw_iq"])


def replay_plot_phase_loop(
    iq_blocks: np.ndarray,
    args: argparse.Namespace,
) -> tuple[ReplaySeries, float]:
    nblocks = int(iq_blocks.shape[0])
    nsamp = int(iq_blocks.shape[1])
    fs = float(args.sample_rate)
    tone_offset_hz = (
        None if args.target_offset_hz is None else float(args.target_offset_hz)
    )
    if tone_offset_hz is None:
        bootstrap = max(1, min(int(args.tone_estimate_blocks), nblocks))
        stacked = iq_blocks[:bootstrap] - np.mean(iq_blocks[:bootstrap], axis=1, keepdims=True)
        tone_offset_hz = float(
            phase_loop.estimate_tone_offset_hz(
                stacked,
                sample_rate_hz=fs,
                dc_exclusion_hz=float(args.dc_exclusion_hz),
            )
        )

    wrapped = np.zeros(nblocks, dtype=np.float64)
    unwrapped = np.zeros(nblocks, dtype=np.float64)
    coherence = np.zeros(nblocks, dtype=np.float64)
    freq_err = np.zeros(nblocks, dtype=np.float64)
    elapsed = np.zeros(nblocks, dtype=np.float64)

    prev_wrapped: float | None = None
    prev_unwrapped = 0.0

    for i in range(nblocks):
        sample_start = i * nsamp
        elapsed[i] = (sample_start + 0.5 * nsamp) / fs
        wrapped_phase, coh = phase_loop.measure_phase(
            iq_blocks[i],
            sample_rate_hz=fs,
            tone_offset_hz=float(tone_offset_hz),
            sample_start=sample_start,
        )
        prev_wrapped, prev_unwrapped = phase_loop.unwrap_phase_step(
            wrapped_phase,
            prev_wrapped,
            prev_unwrapped,
        )
        wrapped[i] = wrapped_phase
        unwrapped[i] = prev_unwrapped
        coherence[i] = coh
        freq_err[i] = 0.0

    return (
        ReplaySeries(
            elapsed_s=elapsed,
            wrapped_phase_rad=wrapped,
            unwrapped_phase_rad=unwrapped,
            coherence=coherence,
            freq_err_hz=freq_err,
        ),
        float(tone_offset_hz),
    )


def replay_open_loop_analysis(
    iq_blocks: np.ndarray,
    args: argparse.Namespace,
) -> ReplaySeries:
    nblocks = int(iq_blocks.shape[0])
    nsamp = int(iq_blocks.shape[1])
    fs = float(args.sample_rate)
    mcfg = base.MeasureConfig(
        fs=fs,
        beacon_hz=float(args.beacon_freq),
        center_hz=float(args.center_freq),
        fft_search_span_hz=float(args.fft_search_span_hz),
    )
    phase_hist = base.PhaseHistory()

    wrapped = np.zeros(nblocks, dtype=np.float64)
    unwrapped = np.zeros(nblocks, dtype=np.float64)
    coherence = np.zeros(nblocks, dtype=np.float64)
    freq_err = np.zeros(nblocks, dtype=np.float64)
    elapsed = np.zeros(nblocks, dtype=np.float64)

    for i in range(nblocks):
        elapsed[i] = (i * nsamp + 0.5 * nsamp) / fs
        t_now = float((i + 1) * nsamp) / fs
        meas = base.build_update_measurement_from_acquire(
            [iq_blocks[i]],
            mcfg,
            phase_hist,
            t_now,
        )
        wrapped[i] = float(meas.phase_meas_rad)
        unwrapped[i] = float(meas.phase_unwrapped_rad)
        coherence[i] = float(meas.coh)
        freq_err[i] = float(meas.freq_err_block_hz)

    return ReplaySeries(
        elapsed_s=elapsed,
        wrapped_phase_rad=wrapped,
        unwrapped_phase_rad=unwrapped,
        coherence=coherence,
        freq_err_hz=freq_err,
    )


def write_replay_csv(path: Path, series: ReplaySeries) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "elapsed_s",
                "wrapped_phase_rad",
                "unwrapped_phase_rad",
                "coherence",
                "freq_err_hz",
            ]
        )
        for i in range(series.elapsed_s.size):
            writer.writerow(
                [
                    f"{series.elapsed_s[i]:.9f}",
                    f"{series.wrapped_phase_rad[i]:.9f}",
                    f"{series.unwrapped_phase_rad[i]:.9f}",
                    f"{series.coherence[i]:.9f}",
                    f"{series.freq_err_hz[i]:.9f}",
                ]
            )


def wrap_to_pi_vec(x: np.ndarray) -> np.ndarray:
    y = (x + np.pi) % (2.0 * np.pi) - np.pi
    y[y <= -np.pi] += 2.0 * np.pi
    return y


def circular_resultant_length(phases: np.ndarray) -> float:
    if phases.size == 0:
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * phases))))


def make_phase_plot(
    path: Path,
    plot_series: ReplaySeries,
    open_series: ReplaySeries,
    tone_offset_hz: float,
) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent_dir(path)
    p = plot_series
    o = open_series

    n = min(p.elapsed_s.size, o.elapsed_s.size)
    wrapped_diff = wrap_to_pi_vec(o.wrapped_phase_rad[:n] - p.wrapped_phase_rad[:n])
    wrapped_rms = float(np.sqrt(np.mean(wrapped_diff ** 2)))

    pu = p.unwrapped_phase_rad[:n] - p.unwrapped_phase_rad[0]
    ou = o.unwrapped_phase_rad[:n] - o.unwrapped_phase_rad[0]
    unwrapped_rms = float(np.sqrt(np.mean((ou - pu) ** 2)))

    p_r = circular_resultant_length(p.wrapped_phase_rad)
    o_r = circular_resultant_length(o.wrapped_phase_rad)
    p_coh = float(np.median(p.coherence))
    o_coh = float(np.median(o.coherence))

    fig, axs = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True)
    axs[0].plot(
        p.elapsed_s, p.wrapped_phase_rad, linewidth=1.25, color="#005f73", label="plot_sdr_phase loop"
    )
    axs[0].plot(
        o.elapsed_s, o.wrapped_phase_rad, linewidth=1.25, color="#ca6702", label="open-loop analysis loop"
    )
    axs[0].set_ylabel("Wrapped Phase (rad)")
    axs[0].set_ylim(-math.pi - 0.2, math.pi + 0.2)
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(loc="upper right")
    axs[0].set_title(
        "Replay Comparison on Identical IQ Blocks\n"
        f"tone_offset={tone_offset_hz:.3f} Hz"
    )

    axs[1].plot(
        p.elapsed_s,
        p.unwrapped_phase_rad - p.unwrapped_phase_rad[0],
        linewidth=1.25,
        color="#005f73",
        label="plot_sdr_phase loop",
    )
    axs[1].plot(
        o.elapsed_s,
        o.unwrapped_phase_rad - o.unwrapped_phase_rad[0],
        linewidth=1.25,
        color="#ca6702",
        label="open-loop analysis loop",
    )
    axs[1].set_xlabel("Sample Time (s)")
    axs[1].set_ylabel("Unwrapped Phase (rad)")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(loc="upper left")

    note = (
        f"wrapped RMS diff={wrapped_rms:.3f} rad | "
        f"unwrapped RMS diff={unwrapped_rms:.3f} rad | "
        f"R: plot={p_r:.3f}, open={o_r:.3f} | "
        f"median coh: plot={p_coh:.3f}, open={o_coh:.3f}"
    )
    fig.text(0.01, 0.01, note, fontsize=10)
    plt.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=160, bbox_inches="tight")

    return {
        "wrapped_rms_diff_rad": wrapped_rms,
        "unwrapped_rms_diff_rad": unwrapped_rms,
        "plot_wrapped_R": p_r,
        "open_wrapped_R": o_r,
        "plot_median_coh": p_coh,
        "open_median_coh": o_coh,
    }


def resolve_prefix(args: argparse.Namespace) -> Path:
    if args.out_prefix is not None:
        return Path(args.out_prefix)
    return Path("beaconLogs") / f"iq_replay_compare_{timestamp_for_filename()}"


def main() -> int:
    args = parse_args()
    prefix = resolve_prefix(args)

    if args.iq_npz is not None:
        iq_npz = Path(args.iq_npz)
        raw_iq = load_iq_npz(iq_npz)
        print(f"Loaded IQ from: {iq_npz}")
    else:
        nblocks = infer_nblocks(args)
        raw_iq = capture_raw_iq(args, nblocks)
        iq_npz = Path(str(prefix) + "_iq.npz")
        save_iq_npz(iq_npz, raw_iq, args)
        print(f"Captured IQ blocks: {raw_iq.shape[0]} x {raw_iq.shape[1]}")
        print(f"Saved IQ to: {iq_npz}")

    iq_blocks = to_complex_iq(raw_iq)

    plot_series, tone_offset_hz = replay_plot_phase_loop(iq_blocks, args)
    open_series = replay_open_loop_analysis(iq_blocks, args)

    plot_csv = Path(str(prefix) + "_plot_loop.csv")
    open_csv = Path(str(prefix) + "_open_loop_loop.csv")
    phase_png = Path(str(prefix) + "_phase_compare.png")
    write_replay_csv(plot_csv, plot_series)
    write_replay_csv(open_csv, open_series)
    metrics = make_phase_plot(phase_png, plot_series, open_series, tone_offset_hz)

    print(f"Plot-loop replay CSV: {plot_csv}")
    print(f"Open-loop replay CSV: {open_csv}")
    print(f"Comparison plot: {phase_png}")
    print(f"wrapped RMS diff (rad): {metrics['wrapped_rms_diff_rad']:.6f}")
    print(f"unwrapped RMS diff (rad): {metrics['unwrapped_rms_diff_rad']:.6f}")
    print(f"wrapped R (plot/open): {metrics['plot_wrapped_R']:.6f} / {metrics['open_wrapped_R']:.6f}")
    print(f"median coherence (plot/open): {metrics['plot_median_coh']:.6f} / {metrics['open_median_coh']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
