#!/usr/bin/env python3
"""Stream SDR data and plot the dominant-tone phase versus time.

This script is built to match the RTL-SDR wrapper exposed by:
https://github.com/AaronParsons/ugradio/blob/main/ugradio_code/src/sdr.py

Key points:
- It uses ``ugradio.sdr.SDR(...)`` directly.
- It consumes the underlying async ``stream(...)`` interface block-by-block.
- It exposes a reusable async generator for downstream processing.
- It exposes both wrapped and unwrapped phase traces.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
from ugradio import sdr as ugradio_sdr


@dataclass
class PhaseMeasurement:
    block_index: int
    unix_time_s: float
    elapsed_s: float
    wrapped_phase_rad: float
    unwrapped_phase_rad: float
    coherence: float
    tone_offset_hz: float
    rf_tone_hz: float
    iq_block: np.ndarray


@dataclass
class QueuedBlock:
    unix_time_s: float
    sample_start: int
    iq_block: np.ndarray


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Stream IQ blocks from an SDR and plot dominant-tone wrapped and unwrapped phase versus time."
    )
    parser.add_argument("--device-index", type=int, default=0, help="RTL-SDR device index.")
    parser.add_argument(
        "--center-freq",
        type=float,
        default=1.42e9,
        help="RF center frequency in Hz (matches ugradio.sdr.SDR default).",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=2.2e6,
        help="Sample rate in Hz (matches ugradio.sdr.SDR default).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.0,
        help="Tuner gain in dB (matches ugradio.sdr.SDR default).",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=16384,
        help="Samples per streaming block.",
    )
    parser.add_argument(
        "--nblocks",
        type=int,
        default=None,
        help="Number of streaming blocks to consume before exiting. Defaults to 128 unless --live-plot is set.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.25,
        help="Wait time after tuning before streaming starts.",
    )
    parser.add_argument(
        "--target-offset-hz",
        type=float,
        default=None,
        help="Known baseband tone offset in Hz. If omitted, the strongest tone is estimated from startup blocks.",
    )
    parser.add_argument(
        "--tone-estimate-blocks",
        type=int,
        default=8,
        help="Startup blocks to buffer for automatic tone detection when --target-offset-hz is omitted.",
    )
    parser.add_argument(
        "--dc-exclusion-hz",
        type=float,
        default=5e3,
        help="Ignore tones within this distance of DC during automatic detection.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=base_dir / "phase_vs_time.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=base_dir / "phase_vs_time.png",
        help="PNG output path.",
    )
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="Update the matplotlib figure while streaming.",
    )
    parser.add_argument(
        "--plot-update-seconds",
        type=float,
        default=20.0,
        help="Update cadence for live plotting, in seconds of sample time.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Keep the final plot window open after streaming completes.",
    )
    args = parser.parse_args()
    if args.nblocks is None and not args.live_plot:
        args.nblocks = 128
    return args


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
    return ugradio_sdr.SDR(
        device_index=args.device_index,
        direct=False,
        center_freq=float(args.center_freq),
        sample_rate=float(args.sample_rate),
        gain=float(args.gain),
    )


async def stream_iq_blocks(
    device: Any,
    nsamples: int,
    nblocks: int | None,
    queue: asyncio.Queue[QueuedBlock | None],
) -> None:
    count = 0
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
                    sample_start=sample_start,
                    iq_block=iq_block,
                )
            )
            sample_start += int(iq_block.size)
            count += 1
            if nblocks is not None and count >= int(nblocks):
                break
    except asyncio.CancelledError:
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        raise
    else:
        await queue.put(None)
    finally:
        try:
            await device.stop()
        except Exception:
            pass


async def iter_queued_blocks(
    queue: asyncio.Queue[QueuedBlock | None],
) -> AsyncIterator[QueuedBlock]:
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item


def estimate_tone_offset_hz(
    iq_blocks: np.ndarray,
    sample_rate_hz: float,
    dc_exclusion_hz: float,
) -> float:
    spectra = np.fft.fft(iq_blocks, axis=1)
    power = np.mean(np.abs(spectra) ** 2, axis=0)
    freqs = np.fft.fftfreq(iq_blocks.shape[1], d=1.0 / float(sample_rate_hz))

    mask = np.ones(freqs.shape, dtype=bool)
    if dc_exclusion_hz > 0:
        mask &= np.abs(freqs) >= float(dc_exclusion_hz)
    if not np.any(mask):
        mask[:] = True

    idx_candidates = np.flatnonzero(mask)
    peak_idx = int(idx_candidates[int(np.argmax(power[mask]))])
    return float(freqs[peak_idx])


def measure_phase(
    iq_block: np.ndarray,
    sample_rate_hz: float,
    tone_offset_hz: float,
    sample_start: int,
) -> tuple[float, float]:
    iq = np.asarray(iq_block, dtype=np.complex64).reshape(-1)
    iq = iq - np.mean(iq)

    n = float(sample_start) + np.arange(iq.size, dtype=np.float64)
    rotator = np.exp(
        -2j * np.pi * float(tone_offset_hz) * n / float(sample_rate_hz)
    ).astype(np.complex64)
    phasor = np.sum(iq * rotator)

    wrapped_phase = float(np.angle(phasor))
    coherence = float(np.abs(phasor) / (np.sum(np.abs(iq)) + 1e-12))
    return wrapped_phase, coherence


def unwrap_phase_step(
    wrapped_phase_rad: float,
    prev_wrapped_phase_rad: float | None,
    prev_unwrapped_phase_rad: float,
) -> tuple[float, float]:
    if prev_wrapped_phase_rad is None:
        return float(wrapped_phase_rad), float(wrapped_phase_rad)

    dphi = (float(wrapped_phase_rad) - float(prev_wrapped_phase_rad) + math.pi) % (2.0 * math.pi) - math.pi
    unwrapped_phase_rad = float(prev_unwrapped_phase_rad + dphi)
    return float(wrapped_phase_rad), unwrapped_phase_rad


def csv_writer(csv_path: Path) -> tuple[Any, csv.writer]:
    ensure_parent_dir(csv_path)
    fh = csv_path.open("w", newline="")
    writer = csv.writer(fh)
    writer.writerow(
        [
            "block_index",
            "unix_time_s",
            "sample_time_s",
            "phase_wrapped_rad",
            "phase_unwrapped_rad",
            "coherence",
            "tone_offset_hz",
            "rf_tone_hz",
        ]
    )
    return fh, writer


def write_measurement_row(writer: csv.writer, measurement: PhaseMeasurement) -> None:
    writer.writerow(
        [
            measurement.block_index,
            f"{measurement.unix_time_s:.9f}",
            f"{measurement.elapsed_s:.9f}",
            f"{measurement.wrapped_phase_rad:.9f}",
            f"{measurement.unwrapped_phase_rad:.9f}",
            f"{measurement.coherence:.9f}",
            f"{measurement.tone_offset_hz:.9f}",
            f"{measurement.rf_tone_hz:.9f}",
        ]
    )


class PhasePlotter:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                "matplotlib is not installed. Install it to render the phase plot."
            ) from exc

        self._plt = plt
        self._args = args
        self._tone_offset_hz: float | None = None

        if args.live_plot:
            plt.ion()

        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        self.wrapped_line, = self.axes[0].plot([], [], color="#005f73", linewidth=1.5)
        self.unwrapped_line, = self.axes[1].plot([], [], color="#ca6702", linewidth=1.2)

        self.axes[0].set_ylabel("Wrapped Phase (rad)")
        self.axes[0].set_ylim(-math.pi - 0.2, math.pi + 0.2)
        self.axes[0].grid(True, alpha=0.25)

        self.axes[1].set_xlabel("Sample Time (s)")
        self.axes[1].set_ylabel("Unwrapped Phase (rad)")
        self.axes[1].grid(True, alpha=0.25)

    def update(
        self,
        time_s: list[float],
        wrapped_phase_rad: list[float],
        unwrapped_phase_rad: list[float],
        tone_offset_hz: float,
    ) -> None:
        self._tone_offset_hz = float(tone_offset_hz)
        self.wrapped_line.set_data(time_s, wrapped_phase_rad)
        self.unwrapped_line.set_data(time_s, unwrapped_phase_rad)

        if time_s:
            x_max = max(time_s[-1], 1e-9)
            self.axes[0].set_xlim(0.0, x_max * 1.02)
            self.axes[1].set_xlim(0.0, x_max * 1.02)

        if unwrapped_phase_rad:
            y_min = float(np.nanmin(unwrapped_phase_rad))
            y_max = float(np.nanmax(unwrapped_phase_rad))
            if math.isclose(y_min, y_max):
                pad = 1.0
            else:
                pad = 0.05 * (y_max - y_min)
            self.axes[1].set_ylim(y_min - pad, y_max + pad)

        self.axes[0].set_title(
            "Dominant-tone phase vs time\n"
            f"center={self._args.center_freq:.3f} Hz, offset={tone_offset_hz:.3f} Hz"
        )
        if self._args.live_plot:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self._plt.pause(0.001)

    def save(self, plot_path: Path) -> None:
        ensure_parent_dir(plot_path)
        self.fig.tight_layout()
        self.fig.savefig(plot_path, dpi=160)

    def poll(self) -> None:
        if self._args.live_plot:
            self._plt.pause(0.001)

    def is_open(self) -> bool:
        if not self._args.live_plot:
            return True
        fignum_exists = getattr(self._plt, "fignum_exists", None)
        fig_number = getattr(self.fig, "number", None)
        if callable(fignum_exists) and fig_number is not None:
            try:
                return bool(fignum_exists(fig_number))
            except Exception:
                pass
        return True

    def close(self) -> None:
        if self._args.show:
            self._plt.ioff()
            self._plt.show()
        else:
            self._plt.close(self.fig)


async def iter_phase_measurements(
    source: AsyncIterator[QueuedBlock],
    args: argparse.Namespace,
) -> AsyncIterator[PhaseMeasurement]:
    tone_offset_hz = (
        None if args.target_offset_hz is None else float(args.target_offset_hz)
    )
    bootstrap_target = 0
    if tone_offset_hz is None:
        if args.nblocks is None:
            bootstrap_target = max(1, int(args.tone_estimate_blocks))
        else:
            bootstrap_target = max(1, min(int(args.tone_estimate_blocks), int(args.nblocks)))

    buffered: list[QueuedBlock] = []
    block_index = 0
    prev_wrapped_phase_rad: float | None = None
    prev_unwrapped_phase_rad = 0.0

    async for queued_block in source:
        unix_time_s = float(queued_block.unix_time_s)
        sample_start = int(queued_block.sample_start)
        iq_block = np.asarray(queued_block.iq_block, dtype=np.complex64).reshape(-1)

        if tone_offset_hz is None:
            buffered.append(
                QueuedBlock(
                    unix_time_s=unix_time_s,
                    sample_start=sample_start,
                    iq_block=iq_block,
                )
            )
            if len(buffered) < bootstrap_target:
                continue

            stacked = np.stack(
                [item.iq_block - np.mean(item.iq_block) for item in buffered],
                axis=0,
            )
            tone_offset_hz = estimate_tone_offset_hz(
                stacked,
                sample_rate_hz=float(args.sample_rate),
                dc_exclusion_hz=float(args.dc_exclusion_hz),
            )

            for item in buffered:
                wrapped_phase, coherence = measure_phase(
                    item.iq_block,
                    sample_rate_hz=float(args.sample_rate),
                    tone_offset_hz=float(tone_offset_hz),
                    sample_start=item.sample_start,
                )
                prev_wrapped_phase_rad, prev_unwrapped_phase_rad = unwrap_phase_step(
                    wrapped_phase,
                    prev_wrapped_phase_rad,
                    prev_unwrapped_phase_rad,
                )
                yield PhaseMeasurement(
                    block_index=block_index,
                    unix_time_s=float(item.unix_time_s),
                    elapsed_s=float(
                        (float(item.sample_start) + 0.5 * float(item.iq_block.size))
                        / float(args.sample_rate)
                    ),
                    wrapped_phase_rad=wrapped_phase,
                    unwrapped_phase_rad=prev_unwrapped_phase_rad,
                    coherence=coherence,
                    tone_offset_hz=float(tone_offset_hz),
                    rf_tone_hz=float(args.center_freq) + float(tone_offset_hz),
                    iq_block=item.iq_block,
                )
                block_index += 1
            buffered.clear()
            continue

        wrapped_phase, coherence = measure_phase(
            iq_block,
            sample_rate_hz=float(args.sample_rate),
            tone_offset_hz=float(tone_offset_hz),
            sample_start=sample_start,
        )
        prev_wrapped_phase_rad, prev_unwrapped_phase_rad = unwrap_phase_step(
            wrapped_phase,
            prev_wrapped_phase_rad,
            prev_unwrapped_phase_rad,
        )
        yield PhaseMeasurement(
            block_index=block_index,
            unix_time_s=float(unix_time_s),
            elapsed_s=float(
                (float(sample_start) + 0.5 * float(iq_block.size)) / float(args.sample_rate)
            ),
            wrapped_phase_rad=wrapped_phase,
            unwrapped_phase_rad=prev_unwrapped_phase_rad,
            coherence=coherence,
            tone_offset_hz=float(tone_offset_hz),
            rf_tone_hz=float(args.center_freq) + float(tone_offset_hz),
            iq_block=iq_block,
        )
        block_index += 1

    if tone_offset_hz is None and buffered:
        stacked = np.stack(
            [item.iq_block - np.mean(item.iq_block) for item in buffered],
            axis=0,
        )
        tone_offset_hz = estimate_tone_offset_hz(
            stacked,
            sample_rate_hz=float(args.sample_rate),
            dc_exclusion_hz=float(args.dc_exclusion_hz),
        )
        for item in buffered:
            wrapped_phase, coherence = measure_phase(
                item.iq_block,
                sample_rate_hz=float(args.sample_rate),
                tone_offset_hz=float(tone_offset_hz),
                sample_start=item.sample_start,
            )
            prev_wrapped_phase_rad, prev_unwrapped_phase_rad = unwrap_phase_step(
                wrapped_phase,
                prev_wrapped_phase_rad,
                prev_unwrapped_phase_rad,
            )
            yield PhaseMeasurement(
                block_index=block_index,
                unix_time_s=float(item.unix_time_s),
                elapsed_s=float(
                    (float(item.sample_start) + 0.5 * float(item.iq_block.size))
                    / float(args.sample_rate)
                ),
                wrapped_phase_rad=wrapped_phase,
                unwrapped_phase_rad=prev_unwrapped_phase_rad,
                coherence=coherence,
                tone_offset_hz=float(tone_offset_hz),
                rf_tone_hz=float(args.center_freq) + float(tone_offset_hz),
                iq_block=item.iq_block,
            )
            block_index += 1


def close_quietly(device: Any) -> None:
    try:
        device.close()
    except Exception:
        pass


async def run_capture(args: argparse.Namespace) -> int:
    if args.nsamples <= 0:
        print("--nsamples must be positive.", file=sys.stderr)
        return 2
    if args.nblocks is not None and args.nblocks <= 0:
        print("--nblocks must be positive when provided.", file=sys.stderr)
        return 2
    if args.tone_estimate_blocks <= 0:
        print("--tone-estimate-blocks must be positive.", file=sys.stderr)
        return 2
    if args.plot_update_seconds <= 0:
        print("--plot-update-seconds must be positive.", file=sys.stderr)
        return 2

    device = None
    csv_fh = None
    producer_task: asyncio.Task[None] | None = None
    previous_sigint: Any = None

    try:
        device = open_sdr(args)
        if args.settle_seconds > 0:
            await asyncio.sleep(float(args.settle_seconds))

        csv_fh, writer = csv_writer(args.csv_out)
        plotter = PhasePlotter(args)

        time_points: list[float] = []
        wrapped_phase_points: list[float] = []
        unwrapped_phase_points: list[float] = []
        coherence_points: list[float] = []
        last_measurement: PhaseMeasurement | None = None
        last_plot_time_s: float | None = None
        stopped_early = False
        stop_requested = False

        if args.live_plot:
            previous_sigint = signal.getsignal(signal.SIGINT)

            def _handle_sigint(_signum, _frame) -> None:
                nonlocal stop_requested
                stop_requested = True

            signal.signal(signal.SIGINT, _handle_sigint)
        else:
            previous_sigint = None

        queue: asyncio.Queue[QueuedBlock | None] = asyncio.Queue(
            maxsize=max(4, min(int(args.nblocks) if args.nblocks is not None else 128, 128))
        )
        producer_task = asyncio.create_task(
            stream_iq_blocks(device, args.nsamples, args.nblocks, queue)
        )

        async for measurement in iter_phase_measurements(iter_queued_blocks(queue), args):
            last_measurement = measurement
            time_points.append(measurement.elapsed_s)
            wrapped_phase_points.append(measurement.wrapped_phase_rad)
            unwrapped_phase_points.append(measurement.unwrapped_phase_rad)
            coherence_points.append(measurement.coherence)

            write_measurement_row(writer, measurement)
            if args.live_plot and (
                last_plot_time_s is None
                or (measurement.elapsed_s - last_plot_time_s) >= float(args.plot_update_seconds)
            ):
                plotter.update(
                    time_points,
                    wrapped_phase_points,
                    unwrapped_phase_points,
                    measurement.tone_offset_hz,
                )
                last_plot_time_s = float(measurement.elapsed_s)

            if args.live_plot and (measurement.block_index + 1) % 16 == 0:
                plotter.poll()
                if not plotter.is_open():
                    stopped_early = True
                    break

            if stop_requested:
                stopped_early = True
                break

            if (measurement.block_index + 1) % 16 == 0:
                await asyncio.sleep(0)

        if producer_task is not None:
            if stopped_early and not producer_task.done():
                producer_task.cancel()
            try:
                await producer_task
            except BaseException:
                pass
            producer_task = None

        if last_measurement is None:
            raise RuntimeError("No SDR data blocks were received.")

        csv_fh.flush()
        plotter.update(
            time_points,
            wrapped_phase_points,
            unwrapped_phase_points,
            last_measurement.tone_offset_hz,
        )
        plotter.save(args.plot_out)
        plotter.close()

        print("Capture backend: ugradio.sdr.SDR")
        print(f"Dominant baseband tone: {last_measurement.tone_offset_hz:.3f} Hz")
        print(f"Estimated RF tone: {last_measurement.rf_tone_hz:.3f} Hz")
        print(f"Median coherence: {float(np.median(coherence_points)):.6f}")
        print(f"CSV written to: {args.csv_out}")
        print(f"Plot written to: {args.plot_out}")
        return 0
    except Exception as exc:
        if producer_task is not None:
            producer_task.cancel()
            try:
                await producer_task
            except BaseException:
                pass
        print(f"signalTest failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.live_plot and previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        if csv_fh is not None:
            try:
                csv_fh.close()
            except Exception:
                pass
        if device is not None:
            close_quietly(device)


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run_capture(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
