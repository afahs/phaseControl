#!/usr/bin/env python3
"""Plot full and 1/6-second wrapped/unwrapped phase tracks from a saved log."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BINARY_MAGIC = b"CWVXCOBIN1\n"
# These are the measured physical phase fields. NCO residual/correction fields
# are intentionally not used by this standard comparison plot.
PHASE_COLUMNS = ("t_rel_s", "phase_rad", "unwrapped_phase_rad")
OPTIONAL_COLUMNS = ("coherence", "amp", "dac_code_cmd")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_csv_log(path: Path) -> dict[str, np.ndarray]:
    columns = PHASE_COLUMNS + OPTIONAL_COLUMNS
    data: dict[str, list[float]] = {name: [] for name in columns}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name in columns:
                data[name].append(safe_float(row.get(name)))
    return {name: np.asarray(values, dtype=np.float64) for name, values in data.items()}


def read_binary_log(path: Path) -> dict[str, np.ndarray]:
    columns = PHASE_COLUMNS + OPTIONAL_COLUMNS
    data: dict[str, list[float]] = {name: [] for name in columns}
    with path.open("rb") as f:
        magic = f.readline()
        if magic != BINARY_MAGIC:
            raise ValueError(f"{path} is not a CW VXCO binary log.")
        header_len_raw = f.read(4)
        if len(header_len_raw) != 4:
            raise ValueError(f"{path} has a truncated binary header.")
        header_len = struct.unpack("<I", header_len_raw)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        numeric_fields = list(header["numeric_fields"])
        string_fields = list(header["string_fields"])
        numeric_struct = struct.Struct("<" + ("d" * len(numeric_fields)))
        field_index = {name: numeric_fields.index(name) for name in columns if name in numeric_fields}

        while True:
            record_len_raw = f.read(4)
            if not record_len_raw:
                break
            if len(record_len_raw) != 4:
                break
            record_len = struct.unpack("<I", record_len_raw)[0]
            payload = f.read(record_len)
            if len(payload) != record_len:
                break
            values = numeric_struct.unpack_from(payload, 0)
            for name in columns:
                index = field_index.get(name)
                data[name].append(values[index] if index is not None else math.nan)

            # Advance over string fields as a sanity check for future format edits.
            offset = numeric_struct.size
            for _ in string_fields:
                if offset + 2 > len(payload):
                    break
                size = struct.unpack_from("<H", payload, offset)[0]
                offset += 2 + size

    return {name: np.asarray(values, dtype=np.float64) for name, values in data.items()}


def read_log(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        magic = f.readline()
    if magic == BINARY_MAGIC:
        return read_binary_log(path)
    return read_csv_log(path)


def sanitize_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")


def finite_phase_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    t = data["t_rel_s"]
    wrapped = data["phase_rad"]
    unwrapped = data["unwrapped_phase_rad"]
    if not np.isfinite(unwrapped).any():
        unwrapped = np.unwrap(wrapped)

    mask = np.isfinite(t) & np.isfinite(wrapped) & np.isfinite(unwrapped)
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError("Log does not contain at least two finite phase samples.")

    out = {
        "t_rel_s": t[mask],
        "phase_rad": wrapped[mask],
        "unwrapped_phase_rad": unwrapped[mask],
    }
    for name in OPTIONAL_COLUMNS:
        values = data.get(name)
        out[name] = values[mask] if values is not None and values.size == t.size else np.full(out["t_rel_s"].shape, math.nan)
    return out


def default_zoom_start(t: np.ndarray, zoom_duration_s: float) -> float:
    t_min = float(np.nanmin(t))
    t_max = float(np.nanmax(t))
    if t_min <= 30.0 and 30.0 + zoom_duration_s <= t_max:
        return 30.0
    return max(t_min, 0.5 * (t_min + t_max - zoom_duration_s))


def plot_phase_track(
    data: dict[str, np.ndarray],
    out_png: Path,
    title: str,
    mask: np.ndarray,
) -> None:
    t = data["t_rel_s"][mask]
    wrapped = data["phase_rad"][mask]
    unwrapped = data["unwrapped_phase_rad"][mask]
    if t.size < 2:
        raise ValueError(f"Selected window for {out_png} has fewer than two samples.")

    unwrapped_rel = unwrapped - unwrapped[0]
    coherence = data["coherence"][mask]
    amp = data["amp"][mask]
    dac = data["dac_code_cmd"][mask]

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(t, wrapped, lw=0.8, color="#1f77b4")
    axes[0].set_ylabel("Wrapped phase (rad)")
    axes[0].set_ylim(-math.pi * 1.08, math.pi * 1.08)
    axes[0].set_title(title)

    axes[1].plot(t, unwrapped_rel, lw=0.8, color="#d62728")
    axes[1].set_ylabel("Unwrapped phase rel. start (rad)")
    axes[1].set_xlabel("Time (s)")

    summary_parts = [f"N={t.size}", f"span={t[-1] - t[0]:.6g}s"]
    if np.isfinite(coherence).any():
        summary_parts.append(f"coh mean={np.nanmean(coherence):.3f}")
    if np.isfinite(amp).any():
        summary_parts.append(f"amp mean={np.nanmean(amp):.3f}")
    if np.isfinite(dac).any():
        summary_parts.append(f"DAC {np.nanmin(dac):.0f}-{np.nanmax(dac):.0f}")
    fig.text(0.01, 0.01, ", ".join(summary_parts), ha="left", va="bottom", fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 2:
        return math.nan
    xm = x[mask]
    ym = y[mask]
    xc = xm - float(np.mean(xm))
    den = float(np.sum(xc * xc))
    if den <= 0.0:
        return math.nan
    return float(np.sum(xc * (ym - float(np.mean(ym)))) / den)


def local_metrics(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    t = data["t_rel_s"][mask]
    unwrapped = data["unwrapped_phase_rad"][mask]
    wrapped = data["phase_rad"][mask]
    dac = data["dac_code_cmd"][mask]
    if t.size < 2:
        return {
            "rows": float(t.size),
            "span_s": math.nan,
            "unwrapped_span_rad": math.nan,
            "slope_hz": math.nan,
            "wrap_count": math.nan,
            "wrapped_abs_median_rad": math.nan,
            "wrapped_abs_median_first_half_rad": math.nan,
            "wrapped_abs_median_second_half_rad": math.nan,
            "wrapped_abs_median_delta_rad": math.nan,
            "dac_min": math.nan,
            "dac_max": math.nan,
        }
    slope_hz = linear_slope(t, unwrapped) / (2.0 * math.pi)
    wrapped_abs = np.abs(wrapped)
    t_mid = float(t[0] + 0.5 * (t[-1] - t[0]))
    first_half = wrapped_abs[t <= t_mid]
    second_half = wrapped_abs[t > t_mid]
    wrapped_abs_median = float(np.nanmedian(wrapped_abs)) if np.isfinite(wrapped_abs).any() else math.nan
    wrapped_abs_first = (
        float(np.nanmedian(first_half)) if first_half.size and np.isfinite(first_half).any() else math.nan
    )
    wrapped_abs_second = (
        float(np.nanmedian(second_half)) if second_half.size and np.isfinite(second_half).any() else math.nan
    )
    return {
        "rows": float(t.size),
        "span_s": float(t[-1] - t[0]),
        "unwrapped_span_rad": float(np.nanmax(unwrapped) - np.nanmin(unwrapped)),
        "slope_hz": float(slope_hz),
        "wrap_count": float(np.floor((np.nanmax(unwrapped) - np.nanmin(unwrapped)) / (2.0 * math.pi))),
        "wrapped_abs_median_rad": wrapped_abs_median,
        "wrapped_abs_median_first_half_rad": wrapped_abs_first,
        "wrapped_abs_median_second_half_rad": wrapped_abs_second,
        "wrapped_abs_median_delta_rad": (
            float(wrapped_abs_second - wrapped_abs_first)
            if np.isfinite(wrapped_abs_first) and np.isfinite(wrapped_abs_second)
            else math.nan
        ),
        "dac_min": float(np.nanmin(dac)) if np.isfinite(dac).any() else math.nan,
        "dac_max": float(np.nanmax(dac)) if np.isfinite(dac).any() else math.nan,
        "wrapped_jumps": float(np.count_nonzero(np.abs(np.diff(wrapped)) > math.pi)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Saved cw_vcxo log, either .bin or .csv.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("first_stage_demo/logs/cleaned"),
        help="Directory for output PNG files.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to the log stem.")
    parser.add_argument("--zoom-start-s", type=float, default=None, help="Start time for the 1/6-second zoom.")
    parser.add_argument("--zoom-duration-s", type=float, default=1.0 / 6.0, help="Zoom duration in seconds.")
    parser.add_argument("--dpi", type=int, default=180, help="Saved image DPI.")
    args = parser.parse_args()

    if args.zoom_duration_s <= 0.0:
        raise ValueError("--zoom-duration-s must be positive.")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": int(args.dpi),
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    data = finite_phase_data(read_log(args.log))
    t = data["t_rel_s"]
    zoom_start = (
        float(args.zoom_start_s)
        if args.zoom_start_s is not None
        else default_zoom_start(t, float(args.zoom_duration_s))
    )
    zoom_stop = zoom_start + float(args.zoom_duration_s)
    full_mask = np.ones(t.shape, dtype=bool)
    zoom_mask = (t >= zoom_start) & (t <= zoom_stop)

    prefix = args.prefix or sanitize_stem(args.log)
    full_out = args.out_dir / f"{prefix}_phase_track_full.png"
    zoom_label = str(round(zoom_start, 6)).replace(".", "p")
    zoom_out = args.out_dir / f"{prefix}_phase_track_one_sixth_{zoom_label}s.png"

    plot_phase_track(data, full_out, f"{prefix}: full phase track", full_mask)
    plot_phase_track(
        data,
        zoom_out,
        f"{prefix}: phase track {zoom_start:.6f}-{zoom_stop:.6f}s",
        zoom_mask,
    )

    print(full_out)
    print(zoom_out)
    print(f"rows={t.size} duration={t[-1] - t[0]:.6f}s zoom_rows={int(np.count_nonzero(zoom_mask))}")
    full_metrics = local_metrics(data, full_mask)
    print(
        "full_metrics "
        + " ".join(
            f"{key}={value:.6g}" if np.isfinite(value) else f"{key}=nan"
            for key, value in full_metrics.items()
        )
    )
    metrics = local_metrics(data, zoom_mask)
    print(
        "zoom_metrics "
        + " ".join(
            f"{key}={value:.6g}" if np.isfinite(value) else f"{key}=nan"
            for key, value in metrics.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
