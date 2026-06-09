#!/usr/bin/env python3
"""Plot cycle-aware phase observer diagnostics from a CW VCXO binary log."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BINARY_MAGIC = b"CWVXCOBIN1\n"
# Keep diagnostics anchored to measured physical phase. NCO residual/correction
# fields are separate diagnostics and should not replace these columns.
DEFAULT_FIELDS = (
    "t_rel_s",
    "phase_rad",
    "unwrapped_phase_rad",
    "f_off_hz",
    "coherence",
    "phase_hold_active",
    "phase_hold_cmd_offset",
    "phase_error_rad",
    "phase_error_raw_rad",
    "phase_error_filtered_rad",
    "phase_error_cycle_rad",
    "phase_error_predicted_rad",
    "phase_error_branch_index",
    "phase_error_branch_residual_rad",
    "phase_error_physical_residual_rad",
    "phase_cycle_update_allowed",
    "phase_cycle_branch_reject_detected",
    "phase_cycle_branch_reject_count",
    "phase_cycle_slip_candidate_detected",
    "phase_cycle_slip_dwell_counter",
    "phase_cycle_slip_dwell_updates",
    "phase_cycle_slip_detected",
    "phase_cycle_slip_count",
    "phase_cycle_max_branch_error_rad",
    "phase_cycle_physical_max_continuity_error_rad",
    "phase_error_update_allowed",
    "dac_code_cmd",
    "control_base_cmd",
    "clock_est_phase_residual_rad",
)


def sanitize_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")


def read_binary_columns(path: Path, fields: Iterable[str]) -> dict[str, np.ndarray]:
    requested = tuple(fields)
    values: dict[str, list[float]] = {name: [] for name in requested}
    with path.open("rb") as f:
        magic = f.readline()
        if magic != BINARY_MAGIC:
            raise ValueError(f"{path} is not a CW VCXO binary log.")
        header_len_raw = f.read(4)
        if len(header_len_raw) != 4:
            raise ValueError(f"{path} has a truncated binary header.")
        header_len = struct.unpack("<I", header_len_raw)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        numeric_fields = list(header["numeric_fields"])
        string_fields = list(header["string_fields"])
        numeric_struct = struct.Struct("<" + ("d" * len(numeric_fields)))
        field_index = {
            name: numeric_fields.index(name) for name in requested if name in numeric_fields
        }

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
            row = numeric_struct.unpack_from(payload, 0)
            for name in requested:
                index = field_index.get(name)
                values[name].append(row[index] if index is not None else math.nan)

            offset = numeric_struct.size
            for _ in string_fields:
                if offset + 2 > len(payload):
                    break
                size = struct.unpack_from("<H", payload, offset)[0]
                offset += 2 + size

    return {name: np.asarray(column, dtype=np.float64) for name, column in values.items()}


def finite_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    t = data["t_rel_s"]
    phase = data["phase_rad"]
    unwrapped = data["unwrapped_phase_rad"]
    return np.isfinite(t) & np.isfinite(phase) & np.isfinite(unwrapped)


def decimate(mask: np.ndarray, max_points: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size <= max_points:
        return mask
    keep = indices[:: int(math.ceil(indices.size / max_points))]
    out = np.zeros_like(mask, dtype=bool)
    out[keep] = True
    return out


def robust_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"rms": math.nan, "p95_abs": math.nan, "span": math.nan}
    return {
        "rms": float(np.sqrt(np.mean(finite * finite))),
        "p95_abs": float(np.percentile(np.abs(finite), 95.0)),
        "span": float(np.nanmax(finite) - np.nanmin(finite)),
    }


def linear_slope_hz(t: np.ndarray, phase: np.ndarray) -> float:
    mask = np.isfinite(t) & np.isfinite(phase)
    if int(np.count_nonzero(mask)) < 2:
        return math.nan
    x = t[mask]
    y = phase[mask]
    xc = x - float(np.mean(x))
    den = float(np.sum(xc * xc))
    if den <= 0.0:
        return math.nan
    slope_rad_s = float(np.sum(xc * (y - float(np.mean(y)))) / den)
    return slope_rad_s / (2.0 * math.pi)


def summarize(data: dict[str, np.ndarray], zoom_mask: np.ndarray) -> dict[str, float]:
    mask = finite_mask(data)
    t = data["t_rel_s"][mask]
    summary: dict[str, float] = {
        "rows": float(t.size),
        "duration_s": float(t[-1] - t[0]) if t.size >= 2 else math.nan,
    }
    for name in ("phase_hold_active", "phase_cycle_update_allowed", "phase_error_update_allowed"):
        values = data[name][mask]
        finite = values[np.isfinite(values)]
        summary[f"{name}_fraction"] = float(np.mean(finite > 0.5)) if finite.size else math.nan

    slip_count = data["phase_cycle_slip_count"][mask]
    finite_slips = slip_count[np.isfinite(slip_count)]
    summary["cycle_slip_count_final"] = float(finite_slips[-1]) if finite_slips.size else math.nan
    slip_flags = data["phase_cycle_slip_detected"][mask]
    summary["cycle_slip_flag_rows"] = float(np.count_nonzero(np.nan_to_num(slip_flags) > 0.5))
    branch_rejects = data["phase_cycle_branch_reject_detected"][mask]
    summary["cycle_branch_reject_rows"] = float(np.count_nonzero(np.nan_to_num(branch_rejects) > 0.5))
    branch_reject_count = data["phase_cycle_branch_reject_count"][mask]
    finite_branch_rejects = branch_reject_count[np.isfinite(branch_reject_count)]
    summary["cycle_branch_reject_count_final"] = (
        float(finite_branch_rejects[-1]) if finite_branch_rejects.size else math.nan
    )
    slip_candidates = data["phase_cycle_slip_candidate_detected"][mask]
    summary["cycle_slip_candidate_rows"] = float(np.count_nonzero(np.nan_to_num(slip_candidates) > 0.5))

    branch = data["phase_error_branch_residual_rad"][mask]
    for key, value in robust_stats(branch).items():
        summary[f"branch_residual_{key}_rad"] = value
    physical = data["phase_error_physical_residual_rad"][mask]
    for key, value in robust_stats(physical).items():
        summary[f"physical_residual_{key}_rad"] = value

    cycle_error = data["phase_error_cycle_rad"][mask]
    for key, value in robust_stats(cycle_error).items():
        summary[f"cycle_error_{key}_rad"] = value

    zoom = zoom_mask & mask
    zt = data["t_rel_s"][zoom]
    zunwrapped = data["unwrapped_phase_rad"][zoom]
    zcycle = data["phase_error_cycle_rad"][zoom]
    zbranch = data["phase_error_branch_residual_rad"][zoom]
    zphysical = data["phase_error_physical_residual_rad"][zoom]
    summary["zoom_rows"] = float(zt.size)
    summary["zoom_span_s"] = float(zt[-1] - zt[0]) if zt.size >= 2 else math.nan
    summary["zoom_unwrapped_span_rad"] = (
        float(np.nanmax(zunwrapped) - np.nanmin(zunwrapped)) if zt.size else math.nan
    )
    summary["zoom_unwrapped_slope_hz"] = linear_slope_hz(zt, zunwrapped)
    for key, value in robust_stats(zcycle).items():
        summary[f"zoom_cycle_error_{key}_rad"] = value
    for key, value in robust_stats(zbranch).items():
        summary[f"zoom_branch_residual_{key}_rad"] = value
    for key, value in robust_stats(zphysical).items():
        summary[f"zoom_physical_residual_{key}_rad"] = value
    return summary


def plot_diagnostics(
    data: dict[str, np.ndarray],
    out_png: Path,
    title: str,
    zoom_start_s: float,
    zoom_duration_s: float,
) -> dict[str, float]:
    base_mask = finite_mask(data)
    if int(np.count_nonzero(base_mask)) < 2:
        raise ValueError("Log does not contain enough finite phase samples.")

    t = data["t_rel_s"]
    zoom_mask = base_mask & (t >= zoom_start_s) & (t < zoom_start_s + zoom_duration_s)
    if int(np.count_nonzero(zoom_mask)) < 2:
        raise ValueError("Selected zoom window has fewer than two finite samples.")
    full_plot_mask = decimate(base_mask, max_points=25000)

    summary = summarize(data, zoom_mask)
    fig, axes = plt.subplots(4, 2, figsize=(15.0, 11.0), constrained_layout=True)

    def x(mask: np.ndarray) -> np.ndarray:
        return data["t_rel_s"][mask]

    unwrapped_rel = data["unwrapped_phase_rad"] - data["unwrapped_phase_rad"][np.flatnonzero(base_mask)[0]]
    cycle = data["phase_error_cycle_rad"]
    cycle_rel = cycle - cycle[np.flatnonzero(np.isfinite(cycle) & base_mask)[0]]
    branch_threshold_values = data["phase_cycle_max_branch_error_rad"][base_mask]
    branch_threshold_values = branch_threshold_values[np.isfinite(branch_threshold_values)]
    branch_threshold = float(branch_threshold_values[-1]) if branch_threshold_values.size else 2.5
    physical_threshold_values = data["phase_cycle_physical_max_continuity_error_rad"][base_mask]
    physical_threshold_values = physical_threshold_values[np.isfinite(physical_threshold_values)]
    physical_threshold = float(physical_threshold_values[-1]) if physical_threshold_values.size else math.nan

    axes[0, 0].plot(x(full_plot_mask), unwrapped_rel[full_plot_mask], lw=0.7, label="raw unwrapped")
    axes[0, 0].plot(x(full_plot_mask), cycle_rel[full_plot_mask], lw=0.7, label="cycle observer")
    axes[0, 0].set_title("Full Run: Relative Phase States")
    axes[0, 0].set_ylabel("rad")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[0, 1].plot(x(zoom_mask), data["phase_rad"][zoom_mask], lw=0.9, label="wrapped raw")
    axes[0, 1].plot(x(zoom_mask), data["phase_error_raw_rad"][zoom_mask], lw=0.9, label="wrapped error")
    axes[0, 1].set_title(f"Zoom: {zoom_start_s:.3f}s to {zoom_start_s + zoom_duration_s:.3f}s")
    axes[0, 1].set_ylabel("rad")
    axes[0, 1].set_ylim(-math.pi * 1.08, math.pi * 1.08)
    axes[0, 1].legend(loc="upper right", fontsize=8)

    axes[1, 0].plot(x(full_plot_mask), data["phase_error_rad"][full_plot_mask], lw=0.7, label="control error")
    axes[1, 0].plot(x(full_plot_mask), data["phase_error_filtered_rad"][full_plot_mask], lw=0.7, label="wrapped filter")
    axes[1, 0].set_title("Full Run: Control Error")
    axes[1, 0].set_ylabel("rad")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    axes[1, 1].plot(x(zoom_mask), data["phase_error_rad"][zoom_mask], lw=0.9, label="control error")
    axes[1, 1].plot(x(zoom_mask), data["phase_error_filtered_rad"][zoom_mask], lw=0.9, label="wrapped filter")
    axes[1, 1].set_title("Zoom: Control Error")
    axes[1, 1].set_ylabel("rad")
    axes[1, 1].legend(loc="upper right", fontsize=8)

    for col, mask, label in ((0, full_plot_mask, "Full Run"), (1, zoom_mask, "Zoom")):
        axes[2, col].plot(
            x(mask),
            data["phase_error_branch_residual_rad"][mask],
            lw=0.75,
            label="branch residual",
        )
        if np.any(np.isfinite(data["phase_error_physical_residual_rad"][mask])):
            axes[2, col].plot(
                x(mask),
                data["phase_error_physical_residual_rad"][mask],
                lw=0.75,
                label="physical residual",
            )
        branch_reject_mask = mask & (np.nan_to_num(data["phase_cycle_branch_reject_detected"]) > 0.5)
        if np.any(branch_reject_mask):
            axes[2, col].scatter(
                x(branch_reject_mask),
                data["phase_error_branch_residual_rad"][branch_reject_mask],
                s=14,
                color="#ff7f0e",
                label="branch reject",
                zorder=3,
            )
        candidate_mask = mask & (np.nan_to_num(data["phase_cycle_slip_candidate_detected"]) > 0.5)
        if np.any(candidate_mask):
            candidate_y = data["phase_error_physical_residual_rad"][candidate_mask]
            if not np.any(np.isfinite(candidate_y)):
                candidate_y = data["phase_error_branch_residual_rad"][candidate_mask]
            axes[2, col].scatter(
                x(candidate_mask),
                candidate_y,
                s=16,
                color="#9467bd",
                label="slip candidate",
                zorder=4,
            )
        slip_mask = mask & (np.nan_to_num(data["phase_cycle_slip_detected"]) > 0.5)
        if np.any(slip_mask):
            axes[2, col].scatter(
                x(slip_mask),
                data["phase_error_branch_residual_rad"][slip_mask],
                s=18,
                color="#d62728",
                label="slip flag",
                zorder=3,
            )
        axes[2, col].axhline(branch_threshold, color="0.35", ls="--", lw=0.8)
        axes[2, col].axhline(-branch_threshold, color="0.35", ls="--", lw=0.8)
        if np.isfinite(physical_threshold):
            axes[2, col].axhline(physical_threshold, color="#1f77b4", ls=":", lw=0.8)
            axes[2, col].axhline(-physical_threshold, color="#1f77b4", ls=":", lw=0.8)
        axes[2, col].set_title(f"{label}: Branch Residual")
        axes[2, col].set_ylabel("rad")
        axes[2, col].legend(loc="upper right", fontsize=8)

    ax = axes[3, 0]
    ax.plot(x(full_plot_mask), data["f_off_hz"][full_plot_mask], lw=0.7, label="f_off")
    ax.set_ylabel("Hz")
    ax2 = ax.twinx()
    ax2.step(x(full_plot_mask), data["dac_code_cmd"][full_plot_mask], where="post", lw=0.7, color="#d62728", label="DAC")
    ax2.plot(
        x(full_plot_mask),
        data["control_base_cmd"][full_plot_mask] + data["phase_hold_cmd_offset"][full_plot_mask],
        lw=0.7,
        color="#9467bd",
        label="base+phase FF",
    )
    ax.set_title("Full Run: Frequency Estimate and DAC")
    ax.set_xlabel("Time (s)")
    ax2.set_ylabel("DAC code")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

    ax = axes[3, 1]
    ax.step(x(zoom_mask), data["phase_hold_active"][zoom_mask], where="post", lw=0.8, label="phase hold")
    ax.step(
        x(zoom_mask),
        data["phase_cycle_update_allowed"][zoom_mask],
        where="post",
        lw=0.8,
        label="cycle update",
    )
    ax.step(
        x(zoom_mask),
        data["phase_cycle_branch_reject_detected"][zoom_mask],
        where="post",
        lw=0.8,
        label="branch reject",
    )
    ax.step(
        x(zoom_mask),
        data["phase_cycle_slip_candidate_detected"][zoom_mask],
        where="post",
        lw=0.8,
        label="slip candidate",
    )
    ax.step(
        x(zoom_mask),
        data["phase_cycle_slip_count"][zoom_mask],
        where="post",
        lw=0.8,
        label="slip count",
    )
    ax.set_title("Zoom: Observer Gates")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("state/count")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(title)
    fig.text(
        0.01,
        0.01,
        (
            f"rows={summary['rows']:.0f}, duration={summary['duration_s']:.3f}s, "
            f"hold={summary['phase_hold_active_fraction']:.3f}, "
            f"cycle_update={summary['phase_cycle_update_allowed_fraction']:.3f}, "
            f"slips={summary['cycle_slip_count_final']:.0f}, "
            f"branch_rejects={summary['cycle_branch_reject_count_final']:.0f}, "
            f"candidates={summary['cycle_slip_candidate_rows']:.0f}, "
            f"branch_rms={summary['branch_residual_rms_rad']:.3f} rad"
        ),
        ha="left",
        va="bottom",
        fontsize=9,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("first_stage_demo/logs/cleaned"))
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--zoom-start-s", type=float, default=150.0)
    parser.add_argument("--zoom-duration-s", type=float, default=1.0 / 6.0)
    args = parser.parse_args()

    data = read_binary_columns(args.log, DEFAULT_FIELDS)
    prefix = args.prefix or sanitize_stem(args.log)
    out_png = args.out_dir / f"{prefix}_cycle_tracking_diagnostics.png"
    summary = plot_diagnostics(
        data,
        out_png=out_png,
        title=prefix,
        zoom_start_s=float(args.zoom_start_s),
        zoom_duration_s=float(args.zoom_duration_s),
    )
    print(out_png)
    for key in sorted(summary):
        print(f"{key}={summary[key]:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
