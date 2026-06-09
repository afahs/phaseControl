#!/usr/bin/env python3
"""Plot phase-measurement acceptance diagnostics for saved NotebookBased runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        try:
            vals.append(float(raw))
        except Exception:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float64)


def as_int(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals: list[int] = []
    for row in rows:
        raw = row.get(key, "")
        try:
            vals.append(int(raw))
        except Exception:
            vals.append(0)
    return np.asarray(vals, dtype=np.int64)


def phase_lock_mask(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([row.get("stage", "") == "phase_lock" for row in rows], dtype=bool)


def plot_phase_traces(
    *,
    monitor_rows: list[dict[str, str]],
    pid_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)

    for ax, rows, title in zip(
        axes,
        [monitor_rows, pid_rows],
        ["Open Loop", "Closed Loop"],
        strict=True,
    ):
        mask = phase_lock_mask(rows)
        t = as_float(rows, "elapsed_s")[mask]
        raw = as_float(rows, "phase_err_raw_rad")[mask]
        filt = as_float(rows, "phase_err_filt_rad")[mask]
        rejected = as_int(rows, "phase_measurement_rejected")[mask].astype(bool)
        accepted = ~rejected

        ax.plot(t, raw, color="#b7bcc4", lw=0.8, alpha=0.45, label="raw all")
        ax.scatter(
            t[accepted],
            raw[accepted],
            s=5,
            color="#1565c0",
            alpha=0.65,
            label="raw accepted",
            rasterized=True,
        )
        ax.plot(t, filt, color="#d9480f", lw=1.25, alpha=0.95, label="filtered")
        ax.axhline(1.0, color="#777777", lw=0.8, ls="--", alpha=0.8)
        ax.axhline(-1.0, color="#777777", lw=0.8, ls="--", alpha=0.8)
        ax.axhline(0.0, color="#333333", lw=0.8, alpha=0.6)
        ax.set_ylabel("Phase Error [rad]")
        ax.set_title(
            f"{title}: accepted {int(np.count_nonzero(accepted))}/{int(t.size)} "
            f"({100.0 * float(np.mean(accepted)):.1f}%)"
        )
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Elapsed Time [s]")
    fig.suptitle("Phase Error Diagnostics in phase_lock")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gate_traces(
    *,
    monitor_rows: list[dict[str, str]],
    pid_rows: list[dict[str, str]],
    output_path: Path,
    innovation_max_rad: float,
    fit_rms_max_rad: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)

    for ax, rows, title in zip(
        axes,
        [monitor_rows, pid_rows],
        ["Open Loop", "Closed Loop"],
        strict=True,
    ):
        mask = phase_lock_mask(rows)
        t = as_float(rows, "elapsed_s")[mask]
        innov = np.abs(as_float(rows, "phase_measure_innovation_rad")[mask])
        fit_rms = as_float(rows, "phase_fit_rms_rad")[mask]
        rejected = as_int(rows, "phase_measurement_rejected")[mask].astype(bool)

        ax.plot(t, innov, color="#6a1b9a", lw=1.0, alpha=0.9, label="|innovation|")
        ax.plot(t, fit_rms, color="#2e7d32", lw=1.0, alpha=0.9, label="fit RMS")
        if np.isfinite(innovation_max_rad) and innovation_max_rad > 0.0:
            ax.axhline(innovation_max_rad, color="#6a1b9a", lw=0.9, ls="--", alpha=0.8)
        if np.isfinite(fit_rms_max_rad) and fit_rms_max_rad > 0.0:
            ax.axhline(fit_rms_max_rad, color="#2e7d32", lw=0.9, ls="--", alpha=0.8)
        if np.any(rejected):
            reject_y = np.nanmax(
                np.concatenate(
                    [
                        innov[np.isfinite(innov)],
                        fit_rms[np.isfinite(fit_rms)],
                        np.asarray([innovation_max_rad, fit_rms_max_rad], dtype=np.float64),
                    ]
                )
            )
            reject_y = max(float(reject_y), 1.0)
            ax.scatter(
                t[rejected],
                np.full(int(np.count_nonzero(rejected)), reject_y * 1.03, dtype=np.float64),
                s=10,
                marker="x",
                color="#c62828",
                alpha=0.75,
                label="rejected",
                rasterized=True,
            )
            ax.set_ylim(top=reject_y * 1.1)
        ax.set_ylabel("Gate Value")
        ax.set_title(title)
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Elapsed Time [s]")
    fig.suptitle("Measurement Gate Diagnostics in phase_lock")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--nsamp", type=int, required=True)
    p.add_argument(
        "--phase-plot-name",
        type=str,
        default="phase_acceptance_diagnostics.png",
    )
    p.add_argument(
        "--gate-plot-name",
        type=str,
        default="phase_gate_diagnostics.png",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    monitor_csv = run_dir / "monitor" / f"nsamp_{int(args.nsamp)}" / f"monitor_nsamp{int(args.nsamp)}.csv"
    pid_csv = run_dir / "pid" / f"nsamp_{int(args.nsamp)}" / f"pid_nsamp{int(args.nsamp)}.csv"
    if not monitor_csv.exists():
        raise FileNotFoundError(f"Missing monitor CSV: {monitor_csv}")
    if not pid_csv.exists():
        raise FileNotFoundError(f"Missing PID CSV: {pid_csv}")

    monitor_rows = load_rows(monitor_csv)
    pid_rows = load_rows(pid_csv)
    comparison_dir = run_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # The thresholds are encoded by the controller settings used for the run.
    # We pass the active defaults here so the saved plot shows the intended gate lines.
    innovation_max_rad = 1.8
    fit_rms_max_rad = 0.8

    phase_plot = comparison_dir / str(args.phase_plot_name)
    gate_plot = comparison_dir / str(args.gate_plot_name)
    plot_phase_traces(
        monitor_rows=monitor_rows,
        pid_rows=pid_rows,
        output_path=phase_plot,
    )
    plot_gate_traces(
        monitor_rows=monitor_rows,
        pid_rows=pid_rows,
        output_path=gate_plot,
        innovation_max_rad=innovation_max_rad,
        fit_rms_max_rad=fit_rms_max_rad,
    )
    print(f"Wrote {phase_plot}")
    print(f"Wrote {gate_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
