from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_summary(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected dict summary in {path}")
    return data


def to_float_series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        try:
            vals.append(float(row.get(key, "nan")))
        except Exception:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float64)


def first_finite(a: np.ndarray) -> float:
    mask = np.isfinite(a)
    if not np.any(mask):
        return float("nan")
    return float(a[np.flatnonzero(mask)[0]])


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.asarray([], dtype=bool)
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def interp_series(x: np.ndarray, y: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    out = np.full(x_query.shape, np.nan, dtype=np.float64)
    mask = finite_mask(x, y)
    if np.count_nonzero(mask) < 2:
        return out
    xv = np.asarray(x[mask], dtype=np.float64)
    yv = np.asarray(y[mask], dtype=np.float64)
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    uniq = np.ones(xv.shape, dtype=bool)
    uniq[1:] = np.diff(xv) > 0.0
    xv = xv[uniq]
    yv = yv[uniq]
    if xv.size < 2:
        return out
    valid = np.isfinite(x_query) & (x_query >= xv[0]) & (x_query <= xv[-1])
    if np.any(valid):
        out[valid] = np.interp(x_query[valid], xv, yv)
    return out


def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if window <= 1:
        return np.copy(a)
    for i in range(a.size):
        j0 = max(0, i - window + 1)
        seg = a[j0 : i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.mean(seg))
    return out


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = finite_mask(x, y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    x0 = xv - np.mean(xv)
    denom = float(np.sum(x0 * x0))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x0 * (yv - np.mean(yv))) / denom)


def first_stage_time(rows: list[dict[str, str]], stage_name: str) -> float:
    stage_key = str(stage_name).strip().lower()
    for row in rows:
        if str(row.get("stage", "")).strip().lower() != stage_key:
            continue
        try:
            return float(row.get("elapsed_s", "nan"))
        except Exception:
            return float("nan")
    return float("nan")


def last_stage_time(rows: list[dict[str, str]], stage_name: str) -> float:
    stage_key = str(stage_name).strip().lower()
    last = float("nan")
    for row in rows:
        if str(row.get("stage", "")).strip().lower() != stage_key:
            continue
        try:
            last = float(row.get("elapsed_s", "nan"))
        except Exception:
            continue
    return float(last)


def format_ns_value(value_ns: float) -> str:
    val = float(value_ns)
    if not np.isfinite(val):
        return "nan"
    abs_val = abs(val)
    if abs_val >= 1e6:
        return f"{val / 1e6:.2f} ms"
    if abs_val >= 1e3:
        return f"{val / 1e3:.1f} us"
    return f"{val:.1f} ns"


def compute_relative_timing_reduction(
    reference_t_s: np.ndarray,
    reference_tau_rel_ns: np.ndarray,
    candidate_t_s: np.ndarray,
    candidate_tau_rel_ns: np.ndarray,
    *,
    min_reference_abs_ns: float = 100.0,
) -> dict[str, np.ndarray]:
    cand_mask = finite_mask(candidate_t_s, candidate_tau_rel_ns)
    t_s = np.asarray(candidate_t_s[cand_mask], dtype=np.float64)
    candidate_tau_ns = np.asarray(candidate_tau_rel_ns[cand_mask], dtype=np.float64)
    reference_tau_ns = interp_series(reference_t_s, reference_tau_rel_ns, t_s)
    reference_abs_ns = np.abs(reference_tau_ns)
    candidate_abs_ns = np.abs(candidate_tau_ns)
    reduction_pct = np.full(t_s.shape, np.nan, dtype=np.float64)
    valid = finite_mask(reference_abs_ns, candidate_abs_ns) & (reference_abs_ns >= float(min_reference_abs_ns))
    if np.any(valid):
        reduction_pct[valid] = 100.0 * (1.0 - (candidate_abs_ns[valid] / reference_abs_ns[valid]))
    return {
        "t_s": t_s,
        "reference_tau_ns": reference_tau_ns,
        "candidate_tau_ns": candidate_tau_ns,
        "reference_abs_ns": reference_abs_ns,
        "candidate_abs_ns": candidate_abs_ns,
        "reduction_pct": reduction_pct,
    }


def metric_at_time(t_query_s: float, t_s: np.ndarray, y: np.ndarray) -> float:
    if not np.isfinite(t_query_s):
        return float("nan")
    out = interp_series(t_s, y, np.asarray([float(t_query_s)], dtype=np.float64))
    if out.size == 0:
        return float("nan")
    return float(out[0])


def compute_timing_series(rows: list[dict[str, str]], center_freq_hz: float) -> dict[str, np.ndarray | float]:
    t = to_float_series(rows, "elapsed_s")
    tau_ns_abs = to_float_series(rows, "observer_tau_err_ns")
    sample_rate_hz = float(rows[0]["sample_rate_hz"])
    sample_period_ns = 1e9 / sample_rate_hz
    tau0 = first_finite(tau_ns_abs)
    tau_ns_rel = tau_ns_abs - tau0 if np.isfinite(tau0) else np.copy(tau_ns_abs)
    offset_samples = tau_ns_rel / sample_period_ns

    observer_tone_offset_hz = to_float_series(rows, "observer_tone_offset_hz")
    observer_rf_hz = float(center_freq_hz) + observer_tone_offset_hz
    update_freq_hz = to_float_series(rows, "observer_freq_err_update_hz")
    slip_samples_per_s = np.full(update_freq_hz.shape, np.nan, dtype=np.float64)
    valid = finite_mask(observer_rf_hz, update_freq_hz) & (np.abs(observer_rf_hz) > 1e-6)
    slip_samples_per_s[valid] = update_freq_hz[valid] * sample_rate_hz / observer_rf_hz[valid]

    return {
        "t_s": t,
        "tau_abs_ns": tau_ns_abs,
        "tau_rel_ns": tau_ns_rel,
        "offset_samples": offset_samples,
        "slip_samples_per_s": slip_samples_per_s,
        "sample_rate_hz": sample_rate_hz,
        "sample_period_ns": sample_period_ns,
    }


def compute_metrics(rows: list[dict[str, str]], summary: dict[str, Any]) -> dict[str, Any]:
    timing = compute_timing_series(rows, float(summary.get("center_freq_hz", 100e6)))
    t = timing["t_s"]
    offs = timing["offset_samples"]
    slips = timing["slip_samples_per_s"]
    mask_offs = np.isfinite(offs)
    mask_slips = np.isfinite(slips)

    metrics: dict[str, Any] = {
        "elapsed_s": float(t[mask_offs][-1] - t[mask_offs][0]) if np.count_nonzero(mask_offs) >= 2 else float("nan"),
        "timing_drift_samples_per_s": float(summary.get("primary_timing_drift_samples_per_s", "nan")),
        "timing_drift_ns_per_s": float(summary.get("primary_timing_drift_ns_per_s", "nan")),
        "offset_end_samples": float(offs[mask_offs][-1]) if np.any(mask_offs) else float("nan"),
        "max_abs_offset_samples": float(np.nanmax(np.abs(offs))) if np.any(mask_offs) else float("nan"),
        "p95_abs_offset_samples": (
            float(np.nanpercentile(np.abs(offs[mask_offs]), 95.0)) if np.any(mask_offs) else float("nan")
        ),
        "offset_end_ns": float(timing["tau_rel_ns"][mask_offs][-1]) if np.any(mask_offs) else float("nan"),
        "max_abs_timing_error_change_ns": (
            float(np.nanmax(np.abs(timing["tau_rel_ns"][mask_offs]))) if np.any(mask_offs) else float("nan")
        ),
        "p95_abs_timing_error_change_ns": (
            float(np.nanpercentile(np.abs(timing["tau_rel_ns"][mask_offs]), 95.0)) if np.any(mask_offs) else float("nan")
        ),
        "mean_abs_slip_samples_per_s": (
            float(np.nanmean(np.abs(slips[mask_slips]))) if np.any(mask_slips) else float("nan")
        ),
        "phase_drift_hz": float(summary.get("observer_phase_drift_slope_full_hz", "nan")),
        "quality_fraction": float(summary.get("quality_fraction", "nan")),
        "median_coherence": float(summary.get("observer_median_coherence", "nan")),
    }

    for band in (0.25, 0.5, 1.0, 2.0, 4.0):
        key = f"fraction_within_{str(band).replace('.', 'p')}_samples"
        metrics[key] = float(np.mean(np.abs(offs[mask_offs]) <= band)) if np.any(mask_offs) else float("nan")

    return metrics


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def write_markdown(
    path: Path,
    run_dir: Path,
    monitor_metrics: dict[str, Any],
    pid_metrics: dict[str, Any],
    comparison_metrics: dict[str, Any] | None = None,
) -> None:
    improvement = float("nan")
    mon = float(monitor_metrics.get("timing_drift_samples_per_s", "nan"))
    pid = float(pid_metrics.get("timing_drift_samples_per_s", "nan"))
    if np.isfinite(mon) and abs(mon) > 1e-12 and np.isfinite(pid):
        improvement = 100.0 * (abs(mon) - abs(pid)) / abs(mon)

    lines = [
        "# Open vs Closed Comparison",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "These timing metrics are inferred from observer phase on the saved SDR stream.",
        "They show relative timing drift and inferred sample offset, but they are not direct dual-device hardware timestamps.",
        "",
        "## Timing Summary",
        "",
        f"- Open-loop inferred timing drift: `{monitor_metrics['timing_drift_samples_per_s']:.6f}` samples/s",
        f"- Closed-loop inferred timing drift: `{pid_metrics['timing_drift_samples_per_s']:.6f}` samples/s",
        f"- Drift improvement: `{improvement:.3f}%`",
        f"- Open-loop max inferred sample offset change: `{monitor_metrics['max_abs_offset_samples']:.3f}` samples",
        f"- Closed-loop max inferred sample offset change: `{pid_metrics['max_abs_offset_samples']:.3f}` samples",
        f"- Open-loop 95th percentile |offset|: `{monitor_metrics['p95_abs_offset_samples']:.3f}` samples",
        f"- Closed-loop 95th percentile |offset|: `{pid_metrics['p95_abs_offset_samples']:.3f}` samples",
        f"- Closed-loop fraction within +/-1 sample: `{pid_metrics['fraction_within_1p0_samples']:.3f}`",
        f"- Closed-loop fraction within +/-0.5 sample: `{pid_metrics['fraction_within_0p5_samples']:.3f}`",
        f"- Open-loop max |relative timing error change|: `{monitor_metrics['max_abs_timing_error_change_ns']:.1f}` ns",
        f"- Closed-loop max |relative timing error change|: `{pid_metrics['max_abs_timing_error_change_ns']:.1f}` ns",
        "",
        "## Interpretation",
        "",
        "The saved data supports that closed loop greatly reduces inferred relative timing drift versus open loop.",
        "It does not, by itself, prove exact same-time sampling at the hardware timestamp level.",
        "",
    ]
    if comparison_metrics:
        handoff_s = float(comparison_metrics.get("pull_in_end_s", float("nan")))
        handoff_reduction_pct = float(comparison_metrics.get("timing_error_reduction_at_pullin_end_pct", float("nan")))
        handoff_open_ns = float(comparison_metrics.get("open_abs_timing_error_at_pullin_end_ns", float("nan")))
        handoff_closed_ns = float(comparison_metrics.get("closed_abs_timing_error_at_pullin_end_ns", float("nan")))
        end_reduction_pct = float(comparison_metrics.get("timing_error_reduction_at_end_pct", float("nan")))
        if np.isfinite(handoff_s) and np.isfinite(handoff_reduction_pct):
            lines.extend(
                [
                    "## Pull-In Snapshot",
                    "",
                    f"- Phase-lock handoff time: `{handoff_s:.3f}` s",
                    (
                        "- At handoff, closed-loop |timing error change| is "
                        f"`{format_ns_value(handoff_closed_ns)}` versus open-loop "
                        f"`{format_ns_value(handoff_open_ns)}`, a reduction of "
                        f"`{handoff_reduction_pct:.2f}%`"
                    ),
                    f"- End-of-run timing-error reduction versus open loop: `{end_reduction_pct:.2f}%`",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def plot_dashboard(
    *,
    output_path: Path,
    run_label: str,
    monitor_rows: list[dict[str, str]],
    pid_rows: list[dict[str, str]],
    monitor_summary: dict[str, Any],
    pid_summary: dict[str, Any],
) -> dict[str, Any]:
    mon = compute_timing_series(monitor_rows, float(monitor_summary.get("center_freq_hz", 100e6)))
    pid = compute_timing_series(pid_rows, float(pid_summary.get("center_freq_hz", 100e6)))
    mon_metrics = compute_metrics(monitor_rows, monitor_summary)
    pid_metrics = compute_metrics(pid_rows, pid_summary)
    timing_reduction = compute_relative_timing_reduction(
        np.asarray(mon["t_s"]),
        np.asarray(mon["tau_rel_ns"]),
        np.asarray(pid["t_s"]),
        np.asarray(pid["tau_rel_ns"]),
    )

    pull_in_end_s = first_stage_time(pid_rows, "phase_lock")
    if not np.isfinite(pull_in_end_s):
        pull_in_end_s = last_stage_time(pid_rows, "pull_in")

    handoff_open_tau_ns = metric_at_time(float(pull_in_end_s), np.asarray(mon["t_s"]), np.asarray(mon["tau_rel_ns"]))
    handoff_closed_tau_ns = metric_at_time(float(pull_in_end_s), np.asarray(pid["t_s"]), np.asarray(pid["tau_rel_ns"]))
    end_t_s = float(pid_metrics.get("elapsed_s", float("nan")))
    end_open_tau_ns = metric_at_time(end_t_s, np.asarray(mon["t_s"]), np.asarray(mon["tau_rel_ns"]))
    end_closed_tau_ns = metric_at_time(end_t_s, np.asarray(pid["t_s"]), np.asarray(pid["tau_rel_ns"]))

    def reduction_from_pair(reference_ns: float, candidate_ns: float) -> float:
        if (not np.isfinite(reference_ns)) or (not np.isfinite(candidate_ns)) or abs(reference_ns) < 1e-9:
            return float("nan")
        return float(100.0 * (1.0 - (abs(candidate_ns) / abs(reference_ns))))

    comparison_metrics = {
        "pull_in_end_s": float(pull_in_end_s),
        "open_abs_timing_error_at_pullin_end_ns": float(abs(handoff_open_tau_ns)) if np.isfinite(handoff_open_tau_ns) else float("nan"),
        "closed_abs_timing_error_at_pullin_end_ns": float(abs(handoff_closed_tau_ns)) if np.isfinite(handoff_closed_tau_ns) else float("nan"),
        "timing_error_reduction_at_pullin_end_pct": reduction_from_pair(handoff_open_tau_ns, handoff_closed_tau_ns),
        "open_abs_timing_error_end_ns": float(abs(end_open_tau_ns)) if np.isfinite(end_open_tau_ns) else float("nan"),
        "closed_abs_timing_error_end_ns": float(abs(end_closed_tau_ns)) if np.isfinite(end_closed_tau_ns) else float("nan"),
        "timing_error_reduction_at_end_pct": reduction_from_pair(end_open_tau_ns, end_closed_tau_ns),
        "median_timing_error_reduction_pct_after_1s": (
            float(np.nanmedian(timing_reduction["reduction_pct"][np.asarray(timing_reduction["t_s"]) >= 1.0]))
            if np.any(np.isfinite(timing_reduction["reduction_pct"][np.asarray(timing_reduction["t_s"]) >= 1.0]))
            else float("nan")
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    def shade_pullin(ax: Any, *, label: str | None = None) -> None:
        if np.isfinite(pull_in_end_s) and pull_in_end_s > 0.0:
            ax.axvspan(0.0, pull_in_end_s, color="#e9c46a", alpha=0.16, label=label)
            ax.axvline(pull_in_end_s, color="#c98f00", linewidth=1.2, linestyle="--", alpha=0.9)

    ax = axes[0, 0]
    shade_pullin(ax, label="Closed-loop pull-in window")
    ax.axhline(0.0, color="#999999", linewidth=0.9, alpha=0.6)
    ax.plot(mon["t_s"], mon["tau_rel_ns"], label="Open loop", color="#b8402a", linewidth=2.0)
    ax.plot(pid["t_s"], pid["tau_rel_ns"], label="Closed loop", color="#006d77", linewidth=2.0)
    ax.set_yscale("symlog", linthresh=100.0)
    ax.set_title("Relative Timing Error Change (symlog)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Delta tau [ns]")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    shade_pullin(ax)
    mon_tau_mag = np.maximum(np.abs(np.asarray(mon["tau_rel_ns"])), 1.0)
    pid_tau_mag = np.maximum(np.abs(np.asarray(pid["tau_rel_ns"])), 1.0)
    ax.plot(mon["t_s"], mon_tau_mag, label="Open loop", color="#b8402a", linewidth=2.0)
    ax.plot(pid["t_s"], pid_tau_mag, label="Closed loop", color="#006d77", linewidth=2.0)
    ax.set_yscale("log")
    ax.set_title("Timing Error Magnitude")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("|Delta tau| [ns]")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    handoff_reduction_pct = float(comparison_metrics.get("timing_error_reduction_at_pullin_end_pct", float("nan")))
    end_reduction_pct = float(comparison_metrics.get("timing_error_reduction_at_end_pct", float("nan")))
    ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                (
                    f"handoff @ {pull_in_end_s:.2f}s: "
                    f"{handoff_reduction_pct:.2f}% lower |Delta tau|"
                )
                if np.isfinite(pull_in_end_s) and np.isfinite(handoff_reduction_pct)
                else "handoff: not available",
                (
                    f"open {format_ns_value(comparison_metrics['open_abs_timing_error_at_pullin_end_ns'])} "
                    f"vs closed {format_ns_value(comparison_metrics['closed_abs_timing_error_at_pullin_end_ns'])}"
                )
                if np.isfinite(handoff_reduction_pct)
                else "",
                (
                    f"end of run: {end_reduction_pct:.2f}% lower |Delta tau|"
                )
                if np.isfinite(end_reduction_pct)
                else "",
            ]
        ).strip(),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"},
    )

    ax = axes[1, 0]
    shade_pullin(ax)
    reduction_t_s = np.asarray(timing_reduction["t_s"])
    reduction_pct = np.asarray(timing_reduction["reduction_pct"])
    ax.axhline(90.0, color="#999999", linewidth=0.9, alpha=0.7, linestyle="--")
    ax.plot(reduction_t_s, reduction_pct, color="#006d77", linewidth=2.0)
    if np.isfinite(pull_in_end_s) and np.isfinite(handoff_reduction_pct):
        ax.scatter([pull_in_end_s], [handoff_reduction_pct], color="#c98f00", s=32, zorder=5)
    if np.isfinite(end_t_s) and np.isfinite(end_reduction_pct):
        ax.scatter([end_t_s], [end_reduction_pct], color="#264653", s=28, zorder=5)
    ax.set_title("Timing Error Reduction Vs Open Loop")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Reduction in |Delta tau| [%]")
    ax.set_ylim(-5.0, 100.5)
    ax.grid(alpha=0.25)
    median_reduction_pct = float(comparison_metrics.get("median_timing_error_reduction_pct_after_1s", float("nan")))
    ax.text(
        0.02,
        0.05,
        (
            f"median reduction after 1 s: {median_reduction_pct:.2f}%"
            if np.isfinite(median_reduction_pct)
            else "median reduction after 1 s: n/a"
        ),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"},
    )

    ax = axes[1, 1]
    shade_pullin(ax)
    pid_t = np.asarray(pid["t_s"])
    pid_dac = to_float_series(pid_rows, "dac_code_cmd")
    pid_pullin_lpf = to_float_series(pid_rows, "pullin_error_lpf_hz")
    ax.plot(pid_t, pid_dac, color="#264653", linewidth=2.0)
    ax.set_title("Closed-Loop Actuation And Pull-In Error")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("DAC code", color="#264653")
    ax.tick_params(axis="y", labelcolor="#264653")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(pid_t, pid_pullin_lpf, color="#e76f51", linewidth=1.75)
    ax2.axhline(0.0, color="#999999", linewidth=1.0, alpha=0.6)
    ax2.set_ylabel("LPF pull-in error [Hz]", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51")

    fig.suptitle(
        f"Open vs Closed Loop Timing Comparison\n{run_label}\n"
        "Shaded region marks the closed-loop pull-in window; lower |Delta tau| means lower inferred timing error.",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "monitor": mon_metrics,
        "closed_loop": pid_metrics,
        "comparison": comparison_metrics,
    }


def infer_paths(run_dir: Path, nsamp: int) -> dict[str, Path]:
    return {
        "monitor_csv": run_dir / "monitor" / f"nsamp_{nsamp}" / f"monitor_nsamp{nsamp}.csv",
        "monitor_summary": run_dir / "monitor" / f"nsamp_{nsamp}" / f"monitor_nsamp{nsamp}_summary.json",
        "pid_csv": run_dir / "pid" / f"nsamp_{nsamp}" / f"pid_nsamp{nsamp}.csv",
        "pid_summary": run_dir / "pid" / f"nsamp_{nsamp}" / f"pid_nsamp{nsamp}_summary.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot saved open-loop vs closed-loop timing comparison.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--nsamp", type=int, default=32768)
    parser.add_argument("--output-name", type=str, default="open_closed_timing_dashboard.png")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    paths = infer_paths(run_dir, int(args.nsamp))
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {key}: {path}")

    monitor_rows = load_csv_rows(paths["monitor_csv"])
    pid_rows = load_csv_rows(paths["pid_csv"])
    monitor_summary = load_summary(paths["monitor_summary"])
    pid_summary = load_summary(paths["pid_summary"])

    comparison_dir = run_dir / "comparison"
    dashboard_path = comparison_dir / str(args.output_name)
    metrics = plot_dashboard(
        output_path=dashboard_path,
        run_label=run_dir.name,
        monitor_rows=monitor_rows,
        pid_rows=pid_rows,
        monitor_summary=monitor_summary,
        pid_summary=pid_summary,
    )

    write_json(comparison_dir / "timing_proxy_metrics.json", metrics)
    write_markdown(
        comparison_dir / "timing_proxy_report.md",
        run_dir,
        metrics["monitor"],
        metrics["closed_loop"],
        metrics.get("comparison"),
    )
    print(f"Wrote dashboard: {dashboard_path}")
    print(f"Wrote metrics: {comparison_dir / 'timing_proxy_metrics.json'}")
    print(f"Wrote report: {comparison_dir / 'timing_proxy_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
