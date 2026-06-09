#!/usr/bin/env python3
"""Evaluate saved NotebookBased runs using settled-window phase metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def to_float_series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        try:
            vals.append(float(raw))
        except Exception:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float64)


def first_finite_series(rows: list[dict[str, str]], *keys: str) -> np.ndarray:
    for key in keys:
        arr = to_float_series(rows, key)
        if np.any(np.isfinite(arr)):
            return arr
    return np.full(len(rows), np.nan, dtype=np.float64)


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    x0 = xv - np.mean(xv)
    denom = float(np.sum(x0 * x0))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x0 * (yv - np.mean(yv))) / denom)


def linear_fit_rms(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    slope = linear_slope(xv, yv)
    if not np.isfinite(slope):
        return float("nan")
    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))
    fit = (slope * (xv - x_mean)) + y_mean
    resid = yv - fit
    return float(np.sqrt(np.mean(resid * resid)))


def bridge_small_bad_gaps(mask: np.ndarray, t_s: np.ndarray, max_bad_gap_s: float) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if out.size == 0 or max_bad_gap_s <= 0.0:
        return out
    i = 0
    while i < out.size:
        if out[i]:
            i += 1
            continue
        j = i
        while j < out.size and (not out[j]):
            j += 1
        left = i - 1
        right = j
        if left >= 0 and right < out.size and out[left] and out[right]:
            gap_dur = float(t_s[right] - t_s[left])
            if np.isfinite(gap_dur) and gap_dur <= float(max_bad_gap_s):
                out[i:j] = True
        i = j
    return out


def infer_paths(run_dir: Path, nsamp: int) -> dict[str, Path]:
    return {
        "monitor_csv": run_dir / "monitor" / f"nsamp_{nsamp}" / f"monitor_nsamp{nsamp}.csv",
        "monitor_summary": run_dir / "monitor" / f"nsamp_{nsamp}" / f"monitor_nsamp{nsamp}_summary.json",
        "pid_csv": run_dir / "pid" / f"nsamp_{nsamp}" / f"pid_nsamp{nsamp}.csv",
        "pid_summary": run_dir / "pid" / f"nsamp_{nsamp}" / f"pid_nsamp{nsamp}_summary.json",
    }


def to_quality_mask(rows: list[dict[str, str]]) -> np.ndarray:
    vals: list[bool] = []
    for row in rows:
        action = str(row.get("action", "")).strip().lower()
        vals.append((not action.startswith("hold_low_")) and (action != "hold_invalid_kv"))
    return np.asarray(vals, dtype=bool)


def to_phase_stage_mask(rows: list[dict[str, str]]) -> np.ndarray:
    vals = [str(row.get("stage", "")).strip().lower() == "phase_lock" for row in rows]
    return np.asarray(vals, dtype=bool)


@dataclass
class SettledMetrics:
    primary_phase_metric_name: str
    locked: bool
    lock_reason: str
    first_good_region_start_s: float
    first_phase_lock_stage_s: float
    settled_start_s: float
    settled_start_idx: int
    best_good_dwell_s: float
    post_lock_duration_s: float
    steady_state_drift_ns_per_s: float
    steady_state_drift_samples_per_s: float
    steady_state_timing_rms_ns: float
    steady_state_mean_abs_phase_err_primary_rad: float
    steady_state_median_abs_phase_err_primary_rad: float
    steady_state_mean_abs_phase_err_filt_rad: float
    steady_state_median_abs_phase_err_filt_rad: float
    steady_state_mean_abs_phase_err_accepted_raw_rad: float
    steady_state_median_abs_phase_err_accepted_raw_rad: float
    steady_state_primary_phase_band_fraction: float
    steady_state_phase_band_fraction: float
    steady_state_accepted_raw_phase_band_fraction: float
    steady_state_quality_fraction: float
    steady_state_phase_stage_fraction: float
    steady_state_good_fraction: float
    steady_state_median_coherence: float
    steady_state_mean_coherence: float
    max_abs_phase_err_before_settle_rad: float
    max_abs_phase_err_after_settle_rad: float
    max_abs_observer_tau_err_ns_after_settle: float
    full_run_primary_phase_band_fraction: float
    full_run_phase_band_fraction: float
    full_run_accepted_raw_phase_band_fraction: float
    full_run_quality_fraction: float
    full_run_phase_stage_fraction: float
    points_after_settle: int


def detect_settled_metrics(
    *,
    rows: list[dict[str, str]],
    sample_rate_hz: float,
    phase_band_rad: float,
    min_lock_dwell_s: float,
    max_bad_gap_s: float,
    require_quality_ok: bool,
    coh_min: float | None,
    require_phase_lock_stage: bool,
) -> SettledMetrics:
    t_s = first_finite_series(rows, "elapsed_s")
    phase_err_filt = first_finite_series(rows, "phase_err_filt_rad", "control_phase_err_filt_rad")
    phase_err_accepted_raw = first_finite_series(rows, "phase_err_accepted_raw_rad")
    use_accepted_raw_primary = bool(np.any(np.isfinite(phase_err_accepted_raw)))
    primary_phase_metric_name = "accepted_raw" if use_accepted_raw_primary else "filtered"
    phase_err_primary = (
        phase_err_accepted_raw if use_accepted_raw_primary else phase_err_filt
    )
    coherence = first_finite_series(rows, "control_coherence", "coherence", "observer_coherence")
    quality_ok = to_quality_mask(rows)
    phase_stage = to_phase_stage_mask(rows)
    observer_tau_ns = first_finite_series(rows, "observer_tau_err_ns")
    phase_reference_valid = to_float_series(rows, "phase_reference_valid")
    phase_reference_mask = (
        np.asarray(phase_reference_valid >= 0.5, dtype=bool)
        if np.any(np.isfinite(phase_reference_valid))
        else np.ones(len(rows), dtype=bool)
    )

    finite_phase_primary = np.isfinite(t_s) & np.isfinite(phase_err_primary)
    finite_phase = np.isfinite(t_s) & np.isfinite(phase_err_filt)
    finite_t = np.isfinite(t_s)
    primary_phase_in_band = finite_phase_primary & (
        np.abs(phase_err_primary) <= float(phase_band_rad)
    )
    phase_in_band = finite_phase & (np.abs(phase_err_filt) <= float(phase_band_rad))
    finite_phase_accepted_raw = np.isfinite(t_s) & np.isfinite(phase_err_accepted_raw)
    accepted_raw_phase_in_band = finite_phase_accepted_raw & (
        np.abs(phase_err_accepted_raw) <= float(phase_band_rad)
    )
    good_mask = np.copy(primary_phase_in_band)
    good_mask &= phase_reference_mask
    if require_quality_ok:
        good_mask &= quality_ok
    if coh_min is not None:
        good_mask &= np.isfinite(coherence) & (coherence >= float(coh_min))
    if require_phase_lock_stage:
        good_mask &= phase_stage
    good_mask = bridge_small_bad_gaps(good_mask, t_s, float(max_bad_gap_s))

    first_phase_lock_stage_s = float("nan")
    phase_stage_idx = np.flatnonzero(phase_stage & finite_t)
    if phase_stage_idx.size:
        first_phase_lock_stage_s = float(t_s[int(phase_stage_idx[0])])

    best_good_dwell_s = 0.0
    first_good_region_start_s = float("nan")
    settled_start_s = float("nan")
    settled_start_idx = -1

    i = 0
    while i < good_mask.size:
        if not good_mask[i]:
            i += 1
            continue
        j = i + 1
        while j < good_mask.size and good_mask[j]:
            j += 1
        seg_dwell_s = float(t_s[j - 1] - t_s[i]) if j - i >= 2 else 0.0
        if np.isfinite(seg_dwell_s):
            best_good_dwell_s = max(best_good_dwell_s, seg_dwell_s)
        if settled_start_idx < 0 and seg_dwell_s >= float(min_lock_dwell_s):
            first_good_region_start_s = float(t_s[i])
            k = i
            while k < j and float(t_s[k] - t_s[i]) < float(min_lock_dwell_s):
                k += 1
            if k >= j:
                k = j - 1
            settled_start_idx = int(k)
            settled_start_s = float(t_s[settled_start_idx])
            break
        i = j

    full_run_primary_phase_band_fraction = (
        float(np.mean(primary_phase_in_band[finite_phase_primary]))
        if np.any(finite_phase_primary)
        else float("nan")
    )
    full_run_phase_band_fraction = float(np.mean(phase_in_band[finite_t])) if np.any(finite_t) else float("nan")
    full_run_accepted_raw_phase_band_fraction = (
        float(np.mean(accepted_raw_phase_in_band[finite_phase_accepted_raw]))
        if np.any(finite_phase_accepted_raw)
        else float("nan")
    )
    full_run_quality_fraction = float(np.mean(quality_ok[finite_t])) if np.any(finite_t) else float("nan")
    full_run_phase_stage_fraction = float(np.mean(phase_stage[finite_t])) if np.any(finite_t) else float("nan")
    max_abs_before = (
        float(np.nanmax(np.abs(phase_err_primary[finite_phase_primary])))
        if np.any(finite_phase_primary)
        else float("nan")
    )

    if settled_start_idx < 0:
        if require_phase_lock_stage and not np.any(phase_stage):
            lock_reason = "never_entered_phase_lock_stage"
        elif best_good_dwell_s < float(min_lock_dwell_s):
            lock_reason = "insufficient_good_dwell"
        else:
            lock_reason = "no_settled_window_found"
        return SettledMetrics(
            primary_phase_metric_name=str(primary_phase_metric_name),
            locked=False,
            lock_reason=lock_reason,
            first_good_region_start_s=float("nan"),
            first_phase_lock_stage_s=float(first_phase_lock_stage_s),
            settled_start_s=float("nan"),
            settled_start_idx=-1,
            best_good_dwell_s=float(best_good_dwell_s),
            post_lock_duration_s=0.0,
            steady_state_drift_ns_per_s=float("nan"),
            steady_state_drift_samples_per_s=float("nan"),
            steady_state_timing_rms_ns=float("nan"),
            steady_state_mean_abs_phase_err_primary_rad=float("nan"),
            steady_state_median_abs_phase_err_primary_rad=float("nan"),
            steady_state_mean_abs_phase_err_filt_rad=float("nan"),
            steady_state_median_abs_phase_err_filt_rad=float("nan"),
            steady_state_mean_abs_phase_err_accepted_raw_rad=float("nan"),
            steady_state_median_abs_phase_err_accepted_raw_rad=float("nan"),
            steady_state_primary_phase_band_fraction=float("nan"),
            steady_state_phase_band_fraction=float("nan"),
            steady_state_accepted_raw_phase_band_fraction=float("nan"),
            steady_state_quality_fraction=float("nan"),
            steady_state_phase_stage_fraction=float("nan"),
            steady_state_good_fraction=float("nan"),
            steady_state_median_coherence=float("nan"),
            steady_state_mean_coherence=float("nan"),
            max_abs_phase_err_before_settle_rad=float(max_abs_before),
            max_abs_phase_err_after_settle_rad=float("nan"),
            max_abs_observer_tau_err_ns_after_settle=float("nan"),
            full_run_primary_phase_band_fraction=float(full_run_primary_phase_band_fraction),
            full_run_phase_band_fraction=float(full_run_phase_band_fraction),
            full_run_accepted_raw_phase_band_fraction=float(full_run_accepted_raw_phase_band_fraction),
            full_run_quality_fraction=float(full_run_quality_fraction),
            full_run_phase_stage_fraction=float(full_run_phase_stage_fraction),
            points_after_settle=0,
        )

    post = slice(settled_start_idx, None)
    t_post = t_s[post]
    pe_primary_post = phase_err_primary[post]
    pe_post = phase_err_filt[post]
    pe_accepted_raw_post = phase_err_accepted_raw[post]
    coh_post = coherence[post]
    q_post = quality_ok[post]
    phase_stage_post = phase_stage[post]
    good_post = good_mask[post]
    tau_post = observer_tau_ns[post]

    finite_post_primary_phase = np.isfinite(t_post) & np.isfinite(pe_primary_post)
    finite_post_phase = np.isfinite(t_post) & np.isfinite(pe_post)
    finite_post_phase_accepted_raw = np.isfinite(t_post) & np.isfinite(pe_accepted_raw_post)
    finite_post_tau = np.isfinite(t_post) & np.isfinite(tau_post)
    post_lock_duration_s = float(max(0.0, float(t_post[-1] - t_post[0]))) if t_post.size >= 2 else 0.0
    drift_ns_per_s = linear_slope(t_post[finite_post_tau], tau_post[finite_post_tau]) if np.count_nonzero(finite_post_tau) >= 2 else float("nan")
    drift_samples_per_s = (
        float(drift_ns_per_s * float(sample_rate_hz) / 1e9) if np.isfinite(drift_ns_per_s) else float("nan")
    )

    max_abs_before = float(np.nanmax(np.abs(phase_err_primary[:settled_start_idx]))) if settled_start_idx > 0 else 0.0
    max_abs_after = (
        float(np.nanmax(np.abs(pe_primary_post[finite_post_primary_phase])))
        if np.any(finite_post_primary_phase)
        else float("nan")
    )
    max_abs_tau_after = float(np.nanmax(np.abs(tau_post[finite_post_tau]))) if np.any(finite_post_tau) else float("nan")

    return SettledMetrics(
        primary_phase_metric_name=str(primary_phase_metric_name),
        locked=True,
        lock_reason="settled_window_found",
        first_good_region_start_s=float(first_good_region_start_s),
        first_phase_lock_stage_s=float(first_phase_lock_stage_s),
        settled_start_s=float(settled_start_s),
        settled_start_idx=int(settled_start_idx),
        best_good_dwell_s=float(best_good_dwell_s),
        post_lock_duration_s=float(post_lock_duration_s),
        steady_state_drift_ns_per_s=float(drift_ns_per_s),
        steady_state_drift_samples_per_s=float(drift_samples_per_s),
        steady_state_timing_rms_ns=float(linear_fit_rms(t_post[finite_post_tau], tau_post[finite_post_tau])),
        steady_state_mean_abs_phase_err_primary_rad=float(np.nanmean(np.abs(pe_primary_post[finite_post_primary_phase]))) if np.any(finite_post_primary_phase) else float("nan"),
        steady_state_median_abs_phase_err_primary_rad=float(np.nanmedian(np.abs(pe_primary_post[finite_post_primary_phase]))) if np.any(finite_post_primary_phase) else float("nan"),
        steady_state_mean_abs_phase_err_filt_rad=float(np.nanmean(np.abs(pe_post[finite_post_phase]))) if np.any(finite_post_phase) else float("nan"),
        steady_state_median_abs_phase_err_filt_rad=float(np.nanmedian(np.abs(pe_post[finite_post_phase]))) if np.any(finite_post_phase) else float("nan"),
        steady_state_mean_abs_phase_err_accepted_raw_rad=float(np.nanmean(np.abs(pe_accepted_raw_post[finite_post_phase_accepted_raw]))) if np.any(finite_post_phase_accepted_raw) else float("nan"),
        steady_state_median_abs_phase_err_accepted_raw_rad=float(np.nanmedian(np.abs(pe_accepted_raw_post[finite_post_phase_accepted_raw]))) if np.any(finite_post_phase_accepted_raw) else float("nan"),
        steady_state_primary_phase_band_fraction=float(np.mean(np.abs(pe_primary_post[finite_post_primary_phase]) <= float(phase_band_rad))) if np.any(finite_post_primary_phase) else float("nan"),
        steady_state_phase_band_fraction=float(np.mean(np.abs(pe_post[finite_post_phase]) <= float(phase_band_rad))) if np.any(finite_post_phase) else float("nan"),
        steady_state_accepted_raw_phase_band_fraction=float(np.mean(np.abs(pe_accepted_raw_post[finite_post_phase_accepted_raw]) <= float(phase_band_rad))) if np.any(finite_post_phase_accepted_raw) else float("nan"),
        steady_state_quality_fraction=float(np.mean(q_post[np.isfinite(t_post)])) if t_post.size else float("nan"),
        steady_state_phase_stage_fraction=float(np.mean(phase_stage_post[np.isfinite(t_post)])) if t_post.size else float("nan"),
        steady_state_good_fraction=float(np.mean(good_post[np.isfinite(t_post)])) if t_post.size else float("nan"),
        steady_state_median_coherence=float(np.nanmedian(coh_post)) if coh_post.size else float("nan"),
        steady_state_mean_coherence=float(np.nanmean(coh_post)) if coh_post.size else float("nan"),
        max_abs_phase_err_before_settle_rad=float(max_abs_before),
        max_abs_phase_err_after_settle_rad=float(max_abs_after),
        max_abs_observer_tau_err_ns_after_settle=float(max_abs_tau_after),
        full_run_primary_phase_band_fraction=float(full_run_primary_phase_band_fraction),
        full_run_phase_band_fraction=float(full_run_phase_band_fraction),
        full_run_accepted_raw_phase_band_fraction=float(full_run_accepted_raw_phase_band_fraction),
        full_run_quality_fraction=float(full_run_quality_fraction),
        full_run_phase_stage_fraction=float(full_run_phase_stage_fraction),
        points_after_settle=int(np.count_nonzero(np.isfinite(t_post))),
    )


def improvement_pct(open_value: float, closed_value: float) -> float:
    if not np.isfinite(open_value) or not np.isfinite(closed_value) or abs(open_value) <= 1e-12:
        return float("nan")
    return float(100.0 * (abs(open_value) - abs(closed_value)) / abs(open_value))


def render_markdown(
    *,
    run_dir: Path,
    nsamp: int,
    output: dict[str, Any],
) -> str:
    monitor_summary = output["monitor_summary"]
    pid_summary = output["pid_summary"]
    monitor_metrics = output["monitor_phase_metrics"]
    pid_metrics = output["pid_phase_metrics"]
    comparison = output["comparison"]
    config = output["config"]

    lines = [
        "# Phase Capability Evaluation",
        "",
        f"Run directory: `{run_dir}`",
        f"nsamp: `{nsamp}`",
        "",
        "This report judges the NotebookBased run by whether it reaches a sustained, phase-bounded regime, not just whether its inferred timing drift improves.",
        "",
        "## Lock Definition",
        "",
        "- Primary decision metric: phase error",
        f"- Primary phase series used: `{pid_metrics['primary_phase_metric_name']}`",
        f"- Phase band: `|phase_err_primary| <= {config['phase_band_rad']}` rad",
        f"- Minimum dwell before declaring settled: `{config['min_lock_dwell_s']}` s",
        f"- Small bad gaps bridged up to: `{config['max_bad_gap_s']}` s",
        f"- Require action-quality gate: `{config['require_quality_ok']}`",
        f"- Extra coherence minimum: `{config['coh_min']}`",
        f"- Require controller to be in `phase_lock` stage: `{config['require_phase_lock_stage']}`",
        "",
        "## Open Baseline",
        "",
        f"- Primary phase metric used: `{monitor_metrics['primary_phase_metric_name']}`",
        f"- Full-run observer timing drift: `{float(monitor_summary.get('primary_timing_drift_samples_per_s', float('nan'))):.6f}` samples/s",
        f"- Full-run phase drift slope: `{float(monitor_summary.get('phase_drift_slope_full_hz', float('nan'))):.6f}` Hz",
        f"- Full-run primary phase-band fraction: `{monitor_metrics['full_run_primary_phase_band_fraction']}`",
        f"- Steady-state mean |primary phase error|: `{monitor_metrics['steady_state_mean_abs_phase_err_primary_rad']}` rad",
        f"- Mean |phase error|: `{float(monitor_summary.get('mean_abs_phase_err_rad', float('nan'))):.6f}` rad",
        f"- Phase-band fraction: `{monitor_metrics['full_run_phase_band_fraction']}`",
        f"- Accepted-raw phase-band fraction: `{monitor_metrics['full_run_accepted_raw_phase_band_fraction']}`",
        "",
        "## Closed-Loop Assessment",
        "",
        f"- Primary phase metric used: `{pid_metrics['primary_phase_metric_name']}`",
        f"- Entered `phase_lock` stage: `{bool(np.isfinite(pid_metrics['first_phase_lock_stage_s']))}`",
        f"- First `phase_lock` stage time: `{pid_metrics['first_phase_lock_stage_s']}` s",
        f"- Locked under this definition: `{pid_metrics['locked']}`",
        f"- Lock reason: `{pid_metrics['lock_reason']}`",
        f"- Best good dwell: `{pid_metrics['best_good_dwell_s']}` s",
        f"- Settled start: `{pid_metrics['settled_start_s']}` s",
        f"- Full-run observer timing drift: `{float(pid_summary.get('primary_timing_drift_samples_per_s', float('nan'))):.6f}` samples/s",
        f"- Full-run phase drift slope: `{float(pid_summary.get('phase_drift_slope_full_hz', float('nan'))):.6f}` Hz",
        f"- Full-run primary phase-band fraction: `{pid_metrics['full_run_primary_phase_band_fraction']}`",
        f"- Steady-state mean |primary phase error|: `{pid_metrics['steady_state_mean_abs_phase_err_primary_rad']}` rad",
        f"- Steady-state median |primary phase error|: `{pid_metrics['steady_state_median_abs_phase_err_primary_rad']}` rad",
        f"- Steady-state primary phase-band fraction: `{pid_metrics['steady_state_primary_phase_band_fraction']}`",
        f"- Mean |phase error|: `{float(pid_summary.get('mean_abs_phase_err_rad', float('nan'))):.6f}` rad",
        f"- Full-run phase-band fraction: `{pid_metrics['full_run_phase_band_fraction']}`",
        f"- Full-run accepted-raw phase-band fraction: `{pid_metrics['full_run_accepted_raw_phase_band_fraction']}`",
        f"- Full-run `phase_lock` stage fraction: `{pid_metrics['full_run_phase_stage_fraction']}`",
        f"- Steady-state drift after settle: `{pid_metrics['steady_state_drift_samples_per_s']}` samples/s",
        f"- Steady-state timing RMS: `{pid_metrics['steady_state_timing_rms_ns']}` ns",
        f"- Steady-state mean |phase error|: `{pid_metrics['steady_state_mean_abs_phase_err_filt_rad']}` rad",
        f"- Steady-state mean |accepted raw phase error|: `{pid_metrics['steady_state_mean_abs_phase_err_accepted_raw_rad']}` rad",
        f"- Steady-state accepted-raw phase-band fraction: `{pid_metrics['steady_state_accepted_raw_phase_band_fraction']}`",
        "",
        "## Comparison",
        "",
        f"- Primary phase error improvement vs open loop: `{comparison.get('primary_phase_err_improvement_pct')}`%",
        f"- Primary phase-band improvement vs open loop: `{comparison.get('primary_phase_band_improvement_abs')}`",
        f"- Timing drift improvement vs open loop: `{comparison['timing_drift_improvement_pct']}`%",
        f"- Phase drift improvement vs open loop: `{comparison['phase_drift_improvement_pct']}`%",
        f"- Mean |phase error| improvement vs open loop: `{comparison['mean_abs_phase_err_improvement_pct']}`%",
        f"- Accepted-raw phase-band improvement vs open loop: `{comparison.get('accepted_raw_phase_band_improvement_abs')}`",
        "",
        "## Interpretation",
        "",
    ]

    if bool(pid_metrics["locked"]):
        lines.extend(
            [
                "This run satisfies the settled-window lock definition, so it does show DPLL-like phase capability on the saved data.",
                "The steady-state fields above are the right place to judge how clean that lock is.",
            ]
        )
    elif str(pid_metrics["lock_reason"]) == "never_entered_phase_lock_stage":
        lines.extend(
            [
                "This run improves inferred timing drift strongly, but it never transitions into the controller's `phase_lock` stage.",
                "That means the saved data supports frequency pull-in and drift reduction, but not a demonstrated phase-locked DPLL regime yet.",
            ]
        )
    else:
        lines.extend(
            [
                "This run reduces drift, but it does not maintain a long enough in-band phase window to count as settled lock under this definition.",
                "It is behaving more like partial pull-in than a clean phase-locked DPLL.",
            ]
        )

    lines.extend(
        [
            "",
            f"Monitor summary: `{monitor_summary.get('csv_path', '')}`",
            f"Closed-loop summary: `{pid_summary.get('csv_path', '')}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved NotebookBased runs using settled phase metrics.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--nsamp", type=int, default=32768)
    parser.add_argument("--phase-band-rad", type=float, default=1.0)
    parser.add_argument("--min-lock-dwell-s", type=float, default=2.0)
    parser.add_argument("--max-bad-gap-s", type=float, default=0.05)
    parser.add_argument("--coh-min", type=float, default=0.025)
    parser.add_argument("--require-quality-ok", action="store_true", default=True)
    parser.add_argument("--ignore-quality-ok", dest="require_quality_ok", action="store_false")
    parser.add_argument("--require-phase-lock-stage", action="store_true", default=True)
    parser.add_argument("--ignore-phase-lock-stage", dest="require_phase_lock_stage", action="store_false")
    parser.add_argument("--json-name", type=str, default="phase_lock_evaluation.json")
    parser.add_argument("--report-name", type=str, default="phase_lock_evaluation.md")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_dir = args.run_dir.resolve()
    paths = infer_paths(run_dir, int(args.nsamp))
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {key}: {path}")

    monitor_rows = load_csv_rows(paths["monitor_csv"])
    pid_rows = load_csv_rows(paths["pid_csv"])
    monitor_summary = load_json(paths["monitor_summary"])
    pid_summary = load_json(paths["pid_summary"])
    sample_rate_hz = float(pid_summary.get("sample_rate_hz", monitor_summary.get("sample_rate_hz", 2.4e6)))

    monitor_metrics = detect_settled_metrics(
        rows=monitor_rows,
        sample_rate_hz=sample_rate_hz,
        phase_band_rad=float(args.phase_band_rad),
        min_lock_dwell_s=float(args.min_lock_dwell_s),
        max_bad_gap_s=float(args.max_bad_gap_s),
        require_quality_ok=bool(args.require_quality_ok),
        coh_min=float(args.coh_min) if args.coh_min is not None else None,
        require_phase_lock_stage=bool(args.require_phase_lock_stage),
    )
    pid_metrics = detect_settled_metrics(
        rows=pid_rows,
        sample_rate_hz=sample_rate_hz,
        phase_band_rad=float(args.phase_band_rad),
        min_lock_dwell_s=float(args.min_lock_dwell_s),
        max_bad_gap_s=float(args.max_bad_gap_s),
        require_quality_ok=bool(args.require_quality_ok),
        coh_min=float(args.coh_min) if args.coh_min is not None else None,
        require_phase_lock_stage=bool(args.require_phase_lock_stage),
    )

    comparison = {
        "primary_phase_metric": str(pid_metrics.primary_phase_metric_name),
        "primary_phase_err_improvement_pct": improvement_pct(
            float(monitor_metrics.steady_state_mean_abs_phase_err_primary_rad),
            float(pid_metrics.steady_state_mean_abs_phase_err_primary_rad),
        ),
        "primary_phase_band_improvement_abs": (
            float(
                pid_metrics.steady_state_primary_phase_band_fraction
                - monitor_metrics.steady_state_primary_phase_band_fraction
            )
            if (
                np.isfinite(float(pid_metrics.steady_state_primary_phase_band_fraction))
                and np.isfinite(float(monitor_metrics.steady_state_primary_phase_band_fraction))
            )
            else float("nan")
        ),
        "timing_drift_improvement_pct": improvement_pct(
            float(monitor_summary.get("primary_timing_drift_samples_per_s", float("nan"))),
            float(pid_summary.get("primary_timing_drift_samples_per_s", float("nan"))),
        ),
        "phase_drift_improvement_pct": improvement_pct(
            float(monitor_summary.get("phase_drift_slope_full_hz", float("nan"))),
            float(pid_summary.get("phase_drift_slope_full_hz", float("nan"))),
        ),
        "mean_abs_phase_err_improvement_pct": improvement_pct(
            float(monitor_summary.get("mean_abs_phase_err_rad", float("nan"))),
            float(pid_summary.get("mean_abs_phase_err_rad", float("nan"))),
        ),
        "accepted_raw_phase_band_improvement_abs": (
            float(
                pid_metrics.steady_state_accepted_raw_phase_band_fraction
                - monitor_metrics.steady_state_accepted_raw_phase_band_fraction
            )
            if (
                np.isfinite(float(pid_metrics.steady_state_accepted_raw_phase_band_fraction))
                and np.isfinite(float(monitor_metrics.steady_state_accepted_raw_phase_band_fraction))
            )
            else float("nan")
        ),
    }

    output = {
        "run_dir": str(run_dir),
        "nsamp": int(args.nsamp),
        "config": {
            "phase_band_rad": float(args.phase_band_rad),
            "min_lock_dwell_s": float(args.min_lock_dwell_s),
            "max_bad_gap_s": float(args.max_bad_gap_s),
            "coh_min": None if args.coh_min is None else float(args.coh_min),
            "require_quality_ok": bool(args.require_quality_ok),
            "require_phase_lock_stage": bool(args.require_phase_lock_stage),
        },
        "monitor_summary": monitor_summary,
        "pid_summary": pid_summary,
        "monitor_phase_metrics": asdict(monitor_metrics),
        "pid_phase_metrics": asdict(pid_metrics),
        "comparison": comparison,
    }

    comparison_dir = run_dir / "comparison"
    out_json = comparison_dir / str(args.json_name)
    out_md = comparison_dir / str(args.report_name)
    write_json(out_json, output)
    write_markdown(
        out_md,
        render_markdown(
            run_dir=run_dir,
            nsamp=int(args.nsamp),
            output=output,
        ),
    )

    print(json.dumps(output["pid_phase_metrics"], indent=2, sort_keys=True))
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
