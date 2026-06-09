#!/usr/bin/env python3
"""Evaluate sweep results using settled-window phase and timing metrics."""

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


def to_bool_series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals: list[bool] = []
    for row in rows:
        raw = str(row.get(key, "")).strip().lower()
        vals.append(raw in {"1", "true", "yes", "y"})
    return np.asarray(vals, dtype=bool)


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


@dataclass
class SettledMetrics:
    locked: bool
    first_good_region_start_s: float
    settled_start_s: float
    settled_start_idx: int
    best_good_dwell_s: float
    post_lock_duration_s: float
    steady_state_drift_ns_per_s: float
    steady_state_drift_samples_per_s: float
    steady_state_timing_rms_ns: float
    steady_state_mean_abs_phase_err_filt_rad: float
    steady_state_median_abs_phase_err_filt_rad: float
    steady_state_phase_band_fraction: float
    steady_state_quality_fraction: float
    steady_state_good_fraction: float
    steady_state_median_coherence: float
    steady_state_mean_coherence: float
    max_abs_phase_err_before_settle_rad: float
    max_abs_phase_err_after_settle_rad: float
    max_abs_observer_timing_proxy_ns_after_settle: float
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
) -> SettledMetrics:
    t_s = to_float_series(rows, "elapsed_s")
    phase_err_filt = to_float_series(rows, "control_phase_err_filt_rad")
    coherence = to_float_series(rows, "control_coherence")
    quality_ok = to_bool_series(rows, "control_quality_ok")
    observer_timing_ns = to_float_series(rows, "observer_timing_proxy_ns")

    finite_phase = np.isfinite(t_s) & np.isfinite(phase_err_filt)
    phase_in_band = finite_phase & (np.abs(phase_err_filt) <= float(phase_band_rad))
    qual_mask = np.copy(phase_in_band)
    if require_quality_ok:
        qual_mask &= quality_ok
    if coh_min is not None:
        qual_mask &= np.isfinite(coherence) & (coherence >= float(coh_min))

    good_mask = bridge_small_bad_gaps(qual_mask, t_s, float(max_bad_gap_s))

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

    if settled_start_idx < 0:
        max_abs_before = float(np.nanmax(np.abs(phase_err_filt[finite_phase]))) if np.any(finite_phase) else float("nan")
        return SettledMetrics(
            locked=False,
            first_good_region_start_s=float("nan"),
            settled_start_s=float("nan"),
            settled_start_idx=-1,
            best_good_dwell_s=float(best_good_dwell_s),
            post_lock_duration_s=0.0,
            steady_state_drift_ns_per_s=float("nan"),
            steady_state_drift_samples_per_s=float("nan"),
            steady_state_timing_rms_ns=float("nan"),
            steady_state_mean_abs_phase_err_filt_rad=float("nan"),
            steady_state_median_abs_phase_err_filt_rad=float("nan"),
            steady_state_phase_band_fraction=float("nan"),
            steady_state_quality_fraction=float("nan"),
            steady_state_good_fraction=float("nan"),
            steady_state_median_coherence=float("nan"),
            steady_state_mean_coherence=float("nan"),
            max_abs_phase_err_before_settle_rad=float(max_abs_before),
            max_abs_phase_err_after_settle_rad=float("nan"),
            max_abs_observer_timing_proxy_ns_after_settle=float("nan"),
            points_after_settle=0,
        )

    post = slice(settled_start_idx, None)
    t_post = t_s[post]
    pe_post = phase_err_filt[post]
    coh_post = coherence[post]
    q_post = quality_ok[post]
    good_post = good_mask[post]
    timing_post = observer_timing_ns[post]

    finite_post_phase = np.isfinite(t_post) & np.isfinite(pe_post)
    finite_post_timing = np.isfinite(t_post) & np.isfinite(timing_post)
    post_lock_duration_s = float(max(0.0, float(t_post[-1] - t_post[0]))) if t_post.size >= 2 else 0.0
    drift_ns_per_s = linear_slope(t_post[finite_post_timing], timing_post[finite_post_timing]) if np.count_nonzero(finite_post_timing) >= 2 else float("nan")
    drift_samples_per_s = (
        float(drift_ns_per_s * float(sample_rate_hz) / 1e9) if np.isfinite(drift_ns_per_s) else float("nan")
    )

    max_abs_before = float(np.nanmax(np.abs(phase_err_filt[:settled_start_idx]))) if settled_start_idx > 0 else 0.0
    max_abs_after = float(np.nanmax(np.abs(pe_post[finite_post_phase]))) if np.any(finite_post_phase) else float("nan")
    max_abs_timing_after = (
        float(np.nanmax(np.abs(timing_post[finite_post_timing]))) if np.any(finite_post_timing) else float("nan")
    )

    return SettledMetrics(
        locked=True,
        first_good_region_start_s=float(first_good_region_start_s),
        settled_start_s=float(settled_start_s),
        settled_start_idx=int(settled_start_idx),
        best_good_dwell_s=float(best_good_dwell_s),
        post_lock_duration_s=float(post_lock_duration_s),
        steady_state_drift_ns_per_s=float(drift_ns_per_s),
        steady_state_drift_samples_per_s=float(drift_samples_per_s),
        steady_state_timing_rms_ns=float(linear_fit_rms(t_post[finite_post_timing], timing_post[finite_post_timing])),
        steady_state_mean_abs_phase_err_filt_rad=float(np.nanmean(np.abs(pe_post[finite_post_phase]))) if np.any(finite_post_phase) else float("nan"),
        steady_state_median_abs_phase_err_filt_rad=float(np.nanmedian(np.abs(pe_post[finite_post_phase]))) if np.any(finite_post_phase) else float("nan"),
        steady_state_phase_band_fraction=float(np.mean(np.abs(pe_post[finite_post_phase]) <= float(phase_band_rad))) if np.any(finite_post_phase) else float("nan"),
        steady_state_quality_fraction=float(np.mean(q_post[np.isfinite(t_post)])) if t_post.size else float("nan"),
        steady_state_good_fraction=float(np.mean(good_post[np.isfinite(t_post)])) if t_post.size else float("nan"),
        steady_state_median_coherence=float(np.nanmedian(coh_post)) if coh_post.size else float("nan"),
        steady_state_mean_coherence=float(np.nanmean(coh_post)) if coh_post.size else float("nan"),
        max_abs_phase_err_before_settle_rad=float(max_abs_before),
        max_abs_phase_err_after_settle_rad=float(max_abs_after),
        max_abs_observer_timing_proxy_ns_after_settle=float(max_abs_timing_after),
        points_after_settle=int(np.count_nonzero(np.isfinite(t_post))),
    )


def rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
    locked = bool(item.get("locked", False))
    drift = float(item.get("steady_state_drift_samples_per_s", float("nan")))
    band = float(item.get("steady_state_phase_band_fraction", float("nan")))
    rms = float(item.get("steady_state_timing_rms_ns", float("nan")))
    settle = float(item.get("settled_start_s", float("nan")))
    if not locked:
        full_run_drift = float(item.get("full_run_observer_timing_drift_samples_per_s", float("nan")))
        best_dwell = float(item.get("best_good_dwell_s", float("nan")))
        med_coh = float(item.get("full_run_median_control_coherence", float("nan")))
        return (
            1,
            -best_dwell if np.isfinite(best_dwell) else float("inf"),
            abs(full_run_drift) if np.isfinite(full_run_drift) else float("inf"),
            -med_coh if np.isfinite(med_coh) else float("inf"),
            float("inf"),
        )
    return (
        0,
        abs(drift) if np.isfinite(drift) else float("inf"),
        -band if np.isfinite(band) else float("inf"),
        rms if np.isfinite(rms) else float("inf"),
        settle if np.isfinite(settle) else float("inf"),
    )


def render_markdown(
    *,
    sweep_root: Path,
    open_summary: dict[str, Any],
    ranked: list[dict[str, Any]],
    phase_band_rad: float,
    min_lock_dwell_s: float,
    max_bad_gap_s: float,
    require_quality_ok: bool,
    coh_min: float | None,
) -> str:
    lines = [
        "# Settled-Window Evaluation",
        "",
        f"Sweep root: `{sweep_root}`",
        "",
        "This evaluator ranks runs by settled behavior rather than whole-run slope.",
        "",
        "## Lock Definition",
        "",
        f"- Filtered control phase error band: `|phase_err_filt| <= {phase_band_rad}` rad",
        f"- Minimum dwell before declaring settled: `{min_lock_dwell_s}` s",
        f"- Small bad gaps bridged up to: `{max_bad_gap_s}` s",
        f"- Require `control_quality_ok`: `{require_quality_ok}`",
        f"- Extra coherence minimum: `{coh_min}`",
        "",
        "## Open Baseline",
        "",
        f"- Full-run observer drift: `{float(open_summary.get('observer_timing_drift_samples_per_s', float('nan'))):.6f}` samples/s",
        f"- Full-run observer drift: `{float(open_summary.get('observer_timing_drift_ns_per_s', float('nan'))):.6f}` ns/s",
        "",
        "## Ranked Results",
        "",
    ]
    for idx, item in enumerate(ranked, start=1):
        lines.extend(
            [
                f"### {idx}. igain={item['integral_gain_codes_per_rad_s']}",
                "",
                f"- Locked: `{item['locked']}`",
                f"- Settled start: `{item['settled_start_s']}` s",
                f"- Best good dwell: `{item['best_good_dwell_s']}` s",
                f"- Settled drift: `{item['steady_state_drift_samples_per_s']}` samples/s",
                f"- Settled drift: `{item['steady_state_drift_ns_per_s']}` ns/s",
                f"- Settled phase-band fraction: `{item['steady_state_phase_band_fraction']}`",
                f"- Settled timing RMS: `{item['steady_state_timing_rms_ns']}` ns",
                f"- Settled mean abs filtered phase error: `{item['steady_state_mean_abs_phase_err_filt_rad']}` rad",
                f"- Settled median coherence: `{item['steady_state_median_coherence']}`",
                f"- Summary: `{item['summary_path']}`",
                "",
            ]
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate sweep results using settled-window phase metrics.")
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--phase-band-rad", type=float, default=1.0)
    parser.add_argument("--min-lock-dwell-s", type=float, default=2.0)
    parser.add_argument("--max-bad-gap-s", type=float, default=0.05)
    parser.add_argument("--coh-min", type=float, default=None)
    parser.add_argument("--require-quality-ok", action="store_true", default=True)
    parser.add_argument("--ignore-quality-ok", dest="require_quality_ok", action="store_false")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    sweep_root = Path(args.sweep_root).resolve()
    open_summary_path = sweep_root / "monitor" / "open_loop_summary.json"
    open_csv_path = sweep_root / "monitor" / "open_loop.csv"
    open_summary = load_json(open_summary_path)
    open_rows = load_csv_rows(open_csv_path)

    closed_candidates = sorted((sweep_root / "pid").glob("igain_*/closed_loop_summary.json"))
    if not closed_candidates:
        raise RuntimeError(f"No closed-loop summaries found under {sweep_root / 'pid'}")

    sample_rate_hz = float(open_summary.get("sample_rate_hz", 2.4e6))
    ranked_items: list[dict[str, Any]] = []
    for summary_path in closed_candidates:
        summary = load_json(summary_path)
        csv_path = summary_path.parent / "closed_loop.csv"
        rows = load_csv_rows(csv_path)
        gain_from_row = float("nan")
        if rows:
            try:
                gain_from_row = float(rows[0].get("integral_gain_codes_per_rad_s", "nan"))
            except Exception:
                gain_from_row = float("nan")
        gain_from_dir = float("nan")
        try:
            gain_from_dir = float(summary_path.parent.name.replace("igain_", "").replace("p", "."))
        except Exception:
            gain_from_dir = float("nan")
        gain_value = float(summary.get("integral_gain_codes_per_rad_s", float("nan")))
        if not np.isfinite(gain_value):
            gain_value = gain_from_row
        if not np.isfinite(gain_value):
            gain_value = gain_from_dir
        settled = detect_settled_metrics(
            rows=rows,
            sample_rate_hz=sample_rate_hz,
            phase_band_rad=float(args.phase_band_rad),
            min_lock_dwell_s=float(args.min_lock_dwell_s),
            max_bad_gap_s=float(args.max_bad_gap_s),
            require_quality_ok=bool(args.require_quality_ok),
            coh_min=float(args.coh_min) if args.coh_min is not None else None,
        )
        item = {
            "integral_gain_codes_per_rad_s": float(gain_value),
            "summary_path": str(summary_path.resolve()),
            "csv_path": str(csv_path.resolve()),
            "full_run_observer_timing_drift_samples_per_s": float(summary.get("observer_timing_drift_samples_per_s", float("nan"))),
            "full_run_observer_timing_drift_ns_per_s": float(summary.get("observer_timing_drift_ns_per_s", float("nan"))),
            "full_run_mean_abs_control_phase_err_rad": float(summary.get("mean_abs_control_phase_err_rad", float("nan"))),
            "full_run_median_control_coherence": float(summary.get("median_control_coherence", float("nan"))),
            **asdict(settled),
        }
        ranked_items.append(item)

    ranked = sorted(ranked_items, key=rank_key)
    output = {
        "sweep_root": str(sweep_root),
        "config": {
            "phase_band_rad": float(args.phase_band_rad),
            "min_lock_dwell_s": float(args.min_lock_dwell_s),
            "max_bad_gap_s": float(args.max_bad_gap_s),
            "coh_min": None if args.coh_min is None else float(args.coh_min),
            "require_quality_ok": bool(args.require_quality_ok),
        },
        "open_summary": open_summary,
        "open_settled_metrics": asdict(
            detect_settled_metrics(
                rows=open_rows,
                sample_rate_hz=sample_rate_hz,
                phase_band_rad=float(args.phase_band_rad),
                min_lock_dwell_s=float(args.min_lock_dwell_s),
                max_bad_gap_s=float(args.max_bad_gap_s),
                require_quality_ok=bool(args.require_quality_ok),
                coh_min=float(args.coh_min) if args.coh_min is not None else None,
            )
        ),
        "results_ranked_by_settled_behavior": ranked,
    }

    out_json = sweep_root / "comparison" / "settled_phase_evaluation.json"
    out_md = sweep_root / "comparison" / "settled_phase_evaluation.md"
    write_json(out_json, output)
    write_markdown(
        out_md,
        render_markdown(
            sweep_root=sweep_root,
            open_summary=open_summary,
            ranked=ranked,
            phase_band_rad=float(args.phase_band_rad),
            min_lock_dwell_s=float(args.min_lock_dwell_s),
            max_bad_gap_s=float(args.max_bad_gap_s),
            require_quality_ok=bool(args.require_quality_ok),
            coh_min=float(args.coh_min) if args.coh_min is not None else None,
        ),
    )

    print(json.dumps(ranked, indent=2, sort_keys=True))
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
