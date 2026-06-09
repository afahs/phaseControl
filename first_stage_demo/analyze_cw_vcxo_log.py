#!/usr/bin/env python3
"""
Pi-safe CSV analysis and plotting for first_stage_demo/cw_vcxo_discipline.py logs.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


def parse_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def is_true_flag(value: str | None) -> bool:
    parsed = parse_float(value)
    return bool(np.isfinite(parsed) and parsed > 0.5)


def safe_text(value: str | None) -> str:
    return "" if value is None else str(value).strip()


def summary_stat_line(label: str, value: float, unit: str = "") -> str:
    if value is None or not np.isfinite(value):
        return f"{label}: nan"
    suffix = f" {unit}" if unit else ""
    return f"{label}: {value:.6f}{suffix}"


def fraction_line(label: str, numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return f"{label}: nan"
    frac_pct = 100.0 * float(numerator) / float(denominator)
    return f"{label}: {frac_pct:.3f} % ({numerator}/{denominator})"


def format_counter(counter: Counter, limit: int = 8) -> str:
    if not counter:
        return ""
    items = [f"{key}={value}" for key, value in counter.most_common(limit)]
    return ", ".join(items)


@dataclass
class CsvInfo:
    csv_path: str
    fieldnames: List[str]
    row_count: int
    duration_s: float
    mode: str
    actuator: str


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = field(default_factory=lambda: float("inf"))
    max_value: float = field(default_factory=lambda: float("-inf"))

    def add(self, value: float) -> None:
        if not np.isfinite(value):
            return
        self.count += 1
        delta = float(value) - self.mean
        self.mean += delta / float(self.count)
        delta2 = float(value) - self.mean
        self.m2 += delta * delta2
        self.min_value = min(self.min_value, float(value))
        self.max_value = max(self.max_value, float(value))

    def mean_value(self) -> float:
        return float(self.mean) if self.count > 0 else float("nan")

    def std_value(self) -> float:
        if self.count <= 0:
            return float("nan")
        return float(math.sqrt(max(self.m2 / float(self.count), 0.0)))

    def span_value(self) -> float:
        if self.count <= 0:
            return float("nan")
        return float(self.max_value - self.min_value)


@dataclass
class RegressionAccumulator:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: float, y: float) -> None:
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        self.count += 1
        self.sum_x += float(x)
        self.sum_y += float(y)
        self.sum_xx += float(x) * float(x)
        self.sum_xy += float(x) * float(y)

    def slope(self) -> float:
        if self.count < 2:
            return float("nan")
        denom = (float(self.count) * self.sum_xx) - (self.sum_x * self.sum_x)
        if abs(denom) < 1.0e-12:
            return float("nan")
        numer = (float(self.count) * self.sum_xy) - (self.sum_x * self.sum_y)
        return float(numer / denom)


class BoundedHistogramQuantile:
    def __init__(self, low: float = 0.0, high: float = math.pi, bins: int = 4096) -> None:
        self.low = float(low)
        self.high = float(high)
        self.bins = max(16, int(bins))
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.total = 0

    def add(self, value: float) -> None:
        if not np.isfinite(value):
            return
        clipped = min(max(float(value), self.low), self.high)
        span = self.high - self.low
        if span <= 0.0:
            index = 0
        else:
            index = int(((clipped - self.low) / span) * float(self.bins))
            index = min(max(index, 0), self.bins - 1)
        self.counts[index] += 1
        self.total += 1

    def percentile(self, pct: float) -> float:
        if self.total <= 0:
            return float("nan")
        target = max(1, int(math.ceil((float(pct) / 100.0) * float(self.total))))
        cumulative = np.cumsum(self.counts, dtype=np.int64)
        index = int(np.searchsorted(cumulative, target, side="left"))
        index = min(max(index, 0), self.bins - 1)
        bin_width = (self.high - self.low) / float(self.bins)
        return float(self.low + ((index + 0.5) * bin_width))


@dataclass
class PlotSeries:
    t_rel_s: List[float] = field(default_factory=list)
    unwrapped_phase_rad: List[float] = field(default_factory=list)
    f_off_hz: List[float] = field(default_factory=list)
    control_cmd: List[float] = field(default_factory=list)
    amp: List[float] = field(default_factory=list)
    coherence: List[float] = field(default_factory=list)
    stride: int = 1


class PlotSampler:
    def __init__(self, row_count: int, max_points: int) -> None:
        self.stride = max(1, int(math.ceil(float(max(1, row_count)) / float(max(1, max_points)))))
        self.series = PlotSeries(stride=self.stride)
        self.last_seen_index = -1
        self.last_sampled_index = -1
        self.last_seen_sample: Optional[tuple[float, float, float, float, float, float]] = None

    def observe_row(self, row_index: int, row: dict[str, str]) -> None:
        sample = (
            parse_float(row.get("t_rel_s")),
            parse_float(row.get("unwrapped_phase_rad")),
            parse_float(row.get("f_off_hz")),
            parse_float(row.get("control_cmd")),
            parse_float(row.get("amp")),
            parse_float(row.get("coherence")),
        )
        self.last_seen_index = int(row_index)
        self.last_seen_sample = sample
        if row_index % self.stride == 0:
            self._append(sample)
            self.last_sampled_index = int(row_index)

    def finalize(self) -> PlotSeries:
        if self.last_seen_sample is not None and self.last_sampled_index != self.last_seen_index:
            self._append(self.last_seen_sample)
            self.last_sampled_index = self.last_seen_index
        return self.series

    def _append(self, sample: tuple[float, float, float, float, float, float]) -> None:
        t_rel_s, phase, f_off, cmd, amp, coh = sample
        self.series.t_rel_s.append(float(t_rel_s))
        self.series.unwrapped_phase_rad.append(float(phase))
        self.series.f_off_hz.append(float(f_off))
        self.series.control_cmd.append(float(cmd))
        self.series.amp.append(float(amp))
        self.series.coherence.append(float(coh))


@dataclass
class AnalysisSummary:
    info: CsvInfo
    plot_series: Optional[PlotSeries]
    f_off_stats: RunningStats
    abs_f_off_stats: RunningStats
    late_f_off_stats: RunningStats
    fine_lock_abs_f_off_stats: RunningStats
    amp_stats: RunningStats
    coherence_stats: RunningStats
    control_cmd_stats: RunningStats
    control_meas_abs_f_off_stats: RunningStats
    fine_lock_abs_control_error_stats: RunningStats
    phase_hold_abs_control_error_stats: RunningStats
    phase_error_abs_stats: RunningStats
    late_phase_error_abs_stats: RunningStats
    phase_error_hist: BoundedHistogramQuantile
    overall_phase_fit: RegressionAccumulator
    fine_lock_phase_fit: RegressionAccumulator
    phase_hold_phase_fit: RegressionAccumulator
    recapture_abs_error_stats: RunningStats
    recapture_estimate_max_abs_error: float
    stage_counts: Counter
    status_counts: Counter
    switch_event_counts: Counter
    recapture_active_rows: int
    phase_hold_active_rows: int
    recapture_valid_estimates: int
    recapture_invalid_estimates: int


def inspect_csv(csv_path: str) -> CsvInfo:
    row_count = 0
    duration_s = float("nan")
    mode = ""
    actuator = ""

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header row.")
        for row in reader:
            if row_count == 0:
                mode = safe_text(row.get("mode"))
                actuator = safe_text(row.get("actuator"))
            row_count += 1
            t_rel_s = parse_float(row.get("t_rel_s"))
            if np.isfinite(t_rel_s):
                duration_s = float(t_rel_s)

    return CsvInfo(
        csv_path=os.path.abspath(csv_path),
        fieldnames=list(reader.fieldnames),
        row_count=int(row_count),
        duration_s=float(duration_s),
        mode=mode,
        actuator=actuator,
    )


def analyze_csv(
    csv_path: str,
    info: Optional[CsvInfo] = None,
    plot_points: Optional[int] = None,
) -> AnalysisSummary:
    if info is None:
        info = inspect_csv(csv_path)

    late_threshold_s = 0.5 * float(info.duration_s) if np.isfinite(info.duration_s) else float("inf")
    plot_sampler = PlotSampler(info.row_count, plot_points) if plot_points is not None else None

    f_off_stats = RunningStats()
    abs_f_off_stats = RunningStats()
    late_f_off_stats = RunningStats()
    fine_lock_abs_f_off_stats = RunningStats()
    amp_stats = RunningStats()
    coherence_stats = RunningStats()
    control_cmd_stats = RunningStats()
    control_meas_abs_f_off_stats = RunningStats()
    fine_lock_abs_control_error_stats = RunningStats()
    phase_hold_abs_control_error_stats = RunningStats()
    phase_error_abs_stats = RunningStats()
    late_phase_error_abs_stats = RunningStats()
    phase_error_hist = BoundedHistogramQuantile()
    overall_phase_fit = RegressionAccumulator()
    fine_lock_phase_fit = RegressionAccumulator()
    phase_hold_phase_fit = RegressionAccumulator()
    recapture_abs_error_stats = RunningStats()
    recapture_estimate_max_abs_error = float("nan")
    stage_counts: Counter = Counter()
    status_counts: Counter = Counter()
    switch_event_counts: Counter = Counter()
    recapture_active_rows = 0
    phase_hold_active_rows = 0
    recapture_valid_estimates = 0
    recapture_invalid_estimates = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header row.")
        for row_index, row in enumerate(reader):
            if plot_sampler is not None:
                plot_sampler.observe_row(row_index, row)

            t_rel_s = parse_float(row.get("t_rel_s"))
            f_off_hz = parse_float(row.get("f_off_hz"))
            amp = parse_float(row.get("amp"))
            coherence = parse_float(row.get("coherence"))
            control_cmd = parse_float(row.get("control_cmd"))
            control_meas_f_off_hz = parse_float(row.get("control_meas_f_off_hz"))
            control_error_hz = parse_float(row.get("control_error_hz"))
            phase_error_rad = parse_float(row.get("phase_error_rad"))
            unwrapped_phase_rad = parse_float(row.get("unwrapped_phase_rad"))
            recapture_estimated_error_hz = parse_float(row.get("recapture_estimated_error_hz"))
            recapture_estimate_valid = parse_float(row.get("recapture_estimate_valid"))

            control_stage = safe_text(row.get("control_stage"))
            status = safe_text(row.get("status"))
            switch_event = safe_text(row.get("control_stage_switch_event"))
            recapture_active = is_true_flag(row.get("recapture_active"))
            phase_hold_active = is_true_flag(row.get("phase_hold_active"))

            f_off_stats.add(f_off_hz)
            abs_f_off_stats.add(abs(f_off_hz) if np.isfinite(f_off_hz) else float("nan"))
            amp_stats.add(amp)
            coherence_stats.add(coherence)
            control_cmd_stats.add(control_cmd)
            control_meas_abs_f_off_stats.add(
                abs(control_meas_f_off_hz)
                if np.isfinite(control_meas_f_off_hz)
                else float("nan")
            )

            if np.isfinite(t_rel_s) and t_rel_s >= late_threshold_s:
                late_f_off_stats.add(f_off_hz)

            if np.isfinite(phase_error_rad):
                abs_phase_error = abs(float(phase_error_rad))
                phase_error_abs_stats.add(abs_phase_error)
                phase_error_hist.add(abs_phase_error)
                if np.isfinite(t_rel_s) and t_rel_s >= late_threshold_s:
                    late_phase_error_abs_stats.add(abs_phase_error)

            if np.isfinite(t_rel_s) and np.isfinite(unwrapped_phase_rad):
                overall_phase_fit.add(t_rel_s, unwrapped_phase_rad)
                if control_stage == "fine_lock":
                    fine_lock_phase_fit.add(t_rel_s, unwrapped_phase_rad)
                if phase_hold_active:
                    phase_hold_phase_fit.add(t_rel_s, unwrapped_phase_rad)

            if control_stage:
                stage_counts[control_stage] += 1
                if control_stage == "fine_lock":
                    fine_lock_abs_f_off_stats.add(
                        abs(f_off_hz) if np.isfinite(f_off_hz) else float("nan")
                    )
                    fine_lock_abs_control_error_stats.add(
                        abs(control_error_hz)
                        if np.isfinite(control_error_hz)
                        else float("nan")
                    )
                if phase_hold_active:
                    phase_hold_abs_control_error_stats.add(
                        abs(control_error_hz)
                        if np.isfinite(control_error_hz)
                        else float("nan")
                    )

            if status:
                status_counts[status] += 1
            if switch_event:
                switch_event_counts[switch_event] += 1

            if recapture_active:
                recapture_active_rows += 1
            if phase_hold_active:
                phase_hold_active_rows += 1

            if np.isfinite(recapture_estimated_error_hz):
                abs_recapture_error = abs(float(recapture_estimated_error_hz))
                recapture_abs_error_stats.add(abs_recapture_error)
                if not np.isfinite(recapture_estimate_max_abs_error):
                    recapture_estimate_max_abs_error = abs_recapture_error
                else:
                    recapture_estimate_max_abs_error = max(
                        recapture_estimate_max_abs_error,
                        abs_recapture_error,
                    )

            is_recapture_estimate_row = status in (
                "recapture_update",
                "recapture_complete",
                "recapture_reject_peak",
            )
            if is_recapture_estimate_row and np.isfinite(recapture_estimate_valid):
                if recapture_estimate_valid > 0.5:
                    recapture_valid_estimates += 1
                else:
                    recapture_invalid_estimates += 1

    plot_series = plot_sampler.finalize() if plot_sampler is not None else None
    return AnalysisSummary(
        info=info,
        plot_series=plot_series,
        f_off_stats=f_off_stats,
        abs_f_off_stats=abs_f_off_stats,
        late_f_off_stats=late_f_off_stats,
        fine_lock_abs_f_off_stats=fine_lock_abs_f_off_stats,
        amp_stats=amp_stats,
        coherence_stats=coherence_stats,
        control_cmd_stats=control_cmd_stats,
        control_meas_abs_f_off_stats=control_meas_abs_f_off_stats,
        fine_lock_abs_control_error_stats=fine_lock_abs_control_error_stats,
        phase_hold_abs_control_error_stats=phase_hold_abs_control_error_stats,
        phase_error_abs_stats=phase_error_abs_stats,
        late_phase_error_abs_stats=late_phase_error_abs_stats,
        phase_error_hist=phase_error_hist,
        overall_phase_fit=overall_phase_fit,
        fine_lock_phase_fit=fine_lock_phase_fit,
        phase_hold_phase_fit=phase_hold_phase_fit,
        recapture_abs_error_stats=recapture_abs_error_stats,
        recapture_estimate_max_abs_error=recapture_estimate_max_abs_error,
        stage_counts=stage_counts,
        status_counts=status_counts,
        switch_event_counts=switch_event_counts,
        recapture_active_rows=recapture_active_rows,
        phase_hold_active_rows=phase_hold_active_rows,
        recapture_valid_estimates=recapture_valid_estimates,
        recapture_invalid_estimates=recapture_invalid_estimates,
    )


def slope_hz_equivalent(fit: RegressionAccumulator) -> float:
    slope_rad_per_s = fit.slope()
    if not np.isfinite(slope_rad_per_s):
        return float("nan")
    return float(slope_rad_per_s / (2.0 * math.pi))


def print_summary(summary: AnalysisSummary) -> None:
    info = summary.info
    total_rows = int(info.row_count)
    fine_lock_rows = int(summary.stage_counts.get("fine_lock", 0))
    pull_in_rows = int(summary.stage_counts.get("pull_in", 0))

    print(f"rows: {total_rows}")
    print(f"mode: {info.mode or 'unknown'}")
    print(f"actuator: {info.actuator or 'unknown'}")
    print(summary_stat_line("duration", info.duration_s, "s"))
    print(summary_stat_line("mean_f_off", summary.f_off_stats.mean_value(), "Hz"))
    print(summary_stat_line("mean_abs_f_off", summary.abs_f_off_stats.mean_value(), "Hz"))
    print(summary_stat_line("std_f_off", summary.f_off_stats.std_value(), "Hz"))
    print(summary_stat_line("late_mean_f_off", summary.late_f_off_stats.mean_value(), "Hz"))
    print(summary_stat_line("overall_phase_slope", slope_hz_equivalent(summary.overall_phase_fit), "Hz"))
    print(
        summary_stat_line(
            "fine_lock_phase_slope",
            slope_hz_equivalent(summary.fine_lock_phase_fit),
            "Hz",
        )
    )
    print(
        summary_stat_line(
            "phase_hold_phase_slope",
            slope_hz_equivalent(summary.phase_hold_phase_fit),
            "Hz",
        )
    )

    if summary.phase_error_abs_stats.count > 0:
        print(
            summary_stat_line(
                "mean_abs_phase_error",
                summary.phase_error_abs_stats.mean_value(),
                "rad",
            )
        )
        print(
            summary_stat_line(
                "p95_abs_phase_error",
                summary.phase_error_hist.percentile(95.0),
                "rad",
            )
        )
        print(
            summary_stat_line(
                "late_mean_abs_phase_error",
                summary.late_phase_error_abs_stats.mean_value(),
                "rad",
            )
        )

    print(summary_stat_line("mean_amp", summary.amp_stats.mean_value()))
    print(summary_stat_line("mean_coherence", summary.coherence_stats.mean_value()))
    print(summary_stat_line("control_span", summary.control_cmd_stats.span_value()))
    print(f"pull_in_rows: {pull_in_rows}")
    print(f"fine_lock_rows: {fine_lock_rows}")
    print(fraction_line("fine_lock_fraction", fine_lock_rows, total_rows))
    print(
        summary_stat_line(
            "fine_lock_mean_abs_f_off",
            summary.fine_lock_abs_f_off_stats.mean_value(),
            "Hz",
        )
    )
    print(
        summary_stat_line(
            "control_update_mean_abs_meas_f_off",
            summary.control_meas_abs_f_off_stats.mean_value(),
            "Hz",
        )
    )
    print(
        summary_stat_line(
            "fine_lock_mean_abs_control_error",
            summary.fine_lock_abs_control_error_stats.mean_value(),
            "Hz",
        )
    )
    print(
        summary_stat_line(
            "phase_hold_mean_abs_control_error",
            summary.phase_hold_abs_control_error_stats.mean_value(),
            "Hz",
        )
    )
    print(fraction_line("phase_hold_active_fraction", summary.phase_hold_active_rows, total_rows))
    print(fraction_line("recapture_active_fraction", summary.recapture_active_rows, total_rows))

    if summary.status_counts:
        print(f"recapture_update_count: {summary.status_counts.get('recapture_update', 0)}")
        print(f"recapture_complete_count: {summary.status_counts.get('recapture_complete', 0)}")
        print(f"recapture_reject_peak_count: {summary.status_counts.get('recapture_reject_peak', 0)}")

    if summary.recapture_abs_error_stats.count > 0:
        print(
            summary_stat_line(
                "mean_abs_recapture_est_error",
                summary.recapture_abs_error_stats.mean_value(),
                "Hz",
            )
        )
        print(
            summary_stat_line(
                "max_abs_recapture_est_error",
                summary.recapture_estimate_max_abs_error,
                "Hz",
            )
        )

    if (summary.recapture_valid_estimates + summary.recapture_invalid_estimates) > 0:
        print(f"valid_recapture_estimates: {summary.recapture_valid_estimates}")
        print(f"invalid_recapture_estimates: {summary.recapture_invalid_estimates}")

    if summary.switch_event_counts:
        print(f"stage_switch_events: {format_counter(summary.switch_event_counts, limit=6)}")
    if summary.status_counts:
        print(f"top_status_counts: {format_counter(summary.status_counts, limit=8)}")


def render_plot_series(
    csv_path: str,
    info: CsvInfo,
    plot_series: PlotSeries,
    out_png: Optional[str] = None,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_png is None:
        out_png = os.path.splitext(os.path.abspath(csv_path))[0] + "_plots.png"
    out_png = os.path.abspath(out_png)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(plot_series.t_rel_s, dtype=np.float64)
    phase = np.asarray(plot_series.unwrapped_phase_rad, dtype=np.float64)
    f_off = np.asarray(plot_series.f_off_hz, dtype=np.float64)
    cmd = np.asarray(plot_series.control_cmd, dtype=np.float64)
    amp = np.asarray(plot_series.amp, dtype=np.float64)
    coh = np.asarray(plot_series.coherence, dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    title = (
        f"CW VCXO Discipline Log: mode={info.mode or 'unknown'}, actuator={info.actuator or 'unknown'}"
        f" | plotted={t.size}/{info.row_count}, stride={plot_series.stride}"
    )

    axes[0].plot(t, phase, lw=1.0)
    axes[0].set_ylabel("Phase [rad]")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, f_off, lw=1.0)
    axes[1].axhline(0.0, color="k", lw=0.8, alpha=0.5)
    axes[1].set_ylabel("f_off [Hz]")
    axes[1].grid(True, alpha=0.3)

    axes[2].step(t, cmd, where="post", lw=1.0)
    axes[2].set_ylabel("Control Cmd")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, amp, lw=1.0, label="amp")
    axes[3].plot(t, coh, lw=1.0, label="coherence")
    axes[3].set_ylabel("Quality")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def create_vcxo_sweep_plot(csv_path: str, info: Optional[CsvInfo] = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if info is None:
        info = inspect_csv(csv_path)
    if info.mode != "vcxo_test":
        return

    stats_by_step: dict[int, dict[str, RunningStats]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_idx = parse_float(row.get("test_step_index"))
            step_cmd = parse_float(row.get("test_step_cmd"))
            f_off = parse_float(row.get("f_off_hz"))
            valid_for_fit = parse_float(row.get("test_valid_for_fit"))
            if not (
                np.isfinite(step_idx)
                and np.isfinite(step_cmd)
                and np.isfinite(f_off)
                and np.isfinite(valid_for_fit)
                and valid_for_fit > 0.5
            ):
                continue
            step_key = int(round(float(step_idx)))
            bucket = stats_by_step.setdefault(
                step_key,
                {"cmd": RunningStats(), "f_off": RunningStats()},
            )
            bucket["cmd"].add(step_cmd)
            bucket["f_off"].add(f_off)

    if len(stats_by_step) < 2:
        return

    unique_steps = sorted(stats_by_step.keys())
    x = np.asarray([stats_by_step[idx]["cmd"].mean_value() for idx in unique_steps], dtype=np.float64)
    y = np.asarray([stats_by_step[idx]["f_off"].mean_value() for idx in unique_steps], dtype=np.float64)
    yerr = np.asarray([stats_by_step[idx]["f_off"].std_value() for idx in unique_steps], dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 2:
        return

    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    xfit = np.linspace(float(np.min(x[valid])), float(np.max(x[valid])), 200)
    out_png = os.path.splitext(os.path.abspath(csv_path))[0] + "_sweep.png"

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label="step means")
    ax.plot(xfit, slope * xfit + intercept, label=f"fit: {slope:.6f} Hz/cmd")
    ax.set_xlabel("Control Command")
    ax.set_ylabel("Mean f_off [Hz]")
    ax.set_title("VCXO Characterization Sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def create_plots(
    csv_path: str,
    out_png: Optional[str] = None,
    plot_points: int = 4000,
) -> str:
    info = inspect_csv(csv_path)
    summary = analyze_csv(csv_path, info=info, plot_points=plot_points)
    if summary.plot_series is None:
        raise RuntimeError("Plot series was not generated.")
    out_path = render_plot_series(csv_path, info, summary.plot_series, out_png)
    create_vcxo_sweep_plot(csv_path, info=info)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pi-safe analysis and plotting for first-stage CW VCXO discipline CSV logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", help="CSV file produced by cw_vcxo_discipline.py")
    parser.add_argument(
        "--out-png",
        default=None,
        help="Output PNG path. Defaults to <csv>_plots.png",
    )
    parser.add_argument(
        "--plot-points",
        type=int,
        default=4000,
        help="Maximum number of decimated samples retained for plotting.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the streaming summary and skip plot generation.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.plot_points <= 0:
        print("--plot-points must be positive.")
        return 2

    info = inspect_csv(args.csv_path)
    summary = analyze_csv(
        args.csv_path,
        info=info,
        plot_points=None if args.summary_only else args.plot_points,
    )
    print_summary(summary)

    if args.summary_only:
        return 0

    if summary.plot_series is None:
        print("Plot generation failed: plot series was not collected.")
        return 1

    try:
        out_png = render_plot_series(args.csv_path, info, summary.plot_series, args.out_png)
        create_vcxo_sweep_plot(args.csv_path, info=info)
    except Exception as exc:
        print(f"Plot generation failed: {exc}")
        return 1

    print(f"Saved plots to {out_png}")
    print(f"Plot samples retained: {len(summary.plot_series.t_rel_s)}")
    print(f"Plot decimation stride: {summary.plot_series.stride}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
