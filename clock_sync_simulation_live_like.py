from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class GateConfig:
    coh_min: float = 0.02
    pullin_quality_min: float = 0.10
    phase_enter_max_err_rad: float = 0.5
    pullin_switch_hz: float = 100.0
    pullin_exit_hz: float = 300.0
    pullin_enter_dwell_updates: int = 3
    pullin_exit_dwell_updates: int = 2


LIVE_CASE_PATHS: dict[str, Path] = {
    "live_replay_never_enters": ROOT
    / "NotebookBased/runs/20260402_170940_live_hybrid_one_min_self_observer/pid/nsamp_32768/pid_nsamp32768.csv",
    "live_replay_brief_entries": ROOT
    / "NotebookBased/runs/20260402_184922_live_hybrid_one_min_open_closed_dwell3/pid/nsamp_32768/pid_nsamp32768.csv",
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _float_column(rows: list[dict[str, str]], key: str, fallback_key: str | None = None) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        if (not raw) and fallback_key is not None:
            raw = row.get(fallback_key, "")
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _int_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values: list[int] = []
    for row in rows:
        raw = row.get(key, "0")
        try:
            values.append(int(float(raw)))
        except (TypeError, ValueError):
            values.append(0)
    return np.asarray(values, dtype=np.int64)


def _stage_column(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([str(row.get("stage", "")) for row in rows], dtype=object)


def _median_elapsed_step(elapsed_s: np.ndarray) -> float:
    if elapsed_s.size < 2:
        return float("nan")
    diffs = np.diff(elapsed_s)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
    return float(np.median(diffs))


def load_live_case(label: str, path: Path) -> dict[str, Any]:
    rows = _load_rows(path)
    elapsed_s = _float_column(rows, "elapsed_s")
    stage = _stage_column(rows)
    actual_in_phase = stage == "phase_lock"
    return {
        "label": label,
        "kind": "live_replay",
        "source_path": str(path),
        "elapsed_s": elapsed_s,
        "dt_s": _median_elapsed_step(elapsed_s),
        "control_coherence": _float_column(rows, "control_coherence", fallback_key="coherence"),
        "control_pullin_quality": _float_column(
            rows, "control_pullin_quality", fallback_key="pullin_quality"
        ),
        "phase_err_filt_rad": _float_column(rows, "phase_err_filt_rad"),
        "pullin_error_lpf_hz": _float_column(rows, "pullin_error_lpf_hz"),
        "actual_in_phase": actual_in_phase,
        "actual_stage_switch_events": _int_column(rows, "stage_switch_event"),
        "actual_actions": np.asarray([str(row.get("action", "")) for row in rows], dtype=object),
    }


def make_toy_case(
    *,
    n_updates: int,
    dt_s: float,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_updates, dtype=np.float64)
    elapsed_s = idx * float(dt_s)
    control_coherence = np.clip(
        0.965 + 0.015 * np.sin(idx / 31.0) + rng.normal(scale=0.008, size=n_updates),
        0.0,
        1.0,
    )
    control_pullin_quality = np.clip(0.992 + rng.normal(scale=0.002, size=n_updates), 0.0, 1.0)
    phase_err_filt_rad = (
        0.46 * np.exp(-idx / 28.0)
        + 0.03 * np.sin(idx / 12.0)
        + rng.normal(scale=0.03, size=n_updates)
    )
    phase_err_filt_rad = np.clip(phase_err_filt_rad, -np.pi, np.pi)
    pullin_error_lpf_hz = 75.0 * np.exp(-idx / 16.0) + rng.normal(scale=3.0, size=n_updates)
    return {
        "label": "toy_phase_only_near_lock",
        "kind": "toy",
        "source_path": "synthetic",
        "elapsed_s": elapsed_s,
        "dt_s": float(dt_s),
        "control_coherence": control_coherence,
        "control_pullin_quality": control_pullin_quality,
        "phase_err_filt_rad": phase_err_filt_rad,
        "pullin_error_lpf_hz": pullin_error_lpf_hz,
        "actual_in_phase": None,
        "actual_stage_switch_events": None,
        "actual_actions": None,
    }


def longest_true_streak(mask: np.ndarray) -> tuple[int, int]:
    best_len = 0
    best_start = -1
    cur_len = 0
    cur_start = 0
    for idx, value in enumerate(mask.astype(bool)):
        if value:
            if cur_len == 0:
                cur_start = idx
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    return best_len, best_start


def emulate_phase_mode(case: dict[str, Any], gate: GateConfig) -> dict[str, Any]:
    coherence = np.asarray(case["control_coherence"], dtype=np.float64)
    pullin_quality = np.asarray(case["control_pullin_quality"], dtype=np.float64)
    phase_err_filt = np.asarray(case["phase_err_filt_rad"], dtype=np.float64)
    pullin_error_lpf = np.asarray(case["pullin_error_lpf_hz"], dtype=np.float64)
    n_updates = coherence.size

    entry_ready = np.zeros(n_updates, dtype=bool)
    emulated_in_phase = np.zeros(n_updates, dtype=bool)
    emulated_switch_events = np.zeros(n_updates, dtype=np.int64)

    in_phase_mode = False
    lock_counter = 0
    unlock_counter = 0
    switch_count = 0

    for idx in range(n_updates):
        phase_coh_ok = bool(
            np.isfinite(coherence[idx]) and (coherence[idx] >= float(gate.coh_min))
        )
        pullin_quality_ok = bool(
            np.isfinite(pullin_quality[idx])
            and (pullin_quality[idx] >= float(gate.pullin_quality_min))
        )
        phase_err_ok = bool(
            np.isfinite(phase_err_filt[idx])
            and (abs(float(phase_err_filt[idx])) <= float(gate.phase_enter_max_err_rad))
        )
        pullin_err_enter_ok = bool(
            np.isfinite(pullin_error_lpf[idx])
            and (abs(float(pullin_error_lpf[idx])) <= float(gate.pullin_switch_hz))
        )
        pullin_err_exit_ok = bool(
            (not np.isfinite(pullin_error_lpf[idx]))
            or (abs(float(pullin_error_lpf[idx])) >= float(gate.pullin_exit_hz))
        )

        enter_ok = phase_coh_ok and pullin_quality_ok and phase_err_ok and pullin_err_enter_ok
        exit_ok = (not phase_coh_ok) or (not pullin_quality_ok) or pullin_err_exit_ok
        entry_ready[idx] = enter_ok

        if not in_phase_mode:
            lock_counter = lock_counter + 1 if enter_ok else 0
            unlock_counter = 0
            if lock_counter >= int(gate.pullin_enter_dwell_updates):
                in_phase_mode = True
                lock_counter = 0
                switch_count += 1
                emulated_switch_events[idx] = 1
        else:
            lock_counter = 0
            unlock_counter = unlock_counter + 1 if exit_ok else 0
            if unlock_counter >= int(gate.pullin_exit_dwell_updates):
                in_phase_mode = False
                unlock_counter = 0
                switch_count += 1
                emulated_switch_events[idx] = 1

        emulated_in_phase[idx] = in_phase_mode

    return {
        "entry_ready": entry_ready,
        "emulated_in_phase": emulated_in_phase,
        "emulated_switch_events": emulated_switch_events,
        "emulated_stage_switches": int(switch_count),
    }


def summarize_case(case: dict[str, Any], gate_results: dict[str, Any]) -> dict[str, Any]:
    dt_s = float(case["dt_s"])
    coherence = np.asarray(case["control_coherence"], dtype=np.float64)
    phase_err_abs = np.abs(np.asarray(case["phase_err_filt_rad"], dtype=np.float64))
    pullin_err_abs = np.abs(np.asarray(case["pullin_error_lpf_hz"], dtype=np.float64))
    entry_ready = np.asarray(gate_results["entry_ready"], dtype=bool)
    emulated_in_phase = np.asarray(gate_results["emulated_in_phase"], dtype=bool)
    longest_streak_updates, _ = longest_true_streak(entry_ready)

    actual_in_phase = case.get("actual_in_phase")
    actual_switch_events = case.get("actual_stage_switch_events")
    actual_stage_switches = None
    actual_phase_mode_fraction = None
    actual_entered_phase = None
    if actual_in_phase is not None:
        actual_in_phase = np.asarray(actual_in_phase, dtype=bool)
        actual_phase_mode_fraction = float(np.mean(actual_in_phase))
        actual_entered_phase = bool(np.any(actual_in_phase))
    if actual_switch_events is not None:
        actual_stage_switches = int(np.sum(np.asarray(actual_switch_events, dtype=np.int64)))

    return {
        "kind": case["kind"],
        "source_path": case["source_path"],
        "updates": int(coherence.size),
        "dt_s": dt_s,
        "median_coherence": float(np.nanmedian(coherence)),
        "p90_coherence": float(np.nanpercentile(coherence, 90.0)),
        "median_abs_phase_err_filt_rad": float(np.nanmedian(phase_err_abs)),
        "p10_abs_phase_err_filt_rad": float(np.nanpercentile(phase_err_abs, 10.0)),
        "median_abs_pullin_error_lpf_hz": float(np.nanmedian(pullin_err_abs)),
        "p10_abs_pullin_error_lpf_hz": float(np.nanpercentile(pullin_err_abs, 10.0)),
        "entry_ready_fraction": float(np.mean(entry_ready)),
        "longest_entry_ready_streak_updates": int(longest_streak_updates),
        "longest_entry_ready_streak_s": float(longest_streak_updates * dt_s),
        "emulated_enters_phase_lock": bool(np.any(emulated_in_phase)),
        "emulated_phase_mode_fraction": float(np.mean(emulated_in_phase)),
        "emulated_stage_switches": int(gate_results["emulated_stage_switches"]),
        "actual_enters_phase_lock": actual_entered_phase,
        "actual_phase_mode_fraction": actual_phase_mode_fraction,
        "actual_stage_switches": actual_stage_switches,
    }


def plot_cases(
    cases: list[dict[str, Any]],
    gate_results_by_label: dict[str, dict[str, Any]],
    gate: GateConfig,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=4, ncols=len(cases), figsize=(5.5 * len(cases), 10), sharex="col")
    if len(cases) == 1:
        axes = np.asarray(axes).reshape(4, 1)

    for col, case in enumerate(cases):
        label = str(case["label"])
        elapsed_s = np.asarray(case["elapsed_s"], dtype=np.float64)
        coherence = np.asarray(case["control_coherence"], dtype=np.float64)
        phase_err_abs = np.abs(np.asarray(case["phase_err_filt_rad"], dtype=np.float64))
        pullin_err_abs = np.abs(np.asarray(case["pullin_error_lpf_hz"], dtype=np.float64))
        gate_results = gate_results_by_label[label]
        entry_ready = np.asarray(gate_results["entry_ready"], dtype=bool).astype(np.float64)
        emulated_in_phase = np.asarray(gate_results["emulated_in_phase"], dtype=bool).astype(np.float64)

        axes[0, col].plot(elapsed_s, coherence, color="tab:blue", lw=1.25)
        axes[0, col].axhline(gate.coh_min, color="tab:red", ls="--", alpha=0.7)
        axes[0, col].set_title(label.replace("_", "\n"))
        axes[0, col].set_ylabel("coherence")
        axes[0, col].set_ylim(-0.02, 1.02)

        axes[1, col].plot(elapsed_s, phase_err_abs, color="tab:orange", lw=1.25)
        axes[1, col].axhline(gate.phase_enter_max_err_rad, color="tab:red", ls="--", alpha=0.7)
        axes[1, col].set_ylabel("|phase_err_filt| (rad)")

        axes[2, col].plot(elapsed_s, pullin_err_abs, color="tab:green", lw=1.25)
        axes[2, col].axhline(gate.pullin_switch_hz, color="tab:red", ls="--", alpha=0.7)
        axes[2, col].set_ylabel("|pullin_error_lpf| (Hz)")

        axes[3, col].step(elapsed_s, entry_ready, where="post", color="tab:purple", label="entry ready")
        axes[3, col].step(
            elapsed_s,
            1.2 + emulated_in_phase,
            where="post",
            color="tab:brown",
            label="emulated phase mode",
        )
        actual_in_phase = case.get("actual_in_phase")
        if actual_in_phase is not None:
            actual_in_phase = np.asarray(actual_in_phase, dtype=bool).astype(np.float64)
            axes[3, col].step(
                elapsed_s,
                2.4 + actual_in_phase,
                where="post",
                color="tab:gray",
                label="actual phase mode",
            )
            axes[3, col].set_yticks([0.5, 1.7, 2.9])
            axes[3, col].set_yticklabels(["entry", "emulated", "actual"])
        else:
            axes[3, col].set_yticks([0.5, 1.7])
            axes[3, col].set_yticklabels(["entry", "emulated"])
        axes[3, col].set_ylim(-0.2, 3.6)
        axes[3, col].set_ylabel("gate state")
        axes[3, col].set_xlabel("elapsed_s")

    axes[3, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Why the phase-only toy case works but the live-like input does not", fontsize=14)
    fig.tight_layout()
    return fig


def run_analysis() -> dict[str, Any]:
    live_cases = [load_live_case(label, path) for label, path in LIVE_CASE_PATHS.items()]
    reference_case = live_cases[-1]
    toy_case = make_toy_case(
        n_updates=int(reference_case["control_coherence"].size),
        dt_s=float(reference_case["dt_s"]),
        seed=0,
    )
    cases = [toy_case, *live_cases]
    gate = GateConfig()

    gate_results_by_label: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {}
    for case in cases:
        gate_results = emulate_phase_mode(case, gate)
        gate_results_by_label[str(case["label"])] = gate_results
        summary[str(case["label"])] = summarize_case(case, gate_results)

    fig = plot_cases(cases, gate_results_by_label, gate)
    return {
        "gate": asdict(gate),
        "summary": summary,
        "figure": fig,
    }


def main() -> int:
    results = run_analysis()
    print(json.dumps({"gate": results["gate"], "summary": results["summary"]}, indent=2, sort_keys=True))
    output_path = ROOT / "clock_sync_simulation_live_like.png"
    results["figure"].savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"saved_figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
