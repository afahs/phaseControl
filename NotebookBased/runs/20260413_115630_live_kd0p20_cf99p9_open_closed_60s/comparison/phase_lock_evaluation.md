# Phase Capability Evaluation

Run directory: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260413_115630_live_kd0p20_cf99p9_open_closed_60s`
nsamp: `8192`

This report judges the NotebookBased run by whether it reaches a sustained, phase-bounded regime, not just whether its inferred timing drift improves.

## Lock Definition

- Primary decision metric: phase error
- Primary phase series used: `accepted_raw`
- Phase band: `|phase_err_primary| <= 1.0` rad
- Minimum dwell before declaring settled: `2.0` s
- Small bad gaps bridged up to: `0.05` s
- Require action-quality gate: `True`
- Extra coherence minimum: `0.025`
- Require controller to be in `phase_lock` stage: `True`

## Open Baseline

- Primary phase metric used: `accepted_raw`
- Full-run observer timing drift: `0.009364` samples/s
- Full-run phase drift slope: `45.457512` Hz
- Full-run primary phase-band fraction: `0.31752577319587627`
- Steady-state mean |primary phase error|: `nan` rad
- Mean |phase error|: `1.583260` rad
- Phase-band fraction: `0.32408930669800234`
- Accepted-raw phase-band fraction: `0.31752577319587627`

## Closed-Loop Assessment

- Primary phase metric used: `accepted_raw`
- Entered `phase_lock` stage: `True`
- First `phase_lock` stage time: `0.08704` s
- Locked under this definition: `False`
- Lock reason: `insufficient_good_dwell`
- Best good dwell: `0.1570133330000001` s
- Settled start: `nan` s
- Full-run observer timing drift: `0.006364` samples/s
- Full-run phase drift slope: `4.239454` Hz
- Full-run primary phase-band fraction: `0.3405275779376499`
- Steady-state mean |primary phase error|: `nan` rad
- Steady-state median |primary phase error|: `nan` rad
- Steady-state primary phase-band fraction: `nan`
- Mean |phase error|: `1.563515` rad
- Full-run phase-band fraction: `0.3530548176768896`
- Full-run accepted-raw phase-band fraction: `0.3405275779376499`
- Full-run `phase_lock` stage fraction: `0.9997585124366095`
- Steady-state drift after settle: `nan` samples/s
- Steady-state timing RMS: `nan` ns
- Steady-state mean |phase error|: `nan` rad
- Steady-state mean |accepted raw phase error|: `nan` rad
- Steady-state accepted-raw phase-band fraction: `nan`

## Comparison

- Primary phase error improvement vs open loop: `nan`%
- Primary phase-band improvement vs open loop: `nan`
- Timing drift improvement vs open loop: `32.035621453602964`%
- Phase drift improvement vs open loop: `90.67380964322693`%
- Mean |phase error| improvement vs open loop: `1.247104181832434`%
- Accepted-raw phase-band improvement vs open loop: `nan`

## Interpretation

This run reduces drift, but it does not maintain a long enough in-band phase window to count as settled lock under this definition.
It is behaving more like partial pull-in than a clean phase-locked DPLL.

Monitor summary: `NotebookBased/runs/20260413_115630_live_kd0p20_cf99p9_open_closed_60s/monitor/nsamp_8192/monitor_nsamp8192.csv`
Closed-loop summary: `NotebookBased/runs/20260413_115630_live_kd0p20_cf99p9_open_closed_60s/pid/nsamp_8192/pid_nsamp8192.csv`
