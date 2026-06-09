# Phase Capability Evaluation

Run directory: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_001120_live_measurement_upgrade_open_closed_60s`
nsamp: `8192`

This report judges the NotebookBased run by whether it reaches a sustained, phase-bounded regime, not just whether its inferred timing drift improves.

## Lock Definition

- Filtered phase error band: `|phase_err_filt| <= 1.0` rad
- Minimum dwell before declaring settled: `2.0` s
- Small bad gaps bridged up to: `0.05` s
- Require action-quality gate: `True`
- Extra coherence minimum: `0.025`
- Require controller to be in `phase_lock` stage: `True`

## Open Baseline

- Full-run observer timing drift: `0.013126` samples/s
- Full-run phase drift slope: `-50.313411` Hz
- Mean |phase error|: `1.575890` rad
- Phase-band fraction: `0.3232864478678276`

## Closed-Loop Assessment

- Entered `phase_lock` stage: `True`
- First `phase_lock` stage time: `1.165653333` s
- Locked under this definition: `True`
- Lock reason: `settled_window_found`
- Best good dwell: `2.0684799999999974` s
- Settled start: `19.76832` s
- Full-run observer timing drift: `0.025202` samples/s
- Full-run phase drift slope: `-15.973381` Hz
- Mean |phase error|: `1.570464` rad
- Full-run phase-band fraction: `0.5674067292286564`
- Full-run `phase_lock` stage fraction: `0.9741359578850995`
- Steady-state drift after settle: `0.0037427300160344647` samples/s
- Steady-state timing RMS: `69.67877408983482` ns
- Steady-state mean |phase error|: `0.8758039899325727` rad

## Comparison

- Timing drift improvement vs open loop: `-91.99286031488693`%
- Phase drift improvement vs open loop: `68.25224056275088`%
- Mean |phase error| improvement vs open loop: `0.34432754353362127`%

## Interpretation

This run satisfies the settled-window lock definition, so it does show DPLL-like phase capability on the saved data.
The steady-state fields above are the right place to judge how clean that lock is.

Monitor summary: `NotebookBased/runs/20260406_001120_live_measurement_upgrade_open_closed_60s/monitor/nsamp_8192/monitor_nsamp8192.csv`
Closed-loop summary: `NotebookBased/runs/20260406_001120_live_measurement_upgrade_open_closed_60s/pid/nsamp_8192/pid_nsamp8192.csv`
