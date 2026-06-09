# Phase Capability Evaluation

Run directory: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_155721_phase_tuning_sweep/trials/trial_007_seed12345`
nsamp: `8192`

This report judges the NotebookBased run by whether it reaches a sustained, phase-bounded regime, not just whether its inferred timing drift improves.

## Lock Definition

- Primary decision metric: phase error
- Primary phase series used: `filtered`
- Phase band: `|phase_err_primary| <= 1.0` rad
- Minimum dwell before declaring settled: `2.0` s
- Small bad gaps bridged up to: `0.05` s
- Require action-quality gate: `True`
- Extra coherence minimum: `0.025`
- Require controller to be in `phase_lock` stage: `True`

## Open Baseline

- Primary phase metric used: `filtered`
- Full-run observer timing drift: `9.531286` samples/s
- Full-run phase drift slope: `397.014756` Hz
- Full-run primary phase-band fraction: `0.31399317406143346`
- Steady-state mean |primary phase error|: `nan` rad
- Mean |phase error|: `1.569647` rad
- Phase-band fraction: `0.31399317406143346`
- Accepted-raw phase-band fraction: `nan`

## Closed-Loop Assessment

- Primary phase metric used: `filtered`
- Entered `phase_lock` stage: `False`
- First `phase_lock` stage time: `nan` s
- Locked under this definition: `False`
- Lock reason: `never_entered_phase_lock_stage`
- Best good dwell: `0.0` s
- Settled start: `nan` s
- Full-run observer timing drift: `-0.098832` samples/s
- Full-run phase drift slope: `-4.116721` Hz
- Full-run primary phase-band fraction: `0.24914675767918087`
- Steady-state mean |primary phase error|: `nan` rad
- Steady-state median |primary phase error|: `nan` rad
- Steady-state primary phase-band fraction: `nan`
- Mean |phase error|: `1.584220` rad
- Full-run phase-band fraction: `0.24914675767918087`
- Full-run accepted-raw phase-band fraction: `nan`
- Full-run `phase_lock` stage fraction: `0.0`
- Steady-state drift after settle: `nan` samples/s
- Steady-state timing RMS: `nan` ns
- Steady-state mean |phase error|: `nan` rad
- Steady-state mean |accepted raw phase error|: `nan` rad
- Steady-state accepted-raw phase-band fraction: `nan`

## Comparison

- Primary phase error improvement vs open loop: `nan`%
- Primary phase-band improvement vs open loop: `nan`
- Timing drift improvement vs open loop: `98.96308117372878`%
- Phase drift improvement vs open loop: `98.9630811737288`%
- Mean |phase error| improvement vs open loop: `-0.9283740332622719`%
- Accepted-raw phase-band improvement vs open loop: `nan`

## Interpretation

This run improves inferred timing drift strongly, but it never transitions into the controller's `phase_lock` stage.
That means the saved data supports frequency pull-in and drift reduction, but not a demonstrated phase-locked DPLL regime yet.

Monitor summary: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_155721_phase_tuning_sweep/trials/trial_007_seed12345/monitor/nsamp_8192/monitor_nsamp8192.csv`
Closed-loop summary: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_155721_phase_tuning_sweep/trials/trial_007_seed12345/pid/nsamp_8192/pid_nsamp8192.csv`
