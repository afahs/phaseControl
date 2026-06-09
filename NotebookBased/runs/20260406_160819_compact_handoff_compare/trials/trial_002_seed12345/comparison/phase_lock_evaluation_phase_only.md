# Phase Capability Evaluation

Run directory: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_160819_compact_handoff_compare/trials/trial_002_seed12345`
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
- Require controller to be in `phase_lock` stage: `False`

## Open Baseline

- Primary phase metric used: `filtered`
- Full-run observer timing drift: `-7.200422` samples/s
- Full-run phase drift slope: `-300.000009` Hz
- Full-run primary phase-band fraction: `0.3174061433447099`
- Steady-state mean |primary phase error|: `nan` rad
- Mean |phase error|: `1.569605` rad
- Phase-band fraction: `0.3174061433447099`
- Accepted-raw phase-band fraction: `nan`

## Closed-Loop Assessment

- Primary phase metric used: `accepted_raw`
- Entered `phase_lock` stage: `True`
- First `phase_lock` stage time: `0.397653333` s
- Locked under this definition: `False`
- Lock reason: `insufficient_good_dwell`
- Best good dwell: `0.0` s
- Settled start: `nan` s
- Full-run observer timing drift: `9.712949` samples/s
- Full-run phase drift slope: `404.682484` Hz
- Full-run primary phase-band fraction: `0.3333333333333333`
- Steady-state mean |primary phase error|: `nan` rad
- Steady-state median |primary phase error|: `nan` rad
- Steady-state primary phase-band fraction: `nan`
- Mean |phase error|: `1.561528` rad
- Full-run phase-band fraction: `0.31143344709897613`
- Full-run accepted-raw phase-band fraction: `0.3333333333333333`
- Full-run `phase_lock` stage fraction: `0.9744027303754266`
- Steady-state drift after settle: `nan` samples/s
- Steady-state timing RMS: `nan` ns
- Steady-state mean |phase error|: `nan` rad
- Steady-state mean |accepted raw phase error|: `nan` rad
- Steady-state accepted-raw phase-band fraction: `nan`

## Comparison

- Primary phase error improvement vs open loop: `nan`%
- Primary phase-band improvement vs open loop: `nan`
- Timing drift improvement vs open loop: `-34.894157092966125`%
- Phase drift improvement vs open loop: `-34.89415709296613`%
- Mean |phase error| improvement vs open loop: `0.5145752652352139`%
- Accepted-raw phase-band improvement vs open loop: `nan`

## Interpretation

This run reduces drift, but it does not maintain a long enough in-band phase window to count as settled lock under this definition.
It is behaving more like partial pull-in than a clean phase-locked DPLL.

Monitor summary: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_160819_compact_handoff_compare/trials/trial_002_seed12345/monitor/nsamp_8192/monitor_nsamp8192.csv`
Closed-loop summary: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_160819_compact_handoff_compare/trials/trial_002_seed12345/pid/nsamp_8192/pid_nsamp8192.csv`
