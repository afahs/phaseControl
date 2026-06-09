# Phase Capability Evaluation

Run directory: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_160819_compact_handoff_compare/trials/trial_004_seed12345`
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
- First `phase_lock` stage time: `0.274773333` s
- Locked under this definition: `False`
- Lock reason: `insufficient_good_dwell`
- Best good dwell: `0.0` s
- Settled start: `nan` s
- Full-run observer timing drift: `8.939675` samples/s
- Full-run phase drift slope: `372.464621` Hz
- Full-run primary phase-band fraction: `0.3545232273838631`
- Steady-state mean |primary phase error|: `nan` rad
- Steady-state median |primary phase error|: `nan` rad
- Steady-state primary phase-band fraction: `nan`
- Mean |phase error|: `1.573260` rad
- Full-run phase-band fraction: `0.30716723549488056`
- Full-run accepted-raw phase-band fraction: `0.3545232273838631`
- Full-run `phase_lock` stage fraction: `0.984641638225256`
- Steady-state drift after settle: `nan` samples/s
- Steady-state timing RMS: `nan` ns
- Steady-state mean |phase error|: `nan` rad
- Steady-state mean |accepted raw phase error|: `nan` rad
- Steady-state accepted-raw phase-band fraction: `nan`

## Comparison

- Primary phase error improvement vs open loop: `nan`%
- Primary phase-band improvement vs open loop: `nan`
- Timing drift improvement vs open loop: `-24.154869952477487`%
- Phase drift improvement vs open loop: `-24.154869952477508`%
- Mean |phase error| improvement vs open loop: `-0.23289087339782613`%
- Accepted-raw phase-band improvement vs open loop: `nan`

## Interpretation

This run reduces drift, but it does not maintain a long enough in-band phase window to count as settled lock under this definition.
It is behaving more like partial pull-in than a clean phase-locked DPLL.

Monitor summary: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_160819_compact_handoff_compare/trials/trial_004_seed12345/monitor/nsamp_8192/monitor_nsamp8192.csv`
Closed-loop summary: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_160819_compact_handoff_compare/trials/trial_004_seed12345/pid/nsamp_8192/pid_nsamp8192.csv`
