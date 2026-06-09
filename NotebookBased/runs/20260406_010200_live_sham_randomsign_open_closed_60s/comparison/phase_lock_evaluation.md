# Phase Capability Evaluation

Run directory: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_010200_live_sham_randomsign_open_closed_60s`
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

- Full-run observer timing drift: `-3.531164` samples/s
- Full-run phase drift slope: `-147.086991` Hz
- Mean |phase error|: `1.540981` rad
- Phase-band fraction: `0.3472976837289105`

## Closed-Loop Assessment

- Entered `phase_lock` stage: `True`
- First `phase_lock` stage time: `0.141653333` s
- Locked under this definition: `False`
- Lock reason: `insufficient_good_dwell`
- Best good dwell: `0.9762133330000002` s
- Settled start: `nan` s
- Full-run observer timing drift: `0.155408` samples/s
- Full-run phase drift slope: `-276.151871` Hz
- Mean |phase error|: `1.551918` rad
- Full-run phase-band fraction: `0.33077513430544897`
- Full-run `phase_lock` stage fraction: `0.7996930161166539`
- Steady-state drift after settle: `nan` samples/s
- Steady-state timing RMS: `nan` ns
- Steady-state mean |phase error|: `nan` rad

## Comparison

- Timing drift improvement vs open loop: `95.5989497741277`%
- Phase drift improvement vs open loop: `-87.7473109713889`%
- Mean |phase error| improvement vs open loop: `-0.7097361238206027`%

## Interpretation

This run reduces drift, but it does not maintain a long enough in-band phase window to count as settled lock under this definition.
It is behaving more like partial pull-in than a clean phase-locked DPLL.

Monitor summary: `NotebookBased/runs/20260406_010200_live_sham_randomsign_open_closed_60s/monitor/nsamp_8192/monitor_nsamp8192.csv`
Closed-loop summary: `NotebookBased/runs/20260406_010200_live_sham_randomsign_open_closed_60s/pid/nsamp_8192/pid_nsamp8192.csv`
