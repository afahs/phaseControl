# Phase Capability Evaluation

Run directory: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_004100_live_kd0p20_open_closed_60s`
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
- Full-run observer timing drift: `0.001376` samples/s
- Full-run phase drift slope: `-33.323746` Hz
- Full-run primary phase-band fraction: `0.31594634873323396`
- Steady-state mean |primary phase error|: `nan` rad
- Mean |phase error|: `1.564497` rad
- Phase-band fraction: `0.31594634873323396`
- Accepted-raw phase-band fraction: `nan`

## Closed-Loop Assessment

- Primary phase metric used: `filtered`
- Entered `phase_lock` stage: `True`
- First `phase_lock` stage time: `0.366933333` s
- Locked under this definition: `True`
- Lock reason: `settled_window_found`
- Best good dwell: `3.05152` s
- Settled start: `11.665066667` s
- Full-run observer timing drift: `0.040329` samples/s
- Full-run phase drift slope: `-5.832347` Hz
- Full-run primary phase-band fraction: `0.6098124839455433`
- Steady-state mean |primary phase error|: `0.8878494652746795` rad
- Steady-state median |primary phase error|: `0.635621436` rad
- Steady-state primary phase-band fraction: `0.6682692307692307`
- Mean |phase error|: `1.575104` rad
- Full-run phase-band fraction: `0.6098124839455433`
- Full-run accepted-raw phase-band fraction: `nan`
- Full-run `phase_lock` stage fraction: `0.9935782173131261`
- Steady-state drift after settle: `0.044370072594118126` samples/s
- Steady-state timing RMS: `244.83896655656815` ns
- Steady-state mean |phase error|: `0.8878494652746795` rad
- Steady-state mean |accepted raw phase error|: `nan` rad
- Steady-state accepted-raw phase-band fraction: `nan`

## Comparison

- Primary phase error improvement vs open loop: `nan`%
- Primary phase-band improvement vs open loop: `nan`
- Timing drift improvement vs open loop: `-2831.7493613092647`%
- Phase drift improvement vs open loop: `82.49792457453545`%
- Mean |phase error| improvement vs open loop: `-0.6779785459928572`%
- Accepted-raw phase-band improvement vs open loop: `nan`

## Interpretation

This run satisfies the settled-window lock definition, so it does show DPLL-like phase capability on the saved data.
The steady-state fields above are the right place to judge how clean that lock is.

Monitor summary: `NotebookBased/runs/20260406_004100_live_kd0p20_open_closed_60s/monitor/nsamp_8192/monitor_nsamp8192.csv`
Closed-loop summary: `NotebookBased/runs/20260406_004100_live_kd0p20_open_closed_60s/pid/nsamp_8192/pid_nsamp8192.csv`
