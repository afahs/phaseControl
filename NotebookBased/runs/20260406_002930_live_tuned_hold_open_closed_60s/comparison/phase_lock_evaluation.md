# Phase Capability Evaluation

Run directory: `/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_002930_live_tuned_hold_open_closed_60s`
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

- Full-run observer timing drift: `0.005964` samples/s
- Full-run phase drift slope: `-8.372411` Hz
- Mean |phase error|: `1.563438` rad
- Phase-band fraction: `0.3911024199462234`

## Closed-Loop Assessment

- Entered `phase_lock` stage: `True`
- First `phase_lock` stage time: `0.954026667` s
- Locked under this definition: `True`
- Lock reason: `settled_window_found`
- Best good dwell: `2.0479999999999983` s
- Settled start: `20.007253333` s
- Full-run observer timing drift: `-0.017952` samples/s
- Full-run phase drift slope: `-2.019587` Hz
- Mean |phase error|: `1.574978` rad
- Full-run phase-band fraction: `0.5646910466582598`
- Full-run `phase_lock` stage fraction: `0.9778058007566204`
- Steady-state drift after settle: `-0.008498845396494517` samples/s
- Steady-state timing RMS: `145.67813532829305` ns
- Steady-state mean |phase error|: `0.9692465087044145` rad

## Comparison

- Timing drift improvement vs open loop: `-201.0134064916867`%
- Phase drift improvement vs open loop: `75.8780662822248`%
- Mean |phase error| improvement vs open loop: `-0.7381149282064435`%

## Interpretation

This run satisfies the settled-window lock definition, so it does show DPLL-like phase capability on the saved data.
The steady-state fields above are the right place to judge how clean that lock is.

Monitor summary: `NotebookBased/runs/20260406_002930_live_tuned_hold_open_closed_60s/monitor/nsamp_8192/monitor_nsamp8192.csv`
Closed-loop summary: `NotebookBased/runs/20260406_002930_live_tuned_hold_open_closed_60s/pid/nsamp_8192/pid_nsamp8192.csv`
