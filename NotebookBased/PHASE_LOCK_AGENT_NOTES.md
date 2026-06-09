# Phase Lock Agent Notes

This note is a durable handoff for future work on the `NotebookBased` DPLL experiments.

It is meant to answer four questions:

1. What problem are we actually trying to solve?
2. Why does the controller currently look the way it does?
3. Which changes helped on live hardware, and which did not?
4. What should a future agent do next without repeating dead ends?

## Problem Statement

The system can now reduce timing and residual phase drift on live SDR data, and it can produce a bounded filtered phase state for sustained periods. However, it still does **not** convincingly produce clean raw hardware phase lock in the sense of driving the raw phase detector output near zero on most blocks.

The core challenge is:

- In simulation, the phase loop works well once started inside its capture region.
- On live hardware, the raw phase measurement is much noisier and more biased than the simulation.
- Because of that, the loop can improve the **filtered control state** without showing equally strong improvement in the **raw per-block phase error**.

Future agents should treat this as a **measurement-and-control-trust problem**, not only a controller-gain problem.

## Files That Matter

- Shared controller: [notebook_based_hybrid_pullin_notebook_phase_dpll.py](/home/rrrodriguez13/phaseControl/NotebookBased/notebook_based_hybrid_pullin_notebook_phase_dpll.py)
- Active wrapper/defaults: [notebook_based_pullin_handoff_30hz_phase_dpll.py](/home/rrrodriguez13/phaseControl/NotebookBased/notebook_based_pullin_handoff_30hz_phase_dpll.py)
- Saved-run evaluator: [evaluate_saved_phase_run.py](/home/rrrodriguez13/phaseControl/NotebookBased/evaluate_saved_phase_run.py)
- Acceptance diagnostic plots: [plot_phase_acceptance_diagnostics.py](/home/rrrodriguez13/phaseControl/NotebookBased/plot_phase_acceptance_diagnostics.py)
- Sim harness: [run_simulated_open_closed.py](/home/rrrodriguez13/phaseControl/NotebookBased/run_simulated_open_closed.py)

## How To Run Live SDR Jobs

The system Python in this workspace does **not** have the SDR dependencies.

The live runs that worked used:

```bash
mkdir -p /tmp/ugradio_import
ln -sfn /home/rrrodriguez13/ugradio/ugradio_code/src /tmp/ugradio_import/ugradio

PYTHONPATH=/tmp/ugradio_import \
  /home/rrrodriguez13/sdr/.env/bin/python \
  NotebookBased/notebook_based_pullin_handoff_30hz_phase_dpll.py ...
```

Without that interpreter and `PYTHONPATH`, `ugradio` or `rtlsdr` imports will fail.

## Current Default Structure

The current default wrapper is still a **pull-in -> phase-lock** design, not a true phase-only startup design.

Current defaults in [notebook_based_pullin_handoff_30hz_phase_dpll.py](/home/rrrodriguez13/phaseControl/NotebookBased/notebook_based_pullin_handoff_30hz_phase_dpll.py):

- `control_measurement_source="adaptive"`
- `nsamp=8192`, `stream_nsamp=16384`
- `phase_detector_chunks=4`
- `ignore_blocks_after_dac_write=1`
- `phase_enter_condition="full"`
- `phase_exit_condition="residual_and_phase"`
- `pullin_switch_hz=30.0`
- `pullin_exit_hz=100.0`
- `phase_hold_kp_hz_per_rad=8.0`
- `phase_hold_ki_hz_per_rad_s=3.0`
- `phase_hold_kd_hz_per_hz=0.20`

Important: predictor-through-holds and soft-innovation acceptance are now implemented in the shared controller, but they are currently **disabled by default** because they did not beat the earlier live locked form consistently.

## Why The Controller Evolved This Way

### 1. Pull-in still matters on real hardware

This was a major lesson.

Simulation showed that pure phase mode can work if the initial residual frequency mismatch is already small. But live data does **not** start close enough for phase-only control to be realistic.

Observed live startup residuals were often on the order of `1.1-1.3 kHz`.

The simulation showed the rough phase-only capture story:

- `0-30 Hz` residual mismatch: generally workable
- `40 Hz`: marginal
- `50-75 Hz`: phase-only no longer centers well

Reason:

- At `nsamp=16384` and `2.4 MHz`, one control update spans about `6.83 ms`
- A `50 Hz` residual mismatch rotates phase by about `2.14 rad` per update
- A `75 Hz` residual mismatch rotates phase by about `3.22 rad` per update

That is too fast for the small-step phase hold law to behave like a fine centering loop.

This is why the current system uses pull-in first, then phase mode only after filtered residual is within about `30 Hz` for dwell.

### 2. The full-block phase detector was too fragile on live data

The original phase detector used a full `16384`-sample block sum. On live hardware, the phase can sweep across a block, which causes phasor cancellation. That gives poor coherence and misleading phase estimates even when the signal is locally fine.

This motivated:

- chunked phase detection with `4 x 4096`
- chunk-fit slope/RMS quality fields
- fit-quality gating before trusting phase mode

This was a real improvement and should be kept.

### 3. `adaptive` control was better than `observer`

The controller can drive from either the `observer` or `adaptive` measurement path.

On live data, `adaptive` modestly improved the phase-lock dwell and filtered phase metrics relative to `observer`. The likely reason is that the observer path can stay biased relative to the actual received tone, especially while the loop is moving the clock.

This is why the active wrapper defaults to `adaptive`.

### 4. Ignoring one post-DAC block helped, but two was too much

Ignoring the next block after a DAC write improved the right metrics:

- better good dwell
- lower filtered phase error
- better phase-band fraction

Ignoring **two** blocks starved the controller too much and made phase locking worse.

So the current default is `ignore_blocks_after_dac_write=1`.

### 5. Measurement gating was necessary to get the first live “locked=true” runs

The major jump in live performance came from:

- chunk-fit measurement
- fit-quality gate
- innovation gate
- adaptive control

This produced the first live runs where the saved evaluator reported `locked=true`.

Key runs:

- [20260406_001120_live_measurement_upgrade_open_closed_60s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_001120_live_measurement_upgrade_open_closed_60s)
- [20260406_004100_live_kd0p20_open_closed_60s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_004100_live_kd0p20_open_closed_60s)

### 6. But raw phase still did not clearly collapse

This is the most important caution.

Even in the better live runs, raw `|phase_err_raw|` did **not** improve dramatically. The system mainly improved the **filtered/accepted** phase state.

That means:

- we do have evidence of real control action
- but we should not overclaim “raw hardware phase lock”

This point matters for future evaluation.

## Evidence That The Loop Is Doing Something Real

There was a legitimate concern that the controller was only “winning by filtering.”

To test that, a sham-control mode was added where the same detector, gate, and controller math ran, but the DAC step sign was randomized.

Run:

- [20260406_010200_live_sham_randomsign_open_closed_60s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_010200_live_sham_randomsign_open_closed_60s)

Result:

- real loop: `locked=true`
- sham loop: `locked=false`

Important sham numbers:

- sham best good dwell: about `0.98 s`
- real best good dwell in the stronger run: about `3.05 s`
- sham phase-band fraction was much worse than the real closed-loop run

So the loop is not **only** filtering or cherry-picking measurements. Actuation does matter.

## Best Reference Runs

### Best “measurement-upgrade” reference

Run:

- [20260406_001120_live_measurement_upgrade_open_closed_60s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_001120_live_measurement_upgrade_open_closed_60s)

Saved evaluation:

- [phase_lock_evaluation.json](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_001120_live_measurement_upgrade_open_closed_60s/comparison/phase_lock_evaluation.json)

Highlights:

- `locked=true`
- best good dwell: about `2.07 s`
- steady-state mean `|phase_err_filt|`: about `0.876 rad`
- steady-state median `|phase_err_filt|`: about `0.745 rad`

### Best `Kd=0.20` reference

Run:

- [20260406_004100_live_kd0p20_open_closed_60s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_004100_live_kd0p20_open_closed_60s)

Saved evaluation:

- [phase_lock_evaluation.json](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_004100_live_kd0p20_open_closed_60s/comparison/phase_lock_evaluation.json)

Highlights:

- `locked=true`
- best good dwell: about `3.05 s`
- steady-state mean `|phase_err_filt|`: about `0.888 rad`
- steady-state median `|phase_err_filt|`: about `0.636 rad`

This is a good reference operating point even though it is not uniformly best on every metric.

## Things That Were Tried And Why

### True single-stage phase-only startup

Why it was tried:

- The phase loop worked in simulation when started near the capture region.

What happened:

- It improved simulation capture range.
- It failed live because the hardware starts far too far away in residual frequency.

Lesson:

- Live startup still needs a coarse pull-in stage.

### Tightened handoff

Why it was tried:

- The old handoff was entering phase mode too early.

What happened:

- It helped handoff quality.
- It did **not** fully solve live phase centering.

Lesson:

- Handoff mattered, but handoff was not the whole problem.

### Stickier exit logic

Why it was tried:

- The loop was bouncing between pull-in and phase-lock.

What happened:

- Stage chatter dropped sharply.
- But staying in phase mode longer did not by itself center the phase.

Lesson:

- Exit logic was a secondary issue, not the root issue.

### Smaller logical `nsamp`

Why it was tried:

- Shorter updates should reduce intrablock phase sweep.

What happened:

- `8192` with `stream_nsamp=16384` improved measurement quality somewhat.
- `4096/8192` made the loop too twitchy.

Lesson:

- `8192 + stream_nsamp=16384` is a reasonable operating point.

### Predictor through rejected/ignored updates

Why it was tried:

- The filtered phase state was freezing during ignored or rejected updates, which could make the next innovation look worse.

What happened:

- It worked mechanically.
- It did not improve live 20 s smoke performance enough to keep enabled by default.

Run:

- [20260406_142600_predictor_closed_smoke20s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_142600_predictor_closed_smoke20s)

Lesson:

- The idea is still plausible, but the simple version was not clearly better.

### Soft innovation rejection

Why it was tried:

- Hard reject/hold can make the loop too brittle.

What happened:

- Fully softening moderate overshoots was too permissive and made phase behavior worse.
- A second version with:
  - partial blending
  - a separate hard reject
  - softening only when fit RMS was very good
  behaved much better than the first soft version, but still did not clearly beat the earlier baseline.

Runs:

- [20260406_143100_softinnovation_closed_smoke20s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_143100_softinnovation_closed_smoke20s)
- [20260406_143600_softinnovation_v2_closed_smoke20s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_143600_softinnovation_v2_closed_smoke20s)

Lesson:

- Soft innovation may still be useful, but it is not yet a default-worthy win.

## Current Interpretation Of The System

The best current interpretation is:

- The loop can reduce timing drift and residual phase drift on live hardware.
- The loop can produce a bounded filtered phase estimate in a way that is causally better than sham control.
- The raw live phase detector is still not clean enough to claim strong raw phase centering on most blocks.

So the system is somewhere between:

- “not working at all”
- and “clean raw phase lock”

It is working as a **real but measurement-limited closed-loop controller**.

## What Future Agents Should Not Repeat

- Do not assume the issue is “just gains.”
- Do not assume simulation success means live pull-in or live phase detector quality is equivalent.
- Do not remove pull-in entirely for live startup.
- Do not rely on system Python for SDR runs.
- Do not treat `mean_abs_phase_err_raw` alone as the only truth metric.
- Do not overclaim victory when only the filtered state improves.

## What Future Agents Should Check First

When evaluating a change, check these in order:

1. Did live `phase_lock` actually engage and stay engaged?
2. Did `phase_err_filt` improve?
3. Did accepted/usable raw phase improve, not just filtered phase?
4. Did the sham-control baseline still fail relative to the real loop?
5. Did the change help on a **60 s open vs closed** run, not only a 20 s smoke?

Useful artifacts:

- evaluator output: `comparison/phase_lock_evaluation.json`
- raw/accepted/filtered plots from [plot_phase_acceptance_diagnostics.py](/home/rrrodriguez13/phaseControl/NotebookBased/plot_phase_acceptance_diagnostics.py)

## Recommended Starting Point For Future Work

If a future agent is resuming from here, the safest starting point is:

1. Use the current reverted defaults in [notebook_based_pullin_handoff_30hz_phase_dpll.py](/home/rrrodriguez13/phaseControl/NotebookBased/notebook_based_pullin_handoff_30hz_phase_dpll.py).
2. Run a fresh **60 s open vs closed** comparison to re-establish the reference.
3. Compare against:
   - [20260406_001120_live_measurement_upgrade_open_closed_60s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_001120_live_measurement_upgrade_open_closed_60s)
   - [20260406_004100_live_kd0p20_open_closed_60s](/home/rrrodriguez13/phaseControl/NotebookBased/runs/20260406_004100_live_kd0p20_open_closed_60s)
4. Only then try one new change at a time.

## Best Open Problems

These are still good next targets:

- Improve the live phase measurement so raw accepted phase is closer to the filtered state.
- Better decide when a phase measurement is trustworthy without becoming too brittle.
- Increase phase-centering authority only when confidence is high.
- Understand why some live trials are much better than others even under similar code settings.

## Short Version For A Future Agent

The controller is no longer in the “basic architecture is wrong” phase.

The big architecture lessons are already known:

- keep pull-in
- use `adaptive`
- use chunk-fit phase detection
- ignore one block after DAC write
- use fit-quality and innovation gating

The main remaining problem is that live raw phase measurements are still too noisy or biased, so the filtered control state improves more than the raw phase does.

Future work should focus on **trusting better measurements**, not blindly adding more stages.
