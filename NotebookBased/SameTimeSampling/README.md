# Same-Time Sampling Prototype

This folder contains a separate timing-lock prototype aimed at the user goal:
keep two clocks sampling at the same instant, not just minimize long-term drift.

## Main Script

- `same_time_sampling_dpll.py`

## What Is Different From The Existing Notebook Loop

- `pull_in` is only for coarse beat-frequency acquisition.
- `timing_capture` uses inferred timing offset `tau` and inferred sample-slip to approach a bounded timing region.
- `timing_lock` is intended to hold the inferred timing error near a fixed setpoint with smaller DAC updates.

The controller therefore acts on:

- filtered inferred timing error in ns
- filtered inferred slip in samples/s

rather than only on coarse pull-in frequency error.

## Important Timing Reference Note

This prototype can only hold a timing reference that it can measure.

- With `--tau-reference-mode first_valid` the loop locks timing relative to the first valid observer measurement in the run.
- With `--tau-reference-mode absolute` it tries to drive the observer-derived timing estimate directly toward `--tau-setpoint-ns`.

So this is designed for same-time sampling, but proving true hardware simultaneity still requires an external calibration or timestamp reference.

## Basic Usage

Monitor:

```bash
/home/rrrodriguez13/sdr/.env/bin/python NotebookBased/SameTimeSampling/same_time_sampling_dpll.py \
  --mode monitor \
  --nsamp 32768 \
  --run-seconds 8
```

Closed loop:

```bash
/home/rrrodriguez13/sdr/.env/bin/python NotebookBased/SameTimeSampling/same_time_sampling_dpll.py \
  --mode pid \
  --nsamp 32768 \
  --run-seconds 8
```

## Suggested First Evaluation

When testing this prototype, focus on:

- `primary_timing_drift_samples_per_s`
- `timing_lock_fraction`
- `timing_lock_updates_applied`
- inferred sample-offset traces derived from `observer_tau_err_ns`
