# NotebookBased DPLL

`notebook_based_pid_dpll.py` is a live SDR controller built around your survey requirements:

- async queue streaming from SDR
- staged control (`pull_in` then `phase_lock`)
- DAC-only actuation via `clock_vcxo_helper.py`
- `0 rad` phase setpoint by default
- `monitor` mode that logs all metrics without applying control
- DAC limits set by percentage (`10%-90%` defaults)

## Folder layout

- `NotebookBased/notebook_based_pid_dpll.py`: live SDR controller
- `NotebookBased/run_reduced_nsamp_sweep.py`: organized monitor/PID sweep runner for smaller `nsamp`
- `NotebookBased/plot_saved_open_closed_phase_comparison.py`: saved-run open vs closed phase dashboard
- `NotebookBased/evaluate_saved_phase_run.py`: saved-run settled phase/lock evaluation
- `NotebookBased/runs/`: new timestamped experiment sessions
- legacy flat CSV/JSON/PNG files remain in `NotebookBased/` for now so older references do not break

## Quick start

Monitor-only baseline (no DAC actuation):

```bash
python3 NotebookBased/notebook_based_pid_dpll.py \
  --mode monitor \
  --center-freq 100e6 \
  --sample-rate 2.4e6 \
  --nsamp 65536 \
  --run-seconds 180 \
  --csv-out NotebookBased/monitor_log.csv \
  --summary-out NotebookBased/monitor_summary.json
```

Closed-loop PID run (DAC control enabled):

```bash
python3 NotebookBased/notebook_based_pid_dpll.py \
  --mode pid \
  --center-freq 100e6 \
  --sample-rate 2.4e6 \
  --nsamp 65536 \
  --run-seconds 180 \
  --csv-out NotebookBased/pid_log.csv \
  --summary-out NotebookBased/pid_summary.json
```

Sweep multiple `nsamp` values (to compare loop-rate behavior):

```bash
python3 NotebookBased/notebook_based_pid_dpll.py \
  --mode pid \
  --center-freq 100e6 \
  --sample-rate 2.4e6 \
  --nsamp-values 16384,32768,65536 \
  --run-seconds 120 \
  --csv-out NotebookBased/sweep_log.csv \
  --summary-out NotebookBased/sweep_summary.json
```

Organized reduced-`nsamp` SDR sweep:

```bash
python3 NotebookBased/run_reduced_nsamp_sweep.py \
  --nsamp-values 8192,16384,32768 \
  --run-seconds 15 \
  --label live_retry
```

This creates a timestamped directory under `NotebookBased/runs/` with separate `monitor/`, `pid/`, and `comparison/` outputs.

## Saved-run phase evaluation

Generate a phase-centric open-vs-closed dashboard for an existing run:

```bash
/home/rrrodriguez13/sdr/.env/bin/python NotebookBased/plot_saved_open_closed_phase_comparison.py \
  --run-dir NotebookBased/runs/20260325_145013_tuned_defaults_v2_32768 \
  --nsamp 32768
```

Evaluate whether a saved run actually reaches a settled phase-locked regime:

```bash
/home/rrrodriguez13/sdr/.env/bin/python NotebookBased/evaluate_saved_phase_run.py \
  --run-dir NotebookBased/runs/20260325_145013_tuned_defaults_v2_32768 \
  --nsamp 32768
```

These write additional artifacts into the run's `comparison/` directory:

- `open_closed_phase_dashboard.png`
- `phase_dashboard_metrics.json`
- `phase_lock_evaluation.json`
- `phase_lock_evaluation.md`

## Synthetic verification

Run the real `NotebookBased` controller against a deterministic synthetic IQ stream and fake DAC plant:

```bash
/home/rrrodriguez13/sdr/.env/bin/python NotebookBased/run_simulated_open_closed.py \
  --run-seconds 20 \
  --nsamp 32768 \
  --sim-step-hz 1500
```

This creates a normal run directory under `NotebookBased/runs/` with:

- monitor and pid CSV/summary files from the real controller path
- `*_truth.csv` files with the synthetic plant's true offset and DAC state
- timing and phase comparison plots
- phase-lock evaluation reports using the same settled-window logic as saved live runs

## Key outputs

- Per-run CSV logs:
  - phase metrics (`phase_wrapped_rad`, `phase_unwrapped_rad`, `phase_err_*`)
  - frequency metrics (`freq_err_intrablock_hz`, `freq_err_update_hz`)
  - control metrics (`u_hz_*`, `dac_step_*`, `stage`, `action`)
- Summary JSON/CSV:
  - phase drift start/end slopes
  - drift reduction amount and percent
  - observer timing-drift slope (`primary_timing_drift_ns_per_s`) for sample-time alignment
  - coherence, phase/frequency error averages

## Fixed Observer Metric

The controller now logs a fixed-reference observer channel (logging-only) while control runs.
Use this as the primary metric for synchronization progress:

- `primary_timing_drift_ns_per_s`
- target behavior is this value moving toward `0`

Optional observer reference control:

- `--observer-fixed-offset-hz` to force a fixed observer tone offset
- if omitted, observer uses the initial tone offset

## Tuning notes

- If correction direction is wrong, flip `--control-sign` (`1.0` to `-1.0`).
- If noisy, raise `--coh-min` or lower `--Kp-hz-per-rad`.
- If slow convergence, increase `--Kf-pullin-hz-per-hz` and/or `--Kp-hz-per-rad` carefully.
- If integral windup appears, lower `--Ki-hz-per-rad-s` or `--integral-limit-hz`.
- The reduced-`nsamp` sweep runner defaults to a less strict PID coherence gate (`0.01`) and mild tone tracking (`--tone-track-alpha 0.05`) because the best prior notebook run only actuated under a lower coherence threshold.
