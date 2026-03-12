# NotebookBased DPLL

`notebook_based_pid_dpll.py` is a live SDR controller built around your survey requirements:

- async queue streaming from SDR
- staged control (`pull_in` then `phase_lock`)
- DAC-only actuation via `clock_vcxo_helper.py`
- `0 rad` phase setpoint by default
- `monitor` mode that logs all metrics without applying control
- DAC limits set by percentage (`10%-90%` defaults)

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
