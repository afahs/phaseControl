# Notebook-Style Live DPLL

This folder contains a live SDR controller that follows the control idea in
`clock_sync_simulation.ipynb` more directly than the staged controllers under
`NotebookBased/`.

## Main Script

- `notebook_style_dpll.py`

## What Matches The Notebook

- A fixed tone reference is used after startup bootstrap.
- Each update uses the block-average complex mix after demodulation.
- Phase error is converted into a per-sample phase fraction:
  - `phase_fraction_cycles_per_sample = phase_err_filt_rad / (2*pi*nsamp)`
- That phase fraction maps directly to an absolute DAC target around
  `dac_center`.
- There is no staged `pull_in` / `phase_lock` state machine.
- There is no retained integral trim or adaptive tone tracking.

In other words, the live controller keeps the notebook's "measure phase from the
mixed block, map it straight into the control voltage" structure, while reusing
the SDR streaming and DAC control plumbing from the existing live scripts.

## Hardware Control Path

- SDR block streaming comes from the same `ugradio.sdr` interface used in
  `NotebookBased/`.
- DAC updates go through `clock_vcxo_helper.ClockVCXO`.
- `pid` mode keeps the existing naming convention, but the controller is
  notebook-style absolute DAC targeting rather than staged PID.
- `monitor` mode runs the full measurement pipeline without writing the DAC.

## Basic Usage

Static validation:

```bash
python3 dpllNotebook/notebook_style_dpll.py --help
```

Monitor run:

```bash
/home/rrrodriguez13/sdr/.env/bin/python dpllNotebook/notebook_style_dpll.py \
  --mode monitor \
  --nsamp 32768 \
  --run-seconds 8
```

Closed-loop run:

```bash
/home/rrrodriguez13/sdr/.env/bin/python dpllNotebook/notebook_style_dpll.py \
  --mode pid \
  --nsamp 32768 \
  --run-seconds 8
```

## Important CLI Knobs

- `--phase-frac-gain-codes`
  - notebook-style gain from phase fraction to DAC codes
  - the default `409500` is the closest direct translation of the notebook's
    `frac / Vscale` mapping with `Vscale=0.01`
- `--dac-center`
  - absolute center code around which the notebook-style target is formed
  - if omitted, the midpoint of the allowed DAC range is used
- `--max-dac-step-per-update`
  - hardware safety clamp that limits how fast the commanded DAC can move
- `--coh-min` and `--pullin-quality-min`
  - quality gates that must pass before `pid` mode applies any DAC update

## Outputs

The script writes a compact CSV and summary JSON.

Key CSV fields:

- `phase_wrapped_rad`, `phase_unwrapped_rad`
- `phase_err_raw_rad`, `phase_err_filt_rad`
- `phase_fraction_cycles_per_sample`
- `coherence`, `pullin_quality`
- `freq_err_intrablock_hz`, `freq_err_update_hz`
- `timing_proxy_ns`
- `dac_center_code`, `dac_target_float`, `dac_target_code`
- `dac_code_cmd`, `dac_code_rb`, `dac_step_req_codes`, `dac_step_applied_codes`

Key summary fields:

- median coherence and pull-in quality
- mean and median absolute phase error
- applied DAC update count and fraction
- timing proxy drift in `ns/s` and `samples/s`

## Tuning Notes

- If the correction direction is wrong, flip `--control-sign`.
- If the DAC barely moves, raise `--phase-frac-gain-codes`.
- If the DAC chatters too much, lower `--phase-frac-gain-codes` or tighten the
  quality thresholds.
- If the loop reacts too abruptly, lower `--max-dac-step-per-update`.
- The timing proxy is still an inferred metric from the same fixed-reference
  phase stream; it is useful for drift evaluation, not as a direct timestamp
  proof of simultaneous sampling.
