# First-Stage CW VCXO Discipline Demo

This directory contains a clean first-stage proof-of-concept for disciplining an SDR reference VCXO to a single external CW tone.

## Files

- `cw_vcxo_discipline.py`: real-time runner with `open_loop`, `closed_loop`, `vcxo_test`, and `vcxo_precision_test`
- `analyze_cw_vcxo_log.py`: CSV analysis and plotting helper
- `test_cw_vcxo_discipline.py`: hardware-free synthetic unittest for pull-in and handoff logic

## Hardware Assumptions

- Signal generator output goes by coax directly into the SDR RF input.
- The SDR reference is tunable through either:
  - an MCP4725 DAC path handled by `clock_vcxo_helper.py`, or
  - a Raspberry Pi PWM output followed by an analog low-pass filter.
- The tone is known ahead of time, either as RF frequency (`--tone-rf`) or as expected digital baseband offset (`--tone-bb` / `--expected-offset-hz`).

## Run Order

### 1. Open-loop first

Use this to confirm the tone is visible, block phasor amplitude is healthy, and the measured residual frequency offset is sensible.

```bash
python3 first_stage_demo/cw_vcxo_discipline.py \
    --mode open_loop \
    --sample-rate 240000 \
    --center-freq 100000000 \
    --tone-rf 100010000 \
    --block-size 16384 \
    --duration-s 60 \
    --out-csv first_stage_demo/logs/open_loop.csv
```

Then inspect the result:

```bash
python3 first_stage_demo/analyze_cw_vcxo_log.py first_stage_demo/logs/open_loop.csv
```

### 2. Characterize the VCXO

Sweep the control command and confirm the inferred frequency shift is monotonic. The script prints an estimated sensitivity in Hz per command unit and a recommended `--invert-control` setting.

```bash
python3 first_stage_demo/cw_vcxo_discipline.py \
    --mode vcxo_test \
    --sample-rate 240000 \
    --center-freq 100000000 \
    --tone-rf 100010000 \
    --block-size 16384 \
    --use-mcp4725 \
    --vcxo-test-start 1700 \
    --vcxo-test-stop 2400 \
    --vcxo-test-step 100 \
    --vcxo-test-dwell-s 5 \
    --vcxo-test-settle-s 1 \
    --out-csv first_stage_demo/logs/vcxo_test.csv
```

### 3. Refine Hz-per-command with a differential test

Use this after a coarse sweep has found the useful control region. It alternates around a center command, re-centers the tone with a short FFT in each dwell, fits phase slope, and estimates Hz-per-command while also modeling slow drift over time.

```bash
python3 first_stage_demo/cw_vcxo_discipline.py \
    --mode vcxo_precision_test \
    --sample-rate 1024000 \
    --center-freq 99900000 \
    --tone-rf 99968578.125 \
    --block-size 2048 \
    --use-mcp4725 \
    --vcxo-precision-center 2050 \
    --vcxo-precision-delta 20 \
    --vcxo-precision-cycles 6 \
    --vcxo-precision-dwell-s 2.0 \
    --vcxo-precision-settle-s 0.5 \
    --vcxo-precision-search-span-hz 5000 \
    --vcxo-precision-fft-samples 65536 \
    --vcxo-precision-min-fit-blocks 80 \
    --out-csv first_stage_demo/logs/vcxo_precision.csv
```

The script writes a companion `*_precision_summary.txt` file with:
- regression-based `estimated_sensitivity_hz_per_command`
- `pairwise_median_hz_per_command`
- `drift_hz_per_s`
- `recommended_invert_control`

### 4. Closed-loop

Closed-loop now uses a phase-first PLL-style architecture by default:

```text
IQ blocks -> tone phase detector -> clock state estimator -> phase controller -> calibrated VCXO actuator model
```

The controller's objective is reducing wrapped phase error. Frequency error is treated as the mechanism that moves phase, not as the final objective.

When using the MCP4725 path, fractional requested DAC codes are realized by first-order dithering between adjacent integer DAC codes. This helps when one physical DAC code is too coarse for the residual frequency correction you need.

The SDR source now uses the `ugradio` persistent async stream path with a bounded block queue instead of repeated `capture_data(...)` polling. The queue depth can be adjusted with `--ugradio-queue-blocks`.

The loop now decimates actuator writes and estimates phase/frequency state continuously. The main knobs are:
- `--control-update-interval-s`: actuator update cadence
- `--vcxo-hz-per-command`: measured physical VCXO sensitivity; negative if increasing DAC code lowers measured baseband frequency
- `--pll-estimator-alpha` / `--pll-estimator-beta`: phase/frequency state-estimator gains
- `--pll-frequency-proportional-gain` / `--pll-frequency-integral-gain`: calibrated frequency-servo gains

Closed-loop now also has a coarse recapture path. If the raw block estimator spends too long gated by low amplitude/coherence or too close to its phase-step ambiguity limit, the loop temporarily stops trusting the block-to-block phase estimate, performs a short FFT-based frequency reacquisition around the expected tone, and uses that coarse error to drive pull-in again.

Closed-loop still has two acquisition states:
- `pull_in`: reduce residual frequency until phase measurements are useful
- `fine_lock`: run the phase-alignment controller, which can intentionally request non-zero residual frequency to move phase toward target

When phase hold is enabled, it waits for a configurable number of settled fine-lock updates before arming, then converts phase error into a clamped and smoothed residual-frequency target. The calibrated actuator model converts that target-tracking error into DAC/PWM command using `--vcxo-hz-per-command`.

The phase path now distinguishes between two behaviors:
- `hold`: small bounded corrections that keep an already-good phase target from drifting
- `slew`: stronger bounded corrections that intentionally create a temporary frequency offset so phase can move toward an explicit target

That distinction matters because holding a phase difference constant only requires `delta_f -> 0`, but reducing an existing phase offset requires a deliberate temporary non-zero `delta_f`.

The fine controller is intentionally centered on phase alignment:
- phase error is converted into `control_target_f_off_hz`
- the single VCXO servo acts on `control_error_hz = clock_est_frequency_hz - control_target_f_off_hz`
- `--vcxo-hz-per-command` converts the desired Hz correction into command units

This matters because reducing an existing phase offset requires the clock to run deliberately fast or slow for a while. A pure frequency loop always tries to drive `f_off_hz` to zero immediately, which can stop phase motion before the two clocks are aligned.

The default handoff policy is:
- enter `fine_lock` after 3 consecutive valid updates with measured `|control_meas_f_off_hz| <= --pull-in-switch-hz`
- push back to `pull_in` after 2 consecutive `fine_lock` updates with measured `|control_meas_f_off_hz| >= --pull-in-exit-hz`

Useful pull-in knobs:
- `--pull-in-switch-hz`: residual threshold for handoff into `fine_lock`
- `--pull-in-exit-hz`: hysteretic threshold for pushback into `pull_in`
- `--pull-in-enter-dwell-updates`: consecutive below-threshold updates required before handoff
- `--pull-in-exit-dwell-updates`: consecutive above-threshold updates required before pushback; set to `0` to disable pushback

Useful physical-servo knobs:
- `--vcxo-hz-per-command`: calibrated Hz per DAC code or PWM duty unit
- `--pll-frequency-proportional-gain`: fraction of estimated frequency error corrected on each update
- `--pll-frequency-integral-gain`: slow command correction for residual bias after calibration
- `--pll-cycle-slip-threshold-rad`: phase-estimator residual threshold for rejecting likely cycle slips

Useful phase-hold knobs:
- `--phase-hold-arm-threshold-hz`: residual-frequency threshold required before the phase term is allowed to engage
- `--phase-hold-arm-dwell-updates`: consecutive fine-lock updates below that threshold before phase hold arms
- `--phase-hold-max-freq-hz`: clamp on the phase-derived frequency setpoint
- `--phase-hold-smoothing-alpha`: one-pole smoothing on the clamped phase-derived setpoint
- `--phase-slew-gain-hz-per-rad`: stronger gain used when an explicit target phase is still far away
- `--phase-slew-max-freq-hz`: separate clamp for deliberate phase slewing
- `--phase-slew-enter-error-rad` / `--phase-slew-exit-error-rad`: hysteresis between slew mode and hold mode

Useful recapture knobs:
- `--recapture-invalid-blocks`: consecutive bad blocks required before coarse reacquisition starts
- `--recapture-ambiguity-fraction`: fraction of the block estimator ambiguity limit treated as unsafe for normal updates
- `--recapture-dwell-blocks`: raw blocks accumulated for each coarse FFT estimate
- `--recapture-search-span-hz`: coarse FFT search half-span around the nominal tone
- `--recapture-release-hz`: coarse residual threshold that returns control to the normal pull-in/fine estimator

Important: `closed_loop` without `--phase-hold-gain-hz-per-rad` is still a frequency-discipline loop. It can flatten long-term phase slope, but it does not actively hold phase to a constant target. If constant phase is the objective, enable phase hold with a non-zero `--phase-hold-gain-hz-per-rad`.

```bash
python3 first_stage_demo/cw_vcxo_discipline.py \
    --mode closed_loop \
    --sample-rate 240000 \
    --center-freq 100000000 \
    --tone-rf 100010000 \
    --block-size 16384 \
    --duration-s 180 \
    --use-mcp4725 \
    --dac-start 2048 \
    --vcxo-hz-per-command -16.74 \
    --pull-in-switch-hz 30 \
    --pull-in-enter-dwell-updates 3 \
    --control-update-interval-s 0.05 \
    --amp-threshold 0.02 \
    --out-csv first_stage_demo/logs/closed_loop.csv \
    --plot
```

For phase hold rather than only frequency pull-in, a practical starting point is:

```bash
python3 first_stage_demo/cw_vcxo_discipline.py \
    --mode closed_loop \
    --sample-rate 1024000 \
    --center-freq 99900000 \
    --tone-rf 99968578.125 \
    --block-size 2048 \
    --duration-s 60 \
    --use-mcp4725 \
    --dac-start 2042.4 \
    --vcxo-hz-per-command -16.74 \
    --invert-control \
    --phase-target-rad 0.0 \
    --phase-hold-gain-hz-per-rad 1.0 \
    --phase-hold-arm-threshold-hz 25 \
    --phase-hold-arm-dwell-updates 12 \
    --phase-hold-max-freq-hz 3.0 \
    --phase-hold-smoothing-alpha 0.3 \
    --phase-slew-gain-hz-per-rad 3.0 \
    --phase-slew-max-freq-hz 10.0 \
    --control-update-interval-s 0.05 \
    --amp-threshold 0.02 \
    --coherence-threshold 0.2 \
    --out-csv first_stage_demo/logs/closed_loop_phase_hold.csv
```

The CSV now records both `phase_hold_enabled` and `phase_hold_active`, plus the phase-hold arm counter, the raw/applied phase-hold frequency setpoints, `clock_est_frequency_hz`, `control_target_f_off_hz`, `control_meas_f_off_hz`, and `control_error_hz`. This lets you confirm whether the phase loop armed and whether the frequency servo is tracking the phase-derived residual-frequency target. `phase_hold_cmd_offset` is retained as a compatibility column and should remain zero in the PLL controller.

If the VCXO sweep showed negative Hz-per-command slope, repeat the closed-loop run with `--invert-control`.

The CSV now records `control_stage`, `control_stage_switch_event`, and the pull-in counters so you can confirm whether a run stayed in pull-in too long, handed off cleanly, or pushed back.

### Measuring Hz per DAC code

The PLL controller is strongest when `--vcxo-hz-per-command` comes from a direct measurement instead of a guessed gain. Common lab setups:

1. Frequency counter method:
   - Feed the SDR reference output, VCXO output, or a clock-derived RF output into a frequency counter with a stable 10 MHz reference.
   - Set DAC code `C0`, wait for thermal/electrical settling, and record frequency `F0`.
   - Set DAC code `C1 = C0 + delta`, wait the same dwell time, and record `F1`.
   - Compute `(F1 - F0) / delta`.
   - Repeat `C0 -> C1 -> C0 -> C0 - delta` several times and use the median slope. The sign matters: if frequency falls when DAC code rises, pass a negative value.

2. Signal generator plus SDR method:
   - Lock the signal generator to a stable reference if available.
   - Inject a fixed CW tone into the SDR.
   - Hold each DAC code and estimate the tone's baseband frequency with `vcxo_precision_test`.
   - Use the generated `estimated_sensitivity_hz_per_command` as `--vcxo-hz-per-command`.

3. Oscilloscope phase-drift method:
   - Put the reference clock and a stable lab reference on two channels.
   - Measure phase slope over time at DAC code `C0`, then at `C1`.
   - Convert phase slope difference to frequency difference: `Hz = (rad/s) / (2*pi)`.

Use a delta large enough that the frequency shift is well above counter resolution and short-term noise, but small enough to stay in the local linear region. For the current MCP4725 setup, `delta=20` codes has been a useful starting point.

### 5. Synthetic smoke test

This does not require SDR hardware. It verifies that the pull-in stage can reduce a synthetic initial residual, hand off after the configured dwell, and log the transition correctly.

```bash
python3 -m unittest discover -s first_stage_demo -p 'test_*.py'
```

## What Success Looks Like

- `open_loop`: unwrapped phase drifts steadily and `f_off_hz` has a non-zero mean.
- `vcxo_test`: the mean `f_off_hz` changes monotonically with command.
- `vcxo_precision_test`: plus/minus dwells give a consistent local Hz-per-command estimate and the fitted drift term is smaller than the command-induced shift.
- `closed_loop` without phase hold: startup begins in `pull_in`, then `control_stage` transitions to `fine_lock`; the late-time mean `f_off_hz` moves closer to zero and the unwrapped phase slope becomes flatter than in open-loop.
- `closed_loop` with phase hold: `phase_hold_active` becomes true, `phase_error_rad` decreases or stays bounded near the target, and `control_error_hz` stays small while `control_meas_f_off_hz` tracks the phase-derived `control_target_f_off_hz`.

## Failure Diagnosis

- No response to VCXO control:
  - check wiring, DAC/PWM output, and whether the SDR reference really uses the controlled VCXO
- Wrong sign:
  - run `vcxo_test` and use the printed `recommended_invert_control`
- Saturation:
  - widen command range or move `--dac-start` closer to the useful control midpoint
- Noisy phase estimate:
  - increase `--block-size`, improve tone level, or tighten RF tuning so the expected tone is closer to the true baseband offset
- Block size too small:
  - frequency estimate becomes noisy even if latency improves
- Block size too large:
  - if residual `|f_off|` approaches `0.5 / T_block`, the phase-difference estimator becomes ambiguous
