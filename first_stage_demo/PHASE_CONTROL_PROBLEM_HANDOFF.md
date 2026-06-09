# Phase Control Problem Handoff

Date: 2026-05-20

This document is a context handoff for a new investigation thread. It describes
the overall goal, what was measured recently, what the data appears to say, and
which questions are still open. It is intentionally architecture-neutral: the
current DAC/VCXO control code is one implementation under test, not an
assumption about the correct final design.

## Overall Goal

The goal is to remove relative phase drift/jitter between a received RF tone and
the local sampling/clock/control reference. The useful end state is not merely a
low fitted frequency offset. The useful end state is a received tone whose phase,
after the chosen correction path is applied, remains stable enough for the
downstream measurement or synchronization task.

There are at least two separable problems:

- Frequency centering: remove the average or slowly varying residual frequency
  offset so phase does not wind linearly.
- Phase stabilization: reduce the remaining phase/timing motion after frequency
  centering.

Open-loop phase slope is not by itself the target metric. A constant incoming
phase can appear as a linear unwrapped phase ramp if the receiver/reference has a
frequency offset. Conversely, removing a linear phase slope can hide real slow
timing wander if that wander is the thing we need to characterize. For that
reason, recent analysis keeps both raw unwrapped phase and detrended residuals
visible.

## Current Hardware/Software Context

The active first-stage experiment uses:

- SDR sampling the RF tone.
- A phase detector on the complex tone estimate.
- A frequency/phase correction loop that can command a DAC/VCXO path.
- Optional open-loop and pseudo-open-loop modes for measurement.

The currently relevant scripts are:

- `first_stage_demo/cw_vcxo_discipline.py`
- `first_stage_demo/find_cw_tone.py`
- `first_stage_demo/analyze_cw_vcxo_log.py`
- `first_stage_demo/DAC_PHASE_ACTUATOR_LIMITS.md`

The live SDR runs require the SDR virtualenv and `PYTHONPATH` setup:

```bash
mkdir -p /tmp/ugradio_import
ln -sfn /home/rrrodriguez13/ugradio/ugradio_code/src /tmp/ugradio_import/ugradio

PYTHONPATH=/tmp/ugradio_import \
  /home/rrrodriguez13/sdr/.env/bin/python \
  first_stage_demo/cw_vcxo_discipline.py ...
```

When analyzing logs, avoid loading entire CSV tables into memory. The recent
logs can be large. Read only the columns needed, preferably with `csv.DictReader`
or chunked processing. The important columns have usually been:

- `t_rel_s`
- `phase_rad`
- `unwrapped_phase_rad`
- `f_off_hz`
- `amp`
- `coherence`
- control fields such as `dac_code_cmd`, `control_stage`, and `status` when
  debugging the loop itself

## Important Tone-Finding Result

A major recent issue was that the assumed tone location was wrong. This caused
low coherence and misleading frequency/phase estimates.

Earlier runs assumed a tone near a baseband offset of about `68578.125 Hz`.
Fresh scanning at the actual experiment center frequency found the useful tone
closer to:

```text
center_freq_hz = 49,915,671.875
tone_bb_ref_hz = 69,106.073533
detected RF   = 49,984,777.9485 Hz
```

Evidence:

- Fixed-center scan file:
  `first_stage_demo/logs/50mhz_fixed_center_tone_scan_20260514.csv`
- Validation run with the corrected tone:
  `first_stage_demo/logs/open_loop_validate_detected_tone_bb_69106hz_20260514.csv`
- The corrected-tone validation produced high coherence, around `0.92-0.99`,
  and amplitude around `0.44-0.47`.

Interpretation: the earlier low-coherence behavior was at least partly a
measurement setup problem, not necessarily a hardware phase-lock problem.
Future work should begin by confirming the tone location before interpreting any
phase-loop result.

## Recent Reference Runs

### Open loop with corrected phase detection

Reference file:

```text
first_stage_demo/logs/real_open_loop_corrected_phase_detector_50mhz_5min_20260506_121811.csv
```

Approximate metrics from prior analysis:

```text
phase slope:     -52.37 Hz
mean coherence:    0.9587
mean amplitude:    0.1419
```

This is a useful baseline for uncompensated drift with the corrected phase
detector, but the tone-bin issue discovered later means runs should be compared
carefully against the corrected detected-tone runs.

### Frequency-only closed loop

Reference file:

```text
first_stage_demo/logs/frequency_only_closed_loop_50mhz_5min_20260505_153905.csv
```

Approximate metrics from prior analysis:

```text
mean f_off:       about 0.61 Hz
timing RMS:       about 5.82 ns
```

This showed that some level of frequency correction is occurring. It did not
prove that the absolute/raw phase was being held near a stable value.

### Full phase correction closed loop

Reference file:

```text
first_stage_demo/logs/full_phase_closed_loop_50mhz_5min_20260506_134847.csv
```

Approximate metrics from prior analysis:

```text
timing RMS:       about 5.77 ns
```

The full phase loop did not clearly improve over the frequency-only loop. Around
`260 s`, plots showed a quick phase drop/correction toward near zero, but that
event looked isolated rather than evidence of sustained phase lock.

### Pseudo-open-loop after 30 s frequency centering, corrected tone

This is the most recent high-confidence measurement because it uses the corrected
detected tone.

Frequency-centering run:

```text
first_stage_demo/logs/frequency_centering_30s_detected_tone_before_pseudo_open_50mhz_20260514.csv
```

Pseudo-open-loop monitor:

```text
first_stage_demo/logs/pseudo_open_loop_after_30s_freq_centering_detected_tone_50mhz_5min_20260514.csv
```

Generated summary:

```text
first_stage_demo/logs/cleaned/pseudo_open_loop_after_30s_freq_centering_detected_tone_metrics_20260514.json
```

Key 30 s centering metrics:

```text
duration:                 29.996 s
phase slope:              -0.132 Hz
mean f_off:               -0.130 Hz
median f_off:              0.229 Hz
mean abs f_off:           32.045 Hz
p95 abs f_off:            86.465 Hz
mean coherence:            0.972
p05 coherence:             0.914
timing residual RMS:      62.18 ns
```

Key pseudo-open-loop metrics after the 30 s centering period:

```text
duration:                299.996 s
phase slope:              21.069 Hz
mean f_off:               23.813 Hz
median f_off:             22.959 Hz
mean abs f_off:           34.877 Hz
p95 abs f_off:            87.892 Hz
p99 abs f_off:           137.674 Hz
mean coherence:            0.975
p05 coherence:             0.927
raw timing peak-to-peak: 142876 ns
detrended timing RMS:      4697 ns
```

Minute-by-minute pseudo-open-loop fitted slopes were not constant:

```text
0-60 s:     31.71 Hz
60-120 s:   14.90 Hz
120-180 s:  26.87 Hz
180-240 s:  12.27 Hz
240-300 s:  32.60 Hz
```

Interpretation: the 30 s frequency centering can briefly get close, but once the
loop is opened the apparent residual frequency wanders over tens-of-seconds
timescales. The measurement coherence is high, so this is not simply the
low-coherence bad-tone-bin failure mode.

Relevant plots:

```text
first_stage_demo/logs/cleaned/pseudo_open_loop_after_30s_freq_centering_detected_tone_phase_frequency_full_20260514.png
first_stage_demo/logs/cleaned/pseudo_open_loop_after_30s_freq_centering_detected_tone_phase_frequency_one_sixth_260s_20260514.png
first_stage_demo/logs/cleaned/pseudo_open_loop_after_30s_freq_centering_detected_tone_psd_20260514.png
```

## What The Data Suggests So Far

### 1. The system can measure the tone cleanly when the tone bin is correct

The corrected-tone pseudo-open-loop run has mean coherence around `0.975`. That
is strong evidence that the very low coherence seen in earlier pseudo-open-loop
tests was caused by using the wrong tone location or search center.

This matters because low coherence can make the wrapped phase look strange and
can make `f_off` estimates unstable. Future analysis should always separate
measurement quality failures from control failures.

### 2. Frequency correction is real but not sufficient

The 30 s frequency-centering run reached a fitted phase slope near zero
(`-0.132 Hz`) over that short interval. That indicates the actuator/control path
can change the clock/tone relationship.

However, the following 5 minute pseudo-open-loop run drifted to an average
residual slope around `21 Hz`, with minute-scale slopes ranging roughly
`12-33 Hz`. So a one-time frequency correction does not hold the system centered
for long.

This could be real clock drift, environmental drift, actuator settling, thermal
motion, measurement bias, or some combination.

### 3. Phase motion has both slow and faster components

The PSD from the corrected-tone pseudo-open-loop run showed large raw unwrapped
timing power at low frequencies. Strong components in the raw timing PSD were in
the rough range:

```text
0.027 Hz to 0.14 Hz
periods about 7 s to 37 s
```

The wrapped phase PSD also had structure in the few-Hz to roughly 10-Hz region:

```text
2.23 Hz  -> period about 0.45 s
4.49 Hz  -> period about 0.22 s
7-11 Hz  -> periods about 0.09-0.14 s
```

The hardware update path was discussed as roughly `150 ms` for a clock update to
be pushed in the relevant setup. A 150 ms update corresponds to about `6.67 Hz`.
That means a slow servo could plausibly address the 7-37 s wander, but it is not
a comfortable actuator for phase content near or above several Hz. This is a
reason to consider separating slow frequency control from fast phase correction.

### 4. The DAC/VCXO path should not be assumed to be a fast phase actuator

The document `first_stage_demo/DAC_PHASE_ACTUATOR_LIMITS.md` argues that the
DAC/VCXO path is useful for slow frequency centering but is too slow for direct
phase cleanup. Earlier measured facts in that note include:

```text
open-loop residual phase changes after detrending:
2 ms:    0.634 rad RMS
10 ms:   1.652 rad RMS
20 ms:   2.378 rad RMS

DAC/VCXO latency:
about 32 ms median, with observed responses around 32 ms and 76 ms
```

Those measurements came from an earlier setup/date and should be revalidated if
the hardware or timing has changed. The architecture-level lesson is still
important: a frequency actuator changes phase only by integration, and latency
strongly limits phase-loop bandwidth.

## Current Leading Explanations For Why Full Phase Lock Is Not Working

These are hypotheses, not conclusions.

### Measurement noise or measurement bias

Evidence for:

- Earlier low-coherence runs were clearly affected by an incorrect tone
  location.
- Wrapped phase can have strange shapes when the tone estimate is not centered,
  coherence is low, or the phase detector sees significant motion within a
  block.

Evidence against it being the only issue:

- Corrected-tone pseudo-open-loop coherence is high, around `0.975`.
- Even with high coherence, residual slope and timing motion remain large over
  the 5 minute pseudo-open-loop run.

What to test:

- Split each block into independent sub-blocks and compare phase estimates. If
  independent estimates disagree more than expected from SNR, measurement noise
  or estimator bias is significant.
- Run two independent receivers or two independent DSP paths on the same data.
  Common phase motion suggests real phase/timing drift; uncorrelated motion
  suggests measurement noise.
- Recompute phase with multiple block sizes. True slow phase drift should be
  broadly consistent; estimator artifacts often change with block length/window.

### Actuator path cannot affect phase on the needed timescale

Evidence for:

- The DAC/VCXO path has measured tens-of-milliseconds physical response in older
  tests.
- The corrected-tone PSD contains phase structure near a few Hz and above.
- Full phase correction did not clearly beat frequency-only correction in prior
  5 minute comparisons.

What to test:

- Inject a known software phase rotation into the DSP path and verify that the
  phase detector reports it immediately and with the correct sign.
- Implement a software NCO/digital phase rotator as a fast correction path, while
  leaving the DAC/VCXO as a slow frequency-centering path.
- Compare raw phase before correction, digitally corrected phase, and DAC-only
  corrected phase on the same capture where possible.

### The controller objective may be mixing frequency and phase correction

Evidence for:

- A low fitted frequency error over a short interval does not guarantee stable
  phase over 5 minutes.
- Full phase correction events can appear to snap phase toward zero without
  sustaining low phase error afterward.
- In closed-loop phase mode, fitting a line to the phase error can be misleading;
  timing error should be computed directly as `phase_err / (2*pi*f_beacon)`, and
  residual frequency should be evaluated separately.

What to test:

- Define separate acceptance metrics for frequency centering and phase stability.
- Report raw unwrapped phase, wrapped phase, direct timing error, and frequency
  residual side by side.
- Run controller variants that are frequency-only, software-phase-only,
  DAC-frequency plus software-phase, and DAC-phase attempts, then compare them on
  the same metrics.

## Recommended Next Steps

1. Start every run with a tone scan or validation.

   Confirm the tone bin before interpreting loop behavior. A wrong tone location
   can produce low coherence and false conclusions.

2. Re-run a high-coherence open-loop baseline using the corrected detected tone.

   The most useful baseline would be a 5 minute open-loop run with no correction,
   using `tone_bb_ref_hz = 69106.073533`, and plots showing wrapped phase, raw
   unwrapped phase, timing, `f_off`, amplitude, and coherence.

3. Separate measurement noise from physical phase motion.

   Use sub-block consistency, alternate block sizes, and ideally independent
   receiver/DSP paths. The goal is to determine whether the phase motion is
   common-mode real motion or estimator noise.

4. Treat slow frequency control and fast phase correction as separate design
   questions.

   A plausible architecture is a slow DAC/VCXO frequency servo plus a fast
   software NCO or digital phase rotator, but this should be tested rather than
   assumed.

5. Characterize actuator dynamics in the current setup.

   Repeat latency/step-response measurements if the hardware or code path has
   changed. Measure command-to-observed-frequency delay, settling time, sign, and
   gain.

6. Use metrics that match the goal.

   Useful plots and metrics:

   - Raw wrapped phase.
   - Raw unwrapped phase.
   - Direct timing error, `phase / (2*pi*50e6)`.
   - Frequency residual, `f_off_hz`.
   - Coherence and amplitude as measurement-quality gates.
   - PSD of raw unwrapped timing and wrapped phase.
   - Short-window slopes, not only one global line fit.

## Useful Commands From The Recent Corrected-Tone Run

30 s frequency-centering run:

```bash
PYTHONPATH=/tmp/ugradio_import /home/rrrodriguez13/sdr/.env/bin/python \
  first_stage_demo/cw_vcxo_discipline.py \
  --mode closed_loop \
  --sample-rate 1024000 \
  --center-freq 49915671.875 \
  --tone-bb 69106.07353314661 \
  --block-size 2048 \
  --duration-s 30 \
  --use-mcp4725 \
  --dac-start 2050 \
  --vcxo-hz-per-command -4.5475 \
  --control-update-interval-s 0.05 \
  --control-average-interval-s 0.1 \
  --pull-in-switch-hz 30 \
  --pull-in-enter-dwell-updates 3 \
  --phase-target-rad 0.0 \
  --phase-hold-gain-hz-per-rad 0.0 \
  --phase-hold-max-freq-hz 0.001 \
  --phase-slew-gain-hz-per-rad 0.0 \
  --phase-slew-max-freq-hz 0.001 \
  --amp-threshold 0.02 \
  --coherence-threshold 0.2 \
  --out-csv first_stage_demo/logs/frequency_centering_30s_detected_tone_before_pseudo_open_50mhz_20260514.csv
```

5 minute pseudo-open-loop monitor:

```bash
PYTHONPATH=/tmp/ugradio_import /home/rrrodriguez13/sdr/.env/bin/python \
  first_stage_demo/cw_vcxo_discipline.py \
  --mode open_loop \
  --sample-rate 1024000 \
  --center-freq 49915671.875 \
  --tone-bb 69106.07353314661 \
  --block-size 2048 \
  --duration-s 300 \
  --amp-threshold 0.02 \
  --coherence-threshold 0.2 \
  --out-csv first_stage_demo/logs/pseudo_open_loop_after_30s_freq_centering_detected_tone_50mhz_5min_20260514.csv
```

Important caveat: in `open_loop` mode, the script uses a null actuator. Passing
`--dac-start` to open-loop mode does not by itself set the DAC. If a real DAC
state must be established before monitoring, do that explicitly with the helper
path or by running a closed-loop/frequency-centering step first.

## Open Questions For A New Investigation

- Is the high-coherence residual phase motion mostly real clock/timing drift, or
  still partly estimator noise?
- Does the current phase detector have block-size or intra-block sweep bias?
- How much of the residual frequency variation is real VCXO/reference drift
  versus measurement jitter in `f_off_hz`?
- What is the present command-to-effect latency and settling time of the
  DAC/VCXO path?
- Would a software NCO/digital phase rotator reduce phase error immediately on
  the same data where DAC-only full phase correction did not?
- Should the final architecture use separate fast phase and slow frequency
  actuators instead of trying to make the DAC/VCXO path do both?
