# Why the DAC/VCXO Path Is Too Slow for Direct Phase Updates

Date: 2026-04-27

This note explains why the DAC-controlled VCXO should not be treated as the
fast phase actuator in this setup. The conclusion is architecture-independent:
it follows from the measured phase disturbance timescale, the measured actuator
latency, and the fact that the DAC only changes oscillator frequency. A
frequency actuator changes phase only after integrating frequency over time.

## Executive Summary

The DAC/VCXO path is useful for slow frequency centering, but it is too slow for
direct phase correction.

Measured facts:

- Open-loop phase, after removing the mean frequency offset, changes by about
  `0.63 rad RMS` in `2 ms`, `1.65 rad RMS` in `10 ms`, and `2.38 rad RMS` in
  `20 ms`.
- The measured DAC step response latency is about `32 ms` median, with observed
  nonzero step estimates around `32 ms` and `76 ms`.
- The I2C/write time itself was only about `1.4-1.6 ms`, so the slow part is the
  physical actuator path: DAC output, analog filtering, VCXO response, SDR clock
  response, and measurement delay.
- Closed-loop runs reduced long-term phase slope from about `-95.77 Hz`
  open-loop to `0.15-0.48 Hz`, but circular phase concentration remained low:
  `R = 0.0147` at 100 MHz and `R = 0.0535` at 50 MHz. `R = 0` is random phase;
  `R = 1` is fixed phase.

The practical implication is that the DAC can reduce frequency error, but it
cannot chase millisecond-scale phase motion. A better architecture is:

- Fast phase actuator: software NCO / digital phase rotation / downstream phase
  correction.
- Slow frequency actuator: DAC/VCXO bias servo that keeps the software NCO
  frequency correction near zero.

## How Fast a Phase Actuator Needs To Be

For any actuator that changes frequency rather than phase directly:

```text
dphi/dt = 2*pi*df
```

So the fastest possible correction time for a phase error `phi` with available
frequency authority `df_max` is:

```text
t_correct >= |phi| / (2*pi*df_max)
```

This is the best case. It assumes the command takes effect instantly. A real
actuator also has latency:

```text
t_total >= actuator_latency + |phi| / (2*pi*df_max)
```

The phase measurements show that residual phase moves on millisecond to
tens-of-milliseconds timescales. If the target is to correct a `1 rad` phase
error before it changes substantially, the correction needs to happen on the
order of a few milliseconds, not tens of milliseconds.

Examples:

```text
If t_correct = 2 ms:  df_max >= 1 / (2*pi*0.002) = 79.6 Hz
If t_correct = 5 ms:  df_max >= 1 / (2*pi*0.005) = 31.8 Hz
If t_correct = 10 ms: df_max >= 1 / (2*pi*0.010) = 15.9 Hz
```

Those numbers are only the frequency authority requirement. They do not include
actuator latency. With a measured latency of about `32-76 ms`, the command
arrives after the relevant phase state has already changed.

Delay also limits feedback bandwidth. A common control rule is that usable
closed-loop bandwidth must stay well below the delay-limited phase-margin point:

```text
bandwidth << 1 / (4 * actuator_latency)
```

Using the measured latency:

```text
T_delay = 32 ms  -> conservative bandwidth below about 8 Hz
T_delay = 76 ms  -> conservative bandwidth below about 3 Hz
```

The observed phase disturbance is important at `2-20 ms`, corresponding to
roughly `50-500 Hz` timescales. A `3-8 Hz` actuator loop cannot directly control
that phase.

## Measured Phase Timescale

The open-loop run was analyzed by fitting and subtracting the mean frequency
slope, then measuring how much residual phase changes over different time
separations.

Source log:

```text
first_stage_demo/logs/real_open_loop_corrected_phase_detector_10min_20260424.csv
```

Key measurements after detrending:

```text
2 ms:   RMS 0.634 rad, p95 1.218 rad
4 ms:   RMS 0.988 rad, p95 1.923 rad
10 ms:  RMS 1.652 rad, p95 3.240 rad
20 ms:  RMS 2.378 rad, p95 4.660 rad
50 ms:  RMS 3.794 rad, p95 7.427 rad
100 ms: RMS 5.362 rad, p95 10.537 rad
```

This means that even if the average frequency offset were perfectly removed,
absolute phase is not stable for long. The fast control problem lives on the
millisecond scale.

![Open-loop phase decorrelation](logs/phase_architecture_report/open_loop_phase_decorrelation_timescale.png)

## Measured DAC/VCXO Latency

A dedicated latency test stepped the DAC between codes `2050` and `2055` while
measuring the SDR tone phase/frequency at `1 ms` block cadence.

Source files:

```text
first_stage_demo/logs/actuator_latency_50mhz_step5_20260427.csv
first_stage_demo/logs/actuator_latency_50mhz_step5_20260427.txt
```

Measured summary:

```text
DAC step: 2050 <-> 2055
nominal slope: -4.5475 Hz/code
expected frequency step: +/-22.7375 Hz
observed frequency step: about 34-37 Hz in this run
write elapsed time: about 1.4-1.6 ms
estimated latency median: 31.96 ms
estimated latency range: 0.00003-75.85 ms
```

The first `0.00003 s` latency estimate is likely a threshold artifact: the
smoothed estimator was already near the crossing threshold. The useful nonzero
responses are approximately `32 ms` and `76 ms`.

The important point is that the software write is not the dominant delay. The
write itself is only around `1.5 ms`. The effective frequency response seen by
the SDR takes tens of milliseconds.

![DAC step response](logs/phase_architecture_report/actuator_step_response_latency.png)

## Why Frequency Centering Worked but Phase Lock Did Not

The closed-loop runs did reduce long-term frequency/phase slope. That shows the
DAC is a useful slow actuator.

However, the phase remained broadly distributed. The circular concentration
metric `R` stayed close to zero:

```text
Open-loop 100 MHz:
  phase slope: -95.766 Hz
  phase R: 0.0007
  mean coherence: 0.875

Closed-loop 100 MHz, split-command architecture:
  phase slope: 0.478 Hz
  phase R: 0.0147
  mean coherence: 0.924

Closed-loop 50 MHz, detected tone:
  phase slope: 0.150 Hz
  phase R: 0.0535
  mean coherence: 0.873
```

Interpretation:

- The slow DAC loop is effective at reducing average frequency error.
- The tone measurement is coherent enough to see the signal.
- The loop still does not hold phase at a fixed angle.
- Therefore the blocker is not simply tone visibility or frequency centering;
  it is the mismatch between phase disturbance timescale and actuator response
  time.

![Frequency centering versus phase lock](logs/phase_architecture_report/frequency_centering_vs_phase_lock_metrics.png)

## Architecture-Independent Reasoning

This result does not depend on the specific controller implementation. Any
architecture that tries to use the DAC/VCXO as the fast phase actuator has the
same constraints:

1. The actuator changes frequency, not phase.
2. Phase correction happens only after integrating the frequency change.
3. The measured phase state changes significantly within `2-20 ms`.
4. The actuator response arrives after roughly `32-76 ms`.
5. A delayed feedback loop must run far below the delay-limited bandwidth.

So a DAC/VCXO-only phase loop is structurally disadvantaged. Increasing phase
gain does not solve this; it mainly increases the chance of late corrections,
overshoot, and command dithering. The actuator is reacting to an old phase
measurement.

## What the Architecture Should Do Instead

Use a two-actuator interpretation, even if there is only one physical clock
actuator:

### Fast Path: Digital Phase Actuator

Use a software NCO, digital mixer, timestamp correction, or downstream phase
rotation as the fast phase actuator. This path can apply phase correction
immediately in the digital domain.

Its goal:

```text
hold observed phase near the target phase
```

This path can operate at the block cadence, for example `1-3 ms`.

### Slow Path: DAC/VCXO Frequency Actuator

Use the DAC only to remove the average frequency bias that accumulates in the
software NCO.

Its goal:

```text
drive long-term NCO frequency correction toward zero
```

This path should have conservative bandwidth, consistent with the measured
`32-76 ms` latency. It should not try to chase block-to-block phase.

### Quality Gating

The phase detector should reject or downweight phase updates when the short
window is not locally coherent. The current code now logs:

```text
phase_window_step_rms_rad
phase_window_max_step_rad
```

and gates phase updates with:

```text
--phase-step-rms-gate-rad
```

The phase integral should only arm after repeated high-confidence, low-step-RMS
phase windows. Otherwise it integrates noise and phase-wrap ambiguity.

## Design Rule from These Measurements

For this setup:

```text
DAC/VCXO: use for slow frequency centering
Software NCO/digital phase: use for phase locking
```

The DAC should not be the primary phase actuator unless the measured phase
disturbance becomes much slower, or the actuator latency is reduced by at least
an order of magnitude.

An architecture that only actuates the DAC can reduce phase drift rate, but it
should not be expected to produce tight absolute phase lock under the measured
conditions.
