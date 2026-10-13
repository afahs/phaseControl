# Implementation and bench validation, October 13, 2026

The software is implemented and deployed through Git. Frequency acquisition has
been demonstrated in short tests. Sustained phase lock and the project's final
performance criteria have **not** been established. No startup service is enabled.

## Bench and evidence locations

- Pi 4 Model B Rev 1.5, Debian 12; modified RTL2832U/R820T-family SDR.
- HP8656A set to nominal 50 MHz; owner expects unmodulated operation. Reference
  selection, timebase option, front-panel modulation settings, and scope model
  remain unverified. See the specification discussion in [design.md](design.md).
- Owner confirms the injected 28.8 MHz drives both tuner and ADC clocks.
- MCP4725 at 0x62 and Si5351B at 0x60. Readback matches the repository's nominal
  27 MHz crystal / 864 MHz PLLB / divide-by-30 configuration and APR 120 ppm.
  Register arithmetic is not an independent measurement of the output frequency.
- Runtime data: `/home/rrrodriguez13/clock-runs/` on `project-pi`, outside Git.
- Every run has binary telemetry and JSON configuration, software revision,
  calibration, source settings, counters, and timing summaries. No CSV or raw IQ
  files were produced.

## Short live tests

Timing values below are histogram p99 upper bounds, excluding scheduled I2C from
DSP time. Callback age excludes unknown upstream USB/driver transport delay.

| Run prefix `implementation-` | Duration | Result | DSP p99 |
|---|---:|---|---:|
| `smoke-001` | 60 s | Requested 9.9 dB gain clipped heavily; 7,477/7,500 blocks rejected. No actuation. | 6.90 ms |
| `smoke-002` | 30 s | Gain 0 dB; one startup rejection, no reported loss, stale blocks, or slip warnings. | 2.75 ms |
| `characterization-001` | 78 s | Six ABBA cycles at 2048 +/-8; 26 writes; initial code restored. | 2.65 ms |
| `closed-001` | 60 s | Residual fell from about +916 Hz at 0.9 s to about -2 Hz at 4.9 s. Sustained phase lock absent. | 2.60 ms |
| `closed-002` | 120 s | 127 invalid blocks and 108 old-estimator slip warnings; locked fraction zero. | 3.35 ms |
| `held-001` | 30 s | DAC held at 2403; 52 invalid blocks and 39 old-estimator slip warnings, despite no actuation. | 5.75 ms |
| `held-002` | 30 s | Corrected unwrapping; 39 quality rejections, one slip warning; no writes or software loss. | 4.15 ms |
| `closed-003` | 60 s | Corrected unwrapping; 60 quality rejections, one slip warning, 307 DAC writes; locked fraction zero. | 3.15 ms |

All listed runs had zero reported application-buffer drops. This does not exclude
unreported device/driver sample loss. These short runs do not establish one-hour
reliability. Hardware-free tests ran concurrently during the first approximately
36 seconds of `closed-002`, so its CPU/timing data are not an isolated benchmark.

The calibration fit gave **-3.1710 Hz/code**, regression standard error
0.0233 Hz/code, local range 2040-2056, residual RMS 4.06 Hz. Correlated residuals
can invalidate the ordinary regression standard error; repeatability has not
been measured. Feedback around code 2403 extrapolated that local gain. Repeat
the calibration around the operating point and characterize nonlinearity before
using it as a full-range model. The stored 104 ms delay is a buffering budget,
not a measured step response or a verified physical upper bound.

The first closed-loop run lost detailed telemetry after its SSH stdout closed.
Control continued, but only 140 records were written and 460 were dropped.
The writer now isolates stdout failure; a regression test verifies that binary
logging continues. Later runs use detached execution with redirected output.

## Diagnosing phase variation

Two compact, receive-only FFT diagnostics at fixed code 2403 found spectral
structure close to the carrier. The four-second capture
`implementation-spectrum-independent.json` had no clipped samples. A separate
rectangular-window estimator measured a 1 ms absolute phase-change p99 of
0.611 rad, maximum about 0.930 rad, and no low-amplitude window pairs. This
confirms phase variation in the received samples using a second algorithm; it
does not identify the generator, clock, receiver, or path as its cause.

It also exposed a weakness in the initial control estimator: predicting the next
block midpoint from the previous 8 ms frequency fit can report false slips under
fast, continuous FM. Commit `2fe897c` instead unwraps adjacent filtered samples
at 32 ksample/s across blocks. A synthetic regression alternates +/-60 Hz every
8 ms with continuous phase and checks recovery without false slips. Earlier
`slip_events` counts must therefore **not** be reported as measured physical
cycle-slip counts. The new flag remains an ambiguity warning, not independent
proof of a lost cycle.

After that correction, `implementation-held-002` rejected 39/3,750 blocks on
the linear-phase single-tone SNR threshold (one also failed coherence and the
adjacent-sample phase test). Callback age p99 was 4.30 ms; application queue
high-water was one; CPU time was 11.01 s over 31.11 s wall time. Thus the
predictor correction improved continuity but did not resolve all quality failures
or meet the DSP target in this particular run. Rejections can reflect nonlinear
phase variation as well as additive noise: this SNR metric is the residual of
a fitted constant-frequency tone over an 8 ms block, not an independently
calibrated RF noise measurement. Do not relax its threshold merely to report lock.

The final `closed-003` check still did not establish phase lock. Command-age p99
was 4.35 ms, I2C p99 0.90 ms, and there were no stale blocks, missed scheduled
updates, or application-buffer drops. The run finished normally at code 2404;
no controller remains running. Detailed telemetry was analyzed directly to JSON
and a PNG in that run directory.

## Simulation with the measured gain

The default synthetic plant is much finer than the observed DAC response. Two
120-second sample-time simulations instead used -3.1710 Hz/code and the 104 ms
delay budget, with an 8 Hz initial offset and 0.002 Hz/s drift. The input pilot
had the simulator's additive noise, without the bench's observed rapid phase
variation. Neither run met the phase-lock criterion:

| Natural frequency | Modeled phase margin | Last 60 s frequency RMS | Last 60 s phase peak-to-peak |
|---|---:|---:|---:|
| 0.03 Hz | 60.42 degrees | 1.84 Hz | 691.76 rad (repeated reacquisition/drift) |
| 0.10 Hz | 48.52 degrees | 1.55 Hz | 4.36 rad |
| 0.30 Hz | 14.52 degrees | Rejected | Rejected |

This identifies quantization/control behavior as a limitation of this model even
with a relatively clean pilot. It does not prove that quantization explains all
bench variation. At this gain, one DAC code changes phase slope by approximately
19.92 rad/s. Evaluate reduced APR and repeat local gain calibration after measuring
warm-up/range requirements; do not recommend a hardware replacement or choose an
integration time based on the fine-resolution default simulation.

Local evidence directories are `data/measured-model-003` and
`data/measured-model-01` (ignored by Git). Their source-settings metadata records
the actual simulated gain and delay. These simulations used the calibration-to-
plant CLI correction before its commit; their Git field alone does not capture
that working-copy change. The same inputs can be rerun with the documented CLI.

## Automated validation

The hardware-free suite covers owned acquisition buffers and
overflow, signed gain, delayed quantized control, output/slew limits, saturation,
holdover/fault, bandwidth rejection, phase/frequency recovery, fast FM, dropout,
clipping, deterministic replay, calibration/restoration, storage stalls and caps,
stdout disconnection, and analytical Allan deviation for a frequency ramp.
Twenty tests passed on the Pi in 31.65 s before the final simulation-calibration
CLI regression was added. The three existing clock-configuration tests also passed.
Automated tests
never open the SDR or I2C devices.

## Remaining acceptance gates

1. Verify clock injection amplitude/coupling, actual VC voltage and filtering,
   generator AM/FM settings and reference selection, and scope capabilities.
2. Establish repeatable phase tracking after the estimator correction; use
   independent observations to locate residual signal-path/clock variation.
3. Measure actuator settling and end-to-end delay, gain across operating range,
   DAC quantization, and warm-up drift. Compare APR 120/60/30 between runs and
   choose the smallest range with measured 25% headroom.
4. Complete one-hour fixed-DAC and matched closed-loop runs, three cold starts,
   and eight-hour endurance. Quantify logging overhead with matched telemetry
   enabled/disabled runs. The sub-4 ms DSP target is not met in every observed
   condition, particularly frequent reacquisition.
5. Exercise deliberate runtime overload, real pilot interruption/attenuation,
   stream restart, and recovery. Simulation covers loss/failure logic but does
   not substitute for these hardware tests.
6. Report gap-aware Allan deviation, phase excursions, DAC activity, latency,
   CPU, and independent scope comparisons over supported averaging times.
   No meaningful long-term Allan-deviation improvement is claimed from these
   short, nonmatched runs.

Service operation follows these interactive gates. Two-station sample alignment
and wireless operation remain later milestones; one receiver and one tone cannot
validate either.
