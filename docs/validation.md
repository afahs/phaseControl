# Implementation and bench validation

## November 11, 2026: 100-code gain map and failed acquisition

Evidence: `~/clock-runs/grid30-20261111`, also copied under local `data/`.
Thirty-three local calibrations cover 500-3600 every 100 codes plus 3137, at APR
30 ppm. Each uses +/-8 codes, three ABBA cycles and 2-second dwells with the first
second excluded. Codes 2200, 2600, and 3500 failed the 10% relative cycle-spread
gate initially and passed six-cycle, 3-second-dwell retries. Rejected recordings
are retained. Accepted cycle spread ranges up to 9.99%; fine differences between
neighboring points are not all resolved. This is a local differential-gain map,
not a precision absolute frequency-versus-code or dynamic response measurement.

Fitted gains span -0.89347 to -0.71578 Hz/code; at 3137 the fit is -0.783008
Hz/code. The controller integrates linearly interpolated local gains and inverts
that response for corrections, preserving the frequency reference at its bias
code. Configured output limits 500-3600 remain within measured map coverage.
The delay field is still the unmeasured 104 ms buffering budget. All 33 entries
are embedded in the feedback run's metadata.

The requested 300-second feedback run started at 3137, completed without a runtime
exception, but **never entered TRACK or reported lock**. It logged all 37,500
records with no drops, stale blocks, or missed scheduled updates. There were 311
invalid blocks (SNR gate; startup also coherence/ambiguity), 1,996 DAC writes,
and repeated ACQUIRE/HOLDOVER transitions. DAC codes ranged 3103-3137, ending
at 3118. There is no captured phase target, so no valid phase-lock timing RMS
can be reported. Zero lock-loss events here means lock was never acquired.

Replay of the recorded measurements reproduced every DAC code and controller
state. Analytic positive/negative nonlinear gain-map inversion, scalar equivalence,
range/step limits, and nonlinear-plant acquisition tests passed. This checks
software arithmetic, not the physical response model. No measurement identifies
the underlying cause of the acquisition oscillation yet. Static gain calibration
has not established adequate acquisition stability, actuator settling, or delay.
The earlier scalar runs had 75-81% locked time; this run performed substantially
worse, with different starting conditions and elapsed bench time. It is not a
controlled isolation of the map's causal effect.

Plots: `gain-map.png` shows cycle spread; `feedback-status.png` shows frequency,
DAC activity, and tone-fit quality, including a 60-70-second detail. Feedback
remains stopped, DAC 3118 and nominal 28.8 MHz/30 ppm verified after the run.
Source was developed/committed locally and deployed through Git. The original
30-test suite and four Pi map tests passed before acquisition; two additional
resume/gating tests passed before resuming the sweep.

## November 11, 2026: operating-point calibration and fixed-DAC baseline

Evidence: `~/clock-runs/local3137-20261111` on `project-pi`, copied locally under
`data/local3137-20261111`. Clock readback matched nominal 28.8 MHz and APR 30 ppm.
Generator panel/reference-lock state was not newly verified. Scripts are
`beacon_clock.calibrate_baseline` and `beacon_clock.analyze_baseline`.

Ten ABBA cycles at 3137 +/-8, with 3-second dwells and the first second excluded
from each dwell, measured **-0.789006 Hz/code**. The ten cycle slopes had sample
standard deviation **0.007330 Hz/code**. This is 3.22% lower magnitude than the
previous -0.815287 Hz/code calibration centered at 2048. The new calibration
applies to codes 3129-3145. Its 104 ms delay remains an unmeasured buffering
budget. Calibration rejected startup and one measurement near 78.73 seconds;
both carried estimator ambiguity warnings, not proof of physical cycle slips.

The subsequent 300-second monitor run had 37,500 complete telemetry records,
zero DAC writes, no dropped or stale blocks, and code 3137 verified before/after.
Monitor records use code 65535 as an unknown-code sentinel; endpoint observations
are saved separately. Its only invalid/slip warning was at startup (0.004 s).
No invalid event recurred near 77-79 seconds. Feedback remains off.

Timing at 50 MHz accumulated approximately +5301 ns over the baseline. To expose
short excursions independently of that drift, subtract a centered 10.008-second
moving mean within each valid segment, omitting window edges and startup FIR
transients. Apply exactly the same filter to the previous closed-loop tracking
records; these metrics are **local residuals, not absolute timing/lock error**:

| Recording | Local residual RMS (ns) | Absolute peak (ns) |
|---|---:|---:|
| Fixed DAC 3137 | 2.434 | 10.768 |
| Previous closed trial 1 | 1.068 | 7.519 |
| Previous closed trial 2 | 1.000 | 8.648 |
| Previous closed trial 3 | 1.032 | 6.977 |

The largest baseline local excursion occurred at 68.22 s; within +/-3 s its
maximum callback age was 3.50 ms and DSP time 3.28 ms. Thus comparable excursions
exist without DAC writes and without a coincident large measured processing
delay. This weakens a DAC-switching-only explanation; it does not locate the
cause among generator/reference, clock/VC supply, receiver/path, or estimation,
nor exclude unmeasured upstream timing effects. One sequential baseline is not
a controlled statistical proof of closed-loop improvement. The new local
calibration has not yet been exercised in another feedback run.

## October 13, 2026 historical validation

The software is implemented and deployed through Git. Frequency acquisition has
been demonstrated in short tests. Sustained phase lock and the project's final
performance criteria have **not** been established. No startup service is enabled.

## Bench and evidence locations

- Pi 4 Model B Rev 1.5, Debian 12; modified RTL2832U/R820T-family SDR.
- HP8656A set to nominal 50 MHz; owner expects unmodulated operation and reports
  an external rubidium reference that feeds no other part of this setup. Its
  model/frequency are unspecified; investigating the rubidium unit is not the
  first priority. Front-panel modulation settings remain to be checked.
  Scope: Tektronix TDS2012 (owner report).
  See the specification discussion in [design.md](design.md).
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
The final 23-test suite passed locally in 8.45 s and on the Pi in 33.10 s at
commit `e605746`. It also covers measured calibration in the simulated plant,
distinct Allan averaging times, and rejection of stale DAC proposals without
diverging from the actual applied output. The three existing clock-configuration
tests passed on the Pi. Automated tests never open the SDR or I2C devices.

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

## Next bench investigations

The primary plot now shows unwrapped relative phase in radians. Each continuous
segment has its own origin; it is not the captured controller setpoint, which
was not retained in these historical records. Existing gaps remain visible.

Investigate two separate questions: phase variation with fixed DAC, and coarse
actuation during feedback. Do not assume either explains the other.

1. **MCP4725, VC wiring, and power.** Scope VC at a fixed code for ripple, then
   observe small bounded code steps for size, settling, and glitches. Compare
   with clock behavior; a flat-looking scope trace only bounds noise above the
   instrument's floor. Check local decoupling and ground connections. Do not add
   an arbitrary RC filter before measuring its effect on loop delay.
2. **Si5351B and injected clock.** Check the 28.8 MHz signal at the SDR input for
   amplitude, offset, ringing, and coupling using a suitably low-loading probe.
   Recalibrate near code 2404, measure response delay, and compare 120/60/30 ppm
   APR between stopped runs after establishing warm-up drift/headroom. Reduced
   range should reduce Hz/code; measure rather than assume exact scaling. The
   configurable VCXO gain/range is documented in the
   [Si5351B datasheet](https://www.skyworksinc.com/-/media/Skyworks/SL/documents/public/data-sheets/Si5351-B.pdf).
3. **SDR analog path and estimation.** With source and DAC held fixed, compare
   input attenuation at fixed tuner gain and compare center-frequency offsets.
   Continue both phase estimators. This tests compression, baseband artifacts,
   and model/quality-gate failures; it does not require the unmodified second SDR.
4. **HP8656A RF output, if variation remains unexplained.** First simply confirm
   reference lock and AM/FM off, and check level/termination. The rubidium source
   feeds only the generator; do not start by treating that source as faulty.
   External reference operation does not by itself establish low RF phase noise.
   If the preceding tests are inconclusive and another suitable RF source is
   available, substitute it at matched received power with the DAC fixed. A
   change in phase statistics implicates the source chain; no change shifts
   attention toward the clock/receiver path.
5. **Then retune and validate.** Use measured response/quantization in simulation,
   log the actual phase setpoint/error in a versioned telemetry update, establish
   short repeatable lock, and proceed to matched one-hour open/closed runs and
   endurance. Consider finer actuation only if reduced APR with adequate headroom
   still leaves excessive measured phase limit cycles.

The [TDS2012 specification](https://www.tek.com/de/datasheet/tds1000-and-tds2000-series-oscilloscopes)
lists two channels, 100 MHz bandwidth, up to 1 GS/s, 2,500-point records, and
50 ppm timebase accuracy. Use it first for signal integrity, VC ripple/steps,
and coarse relative behavior. Disable its 20 MHz bandwidth limit when inspecting
the RF/clock signals. Its absolute frequency readout is insufficient evidence
for a few-hertz correction at 50 MHz. Same-source two-channel measurements can
establish a relative measurement floor; sample spacing alone is not phase
precision. External trigger is not an external frequency-reference input.

Because 50 MHz and 28.8 MHz differ, a simple edge-phase readout between them is
not the desired clock error. A quantitative comparison needs a verified frequency
ratio method, such as suitable coherent division/downconversion, or a reference-
locked counter/phase instrument. Establish the available equipment and its floor
before proposing purchases. No hardware replacement is yet justified by the
current measurements alone.
