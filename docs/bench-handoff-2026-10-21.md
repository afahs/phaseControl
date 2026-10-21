# DAC and phase investigation handoff

This handoff covers the October 21, 2026 side conversation: scope investigation
of the DAC control voltage, a DAC step-and-return experiment, and a five-minute
fixed-DAC phase recording. It supplements `PI_HANDOFF.md` and `docs/validation.md`.

The DAC-to-clock control path responds with the expected polarity. Significant
received phase variation also occurs without DAC writes. The cause of that
variation remains unresolved; neither the scope observations nor these runs
establish that control-voltage ripple is responsible.

## Last verified state

- DAC code **2048**, verified after both completed SDR experiments. Earlier
  documentation saying 2404 describes the previous state and is now stale.
- Si5351B configuration readback matches nominal **28.8 MHz**, APR **120 ppm**,
  assumed 27 MHz crystal, PLLB denominator 1,000,000, VCXO parameter 506266,
  and CLK0 register 111. Register arithmetic is not an independent frequency
  measurement.
- The requested captures finished. No feedback loop or startup service was
  started in this conversation. Check for other activity before subsequent work.
- The modified SDR was detected after the user reconnected USB and successfully
  completed both experiments.
- Thermal equilibrium was not established. Probe connections during the SDR
  recordings were not recorded.
- Capture software revision: `83bd90310bc67a57f2b42f02323128c0f220266d`.

The user wants phase expressed in radians, rather than equivalent timing error.
Use compact binary telemetry and JSON; no CSV or raw IQ was recorded.

## Hardware context

The HP8656A supplies the nominal 50 MHz pilot and uses an external rubidium
reference that feeds no other part of this setup. Unmodulated CW is expected,
but the generator panel settings were not independently verified. The modified
SDR uses the injected 28.8 MHz reference for its clock needs. An MCP4725 drives
the Si5351B VC input directly; measuring the DAC output and VC input would
largely repeat the same measurement.

The available scope is a Tektronix TDS2012. Scope photographs showed AC coupling,
a 20 MHz bandwidth limit, and a 1x probe menu setting on CH2. Matching physical
probe attenuation was not confirmed. The working interpretation was CH1/yellow
on DAC output and CH2/cyan on the DAC supply, following the suggested hookup;
the channel mapping was not explicitly reconfirmed by the user.

## Scope observations

1. At the earlier fixed code 2404, the user measured a steady DC level of
   **1.98 V**. The nominal calculation using a 3.3 V supply gives about 1.94 V;
   the actual supply and voltage measurement accuracy were not calibrated.
2. The user reported about **30 mV peak-to-peak** at the DAC output. Touching
   the probe tip to its ground reduced this to about **7.2 mV peak-to-peak**.
   A floating probe produced a larger signal. Floating-probe pickup is not a
   measurement of control voltage.
3. Slow sweeps looked irregular. A possible 60 Hz component was suggested by
   the user but was never established.
4. At **100 ns/div**, approximately 29 cycles were visible across a 1 us
   screen. This is consistent with a component near the **28.8 MHz clock**;
   the photo does not establish an exact frequency or coupling path. Both
   channels showed a similar period. The scope's automatic frequency readouts
   varied and were not used as reliable measurements of that component.
5. With those settings retained, moving the DAC-output probe tip to local
   ground reduced the waveform to approximately **7 mV peak-to-peak**, while
   its frequency remained similar.

The reduction supports a clock-frequency voltage component at the measured
node as well as residual measurement pickup. It does not identify whether
coupling is through the supply, wiring, local ground, or probe geometry.
Peak-to-peak amplitudes cannot simply be subtracted because their relative
phases are unknown.

An earlier illustrative calculation assumed a real 30 mV sinusoidal control
variation at 60 Hz and used the historical DC tuning sensitivity. That example
gave roughly 2 rad peak-to-peak received phase modulation. **It does not apply
to the observed approximately 28.8 MHz component.** DC tuning gain alone cannot
predict the response to that high-frequency interference. No hardware fault,
filter modification, or replacement was established by the scope work.

## Software added for the scope observation

Commit `83bd903` added `beacon_clock/hold_dac.py` and was deployed through Git.
It reads the DAC at the beginning and end of a bounded interval and otherwise
waits quietly. It performs no DAC or clock writes, no SDR acquisition, and no
periodic I2C transactions. The DAC retains its existing volatile output code.

A 600-second observation was launched at code 2404, with log:

`/home/rrrodriguez13/clock-runs/dac-scope-hold-20261021-a784.log`

Startup was verified; its final log was not retrieved in this conversation.
Endpoint readback cannot exclude intermediate changes by another writer, nor
does it measure analog voltage stability. No controller or estimator source
changes were made for the subsequent SDR experiments.

## Clock reset and unsuccessful first attempt

Before the requested DAC-step test, the DAC unexpectedly read **2048** and the
Si5351B PLLB, output-divider, and VCXO parameter registers repeatedly read zero.
That is consistent with a reset or power cycle; the cause was not established.
The existing `inspect` command failed with `cannot convert float NaN to integer`
when decoding the unconfigured clock.

The documented 28.8 MHz/APR120 configuration was restored using the existing
initialization routine, with no SDR acquisition active. DAC readback remained
2048, and the clock configuration then passed inspection.

The first capture could not open the SDR. USB enumeration had failed, and the
receiver was absent from `lsusb`. It captured zero blocks and made zero DAC
writes. Its evidence remains on the Pi at:

`/home/rrrodriguez13/clock-runs/dac-step-20261021-a8f0/`

After the user reconnected the SDR's USB connection, the receiver enumerated
and the retry succeeded. No USB reset or power-cycle automation was added.

## Completed DAC step and return

Run: `dac-step-20261021-retry01`.

- Duration: 70 seconds of delivered-sample time, approximately 71.1 s wall time.
- Initial/final code: **2048**, with final hardware readback verified.
- Sequence: 5 s at 2048, three ABBA cycles at A=2050 and B=2046 with 5 s per
  dwell, then 5 s at 2048. Consecutive identical dwell codes caused no write.
- Eight volatile DAC writes; no closed-loop feedback.
- 8,750 binary records at 125 Hz, one rejected startup estimate, zero reported
  application-buffer drops, zero stale blocks, and zero telemetry drops.
- DSP p99 upper bound 2.80 ms; callback-age p99 upper bound 3.10 ms.

All eight transitions changed nearby phase slope in the expected direction:
increasing DAC code reduced the slope. Local straight-line phase fits used
windows 0.8 to 0.2 s before and 0.2 to 0.8 s after each command. Their median
inferred sensitivity was approximately **-16.7 rad/s/code**, with individual
values spanning about **-25.4 to -6.7 rad/s/code**. These comparisons include
background variation and are not a validated gain calibration.

The automatic characterization fit returned **-0.7373 Hz/code** and marked
`calibration.json` as `valid: true`. Its residual RMS was 11.16 Hz. Independent
phase-slope contrasts for the three ABBA cycles were **-11.94, +15.27, and
-14.17 rad/s/code**. The inconsistent cycle results show why the automatic
flag is insufficient: **do not use this run's calibration for feedback without
further validation**. The earlier historical -3.171 Hz/code measurement is
also not established as a current, repeatable calibration by this experiment.

The phase plot removes the initial fitted slope, approximately 8503.54 rad/s,
to expose changes. It also shows substantial slope variation while codes are
held constant. The test demonstrates actuation polarity, but does not measure
sub-block glitches or establish settling time. Telemetry associates a command
with the measurement that preceded its write; unknown upstream USB delay
prevents interpreting that as the exact sample time of physical actuation.

## Completed five minute fixed DAC recording

Run: `fixed-dac-20261021-01`, started at **2026-10-21 23:05:54 UTC**.
The existing receive-only `monitor` command ran for 300 seconds of sample time
(approximately 301.1 s wall time). DAC endpoint reads were **2048 before and
after**. There were **zero DAC writes and zero I2C transactions inside the
recording process**.

The capture contains 37,500 records at 125 Hz. It reported no application-buffer
or telemetry losses and no stale blocks. Two estimates were rejected, at about
0.004 s and 87.052 s. Analysis preserves the two resulting valid continuous
segments without bridging the break. The one `slip_events` warning is an
estimator ambiguity flag, not proof of a physical cycle slip. DSP p99 upper
bound was 2.75 ms and callback-age p99 upper bound was 3.05 ms.

For 297 complete, valid one-second windows, a separate phase offset and slope
were fitted and removed from each window:

| Statistic | Phase residual |
|---|---:|
| Median one-second RMS | 0.282 rad |
| 95th percentile of one-second RMS | 0.952 rad |
| Maximum one-second RMS | 2.121 rad |
| Median one-second peak-to-peak | 1.217 rad |
| Final-minute median one-second RMS | 0.210 rad |

These statistics exclude fitted constant frequency error and part of the slow
drift. They are not total phase error, locked-loop performance, or independent
oscillator phase noise. The fitted one-second phase slopes spanned about
**875.7 rad/s** over the run. Minute-by-minute median residual RMS values were
0.303, 0.545, 0.318, 0.229, and 0.210 rad: the final minute was quieter, but
thermal settling and stationarity were not demonstrated.

The fixed-run plot removes a different straight-line trend from each continuous
segment in its top panel. The different segment origins must not be interpreted
as a measured phase jump across the rejected block. Its other panels show
one-second phase slopes relative to their median and one-second detrended RMS.

**Conclusion:** received phase variation persists without software DAC updates.
This leaves analog DAC/supply noise, clock behavior, source/path behavior, and
receiver/estimator effects unresolved. It does not prove the DAC voltage itself
was constant, and does not connect the variation to the scope's 28.8 MHz signal.

## Evidence and reproduction

Completed-run directories are available both on the Pi and as local copies:

- Pi root: `/home/rrrodriguez13/clock-runs/`
- Local root: `C:/Users/Adam/Documents/ChatGPT/ClockSyncing/data/`
- Subdirectories: `dac-step-20261021-retry01/` and `fixed-dac-20261021-01/`.

Each contains `config.json`, `instruments.json`, `console.log`, and
`capture/run.json` plus `capture/telemetry.bin`. The step run additionally has
`capture/calibration.json`, `capture/characterization.json`,
`phase-response.json`, and `phase-response.png`. The fixed run additionally has
`endpoint.json`, `phase-report.json`, `phase-windows.json`, and `fixed-phase.png`.
Runtime data is ignored by Git; committing this handoff does not back it up.
The scope photos remain in the chat and temporary clipboard files, rather than
a curated repository evidence directory.

Both successful captures used 1.024 MS/s, 8,192 complex samples per block,
49.75 MHz center, nominal 50 MHz pilot, 0 dB tuner gain, four USB buffers, eight
application queue blocks, and the existing quality thresholds. Telemetry was
temporarily raised to 125 Hz for these bounded diagnostic captures. The binary
record size is 76 bytes, keeping the five-minute capture near 2.85 MB.

Read the binary records with `beacon_clock.telemetry.read_run`; use
`beacon_clock.analysis.clean_segments` to exclude invalid records and preserve
continuity. The monitor's stored DAC code **65535** means **unobserved**, not an
actual DAC command; use the separate endpoint observations for code 2048.

The custom analysis ran as one-off Python commands using NumPy and Matplotlib;
its JSON and PNG outputs are retained, but no reusable analysis command was
added. The methods and window definitions above describe how to reproduce the
reported statistics. The package's generic `analyze` command is not identical
to these custom reports.

## Outstanding work

1. Treat the current evidence as two findings: expected local actuation polarity
   and unresolved phase variation at fixed DAC. Avoid choosing new loop gains
   from the inconsistent automatic calibration.
2. Establish a reproducible warmed-up baseline, recording generator settings,
   power history, and scope probe connections. Then repeat bounded step tests
   with enough repeats to separate actuation from background drift.
3. Isolate one potential contributor at a time. Candidate investigations include
   low-frequency VC/supply behavior, received level and attenuation at fixed
   tuner gain, or a controlled source/reference substitution if equipment is
   available. These are proposed next steps, not completed experiments.
4. Improve `inspect` so an unconfigured clock produces a clear incompatible
   result instead of a NaN conversion error. Strengthen characterization
   acceptance with repeatability checks; the existing `valid` flag can pass
   despite inconsistent ABBA responses. Neither software change was made here.
5. Extract the useful analysis into a reproducible command if further runs use
   these metrics. Preserve gaps, explicit detrending definitions, and phase units.

Read `PI_HANDOFF.md` before Pi work. Recheck current processes, DAC code, and
clock configuration before actuation. Develop and commit source changes locally
and deploy through Git using `project-pi`; preserve password fallback. Keep
monitor receive-only, runtime data outside tracked source, and services disabled
until interactive validation passes. This handoff does not authorize starting
additional experiments or changing hardware.
