# Beacon clock discipline

Experimental frequency and relative-phase discipline for a modified RTL2832U SDR,
using a 50 MHz pilot, Pi, MCP4725 DAC, and Si5351B 28.8 MHz external clock.
This does **not** establish absolute time or sample alignment between stations.

## Architecture

One persistent SDR acquisition thread copies USB buffers into an owned pool. The
DSP/control worker uses a fixed coarse mixer, stateful FIR, and phase/frequency
estimation. A separate writer saves compact binary telemetry in batches. There is
no CSV output, per-sample Python processing, or file I/O in the control path.

The acquisition queue holds eight 8192-sample blocks at 1.024 MS/s. Overflow drains
stale queued work and marks a new continuity segment. USB timestamps indicate
callback arrival, not sample time; unknown device-side loss remains possible.
The direct `read_bytes_async` API avoids pyrtlsdr's extra asynchronous queue.

Control acquires frequency, then uses a PI phase servo. The setpoint is captured
for each continuous run. Invalid/stale data enters holdover; stream or DAC errors
end the run with a fault summary. Normal actuation writes only the MCP4725 volatile
register. Clock initialization is explicit and happens before streaming.

## Installation and tests

Python 3.10+ and NumPy are required. Hardware additionally needs librtlsdr,
pyrtlsdr, smbus2, USB permissions, and I2C permissions. Matplotlib is optional for
plots. Install locally with `python -m pip install -e '.[hardware,plots]'`.
The Pi already has the dependencies in `/home/rrrodriguez13/sdr/.env/bin/python`.
Use `python -m beacon_clock` directly from this checkout; installation is optional.

```bash
python -m unittest discover -s tests -v
python -m unittest test_setClockFreq -v
python -m beacon_clock defaults --output bench.json
python -m beacon_clock discipline --simulate --duration 120 --output /tmp/beacon-sim
python -m beacon_clock analyze /tmp/beacon-sim --plots
```

On Windows sandboxes, set Python's `tempfile.tempdir` to a writable test directory
if the OS-provided temporary directory is inaccessible. Tests never open hardware.

## Hardware workflow

Read [PI_HANDOFF.md](PI_HANDOFF.md) before deployment. Review the wiring, clock
injection/coupling, generator reference, and scope measurement floor before
interpreting results. The owner reports that the injected reference drives both
the tuner and ADC. DAC supply is 3.3 V; default code limits 410–3685 stay within
10–90% of that supply. The legacy code identifies DAC address 0x62 and clock 0x60;
the inspect command verifies responses rather than scanning the bus.

From the Pi checkout, using its existing Python environment:

```bash
PY=/home/rrrodriguez13/sdr/.env/bin/python
$PY -m beacon_clock inspect
$PY -m beacon_clock spectrum --seconds 4 --output ~/clock-runs/spectrum.json
$PY -m beacon_clock monitor --duration 60 --output ~/clock-runs/smoke
$PY -m beacon_clock analyze ~/clock-runs/smoke --plots
$PY -m beacon_clock monitor --duration 3600 --output ~/clock-runs/baseline
$PY -m beacon_clock characterize --output ~/clock-runs/characterization
$PY -m beacon_clock discipline --duration 60 \
    --calibration ~/clock-runs/characterization/calibration.json \
    --output ~/clock-runs/closed-smoke
```

Every output directory must be new. Monitor never writes the DAC or clock.
The bounded spectrum command keeps IQ in memory and saves only compact JSON,
including a separate 1 ms rectangular-window phase diagnostic. It is independent
of the control estimator, but still measures the same receiver signal path.
Add `--instruments /path/to/instruments.json` to a run to retain a JSON object
containing generator/scope models, reference settings, and bench observations.
Characterization performs a bounded ABBA DAC sweep (default ±8 codes, six cycles,
3-second dwells), excludes the first second of each dwell, fits frequency against
code and linear drift, and restores the starting code. It refuses to produce a
valid calibration when slope is unresolved. Calibration is bound to center/pilot
frequency, expected baseband offset, sample rate, and APR.

Control verifies the existing 28.8 MHz/PLLB/VCXO configuration. If it does not
match, investigate first. `--initialize-clock` explicitly programs the repository's
27 MHz crystal / 864 MHz PLL / divide-by-30 plan before acquisition. It disables
other clock outputs, just like the existing configuration utility. Changing APR
requires a new characterization. Evaluate 120, 60, then 30 ppm between runs and
retain at least 25% headroom for observed warm-up and drift.

The default expected pilot offset is 250 kHz at a 49.75 MHz center. Actual tuner
and digital tuning quantization can cause a constant bias. Independently calibrate
`expected_offset_hz` before treating a zero pilot residual as exact 28.8 MHz.
The Si5351 crystal reference frequency is an assumption until measured.

Default tuner gain is 0 dB. The initial live check clipped at the previously
proposed 9.9 dB setting; clipping is rejected rather than hidden by the estimator.
Set `telemetry_enabled` to false in the JSON configuration to benchmark without
detailed telemetry; a final run summary is still saved.

Controller `bandwidth_hz` means second-order natural frequency, not equivalent
noise bandwidth. Gains include measured Hz/code, and the PI/integrator model
checks at least 45 degrees phase margin including measurement age, frequency
averaging, and ZOH delay. Phase is unwrapped at the filtered 32 ksample/s cadence
across block boundaries; an 8 ms frequency prediction is not used to infer cycles.
The automatically generated calibration uses a **conservative buffering delay
budget**, not a scope-measured transfer function. Scope validation and a complete
actuator sweep remain necessary before claiming a characterized stability margin.
No dithering is enabled; quantized limit cycles must be measured, not hidden.

```bash
$PY -m beacon_clock replay ~/clock-runs/closed-smoke \
    --calibration ~/clock-runs/characterization/calibration.json \
    --output ~/clock-runs/replay.json
```

Replay produces deterministic command digests for 0.03, 0.1, and 0.3 Hz candidates.
It does not simulate the effect of those commands on recorded measurements. Use
the delayed, quantized simulated plant and hardware tests for that comparison.
Passing `--calibration` to `discipline --simulate` applies the measured gain and
delay to both the simulated plant and controller. Without it, the default
synthetic plant uses -0.08 Hz/code and 16 ms delay; that fine-resolution example
does not represent the measured bench DAC gain. Change `bandwidth_hz` in the JSON
configuration to compare candidates. Simulation time is sample time, not wall time.

## Storage and analysis

Each run contains `telemetry.bin` and a small `run.json`. Binary schema version 1
uses fixed little-endian records and integer state/flag codes, defined in
`beacon_clock/telemetry.py`. At 10 records/s, detailed telemetry uses about 2.7 MB/hour.
No raw IQ is saved. Raw-IQ capture is deliberately deferred; recorded-measurement
replay is supported, while deterministic IQ fixtures are generated by simulation.

The writer publishes approximately one-second batches through a bounded queue.
Storage stalls drop telemetry, with sequence numbers and summary counters showing
the loss, rather than blocking control. The default acquisition-output cap is
256 MiB including reserved metadata; reaching it stops detailed recording but
leaves control running. Existing runs are never automatically deleted. Analysis
outputs are separate from this acquisition cap. Abrupt power loss can lose buffered
records; initial metadata remains `final: false` until an orderly close.

Analysis reads binary directly, splits on gaps/invalid data/resets, and computes
overlapping Allan deviation from uniform strided subseries without interpolation
or drift subtraction. At 10 Hz scheduling on 8 ms blocks, every other measurement
normally gives a uniform 200 ms cadence. Reports include the cadence method and
number of terms. Coherence is measured at the pilot frequency and retains drift.
Plots show unwrapped phase deviation in radians as the primary error trace,
relative to the first valid record in each continuous segment. Gaps are not joined;
the arbitrary phase origin after reacquisition is not a measured phase jump.
These historical records do not contain the controller's captured phase setpoint,
so this trace is relative phase variation, not the exact servo setpoint error.
Phase includes reference, propagation, receiver, and clock contributions.
In-loop error is not an independent
measurement of clock accuracy.

## Validation and limits

Automated tests cover both control polarities, delayed quantized actuation, DSP
frequency/phase recovery, clipping/dropout, buffer ownership/overflow, writer
stalls/caps, deterministic replay, and calibration restoration. Run them before
deployment. Hardware acceptance additionally requires:

- One-hour matched open/closed runs and eight-hour endurance, without growing
  queues or normal-load software overflow.
- p99 DSP below 4 ms per 8 ms block, separately reported I2C and command age.
- Pilot interruption/attenuation, restart, DAC failure, and three cold starts.
- Scope comparison of generator and clock, including the 50/28.8 frequency ratio.
- Repeatable actuator gain, voltage limits, latency, warm-up drift, and noise floor.

Service installation is deferred until these interactive gates pass. The unmodified
second SDR is excluded. A future second controllable station requires additional
broadband/coded calibration for sample alignment. Wireless testing must characterize
path delays, multipath, pilot leakage, and dynamic range; one tone has a 20 ns
ambiguity and cannot distinguish clock drift from changing propagation.

See [docs/design.md](docs/design.md) for the research basis and legacy-code review.
See [docs/validation.md](docs/validation.md) for measured results and outstanding gates.
