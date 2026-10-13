# Design and evidence

## Research basis

- [AERA, Nanosecond-level time synchronization](https://arxiv.org/abs/1512.02216):
  multiple continuous tones support offline relative timing corrections, validated
  independently. This does not validate this project's live DAC controller.
- [Corstanje et al., LOFAR timing calibration](https://arxiv.org/abs/1603.08354):
  relative phases of terrestrial transmitters provide precise timing calibration.
- [RFClock, MobiCom 2021](https://genesys-lab.org/papers/RFCLOCK_MOBICOM2021.pdf):
  frequency recovery and time/phase synchronization are separate functions.
- [NIST SP 1065](https://www.nist.gov/publications/handbook-frequency-stability-analysis):
  Allan deviation and correct treatment of phase, frequency, drift, and gaps.
- [Si5351 datasheet](https://www.skyworksinc.com/-/media/Skyworks/SL/documents/public/data-sheets/Si5351-B.pdf)
  and [AN619](https://www.skyworksinc.com/-/media/Skyworks/SL/documents/public/application-notes/AN619.pdf):
  VCXO range, PLLB configuration, and output division.
- [MCP4725](https://www.microchip.com/en-us/product/mcp4725): volatile fast-mode DAC
  writes avoid EEPROM wear during feedback.

## Model

Assuming the tuner and sample clock share fractional error y, the measured
baseband residual relative to its nominal digital reference is approximately
`-pilot_hz*y`, not `-baseband_offset*y`. A measured code-to-residual gain includes
the sign of the complete receiver path. The phase estimator restores its coarse
mixer phase, so a software NCO cannot artificially flatten the reported residual.

The PI model is `dphase/dt = 2*pi*(disturbance + gain*code)`. With natural angular
frequency wn and damping z, frequency-domain proportional and integral gains are
`2*z*wn/(2*pi)` and `wn^2/(2*pi)`. Actuation divides by the measured gain with
negative feedback. Saturation and slew limits use conditional integration.

Residual pilot phase is local to a continuous segment; it is not an absolute
timestamp. Restart/invalid measurements break that segment and require reacquisition.

The coarse mixer is fixed within a segment. Contiguous filtered phases are
unwrapped at 32 ksample/s, including the boundary between blocks; the coarse
mixer phase is then restored at each fitted block midpoint. A regression test
alternates +60/-60 Hz every block while keeping physical phase continuous. The
earlier midpoint-frequency extrapolation incorrectly reported slips in this case.
The current slip flag means a filtered adjacent-sample phase increment exceeded
pi/2; it is an ambiguity warning, not proof or a count of physical cycle slips.
Full-cycle jumps or device-side losses can be invisible to a single tone.

The frequency used for acquisition is a phase slope over approximately 100 ms.
The phase-margin calculation budgets half a control period for that window and
half for the actuator zero-order hold, in addition to age and transport budgets.

## Present reference source

The owner identifies an HP8656A, expected to be unmodulated and using an external
rubidium reference that feeds no other part of the setup, and a Tektronix TDS2012
scope. The rubidium model/frequency are unspecified. The front-panel AM/FM
settings and calibration state have not been independently established.
HP's [8656A manual, Table 1-1 and Table 4-2](https://manuals.chudov.com/HP/HP-8656A-Signal-Generator-1981-WW.pdf)
specifies residual FM even in CW mode: below 15 Hz RMS in a 0.3-3 kHz
post-detection bandwidth, and below 30 Hz RMS in 0.05-15 kHz, for the band
containing 50 MHz. Those are specification limits in different bandwidths from
this loop, not measurements of this instrument and not a diagnosis of observed
phase variation. Establish the generator's reference and observe it independently
before assigning a noise floor to the Si5351B or recommending replacement hardware.

## Legacy review, October 13, 2026

The review was read-only. Existing scripts were not executed or replaced.

The Pi contains experiments under `/home/rrrodriguez13/phaseControl`, including
asynchronous controllers, `first_stage_demo`, `NotebookBased/SameTimeSampling`,
capture/replay experiments, and GPIO/DAC latency tests. Their numerical results,
control laws, same-time claims, gains, and wiring constants remain unverified.

Findings directly inspected in source:

- The first-stage acquisition adapter keeps a persistent stream on a background
  thread and uses a bounded application queue, but waits when that queue is full.
- The active installed pyrtlsdr stream adapter adds a 20-block queue. Its overflow
  path logs lost callback data without structured gap metadata. A separate thread
  schedules asynchronous queue insertions, another potential backlog. Direct
  callbacks avoid this adapter.
- The raw callback buffer can be overwritten after callback return. The new
  acquisition pool therefore owns a copy before returning.
- SameTimeSampling's CSV logger flushes each row. The first-stage binary writer
  still builds/serializes rows in the processing loop. Neither pattern is retained.
- Existing tests bracket I2C commands with GPIO pulses. This is useful for scope
  timing, but GPIO pin wiring must be verified before reuse.
- Legacy helpers identify MCP4725 address 0x62. New code checks targeted register
  reads and does not perform a blind I2C bus scan.

Observed runtime: Pi 4 Model B Rev 1.5, Debian 12, NumPy 2.0.0, SciPy 1.13.1.
The existing SDR environment is `/home/rrrodriguez13/sdr/.env`; the new core needs
only NumPy. The local pyrtlsdr repository HEAD was
`697de751246de021c7c0acaf1c18aaf6baabad16`; librtlsdr repository HEAD was
`cd36c28815592af4080d71181964909125a74f44`. Repository HEAD is not proof of the
installed binary's provenance; record active import paths during validation.

## Acceptance evidence policy

Keep synthetic, replay, live in-loop, and independent scope evidence distinct.
Do not substitute simulated hours for actual endurance, or in-loop phase for
out-of-loop accuracy. Calibration's initial delay budget is conservative and must
be replaced/validated by actual response measurements. Electrical amplitude,
coupling, generator stability, and scope resolution require bench observations.
