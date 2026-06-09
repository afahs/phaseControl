# Correctable Spectrum Comparison

Input trace: `/home/rrrodriguez13/phaseControl/first_stage_demo/logs/pseudo_open_loop_after_30s_freq_centering_detected_tone_50mhz_5min_20260514.csv`

The disturbance PSD is computed from the high-coherence corrected-tone pseudo-open-loop phase trace.
The actuator limits are represented as delay-limited feedback bandwidths; this does not replace a full sine-sweep transfer function, but it tests the structural claim directly against the measured disturbance spectrum.

## Candidate Bandwidth Fractions

| limit | bandwidth (Hz) | detrended unwrapped phase power below | wrapped phase power below |
|---|---:|---:|---:|
| measured median latency, 45deg phase lag hz | 3.911 | 100.00% | 5.38% |
| measured median latency, 90deg phase lag hz | 7.822 | 100.00% | 11.40% |
| measured median latency, 180deg phase lag hz | 15.64 | 100.00% | 23.24% |
| measured median latency, 360deg phase lag hz | 31.29 | 100.00% | 45.50% |
| measured slow latency, 45deg phase lag hz | 1.648 | 99.99% | 2.32% |
| measured slow latency, 90deg phase lag hz | 3.296 | 100.00% | 4.69% |
| measured slow latency, 180deg phase lag hz | 6.592 | 100.00% | 9.55% |
| measured slow latency, 360deg phase lag hz | 13.18 | 100.00% | 19.46% |
| assumed 150 ms effective latency, 45deg phase lag hz | 0.8333 | 99.99% | 1.14% |
| assumed 150 ms effective latency, 90deg phase lag hz | 1.667 | 99.99% | 2.34% |
| assumed 150 ms effective latency, 180deg phase lag hz | 3.333 | 100.00% | 4.73% |
| assumed 150 ms effective latency, 360deg phase lag hz | 6.667 | 100.00% | 9.68% |
| 50 ms update Nyquist | 10 | 100.00% | 14.79% |
| 2 ms software-NCO Nyquist | 250 | 100.00% | 100.00% |

## Generated Plots

- `disturbance_psd_vs_actuator_limits.png`
- `cumulative_phase_power_vs_bandwidth.png`
- `delay_phase_lag_limits.png`

Interpretation: if most disturbance power is above the usable delay-limited bandwidth, a DAC/VCXO feedback loop cannot remove it as a real-time phase actuator even when the phase detector can measure it. A software NCO can still correct higher-frequency phase after sampling because it has a much faster digital actuation path.
