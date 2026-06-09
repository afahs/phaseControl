# FLL/PLL Design Evaluation

Input CSV: `/home/rrrodriguez13/phaseControl/first_stage_demo/logs/pseudo_open_loop_after_30s_freq_centering_detected_tone_50mhz_5min_20260514.csv`

This replay treats the measured high-coherence phase trace as the input to a software NCO correction path.
It evaluates architecture behavior only; it does not claim the slow DAC/VCXO path can realize the PLL bandwidth.

## Controller Designs

- `fll_slow_frequency_centering`: first-order FLL from measured f_off_hz, bandwidth=0.05 Hz, damping=1, max_freq=250 Hz
- `pll_fast_phase_lock`: type-II software PLL/NCO, bandwidth=50 Hz, damping=0.707, max_freq=250 Hz

## Metrics

| case | phase slope (Hz) | timing RMS (ns) | p95 |phase| (rad) | p95 |freq| (Hz) | phase-band fraction | best dwell (s) |
|---|---:|---:|---:|---:|---:|---:|
| `open_loop` | 21.0692 | 4696.85 | 2.98511 | 87.8796 | 0.313393 | 0.12 |
| `fll_slow_frequency_centering` | -0.0297517 | 835.97 | 2.98334 | 72.0326 | 0.32146 | 0.12 |
| `pll_fast_phase_lock` | 7.28011e-06 | 0.68895 | 0.505953 | 20.0345 | 0.99648 | 55.376 |

## Interpretation

- The FLL design is a slow frequency-centering loop. It reduces long-term slope and frequency error but intentionally does not force the instantaneous phase to zero.
- The PLL design is a fast software phase loop. Its useful metric is wrapped phase/timing stability after correction, not just fitted frequency slope.
- Coherence and amplitude are used only as measurement gates; the pseudo-open-loop input already has high mean coherence in the handoff.

Generated plots:

- `fll_pll_replay_timeseries.png`
- `fll_pll_metric_comparison.png`
- `fll_pll_psd_peak_comparison.png`
- `fll_wrapped_phase_full.png`
- `fll_wrapped_phase_one_sixth_second.png`
- `pll_wrapped_phase_full.png`
- `pll_wrapped_phase_one_sixth_second.png`
