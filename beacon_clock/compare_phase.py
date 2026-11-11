"""Compare completed version-2 phase-loop recordings."""
import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import clean_segments
from .control import State
from .telemetry import read_run, LOCKED


def compare(paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(len(paths), 1, sharex=True, sharey=True,
                             figsize=(12, 3 * len(paths)), layout="constrained", squeeze=False)
    spectrum_fig, spectrum_ax = plt.subplots(figsize=(10, 6), layout="constrained")
    reports = []
    for index, path in enumerate(paths):
        meta, rows = read_run(path)
        if meta['version'] != 2 or not meta['final']:
            raise ValueError("comparison requires completed version-2 recordings")
        tracking = rows['state'] == State.TRACK
        segments = clean_segments(rows[tracking])
        tracked = rows[tracking]
        phase = tracked['phase_error']
        phase = phase[np.isfinite(phase)]
        locked = (rows['flags'] & LOCKED) != 0
        breaks = np.r_[True, np.diff(rows['record']) != 1]
        longest = current = 0
        for good, gap in zip(locked, breaks):
            current = (0 if gap else current) + 1 if good else 0
            longest = max(longest, current)
        report = dict(run=str(path), summary=meta['summary'], telemetry=meta['telemetry'],
                      phase_rms_rad=float(np.sqrt(np.mean(phase**2))) if len(phase) else None,
                      phase_abs_p95_rad=float(np.percentile(abs(phase), 95)) if len(phase) else None,
                      phase_abs_max_rad=float(np.max(abs(phase))) if len(phase) else None,
                      longest_recorded_lock_s=longest/meta['config']['telemetry_hz'])
        reports.append(report)
        ax = axes[index, 0]
        for segment in segments:
            s = tracked[segment]
            ax.plot(s['t'], s['phase_error'], color='tab:purple', linewidth=.6)
        ax.fill_between(rows['t'], 0, 1, where=locked, transform=ax.get_xaxis_transform(),
                        color='tab:green', alpha=.10)
        losses = rows['lock_loss'] != 0
        ax.scatter(rows['t'][losses], np.full(np.sum(losses), .97),
                   transform=ax.get_xaxis_transform(), color='tab:red', marker='v', s=18)
        for limit in (-1, -.5, .5, 1):
            ax.axhline(limit, color='gray', linestyle='--', linewidth=.6)
        ax.set_title(f"Trial {index+1} — locked {meta['summary']['locked_fraction']:.1%}; "
                     f"{meta['summary']['lock_loss_events']} lock losses")
        ax.set_ylabel('Phase error (rad)')
        ax.grid(alpha=.2)
        window = np.hanning(1025)[:-1]
        spectra = []
        for segment in segments:
            s = tracked[segment]
            if len(s) < 1024:
                continue
            if not np.allclose(np.diff(s['t']), .008, rtol=0, atol=1e-7):
                raise ValueError("spectral comparison requires uniform 125 Hz records")
            for start in range(0, len(s)-1024+1, 512):
                e = s['phase_error'][start:start+1024]
                power = abs(np.fft.rfft((e-e.mean())*window))**2 / (125*np.sum(window**2))
                power[1:-1] *= 2
                spectra.append(power)
        report['spectral_windows'] = len(spectra)
        if spectra:
            freq = np.fft.rfftfreq(1024, .008)
            spectrum_ax.loglog(freq[1:], np.mean(spectra, axis=0)[1:], label=f'Trial {index+1}')
    axes[-1, 0].set_xlabel('Delivered-sample time (s)')
    fig.suptitle('30 ppm phase-loop comparison — directly recorded phase error\n'
                 'Green shading: controller locked; red markers: lock-loss events; gaps: unavailable phase', fontsize=12)
    fig.savefig(output/'phase-comparison.png', dpi=150)
    spectrum_ax.set(xlabel='Offset frequency (Hz)', ylabel='Phase-error PSD (rad²/Hz)',
                    title='Tracking intervals, including phase excursions\n8.192 s Hann windows, 50% overlap, window means removed')
    spectrum_ax.legend()
    spectrum_ax.grid(which='both', alpha=.2)
    spectrum_fig.savefig(output/'phase-spectra.png', dpi=150)
    plt.close(fig)
    plt.close(spectrum_fig)
    (output/'comparison.json').write_text(json.dumps(reports, indent=2, allow_nan=False))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compare(args.runs, args.output)
