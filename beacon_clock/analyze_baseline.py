"""Offline fixed-DAC timing diagnostics; never bridge invalid measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
from .analysis import clean_segments
from .control import State
from .telemetry import read_run, VALID, STALE


def analyze(root, closed_runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    out = root/'diagnostics'
    out.mkdir(exist_ok=True)
    meta, rows = read_run(root/'baseline')
    if not meta['final'] or not meta['telemetry']['complete']:
        raise ValueError('baseline must be complete')
    scale = 1e9/(2*np.pi*meta['config']['pilot_hz'])
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 11), layout='constrained')
    segment_reports = []
    for ix in clean_segments(rows):
        s = rows[ix]
        t = s['t']
        timing = (s['phase']-s['phase'][0])*scale
        fit = np.polyfit(t-t[0], timing, 1)
        residual = timing-np.polyval(fit, t-t[0])
        axes[0].plot(t, timing, linewidth=.7)
        axes[1].plot(t, residual, linewidth=.7)
        # Non-overlapping 1-second means, confined to valid segments.
        n = len(s)//125*125
        if n:
            axes[2].plot(t[:n].reshape(-1,125).mean(1),
                         s['frequency'][:n].reshape(-1,125).mean(1), linewidth=.8)
        segment_reports.append(dict(start_s=float(t[0]), end_s=float(t[-1]),
                                    slope_ns_per_s=float(fit[0]),
                                    residual_peak_ns=float(abs(residual).max())))
    axes[3].plot(rows['t'], rows['age_s']*1e3, linewidth=.5, label='Callback age')
    axes[3].plot(rows['t'], rows['dsp_s']*1e3, linewidth=.5, label='DSP time')
    axes[3].legend()
    for ax, label in zip(axes, ['Relative timing (ns)', 'Linear-fit residual (ns)',
                               'Frequency error (Hz)', 'Processing time (ms)']):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Delivered-sample time (s)')
    fig.suptitle('30 ppm, fixed DAC 3137, feedback off\nTiming at 50 MHz; independent origin and linear fit per valid segment')
    fig.savefig(out/'baseline-overview.png', dpi=150)
    plt.close(fig)

    paths = [root/'baseline'] + [Path(p) for p in closed_runs]
    fig, axes = plt.subplots(len(paths), 1, sharex=True, sharey=True,
                             figsize=(12, 2.7*len(paths)), layout='constrained', squeeze=False)
    comparison = []
    for i, path in enumerate(paths):
        m, r = read_run(path)
        if m['config']['telemetry_hz'] != 125 or m['config']['pilot_hz'] != meta['config']['pilot_hz']:
            raise ValueError('comparison needs matching pilot and 125 Hz telemetry')
        if i:
            r = r[r['state'] == State.TRACK]
        ax = axes[i,0]
        residuals = []
        for ix in clean_segments(r):
            # The initial FIR midpoint after a reset differs from steady cadence.
            s = r[ix[2:]]
            if len(s) < 1251:
                continue
            if not np.allclose(np.diff(s['t']), .008, rtol=0, atol=1e-7):
                raise ValueError('nonuniform records')
            x = (s['phase']-s['phase'][0])*scale
            # Centered 10.008-second moving mean removes linear frequency drift
            # within each window. Drop the five-second edges; never fill gaps.
            residual = x[625:-625] - np.convolve(x, np.ones(1251)/1251, mode='valid')
            ax.plot(s['t'][625:-625], residual, linewidth=.6, color='tab:blue' if i==0 else 'tab:purple')
            residuals.extend(residual.tolist())
        label = 'Fixed DAC 3137; feedback off' if i==0 else f'Previous closed-loop trial {i}'
        z = np.asarray(residuals)
        metrics = dict(label=label, rms_ns=float(np.sqrt(np.mean(z*z))) if len(z) else None,
                       abs_peak_ns=float(abs(z).max()) if len(z) else None,
                       abs_p95_ns=float(np.percentile(abs(z),95)) if len(z) else None)
        comparison.append(metrics)
        ax.set_title(label)
        ax.set_ylabel('Local residual (ns)')
        ax.grid(alpha=.2)
    axes[-1,0].set_xlabel('Delivered-sample time (s)')
    fig.suptitle('Timing excursions after subtracting a centered 10.008 s moving mean\nSame filter for all runs; gaps and edges excluded; not absolute timing or lock error')
    fig.savefig(out/'local-excursions.png', dpi=150)
    plt.close(fig)

    cm, cr = read_run(root/'calibration')
    slopes = []
    center = 3137
    for cycle in range(10):
        begin = 3+12*cycle
        mask = (cr['t']>=begin) & (cr['t']<begin+12) & (cr['t']%3>=1)
        mask &= ((cr['flags'] & VALID)!=0) & ((cr['flags'] & STALE)==0)
        s = cr[mask]
        expected = np.where(((s['t']-begin)//3).astype(int) % 3 == 0, center+8, center-8)
        s = s[s['code']==expected]
        x = np.column_stack((np.ones(len(s)), s['code'].astype(float)-center, s['t']-begin))
        if len(s)>=30 and np.linalg.matrix_rank(x)==3:
            slopes.append(float(np.linalg.lstsq(x,s['frequency'],rcond=None)[0][1]))
    report = dict(summary=meta['summary'], telemetry=meta['telemetry'], segments=segment_reports,
                  local_residual_method='phase at 50 MHz minus centered 10.008 s moving mean, valid contiguous data only',
                  comparison=comparison, calibration_cycle_slopes_hz_per_code=slopes,
                  calibration_cycle_mean=float(np.mean(slopes)),
                  calibration_cycle_std=float(np.std(slopes,ddof=1)))
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('closed_runs', nargs='*')
    args = parser.parse_args()
    analyze(args.root,args.closed_runs)
