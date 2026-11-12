"""Plot completed local-gain measurements and their cycle repeatability."""
import argparse
import json
from pathlib import Path
import numpy as np
from .telemetry import read_run, VALID


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    points = json.loads((root/'grid-progress.json').read_text())
    cal = json.loads((root/'calibration-map.json').read_text())
    code = np.array([r['center'] for r in points])
    gain = np.array([r['calibration']['hz_per_code'] for r in points])
    spread = np.array([r['cycle_std_hz_per_code'] for r in points])
    if not np.allclose(np.array(cal['gain_table']),np.column_stack((code,gain))):
        raise ValueError('map differs from recorded measurements')
    fig, ax = plt.subplots(figsize=(11,5),layout='constrained')
    ax.errorbar(code,gain,yerr=spread,fmt='o-',markersize=4,capsize=3,
                label='Measured local gain; bars = cycle standard deviation')
    ax.axhline(-.8152865820263218,color='gray',linestyle='--',
               label='Previous scalar calibration near code 2048')
    ax.axvline(3137,color='tab:orange',alpha=.5,label='Starting code for new feedback run')
    ax.set(xlabel='DAC code',ylabel='Local gain (Hz/code)',
           title='30 ppm DAC calibration map: three ABBA cycles, longer retries where needed')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.savefig(root/'gain-map.png',dpi=150)
    plt.close(fig)
    meta, rows = read_run(root/'feedback')
    fig, axes = plt.subplots(3,2,figsize=(13,9),layout='constrained',sharex='col')
    for column,(start,end) in enumerate(((0,300),(60,70))):
        s = rows[(rows['t']>=start)&(rows['t']<=end)]
        axes[0,column].plot(s['t'],s['frequency'],linewidth=.5)
        axes[0,column].axhline(0,color='gray',linewidth=.6)
        axes[1,column].step(s['t'],s['code'],where='post',linewidth=.7)
        axes[2,column].plot(s['t'],s['snr_db'],linewidth=.5)
        bad = (s['flags']&VALID)==0
        axes[2,column].scatter(s['t'][bad],s['snr_db'][bad],s=8,color='tab:red',label='Invalid measurement')
        axes[2,column].axhline(meta['config']['min_snr_db'],color='tab:red',linestyle='--',linewidth=.7)
        axes[0,column].set_title('Full five-minute run' if column==0 else '10-second detail')
        axes[2,column].set_xlabel('Delivered-sample time (s)')
        for row,label in enumerate(('Frequency error (Hz)','DAC code','Tone-fit SNR (dB)')):
            axes[row,column].set_ylabel(label)
            axes[row,column].grid(alpha=.2)
    axes[2,1].legend(fontsize=8)
    captured = np.isfinite(rows['phase_target']).any()
    fig.suptitle(f"Mapped-calibration feedback: {meta['summary']['locked_fraction']:.1%} locked\n"
                 + ('Phase target captured' if captured else 'No phase target captured; acquisition and holdover only'))
    fig.savefig(root/'feedback-status.png',dpi=150)
    plt.close(fig)
    print(json.dumps(dict(points=len(points),gain_min=float(gain.min()),
                          gain_max=float(gain.max()),
                          max_relative_cycle_std=float(np.max(spread/abs(gain))))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    plot(parser.parse_args().root)
