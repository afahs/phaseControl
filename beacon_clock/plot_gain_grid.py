"""Plot completed local-gain measurements and their cycle repeatability."""
import argparse
import json
from pathlib import Path
import numpy as np


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
           title='30 ppm DAC calibration map: three ABBA cycles per point')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.savefig(root/'gain-map.png',dpi=150)
    plt.close(fig)
    print(json.dumps(dict(points=len(points),gain_min=float(gain.min()),
                          gain_max=float(gain.max()),
                          max_relative_cycle_std=float(np.max(spread/abs(gain))))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    plot(parser.parse_args().root)
