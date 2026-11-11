"""Explicit bench sequence: local gain measurement followed by receive-only baseline."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .acquisition import RtlSource
from .characterize import characterize
from .config import Config
from .hardware import Dac
from .runtime import run, version_info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--center', type=int, required=True)
    args = p.parse_args()
    config = Config.load(args.config)
    if not config.dac_min + 8 <= args.center <= config.dac_max - 8:
        raise ValueError('center needs eight codes of headroom')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (root / name).write_text(json.dumps(value, indent=2, allow_nan=False))

    def inspect():
        dac = Dac(config)
        try:
            result = dict(dac_code=dac.read(), clock=dac.inspect_clock())
            if not result['clock']['compatible']:
                raise RuntimeError('clock configuration is incompatible')
            return result
        finally:
            dac.close()

    save('config.json', asdict(config))
    save('before.json', inspect())
    extra = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                 software=version_info(), requested_fixed_code=args.center,
                 generator='HP8656A; last user report: 50 MHz, -30 dBm, no modulation; current panel unverified',
                 reference='User reports external rubidium feeding generator only; lock not independently verified')
    save('instruments.json', extra)
    dac = Dac(config)
    try:
        dac.read()
        while dac.code != args.center:
            dac.write(int(max(dac.code-config.max_step_codes,
                              min(dac.code+config.max_step_codes, args.center))))
            time.sleep(1/config.control_hz)
        time.sleep(5)
        print('CALIBRATION START', flush=True)
        result = characterize(config, RtlSource(config), dac, root/'calibration',
                              cycles=10, step=8, dwell=3., metadata_extra=extra)
        print(json.dumps(result), flush=True)
    finally:
        dac.close()
    before = inspect()
    save('before-baseline.json', before)
    if before['dac_code'] != args.center:
        raise RuntimeError('calibration did not restore requested center')
    time.sleep(5)
    print('BASELINE START: no actuator passed to monitor', flush=True)
    summary = run(config, RtlSource(config), root/'baseline', 300., mode='monitor',
                  metadata_extra=dict(extra, fixed_dac_before=before,
                                      dac_observation='Endpoint readback only; no I2C during monitor'))
    save('after-baseline.json', inspect())
    print(json.dumps(summary), flush=True)
    if summary['failure']:
        raise RuntimeError(summary['failure'])
    after = json.loads((root/'after-baseline.json').read_text())
    if after['dac_code'] != args.center:
        raise RuntimeError('baseline endpoint code differs from requested center')
    print('ALL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
