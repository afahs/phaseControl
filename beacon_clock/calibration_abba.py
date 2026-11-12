"""Matched old/new calibration comparison; only calibration changes between runs."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from .acquisition import RtlSource
from .config import Config, Calibration
from .hardware import Dac
from .runtime import run, version_info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',required=True)
    p.add_argument('--new',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--initial-code',type=int,default=2048)
    p.add_argument('--duration',type=float,default=300.)
    args=p.parse_args()
    c=Config(apr_ppm=30,telemetry_hz=125,dac_min=500,dac_max=3600).validate()
    if not c.dac_min<=args.initial_code<=c.dac_max or not 0<args.duration<=3600:
        raise ValueError('invalid initial code or duration')
    calibrations={label:Calibration.load(path).validate(c)
                  for label,path in (('old',args.old),('new',args.new))}
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (root/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    def report(event,**values):
        print(json.dumps(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**values)),flush=True)
    save('config.json',asdict(c))
    for label,cal in calibrations.items():
        save(f'{label}-calibration.json',asdict(cal))
    protocol=dict(order=['old','new','new','old'],duration_s=args.duration,
                  initial_code=args.initial_code,settle_s=5.,software=version_info(),
                  generator='HP8656A; last reported 50 MHz, -30 dBm, no modulation; current panel/reference lock unverified',
                  comparison='Same software, config, initial code, settling and SDR settings; sequential ABBA reduces but does not eliminate bench drift')
    save('protocol.json',protocol)
    summaries=[]
    dac=Dac(c)
    try:
        for index,label in enumerate(protocol['order'],1):
            dac.read()
            if not dac.inspect_clock()['compatible']:
                raise ValueError('clock does not match 30 ppm configuration')
            while dac.code!=args.initial_code:
                dac.write(int(max(dac.code-c.max_step_codes,min(dac.code+c.max_step_codes,args.initial_code))))
                time.sleep(1/c.control_hz)
            time.sleep(protocol['settle_s'])
            before=dict(dac_code=dac.read(),clock=dac.inspect_clock())
            if before['dac_code']!=args.initial_code or not before['clock']['compatible']:
                raise ValueError('pre-run verification failed')
            name=f'{index}-{label}'
            save(f'before-{name}.json',before)
            report('run_start',index=index,label=label,initial_code=args.initial_code)
            summary=run(c,RtlSource(c),root/name,args.duration,actuator=dac,
                        calibration=calibrations[label],mode='discipline',
                        metadata_extra=dict(protocol,comparison_label=label,before=before))
            after=dict(dac_code=dac.read(),clock=dac.inspect_clock())
            save(f'after-{name}.json',after)
            summaries.append(dict(name=name,summary=summary,after=after))
            save('results.json',summaries)
            report('run_complete',name=name,summary=summary,after=after)
            if summary['failure'] or not after['clock']['compatible']:
                raise RuntimeError('run failed or clock changed; comparison stopped')
        report('all_complete')
    finally:
        dac.close()


if __name__=='__main__':
    main()
