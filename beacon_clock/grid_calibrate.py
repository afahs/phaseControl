"""Measure a 100-code gain grid, validate it, then run five minutes of feedback."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from .acquisition import RtlSource
from .characterize import characterize
from .config import Calibration, Config
from .hardware import Dac
from .runtime import run, version_info
from .telemetry import read_run, VALID, STALE


def cycle_gains(path, center, cycles=3, dwell=2.):
    meta, rows = read_run(path)
    if not meta['telemetry']['complete'] or meta['summary']['failure']:
        raise ValueError('incomplete characterization')
    gains = []
    for cycle in range(cycles):
        begin = dwell+4*dwell*cycle
        mask = (rows['t']>=begin) & (rows['t']<begin+4*dwell) & (rows['t']%dwell>=1.)
        mask &= ((rows['flags']&VALID)!=0) & ((rows['flags']&STALE)==0)
        s = rows[mask]
        expected = np.where(((s['t']-begin)//dwell).astype(int)%3==0,center+8,center-8)
        s = s[s['code']==expected]
        x = np.column_stack((np.ones(len(s)), s['code'].astype(float)-center, s['t']-begin))
        if len(s)<.9*4*(dwell-1)*meta['config']['telemetry_hz'] or np.linalg.matrix_rank(x)!=3:
            raise ValueError('insufficient settled data in an ABBA cycle')
        gains.append(float(np.linalg.lstsq(x,s['frequency'],rcond=None)[0][1]))
    mean, std = float(np.mean(gains)), float(np.std(gains,ddof=1))
    if not all(g*mean>0 for g in gains) or abs(mean)<1e-4 or std>abs(mean)*.10:
        raise ValueError('local gain is unresolved or cycle repeatability exceeds 10%')
    return dict(cycle_gains=gains, mean_hz_per_code=mean, cycle_std_hz_per_code=std)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true', help='reuse accepted points; preserve rejected recordings')
    args = parser.parse_args()
    root = Path(args.output)
    if args.resume:
        if not (root/'grid-progress.json').is_file() or (root/'feedback').exists():
            raise ValueError('resume requires accepted points and no feedback recording')
    else:
        root.mkdir(parents=True, exist_ok=False)
    config = Config(apr_ppm=30, telemetry_hz=125).validate()
    def save(name, data):
        (root/name).write_text(json.dumps(data,indent=2,allow_nan=False))
    def report(event, **data):
        print(json.dumps(dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **data)),flush=True)
    dac = Dac(config)
    def move(code):
        dac.read()
        while dac.code != code:
            dac.write(int(max(dac.code-config.max_step_codes,min(dac.code+config.max_step_codes,code))))
            time.sleep(1/config.control_hz)
    completed = False
    try:
        initial = dac.read()
        clock = dac.inspect_clock()
        if not clock['compatible']:
            raise ValueError('clock is not compatible with 30 ppm')
        extra = dict(clock=clock, software=version_info(),
                     generator='Last user report: HP8656A 50 MHz -30 dBm unmodulated; current panel and external reference lock not independently verified')
        if not args.resume:
            save('config.json',asdict(config))
            save('before.json',dict(dac_code=initial,**extra))
        elif json.loads((root/'config.json').read_text()) != asdict(config):
            raise ValueError('resume configuration differs from saved sweep')
        centers = sorted(set(range(500,3601,100)) | {3137})
        results = json.loads((root/'grid-progress.json').read_text()) if args.resume else []
        if [r['center'] for r in results] != centers[:len(results)]:
            raise ValueError('accepted points are not a prefix of the requested grid')
        for result in results:
            Calibration(**result['calibration']).validate(config)
        report('sweep_started',resumed_points=len(results),software=version_info())
        for index,center in enumerate(centers):
            if index < len(results):
                continue
            move(center)
            time.sleep(2.)
            if not dac.inspect_clock()['compatible']:
                raise ValueError('clock changed during sweep')
            report('point_start',index=index+1,total=len(centers),center=center)
            accepted = False
            for attempt,(cycles,dwell) in enumerate(((3,2.),(6,3.),(10,3.))):
                path = root/(f'code-{center}' if attempt==0 else f'code-{center}-retry-{attempt}')
                if path.exists():
                    continue
                result = characterize(config,RtlSource(config),dac,path,cycles=cycles,step=8,
                                      dwell=dwell,metadata_extra=extra)
                try:
                    Calibration.load(path/'calibration.json').validate(config)
                    repeatability = cycle_gains(path,center,cycles,dwell)
                except ValueError as exc:
                    report('point_rejected',center=center,attempt=attempt,reason=str(exc))
                    continue
                result.update(center=center,cycles=cycles,dwell_s=dwell,recording=path.name,**repeatability)
                accepted = True
                break
            if not accepted:
                raise ValueError(f'code {center} exhausted repeatability attempts')
            results.append(result)
            save('grid-progress.json',results)
            report('point_complete',center=center,**repeatability)
        feedback_config = replace(config,dac_min=500,dac_max=3600).validate()
        local = next(r for r in results if r['center']==3137)
        table = tuple((r['center'],r['calibration']['hz_per_code']) for r in results)
        mapped = replace(Calibration(**local['calibration']),gain_table=table,measured_range=(500,3600))
        mapped.validate(feedback_config)
        save('calibration-map.json',asdict(mapped))
        save('feedback-config.json',asdict(feedback_config))
        move(3137)
        time.sleep(5.)
        before = dict(dac_code=dac.read(),clock=dac.inspect_clock())
        save('before-feedback.json',before)
        if not before['clock']['compatible'] or before['dac_code']!=3137:
            raise ValueError('feedback preflight failed')
        report('feedback_start',duration_s=300.,center=3137)
        summary = run(feedback_config,RtlSource(feedback_config),root/'feedback',300.,
                      actuator=dac,calibration=mapped,mode='discipline',metadata_extra=extra)
        after = dict(dac_code=dac.read(),clock=dac.inspect_clock())
        save('after-feedback.json',after)
        if summary['failure']:
            raise RuntimeError(summary['failure'])
        completed = True
        report('all_complete',summary=summary,after=after)
    finally:
        try:
            if not completed:
                move(3137)
                report('stopped_without_completed_feedback',restored_code=dac.read())
        finally:
            dac.close()


if __name__ == '__main__':
    main()
