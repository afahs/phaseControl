"""Quiet, fixed-code scope observation: no DAC or clock writes.

Run with no other DAC writer active. The MCP4725 retains its current code
without software refresh. Readback occurs only before and after the observation
window so periodic I2C traffic does not contaminate the ripple measurement.
This does not measure voltage or prevent another process from changing the DAC.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import time

from .config import Config
from .hardware import Dac


def observe(dac, config, duration, emit=print, wait=time.sleep):
    if not math.isfinite(duration) or not 1 <= duration <= 1800:
        raise ValueError("duration must be between 1 and 1800 seconds")
    initial = dac.read()
    if not config.dac_min <= initial <= config.dac_max:
        raise ValueError("current DAC code is outside configured operating limits")
    emit(json.dumps(dict(event="observation_started", code=initial,
                         nominal_voltage_v=config.dac_supply_v*initial/4096,
                         duration_s=duration, writes=0,
                         started_utc=datetime.now(timezone.utc).isoformat())))
    try:
        wait(duration)
    except KeyboardInterrupt:
        emit(json.dumps(dict(event="interrupted", writes=0,
                             note="DAC output left unchanged by this script")))
        return 130
    final = dac.read()
    emit(json.dumps(dict(event="observation_finished", initial_code=initial,
                         final_code=final, writes=0,
                         note="Endpoint readback only; scope measures analog stability")))
    if final != initial:
        raise RuntimeError("DAC code changed during observation; investigate another writer or reset")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--duration", type=float, default=600.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or not 1 <= args.duration <= 1800:
        parser.error("duration must be between 1 and 1800 seconds")
    config = Config.load(args.config)
    dac = Dac(config)
    try:
        return observe(dac, config, args.duration,
                       emit=lambda message: print(message, flush=True))
    finally:
        dac.close()


if __name__ == "__main__":
    raise SystemExit(main())
