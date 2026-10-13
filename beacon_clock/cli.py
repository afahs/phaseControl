import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from .acquisition import RtlSource
from .analysis import analyze
from .characterize import characterize
from .config import Config, Calibration
from .control import Controller, phase_margin
from .estimator import Measurement
from .hardware import Dac
from .runtime import run
from .simulation import SimulatedRig
from .telemetry import read_run, VALID, STALE
from .diagnostics import spectrum


def parser():
    p = argparse.ArgumentParser(description="Beacon clock measurement and discipline; binary telemetry only")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("monitor", "discipline", "characterize"):
        s = sub.add_parser(name)
        s.add_argument("--config")
        s.add_argument("--simulate", action="store_true")
        s.add_argument("--output", required=True, help="new run directory; existing runs are never overwritten")
        s.add_argument("--duration", type=float, default=60.)
        s.add_argument("--status", action="store_true")
        s.add_argument("--calibration")
        s.add_argument("--instruments", help="JSON object of instrument models, reference settings, and bench observations")
        s.add_argument("--initialize-clock", action="store_true",
                       help="configure 28.8 MHz once, before acquisition; default verifies existing settings")
        s.add_argument("--step", type=int, default=8)
        s.add_argument("--cycles", type=int, default=6)
        s.add_argument("--dwell", type=float, default=3.)
    s = sub.add_parser("analyze")
    s.add_argument("run")
    s.add_argument("--plots", action="store_true")
    s = sub.add_parser("replay", help="deterministically replay recorded measurements through the controller")
    s.add_argument("run")
    s.add_argument("--calibration", required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("inspect", help="read DAC and clock configuration without writing")
    s.add_argument("--config")
    s = sub.add_parser("spectrum", help="bounded receive-only high-resolution spectrum; no raw IQ saved")
    s.add_argument("--config")
    s.add_argument("--output", required=True)
    s.add_argument("--seconds", type=float, default=2.)
    s = sub.add_parser("defaults")
    s.add_argument("--output", required=True)
    return p


def replay(path, calibration_path, output):
    metadata, rows = read_run(path)
    c = Config(**metadata["config"]).validate()
    if c.telemetry_hz != c.control_hz:
        raise ValueError("measurement replay requires telemetry_hz == control_hz")
    cal = Calibration.load(calibration_path).validate(c)
    result = {}
    for bandwidth in (.03, .1, .3):
        cfg = replace(c, bandwidth_hz=bandwidth)
        margin = phase_margin(cfg, cal.delay_s + cfg.max_age_s)
        if margin < 45:
            result[str(bandwidth)] = dict(phase_margin_deg=margin, rejected=True)
            continue
        initial = metadata["initial_code"]
        controller = Controller(cfg, cal, cal.center_code if initial == 65535 else initial)
        commands = []
        previous = None
        for r in rows:
            if previous is not None and (r["segment"] != previous["segment"] or r["record"] != previous["record"]+1):
                controller.hold()
            m = Measurement(float(r["t"]), float(r["phase"]), float(r["frequency"]),
                            float(r["amplitude"]), float(r["snr_db"]), float(r["coherence"]),
                            float(r["clipping"]), bool(r["flags"] & VALID), int(r["segment"]))
            controller.update(m, float(r["age_s"]))
            commands.append((controller.code, int(controller.state)))
            previous = r
        import hashlib
        import numpy as np
        result[str(bandwidth)] = dict(phase_margin_deg=margin, updates=len(commands),
                                     command_sha256=hashlib.sha256(np.array(commands,dtype="<i4").tobytes()).hexdigest(),
                                     final_code=controller.code)
    payload = dict(caveat="Recorded measurements do not respond to replayed commands; use plant simulation for closed-loop comparisons", candidates=result)
    with Path(output).open("x") as f:
        json.dump(payload, f, indent=2)
    return payload


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "analyze":
            result = analyze(args.run, args.plots)
        elif args.command == "replay":
            result = replay(args.run, args.calibration, args.output)
        elif args.command == "defaults":
            with Path(args.output).open("x") as f:
                json.dump(asdict(Config()), f, indent=2)
            return 0
        else:
            c = Config.load(args.config)
            if args.command == "spectrum":
                result = spectrum(c, RtlSource(c), args.output, args.seconds)
            elif args.command == "inspect":
                dac = Dac(c)
                try:
                    result = dict(dac_code=dac.read(), clock=dac.inspect_clock())
                finally:
                    dac.close()
            else:
                if not 0 < args.duration <= 86400:
                    raise ValueError("duration must be in (0, 86400] seconds")
                source = SimulatedRig(c) if args.simulate else RtlSource(c)
                actuator = None
                extra = json.loads(Path(args.instruments).read_text()) if args.instruments else {}
                if not isinstance(extra, dict):
                    raise ValueError("--instruments must contain a JSON object")
                extra["simulation"] = args.simulate
                try:
                    if args.command != "monitor":
                        actuator = source if args.simulate else Dac(c)
                        if not args.simulate:
                            if args.initialize_clock:
                                actuator.initialize_clock()
                            extra["clock"] = actuator.inspect_clock()
                            if not extra["clock"]["compatible"]:
                                raise ValueError("clock configuration does not match 28.8 MHz/VCXO/APR; inspect before initialization")
                        initial = actuator.read()
                        if not c.dac_min <= initial <= c.dac_max:
                            raise ValueError("current DAC code is outside verified operating range")
                    elif args.initialize_clock:
                        raise ValueError("monitor is receive-only and cannot initialize the clock")
                    if args.command == "characterize":
                        result = characterize(c, source, actuator, args.output, args.cycles,
                                              args.step, args.dwell, args.status, extra)
                    else:
                        cal = Calibration.load(args.calibration) if args.calibration else None
                        if args.command == "discipline" and cal is None:
                            if not args.simulate:
                                raise ValueError("--calibration is required for hardware feedback")
                            cal = Calibration(source.gain, 2048, source.delay_s, c.center_hz,
                                              c.pilot_hz, c.expected_offset_hz, c.sample_rate, c.apr_ppm, True)
                        result = run(c, source, args.output, args.duration, actuator, cal,
                                     args.command, args.status, metadata_extra=extra)
                finally:
                    if actuator:
                        actuator.close()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result.get("failure") else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1
