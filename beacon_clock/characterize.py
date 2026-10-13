from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .config import Calibration
from .runtime import run
from .telemetry import read_run, VALID, STALE


def characterize(config, source, actuator, path, cycles=6, step=8, dwell=3., status=False,
                 metadata_extra=None):
    center = actuator.read()
    if not config.dac_min + step <= center <= config.dac_max - step:
        raise ValueError("current DAC code lacks safe characterization headroom")
    if not 1 <= step <= config.max_step_codes or dwell < 2 or cycles < 3:
        raise ValueError("characterization requires >=3 cycles, >=2 s dwells, and a bounded step")
    # ABBA suppresses slow drift; center points bracket the sequence.
    codes = [center] + [center + sign*step for _ in range(cycles) for sign in (1,-1,-1,1)] + [center]
    def schedule(t):
        return codes[min(len(codes)-1, int(t/dwell))]
    try:
        summary = run(config, source, path, len(codes)*dwell, actuator=actuator,
                      mode="characterize", schedule=schedule, status=status,
                      metadata_extra=metadata_extra)
    finally:
        # Restore the initial code through bounded writes, even on interruption.
        import time
        while actuator.code != center:
            actuator.write(int(max(actuator.code-config.max_step_codes,
                                   min(actuator.code+config.max_step_codes, center))))
            time.sleep(1/config.control_hz)
    metadata, rows = read_run(path)
    elapsed_in_step = rows["t"] % dwell
    expected = np.array([schedule(t) for t in rows["t"]])
    mask = ((rows["flags"] & VALID) != 0) & ((rows["flags"] & STALE) == 0)
    mask &= (elapsed_in_step >= 1.) & (rows["code"] == expected)
    r = rows[mask]
    if len(r) < 30 or summary["failure"]:
        raise RuntimeError("insufficient valid settled characterization; no calibration produced")
    x = np.column_stack((np.ones(len(r)), r["code"].astype(float)-center, r["t"]))
    beta, _, rank, _ = np.linalg.lstsq(x, r["frequency"], rcond=None)
    residual = r["frequency"]-x@beta
    covariance = np.linalg.pinv(x.T@x)*np.sum(residual**2)/max(1,len(r)-3)
    slope_std = float(np.sqrt(covariance[1,1]))
    # Conservative latency budget pending independent scope step-response work:
    # configured USB transfer buffering + application queue + one DSP block.
    delay = (config.usb_buffers + config.queue_blocks + 1)*config.block_s
    valid = bool(rank == 3 and abs(beta[1]) > max(5*slope_std, 1e-4)
                 and not metadata["telemetry"]["capped"] and not metadata["telemetry"]["error"])
    cal = Calibration(float(beta[1]), center, delay, config.center_hz, config.pilot_hz,
                      config.expected_offset_hz, config.sample_rate, config.apr_ppm,
                      valid=valid, slope_std=slope_std, measured_range=(center-step,center+step))
    Path(path, "calibration.json").write_text(json.dumps(asdict(cal), indent=2))
    result = dict(calibration=asdict(cal), drift_hz_per_s=float(beta[2]),
                  residual_rms_hz=float(np.sqrt(np.mean(residual**2))), samples_used=len(r),
                  delay_basis="conservative configured buffering budget, not a measured physical delay",
                  caveat="Regression standard error assumes independent residuals; verify repeatability")
    Path(path, "characterization.json").write_text(json.dumps(result, indent=2))
    return result
