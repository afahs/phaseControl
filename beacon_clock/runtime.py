from dataclasses import asdict
import math
import platform
import queue
import subprocess
import time
from . import __version__
from .control import Controller, State
from .estimator import PilotEstimator
from .telemetry import Writer, VALID, LOCKED, GAP, SLIP, STALE
import numpy as np


class Distribution:
    """Fixed-size latency histogram, 50 us resolution below 100 ms."""
    def __init__(self):
        self.bins = np.zeros(2001, dtype=np.int64)
        self.count = 0
        self.total = self.maximum = 0.

    def add(self, value):
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        self.bins[min(2000, max(0, int(value / .00005)))] += 1

    def report(self):
        return dict(count=self.count, mean_s=self.total/max(1, self.count), max_s=self.maximum,
                    p99_upper_s=float((np.searchsorted(np.cumsum(self.bins), .99*self.count)+1)*.00005) if self.count else 0.,
                    histogram_overflow=int(self.bins[-1]))


def version_info():
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    return dict(package=__version__, git=revision, python=platform.python_version(),
                numpy=np.__version__)


def run(config, source, path, duration, actuator=None, calibration=None,
        mode="monitor", status=False, schedule=None, metadata_extra=None):
    estimator = PilotEstimator(config)
    controller = None
    code = actuator.read() if actuator else 65535  # unknown/unobserved, not a DAC command
    if mode == "discipline":
        controller = Controller(config, calibration, code)
    writer = Writer(path, config,
                    dict(software=version_info(), mode=mode,
                         calibration=asdict(calibration) if calibration else None,
                         initial_code=code, instruments=metadata_extra or {},
                         time_basis="delivered sample index; no hardware timestamps"), status=status)
    dsp, age_stats, i2c_stats, command_age_stats = Distribution(), Distribution(), Distribution(), Distribution()
    start = time.monotonic()
    cpu_start = time.process_time()
    blocks = invalid = stale_count = slips = missed = updates = writes = 0
    locked_blocks = 0
    next_control = next_log = 0.
    segment = 0
    previous_valid = False
    previous_index = None
    previous_source_segment = None
    pending_flags = 0
    last_state = State.MONITOR if not controller else controller.state
    failure = None
    source_started = False
    try:
        source.start()
        source_started = True
        while True:
            try:
                block = source.get(timeout=1.)
            except queue.Empty:
                if controller:
                    controller.hold()
                estimator.reset()
                raise RuntimeError("no SDR data for one second; stream must be restarted")
            try:
                if block.sample_index/config.sample_rate >= duration:
                    break
                block_start = time.perf_counter()
                gap = (block.lost or (previous_index is not None and block.sample_index != previous_index)
                       or (previous_source_segment is not None and block.segment != previous_source_segment))
                previous_index = block.sample_index + config.block_samples
                previous_source_segment = block.segment
                if gap:
                    estimator.reset()
                    if controller:
                        controller.hold()
                measurement = estimator.process(block)
                dsp_s = time.perf_counter() - block_start
                age = time.monotonic() - block.arrival
                stale = age > config.max_age_s
                valid = measurement.valid and not stale
                if gap or (valid and not previous_valid):
                    segment += 1
                if not valid:
                    invalid += 1
                    if controller:
                        controller.hold()
                    if stale:
                        estimator.reset()
                previous_valid = valid
                slips += int(measurement.slip)
                stale_count += int(stale)
                pending_flags |= GAP if gap else 0
                pending_flags |= SLIP if measurement.slip else 0
                pending_flags |= STALE if stale else 0
                dsp.add(dsp_s)
                age_stats.add(age)
                blocks += 1
                i2c_s = 0.
                measurement.segment = segment
                due_control = measurement.t >= next_control
                if due_control:
                    missed += max(0, int((measurement.t - next_control)*config.control_hz))
                    next_control = (math.floor(measurement.t*config.control_hz)+1)/config.control_hz
                    requested = code
                    if controller:
                        requested = controller.update(measurement, age)
                    elif schedule and valid:
                        target = schedule(measurement.t)
                        requested = int(max(code-config.max_step_codes, min(code+config.max_step_codes, target)))
                    if actuator and requested != code and valid:
                        # Last freshness check immediately before the I2C transaction.
                        if time.monotonic() - block.arrival > config.max_age_s:
                            if controller:
                                controller.hold()
                            stale_count += 1
                            valid = False
                            pending_flags |= STALE
                            estimator.reset()
                            previous_valid = False
                        else:
                            begin = time.perf_counter()
                            actuator.write(int(requested))
                            code = int(requested)
                            i2c_s = time.perf_counter() - begin
                            i2c_stats.add(i2c_s)
                            command_age_stats.add(time.monotonic() - block.arrival)
                            writes += 1
                            if age + i2c_s > config.max_age_s:
                                if controller:
                                    controller.hold()
                                valid = False
                                pending_flags |= STALE
                                estimator.reset()
                                previous_valid = False
                    updates += 1
                state = controller.state if controller else State.MONITOR
                locked = bool(controller and controller.locked)
                locked_blocks += int(locked)
                # Events are reflected in continuity/state fields; periodic records
                # remain uniformly scheduled for phase statistics.
                if measurement.t >= next_log:
                    next_log = (math.floor(measurement.t*config.telemetry_hz)+1)/config.telemetry_hz
                    flags = (VALID if valid else 0) | (LOCKED if locked else 0)
                    flags |= pending_flags
                    writer.append((block.sample_index, measurement.t, measurement.phase,
                                   measurement.frequency, segment, measurement.amplitude,
                                   measurement.snr_db, measurement.coherence, measurement.clipping,
                                   age, dsp_s, i2c_s, code, int(state), flags))
                    pending_flags = 0
                last_state = state
            finally:
                source.release(block)
    except KeyboardInterrupt:
        failure = "interrupted"
    except Exception as exc:
        if controller:
            controller.fault()
        last_state = State.FAULT
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if source_started:
            try:
                source.close()
            except Exception as exc:
                failure = failure or str(exc)
        elapsed = time.monotonic() - start
        pool = getattr(source, "pool", source)
        summary = dict(blocks=blocks, invalid_blocks=invalid, stale_blocks=stale_count,
                       slip_events=slips, updates=updates, dac_writes=writes,
                       missed_control_deadlines=missed, locked_fraction=locked_blocks/max(1, blocks),
                       final_code=code, final_state=int(last_state), failure=failure,
                       wall_seconds=elapsed, cpu_seconds=time.process_time()-cpu_start,
                       dsp=dsp.report(), callback_age=age_stats.report(), i2c=i2c_stats.report(),
                       command_age=command_age_stats.report(),
                       dropped_blocks=pool.dropped_blocks, queue_high_water=pool.high_water,
                       source_settings=getattr(source, "settings", {}))
        writer.close(summary)
    return summary
