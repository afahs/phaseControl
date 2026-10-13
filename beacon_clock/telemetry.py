"""Fixed-size binary records; only the writer thread serializes or touches disk."""
from dataclasses import asdict
import json
from pathlib import Path
import queue
import threading
import time
import numpy as np

MAGIC = b"BCLK001\n"
DTYPE = np.dtype([
    ("record", "<u8"), ("sample", "<u8"), ("t", "<f8"),
    ("phase", "<f8"), ("frequency", "<f8"), ("segment", "<u4"),
    ("amplitude", "<f4"), ("snr_db", "<f4"), ("coherence", "<f4"),
    ("clipping", "<f4"), ("age_s", "<f4"), ("dsp_s", "<f4"), ("i2c_s", "<f4"),
    ("code", "<u2"), ("state", "u1"), ("flags", "u1"),
])
VALID, LOCKED, GAP, SLIP, STALE = 1, 2, 4, 8, 16
METADATA_RESERVE = 65528


class Writer:
    def __init__(self, path, config, metadata, status=False, stall_s=0.):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.metadata = metadata
        self.status = status
        self.stall_s = stall_s  # injectable writer latency for tests
        self.batch_size = max(1, int(round(config.telemetry_hz)))
        self.pending = queue.Queue(config.writer_batches)
        self.free = queue.Queue(config.writer_batches + 1)
        for _ in range(config.writer_batches + 1):
            self.free.put(np.empty(self.batch_size, dtype=DTYPE))
        self.active = self.free.get_nowait()
        self.count = self.sequence = 0
        self.dropped = self.written = 0
        self.capped = False
        self.error = None
        self.bytes = len(MAGIC)
        self.finished = threading.Event()
        self.started = threading.Event()
        self.thread = threading.Thread(target=self._work, name="telemetry-writer", daemon=True)
        self.thread.start()
        if not self.started.wait(5) or self.error:
            raise RuntimeError(f"cannot start telemetry: {self.error}")

    def append(self, values):
        if not self.config.telemetry_enabled:
            return
        seq = self.sequence
        self.sequence += 1
        if self.capped or self.error:
            self.dropped += 1
            return
        self.active[self.count] = (seq, *values)
        self.count += 1
        if self.count == self.batch_size:
            self.publish()

    def publish(self):
        if not self.count:
            return
        try:
            replacement = self.free.get_nowait()
        except queue.Empty:
            self.dropped += self.count
            self.count = 0
            return
        try:
            self.pending.put_nowait((self.active, self.count))
            self.active = replacement
        except queue.Full:
            self.free.put_nowait(replacement)
            self.dropped += self.count
        self.count = 0

    def _metadata(self, summary, final):
        payload = dict(self.metadata, format="beacon-clock", version=1,
                       dtype=DTYPE.descr, record_bytes=DTYPE.itemsize,
                       config=asdict(self.config), final=final, summary=summary,
                       telemetry=dict(enabled=self.config.telemetry_enabled,
                                      records_written=self.written, records_dropped=self.dropped,
                                      capped=self.capped, error=self.error,
                                      complete=final and not self.error and not self.capped and not self.dropped))
        encoded = json.dumps(payload, indent=2, allow_nan=False).encode()
        if len(encoded) > METADATA_RESERVE:
            raise ValueError("metadata exceeds reserved output budget")
        (self.path / "run.json").write_bytes(encoded)

    def _work(self):
        try:
            self._metadata({}, False)
            with (self.path / "telemetry.bin").open("xb", buffering=65536) as f:
                f.write(MAGIC)
                self.started.set()
                while not self.finished.is_set() or not self.pending.empty():
                    try:
                        batch, count = self.pending.get(timeout=.1)
                    except queue.Empty:
                        continue
                    try:
                        if self.stall_s:
                            time.sleep(self.stall_s)
                        remaining = max(0, self.config.max_output_bytes - METADATA_RESERVE - self.bytes)
                        accepted = min(count, remaining // DTYPE.itemsize)
                        if accepted < count:
                            self.capped = True
                            self.dropped += count - accepted
                        if accepted:
                            f.write(batch[:accepted].tobytes())
                            self.written += accepted
                            self.bytes += accepted * DTYPE.itemsize
                        if self.status:
                            r = batch[count - 1]
                            try:
                                print(f"t={r['t']:.1f}s state={int(r['state'])} "
                                      f"error={r['frequency']:+.4f}Hz DAC={int(r['code'])} "
                                      f"age={r['age_s']*1000:.1f}ms", flush=True)
                            except (BrokenPipeError, OSError):
                                # An SSH/stdout disconnect must not stop binary logging.
                                self.status = False
                    finally:
                        self.free.put_nowait(batch)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.started.set()

    def close(self, summary):
        self.publish()
        self.finished.set()
        self.thread.join(10)
        if self.thread.is_alive():
            raise RuntimeError("telemetry writer still blocked; initial metadata marks run unfinished")
        # Control/acquisition have stopped; final small JSON write is outside hot path.
        self._metadata(summary, True)


def read_run(path):
    path = Path(path)
    metadata = json.loads((path / "run.json").read_text())
    with (path / "telemetry.bin").open("rb") as f:
        if f.read(len(MAGIC)) != MAGIC or metadata.get("version") != 1:
            raise ValueError("unsupported telemetry format")
    size = (path / "telemetry.bin").stat().st_size - len(MAGIC)
    if size % DTYPE.itemsize:
        raise ValueError("truncated telemetry record")
    records = np.memmap(path / "telemetry.bin", dtype=DTYPE, mode="r", offset=len(MAGIC)) if size else np.empty(0, dtype=DTYPE)
    return metadata, records
