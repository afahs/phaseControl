"""One owner per USB buffer; explicit software loss and bounded queues."""
from dataclasses import dataclass
import queue
import threading
import time
import numpy as np


@dataclass(slots=True)
class Block:
    raw: np.ndarray
    sequence: int = 0
    sample_index: int = 0
    arrival: float = 0.0
    segment: int = 0
    lost: bool = False


class BufferPool:
    def __init__(self, samples, capacity):
        self.samples = samples
        self.ready = queue.Queue(capacity)
        self.free = queue.Queue(capacity + 2)
        for _ in range(capacity + 2):
            self.free.put(Block(np.empty(2 * samples, dtype=np.uint8)))
        self.sequence = self.index = self.segment = 0
        self.dropped_blocks = self.high_water = 0
        self.error = None

    def discard_pending(self):
        dropped = 0
        while True:
            try:
                self.free.put_nowait(self.ready.get_nowait())
                dropped += 1
            except queue.Empty:
                break
        self.dropped_blocks += dropped
        return dropped

    def callback(self, raw, _context=None):
        # ctypes callbacks must never allow exceptions to escape unnoticed.
        try:
            arrival = time.monotonic()
            incoming = np.frombuffer(raw, dtype=np.uint8)
            if incoming.size != self.samples * 2:
                raise ValueError(f"unexpected USB block length: {incoming.size}")
            lost = self.ready.full()
            if lost:
                self.discard_pending()
                self.segment += 1
            block = self.free.get_nowait()
            np.copyto(block.raw, incoming)
            block.sequence, block.sample_index = self.sequence, self.index
            block.arrival, block.segment, block.lost = arrival, self.segment, lost
            self.sequence += 1
            self.index += self.samples
            self.ready.put_nowait(block)
            self.high_water = max(self.high_water, self.ready.qsize())
        except Exception as exc:
            self.error = exc

    def release(self, block):
        self.free.put_nowait(block)


class RtlSource:
    def __init__(self, config):
        self.config = config
        self.pool = BufferPool(config.block_samples, config.queue_blocks)
        self.device = None
        self.thread = None
        self.stopping = threading.Event()
        self.settings = {}

    def start(self):
        from rtlsdr import RtlSdr
        c = self.config
        self.device = RtlSdr(c.device)
        try:
            self.device.DEFAULT_ASYNC_BUF_NUMBER = c.usb_buffers
            self.device.sample_rate = c.sample_rate
            self.device.center_freq = c.center_hz
            self.device.gain = c.gain_db
            self.device.set_agc_mode(False)
            # Some drivers return -2 when zero correction is already selected.
            if self.device.get_freq_correction() != 0:
                self.device.set_freq_correction(0)
            self.settings = dict(sample_rate=self.device.sample_rate,
                                 center_hz=self.device.center_freq,
                                 gain_db=self.device.gain,
                                 usb_buffers=c.usb_buffers)
            self.thread = threading.Thread(target=self._read, name="sdr-acquisition", daemon=True)
            self.thread.start()
        except Exception:
            self.device.close()
            raise

    def _read(self):
        try:
            self.device.read_bytes_async(self.pool.callback, self.config.block_samples * 2)
            if not self.stopping.is_set():
                self.pool.error = RuntimeError("SDR stream ended unexpectedly")
        except Exception as exc:
            if not self.stopping.is_set():
                self.pool.error = exc

    def get(self, timeout=1.0):
        if self.pool.error:
            raise RuntimeError("SDR acquisition failed") from self.pool.error
        return self.pool.ready.get(timeout=timeout)

    def release(self, block):
        self.pool.release(block)

    def close(self):
        self.stopping.set()
        if self.device and self.thread:
            self.device.cancel_read_async()
            self.thread.join(5)
            if self.thread.is_alive():
                raise RuntimeError("SDR reader did not stop; device remains owned by reader")
        if self.device:
            self.device.close()
