from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass
class Config:
    sample_rate: int = 1_024_000
    block_samples: int = 8192
    queue_blocks: int = 8
    usb_buffers: int = 4
    center_hz: float = 49_750_000.0
    pilot_hz: float = 50_000_000.0
    expected_offset_hz: float = 250_000.0
    gain_db: float = 9.9
    device: int = 0
    search_hz: float = 10_000.0
    control_hz: float = 10.0
    telemetry_hz: float = 10.0
    bandwidth_hz: float = 0.1
    damping: float = 0.707
    max_age_s: float = 0.1
    dac_min: int = 410
    dac_max: int = 3685
    max_step_codes: int = 8
    dac_address: int = 0x62
    i2c_bus: int = 1
    dac_supply_v: float = 3.3
    clock_ref_hz: float = 27_000_000.0
    apr_ppm: int = 120
    min_snr_db: float = 10.0
    min_coherence: float = 0.90
    min_amplitude: float = 0.005
    max_clip_fraction: float = 0.01
    max_output_bytes: int = 256 * 1024 * 1024
    writer_batches: int = 4

    @property
    def block_s(self):
        return self.block_samples / self.sample_rate

    def validate(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.sample_rate != 1_024_000:
            raise ValueError("v1 supports the characterized 1.024 MS/s rate only")
        if self.block_samples < 8192 or self.block_samples % 8192:
            raise ValueError("block_samples must be a multiple of 8192 (USB transfer alignment)")
        if not 2 <= self.queue_blocks <= 64 or not 2 <= self.usb_buffers <= 32:
            raise ValueError("invalid bounded acquisition buffers")
        if not 0 < self.control_hz <= 1 / self.block_s:
            raise ValueError("control rate must be positive and <= block rate")
        if not 0 < self.telemetry_hz <= 1 / self.block_s:
            raise ValueError("telemetry rate must be positive and <= block rate")
        if not 0 < self.bandwidth_hz < self.control_hz / 10:
            raise ValueError("loop natural frequency must be below control_hz/10")
        if not .5 <= self.damping <= 2 or not 0 < self.max_age_s <= 1:
            raise ValueError("invalid damping or age limit")
        if not 410 <= self.dac_min < self.dac_max <= 3685:
            raise ValueError("DAC limits must stay within 10-90% of its 3.3 V supply")
        if self.dac_supply_v != 3.3 or self.apr_ppm not in (30, 60, 120):
            raise ValueError("v1 supports 3.3 V DAC and APR 30, 60, or 120")
        if not 1 <= self.max_step_codes <= 64:
            raise ValueError("invalid DAC slew limit")
        if abs(self.expected_offset_hz) + self.search_hz >= .45 * self.sample_rate:
            raise ValueError("pilot search extends beyond usable passband")
        if not 100 <= self.search_hz <= 20_000 or self.pilot_hz <= 0:
            raise ValueError("invalid pilot/search frequency")
        if not 0 < self.min_coherence <= 1 or not 0 < self.max_clip_fraction < 1:
            raise ValueError("invalid quality limits")
        if self.max_output_bytes < 65536 or not 1 <= self.writer_batches <= 16:
            raise ValueError("invalid telemetry limits")
        return self

    @classmethod
    def load(cls, path=None):
        values = json.loads(Path(path).read_text()) if path else {}
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        return cls(**values).validate()


@dataclass
class Calibration:
    hz_per_code: float
    center_code: int
    delay_s: float
    center_hz: float
    pilot_hz: float
    expected_offset_hz: float
    sample_rate: int
    apr_ppm: int
    valid: bool = False
    slope_std: float = 0.0
    measured_range: tuple = (0, 4095)

    def validate(self, config):
        if not self.valid or not math.isfinite(self.hz_per_code) or abs(self.hz_per_code) < 1e-5:
            raise ValueError("a valid measured Hz/code calibration is required")
        if not math.isfinite(self.delay_s) or self.delay_s < 0:
            raise ValueError("invalid calibrated delay")
        for key in ("center_hz", "pilot_hz", "expected_offset_hz", "sample_rate", "apr_ppm"):
            if getattr(self, key) != getattr(config, key):
                raise ValueError(f"calibration mismatch: {key}")
        if not config.dac_min <= self.center_code <= config.dac_max:
            raise ValueError("calibration center outside safe DAC range")
        return self

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
