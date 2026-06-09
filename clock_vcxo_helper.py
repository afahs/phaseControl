#!/usr/bin/env python3
"""
clock_vcxo_helper.py

Si5351 + MCP4725 helper using the SAME low-level I2C register approach
you previously validated (SMBus2).

Implements:
- init_clock_vcxo_28p8mhz(apr_ppm): configure Si5351 CLK0 ~28.800 MHz via VCXO/PLLB path.
- set_clock0_freq_hz(f_hz): coarse I2C frequency change by updating MultiSynth0 (fractional capable).
- set_dac_code(code): fine tuning via MCP4725 VC pin. Returns the commanded code by
  default; pass readback=True when an immediate I2C readback is required.
- dac_code_to_volts(code): convenience conversion.

Notes:
- Si5351 at 0x60 on I2C bus 1
- MCP4725 at 0x62 on I2C bus 1
- Uses PLLB VCXO path (PLLB params at regs 34..41) with c=106 as in your script.
- Keeps PLLB at ~864 MHz and varies MS0 divider to change CLK0 frequency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from smbus2 import SMBus, i2c_msg

# ---------------------------
# Default I2C addresses/bus
# ---------------------------
I2C_BUS_DEFAULT = 1
SI5351_ADDR_DEFAULT = 0x60
MCP4725_ADDR_DEFAULT = 0x62

VDD_DAC_DEFAULT = 3.3
I2C_MARKER_GPIO_PIN_DEFAULT = 37
I2C_MARKER_GPIO_MODE_DEFAULT = "BOARD"
MCP4725_FAST_WRITE_DEFAULT = True
MCP4725_VERIFY_WRITES_DEFAULT = False
SI5351_FAST_CLOCK0_UPDATES_DEFAULT = True
REQUIRE_CLOCK_CONFIG_FOR_DAC_DEFAULT = True


# ---------------------------
# Si5351 register map bits
# ---------------------------
REG0_STATUS = 0
REG3_OE = 3
REG16_CLK0 = 16
REG_PLLRST = 177

# PLLA params: 26..33, PLLB params: 34..41
PLLB_BASE = 34
# MultiSynth0 params: 42..49
MS0_BASE = 42

# VCXO param regs: 162..164
REG_VCXO_0 = 162
REG_VCXO_1 = 163
REG_VCXO_2 = 164


# ---------------------------
# Low-level helpers
# ---------------------------
def _rd(bus: SMBus, addr: int, reg: int) -> int:
    return bus.read_byte_data(addr, reg)


def _wr(bus: SMBus, addr: int, reg: int, val: int) -> None:
    bus.write_byte_data(addr, reg, val & 0xFF)


def _wr_block(bus: SMBus, addr: int, reg: int, vals) -> None:
    bus.i2c_rdwr(i2c_msg.write(addr, [int(reg) & 0xFF] + [int(v) & 0xFF for v in vals]))


def _wait_sys_init_clear(bus: SMBus, addr: int, timeout_s: float = 1.0) -> None:
    t0 = time.time()
    while _rd(bus, addr, REG0_STATUS) & 0x80:
        if time.time() - t0 > timeout_s:
            raise RuntimeError("Si5351 SYS_INIT did not clear (REG0 bit7 stayed high).")
        time.sleep(0.01)


def _params_bytes(p1: int, p2: int, p3: int) -> list[int]:
    return [
        (p3 >> 8) & 0xFF,
        (p3 >> 0) & 0xFF,
        (p1 >> 16) & 0x03,
        (p1 >> 8) & 0xFF,
        (p1 >> 0) & 0xFF,
        ((p3 >> 12) & 0xF0) | ((p2 >> 16) & 0x0F),
        (p2 >> 8) & 0xFF,
        (p2 >> 0) & 0xFF,
    ]


def _write_params(bus: SMBus, addr: int, base: int, p1: int, p2: int, p3: int) -> None:
    """
    Write 8-byte parameter block for PLL/MS registers.
    Matches your previous write_params().
    """
    _wr_block(bus, addr, base, _params_bytes(p1, p2, p3))


def _pll_params(a: int, b: int, c: int):
    """
    Standard Si5351 PLL encoding (same as your script).
    """
    t = (128 * b) // c
    p1 = 128 * a + t - 512
    p2 = 128 * b - c * t
    p3 = c
    return p1, p2, p3


def _ms_frac_params(a: int, b: int, c: int):
    """
    MultiSynth fractional encoding (same structure as PLL).
    Divider = a + b/c.
    """
    t = (128 * b) // c
    p1 = 128 * a + t - 512
    p2 = 128 * b - c * t
    p3 = c
    return p1, p2, p3


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


class _GpioI2cMarker:
    """
    Scope marker for DAC I2C writes.

    Pin numbers default to Raspberry Pi physical-header numbering so "pin 37"
    means header pin 37, not BCM GPIO number 37.
    """

    def __init__(self, pin: int, mode: str = I2C_MARKER_GPIO_MODE_DEFAULT) -> None:
        self.pin = int(pin)
        self.mode = str(mode).upper()
        self._gpio = None
        self._ready = False
        self._warned = False

    def open(self) -> None:
        if self._ready:
            return
        try:
            import RPi.GPIO as GPIO

            if self.mode == "BOARD":
                GPIO.setmode(GPIO.BOARD)
            elif self.mode == "BCM":
                GPIO.setmode(GPIO.BCM)
            else:
                raise ValueError("GPIO marker mode must be BOARD or BCM.")
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO
            self._ready = True
        except Exception as exc:
            if not self._warned:
                print(
                    "WARNING: GPIO I2C marker disabled: {}".format(exc),
                    flush=True,
                )
                self._warned = True
            self._gpio = None
            self._ready = False

    def high(self) -> None:
        if self._ready and self._gpio is not None:
            self._gpio.output(self.pin, self._gpio.HIGH)

    def low(self) -> None:
        if self._ready and self._gpio is not None:
            self._gpio.output(self.pin, self._gpio.LOW)

    def close(self) -> None:
        if self._gpio is None:
            return
        try:
            self.low()
            self._gpio.cleanup(self.pin)
        finally:
            self._gpio = None
            self._ready = False


# ---------------------------
# MCP4725 helpers
# ---------------------------
def _mcp4725_write_dac(bus: SMBus, addr: int, code: int, fast_write: bool = True) -> None:
    """
    MCP4725 DAC-register write, with no EEPROM update.
    """
    code = _clamp_int(code, 0, 4095)
    if fast_write:
        # MCP4725 fast-mode write: no EEPROM, two data bytes, immediate DAC update.
        bus.i2c_rdwr(i2c_msg.write(addr, [(code >> 8) & 0x0F, code & 0xFF]))
    else:
        b0 = 0x40
        b1 = (code >> 4) & 0xFF
        b2 = (code & 0x0F) << 4
        bus.i2c_rdwr(i2c_msg.write(addr, [b0, b1, b2]))


def _mcp4725_readback(bus: SMBus, addr: int) -> int:
    r = i2c_msg.read(addr, 5)
    bus.i2c_rdwr(r)
    data = list(r)
    return (data[1] << 4) | (data[2] >> 4)


# ---------------------------
# ClockVCXO class
# ---------------------------
@dataclass
class ClockVCXO:
    i2c_bus: int = I2C_BUS_DEFAULT
    si5351_addr: int = SI5351_ADDR_DEFAULT
    mcp4725_addr: int = MCP4725_ADDR_DEFAULT
    vdd_dac: float = VDD_DAC_DEFAULT
    i2c_marker_gpio_pin: Optional[int] = I2C_MARKER_GPIO_PIN_DEFAULT
    i2c_marker_gpio_mode: str = I2C_MARKER_GPIO_MODE_DEFAULT
    mcp4725_fast_write: bool = MCP4725_FAST_WRITE_DEFAULT
    verify_dac_writes: bool = MCP4725_VERIFY_WRITES_DEFAULT
    fast_clock0_updates: bool = SI5351_FAST_CLOCK0_UPDATES_DEFAULT
    require_clock_config_for_dac: bool = REQUIRE_CLOCK_CONFIG_FOR_DAC_DEFAULT

    # Internal state
    _bus: Optional[SMBus] = None
    _i2c_marker: Optional[_GpioI2cMarker] = None
    _configured: bool = False
    _pll_b_hz: float = 864e6  # nominal after init (27 MHz * 32) via VCXO path
    _last_clock0_hz: Optional[float] = None
    _last_dac_code: Optional[int] = None

    def __enter__(self) -> "ClockVCXO":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._bus is None:
            self._bus = SMBus(self.i2c_bus)
        if self.i2c_marker_gpio_pin is not None and self._i2c_marker is None:
            self._i2c_marker = _GpioI2cMarker(
                int(self.i2c_marker_gpio_pin),
                self.i2c_marker_gpio_mode,
            )
            self._i2c_marker.open()

    def close(self) -> None:
        if self._i2c_marker is not None:
            try:
                self._i2c_marker.close()
            finally:
                self._i2c_marker = None
        if self._bus is not None:
            try:
                self._bus.close()
            finally:
                self._bus = None

    # -----------------------
    # Public convenience
    # -----------------------
    def dac_code_to_volts(self, code: int) -> float:
        code = _clamp_int(code, 0, 4095)
        return (code / 4095.0) * float(self.vdd_dac)

    def volts_to_dac_code(self, v: float) -> int:
        v = max(0.0, min(float(self.vdd_dac), float(v)))
        return int(round((v / float(self.vdd_dac)) * 4095.0))

    # -----------------------
    # Si5351 init (your proven method)
    # -----------------------
    def init_clock_vcxo_28p8mhz(self, apr_ppm: int = 120) -> None:
        """
        Configure Si5351:
        - PLLB VCXO mode: a=32, b=0, c=106 => ~864 MHz
        - VCXO_PARAM based on APR_PPM (regs 162..164)
        - MS0 divider: 30 (integer) => 28.8 MHz nominal
        - REG16 = 0x6F routing: CLK0 from MS0; MS0 source VCXO/PLLB
        - REG3 = 0xFE enable only CLK0
        """
        if self._bus is None:
            self.open()
        assert self._bus is not None
        bus = self._bus

        _wait_sys_init_clear(bus, self.si5351_addr)

        # Disable all outputs while programming
        _wr(bus, self.si5351_addr, REG3_OE, 0xFF)

        # Program PLLB (VCXO path) with denominator c=106
        a, b, c = 32, 0, 106
        p1, p2, p3 = _pll_params(a, b, c)
        _write_params(bus, self.si5351_addr, PLLB_BASE, p1, p2, p3)

        # VCXO pull range (must be non-zero)
        # Same formula you used: int(round(1.03 * 128 * (a + b/106) * APR_PPM))
        vcxo_param = int(round(1.03 * 128.0 * (a + (b / 106.0)) * int(apr_ppm)))
        _wr(bus, self.si5351_addr, REG_VCXO_0, (vcxo_param >> 0) & 0xFF)
        _wr(bus, self.si5351_addr, REG_VCXO_1, (vcxo_param >> 8) & 0xFF)
        _wr(bus, self.si5351_addr, REG_VCXO_2, (vcxo_param >> 16) & 0x3F)  # top 2 bits must be 0

        # Program MS0 for 28.800 MHz nominal via integer divider 30
        div = 30
        ms_p1 = 128 * div - 512
        ms_p2 = 0
        ms_p3 = 1
        _write_params(bus, self.si5351_addr, MS0_BASE, ms_p1, ms_p2, ms_p3)

        # Route CLK0 from MS0 and set MS0 source = VCXO/PLLB, max drive.
        _wr(bus, self.si5351_addr, REG16_CLK0, 0x6F)

        # Reset PLLB so it re-locks cleanly
        _wr(bus, self.si5351_addr, REG_PLLRST, 0x80)
        time.sleep(0.02)

        # Enable only CLK0 (your convention)
        _wr(bus, self.si5351_addr, REG3_OE, 0xFE)

        # Track nominal PLLB frequency (still VCXO-controlled in reality)
        self._pll_b_hz = 864e6
        self._last_clock0_hz = 28.8e6
        self._configured = True

    # -----------------------
    # Coarse frequency update via I2C (fractional MS0)
    # -----------------------
    def set_clock0_freq_hz(
        self,
        f_hz: float,
        settle_s: float = 0.0,
        *,
        disable_output: Optional[bool] = None,
        reset_pll: Optional[bool] = None,
    ) -> float:
        """
        Coarse change of CLK0 frequency by updating MultiSynth0 divider ratio.
        Uses fractional mode so you can do kHz–10 kHz class steps without huge jumps.

        Keeps PLLB at ~864 MHz (VCXO path), changes MS0 divider:
            f_out = f_pll / (a + b/c)

        Returns commanded frequency (Hz).

        Requires init_clock_vcxo_28p8mhz() called first.
        """
        if not self._configured:
            raise RuntimeError("ClockVCXO not configured. Call init_clock_vcxo_28p8mhz() first.")
        if self._bus is None:
            self.open()
        assert self._bus is not None
        bus = self._bus

        f_out = float(f_hz)
        if not (1e3 <= f_out <= 200e6):
            raise ValueError(f"Requested CLK0 frequency {f_out} Hz seems invalid.")

        f_pll = float(self._pll_b_hz)
        ratio = f_pll / f_out  # divider

        # Si5351 MS divider constraints: a in [4..900] typically (datasheet).
        # We'll clamp conservatively.
        a = int(math.floor(ratio))
        if a < 4:
            a = 4
        if a > 900:
            a = 900

        frac = ratio - float(a)
        if frac < 0:
            frac = 0.0

        # Choose a large-ish c for fine resolution but within 20-bit limit (<= 1,048,575)
        c = 1_000_000
        b = int(round(frac * c))
        if b >= c:
            b = c - 1
        if b < 0:
            b = 0

        p1, p2, p3 = _ms_frac_params(a, b, c)

        fast_update = bool(self.fast_clock0_updates)
        if disable_output is None:
            disable_output = not fast_update
        if reset_pll is None:
            reset_pll = not fast_update

        if disable_output:
            _wr(bus, self.si5351_addr, REG3_OE, 0xFF)
        _write_params(bus, self.si5351_addr, MS0_BASE, p1, p2, p3)
        if reset_pll:
            _wr(bus, self.si5351_addr, REG_PLLRST, 0x80)
            time.sleep(0.01)
        if disable_output:
            _wr(bus, self.si5351_addr, REG3_OE, 0xFE)

        # Best-effort commanded frequency (actual will differ slightly due to quantization/VCXO pull)
        f_cmd = f_pll / (float(a) + (float(b) / float(c)))
        self._last_clock0_hz = f_cmd

        if settle_s and float(settle_s) > 0:
            time.sleep(float(settle_s))

        return float(f_cmd)

    # -----------------------
    # Fine tuning via DAC (VC pin)
    # -----------------------
    def read_dac_code(self) -> int:
        if self._bus is None:
            self.open()
        assert self._bus is not None
        rb = _mcp4725_readback(self._bus, self.mcp4725_addr)
        self._last_dac_code = int(rb)
        return int(rb)

    def set_dac_code(
        self,
        code: int,
        settle_s: float = 0.0,
        *,
        readback: Optional[bool] = None,
    ) -> int:
        """
        Write MCP4725 code and return the applied code.

        By default this returns the clamped command after the write succeeds.
        Set readback=True, or construct with verify_dac_writes=True, to force a
        separate MCP4725 I2C readback for diagnostics.

        Requires init_clock_vcxo_28p8mhz() called first unless constructed with
        require_clock_config_for_dac=False for DAC-only tests.
        """
        if self.require_clock_config_for_dac and not self._configured:
            raise RuntimeError("ClockVCXO not configured. Call init_clock_vcxo_28p8mhz() first.")
        if self._bus is None:
            self.open()
        assert self._bus is not None
        bus = self._bus

        code_i = _clamp_int(int(code), 0, 4095)
        marker = self._i2c_marker
        if marker is not None:
            marker.high()
        try:
            _mcp4725_write_dac(
                bus,
                self.mcp4725_addr,
                code_i,
                fast_write=bool(self.mcp4725_fast_write),
            )
        finally:
            if marker is not None:
                marker.low()
        if settle_s and float(settle_s) > 0:
            time.sleep(float(settle_s))
        if readback is None:
            readback = bool(self.verify_dac_writes)
        if readback:
            return self.read_dac_code()
        self._last_dac_code = code_i
        return int(code_i)
