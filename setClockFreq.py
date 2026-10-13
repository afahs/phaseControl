#!/usr/bin/env python3
"""
setClockFreq.py

Set Si5351 CLK0 to a requested frequency (nominal), while ensuring CLK0 is routed
through MultiSynth0 sourced from the VCXO/PLLB path so your VC/I (driven by DAC)
can still fine-tune it.

Default assumptions match your recent setup:
- I2C bus 1
- Si5351 at 0x60
- Reference ~27 MHz (common on Si5351B VCXO boards)
- Output: CLK0 (pad/output labeled "0")

Notes:
- This script sets a *nominal* frequency using PLLB + MS0.
- Fine tuning via VC/I depends on your VCXO range (APR) and analog wiring.

Requires: smbus2 (pip install smbus2)
"""

import argparse
import math
from fractions import Fraction
from smbus2 import SMBus

# -----------------------------
# Device addresses / registers
# -----------------------------
SI5351_ADDR = 0x60

REG_STATUS = 0
REG_OE     = 3
REG_CLK0   = 16
REG_PLLRST = 177

# PLLB parameter block: 34..41
PLLB_BASE  = 34
# MS0 parameter block: 42..49
MS0_BASE   = 42

# VCXO parameters: 162..164
REG_VCXO_0 = 162
REG_VCXO_1 = 163
REG_VCXO_2 = 164

# -----------------------------
# Low-level helpers
# -----------------------------
def rd(bus: SMBus, reg: int) -> int:
    return bus.read_byte_data(SI5351_ADDR, reg)

def wr(bus: SMBus, reg: int, val: int) -> None:
    bus.write_byte_data(SI5351_ADDR, reg, val & 0xFF)

def wait_sys_init_clear(bus: SMBus, timeout_s: float = 1.0) -> None:
    import time
    t0 = time.time()
    while rd(bus, REG_STATUS) & 0x80:
        if time.time() - t0 > timeout_s:
            raise RuntimeError("Si5351 SYS_INIT did not clear (REG0 bit7 stayed high).")
        time.sleep(0.01)

def write_params_8(bus: SMBus, base: int, p1: int, p2: int, p3: int) -> None:
    """
    Packs (P1,P2,P3) into the standard Si5351 8-byte parameter block.
    """
    wr(bus, base + 0, (p3 >> 8) & 0xFF)
    wr(bus, base + 1, (p3 >> 0) & 0xFF)
    wr(bus, base + 2, (p1 >> 16) & 0x03)
    wr(bus, base + 3, (p1 >> 8) & 0xFF)
    wr(bus, base + 4, (p1 >> 0) & 0xFF)
    wr(bus, base + 5, ((p3 >> 12) & 0xF0) | ((p2 >> 16) & 0x0F))
    wr(bus, base + 6, (p2 >> 8) & 0xFF)
    wr(bus, base + 7, (p2 >> 0) & 0xFF)

def calc_p123(a: int, b: int, c: int):
    """
    Si5351 encoding for PLL and MultiSynth fractional settings:
      P1 = 128*a + floor(128*b/c) - 512
      P2 = 128*b - c*floor(128*b/c)
      P3 = c
    """
    t  = (128 * b) // c
    p1 = 128 * a + t - 512
    p2 = 128 * b - c * t
    p3 = c
    return p1, p2, p3

def decode_reg16(r16: int):
    clk0_src = (r16 >> 2) & 0b11   # bits 3:2
    ms0_src  = (r16 >> 5) & 0b1    # bit 5
    ms0_int  = (r16 >> 6) & 0b1    # bit 6
    pdown    = (r16 >> 7) & 0b1    # bit 7
    idrv     = r16 & 0b11          # bits 1:0
    return clk0_src, ms0_src, ms0_int, pdown, idrv

def build_reg16(ms0_int: bool, drive_8ma: bool = True) -> int:
    """
    REG16 fields:
      bit7  CLK0_PDN   = 0 (power up)
      bit6  MS0_INT    = 1 if integer mode else 0
      bit5  MS0_SRC    = 1 (VCXO/PLLB)
      bits3:2 CLK0_SRC = 11 (MultiSynth0)
      bits1:0 IDRV     = 11 (8 mA) if drive_8ma else 00 (2 mA)
    """
    idrv = 0b11 if drive_8ma else 0b00
    return (0 << 7) | ((1 if ms0_int else 0) << 6) | (1 << 5) | (0b11 << 2) | idrv

def set_vcxo_param(bus: SMBus, a_pll: int, b_pll: int, c_pll: int, apr_ppm: int) -> int:
    """
    Sets VCXO parameter registers (must be non-zero for VC pulling to take effect).
    AN619 section 7 requires c=10^6 and:
      VCXO_Param = 1.03 * (128*a + b/10^6) * APR
    """
    if c_pll != 1_000_000:
        raise ValueError("VCXO PLLB denominator must be 1,000,000.")
    vcxo_param = int(round(1.03 * (128.0 * a_pll + b_pll / c_pll) * apr_ppm))
    wr(bus, REG_VCXO_0, (vcxo_param >> 0) & 0xFF)
    wr(bus, REG_VCXO_1, (vcxo_param >> 8) & 0xFF)
    wr(bus, REG_VCXO_2, (vcxo_param >> 16) & 0x3F)  # top bits must be 0
    return vcxo_param

# -----------------------------
# Main configuration routine
# -----------------------------
def configure_clk0_vcxo(bus: SMBus, fout_hz: float, ref_hz: float, apr_ppm: int):
    """
    Strategy (ordinary MultiSynth mode, without divide-by-4):
      - Choose a fixed PLLB multiplier a_pll (integer) so VCO is in a safe range.
      - Use fractional MultiSynth0 to reach fout.
      - Ensure REG16 routes CLK0 from MS0 and MS0 from VCXO/PLLB.

    AN619 requires PLLB denominator c=10^6 in VCXO mode.
    Keep b_pll=0 for an integer feedback ratio.
    """
    if not math.isfinite(fout_hz) or fout_hz <= 0:
        raise ValueError("Frequency must be finite and > 0 Hz.")
    if not math.isfinite(ref_hz) or not 25e6 <= ref_hz <= 27e6:
        raise ValueError("Si5351B crystal reference must be between 25 and 27 MHz.")
    if not 1 <= apr_ppm <= 240:
        raise ValueError("APR must be between 1 and 240 ppm.")

    # Choose a PLLB multiplier so VCO ~ 800–900 MHz (works well for 27–30 MHz outputs too).
    # Keep simple: pick a_pll so VCO is near 864 MHz by default.
    # For ref=27 MHz => a_pll=32 gives 864 MHz.
    # For other ref, still pick nearest integer to 864/ref.
    a_pll = max(15, min(90, int(round(864_000_000.0 / ref_hz))))
    b_pll = 0
    c_pll = 1_000_000
    vco_hz = ref_hz * a_pll

    # MultiSynth divider needed
    div = vco_hz / fout_hz
    if not (8.0 <= div <= 900.0):
        raise ValueError(
            f"Requested fout implies MS divider {div:.3f}, out of supported range [8,900]. "
            f"Try a different ref_hz or a different output plan."
        )

    # Represent divider as fraction a_ms + b_ms/c_ms
    frac = Fraction(div).limit_denominator(1_048_575)
    a_ms = frac.numerator // frac.denominator
    rem = frac.numerator - a_ms * frac.denominator
    b_ms = rem
    c_ms = frac.denominator

    # Validate the frequency plan before disabling any outputs.
    wr(bus, REG_OE, 0xFF)

    # Program PLLB
    p1, p2, p3 = calc_p123(a_pll, b_pll, c_pll)
    write_params_8(bus, PLLB_BASE, p1, p2, p3)

    # Program VCXO pull range (must be non-zero)
    vcxo_param = set_vcxo_param(bus, a_pll, b_pll, c_pll, apr_ppm)

    # Program MS0
    ms_p1, ms_p2, ms_p3 = calc_p123(a_ms, b_ms, c_ms)
    write_params_8(bus, MS0_BASE, ms_p1, ms_p2, ms_p3)

    # AN619 permits integer mode only for even integer output dividers.
    want_r16 = build_reg16(ms0_int=(b_ms == 0 and a_ms % 2 == 0), drive_8ma=True)
    cur_r16 = rd(bus, REG_CLK0)
    if cur_r16 != want_r16:
        wr(bus, REG_CLK0, want_r16)

    # Reset PLLs (PLLB reset bit is bit 7)
    wr(bus, REG_PLLRST, 0x80)

    # Enable only CLK0: REG3 bit0 = 0 enables CLK0, others disabled (1).
    # REG3 is active-high disable per the Si5351 register map.
    wr(bus, REG_OE, 0xFE)

    # Return details for logging
    achieved = vco_hz / (a_ms + (b_ms / c_ms if c_ms else 0.0))
    return {
        "ref_hz": ref_hz,
        "fout_req_hz": fout_hz,
        "fout_nom_hz": achieved,
        "pllb_a": a_pll,
        "pllb_b": b_pll,
        "pllb_c": c_pll,
        "vco_hz": vco_hz,
        "ms_a": a_ms,
        "ms_b": b_ms,
        "ms_c": c_ms,
        "vcxo_param": vcxo_param,
        "reg16_before": cur_r16,
        "reg16_after": want_r16,
    }

def main():
    ap = argparse.ArgumentParser(
        description="Set Si5351 CLK0 to a target frequency with VCXO/PLLB routing (VC/I enabled)."
    )
    ap.add_argument("frequency", type=float, help="Target CLK0 frequency in Hz (e.g., 28.8e6)")
    ap.add_argument("--bus", type=int, default=1, help="I2C bus number (default: 1)")
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=0x60, help="Si5351 I2C address (default: 0x60)")
    ap.add_argument("--ref", type=float, default=27e6, help="Reference frequency in Hz (default: 27e6)")
    ap.add_argument("--apr", type=int, default=120, help="VCXO pull range parameter in ppm (default: 120)")
    args = ap.parse_args()

    global SI5351_ADDR
    SI5351_ADDR = args.addr

    with SMBus(args.bus) as bus:
        wait_sys_init_clear(bus)

        info = configure_clk0_vcxo(bus, args.frequency, args.ref, args.apr)

        # Read back REG16 decode
        r16 = rd(bus, REG_CLK0)
        clk0_src, ms0_src, ms0_int, pdown, idrv = decode_reg16(r16)

        print("Configured Si5351 CLK0 (VCXO-enabled path):")
        print(f"  Requested: {info['fout_req_hz']:.3f} Hz")
        print(f"  Nominal:   {info['fout_nom_hz']:.3f} Hz  (before DAC pulling)")
        print(f"  REF:       {info['ref_hz']:.3f} Hz")
        print(f"  PLLB:      a={info['pllb_a']} b={info['pllb_b']} c={info['pllb_c']}  -> VCO={info['vco_hz']:.3f} Hz")
        print(f"  MS0:       a={info['ms_a']} b={info['ms_b']} c={info['ms_c']}  -> divider={info['vco_hz']/info['fout_nom_hz']:.9f}")
        print(f"  VCXO_Param: {info['vcxo_param']} (APR={args.apr} ppm)")
        print(f"  REG16:     0x{r16:02X}  (CLK0_SRC={clk0_src:02b}, MS0_SRC={ms0_src}, MS0_INT={ms0_int}, PDN={pdown}, IDRV={idrv:02b})")
        print("  REG3 (OE): 0x%02X (CLK0 enabled only)" % rd(bus, REG_OE))

if __name__ == "__main__":
    main()
