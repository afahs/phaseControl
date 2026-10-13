#!/usr/bin/env python3
"""Alternate CLK0 between 7.2 and 28.8 MHz every five seconds.

Uses the same bus 1, address 0x60, 27 MHz reference and 120 ppm APR
as setClockFreq.py. Press Ctrl+C to stop.
Only CLK0 is enabled. Each switch briefly disables outputs and resets PLLB.
"""

import itertools
import time

from smbus2 import SMBus
from setClockFreq import configure_clk0_vcxo, wait_sys_init_clear

FREQUENCIES_HZ = (7_200_000, 28_800_000)
DWELL_SECONDS = 5.0


def main():
    try:
        with SMBus(1) as bus:
            wait_sys_init_clear(bus)
            print("Alternating CLK0 every 5 seconds. Press Ctrl+C to stop.", flush=True)
            for frequency in itertools.cycle(FREQUENCIES_HZ):
                configure_clk0_vcxo(bus, frequency, 27_000_000, 120)
                print(
                    f"CLK0 set to {frequency / 1e6:.1f} MHz; holding for 5 seconds",
                    flush=True,
                )
                time.sleep(DWELL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
