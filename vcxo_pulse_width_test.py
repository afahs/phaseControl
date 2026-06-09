#!/usr/bin/env python3
"""
vcxo_pulse_width_test.py

Pulse-width settling test for a DAC-controlled VCXO.

Measurement method:
- The Raspberry Pi repeatedly commands a DAC from an old code to a new code.
- The new code is held for a controlled pulse width N milliseconds.
- The DAC is then returned to the old code.
- A frequency counter such as an HP 53131A is observed manually.
- By sweeping N, you can determine whether the counter reaches a threshold
  such as 90% or 95% of the final expected frequency step.

Example:
    python3 vcxo_pulse_width_test.py \\
      --old-code 2050 \\
      --new-code 2055 \\
      --widths-ms 10,20,30,50,75,100,150,200 \\
      --repeats 50 \\
      --old-dwell-s 1.0 \\
      --pause-between

At startup, the script first sets the DAC to the old code, then holds the new
code until Enter is pressed. Use that initial hold to verify the full frequency
step on the counter before beginning the pulse-width sweep.

Important:
- This version is adapted for the existing phaseControl setup:
  MCP4725 DAC on I2C through clock_vcxo_helper.py's fast no-EEPROM path.
- No GPIO marker pin is used.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import clock_vcxo_helper as cvh


SPIN_MARGIN_NS = 300_000  # Busy-wait for final 300 us for tighter pulse timing.
MCP4725_DEFAULT_ADDR = 0x62


def parse_widths_ms(text: str) -> List[float]:
    widths: List[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("Pulse widths must be positive.")
        widths.append(value)
    if not widths:
        raise argparse.ArgumentTypeError("At least one pulse width is required.")
    return widths


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(x)))


class VcxoDac:
    def __init__(
        self,
        i2c_bus: int,
        mcp4725_addr: int,
        clock_apr_ppm: int,
        skip_clock_init: bool,
        readback: bool,
    ) -> None:
        self.i2c_bus = int(i2c_bus)
        self.mcp4725_addr = int(mcp4725_addr)
        self.clock_apr_ppm = int(clock_apr_ppm)
        self.skip_clock_init = bool(skip_clock_init)
        self.readback = bool(readback)
        self.clock: Optional[cvh.ClockVCXO] = None

    def open(self) -> None:
        self.clock = cvh.ClockVCXO(
            i2c_bus=self.i2c_bus,
            mcp4725_addr=self.mcp4725_addr,
            i2c_marker_gpio_pin=None,
            verify_dac_writes=self.readback,
            require_clock_config_for_dac=not self.skip_clock_init,
        )
        self.clock.open()
        if not self.skip_clock_init:
            self.clock.init_clock_vcxo_28p8mhz(apr_ppm=self.clock_apr_ppm)

    def close(self) -> None:
        if self.clock is not None:
            try:
                self.clock.close()
            finally:
                self.clock = None

    def write_dac(self, code: int) -> None:
        """
        Write an MCP4725 DAC code through clock_vcxo_helper.py's fast path.
        """
        if self.clock is None:
            raise RuntimeError("DAC bus is not open.")

        code_i = clamp_int(int(round(code)), 0, 4095)
        self.clock.set_dac_code(code_i, readback=self.readback)


def wait_until_ns(deadline_ns: int) -> None:
    """
    Hybrid wait:
    - Sleep for most of the remaining interval.
    - Busy-wait for the final few hundred microseconds.

    This gives better millisecond pulse timing than using time.sleep() alone.
    """
    while True:
        now = time.perf_counter_ns()
        remaining_ns = deadline_ns - now
        if remaining_ns <= 0:
            return

        if remaining_ns > SPIN_MARGIN_NS:
            sleep_s = (remaining_ns - SPIN_MARGIN_NS) / 1_000_000_000
            time.sleep(sleep_s)
        else:
            while time.perf_counter_ns() < deadline_ns:
                pass
            return


def run_pulse_group(
    dac: VcxoDac,
    old_code: int,
    new_code: int,
    old_dwell_s: float,
    width_ms: float,
    repeats: int,
) -> List[float]:
    high_durations_ms: List[float] = []
    width_ns = int(round(width_ms * 1_000_000))

    for idx in range(1, repeats + 1):
        dac.write_dac(old_code)
        time.sleep(old_dwell_s)

        high_start_ns = time.perf_counter_ns()
        dac.write_dac(new_code)

        high_end_target_ns = high_start_ns + width_ns
        wait_until_ns(high_end_target_ns)

        before_return_ns = time.perf_counter_ns()
        dac.write_dac(old_code)
        after_return_ns = time.perf_counter_ns()

        # Approximate high-code duration as time from just before the new-code
        # I2C write until just before the old-code I2C write. This includes the
        # new-code write time, which is usually what the VCXO control node sees.
        high_duration_ms = (before_return_ns - high_start_ns) / 1_000_000
        high_durations_ms.append(high_duration_ms)

        print(
            f"  pulse {idx:3d}/{repeats}: "
            f"target={width_ms:.3f} ms, "
            f"actual_high={high_duration_ms:.3f} ms, "
            f"return_write={(after_return_ns - before_return_ns) / 1_000_000:.3f} ms",
            flush=True,
        )

    return high_durations_ms


def print_group_header(width_ms: float, old_code: int, new_code: int, repeats: int) -> None:
    print()
    print("=" * 72)
    print(f"Now testing {width_ms:g} ms pulses.")
    print(f"Old DAC code: {old_code}")
    print(f"New DAC code: {new_code}")
    print(f"Repeats:      {repeats}")
    print()
    print("Watch the HP 53131A frequency counter.")
    print("Record whether the reading reaches the 90% or 95% threshold")
    print("of the final frequency step during these pulses.")
    print("=" * 72)
    print()


def summarize_group(width_ms: float, durations_ms: Iterable[float]) -> None:
    durations = list(durations_ms)
    if not durations:
        print("No pulses recorded.")
        return

    print()
    print(f"Summary for target width {width_ms:g} ms:")
    print(f"  actual high duration min:  {min(durations):.3f} ms")
    print(f"  actual high duration mean: {statistics.fmean(durations):.3f} ms")
    print(f"  actual high duration max:  {max(durations):.3f} ms")
    if len(durations) >= 2:
        print(f"  actual high duration stdev: {statistics.stdev(durations):.3f} ms")


def verify_full_step_before_sweep(
    dac: VcxoDac,
    old_code: int,
    new_code: int,
    old_dwell_s: float,
) -> None:
    print()
    print("=" * 72)
    print("Initial full-step verification")
    print(f"The DAC will now hold the new code: {new_code}")
    print("Watch the frequency counter and verify the full frequency change.")
    print("Press Enter when you are ready to return to the old code and start testing.")
    print("=" * 72)
    print()

    dac.write_dac(new_code)
    input("Holding new code. Press Enter to return to old code and continue...")
    dac.write_dac(old_code)
    time.sleep(old_dwell_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pulse-width settling test for a DAC-controlled VCXO."
    )
    parser.add_argument("--i2c-bus", type=int, default=1)
    parser.add_argument(
        "--mcp4725-addr",
        type=lambda s: int(s, 0),
        default=MCP4725_DEFAULT_ADDR,
        help="MCP4725 I2C address. Decimal or hex, e.g. 0x62.",
    )
    parser.add_argument("--clock-apr-ppm", type=int, default=120)
    parser.add_argument(
        "--skip-clock-init",
        action="store_true",
        help="Do not initialize the Si5351/VCXO path before pulsing the DAC.",
    )
    parser.add_argument(
        "--readback",
        action="store_true",
        help="Read MCP4725 after each write. Useful for debugging, but it changes pulse timing.",
    )
    parser.add_argument("--old-code", type=int, default=2050)
    parser.add_argument("--new-code", type=int, default=2055)
    parser.add_argument("--old-dwell-s", type=float, default=1.0)
    parser.add_argument(
        "--widths-ms",
        type=parse_widths_ms,
        default=parse_widths_ms("10,20,30,50,75,100,150,200"),
        help="Comma-separated pulse widths in milliseconds.",
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--pause-between", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.old_dwell_s < 0:
        print("Error: --old-dwell-s must be non-negative.", file=sys.stderr)
        return 2
    if args.repeats <= 0:
        print("Error: --repeats must be positive.", file=sys.stderr)
        return 2
    for label, code in (("old", args.old_code), ("new", args.new_code)):
        if not 0 <= int(code) <= 4095:
            print(f"Error: --{label}-code must be in MCP4725 range 0..4095.", file=sys.stderr)
            return 2

    dac = None

    try:
        dac = VcxoDac(
            i2c_bus=args.i2c_bus,
            mcp4725_addr=args.mcp4725_addr,
            clock_apr_ppm=args.clock_apr_ppm,
            skip_clock_init=args.skip_clock_init,
            readback=args.readback,
        )
        dac.open()

        print("Setting DAC to old code for safe startup.")
        dac.write_dac(args.old_code)
        time.sleep(args.old_dwell_s)

        print()
        print("VCXO pulse-width settling test")
        print(f"I2C bus:      {args.i2c_bus}")
        print(f"MCP4725 addr: 0x{args.mcp4725_addr:02x}")
        print(f"Clock init:   {not args.skip_clock_init}")
        print(f"Readback:     {args.readback}")
        print(f"Old code:     {args.old_code}")
        print(f"New code:     {args.new_code}")
        print(f"Old dwell:    {args.old_dwell_s:.3f} s")
        print(f"Pulse widths: {', '.join(str(w) for w in args.widths_ms)} ms")
        print(f"Repeats:      {args.repeats}")
        print()
        print("Press Ctrl+C at any time to return DAC to old code and exit.")

        verify_full_step_before_sweep(
            dac=dac,
            old_code=args.old_code,
            new_code=args.new_code,
            old_dwell_s=args.old_dwell_s,
        )

        for width_ms in args.widths_ms:
            print_group_header(width_ms, args.old_code, args.new_code, args.repeats)

            if args.pause_between:
                input("Press Enter to start this pulse-width group...")

            durations = run_pulse_group(
                dac=dac,
                old_code=args.old_code,
                new_code=args.new_code,
                old_dwell_s=args.old_dwell_s,
                width_ms=width_ms,
                repeats=args.repeats,
            )
            summarize_group(width_ms, durations)

            print()
            print("Group complete. DAC returned to old code.")
            dac.write_dac(args.old_code)

            if args.pause_between:
                input("Press Enter to continue to the next pulse width...")

        print()
        print("All pulse-width groups complete. Returning DAC to old code.")
        dac.write_dac(args.old_code)
        return 0

    except KeyboardInterrupt:
        print()
        print("Ctrl+C received. Returning DAC to old code before exit.")
        if dac is not None:
            try:
                dac.write_dac(args.old_code)
            except Exception as exc:
                print(f"Warning: failed to restore old DAC code: {exc}", file=sys.stderr)
        return 130

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Attempting to return DAC to old code before exit.", file=sys.stderr)
        if dac is not None:
            try:
                dac.write_dac(args.old_code)
            except Exception as restore_exc:
                print(
                    f"Warning: failed to restore old DAC code: {restore_exc}",
                    file=sys.stderr,
                )
        return 1

    finally:
        if dac is not None:
            try:
                dac.write_dac(args.old_code)
            except Exception:
                pass
            dac.close()


if __name__ == "__main__":
    raise SystemExit(main())
