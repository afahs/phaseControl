"""Set the DAC output code. Run while other DAC writers are stopped."""
import argparse
import sys

from beacon_clock.config import Config
from beacon_clock.hardware import Dac


def main(argv=None):
    config = Config(dac_min=0, dac_max=4095, max_step_codes=4095)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", type=int, help="desired 12-bit DAC code (0–4095), applied directly")
    args = parser.parse_args(argv)
    if not config.dac_min <= args.code <= config.dac_max:
        parser.error(f"code must be between {config.dac_min} and {config.dac_max}")

    try:
        dac = Dac(config)
        try:
            dac.write(args.code)
            current = dac.read()
            if current != args.code:
                raise RuntimeError(f"DAC readback {current} differs from requested {args.code}")
            print(f"DAC code {current} verified")
        finally:
            dac.close()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted during DAC adjustment", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
