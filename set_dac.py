"""Set the DAC output code. Run while other DAC writers are stopped."""
import argparse
import sys
import time

from beacon_clock.config import Config
from beacon_clock.hardware import Dac


def main(argv=None):
    config = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", type=int, help="desired DAC code (410–3685)")
    args = parser.parse_args(argv)
    if not config.dac_min <= args.code <= config.dac_max:
        parser.error(f"code must be between {config.dac_min} and {config.dac_max}")

    try:
        dac = Dac(config)
        try:
            current = dac.read()
            if not config.dac_min <= current <= config.dac_max:
                raise ValueError("starting DAC code is outside configured limits")
            while current != args.code:
                change = max(-config.max_step_codes,
                             min(config.max_step_codes, args.code - current))
                requested = current + change
                dac.write(requested)
                current = dac.read()
                if current != requested:
                    raise RuntimeError(f"DAC readback {current} differs from requested {requested}")
                if current != args.code:
                    time.sleep(1 / config.control_hz)
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
