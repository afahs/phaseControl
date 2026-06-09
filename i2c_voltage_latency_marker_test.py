#!/usr/bin/env python3
"""One-shot GPIO-to-I2C timing test."""

import time

import RPi.GPIO as GPIO
from smbus2 import SMBus, i2c_msg


GPIO_PIN = 37  # Physical Raspberry Pi header pin 37.
I2C_BUS = 1
MCP4725_ADDR = 0x62
START_CODE = 2050
STEP_CODE = 2055


def write_mcp4725(bus: SMBus, code: int) -> None:
    code = max(0, min(4095, int(code)))
    bus.i2c_rdwr(
        i2c_msg.write(
            MCP4725_ADDR,
            [0x40, (code >> 4) & 0xFF, (code & 0x0F) << 4],
        )
    )


def main() -> int:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)

    try:
        with SMBus(I2C_BUS) as bus:
            write_mcp4725(bus, START_CODE)
            time.sleep(0.1)

            GPIO.output(GPIO_PIN, GPIO.HIGH)
            write_mcp4725(bus, STEP_CODE)
            GPIO.output(GPIO_PIN, GPIO.LOW)

        print(f"Done: GPIO {GPIO_PIN} rose, then I2C wrote DAC code {STEP_CODE}.")
    finally:
        GPIO.output(GPIO_PIN, GPIO.LOW)
        GPIO.cleanup(GPIO_PIN)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
