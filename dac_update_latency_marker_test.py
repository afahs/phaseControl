#!/usr/bin/env python3
"""One-shot GPIO-to-DAC-update timing test for the MCP4725."""

import time

import RPi.GPIO as GPIO
from smbus2 import SMBus, i2c_msg


GPIO_PIN = 37  # Physical Raspberry Pi header pin 37.
I2C_BUS = 1
MCP4725_ADDR = 0x62
START_CODE = 2050
STEP_CODE = 2155


def dac_write_message(code: int) -> i2c_msg:
    code = max(0, min(4095, int(code)))
    return i2c_msg.write(
        MCP4725_ADDR,
        [0x40, (code >> 4) & 0xFF, (code & 0x0F) << 4],
    )


def update_dac(bus: SMBus, code: int) -> None:
    bus.i2c_rdwr(dac_write_message(code))


def main() -> int:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)

    try:
        with SMBus(I2C_BUS) as bus:
            update_dac(bus, START_CODE)
            time.sleep(0.1)

            GPIO.output(GPIO_PIN, GPIO.HIGH)
            update_dac(bus, STEP_CODE)
            GPIO.output(GPIO_PIN, GPIO.LOW)

        print(f"Done: DAC updated from {START_CODE} to {STEP_CODE}.")
    finally:
        GPIO.output(GPIO_PIN, GPIO.LOW)
        GPIO.cleanup(GPIO_PIN)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
