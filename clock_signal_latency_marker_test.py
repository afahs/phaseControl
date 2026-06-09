#!/usr/bin/env python3
"""One-shot GPIO-to-clock-response timing test via the MCP4725 VCXO DAC."""

import time

import RPi.GPIO as GPIO

import clock_vcxo_helper as cvh


GPIO_PIN = 37  # Physical Raspberry Pi header pin 37.
I2C_BUS = 1
MCP4725_ADDR = 0x62
START_CODE = 2048
STEP_CODE = 3048
APR_PPM = 120


def main() -> int:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)

    clock = cvh.ClockVCXO(
        i2c_bus=I2C_BUS,
        mcp4725_addr=MCP4725_ADDR,
        i2c_marker_gpio_pin=None,
    )

    try:
        print("Initializing clock output...")
        clock.open()
        clock.init_clock_vcxo_28p8mhz(apr_ppm=APR_PPM)

        clock.set_dac_code(START_CODE)
        time.sleep(0.1)

        GPIO.output(GPIO_PIN, GPIO.HIGH)
        clock.set_dac_code(STEP_CODE)
        GPIO.output(GPIO_PIN, GPIO.LOW)

        print(f"Done: DAC stepped from {START_CODE} to {STEP_CODE}.")
    finally:
        GPIO.output(GPIO_PIN, GPIO.LOW)
        GPIO.cleanup(GPIO_PIN)
        clock.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
