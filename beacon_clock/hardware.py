"""I2C is opened only on explicit hardware operations; no EEPROM writes."""
import math


class Dac:
    def __init__(self, config):
        from smbus2 import SMBus
        self.c = config
        self.bus = SMBus(config.i2c_bus)
        self.code = None

    def read(self):
        from smbus2 import i2c_msg
        message = i2c_msg.read(self.c.dac_address, 5)
        self.bus.i2c_rdwr(message)
        b = list(message)
        if (b[0] >> 1) & 3:
            raise RuntimeError("DAC is in power-down mode")
        self.code = ((b[1] << 4) | (b[2] >> 4)) & 4095
        return self.code

    def write(self, code):
        from smbus2 import i2c_msg
        if not isinstance(code, int) or not self.c.dac_min <= code <= self.c.dac_max:
            raise ValueError("DAC command outside configured limits")
        if self.code is not None and abs(code - self.code) > self.c.max_step_codes:
            raise ValueError("DAC command exceeds slew limit")
        if code != self.code:
            # MCP4725 fast-mode command: PD=00, volatile DAC register only.
            self.bus.i2c_rdwr(i2c_msg.write(self.c.dac_address, [(code >> 8) & 15, code & 255]))
            self.code = code

    def inspect_clock(self):
        def rd(reg):
            return self.bus.read_byte_data(0x60, reg)
        def ratio(base):
            b = [rd(base + i) for i in range(8)]
            p1 = ((b[2] & 3) << 16) | b[3] << 8 | b[4]
            p2 = ((b[5] & 15) << 16) | b[6] << 8 | b[7]
            p3 = ((b[5] & 240) << 12) | b[0] << 8 | b[1]
            return (p1 + 512 + p2 / p3) / 128 if p3 else math.nan, p3
        pll, denominator = ratio(34)
        divider, _ = ratio(42)
        register16 = rd(16)
        pull = rd(162) | rd(163) << 8 | (rd(164) & 63) << 16
        status = rd(0)
        output_hz = self.c.clock_ref_hz * pll / divider
        expected_pull = round(1.03 * (128 * int(pll) + pll - int(pll)) * self.c.apr_ppm)
        # R divider and divide-by-four must be off for the existing frequency plan.
        compatible = (status & 0xC0 == 0 and register16 & 0xBC == 0x2C
                      and rd(3) & 1 == 0 and rd(44) & 0x7C == 0
                      and denominator == 1_000_000 and pull == expected_pull
                      and abs(output_hz - 28_800_000) < .01)
        return dict(status=status, register16=register16, nominal_output_hz=output_hz,
                    vcxo_parameter=pull, pll_denominator=denominator, compatible=compatible,
                    reference_assumed_hz=self.c.clock_ref_hz)

    def initialize_clock(self):
        from setClockFreq import configure_clk0_vcxo, wait_sys_init_clear
        wait_sys_init_clear(self.bus)
        configure_clk0_vcxo(self.bus, 28_800_000, self.c.clock_ref_hz, self.c.apr_ppm)

    def close(self):
        self.bus.close()
