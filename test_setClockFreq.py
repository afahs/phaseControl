"""Register-level checks; no I2C device is opened or modified."""
import unittest

from setClockFreq import configure_clk0_vcxo


class RegisterBus:
    def __init__(self):
        self.registers = {}
        self.writes = []

    def write_byte_data(self, address, register, value):
        self.registers[register] = value
        self.writes.append((register, value))

    def read_byte_data(self, address, register):
        return self.registers.get(register, 0)


def decode_parameters(registers, base):
    data = [registers[base + i] for i in range(8)]
    p1 = ((data[2] & 3) << 16) | (data[3] << 8) | data[4]
    p2 = ((data[5] & 15) << 16) | (data[6] << 8) | data[7]
    p3 = ((data[5] >> 4) << 16) | (data[0] << 8) | data[1]
    ratio = (p1 + 512 + p2 / p3) / 128
    return ratio, p3


class FrequencyTests(unittest.TestCase):
    def test_scope_frequencies_encode_correctly(self):
        for frequency in (7_200_000, 28_800_000):
            with self.subTest(frequency=frequency):
                bus = RegisterBus()
                configure_clk0_vcxo(bus, frequency, 27_000_000, 120)
                pll_ratio, denominator = decode_parameters(bus.registers, 34)
                output_ratio, _ = decode_parameters(bus.registers, 42)
                self.assertEqual(denominator, 1_000_000)
                self.assertEqual(pll_ratio, 32)
                self.assertAlmostEqual(27_000_000 * pll_ratio / output_ratio, frequency)
                self.assertEqual(bus.registers[16], 0x6F)
                self.assertEqual(bus.writes[0], (3, 0xFF))
                self.assertEqual(bus.writes[-1], (3, 0xFE))
                vcxo = sum(bus.registers[162 + i] << (8 * i) for i in range(3))
                self.assertEqual(vcxo, 506266)

    def test_odd_integer_divider_uses_fractional_mode(self):
        bus = RegisterBus()
        configure_clk0_vcxo(bus, 864_000_000 / 31, 27_000_000, 120)
        self.assertEqual(bus.registers[16], 0x2F)

    def test_bad_requests_do_not_disable_outputs(self):
        for frequency, reference, apr in (
            (0, 27e6, 120), (float("nan"), 27e6, 120),
            (200e6, 27e6, 120), (28.8e6, 0, 120), (28.8e6, 27e6, 0),
        ):
            with self.subTest(frequency=frequency, reference=reference, apr=apr):
                bus = RegisterBus()
                with self.assertRaises(ValueError):
                    configure_clk0_vcxo(bus, frequency, reference, apr)
                self.assertEqual(bus.writes, [])


if __name__ == "__main__":
    unittest.main()
