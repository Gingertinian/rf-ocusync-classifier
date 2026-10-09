import unittest

import numpy as np

from capture_io import CaptureError
from perturb_capture import transform


class PerturbationTests(unittest.TestCase):
    def test_noise_is_deterministic_and_uses_capture_power(self):
        samples = np.ones(100000, dtype=np.complex64)
        first, info = transform(samples, 50e6, noise_ratio_db=20)
        second, _ = transform(samples, 50e6, noise_ratio_db=20)
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(info["measured_original_capture_to_added_noise_power_db"], 20, delta=0.05)
        self.assertFalse(info["true_signal_to_noise_ratio_known"])

    def test_frequency_offset_preserves_power(self):
        samples = np.ones(1000, dtype=np.complex64)
        shifted, _ = transform(samples, 10000, frequency_offset_hz=1000)
        np.testing.assert_allclose(np.abs(shifted), 1, atol=1e-6)
        self.assertAlmostEqual(float(np.angle(shifted[1] / shifted[0])), 2 * np.pi / 10, places=6)

    def test_echo_is_one_nonrecursive_delayed_copy(self):
        samples = np.zeros(10, dtype=np.complex64)
        samples[0] = 1
        output, _ = transform(samples, 1000, echo_gain=0.5, echo_delay_samples=2)
        self.assertEqual(output[2], 0.5)
        self.assertEqual(output[4], 0)

    def test_invalid_samples_and_parameters_fail(self):
        for options in ({"noise_ratio_db": float("nan")}, {"frequency_offset_hz": 500},
                        {"iq_gain_ratio": 0}, {"echo_gain": 0.5, "echo_delay_samples": 0}):
            with self.assertRaises(CaptureError):
                transform(np.ones(10, dtype=np.complex64), 1000, **options)


if __name__ == "__main__":
    unittest.main()
