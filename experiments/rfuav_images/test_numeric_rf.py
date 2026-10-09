import unittest

import numpy as np

from capture_io import CaptureError
from numeric_rf import RECIPE, representation


class NumericRFTests(unittest.TestCase):
    def test_scale_invariance(self):
        signal = np.random.default_rng(13).standard_normal(100000).astype(np.complex64)
        np.testing.assert_allclose(representation(signal, 100e6), representation(signal * 3, 100e6), atol=1e-6)

    def test_zero_and_shape_are_safe(self):
        result = representation(np.zeros(1024, dtype=np.complex64), 100e6)
        self.assertEqual(result.shape, (128, 128))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result == 0).all())
        self.assertFalse(RECIPE["legacy_image_checkpoint_compatible"])

    def test_frequency_axis_is_ordered(self):
        time = np.arange(100000) / 100e6
        result = representation(np.exp(2j * np.pi * 20e6 * time).astype(np.complex64), 100e6)
        strongest_bin = int(result.mean(1).argmax())
        self.assertLess(abs(strongest_bin - int(128 * 0.70)), 2)

    def test_nonfinite_and_short_input_are_rejected(self):
        for signal in (np.zeros(100, dtype=np.complex64), np.full(1024, complex(float("nan"), 0))):
            with self.assertRaises(CaptureError):
                representation(signal, 100e6)


if __name__ == "__main__":
    unittest.main()
