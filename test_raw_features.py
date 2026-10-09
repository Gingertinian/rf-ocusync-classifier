import unittest

import numpy as np

from raw_features import CLASSES, WINDOW_SAMPLES, extract, intervals


class RawFeaturesTests(unittest.TestCase):
    def test_zero_input(self):
        self.assertTrue(np.array_equal(extract(np.zeros(WINDOW_SAMPLES, dtype=np.complex64)), np.zeros(516)))

    def test_scale_and_phase_invariance(self):
        rng = np.random.default_rng(42)
        signal = (rng.standard_normal(WINDOW_SAMPLES) + 1j * rng.standard_normal(WINDOW_SAMPLES)).astype(np.complex64)
        np.testing.assert_allclose(extract(signal), extract(signal * (2 + 3j)), atol=1e-4)

    def test_windows_are_disjoint_with_guard_gaps(self):
        split = intervals(26 * WINDOW_SAMPLES)
        sets = [set(split[name]) for name in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertGreater(min(sets[1]) - max(sets[0]), 1)
        self.assertGreater(min(sets[2]) - max(sets[1]), 1)
        self.assertEqual(CLASSES, ("OCU2", "OCU3", "OCU4", "OTHER"))


if __name__ == "__main__":
    unittest.main()
