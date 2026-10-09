import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from raw_features import WINDOW_SAMPLES, read_window
from raw_model import ROOT, RawClassifier, digest


class RawModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = RawClassifier()

    def test_zero_and_dc_return_unknown(self):
        for value in (0, 1 + 2j):
            result = self.classifier.classify_samples(np.full(WINDOW_SAMPLES, value, dtype=np.complex64))
            self.assertEqual(result["label"], "UNKNOWN")
            self.assertFalse(result["accepted"])

    def test_invalid_input_is_rejected(self):
        for signal in (np.zeros(10, dtype=np.complex64), np.full(WINDOW_SAMPLES, np.nan + 1j, dtype=np.complex64)):
            with self.assertRaises(ValueError):
                self.classifier.classify_samples(signal)

    def test_included_recordings(self):
        for case in json.loads((ROOT / "examples.json").read_text()):
            path = ROOT / case["file"]
            self.assertEqual(digest(path), case["sha256"])
            self.assertEqual(self.classifier.classify_file(path)["label"], case["expected"])

    def test_int16_reader(self):
        signal = read_window(ROOT / "signal_01.iq")
        scale = 0.9 / max(np.abs(signal.real).max(), np.abs(signal.imag).max())
        components = np.column_stack((signal.real, signal.imag)) * scale * 32767
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.iq"
            path.write_bytes(components.astype("<i2").tobytes())
            self.assertEqual(self.classifier.classify_file(path, fmt="ci16")["label"], "OCU2")

    def test_other_and_low_margin_are_rejected(self):
        for scores in ([0, 0, 0, 1], [0.4, 0.3, 0.2, 0.1], [0.65, 0.55, 0, 0]):
            self.assertFalse(self.classifier.decide(scores)["accepted"])


if __name__ == "__main__":
    unittest.main()
