import unittest

from detector_model import decision


class DecisionTests(unittest.TestCase):
    def test_known_candidate_below_threshold_abstains(self):
        result = decision([0.89, 0.02, 0.02, 0.02, 0.02, 0.03], 0.90)
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["associated_link_family_lookup"])

    def test_other_is_never_emitted_as_dji(self):
        result = decision([0, 0, 0, 0, 0, 1], 0.90)
        self.assertFalse(result["accepted"])

    def test_accepted_label_has_only_a_specification_lookup(self):
        result = decision([1, 0, 0, 0, 0, 0], 0.90)
        self.assertEqual(result["label"], "DJI AVATA2")
        self.assertEqual(result["associated_link_family_lookup"], "O4")
        self.assertFalse(result["direct_protocol_version_detection"])

    def test_nonfinite_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            decision([float("nan"), 0, 0, 0, 0, 1], 0.90)


if __name__ == "__main__":
    unittest.main()
