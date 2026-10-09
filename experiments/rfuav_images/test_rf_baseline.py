import unittest

from rf_baseline import CLASSES, LINK_FAMILY, select_samples


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.entries = [{"type": "file", "path": f"capture/{index}.jpg"} for index in range(30)]
        self.entries += [{"type": "directory", "path": "capture/other"}, {"type": "file", "path": "capture/notes.txt"}]

    def test_selection_is_repeatable(self):
        self.assertEqual(select_samples(self.entries, 10, 42), select_samples(self.entries, 10, 42))

    def test_selection_is_unique(self):
        self.assertEqual(len({entry["path"] for entry in select_samples(self.entries, 10, 42)}), 10)

    def test_order_does_not_change_selection(self):
        self.assertEqual(select_samples(self.entries, 10, 42), select_samples(list(reversed(self.entries)), 10, 42))

    def test_short_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            select_samples(self.entries, 31, 42)

    def test_families_are_not_five_distinct_generations(self):
        self.assertEqual(len(CLASSES), 5)
        self.assertEqual(len(set(LINK_FAMILY)), 4)
        self.assertEqual(LINK_FAMILY[0], LINK_FAMILY[4])


if __name__ == "__main__":
    unittest.main()
