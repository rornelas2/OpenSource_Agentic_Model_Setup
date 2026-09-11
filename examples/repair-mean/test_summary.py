import unittest

from summary import mean_readings


class MeanReadingsTests(unittest.TestCase):
    def test_regular_readings(self):
        self.assertEqual(mean_readings([2, 4, 6]), 4)

    def test_missing_readings(self):
        self.assertEqual(mean_readings([2, None, 6]), 4)

    def test_zero_is_a_reading(self):
        self.assertEqual(mean_readings([0, None, 6]), 3)

    def test_no_readings(self):
        for values in ([], [None, None]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    mean_readings(values)


if __name__ == '__main__':
    unittest.main()
