import unittest

from os_computer_use.grounding import coerce_point


class CoercePointTest(unittest.TestCase):
    def test_integer_object(self):
        self.assertEqual(coerce_point('{"x": 12, "y": 34}', 100, 80), (12, 34))

    def test_single_element_lists(self):
        self.assertEqual(coerce_point('{"x": [12], "y": [34]}', 100, 80), (12, 34))

    def test_bbox_midpoint(self):
        self.assertEqual(
            coerce_point('{"x": [0, 20], "y": [10, 30]}', 100, 80), (10, 20)
        )

    def test_unit_interval_is_scaled(self):
        self.assertEqual(coerce_point('{"x": 0.5, "y": 0.25}', 200, 100), (100, 25))

    def test_zero_to_thousand_is_scaled_when_outside_the_image(self):
        self.assertEqual(coerce_point('{"x": 500, "y": 250}', 100, 50), (50, 12))

    def test_point_array(self):
        self.assertEqual(coerce_point({"point": [8, 9]}, 100, 80), (8, 9))


if __name__ == "__main__":
    unittest.main()
