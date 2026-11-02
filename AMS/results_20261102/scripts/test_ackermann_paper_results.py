import unittest

from build_ackermann_paper_results import exact_sign_flip, holm_adjust


class StatisticalTests(unittest.TestCase):
    def test_three_positive_pairs_have_only_two_extreme_signs(self):
        observed, p, distribution = exact_sign_flip([1, 2, 4])
        self.assertAlmostEqual(observed, 7/3)
        self.assertEqual(len(distribution), 8)
        self.assertEqual(sum(row['extreme'] for row in distribution), 2)
        self.assertEqual(p, .25)
        self.assertEqual(exact_sign_flip([-1, -2, -4])[1], .25)

    def test_ties_and_zero_statistic_are_not_artificially_significant(self):
        self.assertEqual(exact_sign_flip([0, 0, 0])[1], 1)
        self.assertEqual(exact_sign_flip([1, 2, -3])[1], 1)
        self.assertEqual(exact_sign_flip([0, 1, 2])[1], .5)

    def test_invalid_inputs_fail(self):
        for values in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):exact_sign_flip(values)
        with self.assertRaises(ValueError):holm_adjust([1.1])

    def test_holm_is_monotone_and_preserves_original_order(self):
        actual = holm_adjust([.01, .04, .03])
        for a,b in zip(actual,[.03,.06,.06]):self.assertAlmostEqual(a,b)
        self.assertEqual(holm_adjust([.25]*5),[1.]*5)


if __name__ == '__main__':
    unittest.main()
