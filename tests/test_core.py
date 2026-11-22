import os
import statistics
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_overfitting.core import (
    N_KNOBS, dev_score, heldout_score, hill_climb, run_experiment,
    thresholdout_climb, true_quality,
)

ZERO = tuple([0] * N_KNOBS)
FULL_REAL = tuple(1 if i in (0, 1, 2) else 0 for i in range(N_KNOBS))


class TestTruth(unittest.TestCase):
    def test_true_quality_only_real_knobs(self):
        self.assertEqual(true_quality(ZERO), 0.62)
        self.assertAlmostEqual(true_quality(FULL_REAL), 0.80)
        junk = tuple(1 if i >= 3 else 0 for i in range(N_KNOBS))
        self.assertEqual(true_quality(junk), 0.62)

    def test_dev_score_deterministic_per_config(self):
        self.assertEqual(dev_score(3, ZERO, 25), dev_score(3, ZERO, 25))
        self.assertNotEqual(dev_score(3, ZERO, 25), dev_score(4, ZERO, 25))

    def test_dev_noise_shrinks_with_n(self):
        for n, bound in ((25, 0.30), (400, 0.08)):
            diffs = [abs(dev_score(s, ZERO, n) - 0.62) for s in range(30)]
            self.assertLess(statistics.mean(diffs), bound)

    def test_heldout_close_to_true(self):
        self.assertLess(abs(heldout_score(1, FULL_REAL) - 0.80), 0.05)


class TestClimb(unittest.TestCase):
    def test_hill_climb_counts_evaluations(self):
        r = hill_climb(0, 25)
        self.assertGreaterEqual(r.true, 0.62)
        self.assertEqual(r.evaluations, 1 + 3 * N_KNOBS)

    def test_gap_shrinks_as_dev_set_grows(self):
        rep = run_experiment()
        self.assertGreater(rep["dev_sizes"]["25"]["mean_gap"],
                           rep["dev_sizes"]["400"]["mean_gap"])
        self.assertGreater(rep["dev_sizes"]["400"]["mean_gap"], -0.02)

    def test_thresholdout_reduces_gap(self):
        t = run_experiment()["thresholdout"]
        self.assertLess(t["thresholdout_mean_gap"], t["plain_mean_gap"])
        self.assertGreater(t["mean_queries_used"], 0)
        self.assertLessEqual(t["mean_queries_used"], t["query_budget"])

    def test_thresholdout_budget_exhaustion_flag(self):
        m = thresholdout_climb(0, 25, query_budget=5)
        self.assertEqual(m.queries_used, 5)
        self.assertTrue(m.budget_exhausted)


if __name__ == "__main__":
    unittest.main()
