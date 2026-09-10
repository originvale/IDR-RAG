import unittest
from pathlib import Path

from idr_rag.allocation import hash_tie, precedence_ok
from idr_rag.features import FEATURE_NAMES
from idr_rag.io import group_actions, read_csv
from idr_rag.paper import policy_orders


ROOT = Path(__file__).resolve().parents[1]


class CoreInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_csv(ROOT / "artifacts" / "evaluation_actions_32d.csv")

    def test_state_dimension(self):
        self.assertEqual(len(FEATURE_NAMES), 32)
        self.assertEqual(len(self.rows), 3600)

    def test_prefix_legality(self):
        for order in policy_orders(self.rows).values():
            self.assertTrue(precedence_ok(order))

    def test_exact_quota_pool(self):
        self.assertEqual(len(group_actions(self.rows)), 1800)
        for order in policy_orders(self.rows).values():
            self.assertEqual(len(order), 3600)

    def test_dpa_hash_is_deterministic(self):
        action = self.rows[0]
        self.assertEqual(hash_tie(action), hash_tie(dict(action)))


if __name__ == "__main__":
    unittest.main()
