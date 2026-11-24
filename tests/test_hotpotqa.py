import importlib.util
import sys
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "benchmarks" / "hotpotqa" / "run.py"
spec = importlib.util.spec_from_file_location("hotpot_run", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ScoreTests(unittest.TestCase):
    def test_exact_match_and_support_set(self):
        gold = {"answer": "yes", "supporting_facts": [["A", 0], ["B", 1]]}
        self.assertEqual(module.score({"answer": "Yes", "sp": [["B", 1], ["A", 0]]}, gold),
                         {"answer_em": 1, "support_em": 1, "joint_em": 1})
        self.assertEqual(module.score({"answer": "Yes", "sp": [["A #0", 0], ["B", 1]]}, gold)["support_em"], 0)


if __name__ == "__main__":
    unittest.main()
