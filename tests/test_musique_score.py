import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.musique.run import score


class MuSiQueScoreTest(unittest.TestCase):
    def test_unique_bare_title_maps_to_paragraph(self):
        item = {"answer": "Paris", "answer_aliases": [], "paragraphs": [
            {"idx": 0, "title": "France", "is_supporting": True},
            {"idx": 1, "title": "Elsewhere", "is_supporting": False},
        ]}
        self.assertEqual(score({"answer": "Paris", "sp": [["France", 0]]}, item)["joint_em"], 1)
        self.assertEqual(score({"answer": "Paris", "sp": [["P0 France", 0]]}, item)["joint_em"], 1)

    def test_ambiguous_title_gets_no_support_credit(self):
        item = {"answer": "Paris", "answer_aliases": [], "paragraphs": [
            {"idx": 0, "title": "France", "is_supporting": True},
            {"idx": 1, "title": "France", "is_supporting": False},
        ]}
        self.assertEqual(score({"answer": "Paris", "sp": [["France", 0]]}, item)["support_em"], 0)


if __name__ == "__main__":
    unittest.main()
