import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.browsecomp.search_index import SearchIndex
from benchmarks.browsecomp.run import run_policy, score


class StubClient:
    def __init__(self):
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"action": "search_docs", "query": "orbit origin", "limit": 1})
        if self.calls == 2:
            return json.dumps({"action": "read_docs", "documents": ["c0"]})
        return json.dumps({"action": "conclude", "result": json.dumps(
            {"answer": "Mars", "evidence_docids": ["d1"]})})


class BrowseCompPilotTest(unittest.TestCase):
    def test_search_read_and_score(self):
        document = {"docid": "d1", "text": "The orbit origin in this example is Mars.", "url": ""}
        row = {"query": "What is the orbit origin?", "answer": "Mars",
               "evidence_docs": [document], "gold_docs": [document], "negative_docs": []}
        index = SearchIndex(row)
        prediction, result, calls = run_policy(row, index, StubClient(), "single")
        self.assertEqual(prediction["answer"], "Mars")
        self.assertEqual(calls["read_by_node"]["root"], ["d1"])
        self.assertEqual(score(prediction, row, calls)["evidence_recall"], 1)
        self.assertEqual([event["action"] for event in result.events[:2]],
                         ["search_docs", "read_docs"])


if __name__ == "__main__":
    unittest.main()
