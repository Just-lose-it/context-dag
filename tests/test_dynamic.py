import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_dag import Action, DynamicDAG, JSONAgent


class DynamicTests(unittest.TestCase):
    def test_spawn_read_yield_finalizer(self):
        dag = DynamicDAG("root task")
        def decide(ctx):
            if ctx.node_id == "root":
                return Action("spawn", children=(("a", "task a"), ("b", "task b")))
            if ctx.observation is None:
                return Action("read_dag", verbose=False)
            self.assertEqual(len(ctx.observation["nodes"]), 3)
            self.assertNotIn("prompt", ctx.observation["nodes"][0])
            return Action("yield", result=ctx.prompt)
        result = dag.run(decide, lambda snapshot, store: "+".join(
            n["result"] for n in snapshot["nodes"] if n["status"] == "yielded"))
        self.assertEqual(result.answer, "task a+task b")
        self.assertTrue(result.finalized)
        self.assertEqual(result.nodes["a"].parent, "root")

    def test_conclude_skips_finalizer(self):
        dag = DynamicDAG("answer")
        result = dag.run(lambda ctx: Action("conclude", result="done"),
                         lambda snapshot, store: self.fail("finalizer called"))
        self.assertEqual(result.answer, "done")
        self.assertFalse(result.finalized)

    def test_duplicate_spawn_is_rejected(self):
        dag = DynamicDAG("task")
        with self.assertRaises(ValueError):
            dag.run(lambda ctx: Action("spawn", children=(("a", "one"), ("a", "two"))),
                    lambda snapshot, store: "")

    def test_json_agent_keeps_context_histories(self):
        calls = []
        def complete(messages):
            calls.append(list(messages))
            return '{"action":"yield","result":"ok"}'
        result = DynamicDAG("task").run(JSONAgent(complete), lambda snapshot, store: "final")
        self.assertEqual(result.answer, "final")
        self.assertEqual(calls[0][-1]["content"], "task")

    def test_yield_resumes_after_later_stage(self):
        dag = DynamicDAG("root")
        calls = []
        def decide(ctx):
            calls.append((ctx.node_id, ctx.stage))
            if ctx.node_id == "root":
                return Action("spawn", children=(("a", "wait"), ("b", "spawn")))
            if ctx.node_id == "a":
                return Action("conclude", result="resumed") if ctx.observation else Action("yield")
            if ctx.node_id == "b":
                return Action("spawn", children=(("c", "child"),))
            return Action("yield", result="child done")
        result = dag.run(decide, lambda snapshot, store: "final")
        self.assertEqual(result.answer, "resumed")
        self.assertIn(("a", 2), calls)

    def test_yield_without_later_stage_finalizes(self):
        result = DynamicDAG("root").run(lambda ctx: Action("yield", result="partial"),
                                         lambda snapshot, store: "final")
        self.assertEqual(result.answer, "final")


if __name__ == "__main__":
    unittest.main()
