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
                if ctx.observation and "completed_children" in ctx.observation:
                    self.assertEqual(len(ctx.observation["completed_children"]), 2)
                    return Action("yield")
                return Action("spawn", children=(("a", "task a"), ("b", "task b")))
            if ctx.observation is None:
                return Action("read_dag", verbose=False)
            self.assertEqual(len(ctx.observation["nodes"]), 3)
            self.assertNotIn("prompt", ctx.observation["nodes"][0])
            return Action("yield", result=ctx.prompt)
        result = dag.run(decide, lambda snapshot, store: "+".join(
            n["result"] for n in snapshot["nodes"] if n["status"] == "yielded" and n["result"]))
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
                if ctx.observation:
                    return Action("yield")
                return Action("spawn", children=(("a", "wait"), ("b", "spawn")))
            if ctx.node_id == "a":
                return Action("conclude", result="resumed") if ctx.observation else Action("yield")
            if ctx.node_id == "b":
                if ctx.observation:
                    return Action("yield")
                return Action("spawn", children=(("c", "child"),))
            return Action("yield", result="child done")
        result = dag.run(decide, lambda snapshot, store: "final")
        self.assertEqual(result.answer, "resumed")
        self.assertIn(("a", 2), calls)

    def test_yield_without_later_stage_finalizes(self):
        result = DynamicDAG("root").run(lambda ctx: Action("yield", result="partial"),
                                         lambda snapshot, store: "final")
        self.assertEqual(result.answer, "final")

    def test_parent_receives_short_child_result(self):
        dag = DynamicDAG("parent")
        def decide(ctx):
            if ctx.node_id == "root":
                if ctx.observation:
                    self.assertEqual(ctx.observation["completed_children"][0]["result"], "done")
                    return Action("conclude", result="received")
                return Action("spawn", children=(("child", "work"),))
            return Action("yield", result="done")
        result = dag.run(decide, lambda snapshot, store: "unused")
        self.assertEqual(result.answer, "received")
        self.assertIn("handoff", [event["action"] for event in result.events])

    def test_parent_reads_long_child_result(self):
        dag = DynamicDAG("parent")
        def decide(ctx):
            if ctx.node_id == "root":
                if ctx.observation is None:
                    return Action("spawn", children=(("child", "work"),))
                if "completed_children" in ctx.observation:
                    notice = ctx.observation["completed_children"][0]
                    self.assertEqual(notice["mode"], "reference")
                    self.assertNotIn("result", notice)
                    return Action("read_result", key=notice["store_key"])
                self.assertEqual(ctx.observation["result"], "long result")
                return Action("conclude", result="received")
            return Action("yield", result="long result")
        result = dag.run(decide, lambda snapshot, store: "unused", handoff_limit=4)
        self.assertEqual(result.answer, "received")
        self.assertIn("read_result", [event["action"] for event in result.events])

    def test_nested_child_result_reaches_root(self):
        dag = DynamicDAG("root")
        def decide(ctx):
            if ctx.node_id == "root":
                if ctx.observation:
                    self.assertEqual(ctx.observation["completed_children"][0]["result"], "leaf done")
                    return Action("conclude", result="complete")
                return Action("spawn", children=(("middle", "middle task"),))
            if ctx.node_id == "middle":
                if ctx.observation:
                    return Action("yield", result=ctx.observation["completed_children"][0]["result"])
                return Action("spawn", children=(("leaf", "leaf task"),))
            return Action("yield", result="leaf done")
        self.assertEqual(dag.run(decide, lambda snapshot, store: "unused").answer, "complete")

    def test_verbose_dag_does_not_expose_long_child_result(self):
        dag = DynamicDAG("root")
        def decide(ctx):
            if ctx.node_id == "root":
                if ctx.observation and "completed_children" in ctx.observation:
                    return Action("read_dag", verbose=True)
                if ctx.observation and "nodes" in ctx.observation:
                    child = next(n for n in ctx.observation["nodes"] if n["id"] == "child")
                    self.assertNotIn("result", child)
                    return Action("conclude", result="safe")
                return Action("spawn", children=(("child", "work"),))
            return Action("yield", result="long result")
        result = dag.run(decide, lambda snapshot, store: "unused", handoff_limit=4)
        self.assertEqual(result.answer, "safe")

    def test_read_docs_observation(self):
        dag = DynamicDAG("find evidence")
        def decide(ctx):
            if ctx.observation is None:
                return Action("read_docs", documents=("A",))
            self.assertEqual(ctx.observation["A"], "text")
            return Action("conclude", result="done")
        result = dag.run(decide, lambda snapshot, store: "unused",
                         read_documents=lambda ids, node: {"A": "text"})
        self.assertEqual(result.events[0]["action"], "read_docs")


if __name__ == "__main__":
    unittest.main()
