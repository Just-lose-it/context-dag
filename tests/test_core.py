import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_dag import Artifact, ContextDAG, Node, RoutingDecision, SharedStore


class ContextDAGTests(unittest.TestCase):
    def test_fan_out_and_join(self):
        graph = ContextDAG()
        graph.add_node(Node("a", lambda ctx: Artifact("evidence", "fact", "a")))
        graph.add_node(Node("b", lambda ctx: Artifact("b", ctx.inbox[0].value + " B", "b")))
        graph.add_node(Node("c", lambda ctx: Artifact("c", ctx.inbox[0].value + " C", "c")))
        graph.add_node(Node("d", lambda ctx: sorted(x.value for x in ctx.inbox)))
        for source, target in (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")):
            graph.add_edge(source, target)
        self.assertEqual(graph.execute().outputs["d"], ["fact B", "fact C"])

    def test_store_scope_and_version(self):
        store = SharedStore()
        store.write(Artifact("k", "old", "a", readers=frozenset({"b"})))
        self.assertIsNone(store.read("k", "c"))
        store.write(Artifact("k", "new", "a", version=2, readers=frozenset({"b"})))
        self.assertEqual(store.read("k", "b").value, "new")
        with self.assertRaises(ValueError):
            store.write(Artifact("k", "stale", "a", version=1))

    def test_store_route_requires_explicit_read(self):
        graph = ContextDAG()
        graph.add_node(Node("a", lambda ctx: Artifact("k", 42, "a", readers=frozenset({"b"}))))
        graph.add_node(Node("b", lambda ctx: (len(ctx.inbox), ctx.read("k").value)))
        graph.add_edge("a", "b")
        result = graph.execute(lambda artifact, successors: "store")
        self.assertEqual(result.outputs["b"], (0, 42))

    def test_direct_route_to_selected_successor(self):
        graph = ContextDAG()
        graph.add_node(Node("a", lambda ctx: Artifact("k", "content", "a")))
        graph.add_node(Node("b", lambda ctx: len(ctx.inbox)))
        graph.add_node(Node("c", lambda ctx: len(ctx.inbox)))
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        result = graph.execute(lambda artifact, successors:
                               RoutingDecision("push", frozenset({"b"})))
        self.assertEqual((result.outputs["b"], result.outputs["c"]), (1, 0))
        self.assertEqual(result.events[-1]["delivered_to"], ["b"])

    def test_direct_route_rejects_non_successor(self):
        graph = ContextDAG()
        graph.add_node(Node("a", lambda ctx: Artifact("k", "content", "a")))
        graph.add_node(Node("b", lambda ctx: None))
        with self.assertRaises(ValueError):
            graph.execute(lambda artifact, successors:
                          RoutingDecision("push", frozenset({"b"})))

    def test_cycle_rejected_without_mutating_graph(self):
        graph = ContextDAG()
        graph.add_node(Node("a", lambda ctx: None))
        graph.add_node(Node("b", lambda ctx: None))
        graph.add_edge("a", "b")
        with self.assertRaises(ValueError):
            graph.add_edge("b", "a")
        self.assertEqual(graph.order(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
