import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_dag import Artifact, ContextDAG, Node

graph = ContextDAG()
graph.add_node(Node("research", lambda ctx: Artifact("fact", "A source-backed fact", "research")))
graph.add_node(Node("writer", lambda ctx: "Draft: " + ctx.inbox[0].value))
graph.add_edge("research", "writer")

result = graph.execute()
print(result.outputs["writer"])
print(result.events)
