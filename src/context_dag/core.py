"""Logical context DAG with directed handoffs and scoped shared storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

Route = Literal["push", "store", "both", "hold"]


@dataclass(frozen=True)
class RoutingDecision:
    """Route an artifact, optionally limiting direct delivery to named successors."""

    mode: Route
    targets: frozenset[str] | None = None


@dataclass(frozen=True)
class Artifact:
    key: str
    value: Any
    producer: str
    version: int = 1
    verified: bool = True
    readers: frozenset[str] = frozenset()


@dataclass
class Node:
    id: str
    run: Callable[[ExecutionContext], Any]
    instructions: str = ""


@dataclass
class ExecutionContext:
    node_id: str
    instructions: str
    inbox: tuple[Artifact, ...]
    store: SharedStore

    def read(self, key: str) -> Artifact | None:
        return self.store.read(key, self.node_id)


@dataclass
class RunResult:
    outputs: dict[str, Any]
    artifacts: tuple[Artifact, ...]
    events: tuple[dict[str, Any], ...]


class SharedStore:
    def __init__(self) -> None:
        self._items: dict[str, Artifact] = {}
        self.events: list[dict[str, Any]] = []

    def write(self, artifact: Artifact) -> None:
        previous = self._items.get(artifact.key)
        if previous and artifact.version <= previous.version:
            raise ValueError(f"version must increase for {artifact.key}")
        self._items[artifact.key] = artifact
        self.events.append({"kind": "store", "key": artifact.key, "version": artifact.version})

    def read(self, key: str, reader: str) -> Artifact | None:
        item = self._items.get(key)
        allowed = item is not None and item.verified and (not item.readers or reader in item.readers)
        self.events.append({"kind": "read", "key": key, "reader": reader, "hit": allowed})
        return item if allowed else None


class ContextDAG:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, set[str]] = {}

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node: {node.id}")
        self.nodes[node.id] = node
        self.edges[node.id] = set()

    def add_edge(self, source: str, target: str) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("both edge endpoints must exist")
        self.edges[source].add(target)
        try:
            self.order()
        except ValueError:
            self.edges[source].remove(target)
            raise

    def order(self) -> list[str]:
        remaining = {n: 0 for n in self.nodes}
        for targets in self.edges.values():
            for target in targets:
                remaining[target] += 1
        ready = sorted(n for n, degree in remaining.items() if degree == 0)
        result = []
        while ready:
            node = ready.pop(0)
            result.append(node)
            for target in sorted(self.edges[node]):
                remaining[target] -= 1
                if remaining[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(result) != len(self.nodes):
            raise ValueError("cycle in context DAG")
        return result

    def execute(self, route: Callable[[Artifact, frozenset[str]], Route | RoutingDecision] | None = None) -> RunResult:
        """Execute nodes once in topological order; node code decides store reads."""
        route = route or (lambda artifact, successors: "push")
        store = SharedStore()
        inboxes: dict[str, list[Artifact]] = {node: [] for node in self.nodes}
        outputs: dict[str, Any] = {}
        artifacts: list[Artifact] = []
        events = store.events
        for node_id in self.order():
            node = self.nodes[node_id]
            context = ExecutionContext(node_id, node.instructions, tuple(inboxes[node_id]), store)
            output = node.run(context)
            outputs[node_id] = output
            produced = output if isinstance(output, Artifact) else ()
            if isinstance(produced, Artifact):
                produced = (produced,)
            if isinstance(output, (list, tuple)) and all(isinstance(x, Artifact) for x in output):
                produced = output
            for artifact in produced:
                if artifact.producer != node_id:
                    raise ValueError("artifact producer must match running node")
                choice = route(artifact, frozenset(self.edges[node_id]))
                action = choice.mode if isinstance(choice, RoutingDecision) else choice
                if action not in ("push", "store", "both", "hold"):
                    raise ValueError(f"invalid route: {action}")
                targets = (choice.targets if isinstance(choice, RoutingDecision) else None)
                if targets is not None:
                    if action not in ("push", "both"):
                        raise ValueError("targets require push or both")
                    if not targets <= self.edges[node_id]:
                        raise ValueError("direct targets must be outgoing successors")
                delivered_to = sorted(self.edges[node_id] if targets is None else targets)
                artifacts.append(artifact)
                if action in ("push", "both"):
                    for target in delivered_to:
                        inboxes[target].append(artifact)
                if action in ("store", "both"):
                    store.write(artifact)
                events.append({"kind": "route", "key": artifact.key, "source": node_id,
                               "targets": sorted(self.edges[node_id]), "delivered_to":
                               delivered_to if action in ("push", "both") else [], "action": action})
        return RunResult(outputs, tuple(artifacts), tuple(events))
