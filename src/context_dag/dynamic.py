"""Stage-based dynamic context DAG with a small ReAct-style tool loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from .core import Artifact, SharedStore

ActionName = Literal["read_dag", "search_docs", "read_docs", "read_result", "spawn", "yield", "conclude"]


@dataclass(frozen=True)
class Action:
    name: ActionName
    verbose: bool = False
    children: tuple[tuple[str, str], ...] = ()  # (node_id, prompt)
    documents: tuple[str, ...] = ()
    query: str = ""
    limit: int = 5
    key: str = ""
    result: str = ""


@dataclass
class ContextNode:
    id: str
    prompt: str
    parent: str | None = None
    stage: int = 0
    status: str = "pending"
    result: str = ""


@dataclass(frozen=True)
class DecisionContext:
    node_id: str
    prompt: str
    stage: int
    observation: dict[str, Any] | None
    store: SharedStore


@dataclass(frozen=True)
class DynamicResult:
    answer: str
    nodes: dict[str, ContextNode]
    events: tuple[dict[str, Any], ...]
    finalized: bool


class DynamicDAG:
    def __init__(self, root_prompt: str, root_id: str = "root") -> None:
        self.nodes = {root_id: ContextNode(root_id, root_prompt)}
        self.root_id = root_id
        self.store = SharedStore()
        self.events: list[dict[str, Any]] = []
        self._ran = False
        self._result_versions: dict[str, int] = {}

    def snapshot(self, verbose: bool = False, *, viewer: str | None = None,
                 handoff_limit: int = 256) -> dict[str, Any]:
        nodes = []
        for node in self.nodes.values():
            item = {"id": node.id, "parent": node.parent, "stage": node.stage, "status": node.status}
            if verbose:
                item["prompt"] = node.prompt
                if viewer is None or viewer == node.id or (
                    viewer == node.parent and len(node.result) <= handoff_limit
                ):
                    item["result"] = node.result
                elif node.result:
                    item["result_key"] = f"node/{node.id}/result"
            nodes.append(item)
        return {"nodes": nodes, "edges": [
            {"from": node.parent, "to": node.id} for node in self.nodes.values() if node.parent
        ]}

    def run(
        self,
        decide: Callable[[DecisionContext], Action],
        finalizer: Callable[[dict[str, Any], SharedStore], str],
        *,
        max_nodes: int = 64,
        max_steps: int = 256,
        read_documents: Callable[[tuple[str, ...], str], dict[str, Any]] | None = None,
        search_documents: Callable[[str, int, str], dict[str, Any]] | None = None,
        handoff_limit: int = 256,
    ) -> DynamicResult:
        """Run each stage to completion, then start its spawned children.

        read_dag is a nonterminal tool action. A yielded node resumes after a
        later stage completes, if one exists. conclude ends the task. If all
        work finishes without concluding, a separate finalizer runs.
        """
        if max_nodes < 1 or max_steps < 1 or handoff_limit < 0:
            raise ValueError("limits must be positive")
        if self._ran:
            raise RuntimeError("a DynamicDAG instance can run only once")
        self._ran = True
        steps = 0
        stage = 0
        waiting: dict[str, int] = {}

        def save_result(node: ContextNode) -> None:
            key = f"node/{node.id}/result"
            version = self._result_versions.get(node.id, 0) + 1
            self._result_versions[node.id] = version
            readers = frozenset({node.parent}) if node.parent else frozenset()
            self.store.write(Artifact(key, node.result, node.id, version=version, readers=readers))
            self.events.append({"node": node.id, "action": "store_result", "key": key,
                                "version": version})

        def child_handoff(parent: ContextNode) -> dict[str, Any]:
            children = [node for node in self.nodes.values() if node.parent == parent.id]
            messages = []
            for child in children:
                key = f"node/{child.id}/result"
                direct = len(child.result) <= handoff_limit
                message = {"node": child.id, "store_key": key, "version": self._result_versions[child.id],
                           "mode": "direct" if direct else "reference"}
                if direct:
                    message["result"] = child.result
                messages.append(message)
                self.events.append({"node": child.id, "action": "handoff", "parent": parent.id,
                                    "mode": message["mode"], "key": key})
            return {"completed_children": messages}

        def run_node(node: ContextNode, current_stage: int, observation: dict[str, Any] | None = None) -> str | None:
            nonlocal steps
            node.status = "running"
            while True:
                if steps >= max_steps:
                    raise RuntimeError("maximum ReAct steps exceeded")
                steps += 1
                action = decide(DecisionContext(node.id, node.prompt, current_stage, observation, self.store))
                if action.name == "read_dag":
                    observation = self.snapshot(action.verbose, viewer=node.id,
                                                handoff_limit=handoff_limit)
                    self.events.append({"node": node.id, "action": "read_dag", "verbose": action.verbose})
                    continue
                if action.name == "read_docs":
                    if read_documents is None:
                        raise ValueError("read_docs requires a read_documents tool")
                    if not action.documents:
                        raise ValueError("read_docs requires document IDs")
                    observation = read_documents(action.documents, node.id)
                    self.events.append({"node": node.id, "action": "read_docs",
                                        "documents": list(action.documents)})
                    continue
                if action.name == "search_docs":
                    if search_documents is None:
                        raise ValueError("search_docs requires a search_documents tool")
                    if not action.query.strip() or action.limit < 1:
                        raise ValueError("search_docs requires a query and positive limit")
                    observation = search_documents(action.query, action.limit, node.id)
                    self.events.append({"node": node.id, "action": "search_docs",
                                        "query": action.query, "limit": action.limit})
                    continue
                if action.name == "read_result":
                    if not action.key:
                        raise ValueError("read_result requires a key")
                    artifact = self.store.read(action.key, node.id)
                    observation = {"key": action.key, "result": artifact.value if artifact else None,
                                   "found": artifact is not None}
                    self.events.append({"node": node.id, "action": "read_result", "key": action.key,
                                        "found": artifact is not None})
                    continue
                if action.name == "spawn":
                    if not action.children:
                        raise ValueError("spawn requires children")
                    ids = [child_id for child_id, _ in action.children]
                    if len(ids) != len(set(ids)) or any(child_id in self.nodes for child_id in ids):
                        raise ValueError("child IDs must be new and unique")
                    if len(self.nodes) + len(ids) > max_nodes:
                        raise RuntimeError("maximum node count exceeded")
                    if any(not child_id or not prompt for child_id, prompt in action.children):
                        raise ValueError("children need IDs and prompts")
                    for child_id, prompt in action.children:
                        self.nodes[child_id] = ContextNode(child_id, prompt, node.id, current_stage + 1)
                    node.status = "spawned"
                    waiting.pop(node.id, None)
                    self.events.append({"node": node.id, "action": "spawn", "children": ids})
                    return None
                if action.name == "yield":
                    node.status = "yielded"
                    node.result = action.result
                    save_result(node)
                    waiting[node.id] = current_stage
                    self.events.append({"node": node.id, "action": "yield", "result": action.result})
                    return None
                if action.name == "conclude":
                    node.status = "concluded"
                    node.result = action.result
                    save_result(node)
                    self.events.append({"node": node.id, "action": "conclude", "result": action.result})
                    return action.result
                raise ValueError(f"unknown action: {action.name}")

        while True:
            current = [node for node in self.nodes.values() if node.stage == stage and node.status == "pending"]
            if not current:
                ready_parents = [node for node in self.nodes.values() if node.status == "spawned" and
                                 all(child.status == "yielded" for child in self.nodes.values()
                                     if child.parent == node.id)]
                if ready_parents:
                    for parent in sorted(ready_parents, key=lambda item: -item.stage):
                        if parent.status != "spawned":
                            continue
                        answer = run_node(parent, stage - 1, child_handoff(parent))
                        if answer is not None:
                            return DynamicResult(answer, self.nodes.copy(), tuple(self.events), False)
                    continue
                # A yielded context waits until a later stage has finished.
                resumed = [self.nodes[node_id] for node_id, waited_at in waiting.items() if waited_at < stage - 1]
                if not resumed:
                    break
                for node in resumed:
                    answer = run_node(node, stage - 1, self.snapshot(verbose=True))
                    if answer is not None:
                        return DynamicResult(answer, self.nodes.copy(), tuple(self.events), False)
                continue
            for node in current:
                answer = run_node(node, stage)
                if answer is not None:
                    return DynamicResult(answer, self.nodes.copy(), tuple(self.events), False)
            stage += 1
        answer = finalizer(self.snapshot(verbose=True), self.store)
        self.events.append({"node": "finalizer", "action": "conclude", "result": answer})
        return DynamicResult(answer, self.nodes.copy(), tuple(self.events), True)
