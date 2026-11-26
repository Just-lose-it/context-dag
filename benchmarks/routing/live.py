"""Live DeepSeek comparison of broadcast, targeted handoff, and shared read."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from context_dag import Artifact, ContextDAG, DeepSeekClient, Node, RoutingDecision
from benchmarks.hotpotqa.run import normalize

HERE = Path(__file__).parent


def load_tasks():
    sample = json.loads((HERE / "live_sample.json").read_text(encoding="utf-8"))
    ids = set(sample["shared_ids"] + sample["local_ids"])
    lookup = {item["id"]: item for line in (HERE.parent / "musique/data/musique_ans_v1.0_dev.jsonl").open(encoding="utf-8")
              if (item := json.loads(line))["id"] in ids}
    return sample, lookup


def report(item):
    return "\n".join(f"[{p['idx']}: {p['title']}] {p['paragraph_text']}" for p in item["paragraphs"])


def run_batch(shared, locals_, policy):
    dag = ContextDAG()
    full_report = report(shared)
    dag.add_node(Node("source", lambda ctx: Artifact(
        "shared_report", full_report, "source",
        readers=frozenset(f"consumer_{i}" for i in range(8)))))
    metrics = []

    def consumer(index, item):
        def answer(ctx):
            needs_shared = index == 7
            inbox_report = ctx.inbox[0].value if ctx.inbox else None
            artifact = ctx.read("shared_report") if needs_shared and inbox_report is None else None
            shared_text = inbox_report or (artifact.value if artifact else "")
            if needs_shared:
                question = item["question"]
                gold = [item["answer"], *item["answer_aliases"]]
                local_text = ""
            else:
                step = item["question_decomposition"][0]
                paragraph = next(p for p in item["paragraphs"]
                                 if p["idx"] == step["paragraph_support_idx"])
                question = step["question"]
                gold = [step["answer"]]
                local_text = f"[{paragraph['title']}] {paragraph['paragraph_text']}"
            prompt = (f"Question: {question}\nTask-specific passage: {local_text}\n"
                      f"Shared report: {shared_text}\n"
                      "Answer using the supplied text. Return a JSON object with one field: answer, a short string.")
            client = DeepSeekClient(max_tokens=300)
            start = time.perf_counter()
            try:
                prediction = json.loads(client([
                    {"role": "system", "content": "Answer the question using only supplied material. Return JSON."},
                    {"role": "user", "content": prompt},
                ]))["answer"]
                correct = int(normalize(str(prediction)) in {normalize(str(g)) for g in gold})
                error = None
            except Exception as exc:
                prediction, correct = None, 0
                error = type(exc).__name__ + ": " + str(exc)
            metrics.append({"node": ctx.node_id, "task_id": item["id"], "needs_shared": needs_shared,
                            "prediction": prediction, "correct": correct, "error": error,
                            "prompt_chars": len(prompt), "shared_chars": len(shared_text),
                            "tokens": sum(u.get("total_tokens", 0) for u in client.usages),
                            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in client.usages),
                            "seconds": time.perf_counter() - start})
            return prediction
        return answer

    for index, item in enumerate([*locals_, shared]):
        node_id = f"consumer_{index}"
        dag.add_node(Node(node_id, consumer(index, item)))
        dag.add_edge("source", node_id)
    def route(artifact, successors):
        return {"broadcast": "push", "targeted": RoutingDecision("push", frozenset({"consumer_7"})),
                "store": "store"}[policy]
    start = time.perf_counter()
    result = dag.execute(route)
    return {"policy": policy, "shared_id": shared["id"], "tasks": metrics,
            "elapsed_seconds": time.perf_counter() - start,
            "report_chars": len(full_report),
            "reads": sum(event["kind"] == "read" for event in result.events),
            "delivered_to": next(event["delivered_to"] for event in result.events
                                 if event["kind"] == "route")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results/live.jsonl")
    args = parser.parse_args()
    sample, lookup = load_tasks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {(row["shared_id"], row["policy"])
                     for line in args.output.open(encoding="utf-8") if (row := json.loads(line))}
    locals_ = [lookup[item_id] for item_id in sample["local_ids"]]
    policies = ["broadcast", "targeted", "store"]
    with args.output.open("a", encoding="utf-8") as handle:
        for index, item_id in enumerate(sample["shared_ids"]):
            for policy in policies[index:] + policies[:index]:
                if (item_id, policy) in completed:
                    continue
                row = run_batch(lookup[item_id], locals_[index * 7:(index + 1) * 7], policy)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(index, policy, sum(t["correct"] for t in row["tasks"]),
                      sum(t["tokens"] for t in row["tasks"]), round(row["elapsed_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
