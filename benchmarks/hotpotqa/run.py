"""Small HotpotQA distractor pilot: single, fixed two-worker, dynamic DAG."""

import argparse
import json
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from context_dag import Action, DeepSeekClient, DynamicDAG, JSONAgent


def passages(item):
    return "\n".join(
        f"[{title} #{i}] {sentence.strip()}"
        for title, sentences in item["context"] for i, sentence in enumerate(sentences)
    )


def ask(client, system, user):
    raw = client([{"role": "system", "content": system + " Return a JSON object."},
                  {"role": "user", "content": user}])
    return json.loads(raw)


ANSWER_SCHEMA = ('Return {"answer":"short answer", "sp":[["document title", sentence_index], ...]}. '
                 'For yes/no questions answer exactly yes or no. Cite only supplied sentences. '
                 'Use the document title without the #sentence suffix.')
EVIDENCE_SCHEMA = ('Find useful evidence for this question. Return '
                   '{"evidence":[{"title":"exact document title without #index",'
                   '"index":0,"quote":"..."}, ...]}. Do not answer yet.')


def clean_evidence(response):
    """Convert our displayed [title #index] reference to canonical fields."""
    for evidence in response.get("evidence", []):
        match = re.fullmatch(r"(.+) #(\d+)", str(evidence.get("title", "")))
        if match:
            evidence["title"] = match.group(1)
            evidence["index"] = int(match.group(2))
    return response


def single(item, client):
    return ask(client, ANSWER_SCHEMA, f"Question: {item['question']}\nSentences:\n{passages(item)}")


def fixed(item, client):
    chunks = (item["context"][:5], item["context"][5:])
    evidence = []
    for chunk in chunks:
        local = dict(item, context=chunk)
        evidence.append(clean_evidence(ask(client, EVIDENCE_SCHEMA,
            f"Question: {item['question']}\nSentences:\n{passages(local)}")))
    return ask(client, ANSWER_SCHEMA,
        f"Question: {item['question']}\nEvidence from two workers:\n{json.dumps(evidence, ensure_ascii=False)}")


def dynamic(item, client):
    root = (f"Answer this question using only the given sentences: {item['question']}\n"
            "First inspect the DAG. Then decide whether to spawn up to two evidence workers or answer directly. "
            "If spawning, copy the relevant question and sentence references into each child prompt. "
            "Workers should yield evidence with document title and sentence index. "
            "If concluding, put a JSON object with answer and sp fields in result.\n"
            f"Sentences:\n{passages(item)}")
    dag = DynamicDAG(root)
    agent = JSONAgent(client)
    def finalizer(snapshot, store):
        return json.dumps(ask(client, ANSWER_SCHEMA,
            f"Question: {item['question']}\nNode results:\n{json.dumps(snapshot, ensure_ascii=False)}"))
    result = dag.run(agent, finalizer, max_nodes=4, max_steps=12)
    return json.loads(result.answer), len(result.nodes), list(result.events)


def constrained(item, client):
    """The model plans a document split; the DAG runs two evidence contexts."""
    titles = [title for title, _ in item["context"]]
    title_map = {title: sentences for title, sentences in item["context"]}
    dag = DynamicDAG(f"Investigate: {item['question']}")
    plan = None

    def decide(ctx):
        nonlocal plan
        if ctx.node_id == "root" and ctx.observation is None:
            return Action("read_dag", verbose=False)
        if ctx.node_id == "root":
            plan = ask(client,
                'Split all document titles into two disjoint evidence groups. Return {"group_a":["exact title",...],"group_b":["exact title",...]}. Include every title exactly once. Put likely complementary clues in different groups.',
                f"Question: {item['question']}\nTitles: {json.dumps(titles, ensure_ascii=False)}\nDAG: {json.dumps(ctx.observation)}")
            a, b = plan["group_a"], plan["group_b"]
            if not a or not b or len(a + b) != len(titles) or set(a + b) != set(titles):
                raise ValueError("planner must partition all document titles exactly once")
            return Action("spawn", children=(("worker_a", json.dumps(a)), ("worker_b", json.dumps(b))))
        selected = json.loads(ctx.prompt)
        local = dict(item, context=[(title, title_map[title]) for title in selected])
        evidence = clean_evidence(ask(client, EVIDENCE_SCHEMA,
            f"Question: {item['question']}\nSentences:\n{passages(local)}"))
        return Action("yield", result=json.dumps(evidence, ensure_ascii=False))

    def finalizer(snapshot, store):
        evidence = [node["result"] for node in snapshot["nodes"] if node["status"] == "yielded"]
        return json.dumps(ask(client, ANSWER_SCHEMA,
            f"Question: {item['question']}\nEvidence from two workers:\n{json.dumps(evidence, ensure_ascii=False)}"))

    result = dag.run(decide, finalizer, max_nodes=3, max_steps=8)
    return json.loads(result.answer), len(result.nodes), list(result.events), plan


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", re.sub(r"\b(a|an|the)\b", " ", text)).strip()


def score(prediction, gold):
    answer_em = int(normalize(str(prediction.get("answer", ""))) == normalize(gold["answer"]))
    try:
        predicted = {(str(title), int(index)) for title, index in prediction.get("sp", [])}
    except (TypeError, ValueError):
        predicted = set()
    support_em = int(predicted == {tuple(fact) for fact in gold["supporting_facts"]})
    return {"answer_em": answer_em, "support_em": support_em, "joint_em": answer_em * support_em}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "hotpot_dev_distractor_v1.json")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "pilot.jsonl")
    parser.add_argument("--policies", nargs="+", choices=("single", "fixed", "dynamic", "constrained"),
                        default=("single", "fixed", "dynamic", "constrained"))
    args = parser.parse_args()
    items = json.loads(args.data.read_text(encoding="utf-8"))[args.start:args.start + args.count]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in items:
            row = {"id": item["_id"], "type": item["type"], "level": item["level"]}
            for name in args.policies:
                client = DeepSeekClient(max_tokens=1200)
                try:
                    result = globals()[name](item, client)
                    if name == "dynamic":
                        prediction, nodes, events = result
                        row[name + "_nodes"] = nodes
                        row[name + "_events"] = events
                    elif name == "constrained":
                        prediction, nodes, events, plan = result
                        row[name + "_nodes"] = nodes
                        row[name + "_events"] = events
                        row[name + "_plan"] = plan
                    else:
                        prediction = result
                    row[name] = {"prediction": prediction, **score(prediction, item),
                                 "calls": len(client.usages),
                                 "tokens": sum(x.get("total_tokens", 0) for x in client.usages)}
                except Exception as error:
                    row[name] = {"error": type(error).__name__ + ": " + str(error),
                                 "calls": len(client.usages)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(row["id"], {name: row[name].get("joint_em", "error")
                               for name in args.policies}, flush=True)


if __name__ == "__main__":
    main()
