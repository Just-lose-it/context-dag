"""Frozen chain/branch pilot: single context, flat DAG, optional nested spawn."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from context_dag import Action, DeepSeekClient, DynamicDAG, JSONAgent
from benchmarks.hotpotqa.run import ANSWER_SCHEMA, ask, passages, single
from benchmarks.musique.run import convert, score

HERE = Path(__file__).parent
BUDGET = 24000


def plan_branches(question, client):
    response = ask(client,
        'Identify exactly two useful investigative leads. Return {"branches":["self-contained lead question 1", "self-contained lead question 2"]}. Do not answer the main question.',
        f"Main question: {question['question']}\nAvailable titles: "
        f"{json.dumps([title for title, _ in question['context']], ensure_ascii=False)}")
    branches = response.get("branches", [])
    if not isinstance(branches, list) or len(branches) != 2 or not all(isinstance(x, str) and x.strip() for x in branches):
        raise ValueError("planner must return exactly two nonempty lead questions")
    return branches


def worker_prompt(main_question, lead, titles, *, can_spawn):
    option = ("If this lead asks for two or more distinct facts, first spawn two smaller investigators, "
              "one fact per child. Give each a self-contained question; then combine their evidence. "
              "If the lead asks for only one fact, read documents yourself. "
              if can_spawn else "Do not spawn other nodes; investigate the lead yourself. ")
    return (f"Main question: {main_question}\nYour investigative lead: {lead}\n"
            f"Available document titles: {json.dumps(titles, ensure_ascii=False)}\n"
            "Read relevant documents with read_docs before reporting evidence. " + option +
            "When finished, yield a short JSON string with a finding and evidence references "
            "using exact P-numbered titles and paragraph index 0. Never conclude the whole task.")


def dag_policy(question, client, branches, *, nested, telemetry=None):
    title_map = {title: sentences for title, sentences in question["context"]}
    titles = list(title_map)
    dag = DynamicDAG(f"Investigate: {question['question']}")
    agent = JSONAgent(client)
    depth = {"root": 0, "lead_0": 1, "lead_1": 1}
    counts = {"spawn": 0, "read_docs": 0, "split_checks": 0}
    if telemetry is not None:
        telemetry["dag"] = dag
        telemetry["counts"] = counts

    def read_documents(ids, node_id):
        counts["read_docs"] += 1
        if len(ids) > 5:
            return {"error": "Read at most five document titles at once"}
        invalid = [title for title in ids if title not in title_map]
        if invalid:
            return {"error": f"Unknown titles: {invalid}", "available_titles": titles}
        return {title: [f"#0 {title_map[title][0]}"] for title in ids}

    def decide(ctx):
        if ctx.node_id == "root":
            if ctx.observation is None:
                return Action("spawn", children=tuple(
                    (f"lead_{i}", worker_prompt(question["question"], lead, titles, can_spawn=nested))
                    for i, lead in enumerate(branches)))
            return Action("yield")
        if ctx.observation and "completed_children" in ctx.observation:
            evidence = [notice.get("result") or (ctx.store.read(notice["store_key"], ctx.node_id).value)
                        for notice in ctx.observation["completed_children"]]
            merged = ask(client,
                'Summarize the evidence from your investigators. Return {"finding":"short finding", "evidence":[...]}. Preserve exact paragraph references.',
                f"Lead: {ctx.prompt}\nChild results: {json.dumps(evidence, ensure_ascii=False)}")
            return Action("yield", result=json.dumps(merged, ensure_ascii=False))
        if nested and depth[ctx.node_id] == 1 and ctx.observation is None:
            counts["split_checks"] += 1
            split = ask(client,
                'Decide whether this investigative lead contains two independently researchable factual unknowns. Return {"split":true_or_false,"subquestions":["self-contained question 1","self-contained question 2"]}. Set split=false and subquestions=[] for a single fact. Do not use the gold answer.',
                ctx.prompt)
            questions = split.get("subquestions", [])
            if (split.get("split") is True and isinstance(questions, list) and len(questions) == 2
                    and all(isinstance(q, str) and q.strip() for q in questions)):
                children = []
                for i, subquestion in enumerate(questions):
                    child_id = f"{ctx.node_id}_sub_{i}"
                    depth[child_id] = 2
                    children.append((child_id, worker_prompt(question["question"], subquestion, titles,
                                                             can_spawn=False)))
                counts["spawn"] += 1
                return Action("spawn", children=tuple(children))
        action = agent(ctx)
        if action.name == "conclude":
            return Action("yield", result=action.result)
        if action.name == "spawn":
            if not nested or depth[ctx.node_id] >= 2:
                return Action("yield", result="No verified evidence: further delegation unavailable")
            if not action.children or len(action.children) > 2:
                return Action("yield", result="No verified evidence: invalid delegation")
            children = []
            for i, (_, prompt) in enumerate(action.children):
                child_id = f"{ctx.node_id}_sub_{i}"
                depth[child_id] = depth[ctx.node_id] + 1
                children.append((child_id, worker_prompt(question["question"], prompt, titles, can_spawn=False)))
            counts["spawn"] += 1
            return Action("spawn", children=tuple(children))
        return action

    def finalizer(snapshot, store):
        findings = [{"id": n["id"], "result": n.get("result", "")} for n in snapshot["nodes"]
                    if n["status"] == "yielded" and n.get("result")]
        return json.dumps(ask(client, ANSWER_SCHEMA,
            f"Main question: {question['question']}\nInvestigative findings: "
            f"{json.dumps(findings, ensure_ascii=False)}"), ensure_ascii=False)

    result = dag.run(decide, finalizer, read_documents=read_documents,
                     max_nodes=7, max_steps=28, handoff_limit=2000)
    return json.loads(result.answer), len(result.nodes), counts, list(result.events)


def evaluate_policy(name, question, item, plan=None, plan_usages=None):
    client = DeepSeekClient(max_tokens=1200, total_token_budget=BUDGET)
    if plan_usages:
        client.usages.extend(plan_usages)
    start = time.perf_counter()
    row = {}
    telemetry = {}
    try:
        if name == "single":
            prediction = single(question, client)
        elif name == "single_review":
            draft = single(question, client)
            prediction = ask(client, ANSWER_SCHEMA,
                f"Review this draft against every supplied paragraph. Correct the answer and evidence if needed.\n"
                f"Question: {question['question']}\nDraft: {json.dumps(draft, ensure_ascii=False)}\n"
                f"Paragraphs:\n{passages(question)}")
        else:
            prediction, nodes, counts, events = dag_policy(question, client, plan,
                                                            nested=name == "nested", telemetry=telemetry)
            row.update(nodes=nodes, child_spawns=counts["spawn"], reads=counts["read_docs"],
                       split_checks=counts["split_checks"], events=events)
        row.update(prediction=prediction, **score(prediction, item))
    except Exception as exc:
        row["error"] = type(exc).__name__ + ": " + str(exc)
        if "dag" in telemetry:
            row.update(nodes=len(telemetry["dag"].nodes),
                       child_spawns=telemetry["counts"]["spawn"],
                       reads=telemetry["counts"]["read_docs"],
                       split_checks=telemetry["counts"]["split_checks"],
                       events=list(telemetry["dag"].events))
    row.update(calls=len(client.usages), tokens=sum(u.get("total_tokens", 0) for u in client.usages),
               seconds=round(time.perf_counter() - start, 3))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("chain_4hop1", "branch_4hop3"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ids = json.loads((HERE / "selection_multilevel.json").read_text(encoding="utf-8"))["ids"][args.group][args.start:args.start + args.limit]
    lookup = {item["id"]: item for line in (HERE / "data/musique_ans_v1.0_dev.jsonl").open(encoding="utf-8")
              if (item := json.loads(line))["id"] in ids}
    output = args.output or HERE / "results" / f"multilevel_{args.group}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {row["id"] for line in output.open(encoding="utf-8") if (row := json.loads(line))} if output.exists() else set()
    with output.open("a", encoding="utf-8") as handle:
        for item_id in ids:
            if item_id in completed:
                continue
            item = lookup[item_id]
            question = convert(item)
            row = {"id": item_id, "group": args.group,
                   "split": "development" if args.start + ids.index(item_id) < 2 else "evaluation"}
            for name in ("single", "single_review"):
                row[name] = evaluate_policy(name, question, item)
            planner = DeepSeekClient(max_tokens=1200, total_token_budget=BUDGET)
            try:
                branches = plan_branches(question, planner)
                row["plan"] = branches
                for name in ("flat", "nested"):
                    row[name] = evaluate_policy(name, question, item, branches, planner.usages)
            except Exception as exc:
                for name in ("flat", "nested"):
                    row[name] = {"error": "planner: " + type(exc).__name__ + ": " + str(exc),
                                 "calls": len(planner.usages),
                                 "tokens": sum(u.get("total_tokens", 0) for u in planner.usages)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(item_id, {name: row[name].get("joint_em", "error")
                            for name in ("single", "single_review", "flat", "nested")}, flush=True)


if __name__ == "__main__":
    main()
