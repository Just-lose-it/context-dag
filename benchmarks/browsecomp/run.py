"""Fixed candidate-pool pilot for evidence-conditioned dynamic DAG expansion."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from context_dag import Action, DeepSeekClient, DynamicDAG, JSONAgent
from benchmarks.hotpotqa.run import ask, normalize
from benchmarks.browsecomp.search_index import SearchIndex

HERE = Path(__file__).parent
BUDGET = 16000
ANSWER_SCHEMA = ('Return {"answer":"short final answer","evidence_docids":["docid",...]}. '
                 'Use only evidence found through the supplied search and read tools.')


def load_row(offset):
    return json.loads((HERE / "data" / f"row_{offset}.json").read_text(encoding="utf-8"))


def plan_leads(row, client):
    plan = ask(client,
        'Return {"leads":["self-contained research question 1","self-contained research question 2"]}. Identify two distinct investigative leads without assuming the answer.',
        f"Question: {row['query']}")
    leads = plan.get("leads", [])
    if not isinstance(leads, list) or len(leads) != 2 or not all(isinstance(x, str) and x.strip() for x in leads):
        raise ValueError("planner must return exactly two research leads")
    return leads


def agent_prompt(main_question, lead, *, root=False):
    task = "Solve the main question" if root else f"Investigate this lead: {lead}"
    end = ("Conclude with a JSON string containing answer and evidence_docids. " if root else
           "Yield a concise JSON string containing finding and evidence_docids. ")
    return (f"Main question: {main_question}\n{task}.\n"
            "You have search_docs(query, limit<=5) and read_docs(documents=[chunk_id,...]). "
            "Search with specific clue terms, read promising chunks, and follow newly found entities with new searches. "
            "Use at most four searches and four reads in this context. Cite exact docids from read chunks. "
            + end + "Never guess from the question alone.")


def run_policy(row, index, client, policy, leads=None, telemetry=None):
    nested = policy == "dynamic"
    dag = DynamicDAG(agent_prompt(row["query"], row["query"], root=True))
    agent = JSONAgent(client)
    depth = {"root": 0, "lead_0": 1, "lead_1": 1}
    expanded = set()
    candidate = {}
    calls = {"search": 0, "read": 0, "spawn": 0, "gap_checks": 0, "read_by_node": {}}
    if telemetry is not None:
        telemetry["dag"] = dag
        telemetry["calls"] = calls
    per_node = {}

    def search_documents(query, limit, node):
        calls["search"] += 1
        per_node.setdefault(node, {"search": 0, "read": 0})["search"] += 1
        if per_node[node]["search"] > 4:
            return {"error": "search limit reached in this context"}
        return index.search(query, limit)

    def read_documents(ids, node):
        calls["read"] += 1
        per_node.setdefault(node, {"search": 0, "read": 0})["read"] += 1
        if per_node[node]["read"] > 4:
            return {"error": "read limit reached in this context"}
        result = index.read(ids)
        if "error" not in result:
            calls["read_by_node"].setdefault(node, []).extend(
                index.chunks[chunk_id]["docid"] for chunk_id in ids)
        return result

    def decide(ctx):
        if policy != "single" and ctx.node_id == "root":
            if ctx.observation is None:
                return Action("spawn", children=tuple(
                    (f"lead_{i}", agent_prompt(row["query"], lead)) for i, lead in enumerate(leads)))
            return Action("yield")
        if ctx.observation and "completed_children" in ctx.observation:
            child_results = [notice.get("result") or ctx.store.read(notice["store_key"], ctx.node_id).value
                             for notice in ctx.observation["completed_children"]]
            merged = ask(client,
                'Return {"finding":"short finding","evidence_docids":["docid",...]}. Combine only the supplied findings; preserve exact docids.',
                f"Lead: {ctx.prompt}\nOwn earlier finding: {candidate.get(ctx.node_id, '')}\n"
                f"Child results: {json.dumps(child_results, ensure_ascii=False)}")
            return Action("yield", result=json.dumps(merged, ensure_ascii=False))
        action = agent(ctx)
        if action.name == "spawn":
            return Action("yield", result="No verified finding: direct delegation unavailable")
        if ctx.node_id == "root":
            if action.name == "yield":
                return Action("conclude", result=action.result)
            return action
        if action.name in ("yield", "conclude"):
            if not nested or depth[ctx.node_id] >= 2 or ctx.node_id in expanded:
                return Action("yield", result=action.result)
            candidate[ctx.node_id] = action.result
            calls["gap_checks"] += 1
            gap = ask(client,
                'Evaluate whether a necessary fact for this investigative lead remains unsupported by the finding and cited documents. Return {"missing":["one self-contained factual question",...]}. Give at most two independent missing questions. If the lead is adequately supported, return an empty list. Do not use any gold answer.',
                f"Lead: {ctx.prompt}\nCurrent finding: {action.result}")
            missing = gap.get("missing", [])
            if isinstance(missing, list):
                missing = [question for question in missing[:2] if isinstance(question, str) and question.strip()]
            else:
                missing = []
            if missing:
                expanded.add(ctx.node_id)
                children = []
                for i, subquestion in enumerate(missing):
                    child_id = f"{ctx.node_id}_gap_{i}"
                    depth[child_id] = 2
                    children.append((child_id, agent_prompt(row["query"], subquestion)))
                calls["spawn"] += 1
                return Action("spawn", children=tuple(children))
            return Action("yield", result=action.result)
        return action

    def finalizer(snapshot, store):
        findings = [{"node": node["id"], "result": node.get("result", "")}
                    for node in snapshot["nodes"] if node["status"] == "yielded" and node.get("result")]
        return json.dumps(ask(client, ANSWER_SCHEMA,
            f"Main question: {row['query']}\nFindings: {json.dumps(findings, ensure_ascii=False)}"),
            ensure_ascii=False)

    result = dag.run(decide, finalizer, read_documents=read_documents,
                     search_documents=search_documents, max_nodes=7, max_steps=32,
                     handoff_limit=2000)
    raw = json.loads(result.answer)
    prediction = raw if isinstance(raw, dict) else {"answer": str(raw), "evidence_docids": []}
    return prediction, result, calls


def score(prediction, row, calls):
    answer_em = int(normalize(str(prediction.get("answer", ""))) == normalize(row["answer"]))
    retrieved = {docid for docids in calls["read_by_node"].values() for docid in docids}
    evidence = {doc["docid"] for doc in row["evidence_docs"]}
    gold = {doc["docid"] for doc in row["gold_docs"]}
    cited = {str(docid) for docid in prediction.get("evidence_docids", [])}
    return {"answer_em": answer_em,
            "evidence_recall": len(retrieved & evidence) / len(evidence) if evidence else 0,
            "gold_recall": len(retrieved & gold) / len(gold) if gold else 0,
            "retrieved_docids": sorted(retrieved),
            "cited_evidence": len(cited & evidence), "evidence_count": len(evidence)}


def evaluate(row, index, policy, leads=None, plan_usages=None):
    client = DeepSeekClient(max_tokens=900, total_token_budget=BUDGET)
    if plan_usages:
        client.usages.extend(plan_usages)
    start = time.perf_counter()
    result = {}
    telemetry = {}
    try:
        prediction, dag_result, calls = run_policy(row, index, client, policy, leads, telemetry)
        result.update(prediction=prediction, **score(prediction, row, calls),
                      nodes=len(dag_result.nodes), tool_calls=calls, events=list(dag_result.events))
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        if "dag" in telemetry:
            result.update(nodes=len(telemetry["dag"].nodes),
                          tool_calls=telemetry["calls"],
                          events=list(telemetry["dag"].events))
    result.update(calls=len(client.usages), tokens=sum(u.get("total_tokens", 0) for u in client.usages),
                  seconds=round(time.perf_counter() - start, 3))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("diagnostic", "evaluation"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selection = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
    offsets = selection[args.split + "_offsets"][:args.limit]
    output = args.output or HERE / "results" / f"{args.split}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(line) for line in output.open(encoding="utf-8")] if output.exists() else []
    def payment_blocked(record):
        return any("HTTP Error 402" in str(record.get(name, {}).get("error", ""))
                   for name in ("single", "flat", "dynamic"))
    done = {entry["offset"] for entry in previous if not payment_blocked(entry)}
    with output.open("a", encoding="utf-8") as handle:
        def emit(record):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(record["offset"], {name: record.get(name, {}).get("answer_em", "error")
                                    for name in ("single", "flat", "dynamic")}, flush=True)
        for offset in offsets:
            if offset in done:
                continue
            row = load_row(offset)
            index = SearchIndex(row)
            record = {"offset": offset, "query_id": row["query_id"], "split": args.split,
                      "candidate_docs": len({d["docid"] for group in ("evidence_docs", "gold_docs", "negative_docs")
                                             for d in row[group]})}
            record["single"] = evaluate(row, index, "single")
            if payment_blocked(record):
                emit(record)
                return
            planner = DeepSeekClient(max_tokens=900, total_token_budget=BUDGET)
            try:
                leads = plan_leads(row, planner)
                record["plan"] = leads
                for policy in ("flat", "dynamic"):
                    record[policy] = evaluate(row, index, policy, leads, planner.usages)
                    if payment_blocked(record):
                        emit(record)
                        return
            except Exception as exc:
                for policy in ("flat", "dynamic"):
                    record[policy] = {"error": "planner: " + type(exc).__name__ + ": " + str(exc),
                                      "calls": len(planner.usages),
                                      "tokens": sum(u.get("total_tokens", 0) for u in planner.usages)}
                if payment_blocked(record):
                    emit(record)
                    return
            emit(record)


if __name__ == "__main__":
    main()
