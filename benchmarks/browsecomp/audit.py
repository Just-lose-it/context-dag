"""Print aggregate pilot metrics without disclosing benchmark questions or answers."""

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def read_ids(record):
    by_node = record.get("tool_calls", {}).get("read_by_node", {})
    return {docid for values in by_node.values() for docid in values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("diagnostic", "evaluation"), default="evaluation")
    args = parser.parse_args()
    path = HERE / "results" / f"{args.split}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    print("questions", len(rows))
    for policy in ("single", "flat", "dynamic"):
        results = [row[policy] for row in rows]
        print(policy, {
            "exact_match": sum(result.get("answer_em", 0) for result in results),
            "completed": sum("answer_em" in result for result in results),
            "tokens": sum(result.get("tokens", 0) for result in results),
            "model_calls": sum(result.get("calls", 0) for result in results),
            "searches": sum(result.get("tool_calls", {}).get("search", 0) for result in results),
            "reads": sum(result.get("tool_calls", {}).get("read", 0) for result in results),
            "child_spawns": sum(result.get("tool_calls", {}).get("spawn", 0) for result in results),
            "errors": dict(Counter(result.get("error") for result in results if result.get("error"))),
        })
    print("offset flat_read dynamic_read new_labeled_evidence new_gold child_read child_new_evidence child_new_gold")
    for row in rows:
        source = json.loads((HERE / "data" / f"row_{row['offset']}.json").read_text(encoding="utf-8"))
        evidence = {doc["docid"] for doc in source["evidence_docs"]}
        gold = {doc["docid"] for doc in source["gold_docs"]}
        flat, dynamic = read_ids(row["flat"]), read_ids(row["dynamic"])
        child = {docid for node, ids in row["dynamic"].get("tool_calls", {}).get("read_by_node", {}).items()
                 if "_gap_" in node for docid in ids}
        print(row["offset"], len(flat), len(dynamic), len((dynamic - flat) & evidence),
              len((dynamic - flat) & gold), len(child), len((child - flat) & evidence),
              len((child - flat) & gold))


if __name__ == "__main__":
    main()
