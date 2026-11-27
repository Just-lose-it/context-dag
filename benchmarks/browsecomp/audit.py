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
    parser.add_argument("--selection", type=Path, default=HERE / "selection.json")
    parser.add_argument("--policies", nargs="+", default=("single", "flat", "dynamic"))
    args = parser.parse_args()
    suffix = "" if args.selection.name == "selection.json" else f"_{args.selection.stem}"
    path = HERE / "results" / f"{args.split}{suffix}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    print("questions", len(rows))
    for policy in args.policies:
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
    adaptive = args.policies[-1]
    print(f"offset flat_read {adaptive}_read new_labeled_evidence new_gold child_read child_new_evidence child_new_gold")
    for row in rows:
        source = json.loads((HERE / "data" / f"row_{row['offset']}.json").read_text(encoding="utf-8"))
        evidence = {doc["docid"] for doc in source["evidence_docs"]}
        gold = {doc["docid"] for doc in source["gold_docs"]}
        flat, dynamic = read_ids(row["flat"]), read_ids(row[adaptive])
        child = {docid for node, ids in row[adaptive].get("tool_calls", {}).get("read_by_node", {}).items()
                 if "_gap_" in node for docid in ids}
        print(row["offset"], len(flat), len(dynamic), len((dynamic - flat) & evidence),
              len((dynamic - flat) & gold), len(child), len((child - flat) & evidence),
              len((child - flat) & gold))


if __name__ == "__main__":
    main()
