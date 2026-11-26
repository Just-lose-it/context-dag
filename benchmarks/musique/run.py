"""MuSiQue paragraph-level pilot using the same DeepSeek model and DAG policies."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from context_dag import DeepSeekClient
from benchmarks.hotpotqa.run import adaptive, fixed, normalize, single

HERE = Path(__file__).parent


def convert(item):
    """Make paragraph IDs unique titles; #0 is the sole sentence per paragraph."""
    context = [(f"P{paragraph['idx']} {paragraph['title']}", [paragraph["paragraph_text"]])
               for paragraph in item["paragraphs"]]
    return {"question": item["question"], "context": context}


def score(prediction, item):
    aliases = [item["answer"], *item["answer_aliases"]]
    answer_em = int(normalize(str(prediction.get("answer", ""))) in {normalize(a) for a in aliases})
    gold = {paragraph["idx"] for paragraph in item["paragraphs"] if paragraph["is_supporting"]}
    title_ids = {}
    for paragraph in item["paragraphs"]:
        title_ids.setdefault(paragraph["title"], []).append(paragraph["idx"])
    predicted = set()
    for fact in prediction.get("sp", []):
        if isinstance(fact, list) and len(fact) == 2 and fact[1] == 0:
            title = str(fact[0])
            prefix = title.split(" ", 1)[0]
            if prefix.startswith("P") and prefix[1:].isdigit():
                predicted.add(int(prefix[1:]))
            elif len(title_ids.get(title, [])) == 1:
                predicted.add(title_ids[title][0])
    return {"answer_em": answer_em, "support_em": int(predicted == gold),
            "support_recall": len(predicted & gold) / len(gold),
            "joint_em": int(answer_em and predicted == gold)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("2hop", "4hop"), required=True)
    parser.add_argument("--policies", nargs="+", choices=("single", "fixed", "adaptive"),
                        default=("single", "fixed", "adaptive"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--token-budget", type=int, default=12000)
    args = parser.parse_args()
    selected = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))["ids"][args.group]
    lookup = {item["id"]: item for line in (HERE / "data/musique_ans_v1.0_dev.jsonl").open(encoding="utf-8")
              if (item := json.loads(line))["id"] in selected}
    output = args.output or HERE / "results" / f"{args.group}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item_id in selected:
            item = lookup[item_id]
            question = convert(item)
            row = {"id": item_id, "group": args.group}
            for policy in args.policies:
                client = DeepSeekClient(max_tokens=1200, total_token_budget=args.token_budget)
                try:
                    result = globals()[policy](question, client)
                    prediction = result[0] if policy == "adaptive" else result
                    row[policy] = {"prediction": prediction, **score(prediction, item),
                                   "calls": len(client.usages),
                                   "tokens": sum(u.get("total_tokens", 0) for u in client.usages)}
                    if policy == "adaptive":
                        row["adaptive_nodes"] = result[1]
                        row["adaptive_events"] = result[2]
                except Exception as error:
                    row[policy] = {"error": type(error).__name__ + ": " + str(error),
                                   "calls": len(client.usages),
                                   "tokens": sum(u.get("total_tokens", 0) for u in client.usages)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(item_id, {p: row[p].get("joint_em", "error") for p in args.policies}, flush=True)


if __name__ == "__main__":
    main()
