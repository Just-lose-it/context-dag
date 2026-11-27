"""Audit whether grandchildren retrieved previously unread gold evidence."""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
rows = [json.loads(line) for line in (HERE / "results/multilevel_branch_4hop3.jsonl").open(encoding="utf-8")]
rows = [row for row in rows if row["split"] == "evaluation"]
ids = {row["id"] for row in rows}
gold = {item["id"]: item for line in (HERE / "data/musique_ans_v1.0_dev.jsonl").open(encoding="utf-8")
        if (item := json.loads(line))["id"] in ids}


def read_ids(events, item, *, grandchildren=None):
    """Count only valid read_docs requests; invalid requests returned error observations."""
    valid = {f"P{p['idx']} {p['title']}": p["idx"] for p in item["paragraphs"]}
    found = set()
    for event in events:
        if event.get("action") != "read_docs":
            continue
        is_grandchild = "_sub_" in event["node"]
        if grandchildren is not None and is_grandchild != grandchildren:
            continue
        documents = event["documents"]
        if len(documents) > 5 or any(title not in valid for title in documents):
            continue
        found.update(valid[title] for title in documents)
    return found


def predicted_ids(prediction, item):
    titles = {}
    for paragraph in item["paragraphs"]:
        titles.setdefault(paragraph["title"], []).append(paragraph["idx"])
    found = set()
    for fact in prediction.get("sp", []):
        if not isinstance(fact, list) or len(fact) != 2 or fact[1] != 0:
            continue
        title = str(fact[0])
        match = re.match(r"^P(\d+) ", title)
        if match:
            found.add(int(match.group(1)))
        elif len(titles.get(title, [])) == 1:
            found.add(titles[title][0])
    return found


totals = {"valid_items": 0, "flat_gold_reads": 0, "nested_gold_reads": 0,
          "grandchild_only_gold": 0, "nested_new_vs_flat_gold": 0,
          "grandchild_gold_reads": 0, "flat_final_gold": 0,
          "nested_final_gold": 0, "gold_count": 0}
for row in rows:
    item = gold[row["id"]]
    support = {p["idx"] for p in item["paragraphs"] if p["is_supporting"]}
    flat, nested = row["flat"], row["nested"]
    if "error" in flat or "error" in nested:
        print(row["id"], "unpaired_error", flat.get("error"), nested.get("error"))
        continue
    flat_read = read_ids(flat["events"], item)
    parent_read = read_ids(nested["events"], item, grandchildren=False)
    child_read = read_ids(nested["events"], item, grandchildren=True)
    nested_read = parent_read | child_read
    novel = (nested_read - flat_read) & support
    child_only = (child_read - parent_read) & support
    totals["valid_items"] += 1
    totals["gold_count"] += len(support)
    totals["flat_gold_reads"] += len(flat_read & support)
    totals["nested_gold_reads"] += len(nested_read & support)
    totals["grandchild_gold_reads"] += len(child_read & support)
    totals["grandchild_only_gold"] += len(child_only)
    totals["nested_new_vs_flat_gold"] += len(novel)
    flat_final = predicted_ids(flat["prediction"], item) & support
    nested_final = predicted_ids(nested["prediction"], item) & support
    totals["flat_final_gold"] += len(flat_final)
    totals["nested_final_gold"] += len(nested_final)
    print(row["id"], "gold", sorted(support), "flat_read", sorted(flat_read & support),
          "nested_parent_read", sorted(parent_read & support),
          "grandchild_read", sorted(child_read & support),
          "novel_vs_flat", sorted(novel),
          "flat_final_support", sorted(flat_final),
          "nested_final_support", sorted(nested_final),
          "flat_joint", flat["joint_em"], "nested_joint", nested["joint_em"])
print("TOTAL", totals)
