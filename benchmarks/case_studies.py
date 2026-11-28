"""Verify two post-hoc DAG wins without printing benchmark text or answers."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main():
    browse = rows(ROOT / "benchmarks/browsecomp/results/evaluation.jsonl")
    fixed = next(row for row in browse if row["offset"] == 172)
    assert fixed["split"] == "evaluation"
    assert fixed["single"]["answer_em"] == 0
    assert fixed["flat"]["answer_em"] == 1
    assert "error" in fixed["dynamic"]
    print("fixed_case", {
        "offset": fixed["offset"],
        "answer_em": {name: fixed[name].get("answer_em", 0) for name in ("single", "flat", "dynamic")},
        "tokens": {name: fixed[name]["tokens"] for name in ("single", "flat", "dynamic")},
        "gold_recall": {name: fixed[name].get("gold_recall") for name in ("single", "flat")},
    })

    musique = rows(ROOT / "benchmarks/musique/results/multilevel_branch_4hop3.jsonl")
    dynamic = next(row for row in musique if row["id"].startswith("4hop3__275416_"))
    assert dynamic["split"] == "evaluation"
    assert all(dynamic[name]["joint_em"] == 0 for name in ("single", "single_review", "flat"))
    assert dynamic["nested"]["joint_em"] == 1
    assert dynamic["nested"]["child_spawns"] >= 1
    print("dynamic_case", {
        "id": dynamic["id"],
        "joint_em": {name: dynamic[name]["joint_em"] for name in ("single", "single_review", "flat", "nested")},
        "tokens": {name: dynamic[name]["tokens"] for name in ("single", "single_review", "flat", "nested")},
        "support_recall": {name: dynamic[name]["support_recall"] for name in ("single", "single_review", "flat", "nested")},
        "child_spawns": dynamic["nested"]["child_spawns"],
    })


if __name__ == "__main__":
    main()
