"""Build human-review tables from an already completed held-out evaluation."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

from permission_intent_model import binary_metrics
from train_intent_contrast import ROOT, read_json, read_rows, write_json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ml/data/permission_intent_v3")
    args = parser.parse_args()
    evaluation = args.run / "evaluation"
    report = read_json(evaluation / "comparison_report.json")  # Must exist before any test parsing.
    versions = {name: {r["id"]: r for r in read_rows(evaluation / ("reserved_" + name + "_predictions.jsonl"))}
                for name in ("v2_original", "v3_semantic", "v3_policy")}
    rows = read_rows(args.data_dir / "reserved_test.jsonl")
    protocol = read_json(args.run / "checkpoint/protocol.json")
    per_capability = {}
    for cap in protocol["capability_ids"]:
        entry = {}
        for state, expected, predicted in (("required", "expected_required", "required_capabilities"),
                                           ("forbidden", "expected_forbidden", "forbidden_capabilities")):
            gold = np.array([cap in versions["v3_policy"][r["id"]][expected] for r in rows])
            actual = np.array([cap in versions["v3_policy"][r["id"]][predicted] for r in rows])
            entry[state] = binary_metrics(actual, gold)
            entry[state]["gold_support"] = int(gold.sum())
        per_capability[cap] = entry
    errors = Counter(e for p in versions["v3_policy"].values() for e in p["errors"])
    clarification_gold = [p for p in versions["v3_policy"].values() if p["expected_intent"] == "ambiguous"]
    goals_passed = sum(v["passed"] for v in report["target_results"].values())
    goals_total = len(report["target_results"])
    analysis = {"scope": "Error analysis after final selection; not used for further tuning",
                "human_review": "pending", "error_sample_counts": dict(errors),
                "per_capability": per_capability,
                "ambiguous_semantic_correct": sum(p["intent_status"] == "ambiguous" for p in clarification_gold),
                "ambiguous_policy_requested_clarification": sum(p["decision"] == "clarify" for p in clarification_gold),
                "ambiguous_gold_samples": len(clarification_gold),
                "all_four_group_successes": report["datasets"]["reserved"]["v3_policy"]["scenario_groups_all_roles_and_paraphrases_correct"],
                "experiment_goals_passed": goals_passed, "experiment_goals_total": goals_total,
                "quality_status": ("Passed prospective experiment targets; human and phone acceptance still pending"
                                   if goals_passed == goals_total else
                                   "Failed prospective experiment targets; do not replace existing app profile")}
    write_json(args.run / "error_analysis.json", analysis)
    fields = ["id", "scenario_group_id", "text", "expected_required", "expected_forbidden", "expected_intent"]
    for version in versions:
        fields += [version + "_" + k for k in ("required", "forbidden", "intent", "decision", "errors")]
    fields += ["human_review", "human_decision", "human_notes"]
    with (args.run / "prediction_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            ref = versions["v3_policy"][row["id"]]
            view = {key: ref[key] for key in fields[:6]}
            for version, predictions in versions.items():
                p = predictions[row["id"]]
                for out, source in (("required", "required_capabilities"), ("forbidden", "forbidden_capabilities"),
                                    ("intent", "intent_status"), ("decision", "decision"), ("errors", "errors")):
                    view[version + "_" + out] = p[source]
            for k, value in list(view.items()):
                if isinstance(value, list):
                    view[k] = " ; ".join(value)
            view["human_review"] = "pending"
            writer.writerow(view)
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario_group_id"]].append(row)
    fields = ["scenario_group_id", "summary"]
    for role in ("a", "b"):
        fields += [role + "_" + k for k in ("text1", "text2", "required", "forbidden", "intent")]
    fields += ["human_review", "human_decision", "human_notes"]
    with (args.run / "reserved_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for gid, members in sorted(groups.items()):
            view = {"scenario_group_id": gid, "summary": members[0]["scenario_summary"], "human_review": "pending"}
            for role in ("a", "b"):
                selected = [r for r in members if r["contrast_role"] == role]
                view.update({role + "_text1": selected[0]["text"], role + "_text2": selected[1]["text"],
                             role + "_required": " ; ".join(selected[0]["required_capabilities"]),
                             role + "_forbidden": " ; ".join(selected[0]["forbidden_capabilities"]),
                             role + "_intent": selected[0]["intent_status"]})
            writer.writerow(view)
    print("Saved 280-row prediction review and 70-group contrast review; human review remains pending")


if __name__ == "__main__":
    main()
