"""Human-review artifacts for an already completed locked v4 evaluation."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np

from permission_intent_model import binary_metrics
from train_intent_contrast import read_json, read_rows, write_json
from train_intent_v4 import DATA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.run / "evaluation"
    comparison = read_json(evaluation / "comparison_report.json")  # Must precede test parsing.
    rows = read_rows(DATA / "reserved_test.jsonl")
    versions = {name: {r["id"]: r for r in read_rows(evaluation / ("fresh_reserved_" + name + "_predictions.jsonl"))}
                for name in ("v3_original_policy", "v3_recalibrated_policy", "v4_semantic", "v4_policy")}
    selected = versions["v4_policy"]
    protocol = read_json(args.run / "checkpoint/protocol.json")
    per_capability = {}
    for cap in protocol["capability_ids"]:
        per_capability[cap] = {}
        for state, expected, actual in (("required", "expected_required", "required_capabilities"),
                                        ("forbidden", "expected_forbidden", "forbidden_capabilities")):
            gold = np.array([cap in selected[r["id"]][expected] for r in rows])
            predicted = np.array([cap in selected[r["id"]][actual] for r in rows])
            per_capability[cap][state] = {**binary_metrics(predicted, gold), "gold_support": int(gold.sum())}
    unclear = [p for p in selected.values() if p["expected_intent"] == "ambiguous"]
    denied = [p for p in selected.values() if p["expected_forbidden"]]
    violated = sum("explicit_prohibition_violation" in p["errors"] for p in denied)
    passed = sum(r["passed"] for r in comparison["target_results"].values())
    total = len(comparison["target_results"])
    analysis = {"human_review": "pending", "scope": "After final held-out evaluation; not used for model selection",
                "error_sample_counts": dict(Counter(e for p in selected.values() for e in p["errors"])),
                "per_capability": per_capability, "ambiguous_gold_samples": len(unclear),
                "ambiguous_semantic_correct": sum(p["intent_status"] == "ambiguous" for p in unclear),
                "ambiguous_policy_requested_clarification": sum(p["decision"] == "clarify" for p in unclear),
                "explicitly_forbidden_samples": len(denied), "violations_among_explicit_bans": violated,
                "conditional_violation_rate": violated / len(denied) if denied else None,
                "experiment_goals_passed": passed, "experiment_goals_total": total,
                "quality_status": "Prospective targets passed; human and phone acceptance pending" if passed == total
                                  else "Prospective targets not all met; keep existing phone profile"}
    write_json(args.run / "error_analysis.json", analysis)
    fields = ["id", "family_id", "text", "ambiguity_type", "expected_required", "expected_forbidden", "expected_intent"]
    for version in versions:
        fields += [version + "_" + f for f in ("required", "forbidden", "intent", "decision", "errors")]
    fields += ["human_review", "human_decision", "human_notes"]
    with (args.run / "prediction_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            ref = selected[row["id"]]
            view = {f: ref.get(f) for f in fields[:7]}
            view["ambiguity_type"] = row["ambiguity_type"]
            for version, predictions in versions.items():
                p = predictions[row["id"]]
                for name, key in (("required", "required_capabilities"), ("forbidden", "forbidden_capabilities"),
                                  ("intent", "intent_status"), ("decision", "decision"), ("errors", "errors")):
                    view[version + "_" + name] = p[key]
            for k, v in list(view.items()):
                if isinstance(v, list):
                    view[k] = " ; ".join(v)
            view.update(human_review="pending", human_decision="", human_notes="")
            writer.writerow(view)
    families = defaultdict(list)
    for row in rows:
        families[row["family_id"]].append(row)
    fields = ["family_id", "text1", "text2", "intent_status", "ambiguity_type", "required_capabilities",
              "forbidden_capabilities", "human_review", "human_decision", "human_notes"]
    with (args.run / "reserved_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for fid, members in sorted(families.items()):
            view = {f: members[0].get(f) for f in fields}
            view.update(text1=members[0]["text"], text2=members[1]["text"], human_review="pending", human_decision="", human_notes="")
            for k, v in list(view.items()):
                if isinstance(v, list):
                    view[k] = " ; ".join(v)
            writer.writerow(view)
    product_path = DATA / "product_examples.jsonl"
    if product_path.exists():
        products = read_rows(product_path)
        views = {name: {r["id"]: r for r in read_rows(evaluation / ("product_examples_" + name + "_predictions.jsonl"))}
                 for name in ("v3_original_policy", "v4_semantic", "v4_policy")}
        fields = ["id", "text", "expected_required", "expected_forbidden", "expected_intent"]
        for name in views:
            fields += [name + "_" + key for key in ("required", "forbidden", "intent", "decision", "errors")]
        fields += ["scope", "human_review", "human_decision", "human_notes"]
        with (args.run / "product_example_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in products:
                view = {key: views["v4_policy"][row["id"]].get(key) for key in fields[:5]}
                for name, predictions in views.items():
                    pred = predictions[row["id"]]
                    for label, key in (("required", "required_capabilities"), ("forbidden", "forbidden_capabilities"),
                                       ("intent", "intent_status"), ("decision", "decision"), ("errors", "errors")):
                        view[name + "_" + label] = pred[key]
                for key, value in list(view.items()):
                    if isinstance(value, list):
                        view[key] = " ; ".join(value)
                view.update(scope="Assistant-curated illustration; not independent user or blind acceptance evidence",
                            human_review="pending", human_decision="", human_notes="")
                writer.writerow(view)
    print({"prediction_review_rows": len(rows), "family_review_rows": len(families), "human_review": "pending"})


if __name__ == "__main__":
    main()
