"""Final locked v4 comparison; the fresh holdout is read only after selection."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from evaluate_intent_contrast import evaluate_version
from intent_contrast import calibrate_policy
from intent_optimization import calibrate_v4, subtype_results
from permission_intent_model import decode, encode, labels, load_checkpoint, probabilities
from train_intent_contrast import ROOT, read_json, read_rows, sha256, write_json
from train_intent_v4 import DATA, load_development


def check_profile(checkpoint):
    lock = read_json(checkpoint.parent / "selection_lock.json")
    for name, expected in lock["checkpoint_files_sha256"].items():
        if sha256(checkpoint / name) != expected:
            raise ValueError("Selected profile changed: " + name)
    return lock


def confidence_report(probabilities, statuses):
    p = probabilities[:, 0]
    expected = statuses == 0
    error = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if selected.any():
            error += float(selected.mean() * abs(p[selected].mean() - expected[selected].mean()))
    confident = p >= 0.95
    return {"in_scope_ece_10_bins": error, "in_scope_p_ge_095": int(confident.sum()),
            "incorrect_in_scope_p_ge_095": int((confident & ~expected).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a completed held-out evaluation")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    lock = check_profile(args.run / "checkpoint")
    check_profile(args.baseline)
    splits, ids, hashes = load_development()  # Validates hashes, does not parse the fresh test.
    model, tokenizer, protocol = load_checkpoint(args.run / "checkpoint")
    old, old_tokenizer, original = load_checkpoint(args.baseline)
    if protocol["capability_ids"] != ids or original["capability_ids"] != ids:
        raise ValueError("Capability order differs")
    if tokenizer.get_vocab() != old_tokenizer.get_vocab():
        raise ValueError("Tokenizer vocabulary differs")
    validation = splits["validation"]
    vt, vs = labels(validation, ids)
    vp, vi = probabilities(old, encode(tokenizer, validation, 64), 32)
    recalibrated = copy.deepcopy(original)
    recalibrated["thresholds"], _ = calibrate_v4(vp, vi, vt.numpy(), vs.numpy())
    recalibrated["policy"], _ = calibrate_policy(vp, vi, vt.numpy(), vs.numpy(), validation, recalibrated["thresholds"])
    baseline_lock = {"baseline_profile": read_json(args.baseline.parent / "selection_lock.json"),
                     "recalibrated_protocol": recalibrated, "validation_ids": [r["id"] for r in validation],
                     "reserved_opened": False, "data_hashes": hashes}
    write_json(args.output / "baseline_validation_lock.json", baseline_lock)
    # First new test parsing in this training/evaluation workflow.
    fresh = read_rows(DATA / "reserved_test.jsonl")
    regression = read_rows(ROOT / "ml/data/permission_intent_v3/reserved_test.jsonl")
    manifest = read_json(ROOT / "ml/results/tinybert-intent-v2-20261118/data_manifest.json")
    old_ids = set(manifest["splits"]["test"]["ids"])
    legacy = [r for r in read_rows(ROOT / "ml/data/permission_intent_v2/expansion/all_instructions.jsonl") if r["id"] in old_ids]
    report = {"scope": "AI-authored and independently AI-reviewed fresh cases; human review pending. All 32 output capabilities scored.",
              "old280_scope": "Previously inspected regression only; excluded from v4 train/validation/model selection",
              "product_examples_scope": "Assistant-curated development examples; may resemble known tasks; not independent user evidence or held-out acceptance",
              "selection_lock": lock, "fresh_test_sha256": sha256(DATA / "reserved_test.jsonl"),
              "baseline_validation_lock_sha256": sha256(args.output / "baseline_validation_lock.json"),
              "further_tuning_on_fresh_test": False, "datasets": {}}
    datasets = [("fresh_reserved", fresh), ("known_v3_regression", regression), ("legacy_v2_regression", legacy)]
    if (DATA / "product_examples.jsonl").exists():
        datasets.append(("product_examples", read_rows(DATA / "product_examples.jsonl")))
    for name, rows in datasets:
        encoded = encode(tokenizer, rows, 64)
        targets, statuses = labels(rows, ids)
        targets, statuses = targets.numpy(), statuses.numpy()
        old_cp, old_ip = probabilities(old, encoded, 32)
        new_cp, new_ip = probabilities(model, encoded, 32)
        result = {}
        for version, cp, ip, config, guarded in (
                ("v3_original_semantic", old_cp, old_ip, original, False),
                ("v3_original_policy", old_cp, old_ip, original, True),
                ("v3_recalibrated_semantic", old_cp, old_ip, recalibrated, False),
                ("v3_recalibrated_policy", old_cp, old_ip, recalibrated, True),
                ("v4_semantic", new_cp, new_ip, protocol, False),
                ("v4_policy", new_cp, new_ip, protocol, True)):
            result[version] = evaluate_version(rows, cp, ip, targets, statuses, config, args.output,
                                               name + "_" + version, guarded)
            state, status = decode(cp, ip, config["thresholds"])
            result[version]["ambiguity_subtypes_semantic"] = subtype_results(state, status, targets, statuses, rows)
        result["v3_confidence"] = confidence_report(old_ip, statuses)
        result["v4_confidence"] = confidence_report(new_ip, statuses)
        report["datasets"][name] = result
    previous = read_json(ROOT / "ml/results/tinybert-intent-v3-20261118/evaluation/comparison_report.json")["datasets"]["reserved"]
    for new_name, old_name in (("v3_original_semantic", "v3_semantic"), ("v3_original_policy", "v3_policy")):
        reproduced = report["datasets"]["known_v3_regression"][new_name]
        for key in ("required", "forbidden", "intent_confusion_matrix", "overrequested_samples", "explicit_prohibition_violation_samples"):
            if reproduced[key] != previous[old_name][key]:
                raise ValueError("Original v3 baseline failed to reproduce: " + key)
    report["v3_original_regression_reproduced"] = True
    sem = report["datasets"]["fresh_reserved"]["v4_semantic"]
    pol = report["datasets"]["fresh_reserved"]["v4_policy"]
    observed = {"required_f1_min": sem["required"]["f1"], "forbidden_f1_min": sem["forbidden"]["f1"],
                "ambiguous_recall_min": sem["intent_per_class"]["ambiguous"]["recall"],
                "no_device_recall_min": sem["intent_per_class"]["no_device_capability"]["recall"],
                "out_of_scope_recall_min": sem["intent_per_class"]["out_of_scope"]["recall"],
                "policy_violation_rate_max": pol["explicit_prohibition_violation_rate"],
                "policy_overrequest_rate_max": pol["overrequested_sample_rate"],
                "policy_in_scope_coverage_min": pol["in_scope_acceptance_coverage"],
                "policy_required_recall_min": pol["required"]["recall"]}
    goals = read_json(DATA / "experiment_plan.json")["prospective_targets"]
    report["target_results"] = {name: {"threshold": value, "observed": observed[name],
                                       "passed": observed[name] >= value if name.endswith("_min") else observed[name] <= value}
                                for name, value in goals.items()}
    write_json(args.output / "comparison_report.json", report)
    print({"event": "fresh_evaluation_complete", "samples": len(fresh),
           "targets_passed": sum(t["passed"] for t in report["target_results"].values()), "targets_total": len(goals)})


if __name__ == "__main__":
    main()
