"""Locked comparison on fresh AI-reviewed contrasts; never selects on test results."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

import numpy as np
import torch

from intent_contrast import calibrate_policy, calibrate_semantics, decode_policy, policy_metrics
from permission_intent_model import INTENTS, decode, encode, labels, load_checkpoint, metrics, probabilities
from train_intent_contrast import ROOT, development_data, read_json, read_rows, sha256, write_json


def grouped_result(state, status, targets, statuses, rows):
    result = metrics(state, status, targets, statuses, rows)
    groups = defaultdict(list)
    correct = np.all(state == targets, axis=1) & (status == statuses)
    for i, row in enumerate(rows):
        groups[row.get("scenario_group_id", row["family_id"])].append(i)
    result["scenario_groups"] = len(groups)
    result["scenario_groups_all_roles_and_paraphrases_correct"] = sum(
        bool(correct[indices].all()) for indices in groups.values())
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        strata[row.get("status_stratum") or ("focus_capabilities" if "scenario_group_id" in row else "legacy")].append(i)
    result["by_status_stratum"] = {
        name: metrics(state[indices], status[indices], targets[indices], statuses[indices])
        for name, indices in strata.items()}
    return result


def evaluate_version(rows, cap_probs, intent_probs, targets, statuses, protocol, output, name, guarded):
    if guarded:
        state, status, accepted, decisions = decode_policy(
            cap_probs, intent_probs, protocol["thresholds"], protocol["policy"])
        result = policy_metrics(cap_probs, intent_probs, targets, statuses, rows,
                                protocol["thresholds"], protocol["policy"])
    else:
        state, status = decode(cap_probs, intent_probs, protocol["thresholds"])
        accepted = status == 0
        decisions = np.array(["accept", "clarify", "no_device_capability", "out_of_scope"])[status]
        result = {}
    result.update(grouped_result(state, status, targets, statuses, rows))
    ids = protocol["capability_ids"]
    with (output / (name + "_predictions.jsonl")).open("w") as stream:
        for i, row in enumerate(rows):
            errors = []
            if np.any((state[i] == 1) & (targets[i] == 2)):
                errors.append("explicit_prohibition_violation")
            if np.any((state[i] == 1) & (targets[i] != 1)):
                errors.append("extra_required")
            if np.any((state[i] != 1) & (targets[i] == 1)):
                errors.append("missing_required")
            if np.any((state[i] == 2) != (targets[i] == 2)):
                errors.append("forbidden_set_mismatch")
            if status[i] != statuses[i]:
                errors.append("intent_status_mismatch")
            item = {"id": row["id"], "text": row["text"], "family_id": row["family_id"],
                    "scenario_group_id": row.get("scenario_group_id"),
                    "expected_required": row["required_capabilities"],
                    "expected_forbidden": row["forbidden_capabilities"],
                    "expected_intent": row["intent_status"],
                    "required_capabilities": [c for c, s in zip(ids, state[i]) if s == 1],
                    "forbidden_capabilities": [c for c, s in zip(ids, state[i]) if s == 2],
                    "intent_status": INTENTS[status[i]], "decision": str(decisions[i]),
                    "accepted": bool(accepted[i]), "errors": errors}
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return result


def goal_results(semantic, policy, goals):
    observed = {"required_f1": semantic["required"]["f1"],
                "forbidden_f1": semantic["forbidden"]["f1"],
                "out_of_scope_recall": semantic["intent_per_class"]["out_of_scope"]["recall"],
                "no_device_capability_recall": semantic["intent_per_class"]["no_device_capability"]["recall"],
                "explicit_prohibition_violation_rate": policy["explicit_prohibition_violation_rate"],
                "overrequested_sample_rate": policy["overrequested_sample_rate"],
                "in_scope_acceptance_coverage": policy["in_scope_acceptance_coverage"],
                "required_recall": policy["required"]["recall"]}
    return {name: {**target, "observed": observed[name],
                   "passed": observed[name] >= target["value"] if target["direction"] == "minimum"
                   else observed[name] <= target["value"]}
            for name, target in goals["targets"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ml/data/permission_intent_v3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite an existing held-out evaluation")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    lock = read_json(args.run / "selection_lock.json")
    for file, expected in lock["checkpoint_files_sha256"].items():
        if sha256(args.run / "checkpoint" / file) != expected:
            raise ValueError("Final checkpoint profile changed: " + file)
    for file, key in (("model.safetensors", "weights_sha256"), ("protocol.json", "protocol_sha256")):
        if sha256(args.run / "checkpoint" / file) != lock[key]:
            raise ValueError("Final selection lock changed: " + file)
    freeze = read_json(args.data_dir / "freeze_manifest.json")
    if sha256(args.data_dir / "reserved_test.jsonl") != freeze["reserved_test_sha256"]:
        raise ValueError("Reserved bytes changed after freeze")
    new_model, tokenizer, protocol = load_checkpoint(args.run / "checkpoint")
    old_model, old_tokenizer, original = load_checkpoint(args.baseline)
    if original["capability_ids"] != protocol["capability_ids"]:
        raise ValueError("Capability ordering changed between baselines")
    if tokenizer.get_vocab() != old_tokenizer.get_vocab():
        raise ValueError("Baseline tokenizer differs")
    # Fit the old model's thresholds and abstention on the same development set.
    # No reserved prompts or gold labels have been parsed at this point.
    splits, hashes = development_data("full", args.data_dir, protocol["capability_ids"])
    val = splits["validation"]
    val_targets, val_statuses = labels(val, protocol["capability_ids"])
    vp, vi = probabilities(old_model, encode(tokenizer, val, 64))
    thresholds, _ = calibrate_semantics(vp, vi, val_targets.numpy(), val_statuses.numpy())
    policy, _ = calibrate_policy(vp, vi, val_targets.numpy(), val_statuses.numpy(), val, thresholds)
    recalibrated = copy.deepcopy(original)
    recalibrated.update(thresholds=thresholds, policy=policy)
    baseline_lock = {"checkpoint_sha256": sha256(args.baseline / "model.safetensors"),
                     "original_protocol_sha256": sha256(args.baseline / "protocol.json"),
                     "recalibrated_protocol": recalibrated, "validation_rows": len(val),
                     "validation_ids": [r["id"] for r in val], "data_hashes": hashes,
                     "reserved_test_opened": False,
                     "note": "Baseline weights are unchanged; thresholds/policy fit only on current validation."}
    write_json(args.output / "baseline_validation_lock.json", baseline_lock)
    # First parsing of the fresh reserved test in the root training/evaluation workflow.
    reserved = read_rows(args.data_dir / "reserved_test.jsonl")
    old_manifest = read_json(ROOT / "ml/results/tinybert-intent-v2-20261118/data_manifest.json")
    legacy_ids = set(old_manifest["splits"]["test"]["ids"])
    legacy = [r for r in read_rows(ROOT / "ml/data/permission_intent_v2/expansion/all_instructions.jsonl")
              if r["id"] in legacy_ids]
    report = {"reserved_test_sha256": freeze["reserved_test_sha256"], "selection_lock": lock,
              "baseline_validation_lock_sha256": sha256(args.output / "baseline_validation_lock.json"),
              "targets_sha256": sha256(args.data_dir / "experiment_targets.json"),
              "scope": "AI-authored, independently AI-reviewed focus-eight contrasts; human review pending. All 32 output capabilities scored, including extras outside the focus eight.",
              "legacy_scope": "Previously inspected development regression, not a new blind benchmark.",
              "further_selection_on_reserved": False, "datasets": {}}
    for dataset_name, rows in (("reserved", reserved), ("legacy_regression", legacy)):
        enc = encode(tokenizer, rows, 64)
        targets, statuses = labels(rows, protocol["capability_ids"])
        targets, statuses = targets.numpy(), statuses.numpy()
        old_probs, old_intents = probabilities(old_model, enc)
        new_probs, new_intents = probabilities(new_model, enc)
        results = {}
        for name, caps, intents, config, guarded in (
                ("v2_original", old_probs, old_intents, original, False),
                ("v2_recalibrated_semantic", old_probs, old_intents, recalibrated, False),
                ("v2_recalibrated_policy", old_probs, old_intents, recalibrated, True),
                ("v3_semantic", new_probs, new_intents, protocol, False),
                ("v3_policy", new_probs, new_intents, protocol, True)):
            results[name] = evaluate_version(rows, caps, intents, targets, statuses, config,
                                            args.output, dataset_name + "_" + name, guarded)
        report["datasets"][dataset_name] = results
    report["target_results"] = goal_results(report["datasets"]["reserved"]["v3_semantic"],
                                           report["datasets"]["reserved"]["v3_policy"],
                                           read_json(args.data_dir / "experiment_targets.json"))
    previous_report = args.baseline.parent / "training_report.json"
    if previous_report.exists():
        old = read_json(previous_report)["metrics"]["test"]
        reproduced = report["datasets"]["legacy_regression"]["v2_original"]
        for key in ("required", "forbidden", "overrequested_samples", "explicit_prohibition_violation_samples",
                    "joint_states_and_intent_exact", "intent_confusion_matrix"):
            if old[key] != reproduced[key]:
                raise ValueError("Original v2 legacy result failed to reproduce: " + key)
        report["original_v2_legacy_metrics_reproduced"] = True
    write_json(args.output / "comparison_report.json", report)
    print(json.dumps({"event": "held_out_evaluation_complete", "reserved_rows": len(reserved),
                      "goals_passed": sum(r["passed"] for r in report["target_results"].values()),
                      "goals_total": len(report["target_results"])}))


if __name__ == "__main__":
    main()
