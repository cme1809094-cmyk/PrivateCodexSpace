"""Fine-tune pinned TinyBERT on reviewed candidate data; export standard ONNX.

Experimental text-only task. This does not install models in the HarmonyOS app.
"""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

import fetch_tinybert
import prepare_permission_intent_expansion as data_validation
from permission_intent_model import (INTENTS, STATES, PermissionIntentModel, balanced_weights,
                                     calibrate, decode, encode, labels, load_checkpoint, metrics,
                                     objective, prediction_records, probabilities, save_checkpoint,
                                     selection_score)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare_data(data_dir, seed, capabilities):
    values = data_validation.inspect(data_dir, require_cross_review=True)
    audit = values[-1]
    if audit["errors"]:
        raise ValueError("Data validation failed: " + "; ".join(audit["errors"]))
    manifest = json.loads((data_dir / "split_manifest.json").read_text())
    splits, excluded = {}, {}
    for short, source in (("train", "train_candidate"), ("validation", "validation_candidate"), ("test", "test_candidate")):
        item = manifest["splits"][source]
        path = data_dir / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Stale candidate split: " + source)
        rows = read_rows(path)
        excluded[short] = [r["id"] for r in rows if "context_required" in r["tags"]]
        splits[short] = [r for r in rows if "context_required" not in r["tags"]]
    before = {name: [r["id"] for r in rows] for name, rows in splits.items()}
    families = defaultdict(list)
    for row in splits["train"]:
        families[row["family_id"]].append(row)
    present = {r["intent_status"] for r in splits["validation"]}
    moved, moving = [], set()
    required_before = Counter(c for r in splits["train"] for c in r["required_capabilities"])
    forbidden_before = Counter(c for r in splits["train"] for c in r["forbidden_capabilities"])
    for missing in [status for status in INTENTS if status not in present]:
        candidates = [fid for fid, members in families.items() if members[0]["intent_status"] == missing]
        candidates.sort(key=lambda fid: hashlib.sha256(f"{seed}:{fid}".encode()).hexdigest())
        selected = []
        for fid in candidates:
            chosen = moving | {fid}
            remaining = [r for r in splits["train"] if r["family_id"] not in chosen]
            req = Counter(c for r in remaining for c in r["required_capabilities"])
            deny = Counter(c for r in remaining for c in r["forbidden_capabilities"])
            if any(required_before[c] and not req[c] or forbidden_before[c] and not deny[c] for c in capabilities):
                continue
            moving.add(fid)
            selected.append(fid)
            if len(selected) == 3:
                break
        if len(selected) != 3:
            raise ValueError("Cannot hold out three families for validation status: " + missing)
        moved.append({"intent_status": missing, "family_ids": selected,
                      "ids": [r["id"] for fid in selected for r in families[fid]]})
    splits["validation"] += [r for r in splits["train"] if r["family_id"] in moving]
    splits["train"] = [r for r in splits["train"] if r["family_id"] not in moving]
    for name in splits:
        splits[name] = sorted(splits[name], key=lambda r: r["id"])
    family_sets = {name: {r["family_id"] for r in rows} for name, rows in splits.items()}
    if any(family_sets[a] & family_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Training run has cross-split family leakage")
    if [r["id"] for r in splits["test"]] != sorted(before["test"]):
        raise ValueError("Test membership changed")
    metadata = {
        "source": str(data_dir), "source_validation_sha256": audit["input_sha256"],
        "candidate_manifest_sha256": sha256(data_dir / "split_manifest.json"),
        "human_review": "pending", "frozen_test_set": False, "reference_only_rows_used": 0,
        "input_fields": ["text"], "context_required_excluded_ids": excluded,
        "validation_augmentation": moved,
        "augmentation_reason": "Original validation candidates omit ambiguous and no_device_capability. Hold out three intact train families per missing status, using development data only; retain training capability coverage. Test remains unchanged.",
        "splits": {name: {"rows": len(rows), "families": len(family_sets[name]),
                           "intent_counts": dict(Counter(r["intent_status"] for r in rows)),
                           "ids": [r["id"] for r in rows]} for name, rows in splits.items()},
        "warnings": audit["warnings"],
        "limitations": ["Synthetic data, human review pending; internal candidate evaluation only",
                        "Candidate test has incomplete explicit prohibition coverage",
                        "Newly held-out development families are not a separately human-verified benchmark",
                        "No target/slot/sequence or free-text clarification generation is trained"]
    }
    return splits, metadata


def summarize(cap_probs, intent_probs, targets, intent_targets, rows, thresholds, gate=True):
    states, statuses = decode(cap_probs, intent_probs, thresholds, gate=gate)
    return metrics(states, statuses, targets.numpy(), intent_targets.numpy(), rows)


def export_models(model, tokenizer, test_rows, protocol, output, batch_size):
    directory = output / "onnx"
    directory.mkdir(exist_ok=True)
    report = {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(torch.get_num_threads(), 4)
    options.inter_op_num_threads = 1
    model.eval()
    for length in (16, 32, 64):
        dummy = tokenizer("测试", padding="max_length", max_length=length, return_tensors="pt",
                          return_token_type_ids=False)
        inputs = tuple(dummy[key].to(torch.int32) for key in ("input_ids", "attention_mask"))
        path = directory / f"permission_intent_{length}.onnx"
        torch.onnx.export(model, inputs, path, input_names=["input_ids", "attention_mask"],
                          output_names=["capability_logits", "intent_logits"], opset_version=13, dynamo=False)
        onnx.checker.check_model(str(path))
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        selected = [r for r in test_rows if len(tokenizer(r["text"], truncation=False)["input_ids"]) <= length]
        encoded = encode(tokenizer, selected, length)
        maximum = [0.0, 0.0]
        mismatch = []
        vectors = []
        with torch.inference_mode():
            for i, row in enumerate(selected):
                args = {key: value[i:i + 1].to(torch.int32) for key, value in encoded.items()}
                reference = [value.numpy() for value in model(**args)]
                actual = session.run(None, {key: value.numpy() for key, value in args.items()})
                for j, (left, right) in enumerate(zip(reference, actual)):
                    maximum[j] = max(maximum[j], float(np.max(np.abs(left - right))))
                    np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-4)
                ref_prob = [torch.from_numpy(v).softmax(-1).numpy() for v in reference]
                actual_prob = [torch.from_numpy(v).softmax(-1).numpy() for v in actual]
                ref_state, ref_status = decode(*ref_prob, protocol["thresholds"])
                out_state, out_status = decode(*actual_prob, protocol["thresholds"])
                if not np.array_equal(ref_state, out_state) or not np.array_equal(ref_status, out_status):
                    mismatch.append(row["id"])
                vectors.append({"id": row["id"], "text": row["text"],
                                "input_ids": args["input_ids"][0].tolist(),
                                "attention_mask": args["attention_mask"][0].tolist(),
                                "capability_logits": reference[0][0].tolist(), "intent_logits": reference[1][0].tolist()})
        if mismatch:
            raise ValueError(f"ONNX decoded output differs in bucket {length}: {mismatch}")
        vector_path = directory / f"reference_{length}.jsonl"
        vector_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in vectors))
        report[str(length)] = {"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                               "sha256": sha256(path), "checked_rows": len(selected),
                               "capability_max_abs_logit_error": maximum[0], "intent_max_abs_logit_error": maximum[1],
                               "decoded_mismatch_ids": mismatch, "scope": "PyTorch versus ONNX Runtime CPU; not OMC or NPU"}
        print(json.dumps({"export_bucket": length, **report[str(length)]}), flush=True)
    write_json(output / "onnx_verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ml/data/permission_intent_v2/expansion")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--learning-rate", type=float, default=7e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty output directory; refusing to overwrite a training run")
    args.output.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    for name, expected in fetch_tinybert.FILES.items():
        if sha256(args.base / name) != expected:
            raise ValueError("Pinned pretrained source mismatch: " + name)
    catalog = json.loads((args.data_dir.parent / "capability_catalog.json").read_text())
    capabilities = [cap["id"] for cap in catalog["capabilities"]]
    splits, data_meta = prepare_data(args.data_dir, args.seed, capabilities)
    split_directory = args.output / "run_splits"
    split_directory.mkdir()
    for name, rows in splits.items():
        path = split_directory / (name + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        data_meta["splits"][name]["sha256"] = sha256(path)
    write_json(args.output / "data_manifest.json", data_meta)
    tokenizer = BertTokenizerFast.from_pretrained(args.base, local_files_only=True)
    encoded = {name: encode(tokenizer, rows, 64) for name, rows in splits.items()}
    gold = {name: labels(rows, capabilities) for name, rows in splits.items()}
    train_targets, train_intents = gold["train"]
    capability_weights, cap_counts = balanced_weights(train_targets, len(STATES), power=1.0)
    intent_weights, intent_counts = balanced_weights(train_intents, len(INTENTS))
    if not torch.all(cap_counts > 0) or not torch.all(intent_counts > 0):
        raise ValueError("Training must cover every capability state and all intent classes")
    config = BertConfig.from_pretrained(args.base, local_files_only=True)
    config._attn_implementation = "eager"
    model = PermissionIntentModel(config, len(capabilities))
    encoder, loading = BertModel.from_pretrained(args.base, config=config, local_files_only=True,
                                                attn_implementation="eager", output_loading_info=True)
    if loading["missing_keys"] or loading.get("mismatched_keys"):
        raise ValueError("Encoder weights missing or mismatched: " + str(loading))
    model.encoder = encoder
    if args.initial_checkpoint:
        initial, _, initial_protocol = load_checkpoint(args.initial_checkpoint)
        if initial_protocol["capability_ids"] != capabilities:
            raise ValueError("Initial checkpoint capability order mismatch")
        model.load_state_dict(initial.state_dict(), strict=True)
        del initial
    parameter_groups = []
    for prefix, learning_rate in (("encoder.", args.learning_rate), ("heads", args.head_learning_rate)):
        for decay in (True, False):
            parameters = [p for name, p in model.named_parameters()
                          if (name.startswith("encoder.") == (prefix == "encoder."))
                          and (not (name.endswith("bias") or "LayerNorm.weight" in name)) == decay]
            parameter_groups.append({"params": parameters, "lr": learning_rate, "weight_decay": 0.01 if decay else 0.0})
    optimizer = torch.optim.AdamW(parameter_groups)
    total_steps = math.ceil(len(splits["train"]) / args.batch_size) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
    protocol = {
        "schema_version": 2, "task": "permission_intent_32_tristate_text_only",
        "base_model": fetch_tinybert.MODEL_ID, "base_revision": fetch_tinybert.REVISION,
        "capability_ids": capabilities, "capability_names": {cap["id"]: cap["name"] for cap in catalog["capabilities"]},
        "states": STATES, "intent_statuses": INTENTS, "max_length": 64,
        "thresholds": {"required": 0.5, "forbidden": 0.5},
        "required_gate": "Only in_scope status may emit required capabilities; forbidden states remain available in other statuses",
        "model_inputs": ["input_ids", "attention_mask"],
        "model_outputs": {"capability_logits": [1, len(capabilities), len(STATES)], "intent_logits": [1, len(INTENTS)]},
        "generates_scope_slots_or_question_text": False, "compatible_with_existing_five_label_app": False,
        "human_review": "pending", "candidate_test_frozen": False,
    }
    training_config = {**vars(args), "base": str(args.base), "output": str(args.output), "data_dir": str(args.data_dir),
                       "initial_checkpoint": str(args.initial_checkpoint) if args.initial_checkpoint else None,
                       "initial_checkpoint_sha256": sha256(args.initial_checkpoint / "model.safetensors") if args.initial_checkpoint else None,
                       "max_length": 64, "loss": "inverse-frequency-weighted capability cross entropy + 0.5 sqrt-frequency-weighted intent cross entropy",
                       "selection": "Validation-only score: 0.5 required F1 +0.35 forbidden F1 +0.15 intent macro F1, after tuning two global confidence thresholds; validation loss is diagnostic only",
                       "post_selection_calibration": "Selected epoch retains its validation-only confidence thresholds",
                       "capability_class_counts": cap_counts.tolist(), "capability_class_weights": capability_weights.tolist(),
                       "intent_class_counts": intent_counts.tolist(), "intent_class_weights": intent_weights.tolist(),
                       "trainable_parameters": sum(p.numel() for p in model.parameters()),
                       "source_sha256": {name: sha256(ROOT / "ml" / name) for name in
                                          ("train_permission_intent_v2.py", "permission_intent_model.py")},
                       "encoder_loading_info": loading}
    write_json(args.output / "training_config.json", training_config)
    print(json.dumps({"event": "start", "splits": {s: len(r) for s, r in splits.items()},
                      "parameters": training_config["trainable_parameters"], "threads": args.threads}), flush=True)
    best, bad_epochs = None, 0
    history = args.output / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        loss_sum, seen = 0.0, 0
        for indices in torch.randperm(len(splits["train"])).split(args.batch_size):
            batch = {key: value[indices] for key, value in encoded["train"].items()}
            cap_logits, intent_logits = model(**batch)
            loss = objective(cap_logits, intent_logits, train_targets[indices], train_intents[indices],
                             capability_weights, intent_weights)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(indices)
            seen += len(indices)
        cap_probs, intent_probs = probabilities(model, encoded["validation"], args.batch_size)
        result = summarize(cap_probs, intent_probs, *gold["validation"], splits["validation"], protocol["thresholds"])
        validation_loss = float(objective(torch.from_numpy(cap_probs).clamp_min(1e-12).log(),
                                          torch.from_numpy(intent_probs).clamp_min(1e-12).log(),
                                          *gold["validation"], capability_weights, intent_weights))
        epoch_thresholds, calibrated = calibrate(cap_probs, intent_probs,
                                                 gold["validation"][0].numpy(), gold["validation"][1].numpy())
        score = selection_score(calibrated)
        if best is None or score > best[0] + 1e-5:
            best = (score, epoch, copy.deepcopy(model.state_dict()), calibrated, epoch_thresholds)
            bad_epochs = 0
            recovery_protocol = {**protocol, "thresholds": epoch_thresholds, "selected_epoch": epoch}
            save_checkpoint(model, tokenizer, args.output / "best_checkpoint", recovery_protocol)
        else:
            bad_epochs += 1
        entry = {"epoch": epoch, "training_loss": loss_sum / seen, "validation_score": score,
                 "validation_loss": validation_loss, "validation_metric_score": selection_score(result),
                 "validation": calibrated, "uncalibrated_validation": result, "thresholds": epoch_thresholds,
                 "seconds": time.time() - epoch_start, "best_epoch": best[1]}
        with history.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps({"epoch": epoch, "loss": round(loss_sum / seen, 5), "validation_score": round(score, 5),
                          "required_f1": round(calibrated["required"]["f1"], 4), "forbidden_f1": round(calibrated["forbidden"]["f1"], 4),
                          "joint_exact": calibrated["joint_states_and_intent_exact"], "best_epoch": best[1],
                          "seconds": round(time.time() - epoch_start, 1)}), flush=True)
        if epoch >= args.min_epochs and bad_epochs >= args.patience:
            break
    model.load_state_dict(best[2], strict=True)
    thresholds = best[4]
    protocol["thresholds"] = thresholds
    protocol["selected_epoch"] = best[1]
    checkpoint = args.output / "checkpoint"
    save_checkpoint(model, tokenizer, checkpoint, protocol)
    reloaded, reloaded_tokenizer, reloaded_protocol = load_checkpoint(checkpoint)
    model.eval()
    with torch.inference_mode():
        sample = {key: value[:2] for key, value in encoded["validation"].items()}
        for before, after in zip(model(**sample), reloaded(**sample)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    results, prediction_sets = {}, {}
    for name, rows in splits.items():
        cap_probs, intent_probs = probabilities(reloaded, encoded[name], args.batch_size)
        result = summarize(cap_probs, intent_probs, *gold[name], rows, thresholds)
        result["raw_without_status_gate"] = summarize(cap_probs, intent_probs, *gold[name], rows, thresholds, gate=False)
        per_capability = {}
        pred, status = decode(cap_probs, intent_probs, thresholds)
        from permission_intent_model import binary_metrics
        for i, cid in enumerate(capabilities):
            per_capability[cid] = {"required": binary_metrics(pred[:, i] == 1, gold[name][0].numpy()[:, i] == 1),
                                   "forbidden": binary_metrics(pred[:, i] == 2, gold[name][0].numpy()[:, i] == 2)}
        result["per_capability"] = per_capability
        results[name] = result
        predictions = list(prediction_records(rows, cap_probs, intent_probs, protocol))
        for row, prediction in zip(rows, predictions):
            prediction.update(family_id=row["family_id"], expected_required=row["required_capabilities"],
                              expected_forbidden=row["forbidden_capabilities"], expected_intent=row["intent_status"])
        prediction_sets[name] = predictions
        (args.output / (name + "_predictions.jsonl")).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions))
    report = {
        "schema_version": 2, "stage": "experimental_text_only_retraining", "model_id": fetch_tinybert.MODEL_ID,
        "revision": fetch_tinybert.REVISION, "selected_epoch": best[1], "epochs_run": epoch,
        "thresholds": thresholds, "max_length": 64, "splits": {s: len(r) for s, r in splits.items()},
        "metrics": results, "checkpoint_roundtrip_exact": True, "human_review": "pending",
        "test_is_frozen_independent_benchmark": False, "model_selected_using_test": False,
        "old_five_label_app_updated": False, "omc_compiled": False, "phone_npu_verified": False,
        "scope_constraints_or_question_generation_evaluated": False,
        "limitations": data_meta["limitations"], "elapsed_seconds_before_export": time.time() - start_time,
    }
    write_json(args.output / "training_report.json", report)
    fields = ["id", "family_id", "text", "expected_required", "required_capabilities", "expected_forbidden",
              "forbidden_capabilities", "expected_intent", "intent_status", "needs_clarification"]
    with (args.output / "test_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in prediction_sets["test"]:
            writer.writerow({key: " ; ".join(item[key]) if isinstance(item[key], list) else item[key] for key in fields})
    report["onnx_verification"] = export_models(reloaded, reloaded_tokenizer, splits["test"], reloaded_protocol,
                                                args.output, args.batch_size)
    report["elapsed_seconds"] = time.time() - start_time
    write_json(args.output / "training_report.json", report)
    artifact_manifest = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in sorted(args.output.rglob("*")) if p.is_file()}
    write_json(args.output / "artifact_manifest.json", artifact_manifest)
    print(json.dumps({"event": "complete", "selected_epoch": best[1], "thresholds": thresholds,
                      "test_required_f1": results["test"]["required"]["f1"],
                      "test_forbidden_f1": results["test"]["forbidden"]["f1"],
                      "test_joint_exact": results["test"]["joint_states_and_intent_exact"],
                      "seconds": round(report["elapsed_seconds"], 1)}), flush=True)


if __name__ == "__main__":
    main()
