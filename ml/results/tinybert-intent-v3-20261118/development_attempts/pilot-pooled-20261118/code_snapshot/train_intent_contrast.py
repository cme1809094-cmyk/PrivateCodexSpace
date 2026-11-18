"""Development-only head comparison and reviewed contrastive TinyBERT training.

Reserved prompts are loaded only after weights, thresholds and policy are locked.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

import fetch_tinybert
from permission_intent_model import (INTENTS, STATES, PermissionIntentModel, balanced_weights,
                                     decode, encode, labels, load_checkpoint, metrics, objective,
                                     prediction_records, probabilities, save_checkpoint)
from intent_contrast import (calibrate_policy, calibrate_semantics, contrast_score,
                             packed_group_batches, pair_margin_loss, policy_metrics)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def old_development():
    source = ROOT / "ml/data/permission_intent_v2/expansion/all_instructions.jsonl"
    manifest_path = ROOT / "ml/results/tinybert-intent-v2-20261118/data_manifest.json"
    manifest = read_json(manifest_path)
    by_id = {row["id"]: row for row in read_rows(source)}
    result = {name: [by_id[rid] for rid in manifest["splits"][name]["ids"]]
              for name in ("train", "validation")}
    return result, {"v2_all_sha256": sha256(source), "v2_run_manifest_sha256": sha256(manifest_path)}


def development_data(mode, data_dir, capability_ids):
    splits, hashes = old_development()
    if mode == "pilot":
        focus = set(capability_ids)
        splits = {name: [dict(row,
                              required_capabilities=[c for c in row["required_capabilities"] if c in focus],
                              forbidden_capabilities=[c for c in row["forbidden_capabilities"] if c in focus])
                         for row in rows] for name, rows in splits.items()}
    else:
        audit = read_json(data_dir / "ai_review_report.json")
        freeze = read_json(data_dir / "freeze_manifest.json")
        if audit["unresolved_errors"] or not freeze["frozen_before_training"]:
            raise ValueError("Reviewed data and a pre-training reserved-test freeze are required")
        if sha256(data_dir / "reserved_test.jsonl") != freeze["reserved_test_sha256"]:
            raise ValueError("Reserved-test bytes changed after freeze")
        for filename in ("a_photos_contacts.jsonl", "b_clipboard_recording.jsonl", "c_status_contrasts.jsonl"):
            path = data_dir / "shards" / filename
            if sha256(path) != audit["source_sha256"][filename]:
                raise ValueError("Stale v3 semantic review: " + filename)
            hashes[filename] = sha256(path)
            for row in read_rows(path):
                if row["review"]["cross_review"] != "reviewed":
                    raise ValueError("Incomplete independent review: " + row["id"])
                if row["split_candidate"] not in ("train_candidate", "validation_candidate"):
                    raise ValueError("Reserved example entered development")
                splits["train" if row["split_candidate"] == "train_candidate" else "validation"].append(row)
        hashes.update(audit_sha256=sha256(data_dir / "ai_review_report.json"),
                      freeze_sha256=sha256(data_dir / "freeze_manifest.json"),
                      reserved_test_sha256=freeze["reserved_test_sha256"])
    train_groups = {r.get("scenario_group_id", r["family_id"]) for r in splits["train"]}
    dev_groups = {r.get("scenario_group_id", r["family_id"]) for r in splits["validation"]}
    if train_groups & dev_groups:
        raise ValueError("Contrast group crosses development splits")
    ids = [r["id"] for rows in splits.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate development row IDs")
    return splits, hashes


def result_for(model, encoded, targets, statuses, rows, thresholds, batch_size):
    cap_probs, intent_probs = probabilities(model, encoded, batch_size)
    pred, status = decode(cap_probs, intent_probs, thresholds)
    return metrics(pred, status, targets.numpy(), statuses.numpy(), rows), cap_probs, intent_probs


def train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    for filename, expected in fetch_tinybert.FILES.items():
        if sha256(args.base / filename) != expected:
            raise ValueError("Pinned source mismatch: " + filename)
    contract = read_json(args.data_dir / "contract.json")
    catalog = read_json(ROOT / "ml/data/permission_intent_v2/capability_catalog.json")
    all_ids = [c["id"] for c in catalog["capabilities"]]
    capability_ids = contract["focus_capabilities"] if args.mode == "pilot" else all_ids
    splits, data_hashes = development_data(args.mode, args.data_dir, capability_ids)
    tokenizer = BertTokenizerFast.from_pretrained(args.base, local_files_only=True)
    encoded = {name: encode(tokenizer, rows, 64) for name, rows in splits.items()}
    gold = {name: labels(rows, capability_ids) for name, rows in splits.items()}
    cap_weights, cap_counts = balanced_weights(gold["train"][0], len(STATES), power=1.0)
    status_weights, status_counts = balanced_weights(gold["train"][1], len(INTENTS), power=1.0)
    if not torch.all(cap_counts > 0) or not torch.all(status_counts > 0):
        raise ValueError("Training coverage missing a capability state or intent class")
    config = BertConfig.from_pretrained(args.base, local_files_only=True)
    config.permission_intent_head = args.head
    config.cls_token_id, config.sep_token_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    config.hidden_dropout_prob = args.dropout
    model = PermissionIntentModel(config, len(capability_ids))
    encoder, loading = BertModel.from_pretrained(args.base, config=config, local_files_only=True,
                                                attn_implementation="eager", output_loading_info=True)
    if loading["missing_keys"] or loading.get("mismatched_keys"):
        raise ValueError("Encoder failed to load: " + str(loading))
    model.encoder = encoder
    by_id = {c["id"]: c for c in catalog["capabilities"]}
    descriptions = [by_id[c]["name"] + "。" + by_id[c]["definition"] for c in capability_ids]
    model.initialize_label_queries(tokenizer, descriptions)
    protocol = {"schema_version": 3, "task": "permission_intent_contrast_text_only", "head_type": args.head,
                "base_model": fetch_tinybert.MODEL_ID, "base_revision": fetch_tinybert.REVISION,
                "capability_ids": capability_ids, "capability_names": {c: by_id[c]["name"] for c in capability_ids},
                "states": STATES, "intent_statuses": INTENTS, "max_length": 64,
                "thresholds": {"required": 0.5, "forbidden": 0.5}, "focus_projection_only": args.mode == "pilot",
                "compatible_with_existing_five_label_app": False, "human_review": "pending",
                "generates_scope_slots_or_question_text": False,
                "model_outputs": {"capability_logits": [1, len(capability_ids), 3], "intent_logits": [1, 4]}}
    training = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "data_hashes": data_hashes, "rows": {k: len(v) for k, v in splits.items()},
                "source_sha256": {name: sha256(ROOT / "ml" / name) for name in
                                   ("train_intent_contrast.py", "intent_contrast.py", "permission_intent_model.py")},
                "capability_ids": capability_ids, "capability_class_counts": cap_counts.tolist(),
                "intent_class_counts": status_counts.tolist(), "class_weighting": "inverse frequency",
                "test_read_before_selection": False,
                "selection": "Development only: recall>=0.4 where possible, then 0.45 required F0.5 +0.30 forbidden F1 +0.25 intent macro F1 -0.10 prohibition violation rate",
                "query_initialization": "Pretrained contextual mean of semantic capability name/definition; not permission mappings or example answers"}
    write_json(args.output / "training_config.json", training)
    write_json(args.output / "run_splits.json", {name: [r["id"] for r in rows] for name, rows in splits.items()})
    parameters = []
    for encoder_group, lr in ((True, args.learning_rate), (False, args.head_learning_rate)):
        for decay in (True, False):
            params = [p for n, p in model.named_parameters() if n.startswith("encoder.") == encoder_group
                      and (not (n.endswith("bias") or "LayerNorm.weight" in n)) == decay]
            parameters.append({"params": params, "lr": lr, "weight_decay": 0.01 if decay else 0.0})
    optimizer = torch.optim.AdamW(parameters)
    generator = torch.Generator().manual_seed(args.seed)
    batches_per_epoch = len(list(packed_group_batches(splits["train"], args.batch_size, torch.Generator().manual_seed(args.seed))))
    total_steps = batches_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total_steps * 0.06)), total_steps)
    best, bad = None, 0
    print(json.dumps({"event": "start", "mode": args.mode, "head": args.head, "rows": training["rows"],
                      "capabilities": len(capability_ids), "parameters": sum(p.numel() for p in model.parameters())}), flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        losses = []
        for indices in packed_group_batches(splits["train"], args.batch_size, generator):
            batch = {key: value[indices] for key, value in encoded["train"].items()}
            cap_logits, intent_logits = model(**batch)
            targets, statuses = gold["train"][0][indices], gold["train"][1][indices]
            loss = objective(cap_logits, intent_logits, targets, statuses, cap_weights, status_weights)
            pairs = pair_margin_loss(cap_logits, intent_logits, targets, statuses,
                                     [splits["train"][i] for i in indices.tolist()])
            loss = loss + args.pair_loss_weight * pairs
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        cap_probs, intent_probs = probabilities(model, encoded["validation"], args.batch_size)
        thresholds, result = calibrate_semantics(cap_probs, intent_probs,
                                                 gold["validation"][0].numpy(), gold["validation"][1].numpy())
        score = contrast_score(result)
        if best is None or score > best[0] + 1e-5:
            best = (score, epoch, copy.deepcopy(model.state_dict()), thresholds, result)
            bad = 0
        else:
            bad += 1
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "validation_score": score,
                 "thresholds": thresholds, "validation": result, "best_epoch": best[1], "seconds": time.time() - epoch_start}
        with (args.output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps({"epoch": epoch, "loss": round(entry["loss"], 4), "score": round(score, 4),
                          "required_f1": round(result["required"]["f1"], 4),
                          "required_precision": round(result["required"]["precision"], 4),
                          "forbidden_f1": round(result["forbidden"]["f1"], 4),
                          "status_macro_f1": round(result["intent_macro_f1"], 4), "best_epoch": best[1],
                          "seconds": round(entry["seconds"], 1)}), flush=True)
        if epoch >= args.min_epochs and bad >= args.patience:
            break
    model.load_state_dict(best[2], strict=True)
    protocol.update(thresholds=best[3], selected_epoch=best[1])
    report = {"mode": args.mode, "head": args.head, "selected_epoch": best[1], "epochs_run": epoch,
              "thresholds": best[3], "validation": best[4], "validation_score": best[0],
              "human_review": "pending", "test_opened": False, "data_rows": training["rows"]}
    results = {}
    for name, rows in splits.items():
        result, cap_probs, intent_probs = result_for(model, encoded[name], *gold[name], rows, best[3], args.batch_size)
        results[name] = result
        if args.mode == "full" and name == "validation":
            policy, guarded = calibrate_policy(cap_probs, intent_probs, gold[name][0].numpy(), gold[name][1].numpy(), rows, best[3])
            protocol["policy"] = policy
            report["validation_policy"] = guarded
    report["metrics"] = results
    checkpoint = args.output / "checkpoint"
    save_checkpoint(model, tokenizer, checkpoint, protocol)
    restored, _, _ = load_checkpoint(checkpoint)
    with torch.inference_mode():
        sample = {k: v[:2] for k, v in encoded["validation"].items()}
        model.eval()
        for a, b in zip(model(**sample), restored(**sample)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    report["checkpoint_roundtrip_exact"] = True
    lock = {"weights_sha256": sha256(checkpoint / "model.safetensors"), "protocol_sha256": sha256(checkpoint / "protocol.json"),
            "selected_using": "development data only", "reserved_test_opened": False}
    write_json(args.output / "selection_lock.json", lock)
    write_json(args.output / "training_report.json", report)
    if args.mode == "full" and args.evaluate_reserved:
        evaluate_reserved(restored, tokenizer, protocol, args, report)
    report["seconds"] = time.time() - start
    write_json(args.output / "training_report.json", report)
    print(json.dumps({"event": "complete", "head": args.head, "mode": args.mode, "epoch": best[1],
                      "validation_score": best[0], "seconds": round(report["seconds"], 1)}), flush=True)


def evaluate_reserved(model, tokenizer, protocol, args, report):
    freeze = read_json(args.data_dir / "freeze_manifest.json")
    if sha256(args.data_dir / "reserved_test.jsonl") != freeze["reserved_test_sha256"]:
        raise ValueError("Reserved-test bytes changed after model selection")
    lock = read_json(args.output / "selection_lock.json")
    if sha256(args.output / "checkpoint/model.safetensors") != lock["weights_sha256"]:
        raise ValueError("Selected weights changed before test")
    rows = read_rows(args.data_dir / "reserved_test.jsonl")
    encoded = encode(tokenizer, rows, 64)
    targets, statuses = labels(rows, protocol["capability_ids"])
    cap_probs, intent_probs = probabilities(model, encoded, args.batch_size)
    state, status = decode(cap_probs, intent_probs, protocol["thresholds"])
    report["reserved_semantic"] = metrics(state, status, targets.numpy(), statuses.numpy(), rows)
    report["reserved_policy"] = policy_metrics(cap_probs, intent_probs, targets.numpy(), statuses.numpy(), rows,
                                                protocol["thresholds"], protocol["policy"])
    report["test_opened"] = True
    report["reserved_sha256"] = freeze["reserved_test_sha256"]
    report["reserved_scope"] = "Fresh held-out AI-generated focus-eight contrast cases; not a human-verified full-32 production benchmark"
    predictions = list(prediction_records(rows, cap_probs, intent_probs, protocol))
    for row, prediction in zip(rows, predictions):
        prediction.update(expected_required=row["required_capabilities"], expected_forbidden=row["forbidden_capabilities"],
                          expected_intent=row["intent_status"], scenario_group_id=row["scenario_group_id"])
    (args.output / "reserved_predictions.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--head", choices=("pooled", "label_attention"), required=True)
    parser.add_argument("--base", type=Path, default=Path('/workspace/work/tinybert/downloads'))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ml/data/permission_intent_v3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--pair-loss-weight", type=float, default=0.15)
    parser.add_argument("--evaluate-reserved", action="store_true")
    args = parser.parse_args()
    if args.mode == "pilot" and args.evaluate_reserved:
        raise ValueError("Pilot model selection must never open the reserved test")
    train(args)


if __name__ == "__main__":
    main()
