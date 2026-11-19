"""Audited composition training: data-only control versus gated token evidence.

New reserved prompts are not parsed by this trainer, even after training.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from intent_contrast import calibrate_policy, packed_group_batches, pair_margin_loss
from intent_optimization import (add_training_device_argument, calibrate_v4, focal_objective, optimization_score,
                                 resolve_training_device, training_runtime, trim_padding)
from permission_intent_model import (PermissionIntentModel, balanced_weights, decode, encode, labels,
                                    load_checkpoint, metrics, objective, probabilities, save_checkpoint)
from train_intent_contrast import ROOT, development_data, read_json, read_rows, sha256, write_json
from validate_permission_intent_v2 import normalized


DATA = ROOT / "ml/data/permission_intent_v4"


def load_development():
    freeze = read_json(DATA / "freeze_manifest.json")
    audit = read_json(DATA / "ai_review_report.json")
    if not freeze["frozen_before_training"] or audit["unresolved_errors"]:
        raise ValueError("V4 independent review and pre-training freeze are required")
    if (sha256(DATA / "ai_review_report.json") != freeze["audit_sha256"]
            or sha256(DATA / "experiment_plan.json") != freeze["plan_sha256"]):
        raise ValueError("V4 audit or experiment plan changed after freeze")
    if (len(set(audit["covered_validation_ids"])) != 224
            or len(set(audit["covered_reserved_ids"])) != 320
            or not audit["training_template_review"]["passed"]):
        raise ValueError("Incomplete v4 review")
    for name, expected in freeze["source_sha256"].items():
        path = ROOT / "ml/generate_intent_v4.py" if name == "generate_intent_v4.py" else DATA / name
        if sha256(path) != expected:
            raise ValueError("Frozen v4 source changed: " + name)
    # Only the reserved file's hash was read above, never its prompts or labels.
    catalog = read_json(ROOT / "ml/data/permission_intent_v2/capability_catalog.json")
    cap_ids = [c["id"] for c in catalog["capabilities"]]
    splits, legacy_hashes = development_data("full", ROOT / "ml/data/permission_intent_v3", cap_ids)
    for split, filename in (("train", "train.jsonl"), ("validation", "validation.jsonl")):
        fresh = read_rows(DATA / filename)
        if any(r["split"] != split for r in fresh):
            raise ValueError("Wrong source split")
        splits[split].extend(fresh)
    train_families = {r["family_id"] for r in splits["train"]}
    validation_families = {r["family_id"] for r in splits["validation"]}
    if train_families & validation_families:
        raise ValueError("Family crosses train and validation")
    texts = {split: {normalized(r["text"]) for r in rows} for split, rows in splits.items()}
    if texts["train"] & texts["validation"]:
        raise ValueError("Identical text crosses train and validation")
    return splits, cap_ids, {"v4_freeze": freeze, "legacy": legacy_hashes}


def initialize(initial, variant, tokenizer, cap_ids):
    initial_lock = initial.parent / "selection_lock.json"
    if initial_lock.exists():
        for name, expected in read_json(initial_lock)["checkpoint_files_sha256"].items():
            if sha256(initial / name) != expected:
                raise ValueError("Initial checkpoint profile changed: " + name)
    desired = "pooled" if variant == "pooled" else "pooled_label_attention"
    old, _, protocol = load_checkpoint(initial)
    if protocol["capability_ids"] != cap_ids:
        raise ValueError("Initial capability order differs")
    if old.head_type == desired:
        model = old
    else:
        if old.head_type != "pooled" or desired != "pooled_label_attention":
            raise ValueError("Unsupported profile transfer")
        config = copy.deepcopy(old.encoder.config)
        config.permission_intent_head = desired
        model = PermissionIntentModel(config, len(cap_ids))
        missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
        if set(missing) != {"label_queries", "label_gates"} or unexpected:
            raise ValueError("Transfer changed pretrained classifier parameters")
        lex = read_json(DATA / "lexicon.json")["capabilities"]
        descriptions = ["。".join(lex[c]["positive"]) for c in cap_ids]
        model.initialize_label_queries(tokenizer, descriptions)
    model.encoder.config.hidden_dropout_prob = 0.1
    model.encoder.config.attention_probs_dropout_prob = 0.1
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.1
    return model, protocol


def train(args):
    device = resolve_training_device(args.device)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    splits, cap_ids, data_hashes = load_development()
    _, tokenizer, _ = load_checkpoint(args.initial)
    encoded = {name: encode(tokenizer, rows, 64) for name, rows in splits.items()}
    gold = {name: labels(rows, cap_ids) for name, rows in splits.items()}
    power = 1.0 if args.variant == "pooled" else 0.5
    cap_weights, cap_counts = balanced_weights(gold["train"][0], 3, power=power)
    intent_weights, intent_counts = balanced_weights(gold["train"][1], 4, power=1.0)
    if not torch.all(cap_counts > 0) or not torch.all(intent_counts > 0):
        raise ValueError("Missing capability-state or intent training support")
    model, protocol = initialize(args.initial, args.variant, tokenizer, cap_ids)
    model.to(device)
    cap_weights, intent_weights = cap_weights.to(device), intent_weights.to(device)
    train_targets, train_statuses = (value.to(device) for value in gold["train"])
    protocol = copy.deepcopy(protocol)
    protocol.pop("policy", None)
    protocol.update(schema_version=4, task="permission_intent_v4_text_only", head_type=model.head_type,
                    human_review="pending", compatible_with_existing_five_label_app=False,
                    v4_variant=args.variant, old280_is_known_regression=True)
    source_names = ["train_intent_v4.py", "intent_optimization.py", "permission_intent_model.py", "intent_contrast.py"]
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "runtime": training_runtime(device),
              "rows": {k: len(v) for k, v in splits.items()}, "data_hashes": data_hashes,
              "initial_files_sha256": {p.name: sha256(p) for p in args.initial.iterdir() if p.is_file()},
              "source_sha256": {n: sha256(ROOT / "ml" / n) for n in source_names},
              "capability_state_counts": cap_counts.tolist(), "intent_counts": intent_counts.tolist(),
              "capability_weight_power": power, "intent_weight_power": 1.0,
              "training_padding": "Right padding trimmed within each batch, never text truncated",
              "reserved_prompts_read": False, "human_review": "pending"}
    write_json(args.output / "training_config.json", config)
    write_json(args.output / "run_splits.json", {s: [r["id"] for r in rows] for s, rows in splits.items()})
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for name in source_names:
        (snapshot / name).write_bytes((ROOT / "ml" / name).read_bytes())
    groups = []
    for encoder, lr in ((True, args.learning_rate), (False, args.head_learning_rate)):
        for decay in (True, False):
            params = [p for n, p in model.named_parameters() if n.startswith("encoder.") == encoder
                      and (not (n.endswith("bias") or "LayerNorm.weight" in n)) == decay]
            groups.append({"params": params, "lr": lr, "weight_decay": 0.01 if decay else 0})
    optimizer = torch.optim.AdamW(groups)
    generator = torch.Generator().manual_seed(args.seed)
    steps = len(list(packed_group_batches(splits["train"], args.batch_size, torch.Generator().manual_seed(args.seed))))
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * args.epochs * 0.05)), steps * args.epochs)
    best, bad = None, 0
    print(json.dumps({"event": "start", "variant": args.variant, "rows": config["rows"], "runtime": config["runtime"],
                      "parameters": sum(p.numel() for p in model.parameters())}), flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        losses = []
        for indices in packed_group_batches(splits["train"], args.batch_size, generator):
            batch = trim_padding({k: v[indices] for k, v in encoded["train"].items()})
            cap_logits, intent_logits = model(**{k: v.to(device) for k, v in batch.items()})
            targets, statuses = train_targets[indices], train_statuses[indices]
            if args.variant == "pooled":
                loss = objective(cap_logits, intent_logits, targets, statuses, cap_weights, intent_weights)
            else:
                loss = focal_objective(cap_logits, intent_logits, targets, statuses, cap_weights, intent_weights)
            loss = loss + 0.1 * pair_margin_loss(cap_logits, intent_logits, targets, statuses,
                                                [splits["train"][i] for i in indices.tolist()])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        cp, ip = probabilities(model, encoded["validation"], args.batch_size)
        thresholds, result = calibrate_v4(cp, ip, gold["validation"][0].numpy(), gold["validation"][1].numpy())
        score = optimization_score(result)
        if best is None or score > best[0] + 1e-5:
            best = score, epoch, copy.deepcopy(model.state_dict()), thresholds, result
            bad = 0
        else:
            bad += 1
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "score": score,
                 "thresholds": thresholds, "validation": result, "best_epoch": best[1],
                 "seconds": time.time() - epoch_start}
        with (args.output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps({"epoch": epoch, "loss": round(entry["loss"], 4), "score": round(score, 4),
                          "required_f1": round(result["required"]["f1"], 4),
                          "required_recall": round(result["required"]["recall"], 4),
                          "forbidden_f1": round(result["forbidden"]["f1"], 4),
                          "ambiguous_recall": round(result["intent_per_class"]["ambiguous"]["recall"], 4),
                          "intent_macro_f1": round(result["intent_macro_f1"], 4),
                          "best_epoch": best[1], "seconds": round(entry["seconds"], 1)}), flush=True)
        if epoch >= args.min_epochs and bad >= args.patience:
            break
    model.load_state_dict(best[2], strict=True)
    protocol.update(thresholds=best[3], selected_epoch=best[1])
    report = {"variant": args.variant, "epochs_run": epoch, "selected_epoch": best[1],
              "validation_score": best[0], "validation": best[4], "data_rows": config["rows"],
              "reserved_test_opened": False, "human_review": "pending"}
    for split, rows in splits.items():
        cp, ip = probabilities(model, encoded[split], args.batch_size)
        st, status = decode(cp, ip, protocol["thresholds"])
        report[split] = metrics(st, status, gold[split][0].numpy(), gold[split][1].numpy(), rows)
        if split == "validation":
            protocol["policy"], report["validation_policy"] = calibrate_policy(
                cp, ip, gold[split][0].numpy(), gold[split][1].numpy(), rows, protocol["thresholds"])
    checkpoint = args.output / "checkpoint"
    save_checkpoint(model, tokenizer, checkpoint, protocol)
    restored, _, _ = load_checkpoint(checkpoint)
    with torch.inference_mode():
        sample = {k: v[:2] for k, v in encoded["validation"].items()}
        model.cpu().eval()
        for a, b in zip(model(**sample), restored(**sample)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    report["checkpoint_roundtrip_exact"] = True
    report["seconds"] = time.time() - start
    lock = {"checkpoint_files_sha256": {p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()},
            "weights_sha256": sha256(checkpoint / "model.safetensors"),
            "protocol_sha256": sha256(checkpoint / "protocol.json"),
            "selected_using": "V4 development validation only", "reserved_test_opened": False}
    write_json(args.output / "selection_lock.json", lock)
    write_json(args.output / "training_report.json", report)
    print(json.dumps({"event": "complete", "variant": args.variant, "epoch": best[1],
                      "validation_score": best[0], "seconds": round(report["seconds"], 1)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("pooled", "focal_evidence"), required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261118)
    add_training_device_argument(parser)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
