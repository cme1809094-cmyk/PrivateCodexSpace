"""Partially fine-tune a pinned Chinese NLI teacher without reading the holdout."""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import BertConfig, BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

from intent_contrast import calibrate_policy, packed_group_batches, pair_margin_loss
from intent_optimization import (add_training_device_argument, calibrate_v4, focal_objective, optimization_score,
                                 resolve_training_device, training_runtime, trim_padding)
from permission_intent_model import PermissionIntentModel, balanced_weights, encode, labels, probabilities
from train_intent_contrast import ROOT, read_json, sha256, write_json
from train_intent_v4 import DATA, load_development


def load_teacher(base, cap_ids, delta_dir=None):
    manifest = read_json(base / "source-manifest.json")
    for name in ("pytorch_model.bin", "config.json", "vocab.txt"):
        if sha256(base / name) != manifest["downloaded_files"][name]["sha256"]:
            raise ValueError("Pinned teacher source changed: " + name)
    config = BertConfig.from_pretrained(base, local_files_only=True)
    config.permission_intent_head = "pooled"
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1
    model = PermissionIntentModel(config, len(cap_ids))
    encoder, info = BertModel.from_pretrained(base, config=config, local_files_only=True,
                                              use_safetensors=False, attn_implementation="sdpa", output_loading_info=True)
    if info["missing_keys"] or info.get("mismatched_keys"):
        raise ValueError("Incomplete teacher encoder: " + str(info))
    model.encoder = encoder
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for layer in model.encoder.encoder.layer[-4:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    for p in model.encoder.pooler.parameters():
        p.requires_grad_(True)
    if delta_dir is not None:
        lock = read_json(delta_dir / "teacher_lock.json")
        if sha256(delta_dir / "teacher_delta.safetensors") != lock["delta_sha256"]:
            raise ValueError("Teacher delta changed after validation selection")
        delta = load_file(str(delta_dir / "teacher_delta.safetensors"))
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        if set(delta) != expected:
            raise ValueError("Teacher delta parameter set differs")
        _, unexpected = model.load_state_dict(delta, strict=False)
        if unexpected:
            raise ValueError(unexpected)
    model.eval()
    return model, BertTokenizerFast.from_pretrained(base, local_files_only=True)


def cache_teacher(model, tokenizer, rows, output, batch_size=16):
    encoded = encode(tokenizer, rows, 64)
    device = next(model.parameters()).device
    capabilities, intents, features = [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = trim_padding({k: v[start:start + batch_size] for k, v in encoded.items()})
            batch = {k: v.to(device) for k, v in batch.items()}
            output_encoder = model.encoder(**batch, token_type_ids=torch.zeros_like(batch["input_ids"]), return_dict=False)
            pooled = output_encoder[1]
            capabilities.append(model.capability_classifier(pooled).reshape(-1, model.capability_count, 3).cpu().numpy())
            intents.append(model.intent_classifier(pooled).cpu().numpy())
            features.append(pooled.to(device="cpu", dtype=torch.float16).numpy())
    np.savez_compressed(output, ids=np.asarray([r["id"] for r in rows]),
                        capability_logits=np.concatenate(capabilities), intent_logits=np.concatenate(intents),
                        pooled_features=np.concatenate(features))


def train(args):
    device = resolve_training_device(args.device)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty teacher output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261118)
    np.random.seed(20261118)
    random.seed(20261118)
    splits, caps, hashes = load_development()
    model, tokenizer = load_teacher(args.base, caps)
    model.to(device)
    encoded = {s: encode(tokenizer, rows, 64) for s, rows in splits.items()}
    gold = {s: labels(rows, caps) for s, rows in splits.items()}
    cw, _ = balanced_weights(gold["train"][0], 3, power=1.0)
    iw, _ = balanced_weights(gold["train"][1], 4, power=1.0)
    cw, iw = cw.to(device), iw.to(device)
    train_targets, train_statuses = (value.to(device) for value in gold["train"])
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    config = {"teacher_source": read_json(args.base / "source-manifest.json"), "data_hashes": hashes,
              "runtime": training_runtime(device),
              "extension_plan_sha256": sha256(DATA / "extension_plan.json"),
              "rows": {s: len(rows) for s, rows in splits.items()}, "epochs": args.epochs,
              "trainable_parameters": sum(p.numel() for p in trainable.values()),
              "total_parameters": sum(p.numel() for p in model.parameters()), "trainable_encoder_layers": 4,
              "encoder_lr": 2e-5, "head_lr": 1e-3, "fresh_reserved_read": False,
              "training_source_sha256": sha256(Path(__file__)), "human_review": "pending"}
    write_json(args.output / "training_config.json", config)
    (args.output / "train_permission_teacher.py").write_bytes(Path(__file__).read_bytes())
    groups = [{"params": [p for n, p in trainable.items() if n.startswith("encoder.")], "lr": 2e-5},
              {"params": [p for n, p in trainable.items() if not n.startswith("encoder.")], "lr": 1e-3}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    generator = torch.Generator().manual_seed(20261118)
    steps = len(list(packed_group_batches(splits["train"], 16, torch.Generator().manual_seed(20261118))))
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * args.epochs * 0.1)), steps * args.epochs)
    best = None
    print(json.dumps({"event": "teacher_start", "rows": config["rows"], "runtime": config["runtime"], "trainable_parameters": config["trainable_parameters"],
                      "parameters": config["total_parameters"]}), flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        model.encoder.embeddings.eval()
        for layer in model.encoder.encoder.layer[:-4]:
            layer.eval()
        losses = []
        for step, indices in enumerate(packed_group_batches(splits["train"], 16, generator), 1):
            batch = trim_padding({k: v[indices] for k, v in encoded["train"].items()})
            cap, intent = model(**{k: v.to(device) for k, v in batch.items()})
            target, status = train_targets[indices], train_statuses[indices]
            loss = focal_objective(cap, intent, target, status, cw, iw, gamma=0, smoothing=0.03)
            loss = loss + 0.1 * pair_margin_loss(cap, intent, target, status, [splits["train"][i] for i in indices.tolist()])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite teacher loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            if step % 40 == 0:
                print(json.dumps({"teacher_epoch": epoch, "step": step, "epoch_steps": steps,
                                  "mean_loss": round(float(np.mean(losses)), 4), "elapsed": round(time.time() - epoch_start)}), flush=True)
        cp, ip = probabilities(model, encoded["validation"], 16)
        thresholds, result = calibrate_v4(cp, ip, gold["validation"][0].numpy(), gold["validation"][1].numpy())
        score = optimization_score(result)
        if best is None or score > best[0]:
            best = score, epoch, {n: p.detach().clone() for n, p in trainable.items()}, thresholds, result
        entry = {"epoch": epoch, "score": score, "validation": result, "thresholds": thresholds,
                 "loss": float(np.mean(losses)), "seconds": time.time() - epoch_start, "best_epoch": best[1]}
        with (args.output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps({"teacher_epoch": epoch, "validation_score": round(score, 4),
                          "required_f1": round(result["required"]["f1"], 4), "forbidden_f1": round(result["forbidden"]["f1"], 4),
                          "intent_macro_f1": round(result["intent_macro_f1"], 4), "seconds": round(entry["seconds"])}), flush=True)
    model.load_state_dict(best[2], strict=False)
    save_file({n: p.cpu().contiguous() for n, p in best[2].items()}, str(args.output / "teacher_delta.safetensors"))
    cp, ip = probabilities(model, encoded["validation"], 16)
    policy, guarded = calibrate_policy(cp, ip, gold["validation"][0].numpy(), gold["validation"][1].numpy(), splits["validation"], best[3])
    write_json(args.output / "protocol.json", {"capability_ids": caps, "thresholds": best[3], "policy": policy,
               "teacher_only": True, "source_model": config["teacher_source"]["model_id"]})
    write_json(args.output / "teacher_lock.json", {"delta_sha256": sha256(args.output / "teacher_delta.safetensors"),
               "source_manifest_sha256": sha256(args.base / "source-manifest.json"), "fresh_reserved_opened": False})
    for split, rows in splits.items():
        cache_teacher(model, tokenizer, rows, args.output / ("teacher_" + split + ".npz"))
    report = {"selected_epoch": best[1], "epochs_run": args.epochs, "validation_score": best[0],
              "validation": best[4], "validation_policy": guarded, "thresholds": best[3],
              "fresh_reserved_opened": False, "human_review": "pending", "seconds": time.time() - start}
    write_json(args.output / "training_report.json", report)
    print(json.dumps({"event": "teacher_complete", "selected_epoch": best[1], "validation_score": best[0],
                      "seconds": round(report["seconds"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    add_training_device_argument(parser)
    train(parser.parse_args())
