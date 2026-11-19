"""Distill a selected NLI teacher into the same four-layer mobile profile."""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from transformers import get_linear_schedule_with_warmup

from intent_contrast import calibrate_policy, packed_group_batches
from intent_optimization import (add_training_device_argument, calibrate_v4, focal_objective, optimization_score,
                                 resolve_training_device, training_runtime, trim_padding)
from permission_intent_model import balanced_weights, encode, labels, load_checkpoint, probabilities, save_checkpoint
from train_intent_contrast import ROOT, read_json, sha256, write_json
from train_intent_v4 import DATA, load_development


def correct_teacher_kl(student, teacher, target, weight, temperature=2):
    correct = teacher.argmax(-1) == target
    cell = F.kl_div((student / temperature).log_softmax(-1),
                    (teacher / temperature).softmax(-1), reduction="none").sum(-1)
    selected_weight = weight * correct
    return (cell * selected_weight).sum() / selected_weight.sum().clamp_min(1e-8)


def load_training_cache(path, train_ids):
    with np.load(path, allow_pickle=False) as cache:
        if cache["ids"].tolist() != train_ids:
            raise ValueError("Teacher cache IDs/order differ from frozen training split")
        result = {key: torch.from_numpy(cache[key].astype(np.float32))
                  for key in ("capability_logits", "intent_logits", "pooled_features")}
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise ValueError("Nonfinite teacher cache")
    return result


def clarification_specialist_kl(student, teacher, status, weight, temperature=2):
    others = [0, 2, 3]
    student_binary = torch.stack((torch.logsumexp(student[:, others], -1), student[:, 1]), -1)
    teacher_binary = torch.stack((torch.logsumexp(teacher[:, others], -1), teacher[:, 1]), -1)
    correct_property = ((teacher.argmax(-1) == 1) == (status == 1))
    cell = F.kl_div((student_binary / temperature).log_softmax(-1),
                    (teacher_binary / temperature).softmax(-1), reduction="none").sum(-1)
    selected = weight * correct_property
    return (cell * selected).sum() / selected.sum().clamp_min(1e-8)


def cache_self_predictions(model, encoded, captured):
    caps, intents, features = [], [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded["input_ids"]), 16):
            batch = trim_padding({k: v[start:start + 16] for k, v in encoded.items()})
            cap, intent = model(**{k: v.to(device) for k, v in batch.items()})
            caps.append(cap.cpu().numpy())
            intents.append(intent.cpu().numpy())
            features.append(captured["pooled"].cpu().numpy())
    return {"capability_logits": torch.from_numpy(np.concatenate(caps)),
            "intent_logits": torch.from_numpy(np.concatenate(intents)),
            "pooled_features": torch.from_numpy(np.concatenate(features))}


def train(args):
    device = resolve_training_device(args.device)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty distillation output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261118)
    np.random.seed(20261118)
    random.seed(20261118)
    splits, caps, hashes = load_development()
    model, tokenizer, protocol = load_checkpoint(args.initial)
    model.to(device)
    if protocol["capability_ids"] != caps:
        raise ValueError("Student capability order differs")
    initial_lock = read_json(args.initial.parent / "selection_lock.json")
    for name, digest in initial_lock["checkpoint_files_sha256"].items():
        if sha256(args.initial / name) != digest:
            raise ValueError("Initial student profile changed")
    teacher_report = read_json(args.teacher / "training_report.json")
    initial_report = read_json(args.initial.parent / "training_report.json")
    if args.clarification_specialist:
        teacher_f1 = teacher_report["validation"]["intent_per_class"]["ambiguous"]["f1"]
        student_f1 = initial_report["validation"]["intent_per_class"]["ambiguous"]["f1"]
        if teacher_f1 < student_f1 + 0.03:
            raise ValueError("Teacher failed the clarification-specialist validation gate")
    elif teacher_report["validation_score"] < initial_report["validation_score"] + 0.02:
        raise ValueError("Teacher failed the predeclared validation improvement gate")
    teacher_protocol = read_json(args.teacher / "protocol.json")
    if teacher_protocol["capability_ids"] != caps:
        raise ValueError("Teacher capability order differs")
    teachers = load_training_cache(args.teacher / "teacher_train.npz", [r["id"] for r in splits["train"]])
    if teachers["capability_logits"].shape != (len(splits["train"]), len(caps), 3):
        raise ValueError("Invalid teacher cache shape")
    encoded = {s: encode(tokenizer, rows, 64) for s, rows in splits.items()}
    gold = {s: labels(rows, caps) for s, rows in splits.items()}
    cw, _ = balanced_weights(gold["train"][0], 3, power=0.5)
    iw, _ = balanced_weights(gold["train"][1], 4, power=1.0)
    cw, iw = cw.to(device), iw.to(device)
    train_targets, train_statuses = (value.to(device) for value in gold["train"])
    projection = torch.nn.Linear(model.encoder.config.hidden_size, teachers["pooled_features"].shape[-1])
    captured = {}
    def capture(module, inputs, output):
        captured["pooled"] = output[1]
    handle = model.encoder.register_forward_hook(capture)
    self_teacher = None
    binary_weights = None
    if args.clarification_specialist:
        self_teacher = cache_self_predictions(model, encoded["train"], captured)
        mean = teachers["pooled_features"].mean(0)
        scale = teachers["pooled_features"].std(0).clamp_min(0.02)
        teachers["pooled_features"] = (teachers["pooled_features"] - mean) / scale
        np.savez_compressed(args.output / "teacher_feature_stats.npz", mean=mean.numpy(), scale=scale.numpy())
        x = self_teacher["pooled_features"]
        xm = x.mean(0)
        centered = x - xm
        ridge = torch.linalg.solve(centered.t() @ centered + torch.eye(centered.shape[1]),
                                   centered.t() @ teachers["pooled_features"])
        with torch.no_grad():
            projection.weight.copy_(ridge.t())
            projection.bias.copy_(-xm @ ridge)
        np.savez_compressed(args.output / "self_teacher_train.npz", ids=np.asarray([r["id"] for r in splits["train"]]),
                            **{k: v.numpy() for k, v in self_teacher.items()})
        binary_weights, _ = balanced_weights((gold["train"][1] == 1).long(), 2, power=1.0)
        binary_weights = binary_weights.to(device)
        self_teacher = {key: value.to(device) for key, value in self_teacher.items()}
    teachers = {key: value.to(device) for key, value in teachers.items()}
    projection.to(device)
    capability_indices = torch.arange(len(caps), device=device)
    groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("encoder.")], "lr": 1e-5},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")] + list(projection.parameters()), "lr": 1e-4}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    generator = torch.Generator().manual_seed(20261118)
    steps = len(list(packed_group_batches(splits["train"], 16, torch.Generator().manual_seed(20261118))))
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * args.epochs * 0.05)), steps * args.epochs)
    config = {"initial": str(args.initial), "teacher": str(args.teacher), "data_hashes": hashes,
              "runtime": training_runtime(device),
              "teacher_train_cache_sha256": sha256(args.teacher / "teacher_train.npz"),
              "teacher_report_sha256": sha256(args.teacher / "training_report.json"),
              "extension_plan_sha256": sha256(DATA / "extension_plan.json"),
              "epochs": args.epochs, "temperature": 2, "clarification_specialist": args.clarification_specialist,
              "specialist_plan_sha256": sha256(DATA / "specialist_distillation_plan.json") if args.clarification_specialist else None,
              "mobile_parameters": sum(p.numel() for p in model.parameters()),
              "training_only_projection_parameters": sum(p.numel() for p in projection.parameters()),
              "rows": {s: len(r) for s, r in splits.items()}, "fresh_reserved_read": False, "human_review": "pending",
              "source_sha256": {n: sha256(ROOT / "ml" / n) for n in ("distill_permission_intent.py", "permission_intent_model.py", "intent_optimization.py")}}
    write_json(args.output / "training_config.json", config)
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for name in config["source_sha256"]:
        (snapshot / name).write_bytes((ROOT / "ml" / name).read_bytes())
    best, bad = None, 0
    print(json.dumps({"event": "distillation_start", "rows": config["rows"], "runtime": config["runtime"],
                      "mobile_parameters": config["mobile_parameters"]}), flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        projection.train()
        losses = []
        for indices in packed_group_batches(splits["train"], 16, generator):
            batch = trim_padding({k: v[indices] for k, v in encoded["train"].items()})
            cap, intent = model(**{k: v.to(device) for k, v in batch.items()})
            target, status = train_targets[indices], train_statuses[indices]
            hard = focal_objective(cap, intent, target, status, cw, iw)
            cap_weight = cw[capability_indices, target]
            cap_source = self_teacher["capability_logits"][indices] if self_teacher is not None else teachers["capability_logits"][indices]
            cap_kl = correct_teacher_kl(cap, cap_source, target, cap_weight)
            feature = (1 - F.cosine_similarity(projection(captured["pooled"]), teachers["pooled_features"][indices], dim=-1)).mean()
            if args.clarification_specialist:
                intent_kl = clarification_specialist_kl(intent, teachers["intent_logits"][indices], status,
                                                        binary_weights[(status == 1).long()])
                other_status = (status - 1).clamp_min(0)
                other_kl = correct_teacher_kl(intent[:, [0, 2, 3]], self_teacher["intent_logits"][indices][:, [0, 2, 3]],
                                              other_status, iw[status] * (status != 1))
                loss = 0.6 * hard + 0.3 * 4 * cap_kl + 0.2 * 4 * intent_kl + 0.1 * 4 * other_kl + 0.1 * feature
            else:
                intent_kl = correct_teacher_kl(intent, teachers["intent_logits"][indices], status, iw[status])
                loss = 0.6 * hard + 0.4 * 4 * cap_kl + 0.2 * 4 * intent_kl + 0.1 * feature
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite distillation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projection.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        cp, ip = probabilities(model, encoded["validation"], 16)
        thresholds, result = calibrate_v4(cp, ip, gold["validation"][0].numpy(), gold["validation"][1].numpy())
        score = optimization_score(result)
        if best is None or score > best[0] + 1e-5:
            best = score, epoch, copy.deepcopy(model.state_dict()), copy.deepcopy(projection.state_dict()), thresholds, result
            bad = 0
        else:
            bad += 1
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "validation_score": score,
                 "validation": result, "thresholds": thresholds, "best_epoch": best[1], "seconds": time.time() - epoch_start}
        with (args.output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps({"epoch": epoch, "score": round(score, 4), "required_f1": round(result["required"]["f1"], 4),
                          "forbidden_f1": round(result["forbidden"]["f1"], 4), "intent_macro_f1": round(result["intent_macro_f1"], 4),
                          "ambiguous_recall": round(result["intent_per_class"]["ambiguous"]["recall"], 4), "best_epoch": best[1],
                          "seconds": round(entry["seconds"])}), flush=True)
        if epoch >= 5 and bad >= 3:
            break
    model.load_state_dict(best[2], strict=True)
    projection.load_state_dict(best[3], strict=True)
    handle.remove()
    protocol = copy.deepcopy(protocol)
    protocol.update(thresholds=best[4], selected_epoch=best[1], teacher_distilled=True,
                    teacher_source="IDEA-CCNL/Erlangshen-Roberta-110M-NLI", training_projection_discarded_at_inference=True,
                    clarification_specialist_distilled=args.clarification_specialist)
    cp, ip = probabilities(model, encoded["validation"], 16)
    protocol["policy"], policy_result = calibrate_policy(cp, ip, gold["validation"][0].numpy(), gold["validation"][1].numpy(), splits["validation"], best[4])
    checkpoint = args.output / "checkpoint"
    save_checkpoint(model, tokenizer, checkpoint, protocol)
    save_file({n: p.detach().cpu().contiguous() for n, p in projection.state_dict().items()}, str(args.output / "training_projection.safetensors"))
    restored, _, _ = load_checkpoint(checkpoint)
    with torch.inference_mode():
        model.cpu().eval()
        sample = {k: v[:2] for k, v in encoded["validation"].items()}
        for a, b in zip(model(**sample), restored(**sample)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    write_json(args.output / "selection_lock.json", {"checkpoint_files_sha256": {p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()},
               "weights_sha256": sha256(checkpoint / "model.safetensors"), "protocol_sha256": sha256(checkpoint / "protocol.json"),
               "selected_using": "V4 validation only", "fresh_reserved_opened": False})
    report = {"selected_epoch": best[1], "epochs_run": epoch, "validation_score": best[0], "validation": best[5],
              "validation_policy": policy_result, "checkpoint_roundtrip_exact": True, "fresh_reserved_opened": False,
              "mobile_parameters": config["mobile_parameters"], "human_review": "pending", "seconds": time.time() - start}
    write_json(args.output / "training_report.json", report)
    print(json.dumps({"event": "distillation_complete", "selected_epoch": best[1], "validation_score": best[0],
                      "seconds": round(report["seconds"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--clarification-specialist", action="store_true")
    add_training_device_argument(parser)
    train(parser.parse_args())
