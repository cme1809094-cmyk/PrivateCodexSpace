"""Fit only task heads on fixed train features; fold normalization for inference."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from intent_contrast import calibrate_policy
from intent_optimization import calibrate_v4, optimization_score
from permission_intent_model import balanced_weights, encode, labels, load_checkpoint, probabilities, save_checkpoint
from train_intent_contrast import ROOT, read_json, sha256, write_json
from train_intent_v4 import DATA, load_development
from train_permission_teacher import cache_teacher, load_teacher


def extract_features(model, tokenizer, rows):
    encoded = encode(tokenizer, rows, 64)
    cap_chunks, intent_chunks = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = {k: v[start:start + 16] for k, v in encoded.items()}
            output = model.encoder(**batch, token_type_ids=torch.zeros_like(batch["input_ids"]), return_dict=False)
            pooled = output[1]
            if model.head_type == "pooled":
                feature = pooled
            elif model.head_type == "pooled_label_attention":
                tokens = output[0]
                scores = torch.matmul(tokens, model.label_queries.t()).transpose(1, 2) / (tokens.shape[-1] ** 0.5)
                valid = model._content_mask(batch["input_ids"], batch["attention_mask"])
                attention = scores.masked_fill(~valid[:, None, :], -10000.0).softmax(-1)
                context = torch.tanh(model.encoder.pooler.dense(torch.matmul(attention, tokens)))
                gate = model.label_gates.sigmoid()[None, :, None]
                feature = pooled[:, None, :] * (1 - gate) + context * gate
            else:
                raise ValueError("Head refit requires linear per-capability classifiers")
            cap_chunks.append(feature.numpy())
            intent_chunks.append(pooled.numpy())
    return torch.from_numpy(np.concatenate(cap_chunks)), torch.from_numpy(np.concatenate(intent_chunks))


def cap_linear(feature, weight, bias):
    if feature.ndim == 2:
        return F.linear(feature, weight.flatten(0, 1), bias.flatten()).reshape(-1, weight.shape[0], 3)
    return torch.einsum("bch,csh->bcs", feature, weight) + bias


def fold(weight, bias, mean, scale):
    # Remove the softmax-invariant common class component to keep coefficients small.
    dimension = -2
    weight = weight - weight.mean(dimension, keepdim=True)
    bias = bias - bias.mean(-1, keepdim=True)
    if weight.ndim == 3:
        expanded_scale = scale[None, None, :] if scale.ndim == 1 else scale[:, None, :]
        expanded_mean = mean[None, None, :] if mean.ndim == 1 else mean[:, None, :]
    else:
        expanded_scale, expanded_mean = scale[None, :], mean[None, :]
    final = weight / expanded_scale
    return final, bias - (final * expanded_mean).sum(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("student", "teacher"), required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty head-refit directory")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(20261118)
    splits, caps, hashes = load_development()
    if args.kind == "teacher":
        model, tokenizer = load_teacher(args.base, caps, args.initial)
        protocol = read_json(args.initial / "protocol.json")
        parent = read_json(args.initial / "training_report.json")
    else:
        model, tokenizer, protocol = load_checkpoint(args.initial)
        parent = read_json(args.initial.parent / "training_report.json")
    features = {s: extract_features(model, tokenizer, rows) for s, rows in splits.items()}
    gold = {s: labels(rows, caps) for s, rows in splits.items()}
    cw, _ = balanced_weights(gold["train"][0], 3, power=0.5)
    iw, _ = balanced_weights(gold["train"][1], 4, power=1.0)
    cm = features["train"][0].mean(0)
    cs = features["train"][0].std(0).clamp_min(0.02)
    im = features["train"][1].mean(0)
    ins = features["train"][1].std(0).clamp_min(0.02)
    normalized = {s: ((x - cm) / cs, (z - im) / ins) for s, (x, z) in features.items()}
    original_w = model.capability_classifier.weight.detach().reshape(len(caps), 3, -1)
    original_b = model.capability_classifier.bias.detach().reshape(len(caps), 3)
    expanded_mean = cm[None, None, :] if cm.ndim == 1 else cm[:, None, :]
    expanded_scale = cs[None, None, :] if cs.ndim == 1 else cs[:, None, :]
    initial = [original_w * expanded_scale, original_b + (original_w * expanded_mean).sum(-1),
               model.intent_classifier.weight.detach() * ins[None, :],
               model.intent_classifier.bias.detach() + (model.intent_classifier.weight.detach() * im[None, :]).sum(-1)]
    weights = [torch.nn.Parameter(v.clone()) for v in initial]
    with torch.inference_mode():
        old_cp, old_ip = probabilities(model, encode(tokenizer, splits["validation"], 64), 16)
    thresholds, baseline = calibrate_v4(old_cp, old_ip, gold["validation"][0].numpy(), gold["validation"][1].numpy())
    best = optimization_score(baseline), 0, None, None, thresholds, baseline
    config = {"kind": args.kind, "initial": str(args.initial), "data_hashes": hashes,
              "head_plan_sha256": sha256(DATA / "head_refit_plan.json"), "steps": 100, "regularization": [0.01, 0.1],
              "mean_std_fit_on": "training only", "source_sha256": sha256(Path(__file__)),
              "fresh_reserved_read": False, "mobile_parameters_unchanged": True, "human_review": "pending"}
    write_json(args.output / "training_config.json", config)
    (args.output / "refit_permission_heads.py").write_bytes(Path(__file__).read_bytes())
    print({"event": "head_refit_start", "kind": args.kind, "baseline_score": best[0]}, flush=True)
    for regularization in (0.01, 0.1):
        with torch.no_grad():
            for p, value in zip(weights, initial):
                p.copy_(value)
        optimizer = torch.optim.Adam([{"params": [weights[0], weights[2]], "lr": 0.01},
                                      {"params": [weights[1], weights[3]], "lr": 0.03}])
        for step in range(1, 101):
            logits = cap_linear(normalized["train"][0], weights[0], weights[1])
            intents = F.linear(normalized["train"][1], weights[2], weights[3])
            targets, statuses = gold["train"]
            loss = (F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").reshape_as(targets)
                    * cw[torch.arange(len(caps)), targets]).mean()
            loss = loss + 0.5 * F.cross_entropy(intents, statuses, weight=iw)
            loss = loss + regularization * (weights[0].square().sum() / len(caps) + weights[2].square().sum() / 4)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 5 == 0:
                with torch.inference_mode():
                    cp = cap_linear(normalized["validation"][0], weights[0], weights[1]).softmax(-1).numpy()
                    ip = F.linear(normalized["validation"][1], weights[2], weights[3]).softmax(-1).numpy()
                threshold, result = calibrate_v4(cp, ip, gold["validation"][0].numpy(), gold["validation"][1].numpy())
                score = optimization_score(result)
                if score > best[0] + 1e-5:
                    best = score, step, regularization, [p.detach().clone() for p in weights], threshold, result
                with (args.output / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": step, "regularization": regularization, "score": score,
                                             "thresholds": threshold, "validation": result}) + "\n")
        print({"regularization": regularization, "best_score": best[0], "best_step": best[1]}, flush=True)
    if best[1] == 0:
        write_json(args.output / "training_report.json", {"validation_score": best[0], "validation": baseline,
                   "selected_step": 0, "accepted": False, "reason": "No validation improvement", "fresh_reserved_opened": False})
        print({"event": "head_refit_rejected", "reason": "No validation improvement"}, flush=True)
        return
    cap_w, cap_b = fold(best[3][0], best[3][1], cm, cs)
    int_w, int_b = fold(best[3][2], best[3][3], im, ins)
    with torch.no_grad():
        model.capability_classifier.weight.copy_(cap_w.flatten(0, 1))
        model.capability_classifier.bias.copy_(cap_b.flatten())
        model.intent_classifier.weight.copy_(int_w)
        model.intent_classifier.bias.copy_(int_b)
    x, z = features["validation"]
    fp32 = [cap_linear(x, cap_w, cap_b), F.linear(z, int_w, int_b)]
    quantized = [cap_linear(x.half().float(), cap_w.half().float(), cap_b.half().float()),
                 F.linear(z.half().float(), int_w.half().float(), int_b.half().float())]
    precision_error = max(float((a - b).abs().max()) for a, b in zip(fp32, quantized))
    cp, ip = probabilities(model, encode(tokenizer, splits["validation"], 64), 16)
    threshold, result = calibrate_v4(cp, ip, gold["validation"][0].numpy(), gold["validation"][1].numpy())
    score = optimization_score(result)
    accepted = score > optimization_score(baseline) + 1e-5 and (args.kind == "teacher" or precision_error <= 0.1)
    report = {"validation_score": score, "validation": result, "selected_step": best[1], "regularization": best[2],
              "accepted": accepted, "head_fp16_proxy_max_abs_logit_error": precision_error,
              "precision_scope": "Quantized head inputs/parameters with FP32 accumulation only; not a phone or whole-graph check",
              "fresh_reserved_opened": False, "human_review": "pending", "seconds": time.time() - start}
    write_json(args.output / "training_report.json", report)
    if not accepted:
        print({"event": "head_refit_rejected", "score": score, "head_fp16_error": precision_error}, flush=True)
        return
    protocol = copy.deepcopy(protocol)
    protocol.update(thresholds=threshold, head_refitted=True, normalization_folded_into_linear=True)
    protocol["policy"], report["validation_policy"] = calibrate_policy(cp, ip, gold["validation"][0].numpy(), gold["validation"][1].numpy(), splits["validation"], threshold)
    if args.kind == "teacher":
        save_file({n: p.detach().contiguous() for n, p in model.named_parameters() if p.requires_grad}, str(args.output / "teacher_delta.safetensors"))
        write_json(args.output / "protocol.json", protocol)
        write_json(args.output / "teacher_lock.json", {"delta_sha256": sha256(args.output / "teacher_delta.safetensors"),
                   "source_manifest_sha256": sha256(args.base / "source-manifest.json"), "fresh_reserved_opened": False})
        for split, rows in splits.items():
            cache_teacher(model, tokenizer, rows, args.output / ("teacher_" + split + ".npz"))
    else:
        save_checkpoint(model, tokenizer, args.output / "checkpoint", protocol)
        checkpoint = args.output / "checkpoint"
        write_json(args.output / "selection_lock.json", {"checkpoint_files_sha256": {p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()},
                   "weights_sha256": sha256(checkpoint / "model.safetensors"), "protocol_sha256": sha256(checkpoint / "protocol.json"),
                   "selected_using": "V4 validation only", "fresh_reserved_opened": False})
    write_json(args.output / "training_report.json", report)
    print({"event": "head_refit_complete", "kind": args.kind, "score": score, "head_fp16_error": precision_error}, flush=True)


if __name__ == "__main__":
    main()
