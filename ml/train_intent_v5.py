"""Fit locked v5 data-only and independent-evidence candidates to the same development split."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from intent_contrast import packed_group_batches
from intent_optimization import (add_training_device_argument, focal_objective, optimization_score,
                                 resolve_training_device, training_runtime, trim_padding)
from permission_evidence import (DEFAULT_THRESHOLDS, EVIDENCE_CHANNELS, POLICY, decode_profile,
                                 evidence_loss, evidence_targets, evidence_weights, infer,
                                 initialize_evidence, load_profile)
from permission_intent_model import balanced_weights, binary_metrics, encode, labels, metrics, save_checkpoint
from prepare_intent_v5 import DATA
from train_intent_contrast import ROOT, read_json, read_rows, sha256, write_json
from train_intent_v4 import load_development


VARIANTS = ('data_only', 'dual_evidence')
GRID = (0.4, 0.5, 0.6, 0.7, 0.8)


def verify_data():
    frozen = read_json(DATA / 'freeze_manifest.json')
    for name, digest in frozen['files_sha256'].items():
        if sha256(DATA / name) != digest:
            raise ValueError(f'Frozen v5 file changed: {name}')
    if sha256(ROOT / 'ml/prepare_intent_v5.py') != frozen['generator_sha256']:
        raise ValueError('Frozen data generator changed')
    return frozen


def verify_profile(checkpoint):
    lock = read_json(checkpoint.parent / 'selection_lock.json')
    for name, digest in lock['checkpoint_files_sha256'].items():
        if sha256(checkpoint / name) != digest:
            raise ValueError(f'Checkpoint changed: {checkpoint / name}')
    return lock


def validation_result(decoded, gold, old_count, focused):
    state, status = decoded['semantic_state'], decoded['status']
    cap, intent = gold
    old = metrics(state[:old_count], status[:old_count], cap[:old_count], intent[:old_count])
    actual_state, actual_status = state[old_count:], status[old_count:]
    conflict = np.array([r['kind'] == 'overlap' for r in focused])
    conflict_class = binary_metrics(actual_status == 1, conflict)
    exact = (actual_status == intent[old_count:]) & (actual_state == cap[old_count:]).all(axis=1)
    score = 0.60 * optimization_score(old) + 0.25 * conflict_class['f1'] + 0.15 * float(exact.mean())
    return {'score': score, 'old_validation': old, 'focused_conflict': conflict_class,
            'focused_exact': int(exact.sum()), 'focused_samples': len(focused)}


def calibrate(probabilities, protocol, gold, old_count, focused):
    best = None
    dual = protocol['states'] == EVIDENCE_CHANNELS
    positive, negative = ('requested', 'prohibited') if dual else ('required', 'forbidden')
    for pos in GRID:
        for neg in GRID:
            trial = {**protocol['thresholds'], positive: pos, negative: neg}
            decoded = decode_profile(*probabilities, {**protocol, 'thresholds': trial})
            result = validation_result(decoded, gold, old_count, focused)
            rank = result['score'], -abs(pos - 0.6) - abs(neg - 0.6)
            if best is None or rank > best[0]:
                best = rank, trial, result
    return best[1], best[2]


def train(args):
    device = resolve_training_device(args.device)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty training output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    start = time.monotonic()
    frozen = verify_data()
    initial_lock = verify_profile(args.initial)
    previous, caps, previous_hashes = load_development()
    added_train = read_rows(DATA / 'train.jsonl')
    focused = read_rows(DATA / 'validation.jsonl')
    train_rows = previous['train'] + added_train
    validation = previous['validation'] + focused
    old_count = len(previous['validation'])
    model, tokenizer, protocol = load_profile(args.initial)
    if protocol['capability_ids'] != caps or protocol['schema_version'] != 4:
        raise ValueError('V5 requires the paired selected v4 profile')
    runtime = training_runtime(device)
    dual = args.variant == 'dual_evidence'
    if dual:
        initialize_evidence(model)
        protocol.update(states=EVIDENCE_CHANNELS, thresholds=DEFAULT_THRESHOLDS)
    protocol.update(schema_version=5, task='permission_intent_v5_evidence', v5_variant=args.variant,
                    human_review='pending', policy=POLICY,
                    compatible_with_existing_five_label_app=False,
                    generates_scope_slots_or_question_text=False,
                    output_semantics='capability_channels_are_independent_sigmoids' if dual else 'capability_channels_are_mutually_exclusive')
    model.to(device)
    train_x = encode(tokenizer, train_rows, 64)
    validation_x = encode(tokenizer, validation, 64)
    train_y, train_status = labels(train_rows, caps)
    validation_gold = tuple(t.numpy() for t in labels(validation, caps))
    intent_weights, _ = balanced_weights(train_status, 4, power=1.0)
    intent_weights = intent_weights.to(device)
    train_status = train_status.to(device)
    if dual:
        evidence_y, evidence_known = evidence_targets(train_rows, caps)
        weights = tuple(w.to(device) for w in evidence_weights(evidence_y, evidence_known))
        evidence_y, evidence_known = evidence_y.to(device), evidence_known.to(device)
    else:
        cap_weights, _ = balanced_weights(train_y, 3, power=0.5)
        cap_weights, train_y = cap_weights.to(device), train_y.to(device)
    sources = ('train_intent_v5.py', 'permission_evidence.py', 'permission_intent_model.py',
               'intent_optimization.py', 'intent_contrast.py', 'prepare_intent_v5.py')
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'runtime': runtime,
              'rows': {'train': len(train_rows), 'validation': len(validation), 'new_train': len(added_train),
                       'focused_validation': len(focused)},
              'v5_data': frozen, 'v4_data': previous_hashes, 'initial_weights_sha256': initial_lock['weights_sha256'],
              'source_sha256': {name: sha256(ROOT / 'ml' / name) for name in sources},
              'parameters': sum(p.numel() for p in model.parameters()), 'reserved_prompts_read': False}
    write_json(args.output / 'training_config.json', config)
    snapshot = args.output / 'code_snapshot'
    snapshot.mkdir()
    for name in sources:
        (snapshot / name).write_bytes((ROOT / 'ml' / name).read_bytes())
    groups = [{'params': [p for n, p in model.named_parameters() if n.startswith('encoder.') == encoder],
               'lr': 1e-5 if encoder else 1e-4} for encoder in (True, False)]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    rng = torch.Generator().manual_seed(args.seed)
    generator_for_count = torch.Generator().manual_seed(args.seed)
    total = args.epochs * len(list(packed_group_batches(train_rows, args.batch_size, generator_for_count)))
    schedule = get_linear_schedule_with_warmup(opt, max(1, int(total * 0.05)), total)
    best = None
    print(json.dumps({'event': 'start', 'variant': args.variant, 'rows': config['rows'], 'runtime': runtime}), flush=True)
    for epoch in range(1, args.epochs + 1):
        tick = time.monotonic()
        model.train()
        losses = []
        for indices in packed_group_batches(train_rows, args.batch_size, rng):
            batch = trim_padding({k: v[indices] for k, v in train_x.items()})
            cap, intent = model(**{k: v.to(device) for k, v in batch.items()})
            status = train_status[indices]
            if dual:
                loss = evidence_loss(cap, evidence_y[indices], evidence_known[indices], weights)
                loss += 0.5 * F.cross_entropy(intent, status, weight=intent_weights, label_smoothing=0.03)
            else:
                loss = focal_objective(cap, intent, train_y[indices], status, cap_weights, intent_weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            schedule.step()
            losses.append(float(loss.detach()))
        thresholds, result = calibrate(infer(model, validation_x, dual), protocol, validation_gold, old_count, focused)
        if best is None or result['score'] > best[0]:
            best = result['score'], epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, thresholds, result
        entry = {'epoch': epoch, 'mean_loss': float(np.mean(losses)), 'thresholds': thresholds,
                 'seconds': round(time.monotonic() - tick, 2), **result}
        with (args.output / 'history.jsonl').open('a') as file:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(json.dumps({'epoch': epoch, 'score': round(result['score'], 4), 'old_required_f1': round(result['old_validation']['required']['f1'], 4),
                          'conflict_f1': round(result['focused_conflict']['f1'], 4), 'seconds': entry['seconds']}), flush=True)
    model.cpu().load_state_dict(best[2])
    protocol.update(selected_epoch=best[1], thresholds=best[3],
                    policy_semantics='Scope overlap prompts clarification. Scope review retains evidence without an unconditional action.' if dual
                    else 'Legacy mutually exclusive capability states with calibrated thresholds.')
    checkpoint = args.output / 'checkpoint'
    save_checkpoint(model, tokenizer, checkpoint, protocol)
    restored, _, _ = load_profile(checkpoint)
    x = {key: value[:2] for key, value in validation_x.items()}
    with torch.inference_mode():
        model.eval()
        for a, b in zip(model(**x), restored(**x)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    lock = {'checkpoint_files_sha256': {p.name: sha256(p) for p in sorted(checkpoint.iterdir()) if p.is_file()},
            'weights_sha256': sha256(checkpoint / 'model.safetensors'), 'protocol_sha256': sha256(checkpoint / 'protocol.json'),
            'v5_reserved_sha256': frozen['files_sha256']['reserved_test.jsonl'],
            'utc': datetime.now(timezone.utc).isoformat(), 'reserved_prompts_read': False}
    write_json(args.output / 'selection_lock.json', lock)
    write_json(args.output / 'training_report.json', {'variant': args.variant, 'selected_epoch': best[1],
               'elapsed_seconds': time.monotonic() - start, 'checkpoint_roundtrip_exact': True, 'validation': best[4]})
    print(json.dumps({'event': 'complete', 'variant': args.variant, 'selected_epoch': best[1], 'validation_score': best[0]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261119)
    add_training_device_argument(parser)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    train(args)


if __name__ == '__main__':
    main()
