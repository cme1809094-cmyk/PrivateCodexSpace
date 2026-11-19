"""Compare locked v4 and v5 models on the same focused holdout and old regression."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from permission_evidence import EVIDENCE_CHANNELS, decode_profile, evidence_targets, infer, load_profile
from permission_intent_model import INTENTS, binary_metrics, encode, labels, metrics
from train_intent_contrast import read_json, read_rows, write_json
from train_intent_v4 import DATA as V4, load_development
from train_intent_v5 import DATA, validation_result, verify_data, verify_profile


def tasks(decoded, gold, rows, caps, focused):
    effective, status, decision = decoded['state'], decoded['status'], decoded['decision']
    result = {mode: metrics(decoded[key], status, *gold, rows)
              for mode, key in [('semantic', 'semantic_state'), ('policy', 'state')]}
    conflict = np.array([r.get('kind') == 'overlap' or r.get('ambiguity_type') == 'conflict' for r in rows])
    in_scope = (gold[1] == 0)
    # Exclude scoped cases from the unscoped route's coverage denominator.
    in_scope &= np.array([r.get('kind') != 'disjoint' for r in rows])
    true_banned = (gold[0] == 2)
    violated = ((effective == 1) & true_banned).any(axis=1)
    result['conflict'] = {**binary_metrics(status == 1, conflict),
        'raw_intent_correct_conflicts': int((conflict & (decoded['raw_status'] == 1)).sum()),
        'policy_clarifications': int((conflict & (decision == 'clarify')).sum()),
        'policy_accepted': int((conflict & decoded['accepted']).sum()),
        'forbidden_issued': int((conflict & violated).sum()), 'samples': int(conflict.sum())}
    result['normal_in_scope'] = {'samples': int(in_scope.sum()),
        'accepted': int((in_scope & decoded['accepted']).sum()),
        'coverage': float((in_scope & decoded['accepted']).sum() / max(1, in_scope.sum())),
        'incorrect_clarifications': int((in_scope & (decision == 'clarify')).sum())}
    result['policy']['explicit_ban_samples'] = int(true_banned.any(axis=1).sum())
    result['policy']['explicit_ban_violations'] = int(violated.sum())
    result['policy']['violation_rate_among_bans'] = float(violated.sum() / max(1, true_banned.any(axis=1).sum()))
    expected_decisions = np.array([
        'review_scope' if r.get('kind') == 'disjoint' else ['accept', 'clarify', 'no_device_capability', 'out_of_scope'][s]
        for r, s in zip(rows, gold[1])])
    if focused:
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[row['scenario_group_id']].append(i)
        result['focused'] = {kind: {'rows': int(mask.sum()), 'correct_raw_intent': int((mask & (decoded['raw_status'] == gold[1])).sum()),
                                   'correct_intent': int((mask & (status == gold[1])).sum()),
                                   'policy_accept': int((mask & (decision == 'accept')).sum()),
                                   'policy_clarify': int((mask & (decision == 'clarify')).sum()),
                                   'policy_scope_review': int((mask & (decision == 'review_scope')).sum()),
                                   'policy_correct_decision': int((mask & (decision == expected_decisions)).sum())}
                             for kind in sorted({r['kind'] for r in rows})
                             for mask in [np.array([r['kind'] == kind for r in rows])]}
        correct_decision = decision == expected_decisions
        result['focused_group_all_decisions_correct'] = sum(bool(correct_decision[inds].all()) for inds in groups.values())
        result['focused_groups'] = len(groups)
        if 'requested' in decoded:
            targets, _ = evidence_targets(rows, caps)
            expected = targets.numpy().astype(bool)
            result['evidence'] = {key: binary_metrics(decoded[key], expected[:, :, i])
                                  for i, key in enumerate(('requested', 'prohibited'))}
            for key, overlap in (('overlap', True), ('disjoint', False)):
                joint = (expected[:, :, 0] & expected[:, :, 1]) & (expected[:, :, 2] == overlap)
                result['evidence'][key] = {'gold_cells': int(joint.sum()),
                                           'identified_cells': int((joint & decoded[key]).sum())}
    return result, expected_decisions


def predictions(decoded, decisions, rows, caps, cap_scores, intent_scores):
    def cap_list(mask):
        return [c for c, take in zip(caps, mask) if take]

    for i, row in enumerate(rows):
        item = {'id': row['id'], 'family_id': row['family_id'], 'text': row['text'],
                'kind': row.get('kind', row.get('ambiguity_type')), 'expected_intent': row['intent_status'],
                'expected_required': row['required_capabilities'], 'expected_forbidden': row['forbidden_capabilities'],
                'expected_decision': str(decisions[i]), 'raw_intent': INTENTS[decoded['raw_status'][i]],
                'intent': INTENTS[decoded['status'][i]], 'decision': str(decoded['decision'][i]),
                'required': cap_list(decoded['state'][i] == 1), 'forbidden': cap_list(decoded['state'][i] == 2),
                'in_scope_probability': float(intent_scores[i, 0])}
        if 'requested' in decoded:
            item.update({name: cap_list(decoded[name][i])
                         for name in ('requested', 'prohibited', 'overlap', 'disjoint', 'unresolved_scope')})
            active = decoded['requested'][i] | decoded['prohibited'][i]
            item['active_evidence_scores'] = {c: cap_scores[i, j].tolist() for j, c in enumerate(caps) if active[j]}
        yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--data-only', type=Path, required=True)
    parser.add_argument('--dual', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty evaluation directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    frozen = verify_data()
    development, caps, _ = load_development()
    focus = read_rows(DATA / 'validation.jsonl')
    fresh_hash = frozen['files_sha256']['reserved_test.jsonl']
    profiles = {'v4': args.baseline, 'data_only': args.data_only / 'checkpoint', 'dual_evidence': args.dual / 'checkpoint'}
    locks = {key: verify_profile(path) for key, path in profiles.items()}
    for key, lock in locks.items():
        if key != 'v4' and (lock['v5_reserved_sha256'] != fresh_hash or lock['reserved_prompts_read']):
            raise ValueError(f'Wrong reserved set in {key} lock')
    baseline, tokenizer, protocol = load_profile(profiles['v4'])
    val = development['validation'] + focus
    baseline_validation = validation_result(decode_profile(*infer(baseline, encode(tokenizer, val, 64), False), protocol),
                         tuple(t.numpy() for t in labels(val, caps)), len(development['validation']), focus)
    del baseline
    scores = {'v4': baseline_validation['score']}
    for key in ('data_only', 'dual_evidence'):
        scores[key] = read_json(profiles[key].parent / 'training_report.json')['validation']['score']
    selected = max(scores, key=scores.get)
    selection = {'picked_using': 'highest development score', 'scores': scores, 'selected': selected,
                 'profile_locks': locks, 'v5_reserved_sha256': fresh_hash, 'reserved_prompts_read': False}
    write_json(args.output / 'final_selection.json', selection)
    datasets = {'new64': read_rows(DATA / 'reserved_test.jsonl'), 'known_v4_320': read_rows(V4 / 'reserved_test.jsonl')}
    report = {'selection': selection, 'human_review': 'pending', 'new64_scope': 'Assistant-authored contrast cases, not real usage distribution',
              'known_v4_320_scope': 'Previously observed regression only', 'holdout_used_to_tune': False, 'datasets': {}}
    all_predictions = defaultdict(dict)
    for name, path in profiles.items():
        model, tokenizer, protocol = load_profile(path)
        if protocol['capability_ids'] != caps:
            raise ValueError(f'Capability order changed in {name}')
        dual = protocol['states'] == EVIDENCE_CHANNELS
        for dataset, rows in datasets.items():
            cap, intent = infer(model, encode(tokenizer, rows, 64), dual)
            decoded = decode_profile(cap, intent, protocol)
            gold = tuple(t.numpy() for t in labels(rows, caps))
            result, expected = tasks(decoded, gold, rows, caps, dataset == 'new64')
            report['datasets'].setdefault(dataset, {})[name] = result
            preds = list(predictions(decoded, expected, rows, caps, cap, intent))
            all_predictions[dataset][name] = preds
            (args.output / f'{dataset}_{name}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in preds))
        del model
    # Published v4 comparison is a guard against accidental threshold/decoder drift in this experiment.
    prior = read_json(V4.parents[1] / 'results/tinybert-intent-v4-20261118/evaluation/comparison_report.json')
    for mode in ('semantic', 'policy'):
        reproduced = report['datasets']['known_v4_320']['v4'][mode]
        original = prior['datasets']['fresh_reserved'][f'v4_{mode}']
        for key in ('required', 'forbidden', 'explicit_prohibition_violation_samples', 'intent_confusion_matrix'):
            if reproduced[key] != original[key]:
                raise ValueError(f'Published v4 result changed: {mode}/{key}')
    report['original_v4_reproduced'] = True
    write_json(args.output / 'comparison_report.json', report)
    with (args.output / 'new64_review.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['id', 'text', 'kind', 'expected_requested', 'expected_prohibited', 'expected_decision',
                         'v4_decision', 'data_only_decision', 'dual_decision', 'dual_requested', 'dual_prohibited',
                         'dual_overlap', 'dual_disjoint', 'human_review', 'review_notes'])
        for i, row in enumerate(datasets['new64']):
            views = [all_predictions['new64'][name][i] for name in profiles]
            writer.writerow([row['id'], row['text'], row['kind'], ', '.join(row['requested_evidence']),
                             ', '.join(row['prohibited_evidence']), views[0]['expected_decision'],
                             *(view['decision'] for view in views),
                             *(', '.join(views[-1][key]) for key in ('requested', 'prohibited', 'overlap', 'disjoint')), 'pending', ''])
    print(json.dumps({'selected': selected, 'development_scores': scores,
                      'new64': {name: r['conflict'] for name, r in report['datasets']['new64'].items()},
                      'old_v4_reproduced': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
