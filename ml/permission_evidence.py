"""Independent request and prohibition evidence over one TinyBERT encoder."""
import numpy as np
import torch
from torch.nn import functional as F

from permission_intent_model import STATES, decode, load_checkpoint
from intent_contrast import decode_policy
from train_intent_contrast import read_json


EVIDENCE_CHANNELS = ['requested', 'prohibited', 'scope_overlap']
DEFAULT_THRESHOLDS = {'requested': 0.6, 'prohibited': 0.6, 'scope_overlap': 0.55, 'scope_disjoint': 0.25}
POLICY = {'min_in_scope_probability': 0.9, 'defer_uncertain_required': True}
DECISIONS = np.array(['accept', 'clarify', 'no_device_capability', 'out_of_scope'], dtype='<U24')


def load_profile(path):
    states = read_json(path / 'protocol.json')['states']
    if states not in (STATES, EVIDENCE_CHANNELS):
        raise ValueError(f'Unknown output channels: {states}')
    return load_checkpoint(path, expected_states=states)


def initialize_evidence(model):
    """Reuse v4 logit differences; range overlap starts without a prior."""
    with torch.no_grad():
        classifier = model.capability_classifier
        for parameter in (classifier.weight, classifier.bias):
            values = parameter.reshape(model.capability_count, 3, *parameter.shape[1:])
            previous = values.clone()
            values[:, 0] = previous[:, 1] - previous[:, 0]
            values[:, 1] = previous[:, 2] - previous[:, 0]
            values[:, 2].zero_()


def evidence_targets(rows, capabilities):
    index = {cap: i for i, cap in enumerate(capabilities)}
    target = torch.zeros((len(rows), len(capabilities), 3))
    known = torch.ones_like(target, dtype=torch.bool)
    known[:, :, 2] = False
    for i, row in enumerate(rows):
        denied = row.get('prohibited_evidence', row['forbidden_capabilities'])
        wanted = row.get('requested_evidence', row['required_capabilities'])
        same_scope = row.get('overlapping_capabilities', [])
        disjoint = row.get('disjoint_capabilities', [])
        source = row.get('source')
        # The v4 template's required action is recorded even when the effective required list is empty.
        direct_template = (row['id'].startswith('v4_train_conflict_') and row.get('split') == 'train'
                           and row.get('ambiguity_type') == 'conflict' and isinstance(source, dict)
                           and source.get('kind') == 'conflict_vs_other_action')
        if direct_template:
            cap = source['composition']['left']
            if denied != [cap] or wanted:
                raise ValueError(f'Invalid reviewed v4 conflict template: {row["id"]}')
            wanted = same_scope = [cap]
        elif 'requested_evidence' not in row and row['intent_status'] == 'ambiguous':
            known[i, :, 0] = False
        if set(same_scope) & set(disjoint) or not (set(same_scope) | set(disjoint)) <= (set(wanted) & set(denied)):
            raise ValueError(f'Invalid scope evidence: {row["id"]}')
        for channel, values in enumerate((wanted, denied, same_scope)):
            for cap in values:
                target[i, index[cap], channel] = 1
        for cap in same_scope + disjoint:
            known[i, index[cap], 2] = True
    return target, known


def evidence_weights(target, known):
    positive = (target * known).sum(dim=0)
    negative = ((1 - target) * known).sum(dim=0)
    evidence = (negative[:, :2] / positive[:, :2].clamp_min(1)).sqrt().clamp(1, 12)
    scope = (positive[:, 2].sum() / negative[:, 2].sum().clamp_min(1)).sqrt().clamp(1, 6)
    return evidence, scope


def evidence_loss(logits, target, known, weights):
    proof_weights, scope_negative_weight = weights
    proof_target = target[:, :, :2]
    proof_logits = logits[:, :, :2]
    proof_known = known[:, :, :2]
    wrong_probability = (proof_logits.sigmoid() - proof_target).abs()
    proof = F.binary_cross_entropy_with_logits(proof_logits, proof_target, pos_weight=proof_weights, reduction='none')
    proof = (proof * wrong_probability * proof_known).sum() / proof_known.sum().clamp_min(1)
    scope = F.binary_cross_entropy_with_logits(logits[:, :, 2], target[:, :, 2], reduction='none')
    scope_weight = torch.where(target[:, :, 2] == 0, scope_negative_weight, 1.0) * known[:, :, 2]
    scope = (scope * scope_weight).sum() / scope_weight.sum().clamp_min(1)
    return 0.7 * proof + 0.2 * scope


def infer(model, encoded, dual, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    caps, intents = [], []
    with torch.inference_mode():
        for start in range(0, len(encoded['input_ids']), batch_size):
            x = {k: v[start:start + batch_size].to(device) for k, v in encoded.items()}
            cap, intent = model(**x)
            caps.append((cap.sigmoid() if dual else cap.softmax(-1)).cpu().numpy())
            intents.append(intent.softmax(-1).cpu().numpy())
    return np.concatenate(caps), np.concatenate(intents)


def decode_evidence(cap, intent, thresholds, policy):
    requested = cap[:, :, 0] >= thresholds['requested']
    prohibited = cap[:, :, 1] >= thresholds['prohibited']
    both = requested & prohibited
    overlapping = both & (cap[:, :, 2] >= thresholds['scope_overlap'])
    disjoint = both & (cap[:, :, 2] <= thresholds['scope_disjoint'])
    unresolved = both & ~(overlapping | disjoint)
    raw_status = intent.argmax(axis=-1)
    status = raw_status.copy()
    status[overlapping.any(axis=1)] = 1
    # Project the evidence to the older effective lists without issuing unscoped requests.
    wanted = requested & ~prohibited & (status[:, None] == 0) & ~both.any(axis=1)[:, None]
    semantic = np.where(prohibited, 2, wanted.astype(np.int64))
    decisions = DECISIONS[status].copy()
    in_scope = status == 0
    uncertain = ((cap[:, :, 0] >= 0.5) & ~requested & ~prohibited).any(axis=1)
    defer = intent[:, 0] < policy['min_in_scope_probability']
    if policy['defer_uncertain_required']:
        defer |= uncertain
    decisions[in_scope & defer] = 'clarify'
    decisions[in_scope & disjoint.any(axis=1) & ~unresolved.any(axis=1)] = 'review_scope'
    decisions[in_scope & unresolved.any(axis=1)] = 'clarify'
    accepted = decisions == 'accept'
    state = np.where((semantic == 1) & ~accepted[:, None], 0, semantic)
    return {'state': state, 'semantic_state': semantic, 'status': status, 'raw_status': raw_status,
            'decision': decisions, 'accepted': accepted, 'requested': requested, 'prohibited': prohibited,
            'overlap': overlapping, 'disjoint': disjoint, 'unresolved_scope': unresolved}


def decode_profile(cap, intent, protocol):
    if protocol['states'] == EVIDENCE_CHANNELS:
        return decode_evidence(cap, intent, protocol['thresholds'], protocol['policy'])
    state, status, accepted, decision = decode_policy(cap, intent, protocol['thresholds'], protocol['policy'])
    semantic, _ = decode(cap, intent, protocol['thresholds'])
    return {'state': state, 'semantic_state': semantic, 'status': status, 'raw_status': status,
            'accepted': accepted, 'decision': decision}
