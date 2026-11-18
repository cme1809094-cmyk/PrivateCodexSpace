"""Shared contrastive training and explicit abstention policy helpers."""
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F

from permission_intent_model import decode, metrics


def required_fbeta(result, beta=0.5):
    p, r = result["required"]["precision"], result["required"]["recall"]
    return (1 + beta * beta) * p * r / (beta * beta * p + r) if p + r else 0.0


def contrast_score(result):
    return (0.45 * required_fbeta(result) + 0.30 * result["forbidden"]["f1"]
            + 0.25 * result["intent_macro_f1"] - 0.1 * result["explicit_prohibition_violation_rate"])


def calibrate_semantics(cap_probs, intent_probs, targets, statuses):
    best = None
    for required in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        for forbidden in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            thresholds = {"required": required, "forbidden": forbidden}
            state, status = decode(cap_probs, intent_probs, thresholds)
            result = metrics(state, status, targets, statuses)
            rank = (result["required"]["recall"] >= 0.4, contrast_score(result),
                    -result["explicit_prohibition_violation_samples"], -result["overrequested_samples"])
            if best is None or rank > best[0]:
                best = rank, thresholds, result
    return best[1], best[2]


def packed_group_batches(rows, batch_size, generator):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.get("scenario_group_id", row["family_id"])].append(i)
    grouped = list(groups.values())
    order = torch.randperm(len(grouped), generator=generator).tolist()
    batch = []
    for i in order:
        members = grouped[i]
        if len(members) > batch_size:
            raise ValueError("Batch is smaller than a locked scenario group")
        if batch and len(batch) + len(members) > batch_size:
            yield torch.tensor(batch)
            batch = []
        batch.extend(members)
    if batch:
        yield torch.tensor(batch)


def pair_margin_loss(cap_logits, intent_logits, targets, statuses, rows, margin=1.0):
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        if "scenario_group_id" in row:
            groups[row["scenario_group_id"]][row["contrast_role"]].append(i)
    losses = []
    for roles in groups.values():
        if not roles.get("a") or not roles.get("b"):
            continue
        ia, ib = roles["a"], roles["b"]
        ya, yb = targets[ia[0]], targets[ib[0]]
        a, b = cap_logits[ia].mean(0), cap_logits[ib].mean(0)
        changed = ya != yb
        if changed.any():
            indices = torch.arange(targets.shape[1])[changed]
            separation = ((a[indices, ya[changed]] - a[indices, yb[changed]])
                          + (b[indices, yb[changed]] - b[indices, ya[changed]])) * 0.5
            losses.append(F.relu(margin - separation).mean())
        sa, sb = statuses[ia[0]], statuses[ib[0]]
        if sa != sb:
            a_status, b_status = intent_logits[ia].mean(0), intent_logits[ib].mean(0)
            separation = ((a_status[sa] - a_status[sb]) + (b_status[sb] - b_status[sa])) * 0.5
            losses.append(F.relu(margin - separation))
    return torch.stack(losses).mean() if losses else cap_logits.sum() * 0


def decode_policy(cap_probs, intent_probs, thresholds, policy):
    state, status = decode(cap_probs, intent_probs, thresholds)
    raw_winner = cap_probs.argmax(-1)
    uncertain = np.any((raw_winner == 1) & (cap_probs[:, :, 1] < thresholds["required"]), axis=1)
    accepted = (status == 0) & (intent_probs[:, 0] >= policy.get("min_in_scope_probability", 0.0))
    if policy.get("defer_uncertain_required", False):
        accepted &= ~uncertain
    state[(~accepted[:, None]) & (state == 1)] = 0
    decision = np.array(["accept", "clarify", "no_device_capability", "out_of_scope"])[status].copy()
    decision[(status == 0) & ~accepted] = "clarify"
    return state, status, accepted, decision


def policy_metrics(cap_probs, intent_probs, targets, statuses, rows, thresholds, policy):
    state, status, accepted, decision = decode_policy(cap_probs, intent_probs, thresholds, policy)
    result = metrics(state, status, targets, statuses, rows)
    in_scope = statuses == 0
    result["acceptance_coverage"] = float(accepted.mean())
    result["in_scope_acceptance_coverage"] = float((accepted & in_scope).sum() / max(int(in_scope.sum()), 1))
    result["accepted_samples"] = int(accepted.sum())
    result["operational_clarification_samples"] = int((decision == "clarify").sum())
    if accepted.any():
        result["accepted_subset"] = metrics(state[accepted], status[accepted], targets[accepted], statuses[accepted])
    else:
        result["accepted_subset"] = None
    return result


def calibrate_policy(cap_probs, intent_probs, targets, statuses, rows, thresholds):
    best = None
    for minimum in (0.0, 0.6, 0.75, 0.9, 0.97):
        for defer in (False, True):
            policy = {"min_in_scope_probability": minimum, "defer_uncertain_required": defer}
            result = policy_metrics(cap_probs, intent_probs, targets, statuses, rows, thresholds, policy)
            feasible = result["in_scope_acceptance_coverage"] >= 0.5 and result["required"]["recall"] >= 0.5
            tier = 2 if feasible else 1 if result["in_scope_acceptance_coverage"] >= 0.5 else 0
            rank = (tier, result["in_scope_acceptance_coverage"] if tier == 0 else 0,
                    -4 * result["explicit_prohibition_violation_rate"] - result["overrequested_sample_rate"]
                    + 0.1 * result["required"]["f1"] + 0.1 * result["in_scope_acceptance_coverage"])
            if best is None or rank > best[0]:
                best = rank, policy, result
    return best[1], best[2]
