"""TinyBERT heads and decoding for the experimental 32-capability task."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel, BertTokenizerFast
from safetensors.torch import load_file, save_file


STATES = ["unmentioned", "required", "forbidden"]
INTENTS = ["in_scope", "ambiguous", "no_device_capability", "out_of_scope"]


class PermissionIntentModel(nn.Module):
    def __init__(self, config, capability_count):
        super().__init__()
        config._attn_implementation = "eager"
        self.encoder = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.head_type = getattr(config, "permission_intent_head", "pooled")
        if self.head_type in ("pooled", "pooled_label_attention"):
            self.capability_classifier = nn.Linear(config.hidden_size, capability_count * len(STATES))
            if self.head_type == "pooled_label_attention":
                self.label_queries = nn.Parameter(torch.empty(capability_count, config.hidden_size))
                nn.init.normal_(self.label_queries, std=config.initializer_range)
                self.label_gates = nn.Parameter(torch.full((capability_count,), -2.0))
        elif self.head_type == "label_attention":
            self.label_queries = nn.Parameter(torch.empty(capability_count, config.hidden_size))
            nn.init.normal_(self.label_queries, std=config.initializer_range)
            self.capability_classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 4, config.hidden_size // 2), nn.GELU(),
                nn.Dropout(config.hidden_dropout_prob), nn.Linear(config.hidden_size // 2, len(STATES)))
        else:
            raise ValueError("Unknown permission intent head: " + self.head_type)
        self.intent_classifier = nn.Linear(config.hidden_size, len(INTENTS))
        self.capability_count = capability_count
        for head in (self.capability_classifier, self.intent_classifier):
            for layer in head.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=config.initializer_range)
                    nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=torch.zeros_like(input_ids), return_dict=False)
        pooled = self.dropout(encoded[1])
        if self.head_type == "pooled":
            capabilities = self.capability_classifier(pooled).reshape(-1, self.capability_count, len(STATES))
        elif self.head_type == "pooled_label_attention":
            tokens = encoded[0]
            # Scaled dot product keeps the exported path to ordinary matmuls,
            # softmax and elementwise operations. No label strings at inference.
            scores = torch.matmul(tokens, self.label_queries.transpose(0, 1)).transpose(1, 2)
            scores = scores / (tokens.shape[-1] ** 0.5)
            valid = self._content_mask(input_ids, attention_mask)
            attention = scores.masked_fill(~valid[:, None, :], -10000.0).softmax(-1)
            context = torch.matmul(attention, tokens)
            context = self.dropout(torch.tanh(self.encoder.pooler.dense(context)))
            gate = self.label_gates.sigmoid()[None, :, None]
            features = pooled[:, None, :] * (1 - gate) + context * gate
            weight = self.capability_classifier.weight.reshape(self.capability_count, len(STATES), -1)
            bias = self.capability_classifier.bias.reshape(self.capability_count, len(STATES))
            capabilities = (features[:, :, None, :] * weight[None, :, :, :]).sum(-1) + bias
        else:
            tokens = encoded[0]
            query = F.normalize(self.label_queries, dim=-1)
            keys = F.normalize(tokens, dim=-1)
            scores = torch.matmul(keys, query.transpose(0, 1)).transpose(1, 2) * 10.0
            valid = attention_mask.bool()
            cls_id = getattr(self.encoder.config, "cls_token_id", None)
            sep_id = getattr(self.encoder.config, "sep_token_id", None)
            valid = valid & (input_ids != (101 if cls_id is None else cls_id))
            valid = valid & (input_ids != (102 if sep_id is None else sep_id))
            scores = scores.masked_fill(~valid[:, None, :], -10000.0)
            attention = scores.softmax(dim=-1)
            context = torch.matmul(attention, tokens)
            queries = self.label_queries[None, :, :].expand_as(context)
            features = torch.cat((context, pooled[:, None, :].expand_as(context),
                                  context * queries, (context - queries).abs()), dim=-1)
            capabilities = self.capability_classifier(self.dropout(features))
        return capabilities, self.intent_classifier(pooled)

    def _content_mask(self, input_ids, attention_mask):
        cls_id = getattr(self.encoder.config, "cls_token_id", None)
        sep_id = getattr(self.encoder.config, "sep_token_id", None)
        return (attention_mask.bool() & (input_ids != (101 if cls_id is None else cls_id))
                & (input_ids != (102 if sep_id is None else sep_id)))

    def initialize_label_queries(self, tokenizer, descriptions):
        if self.head_type not in ("label_attention", "pooled_label_attention"):
            return
        self.encoder.eval()
        encoded = tokenizer(descriptions, padding=True, truncation=False, return_tensors="pt",
                            return_token_type_ids=False)
        if encoded["input_ids"].shape[1] > self.encoder.config.max_position_embeddings:
            raise ValueError("Capability descriptions exceed encoder position limit")
        with torch.inference_mode():
            tokens = self.encoder(**encoded, return_dict=False)[0]
            valid = encoded["attention_mask"].bool() & (encoded["input_ids"] != tokenizer.cls_token_id)
            valid &= encoded["input_ids"] != tokenizer.sep_token_id
            mean = (tokens * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
            if self.head_type == "pooled_label_attention":
                mean = F.normalize(mean, dim=-1) * (mean.shape[-1] ** 0.5)
            self.label_queries.copy_(mean)


def labels(rows, capability_ids):
    index = {cid: i for i, cid in enumerate(capability_ids)}
    targets = torch.zeros((len(rows), len(index)), dtype=torch.long)
    intents = torch.empty(len(rows), dtype=torch.long)
    for i, row in enumerate(rows):
        required, forbidden = row["required_capabilities"], row["forbidden_capabilities"]
        if set(required) & set(forbidden):
            raise ValueError("Conflicting required/forbidden labels: " + row["id"])
        for state, caps in ((1, required), (2, forbidden)):
            for cid in caps:
                if cid not in index:
                    raise ValueError("Unknown capability: " + cid)
                targets[i, index[cid]] = state
        intents[i] = INTENTS.index(row["intent_status"])
    return targets, intents


def encode(tokenizer, rows, max_length):
    texts = [row["text"] for row in rows]
    lengths = tokenizer(texts, truncation=False, padding=False, return_length=True)["length"]
    over = [row["id"] for row, length in zip(rows, lengths) if length > max_length]
    if over:
        raise ValueError(f"Refusing silent truncation at {max_length} tokens: {over}")
    encoded = tokenizer(texts, padding="max_length", truncation=False, max_length=max_length,
                        return_tensors="pt", return_token_type_ids=False)
    return encoded


def balanced_weights(targets, class_count, power=0.5):
    counts = torch.stack([(targets == i).sum(dim=0) for i in range(class_count)], dim=-1).float()
    weights = (len(targets) / (class_count * counts.clamp_min(1))).pow(power)
    weights = weights.clamp(0.25 if power == 1 else 0.5, 50.0 if power == 1 else 8.0)
    weights /= ((counts * weights).sum(dim=-1, keepdim=True) / len(targets)).clamp_min(1e-6)
    return weights, counts


def objective(capability_logits, intent_logits, targets, intents, capability_weights, intent_weights):
    losses = F.cross_entropy(capability_logits.reshape(-1, len(STATES)), targets.reshape(-1), reduction="none")
    sample_weights = capability_weights[torch.arange(targets.shape[1], device=targets.device), targets]
    cap_loss = (losses.reshape_as(targets) * sample_weights).mean()
    status_loss = F.cross_entropy(intent_logits, intents, weight=intent_weights)
    return cap_loss + 0.5 * status_loss


def probabilities(model, encoded, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    caps, intents = [], []
    with torch.inference_mode():
        for start in range(0, len(encoded["input_ids"]), batch_size):
            batch = {key: value[start:start + batch_size].to(device) for key, value in encoded.items()}
            cap_logits, intent_logits = model(**batch)
            caps.append(cap_logits.softmax(-1).cpu().numpy())
            intents.append(intent_logits.softmax(-1).cpu().numpy())
    return np.concatenate(caps), np.concatenate(intents)


def decode(capabilities, intents, thresholds, gate=True):
    state = capabilities.argmax(axis=-1)
    status = intents.argmax(axis=-1)
    state[(state == 1) & (capabilities[:, :, 1] < thresholds["required"])] = 0
    state[(state == 2) & (capabilities[:, :, 2] < thresholds["forbidden"])] = 0
    if gate:
        state[(status[:, None] != 0) & (state == 1)] = 0
    return state, status


def binary_metrics(predicted, expected):
    tp = int((predicted & expected).sum())
    fp = int((predicted & ~expected).sum())
    fn = int((~predicted & expected).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def metrics(predicted, status, expected, expected_status, rows=None):
    expected = np.asarray(expected)
    expected_status = np.asarray(expected_status)
    required = binary_metrics(predicted == 1, expected == 1)
    forbidden = binary_metrics(predicted == 2, expected == 2)
    required_exact = np.all((predicted == 1) == (expected == 1), axis=1)
    state_exact = np.all(predicted == expected, axis=1)
    joint = state_exact & (status == expected_status)
    over = np.any((predicted == 1) & (expected != 1), axis=1)
    violations = np.any((predicted == 1) & (expected == 2), axis=1)
    intent_classes = {name: binary_metrics(status == i, expected_status == i) for i, name in enumerate(INTENTS)}
    result = {
        "samples": len(expected), "required": required, "forbidden": forbidden,
        "required_set_exact": int(required_exact.sum()), "required_set_exact_rate": float(required_exact.mean()),
        "all_capability_states_exact": int(state_exact.sum()), "all_capability_states_exact_rate": float(state_exact.mean()),
        "joint_states_and_intent_exact": int(joint.sum()), "joint_states_and_intent_exact_rate": float(joint.mean()),
        "intent_accuracy": float((status == expected_status).mean()),
        "intent_macro_f1": float(np.mean([m["f1"] for m in intent_classes.values()])),
        "intent_per_class": intent_classes,
        "overrequested_samples": int(over.sum()), "overrequested_sample_rate": float(over.mean()),
        "explicit_prohibition_violation_samples": int(violations.sum()),
        "explicit_prohibition_violation_rate": float(violations.mean()),
        "clarification": binary_metrics(status == 1, expected_status == 1),
        "intent_confusion_matrix": [[int(((expected_status == i) & (status == j)).sum())
                                      for j in range(len(INTENTS))] for i in range(len(INTENTS))],
    }
    if rows is not None:
        families = defaultdict(list)
        for row, correct in zip(rows, joint):
            families[row["family_id"]].append(bool(correct))
        result["families"] = len(families)
        result["families_all_paraphrases_correct"] = sum(all(values) for values in families.values())
        result["mean_family_joint_accuracy"] = float(np.mean([np.mean(values) for values in families.values()]))
    return result


def selection_score(result):
    return float(0.5 * result["required"]["f1"] + 0.35 * result["forbidden"]["f1"]
                 + 0.15 * result["intent_macro_f1"])


def calibrate(capabilities, intents, targets, intent_targets):
    best = None
    for required in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        for forbidden in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            thresholds = {"required": required, "forbidden": forbidden}
            states, status = decode(capabilities, intents, thresholds)
            result = metrics(states, status, targets, intent_targets)
            rank = (selection_score(result), -result["explicit_prohibition_violation_samples"],
                    -result["overrequested_samples"], -abs(required - 0.5) - abs(forbidden - 0.5))
            if best is None or rank > best[0]:
                best = rank, thresholds, result
    return best[1], best[2]


def save_checkpoint(model, tokenizer, directory, protocol):
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(directory / "model.safetensors"))
    (directory / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")


def load_checkpoint(directory, *, expected_states=STATES):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["states"] != expected_states or protocol["intent_statuses"] != INTENTS:
        raise ValueError("Unsupported state/intent ordering")
    config = BertConfig.from_pretrained(directory, local_files_only=True)
    model = PermissionIntentModel(config, len(protocol["capability_ids"]))
    model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained(directory, local_files_only=True)
    return model, tokenizer, protocol


def prediction_records(rows, cap_probs, intent_probs, protocol):
    policy = protocol.get("policy")
    if policy is not None:
        from intent_contrast import decode_policy
        states, statuses, accepted, decisions = decode_policy(cap_probs, intent_probs, protocol["thresholds"], policy)
    else:
        states, statuses = decode(cap_probs, intent_probs, protocol["thresholds"])
    for i, row in enumerate(rows):
        required = [cid for cid, state in zip(protocol["capability_ids"], states[i]) if state == 1]
        forbidden = [cid for cid, state in zip(protocol["capability_ids"], states[i]) if state == 2]
        record = {"id": row.get("id"), "text": row["text"], "required_capabilities": required,
                  "forbidden_capabilities": forbidden, "intent_status": INTENTS[statuses[i]],
                  "needs_clarification": bool(statuses[i] == 1),
                  "intent_probabilities": dict(zip(INTENTS, intent_probs[i].tolist()))}
        if policy is not None:
            record.update(decision=str(decisions[i]), accepted=bool(accepted[i]),
                          needs_clarification=bool(decisions[i] == "clarify"))
        yield record
