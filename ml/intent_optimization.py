"""V4 development-only objectives and calibration; no evaluation data readers."""
import os

import numpy as np
import torch
from torch.nn import functional as F

from permission_intent_model import decode, metrics


def add_training_device_argument(parser):
    parser.add_argument("--device", default=os.environ.get("TINYBERT_TRAINING_DEVICE", "cpu"),
                        help="Training device: cpu, cuda, or cuda:<index>")


def resolve_training_device(value):
    device = torch.device(value)
    if device == torch.device("cpu"):
        return device
    if device.type != "cuda":
        raise ValueError("--device must be cpu, cuda, or cuda:<index>")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Check the NVIDIA driver and Docker --gpus option, or select --device cpu.")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} is not visible to PyTorch")
    return torch.device("cuda", index)


def training_runtime(device):
    return {"device": str(device), "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def focal_objective(cap_logits, intent_logits, targets, statuses, cap_weights, intent_weights,
                    gamma=1.5, smoothing=0.03):
    hard = F.cross_entropy(cap_logits.flatten(0, 1), targets.flatten(), reduction="none").reshape_as(targets)
    pt = cap_logits.softmax(-1).gather(-1, targets[:, :, None]).squeeze(-1)
    weight = cap_weights[torch.arange(targets.shape[1], device=targets.device), targets]
    focal = (hard * (1 - pt).pow(gamma) * weight).mean()
    # An unsuppressed smoothing term also regularizes already easy predictions.
    smooth = -cap_logits.log_softmax(-1).mean(-1).mean()
    status_loss = F.cross_entropy(intent_logits, statuses, weight=intent_weights, label_smoothing=smoothing)
    return (1 - smoothing) * focal + smoothing * smooth + 0.5 * status_loss


def optimization_score(result):
    return (0.40 * result["required"]["f1"] + 0.30 * result["forbidden"]["f1"]
            + 0.30 * result["intent_macro_f1"] - 0.20 * result["explicit_prohibition_violation_rate"])


def calibrate_v4(cap_probs, intent_probs, targets, statuses):
    best = None
    grid = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
    for required in grid:
        for forbidden in grid:
            thresholds = {"required": required, "forbidden": forbidden}
            state, status = decode(cap_probs, intent_probs, thresholds)
            result = metrics(state, status, targets, statuses)
            rank = (result["required"]["recall"] >= 0.5, optimization_score(result),
                    -result["explicit_prohibition_violation_samples"], -result["overrequested_samples"])
            if best is None or rank > best[0]:
                best = rank, thresholds, result
    return best[1], best[2]


def trim_padding(batch):
    length = max(2, int(batch["attention_mask"].sum(-1).max()))
    return {key: value[:, :length] for key, value in batch.items()}


def subtype_results(state, status, targets, statuses, rows):
    out = {}
    for subtype in ("conflict", "source", "target", "scope"):
        indices = [i for i, row in enumerate(rows) if row.get("ambiguity_type") == subtype]
        if indices:
            out[subtype] = metrics(state[indices], status[indices], targets[indices], statuses[indices])
    return out
