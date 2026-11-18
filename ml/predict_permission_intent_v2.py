"""Run a trained text-only permission-intent checkpoint on CPU."""
import argparse
import json
from pathlib import Path

import torch

from permission_intent_model import encode, load_checkpoint, prediction_records, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("texts", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, tokenizer, protocol = load_checkpoint(args.checkpoint)
    rows = [{"id": str(i + 1), "text": text} for i, text in enumerate(args.texts)]
    encoded = encode(tokenizer, rows, protocol["max_length"])
    cap_probs, intent_probs = probabilities(model, encoded)
    for record in prediction_records(rows, cap_probs, intent_probs, protocol):
        for kind in ("required", "forbidden"):
            record[kind + "_names"] = [protocol["capability_names"][cid] for cid in record[kind + "_capabilities"]]
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
