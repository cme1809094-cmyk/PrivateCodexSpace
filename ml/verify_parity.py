"""Compare logits and final permissions on the original held-out examples."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def evaluate(session, vectors):
    return np.stack([
        session.run(["logits"], {
            name: np.asarray(vector[name], dtype=np.int32)[None, :]
            for name in ("input_ids", "attention_mask")
        })[0][0]
        for vector in vectors
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    vectors = json.loads(args.vectors.read_text())
    thresholds = np.asarray(json.loads(args.config.read_text())["thresholds"])
    logit_thresholds = np.log(thresholds / (1.0 - thresholds))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3

    def load(path):
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    reference = evaluate(load(args.baseline), vectors)
    expected = reference >= logit_thresholds
    results = {}
    for candidate in args.candidates:
        logits = evaluate(load(candidate), vectors)
        errors = np.abs(logits - reference)
        matches = np.all((logits >= logit_thresholds) == expected, axis=1)
        results[candidate.name] = {
            "examples": len(vectors),
            "max_absolute_logit_error": float(errors.max()),
            "mean_absolute_logit_error": float(errors.mean()),
            "identical_permission_sets": int(matches.sum()),
            "changed_example_indexes": np.where(~matches)[0].tolist(),
        }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(result["changed_example_indexes"] for result in results.values()):
        raise SystemExit("Candidate changed one or more permission sets")


if __name__ == "__main__":
    main()
