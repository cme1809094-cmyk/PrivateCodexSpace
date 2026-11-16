"""Compare ONNX permission decisions against the original MiniRBT model on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    ml = args.project / "ml"
    config = args.project / "entry/src/main/resources/rawfile/model_config.json"
    thresholds = np.array(json.loads(config.read_text())["thresholds"])
    boundaries = np.log(thresholds / (1 - thresholds))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3

    def session(model):
        return ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])

    base = session(ml / "permission.onnx")
    vectors = []
    for vector in json.loads((ml / "test_vectors.json").read_text()):
        inputs = {key: np.asarray([vector[key]], dtype=np.int32) for key in ("input_ids", "attention_mask")}
        logits, = base.run(None, inputs)
        vectors.append((inputs, logits))

    report = {}
    for length in (16, 32, 64):
        candidate = session(args.build / f"l{length}" / "permission_native.onnx")
        samples = matches = 0
        max_error = 0.0
        for inputs, expected in vectors:
            if inputs["attention_mask"][0, length:].any():
                continue
            sliced = {key: value[:, :length] for key, value in inputs.items()}
            actual, = candidate.run(None, sliced)
            samples += 1
            matches += int(np.array_equal(expected >= boundaries, actual >= boundaries))
            max_error = max(max_error, float(np.max(np.abs(expected - actual))))
        report[length] = {"samples": samples, "matches": matches, "max_logit_error": max_error}
    print(json.dumps(report, indent=2))
    if any(item["matches"] != item["samples"] or item["samples"] == 0 for item in report.values()):
        raise SystemExit("Host permission outputs differ from the reference or no vectors matched a bucket.")


if __name__ == "__main__":
    main()
