"""Verify compiled partitions, memory metadata, and host numerical parity."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort


BUCKETS = (16, 32, 64)


def compiled(directory, length, expect_softmax):
    path = directory / "permission.omc"
    model = json.loads((directory / "permission.json").read_text())
    subgraphs = [
        {a["key"]: a["value"] for a in op.get("attr", [])}
        for graph in model["graph"] for op in graph["op"] if op["type"] == "GraphOp"
    ]
    backends = Counter(a["cl_name"]["s"] for a in subgraphs)
    if backends != {"NPUCL": 1}:
        raise ValueError(f"{length}: expected 1 NPUCL subgraph, got {backends}")
    root = model["graph"][0]["op"]
    inputs = {n["name"]: n["output_desc"][0] for n in root if n["type"] == "Data"}
    output = [d for n in root if n["type"] == "NetOutput" for d in n["input_desc"]]
    actual = {name: (d["dtype"], d["shape"]["dim"]) for name, d in inputs.items()}
    expected = {name: (7, [1, length]) for name in ("input_ids", "attention_mask")}
    if actual != expected or [(d["name"], d["dtype"], d["shape"]["dim"]) for d in output] != [
        ("logits", 1, [1, 5])
    ]:
        raise ValueError(f"Unexpected input/output metadata for {length}")
    result = {
        "subgraphs": dict(backends),
        "compiler_subgraph_memory_bytes": subgraphs[0]["memory_size"]["i"],
        "model_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if expect_softmax:
        marker = "stubName = executor_softmax_5d_w_tiling0_fp16,"
        lines = (directory / "compiler.log").read_text().splitlines()
        if not any(marker in line and "SaveKernelBinInfo" in line for line in lines):
            raise ValueError(f"{directory.name}: expected Kirin 9030 NPU Softmax kernel")
        result["softmax_kernel"] = "executor_softmax_5d_w_tiling0_fp16"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=["kirin9030", "kirin9050"], default="kirin9050")
    parser.add_argument("--include-baseline", action="store_true")
    args = parser.parse_args()
    ml = Path(__file__).resolve().parent
    raw = ml.parent / "entry/src/main/resources/rawfile"
    vectors = json.loads((ml / "test_vectors.json").read_text())
    thresholds = np.array(json.loads((raw / "model_config.json").read_text())["thresholds"])
    boundary = np.log(thresholds / (1 - thresholds))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1

    def session(path):
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    base = session(ml / "permission.onnx")
    candidates = {n: session(args.work / f"l{n}/permission.onnx") for n in BUCKETS}
    expect_softmax = args.platform == "kirin9030"
    report = {n: compiled(args.work / f"l{n}", n, expect_softmax) for n in BUCKETS}
    baseline = None
    if args.include_baseline:
        directory = args.work / "lbaseline"
        baseline = compiled(directory, 64, expect_softmax)
        baseline.update(heldout_samples=0, max_logit_error=0, random_mask_samples=0, random_max_logit_error=0)
        baseline_candidate = session(directory / "permission.onnx")
    for length in BUCKETS:
        report[length].update(heldout_samples=0, max_logit_error=0, random_mask_samples=0,
                              random_max_logit_error=0)
    selected = Counter()
    for vector in vectors:
        inputs = {k: np.asarray([vector[k]], np.int32) for k in ("input_ids", "attention_mask")}
        count = int(inputs["attention_mask"].sum())
        selected[next(n for n in BUCKETS if n >= count)] += 1
        logits, = base.run(None, inputs)
        if baseline is not None:
            compare(baseline_candidate, inputs, logits, 64, boundary, baseline, False)
        for n, candidate in candidates.items():
            if count <= n:
                compare(candidate, inputs, logits, n, boundary, report[n], False)
    rng = np.random.default_rng(2026)
    for n, candidate in candidates.items():
        for count in range(1, n + 1):
            ids = rng.integers(104, 21128, size=(1, 64), dtype=np.int32)
            mask = np.zeros((1, 64), dtype=np.int32)
            mask[0, rng.choice(n, size=count, replace=False)] = 1
            inputs = {"input_ids": ids, "attention_mask": mask}
            logits, = base.run(None, inputs)
            compare(candidate, inputs, logits, n, boundary, report[n], True)
            if n == 64 and baseline is not None:
                compare(baseline_candidate, inputs, logits, 64, boundary, baseline, True)
    out = {
        "platform": args.platform,
        "device_tested": False,
        "reference": "original 64-token MiniRBT ONNX, FP32",
        "buckets": report,
        "heldout_routing": dict(selected),
    }
    if baseline is not None:
        out["baseline_64_all_queries"] = baseline
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def compare(candidate, inputs, reference, length, thresholds, summary, random):
    got, = candidate.run(None, {k: v[:, :length] for k, v in inputs.items()})
    np.testing.assert_allclose(got, reference, atol=2e-5, rtol=1e-4)
    np.testing.assert_array_equal(got >= thresholds, reference >= thresholds)
    sample_key, error_key = (
        ("random_mask_samples", "random_max_logit_error") if random
        else ("heldout_samples", "max_logit_error")
    )
    summary[sample_key] += 1
    summary[error_key] = max(summary[error_key], float(np.max(np.abs(got - reference))))


if __name__ == "__main__":
    main()
