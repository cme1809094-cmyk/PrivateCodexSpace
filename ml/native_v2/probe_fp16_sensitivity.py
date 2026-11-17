"""Probe FP16 rounding sensitivity on CPU; this does not emulate Kirin kernels."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference
import onnxruntime as ort


POLICIES = ("baseline", "weights", "conv", "norm", "except_norm", "all")


def rounded_graph(source, policy):
    model = copy.deepcopy(source)
    if policy == "baseline":
        return model
    float_values = {
        value.name for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
        if value.type.tensor_type.elem_type == TensorProto.FLOAT
    }
    for tensor in model.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT:
            data = numpy_helper.to_array(tensor).astype(np.float16).astype(np.float32)
            tensor.CopyFrom(numpy_helper.from_array(data, tensor.name))
    nodes = []
    for original in model.graph.node:
        node = copy.deepcopy(original)
        is_norm = "/LayerNorm/" in node.name
        selected = (policy == "all" or (policy == "conv" and node.op_type == "Conv")
                    or (policy == "norm" and is_norm) or (policy == "except_norm" and not is_norm))
        casts = []
        if selected:
            for index, output in enumerate(original.output):
                if output not in float_values:
                    continue
                before, half = output + "__probe_before", output + "__probe_half"
                node.output[index] = before
                casts.extend((
                    helper.make_node("Cast", [before], [half], name=half, to=TensorProto.FLOAT16),
                    helper.make_node("Cast", [half], [output], name=output + "__probe_restore", to=TensorProto.FLOAT),
                ))
        nodes.append(node)
        nodes.extend(casts)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="Standard permission_native.onnx, not the custom-op adapted.onnx")
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = shape_inference.infer_shapes(onnx.load(args.graph))
    length = source.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    if length not in (16, 32, 64):
        raise ValueError("Expected a fixed 16/32/64-token graph")
    thresholds = np.array(json.loads(args.config.read_text())["thresholds"])
    boundaries = np.log(thresholds / (1 - thresholds))
    vectors = json.loads(args.vectors.read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    result = {
        "scope": "CPU per-node FP16 round-trip sensitivity; not Kirin arithmetic, kernel emulation or a device result",
        "graph_sha256": hashlib.sha256(args.graph.read_bytes()).hexdigest(),
        "length": length, "thresholds": thresholds.tolist(), "device_tested": False, "policies": {}
    }
    for policy in POLICIES:
        model = rounded_graph(source, policy)
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
        rows = []
        for case_id, vector in enumerate(vectors, 1):
            if any(vector["attention_mask"][length:]):
                continue
            inputs = {key: np.array([vector[key][:length]], np.int32) for key in ("input_ids", "attention_mask")}
            actual = session.run(None, inputs)[0].reshape(-1)
            expected = np.array(vector["logits"])
            finite = bool(np.isfinite(actual).all())
            rows.append({
                "case_id": case_id, "finite": finite,
                "labels_match": finite and bool(np.array_equal(actual >= boundaries, expected >= boundaries)),
                "max_logit_error": float(np.max(np.abs(actual - expected))) if finite else None,
            })
        values = [row["max_logit_error"] for row in rows if row["finite"]]
        summary = {
            "checked": len(rows), "matched": sum(row["labels_match"] for row in rows),
            "nonfinite": sum(not row["finite"] for row in rows), "max_logit_error": max(values) if values else None,
            "label_mismatch_case_ids": [row["case_id"] for row in rows if not row["labels_match"]],
        }
        result["policies"][policy] = summary
        print(policy + ": " + json.dumps(summary), flush=True)
    if not result["policies"]["baseline"]["checked"]:
        raise ValueError("No reference vectors fit this bucket")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
