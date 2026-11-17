"""Measure decomposed LayerNorm centered values against the FP16 squaring range on CPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = shape_inference.infer_shapes(onnx.load(args.graph))
    length = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    names = [node.output[0] for node in model.graph.node if node.op_type == "Sub" and "/LayerNorm/" in node.name]
    if not names:
        raise ValueError("No decomposed LayerNorm centers found")
    values = {value.name: value for value in model.graph.value_info}
    for name in names:
        model.graph.output.append(values[name])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    rows = []
    for index, vector in enumerate(json.loads(args.vectors.read_text()), 1):
        if any(vector["attention_mask"][length:]):
            continue
        inputs = {key: np.array([vector[key][:length]], np.int32) for key in ("input_ids", "attention_mask")}
        outputs = session.run(names, inputs)
        active = np.array(vector["attention_mask"][:length], bool)
        stages = []
        for name, data in zip(names, outputs):
            if data.ndim != 3 or data.shape[0] != 1:
                raise ValueError("Unexpected LayerNorm center layout")
            chosen = data[:, active, :] if data.shape[1] == length else data
            stages.append({"name": name, "all_token_absmax": float(np.max(np.abs(data))),
                           "active_token_absmax": float(np.max(np.abs(chosen)))})
        rows.append({"case_id": index, "stages": stages})
    result = {
        "scope": "CPU FP32 intermediate ranges; not phone values or a proof of hardware overflow",
        "graph_sha256": hashlib.sha256(args.graph.read_bytes()).hexdigest(),
        "length": length, "fp16_max": float(np.finfo(np.float16).max),
        "fp16_square_safe_abs": math.sqrt(float(np.finfo(np.float16).max)),
        "max_all_token_centered_abs": max(stage["all_token_absmax"] for row in rows for stage in row["stages"]),
        "max_active_token_centered_abs": max(stage["active_token_absmax"] for row in rows for stage in row["stages"]),
        "cases": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Maximum centered abs (all/active):", result["max_all_token_centered_abs"], result["max_active_token_centered_abs"])


if __name__ == "__main__":
    main()
