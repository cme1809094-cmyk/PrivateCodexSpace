"""Prepare static FP16 attention and map the ONNX nodes through the Kirin V2 adapter."""
import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def prepare(source):
    model = shape_inference.infer_shapes(onnx.load(source))
    graph = model.graph
    layers = {int(m[1]) for n in graph.node
              if (m := re.search(r"/encoder/layer\.(\d+)/", n.name))}
    if not layers or layers != set(range(len(layers))):
        raise ValueError(f"Unexpected encoder layers {layers}")
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in (*graph.input, *graph.value_info, *graph.output)}
    nodes = []
    count = Counter()
    for n in graph.node:
        if n.name == "/classifier/bert/Constant_3":
            n.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array(-10000, np.float32)))
        if "/attention/self/" not in n.name or n.op_type not in {"MatMul", "Softmax"}:
            nodes.append(n)
            continue
        count[n.op_type] += 1
        prefix = n.name + "/native"

        def op(kind, args, suffix, output=None, **kwargs):
            name = prefix + "/" + suffix
            target = output or name + "_out"
            nodes.append(helper.make_node(kind, args, [target], name=name, **kwargs))
            return target

        def reshape(value, dims, suffix):
            name = prefix + "/" + suffix + "_dims"
            graph.initializer.append(numpy_helper.from_array(np.array(dims, np.int64), name))
            return op("Reshape", [value, name], suffix)

        inputs = []
        for i, value in enumerate(n.input):
            s = shapes[value]
            if len(s) != 4 or s[0] != 1 or min(s) <= 0:
                raise ValueError(f"Unexpected attention shape: {value} {s}")
            dims = s[1:] if n.op_type == "MatMul" else [s[1] * s[2], s[3]]
            flat = reshape(value, dims, f"pack_{i}")
            inputs.append(op("Cast", [flat], f"fp16_{i}", to=TensorProto.FLOAT16))
        attrs = {"axis": -1} if n.op_type == "Softmax" else {}
        result = op(n.op_type, inputs, n.op_type + "V2", **attrs)
        result = op("Cast", [result], "fp32", to=TensorProto.FLOAT)
        dims = prefix + "/unpack_dims"
        graph.initializer.append(numpy_helper.from_array(np.array(shapes[n.output[0]], np.int64), dims))
        op("Reshape", [result, dims], "unpack", output=n.output[0])
    if count != {"MatMul": 2 * len(layers), "Softmax": len(layers)}:
        raise ValueError(f"Unexpected BERT operator count: {count}")
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.value_info[:]
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("standard", type=Path)
    parser.add_argument("adapted", type=Path)
    args = parser.parse_args()
    model = prepare(args.source)
    onnx.save(model, args.standard)

    for node in model.graph.node:
        if node.name.endswith("/native/MatMulV2"):
            node.op_type = "NativeAttentionMatmul"
        elif node.name.endswith("/native/SoftmaxV2"):
            node.op_type = "NativeAttentionSoftmax"
            del node.attribute[:]
    onnx.save(model, args.adapted)


if __name__ == "__main__":
    main()
