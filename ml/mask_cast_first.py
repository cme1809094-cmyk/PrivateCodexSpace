"""Cast the binary attention mask before changing its rank."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def rewrite(source, destination):
    model = onnx.load(source)
    graph = model.graph
    names = {node.name for node in graph.node}
    cast = "/classifier/bert/Cast"
    reshapes = {"/classifier/bert/Unsqueeze_output_0", "/classifier/bert/Unsqueeze_1_output_0"}
    penalty = "/classifier/bert/Constant_3"
    if not (reshapes | {cast, penalty}).issubset(names):
        raise ValueError("Unexpected MiniRBT mask pattern")
    mask = next(v for v in graph.input if v.name == "attention_mask")
    length = mask.type.tensor_type.shape.dim[1].dim_value
    if length not in (16, 32, 64):
        raise ValueError(f"Unexpected sequence length {length}")
    graph.initializer.append(numpy_helper.from_array(np.array([1, 1, 1, length], np.int64), "mask_shape_9030"))
    nodes = []
    for node in graph.node:
        if node.name in reshapes:
            continue
        if node.name == cast:
            nodes.extend([
                helper.make_node("Cast", ["attention_mask"], ["mask_float_9030"], name="mask_cast_9030",
                                 to=TensorProto.FLOAT),
                helper.make_node("Reshape", ["mask_float_9030", "mask_shape_9030"], node.output,
                                 name="mask_reshape_9030"),
            ])
            continue
        if node.name == penalty:
            node.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array(-10000, np.float32)))
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    referenced = {name for node in nodes for name in node.input}
    initializers = [tensor for tensor in graph.initializer if tensor.name in referenced]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.value_info[:]
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    rewrite(args.source, args.destination)
