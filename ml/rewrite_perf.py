"""Specialize MiniRBT sequence shape and remove final-layer non-CLS queries."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


ATTENTION = "/classifier/bert/encoder/layer.5/attention"
EMBEDDINGS = "/classifier/bert/embeddings"
CONTEXT_SHAPE = "/attention/self/Constant_4"
INPUT_LENGTH = 64


def replace_constant(node, array):
    node.attribute[0].t.CopyFrom(numpy_helper.from_array(array))


def rewrite(source, length, cls_only):
    model = onnx.load(source)
    graph = model.graph
    for inp in graph.input:
        shape = inp.type.tensor_type.shape.dim
        if inp.name not in {"input_ids", "attention_mask"} or shape[1].dim_value != INPUT_LENGTH:
            raise ValueError("Expected two MiniRBT inputs of length 64")
        shape[1].dim_value = length

    selected = None
    nodes = []
    for node in graph.node:
        name = node.name
        if name in {"/Constant", EMBEDDINGS + "/Constant_1"}:
            tensor = numpy_helper.to_array(node.attribute[0].t)
            if tensor.shape != (1, INPUT_LENGTH):
                raise ValueError(f"Unexpected constant shape for {name}")
            replace_constant(node, tensor[:, :length])
        if name.endswith(CONTEXT_SHAPE):
            shape = numpy_helper.to_array(node.attribute[0].t).copy()
            if tuple(shape) != (1, INPUT_LENGTH, 256):
                raise ValueError(f"Unexpected context shape for {name}")
            shape[1] = 1 if cls_only and name.startswith(ATTENTION) else length
            replace_constant(node, shape)
        if cls_only and name == ATTENTION + "/self/query/MatMul":
            selected = ATTENTION + "/cls_input"
            previous = node.input[0]
            indices = selected + "_indices"
            graph.initializer.append(numpy_helper.from_array(np.array([0], np.int64), indices))
            nodes.append(helper.make_node("Gather", [previous, indices], [selected], name=selected, axis=1))
            node.input[0] = selected
        if cls_only and name == ATTENTION + "/output/Add":
            if selected is None or not node.input[1].endswith("layer.4/output/LayerNorm/Add_1_output_0"):
                raise ValueError("Unexpected final-layer residual")
            node.input[1] = selected
        nodes.append(node)

    if cls_only and selected is None:
        raise ValueError("Final-layer CLS branch was not rewritten")
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.value_info[:]
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--length", choices=[16, 32, 64], type=int, default=64)
    parser.add_argument("--all-final-queries", action="store_true")
    args = parser.parse_args()
    onnx.save(rewrite(args.source, args.length, not args.all_final_queries), args.output)


if __name__ == "__main__":
    main()
