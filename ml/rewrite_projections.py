"""Rewrite constant-weight MiniRBT projections for the Kirin compiler."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def rewrite(source, method):
    model = shape_inference.infer_shapes(onnx.load(source))
    graph = model.graph
    shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in (*graph.input, *graph.value_info, *graph.output)
    }
    constants = {tensor.name: tensor for tensor in graph.initializer}
    rewritten = []
    replaced = Counter()

    def const(name, values):
        graph.initializer.append(numpy_helper.from_array(np.asarray(values), name))
        return name

    for node in graph.node:
        if node.op_type != "MatMul" or node.input[1] not in constants:
            rewritten.append(node)
            continue
        x, weight = node.input
        y = node.output[0]
        input_shape = shapes[x]
        weight_shape = list(constants[weight].dims)
        if len(input_shape) != 3 or input_shape[0] != 1 or len(weight_shape) != 2:
            rewritten.append(node)
            continue
        batch, tokens, channels = input_shape
        _, outputs = weight_shape
        if not all((tokens, channels, outputs)) or channels != weight_shape[0]:
            raise ValueError(f"Unexpected shape for {node.name}")
        prefix = node.name + "/kirin"
        if method == "flatten":
            flat_shape = const(prefix + "/flat_shape", np.array([tokens, channels], dtype=np.int64))
            output_shape = const(prefix + "/output_shape", np.array([batch, tokens, outputs], dtype=np.int64))
            rewritten += [
                helper.make_node("Reshape", [x, flat_shape], [prefix + "/flat"], name=prefix + "/Flatten"),
                helper.make_node("MatMul", [prefix + "/flat", weight], [prefix + "/projected"],
                                 name=prefix + "/MatMul2D"),
                helper.make_node("Reshape", [prefix + "/projected", output_shape], [y], name=prefix + "/Restore"),
            ]
        else:
            kernel = numpy_helper.to_array(constants[weight]).T.reshape(outputs, channels, 1, 1).copy()
            kernel_name = const(prefix + "/kernel", kernel)
            conv_shape = const(prefix + "/conv_shape", np.array([batch, channels, 1, tokens], dtype=np.int64))
            output_shape = const(prefix + "/output_shape", np.array([batch, outputs, tokens], dtype=np.int64))
            rewritten += [
                helper.make_node("Transpose", [x], [prefix + "/channels"], name=prefix + "/ToChannels",
                                 perm=[0, 2, 1]),
                helper.make_node("Reshape", [prefix + "/channels", conv_shape], [prefix + "/image"],
                                 name=prefix + "/ToNCHW"),
                helper.make_node("Conv", [prefix + "/image", kernel_name], [prefix + "/projected"],
                                 name=prefix + "/Conv1x1", kernel_shape=[1, 1]),
                helper.make_node("Reshape", [prefix + "/projected", output_shape], [prefix + "/squeezed"],
                                 name=prefix + "/DropHeight"),
                helper.make_node("Transpose", [prefix + "/squeezed"], [y], name=prefix + "/Restore",
                                 perm=[0, 2, 1]),
            ]
        replaced[node.op_type] += 1
    if not replaced:
        raise ValueError("No constant-weight 3D projections were found")
    del graph.node[:]
    graph.node.extend(rewritten)
    referenced = {name for node in graph.node for name in node.input}
    initializers = [tensor for tensor in graph.initializer if tensor.name in referenced]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model, replaced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", choices=["flatten", "conv"], required=True)
    args = parser.parse_args()
    model, replaced = rewrite(args.source, args.method)
    onnx.save(model, args.output)
    print(f"Rewrote {sum(replaced.values())} projections using {args.method}")


if __name__ == "__main__":
    main()
