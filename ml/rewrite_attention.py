"""Test operator and layout alternatives on the fixed-shape MiniRBT graph."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def rewrite(source, options):
    model = shape_inference.infer_shapes(onnx.load(source))
    graph = model.graph
    shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in (*graph.input, *graph.value_info, *graph.output)
    }
    weights = {tensor.name: tensor for tensor in graph.initializer}
    nodes = []

    def emit(op, inputs, output, **attrs):
        nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))
        return output

    def init(name, values, dtype=None):
        graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
        return name

    def reshape(value, shape, output):
        return emit("Reshape", [value, init(output + "_shape", shape, np.int64)], output)

    for node in graph.node:
        y = node.output[0]
        prefix = node.name + "/experiment"
        if node.op_type == "MatMul" and "broadcast" in options:
            a, b = node.input
            if len(shapes.get(a, [])) == 4 and len(shapes.get(b, [])) == 4:
                batch, heads, rows, width = shapes[a]
                b_batch, b_heads, b_width, cols = shapes[b]
                assert (batch, heads, width) == (b_batch, b_heads, b_width)
                a = reshape(a, [batch, heads, rows, width, 1], prefix + "/broadcast_a")
                b = reshape(b, [batch, heads, 1, width, cols], prefix + "/broadcast_b")
                product = emit("Mul", [a, b], prefix + "/products")
                axes = init(prefix + "/axes", [3], np.int64)
                emit("ReduceSum", [product, axes], y, keepdims=0)
                continue
        if node.op_type == "MatMul" and "rank3" in options:
            x, weight = node.input
            if len(shapes.get(x, [])) == 4 and len(shapes.get(weight, [])) == 4:
                assert shapes[x][0] == shapes[weight][0] == shapes[y][0] == 1
                x = reshape(x, shapes[x][1:], prefix + "/a3d")
                weight = reshape(weight, shapes[weight][1:], prefix + "/b3d")
                projected = emit("MatMul", [x, weight], prefix + "/product")
                reshape(projected, shapes[y], y)
                continue
        if node.op_type == "Softmax" and "softmax" in options:
            x = node.input[0]
            axis = next(helper.get_attribute_value(a) for a in node.attribute if a.name == "axis")
            maximum = emit("ReduceMax", [x], prefix + "/max", axes=[axis], keepdims=1)
            if "reciprocal" in options:
                negative = emit("Neg", [maximum], prefix + "/negative_max")
                centered = emit("Add", [x, negative], prefix + "/centered")
            else:
                centered = emit("Sub", [x, maximum], prefix + "/centered")
            exponent = emit("Exp", [centered], prefix + "/exp")
            axes = init(prefix + "/axes", [axis], np.int64)
            summed = emit("ReduceSum", [exponent, axes], prefix + "/sum", keepdims=1)
            if "reciprocal" in options:
                reciprocal = emit("Reciprocal", [summed], prefix + "/reciprocal_sum")
                emit("Mul", [exponent, reciprocal], y)
            else:
                emit("Div", [exponent, summed], y)
            continue
        if node.op_type == "Unsqueeze" and "/classifier/bert/Unsqueeze" in node.name and "maskreshape" in options:
            reshape(node.input[0], shapes[y], y)
            continue
        if node.op_type == "Gemm" and "gemm" in options:
            x, weight, bias = node.input
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            assert attrs == {"alpha": 1.0, "beta": 1.0, "transB": 1}
            w = numpy_helper.to_array(weights[weight])
            out_channels, in_channels = w.shape
            kernel = init(prefix + "/kernel", w.reshape(out_channels, in_channels, 1, 1))
            image = reshape(x, [1, in_channels, 1, 1], prefix + "/image")
            conv = emit("Conv", [image, kernel, bias], prefix + "/conv", kernel_shape=[1, 1])
            reshape(conv, [1, out_channels], y)
            continue
        if "/LayerNorm/" in node.name and "norm" in options:
            if node.op_type == "Pow":
                emit("Mul", [node.input[0], node.input[0]], y)
                continue
            if node.op_type == "Div":
                reciprocal = emit("Reciprocal", [node.input[1]], prefix + "/reciprocal")
                emit("Mul", [node.input[0], reciprocal], y)
                continue
        is_attention = "/attention/self/" in node.name and node.op_type in {"MatMul", "Softmax"}
        if (node.op_type == "Conv" and "fp16conv" in options) or (is_attention and "fp16attention" in options):
            inputs = []
            for idx, name in enumerate(node.input):
                new_name = prefix + f"/input{idx}_fp16"
                if name in weights:
                    inputs.append(init(new_name, numpy_helper.to_array(weights[name]).astype(np.float16)))
                else:
                    inputs.append(emit("Cast", [name], new_name, to=TensorProto.FLOAT16))
            adjusted = onnx.NodeProto()
            adjusted.CopyFrom(node)
            adjusted.input[:] = inputs
            adjusted.output[0] = prefix + "/output_fp16"
            nodes.append(adjusted)
            emit("Cast", adjusted.output, y, to=TensorProto.FLOAT)
            continue
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input}
    initializers = [tensor for tensor in graph.initializer if tensor.name in used]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.value_info[:]
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "transforms", nargs="+",
        choices=["rank3", "broadcast", "softmax", "reciprocal", "maskreshape",
                 "gemm", "norm", "fp16conv", "fp16attention"],
    )
    args = parser.parse_args()
    onnx.save(rewrite(args.source, set(args.transforms)), args.output)


if __name__ == "__main__":
    main()
