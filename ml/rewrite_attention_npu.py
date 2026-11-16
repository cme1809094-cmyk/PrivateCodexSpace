"""Lower MiniRBT attention to fixed-shape ONNX primitives for Kirin 9050."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


class Lowering:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = []
        self.counter = 0
        self.scope = ""

    def name(self, stem):
        self.counter += 1
        return f"{self.scope}/{stem}_{self.counter}"

    def constant(self, stem, values, dtype=np.float32):
        name = self.name(stem)
        self.graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
        return name

    def op(self, kind, *inputs, output=None, **attrs):
        name = self.name(kind)
        output = output or name + ":0"
        self.nodes.append(helper.make_node(kind, inputs, [output], name=name, **attrs))
        return output

    def reshape(self, x, dims):
        return self.op("Reshape", x, self.constant("shape", dims, np.int64))

    def tile(self, x, repeats):
        return self.op("Tile", x, self.constant("repeats", repeats, np.int64))

    def matmul(self, a, b, lhs, rhs, output=None):
        batch, heads, rows, inner = lhs
        if rhs[:3] != (batch, heads, inner) or batch != 1 or min(*lhs, *rhs) <= 0:
            raise ValueError(f"Unexpected attention matrix shapes {lhs} and {rhs}")
        cols = rhs[-1]
        a = self.op("Transpose", a, perm=[0, 1, 3, 2])
        a = self.reshape(a, [batch, heads * inner, rows, 1])
        b = self.reshape(b, [batch, heads * inner, 1, cols])
        a = self.tile(a, [1, 1, 1, cols])
        b = self.tile(b, [1, 1, rows, 1])
        products = self.op("Mul", a, b)
        weight = self.constant("reduce_head", np.ones((heads, inner, 1, 1)))
        return self.op("Conv", products, weight, output=output, kernel_shape=[1, 1], group=heads)

    def masked_softmax(self, scores, mask, dims, output=None):
        batch, heads, rows, cols = dims
        mask = self.op("Cast", mask, to=TensorProto.FLOAT)
        mask = self.reshape(mask, [batch, 1, 1, cols])
        mask = self.tile(mask, [1, heads, rows, 1])
        scores = self.op("Mul", scores, mask)
        # -10000 is exactly representable in FP16. The exponent of a masked
        # score underflows to zero. If every token is masked, the row is uniform.
        penalty = self.op("Sub", mask, self.constant("one", np.ones(dims)))
        penalty = self.op("Mul", penalty, self.constant("penalty", np.full(dims, 10000)))
        scores = self.op("Add", scores, penalty)
        maximum = self.op("ReduceMax", scores, axes=[3], keepdims=1)
        maximum = self.tile(maximum, [1, 1, 1, cols])
        centered = self.op("Sub", scores, maximum)
        numerator = self.op("Exp", centered)
        axes = self.constant("axis", [3], np.int64)
        denominator = self.op("ReduceSum", numerator, axes, keepdims=1)
        denominator = self.tile(denominator, [1, 1, 1, cols])
        return self.op("Div", numerator, denominator, output=output)


def clean(model, nodes):
    graph = model.graph
    needed = {value.name for value in graph.output}
    retained = []
    for node in reversed(nodes):
        if needed.intersection(node.output):
            needed.update(node.input)
            retained.append(node)
    initializers = [w for w in graph.initializer if w.name in needed]
    del graph.node[:]
    graph.node.extend(reversed(retained))
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.value_info[:]
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def lower(source, parts):
    model = shape_inference.infer_shapes(onnx.load(source))
    graph = model.graph
    shapes = {
        v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        for v in (*graph.input, *graph.value_info, *graph.output)
    }
    producers = {y: n for n in graph.node for y in n.output}
    lower = Lowering(graph)
    replaced = {"matmul": 0, "softmax": 0}
    for node in graph.node:
        is_attention = "/attention/self/" in node.name
        y = node.output[0]
        if is_attention and node.op_type == "MatMul" and "matmul" in parts:
            a, b = node.input
            lhs, rhs = shapes[a], shapes[b]
            if len(lhs) == len(rhs) == 4:
                lower.scope = node.name + "/npu"
                lower.matmul(a, b, lhs, rhs, output=y)
                replaced["matmul"] += 1
                continue
        if is_attention and node.op_type == "Softmax" and "softmax" in parts:
            add = producers[node.input[0]]
            axis = next((a.i for a in node.attribute if a.name == "axis"), -1)
            if add.op_type != "Add" or add.input[1] != "/classifier/bert/Mul_output_0" or axis not in (-1, 3):
                raise ValueError(f"Unexpected masked Softmax pattern at {node.name}")
            lower.scope = node.name + "/npu"
            lower.masked_softmax(add.input[0], "attention_mask", shapes[y], output=y)
            replaced["softmax"] += 1
            continue
        lower.nodes.append(node)
    for part in parts:
        expected = 12 if part == "matmul" else 6
        if replaced[part] != expected:
            raise ValueError(f"Expected {expected} {part} rewrites, got {replaced[part]}")
    return clean(model, lower.nodes)


def attention_probe():
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
        for name, dims in [
            ("q", [1, 8, 64, 32]), ("k", [1, 8, 32, 64]), ("v", [1, 8, 64, 32]),
        ]
    ]
    inputs.append(helper.make_tensor_value_info("mask", TensorProto.INT32, [1, 64]))
    out = helper.make_tensor_value_info("context", TensorProto.FLOAT, [1, 8, 64, 32])
    graph = helper.make_graph([], "MiniRBT_attention_core", inputs, [out])
    lower = Lowering(graph)
    lower.scope = "qk"
    scores = lower.matmul("q", "k", (1, 8, 64, 32), (1, 8, 32, 64))
    scale = lower.constant("scale", np.full((1, 8, 64, 64), 1 / np.sqrt(32)))
    scores = lower.op("Mul", scores, scale)
    lower.scope = "masked_softmax"
    prob = lower.masked_softmax(scores, "mask", (1, 8, 64, 64))
    lower.scope = "pv"
    lower.matmul(prob, "v", (1, 8, 64, 64), (1, 8, 64, 32), output="context")
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    return clean(model, lower.nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--parts", nargs="+", choices=["matmul", "softmax"], default=["matmul", "softmax"])
    args = parser.parse_args()
    model = lower(args.source, set(args.parts)) if args.source else attention_probe()
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
