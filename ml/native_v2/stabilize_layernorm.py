"""Rescale decomposed BERT LayerNorm to reduce FP16 square overflow, preserving FP32 semantics."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def stabilize(source, scale):
    if not 0 < scale <= 1 or math.frexp(scale)[0] != 0.5:
        raise ValueError("Scale must be a positive power of two no greater than one")
    model = copy.deepcopy(source)
    graph = model.graph
    centers = {node.output[0]: node.name.rsplit("/", 1)[0]
               for node in graph.node if node.op_type == "Sub" and "/LayerNorm/" in node.name}
    if not centers:
        raise ValueError("No decomposed BERT LayerNorm centers found")
    epsilon_nodes = {prefix + "/Constant_1" for prefix in centers.values()}
    if len(epsilon_nodes) != len(centers):
        raise ValueError("LayerNorm prefixes must be unique")
    for value, prefix in centers.items():
        consumers = [node for node in graph.node if value in node.input]
        if len(consumers) != 2 or any(not node.name.startswith(prefix + "/") for node in consumers):
            raise ValueError("Unexpected consumers of centered LayerNorm input " + value)
        square = [node for node in consumers if node.op_type == "Mul" and list(node.input) == [value, value]]
        if len(square) != 1:
            raise ValueError("Expected centered-square Mul from rewrite_attention.py norm")
    factor = "__stable_layernorm_scale"
    if any(tensor.name == factor for tensor in graph.initializer):
        raise ValueError("LayerNorm already has a scale initializer")
    graph.initializer.append(numpy_helper.from_array(np.array(scale, np.float32), factor))
    replacements = {}
    nodes = []
    changed_epsilon = set()
    for original in graph.node:
        node = copy.deepcopy(original)
        for index, value in enumerate(node.input):
            if value in replacements:
                node.input[index] = replacements[value]
        if node.name in epsilon_nodes:
            if node.op_type != "Constant" or len(node.attribute) != 1:
                raise ValueError("Unexpected LayerNorm epsilon node")
            epsilon = numpy_helper.to_array(node.attribute[0].t)
            if epsilon.size != 1 or epsilon.dtype != np.float32 or not float(epsilon.reshape(-1)[0]) > 0:
                raise ValueError("Expected a positive scalar FP32 LayerNorm epsilon")
            node.attribute[0].t.CopyFrom(numpy_helper.from_array(epsilon * np.float32(scale * scale)))
            changed_epsilon.add(node.name)
        nodes.append(node)
        value = original.output[0]
        if value in centers:
            scaled = value + "/stable_scaled"
            nodes.append(helper.make_node("Mul", [value, factor], [scaled], name=scaled))
            replacements[value] = scaled
    if changed_epsilon != epsilon_nodes:
        raise ValueError("Not all LayerNorm epsilons were rescaled")
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.value_info[:]
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model, {
        "scale": scale, "epsilon_scale": scale * scale, "layernorm_count": len(centers),
        "centered_outputs": sorted(centers), "fp16_square_safe_centered_abs": math.sqrt(float(np.finfo(np.float16).max)) / scale,
        "formula": "(s*d)/sqrt(mean((s*d)^2)+s^2*eps), d=x-mean(x), s>0",
        "device_tested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    model, report = stabilize(onnx.load(args.source), args.scale)
    onnx.save(model, args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("Rescaled " + str(report["layernorm_count"]) + " LayerNorm centers; scale=" + str(args.scale))


if __name__ == "__main__":
    main()
