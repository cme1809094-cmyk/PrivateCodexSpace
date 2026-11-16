"""Check the V2 types, attributes and FP16 shapes in an exported OMG intermediate graph."""
import argparse
from collections import Counter
import json
from pathlib import Path


def verify(length, graph):
    counts = Counter()
    layouts = Counter()
    nodes = []
    for node in graph:
        tail = node["name"].split("/")[-3:]
        if tail[1:2] != ["native"]:
            continue
        layer = node["name"].split("/encoder/layer.")[1].split("/")[0]
        query = 1 if layer == "5" else length
        op = node["type"]
        attrs = {entry["key"]: entry["value"] for entry in node.get("attr", [])}
        if tail[2] == "SoftmaxV2":
            expected = [[8 * query, length]] * 2
            valid = (
                op == "SoftmaxV2" and attrs["axes"]["list"]["i"] == [-1]
                and attrs["half_to_float"]["b"] is False
            )
        elif tail[2] == "MatMulV2":
            expected = (
                [[8, query, 32], [8, 32, length], [8, query, length]]
                if tail[0] == "MatMul"
                else [[8, query, length], [8, length, 32], [8, query, 32]]
            )
            valid = (
                op == "BatchMatMulV2" and attrs["adj_x1"]["b"] is False
                and attrs["adj_x2"]["b"] is False
            )
        else:
            continue

        descs = node["input_desc"] + node["output_desc"]
        shapes = [desc["shape"]["dim"] for desc in descs]
        if not valid or shapes != expected or any(desc["dtype"] != 2 for desc in descs):
            raise ValueError(f"Unexpected V2 type, attributes, or FP16 shapes for {node['name']}")
        counts[op] += 1
        layouts.update(desc["layout"] for desc in descs)
        nodes.append({"name": node["name"], "type": op, "shapes": shapes})
    if counts != {"BatchMatMulV2": 12, "SoftmaxV2": 6}:
        raise ValueError(f"Unexpected V2 node counts {counts}")
    return {"length": length, "counts": counts, "ir_layout_labels": layouts, "nodes": nodes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("length", type=int, choices=(16, 32, 64))
    parser.add_argument("graph", type=Path)
    args = parser.parse_args()
    document = json.loads(args.graph.read_text())
    print(json.dumps(verify(args.length, document["graph"][0]["op"]), indent=2))
