"""Check all MiniRBT V2 nodes against emitted NPU tasks and partition metadata."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


TILING = re.compile(r"Do general op tiling success.*OpType:\[([^]]+)\], OpName:\[([^]]+)\]")
RUNTIME = re.compile(r"set smDesc kernel_type of OP\[([^,]+),(BatchMatMulV2|SoftmaxV2)\] == (\d+), set blockDim == (\d+)")
KERNEL = re.compile(r"SaveKernelBinInfo.*stubName = ([^,]+), kernelSize = (\d+)")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory, length):
    frontend = json.loads((directory / "frontend-verified.json").read_text())
    layers = frontend.get("architecture", {}).get("num_hidden_layers", 6)
    expected = {"BatchMatMulV2": 2 * layers, "SoftmaxV2": layers}
    check(frontend["counts"] == expected, f"{length}: wrong frontend V2 counts")
    wanted = {n["name"]: n["type"] for n in frontend["nodes"]}
    log = (directory / "backend.log").read_text()
    lines = log.splitlines()
    tiling, runtime, kernels = {}, {}, {}
    for number, line in enumerate(lines, 1):
        match = TILING.search(line)
        if match and match[2] in wanted:
            tiling[match[2]] = {"type": match[1], "line": number}
        match = RUNTIME.search(line)
        if match:
            check(match[1] not in runtime, f"duplicate runtime task: {match[1]}")
            runtime[match[1]] = {"type": match[2], "kernel_type": int(match[3]),
                                 "block_dim": int(match[4]), "line": number}
        match = KERNEL.search(line)
        if match:
            kernels[match[1]] = {"bytes": int(match[2]), "line": number}
    check(set(tiling) == set(wanted) == set(runtime), f"{length}: missing or extra V2 tasks")
    check(Counter(n["type"] for n in runtime.values()) == expected, "wrong runtime V2 counts")
    check("CoreNum is 0" not in log, f"{length}: zero-core tiling error")
    binary = (directory / "mobile.omc").read_bytes()
    native_kernels = {k: v for k, v in kernels.items() if k.startswith(tuple(t + "_" for t in expected))}
    for kind in expected:
        check(any(k.startswith(kind + "_") for k in native_kernels), f"{length}: no {kind} binary")
    for name, info in native_kernels.items():
        check(info["bytes"] > 0 and name.encode() in binary, f"{length}: kernel absent from OMC: {name}")
    legacy = [k for k in kernels if k.startswith("executor_softmax")]
    check(not legacy, f"{length}: legacy Softmax emitted")
    model = json.loads((directory / "mobile.json").read_text())
    ops = [op for graph in model["graph"] for op in graph["op"]]
    subgraphs = [{a["key"]: a["value"] for a in op.get("attr", [])}
                 for op in ops if op["type"] == "GraphOp"]
    partitions = Counter(a["cl_name"]["s"] for a in subgraphs)
    check(partitions == {"NPUCL": 1}, f"{length}: CPU or other graph partitions: {partitions}")
    inputs = {op["name"]: op["output_desc"][0] for op in ops if op["type"] == "Data"}
    check({name: (d["dtype"], d["shape"]["dim"]) for name, d in inputs.items()} ==
          {name: (7, [1, length]) for name in ("input_ids", "attention_mask")}, "wrong model inputs")
    outputs = [d for op in ops if op["type"] == "NetOutput" for d in op["input_desc"]]
    check([(d["name"], d["dtype"], d["shape"]["dim"]) for d in outputs] ==
          [("logits", 1, [1, 5])], "wrong model output")
    nodes = []
    for node in frontend["nodes"]:
        name, kind = node["name"], node["type"]
        check(tiling[name]["type"] == runtime[name]["type"] == kind, f"{name}: mismatched type")
        check(runtime[name]["kernel_type"] == 0 and runtime[name]["block_dim"] > 0,
              f"{name}: invalid NPU task metadata")
        nodes.append({**node, "backend_tiling_line": tiling[name]["line"],
                      "backend_npu_task_line": runtime[name]["line"],
                      "block_dim": runtime[name]["block_dim"],
                      "emitted_kernels_for_type": [k for k in native_kernels if k.startswith(kind + "_")]})
    return {"length": length, "passed": True, "device_tested": False,
            "model": str(directory / "mobile.omc"), "bytes": len(binary),
            "sha256": hashlib.sha256(binary).hexdigest(),
            "backend_log": str(directory / "backend.log"), "model_dump": str(directory / "mobile.json"),
            "partitions": {"NPUCL": partitions["NPUCL"], "CPUCL": partitions["CPUCL"]},
            "compiler_subgraph_memory_bytes": subgraphs[0]["memory_size"]["i"],
            "native_node_counts": dict(Counter(n["type"] for n in nodes)),
            "native_kernels": native_kernels, "legacy_softmax_kernels": legacy, "nodes": nodes,
            "evidence_scope": "Per-node vendor tiling success and NPU RuntimeConfig, emitted native kernel binaries present in OMC, compiled partition metadata; no device execution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    result = {str(n): verify(build / f"l{n}", n) for n in (16, 32, 64)}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for n, r in result.items():
        print(f"{n}: {r['native_node_counts']}, partitions={r['partitions']}, bytes={r['bytes']}, PASS")


if __name__ == "__main__":
    main()
