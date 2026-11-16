"""Exercise the installed Kirin9030 host tilers with MiniRBT attention shapes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def tensor(name, desc):
    return {
        "name": name,
        "dtype": desc["type"],
        "format": desc["format"],
        "shape": desc["shape"],
        "ignore": False,
        "ori_format": desc["format"],
        "ori_shape": desc["shape"],
        "data_file": "",
    }


def run_case(spec_path, out, argv, env):
    spec, = json.loads(spec_path.read_text())
    name = spec_path.stem
    op = spec["op"]
    case = out / name
    case.mkdir()
    inputs = ("x",) if op == "SoftmaxV2" else ("x1", "x2")
    write_json(case / "inputs.json", [tensor(key, desc) for key, desc in zip(inputs, spec["input_desc"], strict=True)])
    write_json(case / "outputs.json", [tensor("y", spec["output_desc"][0])])
    attrs = [{"name": attr["name"], "dtype": attr["type"], "value": attr["value"]} for attr in spec["attr"]]
    write_json(case / "attrs.json", attrs)
    with (case / "tiling.log").open("w") as log:
        process = subprocess.run(
            [*argv, op, "kirin9030", str(case / "tiling.bin"), str(case / "run.bin"),
             "inputs.json", "outputs.json", "attrs.json"],
            cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    report = {"op": op, "inputs": [desc["shape"] for desc in spec["input_desc"]], "exit_code": process.returncode}
    if process.returncode == 0 and (case / "run.bin").is_file():
        result = json.loads((case / "run.bin").read_text())
        report.update(block_dim=result["block_dim"], tiling_key=result["tiling_key"])
        report["tiling_bytes"] = len(bytes.fromhex(result["tiling_data"]))
        report["passed"] = report["block_dim"] > 0 and report["tiling_bytes"] > 0
    else:
        report["passed"] = False
        log = (case / "tiling.log").read_text(errors="replace")
        if "CoreNum is 0" in log:
            report["error"] = "CoreNum is 0"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cann_include", type=Path)
    parser.add_argument("mobile_ddk_root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    ddk = args.mobile_ddk_root.resolve()
    tools = ddk / "tools"
    master = tools / "tools_omg/master/lib64"
    vendor = (tools / "platform/kirin9030/lib64").resolve()
    tiling = vendor / "libai_npucore_ascendc.so"
    exe = tools / "tools_ascendc/package/ascendc_tiling_tool"
    args.out.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="tiling.", dir=args.out.resolve()))
    print(f"Tiling logs and results in {out}", flush=True)
    schema = out / "native_tiling_schema.so"
    with (out / "build.log").open("w") as log:
        subprocess.run(
            ["g++", "-std=c++17", "-shared", "-fPIC", "-I", str(args.cann_include.resolve()),
             str(here / "plugin/native_tiling_schema.cpp"), "-L", str(master), "-lai_register",
             "-lgraph_host", "-Wl,--no-as-needed", str(tiling), "-o", str(schema)],
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    env = os.environ.copy()
    env.update(DDK_PATH=str(ddk), SOC_VERSION="kirin9030")
    for key, paths in (("LD_LIBRARY_PATH", (master, vendor)),
                       ("PYTHONPATH", (vendor.parent / "ops/impl",))):
        current = env.get(key)
        values = [str(path) for path in paths]
        if current:
            values.append(current)
        env[key] = ":".join(values)
    command = [env.get("OMG_LOADER", "/lib64/ld-linux-x86-64.so.2"), str(exe), str(schema)]
    results = {
        spec.stem: run_case(spec, out, command, env)
        for spec in sorted((here / "specs").glob("*.json"))
        if spec.stem != "add"
    }
    write_json(out / "report.json", results)
    print(f"Passed {sum(result['passed'] for result in results.values())}/{len(results)} host tiling cases")
    if any(not result["passed"] for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
