"""Validate the bundled .ms with the Linux MindSpore Lite CPU runtime."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lite", type=Path, help="Extracted Linux MindSpore Lite directory")
    parser.add_argument("--vectors", type=Path, default=root / "ml/test_vectors.json")
    parser.add_argument("--model", type=Path, default=root / "entry/src/main/resources/rawfile/permission.ms")
    parser.add_argument("--config", type=Path, default=root / "entry/src/main/resources/rawfile/model_config.json")
    args = parser.parse_args()
    runtime = args.lite.resolve() / "runtime"
    rows = json.loads(args.vectors.read_text(encoding="utf-8"))
    model_config = args.config
    thresholds = np.asarray(json.loads(model_config.read_text(encoding="utf-8"))["thresholds"])
    env = os.environ.copy()
    paths = [str(runtime / "lib"), str(runtime / "third_party/glog")]
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / "verify"
        inputs = Path(directory) / "inputs.bin"
        np.asarray([row["input_ids"] + row["attention_mask"] for row in rows], dtype="<i4").tofile(inputs)
        subprocess.run(
            ["g++", "-std=c++17", "-I", runtime, root / "ml/verify_ms.cc",
             "-L", runtime / "lib", "-lmindspore-lite", "-o", executable], env=env, check=True,
        )
        actual = subprocess.check_output([executable, args.model, inputs], env=env, text=True)
    logits = np.asarray([[float(value) for value in line.split()] for line in actual.splitlines()])
    reference = np.asarray([row["logits"] for row in rows])
    np.testing.assert_allclose(logits, reference, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(1 / (1 + np.exp(-logits)) >= thresholds,
                                  1 / (1 + np.exp(-reference)) >= thresholds)
    print(f"Matched {len(rows)} requests. Maximum absolute logit error: {np.max(np.abs(logits-reference)):.8g}")


if __name__ == "__main__":
    main()
