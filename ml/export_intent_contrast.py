"""Export locked intent checkpoint and verify both semantic and guarded CPU parity."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from intent_contrast import decode_policy
from permission_intent_model import load_checkpoint
from train_intent_contrast import ROOT, read_json, read_rows, sha256, write_json
from train_permission_intent_v2 import export_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "ml/data/permission_intent_v3/reserved_test.jsonl")
    args = parser.parse_args()
    torch.set_num_threads(4)
    lock = read_json(args.run / "selection_lock.json")
    for file, expected in lock["checkpoint_files_sha256"].items():
        if sha256(args.run / "checkpoint" / file) != expected:
            raise ValueError("Selected checkpoint profile changed: " + file)
    for name, key in (("model.safetensors", "weights_sha256"), ("protocol.json", "protocol_sha256")):
        if sha256(args.run / "checkpoint" / name) != lock[key]:
            raise ValueError("Selected artifact changed before export: " + name)
    model, tokenizer, protocol = load_checkpoint(args.run / "checkpoint")
    reserved = read_rows(args.data)
    smoke = [{"id": "export_smoke_" + str(i), "text": text}
             for i, text in enumerate(("录音。", "拍张照片。", "不要录音。"))]
    # These short, unlabeled engineering inputs ensure even a bucket with no
    # fitting held-out prompt gets a numerical check. They are not accuracy cases.
    report = export_models(model, tokenizer, reserved + smoke, protocol, args.run, 32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    for length, entry in report.items():
        entry["held_out_rows_that_fit"] = sum(
            len(tokenizer(row["text"], truncation=False)["input_ids"]) <= int(length) for row in reserved)
        entry["unlabeled_engineering_smoke_rows"] = len(smoke)
        session = ort.InferenceSession(str(args.run / entry["path"]), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        mismatches = []
        for row in read_rows(args.run / "onnx" / ("reference_" + length + ".jsonl")):
            inputs = {key: np.asarray([row[key]], dtype=np.int32) for key in ("input_ids", "attention_mask")}
            actual = session.run(None, inputs)
            reference = [np.asarray([row[key]], dtype=np.float32)
                         for key in ("capability_logits", "intent_logits")]
            ref_probs = [torch.from_numpy(v).softmax(-1).numpy() for v in reference]
            out_probs = [torch.from_numpy(v).softmax(-1).numpy() for v in actual]
            ref_decoded = decode_policy(*ref_probs, protocol["thresholds"], protocol["policy"])
            out_decoded = decode_policy(*out_probs, protocol["thresholds"], protocol["policy"])
            if not all(np.array_equal(a, b) for a, b in zip(ref_decoded, out_decoded)):
                mismatches.append(row["id"])
        entry["policy_decision_mismatch_ids"] = mismatches
        if mismatches:
            raise ValueError("Guarded decisions differ in bucket " + length)
    report["verification_scope"] = "ONNX Runtime CPU numerical/decision parity only; OMC conversion and physical NPU are not verified."
    report["input_data_sha256"] = sha256(args.data)
    write_json(args.run / "onnx_verification.json", report)
    print(json.dumps({"event": "export_and_policy_parity_passed", "buckets": [16, 32, 64]}))


if __name__ == "__main__":
    main()
