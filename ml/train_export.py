"""Fine-tune MiniRBT on the bundled synthetic requests and export ONNX."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from dataset import CAPABILITIES, load_split

MODEL_ID = "hfl/minirbt-h256"
REVISION = "0ba7e5a947e3898b65bc742048790acd54fecf76"
LENGTH = 64
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "entry/src/main/resources/rawfile"


class ExportModel(torch.nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, input_ids, attention_mask):
        return self.classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
            return_dict=False,
        )[0]


def encode(tokenizer, rows, length):
    encoded = tokenizer(
        [row[0] for row in rows], padding="max_length", truncation=True,
        max_length=length, return_tensors="pt", return_token_type_ids=False,
    )
    return encoded, torch.tensor([row[1] for row in rows], dtype=torch.float32)


def scores(model, encoded):
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(**encoded).logits).cpu().numpy()


def evaluate(probabilities, labels, thresholds):
    predicted = probabilities >= np.asarray(thresholds)
    expected = labels.numpy().astype(bool)
    exact = np.all(predicted == expected, axis=1)
    over = np.any(predicted[:, :4] & ~expected[:, :4], axis=1)
    return {
        "samples": len(labels), "exact": int(exact.sum()), "overpredicted": int(over.sum()),
        "missed_by_label": ((~predicted) & expected).sum(axis=0).tolist(),
    }


def tune_thresholds(probabilities, labels):
    expected = labels.numpy().astype(bool)
    thresholds = []
    for i in range(len(CAPABILITIES)):
        best = None
        for threshold in np.arange(0.25, 0.86, 0.025):
            predicted = probabilities[:, i] >= threshold
            fp = np.sum(predicted & ~expected[:, i])
            fn = np.sum(~predicted & expected[:, i])
            ranking = (2 * fp + fn, fp, abs(threshold - 0.55))
            if best is None or ranking < best[0]:
                best = (ranking, threshold)
        thresholds.append(round(float(best[1]), 3))
    return thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=MODEL_ID)
    parser.add_argument("--source-id", help="Public model ID to record when --base is a local snapshot")
    parser.add_argument("--revision", help="Pinned source revision")
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "model-build")
    parser.add_argument("--raw", type=Path, default=RAW, help="Destination for model-specific vocab and thresholds")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    random.seed(2026)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.raw
    raw.mkdir(parents=True, exist_ok=True)

    revision = args.revision or (REVISION if args.base == MODEL_ID else None)
    kwargs = {"revision": revision} if revision else {}
    tokenizer = BertTokenizerFast.from_pretrained(args.base, **kwargs)
    config = BertConfig.from_pretrained(args.base, **kwargs)
    config.num_labels = len(CAPABILITIES)
    config.problem_type = "multi_label_classification"
    config.id2label = dict(enumerate(CAPABILITIES))
    config.label2id = {k:i for i,k in enumerate(CAPABILITIES)}
    model, loading = BertForSequenceClassification.from_pretrained(
        args.base, config=config, attn_implementation="eager", output_loading_info=True, **kwargs,
    )
    missing_encoder = [key for key in loading["missing_keys"]
                       if key.startswith("bert.") and not key.startswith("bert.pooler.")]
    if missing_encoder or loading.get("mismatched_keys"):
        raise ValueError(f"Pretrained encoder weights were not loaded: {loading}")
    print(f"Loaded {config.num_hidden_layers} layers, {config.num_attention_heads} heads, hidden={config.hidden_size}", flush=True)
    train_rows = load_split("train")
    train_inputs, train_labels = encode(tokenizer, train_rows, 32)
    val_inputs, val_labels = encode(tokenizer, load_split("validation"), LENGTH)
    test_rows = load_split("test")
    test_inputs, test_labels = encode(tokenizer, test_rows, LENGTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=0.01)
    best = None

    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(train_rows)).split(12):
            inputs = {key: value[indices] for key, value in train_inputs.items()}
            loss = model(**inputs, labels=train_labels[indices]).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_prob = scores(model, val_inputs)
        thresholds = tune_thresholds(val_prob, val_labels)
        metric = evaluate(val_prob, val_labels, thresholds)
        ranking = (metric["exact"] - metric["overpredicted"], metric["exact"],
                   -metric["overpredicted"], -float(np.mean((val_prob - val_labels.numpy())**2)))
        if best is None or ranking > best[0]:
            best = (ranking, copy.deepcopy(model.state_dict()), thresholds, epoch + 1, metric)
        if (epoch + 1) % 6 == 0:
            print(f"epoch {epoch + 1}: {metric}", flush=True)

    _, state, thresholds, epoch, validation = best
    model.load_state_dict(state)
    model.eval()
    checkpoint = args.output / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    tokenizer.save_vocabulary(str(raw))

    wrapper = ExportModel(model).eval()
    sample = tuple(test_inputs[key][:1].to(torch.int32) for key in ("input_ids", "attention_mask"))
    onnx_path = args.output / "permission.onnx"
    torch.onnx.export(
        wrapper, sample, onnx_path, input_names=["input_ids", "attention_mask"],
        output_names=["logits"], opset_version=13, dynamo=False,
    )
    onnx.checker.check_model(onnx_path)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(onnx_path), session_options, providers=["CPUExecutionProvider"])
    vectors = []
    max_error = 0.0
    for i, (phrase, labels) in enumerate(test_rows):
        inputs = [test_inputs[key][i:i+1].to(torch.int32) for key in ("input_ids", "attention_mask")]
        with torch.inference_mode():
            reference = wrapper(*inputs).numpy()
        feed = {key: value.numpy() for key, value in zip(("input_ids", "attention_mask"), inputs)}
        actual = session.run(["logits"], feed)[0]
        max_error = max(max_error, float(np.max(np.abs(reference - actual))))
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)
        vectors.append({"text": phrase, "labels": labels, "input_ids": inputs[0][0].tolist(),
                        "attention_mask": inputs[1][0].tolist(), "logits": reference[0].tolist()})
    test_prob = scores(model, test_inputs)
    test = evaluate(test_prob, test_labels, thresholds)
    report = {"checkpoint": args.source_id or args.base, "revision": revision, "synthetic_data": True,
              "architecture": {k:getattr(config,k) for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "vocab_size")},
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "pretrained_loading": loading,
              "labels": CAPABILITIES, "training_samples": len(train_rows), "selected_epoch": epoch,
              "thresholds": thresholds, "validation": validation, "test": test,
              "pytorch_onnx_max_abs_logit_error": max_error}
    (raw / "model_config.json").write_text(
        json.dumps({"thresholds": thresholds}, ensure_ascii=False) + "\n", encoding="utf-8")
    (raw / "hiai_reference.json").write_text(
        json.dumps([{"text": row["text"], "logits": row["logits"]} for row in vectors],
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "golden.json").write_text(json.dumps(vectors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "test_vectors.json").write_text(json.dumps(vectors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
