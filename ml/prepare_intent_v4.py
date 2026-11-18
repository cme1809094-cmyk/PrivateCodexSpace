"""Validate development data and freeze hashes without reading reserved prompts."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from pathlib import Path

from transformers import BertTokenizerFast

from train_intent_contrast import ROOT, read_json, read_rows, sha256, write_json
from validate_permission_intent_v2 import normalized
from dataset import FAMILIES


DATA = ROOT / "ml/data/permission_intent_v4"


def canonical(row):
    return (row["intent_status"], tuple(sorted(row["required_capabilities"])),
            tuple(sorted(row["forbidden_capabilities"])), row["ambiguity_type"])


def inspect(require_review=False, tokenizer_path=None):
    catalog = read_json(ROOT / "ml/data/permission_intent_v2/capability_catalog.json")
    caps = {c["id"] for c in catalog["capabilities"]}
    old = read_rows(ROOT / "ml/data/permission_intent_v2/expansion/all_instructions.jsonl")
    for p in (ROOT / "ml/data/permission_intent_v3/shards").glob("*.jsonl"):
        old += read_rows(p)
    old += read_rows(ROOT / "ml/data/permission_intent_v3/reserved_test.jsonl")  # Known v3 regression only.
    previous = {normalized(r["text"]) for r in old}
    for families in FAMILIES.values():
        for _, phrases in families:
            previous.update(normalized(text) for text in phrases.split("|"))
    splits = {s: read_rows(DATA / (s + ".jsonl")) for s in ("train", "validation")}
    errors, seen_ids, seen_texts, families = [], set(), set(), defaultdict(list)
    def check(condition, message):
        if not condition:
            errors.append(message)
    for split, rows in splits.items():
        for row in rows:
            rid = row["id"]
            check(rid not in seen_ids, "Duplicate ID: " + rid)
            seen_ids.add(rid)
            key = normalized(row["text"])
            check(key not in seen_texts and key not in previous, "Exact prior/new text duplicate: " + rid)
            seen_texts.add(key)
            check(row["split"] == split, "Split mismatch: " + rid)
            req, deny = set(row["required_capabilities"]), set(row["forbidden_capabilities"])
            check(not req & deny and (req | deny).issubset(caps), "Invalid capability states: " + rid)
            check(row["intent_status"] in ("in_scope", "ambiguous", "no_device_capability", "out_of_scope"), "Invalid intent: " + rid)
            check(row["intent_status"] == "in_scope" or not req, "Nonexecuting intent requires capability: " + rid)
            check(row["ambiguity_type"] in ("conflict", "source", "target", "scope") if row["intent_status"] == "ambiguous"
                  else row["ambiguity_type"] is None, "Invalid ambiguity subtype: " + rid)
            check(row.get("context", {}) == {}, "Hidden context outside text-only task: " + rid)
            check(row["review"]["human_review"] == "pending", "Unsubstantiated human approval: " + rid)
            if require_review:
                expected = "template_reviewed" if split == "train" else "reviewed"
                check(row["review"]["cross_review"] == expected, "Review incomplete: " + rid)
            families[row["family_id"]].append(row)
    for fid, rows in families.items():
        check(len(rows) == 2, "Family must contain two paraphrases: " + fid)
        check(len({r["split"] for r in rows}) == 1 and len({canonical(r) for r in rows}) == 1,
              "Family leaks or gold drifts: " + fid)
    val = splits["validation"]
    check(len(val) == 224 and len({r["family_id"] for r in val}) == 112, "Validation quota")
    check(Counter(r["intent_status"] for r in val) == {"in_scope": 128, "ambiguous": 32, "no_device_capability": 32, "out_of_scope": 32}, "Validation status quota")
    for cap in caps:
        check(len({r["family_id"] for r in val if cap in r["required_capabilities"]}) >= 2,
              "Insufficient validation capability families: " + cap)
        for field in ("required_capabilities", "forbidden_capabilities"):
            check(any(cap in r[field] for r in splits["train"]), "Missing augmented train state: " + cap + "/" + field)
    if tokenizer_path is None:
        packaged = ROOT / "run/checkpoint"
        tokenizer_path = packaged if packaged.exists() else Path('/workspace/work/tinybert/downloads')
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
    all_rows = splits["train"] + val
    lengths = tokenizer([r["text"] for r in all_rows], truncation=False, return_length=True)["length"]
    check(max(lengths) <= 64, "An instruction would be truncated")
    report = {"passed": not errors, "errors": errors, "rows": {s: len(r) for s, r in splits.items()},
              "source_sha256": {s + ".jsonl": sha256(DATA / (s + ".jsonl")) for s in splits},
              "max_tokens_with_special": max(lengths), "reserved_prompts_read": False, "human_review": "pending"}
    if require_review:
        audit = read_json(DATA / "ai_review_report.json")
        check(audit["unresolved_errors"] == 0 and audit["training_template_review"]["passed"], "Unresolved independent review")
        check({r["id"] for r in val} == set(audit["covered_validation_ids"]), "Validation review coverage differs")
        for name, digest in report["source_sha256"].items():
            check(audit["source_sha256"][name] == digest, "Stale independent review: " + name)
        report.update(passed=not errors, errors=errors)
    return splits, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-review", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--tokenizer", type=Path, help="Local paired checkpoint/tokenizer; packaged run/checkpoint is auto-detected")
    args = parser.parse_args()
    splits, report = inspect(args.require_review or args.freeze, args.tokenizer)
    write_json(DATA / "development_validation.json", report)
    if not report["passed"]:
        raise ValueError(report["errors"])
    for split, rows in splits.items():
        with (DATA / (split + "_review.csv")).open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["id", "family_id", "text", "intent_status", "required_capabilities", "forbidden_capabilities", "ambiguity_type", "human_review", "human_notes"]
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                view = {k: row.get(k) for k in fields}
                view.update(human_review="pending", human_notes="")
                writer.writerow(view)
    if args.freeze:
        if (DATA / "freeze_manifest.json").exists():
            raise ValueError("Refusing to replace an existing freeze")
        audit = read_json(DATA / "ai_review_report.json")
        manifest = read_json(DATA / "reserved_test_manifest.json")
        if not manifest["frozen"] or len(set(audit["covered_reserved_ids"])) != 320:
            raise ValueError("Reserved author/reviewer has not finalized the holdout")
        hashes = {}
        for name in ("lexicon.json", "train.jsonl", "validation.jsonl", "reserved_test.jsonl", "generate_intent_v4.py"):
            path = ROOT / "ml" / name if name.endswith(".py") else DATA / name
            hashes[name] = sha256(path)
            if audit["source_sha256"][name] != hashes[name]:
                raise ValueError("Stale source review: " + name)
        if manifest["sha256"] != hashes["reserved_test.jsonl"]:
            raise ValueError("Reserved manifest changed")
        write_json(DATA / "freeze_manifest.json", {"frozen_before_training": True,
                   "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": hashes,
                   "audit_sha256": sha256(DATA / "ai_review_report.json"), "plan_sha256": sha256(DATA / "experiment_plan.json"),
                   "root_read_reserved_before_selection": False, "human_review": "pending"})
    print("PASS: v4 development checked; reserved prompts not parsed")


if __name__ == "__main__":
    main()
