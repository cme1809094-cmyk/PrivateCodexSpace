"""Freeze independently reviewed sources without parsing any reserved prompts."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from prepare_intent_contrast_data import inspect
from train_intent_contrast import ROOT, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ml/data/permission_intent_v3")
    parser.add_argument("--head-comparison", type=Path, required=True)
    args = parser.parse_args()
    destination = args.data_dir / "freeze_manifest.json"
    if destination.exists():
        raise ValueError("Refusing to replace a pre-training freeze")
    _, _, checked = inspect(args.data_dir, require_review=True)
    if not checked["passed"]:
        raise ValueError(checked["errors"])
    audit = read_json(args.data_dir / "ai_review_report.json")
    if (audit["unresolved_errors"] or len(set(audit["covered_ids"])) != 1400
            or len(set(audit["covered_group_ids"])) != 350):
        raise ValueError("All 1400 rows / 350 groups must be independently AI-reviewed")
    digest = sha256(args.data_dir / "reserved_test.jsonl")
    if audit["source_sha256"]["reserved_test.jsonl"] != digest:
        raise ValueError("Reserved-file independent review is stale")
    manifest = read_json(args.data_dir / "reserved_test_manifest.json")
    if manifest["sha256"] != digest or not manifest["frozen"]:
        raise ValueError("Reserved owner must finalize its reviewed manifest first")
    choice = read_json(args.head_comparison)
    value = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "frozen_before_training": True,
             "reserved_test_sha256": digest, "reserved_rows": 280, "reserved_groups": 70,
             "audit_sha256": sha256(args.data_dir / "ai_review_report.json"),
             "development_source_sha256": checked["source_sha256"],
             "head_comparison_sha256": sha256(args.head_comparison), "chosen_head": choice["chosen_head"],
             "experiment_targets_sha256": sha256(args.data_dir / "experiment_targets.json"),
             "root_read_reserved_prompts_before_selection": False, "human_review": "pending",
             "scope": "AI-reviewed focused contrast experiment; not a human-verified full-32 benchmark"}
    write_json(destination, value)
    print("PASS: reviewed sources and reserved bytes frozen; reserved prompts not parsed")


if __name__ == "__main__":
    main()
