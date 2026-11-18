"""Validate development shards without exposing reserved prompts to developers."""
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path

from validate_permission_intent_v2 import normalized
from dataset import FAMILIES


ROOT = Path(__file__).resolve().parents[1]
GOLD = ("context", "actions", "intent_status", "required_capabilities", "forbidden_capabilities",
        "default_state_for_other_capabilities", "constraints", "needs_clarification", "clarification_question")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(row, fields):
    return json.dumps({key: row[key] for key in fields}, ensure_ascii=False, sort_keys=True)


def inspect(data_dir, require_review=False):
    contract = json.loads((data_dir / "contract.json").read_text())
    catalog = json.loads((ROOT / "ml/data/permission_intent_v2/capability_catalog.json").read_text())
    caps = {c["id"] for c in catalog["capabilities"]}
    original = [json.loads(line) for line in (ROOT / "ml/data/permission_intent_v2/expansion/all_instructions.jsonl").read_text().splitlines()]
    prior_texts = {normalized(row["text"]): row["id"] for row in original}
    for split, families in FAMILIES.items():
        for _, phrases in families:
            for i, text in enumerate(phrases.split("|")):
                prior_texts[normalized(text)] = f"historical:{split}:{i}"
    errors, rows, hashes = [], [], {}
    def check(condition, message):
        if not condition:
            errors.append(message)
    for author, spec in contract["authors"].items():
        if author == "reserved_test_owner":
            continue
        path = data_dir / spec["file"]
        if not path.exists():
            errors.append("Missing development shard: " + path.name)
            continue
        selected = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        hashes[path.name] = sha256(path)
        check(len(selected) == spec["rows"], "Author row quota: " + author)
        rows.extend(selected)
        stratifier = "focus_capability" if "capabilities" in spec else "status_stratum"
        values = spec.get("capabilities", spec.get("status_strata"))
        for value in values:
            subset = [row for row in selected if row[stratifier] == value]
            for split, key in (("train_candidate", "train_groups_per_"), ("validation_candidate", "validation_groups_per_")):
                expected = spec[key + ("capability" if stratifier == "focus_capability" else "status")]
                check(sum(row["split_candidate"] == split for row in subset) == expected * 4,
                      f"Author stratum/split quota: {author}/{value}/{split}")
    groups = defaultdict(list)
    seen_ids, seen_texts = set(), {}
    for row in rows:
        rid = row.get("id")
        check(rid not in seen_ids, "Repeated row ID: " + str(rid))
        seen_ids.add(rid)
        for field in contract["additional_required_fields"] + list(GOLD) + ["id", "text", "primary_group", "tags", "source", "review"]:
            check(field in row, str(rid) + ": missing " + field)
        check(row.get("schema_version") == contract["schema_version"], str(rid) + ": wrong schema")
        text = row.get("text", "")
        check(isinstance(text, str) and bool(text.strip()), str(rid) + ": empty text")
        key = normalized(text)
        check(key not in prior_texts, f"Old text duplicated: {rid}/{prior_texts.get(key)}")
        check(key not in seen_texts, f"New text duplicated: {rid}/{seen_texts.get(key)}")
        seen_texts[key] = rid
        for field in ("required_capabilities", "forbidden_capabilities", "constraints", "tags"):
            check(isinstance(row[field], list) and all(isinstance(x, str) for x in row[field]), f"{rid}: invalid {field}")
        req, deny = set(row["required_capabilities"]), set(row["forbidden_capabilities"])
        check((req | deny).issubset(caps), str(rid) + ": unknown global capability")
        check(not req & deny, str(rid) + ": contradictory state labels")
        check(row["default_state_for_other_capabilities"] == "unmentioned", str(rid) + ": wrong default state")
        check(row["intent_status"] in ("in_scope", "ambiguous", "no_device_capability", "out_of_scope"), str(rid) + ": invalid status")
        if row["intent_status"] in ("ambiguous", "no_device_capability", "out_of_scope"):
            check(not req, str(rid) + ": nonexecuting intent requires capability")
        clarify = row["needs_clarification"]
        check(isinstance(clarify, bool) and clarify == (row["intent_status"] == "ambiguous"), str(rid) + ": invalid clarification flag")
        check(isinstance(row["clarification_question"], str) and bool(row["clarification_question"].strip()) if clarify
              else row["clarification_question"] is None, str(rid) + ": invalid clarification question")
        check(isinstance(row["context"], dict), str(rid) + ": context must be explicit object")
        check("context_required" not in row["tags"], str(rid) + ": hidden context is outside this text-only task")
        check(isinstance(row["actions"], list) and all(isinstance(a, dict) and all(isinstance(a.get(k), str) and a[k].strip()
                    for k in ("verb", "resource", "scope")) for a in row["actions"]), str(rid) + ": invalid actions")
        check(row["split_candidate"] in ("train_candidate", "validation_candidate"), str(rid) + ": reserved input in development")
        check(row["contrast_role"] in ("a", "b"), str(rid) + ": invalid contrast role")
        check(row["family_id"] == row["scenario_group_id"] + "_" + row["contrast_role"], str(rid) + ": family/group mismatch")
        check(row["focus_capability"] in contract["focus_capabilities"] or row["focus_capability"] is None, str(rid) + ": invalid focus")
        check(row["status_stratum"] in (None, "ambiguous", "no_device_capability", "out_of_scope"), str(rid) + ": invalid status stratum")
        if row["status_stratum"]:
            check(row["intent_status"] == (row["status_stratum"] if row["contrast_role"] == "a" else "in_scope"), str(rid) + ": wrong status contrast")
        check(row["source"].get("type") == "agent_generated", str(rid) + ": provenance missing")
        check(row["review"].get("human_review") == "pending", str(rid) + ": false human approval")
        check(row["review"].get("semantic_ai") == "reviewed", str(rid) + ": author review incomplete")
        check(row["review"].get("cross_review") in ("pending", "reviewed"), str(rid) + ": invalid cross review")
        if require_review:
            check(row["review"]["cross_review"] == "reviewed", str(rid) + ": independent review incomplete")
        groups[row["scenario_group_id"]].append(row)
    for gid, members in groups.items():
        check(len(members) == 4, str(gid) + ": group must have four rows")
        check(len({row["split_candidate"] for row in members}) == 1, str(gid) + ": group leaks across splits")
        check(len({row["scenario_summary"] for row in members}) == 1, str(gid) + ": inconsistent group summary")
        roles = {role: [row for row in members if row["contrast_role"] == role] for role in ("a", "b")}
        for role, subset in roles.items():
            check(len(subset) == 2 and len({signature(row, GOLD) for row in subset}) == 1, str(gid) + ": role meaning drifts: " + role)
        if all(roles.values()):
            check(signature(roles["a"][0], ("required_capabilities", "forbidden_capabilities", "intent_status")) !=
                  signature(roles["b"][0], ("required_capabilities", "forbidden_capabilities", "intent_status")),
                  str(gid) + ": contrast changes no model target")
    counts = Counter(row["split_candidate"] for row in rows)
    check(len(rows) == 1120 and len(groups) == 280, "Development total must be 1120 rows / 280 groups")
    check(counts == {"train_candidate": 896, "validation_candidate": 224}, "Development split counts mismatch")
    if require_review:
        audit = json.loads((data_dir / "ai_review_report.json").read_text())
        check(audit["unresolved_errors"] == 0, "Independent review has unresolved errors")
        check(set(seen_ids).issubset(audit["covered_ids"]), "Independent review row coverage incomplete")
        for name, digest in hashes.items():
            check(audit["source_sha256"].get(name) == digest, "Stale review for " + name)
    report = {"schema_version": contract["schema_version"], "passed": not errors, "errors": errors,
              "development_rows": len(rows), "development_groups": len(groups), "split_counts": dict(counts),
              "source_sha256": hashes, "status_counts": dict(Counter(row["intent_status"] for row in rows)),
              "reserved_prompts_read": False, "human_review": "pending", "training_performed_by_this_check": False}
    return rows, groups, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ml/data/permission_intent_v3")
    parser.add_argument("--require-review", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    rows, groups, report = inspect(args.data_dir, args.require_review)
    (args.data_dir / "development_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["errors"]:
        raise SystemExit("\n".join(report["errors"]))
    if args.render:
        (args.data_dir / "development.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rows, key=lambda r: r["id"])))
        fields = ["scenario_group_id", "split", "summary", "a_text1", "a_text2", "a_required", "a_forbidden", "a_intent",
                  "b_text1", "b_text2", "b_required", "b_forbidden", "b_intent", "human_review", "human_decision", "human_notes"]
        with (args.data_dir / "development_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for gid, members in sorted(groups.items()):
                view = {"scenario_group_id": gid, "split": members[0]["split_candidate"], "summary": members[0]["scenario_summary"], "human_review": "pending"}
                for role in ("a", "b"):
                    selected = [r for r in members if r["contrast_role"] == role]
                    view.update({role + "_text1": selected[0]["text"], role + "_text2": selected[1]["text"],
                                 role + "_required": " ; ".join(selected[0]["required_capabilities"]),
                                 role + "_forbidden": " ; ".join(selected[0]["forbidden_capabilities"]), role + "_intent": selected[0]["intent_status"]})
                writer.writerow(view)
    print(f"PASS: {len(rows)} development rows, {len(groups)} locked contrast groups; reserved prompts unopened")


if __name__ == "__main__":
    main()
