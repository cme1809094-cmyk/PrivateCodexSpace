"""Validate, merge and render the permission-intent expansion candidate splits.

This prepares review data. It never trains a model or freezes a benchmark.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import validate_permission_intent_v2 as seed_validator
from dataset import FAMILIES as HISTORICAL_FAMILIES


GOLD_FIELDS = ("primary_group", "context", "actions", "intent_status",
               "required_capabilities", "forbidden_capabilities",
               "default_state_for_other_capabilities", "constraints",
               "needs_clarification", "clarification_question")
SPLITS = ("train_candidate", "validation_candidate", "test_candidate")
LEXICAL_THRESHOLD = 0.45


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def fingerprint(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def text_grams(text):
    text = seed_validator.normalized(text)
    return {text[i:i + 2] for i in range(len(text) - 1)}


def near_pairs(families):
    """Screen lexical overlap. Human/AI review must decide semantic leakage."""
    items = list(families.items())
    grams = {fid: [text_grams(row["text"]) for row in rows] for fid, rows in items}
    results = []
    for i, (left, lrows) in enumerate(items):
        for right, rrows in items[i + 1:]:
            if lrows[0]["split_candidate"] == rrows[0]["split_candidate"]:
                continue
            best, match = 0.0, None
            for li, lg in enumerate(grams[left]):
                for ri, rg in enumerate(grams[right]):
                    union = lg | rg
                    score = len(lg & rg) / len(union) if union else 0.0
                    if score > best:
                        best, match = score, (lrows[li]["id"], rrows[ri]["id"])
            if best >= LEXICAL_THRESHOLD:
                results.append({"left_family": left, "right_family": right,
                                "left_split": lrows[0]["split_candidate"],
                                "right_split": rrows[0]["split_candidate"],
                                "bigram_jaccard": round(best, 6),
                                "left_example": match[0], "right_example": match[1]})
    return sorted(results, key=lambda x: (-x["bigram_jaccard"], x["left_family"], x["right_family"]))


def inspect(data_dir, require_cross_review=False):
    parent = data_dir.parent
    contract = read_json(data_dir / "contract.json")
    base_contract, catalog, legacy, base_report = seed_validator.inspect(parent)
    errors, warnings = [], []
    def check(condition, message):
        if not condition:
            errors.append(message)
    def string_list(value, where):
        check(isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value),
              where + ": expected nonempty-string list")
        return value if isinstance(value, list) else []
    check(base_report["structure_passed"], "Original seed/catalog validation failed")
    cap_ids = {cap["id"] for cap in catalog["capabilities"]}
    groups = set(base_contract["new_groups"]) | set(base_contract["retained_groups"]) | set(base_contract["corpus_extra_groups"])
    rows, input_hashes = [], {}
    author_counts = {}
    for author, spec in contract["authors"].items():
        path = data_dir / "shards" / (author + ".jsonl")
        if not path.exists():
            check(False, "Missing author shard: " + path.name)
            continue
        authored = read_rows(path)
        rows.extend(authored)
        input_hashes[str(path.relative_to(data_dir))] = sha256(path)
        author_counts[author] = len(authored)
        check(len(authored) == spec["rows"], "Author quota mismatch: " + author)
        check(all(r.get("primary_group") in spec["groups"] for r in authored), "Author group mismatch: " + author)
        if "families_per_group" in spec:
            for group in spec["groups"]:
                selected = [r for r in authored if r.get("primary_group") == group]
                check(len(selected) == spec["families_per_group"] * contract["family_size"], "Group quota: " + group)
                for split, count in spec["family_splits_per_group"].items():
                    check(sum(r.get("split_candidate") == split for r in selected) == count * contract["family_size"],
                          f"Group/split quota: {group}/{split}")
        else:
            for split, count in spec["family_splits"].items():
                check(sum(r.get("split_candidate") == split for r in authored) == count * contract["family_size"],
                      "Author/split quota: " + author + "/" + split)
    ids, seen, families = set(), {}, defaultdict(list)
    legacy_ids = {row["id"] for row in legacy}
    for row in legacy:
        seen[(seed_validator.normalized(row["text"]), fingerprint(row["context"]))] = row["id"]
    split_counts, statuses, review_counts = Counter(), Counter(), Counter()
    positive, prohibited = Counter(), Counter()
    per_split_positive = {s: Counter() for s in SPLITS}
    per_split_prohibited = {s: Counter() for s in SPLITS}
    context_required_counts, context_present_counts = Counter(), Counter()
    for row in rows:
        rid = row.get("id")
        check(isinstance(rid, str) and bool(rid) and rid not in ids, "Invalid/duplicate row ID: " + str(rid))
        check(rid not in legacy_ids, "New ID collides with original seed: " + str(rid))
        ids.add(rid)
        for key in base_contract["corpus_required_fields"] + contract["additional_required_fields"]:
            check(key in row, str(rid) + ": missing " + key)
        check(row.get("schema_version") == contract["schema_version"], str(rid) + ": schema_version")
        check(row.get("primary_group") in groups, str(rid) + ": unknown primary_group")
        text = row.get("text", "")
        check(isinstance(text, str) and bool(text.strip()), str(rid) + ": empty text")
        context = row.get("context")
        check(isinstance(context, dict), str(rid) + ": context must be explicit object")
        key = (seed_validator.normalized(text), fingerprint(context))
        check(key not in seen, f"Duplicate normalized text/context: {rid} / {seen.get(key)}")
        seen[key] = rid
        req = string_list(row.get("required_capabilities"), str(rid) + "/required")
        deny = string_list(row.get("forbidden_capabilities"), str(rid) + "/forbidden")
        check(len(req) == len(set(req)) and len(deny) == len(set(deny)), str(rid) + ": repeated capability")
        check(set(req + deny).issubset(cap_ids), str(rid) + ": unknown capability")
        check(not set(req) & set(deny), str(rid) + ": required/forbidden conflict")
        check(row.get("default_state_for_other_capabilities") == "unmentioned", str(rid) + ": wrong default state")
        string_list(row.get("constraints"), str(rid) + "/constraints")
        string_list(row.get("tags"), str(rid) + "/tags")
        actions = row.get("actions")
        check(isinstance(actions, list), str(rid) + ": actions must be an array")
        for action in actions or []:
            check(isinstance(action, dict) and all(isinstance(action.get(k), str) and action[k].strip()
                                                  for k in ("verb", "resource", "scope")), str(rid) + ": malformed action")
        status, clarify = row.get("intent_status"), row.get("needs_clarification")
        question = row.get("clarification_question")
        check(status in base_contract["intent_status_values"], str(rid) + ": invalid status")
        check(isinstance(clarify, bool), str(rid) + ": clarification flag must be bool")
        if clarify:
            check(status == "ambiguous" and isinstance(question, str) and question.strip(), str(rid) + ": invalid clarification")
        else:
            check(status != "ambiguous" and question is None, str(rid) + ": unexpected clarification")
        if status in ("no_device_capability", "out_of_scope"):
            check(not req, str(rid) + ": unsupported status must not require capabilities")
        check(row.get("source", {}).get("type") == "agent_generated", str(rid) + ": generation provenance missing")
        review = row.get("review", {})
        check(review.get("semantic_ai") == "reviewed", str(rid) + ": missing author semantic review")
        check(review.get("cross_review") in ("pending", "reviewed"), str(rid) + ": invalid cross_review")
        check(review.get("human_review") == "pending", str(rid) + ": no human approval may be inferred")
        check(isinstance(review.get("notes"), list), str(rid) + ": review notes must be an array")
        if require_cross_review:
            check(review.get("cross_review") == "reviewed", str(rid) + ": independent review incomplete")
        review_counts[review.get("cross_review")] += 1
        fid = row.get("family_id")
        check(isinstance(fid, str) and bool(fid), str(rid) + ": invalid family_id")
        check(isinstance(row.get("family_summary"), str) and bool(row["family_summary"].strip()), str(rid) + ": empty family_summary")
        families[fid].append(row)
        split = row.get("split_candidate")
        check(split in SPLITS, str(rid) + ": invalid split_candidate")
        split_counts[split] += 1
        if "context_required" in row["tags"]:
            context_required_counts[split] += 1
            check(bool(context), str(rid) + ": context_required cannot have empty context")
        if context:
            context_present_counts[split] += 1
        statuses[status] += 1
        positive.update(req)
        prohibited.update(deny)
        if split in SPLITS:
            per_split_positive[split].update(req)
            per_split_prohibited[split].update(deny)
    for fid, members in families.items():
        check(len(members) == contract["family_size"], "Family must contain five paraphrases: " + str(fid))
        check(len({r["split_candidate"] for r in members}) == 1, "Family leaks across splits: " + str(fid))
        check(len({r["family_summary"] for r in members}) == 1, "Inconsistent family summary: " + str(fid))
        check(len({fingerprint({k: r[k] for k in GOLD_FIELDS}) for r in members}) == 1,
              "Paraphrases change gold meaning: " + str(fid))
    check(len(rows) == contract["new_rows"], "New row total mismatch")
    check(len(legacy) == contract["legacy_reference_rows"], "Legacy reference total mismatch")
    check(len(families) == contract["new_families"], "Scenario-family total mismatch")
    for split in SPLITS:
        check(split_counts[split] == contract["split_targets"][split], "Split row quota mismatch: " + split)
        for cid in sorted(cap_ids):
            check(per_split_positive[split][cid] > 0, f"Missing positive coverage: {split}/{cid}")
    for cid in sorted(cap_ids):
        check(prohibited[cid] > 0, "Missing explicit whole-capability prohibition: " + cid)
    missing_test_denials = sorted(cid for cid in cap_ids if not per_split_prohibited["test_candidate"][cid])
    if missing_test_denials:
        warnings.append("Test candidates do not yet cover explicit whole-capability prohibitions for: "
                        + ", ".join(missing_test_denials))
    pairs = near_pairs(families) if not errors else []
    audit_path = data_dir / "ai_review_report.json"
    audit = read_json(audit_path) if audit_path.exists() else None
    if require_cross_review:
        check(audit is not None, "Independent audit report missing")
        if audit:
            check(set(audit.get("covered_ids", [])) == ids and len(audit.get("covered_ids", [])) == len(rows),
                  "Independent audit does not cover every new row")
            check(set(audit.get("covered_family_ids", [])) == set(families)
                  and len(audit.get("covered_family_ids", [])) == len(families), "Independent audit family coverage mismatch")
            for author in contract["authors"]:
                name = author + ".jsonl"
                expected = audit.get("shard_sha256", {}).get(name)
                check(expected == sha256(data_dir / "shards" / name), "Stale independent review hash: " + name)
            check(audit.get("review", {}).get("human_review") == "pending", "Audit must not claim human approval")
            check(audit.get("unresolved_errors", -1) == 0, "Independent audit has unresolved errors")
            for field, path in (("catalog_sha256", parent / "capability_catalog.json"),
                                ("legacy_sha256", parent / "representative_instructions.jsonl"),
                                ("contract_sha256", data_dir / "contract.json")):
                check(audit.get(field) == sha256(path), "Stale audit source hash: " + field)
    input_hashes.update({"contract.json": sha256(data_dir / "contract.json"),
                         "../capability_catalog.json": sha256(parent / "capability_catalog.json"),
                         "../representative_instructions.jsonl": sha256(parent / "representative_instructions.jsonl")})
    historical = [(split, phrase) for split, entries in HISTORICAL_FAMILIES.items()
                  for _, phrases in entries for phrase in phrases.split("|")]
    old_by_text = defaultdict(list)
    for split, phrase in historical:
        old_by_text[seed_validator.normalized(phrase)].append({"old_split": split, "text": phrase})
    historical_overlap = [{"id": row["id"], "split_candidate": row["split_candidate"],
                           "matches": old_by_text[seed_validator.normalized(row["text"])],
                           "handling": "Review training history and new label definitions before benchmark freezing"}
                          for row in rows if seed_validator.normalized(row["text"]) in old_by_text]
    input_hashes["../../../dataset.py"] = sha256(Path(__file__).with_name("dataset.py"))
    if audit_path.exists():
        input_hashes["ai_review_report.json"] = sha256(audit_path)
    report = {
        "schema_version": contract["schema_version"], "stage": contract["stage"],
        "structure_passed": not errors, "errors": errors, "warnings": warnings,
        "input_sha256": input_hashes, "author_counts": author_counts,
        "new_instructions": len(rows), "legacy_reference_instructions": len(legacy),
        "total_instructions": len(rows) + len(legacy), "scenario_families": len(families),
        "family_size": contract["family_size"], "split_counts": dict(split_counts),
        "intent_status_counts": dict(statuses), "cross_review_counts": dict(review_counts),
        "context_required_counts_by_split": dict(context_required_counts),
        "context_present_counts_by_split": dict(context_present_counts),
        "positive_counts_by_split": {s: dict(sorted(c.items())) for s, c in per_split_positive.items()},
        "prohibition_counts_by_split": {s: dict(sorted(c.items())) for s, c in per_split_prohibited.items()},
        "cross_split_lexical_candidates": pairs,
        "lexical_screening_note": "Maximum character-bigram Jaccard >=0.45 across paraphrases. A candidate is not proof of leakage; low similarity is not proof of semantic independence.",
        "human_review": "pending", "human_approved_instructions": 0,
        "frozen_test_set": False, "training_performed": False, "model_accuracy_measured": False,
        "semantic_leakage_review": "ai_review_report.json; human review and future merged-corpus audit remain required"
    }
    report["historical_five_label_data_check"] = {
        "rows": len(historical), "normalized_exact_matches": historical_overlap,
        "note": "Lexical screen only. Historical labels are incompatible with the expanded taxonomy; old out_of_scope entries include now-supported tasks. Do not mix old labels into new training or claim semantic independence from historical fine-tuning."
    }
    return contract, catalog, legacy, rows, families, report


def render(data_dir, contract, catalog, legacy, rows, families, report):
    rows = sorted(rows, key=lambda r: r["id"])
    references = [dict(row, split_candidate="reference_only") for row in legacy]
    all_rows = references + rows
    write_rows(data_dir / "expanded_instructions.jsonl", rows)
    write_rows(data_dir / "all_instructions.jsonl", all_rows)
    split_dir = data_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    manifest = {"stage": contract["stage"], "frozen": False, "human_review": "pending",
                "training_performed": False, "legacy_policy": "reference_only; excluded from training and tuning",
                "candidate_use_only": True, "model_input_fields": ["text", "context when explicitly supported"],
                "never_model_inputs": ["required_capabilities", "forbidden_capabilities", "actions", "constraints",
                                       "intent_status", "needs_clarification", "clarification_question",
                                       "family_summary", "family_id", "split_candidate", "tags", "review"],
                "splits": {}}
    for split in (*SPLITS, "reference_only"):
        selected = [row for row in all_rows if row["split_candidate"] == split]
        path = split_dir / (split + ".jsonl")
        write_rows(path, selected)
        manifest["splits"][split] = {"rows": len(selected), "path": str(path.relative_to(data_dir)),
                                     "sha256": sha256(path), "ids": [r["id"] for r in selected],
                                     "family_ids": sorted({r["family_id"] for r in selected if "family_id" in r})}
    write_json(data_dir / "split_manifest.json", manifest)
    family_manifest = []
    for fid, members in sorted(families.items()):
        first = members[0]
        family_manifest.append({"family_id": fid, "summary": first["family_summary"],
                                "split_candidate": first["split_candidate"], "primary_group": first["primary_group"],
                                "ids": [r["id"] for r in members],
                                "gold": {k: first[k] for k in GOLD_FIELDS}})
    write_json(data_dir / "family_manifest.json", family_manifest)
    names = {cap["id"]: cap["name"] for cap in catalog["capabilities"]}
    fields = ["id", "family_id", "family_summary", "split_candidate", "primary_group", "text", "context",
              "intent_status", "required_capabilities", "required_names", "forbidden_capabilities", "forbidden_names",
              "constraints", "needs_clarification", "clarification_question", "cross_review", "human_review",
              "human_decision", "human_notes"]
    tests = [row for row in rows if row["split_candidate"] == "test_candidate"]
    for filename, selected in (("instruction_review.csv", all_rows), ("test_review.csv", tests)):
        with (data_dir / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for item in selected:
                view = {key: item.get(key, "") for key in fields}
                view["context"] = fingerprint(item["context"])
                for key in ("required_capabilities", "forbidden_capabilities", "constraints"):
                    view[key] = " ; ".join(item[key])
                view["required_names"] = " ; ".join(names[c] for c in item["required_capabilities"])
                view["forbidden_names"] = " ; ".join(names[c] for c in item["forbidden_capabilities"])
                view.update(cross_review=item["review"]["cross_review"], human_review="pending", human_decision="", human_notes="")
                writer.writerow(view)
    family_fields = ["family_id", "split_candidate", "primary_group", "summary", "context", "required_names",
                     "forbidden_names", "constraints", "clarification_question", "variant_1", "variant_2", "variant_3",
                     "variant_4", "variant_5", "human_review", "human_decision", "human_notes"]
    with (data_dir / "family_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=family_fields, lineterminator="\n")
        writer.writeheader()
        for fid, members in sorted(families.items()):
            first = members[0]
            view = {"family_id": fid, "split_candidate": first["split_candidate"], "primary_group": first["primary_group"],
                    "summary": first["family_summary"], "context": fingerprint(first["context"]),
                    "required_names": " ; ".join(names[c] for c in first["required_capabilities"]),
                    "forbidden_names": " ; ".join(names[c] for c in first["forbidden_capabilities"]),
                    "constraints": " ; ".join(first["constraints"]), "clarification_question": first["clarification_question"],
                    "human_review": "pending", "human_decision": "", "human_notes": ""}
            view.update({"variant_" + str(i): member["text"] for i, member in enumerate(members, 1)})
            writer.writerow(view)
    write_json(data_dir / "test_set_overview.json", {
        "split": "test_candidate", "rows": len(tests), "scenario_families": len({r["family_id"] for r in tests}),
        "paraphrases_per_family": contract["family_size"], "frozen": False, "human_review": "pending",
        "intent_status_counts": dict(Counter(r["intent_status"] for r in tests)),
        "primary_group_counts": dict(Counter(r["primary_group"] for r in tests)),
        "context_required_rows": sum("context_required" in r["tags"] for r in tests),
        "positive_capability_counts": report["positive_counts_by_split"]["test_candidate"],
        "prohibition_capability_counts": report["prohibition_counts_by_split"]["test_candidate"],
        "capabilities_without_explicit_prohibition_test": sorted(
            cap["id"] for cap in catalog["capabilities"]
            if not report["prohibition_counts_by_split"]["test_candidate"].get(cap["id"], 0)),
        "candidate_scope": "Synthetic internal review candidates, not a blind real-user benchmark or measured model accuracy",
        "metrics_to_measure_later": ["required capability precision/recall/F1", "required set exact match",
                                     "forbidden capability violation rate", "intent status accuracy",
                                     "clarification precision/recall", "per-family and per-capability breakdown"],
        "not_automatically_scored": ["free-text scope/sequence constraints without a defined evaluator",
                                    "platform runtime authorization", "NPU numerical parity and latency"]
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data/permission_intent_v2/expansion")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--require-cross-review", action="store_true")
    args = parser.parse_args()
    values = inspect(args.data_dir, args.require_cross_review)
    report = values[-1]
    write_json(args.data_dir / "validation_report.json", report)
    if report["errors"]:
        raise SystemExit("\n".join(report["errors"]))
    if args.render:
        render(args.data_dir, *values[:-1], report)
    print(f"PASS: {report['total_instructions']} rows, {report['scenario_families']} new families; "
          f"new split counts={report['split_counts']}; human-approved=0; not frozen")
    if report["warnings"]:
        print("Documented coverage warnings:", len(report["warnings"]))


if __name__ == "__main__":
    main()
