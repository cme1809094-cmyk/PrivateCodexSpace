"""Validate and render the capability catalog and permission-intent seed corpus."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse

PLATFORMS = ("harmonyos", "android")
AVAILABILITY = {"public", "conditional", "restricted", "needs_verification"}
VERIFICATION = {"official_docs_verified", "sdk_verified", "openharmony_reference_only", "needs_verification"}
ANDROID_LIBRARY_PERMISSIONS = {"com.google.android.gms.permission.ACTIVITY_RECOGNITION"}


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower()
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def inspect(data_dir):
    contract = json.loads((data_dir / "contract.json").read_text())
    catalog = json.loads((data_dir / "capability_catalog.json").read_text())
    rows = [json.loads(line) for line in (data_dir / "representative_instructions.jsonl").read_text().splitlines() if line.strip()]
    errors, warnings = [], []
    def check(condition, message):
        if not condition:
            errors.append(message)
    def string_list(value, where):
        check(isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value),
              where + " must be a list of nonempty strings")
        return value if isinstance(value, list) else []
    expected_groups = {**contract["new_groups"], **contract["retained_groups"]}
    expected_ids = {item for group in expected_groups.values() for item in group}
    groups, caps, sources = catalog.get("groups", []), catalog.get("capabilities", []), catalog.get("sources", [])
    check(catalog.get("schema_version") == contract["schema_version"], "Catalog schema_version mismatch")
    group_ids, cap_ids, source_ids = [g.get("id") for g in groups], [c.get("id") for c in caps], [s.get("id") for s in sources]
    check(len(group_ids) == len(set(group_ids)) and set(group_ids) == set(expected_groups), "Group IDs must match contract")
    check(len(cap_ids) == len(set(cap_ids)) and set(cap_ids) == expected_ids, "Capability IDs must match contract")
    check(len(source_ids) == len(set(source_ids)), "Duplicate source IDs")
    for source in sources:
        sid, url = source.get("id"), source.get("url", "")
        check(bool(sid) and isinstance(url, str) and bool(url.strip()), "Invalid source " + str(sid))
        check(urlparse(url).scheme in ("https", "http", "sdk", "file") or url.startswith("/workspace/"),
              "Source needs official URL or SDK locator: " + str(sid))
        check(bool(source.get("checked_on")), "Source needs checked_on: " + str(sid))
    rule_ids, unresolved = [], []
    availability_counts, verification_counts = Counter(), Counter()
    for cap in caps:
        cid, group = cap.get("id"), cap.get("group")
        check(group in expected_groups and cid in expected_groups.get(group, []), "Capability/group mismatch: " + str(cid))
        check(bool(cap.get("name")) and bool(cap.get("definition")), "Missing name/definition: " + str(cid))
        for platform in PLATFORMS:
            rules = cap.get("platforms", {}).get(platform, {}).get("rules", [])
            check(isinstance(rules, list) and bool(rules), f"Missing rules: {cid}/{platform}")
            for rule in rules:
                rid = rule.get("rule_id")
                rule_ids.append(rid)
                check(isinstance(rid, str) and bool(rid), f"Missing rule_id: {cid}/{platform}")
                check(bool(rule.get("route")), "Missing route: " + str(rid))
                availability, verification = rule.get("availability"), rule.get("verification_status")
                check(availability in AVAILABILITY, "Invalid availability: " + str(rid))
                check(verification in VERIFICATION, "Invalid verification status: " + str(rid))
                availability_counts[platform + ":" + str(availability)] += 1
                verification_counts[platform + ":" + str(verification)] += 1
                minimum, maximum = rule.get("api_min"), rule.get("api_max")
                check(minimum is None or isinstance(minimum, int) and minimum > 0, "Invalid api_min: " + str(rid))
                check(maximum is None or isinstance(maximum, int) and maximum > 0, "Invalid api_max: " + str(rid))
                if isinstance(minimum, int) and isinstance(maximum, int):
                    check(maximum >= minimum, "Reversed API range: " + str(rid))
                manifest = string_list(rule.get("manifest_permissions"), str(rid) + "/manifest_permissions")
                runtime = string_list(rule.get("runtime_permissions"), str(rid) + "/runtime_permissions")
                for field in ("special_access", "scoped_authorization", "conditions", "evidence_source_ids"):
                    string_list(rule.get(field), str(rid) + "/" + field)
                check(set(runtime).issubset(manifest), "Runtime permission lacks declaration: " + str(rid))
                candidates = string_list(rule.get("candidate_permissions", []), str(rid) + "/candidate_permissions")
                for permission in manifest + runtime + candidates:
                    prefix = "ohos.permission." if platform == "harmonyos" else "android.permission."
                    check(permission.startswith(prefix) or platform == "android" and permission in ANDROID_LIBRARY_PERMISSIONS,
                          "Wrong platform permission: " + str(rid) + "/" + permission)
                if rule.get("executable") is False:
                    check(not manifest and not runtime, "Blocked candidate must not expose an executable request list: " + str(rid))
                refs = rule.get("evidence_source_ids", [])
                check(bool(refs) and set(refs).issubset(source_ids), "Missing/unknown evidence: " + str(rid))
                if availability == "public":
                    check(verification != "needs_verification", "Unverified mapping cannot claim public availability: " + str(rid))
                if cid == "media.photo.pick_one":
                    broad = ("READ_IMAGEVIDEO", "READ_MEDIA_IMAGES", "READ_EXTERNAL_STORAGE", "MANAGE_EXTERNAL_STORAGE")
                    check(not any(any(part in p for part in broad) for p in manifest + runtime),
                          "Single-photo picker must not request broad media/storage access: " + str(rid))
                    check(bool(rule.get("scoped_authorization")), "Picker must retain scoped authorization: " + str(rid))
                if platform == "harmonyos" and availability == "public":
                    check("ohos.permission.NOTIFICATION_CONTROLLER" not in manifest + runtime,
                          "Ordinary publication cannot request system notification control: " + str(rid))
                if availability in ("restricted", "needs_verification") or verification in ("needs_verification", "openharmony_reference_only"):
                    unresolved.append({"capability_id": cid, "platform": platform, "rule_id": rid,
                                       "availability": availability, "verification_status": verification})
    check(len(rule_ids) == len(set(rule_ids)), "Duplicate rule IDs")
    check(catalog.get("review", {}).get("human_review") == "pending", "Catalog must not claim human approval")
    row_ids, texts = [], []
    group_counts, tags, statuses = Counter(), Counter(), Counter()
    required_counts, forbidden_counts = Counter(), Counter()
    cross_reviews, human_reviews = Counter(), Counter()
    allowed_groups = set(expected_groups) | set(contract["corpus_extra_groups"])
    for row in rows:
        rid = row.get("id")
        row_ids.append(rid)
        for key in contract["corpus_required_fields"]:
            check(key in row, f"{rid}: missing {key}")
        check(row.get("schema_version") == contract["schema_version"], f"{rid}: schema mismatch")
        text = row.get("text", "")
        check(isinstance(text, str) and bool(text.strip()), f"{rid}: empty instruction")
        texts.append(normalized(text))
        group = row.get("primary_group")
        check(group in allowed_groups, f"{rid}: unknown primary group")
        group_counts[group] += 1
        req = string_list(row.get("required_capabilities"), str(rid) + "/required")
        denied = string_list(row.get("forbidden_capabilities"), str(rid) + "/forbidden")
        check(len(req) == len(set(req)) and len(denied) == len(set(denied)), f"{rid}: duplicate capability")
        check(set(req + denied).issubset(expected_ids), f"{rid}: unknown capability")
        check(not set(req) & set(denied), f"{rid}: required/forbidden conflict")
        required_counts.update(req)
        forbidden_counts.update(denied)
        check(row.get("default_state_for_other_capabilities") == "unmentioned", f"{rid}: unspecified must remain unmentioned")
        check(isinstance(row.get("context"), dict), f"{rid}: context must be an object")
        check(isinstance(row.get("actions"), list), f"{rid}: actions must be an array")
        for action in row.get("actions", []):
            check(isinstance(action, dict) and all(isinstance(action.get(k), str) and action[k].strip()
                                                for k in ("verb", "resource", "scope")), f"{rid}: invalid action")
        string_list(row.get("constraints"), str(rid) + "/constraints")
        tags.update(string_list(row.get("tags"), str(rid) + "/tags"))
        status = row.get("intent_status")
        statuses[status] += 1
        check(status in contract["intent_status_values"], f"{rid}: invalid intent_status")
        clarify, question = row.get("needs_clarification"), row.get("clarification_question")
        check(isinstance(clarify, bool), f"{rid}: needs_clarification must be boolean")
        if clarify:
            check(status == "ambiguous" and isinstance(question, str) and bool(question.strip()), f"{rid}: clarification needs ambiguous status/question")
        else:
            check(status != "ambiguous" and question is None, f"{rid}: unexpected blocking question")
        if status in ("out_of_scope", "no_device_capability"):
            check(not req, f"{rid}: status cannot imply a required catalog capability")
        check(row.get("source", {}).get("type") == "agent_generated", f"{rid}: source must disclose generation")
        review = row.get("review", {})
        check(review.get("semantic_ai") == "reviewed", f"{rid}: missing semantic AI review")
        check(review.get("cross_review") in ("pending", "reviewed"), f"{rid}: invalid cross review")
        check(review.get("human_review") == "pending", f"{rid}: human review must remain pending")
        check(isinstance(review.get("notes"), list), f"{rid}: review notes must be an array")
        cross_reviews[review.get("cross_review")] += 1
        human_reviews[review.get("human_review")] += 1
        if re.search(r"(?:ohos|android)\.permission\.", text):
            warnings.append(f"{rid}: permission string in user text; check realism")
    check(all(isinstance(x, str) and x for x in row_ids) and len(row_ids) == len(set(row_ids)), "Invalid/duplicate row IDs")
    check(len(texts) == len(set(texts)), "Duplicate normalized instructions")
    check(len(rows) == contract["corpus_target"]["total"], "Instruction total mismatch")
    for group in contract["new_groups"]:
        check(group_counts[group] == contract["corpus_target"]["each_new_group"], "New group quota mismatch: " + group)
    for cid in expected_ids:
        check(required_counts[cid] > 0, "Missing positive capability example: " + cid)
        if not forbidden_counts[cid]:
            warnings.append("No explicit prohibition example: " + cid)
    report = {
        "schema_version": contract["schema_version"], "stage": contract["stage"], "structure_passed": not errors,
        "input_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in
                         ("contract.json", "capability_catalog.json", "representative_instructions.jsonl")},
        "errors": errors, "warnings": warnings, "new_groups": len(contract["new_groups"]),
        "retained_groups": len(contract["retained_groups"]), "atomic_capabilities": len(caps),
        "mapping_rules": len(rule_ids), "evidence_sources": len(sources), "instructions": len(rows),
        "group_counts": dict(sorted(group_counts.items())), "intent_status_counts": dict(sorted(statuses.items())),
        "tag_counts": dict(sorted(tags.items())), "positive_capability_counts": dict(sorted(required_counts.items())),
        "prohibition_capability_counts": dict(sorted(forbidden_counts.items())),
        "mapping_availability": dict(sorted(availability_counts.items())),
        "mapping_verification": dict(sorted(verification_counts.items())), "rules_requiring_attention": unresolved,
        "cross_review_counts": dict(cross_reviews), "human_review_counts": dict(human_reviews),
        "human_approved_instructions": 0, "frozen_evaluation_set": False, "model_training_performed": False,
        "scope": "Structure/reference/annotation checks; not human approval, actual permission execution or model accuracy"
    }
    return contract, catalog, rows, report


def render(data_dir, contract, catalog, rows, report):
    by_id = {cap["id"]: cap for cap in catalog["capabilities"]}
    def short_rule(rule):
        lo, hi = rule["api_min"], rule["api_max"]
        parts = [f"API {lo or '?'}" + ("+" if hi is None else f"-{hi}"),
                 rule["route"], rule["availability"], rule["verification_status"]]
        for key in ("manifest_permissions", "runtime_permissions", "special_access", "scoped_authorization", "candidate_permissions", "conditions"):
            if rule.get(key):
                parts.append(key + "=" + "; ".join(rule[key]))
        return " | ".join(parts)
    fields = ["capability_id", "name", "group", "platform", "rule_id", "api_min", "api_max", "route", "availability",
              "verification_status", "manifest_permissions", "runtime_permissions", "special_access", "scoped_authorization",
              "candidate_permissions", "conditions", "evidence_source_ids", "human_review", "human_decision", "human_notes"]
    with (data_dir / "capability_rules_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cap in catalog["capabilities"]:
            for platform in PLATFORMS:
                for rule in cap["platforms"][platform]["rules"]:
                    row = {key: rule.get(key, "") for key in fields}
                    row.update(capability_id=cap["id"], name=cap["name"], group=cap["group"], platform=platform,
                               human_review="pending", human_decision="", human_notes="")
                    writer.writerow({k: " ; ".join(v) if isinstance(v, list) else v for k, v in row.items()})
    fields = ["id", "primary_group", "text", "context", "intent_status", "required_capabilities", "forbidden_capabilities",
              "constraints", "needs_clarification", "clarification_question", "harmonyos_rule_summary", "android_rule_summary",
              "semantic_ai", "cross_review", "human_review", "human_decision", "human_notes"]
    with (data_dir / "instruction_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in rows:
            row = {key: item.get(key, "") for key in fields}
            row["context"] = json.dumps(item["context"], ensure_ascii=False, sort_keys=True)
            for key in ("required_capabilities", "forbidden_capabilities", "constraints"):
                row[key] = " ; ".join(item[key])
            for platform in PLATFORMS:
                row[platform + "_rule_summary"] = "\n".join(
                    cid + ": " + short_rule(rule) for cid in item["required_capabilities"]
                    for rule in by_id[cid]["platforms"][platform]["rules"])
            row.update(item["review"])
            row["human_decision"], row["human_notes"] = "", ""
            writer.writerow({key: row.get(key, "") for key in fields})
    lines = [
        "# 首批能力字典与代表性指令", "",
        f"本版覆盖{report['new_groups']}个新增能力组、{report['retained_groups']}个保留组，共"
        f"{report['atomic_capabilities']}项能力、{report['mapping_rules']}条平台映射和{report['instructions']}条种子指令。",
        "", "条目由子agent生成，AI交叉复核状态见独立报告；人工审核仍为pending。结构校验通过不等于人工审核或平台实际可调用。",
        "", "## 文件", "",
        "- [capability_catalog.json](capability_catalog.json)：能力定义、平台版本条件、授权路径、限制与证据来源。",
        "- [representative_instructions.jsonl](representative_instructions.jsonl)：指令、语义三态、范围条件与追问；以能力ID引用字典。",
        "- [capability_rules_review.csv](capability_rules_review.csv)：逐条平台规则的人审表。",
        "- [instruction_review.csv](instruction_review.csv)：逐条指令、平台映射条件的人审表。",
        "- [validation_report.json](validation_report.json)：数量、去重、引用、标注一致性及待核实映射统计。",
        "- [ai_review_report.json](ai_review_report.json)：独立子agent的语义/平台交叉复核记录。",
        "", "## 语义和平台映射", "",
        "required表示任务所需能力，不表示系统已授权。forbidden只记录用户明确禁止的完整能力；"
        "未提及保持unmentioned。模态、目标、范围、持续时间的部分限制放在constraints，避免扩大成整项能力禁止。",
        "", "模型学习能力与语义，平台权限统一按字典的API、route、开放级别和交互条件解释。"
        "candidate_permissions不属于可执行请求方案。restricted/needs_verification及仅OpenHarmony参考的映射仍须核实，不能直接作为普通HarmonyOS应用的申请依据。",
        "", "在线服务INTERNET属于实现依赖，不等于需要设备定位。系统选择器URI授权仍需要用户选择；"
        "取消、拒绝或认证失败不扩大其他访问范围。",
        "", "## 审核", "",
        "先审动作、范围、否定、歧义和上下文，再审API及实现route。CSV中的human_decision/human_notes留给人工填写；"
        "human_review保持pending，不能因一次格式确认批量标成人工审核通过。",
        "", "CSV是生成视图。填写前请另存审核副本；再次运行--render会重新生成并覆盖这两份CSV，"
        "不会自动把表格意见写回JSON/JSONL。修改canonical数据后须重新交叉复核。",
        "", "本批用于人工确认和后续扩写，尚未分配训练/验证/测试集合，也未重新训练Demo。"
        "后续应按语义场景族拆分，避免近似改写泄漏到独立验收集。",
        "", "## 能力索引", "", "| 能力ID | 名称 | 分组 |", "| --- | --- | --- |"]
    for cap in catalog["capabilities"]:
        lines.append("| " + " | ".join(str(cap[k]).replace("|", "\\|") for k in ("id", "name", "group")) + " |")
    lines += ["", "复现检查：", "", "    python3 ml/validate_permission_intent_v2.py --render", ""]
    (data_dir / "README.md").write_text("\n".join(lines))


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "ml/data/permission_intent_v2")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    contract, catalog, rows, report = inspect(args.data_dir)
    (args.data_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["errors"]:
        raise SystemExit("\n".join(report["errors"]))
    if args.render:
        render(args.data_dir, contract, catalog, rows, report)
    print(f"PASS: {report['atomic_capabilities']} capabilities, {report['mapping_rules']} rules, "
          f"{report['instructions']} seed instructions; human-approved=0")
    if report["warnings"]:
        print("Warnings requiring review:", len(report["warnings"]))


if __name__ == "__main__":
    main()
