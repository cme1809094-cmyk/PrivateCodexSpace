"""Generate training-only compositions from a finite, reviewable action lexicon.

This is controlled augmentation, not independently authored user evidence.
Validation and reserved tests are written separately and never generated here.
"""
from collections import Counter
import json
from pathlib import Path

from transformers import BertTokenizerFast
from train_intent_contrast import ROOT, read_json, read_rows, sha256, write_json
from validate_permission_intent_v2 import normalized


DATA = ROOT / "ml/data/permission_intent_v4"
AMBIGUOUS = {
    "source": [
        ("要交一张门牌照片，但我没定用现有图片还是现场相机拍。", "门牌图片用已有照片还是新拍的，我还没选来源。"),
        ("电话从我手写的便签取还是查手机通讯录，我还没决定。", "联系人号码用便签上的还是查询通讯录，来源还没定。"),
        ("日期用我提供的表格还是手机日历里的记录，我还没选。", "我没决定从表格还是设备日历找日期，你先别猜来源。"),
        ("是读取设备计步器还是看应用已存的步数，我没有选定。", "步数用应用旧记录还是现场读手机传感器，来源未定。"),
        ("路程起点用手填坐标还是取设备位置，我还没定。", "起点要手工提供还是现场定位，我没决定来源。"),
        ("文字是我手动输入还是应用读剪贴板，我还没选择。", "输入框要手填还是应用取剪贴板，来源还没定下来。"),
        ("提醒日期用我手填的日期还是先读手机日历取得，我还没决定。", "提醒日期来自我的手工输入还是手机日历，我尚未选定来源。"),
        ("声音用应用里那段音频还是重新录现场，我还没选。", "我没定要现成音频还是新录声音，先别自行选择。")],
    "target": [
        ("手机通讯录有两个同名不同号码的人，我没选哪位，却要查他的邮箱。", "同名两位联系人都有邮箱，我还没选查询哪一个。"),
        ("系统照片面板里有两张很像的凭证，我没选哪张，让你用那张。", "两张凭证照片都在候选里，我没有指定要用哪张。"),
        ("手机日历有两个重名事件，我没指出哪条就让你删除那个。", "两条日历事件名字一样，删除对象还没选定。"),
        ("配对列表有两个同名蓝牙音箱，我没有选连接哪一个。", "两个已配对音箱同名，这次要连谁我还没决定。"),
        ("有两个已存无线网络可选，我没说该连接哪个。", "手机里两份网络配置都能用，连接目标我还没选。"),
        ("两枚空白NFC标签都放在面前，我没指明该写入哪枚。", "要写的NFC标签有两个候选，目标我还没有指定。"),
        ("有两项已排好的本地提醒，我只说修改那个但没选中任何一项。", "待修改提醒有两条，具体哪一条我还未指定。"),
        ("两个文件都已在选择窗口里，我没选哪个却说打开那份。", "候选文档有两份，打开对象还没有确定。")],
    "scope": [
        ("我要查设备日历，但查今天还是整个下月，时间范围尚未定。", "日历读取要覆盖今天还是下个月，我还没有决定范围。"),
        ("要从照片库找照片，但仅看上周还是查看全年，范围还没选。", "图库检索到底上周还是全年，我尚未确定时间范围。"),
        ("要读手机步数，但从现在起计数还是取当前累计，我没选。", "步数要现在的累计还是从此刻重新观察，我没定范围。"),
        ("要改联系人备注，但改选中一人还是所有匹配的人，范围没定。", "备注修改覆盖一个联系人还是全部匹配记录，我还没决定。"),
        ("我要后台记位置，但只记这趟路还是持续到明天，时长未定。", "后台定位记录到本次行程结束还是明天，范围我还没定。"),
        ("要录现场声音，但只录开场还是整场会议，时段还没定。", "麦克风录音覆盖开场还是完整会议，我没有选定时段。"),
        ("要删除日历事件，但删这一次还是连重复系列一起，我没决定。", "日历删除针对本次事件还是整个重复系列，范围未选。"),
        ("要把文字复制进剪贴板，但复制标题还是正文，内容范围没选。", "剪贴板里该放标题还是正文，我还没确定内容范围。")]
}
OUTSIDE = [
    ("读取本机短信收件箱的最近十条短信。", "把手机短信列表里的最新十条消息列出来。"),
    ("用手机号码直接给13900139022发送短信。", "请通过手机短信把一句问候发到13900139022。"),
    ("拨打电话13900139023。", "直接发起到13900139023的手机通话。"),
    ("读出本机最近一次来电号码。", "查手机通话记录里最后一通来电。"),
    ("把手机当前整个屏幕截成一张图。", "截取当前屏幕内容生成截图。"),
    ("录制手机屏幕上现在操作的全过程。", "把屏幕操作录成视频，不是拍现场画面。"),
    ("读取邮箱收件箱里最新的一封邮件正文。", "去我的邮箱找到最新邮件并取正文。"),
    ("把这句话自动发进另一个聊天应用的指定会话。", "替我在别的聊天应用指定窗口发送这段文字。"),
    ("导出手机系统保存的原始指纹模板。", "把系统里的原始指纹特征取出来给我。"),
    ("提取手机认证系统保存的人脸原始模板。", "将系统保存的原始人脸认证特征导出。"),
    ("把银行支付卡的安全凭据复制进另一张卡。", "复制支付卡内部的安全卡凭据用于另一张卡。"),
    ("从健康账户调出完整的历史病历。", "读取健康账号中的全部既往医疗档案。"),
    ("更改SIM卡的实名登记资料。", "把当前SIM卡的实名注册人资料改掉。"),
    ("查看SIM卡实名登记的完整证件信息。", "读出运营商SIM卡登记中的证件详情。"),
    ("把另一个应用设为系统默认短信程序。", "更换手机的默认短信应用。"),
    ("在其他应用上方持续显示一个悬浮窗。", "给别的应用界面上层添加一直悬浮的窗口。"),
    ("覆盖改写手机图库里已存在照片的原文件。", "直接修改公共图库已有照片的原始文件。"),
    ("从手机照片库删除一张已有照片。", "把公共相册里的那张旧照片永久删除。"),
    ("查询我的银行卡当前余额。", "连接银行账户并读取现在可用余额。"),
    ("自动读取另一个聊天应用的聊天历史。", "取出别的聊天软件保存的对话记录。"),
    ("调出手机系统里所有应用的使用时长。", "读取各个手机应用的详细使用时间统计。"),
    ("修改手机系统的默认浏览器。", "把另一个浏览器设成系统默认。"),
    ("从应用商店自动安装一款新软件。", "替我下载安装新的手机应用。"),
    ("把系统的移动数据开关直接关掉。", "关闭手机蜂窝移动网络的数据开关。")]


def build():
    if (DATA / "freeze_manifest.json").exists():
        raise ValueError("Refusing to overwrite frozen training data")
    lex = read_json(DATA / "lexicon.json")["capabilities"]
    tokenizer = BertTokenizerFast.from_pretrained('/workspace/work/tinybert/downloads', local_files_only=True)
    previous = read_rows(ROOT / "ml/data/permission_intent_v2/expansion/all_instructions.jsonl")
    previous += read_rows(ROOT / "ml/data/permission_intent_v3/reserved_test.jsonl")
    for p in (ROOT / "ml/data/permission_intent_v3/shards").glob("*.jsonl"):
        previous += read_rows(p)
    seen = {normalized(r["text"]) for r in previous}
    rows, skipped = [], Counter()

    def add_group(gid, cases, kind, composition):
        members = []
        for role, texts, required, forbidden, status, subtype in cases:
            for variant, text in enumerate(texts, 1):
                members.append({"id": gid + "_" + role + str(variant), "family_id": gid + "_" + role,
                                "scenario_group_id": gid, "contrast_role": role, "split": "train", "text": text,
                                "required_capabilities": required, "forbidden_capabilities": forbidden,
                                "intent_status": status, "ambiguity_type": subtype, "context": {},
                                "source": {"type": "controlled_composition", "kind": kind, "composition": composition},
                                "review": {"semantic_ai": "template_generated", "cross_review": "pending", "human_review": "pending"}})
        lengths = tokenizer([r["text"] for r in members], truncation=False, return_length=True)["length"]
        keys = [normalized(r["text"]) for r in members]
        if max(lengths) > 64:
            skipped["over_64_group"] += 1
            return
        if len(set(keys)) != len(keys) or any(k in seen for k in keys):
            skipped["exact_duplicate_group"] += 1
            return
        seen.update(keys)
        rows.extend(members)

    caps = list(lex)
    for i, cap in enumerate(caps):
        p, q = lex[cap]["positive"]
        n = lex[cap]["negative"]
        add_group(f"v4_train_direct_{i:02}", [("a", [f"请{p}。", f"我现在需要{q}。"], [cap], [], "in_scope", None)], "direct", {"capability": cap})
        add_group(f"v4_train_local_{i:02}", [("a", [f"只高亮应用中已有的文字，禁止{n}。", f"把已在应用页面的文字高亮，绝对不能{n}。"], [], [cap], "no_device_capability", None)], "local_and_ban", {"capability": cap})
        for offset in (1, 5, 11, 17):
            other = caps[(i + offset) % len(caps)]
            op, oq = lex[other]["positive"]
            on = lex[other]["negative"]
            gid = f"v4_train_swap_{i:02}_{offset:02}"
            add_group(gid, [("a", [f"请{p}，同时禁止{on}。", f"我需要{q}，{on}这件事不要做。"], [cap], [other], "in_scope", None),
                            ("b", [f"请{op}，同时禁止{n}。", f"我需要{oq}，{n}这件事不要做。"], [other], [cap], "in_scope", None)], "polarity_swap", {"left": cap, "right": other})
            gid = f"v4_train_conflict_{i:02}_{offset:02}"
            conflict = {
                1: [f"我禁止{n}，却又要求{p}。", f"不允许{n}，但同时必须{q}。"],
                5: [f"必须{p}，但{n}一律不准。", f"要求{q}，同时{n}仍是禁止的。"],
                11: [f"{n}被禁止，另一条要求却是{p}。", f"已经明确禁止{n}，又要求{q}。"],
                17: [f"现在要{p}，不过我仍禁止{n}。", f"这次要{q}，但仍不准{n}。"]}[offset]
            consistent = {
                1: [f"我禁止{n}，却又要求{op}。", f"不允许{n}，但同时必须{oq}。"],
                5: [f"必须{op}，但{n}一律不准。", f"要求{oq}，同时{n}仍是禁止的。"],
                11: [f"{n}被禁止，另一条要求却是{op}。", f"已经明确禁止{n}，又要求{oq}。"],
                17: [f"现在要{op}，不过我仍禁止{n}。", f"这次要{oq}，但仍不准{n}。"]}[offset]
            # Equal discourse markers in contradictory and consistent requests
            # prevent the word "but" from being a shortcut for clarification.
            add_group(gid, [("a", conflict, [], [cap], "ambiguous", "conflict"),
                            ("b", consistent, [other], [cap], "in_scope", None)], "conflict_vs_other_action", {"left": cap, "right": other})
    for subtype, pairs in AMBIGUOUS.items():
        for i, pair in enumerate(pairs):
            add_group(f"v4_train_unclear_{subtype}_{i:02}", [("a", list(pair), [], [], "ambiguous", subtype)], "authored_ambiguity", {"subtype": subtype})
    for i, pair in enumerate(OUTSIDE):
        add_group(f"v4_train_outside_{i:02}", [("a", list(pair), [], [], "out_of_scope", None)], "authored_outside", {})
    DATA.mkdir(exist_ok=True)
    destination = DATA / "train.jsonl"
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    report = {"rows": len(rows), "groups": len({r["scenario_group_id"] for r in rows}),
              "status_counts": dict(Counter(r["intent_status"] for r in rows)), "skipped_groups": dict(skipped),
              "sha256": sha256(destination), "lexicon_sha256": sha256(DATA / "lexicon.json"),
              "max_tokens_including_special": max(tokenizer([r["text"] for r in rows], return_length=True)["length"]),
              "review_method": "Finite atom/template review and programmatic validation; not a per-row human audit", "human_review": "pending"}
    write_json(DATA / "train_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    build()
