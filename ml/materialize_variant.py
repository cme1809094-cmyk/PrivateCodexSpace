"""Create an isolated HarmonyOS project with a variant's models and metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    variant, destination = args.variant.resolve(), args.destination.resolve()
    if destination == root or root in destination.parents or variant in destination.parents:
        raise ValueError("Choose a destination outside the source checkout")
    if destination.exists():
        raise ValueError("Destination already exists; choose a new project directory")
    manifest = json.loads((variant / "variant.json").read_text())
    for artifact in manifest["artifacts"]:
        source = variant / artifact["source"]
        if not source.is_file() or source.stat().st_size != artifact["bytes"] or digest(source) != artifact["sha256"]:
            raise ValueError(f"Missing or invalid {source}; run git lfs pull")
        target = destination / artifact["target"]
        if destination not in target.resolve().parents:
            raise ValueError("Artifact target escapes destination")
    shutil.copytree(root, destination, ignore=shutil.ignore_patterns(
        ".git", "variants", "evidence", "*.om", "*.omc", "*.ms", "*.onnx", "__pycache__",
        "model-build", "build", "oh_modules", ".hvigor", ".idea", ".cxx", "local.properties",
        "model-switch-verification.json"))
    for artifact in manifest["artifacts"]:
        target = destination / artifact["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(variant / artifact["source"], target)
    evidence = destination / "ml/native_v2/evidence" / manifest["id"]
    shutil.copytree(variant / "evidence", evidence)
    report_path = destination / "ml/kirin9030-native-v2-verification.json"
    report = json.loads(report_path.read_text())
    for length, bucket in report["buckets"].items():
        bucket["model"] = f"entry/src/main/resources/rawfile/permission_{length}.om"
        bucket["backend_log"] = f"ml/native_v2/evidence/{manifest['id']}/l{length}/backend.log"
        bucket["model_dump"] = f"ml/native_v2/evidence/{manifest['id']}/l{length}/mobile.json"
    report["mindspore_lite_cpu"]["evidence"] = f"ml/native_v2/evidence/{manifest['id']}/ms-verify.log"
    report["training_metrics"] = "ml/metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    model_manifest = {"source_variant": manifest["id"], "device_tested": False, "artifacts": [
        {"path": a["target"], "bytes": a["bytes"], "sha256": a["sha256"]}
        for a in manifest["artifacts"] if Path(a["source"]).suffix in {".onnx", ".om", ".ms"}]}
    (destination / "ml/artifact-manifest.json").write_text(json.dumps(model_manifest, indent=2) + "\n")
    profile = destination / "AppScope/app.json5"
    profile.write_text(re.sub(r'"bundleName"\s*:\s*"[^"]+"',
                             '"bundleName": "' + manifest["bundle_name"] + '"', profile.read_text()))
    for path in (destination / "AppScope/resources/base/element/string.json",
                 destination / "entry/src/main/resources/base/element/string.json"):
        strings = json.loads(path.read_text())
        for item in strings.get("string", []):
            if item["name"] in {"app_name", "EntryAbility_label", "EntryAbility_desc"}:
                item["value"] = manifest["display_name"] + " 权限识别"
        path.write_text(json.dumps(strings, ensure_ascii=False, indent=2) + "\n")
    raw = destination / "entry/src/main/resources/rawfile"
    # The same runtime is driven by a single-profile catalog, without rewriting
    # ArkTS source strings or retaining unused metadata from the combined app.
    shutil.rmtree(raw / "models", ignore_errors=True)
    catalog = json.loads((root / "entry/src/main/resources/rawfile/model_catalog.json").read_text())
    selected = next(p for p in catalog["profiles"] if p["id"] == manifest["id"])
    selected["assetPrefix"] = ""
    for asset in selected["metadata"]:
        source = variant / asset["file"]
        if source.is_file():
            shutil.copy2(source, raw / asset["file"])
    (raw / "model_catalog.json").write_text(json.dumps(
        {"defaultModel": manifest["id"], "profiles": [selected]}, ensure_ascii=False, indent=2) + "\n")
    (destination / "README.md").write_text(
        "# " + manifest["display_name"] + " 权限识别 Demo\n\n"
        "本工程由 ml/materialize_variant.py 生成，包含配套的三档原生 V2 NPU 模型、CPU 回退模型、词表、阈值和自检向量。1.1.0 支持自动／16／32／64 档位，点击「验收当前模型三档 NPU」后复制验收 JSON。验收逐档检查 27/27、33/33、33/33 参考权限一致性及最大绝对 logit 差 ≤0.1（暂定工程阈值），并记录预热 5 次、测量 50 次的 P50/P95。验收禁止 CPU 回退。\n\n"
        "包名：`" + manifest["bundle_name"] + "`。请在 DevEco Studio 中构建、配置调试签名并安装到 Mate 80 Pro Max。\n\n"
        "OMC 需要 CANN 的 HIAI_F 后端。应用会先动态加载 CANN Foundation 并记录版本，再枚举后端和检查模型兼容性。通用 NNRT 加速器不等于支持 OMC；不可用时会明确记录原因并回退到同模型 CPU。更新应用时需 Clean/Rebuild native 和 ArkTS。\n\n"
        "若回退提示包含 failed at / exception_type，请保留完整内容。1.0.6 起提示包含最后调用、返回状态、异常类型和 CANN 版本。1.0.7 的 HiLog Tag 为 PermissionNPU，消息前缀为 [PNPU]；显示 Info/Warn/Error，保留 [RUNTIME] 到首条 [ERROR] 和 [FALLBACK]。命令行可用 hdc hilog -T PermissionNPU。1.0.8 以 [TENSOR] 记录输入 rank/shape，并支持连续 token 向量的单例维度表示；应用校验错误保留原始消息。1.0.9 修复 N-API 长度返回字节数的兼容性，[INPUT] 每会话记录一次输入长度和范围。真机 RunSync 和自检仍需验证。\n\n"
        "离线编译与主机数值对照已通过；HAP 构建、签名和真机精度/性能尚未验证。\n")
    print(destination)


if __name__ == "__main__":
    main()
