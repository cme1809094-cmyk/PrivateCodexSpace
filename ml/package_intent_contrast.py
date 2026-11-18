"""Build two independently extractable Drive-sized experiment archives."""
import argparse
import json
from pathlib import Path
import zipfile

from train_intent_contrast import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    main_files, extra_files = {}, {}
    for path in args.run.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(args.run)
        if relative.name == "artifact_manifest.json":
            continue
        target = "run/" + relative.as_posix()
        if relative.parts[0] == "onnx" and any(relative.name.endswith(s) for s in
                ("_16.onnx", "_32.onnx", "_16.jsonl", "_32.jsonl")):
            extra_files[target] = path
        else:
            main_files[target] = path
    # A lightweight repository snapshot preserves the trainer's relative paths.
    for path in (ROOT / "ml").glob("*.py"):
        main_files[path.relative_to(ROOT).as_posix()] = path
    main_files["ml/requirements-training-cpu.txt"] = ROOT / "ml/requirements-training-cpu.txt"
    for directory in ("ml/data/permission_intent_v3",):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file():
                main_files[path.relative_to(ROOT).as_posix()] = path
    for name in ("ml/data/permission_intent_v2/capability_catalog.json",
                 "ml/data/permission_intent_v2/expansion/all_instructions.jsonl",
                 "ml/results/tinybert-intent-v2-20261118/data_manifest.json",
                 "entry/src/main/ets/model/WordPiece.ets", "tests/test_intent_training.py",
                 "docs/tinybert-intent-v3-training.md"):
        main_files[name] = ROOT / name
    for path in args.experiments.rglob("*"):
        if not path.is_file() or args.run in path.parents or path.suffix in (".safetensors", ".zip", ".onnx"):
            continue
        if (any(part == "checkpoint" for part in path.relative_to(args.experiments).parts)
                and path.name not in ("config.json", "protocol.json")):
            continue
        # Include all attempted configurations, with weights only for the winner.
        rel = path.relative_to(args.experiments)
        if rel.parts[0].startswith(("pilot-", "full-")) or len(rel.parts) == 1:
            main_files["development_attempts/" + rel.as_posix()] = path
    manifest = {"scope": "Selected checkpoint plus standard ONNX; human review pending; not a phone-ready OMC profile",
                "files": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                          for name, path in sorted({**main_files, **extra_files}.items())}}
    write_json(args.run / "artifact_manifest.json", manifest)
    main_files["run/artifact_manifest.json"] = args.run / "artifact_manifest.json"
    downloads = {}
    for name, files in (("TinyBERT-intent-v3-model-and-64-20261118.zip", main_files),
                        ("TinyBERT-intent-v3-onnx-16-32-20261118.zip", extra_files)):
        destination = args.output / name
        if destination.exists():
            raise ValueError("Refusing to overwrite an existing archive: " + str(destination))
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for target, path in sorted(files.items()):
                archive.write(path, target)
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
        if destination.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("Archive exceeds Drive tool's 100 MiB limit: " + name)
        downloads[name] = {"local_path": str(destination.resolve()), "bytes": destination.stat().st_size,
                           "sha256": sha256(destination), "zip_crc_passed": True}
    write_json(args.output / "downloads.json", downloads)
    print(json.dumps(downloads))


if __name__ == "__main__":
    main()
