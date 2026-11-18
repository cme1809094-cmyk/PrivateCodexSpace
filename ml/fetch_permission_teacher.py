"""Recover the public, revision-pinned teacher download cache, checking every byte."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


SOURCE = Path(__file__).resolve().parent / "data/permission_intent_v4/teacher_source.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest_bytes = SOURCE.read_bytes()
    manifest = json.loads(manifest_bytes)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["downloaded_files"].items():
        if name not in ("README.md", "config.json", "pytorch_model.bin", "vocab.txt"):
            raise ValueError("Unexpected pinned teacher filename")
        target = args.output / name
        valid = target.exists() and target.stat().st_size == expected["bytes"] and digest(target) == expected["sha256"]
        if not valid:
            if args.verify_only:
                raise ValueError("Missing or mismatched teacher cache: " + name)
            url = "https://huggingface.co/{}/resolve/{}/{}".format(manifest["model_id"], manifest["revision"], name)
            temporary = target.with_suffix(target.suffix + ".download")
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                if temporary.stat().st_size != expected["bytes"] or digest(temporary) != expected["sha256"]:
                    raise ValueError("Teacher download size/SHA256 verification failed: " + name)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        print("Verified " + name, flush=True)
    (args.output / "source-manifest.json").write_bytes(manifest_bytes)


if __name__ == "__main__":
    main()
