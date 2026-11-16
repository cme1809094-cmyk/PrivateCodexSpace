"""Verify model bytes after Git LFS checkout, using the published SHA-256 manifest."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "ml/artifact-manifest.json").read_text())
    failures = []
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        if not path.is_file():
            failures.append(f"Missing: {artifact['path']}")
            continue
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if size != artifact["bytes"] or digest.hexdigest() != artifact["sha256"]:
            failures.append(f"Size or SHA-256 mismatch: {artifact['path']}")
        else:
            print(f"OK {artifact['path']}")
    if failures:
        raise SystemExit("\n".join(failures) + "\nRun git lfs pull to materialize model files.")
    print(f"Verified {len(manifest['artifacts'])} model files.")


if __name__ == "__main__":
    main()
