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
    # Check catalog identities and the metadata packaged with each model, too.
    raw = root / "entry/src/main/resources/rawfile"
    catalog = json.loads((raw / "model_catalog.json").read_text())
    published = {a["path"]: a for a in manifest["artifacts"]}
    for profile in catalog["profiles"]:
        base = raw / profile["assetPrefix"]
        for asset in profile["assets"]:
            suffix = "baseline" if asset["bucket"] == 0 else str(asset["bucket"])
            path = str((base / f"permission_{suffix}.om").relative_to(root))
            known = published.get(path)
            if not known or any(known[key] != asset[key] for key in ("bytes", "sha256")):
                failures.append(f"Catalog identity mismatch: {path}")
        for asset in profile["metadata"]:
            path = base / asset["file"]
            data = path.read_bytes()
            if len(data) != asset["bytes"] or hashlib.sha256(data).hexdigest() != asset["sha256"]:
                failures.append(f"Paired metadata mismatch: {path.relative_to(root)}")
        print(f"Checked catalog and paired metadata: {profile['id']}")
    if failures:
        raise SystemExit("\n".join(failures) + "\nRun git lfs pull to materialize model files.")
    print(f"Verified {len(manifest['artifacts'])} model files.")


if __name__ == "__main__":
    main()
