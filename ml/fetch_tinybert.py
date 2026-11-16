"""Download and checksum the pinned official Chinese TinyBERT 4L snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


MODEL_ID = "huawei-noah/TinyBERT_4L_zh"
REVISION = "152bf15d86715b41dd89c1e03cf5664963d9b005"
FILES = {
    "config.json": "d93cd853e1be7f53fad2cd59b78471480af6dff1888cc26064f011ce448dd468",
    "vocab.txt": "366fe2b1318c774fd28bdea58dbcfbd6d6aeae313fdcd9063d170cdc602fd51a",
    "pytorch_model.bin": "b4aec782e4a56727bb6ba1dcac8e6c3319460e58e11254f7e43a2dccee0c19c1",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = args.destination / name
        if path.is_file() and sha256(path) == expected:
            print(f"Verified existing {name}")
            continue
        temporary = path.with_name(path.name + ".part")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if sha256(temporary) != expected:
            raise ValueError(f"SHA-256 mismatch for {name}; retained {temporary}")
        temporary.replace(path)
        print(f"Downloaded and verified {name}")
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "sha256": FILES}
    (args.destination / "pinned-source.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
