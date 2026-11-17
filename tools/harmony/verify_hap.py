"""Verify the actual unsigned HAP contains bytecode, arm64 native code and paired model resources."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hap", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "ml/artifact-manifest.json").read_text())
    app = json.loads((root / "AppScope/app.json5").read_text())["app"]
    raw_prefix = "entry/src/main/"
    checked = []
    with zipfile.ZipFile(args.hap) as package:
        names = package.namelist()
        bytecode = [name for name in names if name.startswith("ets/") and name.endswith(".abc")]
        if not bytecode:
            raise ValueError("No compiled ArkTS bytecode in HAP")
        module = json.loads(package.read("module.json"))
        actual_app = module["app"]
        for field in ("bundleName", "versionName", "versionCode"):
            if actual_app[field] != app[field]:
                raise ValueError("Packaged app identity mismatch: " + field)
        native = "libs/arm64-v8a/libpermission_hiai.so"
        elf = package.read(native)
        if elf[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", elf, 18)[0] != 183:
            raise ValueError("Native bridge must be ELF64 little-endian AArch64")
        for item in manifest["artifacts"]:
            if not item["path"].startswith("entry/src/main/resources/rawfile/"):
                continue
            name = item["path"][len(raw_prefix):]
            data = package.read(name)
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Packaged model identity mismatch: " + name)
            checked.append(name)
        catalog_bytes = package.read("resources/rawfile/model_catalog.json")
        if catalog_bytes != (root / "entry/src/main/resources/rawfile/model_catalog.json").read_bytes():
            raise ValueError("Packaged model catalog differs from source")
        catalog = json.loads(catalog_bytes)
        for profile in catalog["profiles"]:
            for item in profile["metadata"]:
                name = "resources/rawfile/" + profile["assetPrefix"] + item["file"]
                data = package.read(name)
                if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("Packaged metadata mismatch: " + name)
    result = {
        "passed": True, "hap": str(args.hap.resolve()), "bytes": args.hap.stat().st_size,
        "sha256": hashlib.sha256(args.hap.read_bytes()).hexdigest(),
        "app": app, "compiled_arkts": bytecode, "native_bridge": native, "native_architecture": "AArch64",
        "packaged_model_files": checked, "model_profiles": [p["id"] for p in catalog["profiles"]],
        "signed": False, "device_tested": False
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("PASS: HAP identity, ArkTS bytecode, AArch64 native bridge and " + str(len(checked)) + " paired model binaries")
    print(args.hap.resolve())


if __name__ == "__main__":
    main()
