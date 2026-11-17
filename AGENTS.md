# Project validation

This is the combined MiniRBT / TinyBERT HarmonyOS app. Preserve each model's
weights, vocabulary, thresholds and reference vectors as a paired profile.

For changes to ArkTS, native C++, type declarations or build configuration,
run a complete unsigned HAP build before reporting the change as validated:

```sh
python3 tools/harmony/setup.py --root /workspace/work/harmony-build
bash tools/harmony/build.sh /workspace/work/harmony-build
```

The installer pins HarmonyOS 5.0.0.71 / API 12, its bundled build tools and
JDK 17. A normal TypeScript compiler does not enforce ArkTS restrictions and
cannot replace the HAP build. Do not throw a caught arbitrary value from ArkTS;
normalize it to a new Error while preserving the original message.

OHPM uses its official ~/.ohpm cache. In a restricted execution sandbox, grant
write access to that cache directory when running the build; do not reassign HOME.

Run the relevant host routing regression with Node 22+ when changing the model
controller. The SDK's bundled Node 18 is for building, not TypeScript stripping:

```sh
node --experimental-strip-types tests/model_switch_test.mjs
```

Do not commit SDKs, signing material or generated HAP/build directories. An
unsigned HAP and host numerical comparisons do not prove phone NPU execution.
