# NNRT device selection regression checks

## Model switching and acceptance (1.1.1)

```sh
node --experimental-strip-types tests/model_switch_test.mjs
python3 ml/verify_artifacts.py
```

The switching suite exercises the actual ArkTS PermissionModel and WordPiece sources with mocked platform bindings. Both real vocabularies are checked against all 33 paired input-id/mask vectors. Nineteen scenarios cover default selection, all model/bucket combinations, session isolation/release, routing, manual overflow, same-model CPU fallback, metadata failure recovery, exclusive operations, and acceptance gates. Acceptance failures include non-finite/short/wrong-label outputs, excessive logit drift despite identical labels, incomplete reference coverage, Build/RunSync errors and missing NPU. Version 1.1.1 also checks that cleanup failure preserves the original validation error. Mock timing is only used to test control flow; it is not a phone performance measurement.

The native suite now has 75 scenarios. The six additions call the actual Dispose/Predict exports, checking worker release, idempotence, use-after-dispose rejection, ownership retained by queued inference, restored ownership after queue failure and invalid wrappers. The CMake host DSO still uses placeholder libraries and must never be packaged as a HarmonyOS binary.

These host tests reproduce the reported `NPU_ohos.boot.hardware.Kirin9020_v2_0` backend name. They mock device enumeration and ArkTS platform bindings; they do not execute the compiled OMC or replace an on-phone test.

Compile the C++ test with the real NNRT SDK headers (or the public OpenHarmony NNRT C headers) available under `SDK_INCLUDE/neural_network_runtime/`:

```sh
g++ -std=c++17 -Wall -Wextra -Werror \
  -I"$SDK_INCLUDE" -Ientry/src/main/cpp \
  tests/nnrt_device_test.cpp -o /tmp/permission-nnrt-device-test
/tmp/permission-nnrt-device-test
```

With Node 22+ supporting TypeScript stripping:

```sh
node --no-warnings --experimental-strip-types tests/nnrt_startup_test.mjs
```

Hardware accelerator detection is separate from offline OMC capability. The device cases check that the reported generic Kirin backend is detectable but is not used for CANN offline buffers, while HIAI_F is eligible. They also cover CPU/GPU rejection, empty/error inventories and startup CPU fallback. The ArkTS test injects a string rejection from model loading and verifies that its diagnostic survives fallback.

Run the actual native worker/completion callbacks with NNRT/N-API failures injected:

```sh
python3 tests/run_native_error_test.py --sdk-include "$SDK_INCLUDE"
```

The runner builds a host-only fake CANN library in a temporary directory. Never package that fixture in the app. The original 21 scenarios cover Build status failures, incompatible models, unsupported offline backends, standard/integer/string/unknown exceptions, JS Error/string allocation failures, pending JS exceptions, result conversion/resolve failures and cancellation. They also inject failures at compatibility checking, executor construction, input descriptor/tensor creation, tensor validation and RunSync. Failed jobs retain their API and return status after constructor cleanup, plus the exception ABI name; successful mock load/predict still resolves. The ArkTS test verifies that the detailed rejection survives all three bucket attempts and CPU fallback. These are host simulations, not evidence of successful phone execution.

GCC and Clang runs passed. A host CMake smoke build with public SDK headers and placeholder link libraries also confirmed that injected `-fignore-exceptions`/`-fno-exceptions` options were removed and the DSO contained catch/unwind tables. The post-build checker rejects a deliberately built DSO without catch handlers. These checks do not build a deployable HAP or establish phone/model compatibility.

The device type API is documented in [HarmonyOS NNRT C API](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h/).

Since 1.0.7, native and ArkTS diagnostics use HiLog tag `PermissionNPU` and message prefix `[PNPU]`. The ArkTS host mock captures HiLog calls and checks that detailed fallback messages retain that tag across all three buckets. Native fault-injection output includes `[ERROR]` at Error level; detailed tensor calls use Debug and compatibility/Build/executor milestones use Info. The worker's failure is not logged again when rejecting its promise. See the root README for filters and stage labels.

Version 1.0.8 adds 23 scenarios (44 total): all three token lengths in four equivalent layouts are loaded through the real bridge callbacks, input ids/masks are checked byte for byte, and returned logits are checked. Eleven rejection cases cover non-vector shapes, unsupported/zero/dynamic dimensions, empty/null shapes, excessive rank, mismatched ids/mask lengths and wrong element counts. Bridge-owned validation uses an explicitly caught fixed-buffer BridgeError and retains the tensor shape plus the failed validation stage. The phone log proved compatibility/Build/executor creation succeeded for MiniRBT, then its old rank-2 guard failed; the actual phone shape and on-device inference still require a rebuilt-app check.

Version 1.0.9 adds 25 N-API boundary scenarios (69 total). Twelve call the exported Predict callback through argument decoding, promise/work creation, Execute/Complete and NNRT RunSync mocks: 16/32/64 token buckets, API lengths of 64 elements or 256 bytes, and zero/nonzero byte offsets. JS backing buffers are cleared after queueing; the worker still receives the expected ids/masks and logits. Thirteen invalid cases never queue or run, including a 256-element Int32Array whose reported length is ambiguously 256, wrong element types, inconsistent view sizes, short/null buffers, misaligned/overflowing offsets, and N-API property/buffer errors. This closes the previous test gap where inputs were assigned directly to Job and the real ReadTokens/Predict boundary was bypassed.

A complete unsigned HAP build with HarmonyOS 5.0.0.71 / API 12 passed for 1.1.1; this includes the real ArkTS checker, AArch64 C++ build and packaged model verification. See `tools/harmony/README.md` and `ml/harmony-build-verification.json`. These build checks still do not establish phone NPU execution.
