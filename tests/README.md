# NNRT device selection regression checks

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

The runner builds a host-only fake CANN library in a temporary directory. Never package that fixture in the app. Eleven cases cover Build status failures, incompatible models, unsupported offline backends, standard/unknown vendor exceptions, JS Error/string allocation failures, pending JS exceptions, result conversion/resolve failures and cancellation. Each checks promise rejection and async-work cleanup without process abort.

GCC and Clang runs passed. A host CMake smoke build with public SDK headers and placeholder link libraries also confirmed that injected `-fignore-exceptions`/`-fno-exceptions` options were removed and the DSO contained catch/unwind tables. The post-build checker rejects a deliberately built DSO without catch handlers. These checks do not build a deployable HAP or establish phone/model compatibility.

The device type API is documented in [HarmonyOS NNRT C API](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h/).
