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

The cases cover non-HIAI accelerator names, the legacy HIAI_F name, mixed CPU/GPU/accelerator inventories, misleading names on non-accelerator devices, empty or invalid inventories, device-type query failures, and startup fallback/diagnostic behavior. Native syntax was also checked against public OpenHarmony NNRT and N-API headers; a HarmonyOS SDK application build and phone execution remain necessary.

The device type API is documented in [HarmonyOS NNRT C API](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/capi-neural-network-core-h/).
