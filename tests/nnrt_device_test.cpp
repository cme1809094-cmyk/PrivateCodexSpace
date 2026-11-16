#include "nnrt_device.h"

#include <cassert>
#include <functional>
#include <iostream>

namespace {
std::vector<permission_nnrt::Device> hardware;
std::vector<size_t> ids;
OH_NN_ReturnCode typeStatus = OH_NN_SUCCESS;
bool invalidIds = false;

void ExpectFailure(const std::string &message)
{
    bool failed = false;
    try {
        (void)permission_nnrt::SelectAccelerator();
    } catch (const std::runtime_error &error) {
        failed = std::string(error.what()).find(message) != std::string::npos;
    }
    assert(failed);
}
} // namespace

extern "C" OH_NN_ReturnCode OH_NNDevice_GetAllDevicesID(const size_t **out, uint32_t *count)
{
    ids.clear();
    for (const auto &device : hardware) ids.push_back(device.id);
    *out = invalidIds ? nullptr : ids.data();
    *count = static_cast<uint32_t>(ids.size());
    return OH_NN_SUCCESS;
}

extern "C" OH_NN_ReturnCode OH_NNDevice_GetName(size_t id, const char **name)
{
    assert(*name == nullptr);
    for (const auto &device : hardware) {
        if (device.id == id) {
            *name = device.name.c_str();
            return OH_NN_SUCCESS;
        }
    }
    return OH_NN_INVALID_PARAMETER;
}

extern "C" OH_NN_ReturnCode OH_NNDevice_GetType(size_t id, OH_NN_DeviceType *type)
{
    if (typeStatus != OH_NN_SUCCESS) return typeStatus;
    for (const auto &device : hardware) {
        if (device.id == id) {
            *type = device.type;
            return OH_NN_SUCCESS;
        }
    }
    return OH_NN_INVALID_PARAMETER;
}

int main()
{
    // Reproduce the phone's actual backend name with CPU/GPU before it.
    const std::string reported = "NPU_ohos.boot.hardware.Kirin9020_v2_0";
    hardware = {{4, "CPU", OH_NN_CPU}, {5, "GPU", OH_NN_GPU}, {42, reported, OH_NN_ACCELERATOR}};
    auto selected = permission_nnrt::SelectAccelerator();
    assert(selected.id == 42 && selected.name == reported);
    assert(permission_nnrt::DescribeDevice(selected).find(reported) != std::string::npos);
    assert(permission_nnrt::DescribeDevice(selected).find("ACCELERATOR") != std::string::npos);

    hardware = {{9, "HIAI_F", OH_NN_ACCELERATOR}};
    assert(permission_nnrt::SelectAccelerator().id == 9);

    hardware = {{19, "A future vendor backend", OH_NN_ACCELERATOR}};
    assert(permission_nnrt::SelectAccelerator().id == 19);

    // A convincing name must not override CPU/GPU or unknown device types.
    hardware = {{1, "NPU_fake", OH_NN_CPU}, {2, "HIAI_F", OH_NN_GPU}, {3, reported, OH_NN_OTHERS}};
    ExpectFailure("No NNRT accelerator");
    ExpectFailure("type CPU");

    hardware.clear();
    ExpectFailure("enumerated: none");

    hardware = {{42, reported, OH_NN_ACCELERATOR}};
    typeStatus = OH_NN_FAILED;
    ExpectFailure("OH_NNDevice_GetType(42) failed");
    typeStatus = OH_NN_SUCCESS;

    invalidIds = true;
    ExpectFailure("NNRT returned no device IDs");
    std::cout << "PASS: reported Kirin backend, legacy and arbitrary names, CPU/GPU rejection, empty inventory, query error, invalid inventory\n";
}
