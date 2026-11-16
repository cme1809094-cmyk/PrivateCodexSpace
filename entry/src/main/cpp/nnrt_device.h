#ifndef PERMISSION_NNRT_DEVICE_H
#define PERMISSION_NNRT_DEVICE_H

#include <neural_network_runtime/neural_network_core.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace permission_nnrt {
struct Device {
    size_t id;
    std::string name;
    OH_NN_DeviceType type;
};

inline const char *DeviceTypeName(OH_NN_DeviceType type)
{
    switch (type) {
        case OH_NN_CPU: return "CPU";
        case OH_NN_GPU: return "GPU";
        case OH_NN_ACCELERATOR: return "ACCELERATOR";
        default: return "OTHERS";
    }
}

inline std::string DescribeDevice(const Device &device)
{
    return device.name + " (ID " + std::to_string(device.id) + ", type " + DeviceTypeName(device.type)
        + "=" + std::to_string(static_cast<int>(device.type)) + ")";
}

inline void CheckDeviceQuery(OH_NN_ReturnCode code, const std::string &operation)
{
    if (code != OH_NN_SUCCESS) {
        throw std::runtime_error(operation + " failed (" + std::to_string(code) + ")");
    }
}

inline std::vector<Device> ListDevices()
{
    const size_t *ids = nullptr;
    uint32_t count = 0;
    CheckDeviceQuery(OH_NNDevice_GetAllDevicesID(&ids, &count), "OH_NNDevice_GetAllDevicesID");
    if (count != 0 && !ids) {
        throw std::runtime_error("NNRT returned no device IDs");
    }
    std::vector<Device> devices;
    for (uint32_t i = 0; i < count; ++i) {
        const char *name = nullptr;
        OH_NN_DeviceType type = OH_NN_OTHERS;
        const std::string id = std::to_string(ids[i]);
        CheckDeviceQuery(OH_NNDevice_GetName(ids[i], &name), "OH_NNDevice_GetName(" + id + ")");
        CheckDeviceQuery(OH_NNDevice_GetType(ids[i], &type), "OH_NNDevice_GetType(" + id + ")");
        devices.push_back({ids[i], name ? name : "<unnamed>", type});
    }
    return devices;
}

inline Device SelectAccelerator()
{
    const auto devices = ListDevices();
    for (const auto &device : devices) {
        if (device.type == OH_NN_ACCELERATOR) {
            return device;
        }
    }
    std::string message = "No NNRT accelerator is available; enumerated: ";
    for (const auto &device : devices) {
        message += DescribeDevice(device) + "; ";
    }
    throw std::runtime_error(message + (devices.empty() ? "none" : ""));
}
} // namespace permission_nnrt
#endif
