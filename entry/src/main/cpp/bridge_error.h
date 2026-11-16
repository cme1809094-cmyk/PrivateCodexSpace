#ifndef PERMISSION_BRIDGE_ERROR_H
#define PERMISSION_BRIDGE_ERROR_H

#include <cstdio>
#include <string>

namespace permission_nnrt {
// Application validation failures must retain their own message even when a
// device's C++ runtime does not match std::runtime_error to std::exception.
// This type is constructed and caught by the bridge, without a libc++ base.
class BridgeError final {
public:
    explicit BridgeError(const char *message) noexcept
    {
        std::snprintf(message_, sizeof(message_), "%s", message ? message : "<no message>");
    }
    explicit BridgeError(const std::string &message) noexcept : BridgeError(message.c_str()) {}
    const char *what() const noexcept { return message_; }

private:
    char message_[768]{};
};
} // namespace permission_nnrt
#endif
