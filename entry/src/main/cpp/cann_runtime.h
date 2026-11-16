#ifndef PERMISSION_CANN_RUNTIME_H
#define PERMISSION_CANN_RUNTIME_H

#include "native_log.h"
#include <cstdio>
#include <dlfcn.h>
#include <exception>

namespace permission_nnrt {
// CANN's public C ABI is loaded lazily so an absent Foundation library still
// permits the application's CPU fallback. Keep the handle for process lifetime:
// registered NNRT backend objects can reference code in this library.
class CannRuntime {
public:
    using VersionFunction = const char *(*)();
    using CompatibilityFunction = int (*)(const void *, size_t);

    CannRuntime() noexcept
    {
        try {
            handle_ = dlopen("libhiai_foundation.so", RTLD_NOW | RTLD_GLOBAL);
            if (!handle_) {
                const char *detail = dlerror();
                std::snprintf(error_, sizeof(error_), "Cannot load CANN Foundation: %s", detail ? detail : "dlopen failed");
                PermissionLog(error_);
                return;
            }
            auto version = reinterpret_cast<VersionFunction>(dlsym(handle_, "HMS_HiAI_GetVersion"));
            compatibility_ = reinterpret_cast<CompatibilityFunction>(dlsym(handle_, "HMS_HiAICompatibility_CheckFromBuffer"));
            if (!version) {
                std::snprintf(error_, sizeof(error_), "CANN Foundation has no HMS_HiAI_GetVersion symbol");
                PermissionLog(error_);
                return;
            }
            const char *value = version();
            std::snprintf(version_, sizeof(version_), "%s", value ? value : "<unknown>");
            available_ = true;
            char message[256];
            std::snprintf(message, sizeof(message), "CANN Foundation loaded, version=%s", version_);
            PermissionLog(message);
        } catch (const std::exception &error) {
            std::snprintf(error_, sizeof(error_), "CANN initialization exception: %s", error.what());
            available_ = false;
            PermissionLog(error_);
        } catch (...) {
            std::snprintf(error_, sizeof(error_), "Unknown CANN initialization exception");
            available_ = false;
            PermissionLog(error_);
        }
    }

    bool Available() const noexcept { return available_; }
    const char *Error() const noexcept { return error_; }
    const char *Version() const noexcept { return version_; }
    bool HasCompatibilityCheck() const noexcept { return compatibility_ != nullptr; }
    int CheckCompatibility(const void *data, size_t size) const { return compatibility_(data, size); }

private:
    void *handle_ = nullptr;
    CompatibilityFunction compatibility_ = nullptr;
    bool available_ = false;
    char error_[512]{};
    char version_[128]{};
};

inline const CannRuntime &GetCannRuntime()
{
    static CannRuntime runtime;
    return runtime;
}
} // namespace permission_nnrt
#endif
