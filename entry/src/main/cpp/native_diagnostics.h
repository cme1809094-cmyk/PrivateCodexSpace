#ifndef PERMISSION_NATIVE_DIAGNOSTICS_H
#define PERMISSION_NATIVE_DIAGNOSTICS_H

#include "native_log.h"
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <utility>
#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
#define PERMISSION_HAS_CXXABI 1
#endif

namespace permission_nnrt {
// Call only inside a catch handler. Report the ABI name without casting an
// unknown exception object or allocating memory while handling a failure.
inline const char *CurrentExceptionType() noexcept
{
#if defined(PERMISSION_HAS_CXXABI)
    const auto *type = abi::__cxa_current_exception_type();
    if (type) return type->name();
#endif
    return "<unavailable>";
}

// A Job owns this record, so it survives Session constructor unwinding. Never
// update it from cleanup: the useful position is the original failed call.
class NativeTrace {
public:
    explicit NativeTrace(bool verbose = false) noexcept : verbose_(verbose) {}

    void SetRuntimeVersion(const char *version) noexcept
    {
        std::snprintf(runtime_, sizeof(runtime_), "%s", version && *version ? version : "<unavailable>");
    }

    template <typename Function>
    auto Call(const char *operation, Function &&function, int index = -1)
    {
        if (index < 0) std::snprintf(stage_, sizeof(stage_), "%s", operation);
        else std::snprintf(stage_, sizeof(stage_), "%s[index=%d]", operation, index);
        std::snprintf(returned_, sizeof(returned_), "not returned");
        // Keep initialization milestones at Info; tensor API detail is Debug.
        const auto level = std::strcmp(operation, "HMS_HiAICompatibility_CheckFromBuffer") == 0
            || std::strcmp(operation, "OH_NNCompilation_Build") == 0
            || std::strcmp(operation, "OH_NNExecutor_Construct") == 0
            ? PermissionLogLevel::Info : PermissionLogLevel::Debug;
        if (verbose_) PermissionLog(stage_, "API", level);
        auto result = std::forward<Function>(function)();
        using Result = decltype(result);
        if constexpr (std::is_enum_v<Result> || std::is_integral_v<Result>) {
            std::snprintf(returned_, sizeof(returned_), "code=%lld", static_cast<long long>(result));
        } else if constexpr (std::is_pointer_v<Result>) {
            std::snprintf(returned_, sizeof(returned_), "%s", result ? "non-null" : "null");
        } else {
            std::snprintf(returned_, sizeof(returned_), "returned");
        }
        if (verbose_) {
            char message[288];
            std::snprintf(message, sizeof(message), "%s -> %s", stage_, returned_);
            PermissionLog(message, "API", level);
        }
        return result;
    }

    void Failure(char *buffer, size_t size, const char *operation, const char *detail) const noexcept
    {
        std::snprintf(buffer, size, "%s failed at %s (%s); exception_type=%.160s; CANN=%s; bridge=1.0.7; detail=%s",
            operation, stage_, returned_, CurrentExceptionType(), runtime_, detail ? detail : "<no message>");
    }

private:
    char stage_[192] = "worker entry";
    char returned_[64] = "not called";
    char runtime_[128] = "<unavailable>";
    bool verbose_;
};
} // namespace permission_nnrt
#endif
