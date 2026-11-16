#ifndef PERMISSION_NATIVE_LOG_H
#define PERMISSION_NATIVE_LOG_H

#include <cstdio>
#if defined(__OHOS__)
#include <hilog/log.h>
#endif

enum class PermissionLogLevel { Debug, Info, Warn, Error };

inline void PermissionLog(const char *message, const char *stage = "NATIVE",
    PermissionLogLevel level = PermissionLogLevel::Info) noexcept
{
#if defined(__OHOS__)
    constexpr LogLevel levels[] = {LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR};
    OH_LOG_Print(LOG_APP, levels[static_cast<unsigned>(level)], 0xFF00, "PermissionNPU",
        "[PNPU][%{public}s] %{public}s", stage, message);
#else
    constexpr const char *levels[] = {"D", "I", "W", "E"};
    std::fprintf(stderr, "%s PermissionNPU: [PNPU][%s] %s\n", levels[static_cast<unsigned>(level)], stage, message);
#endif
}

inline void PermissionError(const char *message, const char *stage = "ERROR") noexcept
{
    PermissionLog(message, stage, PermissionLogLevel::Error);
}
#endif
