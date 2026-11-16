#ifndef PERMISSION_NATIVE_LOG_H
#define PERMISSION_NATIVE_LOG_H

#include <cstdio>
#if defined(__OHOS__)
#include <hilog/log.h>
#endif

inline void PermissionLog(const char *message) noexcept
{
#if defined(__OHOS__)
    OH_LOG_Print(LOG_APP, LOG_INFO, 0xFF00, "PermissionHiAI", "%{public}s", message);
#else
    std::fprintf(stderr, "PermissionHiAI: %s\n", message);
#endif
}
#endif
