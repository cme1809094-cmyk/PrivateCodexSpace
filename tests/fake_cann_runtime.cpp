// Host-only fixture for native bridge fault injection. Never package this library.
#include <cstddef>
extern "C" int PermissionTestCompatibility();
extern "C" const char *HMS_HiAI_GetVersion() { return "host-test-only"; }
extern "C" int HMS_HiAICompatibility_CheckFromBuffer(const void *, size_t)
{
    return PermissionTestCompatibility();
}
