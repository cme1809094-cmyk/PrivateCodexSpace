cmake_minimum_required(VERSION 3.16)
project(PermissionHiAI LANGUAGES CXX)

# This bridge catches native errors at N-API/FFRT boundaries. Some parent
# toolchains accept try/catch but discard handlers with -fignore-exceptions.
foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
                 CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_MINSIZEREL)
    string(REPLACE "-Xclang -fignore-exceptions" "" ${flag_var} "${${flag_var}}")
    string(REPLACE "-fignore-exceptions" "" ${flag_var} "${${flag_var}}")
    string(REPLACE "-fno-exceptions" "" ${flag_var} "${${flag_var}}")
endforeach()

add_library(permission_hiai SHARED permission_hiai.cpp)
target_compile_features(permission_hiai PRIVATE cxx_std_17)
target_compile_options(permission_hiai PRIVATE -Wall -Wextra -fexceptions
    -funwind-tables -fno-omit-frame-pointer -g)
target_link_libraries(permission_hiai PRIVATE ace_napi.z neural_network_core ${CMAKE_DL_LIBS})
if(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
    target_link_libraries(permission_hiai PRIVATE hilog_ndk.z)
endif()

# -fignore-exceptions can leave __EXCEPTIONS defined. Verify that the resulting
# library really contains catch handlers before allowing it to be packaged.
add_custom_command(TARGET permission_hiai POST_BUILD
    COMMAND ${CMAKE_COMMAND} "-DNM_TOOL=${CMAKE_NM}"
        "-DNATIVE_LIBRARY=$<TARGET_FILE:permission_hiai>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_exception_support.cmake"
    VERBATIM)
