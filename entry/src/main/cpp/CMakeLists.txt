cmake_minimum_required(VERSION 3.16)
project(PermissionHiAI LANGUAGES CXX)

add_library(permission_hiai SHARED permission_hiai.cpp)
target_compile_features(permission_hiai PRIVATE cxx_std_17)
target_compile_options(permission_hiai PRIVATE -Wall -Wextra)
target_link_libraries(permission_hiai PRIVATE ace_napi.z neural_network_core)
