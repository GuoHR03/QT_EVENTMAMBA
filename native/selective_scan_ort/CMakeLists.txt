cmake_minimum_required(VERSION 3.24)

if(WIN32)
    set(
        CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
        CACHE STRING "CUDA compiler flags" FORCE
    )
endif()

project(eventmamba_selective_scan LANGUAGES CXX CUDA)

if(NOT DEFINED ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "Set ONNXRUNTIME_INCLUDE_DIR to the directory containing onnxruntime_c_api.h")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(
    eventmamba_selective_scan
    SHARED
    selective_scan_custom_op.cu
    hierarchical_fps_custom_op.cpp
)
target_include_directories(eventmamba_selective_scan PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_compile_features(eventmamba_selective_scan PRIVATE cxx_std_17)
target_link_libraries(eventmamba_selective_scan PRIVATE CUDA::cudart)

set_target_properties(eventmamba_selective_scan PROPERTIES
    CUDA_ARCHITECTURES "75;86"
    CUDA_RUNTIME_LIBRARY Shared
    OUTPUT_NAME "eventmamba_selective_scan"
)

if(MSVC)
    target_compile_options(eventmamba_selective_scan PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/fp:precise>
    )
else()
    target_compile_options(eventmamba_selective_scan PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
        $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    )
endif()
