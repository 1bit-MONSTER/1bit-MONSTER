cmake_minimum_required(VERSION 3.16)
project(npu-infer C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Find XRT (default: system XRT). For the runlist-capable RuntimeLayerEngine,
# point these at a XRT >= 2.25 install (e.g. -DXRT_COREUTIL_PATH=/path/to/libxrt_coreutil.so):
find_library(XRT_COREUTIL xrt_coreutil HINTS ${XRT_COREUTIL_PATH} $ENV{XRT_COREUTIL_PATH})
find_library(XRT_XRT_CORE xrt_core  HINTS ${XRT_CORE_PATH}    $ENV{XRT_CORE_PATH})
find_package(Threads REQUIRED)

# The per-ctx layer-ELF RuntimeLayerEngine (issue #1776 / HRX runlist) needs a
# runlist-capable XRT (>= 2.25). Opt-in so the default (system XRT 2.21.75)
# build is unaffected.
option(BUILD_RUNTIME_LAYER "Build the RuntimeLayerEngine (requires runlist-capable XRT >= 2.25)" OFF)

# Build model library
add_library(npu_model STATIC
    src/model.c
)
target_include_directories(npu_model PUBLIC include)

# Build inference engine library
add_library(npu_engine STATIC
    src/engine.cpp
)
if(BUILD_RUNTIME_LAYER)
    target_sources(npu_engine PRIVATE src/runtime_layer.cpp)
endif()
target_include_directories(npu_engine PUBLIC include)
target_link_libraries(npu_engine npu_model ${XRT_COREUTIL} Threads::Threads)

# Test weight packer
add_executable(test_weight_packer
    src/test_weight_packer.c
)
target_link_libraries(test_weight_packer npu_model m)

# Original test model
add_executable(test_model
    src/test_model.c
    src/model.c
)
target_include_directories(test_model PRIVATE include)
target_link_libraries(test_model m)

# Instruction dumper (C++)
add_executable(dump_instrs tools/dump_instrs.cpp)
target_link_libraries(dump_instrs dl)
set_target_properties(dump_instrs PROPERTIES LINKER_LANGUAGE CXX)

# Full inference engine
add_executable(npu_infer
    src/main.cpp
    src/engine.cpp
    src/model.c
)
if(BUILD_RUNTIME_LAYER)
    target_sources(npu_infer PRIVATE src/runtime_layer.cpp)
endif()
target_include_directories(npu_infer PRIVATE include)
target_link_libraries(npu_infer npu_model ${XRT_COREUTIL} ${XRT_XRT_CORE} Threads::Threads m dl uuid)
target_compile_options(npu_infer PRIVATE -O3 -mavx2 -mfma -mavx512f -ffast-math -march=native -funroll-loops -fno-math-errno -fvect-cost-model=unlimited -g)
