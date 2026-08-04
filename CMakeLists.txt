cmake_minimum_required(VERSION 3.21)

# Pin HIP compiler to TheRock's clang for native gfx1151. Discover TheRock
# from multiple locations in priority order:
#   1. /opt/rocm-therock — TheRock nightly C++ SDK (this repo)
#   2. THEROCK_PIP_ROOT env var — pip-installed TheRock
#   3. lemonade cache (~/.cache/lemonade/bin/therock)
#   4. legacy therock path (~/therock/build/dist/rocm)
#   5. system ROCm (/opt/rocm)
set(_RC_THEROCK_OPT     "/opt/rocm-therock" CACHE PATH "TheRock opt install root")
set(_RC_THEROCK_PIP     "$ENV{HOME}/.cache/pip/therock" CACHE PATH "pip-installed TheRock root")
set(_RC_LEMONADE_THEROCK "$ENV{HOME}/.cache/lemonade/bin/therock")
set(_RC_THEROCK_LEGACY "$ENV{HOME}/therock/build/dist/rocm")
set(_RC_SYSTEM_BIN     "/opt/rocm/lib/llvm/bin")

set(_RC_LLVM_BIN "")
set(_RC_ROCM_DEFAULT "")

# 1. /opt/rocm-therock — TheRock nightly C++ SDK (system-wide install)
if(EXISTS "${_RC_THEROCK_OPT}/bin/amdclang++")
    set(_RC_LLVM_BIN "${_RC_THEROCK_OPT}/bin")
    set(_RC_ROCM_DEFAULT "${_RC_THEROCK_OPT}")
    message(STATUS "rocm-cpp: using TheRock opt at ${_RC_ROCM_DEFAULT}")
endif()

# 2. Try pip-installed TheRock (from virtual env) — check for rocm-sdk path
if(NOT _RC_LLVM_BIN AND DEFINED ENV{THEROCK_PIP_ROOT})
    set(_RC_PIP_PATH "$ENV{THEROCK_PIP_ROOT}")
    if(EXISTS "${_RC_PIP_PATH}/bin/amdclang++")
        set(_RC_LLVM_BIN "${_RC_PIP_PATH}/bin")
        set(_RC_ROCM_DEFAULT "${_RC_PIP_PATH}")
        message(STATUS "rocm-cpp: using pip TheRock at ${_RC_ROCM_DEFAULT}")
    endif()
endif()

# 3. Lemonade cache
if(NOT _RC_LLVM_BIN)
    file(GLOB _RC_THEROCK_DIRS LIST_DIRECTORIES true "${_RC_LEMONADE_THEROCK}/gfx*")
    list(SORT _RC_THEROCK_DIRS)
    list(REVERSE _RC_THEROCK_DIRS)
    list(LENGTH _RC_THEROCK_DIRS _RC_THEROCK_N)
    if(_RC_THEROCK_N GREATER 0)
        list(GET _RC_THEROCK_DIRS 0 _RC_THEROCK_DIR)
        if(EXISTS "${_RC_THEROCK_DIR}/bin/clang++")
            set(_RC_LLVM_BIN "${_RC_THEROCK_DIR}/bin")
            set(_RC_ROCM_DEFAULT "${_RC_THEROCK_DIR}")
            message(STATUS "rocm-cpp: using TheRock at ${_RC_ROCM_DEFAULT}")
        endif()
    endif()
endif()

# 4. Legacy therock path
if(NOT _RC_LLVM_BIN)
    if(EXISTS "${_RC_THEROCK_LEGACY}/lib/llvm/bin/clang++")
        set(_RC_LLVM_BIN "${_RC_THEROCK_LEGACY}/lib/llvm/bin")
        set(_RC_ROCM_DEFAULT "${_RC_THEROCK_LEGACY}")
        message(STATUS "rocm-cpp: using legacy TheRock at ${_RC_ROCM_DEFAULT}")
    endif()
endif()

# 5. System ROCm
if(NOT _RC_LLVM_BIN)
    if(EXISTS "${_RC_SYSTEM_BIN}/amdclang++")
        set(_RC_LLVM_BIN "${_RC_SYSTEM_BIN}")
        set(_RC_ROCM_DEFAULT "/opt/rocm")
        message(STATUS "rocm-cpp: using system ROCm at ${_RC_ROCM_DEFAULT}")
    else()
        message(FATAL_ERROR
            "rocm-cpp needs an AMD clang: set THEROCK_PIP_ROOT env var, install "
            "via 'pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \"rocm[libraries,devel,device-gfx1151]\"', "
            "or install system ROCm at /opt/rocm.")
    endif()
endif()

set(CMAKE_HIP_COMPILER "${_RC_LLVM_BIN}/amdclang++" CACHE FILEPATH "HIP compiler")
set(CMAKE_C_COMPILER   "${_RC_LLVM_BIN}/amdclang"   CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${_RC_LLVM_BIN}/amdclang++" CACHE FILEPATH "C++ compiler")

# The ROCm compiler is under /opt/rocm* but the HIP runtime (hip-runtime-amd deb)
# installs cmake configs at the Debian multiarch path.  CMake's compiler-driven
# arch detection produces "x86_64-unknown-linux-gnu" which doesn't match the
# Debian "x86_64-linux-gnu".  We create a local stub with the expected structure
# pointing into /usr so the project(LANGUAGES HIP) detection succeeds.
# Local stub for Debian-multiarch HIP runtime (created by the local
# setup; never committed). Only force the stub when it actually exists —
# CI runners with a real ROCm install must keep default detection.
set(_1BIT_ROCM_STUB "${CMAKE_CURRENT_SOURCE_DIR}/.rocm-stub")
if(EXISTS "${_1BIT_ROCM_STUB}")
    set(CMAKE_HIP_COMPILER_ROCM_ROOT "${_1BIT_ROCM_STUB}"
        CACHE PATH "HIP ROCm root (local stub)" FORCE)
endif()

project(1bit-systems LANGUAGES C CXX HIP)

# ── Optional: CUDA (NVIDIA GPU) backend ──
option(USE_CUDA "Build the CUDA GPU backend for NVIDIA GPUs" OFF)
set(RCPP_HAVE_CUDA FALSE)
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(RCPP_HAVE_CUDA TRUE)
        message(STATUS "CUDA: Toolkit found at ${CUDAToolkit_INCLUDE_DIRS} — building CUDA backend")
    else()
        message(WARNING "USE_CUDA=ON but CUDA Toolkit not found — skipping CUDA backend")
    endif()
endif()

# ── Optional: Metal (Apple GPU) backend ──
option(USE_METAL "Build the Metal GPU backend for Apple Silicon" OFF)
set(RCPP_HAVE_METAL FALSE)
if(USE_METAL AND APPLE)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(METAL_PERFORMANCE_SHADERS_LIBRARY MetalPerformanceShaders REQUIRED)
    find_library(METAL_PERFORMANCE_SHADERS_GRAPH_LIBRARY MetalPerformanceShadersGraph)
    set(RCPP_HAVE_METAL TRUE)
    message(STATUS "Metal: Frameworks found at ${METAL_LIBRARY} — building Metal backend")
elif(USE_METAL AND NOT APPLE)
    message(WARNING "USE_METAL=ON but this is not an Apple platform — skipping Metal backend")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force-include the HIP runtime wrapper header which sets up device math
# built-ins (__ocml_*) before <cmath> can shadow them with __host__
# declarations.  This is what hipcc does internally; replicating it here
# lets us use amdclang++ directly while keeping math functions working
# in __global__ functions.
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -include __clang_hip_runtime_wrapper.h")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wno-unused-command-line-argument")

enable_testing()

# NDEBUG only for Release — let Debug builds keep assert() guards
add_compile_definitions($<$<CONFIG:Release>:NDEBUG>)

# -- Discover TheRock SDK under Python site-packages (any Python version) --
# TheRock's ROCm SDK may install under a Python-versioned path like
#   lib/python3.X/site-packages/_rocm_sdk_devel/
# Instead of hardcoding a specific Python version (issue #355), search
# under all Python site-packages for the _rocm_sdk_devel directory.
# Must run BEFORE find_package(hip) below — it supplies THEROCK_SDK_BASE,
# the only PATHS entry that matches the pip/venv TheRock layout.
function(_find_therock_sdk_base out_var)
    if(DEFINED CACHE{THEROCK_SDK_BASE})
        set(${out_var} "${THEROCK_SDK_BASE}" PARENT_SCOPE)
        return()
    endif()
    # Search common locations, preferring newest Python version
    foreach(_base IN ITEMS "$ENV{HOME}/.local/lib" "/usr/local/lib" "/usr/lib" "${_RC_THEROCK_OPT}/lib")
        file(GLOB _py_dirs LIST_DIRECTORIES true "${_base}/python*")
        list(SORT _py_dirs)
        list(REVERSE _py_dirs)
        foreach(_py_dir IN LISTS _py_dirs)
            set(_candidate "${_py_dir}/site-packages/_rocm_sdk_devel")
            if(EXISTS "${_candidate}/include/rocwmma/rocwmma.hpp" OR EXISTS "${_candidate}")
                set(${out_var} "${_candidate}" PARENT_SCOPE)
                set(THEROCK_SDK_BASE "${_candidate}" CACHE INTERNAL "Found TheRock SDK base")
                message(STATUS "rocm-cpp: TheRock SDK found at ${_candidate}")
                return()
            endif()
        endforeach()
    endforeach()
    # Fallback: system ROCm
    if(EXISTS "/opt/rocm/lib/llvm/bin/amdclang++")
        set(${out_var} "/opt/rocm" PARENT_SCOPE)
        return()
    endif()
    set(${out_var} "" PARENT_SCOPE)
endfunction()

_find_therock_sdk_base(THEROCK_SDK_BASE)

set(ROCM_ROOT "${_RC_ROCM_DEFAULT}" CACHE PATH "ROCm install root")
list(PREPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}" "${ROCM_ROOT}/lib/cmake")

find_package(hip REQUIRED CONFIG
    PATHS "${ROCM_ROOT}"
    # System-wide HIP runtime (hip-runtime-amd package)
    /usr
    /usr/lib/x86_64-linux-gnu
    # TheRock pip-installed SDK
    "${THEROCK_SDK_BASE}"
)

set(ROCWMMA_FOUND FALSE)
foreach(_rc_base "${_RC_PIP_PATH}" "${_RC_THEROCK_DIR}" "${_RC_THEROCK_LEGACY}" "/opt/rocm-therock" "/opt/rocm")
    if(_rc_base AND EXISTS "${_rc_base}/include/rocwmma/rocwmma.hpp")
        set(ROCWMMA_INCLUDE_DIR "${_rc_base}/include")
        set(ROCWMMA_FOUND TRUE)
        message(STATUS "rocm-cpp: rocWMMA found at ${ROCWMMA_INCLUDE_DIR}")
        break()
    endif()
endforeach()

if(NOT ROCWMMA_FOUND)
    file(GLOB_RECURSE _RC_ROCWMMA_HPP "$ENV{HOME}/.local/lib/python*/site-packages/_rocm_sdk_devel/include/rocwmma/rocwmma.hpp")
    list(LENGTH _RC_ROCWMMA_HPP _RC_ROCWMMA_N)
    if(_RC_ROCWMMA_N GREATER 0)
        list(GET _RC_ROCWMMA_HPP 0 _RC_ROCWMMA_PATH)
        get_filename_component(ROCWMMA_INCLUDE_DIR "${_RC_ROCWMMA_PATH}" DIRECTORY)
        get_filename_component(ROCWMMA_INCLUDE_DIR "${ROCWMMA_INCLUDE_DIR}" DIRECTORY)
        set(ROCWMMA_FOUND TRUE)
        message(STATUS "rocm-cpp: rocWMMA found at ${ROCWMMA_INCLUDE_DIR} (pip fallback)")
    endif()
endif()
if(NOT ROCWMMA_FOUND)
    message(STATUS "rocm-cpp: rocWMMA NOT found — WMMA intrinsics disabled")
endif()


include(FetchContent)
FetchContent_Declare(cpp_httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.47.0)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3)
FetchContent_Declare(ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG        v5.0.0)
set(FTXUI_BUILD_DOCS     OFF CACHE INTERNAL "")
set(FTXUI_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(FTXUI_BUILD_TESTS    OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(cpp_httplib nlohmann_json ftxui)

set(RCPP_HAVE_CK FALSE)
find_package(composable_kernel QUIET CONFIG PATHS "${ROCM_ROOT}")
if(composable_kernel_FOUND)
    set(RCPP_HAVE_CK TRUE)
    message(STATUS "rocm-cpp: composable_kernel FOUND — building ck_gemm.cpp")
else()
    message(STATUS "rocm-cpp: composable_kernel NOT found — skipping ck_gemm.cpp, using standalone prefill")
endif()

# ── CUDA engine static library (libcuda_engine.a) ──
# Follows the same pattern as rocm_cpp (HIP) but for CUDA.
# The engine .cu file is compiled with CUDA; backend_cuda.cpp (pure C++
# orchestration calling extern "C" wrappers) goes into backend_manager.
if(RCPP_HAVE_CUDA)
    enable_language(CUDA)
    cuda_add_library(cuda_engine STATIC src/cuda_engine.cu)
    target_include_directories(cuda_engine PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_compile_options(cuda_engine PRIVATE -O3)
    target_link_libraries(cuda_engine PUBLIC CUDA::cudart CUDA::cublas)
    message(STATUS "CUDA engine: libcuda_engine.a built — backend_cuda.cpp goes in backend_manager")
endif()

# ── Metal backend ──
# On macOS, backend_metal.mm is added to backend_manager (same as
# backend_mamba1.cpp in PR #579). The Metal frameworks are linked into
# backend_manager so all consumers get Metal transitively.
if(RCPP_HAVE_METAL)
    message(STATUS "Metal backend: backend_metal.mm added via backend_manager")
endif()

option(USE_VULKAN "Build the portable Vulkan backend proof (test_vulkan_gemv)" OFF)
set(RCPP_HAVE_VULKAN FALSE)
if(USE_VULKAN)
    find_package(Vulkan QUIET)
    find_program(GLSLC_EXECUTABLE glslc)
    if(Vulkan_FOUND AND GLSLC_EXECUTABLE)
        set(RCPP_HAVE_VULKAN TRUE)
        message(STATUS "rocm-cpp: Vulkan FOUND (${Vulkan_LIBRARY}), glslc at ${GLSLC_EXECUTABLE} — building test_vulkan_gemv")
    else()
        message(WARNING "rocm-cpp: USE_VULKAN=ON but Vulkan SDK or glslc not found — skipping test_vulkan_gemv")
    endif()
endif()

# Auto-detect GPU architecture.  WMMA wave32 intrinsics
# (__builtin_amdgcn_wmma_*) are only available on gfx11 targets.
# gfx1201 (RDNA4) and gfx1036 (RDNA2) don't support them — the WMMA
# kernels in prefill_standalone, wmma_peak_probe, and ternary_gemv_wmma
# are excluded via WMMA_WAVE32_DISABLED when building for those targets.
if(NOT CMAKE_HIP_ARCHITECTURES)
    set(_DETECTED_ARCH "")
    execute_process(
        COMMAND bash -c "rocminfo 2>/dev/null | grep -oP 'gfx[0-9a-z]+' | head -1"
        OUTPUT_VARIABLE _DETECTED_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_DETECTED_ARCH)
        set(CMAKE_HIP_ARCHITECTURES "${_DETECTED_ARCH}")
    else()
        set(CMAKE_HIP_ARCHITECTURES "gfx1151")
    endif()
endif()
set(CMAKE_HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "GPU arch(es) to build")

# WMMA wave32 intrinsics (__builtin_amdgcn_wmma_*) need gfx11-insts which
# are only available on gfx11 targets (gfx1150, gfx1151, etc.).  For gfx10
# (RDNA2) and gfx12 (RDNA4) we exclude the WMMA kernel sources and define
# WMMA_WAVE32_DISABLED so any included headers can provide scalar fallbacks.
string(REGEX MATCH "gfx11[0-9a-z]+" _HAS_WMMA_ARCH "${CMAKE_HIP_ARCHITECTURES}")
if(NOT _HAS_WMMA_ARCH)
    add_compile_definitions(WMMA_WAVE32_DISABLED)
    message(STATUS "rocm-cpp: GPU ${CMAKE_HIP_ARCHITECTURES} — WMMA wave32 disabled, scalar fallbacks active")
endif()

# -- librocm_cpp : shared library exposing the C API ----------------------------
set(RCPP_SOURCES
    src/ternary_gemv_launchers.hip
    src/prefill_standalone.hip
    src/wmma_peak_probe.hip
    src/prim_kernels.hip
    src/kv_cache_attn.hip
    src/kv_cache_attn_i8.hip
    src/kv_cache_attn_fd.hip
    src/kv_cache_attn_fd_rotor.hip
    src/kv_rotorquant.cpp
    src/h1b_loader.cpp
    src/gguf_loader.cpp
    src/onnx_loader.cpp
    src/tokenizer.cpp
    src/ternary_gemm_smallm.hip
    src/ternary_gemm_smallm_scalar_ref.hip
    src/sherry_gemv.hip
    src/sherry_gemv_scalar_ref.hip
    src/bonsai_q1_gemv.hip
    kernels/bonsai_q1_gemv_soa.hip
    src/bonsai_tq2_gemv.hip
    src/bonsai_gemv_scalar_ref.hip
    src/medusa_tree_attn.hip
    src/medusa_small_m_gemv.hip
    kernels/ternary_gemv_phase5_dot4.hip
    kernels/ternary_gemv_phase5_halo.hip
    kernels/ternary_gemv_sherry.hip
    kernels/ternary_gemv_tq1_halo.hip
    kernels/ternary_gemv_wmma.hip
    kernels/ternary_gemv_block_scaled.hip
    kernels/ternary_gemv_q1_0.hip
    kernels/bitnet_gguf_gemv.hip
    kernels/iq_gemv.hip
    kernels/rotorquant_pack.hip
    kernels/hadamard_rotate_butterfly.hip
    kernels/zaya_moe_ternary_gemv.hip
    kernels/zaya_moe_tiled_gemv.hip
    kernels/zaya_moe_wmma_batched.hip
    kernels/wmma_i8_gemv.hip
    kernels/fp8_wmma_gemv.hip
    src/zaya_moe_launcher.hip
    src/mamba1_engine.hip
    src/bonsai_q1_1024.hip
    src/bonsai_tq2_1024.hip
    src/fused_gemv_tq2_1024.hip
    src/bonsai_tq2_1024_opt.hip
    kernels/vl_resize_norm.hip
    kernels/whisper_kernels.hip
    kernels/deepseek_mla.hip
    kernels/vitallm_sparse_kv.hip
    kernels/twla_int4_gemv.hip
    kernels/turboquant_kv.hip
    kernels/tq2_bwopt.hip
    src/laguna_gemv.hip
    src/laguna_fused.hip
    src/laguna_ops.hip
    src/mamba2_kernels.cpp
    src/mamba2_kernels.hip
    src/zamba2_engine.cpp
    src/zamba2_engine_hip.hip)
if(RCPP_HAVE_CK)
    list(APPEND RCPP_SOURCES src/ck_gemm.cpp)
endif()
# When WMMA/dot8 is disabled, provide stub implementations so the
# library still links.  The stubs return errors; callers use scalar fallbacks.
if(NOT _HAS_WMMA_ARCH)
    list(APPEND RCPP_SOURCES src/wmma_stubs.hip)
endif()

# NPU backend (worker subprocess) + Q4NX reader — compiled into shared lib
list(APPEND RCPP_SOURCES src/backend_npu.cpp)
list(APPEND RCPP_SOURCES src/q4nx_reader.cpp)

# -- libgguf_reader : single shared GGUF parser (no HIP dependency) -----------
# Split into its own tiny static lib (rather than compiled directly into both
# backend_manager and rocm_cpp) so targets that link both don't hit duplicate-
# symbol errors.
add_library(gguf_reader STATIC src/gguf_reader.cpp)
target_include_directories(gguf_reader PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# -- gguf_to_onebp : pure-C++ GGUF -> 1BP converter (Q4NX + TQ2 ternary) -------
# Replaces the old tools/gguf_to_onebp.py. No Python anywhere in the convert
# path. `--tq2` emits symmetric 2-bit ternary (half the size of Q4NX).
add_executable(gguf_to_onebp tools/gguf_to_onebp.cpp)
target_link_libraries(gguf_to_onebp PRIVATE gguf_reader)
target_compile_options(gguf_to_onebp PRIVATE -O3 -std=c++23)

# -- libvl_image : lightweight image loading + preprocessing (no HIP dep) ----
# Uses stb_image.h (vendored at third_party/stb/) for decode. No OpenCV dep.
# Provides: vl_load_image, vl_resize_normalize, vl_decode_base64_image, etc.
add_library(vl_image STATIC src/vl_processor.cpp)
target_include_directories(vl_image PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(vl_image PRIVATE -O3 -std=c++23 -Wno-unused-result)
# vl_processor.cpp uses cpp-httplib as its HTTP client (SSRF-safe downloads,
# issues #1278/#1291) — header-only, propagated to all consumers.
target_link_libraries(vl_image PUBLIC httplib::httplib)


# -- libonebp_model : 1BP model loader (used by vision_encoder and other targets) --
add_library(onebp_model STATIC src/onebp_model.cpp)
target_include_directories(onebp_model PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(onebp_model PRIVATE -O3 -std=c++23 -Wno-unused-result)

# -- libvision_encoder : ViT encoder + vision-text projector (CPU, no HIP dep) --
add_library(vision_encoder STATIC src/vision_encoder.cpp)
target_include_directories(vision_encoder PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(vision_encoder PRIVATE -O3 -march=native -std=c++23 -Wno-unused-result)
target_link_libraries(vision_encoder PUBLIC vl_image backend_manager onebp_model)

# -- libwhisper : Speech-to-text (Whisper) ----------------------------------------------
add_library(whisper STATIC src/whisper.cpp)
target_include_directories(whisper PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(whisper PRIVATE -O3 -march=native -std=c++23 -Wno-unused-result)
target_link_libraries(whisper PUBLIC backend_manager)

# -- libpixtral : Pixtral (Mistral) vision-language connector --------------------------
add_library(pixtral STATIC src/pixtral.cpp)
target_include_directories(pixtral PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(pixtral PRIVATE -O3 -march=native -std=c++23 -Wno-unused-result)
target_link_libraries(pixtral PUBLIC vision_encoder backend_manager vl_image)


add_library(rocm_cpp SHARED ${RCPP_SOURCES})
if(ROCWMMA_FOUND)
    target_include_directories(rocm_cpp SYSTEM PRIVATE "${ROCWMMA_INCLUDE_DIR}")
endif()
target_sources(rocm_cpp PRIVATE src/prefill_dispatcher.cpp)

# -- zinc_cpp : C++ ZINC GPU backend (Vulkan compute inference)
#    Replaces the Zig-based ZINC project (engine/gpu/) which has been
#    fully reverse-engineered and ported to C++. Builds from the
#    engine/gpu/zinc_cpp/ subdirectory.
add_subdirectory(engine/gpu/zinc_cpp zinc_cpp_build)
set(ZINC_CPP_AVAILABLE TRUE)

# -- llama.cpp ggml-vulkan integration (MIT License) -----------------------------------
set(GGML_VULKAN_DIR "${CMAKE_SOURCE_DIR}/third_party/llama.cpp")
set(GGML_BUILD_DIR "${GGML_VULKAN_DIR}/build")

if(EXISTS "${GGML_BUILD_DIR}/ggml/src/ggml-vulkan/libggml-vulkan.a")
    add_library(ggml_vulkan STATIC IMPORTED GLOBAL)
    set_target_properties(ggml_vulkan PROPERTIES IMPORTED_LOCATION "${GGML_BUILD_DIR}/ggml/src/ggml-vulkan/libggml-vulkan.a")
    add_library(ggml STATIC IMPORTED GLOBAL)
    set_target_properties(ggml PROPERTIES IMPORTED_LOCATION "${GGML_BUILD_DIR}/ggml/src/libggml.a")
    add_library(ggml_cpu STATIC IMPORTED GLOBAL)
    set_target_properties(ggml_cpu PROPERTIES IMPORTED_LOCATION "${GGML_BUILD_DIR}/ggml/src/libggml-cpu.a")
    add_library(ggml_base STATIC IMPORTED GLOBAL)
    set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${GGML_BUILD_DIR}/ggml/src/libggml-base.a")
    target_include_directories(ggml INTERFACE "${GGML_VULKAN_DIR}/ggml/include")
    message(STATUS "llama.cpp ggml-vulkan: FOUND")
else()
    message(WARNING "llama.cpp ggml-vulkan not built — cd third_party/llama.cpp && cmake -B build -G Ninja -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF && cmake --build build -j8")
endif()

# -- libbackend_manager : Backend abstraction layer (requires rocm_cpp for HIP) ---------
set(BACKEND_MANAGER_SOURCES
    src/backend_factory.cpp
    src/backend_manager.cpp
    src/backend_monitor.cpp
    src/backend_plugin.cpp
    src/backend_cpu.cpp
    src/backend_generic.cpp
    src/backend_mamba1.cpp
    src/deepseek.cpp
    src/backend_laguna.cpp
    src/backend_zamba2.cpp
    src/simple_tokenizer.cpp
    src/q4nx_reader.cpp
    src/safetensors_reader.cpp
    src/model_discovery.cpp
    src/model_router.cpp
    src/dynamic_router.cpp)

if(RCPP_HAVE_CUDA)
    list(APPEND BACKEND_MANAGER_SOURCES src/backend_cuda.cpp)
endif()

if(RCPP_HAVE_METAL)
    list(APPEND BACKEND_MANAGER_SOURCES src/backend_metal.mm)
endif()

add_library(backend_manager STATIC ${BACKEND_MANAGER_SOURCES})
target_link_libraries(backend_manager PUBLIC gguf_reader)

# OpenMP: parallelizes the generic CPU decode matmul (src/backend_generic.cpp).
# Optional — the code still compiles/runs single-threaded without it.
#
# CMake's own find_library() probe for OpenMP_omp_LIBRARY is unreliable in
# this environment: with ROCM_ROOT prepended onto CMAKE_PREFIX_PATH, it either
# (a) caches a nonexistent "<arch-triple>" subdirectory guess under the SDK's
# llvm lib dir that this TheRock layout doesn't ship, which then survives
# stale across SDK reinstalls/version bumps (TheRock is a nightly package) and
# silently breaks every OpenMP-linked target until someone manually purges the
# cache, or (b) fails to find OpenMP at all once that stale entry is cleared.
# Reproduced and confirmed 2026-07-25 (see PR that added this comment).
# Fix: locate libomp.so ourselves by globbing the compiler's own install tree
# (layout-agnostic — no hardcoded python/package version) plus common system
# dirs, seed the cache with a verified-existing path, then let find_package()
# validate/use it instead of trusting its own prefix-path-driven guess.
if(DEFINED OpenMP_omp_LIBRARY AND NOT EXISTS "${OpenMP_omp_LIBRARY}")
    message(STATUS "Stale OpenMP_omp_LIBRARY (${OpenMP_omp_LIBRARY}) does not exist — re-detecting OpenMP")
    unset(OpenMP_omp_LIBRARY CACHE)
    unset(OpenMP_CXX_FOUND CACHE)
    unset(OpenMP_C_FOUND CACHE)
    unset(OpenMP_FOUND CACHE)
endif()
if(NOT DEFINED OpenMP_omp_LIBRARY OR NOT EXISTS "${OpenMP_omp_LIBRARY}")
    file(GLOB_RECURSE _RC_LIBOMP_CANDIDATES
        "${ROCM_ROOT}/lib/*/libomp.so"
        "/usr/lib/*/libomp.so"
        "/usr/lib/llvm-*/lib/libomp.so")
    if(_RC_LIBOMP_CANDIDATES)
        list(GET _RC_LIBOMP_CANDIDATES 0 _RC_LIBOMP)
        set(OpenMP_omp_LIBRARY "${_RC_LIBOMP}" CACHE FILEPATH "Path to the omp library for OpenMP" FORCE)
        message(STATUS "rocm-cpp: pre-seeding OpenMP_omp_LIBRARY=${_RC_LIBOMP}")
    endif()
    unset(_RC_LIBOMP_CANDIDATES)
endif()
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(backend_manager PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "backend_manager: OpenMP found — generic CPU decode is multi-threaded")
else()
    message(STATUS "backend_manager: OpenMP NOT found — generic CPU decode stays single-threaded")
endif()

# -- ppl_generic : perplexity gate for the generic CPU backend (WS-00/WS-05) --
# Quality harness for 1BP models (GenericBackend::compute_ppl). CI's ppl-gate
# job runs it on a fresh gguf_to_onebp conversion and asserts the window —
# catches catastrophic quant corruption in 30 s (measured on the 48-sample
# gate set: Q4NX-from-Q8_0 ppl 16.34 vs TQ2-of-dense ppl 1.1e8).
add_executable(ppl_generic
    research/ws00-baseline/ppl_generic.cpp
    src/model_discovery.cpp
    src/gguf_reader.cpp
    src/q4nx_reader.cpp
    src/safetensors_reader.cpp)
target_include_directories(ppl_generic PRIVATE src include engine/npu/src)
target_compile_options(ppl_generic PRIVATE -O3 -march=native -std=c++23)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ppl_generic PRIVATE OpenMP::OpenMP_CXX)
endif()

# -- gguf_htok / make_gate : per-vocab ppl gate sets (issue #1243) --
# gguf_htok exports a GGUF tokenizer to .htok v2; make_gate tokenizes the
# gate corpus with it into ppl_generic's jsonl format.
add_executable(gguf_htok tools/gguf_htok.cpp src/gguf_reader.cpp)
target_include_directories(gguf_htok PRIVATE src include)
target_compile_options(gguf_htok PRIVATE -O2 -std=c++23)
add_executable(make_gate tools/make_gate.cpp src/tokenizer.cpp)
target_include_directories(make_gate PRIVATE src include)
target_compile_options(make_gate PRIVATE -O2 -std=c++23)

target_include_directories(backend_manager PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_include_directories(backend_manager PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_include_directories(backend_manager PRIVATE "${ROCM_ROOT}/include")
target_compile_definitions(backend_manager PRIVATE __HIP_PLATFORM_AMD__=1)
target_link_libraries(backend_manager PUBLIC dl rocm_cpp hip::host hip::device)
if(RCPP_HAVE_CUDA)
    target_link_libraries(backend_manager PUBLIC cuda_engine)
endif()
if(RCPP_HAVE_METAL)
    target_link_libraries(backend_manager PRIVATE ${METAL_LIBRARY} ${METAL_PERFORMANCE_SHADERS_LIBRARY})
    target_compile_options(backend_manager PRIVATE -fobjc-arc)
endif()
target_compile_options(backend_manager PRIVATE -O3 -march=native -Wall -Wno-unused-result -std=c++23)
# NOTE: no -ffast-math here (deliberately) — it implies -ffinite-math-only and
# reassociation; it made the generic backend's expf-based SiLU produce NaN for
# large SwiGLU gates (issue #1243 re-conversion gate: Qwen3-8B ppl 58k vs 7.35
# without it). Same class of bug as #825/#1277.

# Try to find libzinc.so for the real GGUF tokenizer C ABI.
# If not found, ZINC_DISABLED remains set and the C++ stubs in
# simple_tokenizer.cpp provide the fallback (issue #798).
find_library(ZINC_LIB zinc PATHS /home/bcloud/zinc/zig-out/lib /opt/zinc/lib /usr/local/lib /usr/lib)
if(ZINC_LIB)
    message(STATUS "Found libzinc.so: ${ZINC_LIB} — using real GGUF tokenizer")
    target_link_libraries(backend_manager PRIVATE ${ZINC_LIB})
    # ZINC_DISABLED is NOT set — simple_tokenizer.cpp uses real extern "C" calls
else()
    message(STATUS "libzinc.so not found — using C++ tokenizer stubs (ZINC_DISABLED)")
    target_compile_definitions(backend_manager PRIVATE ZINC_DISABLED=1)
endif()

if(ZINC_CPP_AVAILABLE)
    target_sources(backend_manager PRIVATE src/backend_zinc.cpp)
    target_link_libraries(backend_manager PUBLIC zinc_cpp)
    target_include_directories(backend_manager PRIVATE engine/gpu/zinc_cpp/include)
endif()

set(RCPP_HIP_SOURCES ${RCPP_SOURCES} src/prefill_dispatcher.cpp)
list(FILTER RCPP_HIP_SOURCES INCLUDE REGEX "\\.hip$|prefill_dispatcher\\.cpp$")
if(RCPP_HAVE_CK)
    list(APPEND RCPP_HIP_SOURCES src/ck_gemm.cpp)
endif()
set_source_files_properties(${RCPP_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
target_include_directories(rocm_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_include_directories(rocm_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
target_include_directories(rocm_cpp SYSTEM PRIVATE "${ROCM_ROOT}/include")
target_compile_definitions(rocm_cpp PRIVATE MAMBA1_ENGINE_AS_LIB=1)
target_compile_options(rocm_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math -munsafe-fp-atomics
        -Wno-unused-result>)
target_link_libraries(rocm_cpp PRIVATE hip::host hip::device gguf_reader)
set_target_properties(rocm_cpp PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

if(RCPP_HAVE_CK)
    add_executable(test_ck_gemm tests/test_ck_gemm.cpp)
    set_source_files_properties(tests/test_ck_gemm.cpp PROPERTIES LANGUAGE HIP)
    target_compile_options(test_ck_gemm PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
    target_link_libraries(test_ck_gemm PRIVATE rocm_cpp hip::host hip::device)
    set_target_properties(test_ck_gemm PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()

if(RCPP_HAVE_CK)
    add_executable(test_standalone tests/test_standalone.cpp)
    set_source_files_properties(tests/test_standalone.cpp PROPERTIES LANGUAGE HIP)
    target_compile_options(test_standalone PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
    target_link_libraries(test_standalone PRIVATE rocm_cpp hip::host hip::device)
    set_target_properties(test_standalone PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()


function(rocm_hip_target TARGET)
    add_executable(${TARGET} ${ARGN})
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
    target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
    target_link_libraries(${TARGET} PRIVATE rocm_cpp hip::host hip::device)
    set_target_properties(${TARGET} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endfunction()

rocm_hip_target(test_standalone_nock tests/test_standalone_nock.cpp)
rocm_hip_target(test_prim_and_attn tests/test_prim_and_attn.cpp)
rocm_hip_target(test_ternary_gemm_smallm tests/test_ternary_gemm_smallm.cpp)
rocm_hip_target(test_sherry_gemv tests/test_sherry_gemv.cpp)
rocm_hip_target(test_bonsai_gemv tests/test_bonsai_gemv.cpp)
rocm_hip_target(bench_bonsai_tile8 tests/bench_bonsai_tile8.cpp)
rocm_hip_target(bench_bonsai_tile8_model tests/bench_bonsai_tile8_model.cpp)
rocm_hip_target(bench_bonsai_q1_1024 tests/bench_bonsai_q1_1024.cpp)
rocm_hip_target(bench_bonsai_tq2_1024 tests/bench_bonsai_tq2_1024.cpp)
rocm_hip_target(bench_fused_tq2_1024 tests/bench_fused_tq2_1024.cpp)

add_executable(bench_cb_compare tests/bench_cb_compare.cpp)
set_source_files_properties(tests/bench_cb_compare.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_cb_compare PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_cb_compare PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_cb_compare PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

if(RCPP_HAVE_VULKAN)
    set(VK_SHADER_DIR "${CMAKE_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${VK_SHADER_DIR}")
    set(VK_SHADER_SOURCES kernels/vulkan/dmmv_tq2_bonsai.comp kernels/vulkan/dmmv_q1_bonsai.comp)
    set(VK_SHADER_SPVS "")
    foreach(_vk_src ${VK_SHADER_SOURCES})
        get_filename_component(_vk_name "${_vk_src}" NAME_WE)
        set(_vk_spv "${VK_SHADER_DIR}/${_vk_name}.spv")
        add_custom_command(OUTPUT "${_vk_spv}"
            COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -o "${_vk_spv}" "${CMAKE_SOURCE_DIR}/${_vk_src}"
            DEPENDS "${CMAKE_SOURCE_DIR}/${_vk_src}"
            COMMENT "glslc ${_vk_src} -> ${_vk_name}.spv")
        list(APPEND VK_SHADER_SPVS "${_vk_spv}")
    endforeach()
    add_custom_target(vulkan_shaders DEPENDS ${VK_SHADER_SPVS})
    add_executable(test_vulkan_gemv tests/test_vulkan_gemv.cpp)
    target_link_libraries(test_vulkan_gemv PRIVATE Vulkan::Vulkan)
    target_compile_definitions(test_vulkan_gemv PRIVATE VK_SHADER_DIR="${VK_SHADER_DIR}")
    add_dependencies(test_vulkan_gemv vulkan_shaders)

    # video-lora: pure-C++ Vulkan inference backend (SD1.5/AnimateDiff ops),
    # built into the single binary. Compiles its own shaders via glslc.
    # Linked into zaya_server after its definition below.
    add_subdirectory(tools/video-lora/vulkan)
    set(VIDEO_LORA_VK_AVAILABLE TRUE)
endif()

rocm_hip_target(test_sherry_e2e tests/test_sherry_e2e.cpp)
rocm_hip_target(test_zaya_moe_gemv tests/test_zaya_moe_gemv.cpp)
rocm_hip_target(test_block_scaled_ternary tests/test_block_scaled_ternary.cpp)

add_executable(test_cca_attn tests/test_cca_attn.cpp kernels/zaya_cca_attn.hip)
set_source_files_properties(tests/test_cca_attn.cpp kernels/zaya_cca_attn.hip PROPERTIES LANGUAGE HIP)
target_compile_options(test_cca_attn PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_cca_attn PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_cca_attn PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

rocm_hip_target(zaya_full tests/zaya_full.cpp)

# -- zaya_gpu_decode : GPU decode benchmarking tool (not a test) -------------
add_executable(zaya_gpu_decode tests/zaya_gpu_decode.cpp kernels/zaya_cca_attn.hip)
set_source_files_properties(tests/zaya_gpu_decode.cpp kernels/zaya_cca_attn.hip PROPERTIES LANGUAGE HIP)
target_compile_options(zaya_gpu_decode PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_include_directories(zaya_gpu_decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(zaya_gpu_decode PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(zaya_gpu_decode PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(test_lora_merge tests/test_lora_merge.cpp src/zaya_engine.cpp)
set_source_files_properties(tests/test_lora_merge.cpp src/zaya_engine.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(test_lora_merge PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/kernels)
target_compile_options(test_lora_merge PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_lora_merge PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_lora_merge PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- zaya_server : multi-model multi-backend inference server (main binary,
#    NOT a test — not registered with add_test(), see install() below) ------
set(ZAYA_SERVER_SRC tests/zaya_server.cpp tests/backends/backend_cpu.cpp tests/backends/backend_npu.cpp tests/backends/backend_generic.cpp tests/backends/backend_zamba2_adapter.cpp src/backend_zamba2.cpp)
if(RCPP_HAVE_VULKAN)
    list(APPEND ZAYA_SERVER_SRC tests/backends/backend_vulkan.cpp)
endif()
if(ZINC_CPP_AVAILABLE)
    list(APPEND ZAYA_SERVER_SRC tests/backends/backend_zinc.cpp)
endif()
# HIP backend: adapter (InferenceBackend) + canonical src (Backend + zaya_engine)
list(APPEND ZAYA_SERVER_SRC tests/backends/backend_hip_adapter.cpp src/backend_hip.cpp src/zaya_engine.cpp)
set_source_files_properties(tests/zaya_server.cpp tests/backends/backend_hip_adapter.cpp src/backend_hip.cpp src/zaya_engine.cpp PROPERTIES LANGUAGE HIP)
# zaya_server is no longer a standalone add_executable target — its full
# source list (ZAYA_SERVER_SRC, kept above) is compiled into onebin/1bit
# only; run it via `1bit zaya` (see tools/onebin.cpp). No second binary,
# no symlink — the standalone target was pure duplication of onebin.

rocm_hip_target(test_wmma_i8_gemv tests/test_wmma_i8_gemv.cpp)
rocm_hip_target(test_bonsai_e2e tests/test_bonsai_e2e.cpp)

add_executable(test_medusa_skeleton tests/test_medusa_skeleton.cpp)
target_compile_options(test_medusa_skeleton PRIVATE -Wall -Wextra -Wno-unused-result)
target_link_libraries(test_medusa_skeleton PRIVATE rocm_cpp)

# -- test_backend: CPU-only backend validation (issue #347) --
# Previously only had a broken standalone build command that couldn't link
# due to transitive HIP deps through backend_factory → backend_generic → gguf_loader.
# Now wired as a proper ninja target linking backend_manager (which already handles
# the rocm_cpp dependency). Run without a model to see usage.
# -- bench_hip_1bp: GPU 1BP engine standalone benchmark --
add_executable(bench_hip_1bp tools/bench_hip_1bp.cpp src/backend_hip_1bp.cpp src/kv_cache_attn.hip)
target_include_directories(bench_hip_1bp PRIVATE src include engine/fusion/zero_copy engine/npu/src)
target_compile_options(bench_hip_1bp PRIVATE -O2 -std=c++23 -D__HIP_PLATFORM_AMD__ -DRCPP_HAVE_FUSION_ZERO_COPY)
target_link_libraries(bench_hip_1bp PRIVATE hip::host hip::device hipblas rocm_cpp)  # keep rocm_cpp; kernel-name collision fixed by h1bp_ prefix (issue #1212)
if(THEROCK_HIPBLAS_LIB)
    get_filename_component(_l "${THEROCK_HIPBLAS_LIB}" DIRECTORY)
    set_property(TARGET bench_hip_1bp APPEND PROPERTY LINK_OPTIONS "-Wl,-rpath,${_l}")
endif()
set_source_files_properties(tools/bench_hip_1bp.cpp PROPERTIES LANGUAGE CXX)
set_target_properties(bench_hip_1bp PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

add_executable(test_backend tests/test_backend.cpp)
target_include_directories(test_backend PRIVATE src)
target_compile_options(test_backend PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(test_backend PRIVATE backend_manager)

set(BONSAI_H1B_MODEL "" CACHE FILEPATH "Path to a Bonsai .h1b model for test_bonsai_e2e (empty = skip)")
set(SHERRY_H1B_MODEL "" CACHE FILEPATH "Path to a Sherry .h1b model for test_sherry_e2e (empty = skip)")

add_test(NAME test_medusa_skeleton         COMMAND test_medusa_skeleton)
set_tests_properties(test_medusa_skeleton   PROPERTIES LABELS "host")

# -- test_gguf_reader_dequant: no GPU, no downloaded model, no Python at ------
# test time. Byte-exact dequant check for every dtype gguf_reader.cpp
# supports, against reference values baked in from the Python `gguf`
# package. Runs unconditionally in every CI invocation.
add_executable(test_gguf_reader_dequant tests/test_gguf_reader_dequant.cpp)
target_include_directories(test_gguf_reader_dequant PRIVATE tests/)
target_link_libraries(test_gguf_reader_dequant PRIVATE gguf_reader)
add_test(NAME test_gguf_reader_dequant COMMAND test_gguf_reader_dequant)
set_tests_properties(test_gguf_reader_dequant PROPERTIES LABELS "host")

# -- test_tokenizer_logprobs: tokenizer logprob API (#81) -- restored after --
# being dropped in the dead-code purge (issue #1341). The API is live
# (unified_server prompt-logprob path: encode_with_logprobs) so it needs
# coverage. No GPU, no model download; builds tokenizer.cpp standalone.
add_executable(test_tokenizer_logprobs tests/test_tokenizer_logprobs.cpp src/tokenizer.cpp)
target_include_directories(test_tokenizer_logprobs PRIVATE include include/rocm_cpp)
target_compile_options(test_tokenizer_logprobs PRIVATE -O2 -std=c++17 -Wall)
target_link_libraries(test_tokenizer_logprobs PRIVATE m)
add_test(NAME test_tokenizer_logprobs COMMAND test_tokenizer_logprobs)
set_tests_properties(test_tokenizer_logprobs PROPERTIES LABELS "host")

add_test(NAME test_backend                 COMMAND test_backend)
set_tests_properties(test_backend           PROPERTIES LABELS "host" SKIP_RETURN_CODE 77)

# test_backend needs weights at /tmp/zaya_weights or the passed path
# Skips gracefully (exit 77) when weights aren't found.

add_test(NAME test_standalone_nock         COMMAND test_standalone_nock)
set_tests_properties(test_standalone_nock   PROPERTIES LABELS "gpu_required" SKIP_RETURN_CODE 77)
add_test(NAME test_prim_and_attn           COMMAND test_prim_and_attn)
set_tests_properties(test_prim_and_attn     PROPERTIES LABELS "gpu_required" SKIP_RETURN_CODE 77)
add_test(NAME test_ternary_gemm_smallm     COMMAND test_ternary_gemm_smallm)
set_tests_properties(test_ternary_gemm_smallm PROPERTIES LABELS "gpu_required" SKIP_RETURN_CODE 77)
add_test(NAME test_sherry_gemv             COMMAND test_sherry_gemv)
set_tests_properties(test_sherry_gemv       PROPERTIES LABELS "gpu_required" SKIP_RETURN_CODE 77)
add_test(NAME test_bonsai_gemv             COMMAND test_bonsai_gemv)
set_tests_properties(test_bonsai_gemv       PROPERTIES LABELS "gpu_required" SKIP_RETURN_CODE 77)
add_test(NAME test_zaya_moe_gemv           COMMAND test_zaya_moe_gemv)
set_tests_properties(test_zaya_moe_gemv     PROPERTIES LABELS "gpu_required" SKIP_RETURN_CODE 77)
add_test(NAME test_block_scaled_ternary    COMMAND test_block_scaled_ternary)
set_tests_properties(test_block_scaled_ternary PROPERTIES LABELS "gpu_required" SKIP_RETURN_CODE 77)
add_test(NAME test_cca_attn                COMMAND test_cca_attn)
set_tests_properties(test_cca_attn          PROPERTIES LABELS "gpu_required" SKIP_RETURN_CODE 77)
add_test(NAME test_wmma_i8_gemv            COMMAND test_wmma_i8_gemv)
set_tests_properties(test_wmma_i8_gemv     PROPERTIES LABELS "gpu_required" SKIP_RETURN_CODE 77)

add_test(NAME test_bonsai_e2e              COMMAND test_bonsai_e2e ${BONSAI_H1B_MODEL})
set_tests_properties(test_bonsai_e2e       PROPERTIES LABELS "gpu_required;model_required;e2e" SKIP_RETURN_CODE 77)
add_test(NAME test_sherry_e2e              COMMAND test_sherry_e2e ${SHERRY_H1B_MODEL})
set_tests_properties(test_sherry_e2e       PROPERTIES LABELS "gpu_required;model_required;e2e" SKIP_RETURN_CODE 77)

# -- test_generic_attention: regression guard for the generic (non-Zaya)
# attention path (fixed in #474/#486). Uses the generic CPU backend, no
# HIP dependency. Renamed from test_hip_generic_attention after the HIP
# backend was consolidated (Zaya only — generic GGUF models go through
# backend_generic).
set(GGUF_TEST_MODEL "" CACHE FILEPATH "Path to a non-Zaya GGUF model for test_generic_attention (empty = skip)")
add_executable(test_generic_attention tests/test_generic_attention.cpp tests/backends/backend_generic.cpp)
target_compile_options(test_generic_attention PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_include_directories(test_generic_attention PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/backends
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_generic_attention PRIVATE backend_manager)
add_test(NAME test_generic_attention COMMAND test_generic_attention ${GGUF_TEST_MODEL})
set_tests_properties(test_generic_attention PROPERTIES LABELS "model_required;e2e" SKIP_RETURN_CODE 77)

add_executable(bitnet_decode tools/bitnet_decode.cpp)
set_source_files_properties(tools/bitnet_decode.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bitnet_decode PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bitnet_decode PRIVATE rocm_cpp hip::host hip::device httplib::httplib nlohmann_json::nlohmann_json)
set_target_properties(bitnet_decode PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(hadamard_export tools/hadamard_export.cpp)
target_compile_options(hadamard_export PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(hadamard_export PRIVATE rocm_cpp)

add_executable(bitnet_debate tools/bitnet_debate.cpp)
set_source_files_properties(tools/bitnet_debate.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bitnet_debate PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bitnet_debate PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bitnet_debate PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_sherry tools/bench_sherry.cpp)
set_source_files_properties(tools/bench_sherry.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_sherry PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_sherry PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_sherry PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_mamba2_kernels tools/bench_mamba2_kernels.cpp)
set_source_files_properties(tools/bench_mamba2_kernels.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_mamba2_kernels PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_mamba2_kernels PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_mamba2_kernels PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_zamba2_e2e tools/bench_zamba2_e2e.cpp)
set_source_files_properties(tools/bench_zamba2_e2e.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_zamba2_e2e PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_include_directories(bench_zamba2_e2e PRIVATE src/)
target_link_libraries(bench_zamba2_e2e PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_zamba2_e2e PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_kv_i8 tools/bench_kv_i8.cpp)
set_source_files_properties(tools/bench_kv_i8.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_kv_i8 PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_kv_i8 PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_kv_i8 PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_kv_fd tools/bench_kv_fd.cpp)
set_source_files_properties(tools/bench_kv_fd.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_kv_fd PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_kv_fd PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_kv_fd PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_rotor tools/bench_rotor.cpp)
set_source_files_properties(tools/bench_rotor.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_rotor PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_rotor PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_rotor PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_prefill_variants tools/bench_prefill_variants.cpp)
set_source_files_properties(tools/bench_prefill_variants.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_prefill_variants PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math -munsafe-fp-atomics
        -Wno-unused-result>)
target_link_libraries(bench_prefill_variants PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_prefill_variants PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# unified_server sources — CUDA and Metal come through backend_manager
# (PR #579 pattern), not listed here. model_router.cpp now lives in
# BACKEND_MANAGER_SOURCES (backend_manager's benchmark_all() calls
# select_backend_route(), so every backend_manager consumer needs it, not
# just unified_server — see the CMake OpenMP-cache fix commit for the sibling
# lesson about consumer targets silently missing a symbol).
set(UNIFIED_SERVER_SOURCES
    tools/unified_server.cpp
    src/strategy_engine.cpp
    src/agent_watchdog.cpp
    src/backend_hip.cpp
    src/backend_hip_1bp.cpp
    src/unified_pool.cpp
    src/backend_vulkan_hpp.cpp
    src/zaya_engine.cpp
    src/backend_npu.cpp
    src/backend_npu_flm.cpp
    src/backend_generic.cpp
)

# ggml_vulkan backend needs llama.cpp's static libs (GGML_BUILD_DIR); only
# compile it when they exist — the factory is dlsym-dispatched, so an absent
# backend just registers unavailable at runtime.
if(TARGET ggml_vulkan)
    list(APPEND UNIFIED_SERVER_SOURCES src/backend_ggml_vulkan.cpp)
endif()

# Fused GPU+NPU backend (attention on GPU ∥ FFN on NPU, zero-copy SharedBO)
# needs the XRT + AIEBU dev headers at compile time. CI images ship the
# XRT/AIEBU cmake configs but NOT the headers (apt libxrt-dev is a different,
# older XRT without xrt_bo.h), so gate on an actual header compile test.
# The factories are dlsym-dispatched (backend_manager.cpp), so an absent
# backend just registers as unavailable at runtime — safe to exclude.
find_package(xrt QUIET)
find_package(aiebu QUIET)
include(CheckIncludeFile)
check_include_file("xrt/xrt_bo.h" _RC_XRT_BO_H)
set(_RC_HAVE_FUSED_BACKEND FALSE)
if(hip_FOUND AND xrt_FOUND AND aiebu_FOUND AND _RC_XRT_BO_H)
    set(_RC_HAVE_FUSED_BACKEND TRUE)
    list(APPEND UNIFIED_SERVER_SOURCES src/backend_fused.cpp src/backend_fused_npu.cpp)
    message(STATUS "rocm-cpp: fused GPU+NPU backend enabled (XRT + AIEBU headers present)")
else()
    message(STATUS "rocm-cpp: fused GPU+NPU backend disabled (XRT headers or AIEBU missing)")
endif()

# unified_server sources (UNIFIED_SERVER_SOURCES, kept above) compile into
# onebin/1bit only — run via `1bit unified` (see tools/onebin.cpp). No
# standalone add_executable here: it was pure duplication of onebin, the
# same problem the zaya_server split had.
# CUDA/Metal come through backend_manager (PR #579 pattern), so they don't
# appear in sources or link line here.
set_source_files_properties(src/backend_hip.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/backend_hip_1bp.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/zaya_engine.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/backend_generic.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(tools/unified_server.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(src/backend_npu.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(src/backend_npu_flm.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(src/unified_pool.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(src/backend_fused_npu.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(src/backend_fused.cpp PROPERTIES LANGUAGE HIP)
# FastFlowLM (flm) is now an official AMD/ROCm project, shipped prebuilt:
#   - built into the TheRock dist (7.14 / 7.15a) at <therock-root>/bin/flm
#   - as the fastflowlm .deb (github.com/FastFlowLM/FastFlowLM releases — the
#     same binary Lemonade Server tracks and installs on PATH)
# Prefer a shipped binary; the submodule build below is only a fallback for
# environments without either.
set(FLM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/FastFlowLM/src")
set(FLM_BUILD_DIR "${FLM_SOURCE_DIR}/build")
set(FLM_BINARY "${FLM_BUILD_DIR}/flm")
set(FLM_CONFIG "${FLM_SOURCE_DIR}/model_list.json")
set(FLM_XCLBINS "${FLM_SOURCE_DIR}/xclbins")

foreach(_flm_root IN ITEMS "${_RC_ROCM_DEFAULT}" "/opt/fastflowlm" "/opt/rocm")
    if(_flm_root AND EXISTS "${_flm_root}/bin/flm")
        set(FLM_BINARY "${_flm_root}/bin/flm")
        if(EXISTS "${_flm_root}/etc/flm/model_list.json")
            set(FLM_CONFIG "${_flm_root}/etc/flm/model_list.json")
        endif()
        if(IS_DIRECTORY "${_flm_root}/share/flm/xclbins")
            set(FLM_XCLBINS "${_flm_root}/share/flm/xclbins")
        endif()
        message(STATUS "flm: using ${FLM_BINARY} (TheRock / FastFlowLM package)")
        break()
    endif()
endforeach()

# PATH lookup — the fastflowlm .deb symlink (/usr/bin/flm) and lemonade-
# managed installs resolve here, matching Lemonade's own find_flm_in_path().
if(NOT FLM_BINARY AND EXISTS "/usr/bin/flm")
    set(FLM_BINARY "/usr/bin/flm")
    if(EXISTS "/opt/fastflowlm/etc/flm/model_list.json")
        set(FLM_CONFIG "/opt/fastflowlm/etc/flm/model_list.json")
    endif()
    if(IS_DIRECTORY "/opt/fastflowlm/share/flm/xclbins")
        set(FLM_XCLBINS "/opt/fastflowlm/share/flm/xclbins")
    endif()
    message(STATUS "flm: using ${FLM_BINARY} (on PATH)")
endif()

if(NOT EXISTS "${FLM_BINARY}")
    message(STATUS "flm: no TheRock FLM found — building FastFlowLM from submodule...")
    execute_process(COMMAND ${CMAKE_COMMAND} -B build -G Ninja
        -DFLM_VERSION=0.9.46 -DNPU_VERSION=1.1.2
        WORKING_DIRECTORY ${FLM_SOURCE_DIR}
        OUTPUT_QUIET ERROR_QUIET)
    execute_process(COMMAND ${CMAKE_COMMAND} --build build -j8
        WORKING_DIRECTORY ${FLM_SOURCE_DIR}
        OUTPUT_QUIET ERROR_QUIET)
    if(EXISTS "${FLM_BINARY}")
        message(STATUS "flm: built at ${FLM_BINARY}")
    else()
        message(WARNING "flm: build failed — FLM backend unavailable")
    endif()
endif()

# Add flm as a build dependency so rebuilds happen automatically.
# Guard: only add build_flm target if the binary was successfully built during configure.
# FLM_*_PATH macros are always defined (with configure-time paths) so that
# backend_npu_flm.cpp compiles regardless of FLM availability. The C++ code
# already has a runtime fallback to /opt/rocm/bin/flm.
if(EXISTS "${FLM_BUILD_DIR}/flm")
    add_custom_target(build_flm ALL
        COMMAND ${CMAKE_COMMAND} --build ${FLM_BUILD_DIR} -j8
        WORKING_DIRECTORY ${FLM_SOURCE_DIR}
        COMMENT "Building FastFlowLM NPU engine"
        BYPRODUCTS ${FLM_BINARY}
    )
    # ── hotpatch target: update FLM submodule to latest main + rebuild ──
    # Keeps npu_flm backend on the latest ROCm/FLM source.
    # Run: cmake --build build --target update_flm
    add_custom_target(update_flm
        COMMAND git submodule update --remote third_party/FastFlowLM
        COMMAND cd ${FLM_SOURCE_DIR} && cmake -B build -G Ninja
            -DFLM_VERSION=0.9.46 -DNPU_VERSION=1.1.2
        COMMAND cd ${FLM_SOURCE_DIR} && cmake --build build -j8
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Updating FLM submodule to latest + rebuilding"
        # no BYPRODUCTS: build_flm already declares ${FLM_BINARY}, and a
        # second rule for the same output breaks the Ninja generator (issue #1284)
    )
else()
    message(STATUS "flm: FLM comes from TheRock or is unavailable — skipping submodule build targets")
endif()

# ── Lemonade SDK embed ─────────────────────────────────────────────────────
# Lemonade's server core (vendored from github.com/lemonade-sdk/lemonade, see
# third_party/lemonade/UPSTREAM.md): all 14 Lemonade backends (llamacpp, flm,
# whispercpp, sd-cpp, kokoro, ryzenai-llm, vllm, ...) and its policy-based
# Router, embedded in-process — `1bit unified --lemonade` hands off to the
# full Lemonade server; unified_router uses the same routing engine for
# NPU/GPU decisions. onebin links lemonade-server-core itself, further down.
option(EMBED_LEMONADE "Embed the Lemonade SDK server core" ON)
if(EMBED_LEMONADE)
    set(BUILD_WEB_APP OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/lemonade EXCLUDE_FROM_ALL)
endif()

# ── Gaia C++ agent framework (AMD official) ────────────────────────────────
# Vendored from amd/gaia v0.22.0 cpp/ tree (third_party/gaia/UPSTREAM.md):
# gaia::gaia_core + gaia-bash coding agent. Replaces the hand-rolled agent
# layer (onebit/jarvis) — the agent talks to the engine through the
# Lemonade-compatible endpoint (`1bit unified --lemonade serve`, :13305).
# TUI (FTXUI) off: engine has its own console; flip with -DGAIA_BUILD_TUI=ON.
# gaia-bash's own standalone binary is off too — its sources are linked
# into onebin instead (see EMBED_GAIA_CPP block further down); no second
# binary, matching the zaya_server/unified_server fix above.
option(EMBED_GAIA_CPP "Build the vendored Gaia C++ agent framework" ON)
if(EMBED_GAIA_CPP)
    set(GAIA_BUILD_TUI OFF CACHE BOOL "" FORCE)
    set(GAIA_BUILD_BASH_AGENT OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/gaia)
endif()
# unified_server's remaining link/rpath config (backend_manager, hipblas,
# ZINC, TheRock rpath) lives on the onebin target only now — see the onebin
# block further down. No standalone unified_server executable, no
# zaya_agent copy target, no symlinks: the source lists (kept above) are
# compiled into onebin/1bit and nothing else.
# TheRock SDK runtime lib path — find libhipblas.so.3 and add its dir to rpath
# (consumed by onebin's own rpath block further down).
find_library(THEROCK_HIPBLAS_LIB hipblas PATHS
    "${THEROCK_SDK_BASE}/lib"
    "/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib"
    "/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_libraries/lib"
    NO_DEFAULT_PATH)
if(THEROCK_HIPBLAS_LIB)
    get_filename_component(THEROCK_LIB_DIR "${THEROCK_HIPBLAS_LIB}" DIRECTORY)
endif()


add_executable(backend_demo tools/backend_demo.cpp)
target_include_directories(backend_demo PRIVATE src/)
target_compile_options(backend_demo PRIVATE -O3 -march=native -Wall -Wno-unused-result -std=c++23)
target_link_libraries(backend_demo PRIVATE backend_manager dl)

add_executable(test_mamba1_backend tools/test_mamba1_backend.cpp)
target_include_directories(test_mamba1_backend PRIVATE src/ include/)
target_compile_options(test_mamba1_backend PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(test_mamba1_backend PRIVATE backend_manager dl)

add_executable(test_mamba2_kernels tools/test_mamba2_kernels.cpp)
target_include_directories(test_mamba2_kernels PRIVATE src/ include/)
target_compile_options(test_mamba2_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-unused-result -std=c++23>
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
set_source_files_properties(tools/test_mamba2_kernels.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(test_mamba2_kernels PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_mamba2_kernels PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# Register as ctest (requires GGUF model path via env or default)
set(MAMBA1_TEST_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/blackmamba-1.5b.gguf" CACHE FILEPATH "GGUF model for test_mamba1_backend")
if(MAMBA1_TEST_MODEL AND EXISTS "${MAMBA1_TEST_MODEL}")
    add_test(NAME test_mamba1_backend_regression COMMAND test_mamba1_backend ${MAMBA1_TEST_MODEL} 8)
    set_tests_properties(test_mamba1_backend_regression PROPERTIES LABELS "gpu_required;model_required;regression" SKIP_RETURN_CODE 77)
endif()

add_test(NAME test_mamba2_kernels_validation COMMAND test_mamba2_kernels)
set_tests_properties(test_mamba2_kernels_validation PROPERTIES LABELS "gpu_required")

# ── Zamba2 regression test (attention correctness, issue #946) ──
add_executable(test_zamba2_regression tools/test_zamba2_regression.cpp)
target_include_directories(test_zamba2_regression PRIVATE src/ include/)
target_compile_options(test_zamba2_regression PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(test_zamba2_regression PRIVATE backend_manager rocm_cpp)
set(ZAMBA2_TEST_MODEL "models/zamba2-1.2b-instruct-v2-q4_0.gguf" CACHE FILEPATH "GGUF model for test_zamba2_regression")
if(ZAMBA2_TEST_MODEL AND EXISTS "${ZAMBA2_TEST_MODEL}")
    add_test(NAME test_zamba2_regression COMMAND test_zamba2_regression ${ZAMBA2_TEST_MODEL})
    set_tests_properties(test_zamba2_regression PROPERTIES LABELS "model_required;regression" SKIP_RETURN_CODE 77)
endif()

# ── CUDA backend test tool (matches test_mamba1_backend pattern from PR #579) ──
# Links against backend_manager which transitively provides cuda_engine.
if(RCPP_HAVE_CUDA)
    add_executable(test_cuda_backend tools/test_cuda_backend.cpp)
    target_include_directories(test_cuda_backend PRIVATE src/ include/)
    target_compile_options(test_cuda_backend PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
    target_link_libraries(test_cuda_backend PRIVATE backend_manager)
endif()

# ── Metal backend test tool (matches test_mamba1_backend pattern from PR #579) ──
if(RCPP_HAVE_METAL)
    add_executable(test_metal_backend tools/test_metal_backend.cpp)
    target_include_directories(test_metal_backend PRIVATE src/ include/)
    target_compile_options(test_metal_backend PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
    target_link_libraries(test_metal_backend PRIVATE backend_manager)
endif()
if(ZINC_CPP_AVAILABLE)
    target_link_libraries(backend_demo PRIVATE zinc_cpp)
endif()

add_executable(vision_qwen2vl_poc tools/vision_qwen2vl_poc.cpp)
target_include_directories(vision_qwen2vl_poc PRIVATE src/)
target_compile_options(vision_qwen2vl_poc PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(vision_qwen2vl_poc PRIVATE backend_manager vl_image dl)

# -- vision_server : HTTP server for vision-language inference ---------------
# OpenAI-compatible /v1/chat/completions with image_url support.
# Auto-detects GPU (HIP) and falls back to CPU for image preprocessing.
# Requires a VL-capable text model + mmproj GGUF file.
# No standalone add_executable — tools/vision_server.cpp compiles into
# onebin/1bit only; run via `1bit vision`.

# -- vl_logit_check : logit-level 1BP-vs-BF16 decoder validation -----------
add_executable(vl_logit_check tools/vl_logit_check.cpp src/tokenizer.cpp)
target_include_directories(vl_logit_check PRIVATE src/ include/ engine/npu/src)
target_compile_options(vl_logit_check PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(vl_logit_check PRIVATE backend_manager onebp_model dl)

# -- unified_router : NPU+GPU content-aware routing proxy -------------------
# Pure C++ replacement for unified-router.py. No standalone add_executable
# — tools/unified_router.cpp compiles into onebin/1bit only; run via
# `1bit router`. Its EMBED_LEMONADE linkage lives on the onebin target.

# -- onebin : the single binary --------------------------------------------
# Every entry point in one ELF: zaya_server + unified_server + unified_router
# + jarvis_server + vision_server + onebit + onebitd, dispatched by argv[0]
# (legacy symlink names) or subcommand (zaya / unified / router / lemonade /
# jarvis / vision / onebitd / agent CLI). Produces build/1bit.
set(ONE_BIN_SOURCES
    tools/onebin.cpp
    ${ZAYA_SERVER_SRC}
    ${UNIFIED_SERVER_SOURCES}
    tools/unified_router.cpp
    tools/jarvis_server.cpp
    tools/jarvis/rag.cpp
    tools/jarvis/tools.cpp
    tools/jarvis/routing.cpp
    tools/jarvis/planner.cpp
    tools/jarvis/beacon.cpp
    tools/jarvis/tts.cpp
    tools/jarvis/audio_out.cpp
    tools/jarvis/audio_stream.cpp
    tools/jarvis/codec_tts.cpp
    tools/jarvis/persona.cpp
    tools/jarvis/vad.cpp
    tools/jarvis/context.cpp
    tools/jarvis/auth.cpp
    tools/jarvis/usage.cpp
    tools/jarvis/billing.cpp
    src/codec_decoder.cpp
    tools/vision_server.cpp
    tools/onebit.cpp
    tools/onebitd.cpp)
add_executable(onebin ${ONE_BIN_SOURCES})
set_target_properties(onebin PROPERTIES OUTPUT_NAME "1bit")
install(TARGETS onebin RUNTIME DESTINATION bin)
# Language props: union of zaya_server + unified_server
set_source_files_properties(tests/zaya_server.cpp tests/backends/backend_hip_adapter.cpp
    src/backend_hip.cpp src/zaya_engine.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/backend_hip_1bp.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/backend_fused.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/backend_npu.cpp src/backend_npu_flm.cpp
    src/unified_pool.cpp src/backend_fused_npu.cpp src/backend_generic.cpp
    tools/unified_server.cpp tools/unified_router.cpp tools/onebit.cpp
    tools/onebitd.cpp tools/onebin.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(onebin PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/kernels
    ${CMAKE_SOURCE_DIR}/tools
    ${CMAKE_SOURCE_DIR}/engine/npu/src
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    # tests/ dirs last: only tests/*.cpp need the InferenceBackend interface;
    # putting them first shadows src/backend.h for backend_manager.h's
    # quoted "backend.h" include (detect_backends() overload collision).
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/backends)
target_compile_options(onebin PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-unused-result -std=c++23 -fopenmp>)
target_link_libraries(onebin PRIVATE
    backend_manager vl_image vision_encoder whisper dl
    rocm_cpp hip::host hip::device httplib::httplib
    nlohmann_json::nlohmann_json hipblas)
target_link_options(onebin PRIVATE -rdynamic -luuid)
if(TARGET ggml_vulkan)
    target_link_libraries(onebin PRIVATE ggml_vulkan ggml ggml_cpu ggml_base
        "${GGML_BUILD_DIR}/src/libllama.a"
        "${GGML_BUILD_DIR}/common/libllama-common.a"
        "${GGML_BUILD_DIR}/common/libllama-common-base.a"
        -lvulkan -ldl -lpthread -lm)
endif()
if(EMBED_LEMONADE)
    target_link_libraries(onebin PRIVATE lemonade-server-core)
    target_compile_definitions(onebin PRIVATE EMBED_LEMONADE=1)
endif()
# Gaia C++ agent loop (third_party/gaia) folded into the single ELF instead
# of shipping gaia-bash as a second binary — see third_party/gaia/UPSTREAM.md
# local patch #3 for the ONE_BIN_DISPATCH guard on agents/bash/main.cpp.
if(EMBED_GAIA_CPP)
    target_sources(onebin PRIVATE
        third_party/gaia/agents/bash/main.cpp
        third_party/gaia/agents/bash/bash_agent.cpp
        third_party/gaia/agents/bash/bash_tools.cpp
        third_party/gaia/agents/bash/api_server.cpp
        third_party/gaia/agents/bash/mcp_server.cpp)
    target_include_directories(onebin PRIVATE third_party/gaia/agents/bash)
    target_link_libraries(onebin PRIVATE gaia::gaia_core)
    if(OpenSSL_FOUND)
        target_compile_definitions(onebin PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
        target_link_libraries(onebin PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    endif()
endif()
target_compile_definitions(onebin PRIVATE
    ONE_BIN_DISPATCH=1
    ROCM_CPP_STATIC_HIP=1
    ROCM_CPP_STATIC_NPU=1
    FLM_BINARY_PATH="${FLM_BINARY}"
    FLM_CONFIG_PATH="${FLM_CONFIG}"
    FLM_XCLBIN_PATH="${FLM_XCLBINS}")
if(ZINC_CPP_AVAILABLE)
    target_link_libraries(onebin PRIVATE zinc_cpp)
    target_include_directories(onebin PRIVATE engine/gpu/zinc_cpp/include)
endif()
set_target_properties(onebin PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
set_property(TARGET onebin APPEND PROPERTY LINK_OPTIONS "-Wl,-rpath,\$ORIGIN")
if(THEROCK_HIPBLAS_LIB)
    set_property(TARGET onebin APPEND PROPERTY
        LINK_OPTIONS "-Wl,-rpath,${THEROCK_LIB_DIR}")
endif()
if(EXISTS "${FLM_BUILD_DIR}/flm")
    add_dependencies(onebin build_flm)
endif()

# -- onebitd : 1bit inference daemon (spawns backend, proxies HTTP) --------
# No standalone add_executable — tools/onebitd.cpp compiles into onebin/1bit
# only; run via `1bit onebitd` / `1bit daemon`.

# -- jarvis_server : local private AI agent server (RAG, tools, planner) ---
# C++ port of the deleted jarvis/ Python package (recovered from git
# history at c252174aa~1 — see .claude/plans/swirling-tumbling-frost.md).
# Talks to unified_server (127.0.0.1:8088) and Ollama over HTTP rather than
# embedding a second BackendManager; own port 8080.
# ── Optional: ONNX Runtime for codec decoder (voice cloning TTS) ──
# The codec decoder compiles without it — decode() returns empty PCM.
# Set ONNXRUNTIME_ROOT_DIR to help find_package locate the SDK.
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "ONNX Runtime install root")
find_package(onnxruntime QUIET HINTS "${ONNXRUNTIME_ROOT_DIR}")
if(onnxruntime_FOUND)
    message(STATUS "codec: ONNX Runtime FOUND — building codec decoder with real inference")
    add_definitions(-DUSE_ONNXRUNTIME)
    target_link_libraries(onebin PRIVATE onnxruntime::onnxruntime)
else()
    message(STATUS "codec: ONNX Runtime NOT found — codec decoder will return empty PCM")
endif()

# No standalone add_executable — jarvis_server.cpp + tools/jarvis/*.cpp +
# src/codec_decoder.cpp (all listed in ONE_BIN_SOURCES above) compile into
# onebin/1bit only; run via `1bit jarvis` / `1bit voice`.

# -- zaya1_vl_demo : ZAYA1-VL-8B vision-language inference demo -------------
add_executable(zaya1_vl_demo tools/zaya1_vl_demo.cpp)
target_include_directories(zaya1_vl_demo PRIVATE src/ include/)
target_compile_options(zaya1_vl_demo PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(zaya1_vl_demo PRIVATE
    backend_manager vision_encoder vl_image dl
    ${CMAKE_DL_LIBS})

# -- whisper_demo : Whisper speech-to-text demo -----------------------------------------
add_executable(whisper_demo tools/whisper_demo.cpp)
target_include_directories(whisper_demo PRIVATE src/ include/)
target_compile_options(whisper_demo PRIVATE -O3 -Wall -Wno-unused-result -std=c++23 -fopenmp)
target_link_libraries(whisper_demo PRIVATE
    whisper backend_manager vl_image dl
    ${CMAKE_DL_LIBS})

# -- whisper_ggml_to_gguf : one-off asset-prep converter (legacy ggml.bin
# whisper models -> this project's own strict-GGUF format). Not part of
# the runtime product; only used to prepare a model file for jarvis_server.
add_executable(whisper_ggml_to_gguf tools/whisper_ggml_to_gguf.cpp)
target_compile_options(whisper_ggml_to_gguf PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)

# -- test_vit_encoder : Standalone ViT encoder test ---------------------------------
add_executable(test_vit_encoder tools/test_vit_encoder.cpp)
target_include_directories(test_vit_encoder PRIVATE src/ include/)
target_compile_options(test_vit_encoder PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(test_vit_encoder PRIVATE
    vision_encoder backend_manager vl_image dl
    ${CMAKE_DL_LIBS})

# -- vl_pipeline_test : End-to-end VL pipeline architecture test -------------------
add_executable(vl_pipeline_test tools/vl_pipeline_test.cpp)
target_include_directories(vl_pipeline_test PRIVATE src/ include/)
target_compile_options(vl_pipeline_test PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(vl_pipeline_test PRIVATE
    vision_encoder backend_manager vl_image dl
    ${CMAKE_DL_LIBS})


# -- onebit : NPU-native terminal coding agent CLI -------------------------
# No standalone add_executable — tools/onebit.cpp compiles into onebin/1bit
# only; onebit's main() lives inside build/1bit (OUTPUT_NAME "1bit"). Run
# via `1bit` with no recognized subcommand, or `1bit chat`/`up`/`down`/etc.

add_executable(bitnet_tui tools/bitnet_tui.cpp)
target_link_libraries(bitnet_tui PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    ftxui::screen ftxui::dom ftxui::component)

if(RCPP_HAVE_CK)
    add_subdirectory(ck-prefill)
endif()


# -- npu_engine_hybrid : NPU+GPU hybrid inference engine ------------------
# Guarded: requires a pre-built dequant_q4nx.o from the FastFlowLM toolchain
# which may not exist in all clones. Skipped silently when absent.
if(EXISTS "${CMAKE_SOURCE_DIR}/engine/npu/build/dequant_q4nx.o")
    add_executable(npu_engine_hybrid
        engine/npu/src/npu_engine_hybrid.cpp
        engine/npu/build/dequant_q4nx.o
)
else()
    message(STATUS "npu_engine_hybrid: engine/npu/build/dequant_q4nx.o not found — skipping NPU hybrid target")
endif()
if(TARGET npu_engine_hybrid)
    target_include_directories(npu_engine_hybrid PRIVATE
        ${CMAKE_SOURCE_DIR}/engine/npu
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR})
    # -ffast-math implies -ffinite-math-only and compiles the std::isfinite
    # NaN/Inf masks out (issue #1277) — keep them alive.
    target_compile_options(npu_engine_hybrid PRIVATE
        -O3 -march=native -std=c++23 -fopenmp)
    target_link_libraries(npu_engine_hybrid PRIVATE xrt_coreutil uuid rt pthread)
    target_link_options(npu_engine_hybrid PRIVATE
        -L${THEROCK_SDK_BASE}/lib -lamdhip64 -Wl,-rpath,${THEROCK_SDK_BASE}/lib -fopenmp)
    set_target_properties(npu_engine_hybrid PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/engine/npu/build)
endif()

# -- qwen36_moe_probe : native NPU MoE expert routing validation ---------
# Standalone probe that runs ONE real Qwen3.6-35B-A3B MoE layer FFN on the
# NPU end-to-end. Guarded: requires XRT + NPU runtime.
if(EXISTS "/opt/xilinx/xrt/lib/libxrt_coreutil.so")
    add_executable(qwen36_moe_probe
        tools/qwen36_moe_probe.cpp
        engine/npu/src/dequant_q4nx.cpp
        engine/npu/src/gemm_npu_instructions.cpp
    )
else()
    message(STATUS "qwen36_moe_probe: XRT not found — skipping MoE probe target")
endif()
if(TARGET qwen36_moe_probe)
    target_include_directories(qwen36_moe_probe PRIVATE
        ${CMAKE_SOURCE_DIR}/engine/npu/src
        ${CMAKE_SOURCE_DIR}/npu-infer/include
        ${CMAKE_SOURCE_DIR}/third_party/FastFlowLM/src/include
        ${CMAKE_SOURCE_DIR}/third_party/FastFlowLM/src/include/npu_utils)
    target_compile_options(qwen36_moe_probe PRIVATE
        -O2 -std=c++20 -fopenmp)
    target_link_libraries(qwen36_moe_probe PRIVATE
        xrt_coreutil xrt_core aiebu uuid m dl pthread)
    target_link_options(qwen36_moe_probe PRIVATE
        -L/opt/xilinx/xrt/lib -Wl,-rpath,/opt/xilinx/xrt/lib -fopenmp)
    set_target_properties(qwen36_moe_probe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
endif()

# -- libfusion_zero_copy: zero-copy NPU<->GPU shared-buffer substrate --------
# Promotes the proven SharedBO/PipelineOverlap primitives (engine/fusion/zero_copy/,
# see its README — hardware-verified zero-copy, 3/3 runs, zero IO_PAGE_FAULTs)
# from "only reachable via the test_pipeline_real binary" to a real linkable
# library, so backend_manager and future hybrid GPU+NPU backends can use them
# without pulling in test_pipeline_real.cpp's demo/harness code.
find_package(aiebu QUIET)
find_package(xrt QUIET)
if(_RC_HAVE_FUSED_BACKEND)
    if(EXISTS "${CMAKE_SOURCE_DIR}/engine/fusion/zero_copy/shared_bo.cpp" AND
       EXISTS "${CMAKE_SOURCE_DIR}/engine/fusion/zero_copy/pipeline_overlap.cpp")
        add_library(fusion_zero_copy STATIC
            engine/fusion/zero_copy/shared_bo.cpp
            engine/fusion/zero_copy/pipeline_overlap.cpp
        )
        target_include_directories(fusion_zero_copy PUBLIC
            engine/fusion/zero_copy
            ${CMAKE_SOURCE_DIR}/include
            ${THEROCK_SDK_BASE}/include
            ${XRT_INCLUDE_DIRS})  # xrt/xrt_bo.h etc. — consumed by fused backend in unified_server/onebin
        target_compile_options(fusion_zero_copy PRIVATE
            -g -O3 -march=native -fno-signed-zeros -fno-trapping-math -fassociative-math
            -freciprocal-math -fno-math-errno -fno-rounding-math
            -std=c++23 -D__HIP_PLATFORM_AMD__)
        target_link_libraries(fusion_zero_copy PUBLIC
            xrt_coreutil xrt_core uuid rt pthread
            /usr/lib/x86_64-linux-gnu/libaiebu.a)

        # Link fusion_zero_copy into unified_server for fused backend
        if(TARGET unified_server)
            target_link_libraries(unified_server PRIVATE fusion_zero_copy)
            target_include_directories(unified_server PRIVATE engine/fusion/zero_copy)
            target_compile_definitions(unified_server PRIVATE RCPP_HAVE_FUSION_ZERO_COPY=1)
        endif()

        # onebin is defined earlier in this file, so it cannot see this
        # target yet — wire it up here, after fusion_zero_copy exists.
        if(TARGET onebin)
            target_link_libraries(onebin PRIVATE fusion_zero_copy)
            target_include_directories(onebin PRIVATE engine/fusion/zero_copy)
            target_compile_definitions(onebin PRIVATE RCPP_HAVE_FUSION_ZERO_COPY=1)
        endif()

        # bench_fused: Fused GPU attn ∥ NPU FFN benchmark
        add_executable(bench_fused tools/bench_fused.cpp src/backend_fused.cpp src/backend_fused_npu.cpp src/kv_cache_attn.hip)
        set_source_files_properties(src/backend_fused.cpp PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/backend_fused_npu.cpp PROPERTIES LANGUAGE CXX)
        set_source_files_properties(src/kv_cache_attn.hip PROPERTIES LANGUAGE HIP)
        target_include_directories(bench_fused PRIVATE src include engine/fusion/zero_copy engine/npu/src)
        target_compile_options(bench_fused PRIVATE -O2 -std=c++23 -D__HIP_PLATFORM_AMD__ -DRCPP_HAVE_FUSION_ZERO_COPY)
        target_link_libraries(bench_fused PRIVATE fusion_zero_copy hip::host hip::device hipblas rocm_cpp pthread)
        set_source_files_properties(tools/bench_fused.cpp PROPERTIES LANGUAGE CXX)
        set_target_properties(bench_fused PROPERTIES
            HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
    endif()
endif()

# -- bench_ggml_vk: GGML-Vulkan backend benchmark (llama.cpp MIT License) ---
if(TARGET ggml_vulkan)
    add_executable(bench_ggml_vk tools/bench_ggml_vk.cpp src/backend_ggml_vulkan.cpp)
    target_include_directories(bench_ggml_vk PRIVATE
        include
        src
        third_party/llama.cpp/include
        third_party/llama.cpp/ggml/include)
    target_link_options(bench_ggml_vk PRIVATE -Wl,--allow-multiple-definition)
    target_link_libraries(bench_ggml_vk PRIVATE
        "${GGML_BUILD_DIR}/ggml/src/ggml-vulkan/libggml-vulkan.a"
        "${GGML_BUILD_DIR}/ggml/src/libggml.a"
        "${GGML_BUILD_DIR}/ggml/src/libggml-cpu.a"
        "${GGML_BUILD_DIR}/ggml/src/libggml-base.a"
        "${GGML_BUILD_DIR}/src/libllama.a"
        -lvulkan -ldl -lpthread -lm)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(bench_ggml_vk PRIVATE OpenMP::OpenMP_CXX)
    endif()
    set_target_properties(bench_ggml_vk PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
    message(STATUS "bench_ggml_vk: GGML-Vulkan benchmark enabled")
endif()

# -- test_pipeline_real: GPU+NPU fused pipeline with real kernels (#446) ------
# Requires HIP, XRT, and the zero-copy pipeline sources.
# Uses relative paths (no variable indirection) so the CI reference checker
# can verify they resolve.
if(hip_FOUND AND xrt_FOUND AND aiebu_FOUND)
    if(TARGET fusion_zero_copy)
        add_executable(test_pipeline_real
            engine/fusion/zero_copy/test_pipeline_real.cpp
            src/kv_cache_attn.hip
        )
        target_include_directories(test_pipeline_real PRIVATE
            engine/fusion/zero_copy
            ${CMAKE_SOURCE_DIR}/include
            ${THEROCK_SDK_BASE}/include)
        # NOTE: no -ffast-math here (deliberately) — it implies -ffinite-math-only,
        # which makes std::isfinite()/NaN-guards throughout this file undefined
        # behavior and was the root cause of the test_pipeline_real SEGFAULT (#825).
        # This subset gives the same perf without disabling NaN/Inf semantics.
        target_compile_options(test_pipeline_real PRIVATE
            -g -O3 -march=native -fno-signed-zeros -fno-trapping-math -fassociative-math
            -freciprocal-math -fno-math-errno -fno-rounding-math
            -std=c++23 -fopenmp -D__HIP_PLATFORM_AMD__)
        target_link_libraries(test_pipeline_real PRIVATE
            fusion_zero_copy
            xrt_coreutil xrt_core uuid rt pthread
            /usr/lib/x86_64-linux-gnu/libaiebu.a)
        target_link_options(test_pipeline_real PRIVATE
            -L${THEROCK_SDK_BASE}/lib -lamdhip64 -Wl,-rpath,${THEROCK_SDK_BASE}/lib -fopenmp)
        set_target_properties(test_pipeline_real PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/engine/fusion/zero_copy
            HIP_ARCHITECTURES "gfx1151")
        add_test(NAME test_pipeline_real COMMAND test_pipeline_real)
        set_tests_properties(test_pipeline_real PROPERTIES
            LABELS "gpu_required;npu_required" SKIP_RETURN_CODE 77
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

        # -- test_npu_ffn_real_weights: NPU FFN correctness vs real model weights --
        # Loads real blk.0 FFN weights from models/Qwen3-0.6B.1bp (shape-matched
        # to this pipeline demo's H=1024/IM=3072) and checks the NPU's
        # int8-quantized GEMM output against a CPU float32 reference computed
        # from the same real, dequantized weights (see file header for the
        # GGUF [out,in] -> packB [K,N] transpose this depends on).
        add_executable(test_npu_ffn_real_weights
            engine/fusion/zero_copy/test_npu_ffn_real_weights.cpp
        )
        target_include_directories(test_npu_ffn_real_weights PRIVATE
            engine/fusion/zero_copy
            ${CMAKE_SOURCE_DIR}/include
            ${THEROCK_SDK_BASE}/include)
        target_compile_options(test_npu_ffn_real_weights PRIVATE
            -g -O2 -std=c++23 -D__HIP_PLATFORM_AMD__)
        target_link_libraries(test_npu_ffn_real_weights PRIVATE
            xrt_coreutil xrt_core uuid rt pthread
            /usr/lib/x86_64-linux-gnu/libaiebu.a)
        set_target_properties(test_npu_ffn_real_weights PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/engine/fusion/zero_copy)
        if(EXISTS "${CMAKE_SOURCE_DIR}/models/Qwen3-0.6B.1bp")
            add_test(NAME test_npu_ffn_real_weights COMMAND test_npu_ffn_real_weights
                     ${CMAKE_SOURCE_DIR}/models/Qwen3-0.6B.1bp)
            set_tests_properties(test_npu_ffn_real_weights PROPERTIES
                LABELS "npu_required;model_required;regression" SKIP_RETURN_CODE 77
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        endif()
    else()
        message(STATUS "test_pipeline_real: shared_bo.cpp or pipeline_overlap.cpp not found — skipping")
    endif()
else()
    message(STATUS "test_pipeline_real: HIP/XRT/aiebu not all found — skipping GPU+NPU pipeline test")
endif()

# -- DSpark full pipeline: Eagle3 draft (NPU/CPU) + GPU verify ---------------
add_executable(dspark_full
    tools/dspark_full.cpp
    engine/fusion/cpu_layer.cpp)
target_include_directories(dspark_full PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/fusion
    ${CMAKE_SOURCE_DIR}/spec-decode/draft
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR})
target_compile_options(dspark_full PRIVATE
    -O3 -march=native -std=c++23 -fopenmp)
target_link_libraries(dspark_full PRIVATE
    rocm_cpp
    ${CMAKE_DL_LIBS} m pthread)
target_link_options(dspark_full PRIVATE
    -L${THEROCK_SDK_BASE}/lib -lamdhip64 -Wl,-rpath,${THEROCK_SDK_BASE}/lib)
set_target_properties(dspark_full PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tools)

# -- DSpark GPU bench: full GPU-accelerated speculative decode --------------
find_package(hipblas QUIET)
if(hipblas_FOUND)
add_executable(dspark_gpu_bench
    tools/dspark_gpu_bench.cpp
    engine/fusion/cpu_layer.cpp)
target_include_directories(dspark_gpu_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/fusion
    ${CMAKE_SOURCE_DIR}/spec-decode/draft
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/spec-decode
    ${THEROCK_SDK_BASE}/include)
target_compile_options(dspark_gpu_bench PRIVATE
    -O3 -march=native -std=c++23 -fopenmp -D__HIP_PLATFORM_AMD__)
target_link_libraries(dspark_gpu_bench PRIVATE
    rocm_cpp
    ${CMAKE_DL_LIBS} m pthread hipblas)
target_link_options(dspark_gpu_bench PRIVATE
    -L${THEROCK_SDK_BASE}/lib -lamdhip64 -lhipblas -Wl,-rpath,${THEROCK_SDK_BASE}/lib)
set_target_properties(dspark_gpu_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tools)
else()
    message(STATUS "rocm-cpp: hipblas NOT found — skipping dspark_gpu_bench (needs ROCm hipblas)")
endif()

# -- Q4NX -> TRG converter --
add_executable(trg_save
    tools/trg_save.cpp
    engine/fusion/cpu_layer.cpp)
target_include_directories(trg_save PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/fusion
    ${CMAKE_SOURCE_DIR})
target_compile_options(trg_save PRIVATE
    -O3 -march=native -std=c++23 -fopenmp)
target_link_libraries(trg_save PRIVATE
    ${CMAKE_DL_LIBS} m pthread)
set_target_properties(trg_save PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tools)

# -- OSCAR calibration tool --
add_executable(oscar_calib
    tools/oscar_calib.cpp
    engine/fusion/cpu_layer.cpp)
target_include_directories(oscar_calib PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/fusion
    ${CMAKE_SOURCE_DIR})
target_compile_options(oscar_calib PRIVATE
    -O3 -march=native -std=c++23 -fopenmp)
target_link_libraries(oscar_calib PRIVATE
    ${CMAKE_DL_LIBS} m pthread)
set_target_properties(oscar_calib PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tools)

# -- OSCAR INT2 quant + decode HIP library --
add_library(oscar SHARED
    src/oscar.cpp
    kernels/oscar_quant.hip)
target_include_directories(oscar PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR})
target_compile_options(oscar PRIVATE
    -O3 -ffast-math -std=c++23)
target_link_libraries(oscar PRIVATE
    ${CMAKE_DL_LIBS} m pthread)
target_link_options(oscar PRIVATE
    -L${THEROCK_SDK_BASE}/lib -lamdhip64 -Wl,-rpath,${THEROCK_SDK_BASE}/lib)

add_executable(test_oscar_attn_decode tests/test_oscar_attn_decode.cpp)
target_include_directories(test_oscar_attn_decode PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_oscar_attn_decode PRIVATE -O2 -Wno-unused-result)
target_link_libraries(test_oscar_attn_decode PRIVATE oscar hip::host hip::device)
set_target_properties(test_oscar_attn_decode PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
add_test(NAME test_oscar_attn_decode COMMAND test_oscar_attn_decode)
set_tests_properties(test_oscar_attn_decode PROPERTIES LABELS "gpu_required" SKIP_RETURN_CODE 77)

# -- tq2_to_q4nx : Convert 1BP TQ2 to Q4NX format for NPU inference ---------
add_executable(tq2_to_q4nx tools/tq2_to_q4nx.cpp)
target_include_directories(tq2_to_q4nx PRIVATE src/ ${CMAKE_SOURCE_DIR}/include)
target_compile_options(tq2_to_q4nx PRIVATE -O3 -std=c++23)
target_link_libraries(tq2_to_q4nx PRIVATE onebp_model pthread)

# ═══════════════════════════════════════════════════════════════════════════
# TRACK 3: LoRA Adapter Runtime
# ═══════════════════════════════════════════════════════════════════════════
# Hot-loadable GGUF LoRA adapters for any 1BP model.
option(USE_LORA "Build LoRA adapter runtime support" ON)
if(USE_LORA)
    add_library(lora_runtime STATIC src/lora.cpp)
    target_include_directories(lora_runtime PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_compile_options(lora_runtime PRIVATE -O3 -std=c++23 -fopenmp)
    target_link_libraries(lora_runtime PUBLIC gguf_reader)
    
    # LoRA merge tool: merge LoRA weights permanently into a GGUF
    add_executable(lora_merge tools/lora_merge.cpp)
    target_include_directories(lora_merge PRIVATE src/ include/)
    target_compile_options(lora_merge PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
    target_link_libraries(lora_merge PRIVATE lora_runtime backend_manager)
    
    message(STATUS "lora: LoRA adapter runtime enabled")
endif()

# ═══════════════════════════════════════════════════════════════════════════
# TRACK 1: stable-diffusion.cpp Integration (Diffusion Bridge)
# ═══════════════════════════════════════════════════════════════════════════
# Image/video generation via stable-diffusion.cpp submodule.
# Set -DUSE_DIFFUSION=ON and init submodule at third_party/stable-diffusion.cpp/
option(USE_DIFFUSION "Build stable-diffusion.cpp integration" OFF)
if(USE_DIFFUSION)
    # Find sd.cpp library (built separately or via submodule)
    # The sd.cpp library is named libstable-diffusion.a. build-rocm is the
    # ROCm/HIP build, build-webm the webm-enabled CPU build, plain build/ the
    # fallback. All three are produced by the sd.cpp submodule's own CMake.
    set(SD_CPP_DIRS
        "${CMAKE_SOURCE_DIR}/third_party/stable-diffusion.cpp/build-rocm"
        "${CMAKE_SOURCE_DIR}/third_party/stable-diffusion.cpp/build-webm"
        "${CMAKE_SOURCE_DIR}/third_party/stable-diffusion.cpp/build")
    find_library(SD_CPP_LIB stable-diffusion PATHS ${SD_CPP_DIRS}
        /usr/local/lib /usr/lib NO_DEFAULT_PATH)
    if(NOT SD_CPP_LIB)
        find_library(SD_CPP_LIB stable-diffusion PATHS ${SD_CPP_DIRS}
            /usr/local/lib /usr/lib)
    endif()
    
    if(SD_CPP_LIB)
        # Diffusion bridge library
        add_library(diffusion_bridge STATIC src/diffusion_bridge.cpp
            third_party/stable-diffusion.cpp/examples/common/media_io.cpp
            third_party/stable-diffusion.cpp/examples/common/log.cpp)
        target_include_directories(diffusion_bridge PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/ggml/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/examples/common>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty>
            $<INSTALL_INTERFACE:include>)
        target_compile_options(diffusion_bridge PRIVATE -O3 -std=c++23)
        # Also find ggml libs that sd.cpp depends on
        set(SD_GGML_DIRS)
        foreach(d IN LISTS SD_CPP_DIRS)
            list(APPEND SD_GGML_DIRS "${d}/ggml/src")
        endforeach()
        find_library(SD_GGML_LIB ggml PATHS ${SD_GGML_DIRS} NO_DEFAULT_PATH)
        find_library(SD_GGML_CPU_LIB ggml-cpu PATHS ${SD_GGML_DIRS} NO_DEFAULT_PATH)
        find_library(SD_GGML_BASE_LIB ggml-base PATHS ${SD_GGML_DIRS} NO_DEFAULT_PATH)
        find_library(SD_GGML_HIP_LIB ggml-hip PATHS
            "${CMAKE_SOURCE_DIR}/third_party/stable-diffusion.cpp/build-rocm/ggml/src/ggml-hip"
            NO_DEFAULT_PATH)
        
        # Optional WebM/WebP video containers: light up automatically when the
        # sd.cpp submodule build has SD_WEBM=ON (build-webm/build-rocm, produces
        # libwebm.a/libwebp.a/libwebpmux.a in its build dir). Without them the
        # encoder falls back to MJPG AVI, same as sd.cpp's own server default.
        set(SD_WEBM_BUILD_DIR "${CMAKE_SOURCE_DIR}/third_party/stable-diffusion.cpp/build-webm")
        find_library(SD_WEBM_LIB webm PATHS ${SD_WEBM_BUILD_DIR}/thirdparty/libwebm NO_DEFAULT_PATH)
        find_library(SD_WEBP_LIB webp PATHS ${SD_WEBM_BUILD_DIR}/thirdparty/libwebp NO_DEFAULT_PATH)
        find_library(SD_WEBPMUX_LIB webpmux PATHS ${SD_WEBM_BUILD_DIR}/thirdparty/libwebp NO_DEFAULT_PATH)
        find_library(SD_SHARPYUV_LIB sharpyuv PATHS ${SD_WEBM_BUILD_DIR}/thirdparty/libwebp NO_DEFAULT_PATH)
        if(SD_WEBM_LIB AND SD_WEBP_LIB AND SD_WEBPMUX_LIB)
            target_compile_definitions(diffusion_bridge PRIVATE SD_USE_WEBM SD_USE_WEBP)
            target_include_directories(diffusion_bridge PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty/libwebm>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty/libwebp/src>)
            target_link_libraries(diffusion_bridge PUBLIC ${SD_WEBM_LIB} ${SD_WEBP_LIB} ${SD_WEBPMUX_LIB} ${SD_SHARPYUV_LIB})
            message(STATUS "diffusion: webm/webp video containers enabled")
        else()
            message(STATUS "diffusion: webm/webp not found, video outputs use MJPG AVI")
        endif()
        
        # ROCm build (build-rocm): static libs don't carry runtime deps, so
        # link the HIP runtime explicitly and rpath it into the server.
        # Same TheRock SDK location the top of this file pins for HIP builds.
        set(_RC_SDK_LIB "/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_libraries/lib")
        set(_RC_SDK_CORE "/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib")
        # Versioned .so names only (no unversioned symlinks) — link exact paths.
        set(SD_AMD_HIP64_LIB "${_RC_SDK_CORE}/libamdhip64.so.7")
        set(SD_ROCBLAS_LIB "${_RC_SDK_LIB}/librocblas.so.5")
        set(SD_HIPBLAS_LIB "${_RC_SDK_LIB}/libhipblas.so.3")
        if(SD_CPP_LIB MATCHES "build-rocm")
            target_link_libraries(diffusion_bridge PUBLIC ${SD_AMD_HIP64_LIB} ${SD_ROCBLAS_LIB} ${SD_HIPBLAS_LIB})
            set(_SD_HIP_RPATH "${_RC_SDK_LIB}:${_RC_SDK_CORE}")  # applied to image_server below
            message(STATUS "diffusion: ROCm/HIP runtime linked (${_RC_SDK_LIB})")
        endif()
        
        target_link_libraries(diffusion_bridge PUBLIC
            ${SD_CPP_LIB}
            ${SD_GGML_LIB}
            ${SD_GGML_CPU_LIB}
            ${SD_GGML_BASE_LIB}
            ${SD_GGML_HIP_LIB}
            vl_image m pthread)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(diffusion_bridge PUBLIC OpenMP::OpenMP_CXX)
        endif()
        
        # Image generation server (OpenAI-compatible /v1/images/*)
        add_executable(image_server tools/image_server.cpp)
        target_include_directories(image_server PRIVATE src/ include/)
        target_compile_options(image_server PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
        target_link_libraries(image_server PRIVATE
            diffusion_bridge vl_image backend_manager
            httplib::httplib nlohmann_json::nlohmann_json)
        if(_SD_HIP_RPATH)
            set_target_properties(image_server PROPERTIES BUILD_RPATH "${_SD_HIP_RPATH}")
        endif()
        
        message(STATUS "diffusion: stable-diffusion.cpp integration enabled (${SD_CPP_LIB})")
    else()
        message(STATUS "diffusion: sd.cpp library not found — set -DUSE_DIFFUSION=OFF or build sd.cpp first")
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════
# TRACK 4: audio.cpp Integration (Audio Bridge)
# ═══════════════════════════════════════════════════════════════════════════
# TTS, STT, music generation, voice cloning, VAD via audio.cpp submodule.
# Set -DUSE_AUDIO_CPP=ON and init submodule at third_party/audio.cpp/
option(USE_AUDIO_CPP "Build audio.cpp integration" OFF)
if(USE_AUDIO_CPP)
    set(AUDIO_CPP_DIR "${CMAKE_SOURCE_DIR}/third_party/audio.cpp/build")
    find_library(AUDIO_CPP_LIB engine_runtime PATHS ${AUDIO_CPP_DIR} NO_DEFAULT_PATH)
    if(NOT AUDIO_CPP_LIB)
        find_library(AUDIO_CPP_LIB engine_runtime PATHS ${AUDIO_CPP_DIR})
    endif()
    
    find_library(AUDIO_GGML_LIB ggml PATHS ${AUDIO_CPP_DIR}/ggml/src NO_DEFAULT_PATH)
    find_library(AUDIO_GGML_CPU_LIB ggml-cpu PATHS ${AUDIO_CPP_DIR}/ggml/src NO_DEFAULT_PATH)
    find_library(AUDIO_GGML_BASE_LIB ggml-base PATHS ${AUDIO_CPP_DIR}/ggml/src NO_DEFAULT_PATH)
    
    if(AUDIO_CPP_LIB)
        # Audio bridge library
        add_library(audio_bridge STATIC src/audio_bridge.cpp)
        target_include_directories(audio_bridge PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        target_include_directories(audio_bridge PRIVATE
            ${AUDIO_CPP_DIR}/../include  # audio.cpp headers
            ${AUDIO_CPP_DIR}/../ggml/include)
        target_compile_options(audio_bridge PRIVATE -O3 -std=c++23)
        target_link_libraries(audio_bridge PUBLIC
            ${AUDIO_CPP_LIB}
            ${AUDIO_GGML_LIB}
            ${AUDIO_GGML_CPU_LIB}
            ${AUDIO_GGML_BASE_LIB}
            m pthread)
        
        # jarvis_server is no longer a standalone target (see onebin) —
        # give onebin native audio.cpp TTS instead of Piper.
        if(TARGET onebin)
            target_link_libraries(onebin PRIVATE audio_bridge)
        endif()
        
        message(STATUS "audio: audio.cpp integration enabled (${AUDIO_CPP_LIB})")
    else()
        message(STATUS "audio: engine_runtime library not found — set -DUSE_AUDIO_CPP=OFF or build audio.cpp first")
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════
# Smoke tests for new tracks
# ═══════════════════════════════════════════════════════════════════════════

# LoRA runtime smoke test (no model required, tests error paths)
add_executable(test_lora_runtime tests/test_lora_runtime.cpp)
target_include_directories(test_lora_runtime PRIVATE include/)
target_compile_options(test_lora_runtime PRIVATE -O3 -std=c++23)
find_package(OpenMP QUIET)
target_link_libraries(test_lora_runtime PRIVATE lora_runtime gguf_reader)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_lora_runtime PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME test_lora_runtime COMMAND test_lora_runtime)
set_tests_properties(test_lora_runtime PROPERTIES LABELS "host")

# Diffusion bridge smoke test (no model required, tests error paths)
if(USE_DIFFUSION)
    add_executable(test_diffusion_bridge tests/test_diffusion_bridge.cpp)
    target_include_directories(test_diffusion_bridge PRIVATE include/)
    target_compile_options(test_diffusion_bridge PRIVATE -O3 -std=c++23)
    target_link_libraries(test_diffusion_bridge PRIVATE diffusion_bridge vl_image)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_diffusion_bridge PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME test_diffusion_bridge COMMAND test_diffusion_bridge)
    set_tests_properties(test_diffusion_bridge PROPERTIES LABELS "host")
    message(STATUS "tests: diffusion bridge smoke test enabled")
endif()

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2.3 Commercial API tests (auth + billing)
# ═══════════════════════════════════════════════════════════════════════════

# Auth module unit tests
add_executable(test_auth
    tests/test_auth.cpp
    tools/jarvis/auth.cpp)
target_include_directories(test_auth PRIVATE tools/ src/ include/)
target_compile_options(test_auth PRIVATE -O3 -std=c++23)
target_link_libraries(test_auth PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME test_auth COMMAND test_auth)
set_tests_properties(test_auth PROPERTIES LABELS "host")

# Billing module unit tests
add_executable(test_billing
    tests/test_billing.cpp
    tools/jarvis/auth.cpp
    tools/jarvis/billing.cpp)
target_include_directories(test_billing PRIVATE tools/ src/ include/)
target_compile_options(test_billing PRIVATE -O3 -std=c++23)
target_link_libraries(test_billing PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME test_billing COMMAND test_billing)
set_tests_properties(test_billing PROPERTIES LABELS "host")

message(STATUS "tests: Phase 2.3 Commercial API tests enabled")

# ═══════════════════════════════════════════════════════════════════════════
# TRACK 2: New 1BP model architecture additions
# ═══════════════════════════════════════════════════════════════════════════
message(STATUS "models: 35+ architectures supported for 1BP conversion")

# ═══════════════════════════════════════════════════════════════════════════
# NPU Engine (XRT required)
# ═══════════════════════════════════════════════════════════════════════════
find_package(XRT QUIET)
if(XRT_FOUND)
    add_subdirectory(engine/npu EXCLUDE_FROM_ALL)
    message(STATUS "npu: XRT found — NPU engine targets available via 'cmake --build build --target npu_engine_universal'")
else()
    message(STATUS "npu: XRT NOT found — skipping NPU engine build")
endif()



# ═══════════════════════════════════════════════════════════════════════════
# Zaya codec decoder (issue #1368): RVQ-VAE GGUF decoder + self-check
# Dependency-light CPU decoder (stdlib + gguf_reader). GGUF file is produced
# by tools/export_codec_gguf.py.
# ═══════════════════════════════════════════════════════════════════════════
add_library(zaya_codec STATIC src/zaya_codec.cpp)
target_include_directories(zaya_codec PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(zaya_codec PRIVATE -O3 -std=c++17)
target_link_libraries(zaya_codec PUBLIC gguf_reader)

add_executable(test_zaya_codec tests/test_zaya_codec.cpp)
target_include_directories(test_zaya_codec PRIVATE include/)
target_compile_options(test_zaya_codec PRIVATE -O3 -std=c++17)
target_link_libraries(test_zaya_codec PRIVATE zaya_codec)
