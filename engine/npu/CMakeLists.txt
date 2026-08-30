cmake_minimum_required(VERSION 3.21)
project(npu-engine LANGUAGES C CXX)

# ── Toolchain ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm/HIP used for TheRock amdclang++ runtime (tools only, not the engine)
find_package(hip REQUIRED)
find_package(XRT QUIET)

# ── npu_engine_universal (flagship — all ops: QKV, O, GU, D, attention) ──
set(NPU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(NPU_ENGINE_SOURCES
    ${NPU_SRC_DIR}/npu_engine_universal.cpp
    ${NPU_SRC_DIR}/dequant_q4nx.cpp
    ${NPU_SRC_DIR}/gemm_npu_instructions.cpp
    ${NPU_SRC_DIR}/zaya_decode.cpp
)
# zaya_decode.cpp — the Zaya decode path (CCA attention on CPU, MoE FFN on
# NPU, NPU_FUSED=1 fused GU->SiLU->D mode). npu_engine_universal dispatches
# to zaya_decode_main when the model header contains "zaya", so it must be
# linked in. Needs the generators/ include dir (silu_quant.h, the dual-
# compiled fused-kernel contract) and -mavx2 (zaya_moe_cpu.h AVX2 host amax).
# flm_bridge.cpp (FLM dlopen wrapper) was removed 2026-08-15: no engine code
# calls FlmBridge — HybridFlmCtx generates instructions via the open-source
# gemm_generate_sequence_i8_split(), and the pre-compiled .txt streams load
# through aiebu. FLM libs (libgemm.so et al) are never dlopened at runtime.

add_executable(npu_engine_universal ${NPU_ENGINE_SOURCES})

# ── #embed freshness (issue #1957) ────────────────────────────────────────
# npu_embedded.h bakes the xclbin/insts artifacts into the binary at compile
# time (P1967 #embed). If an artifact is regenerated (e.g. a new
# build_iron_cascade*.sh / build_attn.sh run) the embed TU MUST recompile —
# otherwise the binary silently ships the old kernel. OBJECT_DEPENDS makes
# any artifact change trigger a rebuild of zaya_decode.cpp.o (the TU that
# pulls in npu_embedded.h via npu_attn_ctx.h).
set(NPU_EMBEDDED_ARTIFACTS
    ${CMAKE_CURRENT_SOURCE_DIR}/xclbins/attn.xclbin
    ${CMAKE_CURRENT_SOURCE_DIR}/xclbins/attn_insts.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/xclbins/final_cascade_fused.xclbin
    ${CMAKE_CURRENT_SOURCE_DIR}/xclbins/insts_cascade_fused.txt
)
set_source_files_properties(${NPU_SRC_DIR}/zaya_decode.cpp PROPERTIES
    OBJECT_DEPENDS "${NPU_EMBEDDED_ARTIFACTS}")

# Configure-time stale-embed warning: if an artifact is NEWER than the last
# configure (i.e. it was regenerated after this build tree was configured),
# tell the operator loudly that the next build will re-embed the new bytes.
# This catches "regenerated but forgot to rebuild" with an actionable message
# instead of a silent stale embed.
foreach(_art ${NPU_EMBEDDED_ARTIFACTS})
    if(EXISTS "${_art}")
        file(TIMESTAMP "${_art}" _art_mtime "%s")
        if(EXISTS "${CMAKE_BINARY_DIR}/CMakeCache.txt")
            file(TIMESTAMP "${CMAKE_BINARY_DIR}/CMakeCache.txt" _cache_mtime "%s")
            if(_art_mtime GREATER _cache_mtime)
                message(WARNING "npu-embed: '${_art}' is NEWER than the last CMake configure — the previous build's embedded copy is stale. The next build will re-embed it (OBJECT_DEPENDS). If this binary is already deployed, rebuild before shipping.")
            endif()
        endif()
    endif()
endforeach()


target_include_directories(npu_engine_universal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/generators
    ${CMAKE_CURRENT_SOURCE_DIR}/../../npu-infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include   # npu_instr.hpp, npu_app.hpp
)

if(EXISTS "/opt/xilinx/xrt/lib/libxrt_coreutil.so")
    # NPU runtime from /opt/xilinx/xrt (XRT 2.26). The apt system XRT's
    # libxrt_coreutil dlopens a nonexistent /usr/lib/lib/libxrt_core.so.2 at
    # runtime — always prefer the /opt tree when present (matches the probe).
    target_include_directories(npu_engine_universal PRIVATE /opt/xilinx/xrt/include)
    target_link_libraries(npu_engine_universal
        /opt/xilinx/xrt/lib/libxrt_coreutil.so
        /opt/xilinx/xrt/lib/libxrt_core.so)
    target_compile_definitions(npu_engine_universal PRIVATE -DXRT_ENABLE)
    set_target_properties(npu_engine_universal PROPERTIES
        BUILD_RPATH /opt/xilinx/xrt/lib INSTALL_RPATH /opt/xilinx/xrt/lib)
elseif(XRT_FOUND)
    target_include_directories(npu_engine_universal PRIVATE ${XRT_INCLUDE_DIRS})
    target_link_libraries(npu_engine_universal ${XRT_CORE_LIBRARIES})
    target_compile_definitions(npu_engine_universal PRIVATE -DXRT_ENABLE)
endif()

option(ONEBP_SUPPORT "Enable native 1BP model format loading" ON)
if(ONEBP_SUPPORT)
    target_compile_definitions(npu_engine_universal PRIVATE ONEBP_SUPPORT)
endif()

target_link_libraries(npu_engine_universal
    m dl pthread uuid
    OpenMP::OpenMP_CXX
    aiebu
)

target_compile_options(npu_engine_universal PRIVATE
    -O3 -ffast-math -march=native -funroll-loops -fno-math-errno -mavx2
    -Wno-unused-parameter -Wno-unused-variable -Wno-nan-infinity-disabled -Wno-implicit-const-int-float-conversion
)

install(TARGETS npu_engine_universal DESTINATION bin)

# ── Optional: torch2aie integration for bf16 kernel generation ──
set(NPU_TORCH2AIE_DIR "$ENV{HOME}/torch2aie" CACHE PATH "Path to torch2aie repo root")
if(EXISTS "${NPU_TORCH2AIE_DIR}/examples/helper.h")
    target_include_directories(npu_engine_universal PRIVATE
        ${NPU_TORCH2AIE_DIR}/iron
        ${NPU_TORCH2AIE_DIR}/examples
    )
    target_compile_definitions(npu_engine_universal PRIVATE NPU_HAVE_TORCH2AIE=1)
    message(STATUS "npu-bf16-engine: using torch2aie headers at ${NPU_TORCH2AIE_DIR}/examples")
endif()

# ── npu_engine_cb (v4) — dynamic instructions, xrt::ext::kernel ──
find_package(OpenMP REQUIRED)
find_library(AIEBU aiebu)
find_library(XRT_COREUTIL xrt_coreutil)

set(NPU_CB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/npu_engine_cb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dequant_q4nx.cpp
)

add_executable(npu_engine_cb ${NPU_CB_SOURCES})

target_include_directories(npu_engine_cb PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../npu-infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include   # npu_instr.hpp, npu_app.hpp
)

target_link_libraries(npu_engine_cb
    ${XRT_COREUTIL}
    m dl pthread uuid
)

target_compile_options(npu_engine_cb PRIVATE
    -O3 -ffast-math -march=native -funroll-loops -fno-math-errno
    -Wno-unused-parameter -Wno-unused-variable
)

target_compile_definitions(npu_engine_cb PRIVATE
    XCLBIN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/xclbins"
)

install(TARGETS npu_engine_cb DESTINATION bin)
