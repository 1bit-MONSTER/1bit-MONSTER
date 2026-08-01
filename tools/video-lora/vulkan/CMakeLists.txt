cmake_minimum_required(VERSION 3.21)
project(video-lora-vk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Vulkan + shader compiler ──
find_package(Vulkan REQUIRED)
# glslc (Vulkan SDK) or glslangValidator (Ubuntu glslang-tools) both work
find_program(GLSLC_EXECUTABLE glslc)
find_program(GLSLANG_VALIDATOR_EXECUTABLE glslangValidator)
if(GLSLC_EXECUTABLE)
    set(SHADER_COMPILER ${GLSLC_EXECUTABLE})
    set(SHADER_COMPILER_ARGS --target-env=vulkan1.2 -O)
elseif(GLSLANG_VALIDATOR_EXECUTABLE)
    set(SHADER_COMPILER ${GLSLANG_VALIDATOR_EXECUTABLE})
    set(SHADER_COMPILER_ARGS -V --target-env vulkan1.2)
else()
    message(FATAL_ERROR "video-lora: neither glslc nor glslangValidator found — install the Vulkan SDK (shaderc) or glslang-tools")
endif()

# ── Compile GLSL .comp → SPIR-V .spv at build time ──
set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(GLOB SHADER_SOURCES ${SHADER_SRC_DIR}/*.comp)
if(NOT SHADER_SOURCES)
    message(FATAL_ERROR "video-lora: no .comp shaders found in ${SHADER_SRC_DIR}")
endif()

set(SPV_FILES "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPV_OUTPUT ${SHADER_OUT_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SPV_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUT_DIR}
        COMMAND ${SHADER_COMPILER} ${SHADER_COMPILER_ARGS} -o ${SPV_OUTPUT} ${SHADER}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME}.comp → .spv")
    list(APPEND SPV_FILES ${SPV_OUTPUT})
endforeach()
add_custom_target(video_lora_shaders ALL DEPENDS ${SPV_FILES})

# ── Static library (links into the single zaya_server binary) ──
add_library(video_lora_vk STATIC
    src/vl_engine.cpp
)
target_include_directories(video_lora_vk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)
target_link_libraries(video_lora_vk PUBLIC Vulkan::Vulkan)
target_compile_options(video_lora_vk PRIVATE -O2 -Wall -Wno-unused-result)
target_compile_definitions(video_lora_vk PRIVATE
    VIDEO_LORA_SHADER_DIR="${SHADER_OUT_DIR}")
add_dependencies(video_lora_vk video_lora_shaders)

# ── CLI / selftest ──
add_executable(video_lora_vk_cli src/main.cpp)
target_link_libraries(video_lora_vk_cli PRIVATE video_lora_vk)
add_dependencies(video_lora_vk_cli video_lora_shaders)

# ── Test: selftest verifies every op vs CPU reference ──
enable_testing()
add_test(NAME video_lora_selftest
         COMMAND video_lora_vk_cli --selftest --shaders ${SHADER_OUT_DIR})
