#pragma once
// backend_ggml_vulkan.h — llama.cpp Vulkan backend wrapper.
// Uses ggml-vulkan (MIT License) via llama.cpp API for high-performance inference.
//
// The factory self-stubs to nullptr unless the including target was built with
// GGML_VULKAN_LINKED (defined by CMakeLists.txt only inside if(TARGET
// ggml_vulkan), i.e. when third_party/llama.cpp was pre-built with
// GGML_VULKAN=ON). An absent backend must be a runtime "unavailable" (the
// dlsym/fallback design in backend_manager.cpp), never a link error.
//
// Do NOT key the stub off __has_include("llama.h"): recursive-submodule CI
// clones llama.cpp's headers without ever building its static libs, so header
// presence says nothing about whether the backend was compiled in — that
// mismatch produced `undefined symbol: create_ggml_vulkan_backend` when
// linking onebin.

#include "backend.h"

#ifdef GGML_VULKAN_LINKED
extern "C" Backend* create_ggml_vulkan_backend();
#else
static inline Backend* create_ggml_vulkan_backend() { return nullptr; }
#endif
