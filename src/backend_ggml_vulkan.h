#pragma once
// backend_ggml_vulkan.h — llama.cpp Vulkan backend wrapper.
// Uses ggml-vulkan (MIT License) via llama.cpp API for high-performance inference.
// The factory self-stubs when llama.h is unreachable (CI builds without the
// submodule), so unified_server links unconditionally.

#include "backend.h"

#ifdef __has_include
#  if __has_include("llama.h")
extern "C" Backend* create_ggml_vulkan_backend();
#  else
static inline Backend* create_ggml_vulkan_backend() { return nullptr; }
#  endif
#else
extern "C" Backend* create_ggml_vulkan_backend();
#endif
