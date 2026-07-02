/** Main16 scheduler kernel for Qwen3-0.6B — recompiled with 0.6B phase dimensions */
#define QWEN3_06B 1
#include "qwen3_constants_06b.h"
// Override namespace: use qwen3_06b constants in qwen3 namespace
// for compatibility with existing kernel source.
#undef qwen3
#define qwen3 qwen3
// Include the generic kernel implementation
#include "/home/bcloud/torch2aie/examples/qwen3-decode-layer/qwen3_decode_kernels.cc"
