// hip_math_compat.h — Redirect standard math functions to HIP device built-ins.
//
// When <cmath> is included in HIP device code (-x hip), its declarations are
// marked __host__ and cannot be called from __global__ functions.  The ROCm
// device library provides equivalent __ocml_* intrinsics.
//
// This header defines inline wrappers that dispatch to the correct
// implementation depending on context (device vs host).  Include it
// AFTER all other includes in .hip files that have this issue.

#pragma once

#ifdef __HIP_DEVICE_COMPILE__

// Device built-in redirects via __ocml_*
__device__ inline float  _hip_expf(float x)   { return __ocml_exp_f32(x); }
__device__ inline float  _hip_exp2f(float x)  { return __ocml_exp2_f32(x); }
__device__ inline float  _hip_logf(float x)   { return __ocml_log_f32(x); }
__device__ inline float  _hip_log2f(float x)  { return __ocml_log2_f32(x); }
__device__ inline float  _hip_sqrtf(float x)  { return __ocml_sqrt_f32(x); }
__device__ inline float  _hip_rsqrtf(float x) { return __ocml_rsqrt_f32(x); }
__device__ inline float  _hip_powf(float x, float y) { return __ocml_pow_f32(x, y); }
__device__ inline float  _hip_sinf(float x)   { return __ocml_sin_f32(x); }
__device__ inline float  _hip_cosf(float x)   { return __ocml_cos_f32(x); }
__device__ inline float  _hip_tanf(float x)   { return __ocml_tan_f32(x); }
__device__ inline float  _hip_fabsf(float x)  { return __ocml_fabs_f32(x); }
__device__ inline float  _hip_fmaxf(float x, float y) { return __ocml_fmax_f32(x, y); }
__device__ inline float  _hip_fminf(float x, float y) { return __ocml_fmin_f32(x, y); }
__device__ inline float  _hip_roundf(float x) { return __ocml_round_f32(x); }
__device__ inline float  _hip_floorf(float x) { return __ocml_floor_f32(x); }
__device__ inline float  _hip_ceilf(float x)  { return __ocml_ceil_f32(x); }
__device__ inline float  _hip_fmaf(float x, float y, float z) { return __ocml_fma_f32(x, y, z); }
__device__ inline float  _hip_tanhf(float x)  { return __ocml_tanh_f32(x); }

// These names are available in both device and host contexts.
// In device code (_hip_* calls __ocml_*), in host code (regular <cmath>).

#else
// Host-only fallback: use standard <cmath> functions.
#include <cmath>

__host__ inline float  _hip_expf(float x)   { return std::expf(x); }
__host__ inline float  _hip_exp2f(float x)  { return std::exp2f(x); }
__host__ inline float  _hip_logf(float x)   { return std::logf(x); }
__host__ inline float  _hip_log2f(float x)  { return std::log2f(x); }
__host__ inline float  _hip_sqrtf(float x)  { return std::sqrtf(x); }
__host__ inline float  _hip_rsqrtf(float x) { return 1.0f / std::sqrtf(x); }
__host__ inline float  _hip_powf(float x, float y) { return std::powf(x, y); }
__host__ inline float  _hip_sinf(float x)   { return std::sinf(x); }
__host__ inline float  _hip_cosf(float x)   { return std::cosf(x); }
__host__ inline float  _hip_tanf(float x)   { return std::tanf(x); }
__host__ inline float  _hip_fabsf(float x)  { return std::fabsf(x); }
__host__ inline float  _hip_fmaxf(float x, float y) { return std::fmaxf(x, y); }
__host__ inline float  _hip_fminf(float x, float y) { return std::fminf(x, y); }
__host__ inline float  _hip_roundf(float x) { return std::roundf(x); }
__host__ inline float  _hip_floorf(float x) { return std::floorf(x); }
__host__ inline float  _hip_ceilf(float x)  { return std::ceilf(x); }
__host__ inline float  _hip_fmaf(float x, float y, float z) { return std::fmaf(x, y, z); }
__host__ inline float  _hip_tanhf(float x)  { return std::tanhf(x); }

#endif // __HIP_DEVICE_COMPILE__
