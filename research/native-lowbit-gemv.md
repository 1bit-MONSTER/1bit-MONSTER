# Native Low-Bit GGUF GEMV — program roadmap

> Branch: `feat/native-lowbit-gemv`. Goal: run **every** low-bit GGUF weight
> format with **native dequant-in-kernel GEMVs** on the custom HIP lane
> (gfx1151 Strix Halo) — no CPU f32 dequant round-trip, no conversion to a
> project format. "Native" = the GPU kernel decodes the GGUF block layout
> itself; the loader memcpys raw blocks to device.
>
> Rule of the program: **every format lands with a bit-exact selfcheck
> (kernel-dequant port vs vendored llama.cpp `dequantize_row_*`) as its first
> commit** — the `Testing/iq1_selfcheck.cpp` pattern. No kernel without its
> harness.

## Why this exists

`rcpp_bitnet_load_gguf` (the GGUF-direct loader in `src/gguf_loader.cpp`)
today dequants every tensor to f32 and uploads fp16 — CPU round-trip, and the
native kernels (`q4k_gemv.hip`, `iq_gemv.hip`, `bitnet_gguf_gemv.hip`) were
built ahead of the loader/dispatch work that would feed them. `q4k_gemv.hip`
is not even in CMakeLists. The dispatch in `tools/bitnet_decode.cpp`
(weight_format → gemv fn) is the seam all of these plug into.

## Status matrix

Legend: selfcheck = `Testing/*_selfcheck.cpp` bit-exact vs llama.cpp ref.
kernel = dequant-in-GEMV `.hip`. loader = keep-packed GGUF branch.
dispatch = weight_format value + bitnet_decode arm.

| GGUF dtype | bpw | block (B) | kernel | selfcheck | loader | dispatch | status |
|---|---|---|---|---|---|---|---|
| TQ1_0 (34) | 1.69 | 54 | `bitnet_gguf_gemv.hip` | — | — | ✅ | partial — loader missing |
| TQ2_0 (35) | 2.06 | 66 | `bitnet_gguf_gemv.hip` | — | — | ✅ | partial — loader missing |
| Q1_0 (41) | 1.0 | 18 | `ternary_gemv_q1_0.hip` | — | — | ✅ | partial |
| IQ1_S (18) | 1.56 | 50 | `iq_gemv.hip` | ✅ (`iq1_selfcheck`) | ❌ | ❌ | kernel done, unwired |
| IQ1_M (23) | 1.75 | 56 | `iq_gemv.hip` | ✅ | ❌ | ❌ | kernel done, unwired |
| Q4_K (12) | 4.5 | 144 | `q4k_gemv.hip` | ✅ (`q4k_selfcheck`) | ❌ | 🔄 | kernel fixed + wired to CMake/launcher/enum; loader+dispatch next |
| Q2_K (10) | 2.56 | 84 | 🔲 | 🔲 | ❌ | ❌ | — |
| Q3_K (11) | 3.44 | 110 | 🔲 | 🔲 | ❌ | ❌ | — |
| Q5_K (13) | 5.06 | 176 | 🔲 | 🔲 | ❌ | ❌ | — |
| Q6_K (14) | 6.56 | 210 | 🔲 | 🔲 | ❌ | ❌ | — |
| Q8_0 (8) | 8.0 | 34 | 🔲 | 🔲 | ❌ | ❌ | — |
| IQ4_NL (19) | 4.5 | 22/32el | 🔲 | 🔲 | ❌ | ❌ | — |
| IQ4_XS (22) | 4.25 | 214 | 🔲 | 🔲 | ❌ | ❌ | — |
| IQ2_XXS (16) | 2.06 | 166 | 🔲 | 🔲 | ❌ | ❌ | — |
| IQ2_S (21) | 2.7 | 214 | 🔲 | 🔲 | ❌ | ❌ | — |
| IQ3_XXS (17) | 3.44 | 198 | 🔲 | 🔲 | ❌ | ❌ | — |
| IQ3_S (20) | 3.5 | 238 | 🔲 | 🔲 | ❌ | ❌ | — |
| MXFP4 (39) | 4.25 | 208 | 🔲 | 🔲 | ❌ | ❌ | — |

GGUF dtype numbers above are the project's `GGUF_DTYPE_*` enum values from
`include/gguf_reader.h` (fork numbering for BF16 etc.), NOT llama.cpp's
`GGML_TYPE_*`.

## Block layouts — source of truth

All layouts are taken from the **vendored** `third_party/llama.cpp/ggml/src/ggml-common.h`
+ `ggml-quants.c` (dequantize_row_*), which is the bit-exact reference each
selfcheck cross-checks against.

### Q4_K — `block_q4_K`, 256 el / 144 B
```
ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qs[128];
per 64-el group (j += 64): get_scale_min_k4(is, scales, &sc, &m):
  is < 4 : sc = scales[is]&63;  m = scales[is+4]&63
  is >= 4: sc = (scales[is+4]&0xF) | ((scales[is-4]>>6)<<4)
           m  = (scales[is+4]>>4) | ((scales[is]>>6)<<4)
d1 = d*sc; m1 = dmin*m;  y = d1*(qs[..]&0xF) - m1 | d2*(qs[..]>>4) - m2
```

### Q6_K — `block_q6_K`, 256 el / 210 B
```
ggml_half d; uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16];
scales[0..15] map to 8 scale bytes: get_scale_min_k4-style (sc = scales[is]&... | ...)
ql: 4-bit low; qh: 2-bit high;  x = (ql | qh<<4) - 32;  y = d * sc * x
```

### Q8_0 — `block_q8_0`, 32 el / 34 B
```
ggml_half d; int8_t qs[32];  y = d * qs[i]
```

### Q2_K — `block_q2_K`, 256 el / 84 B
```
ggml_half d; ggml_half dmin; uint8_t scales[16]; uint8_t qs[64];
scales: 4-bit per 32-el sub-block (2 per byte), m from high nibbles (get_scale_min_k4)
qs: 2-bit codes, 4 per byte; y = d*sc*(qs) - dmin*m
```

### Q3_K — `block_q3_K`, 256 el / 110 B
```
ggml_half d; uint8_t qs[64]; uint8_t qh[32]; int8_t scales[12];
get_scale_min_k4; q = qs | (qh<<2) sign-extended 4-bit; y = d*sc*q - dmin*m
```

### Q5_K — `block_q5_K`, 256 el / 176 B
```
ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128];
y = d*sc*( (qs&0xF) | ((qh bit)<<4) ) - dmin*m
```

### IQ2_XXS — 256 el / 166 B — 8x `uint16_t` grid idx (11-bit LUT) + fp16 d + scales[2]
### IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S — grid-LUT + 4-bit/3-bit codes, superblock scales
### IQ4_NL — 32 el / 22 B — 4-bit codebook (LUT grid), fp16 d
### IQ4_XS — 256 el / 214 B — grid + per-subblock 6-bit scales
### MXFP4 — 32 el / 208 B — block_mxfp4: uint8_t x[32]; uint8_t e8m0_bits_n[4]; fp32 scale
(layouts pinned in the selfcheck for each format at implementation time —
never transcribed by hand into kernels without the selfcheck gate.)

## Dispatch seam

`tools/bitnet_decode.cpp` — `ternary_gemv_i8` lambda (weight_format →
kernel fn, ~line 558) is the model-run dispatch; `RCPP_WEIGHT_FORMAT_*` enum
lives in `include/rocm_cpp/bitnet_model.h`. Native-format weights need:
1. New `RCPP_WEIGHT_FORMAT_*` enum values (e.g. `RCPP_WEIGHT_FORMAT_Q4_K`).
2. `rcpp_bitnet_load_gguf` keep-packed branch per dtype (memcpy raw GGUF
   block bytes to device, set format) instead of dequant-to-fp16.
3. Dispatch arm in `bitnet_decode` + launcher in
   `src/ternary_gemv_launchers.hip` + decl in `include/rocm_cpp/ck_gemm.h`.
4. `kernels/<fmt>_gemv.hip` in `CMakeLists.txt` RCPP_SOURCES.

## Build order (each = selfcheck commit → kernel/wiring commit)

1. Q4_K — orphan kernel exists; selfcheck vs `dequantize_row_q4_K`, CMake,
   enum, keep-packed loader, dispatch.
2. IQ1_S/IQ1_M — kernels + selfcheck exist; enum + loader + dispatch only.
3. Q6_K + Q8_0 — same QK_K=256 superblock family; most-used "safe" layers.
4. Q2_K/Q3_K/Q5_K — complete the K-family.
5. IQ4_NL/XS — codebook formats, very popular on HF.
6. IQ2_XXS/XS/S + IQ3_XXS/S — grid-LUT formats (hardest; IQ1 grid precedent).
7. MXFP4 — new block-fp4 format.

## Verification protocol

- Selfcheck: plain `g++` (no GPU), random + edge-case blocks, bit-exact float
  compare vs the vendored llama.cpp reference (like `Testing/iq1_selfcheck.cpp`).
- Kernel compile: `/opt/rocm-therock/bin/hipcc` for gfx1151.
- End-to-end: real Q4_K GGUF on disk (`~/models/Qwen3-0.6B-Q4_K_M.gguf`) →
  `gguf_to_onebp`/bitnet_decode native path, top-token parity vs CPU ref.
