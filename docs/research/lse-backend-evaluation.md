# LSE (Lemon Seed Engine) as a 1bit-MONSTER backend — evaluation

**Date:** 2026-08-29 · **Box:** strixhalo (AMD Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`, ROCm 10.1 TheRock, HRX runtime)
**Engine:** LSE v0.2.1 (build `c370793`+census fix), HRX backend, JIT kernels via comgr
**Model:** `mlx-community/Qwen3.6-35B-A3B-4bit` (MLX group-affine 4-bit, 19.1 GB, 40 layers, 256 experts top-8, GDN hybrid)

## TL;DR

LSE loads the Qwen3.6-35B-A3B MLX checkpoint **natively, zero conversion**, and
wins decode throughput vs our NPU FastFlowLM stack **at short context
(~2.5×)**, but decode decays steeply with context and loses to the NPU beyond
~1.5–2k. It is a *complementary backend*, not a replacement. The strongest fit
is `backend_lse` as a new router slot for MLX-format / Qwen3.5-family GPU
inference at short context.

## Measured (LSE, warm JIT cache)

| Context (tok) | Decode tok/s | Prefill tok/s |
|---:|---:|---:|
| ~10 | 28.5–29.7 | — |
| 125 | 22.05 | 76.7 |
| 537 | 16.72 | 122.6 |
| 1106 | 10.68 | 129.7 |
| 2292 | 6.52 | 130.8 |
| 3414 | 4.66 | 133.2 |

**2026-08-29 (backend_lse, through the 1bit server):** lemonseed-1.5b MLX
short-context decode ≈ 36 tok/s end-to-end via `POST /v1/completions`
(32 tokens in 0.88 s incl. HTTP overhead) — consistent with the eval's
~102 tok/s direct-engine figure for the 1.5B at near-zero context after
server/JSON overhead. Recorded in `site/benchmarks.json` as `lse_gpu`.

NPU FastFlowLM v0.9.46 baseline (same box, `q4nx` weights): decode 11.66 @1k,
12.17 @2k, 11.85 @4k, 11.30 @8k, 8.82 @32k; prefill 98–266 tok/s.

**Cross-over:** LSE decode ≈ NPU at ~1–1.5k context; NPU wins beyond.
LSE prefill ≈ NPU (slightly below at long context).

**Why decode decays:** the 5 full-attention layers (interval 4) pay O(context)
per step with a growing KV cache; decode is launch-bound (111k kernel groups).
This is the area with headroom (fusion / sliding-window / flash path).

**Cold-start note:** first load JIT-compiles kernels per shape (143–312
compiles ≈ 50–125 s). Disk cache (`~/.cache/lse`) makes subsequent runs free;
the engine also emits HIP sources under `build/hip` for inspection.

## Coverage matrix (our data vs LSE kernels)

| Our asset | LSE loads? | Note |
|---|---|---|
| Qwen3.6-35B-A3B MLX 4-bit | **Yes** | `qwen3.5-moe` kernel, verified end-to-end |
| lemonseed-1.5b (MLX) | **Yes** | ~102 tok/s decode (short ctx) |
| Qwen3.5/3.6/3.8 dense MLX | Yes | `qwen3.5` kernel (GDN hybrid family) |
| Qwen3-0.6B/4B/8B MLX 4-bit | **No** | verified: no GDN marker → "no registered architecture" |
| Zaya 1-8B (1BP/Q4NX/GGUF) | No | arch + format unsupported |
| Qwen2.5 GGUF / any GGUF | No | LSE reads MLX group-affine + bf16/f16/f32 only |
| BitNet/Bonsai ternary TQ2 | No | no ternary kernels |
| NPU routes (xclbin, FLM) | n/a | LSE has no NPU backend |

## Integration design (recommended)

Add `backend_lse` to the existing `BackendManager` router as a new route:

- **Process boundary (recommended):** spawn `lse` / `lse-server` subprocess,
  text-level via `generate_text`/`continue_text` — the same pattern as
  `backend_npu_flm`. Keeps 1bit-MONSTER on C++23/g++/clang/msvc; quarantines
  LSE's g++-16 + C++26 + `-freflection` + `std::meta` (unstable API) risk.
- Route priority: MLX-format checkpoints of supported archs → `lse` first
  (only backend that can read MLX today); other formats unchanged.
- Perf selection: use LSE for decode at short context (<1k), fall back to NPU
  for long context, or accept the decay per workload.

## Blockers / risks

1. **License:** LSE is custom non-commercial (Exhibit A: AMD, Lemonade SDK).
   1bit-MONSTER is MIT. Embedding needs a written license from Geramy Loveless.
2. **Runtime deps:** LSE needs ROCm + HRX + comgr at runtime; not suitable for
   `-pi` / `-iso` / NPU-only targets (keep pre-built kernels there).
3. **Toolchain:** C++26 + P2996 reflection behind `-freflection`; g++-16 only
   (clang/MSVC have no P2996); `std::meta` API has drifted between g++-16
   snapshots (observed `member_count_of` removal).
4. **MTP:** checkpoint config declares `mtp_num_hidden_layers: 1` but MLX 4-bit
   ships no MTP weights → speculative decode unavailable with this artifact.
5. **Vision:** 333 vision tensors refused (text-only build) — fine for text.

## Recommendation

Proceed with `backend_lse` as an **additive** backend (subprocess, MLX lane,
short-context decode) — it is the only backend in our tree that can read the
MLX 4-bit ecosystem, and it beats the NPU 2.5× where it wins. Do **not**
replace existing backends: NPU owns long-context decode and the constrained
targets; GGUF/1BP/Zaya routes are unaffected. Gate on the license question
first.
