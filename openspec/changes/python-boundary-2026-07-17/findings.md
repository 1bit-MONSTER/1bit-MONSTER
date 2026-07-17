# Findings — Python↔Native boundary, verified on hardware (2026-07-17)

Host: Strix Halo (gfx1151 + XDNA2). All results below are from direct build/run,
not inference from source reading. Reproduction commands included.

## 1. The runtime Python touch-points (what actually sits in the request path)

| File | Role | In LLM hot path? |
|---|---|---|
| `daemon/npu-cppd.py` | OpenAI HTTP API → NPU | **YES** — tokenizes per request AND spawns `python3 tools/npu_runner.py` (torch venv) as the decode loop |
| `unified-router.py` | keyword router / proxy in front of lemond | request path, but **glue only** (no tokenize, no inference) |
| `engine/lora/*.py`, `engine/npu/kernel/*.py`, `engine/fusion/hf_to_q4nx.py`, `engine/fusion/export_tokenizer.py`, `engine/fusion/tokenize.py` | training / model-convert / kernel-gen / tokenizer-export | **NO** — offline/build-time |

The C++/HIP engine core itself is Python-free, as branded. The leak is the **serving
layer**: one daemon still tokenizes in Python and runs its NPU decode loop in torch.

## 2. Native tokenizer — VERIFIED READY (bit-exact)

Source: `engine/fusion/tokenize.cpp` (pure C++17, no deps; C ABI + CLI).

```bash
c++ -std=gnu++17 -O2 -o /tmp/tokenize engine/fusion/tokenize.cpp
TOK=~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json
/tmp/tokenize --model "$TOK" --encode "Hello, world!"   # -> 9707 11 1879 0
/tmp/tokenize --model "$TOK" --decode 9707 11 1879 0     # -> Hello, world!
```

`9707 11 1879 0` is **identical** to the Python HF reference documented in
`engine/fusion/tokenize.py`. Supports ByteLevel/Metaspace/Split pre-tokenizers,
GPT-2 byte-level, BPE merge-ranks, byte fallback.

**C ABI available for direct linking:**
```c
void* tokenizer_load(const char* json_path);
int   tokenizer_encode(void* tok, const char* text, int* out_ids, int max_ids);
void  tokenizer_free(void* tok);
```

→ Ready to replace `src/server/rest_handler.cpp:518 simple_tokenize()` (a word-split
stub whose own comment says "In production, this would use the model's actual
tokenizer"), and to remove the `from tokenizers import Tokenizer` / tokenize-subprocess
paths from the daemons.

## 3. Native NPU decode engine — BUILDS + RUNS, but BLOCKED by 2 bugs

Source: `npu-infer/src/npu_engine_stdio.cpp` (per-token stdio stepper; protocol
`{"token":N}` / `{"continue":true}` / `{"reset":true}` → `{"token":M,"ms":X}`).

**Build (verified clean):**
```bash
gcc -std=c11 -O3 -march=native -ffast-math -c engine/npu/src/dequant_q4nx.c -o /tmp/dequant.o
c++ -std=gnu++17 -O3 -march=native -ffast-math -g -I npu-infer/include \
    -o /tmp/npu_engine_stdio npu-infer/src/npu_engine_stdio.cpp /tmp/dequant.o -lxrt_coreutil
```

**Run (verified on XDNA2):** expects xclbins in a relative `int8/` dir; hardcoded to
Qwen3-0.6B dims (H=1024, NC=28, NV=151936).

- With `int8_32tile_v6` xclbins → **NPU hang** (>120 s for 1 token). dmesg:
  ```
  amdxdna: AMD-Vi: IO_PAGE_FAULT ...
  aie2_config_cu: Lookup GEM object failed
  aie2_hwctx_restart: Config cu failed, ret -22
  ```
  Root cause class: NPU DMA to IOMMU-rejected addresses; `r.wait()` (line 44, no
  timeout) then blocks forever. Same failure family as the 2026-07-14 fusion audit.

- With base `int8_32tile` xclbins → **works**: `{"token":9707,"ms":2242}` then
  `free(): invalid size` (heap corruption on the C++ side).

**Fixes:**
1. **`free(): invalid size`** — FIXED (2026-07-17). ASan showed it is NOT a data heap
   bug: it's a null-vtable SEGV in `I8Ctx::~I8Ctx()` at exit, destroying
   `shared_ptr<xrt::xclbin_impl>`. Root cause = XRT static-destruction-order fiasco
   (XRT globals torn down before these local dtors run). Fix = `std::_Exit(0)` after
   the stdin loop, skipping the cross-boundary teardown (OS reclaims). Token gen itself
   was already correct; only shutdown crashed. Verified: exit 0, no ASan/glibc error.
   NOTE: output repeats token 9707 — separate sampling/RoPE bug (tracked T10).
2. **xclbin pinning** — the engine hardcodes `int8/` + `qwen3_0_6b` names; `v6`
   IOMMU-faults. Pin a known-good set (base `int8_32tile`) and add a `wait()` timeout
   so a bad kernel dispatch fails loudly instead of hanging.

Also: lm_head runs the full 151,936-vocab dot product on **CPU** (~2.2 s/token). That
belongs on GPU/NPU for a "perfect" engine.

**Alternative:** retarget the already-validated FLM engine (94 tok/s) instead of
hardening this orphan.

## 4. Router — PORTED TO RUST + TESTED

`unified-router.py` → std-only Rust (zero external crates, 427 KB static binary).
Reference impl in `reference/unified-router-rs/` in this change dir. Behavior verified
identical against a mock backend:

| Input | Python logic | Rust result |
|---|---|---|
| `auto` + plain greeting | → NPU (SMALL) | ✅ `qwen3-0.6b-FLM` |
| `auto` + "code" keyword | → GPU (BIG) | ✅ `Qwen3-8B-4bit` |
| `user.Unified` + tools | → GPU | ✅ |
| explicit `npu` / `gpu` | → SMALL / BIG | ✅ |
| unknown model | passthrough unchanged | ✅ |
| GET | passthrough | ✅ |

Recommended: fold this routing policy into the existing `rust/onebit` frontend rather
than shipping a second router binary long-term. The std-only version is the portable
reference / drop-in.
