# Context 4096 → 8191, unblocked: the fault was two 4096-sized host structures (2026-09-15)

`RESULTS-ctx8192-blocked-2026-09-15.md` recorded this as blocked, with the 8192
attention capture on disk and unwired, and left a list of next steps. The first
entry on that list — "is the capture structurally the same kernel?" — was already
answered (it is, exactly 2x). The second and third were wrong in an instructive
way: **the attention kernel was never involved.**

## The experiment that redirected it

Run the same 4200-token prefill with the **CPU attention reference** instead of
any capture. If the fault were in the kernel, this should be correct.

    Prefill 4200 [bf16]   253253 ms (60.298 ms/tok)
      [0] boot=49691        <-- the SAME wrong token the 8192 kernel gave
    double free or corruption (out)   <-- and the same corruption

Identical symptoms with no capture in the loop. The kernel was innocent, and the
fault was in the bf16 prefill path — i.e. in code that runs above 4096 for *any*
attention implementation.

## What was actually wrong: two structures sized for exactly 4096

**1. The RoPE table — a read overrun** (`npu_engine_universal.cpp`).

```cpp
ri(HD, cfg.rope_theta, 4096);      // rc/rs hold positions 0..4095
...
static inline void ra(float* x, int hd, int p) { ... rc[p*hd+d] ... }   // no bound check
```

`ra` is called with `p = sp + pi`, so a 4200-token prompt indexed `rc` up to
position 4199 — 13,184 floats past the end of the table. Wrong cos/sin, hence
wrong RoPE, hence a wrong answer. Read-only, so it corrupted nothing.

**2. The host K/V caches — a write overrun.**

```cpp
int kv_size = 4096*NKV*HD;
...
memcpy(&kv_caches[l][0].k[(sp+pi)*NKV*HD + kvh*HD], ks, HD*4);   // pi up to npt-1
```

For 4200 tokens that wrote ~105k floats past the end of **every layer's** cache —
28 overruns per run. That is the `double free or corruption (out)`, and because
the overrun lands on adjacent heap, the *victim* of the corruption depends on the
allocation layout rather than on the input — which is why the answer was the same
constant `49691` at 4200, 5000 and 7000.

Both now use the run's own length: a `ctx_need = max(4096, prompt + generated)`
computed once, before either is built. With the cap at its default the value is
4096 and nothing changes.

## Result: 4200 … 8191 all match FLM, on the NPU kernel

Boot token vs FLM's own runtime on the same tokens (nh16, Qwen3-0.6B):

| npt | native | FLM | |
|---:|---:|---:|---|
| 4200 | 5381 | 5381 | ✓ |
| 4600 | 220 | 220 | ✓ |
| 5000 | 30566 | 30566 | ✓ |
| 6000 | 220 | 220 | ✓ |
| 7000 | 1011 | 1011 | ✓ |
| 8191 | 59277 | 59277 | ✓ |

and the same prefill that took **252 seconds** on the CPU reference at 4200 now
takes **3.3 s** on the 8192 capture — 77x, which is what makes the range usable.

## The default now reaches 8191 for the nh16 shapes

The prompt cap is `8193 - ng` (the whole run has to fit the 8192-token KV window
the artifacts are baked for) when the shape is nh16 — the only shape with an 8192
capture. Verified with **no environment at all**:

| invocation | result |
|---|---|
| 0.6B, 5000 tokens, 2 decode | `Prefill 5000 [bf16]`, 3472 ms, boot 30566 = FLM |
| 0.6B, 8191 tokens, 2 decode | `Prefill 8191 [bf16]`, 4153 ms, boot 59277 = FLM |
| 1.7B, 5000 tokens, 2 decode | `Prefill 5000 [bf16]`, 4486 ms, boot 30566 = FLM |
| 4B (nh32), 5000 tokens | announced cap → `Prefill 4095 [bf16]` |

`NPU_PROMPT_MAX` overrides the cap either way.

The nh32 shapes (4B/8B/VL-4B) are left at 4095 because they lack both halves of
the 8192 support: no 8192 capture, and their region stride is 4096 tokens
(2097152 u16, which the existing nh32 attention captures are baked with), so the
K/V layout cannot address 8192 tokens. Lifting the cap for them without both
would move a correct-answer path onto a broken one.

Regressions checked after the change, all unchanged: bf16 gates
256/1024/2048/4095 → 1614/25/220/44353; the default at 1024 still takes the
auto-selected bf16+unified path; `NPU_RUNLIST=1` still takes the runlist path.

## A side effect this made prominent: generation writes into the checkout

Reaching 8191 by default means the unified decode needs per-context ELFs at
`npt+1 … npt+ng`, and the shipped sets stop at 2200 — so a long prompt now
generates a window on demand (round 6's `RT_ELF_GEN`/`RT_ELF_MODEL` auto-config).
The default ELF directory is `npu-infer/captures/txn-elfs*` **inside the
checkout**, so that wrote 1540 untracked files during this round's verification
(512 per long-prompt run, ~35 MB). They were removed again, and they are
regenerable, but the default should not write into a git working tree at all.

Until that is fixed, **set `NPU_LAYER_ELF_DIR` to a scratch directory for long
runs**, or accept the untracked files and `git clean` them. The fix is to point
generation at a cache directory (e.g. `~/.cache/1bit-monster/elfs/<model>/`,
seeded with symlinks to the shipped set) whenever the caller has not chosen an
ELF dir — the same place round 6's `ensure_elf_gen_env` already resolves the
generator from. It is the next hygiene item, not a correctness one: the tokens
and the timings above do not depend on where the ELFs live.

## Still open

- **nh32 to 8192** needs its own 8192 capture *and* a region-stride change for
  `H=2560/H=4096`, which the <=4096 nh32 captures are baked against — so both
  changes have to land together.
- **Beyond 8192** is unreachable: the layer ELFs are baked at `MAX_L=8192`
  (the generator asserts `L <= MAX_L + 1`), so 8192 tokens is the window.
- The 8192 capture is nh16-only. Its identification, like the others, is the
  interposer index (`elf_0012`) plus the affine size model (predicted 770067,
  captured 770064).

## Reproduce

```sh
export NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins
# no environment needed for nh16 up to 8191 now:
engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 2 ~/npu-build/parity/ids5000.txt
# force the old cap, or raise it for a shape that has no 8192 capture:
NPU_PROMPT_MAX=4095 ... / NPU_PROMPT_MAX=8191 ...
```
