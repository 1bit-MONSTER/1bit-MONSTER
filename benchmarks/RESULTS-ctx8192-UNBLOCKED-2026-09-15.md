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

**FIXED (same day).** `ensure_elf_gen_env` now sends generated contexts to a
cache directory — `$XDG_CACHE_HOME/1bit-monster/elfs/<model>/`, or
`~/.cache/1bit-monster/elfs/<model>/` — whenever the caller has not set
`NPU_LAYER_ELF_DIR`, which is still never second-guessed. The shipped set is
symlinked into the cache once as the baseline, so a hit costs nothing and a miss
still generates.

Verified: a default 5000-token run reports

    [runlist] generated ELFs go to /home/bcloud/.cache/1bit-monster/elfs/Qwen3-0.6B-NPU2
    RuntimeLayer: generating missing ELFs ctx=5001..5256 (... 5001 5256 8192)

and `git status npu-infer/captures/` is **empty** afterwards, where the same run
previously left 512 untracked pairs. The cache is 52 MB for that model. An
explicit `NPU_LAYER_ELF_DIR` still suppresses the redirect entirely.

## nh32 too: 4B / 8B / VL-4B reach 8191

The nh16 shapes got the range first. nh32 shares the tokenizer and layer geometry
(H=2560 for 4B and VL-4B, H=4096 for 8B) but needed both halves of the support
before the cap could be raised for them:

- **The capture.** `attn_mha_8192_nh32.elf`, 1400848 B, sha256
  `20fd31b373187c9c…`, taken from Qwen3-4B with the same harness. Identified the
  same way — `elf_0012`, and the nh32 affine size model (2960 + 170.641·L)
  predicts 1400851 against 1400848, the four committed nh32 captures landing at
  −0.4 / −0.8 / −1.5 / −3.1 B.
- **The region stride.** The engine's H table gives H=2560/H=4096 a 2097152 u16
  region because the ≤4096 nh32 captures were taken at `MAX_L=4096`. Above 4096
  the stride must be 4194304, to match the 8192 capture; with the narrow one the
  kernel reads past the region, which is wrong rather than slow. The stride now
  follows the capture that will run.
- **The cap.** `8193-ng` now applies to any shape whose 8192 capture is on disk —
  a file-existence check, not a shape list, so a missing capture degrades to the
  old 4095 cap rather than to an eight-minute CPU-attention prefill.

Verified against FLM's own runtime on the same tokens:

| model | length | native | FLM |
|---|---:|---:|---:|
| Qwen3-4B | 4200 / 5000 / 7000 / 8191 | 5381 / 30566 / 1011 / 59277 | same |
| Qwen3-8B | 5000 / 8191 | 30566 / 59277 | same |
| Qwen3-VL-4B | 4095 / 4200 / 4600 / 4700 / 4800 / 4900 / 5100 / 5200 | 59277 / 5381 / 220 / 9628 / 25 / 15 / 220 / 13 | same |

Held: nh16 gates 256/1024/2048/4095 → 1614/25/220/44353; nh32 gates (4B
1024/2048/4095) → 220/220/59277; and `git status npu-infer/captures/` stays
empty, so the ELF cache redirect holds through all of it.

### One position disagrees, and it is written down rather than smoothed over

**Qwen3-VL-4B at exactly npt=5000 returns 30566 where FLM returns 11211**, both
deterministic across repeats. It is **not** a tie: native's own margin there is
**2.625 logits**, with 11211 as the runner-up.

Two things point to numeric drift at a near-degenerate position rather than a
defect. VL-4B matches FLM at every other length tested — 4095, 4200, 4600, 4700,
4800, 4900, 5100, 5200 — including both sides of 5000. And VL-4B and 4B have
**identical** configs (H 2560, 36 layers, 32/8 heads, head_dim 128, IM 9728,
rope_theta 1e6, no rope_scaling) on a prompt that is `ids1024` repeated five
times: the two models agree with each other, and with FLM, at 4900 and 5100.
Only VL-4B at 5000 splits, and 4B at 5000 returns native's answer in both
implementations.

**The control was run, and native is self-consistent.** The same experiment that
localised the >4096 fault — swap the attention for the CPU reference and see
whether the answer moves — gives, for VL-4B at npt=5000:

| path | token |
|---|---:|
| native bf16 prefill + NPU attention | **30566** |
| native bf16 prefill + **CPU attention reference** | **30566** |
| FLM's own runtime | 11211 |

Two independent native attention implementations agree, so **the attention step is
not the variable** and this is not a native attention defect. What remains is
FLM-vs-native numeric drift at a position where native's top-2 are 2.625 apart —
and the disagreement is with a model whose config is identical to 4B's, at one
length out of nine tested, while 4B at that same length returns native's answer in
*FLM* as well.

(For the record the CPU run is expensive at this length: 952863 ms, 190.6 ms/tok,
16 minutes for a 5000-token prefill — which is why it belongs behind an
environment variable and not in a default.)

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
