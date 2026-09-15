# The ctx-2200 cliff: the default engine dropped to 2 tok/s past the shipped ELF range (2026-09-15)

Found while checking what a *plain* invocation does at a long prompt, after two
rounds of measurements that always passed `NPU_LAYER_ELF_DIR` explicitly. The
answer was not what those rounds assumed.

## The cliff

The shipped per-context ELF sets (`npu-infer/captures/txn-elfs*`) stop at
**ctx 2200**. The runlist prefill walks one whole-layer forward per prompt token,
so a longer prompt needs the whole missing range, and the first absent context
fails the path:

```
RuntimeLayer: missing layer ELF for ctx=2201
[runlist] prefill forward ctx=2201 failed
[runlist] whole-layer path failed (rc=1); falling back to split path
=== Prefill 2500 [fallback] ===
=== 509.4 ms/tok (2 tok/s) | boot=7ms batches=1 tokens=2 ===
```

A 2500-token prompt on Qwen3-0.6B: **2 tok/s** on the 112-launch split path
against **57 tok/s** on the runlist path the same model gets inside the shipped
range. A **28× cliff**, on a model FastFlowLM serves at 32k, and every one of
the previous rounds' measurements would have missed it because they pointed the
engine at a 4200-context set in `~/npu-build`.

The fallback banner is printed, but the last line a reader sees is `2 tok/s` —
which reads as a property of the engine, not of a missing file.

## The fix

1. **`runtime_layer.cpp`** — the lazy build generated **one context per
   invocation**. It now generates a **window** (`RT_ELF_WINDOW`, default 256),
   because the cost is per-invocation, not per-context: measured **0.225 s for
   256 contexts in one call**, so a 300-context gap is two spawns instead of
   three hundred.

2. **`npu_runlist_bridge.cpp`** — `ensure_elf_gen_env()` fills in `RT_ELF_GEN`
   and `RT_ELF_MODEL` when the caller has not, resolving the generator from
   `/proc/self/exe`'s directory and then the source tree. An explicit setting
   always wins, and the run prints the command before executing it.

3. **`build_npu.sh`** — builds `gen_layer_elfs` into `engine/npu/build/`, the
   first place (2) looks, so a plain build needs no environment. It links
   `gemma_text_npu` for its family switch, which the engine's `LIBS` does not
   carry.

## Verified

Qwen3-0.6B, 2500-token prompt, **shipped-range ELF set** (max ctx 2200):

```
=== Prefill 2500 [runlist] ===
RuntimeLayer: generating missing ELFs ctx=2201..2456 (... 2201 2456 8192)
Prefill: 38601ms (15 ms/tok)
=== 9.0 ms/tok (111 tok/s) | tokens=2 ===
```

**2 tok/s → 111 tok/s**, and the run stays on the runlist path instead of
falling back. Four independent checks that this is correct and not merely alive:

| check | result |
|---|---|
| generated `layer_ctx2300.elf` vs the independently built 4200-context set | **byte-identical** (`bdeeed65f73cdb21`) |
| decode tokens, generated set vs verified 4200 set (2500-token prompt) | `19 13 16 24 220 29594` — identical |
| from a genuinely **empty** ELF dir | generated 1..2500 in ten windows, same result |
| true default (no env at all) at 2500 tokens | `Prefill 2500 [runlist]`, 111 tok/s |
| unified path at 4095 with the shipped set | works, boot/tokens `44353 16 4489 58907` = the verified run |

## Side effect worth knowing

Generation writes into the ELF directory, which by default is
`npu-infer/captures/txn-elfs/` **inside the checkout**. It is visible (the
command is printed) and regenerable in ~0.5 s, and the files created during
verification were removed rather than left untracked. A cache directory outside
the checkout is the tidy fix and is the prerequisite for any long-context work:
a 8192-context set is ~1 GB, which is fine in a cache and not fine in a git
working tree.

## What this leaves as the biggest remaining gap

The cliff is gone, but the **default is still the slow-prefill path**. The
runlist prefill costs ~15 ms/token (0.6B) so reaching 4095 takes 70 s against
FLM's 1.9 s — **36×**, and the same order at every size. The unified path
(`NPU_PREFILL_BF16=1 NPU_UNIFIED=1`) fixes that at 4k (see
`RESULTS-native-4k-decode-and-unified-2026-09-15.md`) and is now verified for all
five dense/VL models, but it is opt-in.

The crossover is worth stating, because it is not "always unified": at 1k,
runlist is 13.8 s prefill + 11.4 ms/tok decode against unified's 0.55 s + 13.8
ms/tok, so unified wins for any output shorter than ~2000 tokens and loses only
for a *short* prompt with a *very* long generation (10-token prompt, 1000 output
tokens: 11.5 s runlist vs 13.8 s unified). Making unified the default is
therefore a real decision with a threshold in it, not a free win — and it is the
next thing, together with the cache-dir change above.
