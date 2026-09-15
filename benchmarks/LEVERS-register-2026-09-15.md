# Levers register — matching FastFlowLM, broken into small wins (2026-09-15)

One page to answer "how do we get to FLM parity, and what is left". Every number
here is measured on this box and reproducible from the command in the row; a
number without a command is a claim, and this file does not carry those.

## 1. Where the objective stands

Supported set: **Qwen3-0.6B / 1.7B / 4B / 8B, Qwen3-VL-4B, Llama-3.1-8B**. Within
the KV window (1…8191 tokens) native meets-or-beats FLM on prefill, TTFT and
decode; beyond 8192 neither the layer ELFs nor the attention captures exist.

| model | native prefill | FLM prefill | native dec | FLM dec | tokens |
|---|---:|---:|---:|---:|---|
| 0.6B @1k | **591 ms** | 744 ms | 13.3 ms/tok | 13.4 | 25 220 16 13 |
| 0.6B @4095 | **1700 ms** | 1891 ms | **19.1** | 20.3 | = FLM |
| 4B @4095 | **5556 ms** | 6464 ms | **58.6** | 64.2 | 7/8 (step 8 a tie) |
| 8B @4095 | **8070 ms** | 9029 ms | **94.2** | 104.3 | 8/8 |
| VL-4B @4095 | **5566 ms** | 6528 ms | **58.5** | 64.2 | 8/8 |

Run any row with the default invocation — no environment — against
`NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1` for FLM's own compute.

## 2. Small wins already landed

Each was a small, independently verified change. Ordered as they happened.

| # | win | measured effect | command / where |
|---|---|---|---|
| 1 | 4096-context attention captures (nh16 + nh32) | 2049–4095 served by the NPU instead of a **240614 ms** CPU attention pass: **142×** at 0.6B/4095 | `RESULTS-native-4096-context-2026-09-15.md` |
| 2 | the ctx-2200 cliff | a prompt past the shipped ELF set ran at **2 tok/s**; now **111 tok/s** (28×) | `RESULTS-ctx2200-cliff` |
| 3 | `NPU_UNIFIED` was unreachable (the runlist block returned first) | makes the fast-prefill+fast-decode combination reachable at all: prefill 25×/40× | `RESULTS-native-4k-decode-and-unified` |
| 4 | the unified KV handoff used the session's region stride | nh32 decode was wrong from the first step; now 8/8 tokens | same doc |
| 5 | `RT_ARGMAX_MARGIN` | turned "mixed fidelity" into "these are bf16 ties" — 6 disagreements, every gap ≤ 0.25 logits vs 0.875–4.375 where they agree | `RESULTS-token-disagreements-are-ties` |
| 6 | the default path picks the fast prefill | TTFT **13353 → 591 ms** at 1k (22.6×) and **69109 → 1700 ms** at 4k (40.6×); the default now beats FLM on prefill/TTFT | `RESULTS-default-path` |
| 7 | two 4096-sized host structures (RoPE tables, host KV caches) | all of >4096 was broken — wrong token *and* heap corruption; 4200 now correct and **252 s → 3.3 s** (77×) | `RESULTS-ctx8192-UNBLOCKED` |
| 8 | nh32 8192 capture + region stride + cap | **all six models reach 8191**; 4B/8B/VL-4B match FLM at 4200…8191 | same doc |
| 9 | generated ELFs go to `~/.cache/1bit-monster/elfs/` | long prompts stopped writing 512 untracked files per run into the checkout | `5d0b84535` |

The pattern worth keeping: **seven of the nine were not optimisations.** They were
a missing artifact, a shadowed code path, a wrong constant, or a silent fallback.
The engine's ceiling was already high; what it lacked was *being driven*.

## 3. Levers still on the table

Ranked by (value to the objective) / (effort). Each has the gate that decides
whether it is worth continuing — no lever here should be started without its gate.

### L1 — fix the softmax arity and measure the generated attention *(route past 8192)*
`n1_core_attn.py:134` hard-wires **four** C1 tiles; `n_n = N // 128`, so **N=512 is
the largest N the generator can emit** and the `-N` flag is not a parameter.
**Fix:** generalise the operand list to `n_n` tiles, confirm the AIE softmax kernel
accepts that arity.
**BLOCKED, and not by the arity.** A control build at the *default* N=512 fails
too (`design.mlir:712:44: error: expected ')'`), so the first blocker is
**toolchain drift**: the installed mlir-aie carries a local WIP patch
(`1e6b70af0`, 2026-08-29, `AIELowerDynamicBDPool + BdLowering`) whose emitted
`aie.dma_bd` uses the old positional form
(`aie.dma_bd(%arg0 : memref<32768xi8>, 0, 512, [<size = 1, …>])`) while the current
dialect wants `offset =`/`len =`. The shipped xclbin predates the patch; the
`aiecc` binary (2026-09-12) does not. **That is another repository with another
lane's WIP on it — it needs a decision, not a revert from here.**
**Gate:** with the toolchain unblocked — build `NPU_ATTN_N=1024` and time it
against the captured `attn_mha_1024_nh16.elf` on the same shape and prompt. Within
~1.5× of the capture, N=16384 is the route past 8192; another 1200× (this repo has
that precedent for a generated attention ELF) closes the lever.

### L2 — a Nanbeige nh20 capture at 4096 *(the one non-dense family already close)*
Nanbeige's default i8 path matches FLM exactly (`1033 @1024`, `5938 @256`); only
its **bf16 arm** falls to the broken nh20 kernel. A correct nh20 capture is a
drop-in file by the existing shape naming — the same one-file change as L1's
neighbours, for a family that is otherwise already at parity.

### L3 — `kv_quant.h`: INT4 KV *(storage term, not a quick win)*
Declared, never implemented; its doc claims 6.4× KV memory and 16K in <600 MB. It
does **not** by itself lift the 8192 window (the kernels bake the layout), so it is
only worth starting *after* L1 decides whether the generated attention is viable —
with which it composes.

### L4 — the fused dense FFN (`NPU_QWEN_I4` + `NPU_GUSILU_BF16PAIR`)
A fused int4 GU→SiLU for the dense FFN, self-described as opt-in "until its
per-weight fused corr gate passes". It targets the int8 split path, which nothing
uses by default now, so its ceiling is low — but the flags are a **pair** and the
gate is named, so it is a bounded piece of work for whoever wants it.

### L5 — the second toolchain install *(250 GB of "still needed?")*
`Xilinx2025/2025.2` (75 GB) is *required* — the kernel builders compile against its
`Vitis/aietools/include`, and `aie_clang++` exists only there. `Xilinx/2026.1`
(142 GB) supplies `aiecc`'s PATH. Both are live; neither is a cleanup candidate.

## 4. The Vitis install — repaired, and how to tell

Ten-plus builders under `engine/npu/generators/` compile AIE kernels with
`-I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include`, **a path that does not
exist**: the 2025.2 install lives at `/home/bcloud/Xilinx2025/2025.2`. Nothing at
runtime notices, because the xclbins are prebuilt — which is why it stayed
undiscovered.

**Repair (done):** `/home/bcloud/Xilinx/2025.2 -> /home/bcloud/Xilinx2025/2025.2`,
a symlink. It fixes every reference at once, touches no script, and is reversible.

**Verification — three checks, not one:**

| check | result |
|---|---|
| every 2025.2 path the repo references resolves | `/home/bcloud/Xilinx/2025.2`, `…/Vitis/aietools/include` — both present |
| an AIE kernel compiles through them | `clang++ --target=aie2p-none-unknown-elf -I …/aietools/include -c` → `t.o`, 684 B |
| toolchain binaries present | `v++`, `vitis`, `xchesscc`, `aiecompiler`, `aiebu-asm`, `aie_clang++` all present in 2025.2; 2026.1 has all but `aie_clang++` |
| licence | `/home/bcloud/.Xilinx/Xilinx.lic`, `XILINXD_LICENSE_FILE=/home/bcloud/.Xilinx` |

`settings64.sh` is **missing** from the 2025.2 install and from the 2026.1 top
level (only `Xilinx/2026.1/Vitis/settings64.sh` exists). That is why every script
here sets `PATH`/`LD_LIBRARY_PATH` explicitly instead of sourcing one — and why the
repair is a path fix rather than a reinstall.

## 5. Instruments to measure the next thing

| instrument | what it answers |
|---|---|
| `RT_ARGMAX_MARGIN=1` | is a token disagreement a tie? prints top-2 logits and the gap per step |
| `NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1` | FLM's own compute in this harness — the reference side of every table |
| `NPU_ATTN_CPU=1` | the independent attention reference; the control that localised the >4096 fault |
| `NPU_RUNLIST=1` | the byte-exact int8 path (FLM's own per-ctx layer kernels, driven from here) — the third opinion |
| `NPU_PROMPT_MAX=<n>` | raise the prompt cap deliberately, for lengths without a fast path |
| `NPU_ELF_DEBUG=1` | show where generated per-context ELFs land |
