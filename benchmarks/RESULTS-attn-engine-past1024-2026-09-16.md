# Generated attention driven through the engine past 1024 keys, gated

Goal `mu35shsg-i3hlyi`, task `attn-engine`. Date 2026-09-16 ~01:55 ADT.
Box: strixhalo. This closes "the engine-side run is not done" from the L1 record.

## The kernel: a generated N=2048 int8 attention, Zaya's shape

Zaya1-8B's attention is `nq=8, nkv=2, hd=128` (config.json), which is exactly the
default shape of `engine/npu/generators/n1_core_attn.py` (`-c 8` q heads,
`-b 2`). Built by the generator with `N=2048` (the shipped build is N=512;
the L1 lane's chunked build was N=1024):

```
cd ~/1bit-MONSTER-goal/engine/npu/generators
mkdir -p /tmp/attn_v4_2048
# build_attn.sh with the aiecc output paths retargeted to /tmp/attn_v4_2048
NPU_ATTN_N=2048 bash build_attn.sh        # -> /tmp/attn_v4_2048/attn.{xclbin,insts.txt}
```

Outputs: `attn.xclbin` 134096 B, `attn_insts.txt` 179760 B.

## Kernel gate: NPU == EMU at seq=2048

The bench that carries the L1 gate (`/tmp/ck XCLBIN INSTS SEQ`, `NPU_ATTN_EMU=1`
for the host contract):

```
NPU_ATTN_MAX_SEQ=2048 /tmp/ck /tmp/attn_v4_2048/attn.xclbin /tmp/attn_v4_2048/attn_insts.txt 2048 2
NPU_ATTN_EMU=1 NPU_ATTN_MAX_SEQ=2048 /tmp/ck /tmp/attn_v4_2048/attn.xclbin /tmp/attn_v4_2048/attn_insts.txt 2048 2
```

```
NPU: seq=2048  max_abs_err=5.552646e-02  C2 2/2 non-zero   ms_per_call=6.651
EMU: seq=2048  max_abs_err=5.552646e-02
```

**NPU max_abs_err == EMU max_abs_err, to the digit.** The kernel implements its
contract at 2048 keys.

## Engine run 1 — 1100-token prompt (past 1024), no clamp

`npu_engine_zr1` takes token ids as argv (the dedicated Zaya path):

```
cd ~/1bit-MONSTER-goal
NPU_ATTN=1 NPU_ATTN_MAX_SEQ=2048 \
NPU_ATTN_XCLBIN=/tmp/attn_v4_2048/attn.xclbin \
NPU_ATTN_INSTS=/tmp/attn_v4_2048/attn_insts.txt \
  ./engine/npu/build/npu_engine_zr1 /home/bcloud/models/zaya1-8b.q4nx $(tr '\n' ' ' </tmp/ids1100.txt)
```

```
AttnCtx: xp=/tmp/attn_v4_2048/attn.xclbin instr=44940 words
NPU attention ready (attn.xclbin, 20 layers, MAX_SEQ=2048)
[MoE L1 dbg] corr=0.999342 maxdiff=0.022679 (cpu rms=0.1930 npu rms=0.1923)
[EMB dbg] corr=1.0000000 maxdiff=0.000062 argmax 30334 vs 30334 (SAME)
[perf] 8 tokens in 1976 ms (247.0 ms/tok, 4.0 tok/s)      rc=0
```

No clamp warning: every attention call ran at `seq <= 2048`.

## Engine run 2 — the seq is real, and the window is the kernel's baked N

`zaya_decode.cpp` sets `seq = (KV length) + 1` before each `attn_ctx.run`, so the
prompt length must reach the kernel. Two facts from a 2100-token prompt with the
same kernel (`NPU_ATTN_DIAG=1 NPU_ATTN_DIAG_POS=1099`):

```
[ATTN L0 dbg] corr=1.000000 maxdiff=0.000000 (cpu rms=1.4290 npu rms=1.4290)
AttnCtx: WARN seq=2049 > MAX_SEQ=2048 — clamping (results wrong past the kernel's baked N)
```

- At position 1099 the NPU attention ran at **seq≈1100, past 1024 keys**, and its
  output matched the CPU scan **corr=1.000000, maxdiff=0.000000** — the gated
  output for the >1024 run.
- The `seq=2049 > MAX_SEQ=2048` clamp is the proof that the prompt length drives
  `seq` (it is not silently ignored), and it names the bound: the kernel's baked N.

## Verdict

**The generated chunked attention kernel is engine-driven past 1024 keys with
gated output.** The gate at seq≈1100 is attention corr = 1.000000 (NPU int8 vs CPU)
and the layer-level `[MoE L1 dbg] corr = 0.999342`; the kernel-level gate is
NPU == EMU at seq=2048. The remaining window limit is the baked N (2048 here, 4096
next), and past it the engine clamps — which is `task-beyond8192`'s subject, not a
hidden defect.

Artifacts: `/tmp/attn_v4_2048/attn.{xclbin,insts.txt}`; logs `/tmp/zr1_1100.log`,
`/tmp/zr1_2100.log`, `/tmp/ck2048_{npu,emu}.log` on the box; bench `ck` and the
generator are committed (`engine/npu/generators/n1_core_attn.py`, `build_attn.sh`).
