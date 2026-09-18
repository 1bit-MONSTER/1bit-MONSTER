# Family attention: the generated nh20/nkv4 head-block kernel fails its gate

Goal `mu35shsg-i3hlyi`, tasks `family-nanbeige` / `family-host` (prerequisite
evidence). Date 2026-09-16 ~02:25 ADT. One serial accel0 window.

The family worktree's generator (`~/wt/family-head-block/engine/npu/generators/n1_core_attn.py`)
does have the head-block loop — `--heads` greater than `--cols` runs multiple
passes, and its own comment names the case "Nanbeige (nh20, nkv4, cols4)". The
built artifact is
`engine/npu/xclbins/attn_gen_2048_nh20_hd128.{xclbin,insts.txt}` (48698 B /
449376 B).

Benched with the env-driven driver (`/tmp/ck2`, `CK_NQ=20 CK_NKV=4 CK_HD=128
NPU_ATTN_MAX_SEQ=2048 NPU_ATTN_COLS=4`), same kernel, NPU vs the shipped softmax
contract (EMU):

| seq | NPU | EMU | verdict |
|---:|---|---|---|
| 2048 | C2 **0/2 non-zero**, `max_abs_err=1.645789e+03` (all-zero-output err 3.06e-02), 18.9 ms/call | `max_abs_err=8.575258e-02` | **FAIL** |
| 513 | **`free(): invalid size`** (host heap corruption) | `max_abs_err=2.377548e-01` | **CRASH on NPU path** |

So:
- the nh20 head-block kernel does not write C2 at all at 2048 keys (the NPU
  output is garbage/zeros while EMU is the expected ~8.6e-02), and
- at the first chunked boundary (seq=513) the NPU path corrupts the host heap.

**`task-family-nanbeige`'s bench gate ("NPU max_abs_err == EMU max_abs_err,
non-zero C2") is therefore NOT met** — and the failure is not a small numeric
gap; it is a missing writeback plus a host-side overflow. The heap corruption at
the chunked boundary is exactly the class of layout mismatch `task-family-host`
exists to fix (a host buffer sized for the wrong `nkv`/`cols`/`hd` versus what
the kernel reads/writes), so `family-host` is the prerequisite, not a follow-on.

This supersedes the earlier "engine-side run is not done" phrasing: the earlier
`task-attn-engine` result was on the nh8/nkv2 Zaya shape, which is the shape the
generator's default and the AttnCtx host code were written for. The nh20/nkv4
multi-pass path has never produced a valid C2.

Reproduce:

```
XC=~/wt/family-head-block/engine/npu/xclbins/attn_gen_2048_nh20_hd128.xclbin
IN=~/wt/family-head-block/engine/npu/xclbins/attn_gen_2048_nh20_hd128_insts.txt
export CK_NQ=20 CK_NKV=4 CK_HD=128 NPU_ATTN_MAX_SEQ=2048 NPU_ATTN_COLS=4
/tmp/ck2 "$XC" "$IN" 2048 2                 # NPU: C2 0/2
NPU_ATTN_EMU=1 /tmp/ck2 "$XC" "$IN" 2048 2  # EMU: 8.575258e-02
/tmp/ck2 "$XC" "$IN" 513 2                  # NPU: free(): invalid size
```

Logs on the box: this run's stdout is summarized above; the bench is
`engine/npu/tools/attn_kernel_bench.cpp` (env-driven NQ/NKV/HD) and the generator
is committed in the family branch.

## CORRECTION (2026-09-16 02:30 ADT): the gate PASSES — the binary was stale

The FAIL rows above are an artifact of a **stale bench binary**, not the kernel.
`/tmp/ck2` was built Sep 15 18:55; the generated kernel was rebuilt Sep 15 20:03,
so the bench and the xclbin described different layouts. Rebuilt the bench from
the current `engine/npu/tools/attn_kernel_bench.cpp` +
`engine/npu/src/npu_attn_ctx.h` (`/tmp/ck2_new`, Sep 16 02:26) and re-ran the
same cases:

| seq | NPU | EMU | verdict |
|---:|---|---|---|
| 2048 | C2 **2/2**, `max_abs_err=8.575258e-02`, 10.05 ms/call | `8.575258e-02` | **PASS (identical)** |
| 513 | C2 **2/2**, `max_abs_err=2.377548e-01`, 6.01 ms/call | `2.377548e-01` | **PASS (identical)** |

No `free(): invalid size`. The generated nh20/nkv4/cols4 head-block kernel
implements its contract at both the chunked boundary (513) and 2048 keys —
**NPU == EMU to the digit**. So `task-family-nanbeige`'s bench gate ("NPU
max_abs_err == EMU max_abs_err, non-zero C2") **IS met**, and the earlier
conclusion is retracted.

The tell was already in the first run: the EMU arm gave a sane 8.575258e-02 while
the NPU arm gave 1.6e+03, and an ASan build of the same code reported no overflow
at all — a binary/kernel mismatch, not a kernel bug. Rebuild the bench after any
generator change.
