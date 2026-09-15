# Context 4096 → 8192: capture obtained, one real bug fixed, still blocked (2026-09-15)

FastFlowLM's published Qwen3 tables reach 32k; the native engine caps a prompt at
4095. This round went after the next doubling, got the artifact and found a real
bug on the way, but **did not** get a working >4096 path. It is written up because
the negative result is precise and the next attempt should not repeat it.

## What was built, and it is good

An 8192-context attention capture, made the same way as the 256/1024/2048/4096
ones: rebuild `run_qwen3_prefill` with `qwen3_npu model(config, &npu, 8192)`,
prefill 8192 real tokens under `cap_interposer.so` with `CAP_NO_SYNC=1`.

    engine/npu/xclbins/attn_mha_8192_nh16.elf   770064 B
    sha256 a21ac6c1df991dd8b34300021c877e7ded93c8cf9e4965a485e1f884f677a48c

Identified by the same two measured checks as every other capture:

| check | result |
|---|---|
| interposer index | `elf_0012`, the attention kernel in all five captures |
| affine size model, `2960 + 93.641·L` (fit on the committed 1024/2048 pair) | predicts 770067, **actual 770064** (−3.1 B; the same model reproduces 256/1024/2048/4096 to −0.4…−4.1 B) |
| self-validation | interposed run printed `GREEDY_NEXT: 16`, the uninterrupted run's value |

FLM's own runtime works here: 8192 tokens → 16, and 8191 tokens → 59277.

## The bug found on the way (fixed, and it was latent before this round)

Wiring the capture into a `(4096, 8192]` slot and lifting the 4095 prompt cap made
the engine **segfault inside layer 0** at npt=4200. The cause: `run_attn` sized the
act/out BOs as `attn_tokens * qout`, but a capture writes **its own slot length L
rows**, so a 4200-row call handed the 8192-row kernel a BO half the size it writes.

That is not only an 8192 problem. **At npt=4095 the existing 4096-slot kernel
writes one row past the end of its BO** — the same overrun, one row instead of
4000, which is why it never faulted. The BO is now sized by the slot the selector
picks (the smallest slot covering `attn_tokens`), which fixes both and does not
grow the short-context allocations.

## What still does not work

After the fix, no crash — but npt 4200, 5000 and 7000 **all return the same
token, 49691**, i.e. the attention output does not depend on the prompt, and the
process then aborts at exit with `double free or corruption (out)`.

The capture itself is not the problem:

- forced through the **4096** slot at npt=4095 it returns **44353**, the
  known-good native value for that prompt — so the kernel is correct, and correct
  *below* its captured length;
- the KV region stride the engine uses for this shape is 4194304 u16 = 8192
  tokens, which is exactly what the capture's bake implies and exactly what the
  ≤4096 path already runs successfully with;
- the failure is not length-generic either: it is specifically `attn_tokens >
  4096`, i.e. the branch this round added.

So the fault is in how the engine drives a capture whose slot is longer than the
existing ones — not in the artifact.

## Reverted, and why

The slot and the cap are **back to 4096**. Shipping the wiring as it stood would
have replaced a slow-but-correct CPU reference with a fast wrong answer for every
prompt in (4095, 8192], which is exactly the failure mode this file exists to
avoid. The capture stays on disk, deliberately **not loaded**, and the code
comment says so.

The prompt cap stays at 4095 for the same reason: lifting it without a working
>4096 kernel sends long prompts down a path whose attention is either the CPU
reference (measured at ~59 ms/token at 4095, so ~8 minutes for 8191 tokens) or
the broken slot.

## Next attempt: what is left to check, in order

1. **Is the 8192 capture structurally the same kernel?** FLM could switch
   attention strategy above some length, in which case `elf_0012` at 8192 is a
   different object that merely happens to scale with L. Decode it with
   `npu-infer/tools/decode_txn --decode-only` and compare its `op_histogram` and
   patch count against the 4096 capture's — the tool already reads these and a
   difference would settle it in one command.
2. **The `attn_rows` convention above 4096.** The engine passes `rows =
   attn_tokens` in one call. FLM's prefill may chunk rows in that range, and a
   kernel captured from a chunked walk would not accept 4200 rows in one call.
   The distinguishing experiment is to drive the 8192 capture with `rows = 256`
   and a growing key prefix and see whether the output becomes input-dependent.
3. **The heap corruption is unexplained and may be the same fault.** It is
   independent of the token being wrong (it happens at exit) and it is the only
   other symptom; a 4200-row run with the CPU attention reference would say
   whether it lives in the bf16 prefill above 4096 rather than in the kernel.
4. Only then re-lift the cap — the diff is two edits (the cap bound and the
   selector branch) and is described here.

## Kept from this round

- `attn_mha_8192_nh16.elf` — the capture, documented and not wired.
- The act/out BO sizing fix, which corrects a latent one-row overrun at npt=4095
  in the *existing* 4096 slot.
- `RT_ELF_WINDOW` clamping to the generator's own domain (`L <= MAX_L + 1`): a
  window that ran past MAX_L aborted the generator child with an assertion
  instead of failing cleanly, which is how the first 8192 decode attempt failed.
