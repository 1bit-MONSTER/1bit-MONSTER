# MoE 35B arg map — what is established, and the one contradiction that is not

Round-15 working notes. Recorded because I made TWO self-corrections in this span and the
next reader must not re-derive them. **Read the contradiction at the end before trusting
the arg-order fix in 1ae82a6e1 as complete.**

## Established (independently, more than one way)

**1. Only the activation BO alters the output.** Post-fix order, zeroing one BO at a time:

| variant | logits |
| --- | --- |
| BASE | argmax=193722 max=0.0242 NaN=0 |
| ZERO_ACT | **ALL ZERO — changed** |
| ZERO_WEIGHTS | identical to base |
| ZERO_ROUTER | identical to base |
| ZERO_NORMS | identical to base |

Three of the five BOs the harness binds have **no effect on the output at all**.

**2. The patch table's per-argument requirements** (`decode_txn.cpp`, using `arg_offset` —
the offset INTO the argument buffer, which an earlier pass of mine ignored and so reported
nonsense extents; `buffer_offset` is 0 for every patch and means something else):

| arg_idx | patches | max(arg_offset+len) | distinct coverage | harness BO of that size |
| --- | --- | --- | --- | --- |
| 0 | 544 | **481,878,528** | 4,110,848 B / 241 ranges | weight 481,935,360 |
| 1 | 4 | 1,024 | 1,024 | — |
| 2 | 4 | 45,056 | 35,840 | — |
| 3 | 70 | **5,064,192** | 1,294,464 | norms 5,242,880 |
| 4 | 8 | 573,440 | 536,576 | — |

patch-arg0's requirement is 481,878,528 B against a weight BO of 481,935,360 B — a match to
within 56 KB, i.e. it is the weight BO. patch-arg3's 5,064,192 B against the 5,242,880 B norms
BO is likewise the norms BO. **A full MoE layer must read its expert weights from DDR**, and
patch-arg0 doing exactly that is the only such traffic in the stream.

**3. The ELF's total DDR traffic is 13.75 MB** (11.86 MB MM2S, 1.13 MB S2MM) over 630
BLOCKWRITEs. This is NOT a defect: region A (478,146,560 B) is the **shared expert pool for all
40 layers**, so one layer is ~11.95 MB, which is what the ELF reads. An earlier note of mine
called this "35x short" by comparing a single layer against the whole pool — wrong.

**4. FLM's own binding, from the Qwen3-4B capture** (`~/.dsh/scratch/mesh/argmap.txt`, and the
same shape in `captures/capture_manifest-2026-09-03.log`):

```
idx=0 bytes=4 val=0x3   idx=3 size=1048576     <- act
idx=1 bytes=4 val=0x0   idx=4 size=63963136    <- weights
idx=2 bytes=4 val=0x0   idx=5 size=1048576
                        idx=6 size=1048576
                        idx=7 size=134217728   <- kv
```

**5. The vendor kernel ABI** (`decode_txn.cpp:16-18`): `opcode=0, instr_bo=1, ninstr=2,
act=3, ws=4, w1=5, w2=6, kv=7`.

**6. No MoE capture survives.** The 4B capture was in `/tmp/cap4b_real/` and is gone with
`/tmp`. There is no MoE-model SETARG log anywhere on disk. Facts 4 and 5 are therefore about
the **dense 4B** model and the **mm kernel** respectively — neither is the 35B MoE layer ELF.

## The contradiction that is NOT resolved

- Patch-arg0 requires exactly the weight BO, and the original harness bound `arg3 = weight`.
  If patch `arg_idx` k maps to XRT arg k, the original binding was right for the weights — and
  it produced ALL NaN. If it maps to XRT arg k+3, then `arg3=weight` puts the weight BO where
  patch-arg0 lives, which is also "right", and that also produced NaN.
- My swap (`arg3=act, arg4=weight`) **removed the NaN** but puts a 1 MB BO where patch-arg0
  needs 481 MB, and puts act at patch-arg3 which needs 5 MB (the norms size).
- So the swap that fixed the symptom is **inconsistent with the patch table's own
  requirements**. It is a real, reproducible A/B (legacy order reproduces NaN on demand), but
  it is not explained, and "finite" is not "correct".

**Conclusion: the arg map is still not established for this ELF, and the round-15 fix should be
treated as a symptom change, not a root-cause fix.** The authoritative resolution is FLM's own
binding for the **35B MoE** model, which requires re-capturing FLM on that model — the 4B log
cannot be transferred, because arg order is per-kernel caller order with no intrinsic meaning
(`cap_interposer.cpp:219-226`).

Also stale: the goal's note that the harness "packs LAYER 0 while running the LAYER 1 ELF" is
no longer true — a run prints `packing LAYER 1 weights (moe_layer_ctx1.elf)` and
`engine init OK (packed layer 1)`.
