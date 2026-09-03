## 35B MoE forward integration — complete map (Round 37 continued)

Everything needed to wire the 35B into the engine's runtime_layers path:

### ELF generation (DONE — gen_layer_elfs_moe)
- qwen3_6_moe_desc::build(LM_Config&) fills the layout from the config
  (NO Q4NX / load_weights — bypasses the upstream reorder_cpy crash).
- qwen3_6_moe_npu_sequence(desc, config, MAX_L) + gen_layer_seq(seq, L,
  is_full, false): is_full=true -> full-attn mha (20566 words), false ->
  GateDeltaNet linear (24636 words). Every 4th layer is full_attention.
- gen_lm_head_seq(seq, 0, 0): 260 words -> moe_lm_head.elf.
- Verified: linear layers identical, full-attn layers differ by 4 bytes
  (per-layer weight offsets), ELFs assemble via aiebu.

### Kernel ABI (DONE — probe_kernel_args)
- layer.xclbin MLIR_AIE kernel = (opcode, instr, ninstr, bo0..bo4) — the
  SAME 5-BO ABI as the 0.6B: bo0=act, bo1=weights, bo2=i5, bo3=i6, bo4=kv/state.

### Device-address map (from the layer TXN BDs)
The NPU2 BDs carry DEVICE addresses (deterministic per allocation order —
the 0.6B engine's BOs land at the same addresses as the runtime's, which is
why its ELFs work byte-exact). 35B L0 TXN:
- 0x40000000 (1 GB): the act region (6 refs at 16 MB? -> small BOs)
- 0x2000000 (32 MB): kv/state regions (632 refs)
- 0xe000000 (224 MB): the MoE weight region base (550 refs)
- 0x1bc00000-0x1ca00000 (444-459 MB): per-expert blocks (4 MB steps)
- 0xc0000000 (3 GB): the linear-attn state (560 refs)
- BD@62: len=167891008 (160 MB) off=0x30 — the up_exps copy (160 MB)

### Weight-BO packing (NEXT)
The engine packs the q4nx tensors into the weight BO at the TXN addresses:
map each tensor (qkv_proj 17.8 MB, gate_proj 8.9 MB, up/down/gate_exps
160 MB each, share_* , moe_router 1 MB, shared_expert_gate 4 KB, ssm_*) to
its BD address+len. The desc memory dump (tools/dump_moe_desc.cpp) exposes
the config dims + weight BO addresses (0x1ccc4000/0x1cb9c000) as a
cross-check.

### Forward loop
Per layer: ensure_layer_kernel(ctx) + run the 5-BO ABI with the 35B BOs
(act, weight[L], i5[L], i6[L], kv/state). RoPE via i6[0:128] host writes
(update_rope_i6 with theta 1e7 per the config rope_parameters), lm_head ELF
at the end. The linear-attn layers need the qk/kk state BOs (addr_qk 9216,
addr_kv 36864, addr_kk 45056 from the config) managed alongside the KV.

## Round 38 — the expert reorder formula is SOLVED (byte-exact vs the runtime)

The "reorder formula" piece is done. **tools/verify_moe_reorder.cpp** calls the
runtime's own `qwen3_6_reorder_cpy` (constprop.2 clone, dtype=8) on real
up_exps tiles and compares byte-for-byte:

```
reorder formula (trim 5120->4736 + A/B interleave): PASS — byte-exact
row 0 == tile 0 [0:4736]:  yes
row 1 == tile 8 [0:4736]:  yes
row 2 == tile 1 [0:4736]:  yes
```

### The formula
- Each file 5120-B Q4NX tile is trimmed to **4736 B = tile[0:4736]**
  (`[512 B scales][512 B zps][3712 B packed]`; the last 384 B — packed tail —
  is dropped).
- The reorder processes **16-row blocks** (75776 B): `out[o] = trimmed[o/2 +
  8*(o%2)]` — i.e. `[A0,B0,A1,B1,...,A7,B7]` where A = first 8 rows, B = next
  8 (the A/B half interleave visible in the constprop.2 disassembly: two
  memcpys of 4736 per iteration, src halves 37888 B apart).
- Per expert tensor: 32768 rows x 4736 B = 155,189,248 B = 2048 blocks —
  EXACTLY the desc's gate_exps offset delta (up_exps@0 → gate_exps@148 MiB).
- Block structure matches the layer TXN weight BDs: 18944-B reads (4 rows)
  at 75776-B strides (16-row blocks).

### Weight-BO map (layer 0, from desc descriptors: 8 words, OFF at word 8)
| tensor | OFF | size |
|---|---|---|
| up_exps_proj | 0x0 | 155,189,248 (32768x4736) |
| gate_exps_proj | 0x9400000 | 155,189,248 |
| down_exps_proj | 0x12800000 | 155,189,248 |
| share_up_exps_proj | 0x1bc00000 | |
| share_down_exps_proj | 0x1bd28000 | |
| share_gate_exps_proj | 0x1bc94000 | |
| moe_router | 0x3000 | 12 KiB |
| shared_expert_gate | 0x2000 | 8 KiB |
| linear_attn.qkv_proj | 0x1bdbc000 | 11,796,480 = 2304 x 5120 (out padded 8704→9216) |
| self_attn.gate_proj | 0x1c6fc000 | |
| ssm_a / ssm_alpha_proj | 0x10100 / 0x10200 | small (norm region) |

### Next
- **qkv_proj**: the runtime layout is 5120-B tiles (NOT 4736) with the out dim
  padded 8704 → 9216 (36 col-tiles x 64 row-tiles = 2304). The file rows are
  8704 B — the trim/reorder for this path is still open (different generator).
- Engine integration: extend LayerWeights with the MoE tensors (up/gate/down
  exps, share_*, moe_router, shared_expert_gate, qkv, ssm_*), build the
  per-layer weight BO (3 expert tensors dominate: 465 MB/layer), then the
  RuntimeLayerEngine forward loop (5-BO ABI + per-context moe_layer_ctxN.elf).

## Round 43 — expert-region BO layout SOLVED byte-exact vs the runtime capture (2026-09-03)

### Asset: moe-cap = the COMPLETE 40-layer runtime weight capture

`/home/bcloud/.cache/moe-cap/` holds **40 x 536,870,912-B layer weight BOs**
(bo_to_0140.., Sep-2 00:51) captured from the real runtime's load_weights via
the round-35 wait-hook interposer. That means load_weights DID complete for
all 40 layers on Sep-2 — qualifying the 35b-moe-load-crash doc (the crash is
env/version-specific, not universal). Today the live runtime is additionally
blocked by the missing VLM tower (config `vision_model_weight` =
vision_weight.q4nx, absent locally — the model_list entry is `vlm: true`).
The captures remain the ground-truth oracle for engine-side packing.

### Solved: up/gate/down expert region (rows 0..100959 of the 512 MB BO)

Byte-verified (tools/verify_moe_bo_layout.py) against bo_to_0158 (= layer 6):

- Expert tensor file data is read as **4736-B rows at 4736-B strides from
  file offset 3912** (row j = file[3912+4736j : +4736]) — NOT 5120-tile
  trimmed rows as the earlier doc assumed. Each up/gate/down file is
  167,772,160 B = 35423 usable windows.
- **Boundary splice rows** pack the previous tensor's last 824 B with the
  next tensor's first 3912 B into one 4736-B row:
  row 0 = share_up[-824:]+up[:3912]; row 32 = down[-824:]+gate[:3912];
  row 65536 = [824 B unresolved] + down[:3912].
- up/gate: 31-row block 0 then alternating 32-row blocks, k-order
  gen_k_up ([7,15,23] + cols0-6, then col7/cols0-6 per block); windows
  0..32766 placed (32767). Rows 1..65535.
- down: ALL 35423 windows, order per **8-window group [1,3,5,0,2,4,6,7]**
  (+8 per group) — empirically exact with ZERO violations across all
  35423 rows. Rows 65537..100959.
- Result: rows 0..100959 byte-identical to the capture **except one
  824-B fragment at row 65536** (runtime boundary artifact; not file bytes
  of any layer-6 tensor — checked up/gate/down/share_*/router/gate/qkv).
  99.9998% of the 458 MB expert region.

### Caveat on the old packer

`model.c`'s `npu_pack_moe_experts` (Round-38 16-row A/B interleave on
5120-tile trims) does NOT reproduce this layout — the earlier
"byte-exact on up+gate+down" claim was premature (the old
verify_moe_packer.py matched 67%: up+gate only). The verified spec above
supersedes it; the engine packer for the 35B must be built from
verify_moe_bo_layout.py's structure.

### Still open
- row-65536's 824-B fragment origin.
- rows 102624..113358 (share_up/down/gate + qkv + self-attn gate + router +
  ssm + norms + splices; 10,736 rows ≈ 50.8 MB) — qkv/ssm_out/share_* need
  their own in-BO form (Round 45: NOT raw file bytes in any convention).

## Round 45 — qkv/ssm_out/share tail: exhaustive negative + sharpened attack (2026-09-03)

### Result: no linear-attn/share tensor file bytes exist RAW in any captured BO

Searched every kept capture in `/home/bcloud/.cache/moe-cap/` (all 40 x 512 MB
layer BOs, the 542,113,792-B BO, and the assorted bo_from/actpost/waitpost
files) for qkv_proj / ssm_out_proj / share_{up,gate,down}_exps file content at
every plausible convention (4736-windows @0/@3912, 5120-tile trims, 8704-row
trims, raw spans 512/4096/4736/5120/8704) — **zero hits**. The share_* experts
are 8704-B-row I8 tensors (shape [16,8,8704]/[64,2,8704] etc.), yet their 824-B
fragments appear in the expert region splices (row 0 = share_up[-824:]...) while
their main bodies do not appear anywhere in raw form → they (and qkv/ssm_out)
are **transformed into the BO** (likely the 8704→9216 column-padded tile form,
the doc's flagged "different generator").

### Corrections to the earlier map

- The Round-38 desc-word OFF table is the **logical layout** (up@0, gate@155 MiB,
  ...) — NOT the BO-local physical layout. The physical (capture) layout packs
  tensors with 824/3912 splice rows and 31/32-row blocks: gate's content sits at
  BO-local rows 33..65535 (~156 KiB..310 MiB), not 155 MiB. The engine packer
  must use the physical layout (verify_moe_bo_layout.py), never the desc table.
- Full-attention layers (every 4th: 3,7,11,...,39) use the bigger
  542,113,792-B weight BO (3 such syncs in the manifest); linear layers use the
  536,870,912-B BO. Layer 6 (analysed) is linear_attention.

### Manifest BO-traffic histogram (decisive for the fresh-capture attack)

From capture_manifest.log (moe-cap): 245 x 536,870,912 (layer BO syncs), 3 x
542,113,792 (full-attn), 121 x 134,217,728 (staging/state BOs), 363 x 3 MiB +
182 x 5 MiB + 263 x 2 MiB (per-expert staging), 5 x 9,437,184 (ssm_out/gate-
class uploads), **no 17.8-MB sync** → qkv_proj travels inside the 512-MB layer
BO in transformed form.

### Attack plan for the qkv transform (next round, NPU-free)

1. Reverse the in-BO qkv/share form from the **generated moe layer ELFs**:
   `gen_layer_elfs_moe` emits the runtime's own sequence/TXN BDs — the weight
   BD (addr, len) for the qkv reads give the exact in-BO destination of each
   qkv byte; compare against file bytes to derive the tile transform. Plus
   `tools/dump_moe_desc.cpp` exposes the desc word table + addr_qk/addr_kv
   cross-checks.
2. If (1) stalls: fresh full-load interposer capture keeping ALL bo_to files
   (~130 GB; 528 GB free) — requires resolving the vision_weight.q4nx blocker
   (runtime dies opening the missing VLM tower today; the Sep-2 capture proves
   a full load ran then).
3. Runtime source fix (35b-moe-load-crash) needs the FastFlowLM source — not
   on-box; blocked without an upstream fetch.

## Round 46 — ARCHITECTURE CORRECTION from the layer TXN: the layer ELF does NOT read the experts (2026-09-03)

Decoded the layer-6 linear TXN (moe_layer_ctx6.txn, 24636 words, generated by
`gen_layer_elfs_moe` = the runtime's own `qwen3_6_moe_npu_sequence`) — the
kernel's weight BDs (630 DDR_PATCHes, raw device addresses in word 10):

| region | device range | reads | content class |
|---|---|---|---|
| A | 0x0 .. ~0x4cb200 (~5 MB) | 1024/3072/16512/12288/32768/37888/524288-B (150) | norms, moe_router (0x3000), shared_expert_gate, ssm smalls |
| B | 0x1bc00000 .. 0x1cb8e200 (16,310,784 B) | 416 x 18944 + 64 x 4736 (480) | share_* / qkv / ssm_out / gate_proj region (doc logical table: share_up 0x1bc00000, share_down 0x1bd28000, qkv 0x1bdbc000, gate_proj 0x1c6fc000 — ALL present) |

**The 465 MB up/gate/down expert region (byte-verified content in the captured
512 MB BO, Round 43) is NOT referenced by the layer ELF at all.**

### Consequences (corrects the integration architecture)

1. The captured 512 MB BO is the **expert weight pool**, read by SEPARATE
   per-token routed-expert GEMM kernels — not by the layer.xclbin forward ELF.
   The layer ELF reads only regions A+B.
2. Region B (device 0x1bc00000, ~16 MB) is a **separate BO** — its device
   range cannot map into the 512 MB capture (offset arithmetic excludes it) →
   this explains the Round-45 exhaustive negative: qkv/ssm_out/share raw
   bytes were never in the captured files because they live in a second,
   untetained BO in transformed form.
3. Engine forward for the 35B therefore needs: the layer ELF + regions A+B BOs
   + the expert GEMM kernels (runtime's own expert sequences, or the
   engine's per-shape mm path on the expert pool BO). Broader than the 0.6B
   single-BO model.

### Sharpened next steps

1. Determine the 512 MB BO's device base + capture regions A+B: run ONE layer
   forward under the wait-hook interposer (xrt::bo sync capture) on a
   text-only path, or a fresh full load once vision_weight.q4nx is present,
   retaining ALL bo_to sizes (incl. ~16 MB and ~5 MB ones — the manifest
   shows 9,437,184-B and 8,388,608-B syncs exist).
2. Expert GEMM kernels: enumerate the runtime's expert sequences
   (qwen3_6_moe_npu_sequence exports) for the per-token routed path — the
   0.6B-style replay then extends to the experts on the verified pool BO.

## Round 44 — gate_proj tail mapped; BO coverage 90.5% (2026-09-03)

Extended the row map past the expert region with the same signature method
(tools/map_moe_bo_rows.py — committed for continuation):

- **self_attn.gate_proj SOLVED**: rows 100960..102623 (1663 rows) = 4736-B
  windows at 4736-stride from file offset 3912 (same convention as the
  experts), j range 0..1878 of ~1880 windows. Its window ORDER is not the
  simple row order (2 runs — interleaved/spliced) — permutation rule still
  to derive, same as down was before Round 43.
- **Whole-BO coverage now 102,620/113,359 rows = 90.5%** (up 32767, gate
  32767, down 35423, gate_proj 1663, splice rows).
- **Remaining: rows 102624..113358 (10,736 rows ≈ 50.8 MB)** =
  linear_attn.qkv_proj (17.8 MB file) + linear_attn.ssm_out_proj (8.9 MB) +
  mlp.share_{up,gate,down}_exps_proj (3 x 1.11 MB) + moe_router (1 MB) +
  norms/ssm small tensors + splices. qkv/ssm_out/share_* match NO tested
  convention (4736-window @0/@3912, 5120-tile trim, 8704-row trim) — they
  need the column-padded (8704→9216) tile transform, the "different
  generator" flagged as open in the map above. Next: reverse that generator
  from the capture.

## Round 47 — runtime 35B text load UNBLOCKED + fresh full capture landed (2026-09-03)

### The runtime loads the full 35B text tower now

`LM_Config::_resolve_paths` sets `is_vlm = !vision_model_weight.empty()` — the
model dir's config.json names `vision_weight.q4nx` (missing locally; the model
list entry is `vlm: true`), which made load_weights die opening the vision
tower before any text loading. Setting `vision_model_weight: ""` in the model
dir config.json → text-only load **completes**: "load_weights done", prefill(2
tokens) runs, full 248320-vocab logits out (`/tmp/txn_decode/moe_logits.bin`,
bf16 — the 35B's first real runtime output captured). Config restored after.
The "35B dead on load" ⛔ is RESOLVED for text; the runtime is a live 35B
reference. (The `35b-moe-load-crash` SIGSEGV is a real latent
heap-layout-dependent bug in `qwen3_6_reorder_cpy` on small BF16 vectors —
seen intermittently WITH and WITHOUT interposers; the crash doc + a gdb
repro today both hit it at `load_linear_weights`.)

### Fresh full-load capture: /home/bcloud/.cache/moe-cap4 (73 GB, 854 files)

The wait-hook interposer (cap_interposer.so) triggers the latent reorder bug
on ~half the runs (layout-dependent) — retry until "load_weights done"
(attempt 2 of 6 succeeded). Captured with CAP_MAX_KEEP=67108864 (note: the
512MB expert-pool writes bypass the filter in the runlist post-dump path —
120 x 512MB files are present). Contents: 415 bo_* (incl. 5MB x90, 9MB x6,
8MB x4, 2-3MB staging), 19 insts, 27 preinsts, 15 extsmall, 35 elf, plus the
forward's post/waitpost (168/173) + actpost + kvpost (128MB) dumps = the 35B's
first real on-device forward capture.

### ssm_out transform structure identified (in bo_to_0160_5242880.bin)

layer-6 `linear_attn.ssm_out_proj` (1024 x 8704-B file rows) is stored in a
5,242,880-B BO as **4736-B windows from file offset 0** (NOT the experts'
3912 head offset): window j=0 at BO byte 328,192; j=0..15 sequential at
+9472 (2-row) stride, then the order permutes (group structure, same class
as the expert up/gate/down orders). BO content is NOT row-aligned (weights
placed at byte granularity with cross-tensor splices). qkv_proj raw bytes
still not found in ANY capture file (even multi-offset probes) — its in-BO
form is the deepest remaining transform.

### Next (tooling ready: tools/map_moe_bo_rows.py signature method)

Map the moe-cap4 small/medium BOs (5MB/9MB/8MB x N) window-by-window vs the
layer-6 (and layer-0/3) tensor files: derive each tensor's window order in
its BO (ssm_out done structurally; then ssm_out full order, share_*, qkv,
gate_proj) → complete region A+B spec → engine packer for the linear-attn
BOs. Runtime load with vision-off config remains the reference generator.

## Round 48 — per-layer load BO map SOLVED; qkv transform bounded (2026-09-03)

### The load's per-layer BO structure (from moe-cap4 seq + content matching)

Every LINEAR layer uploads, in order: `[512 MB expert pool]` + `[2 MB]` +
`[5 MB]`. The 512 MB pool seq = 140 + 3L with full-attn layers (3,7,11..)
skipped (they use 542,113,792-B BOs). Content-verified: layer 6 = pool
bo_to_0158 (the R43 ground truth), 2 MB bo_to_0159, 5 MB bo_to_0160
(ssm_out). self_attn.gate_proj lives INSIDE the pool (rows 100960..102623,
R44). The 2 MB BO = the region-A class (norms/router/ssm smalls); the 5 MB
BO = the ssm_out windowed BO.

### ssm_out: window order is 2-row-strided then permuted

In bo_to_0160 (5 MB): ssm_out (1024 x 8704-B file rows) windows j=0..15 sit
at BO byte 328192 + j·9472 (2-row stride, window = file[4736j:+4736] from
file offset 0), then the order permutes (group structure). BO content is
placed at byte granularity — NOT row-aligned (BO local 328192 is not a 4736
multiple; cross-tensor splices fill the gaps).

### qkv: no window/offset convention matches ANY capture (bounded)

qkv_proj (2048 x 8704-B rows) raw/windowed content (offsets 0/3912, strides
4736/8704, widths 4736/8704/2048/4096) is absent from every moe-cap4 file
incl. layer 6's own 512 MB pool + 2 MB + 5 MB BOs. Its in-BO form is a
per-tensor transform (not the expert/simple-window class). The runtime's
`qwen3_6_reorder_cpy` is the probable producer — verify_moe_reorder.cpp
calls it directly (dlopen + hardcoded offset to the constprop.2 clone,
`(gen_layer_seq−0x97ad0)+0x68b80`), verified byte-exact for up_exps tiles.
Note a discrepancy to resolve: R38's reorder-on-trimmed-tiles A/B
interleave output ≠ R43's captured window layout — one of the two
verification framings mislabels its ground truth; reorder_cpy's actual
row-source convention needs re-deriving per tensor shape.

### Next (bounded experiment, ~1 session)

Call qwen3_6_reorder_cpy directly on qkv rows with the load_linear_weights
arg conventions (dtype/flags per the crash evidence: dtype=8 family) and
match the output against the layer-6 BOs; once one qkv/ssm_out row shape is
byte-verified, the region A+B packer spec follows (per-tensor orders are the
same mapping class already solved for up/gate/down).

## Round 49 — PARKED: qkv transform blocked; reorder_cpy ruled out as pool producer (2026-09-03)

Three further negatives close this RE sub-thread for now:

1. **reorder_cpy output does NOT appear in the captured expert pool.** The
   verify_moe_reorder direct-call (dlopen + `(gen_layer_seq−0x97ad0)+0x68b80`,
   still PASSES on the current lib) emits 16 rows of A/B-interleaved trimmed
   tiles for up_exps; those exact bytes are absent from the layer-0 pool
   (bo_to_0140). The pool is filled by a plain window-copy in gen_k_up order
   (R43's byte-verified model) — the R38 "reorder produces the pool layout"
   framing is superseded; reorder_cpy serves a different (linear-attn)
   purpose, if any.
2. **reorder_cpy direct calls on qkv rows segfault** (geometry-specific: the
   16x8704/4736-trim geometries crash the clone — consistent with its
   documented latent chunk-overflow bug). Blind function probing is not
   productive.
3. qkv raw/windowed/reordered content matches NO captured file under any
   tested geometry after R43-49 (pool/2MB/5MB BOs, all conventions).

### Recommendation — park the byte-exact 35B runtime-replication lane

The npu-infer 35B runtime-replication (byte-identical weight BOs) is blocked
on one transform (qkv/region A+B) behind: (a) no FastFlowLM source on-box,
(b) a latent runtime bug, (c) exhaustive-negative content probes. The lane's
hard-won assets remain valuable: expert-pool layout byte-verified (R43),
load architecture + per-layer BO map (R46-48), runtime text-load unlock +
73GB capture oracle (R47). Genuine options, in order of leverage:

1. **Runtime-as-server**: the 35B text tower now loads + prefills on the
   runtime (vision-off config) — usable as the 35B reference/serving path
   directly (npu-infer engine replication optional).
2. **1bit production engine**: npu_engine_universal.cpp's own NPU_MOE path
   (per-expert kernels + packers, #1473 line) already targets the 35B with
   its own xclbins — the npu-infer replication is not the only route.
3. Re-engage the qkv transform only if runtime source becomes available
   (upstream fetch) or a captured qkv BO is obtained (interposer capture
   requires beating the ASLR-dependent reorder bug ~50% of the time).

## Round 50 — 100% BYTE-VERIFIED: today's layout is clean, all 30 linear-layer pools + 5MB BO (2026-09-03)

### The current runtime's layout DIFFERS from the Sep-2 capture (R43 spec superseded)

The Sep-2 moe-cap (R43: w3912 windows, splice rows, 31-row first blocks,
824-B fragments) is a STALE runtime state. Today's runtime (moe-cap4,
vision-off load) produces a far simpler layout — and it reconstructs
byte-for-byte:

### 512 MB expert pool (ALL 30 linear layers = 100.000000%)

rows 0..65535 — alternating 32-row up/gate blocks (1024 each), windows from
FILE OFFSET 0 (stride 4736, j 0..32767), block order `j = base+8·(i%4)+i//4`;
rows 65536..100959 — down, ALL 35424 windows, 8-groups `[0,2,4,6,1,3,5,7]`;
rows 100960..102623 — self_attn.gate_proj (1664 windows: j ∈ {0..7, 224..1879}
in interleaved (v, v+8 mod 1880) pairs from v=224); rows 102624+ = ZEROS
(allocator slack — the 50.8MB "mystery tail" of R44-48 was empty padding).

Byte-verified: layers 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,
26,28,29,30,32,33,34,36,37,38 — ALL 100.000000% vs their moe-cap4 pools.
(tools/verify_moe_current_layout.py)

### 5 MB linear-attn BO (100.000000%)

head 328,192 B = byte-packed [ssm_conv1d 64K][ssm_norm 256][ssm_a 128]
[ssm_dt.bias 128][ssm_alpha_proj 128K][ssm_beta_proj 128K]; then ssm_out_proj
windows from FILE OFFSET 0 in 32-row blocks order `j = base+16·(i%2)+i//2`
(windows 0..1045). Rest zero. No tail, no gaps.

### Byte-flip differentials (decisive method, R50)

One-byte flips in model.q4nx + recapture + 3-way diff vs run-variance isolated
the effect precisely: qkv data byte0 = a loader control byte (0 → qkv load
skipped, downstream 1MB BOs stay default-filled); qkv payload bytes (deep
flips) NEVER change any synced BO → qkv's content is written map-only
(never xrtBOSync'd) and is invisible to sync-capture. This explains every
prior qkv negative (R45-49).

### Remaining (1 item): the 2 MB per-layer BO

bo_to_0159-equiv (per-layer, differs across layers, ~2MB): content matches no
raw tensor bytes/windows and is not clean bf16 (value mode 0x39; int8-ish)
— believed qkv-family derived state. All other per-layer content (pool +
5MB linear-attn) is now byte-verified 100%.

## Round 51 — the 2MB BO IS qkv: int16 fixed-point, value-multiset 99.996% (2026-09-03)

Closed the identity question on the 2MB per-layer BO:

- The data is **int16 fixed-point** (NOT int8, NOT bf16): every other byte = the
  value's high byte; the dominant 0x39 high byte = 0x39xx/16384 ≈ 0.90 weights.
  NaN in bf16 view = exponent patterns, not float.
- **Value multiset overlap with the layer's `linear_attn.qkv_proj.weight` file =
  99.996%** (unique-value overlap); int16 stats nearly identical (mean 1001 vs
  990, std 13614 vs 13629). The 2MB BO = a per-layer selection/permutation of
  the qkv file's fixed-point values.
- qkv file = 8,912,896 int16 values (17.8MB, 2048 rows x 4352 values); the 2MB
  BO holds 1,048,576 of them (~1/8.5). Linear-attn dims (config): Q/K heads
  16x128, V 32x128 — 1,048,576 = 2048x512 = plausibly one projection slice.
- The exact selection + order (its permutation) is not derivable statically:
  qkv payload is map-written (never xrtBOSync'd — Round-50 byte-flip
  differentials) so no sync-capture exists; the runtime's qkv kernel-format
  transform is closed-source.

State: ALL per-layer BO content EXCEPT the 2MB BO permutation is byte-verified
100% (30 pools + 5MB linear-attn, Round 50). The 2MB BO identity is now pinned
(qkv int16 fixed-point, kernel-format buffer); reproducing its exact bytes
requires the runtime's qkv format generator (source) or a map-write capture
(xrt::bo::map interposition — a possible future interposer extension).

## Round 52 — qkv-format BOs CAPTURED as layer-kernel args (2026-09-03)

Extended the interposer (CAP_DUMP_BIG: pre-exec arg-BO dump deduped by size,
no >3MB skip) and recaptured the load+forward (moe-cap8, ASLR retry #4). The
layer kernel's runlist args include the per-layer qkv-format 2MB BOs:

- preinsts_001_*_2097152 = runlist-1 (token-1) exec args; the i5 2MB BOs have
  value multisets FULLY contained in the layer's qkv_proj file (coverage
  1.0000 for layer 0's qkv — two variants at int16 mean 974 and 983 = two
  qkv subsets). Layer 0's copies archived to moe-cap4/qkv_l0_fmt_bo{,2}.bin.
- Their internal permutation (which qkv elements + order) resisted col-slice,
  row-block and [256,8,4352] axis-rotation hypotheses — the layout is a
  deeper block arrangement tied to the linear-attn MM geometry.
- Note: the earlier moe-cap4 0159 2MB BO (mean ~1000, 0x39 fixed-point
  pattern) is the layer-6 sibling of these captured layer-0 args.

Status: the qkv kernel-format BOs are now OBSERVABLE (map-hook or this
arg-dump path), so the remaining permutation is a mapping puzzle on real
captures — no longer a capture problem. The 100% packer spec awaits that
permutation; everything else per-layer is byte-verified.

## Round 53 — qkv permutation: exhaustive structural negatives (2026-09-03)

With the layer-0 kernel-format captures in hand (moe-cap4/qkv_l0_fmt_bo{,2}.
bin), tested every tractable structural hypothesis for the qkv → kernel
layout:

- not a contiguous file slice (head/tail)
- not a column-slice (any 512-col window, contiguous or strided), not rows
- not a [256,8,4352] axis rotation / blocked col slice / transposed
  [4352,2048] out-row slice
- d-rows (2048x512) not contained in single qkv rows; d-columns not equal to
  single qkv columns → values interleave across qkv rows/cols at fine
  (sub-512) granularity
- d/d2 value composition (68-69% |v|>5000, ~53% positive, mean 974-983) is a
  near-uniform representative sample of qkv's (72.7% / 53.5% / 974) — the
  selection is spread across the whole tensor

Also: the runlist-1 i5 2MB arg BOs are SHARED buffers (same bo pointer across
runs/layers: 54790 x3, 55770 x2, 55f60 x3) — they are the layer kernel's
shared norm/state args, NOT the per-layer qkv BOs (which sync as bo_to_0159-
equiv, mean ~1000 for layer 6).

Conclusion: the qkv kernel-format permutation is a fine-grained
rearrangement produced by the runtime's closed weight-prep (qwen3_6_reorder
family). Deriving it byte-exactly requires the runtime source or a
considerably deeper differential campaign (e.g., per-value flips mapped
through the BO). Everything else in the per-layer layout is byte-verified
100% (R50).

## Round 54 — CORRECTION: the 2MB BO overlap was codebook coincidence, not qkv derivation (2026-09-03)

The "99.996% qkv value overlap" (R51) is RETRACTED: all int16-quantized tensors
(qkv, ssm_out, gate/up/down_exps...) share one ~65K-value codebook, so unique-
value overlap vs ANY of them is ~100%. The overlap proved nothing about qkv.

What IS established about the 2MB per-layer BOs (bo_to_141+3L-equiv):

- Two families across the load: mean ~+1000 / std ~13600 (the qkv/ssm_out/
gate_proj/o_proj projection-value family) AND mean ~-440 / std ~11666 (a
different scale — present at ~11 seqs ≈ the full-attn layers + boundary).
- 1,048,576 int16 values = 2048x512 — a slice of the 8.9M/4.46M-value
  projection tensors, but matching no file windowing/slice/permutation
  (R48-53 tests).
- True per-tensor int16 sizes (correcting earlier confusion): qkv 8,912,896;
  ssm_out 4,456,448; self_attn.gate_proj 4,456,448; full-attn L3 q/k/v/o =
  8,912,896/557,056/557,056/4,456,448.

Status: the 2MB BO's exact content/role is OPEN (black-box analysis
exhausted without the runtime source). Not qkv-derived in any simple way;
a projection-family slice in the runtime's kernel format. Everything else
per-layer (pool + 5MB BO) stays byte-verified 100% (R50).

## Round 55 — external Q4NX spec + prefill/decode architecture (arXiv 2602.06063)

Read + banked the Clemson/URI NPU paper (docs/research/q4nx-npu-paper-2602-06063.md):

1. **Canonical Q4NX spec**: 32x256 int4 blocks, dequant w_i = d_g*wq_i + m_g
   (g=32), block = 256 bf16 scales + 256 bf16 offsets + packed int4 = 5120 B.
   ⚠ FastFlowLM's variant files differ (experts ARE 5120-tiled 32768x5120;
   linear-attn 8704-B-row tensors are NOT) — canonical spec = reference only.
2. **Prefill/decode split explains our buffer archaeology**: prefill dequantizes
   Q4NX→bf16 written to DDR (map-written, never xrtBOSync'd — the R50-54
   invisible qkv payloads = PREFILL-side bf16 intermediates); decode keeps
   weights QUANTIZED with fused dequant (FusedDQP, 32x256 tiles / 16x8
   sub-blocks = the 8/16/32-row blocks in the captured BOs). R51-54 int16
   analyses were partially confounded by reading mixed int4+bf16 content as
   uniform int16.
3. Benchmarks (Gemma3 1B/4B, Ryzen AI 7 350): 5.2x prefill / 4.8x decode vs
   iGPU — cross-platform calibration only. No code released (review-anonymous).

## Round 56 — 35B runtime public API returns a deterministic non-logits buffer (2026-09-03)

Attempted the runtime-as-server 35B generation demo (qwen3_6_generate harness:
vision-off config, chat-prompt prefill + autoregressive greedy loop). Load
succeeds (ASLR-retry ~attempt 3; the latent reorder segfault is ~50%+ per run),
prefill ~1.3 s, forward ~60 ms/tok (~17 tok/s wall — the 3B-active experts are
fast). BUT:

- `qwen3_6_moe_npu::prefill()/forward()` return a 248320-bf16 buffer that is
  100% NaN-as-bf16 / fp16-garbage (mean 39543, near fp16 max), and the content
  is DETERMINISTIC across different prompts (2-token dummy == 32-token chat) —
  it is not input-derived logits.
- Token 0 ('!') argmaxes everywhere → the returned buffer is a fixed
  scratch/internal buffer, not the lm_head output. The real logits path for the
  closed moe model needs an internal call not exposed in qwen3_6_moe_npu.hpp
  (no public get_logits/sample on this class, unlike the runtime-layer path).
- Conclusion: harness-level 35B generation through the public lib API is
  blocked at the closed-lib boundary. A working 35B server needs the flm
  serve binary (not on-box) or the lib source. The architecture/assets from
  R43-55 (byte-verified packer specs, capture oracle, runtime load unlock)
  remain the banked value of the lane.
