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

## Round 57 — "no blockers, undiscovered code": Impl::get_logits found + disassembled (2026-09-03)

Direct .so mining (the closed lib is callable/readable — it was never a wall):

- **`qwen3_6_moe_npu::Impl::get_logits` is an EXPORTED symbol**
  (`_ZN15qwen3_6_moe_npu4Impl10get_logitsER6bufferIN8biovault10bfloat16_tEE`,
  addr 0x6cfb0). The public class holds `Impl* _impl` at +0x30 (confirmed via
  wrapper disassembly `mov 0x30(%rdi),%rdi` on get_current_context_length /
  clear_context).
- **get_logits's disassembly reveals its semantics**: it memcpy's
  `buffer.data (+0x10)` → an internal activation BO (impl+0x4d0, byte count
  from impl+0xb8), then runs an **xrt::run (the lm_head kernel**, run object at
  impl+0x5b8), syncs the lm_head output BO (impl+0x528, dir FROM), and writes
  the result back to the buffer (+0x10). So get_logits COMPUTES lm_head over an
  input activation buffer — it is not a "fetch after forward" getter.
- My first call attempt SEGV'd (rc=139): ABI/semantics mismatch (the function
  reads impl-relative fields at +0x4d0..+0x5b8 and expects an activation
  buffer, not a logits sink). Calling it correctly = feed the post-layer
  activation buffer the way the flm server does.
- forward()'s returned 248320-bf16 buffer remains deterministic garbage — the
  real lm_head path lives in get_logits/Impl, now located.

Next (concrete): disassemble Impl::forward (0x73c10) to find which internal
buffer holds the final hidden state + how the server drives forward→get_logits;
call get_logits with that activation buffer (matching the .so's own usage) →
real 35B logits → generation.

## Round 58 — the logits path IS in the lib: _prefill_with_mm calls get_logits (2026-09-03)

Continued the .so mining (no blockers):

- get_logits (0x6cfb0) has exactly ONE caller inside the lib:
  **qwen3_6_moe_npu::Impl::_prefill_with_mm** (0x76320, call at 0x76d94). The
  MM prefill path runs the layers, then calls get_logits to compute lm_head.
- The call site shows the real convention (nonstandard regs): rdi = a REAL
  stack-constructed buffer<bf16> (vtable set from _ZTV6buffer...), rsi = the
  Impl, rdx = an input buffer whose data(+0x10) get_logits memcpy's into the
  lm_head input BO (impl+0x4d0, count impl+0xb8), runs the lm_head xrt::run
  (impl+0x5b8), syncs the output BO (impl+0x528 FROM), writes logits back.
- buffer<bf16> = the REAL header class with a VTABLE at offset 0 + device
  members — my bare 4-field Buf (vtable=nullptr) was why the direct call
  SEGV'd. Need real buffer<bf16> objects (construct via buffer.hpp).
- Public prefill/forward return buffers decode as all-NaN bf16 in EVERY path
  (32/44/382-token prompts) — the returned buffer's data pointer is not
  host-valid logits; the real logits materialize through get_logits only.

Next (concrete): construct real header buffer<bf16> objects and drive
prefill → get_logits(act_buf, impl, out_buf) exactly as _prefill_with_mm
does (need the final-hidden-state buffer as the rdx input — likely the
prefill-returned buffer read as the act, or impl+0x4d0's content). The
lm_head path is now fully mapped at the instruction level; it is a matter of
replicating the call, not discovering it.

## Round 59 — 35B text-only forward = NaN at the layer level (runtime-lib, definitive)

Isolated WHERE the 35B breaks with a clean discriminator chain:

- get_logits + lm_head work PERFECTLY on a clean input: feeding a 2048-ones
  buffer → real 248320-vocab logits (argmax 7535, 0 NaN). The lm_head BO
  chain (impl+0x4d0 input, impl+0x528 output BO) is functional.
- The loaded weights are CORRECT: fresh interposer capture (moe-cap9) pools
  for layers 0/6/30 = 100.000000% match vs the byte-verified R50 spec. NOT
  silent reorder corruption.
- The LAYER forward is genuinely NaN: prefill's returned buffer = 248320 host
  bf16 all-NaN (no device BO backing — a stub), and the impl's own lm_head
  output BO (0x528, 524288 elems) after MM-prefill = NaN over the vocab
  region. With correct weights + a working lm_head, the NaN must originate in
  the layer kernels/state under the text-only (vision-off) configuration of
  this lib build.

Conclusion: the FastFlowLM lib's 35B hybrid forward is non-functional in the
text-only configuration available on-box (matches the "35B dead on load" doc
in spirit — the model cannot be served via this lib even past the crash). The
lib's mm/mv prefills, expert pools and lm_head are individually sound; the
layer pipeline (linear-attn/ssm state, hybrid routing) NaNs. A working 35B
runtime needs the flm server binary with the proper (vision-capable) model
set, or lib source. The lane's verified assets (R50 specs, get_logits map,
pools-correct proof) remain banked.

## Round 60 — the engine (not the runtime) is now the 35B path: replay design (2026-09-03)

With the runtime's text-only forward proven NaN at the layer level (R59) and
the model not servable via the lib, the npu-infer ENGINE becomes the only
route to a working 35B. Its prerequisites are mostly banked:

- moe per-context layer ELFs: gen_layer_elfs_moe ✓ (moe_layer_ctxN.elf +
  moe_lm_head.elf, the runtime's own sequence generator)
- decode-side weight layouts: R50 byte-verified spec (pool + 5MB linear BO),
  + capture oracle /home/bcloud/.cache/moe-cap4 = the runtime's ACTUAL
  per-layer BOs (pool + 2MB + 5MB + 128MB state + smalls) — the replay can
  feed the CAPTURED bytes directly, bypassing the unknown 2MB/qkv format
  entirely (the 0.6B replay path pattern: captured BOs as weights).
- 5-BO kernel ABI + RuntimeLayerEngine machinery (0.6B, proven).

### Engine 35B forward design (replay path)

1. Driver allocates per-layer BOs to land at the device addresses the moe
   ELF TXNs reference (0x40000000 act, 0xe000000-region expert pool, 0x1bc00000
   region-B, 0x2000000 + 0xc0000000 kv/linear state) by replicating the
   runtime's BO allocation order (the 0.6B determinism trick) — needs a probe
   run to confirm address determinism for the 35B BO set.
2. Load the captured per-layer BO bytes (moe-cap4) as weights — no qkv format
   decoding needed for a replay.
3. Per layer: submit moe_layer_ctxN.elf via the 5-BO ABI (act, weights, i5,
   i6, kv/state); the hybrid linear layers additionally need the
   linear-attn state BOs (addr_qk/addr_kv/addr_kk regions).
4. Validation: the runtime is NaN (no byte reference), so validate by
   DETERMINISM + non-NaN + (once generating) coherence — the engine becomes
   the first working 35B on this box if the kernels are correct.

### Feasibility notes / risks
- Device-address replication for the 35B BO set (40 layers x several BOs) is
  the delicate unknown — needs an empirical probe (allocate in the driver in
  the runtime's order, compare landed addresses vs the ELF's BD addresses).
- The linear-attn state (GateDeltaNet: qk/kk/conv state, addr_* config) needs
  its own BO layout — partially mapped (128MB state BOs in moe-cap4).
- Multi-session effort; the single-layer run (one linear layer, captured BOs,
  generated ELF, determinism check) is the first milestone.

This closes the R43-60 35B arc with a concrete, unblocked-by-design path:
engine replay with captured BOs, validated by determinism since the runtime
is NaN.

## Round 61 — 35B engine replay probe: device-VA question is the crux (2026-09-03)

Started the Round-60 milestone (single linear layer via captured BOs + moe
ELF). Built an allocation-order probe (addr_probe.cpp) replicating the
runtime's BO sequence from the moe-cap4 EXTBO manifest.

Result: `xrt::ext::bo::address()` returns HOST-backing ranges
(0x7f54xxxxxxxx descending per allocation) — NOT the low device-VA space the
moe ELF TXNs reference (act 0x40000000, region B 0x1bc00000, state
0x2000000/0xc0000000). With PASID/SVM the driver's device-VA assignment for
system-memory BOs is the unresolved variable; whether ext::bo addresses map
1:1 to the ELF's expected VAs (the 0.6B determinism trick) needs a driver-
level probe (amdxdna BO-info ioctl / xclbin hwctx VA query), not a
userspace address() read on this build.

The replay milestone is therefore parked on one question: the amdxdna
device-VA scheme for ext::bo. Everything else for the 35B engine replay is
banked (captured BOs, moe ELFs, verified content, 5-BO ABI). The probe
utility (alloc order → addresses) is the starting point once the VA query
is resolved.

## Round 62 — driver-VA probe: GET_BO_INFO located; raw-ioctl + interposer results (2026-09-03)

Attacked the device-VA question at the driver level:

- The amdxdna ioctl header is IN-REPO (npu-infer/include/npu_utils/amdxdna_accel.h):
  `amdxdna_drm_get_bo_info` returns `xdna_addr` (the XDNA device VA) + vaddr/
  map_offset; CREATE_BO takes size/type/vaddr. Types: INVALID/SHMEM/DEV_HEAP/DEV/
  CMD/DMA.
- Raw CREATE_BO on /dev/accel/accel0 fails EINVAL for all types without the
  proper hwctx setup (XRT's xclbin→hwctx handshake is a prerequisite — not
  replicable in a few lines).
- An LD_PRELOAD ioctl interposer on the runtime catches calls on fd 3 =
  /dev/accel/accel0, but the request numbering (0xc0206443-style: nr bits =
  100, type byte 0x20) does not trivially equal DRM_COMMAND_BASE+enum — the
  driver's _IOC encoding needs decoding from the module. Interposers also
  destabilize the 35B load (segfaults), making iteration slow.
- No amdxdna debugfs BO map exists (debugfs/dri = the iGPU only).

State: the device-VA question (do replicated BOs land at the moe ELF's
0x40000000/0x1bc00000... addresses?) remains OPEN. The tools are banked:
GET_BO_INFO struct + header, the allocation-order probe, the ioctl interposer.
Next concrete step: decode the accel ioctl numbers from the loaded amdxdna.ko
(ioctl dispatch) or drive the probe through the runlist-capable XRT 2.26 at
/usr/local/xrt-runlist (built for this driver), which may expose correct VAs
for XRT-created BOs.

## Round 63 — driver probe: deterministic allocation map captured (2026-09-03)

The ioctl interposer (corrected with the debugfs-confirmed request numbers
CREATE_BO=0xC0206443, GET_BO_INFO=0xC0206444) captured the runtime's FULL
1576-BO allocation sequence + per-BO DRM map offsets:

- First: DEV_HEAP 64MB @ map 0x100000000; then DEV BOs (no map); then the
  40x 512MB SHMEM pools SEQUENTIALLY from 0x104000000 (512MB stride:
  0x104000000, 0x124000000, 0x144000000...) — exactly the load order from
  the moe-cap4 manifest. Smaller BOs (2MB @0x464000000 etc.) interleave at
  deterministic positions.
- get_bo_info.xdna_addr stays 0 even after BO use on this driver build
  (not populated); the DRM map_offset = the deterministic per-allocation
  handle. vaddr = 0 for the runtime's BOs (driver-managed).
- Debugfs /sys/kernel/debug/accel/0000:c6:00.1/ exposes ioctl_id (confirms
  the request numbers), ctx_rq, get_app_health, powerstate, telemetry*.

Implication for the replay: the allocation MAP is deterministic and now
known (replicate the sizes/order → same map offsets). Whether the map offset
= the device-VA basis the moe ELF TXNs reference collapses into the
empirical single-layer run: allocate the runtime's BO sequence, load the moe
layer ELF, submit — if the kernel reads the right content, the map offsets
ARE the VA basis and the replay works. Tools banked: ioctl interposer +
allocation map + the exact request numbers.

## Round 64 — replay allocation replication DIVERGES at seq 1 (2026-09-03)

The single-layer probe's determinism test: allocate the runtime's BO sizes in
order via XRT, compare the driver CREATE_BO sequence to the runtime's (R63
log). Result: **divergence at seq 1** — the runtime's XRT creates internal
DEV BOs (~400KB/268KB: xclbin/insts buffers) before the pools; a bare
XRT ext::bo sequence creates a different internal BO (64MB SHMEM scratch)
instead. XRT's internal allocations (xclbin load, insts buffers, hwctx
setup) are part of the driver sequence and cannot be skipped.

Conclusion: reproducing the runtime's device-VA layout from userspace
requires replicating the ENTIRE XRT call sequence including its internal
BOs — i.e., driving the identical init path as the runtime, which is fragile
and offers no advantage over the runtime itself. The moe ELFs' addresses are
bound to the runtime's full allocation context.

Realistic paths for the 35B engine replay, in order:
1. Drive the same XRT init sequence (register xclbin -> hwctx -> the runtime's
   BO set incl. internal DEV BOs) then layer ELFs — a full re-implementation
   of the runtime's init, fragile but self-contained.
2. Runtime source or the flm server binary (the model's proper serving path).
3. Accept the 35B runtime is NaN/broken on this lib (R59) until a fixed lib
   or the Q4NX open-source release (arXiv 2602.06063 group) arrives.

The R43-64 bank: byte-verified packer specs (pools+linear, R50), the capture
oracle, the get_logits/lm_head map, the driver allocation map + ioctl numbers
(R63), and this divergence finding. The 35B lane is at a genuine
architectural boundary for userspace replication.

## Round 68 — vision tower tested: NaN persists with the FULL model (2026-09-03)

The last untested variable: R59's NaN was measured in TEXT-ONLY (vision-off)
mode. Downloaded vision_weight.q4nx (0.94GB, exact HF size) → model dir now
complete (all 5 files). Ran the full config (is_vlm=1, vision tower loads
clean, load_weights DONE):

- prefill 1.2s, act.size=248320
- get_logits over the post-prefill state: argmax 0, **248320/248320 NaN**

The layer forward NaNs identically with the vision tower present. The
FastFlowLM lib's 35B hybrid forward is definitively non-functional on this
build for BOTH text-only and full (vision-capable) configurations. Not a
missing-tensor / config artifact.

Conclusion: the 35B cannot run via this lib at all (crash ~50% of loads
[ASLR bug] + NaN forward otherwise). The vision file stays on-disk for a
future fixed lib / the flm server. This closes the runtime-as-server thread
with finality; the engine replay path remains the only 35B route and it is
blocked at the device-VA wall (R61-65). Assets + docs banked.

## Round 69 — task-1 ELF verification (2026-09-10, goal mtusoiy1-cfdhqr)

Re-verified the generated 35B per-ctx MoE ELF set (npu-infer/captures/
txn-elfs-moe35b/) with a self-contained TXN decoder (header-skip + op
dispatch ported from tools/decode_txn.cpp):

- 40 layer TXNs, exactly 30 linear (24636 words, 630 DDR_PATCHes) +
  10 full-attention (20566 words, 530 DDR_PATCHes). Full-attn layers =
  {3,7,11,15,19,23,27,31,35,39} — every 4th layer, matching config
  layer_types. ELF sizes: linear 103104 B, full 86208 B.
- lm_head ELF = 1680 B (260-word sequence + aiebu header), present.
- Linear layer DDR_PATCH arg_idx histogram {0:544, 1:4, 2:4, 3:70, 4:8};
  full-attn {0:512, 1:4, 2:4, 3:2, 4:8}. arg_idx 0 (the weight BO) carries
  arg_off spanning regions 0x0 (region A: 150/82 patches — norms/router/
  ssm smalls), 0x1bc00000.. (region B share_*/qkv), 0x1c6fc000.. (region B
  gate_proj). The 465 MB expert pool (up/gate/down) is NOT referenced by
  the layer ELF — confirms Round 46 (experts are separate per-token GEMMs).
- arg_idx 0 offsets reach 0x1cb89800 (~460 MB) → the layer ELF's weight BO
  is a single ~460 MB BO holding regions A+B at offsets 0x0 and 0x1bc00000
  respectively (NOT two separate BOs — R46's "separate BO" reading was the
  device-address vs BO-offset ambiguity; the DDR_PATCH arg_off is a BO
  offset, patched onto the arg-0 BO base at submit).

=> task-1 (generate + verify 35B per-ctx MoE ELFs + lm_head ELF) is COMPLETE.

## Round 70 — task-2 MoE weight-BO packing landed in C, byte-verified (2026-09-10)

Ported the Round-50 layout to npu-infer/src/model.c (the engine's packer),
extended the model loader for the MoE/linear-attn tensor names, and verified
the C output byte-identical to the Python reference (which R50 had already
byte-verified against the moe-cap4 runtime captures):

- `LayerWeights` now carries up/gate/down_exps, share_*_exps, moe_router,
  shared_expert_gate, self_attn.gate_proj, qkv_proj, ssm_out/conv1d/norm/a/
  dt.bias/alpha/beta (18 new descriptors). Parser also accepts the suffixless
  F32 tensors `*.ssm_a` and `*.bias` (they are part of the 5 MB BO head).
- `npu_pack_moe_expert_pool(bo, mw, L)`: 512 MB pool, rows 0..100959 =
  478,146,560 B — up/gate alternating 32-row blocks (j = base+8*(i%4)+i/4),
  down in 8-groups [0,2,4,6,1,3,5,7].
- `npu_pack_moe_linear5_bo(bo, mw, L)`: 5,242,880 B — 328,192-B head
  (ssm_conv1d/norm/a/dt.bias/alpha/beta) + ssm_out windows (j = base+16*(i%2)+i/2).
- VERIFIED byte-identical for layers 0,1,6,20,30,38 (pool + 5 MB) vs the
  Python reference (tools/test_moe_pack.c + verify_moe_current_layout.py).

Still open (documented, blocked on the unidentified transforms / missing
captures): self_attn.gate_proj pool rows 100960..102623 (order under-
specified), the 2 MB qkv-format BO (R50-54), and the full-attn layers'
542 MB BO. These do not block the expert-pool + linear-attn core.

## Round 71 — DDR_PATCH offsets are BO-relative; device-VA wall de-risked (2026-09-10)

Decoded the DENSE 0.6B layer TXN (txn-elfs/layer_ctx1.txn) with the same
decoder as a control, then compared arg_idx semantics against the MoE TXNs:

| arg_idx | dense 0.6B (proven ABI) | MoE linear L0 | MoE full L3 |
|---|---|---|---|
| 0 | act (4 patches @0) | **weight** 544 patches, 0x0..0x1cb89800 (~460 MB) | weight 512 patches, 0x0..0x1cc9f000 |
| 1 | **weight** 384 patches, 0x0..0x942000 (~9.7 MB) | 4 patches @0 | 4 patches @0 |
| 2 | i5 (2 @0) | 4 patches, 0x0/0x3000 (moe_router) | same |
| 3 | i6 (2 @0) | 70 patches, 0x0..0x4cb200 (~5 MB) | 2 @0 |
| 4 | kv 16 patches, 0x0..0x1800000 (25 MB) | 8 patches @0..0xc000 | 8 @0..0x800800 |

Conclusions (corrects R46 / R61-64):

1. **DDR_PATCH arg_off is a BO-relative offset**, patched onto the arg BO base
   at submit — the dense path proves it (weight offsets 0..0x942000 into the
   10 MB BO, kv 0..0x1800000 into the 32 MB BO, all relative). The MoE ELF's
   arg_off values are likewise all ≤ 0x1cc9f000 (460 MB) with NONE in the
   R46 "absolute" ranges (0x40000000 act / 0x2000000 kv / 0xe000000 experts /
   0xc0000000 state). The R61-64 "device-VA wall" was a misreading of the
   arg_off field — the engine does NOT need to replicate absolute addresses.
2. **The MoE layer kernel has a DIFFERENT BO arg order than the dense kernel.**
   In the MoE ELF arg_idx 0 = the ~460 MB weight BO (region A at offset 0,
   region B at offset 0x1bc00000 — the 439 MB gap between them is where the
   expert pool lives, unreferenced by the layer ELF, matching R46/R50).
   The dense kernel puts act at arg_idx 0 and weight at arg_idx 1.
3. Region B (share_*/qkv/ssm_out/gate_proj) is NOT a separate BO — it is
   offsets 0x1bc00000..0x1cb8e200 inside the SAME weight BO (arg_idx 0).

=> task-3 wiring is unblocked at the address layer: allocate one ~460 MB
   weight BO + act + router/norms smalls + kv in the MoE kernel's arg order,
   submit via xrt::runlist (the proven dense mechanism). Remaining task-3
   unknowns: the MoE kernel's exact arg→BO map (act/kv slots) and the
   per-token expert GEMM sequences (setup_expert_up/down_gate_q4k exports).

## Round 72 — region B layout is decodable from the ELF's own BDs (2026-09-10)

The layer ELF's weight-BD list (arg_idx 0) IS the region A+B packing oracle —
no captures needed. Extracted the full unique (arg_off, len) set for the
linear layer (242 entries):

- Region A: off 0x0 (len 4736 + 18944 = 1- and 4-row reads) — norms/router/
  ssm smalls.
- Region B: off 0x1bc00000..0x1cb89800, all 4736-B reads at 0x12800
  (16-row) strides — the share_*/qkv/ssm_out/gate_proj array. Region B =
  0xf89800 = 3,444 rows × 4736 (16,310,784 B), matching R46's 16,310,784 B.
  Sub-tensor row spans (from the desc logical offsets): share_up rows
  0..255, share_down 256..383, share_gate 384..511(?), qkv @ row 384,
  gate_proj @ row ~2432..; the BDs' 16-row strides give the interleave
  (same window-order class as the expert pool).

=> The remaining task-2 pieces (region B qkv/ssm_out/share_* and the
   self_attn.gate_proj rows) can be derived from the ELF BD list + the
   tensor 4736-row windows — a bounded, capture-free mapping job. This is
   the concrete next step to close task-2, followed by the task-3 arg→BO
   wiring (MoE kernel order: arg0=weight, arg1=act?, arg2=router,
   arg3=norms, arg4=kv/state).

## Round 73 — MoE layer-kernel arg→BO map determined from DMA directions (2026-09-10)

Annotated the linear layer TXN's DDR_PATCHes with DMA direction (from the
queue-write registers, MM2S=BO→tile read, S2MM=tile→BO write) and bucketed
per arg_idx. This yields the MoE layer kernel's host-BO order (arg_idx = BO
slot, kernel slot = 3 + arg_idx):

| arg_idx | MM2S | S2MM | role | BO |
|---|---|---|---|---|
| 0 | 480 | 0 | read-only | **weight BO** (~460 MB: region A @0, region B @0x1bc00000) |
| 1 | 2 | 2 | read+write | **act BO** (hidden state) |
| 2 | 4 | 0 | read-only | **router + shared_expert_gate BO** (moe_router @0x3000) |
| 3 | 68 | 2 | mostly read | **norms/ssm smalls BO** (~5 MB) |
| 4 | 2 | 6 | mostly write | **kv/linear-state BO** |

=> The MoE kernel arg order is DIFFERENT from the dense kernel: weight is
   slot 3 (dense: act is slot 3). Wiring (task-3) is now concrete:
     run.set_arg(3, weight) ; set_arg(4, act) ; set_arg(5, router) ;
     set_arg(6, norms) ; set_arg(7, kv_state)
   plus the separate per-token expert GEMM kernels reading the expert-pool
   BO (task-2). Remaining: exact region A/B byte offsets per tensor (the
   BD list, Round 72) + the expert GEMM sequence ABI.

## Round 74 — per-arg BO layouts (BD len/offset map, linear layer) (2026-09-10)

Full (arg_idx, buffer_length, arg_off) BD map for the linear layer (moe_layer_ctx0.txn):

- **arg1 = act**: 4 × 1024-B reads @0 (the 2048-bf16 hidden state in 4 chunks).
- **arg2 = router**: 2 × 3072-B @0 + 2 × 32768-B @0x3000 (moe_router/shared_gate
  DMA — the `len` is per-iteration, not total; router tensor = 1 MB).
- **arg3 = norms/ssm smalls**: 2 × 12288 @0, 2 × 32768 @0x10200 + @0x30200,
  then 34 × 4736-B rows at 0x25000 (32-row) stride from 0x50200 — the
  ssm_conv1d/alpha/beta + norms packed into ~5 MB (0x0..0x4cb200).
- **arg4 = kv/state**: 4 × 12288 @0 + 4 × 524288 @0xc000 (the GateDeltaNet
  linear-attn state, 512 KB, at offset 0xc000).
- **arg0 = weight BO**: region A (off 0x0: 32 × 18944-B 4-row + 32 × 4736-B
  1-row reads) + region B (off 0x1bc00000..0x1cb89800: 480 × 4736-B reads at
  16-row stride — share_*/qkv/ssm_out/gate_proj in 4736-row window form).

The remaining bit to fully close the packer is the per-BD DMA dims (the
2D descriptor strides) for region A + B, which fix the exact window order —
decode_txn.cpp's patch table already emits these; the next session runs it
in --decode-only mode over the MoE .txn files and matches BDs to tensor
4736-row windows. Then the task-3 wiring code (arg0..arg4 order above) +
the expert GEMM sequences can be landed and validated on the NPU.

## Round 75 — arg3 (norms BO) == task-2 5MB linear-attn BO (2026-09-10)

Cross-check of the arg3 BD offsets against the task-2 5 MB BO layout confirms
they are the SAME BO: arg3 reads ssm_conv1d @0 (12288-B chunk), ssm_alpha
@0x10200, ssm_beta @0x30200 (32768-B chunks), then ssm_out 4736-rows starting
at 0x50200 — and 0x50200 = 328,192 B = exactly the task-2 head size
(ssm_conv1d 65536 + ssm_norm 256 + ssm_a 128 + ssm_dt.bias 128 +
ssm_alpha 131072 + ssm_beta 131072). So `npu_pack_moe_linear5_bo` (task-2,
byte-verified) IS the layer kernel's arg3 BO. Remaining arg-0 content to
resolve: the input/post_attention_layernorm (4096 B each) placement (region A,
arg0's 64 @0x0 patches) and the region B window transforms (share/qkv/ssm_out/
gate_proj — the closed reorder_cpy, still the known blocker from R45-54).

## Round 76 — expert GEMM sequence ABI + per-token MoE flow (2026-09-10)

Disassembled the MoE expert sequence generators (libqwen3_6_moe_npu.so):

- `_gen_sequence` (full-attn layer, 0x971d0) emits, in order:
  `_send_hidden_states` → `_send_rms_weights` → `_send_rope_rms_weights` →
  `_receive_kv_cache` → 4× `_move_weights` (q/k/v/o or gate/up/down via
  weight_desc at +0x378/+0x480/+0x638/+0x5e0) → `_move_kv_cache` →
  **`setup_expert_up_gate_q4k` ×2 (e=4,0xc) + `setup_expert_down_gate_q4k`
  ×2 (e=6,0xe)** — i.e. the layer ELF (task-1) ALSO carries the SHARED
  experts' dequant+mm, reading their weights from the weight BO.
- `setup_expert_{up,down}_gate_q4k(seq, a, expert_idx, c, d, e, woff)`:
  c/d = weight_desc dims (fields 0x58/0x10), e = DMA channel/BD id
  (4/6/0xc/0xe = 2 experts × up_gate/down), woff = u64 weight offset. Each
  emits ONE `npu_dma_memcpy_nd`; tile math uses 0x4a00 (=18944=4×4736) and
  0x1280 (=4736) — the SAME 4736-row convention as task-2's pool.
- `send_manual_expert_{up,down}_gate_q4k(seq, a, expert_idx, c, d, e, woff)`
  — the PER-TOKEN routed expert GEMMs (top-8), reached through the vtable
  (reloc 0x1d5408 → 0x197af0), expert_idx indexes a runtime per-expert
  offset table at 0x1d5d58 (filled by desc.build).
- `cpu_func::_moe_gate(float const*, int, int, int)` — the CPU router
  (OpenMP-parallel), returns `moe_routing` (top-k expert indices + weights).

Per-token MoE forward = CPU router → 8× `send_manual_expert_up_gate` +
8× `send_manual_expert_down` (routed experts) → layer ELF (attention +
shared experts + norms + router) → lm_head. The single-launch runlist
(task-3) = batch all of these into ONE xrt::runlist submit/token; the
layer ELF + lm_head ELFs are already generated (task-1), and the expert
pool BO is already packed (task-2).

## Round 77 — ALL 35B xclbins are the SAME "MLIR_AIE" kernel → one-runlist is feasible (2026-09-10)

Compared the 35B xclbins (layer / lm_head / dequant_mm / conv / mm) with
xclbinutil + kernel-XML extraction:

- Every xclbin exposes the SAME kernel: `MLIR_AIE`, `dpu_kernel_id="0x901"`,
  instance `MLIRAIE`, identical arg signature
  `(opcode u64, instr char*, ninstr u32, bo0..bo4 void*)`.
- The dense 0.6B layer.xclbin is the SAME kernel design (0x901 / MLIRAIE).

=> There is ONE generic AIE kernel; the per-run instruction stream (layer,
   expert dequant+mm, lm_head) is a runtime-loaded ELF (aiebu-assembled from
   the sequence generators). The arg SEMANTICS (bo0=weight vs bo0=act) are
   set by the ELF's DDR_PATCH arg_idx — the sequence generator, not the
   xclbin — which is why the MoE layer ELF differs from the dense ELF in
   arg order (Round 73).

Consequence for task-3: a single hwctx (one registered xclbin) can host ALL
the MoE sub-kernels as modules, so the per-token forward —
   8× send_manual_expert_{up,down}_gate + layer ELF + lm_head ELF
— CAN be batched into ONE xrt::runlist submit/token, exactly like the dense
path batches 28 layer runs + lm_head. This closes the "is one-runlist even
possible" question affirmatively. Remaining: pin the send_manual_expert_*
call args + write the per-token sequence harness, then validate on the NPU.

## Round 78 — the runtime's own expert forward ALREADY uses a runlist (2026-09-10)

Disassembled `qwen3_6_moe_expert_prefill_context::forward` (0x897f0). Its
per-token expert path is:

    gen_dequant_mm_512 ×3        (the routed-expert dequant+mm sequences)
    npu_app::create_run ×2       (up_gate + down runs)
    xrt::runlist::reset() → add(run) ×2 → execute()
    cpu_func::_moe_gate          (CPU router, top-k)
    cpu_func::_pack_moe_tasks    (pack expert tasks)
    cpu_func::_sigmoid / _elementwise_mul / _exp_glu   (activations)

So the runtime ALREADY batches the routed expert GEMMs into a single
xrt::runlist submit — the runlist mechanism is proven in-production for the
experts. The per-token decode is therefore ~3 separate submits:
  (1) expert runlist, (2) layer ELF run, (3) lm_head run (get_logits).
The objective "one xrt::runlist submit/token" = MERGE (1)+(2)+(3) into ONE
runlist — architecturally valid since all three are the same MLIR_AIE kernel
(Round 77) and the runtime's runlist already spans the 2 expert runs.

Also pins the expert GEMM generator: the per-token routed experts use
`gen_dequant_mm_512(npu_sequence*, u32, u32, u32, u64, u64, int, flm_dtype_t)`
(NOT send_manual_expert_* — those are the shared/one-time path). Its 3 call
sites + 2 create_run's = the up_gate/down GEMM pair per routed expert.

## Round 79 — region B transform DERIVED: reorder_cpy = 16-window A/B interleave (2026-09-10)

Called the runtime's own `qwen3_6_reorder_cpy` (constprop.2 clone, dtype=8)
on the 8704-row region-B tensors with the args captured from the live load
(R47 gdb), via a new probe (tools/verify_moe_reorder_qkv.cpp). It works
(no segfault with the R47 args) and its output is byte-exact against a
SIMPLE formula:

    out[o] = in[o//2 + 8*(o%2)]   per 16-window block
    window = 4736-B slice from file offset 0 (stride 4736)

(verified for qkv windows 0..15: all 16 rows match). Per-tensor reorder
output sizes (layer 0):

    share_up/gate/down  (n=512)   -> 236 windows each (14 blk + 12)
    qkv                 (n=2048)  -> 3760 windows (235 blk)
    gate_proj           (n=4096)  -> 1888 windows (118 blk)

This is a THIRD window order, distinct from the expert pool (32-row
`8*(i%4)+i//4`) and the 5 MB BO (32-row `16*(i%2)+i//2`) — matching R49's
"reorder_cpy serves a different (linear-attn) purpose". The region-B
transform is no longer "closed": it is the reorder_cpy 16-window A/B
interleave, reproducible in C. Remaining to close the packer: the exact
reorder-output → region-B BO offset mapping (the ELF BD 16-row/4-row stride
pattern reads the A/B halves via 2D DMA), i.e. reconcile the 6356 total
reorder windows with the 3444-row region-B span (desc offsets qkv@384,
gate_proj@2432 — the logical table vs physical interleave).

## Round 81 — fresh re-capture landed (77 GB); task-2 pool byte-confirmed (2026-09-10)

Ran the runtime load under the lean interposer (vision-on config, ASLR
retry #1 succeeded): `/home/bcloud/.cache/moe-cap-rb` = 77 GB, 523 sync +
34 preinsts (CAP_DUMP_BIG) + 35 ELF + 276/283 post/waitpost dumps.
Structure matches R48: 40 × 512 MB expert pools + 4 × 542 MB full-attn pools
+ 30 × 5 MB + 47 × 2 MB + 92 × 3 MB + 27 × 10 MB + 54 × 11 MB + state BOs.

**Verification (the point):** layer-0's 512 MB pool rows 0..100959 are
**byte-identical** to `npu_pack_moe_expert_pool` (task-2) — sha256
`9752f7aa…` matches on a FRESH capture (not just the R50 reference), and
the gate_proj region (rows 100960..102623) is non-zero (7.8 MB) — confirms
gate_proj lives in the pool (R50) and task-2's packing is correct.

**Remaining blocker reconfirmed:** a 128-B slice of share_up's raw bytes is
absent from ALL 412 bo_to sync dumps → the share_* (and qkv) tensors are
VALUE-transformed (not reordered), exactly as R45/R50-54 concluded. The
reorder_cpy constprop.0 clone (0x7abb0) — the likely producer — hangs on
direct call, and constprop.2's A/B interleave (Round 79) preserves raw
bytes, so it is NOT the region-B producer. The capture oracle now on disk
(moe-cap-rb) holds the transformed share_*/qkv bytes for a future replay
path (use-captured-bytes-directly), but the int8→int16 dequant scale for
qkv (R51-54) and the share_* transform remain the closed-source unknowns.

## Round 82 — layer-ELM weight BO is map-written; region B uncapturable (2026-09-10)

Analyzed the fresh capture's allocation manifest + the CAP_DUMP_BIG preinsts:

- EXTBO order: 40 × 512 MB pools, then per layer a repeating
  [2 MB, 5 MB, 3 MB]×4 + [2 MB, 1 MB, 128 MB] set. The 40 pools are the
  expert pools (sync'd, byte-confirmed R81); there is NO separate ~460 MB
  layer-ELF weight BO among the sync'd EXTBOs.
- preinsts_001 (runlist-1 arg dump) shows the EXPERT GEMM runlist: pool at
  slot i5 (512 MB), full-attn pool at i4 (542 MB), act/workspace/norm
  smalls elsewhere — NOT the layer ELF's arg map (R73).
- desc hexdump (dump_moe_desc): the weight BO device base 0x1ccc4000 /
  0x1cb9c000 appear as desc fields — confirming the DDR_PATCH arg_off is a
  DESC-LOGICAL offset remapped onto a map-written weight BO, not a BO-relative
  offset into a sync'd BO (corrects Round 71's dense-derived generalization).

=> The layer-ELF weight BO (region A+B, incl. the transformed share_*/qkv) is
   map-written (never xrtBOSync'd) and therefore absent from the sync capture
   AND from the preinsts arg-dump (which only shows the expert GEMM runlist).
   This is the same conclusion as R50's byte-flip differentials (qkv map-only).
   The share_*/qkv value-transform remains the closed-source blocker; the
   capture oracle (moe-cap-rb) holds the expert pools + linear BOs (both
   byte-verified) but NOT the layer-ELF region-B content.

## Round 83 — region-B transform DERIVED: 8704-tile trim + A/B interleave (2026-09-10)

Re-examined the blocker and found it is NOT a closed value-transform. The
layer-ELF region-B content is:

    tile[i] = row[i][0:4736]            (trim each 8704-B Q8_0 tile)
    out[o]  = tile[o/2 + 8*(o%2)]        (A/B interleave, 16-tile blocks)

placed at the desc offsets (share_up@0, share_gate@128, share_down@256,
qkv@384, gate_proj@2432 → 3456 rows = 16,367,616 B). The tile counts match
the desc allocations EXACTLY (128/128/128/2048/1024) — and the ELF BD reads
(Round 72) follow those same desc offsets, so the desc table IS the physical
region-B layout here. The earlier R45 "8704-row trims" negative did NOT
include the A/B interleave; the "4736-window slicing" behavior I saw in
Round 79 was reorder_cpy on RAW (untrimmed) qkv — the runtime pre-trims the
tiles first (load_linear_weights), then reorder_cpy only shuffles them.

Implemented + committed: `npu_pack_moe_region_b()` (model.c), structure
verified (tile0/tile8/tile1 interleave + per-tensor offsets match). The
region-B weight packing is no longer a blocker — the last weight-BO piece
(expert pool + 5MB linear + region B) is now packable from the model file.
Byte-exactness vs the runtime remains gated on task-4 (moe_ffn_cpu), since
the runtime's weight BO is map-written (R82).

## Round 84 — CPU reference banked (task-4 correctness baseline) (2026-09-10)

Ran tools/qwen36_full_ref.py (the moe_ffn_cpu-equivalent numpy reference):

    prompt [151644] -> greedy next token 76740
    first-position per-layer hidden states saved (40 x 2048)
      -> /tmp/ref_first_hidden.npy

This is the ground truth for the task-4 correctness gate (token parity + the
per-layer hidden-state byte compare). The engine side (MoERuntimeLayerEngine +
routed-expert GEMMs) must reproduce token 76740 and the 40 hidden states once
the routed experts are wired.

## Round 85 — expert-forward args pinned: gen_dequant_mm_512 = shared experts (2026-09-10)

Dumped the desc's per-layer weight entries (desc.0x70, 40 entries) with a
heap-allocated desc (tools/dump_moe_experts.cpp approach):

- desc.0x10 = 2048 (H), desc.0x58 = 512 (IM_EXP).
- gen_dequant_mm_512's woff2 offsets (entry.0x2b8/0x208/0x158) =
  0x1bc00000 / 0x1bc94000 / 0x1bd28000 = share_up / share_gate / share_down
  (the region-B desc offsets). Only entries 0 and 3 are populated — the
  per-layer shared-expert entries.

=> The expert_prefill_context's gen_dequant_mm_512 ×3 is the SHARED-expert
   FFN (share_up/gate/down GEMMs, reading region B), NOT the routed experts.
   The routed experts are the vtable-dispatched send_manual_expert_* path on
   the expert-pool BO (task-2). Full per-token forward =
   CPU router -> shared FFN (gen_dequant_mm_512, region B) -> routed FFN
   (send_manual_expert_*, expert pool) -> layer ELF (attention/norms/router)
   -> lm_head. Region-B offsets for the shared FFN are now pinned.
