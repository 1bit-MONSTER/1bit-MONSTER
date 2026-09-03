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
- rows 100960..113358 (share_up/down/gate + qkv + self-attn gate + router +
  ssm + norms + splices; 12,399 rows ≈ 58.7 MB) — same signature-mapping
  approach applies (tools/verify_moe_bo_layout.py machinery).

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
