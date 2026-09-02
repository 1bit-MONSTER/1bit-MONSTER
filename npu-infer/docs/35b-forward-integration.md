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
