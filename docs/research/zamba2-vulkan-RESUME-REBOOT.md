# RESUME NOTE — 2026-08-03 (RESOLVED)

> RESOLVED: validate_zamba2_vulkan now passes token-for-token on both
> 1.2B and 2.7B (0 mismatches, ~51/97 ms per token). The device-lost
> hang was an FFN buffer sizing bug (GpuBuffer::size() returns bytes);
> the token mismatches were scan state/x indexing, descriptor binding
> shifts for null-output dispatches, and the silu_mul→swiglu shader_map
> alias. See commit c5436c1. Remaining from the old plan: bench vs HIP,
> README ZINC column (#1461).

# Why reboot (original)
validate_zamba2_vulkan reproducibly failed with RADV context loss
("CS has been cancelled because the context is lost") on the first vkQueueSubmit
during Zamba2VK init — likely a GPU hang left by the in-flight Vulkan port
(agent-cf's validation run). No amdgpu errors visible (dmesg restricted).

## State preserved
- My CPU correctness work: committed + pushed on `fix/zamba2-cpu-correctness`
  (a97a877), PR #1462 open. CPU path verified vs numpy reference (top-3
  logits identical, 38 layers ~1e-5).
- agent-cf's Vulkan port: UNCOMMITTED files in the working tree (safe on disk):
  - src/backend_zamba2_vulkan.cpp (new backend, ZAMBA2_VK=1 gate)
  - engine/gpu/zinc_cpp/src/shaders/{mamba2_scan,mamba_conv1d,group_rms_norm,mamba_rope}.comp
  - src/model_router.cpp, src/backend_manager.cpp (wiring)
  - engine/gpu/zinc_cpp/{include/compute_engine.h,src/compute_engine.cpp} (dispatch_off etc.)
  - CMakeLists.txt (validate_zamba2_vulkan target)
  - tools/validate_zamba2_vulkan.cpp
  - NOTE: agent-cf's session dies with the box; its files may be mid-edit.

## Post-reboot plan (all items 1-3 done in c5436c1)
1. Fix mamba2_scan.comp line 64: `A_bar = exp(dt_sp * (-exp(pr[pr_l + head])))`
   → `exp(dt_sp * pr[pr_l + head])` — mirrors the FIXED CPU convention
   (GGUF ssm.a is already -exp(A_log), #1460/PR1462).
2. Restructure the hybrid path in backend_zamba2_vulkan.cpp to the verified
   reference (currently mirrors the OLD buggy CPU structure):
   concat(hidden, embedding) → post_attention_norm(2*d_model) → QKV(input 2*d_model)
   → mamba_rope → flash_attn (scale sqrt(2/head_dim)!) → o_proj → ffn_norm →
   SiLU FFN → ssm_mix → mamba decoder. Needs an `embedding_` copy buffer.
3. Rebuild validate_zamba2_vulkan + rerun vs CPU reference (should be
   token-for-token identical now that the CPU is verified).
4. Then: bench tok/s vs HIP; fix README ZINC column (#1461).

## UPDATE 22:45 — validation state on lavapipe (software Vulkan, no GPU risk)

### Fixed and verified so far (all on the lavapipe path)
1. mamba2_scan.comp: A-convention (exp(dt_sp*A), not -exp) — #1460
2. mamba2_scan.comp: s_red[8] → s_red[64] (subgroup-size-1 → 64 subgroups → OOB LDS)
3. backend reset(): fill_zero offset-0 bugs — was clobbering A/D/dt_bias and conv
   weights (params state region starts at 3*n_head*n_layers*4; conv state at
   n_layers*(d_conv*conv_dim+conv_dim)*4)
4. pre_ff_norm alloc'd d_model but receives the 4096-float concat norm (OOB
   upload — likely the pre-reboot GPU-wedge cause); qw/kw/vw alloc'd n_at*d_model
   but weights are n_at*2*d_model
5. pure-path missing copy hidden→tmp before mamba_block (read stale tmp_)
6. flash_attn scale: added push-constant override (pc.scale != 1.0) for
   Zamba2's sqrt(2/head_dim)
7. Hybrid branch rewritten to reference structure (concat → post_attn_norm(2d)
   → QKV(2d) → rope → flash_attn(sqrt(2/hd)) → o_proj → ffn_norm → FFN →
   ssm_mix → mamba decoder)

### UNRESOLVED (agent-ce actively debugging)
GPU L0 in_proj (gemv_f32) output wrong AND nondeterministic across runs:
- normed tmp_ readback sometimes byte-exact-correct (matches CPU
  [-0.45261, 0.22722, ...]), sometimes garbage (run-to-run, same binary)
- when normed is correct, gemv y[0..3] = [-0.079, 0.437, -0.341, -0.208] vs
  CPU [-0.376, 0.224, -0.437, -0.912]; partial-K=256 + x_off=7 comes closest
  (err 0.0015) but nothing matches exactly
- YET the scan output y_inner[0..3] is ~right (0.0098 vs 0.0099) — implying
  the conv's xBC input (in_proj rows 2048+) is RIGHT while row 0 is wrong
- suspicion: gemv row-0/descriptor/binding issue OR uninitialized read;
  NOT the shader math (dense lm_head path validated with same shader)
- validator segfaults at exit in ~Zamba2VulkanBackend → vkDestroyBuffer abort
  (double-destroy of GpuBuffers in layers_/scratch — check the move/assign
  paths in init(); GpuBuffer itself has correct move semantics)

### Coordination
- agent-cf (port author) dead; agent-ce ALIVE and editing the same files —
  edits get clobbered between agents; take live-workspace locks or serialize.
- Do NOT run validate on the real AMD GPU: it wedges the device (6 resets).
  Use VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json (lavapipe).
