# FastFlowLM Qwen3-0.6B weight-BO capture (2026-09-01)
# Real runtime (test_qwen3_npu, XRT path) with LD_PRELOAD interposer on
# reorder_cpy + xrt::bo::sync. Full model run (prefill+decode).
# 477 reorder_cpy calls, 903 FROM-device BO syncs.
# BO size histogram: {1048576: 679, 2097152: 112, 6291456: 56, 134217728: 56}
#
# Layer-0 mapping (event order):
#   R0  q_proj   -> B0  (1MB f32 = 256x1024)
#   R1  k_proj   -> B1  (1MB)
#   R2  v_proj   -> B2  (1MB)
#   R3  o_proj   -> B4/B5 (1MB blocks)
#   kv cache     -> B3  (128MB)
#   gate/up      -> B6/B7 (2MB)
#   down         -> B8  (6MB)
# The q_proj reorder output (permuted tiles) is in rc_dst_0_1310720.bin.

## B0 formula status (round-28c)

B0 (the first 1MB f32 BO) is a load-time weight buffer — byte-identical across
runs. Its values (range [-6.3, 13.1]) do NOT match any host-side dequant of
the q_proj tiles:

- (q - zp)*scale (g-major or row-major, signed/unsigned/zp-fold): corr ~0.001.
- int8-requant (per-col/per-row S_col = max/127): maxdiff > 127.
- Any q*s+z combination from the tile's 512 bf16 scale/zero bytes: 0 matches
  for B0[0,0..2] (the required effective scales ~0.101/-0.103 do not exist in
  the tile).
- The q_proj dequant output should be 2048x1024 f32 = 8 MB (8 blocks of 1MB);
  B0 is one 256-row block, but its values are not a (q-zp)*scale of those rows.

Conclusion: the dequant kernel (dequant.xclbin) produces a weight
representation that differs from the host-side torch2aie dequant — likely a
device-side fixed-point or (a,b)-pair form. The CAPTURE itself is the
deliverable: the exact f32 weight BOs the NPU consumes, reproducible by
re-running the interposer. The npu-infer mm test can feed these BOs directly
without knowing the kernel's internal formula.

## Round-28d — B0 identity exhausted

Systematic matching of B0 against every layer-0 projection's (q-zp)*scale
dequant (q/k/v/o/gate/up/down, all 256-row blocks, k in {1, 22}), the
transpose x22 form, int8-requant variants, and brute-force tile-byte
combinations: best meandiff ~0.058 (noise level; a real match would be ~0).
B0 is uncorrelated with decode-time activation syncs (corr -0.005) but is a
deterministic load-time buffer (byte-identical across runs). Conclusion: the
1MB BO at sync #0 is not the q_proj's host-dequant; either the event pairing
differs (async dequant runs) or the dequant kernel emits a genuinely different
representation. The complete capture (903 BOs, /tmp/cap) remains the
deliverable — feed the mm kernel the BOs directly; the internal formula is
not required for the feed path.
