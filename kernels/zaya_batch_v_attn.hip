// zaya_batch_v_attn.hip — Minimal V-only attention for batch path (cca_attn_kernel stub replacement).
//
// Broadcasts V from NKV→NQ heads (GQA) so o_proj can produce correct H-dim output.
// This gives degenerate (first-token-only, no cross-token attention) but VALID output,
// which is much better than the uninitialized-memory garbage the empty stub produced.
// For real attention in batch, add per-sequence KV caches.
//
// Called from zaya_forward_batch after projections + v_interleave fill d_tmp[QD:QD+KD] with V.
// Input v[NKV*HD], output out[NQ*HD].
__global__ void gqa_broadcast_k(const __half* __restrict__ v, __half* __restrict__ out, int nq, int nkv, int hd) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = nq * hd;
    if (idx >= total) return;
    int h = idx / hd, d = idx % hd;
    int kv = h / (nq / nkv);
    out[idx] = v[kv * hd + d];
}
