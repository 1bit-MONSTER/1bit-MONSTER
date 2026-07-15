// zaya_skip_fixup.hip — Post-router fixup for the MoD skip expert (idx == n_exp = 16).
// If the router selects the 17th expert (skip → idx >= n_exp), this kernel:
//   1. clamps expert_idx to 0 (valid expert; the ×0 prob makes its output vanish)
//   2. zeros d_tmp (so residual_scale gets 0 expert contribution)
//   3. sets skip_flag = 1 (host uses this to conditionally skip the wmma section)
// The host must synchronize before reading skip_flag.
__global__ void fixup_skip_expert_kernel(int* expert_idx, __half* d_tmp, int* skip_flag, int n_exp, int n_exp_plus1, int h) {
    int idx = *expert_idx;
    if (idx == n_exp || idx >= n_exp_plus1) {
        *expert_idx = 0;
        for (int i = threadIdx.x; i < h; i += blockDim.x)
            d_tmp[i] = __float2half(0.0f);
        *skip_flag = 1;
    } else {
        *skip_flag = 0;
    }
}
