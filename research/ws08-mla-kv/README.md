# WS-08 — MLA & KV Cache

**Status:** 🔄 P0 gauge probe done — see `FINDINGS.md` + `codec_gauge_probe.py`
**Papers:** 2606.16310 (QK-Normed MLA), 2607.23054 (MLA-Bottleneck), 2607.27269 (MLA-Draft-Functional), 2607.12550 (JoLT), 2607.01831 (Lynx), 2607.20538 (Codec-Gauge), 2504.19874 (TurboQuant), 2607.06601 (TriRoute)
**Owner:** kv/mla

## Goal

Own the MLA path (Kimi K3: 24/93 gated-MLA layers + 69/93 KDA layers) and push KV compression beyond today's FD/i8 benches.

## Theory

QK-Normed MLA (2606.16310) is a free implementation: RMSNorm decomposes into static affine (absorb into query projection) + dynamic RMS statistic (one inverted scalar per token) — **QK normalization without full key caching**. MLA-Bottleneck (2607.23054) tells us what the cKV latent keeps/discards. JoLT (2607.12550) is the strongest published KV compressor: third-order tensor view, joint low-rank + quantized residual. Lynx (2607.01831) enables decode-before-full-transfer for cross-device KV handoff. Codec-Gauge (2607.20538) is a cheap post-training layer — learned orthogonal gauges around existing quant backends.

## Tasks

### P0
- [x] `codec_gauge_probe.py` — orthogonal channel gauge → **2.96× lower MSE at int8, 1.57× at int4** on correlated KV channels; must pair with per-channel scales (FINDINGS.md)
- [ ] QK-Normed MLA absorption on Kimi K3 gated-MLA decode; verify numerically vs reference
  - [x] Mechanism reference test (arXiv:2606.16310 App. E): `qk_norm_absorb_probe.py` — absorbed latent path ≡ explicit full-key QK RMSNorm, max diff 3.6e-14 (fp64) / 3.1e-5 (fp32); blockwise content+RoPE ✓; scalar cache = 0.78% of full-key cache

### P1 (next)
- [ ] Rerun gauge probe on real KV dumps (`tools/capture_attn.cpp`); if ~3× holds, add per-head learned gauge to the KV quant path (128×128 rotation, write-time apply, read-time invert)
- [ ] JoLT-style tensor decomposition for KV at long context (bench vs TurboQuant 3.5-bit/channel)

### P2 (if the bet pays off)
- [ ] Lynx-style progressive KV handoff for NPU↔GPU↔CPU transfer

## Validation

- KV bytes/token; ppl at 32k context; decode tok/s at 32k vs 57.1 GB/s FD baseline
- Kimi K3 layer-level correctness vs reference
