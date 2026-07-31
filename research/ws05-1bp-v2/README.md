# WS-05 — 1BP v2: Expanded-Rank Ternary Format

**Status:** 🔄 P0 probe done — see `FINDINGS.md` and `externd_probe.py`
**Papers:** 2607.13511 (ExTernD), 2604.03336 (NativeTernary), 2602.06694 (NanoQuant), 2502.02631 (ParetoQ), 2606.26650 (CAT-Q)
**Owner:** format

## Goal

Evaluate ExTernD-style factorization (`A ≈ B·diag(D)·C`, ternary B/C, expanded rank k = μ·min(m,n)) as a 1BP v2 extension — "ternary with tunable accuracy."

## Theory

ExTernD's central claim: expanded rank (μ > 1) lets later factor planes correct earlier quantization error, with residual **provably** monotone-decreasing → approaches bf16 accuracy arbitrarily closely. NativeTernary (2604.03336) informs the wire format (2.0 bits/weight, 460× less framing than GGUF); NanoQuant (2602.06694) is the sub-1-bit low-rank cousin; ParetoQ (2502.02631) frames where 2-bit/ternary sits in the accuracy-hw tradeoff.

## Tasks

### P0 (do now) ✅
- [x] `externd_probe.py` — real BF16 LLM weights; **result: monotone decrease validated; TQ2 + 64 correction planes = −15% error at +0.51 b/w; full ExTernD loses to direct ternary at equal bitrate** (see FINDINGS.md)

### P1 (next)
- [ ] ppl confirmation on Bonsai-1.7B (WS-00 harness): TQ2 vs TQ2+64 planes
- [ ] 1BP v2 converter + C++ decoder: correction-plane extension (TQ2 core + n_planes × ternary B/C + fp32 d), versioned in the 1BP header

### P2 (if the bet pays off)
- [ ] ship/kill decision vs GGUF Q2_K on 4 models; wire into WS-06 precision router for sensitive layers

## Validation

- ppl vs FP16 for 0 / 16 / 64 / 256 correction planes
- tok/s impact of extra factor-plane matmuls (B·diag(D)·C = two ternary matmuls + scaling → reuse TQ2 kernels)
