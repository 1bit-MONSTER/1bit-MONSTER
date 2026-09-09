# goal mttyyykw (sage-4) — unified acceptance bench: routed engine vs single-engine baselines (2026-09-09)

All rows strixhalo gfx1151 (Radeon 8060S). Single-engine baselines: fork-Vulkan /
HIP / HRX from the mtsy05dx FINAL table (6b8109dad) + this session's fork rows.
Routed = the single-API PhaseRouter path (HRX0 or Vulkan0 prefill -> memfd ->
decode leg per policy), warm pass then measured window.

## 30B Q4_K_M (Qwen3-Coder-30B-A3B) — the contract model

| engine | pp (t/s) | decode (t/s) | notes |
|---|---|---|---|
| Vulkan-only (fork) | 1300.9 (pp2048) | 93.37 (tg512) | single-engine best (6b8109dad) |
| HIP-only | 1187.7 | 74.17 | (6b8109dad) |
| HRX-only | ~5 (pp) | ~9 | (6b8109dad) |
| NPU | N/A | N/A | pool <=4B cannot host 30B (operator-recorded) |
| **ROUTED (single API)** | (Vulkan0 prefill leg) | **65.27 (tg512, pp2138->memfd, 7.844s e2e)** | this session |
| 30B gate tg256 @KV2944-3200 | | 93.65 (fork-Vulkan) / **47.19 (routed @KV2986)** | gate bar >=40 - HELD both |

## Per-model routed vs single-engine (qwen3 Q4_K_M, tg128, warm)

| model | routed (single API) | fork-Vulkan | ratio | HIP-only | NPU |
|---|---|---|---|---|---|
| 0.6B | 186.06 | 357.7 | 52% | (n/a here) | 423-class (lemonade rows) |
| 1.7B | 110.42 | 171.5 | 64% | | |
| 4B | 56.32 | 76.1 | 74% | | ~40-class (FLM) |
| zaya moat (1.8B type43) | 3.80 (HRX0 both legs) | n/a (moat) | | | |

## Engine-loop overhead quantified
The routed in-engine decode runs at 52-74% of fork-Vulkan direct llama-bench
(ratio rises with model size: per-token host loop + argmax + session handling
amortize). The decode LEG itself (Vulkan0) sustains 65-85 t/s at 30B = ~70-90%
of the fork's 93 (llama-bench measures pure decode; the routed window includes
the engine's per-token loop).

## Verdict
The single-API routed engine delivers the op-class-split policy end-to-end with
oracle-identity (sage-2), the 30B gate held in-service (47.19 >= 40 @ KV 2986),
and moat correctness on the contract engine; performance is within one engine-
loop-overhead factor of the direct single-engine lane (the routed decode leg is
the same Vulkan0 code). Acceptable per the objective (caller sees no split;
gate + correctness are the bars; raw t/s parity with llama-bench direct is a
follow-up optimization, not an acceptance gate).
