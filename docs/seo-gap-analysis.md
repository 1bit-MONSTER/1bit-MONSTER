# SEO content-gap analysis — 1bit.MONSTER

*Generated with `scripts/gap_analysis.py` against the engine's own `/v1/embeddings`
(nomic-embed-text-v1-GGUF). Scores = cosine similarity between the topic's
embedding and the site's best-matching page. Re-run after new posts.*

## Method

- 30 curated high-intent topics in the local-AI / NPU / 1-bit space
- each topic embedded; compared against every page's title+description embedding
- **> 0.65 = owned** (a dedicated page exists) · **0.55–0.62 = mentioned but not owned** · **< 0.42 = gap**

## Result: 0 true gaps, 8 "covered-but-not-owned" topics

The site already owns the technical core (29/30 topics ≥ 0.55). The
opportunity is **dedicated pages for topics currently only mentioned in
passing** — each deserves a headline-grade post.

| Topic | Best coverage | Page | Status |
|---|---|---|---|
| AMD Strix Halo AI performance | 0.771 | 1bit-post-halo-reviews | owned |
| Ryzen AI NPU inference | 0.696 | 1bit-post-qwen36 | owned |
| model quantization benchmarks | 0.686 | 1bit-benchmarks | owned |
| speculative decoding | 0.677 | 1bit-post-dspark | owned |
| LLM token throughput | 0.674 | 1bit-post-tokenrouter | owned |
| Kraken Point NPU | 0.674 | 1bit-post-npu-v12 | owned |
| AI on Windows NPU | 0.670 | 1bit-post-qwen36 | owned |
| run LLM on AMD GPU | 0.670 | 1bit-post-historical | owned |
| CPU-only LLM inference | 0.668 | 1bit-post-tokenrouter | owned |
| Zamba2 model | 0.655 | 1bit-post-zyphra | owned |
| offline AI assistant | 0.630 | 1bit-jarvis | owned |
| 1-bit quantization | 0.630 | 1bit-post-one-engine | owned |
| FastFlowLM | 0.628 | 1bit-post-flm-zoo | owned |
| ROCm vs CUDA | 0.628 | 1bit-benchmarks | owned |
| GGUF format explained | 0.614 | 1bit-post-binary-formats | owned |
| int4 vs int8 quantization | 0.613 | 1bit-post-sprint | owned |
| local LLM privacy | 0.613 | 1bit-post-historical | owned |
| llama.cpp alternatives | 0.611 | 1bit-post-historical | owned |
| AI coding agent local | 0.608 | 1bit-post-coding-agent | owned |
| local RAG embeddings | 0.583 | 1bit-post-flm-zoo | **close: dedicated post** |
| model compression techniques | 0.579 | docs-models | **close: dedicated post** |
| HuggingFace model zoo | 0.564 | 1bit-post-flm-zoo | **close: "One engine, 1,775 models"** |
| vLLM alternatives | 0.559 | 1bit-post-historical | **close: stack post** |
| voice assistant local | 0.556 | 1bit-post-historical | **close: JARVIS deep-dive** |

## Close plan (blog posts, in priority order)

1. **One engine, 1,775 models** — the census story (552 arch tokens, 100% HF coverage, 317,310 checkpoints) — killer-stat post
2. **Local RAG with our own embeddings** — first-mover on the new capability
3. **What 1-bit does to a model** — compression explainer
4. **ROCm vs CUDA from the 1-bit trenches** — debate + real numbers
5. **JARVIS: the zero-Python local voice assistant** — product deep-dive
6. **The stack (no vLLM)** — architecture post
