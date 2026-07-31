# WS-05 — Legacy 1BP Catalog Audit (2026-07-31, issue #1243)

**Gate:** `ppl_generic` (build/ppl_generic) + 48-sample gate set
(`research/ws00-baseline/samples/ppl-gate-48.jsonl`, Qwen3-vocab tokenized —
ppl numbers below are only meaningful for Qwen-vocab models; other families
need per-vocab sample sets). Header/structure scan via `build/onebp_loader`.

## Structural scan (all 59 local models)

**`token_embd.weight` is absent from 41 of 59 files.** Exact-name scan of the
full tensor list per file:

| Group | Files | Status |
|---|---|---|
| ✅ embd present (18) | CodeLlama-7B, Llama-2-7B/13B, Mistral-7B-v0.2/v0.3, Phi-3-mini, Phi-3.5-mini, Qwen2.5-0.5B, Qwen3-0.6B (TQ2), SmolLM2-135M/360M/1.7B, TinyLlama-1.1B, Mage-VL-4B, Llama-3.2-1B (H=1152 header, embd found) | loadable |
| ❌ embd missing (41) | DeepSeek-Coder-V2-Lite, DeepSeek-R1-0528-Qwen3-8B, DeepSeek-R1-Distill-{Llama-8B,Qwen-7B,Qwen-14B,Qwen-32B}, Dolphin3.0-Llama3.1-8B, Falcon3-{1B,3B,7B,10B}, Gemma-2-2B, Gemma-3-{1B,4B,12B}, Gemma-4-{26B-A4B,31B}, Granite-3.2-8B, KAT-Coder-V2.5, Laguna-S-2.1, Ling-2.6-Flash, Llama-3.1-8B, Llama-3.2-3B, Ministral-8B, Mistral-Small-3.1-24B, Nemotron-3-Super-120B-A12B, North-Mini-Code-1.0, OLMo-2-{1124-7B,0325-32B}, Ornith-35B, Phi-4-mini, Qwen2.5-{3B,7B}, Qwen2.5-VL-{3B,7B}, Qwen2-VL-7B, Qwen3-{4B,8B}, Qwen3.5-{4B,9B}, Qwen3.6-{27B,35B-A3B}, Qwen3-Coder-30B-A3B, Qwen3-VL-{2B,4B} | **cannot load — missing token_embd** |

**Root cause:** the C++ converter's pre-07-31 tensor cap (200M elements) silently
dropped embeddings above it (fixed 2026-07-31, commit 5f407bea2 — but no files
were re-converted). Some files below the cap are also affected (e.g. Falcon3-10B,
173M elements), so a second mechanism (naming/ordering in older converter
generations) is suspected — per-file converter provenance is not recorded.

## ppl measurements (Qwen-vocab sample set, 48 samples / 708 tokens)

| Model | quant | PPL | Verdict |
|---|---|---|---|
| **Qwen3-0.6B-q8-q4nx.1bp** (C++ converter, Q8_0) | Q4NX | **16.34** | ✅ reference (matches fresh conversion 16.3376) |
| Qwen3-0.6B.1bp (python, Q4_K_M) | TQ2 | **5.9e8** | ❌ destroyed (known verdict; 3.7e8 on 5027-token set) |
| Qwen2.5-0.5B-Instruct.1bp (python, Q4_K_M) | Q4NX | **2.0e7** | ❌ destroyed |
| Qwen3-4B.1bp (C++ converter pre-cap-fix) | Q4NX | n/a (init fails) | ❌ token_embd dropped |

## Qwen2.5-0.5B additional defect

Header: `H=896 NH=14 NKV=2 HD=64` — q_proj rows = NH×HD = 896 = H, but real
Qwen2.5-0.5B has head_dim **128** (q_proj 1792). The python converter inferred
`head_dim = hidden/n_heads` instead of reading config. Even if the values were
healthy, attention geometry is wrong.

## Conclusions & actions

1. **The legacy Q4_K_M-python-converted Qwen-family files are garbage** (2 of 2
   tested destroyed; 0/2 usable). The FINDINGS3 suspicion of the python Q4_K_M
   path is now empirically confirmed for value AND structure (head_dim).
2. **41/59 files cannot load at all** (missing embeddings). The HF-hosted
   catalog entries for these are broken artifacts.
3. **Action (open, big):** re-convert the affected catalog with the current C++
   converter (Q8_0/bf16 sources, post-5f407bea2) and re-run the ppl gate per
   family. This is a batch job (tens of GB of GGUF downloads); the gate now
   exists to validate every output.
4. Non-Qwen families need per-vocab gate sample sets (tokenize with each
   model's own GGUF vocab) before ppl claims can be made for them.
