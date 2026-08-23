# 1BP Catalog Re-conversion Report (issue #1243)
Run: 2026-08-02T13:57Z — converter @ 77ab04bb9
Gate: Qwen-vocab → ppl_generic /home/bcloud/projects/1bit-monster/research/ws00-baseline/samples/ppl-gate-48.jsonl; others → structural only

| Model | Source GGUF | PPL (Qwen gate) | verify_1bp | token_embd | Status |
|---|---|---|---|---|---|
| Falcon3-1B-Instruct | Falcon3-1B-Instruct-Q8_0.gguf | - | PASS | YES | OK |
| Qwen3-8B | Qwen3-8B-Q8_0.gguf | 7.5845 | PASS | YES | OK |
| Qwen3-4B | Qwen3-4B-Q8_0.gguf | 8.7478 | PASS | YES | OK |
| Llama-3.1-8B-Instruct | Meta-Llama-3.1-8B-Instruct-Q8_0.gguf | - | PASS | YES | OK |
