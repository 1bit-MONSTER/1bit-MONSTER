# FastFlowLM NPU lane — results (/opt/fastflowlm/bin/flm)

Method: `flm serve <tag> -p <port>` → `POST /v1/chat/completions`
(`temperature 0`, greedy). Prompt "The capital of France is".

| Model (tag) | Result | Decode tok/s | Output |
|---|---|---|---|
| qwen3:0.6b | ✅ PASS | 73.9 | "The capital of France is **Paris**." |
| qwen3:1.7b | ✅ PASS (thinking) | 43.3 | "<think> Okay, the user is asking about the capital of France…" (coherent) |
| qwen3:4b | ✅ PASS (thinking) | 19.6 | "<think> Okay, the user is asking…" (coherent) |
| llama3.2:1b | ✅ PASS | 59.9 | "The capital of France is Paris." |
| qwen3.6-moe:35b-a3b | ⚠️ runs, degenerate output | 16.4 | "////////////////////////" |

## Notes
- FLM validates NPU stack `ready:true` (/dev/accel/accel0, 8 cols, fw 1.1.2.65).
- 1.7B/4B are base (non-Instruct) Qwen3 with thinking enabled — output is coherent
  reasoning text, truncated at max_tokens=16 before the answer.
- 35B-A3B MoE runs (loads + decodes 16.4 tok/s) but greedy output for this prompt
  is the classic degenerate "////" repetition; a second serve attempt died during
  model load (server-side crash). Recorded as "runs, not verified correct".
