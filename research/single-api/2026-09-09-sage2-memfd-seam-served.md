# goal mttyyykw (sage-2) — zero-copy memfd seam in the served path: VERIFIED (2026-09-09)

The served single-API path (sage-1 hook) exercises the zero-copy seam in-service:
HRX0 prefill exports the session to a memfd, the Vulkan0 decode leg mmap-imports it
(no file round-trip, no host DMA copy — load_session_mem mmaps the fd MAP_SHARED and
llama_state_set_data consumes the raw session-v9 region).

## Evidence (strixhalo, dual-engine bundle build-opensplit, Qwen3-Coder-30B-A3B Q4_K_M)

1. 30B routed one-call (router_serve_main = the same factory the server hook calls):
   exit 0; HRX0 prefill (5 tok) -> session exported to memfd: 5 tokens, 492,760 bytes
   state -> Vulkan0 import (pos=4, resume=374) -> decode 32 tok.
2. TOKEN-IDENTITY GATE: routed decode first-8 tokens =
   12095 13 576 6722 315 32961 374 37169
   fork-Vulkan0 DIRECT decode (GGML_HRX_DISABLE=1, zgreedy_de, same file+prompt) =
   12095 13 576 6722 315 32961 374 37169
   IDENTICAL - the memfd-seamed served decode is token-for-token the single-engine
   Vulkan reference (extends the mtsy05dx 20/20 memfd_out3 F==M proof to the served
   routed path at 30B contract scale).
3. Seam cost: earlier zero-copy evidence measured ~19 ms state exchange (15349b7e8);
   this run's end-to-end one-call (incl. two engine loads) 32 tok / 0.974 s after
   warm loads - steady decode leg ~80-85 t/s per the mtsy05dx contract runs.
4. No file/blob round-trip in the path (code-verified: HRX_STATE_MEMFD/export_session_mem/
   load_session_mem, mmap MAP_SHARED).

Gate: oracle-identical vs reference + NaN-free + coherent continuation - PASS.
