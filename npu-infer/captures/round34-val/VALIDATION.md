# Round-34 validation — replay-vs-runtime receipts (2026-09-03)

Fresh real-runtime capture (run_qwen3_npu harness, XRT path, amd-oss libs,
model Qwen3-0.6B-NPU2, token 1000, first forward) + on-device replays with
byte-verified inputs. Every number below was re-measured on this capture.

## Setup / commands

    # capture (xrt::bo::sync + runlist/ELF interposer)
    LD_PRELOAD=/tmp/cap_int.so CAP_DIR=/tmp/val ./run_qwen3_npu <model_dir> 1
    # per-arm norm capture (idx5/idx6 content at every set_arg)
    LD_PRELOAD=/tmp/norm_cap.so ./run_qwen3_npu <model_dir> 1

## Inputs verified byte-identical (this capture)

| Input | Check | Result |
|---|---|---|
| per-layer weight BOs (28 x 10 MB) | npu_pack_layer_bo vs bo_to_*_10485760 | **28/28 IDENTICAL** (0 diff bytes) |
| lm_head weight BO (94 MB) | bo_to_0141_98566144.bin fed straight | used as-is |
| per-layer norm buffers (i5/i6) | generated i5=[iln@0\|paln@2048], i6=[64x1.0@0\|64x0.0\|q_norm@256\|k_norm@512] vs arm-time capture (arms 30-85) | **112/112 IDENTICAL** |
| token-1000 embedding | captured pre-runlist act [0:2048] | fed as-is |
| per-ctx layer ELF | fresh elf_0001 md5 2e50c6ea... == captures/txn-elfs/elf_0001_layer.bin | IDENTICAL |
| kv | zero at load AND at layer-run arm time | zeros fed |

## Outputs measured

| Measurement | Result |
|---|---|
| replay determinism (two 28-layer runs) | bit-identical |
| 28-layer replay act vs runtime act | corr **0.997891**, std 194.46 vs 188.62, bit-diff 999/1024 |
| lm_head(runtime act) vs logits_1000.bin | corr **0.999968**, argmax **144370 == runtime** |
| lm_head(replay act) vs logits_1000.bin | corr 0.998033, argmax 397 (flips) |
| runtime act write geometry | diff vs pre = exactly [0:2047] (2048 B in place) |
| runtime logits write geometry | diff = [0:303871] (151936 bf16), == logits_1000.bin |

## Interpretation

1. The layer-chain replay is input-complete (everything the runtime feeds the
   layer kernels is byte-verified identical) yet lands at corr 0.9979 — the
   residual originates at layer 0 itself (single-layer corr 0.999877,
   ~1% delta magnitude, uniform, deterministic) and compounds through the
   outlier-amplifying tail. Suspect: runtime load_weights-time device/AIE
   tile state not replicated by a fresh-hwctx replay.
2. lm_head is closed: with the runtime's own final act it reproduces the
   runtime's logits at corr 0.99997 with the same argmax.
3. RESOLVED: the engine/replay/replay-correlation "0.998 vs runtime" was
   RUNTIME DRIFT, not an engine error. The machine rebooted 2026-09-02
   23:19 (kernel 7.2.0-next-20260821; round-37 ran pre-reboot). The
   runtime's own Sep-1 artifacts (/home/bcloud/.cache/rtcap/: bo_from_0199
   logits argmax 397 @ 12.8125; actpost_002 acts std 194.46) are
   BYTE-IDENTICAL to this replay's outputs (argmax 397 @ 12.8125; act
   std 194.46). corr(Sep-1 runtime, today runtime) = 0.99803 (logits) and
   0.997891 (act) — exactly the replay-vs-today numbers. So the replay
   (and round-37's engine) were exact against the round-37-era runtime
   (corr 1.0); the runtime drifted post-reboot (logits now argmax 144370
   @ 13.25, act std 188.62). The layer-0 ~1% delta was this drift at
   layer granularity. Re-gate against the round-37-era kernel/state for
   byte-parity with today's runtime.

## Evidence kept

- capture_manifest.log (this capture)
- emb.bin (token-1000 embedding, 2048 B)
- fnorm.bin (final model.norm, 2048 B @ BO[0])
- act_post.bin (runtime post-forward act, 1 MB)
- replay28.bin (hand-rolled 1-pass 28-layer replay act, 1 MB)
- (heavy: 28x10MB weight BOs, 94MB lm_head, kv, 112 norm files — reproducible
  via the commands above; not persisted)
