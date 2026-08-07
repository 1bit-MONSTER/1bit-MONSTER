# One Heap, One API Call — Linux NPU Pivot Plan

Status: **DRAFT** (scout + plan)
Date: 2026-08-06
Author: scout/planner
Source tree mirrors: ryzen `~/onnx-extract/` (Windows RE corpus), `~/1bit-systems` (engine dev)

---

## 1. The Goal (restated)

> Single engine → single memory pool of system RAM → single API call to the NPU.
> Windows has it. Linux doesn't yet. Match Windows' memory + dispatch model.

Concretely, on **every inference step**:
- **One** memory pool (a single device heap carve-out, not N tensor BOs),
- **One** api call (one `EXEC_CMD` ioctl / one mailbox `CHAIN_EXEC_DPU` 0x13 message)
  carrying the whole command chain for the layer/model step.

---

## 2. What Windows Actually Does (verified from the RE corpus)

Source: `~/onnx-extract/WINDOWS-NPU-RE-NOTES.md` (sections 2, 12, 13) + `xclbin-dump/`.

### The one heap
- Windows MCDM kernel (ipustack.sys, IpuMcdmDriver) grants **ONE device heap per
  process** via the ComputeAccelerator KMD.
- Heap = the **firmware-defined 64 MB aperture at `0x4000000`** (DRAM window).
- All tensor/cmd allocations are **`xrt::bo` carved out of that single heap**
  (bo/bo_sram/bo_cmd shims) — no per-tensor kernel-level heaps.

### The one API call
- Runtime builds one command BO (`ERT_START_NPU` opcode 20) with `inst_buf` (the
  PDI `aie_image` microcode) + ≤34 shape args.
- Submit = **one mailbox message `CHAIN_EXEC_DPU` (0x13)** → NPU firmware walks
  the chain and runs the DPU. Measured `cmd-chain-latency ~10–12 us`.

### Firmware identity
- Windows `ipustack.sys` is an **MCDM front-end to the SAME NPU firmware** as the
  Linux `amdxdna` driver. Opcode tables match (`aie2_msg_priv.h`).
- Live-proof: same box (Ryzen AI Max+ 395, BDF 00c6:00.01.1) ran `xrt-smi validate`
  — 11/11 passed, **51.3 TOPS** live GEMM through the DPU.

### Kernel dispatch
- ONNX subgraph → fingerprint (`config_xclbin.txt`, 42/42) → kernel PDI blob
  (Windows: **46 embedded blobs**; Linux EP: loaded from disk) → DPU runner
  (`vadd` @ 0x80000, `AP_CTRL_CHAIN`).

---

## 3. Where Linux Currently Falls Short (verified gaps)

### Gap A — FLM allocates per-tensor BOs, not from one heap
`third_party/FastFlowLM/src/include/npu_utils/npu_utils.hpp` + `buffer.hpp`:
- `npu_app::create_bo_buffer(size)` → one `xrt::bo` / `buffer<T>` **per tensor**.
- The engine (`engine/npu/src/npu_engine_universal.cpp`) similarly does
  `make_unique<xrt::bo>` for **bA, bC, layerB[], bQ, bK, bV, bOut** — each a
  separate BO, each `sync()` + separate launch.
- **Not** a single device-heap carve. Many BOs → `SYNC_BO` per buffer → no "one
  heap / one api".

### Gap B — per-KPI synch + per-subgraph submit
- Code does `bA->sync(SYNC_TO_DEVICE)`, `bC->sync(FROM_DEVICE)`, `run.wait()` per
  GEMM — serialized, not a single chained dispatch.
- Windows does one command chain; Linux FLM issues one `ext::kernel(...)` submit
  per GEMM, per layer.

### Gap C — kernel-side support IS present but unused
Linux `amdxdna` driver (mirror: `~/onnx-extract/amdxdna-src/`) **already has**:
- `amdxdna_drm_create_dev_heap` → `client->dev_heap` (one BO per client), carved
  from `AIE2_DEVM_BASE 0x4000000` / `AIE2_DEVM_SIZE 64 MB` (npu1/4/5_regs.c). ✅
- `BO_DEV_HEAP` / `BO_DEVBO` (`amdxdna_gem.c`), EXEC_CMD chain path, mailbox
  `CHAIN_EXEC_DPU 0x13`. ✅

So the **driver contract already mirrors Windows**. The gap is **userspace**: FLM
goes through XRT with one BO per tensor instead of carving from `dev_heap` and
submitting one chain.

---

## 4. The Target Architecture (Linux, matches Windows)

```
engine memory ("single pool"):
  1 × amdxdna DEV_HEAP BO (64 MB @ 0x4000000, per process)
  └─ tensors/weights/cmd-slot are OFFSETS carved from that one BO
     (drm_mm allocator inside the pool, like the kernel already does)

dispatch ("single api call"):
  1 command BO: ERT_START_NPU(20) cmd_chain_slot_dpu { inst_buf=PDI, args≤34 }
  → 1 ioctl EXEC_CMD → 1 mailbox CHAIN_EXEC_DPU(0x13) → firmware runs chain
```

### Layers to touch
| Layer | Change | File(s) |
|-------|--------|---------|
| 1. Buffer | replace per-tensor `buffer<T>`/`xrt::bo` with a **pool allocator** over one `BO_DEV_HEAP` | `buffer.hpp`, `npu_utils.hpp` |
| 2. Submission | chain multiple DPU ops into ONE command BO / ONE EXEC_CMD | `npu_instr_utils.hpp`, runner |
| 3. Engine | `npu_engine_universal.cpp`: allocate all layer weights + kv from the pool; single submit per model step | `engine/npu/src/` |
| 4. Kernel sync | single `SYNC_BO` for the heap, not per-buffer | pool allocator |

---

## 5. Verification / Acceptance

- `dmesg` shows one `dev_heap` alloc; tensor offsets all within it.
- E2E: run a Qwen3-0.6B layer on the NPU; assert **1 `ioctl EXEC_CMD` per
  decoder-token step** (trace via `perf trace` / `strace -e ioctl`).
- cmd-chain latency target: match Windows (`~10–12 us`; FLM `26 us` currently).
- Existing `test_gemm_*` + `run-2026*.md` benches still pass.

---

## 6. Environment / Kernel Risk

- Current kernel here: **7.0.0-29-generic** (strix box). Driver device heap model
  is upstream, so no kernel swap needed on paper.
- **Fallback**: user flagged switch to **mainline kernel 7.1.5** if the `amdxdna`
  ioctl surface in 7.0.0 lacks `CREATE_DEV_HEAP` or chain-slot ergonomics. Check
  `/usr/include/drm/amdxdna_accel.h` for `DRM_AMDXDNA_CREATE_BO` + `EXEC_CMD` +
  `BO_DEV_HEAP` flags before committing to the pool design.
- Same firmware on both; `ipustack.sys` = MCDM, `amdxdna` = DRM accel. **No
  firmware/bitstream change required** — it's a userspace pooling + chaining job.

---

## 7. Risks / Open Questions

1. XRT's own `dev_heap` exposure: does xrt::bo already carve from dev_heap when
   `AMDXDNA_BO_DEV_HEAP` flag is used? (Or must we drop to raw ioctl for the pool?)
2. Max chain depth per EXEC_CMD (34 args; chain of DPU slots) — cap per layer.
3. KV-cache lives in the heap too → eviction/save-restore via preemption path.
4. `engine/npu` (MLIR-AIE xclbins, "MLIR_AIE" kernel) vs FLM path both need the
   pool — unify on one allocator interface.

---

## 8. Suggested Order (worker checklist)

1. Confirm kernels' `amdxdna_accel.h` supports `BO_DEV_HEAP` + chain EXEC_CMD.
2. Implement `npu_pool` (one DEV_HEAP BO + drm_mm-style offset allocator).
3. Rewire `buffer<T>` to be an offset-view into `npu_pool` (no separate BO).
4. Add `chain_submit()` building 1 cmd BO from N DPU slots → 1 EXEC_CMD.
5. Port `engine/npu` + FLM runner onto pool+chain.
6. Verify via strace (1 ioctl/token), dmesg (1 heap), benches (latency/tput).

### VERIFIED on strix, kernel 7.1.5 (2026-08-06)

`pool_probe` passes end-to-end: one 64MB DEV_HEAP, hwctx, 3 slices carved
inside the window, ONE EXEC_CMD with an ERT_CMD_CHAIN of 3 ERT_START_NPU
sub-commands → ONE mailbox CHAIN_EXEC_NPU (0x18) → firmware executes →
syncobj wait returns.

Requirements discovered (7.1.5, Strix XDNA2):
- **force_iova=1** module param is REQUIRED: with SVA (default), MAP_HOST_BUFFER
  receives the DEV_HEAP's invalid UVA (0xffffffffffffffff) → firmware
  INVALID_PARAM (0x4000003). Persisted in `/etc/modprobe.d/amdxdna.conf`.
- DEV_HEAP size must be the full 64MB window (multiples of 0x4000000).
- cmd header layout: op in GENMASK(27,23), **count in GENMASK(22,12)** = payload
  length in u32 words (incl. CU-mask words); CU mask word(s) precede the
  amdxdna_cmd_start_npu payload; sub-cmd count = 1 + 6 + n_args.
- Completion: WAIT_CMD ioctl is REMOVED in 7.1.5 → wait on the DRM syncobj
  returned by CREATE_HWCTX (syncobj_handle); timeout_nsec is an ABSOLUTE
  CLOCK_MONOTONIC deadline.
- SYNC_BO FROM_DEVICE is the debug-BO dump path — not part of the pool.
- A real PDI (e.g. fused_insts/layer_L1.bin) requires the engine's tensor args;
  with garbage args the DPU hangs. Probe uses a zeroed inst buffer.
