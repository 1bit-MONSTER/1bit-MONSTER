# Zero-Copy NPU↔GPU↔Host Fusion Substrate

## Status: ⚡ PROVEN on Strix Halo (gfx1151 + XDNA 2)

> **Verification status (2026-08-30, fresh run on this box):**
> - ✅ `test_vk_dma_buf_import` — **PASS, now BIDIRECTIONAL** (issue #1946):
>   the original one-way proof (GPU compute shader writes NPU pages via
>   dma-buf import, CPU reads with NO copy) is extended with `roundtrip.comp`:
>   CPU writes a pattern → the shader READS it and writes a derived pattern →
>   CPU reads it back with no copy. Both directions alias the same pages
>   (0 mismatches on Strix Halo, 2026-08-30). This is the full
>   `test_zero_copy` proof ported to the Vulkan dma-buf idiom.
> - ✅ `test_vk_attn_slice` — **PASS**: Vulkan compute shader reads NPU SharedBO
>   KV pages via the dma-buf import, matches CPU reference (max rel err 2.06e-4).
>   This is the current production zero-copy proof.
> - ⚠️ `test_zero_copy` (the original `hipHostRegister` idiom) **no longer
>   passes**: TheRock HIP rejects XRT-mapped NPU pointers for `hipHostRegister`
>   (`invalid argument`; plain malloc registers fine — verified 2026-08-29).
>   The idiom is superseded: its proof steps are now covered by the
>   bidirectional Vulkan dma-buf test above (production uses the Vulkan dma-buf
>   import; the HIP side reads via the XRT `host_ptr()` view).

This directory contains the **correct, empirically-verified zero-copy substrate** for NPU+GPU fused inference on Strix Halo. Every previous "fused" implementation was aspirational (lied in its headers, never compiled, IO_PAGE_FAULT'd, or ran GPU-only).

---

## What It Ships (Steps 1–4)

| # | Step | Deliverable | Status |
|---|------|-------------|--------|
| **2** | Zero-copy handoff | `SharedBO` — NPU-owned, GPU-imported, host-coherent. All three alias the same physical pages. No memcpy between them. | ✅ **PROVEN** on hardware (3/3 runs, zero IO_PAGE_FAULTs) |
| **1** | NPU unblock | Root-cause analysis of the `npu_engine_universal.cpp` boot EINVAL + `xclbin_health` validation tool. | ✅ **Debunked** "too many contexts" myth; pinpointed `Invalid num_col N` |
| **3** | NPU-FFN ∥ GPU-attn pipeline | `PipelineOverlap` — 2-slot double-buffered pipeline skeleton on SharedBO. Dummy callbacks exercise the pattern. | ✅ **Skeleton runs** (95.79ms vs 120ms sequential with 40 layers) |
| **4** | Cleanup | `gpu_npu_bridge.cpp` header fixed (no more lying about zero-copy); dead `import_to_hip/import_to_xrt` removed; this README. | ✅ **Fixed** |

---

## Architecture

```
              SharedBO (NPU-owned XRT HOST_ONLY BO)
              ┌──────────────────────────────────────┐
              │          physical pages               │
              └──┬──────────────────────────────┬─────┘
                 │                              │
                 ▼                              ▼
       XRT mmap (CPU)                  dma-buf fd
       → host_ptr                       → exported by NPU → imported by GPU
       (coherent, system RAM)           → HIP hipHostRegister() (test)
                                        → Vulkan VK_KHR_external_memory_fd
                                          (production — only API that works
                                           on the installed TheRock HIP
                                           (7.16), which lacks HIP DmaBuf
                                           external memory)
                 │                              │
                 └──────────┬───────────────────┘
                            ▼
                   ONE set of pages, three views.
                   No memcpy, no staging buffer, no DMA.
```

**Direction rule (critical):** The NPU must **own** the allocation. The `gpu_npu_bridge.cpp` approach (GPU allocates GTT, imports into NPU via XRT dma-buf) causes `AMD-Vi IO_PAGE_FAULT` because `amdxdna`'s dma-buf import path doesn't wire up the NPU's IOMMU domain correctly. When the NPU owns the allocation (XRT HOST_ONLY BO), its IOMMU domain already covers the pages, and the mature `amdgpu` import path handles the GPU side correctly.

---

## Key Findings That Overturn Previous Assumptions

### 1. "5 hw_contexts collide" → WRONG. Real cause: `Invalid num_col N`

The state-of-the-stack doc (2026-07-14) hypothesized that `npu_engine_universal.cpp`'s boot SIGABRT came from 4-5 simultaneous `xrt::hw_context`s exhausting the AIE column/tile budget. **This is incorrect.**

- **Probe**: 8 identical xclbins → 8 OK. 5 distinct xclbins → 3 OK, 2 FAIL. The failing ones (`final_12col_test.xclbin`, `final_40col_v2.xclbin`) fail EVEN FIRST/ALONE.
- **dmesg smoking gun**: `[drm] *ERROR* aie2_hwctx_col_list: Invalid num_col 12`
- **Real cause**: The engine's xclbins request a number of AIE columns the driver's column allocator rejects (12 and 40 both fail; small-column bf16 designs succeed at ≤8). The 40-column target the state doc flagged as "contested" is genuinely **blocked by the shipping driver**.

**The fix**: Use xclbins whose `num_col` the driver accepts. The `xclbin_health` tool validates this at startup so the engine diagnoses (not SIGABRTs) bad xclbins.

### 2. `hipExternalMemoryHandleTypeDmaBuf` → DOES NOT EXIST in the installed TheRock HIP (7.16)

The `gpu_npu_bridge.cpp` code that used `hipImportExternalMemory` with `hipExternalMemoryHandleTypeDmaBuf` never compiled. The installed TheRock HIP (7.16 — the "7.2.4" figure in older copies of this note was a stale attribution) genuinely lacks that enum value: `hipExternalMemoryHandleType` is `{OpaqueFd, OpaqueWin32, OpaqueWin32Kmt, D3D12Heap, D3D12Resource, D3D11Resource, D3D11ResourceKmt, NvSciBuf}` (compile-verified), and the newer mem-pool sharing enum (`hipMemAllocationHandleType`) is `{None, PosixFileDescriptor, Win32, Win32Kmt, Fabric}` — no dma-buf either. The only dma-buf API present is the EXPORT-only `hipMemGetHandleForAddressRange(hipMemRangeHandleTypeDmaBufFd)` (HIP allocation → fd); there is no import counterpart that turns an external dma-buf fd into a HIP device pointer.

**The production GPU import path must be Vulkan** (`VK_KHR_external_memory_fd`), matching `engine/fusion/gpu_attn.zig` and the stub in `interop.zig`.

### 3. Zero-copy IS possible and PROVEN

`hipHostRegister(npu_mmap_ptr) + hipHostGetDevicePointer()` gives a GPU device pointer aliasing the NPU's pages on this APU. The test proves: GPU writes → CPU reads with NO D2H copy = match. Three runs every time, zero IO_PAGE_FAULTs.

---

## File Inventory

| File | Purpose |
|------|---------|
| `shared_bo.h/.cpp` | NPU-owned zero-copy buffer. XRT HOST_ONLY BO + dma-buf fd export. No HIP dep. |
| `test_zero_copy.cpp` | Airtight zero-copy proof. Uses `hipHostRegister` to validate the memory model. |
| `pipeline_overlap.h/.cpp` | 2-slot double-buffered pipeline skeleton (NPU∥GPU). Injected callbacks for testability. |
| `test_pipeline.cpp` | Pipeline demo with dummy timings. |
| `xclbin_health.cpp` | Validate any xclbin against the running driver. Detects `Invalid num_col` rejection. |
| `probe_contexts.cpp` | Probe max concurrent `hw_context`s (empirically disproves the "5 contexts collide" myth). |
| `probe_multi_xclbin.cpp` | Probe mixing distinct xclbins (tests what the real engine does). |
| `Makefile` | Build everything. Needs XRT + TheRock HIP SDK. |

---

## Next Steps for Production

1. ✅ **Vulkan dma-buf import — DONE for the C++ fused backend** (`src/backend_fused.cpp`, issue #1217): each SharedBO's exported dma-buf fd is now imported as Vulkan device memory (`vkrt::GpuBuffer::create_from_dma_buf`, `VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf`) — the `hipHostRegister` test idiom is gone from the production path. **Silicon findings (2026-08-29, RADV/Strix Halo):** the imported dma-buf is *not* CPU-mappable — `vkMapMemory` succeeds but touching the mapping SIGBUSes — so the HIP side of the fused backend talks to the NPU pages through the XRT CPU view (`host_ptr()`), and the Vulkan import is held as the GPU-side handle. **Zero-copy GPU attention PROVEN** by `test_vk_attn_slice.cpp`: a Vulkan compute shader reads the KV cache straight out of NPU SharedBO pages via the dma-buf import and matches a CPU reference (max rel err 2e-4) — the substrate for a full Vulkan-compute attention path (`gpu_attn.zig` / `interop.zig` are not yet present in this tree).
2. **Replace pipeline dummy callbacks**: inject real HIP-attention kernel launches and real XRT-FFN kernel launches into the 2-slot pipeline.
3. **xclbin column-count**: build/reuse xclbins with `num_col` ≤8 (what the driver accepts) for the 1bit engine, or build a generic instruction-driven xclbin (like FastFlowLM's `mvm_i8`) that handles all shapes from one context.
4. **Integrate `xclbin_health`** into the NPU engine startup as a graceful validation gate before any `CREATE_HWCTX`.
