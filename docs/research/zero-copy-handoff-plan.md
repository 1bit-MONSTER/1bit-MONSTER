# zero-copy-handoff — scoped technical plan (goal mtsy05dx task 3) — 2026-09-08

Contract: KV/state handoff between the XDNA NPU engine and the gfx1151 HRX
engine with NO host DMA copy: same-family state in shared physical memory
(dma-buf or equivalent); no file/blob round trip; measured handoff cost <<
compute phase.

## Proven substrate (engine/fusion/zero_copy README, 2026-08-30)

- NPU must OWN the allocation: XRT HOST_ONLY BO + dma-buf fd export
  (SharedBO). NPU-owned pages are covered by amdxdna IOMMU; GPU import via
  amdgpu dma-buf is mature.
- Vulkan import PROVEN bidirectional: VK_KHR_external_memory_fd +
  VK_EXT_external_memory_dma_buf; compute shader reads NPU KV SharedBO pages,
  CPU aliases via XRT host_ptr(), 0 mismatches, max rel err 2.06e-4
  (test_vk_attn_slice, test_vk_dma_buf_import).
- HIP dma-buf import DOES NOT EXIST on TheRock HIP 7.16 (no
  hipExternalMemoryHandleTypeDmaBuf; hipMemGetHandleForAddressRange is
  export-only). => HRX (HIP-bound libhrx) cannot import the fd directly.

## Key fact established this checkpoint

The fork dual-backend process (goal/hrx-collapse, build-opensplit) co-loads
HRX + Vulkan on the SAME gfx1151 UMA device (llama-bench "backend HRX,Vulkan"
column; both ggml_hrx init OK and ggml_vulkan Found in one process). On UMA
the Vulkan-imported NPU pages and HRX device memory alias the same physical
DRAM.

## Candidate routes (to probe in order)

A. Vulkan-import bridge: NPU SharedBO fd -> Vulkan import (proven) -> hand
   the underlying physical pages to HRX kernels. Open question: can libhrx
   (IREE HAL + HIP binding on TheRock) consume a Vulkan VkDeviceMemory /
   host_ptr view as a device buffer? Probe: pass the Vulkan-imported buffer's
   host pointer (UMA => same VA/physical) into an HRX dispatch and check
   kernel reads. This is the SAME physical memory, so if HRX accepts a
   host-visible pointer (hsa_amd_memory_lock-style or HIP host ptr), no copy
   occurs.

B. Host-shared window: NPU SharedBO already exposes host_ptr() (CPU aliases
   NPU pages, proven). If HRX decode kernels can read host memory directly on
   UMA (no staging), the KV handoff is CPU-pointer handoff with no DMA copy.
   Probe: HRX dispatch reading from an NPU SharedBO host_ptr region.

C. (fallback) dma-buf into libhrx/IREE HAL if a newer ROCm HSA/IREE path
   supports import (not present in TheRock today; revisit if toolchain moves).

## Deliverables / gates

1. Probe A or B on strixhalo: NPU-owned KV buffer -> HRX decode reads it,
   token-identical vs CPU oracle, no host memcpy in the handoff.
2. Replace the llama_state 292MB blob file round-trip (D2 path) with the
   shared-memory handoff for the hybrid HIP/NPU-prefill -> HRX-decode flow.
3. Measure: handoff cost << compute phase (prefill+decode), record table.
4. Commit harness + numbers on the fork/engine line.

## Artifacts today

- ~/1bit-MONSTER/engine/fusion/zero_copy/ (SharedBO, vk dma-buf proofs)
- fork build-opensplit (HRX+Vulkan co-tenancy)
- D2 blob path documented in ~/1bit-MONSTER/docs/research/hybrid-prefill-decode.md
