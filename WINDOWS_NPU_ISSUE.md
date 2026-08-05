# [Windows NPU] Complete XDNA2 submission shim with reverse-engineered driver constants

## Summary

The XDNA 2 NPU path (committed in `b1b587523`, #1468) is **Linux-only**. The entire submission-to-NPU flow is built on the Linux **DRM `amdxdna` ioctl ABI** (`npu-infer/include/npu_utils/amdxdna_accel.h`), which does not exist on Windows. There is no `_WIN32` path, no Windows NPU shim, and no reverse-engineered Windows driver constants anywhere in `npu-infer/` (grep for `_WIN32`/`win` yields nothing).

To run 1bit.systems NPU inference on Windows Strix Halo machines we need a Windows submission shim that talks to AMD's Windows XDNA 2 driver stack instead of the Linux DRM ioctls — and the command/control constants for that stack are **not documented publicly**; they must be reverse-engineered from the shipped Windows drivers / libraries.

## Scope

### 1. Windows submission shim (`npu-infer/`)

Port of the (currently blank) submission path so the existing model pipeline can run unchanged on Windows:

- `amdxdna_accel.h` → a Windows equivalent of the DRM ioctl ABI (device open/close, hardware context create/destroy, BO create/info/sync, fence create, command-submission ring).
- Reuse the existing instruction senders (`npu_cmd_*.hpp`, DMA channel + BD slot management) as-is — only the lower "submit to driver" layer is platform-specific.
- Guard with an abstraction header so the rest of the engine stays ABI-agnostic.

### 2. Reverse-engineered Windows driver constants (`needs-re-constants`)

The Linux ioctl numbers (`DRM_AMDXDNA_*`, `enum amdxdna_drm_ioctl_id`) and the `struct amdxdna_drm_*` layouts are Linux-specific. On Windows AMD exposes the NPU through a different driver interface. To reach parity we must:

- Identify the Windows driver interface surface (DXCore device + the AMD NPU user-mode/KMD protocol / memory-mapped command ring).
- Reverse the magic numbers, command opcodes, and struct layouts needed to submit a column/block to an XDNA 2 tile on Windows.
- Document each derived constant with the source artifact it was recovered from (mirroring the Linux reverse-engineering notes in `docs/research/npu/AMD-XDNA-COLUMN-UNLOCK-KNOWLEDGE-DUMP.md`).

### 3. Validation

- Build the shim under MSVC / MinGW (watch for `__linux__`/`__KERNEL__` assumptions — the header has a `#ifdef __KERNEL__` branch already).
- Smoke-test BO creation + a minimal GEMM submit on Strix Halo Windows.
- If no Windows hardware is available at build time, keep the shim **compile-time gated** and unit-test the constant/handshake layer with a mock driver.

## Files touched

| File | Change |
|------|--------|
| `npu-infer/include/npu_utils/amdxdna_accel.h` | gate Linux DRM includes behind non-Windows; add Windows submission header |
| `npu-infer/include/npu_utils/` (new) | Windows NPU driver abstraction + recovered constants header |
| `npu-infer/src/` | submission-layer split (Linux ioctl vs Windows) |

## Blockers / open questions

- Likely no Windows XDNA 2 dev box in-repo for end-to-end validation — need hardware or a mock.
- Driver interface surface is proprietary/unstable; constants may need re-derivation per driver release.
- Is there a need for a signed user-mode driver interop (DXCore gains access)? Confirm target Windows build (10/11) and driver baseline.

## Definition of done

- [ ] Windows shim compiles (MSVC + MinGW) with zero `__linux__` leaks in the shared pipeline.
- [ ] Windows driver constants reverse-engineered and documented (`needs-re-constants` label = still in progress).
- [ ] BO create + minimal GEMM submission verified (hardware or validated mock).
- [ ] No regression on the existing Linux DRM path.
