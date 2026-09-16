# Installation

> This page exists because issue #1 linked to it before it was written. For the
> full, maintained instructions see [`docs/guides/getting-started.md`](../guides/getting-started.md)
> (running the server) and [`docs/guides/building.md`](../guides/building.md) (building from
> source). This page is the short version plus the one thing that will actually
> bite you.

## ⚠️ Kernel first (issue #1)

On **Strix Halo (gfx1151)**, Linux **6.19.x** kernels have a reproducible
`amdgpu` OPTC CRTC hang under sustained NPU/GPU load. Use a confirmed-stable
kernel before anything else:

- ✅ **6.18.22-lts** (confirmed stable)
- ✅ **7.x** series (confirmed stable)
- ❌ **6.19.x** (hangs mid-inference)

```bash
uname -r          # check your running kernel
```

`install.sh` warns automatically if it detects 6.19.x.

## Requirements

| Component | Requirement                                    |
|-----------|------------------------------------------------|
| Hardware  | AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151)    |
| OS        | Ubuntu 24.04 LTS or later (Arch/CachyOS work)  |
| Kernel    | 6.18.22-lts or 7.x (not 6.19.x)                |
| ROCm      | TheRock 7.15.0a (pip)                            |

## NPU runtime (the engine's own worker)

NPU inference runs on the engine's own FLM-free C++ engine: `src/backend_npu.cpp`
fork/execs **`npu_engine_universal`** and drives the pre-compiled xclbins (GEMM,
attention), with CPU fallback for RoPE/norm/residual. It does **not** need
FastFlowLM.

| Item | Default | Override |
|------|---------|----------|
| `npu_engine_universal` worker | `./npu_engine_universal` (cwd-relative) | `NPU_ENGINE_BIN` |

```bash
cmake --build build --target npu_engine_universal   # needs XRT
export NPU_ENGINE_BIN=$PWD/build/npu_engine_universal
```

If the worker is missing, the NPU lane stays off and the engine runs on CPU/GPU.
FastFlowLM remains available as an *optional* second lane (`npu_flm`, via
`NPU_FLM_BIN` / `NPU_FLM_CONFIG` / `NPU_FLM_XCLBINS`, Q4NX-only) — a convenience
for that lane, never a requirement for the NPU. See
[NPU architecture](npu-architecture.md) and
[Vendored: ROCm/FastFlowLM](../vendored-fastflowlm.md).

## Quick install

```bash
git clone https://github.com/1bit-MONSTER/1bit-MONSTER
cd 1bit-monster
./install.sh          # warns on a 6.19.x kernel; use --skip-rocm to reuse prebuilt libs
```

## ⚠️ NPU 40-column unlock

On Strix Halo the NPU defaults to **8 columns** (32 tiles). To unlock the
full **40 columns** (160 tiles) without disabling Secure Boot, pass
`amdxdna.aie2_max_col=40` via the kernel command line in GRUB.

See the [full boot configuration guide](boot-configuration.md) for
step-by-step instructions.

## Secure Boot

**You do not need to disable Secure Boot**, and the docs recommend against it —
disabling it weakens the boot chain for no benefit.

- The 40-column unlock is passed on the **kernel command line** (GRUB), which is
  covered by the signed shim+GRUB chain and applied at module init — so it works
  with Secure Boot **on**.
- On a stock Ubuntu kernel the `amdxdna` module is already signed by Ubuntu's
  key; nothing to do. Only a **custom/patched** `amdxdna.ko` needs signing with
  an enrolled MOK key (`mokutil --import` + `kmodsign`).
- GPU/HIP-only use needs no Secure Boot change at all (userspace ROCm).
- The one exception is **AIE simulator debugging**, which does require Secure
  Boot disabled / lockdown `none` ([aiesim debugging](../aiesim-debugging.md)).

Check your state with `mokutil --sb-state`; details in
[Boot Configuration](boot-configuration.md#what-not-to-do).

## See also

- [Getting Started](../guides/getting-started.md)
- [Building from source](../guides/building.md)
- [Boot Configuration](boot-configuration.md) — 40-column NPU unlock with Secure Boot on
- [NPU architecture](npu-architecture.md) — the native (`npu_xrt`) NPU lane
- [Vendored: ROCm/FastFlowLM](../vendored-fastflowlm.md) — the optional FLM lane
- [Network Topology](Network-Topology.md)
