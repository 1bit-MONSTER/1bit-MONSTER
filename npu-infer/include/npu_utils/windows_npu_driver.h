// SPDX-License-Identifier: MIT
/*
 * windows_npu_driver.h — Windows XDNA2 NPU driver abstraction for npu-infer.
 *
 * Issue #1504: the Linux submission path (amdxdna_accel.h) is built on the
 * DRM ioctl ABI which does not exist on Windows. This header provides the
 * Windows-side driver surface, with the constants below reverse-engineered
 * from AMD's shipped Windows driver package (issue #1504 "needs-re-constants").
 *
 * ============================================================================
 * REVERSE-ENGINEERING NOTES — source artifacts (all recovered 2026-08-11)
 * ============================================================================
 * Package: NPU_RAI_376_WHQL.zip (26,936,876 B) from
 *   https://download.amd.com/opendownload/RyzenAI/Driver/NPU_RAI_376_WHQL.zip
 *   (linked from https://ryzenai.docs.amd.com/en/latest/inst.html, "Windows
 *   NPU driver"; the RAI_280_WHQL sibling carries the same interface).
 *
 * Driver binaries analyzed:
 *   npu_mcdm_stack_prod/ipustack.sys    500,840 B  KMD (kernel driver)
 *   npu_mcdm_stack_prod/xrt_core.dll    592,232 B  XRT device-query UMD
 *   npu_mcdm_stack_prod/RadeonML_IPU.dll            raw control-plane UMD
 *   npu_mcdm_stack_prod/vitis-ai-runtime2.dll       raw control-plane UMD
 *   npu_mcdm_stack_prod/kipudrv.inf                driver package INF
 *
 * Findings (each constant below cites its artifact):
 *
 * 1. DRIVER IDENTITY (kipudrv.inf)
 *    Service:   IpuMcdmDriver  ->  ServiceBinary %13%\ipustack.sys
 *    DriverVer: 04/04/2026, 32.00.20101.3760
 *    Class:     ComputeAccelerator, ClassGuid {F01A9D53-3FF6-48D2-9F97-
 *               C8A7004BE10C}  (use with SetupDiGetClassDevs to enumerate
 *               the device instead of a hardcoded path).
 *    The sys also names a second service \Registry\Machine\System\Current
 *    ControlSet\Services\IpuWdfDriver and the kernel-mode control device
 *    \Device\KipuDrvControlDevice (wide string in ipustack.sys).
 *
 * 2. CONTROL DEVICE (ipustack.sys wide strings)
 *    Kernel device:  \Device\KipuDrvControlDevice
 *    User-mode path: \\?\GLOBALROOT\Device\KipuDrvControlDevice
 *    (CreateFileW GENERIC_READ|WRITE; both RadeonML_IPU.dll and
 *    vitis-ai-runtime2.dll build their handle this way — they carry the
 *    \\?\GLOBALROOT prefix literal and import CreateFileW next to
 *    DeviceIoControl.)
 *
 * 3. IOCTL CODES (DeviceIoControl call sites, disassembly)
 *    Family: CTL_CODE(DeviceType=0x9, fn, METHOD_BUFFERED, FILE_ANY_ACCESS)
 *    - kIoctlParam = 0x000900A4  (fn 41)
 *      Recovered at RadeonML_IPU.dll +0x23a9d7: edx=0x900a4, in-buffer r8,
 *      nInBufferSize r9d = *(u16*)(in+4) + 8  (small param struct:
 *      [0:u16 size][4:u16 param id]..., total len = size+8). 322 code
 *      occurrences in ipustack.sys.
 *    - kIoctlQueryMonitor = 0x000900A8  (fn 42)
 *      Recovered at RadeonML_IPU.dll +0x23a3c7 and vitis-ai-runtime2.dll
 *      +0x4b6666/+0x4b7ad7/+0x4b9df9/+0x4d805b: edx=0x900a8, no in-buffer
 *      (r8=0, r9=0), out-buffer with nOutBufferSize=0x4000 (16 KiB). The
 *      KMD string NPUMonitorControlOption sits in the same driver — this is
 *      the NPU monitor/status query. 525 code occurrences in ipustack.sys.
 *
 * 4. SUBMISSION PLANE — MCDM, not raw ioctls
 *    The package is npu_mcdm_stack_prod: command submission goes through
 *    Microsoft's Compute Driver Model (DXCore -> D3D12 -> dxgkrnl ->
 *    ipustack.sys). RadeonML_DirectML.dll imports dxgi/d3d12 (5 symbols);
 *    the raw control device above carries only management/monitor ioctls.
 *    There is NO Windows equivalent of DRM_AMDXDNA_EXEC_CMD as a raw ioctl:
 *    the equivalent of the Linux command ring is the D3D12 MCDM queue
 *    (ID3D12CommandQueue on the DXCore NPU device). The D3D12 API surface
 *    is documented by Microsoft; the device-specific fence/ring details
 *    still need validation on a live Windows Strix Halo box before the
 *    submission shim can be completed (see TODO at the bottom).
 *
 * ============================================================================
 * USAGE
 * ============================================================================
 * Compile-time gated behind _WIN32; including this header on non-Windows
 * builds is a no-op (used by the mock test in tests/).
 */
#ifndef NPU_INFER_WINDOWS_NPU_DRIVER_H_
#define NPU_INFER_WINDOWS_NPU_DRIVER_H_

#include <cstdint>

// Windows API headers MUST be included outside any namespace: winnt.h cascades
// into the C++ stdlib headers (via x86intrin.h -> mm_malloc.h -> <stdlib.h>),
// which break when parsed inside a user namespace (::abs/::div_t not declared).
#if defined(_WIN32) || defined(__WINDOWS__)
#include <windows.h>
#include <winioctl.h>

#include <cstring>
#include <string>
#include <vector>
#endif

namespace windows_npu {

// -- 2. Control device ------------------------------------------------------
// (wide string; content verified by tests/test_windows_npu_driver.cpp)
inline const wchar_t* kIpuControlDevicePath =
    L"\\\\?\\GLOBALROOT\\Device\\KipuDrvControlDevice";

// -- 3. IOCTL family --------------------------------------------------------
// CTL_CODE(0x9, fn, METHOD_BUFFERED /*0*/, FILE_ANY_ACCESS /*0*/) = 0x90000 + fn*4
inline constexpr std::uint32_t kIoctlParam = 0x000900A4u;        // fn 41
inline constexpr std::uint32_t kIoctlQueryMonitor = 0x000900A8u; // fn 42
inline constexpr std::uint32_t kMonitorBufferBytes = 0x4000u;    // 16 KiB

// Windows CTL_CODE macro (winioctl.h), kept here so the recovered constants
// can be re-verified on any host by the mock test.
inline constexpr std::uint32_t ctl_code(std::uint32_t dev_type,
                                        std::uint32_t function,
                                        std::uint32_t method,
                                        std::uint32_t access) {
  return (dev_type << 16) | (access << 14) | (function << 2) | method;
}

#if defined(_WIN32) || defined(__WINDOWS__)

// -- Control-plane driver wrapper (raw ioctl equivalent) --------------------
class Driver {
 public:
  Driver() = default;
  ~Driver() { close(); }
  Driver(const Driver&) = delete;
  Driver& operator=(const Driver&) = delete;

  // Opens \Device\KipuDrvControlDevice. Returns false on failure (GetLastError
  // holds the NTSTATUS-encoded error).
  bool open() {
    if (handle_ != INVALID_HANDLE_VALUE) return true;
    handle_ = ::CreateFileW(kIpuControlDevicePath, GENERIC_READ | GENERIC_WRITE,
                            0, nullptr, OPEN_EXISTING, 0, nullptr);
    return handle_ != INVALID_HANDLE_VALUE;
  }

  void close() {
    if (handle_ != INVALID_HANDLE_VALUE) {
      ::CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
    }
  }

  HANDLE handle() const { return handle_; }

  // Raw DeviceIoControl passthrough; callers own the buffers.
  bool ioctl(std::uint32_t code, void* in, std::uint32_t in_bytes,
             void* out, std::uint32_t out_bytes, std::uint32_t* returned) {
    DWORD ret = 0;
    BOOL ok = ::DeviceIoControl(handle_, code, in, in_bytes, out, out_bytes,
                                &ret, nullptr);
    if (returned) *returned = ret;
    return ok != FALSE;
  }

  // kIoctlParam: in-buffer layout [0:u16 param_id?][4:u16 size] (len = size+8).
  bool set_param(std::uint16_t param_id, const void* payload,
                 std::uint16_t payload_bytes) {
    if (!open()) return false;
    std::uint32_t in_bytes = 4 + payload_bytes;
    std::vector<std::uint8_t> buf(in_bytes);
    buf[0] = static_cast<std::uint8_t>(param_id & 0xff);
    buf[1] = static_cast<std::uint8_t>(param_id >> 8);
    buf[2] = static_cast<std::uint8_t>(payload_bytes & 0xff);
    buf[3] = static_cast<std::uint8_t>(payload_bytes >> 8);
    if (payload_bytes) std::memcpy(buf.data() + 4, payload, payload_bytes);
    return ioctl(kIoctlParam, buf.data(), in_bytes, nullptr, 0, nullptr);
  }

  // kIoctlQueryMonitor: no in-buffer, 16 KiB out-buffer.
  bool query_monitor(void* out, std::uint32_t out_bytes,
                     std::uint32_t* returned) {
    if (!open()) return false;
    return ioctl(kIoctlQueryMonitor, nullptr, 0, out, out_bytes, returned);
  }

 private:
  HANDLE handle_ = INVALID_HANDLE_VALUE;
};

// -- 4. Submission plane (MCDM) ---------------------------------------------
// TODO(hardware): the command-submission shim maps onto the DXCore MCDM
// device (ID3D12Device on the NPU adapter, exposed via the ComputeAccelerator
// class GUID {F01A9D53-3FF6-48D2-9F97-C8A7004BE10C}) with ID3D12CommandQueue
// as the command ring equivalent. The D3D12 calls are documented; the
// XDNA2-specific fence/ring setup must be validated on a live Windows Strix
// Halo box (minisforum 192.168.50.61) before the EXEC_CMD equivalent can be
// implemented here.

#endif  // _WIN32 || __WINDOWS__

}  // namespace windows_npu

#endif  // NPU_INFER_WINDOWS_NPU_DRIVER_H_
