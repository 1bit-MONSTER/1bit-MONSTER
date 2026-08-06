// npu_pool.h — one device heap, one API call, on Linux amdxdna.
//
// Achievement of the pivot goal: a SINGLE device memory pool (one
// AMDXDNA_BO_DEV_HEAP carved into kva) with sub-BOs sliced from it by the
// kernel's own drm_mm — plus a batched EXEC_CMD (cmd_count>1) submit = the
// "one API call" per inference step. This bypasses XRT because xrt::bo has no
// DEV_HEAP flag; the driver contract (amdxdna_gem.c) is authoritative:
//
//   CREATE_BO type=DEV_HEAP  → one heap BO, drm_mm over 64MB @ AIE2_DEVM_BASE
//   CREATE_BO type=DEV       → sliced from client->dev_heap (userptr=kva+off)
//   EXEC_CMD cmd_count=1     → one ioctl; chain of N DPU sub-commands lives
//                              INSIDE one BO_CMD (ERT_CMD_CHAIN), matching the
//                              Windows mailbox model (driver rejects >1 here)
//
// Flags: DEV_HEAP size capped at dev_mem_size (64MB on npu1/4/5).
// ponytail: raw ioctl FD, safe only because every op is a single syscall under
//   one client; no per-BO locking needed when the pool owns the fd.
#ifndef NPU_POOL_H
#define NPU_POOL_H

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <drm/drm.h>
#include "amdxdna_accel.h"

namespace npu {

// Kernel-internal command structs (amdxdna_ctx.h) — not in the UAPI header.
// Opcode/format verified against amdxdna-src on the dev box.
enum ert_cmd_opcode {
    ERT_CMD_CHAIN = 19,
    ERT_START_NPU = 20,
};

struct amdxdna_cmd {
    uint32_t header;
    uint32_t data[];
};

struct amdxdna_cmd_chain {
    uint32_t command_count;
    uint32_t submit_index;
    uint32_t error_index;
    uint32_t reserved[3];
    uint64_t data[];   // N cmd BO handles
};

struct amdxdna_cmd_start_npu {
    uint64_t buffer;       // instruction buffer device address
    uint32_t buffer_size;
    uint32_t prop_count;
    uint32_t prop_args[];  // properties + regular kernel args
};

// One slice of the device heap (a kva window + xdna device addr).
struct span {
    uint8_t* host;      // CPU kva (heap base + offset)
    uint64_t dev_addr;  // device address (0x4000000 + offset)
    uint64_t size;
    uint32_t handle;    // DRM BO handle (for EXEC_CMD args / info)
};

// The single device heap for the process.
// ponytail: one heap per fd; multi-process/multi-device not needed yet —
//   generalize to a map keyed by <fd,dev> if pooling across NPUs matters.
class pool {
public:
    explicit pool(const char* dev = "/dev/accel/accel0", size_t heap_mb = 48,
                  uint32_t num_tiles = 32)   // 8 cols × 4 core rows (full NPU)
        : dev_(dev), num_tiles_(num_tiles)
    {
        fd_ = ::open(dev, O_RDWR | O_CLOEXEC);
        if (fd_ < 0)
            throw std::runtime_error(std::string("open ") + dev + ": " + strerror(errno));

        // ORDER MATTERS: the kernel requires DEV_HEAP to exist BEFORE hwctx
        // (aie2_hwctx_init fails -ENOENT if client->dev_heap is null).
        // 7.1.5 driver: dev heap size must be a multiple of the full
        // dev_mem window (0x4000000 = 64MB) — partial heaps are rejected.
        if (heap_mb * 1024 * 1024 % (64ull * 1024 * 1024) != 0 ||
            heap_mb * 1024 * 1024 > 64ull * 1024 * 1024)
            throw std::runtime_error("heap must be a full 64MB window");

        uint32_t h = 0;
        if (!create_bo(AMDXDNA_BO_DEV_HEAP, heap_mb * 1024 * 1024, h))
            throw std::runtime_error("DEV_HEAP create failed");
        heap_handle_ = h;
        heap_size_ = heap_mb * 1024 * 1024;

        // mmap the heap NOW: this sets the driver's heap->mem.userptr
        // (amdxdna_gem_obj_mmap), which aie2_hwctx_init REQUIRES when it
        // allocates its internal cmd_bufs from the heap and maps host buf.
        {
            bo_addr_t bi{};
            if (bo_info(heap_handle_, &bi) != 0)
                throw std::runtime_error("DEV_HEAP info failed");
            heap_base_ = static_cast<uint8_t*>(
                ::mmap(nullptr, heap_size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                       fd_, bi.map_offset));
            if (heap_base_ == MAP_FAILED)
                throw std::runtime_error(std::string("mmap heap: ") + strerror(errno));
            heap_dev_base_ = bi.xdna_addr;
        }

        // Create a hardware context (plain, no QoS/UMQ/log). qos_p must point
        // at a valid (zeroed) QoS struct — the kernel unconditionally copies
        // sizeof(amdxdna_qos_info)=24B from it. num_tiles must be >0 and a
        // multiple of core row count (4 on Strix) — aie2_hwctx_col_list
        // rejects 0.
        struct amdxdna_qos_info qos{};
        struct amdxdna_drm_create_hwctx c{};
        c.qos_p = reinterpret_cast<uint64_t>(&qos);
        c.max_opc = 0; c.num_tiles = num_tiles_; c.mem_size = 0;
        if (::ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &c) != 0)
            throw std::runtime_error(std::string("CREATE_HWCTX: ") + strerror(errno));
        hwctx_ = c.handle;
        // Kernel 7.1+: command completion is signaled via a DRM syncobj
        // (the WAIT_CMD ioctl was removed).
        syncobj_ = c.syncobj_handle;
    }

    ~pool() {
        if (heap_base_ != nullptr && heap_base_ != MAP_FAILED)
            ::munmap(heap_base_, heap_size_);
        for (auto& m : cmd_maps_) {
            if (m.ptr != nullptr && m.ptr != MAP_FAILED)
                ::munmap(m.ptr, m.size);
        }
        for (auto& s : slices_) {
            struct drm_gem_close c{ .handle = s.handle, .pad = 0 };
            ::ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &c);
        }
        if (heap_handle_) {
            struct drm_gem_close c{ .handle = heap_handle_, .pad = 0 };
            ::ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &c);
        }
        if (fd_ >= 0) ::close(fd_);
    }

    // Allocate a slice from the single heap.
    span alloc(size_t size) {
        if (size == 0) throw std::runtime_error("alloc size 0");
        size = (size + 4095) & ~4095ull;

        uint32_t h = 0;
        if (!create_bo(AMDXDNA_BO_DEV, size, h))
            throw std::runtime_error("DEV bo create failed");

        bo_addr_t bi{};
        if (bo_info(h, &bi) != 0)
            throw std::runtime_error("DEV bo info failed");

        span s;
        s.handle  = h;
        s.size    = size;
        s.host    = heap_base_ + (bi.xdna_addr - heap_dev_base_);
        s.dev_addr = bi.xdna_addr;
        slices_.push_back(s);
        return s;
    }

    // Create a command BO of the given opcode with a start_npu payload.
    // kva writeable via the returned host pointer.
    // NOTE: BO_CMD is NOT carved from the device heap — it is separate host
    // memory with its own mmap (map_offset); the heap-window offset trick
    // used for DEV slices does not apply.
    span alloc_cmd(size_t size) {
        if (size == 0) throw std::runtime_error("alloc_cmd size 0");
        size = (size + 4095) & ~4095ull;

        uint32_t h = 0;
        if (!create_bo(AMDXDNA_BO_CMD, size, h))
            throw std::runtime_error("CMD bo create failed");

        bo_addr_t bi{};
        if (bo_info(h, &bi) != 0)
            throw std::runtime_error("CMD bo info failed");

        uint8_t* host = static_cast<uint8_t*>(
            ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, bi.map_offset));
        if (host == MAP_FAILED)
            throw std::runtime_error(std::string("mmap cmd bo: ") + strerror(errno));
        cmd_maps_.push_back({host, size});

        span s;
        s.handle  = h;
        s.size    = size;
        s.host    = host;
        s.dev_addr = bi.xdna_addr;
        slices_.push_back(s);
        return s;
    }

    // ONE API CALL: submit a chain of N DPU sub-commands via one EXEC_CMD
    // whose cmd BO is an ERT_CMD_CHAIN pointing at N sub BO_CMDs. The driver
    // walks the chain and issues ONE mailbox MSG_OP_CHAIN_EXEC_DPU (0x13) —
    // the same path Windows uses. cmd_count stays 1 (this driver rejects >1).
    //
    // sub_cmds: each is a BO_CMD whose kva holds
    //   amdxdna_cmd{header: op=ERT_START_NPU(20), count=1} +
    //   amdxdna_cmd_start_npu{buffer=inst_dev_addr, buffer_size, prop_count,
    //                          prop_args[prop_count]}
    // Returns the seq to wait on.
    uint64_t submit_chain(std::vector<uint32_t> sub_cmds) {
        if (sub_cmds.empty())
            throw std::runtime_error("submit_chain: empty");

        // 1) build the chain BO (BO_CMD) whose payload is cmd_chain + N handles.
        //    header count = payload length in u32 words (driver: amdxdna_cmd_get_payload).
        const size_t n = sub_cmds.size();
        const size_t chain_bytes = sizeof(amdxdna_cmd_chain) + n * sizeof(uint64_t);
        span cc_bo = alloc_cmd(sizeof(uint32_t) /*header*/ + chain_bytes);
        amdxdna_cmd* cc_cmd = reinterpret_cast<amdxdna_cmd*>(cc_bo.host);
        // header: op in GENMASK(27,23), COUNT in GENMASK(22,12) — count is
        // the payload length in u32 words (driver: amdxdna_cmd_get_payload).
        cc_cmd->header = (19 /*ERT_CMD_CHAIN*/ << 23) |
                         (uint32_t)((chain_bytes / sizeof(uint32_t)) << 12);
        amdxdna_cmd_chain* cc = reinterpret_cast<amdxdna_cmd_chain*>(cc_cmd->data);
        cc->command_count = n;
        cc->submit_index = 0;
        cc->error_index = 0;
        for (size_t i = 0; i < n; i++)
            cc->data[i] = sub_cmds[i];
        sync(cc_bo.handle, SYNC_DIRECT_TO_DEVICE, 0, cc_bo.size);

        // 2) ONE EXEC_CMD (cmd_count=1) referencing the chain BO
        struct amdxdna_drm_exec_cmd e{};
        e.hwctx = hwctx_;
        e.type = AMDXDNA_CMD_SUBMIT_EXEC_BUF;
        e.cmd_count = 1;
        e.cmd_handles = cc_bo.handle;   // driver casts pointer→handle for count=1
        e.arg_count = 0;
        if (::ioctl(fd_, DRM_IOCTL_AMDXDNA_EXEC_CMD, &e) != 0)
            throw std::runtime_error(std::string("EXEC_CMD chain: ") + strerror(errno));
        return e.seq;
    }

    // Build a sub-command BO (BO_CMD) with a start_npu payload.
    // NPU-path contract (amdxdna_cmd_get_cu_idx / amdxdna_cmd_get_payload):
    // payload starts with num_masks CU-mask words (bitmask of target columns)
    // followed by amdxdna_cmd_start_npu; header count = payload words.
    span make_dpu_cmd(uint32_t cu_mask, uint64_t inst_dev_addr,
                      uint32_t inst_size, std::vector<uint32_t> prop_args) {
        const size_t sn_words = sizeof(amdxdna_cmd_start_npu) / sizeof(uint32_t);
        const size_t sz = sizeof(amdxdna_cmd) + sizeof(uint32_t) /*cu_mask*/ +
                          sizeof(amdxdna_cmd_start_npu) +
                          prop_args.size() * sizeof(uint32_t);
        span bo = alloc_cmd(sz);
        amdxdna_cmd* cmd = reinterpret_cast<amdxdna_cmd*>(bo.host);
        cmd->header = (20 /*ERT_START_NPU*/ << 23) |
                      (uint32_t)((1 /*mask*/ + sn_words + prop_args.size()) << 12);
        cmd->data[0] = cu_mask;
        amdxdna_cmd_start_npu* sn =
            reinterpret_cast<amdxdna_cmd_start_npu*>(cmd->data + 1);
        sn->buffer = inst_dev_addr;
        sn->buffer_size = inst_size;
        sn->prop_count = prop_args.size();
        for (size_t i = 0; i < prop_args.size(); i++)
            sn->prop_args[i] = prop_args[i];
        sync(bo.handle, SYNC_DIRECT_TO_DEVICE, 0, bo.size);
        return bo;
    }

    // Wait for command completion (ONE wait, matching one submit).
    // Kernel 7.1+ signals completion through the hwctx's DRM syncobj.
    // timeout_nsec is an ABSOLUTE CLOCK_MONOTONIC deadline, not a duration.
    int wait(uint64_t seq, uint32_t timeout_ms) {
        (void)seq;  // single syncobj per hwctx; one submit in flight at a time
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        int64_t deadline = ts.tv_sec * 1000000000ll + ts.tv_nsec +
                           static_cast<int64_t>(timeout_ms) * 1000000;
        struct drm_syncobj_wait w{};
        w.handles = reinterpret_cast<uint64_t>(&syncobj_);
        w.count_handles = 1;
        w.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
        w.timeout_nsec = deadline;
        return ::ioctl(fd_, DRM_IOCTL_SYNCOBJ_WAIT, &w);
    }

    int sync(uint32_t handle, uint32_t dir, uint64_t off, uint64_t size) {
        struct amdxdna_drm_sync_bo s{};
        s.handle = handle; s.direction = dir; s.offset = off; s.size = size;
        return ::ioctl(fd_, DRM_IOCTL_AMDXDNA_SYNC_BO, &s);
    }

    int fd() const { return fd_; }
    uint32_t hwctx() const { return hwctx_; }
    uint32_t syncobj() const { return syncobj_; }
    uint32_t heap_handle() const { return heap_handle_; }

private:
    struct bo_addr_t {
        uint64_t map_offset;
        uint64_t vaddr;
        uint64_t xdna_addr;
    };
    struct drm_gem_close { uint32_t handle; uint32_t pad; };
    struct cmd_map { uint8_t* ptr; size_t size; };

    bool create_bo(uint32_t type, uint64_t size, uint32_t& out_handle) {
        struct amdxdna_drm_create_bo b{};
        b.type = type; b.size = size; b.flags = 0; b.vaddr = 0;
        if (::ioctl(fd_, DRM_IOCTL_AMDXDNA_CREATE_BO, &b) != 0)
            return false;
        out_handle = b.handle;
        return true;
    }

    int bo_info(uint32_t handle, bo_addr_t* out) {
        struct amdxdna_drm_get_bo_info i{};
        i.handle = handle;
        if (::ioctl(fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &i) != 0)
            return -1;
        out->map_offset = i.map_offset;
        out->vaddr = i.vaddr;
        out->xdna_addr = i.xdna_addr;
        return 0;
    }

    const char* dev_;
    uint32_t num_tiles_ = 32;
    int fd_ = -1;
    uint32_t hwctx_ = 0;
    uint32_t syncobj_ = 0;
    uint32_t heap_handle_ = 0;
    size_t heap_size_ = 0;
    uint8_t* heap_base_ = nullptr;   // cpu map
    uint64_t heap_dev_base_ = 0;     // xdna addr of heap
    std::vector<cmd_map> cmd_maps_;
    std::vector<span> slices_;
};

} // namespace npu

#endif // NPU_POOL_H
