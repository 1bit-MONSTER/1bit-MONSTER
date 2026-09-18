// runtime_layer_moe.cpp — MoERuntimeLayerEngine: the single-launch whole-layer
// runlist path for Qwen3.6-35B-A3B (MoE). See include/runtime_layer_moe.h.
#include "runtime_layer_moe.h"
#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_kernel.h>   // xrt::runlist

extern "C" int64_t npu_pack_moe_expert_pool(uint8_t*, ModelWeights*, int);
extern "C" int64_t npu_pack_moe_region_b(uint8_t*, ModelWeights*, int);
extern "C" int64_t npu_pack_moe_linear5_bo(uint8_t*, ModelWeights*, int);
extern "C" int64_t npu_pack_moe_router_bo(uint8_t*, ModelWeights*, int);
extern "C" int npu_pack_lmhead_bo(uint8_t*, ModelWeights*, const ModelConfig*);

static bool read_file(const char* path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize(sz);
    size_t br = fread(out.data(), 1, sz, f);
    fclose(f);
    return br == (size_t)sz;
}

static inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16; float f; memcpy(&f, &bits, 4); return f;
}

// region B base (desc-logical offset of share_up) within the arg-0 weight BO.
//
// The base is a property of the ELF, not of the model, and it DIFFERS BY LIBRARY
// VERSION: the committed v1.0.x capture reads region B at 0x1BC00000, while the
// v0.9.46-regenerated ELF reads it at 0x1E000000. Sizing the BO for one layout and
// running the other ELF puts every weight read out of bounds, which does not fail
// cleanly -- it hangs the runlist and reports ERT_CMD_STATE_TIMEOUT. So this is
// overridable, and MOE_REGION_B_BASE must be set to match whichever ELF is used.
// Measured with decode_txn: the v0.9.46 ELF's arg0 reads end at 533,293,056 B, which
// is 51,357,696 B past the v1.0.x-sized BO.
static size_t region_b_base() {
    if (const char* e = getenv("MOE_REGION_B_BASE")) return (size_t)strtoull(e, nullptr, 0);
    return 0x1bc00000ull;
}
// arg-0 weight BO size: region B base + region B content (3456 rows x 4736).
// MOE_WEIGHT_BO_BYTES overrides, for testing an ELF whose read extent is larger.
static size_t weight_bo_bytes() {
    if (const char* e = getenv("MOE_WEIGHT_BO_BYTES")) return (size_t)strtoull(e, nullptr, 0);
    return region_b_base() + 3456ull * 4736ull;
}

MoERuntimeLayerEngine::MoERuntimeLayerEngine() {}
MoERuntimeLayerEngine::~MoERuntimeLayerEngine() {}

bool MoERuntimeLayerEngine::init(xrt::device& dev, ModelWeights* mw, const ModelConfig& cfg,
                                 const char* layer_elf_dir, const char* lmhead_elf_path,
                                 const char* xclbin_path, int layer) {
    dev_ = &dev; mw_ = mw; cfg_ = cfg;
    elf_dir_ = layer_elf_dir ? layer_elf_dir : "";
    lmhead_elf_path_ = lmhead_elf_path ? lmhead_elf_path : "";
    // The weight/router/norms BOs below hold ONE layer's weights, and the ELF run
    // by forward() must be that same layer. Record it so a mismatch is refused
    // instead of silently computing layer L's program with layer M's weights.
    packed_layer_ = layer;

    // ---- xclbin + hw context (the 35B layer.xclbin, "MLIR_AIE" kernel) ----
    FILE* f = fopen(xclbin_path, "rb");
    if (!f) { fprintf(stderr, "MoERuntimeLayer: cannot open %s\n", xclbin_path); return false; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> raw(fsz);
    fread(raw.data(), 1, fsz, f); fclose(f);
    auto xclbin = std::make_unique<xrt::xclbin>(raw);
    dev.register_xclbin(*xclbin);
    hwctx_ = std::make_unique<xrt::hw_context>(dev, xclbin->get_uuid());
    if (!ensure_layer_kernel(layer)) return false;
    fprintf(stderr, "MoERuntimeLayer: packing LAYER %d weights (moe_layer_ctx%d.elf)\n",
            layer, layer);

    // ---- act BO (hidden state, 2048 bf16) ----
    bo_act_ = std::make_unique<xrt::ext::bo>(dev, 1048576);
    memset(bo_act_->map(), 0, 1048576);
    bo_act_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- weight BO (~460 MB): region B at 0x1bc00000 (region A TODO) ----
    bo_weight_ = std::make_unique<xrt::ext::bo>(dev, weight_bo_bytes());
    uint8_t* w = static_cast<uint8_t*>(bo_weight_->map());
    memset(w, 0, weight_bo_bytes());
    // ---- region A = the EXPERT POOL (addenda 63-75) ----
    // The runtime's arg-3 head is a UNIT-INTERLEAVED EXPERT IMAGE, not the layernorm/conv1d/ssm
    // tensors (those live in the norms BO via npu_pack_moe_linear5_bo). Verified exhaustively
    // against the runtime's own captured arg-3 BO: units 0..16383 up/gate and units 16384..24575
    // down, 98,304 windows total = 465,567,744 B = region-B's base EXACTLY, every slot BAD=0.
    // npu_pack_moe_expert_pool implements exactly those two orders. It writes 478,146,560 B, which
    // runs past 0x1bc00000, but the region-B pack below is issued AFTER this one and covers the
    // whole overlap, so region-B ends up intact.
    {
        int64_t pa = npu_pack_moe_expert_pool(w, mw_, layer);
        if (pa <= 0) { fprintf(stderr, "MoERuntimeLayer: expert pool pack failed\n"); return false; }
        fprintf(stderr, "MoERuntimeLayer: region-A = EXPERT POOL (%lld B, verified layout)\n",
                (long long)pa);
    }
    int64_t rb = npu_pack_moe_region_b(w + region_b_base(), mw_, layer);
    if (rb != (int64_t)3456 * 4736) {
        fprintf(stderr, "MoERuntimeLayer: region-B pack failed (%lld)\n", (long long)rb);
        return false;
    }
    // MOE_ZERO_WEIGHTS: zero the packed weight BO AFTER packing. This is the decisive
    // test for "an inf in the packed weights poisons everything": 0 * inf = NaN, so if
    // the NaN DISAPPEARS with zeroed weights the weights were the source, and if it
    // PERSISTS it is not. Note this also removes any real weight contribution, so a
    // non-NaN result here is not a working layer -- it is only evidence about the source.
    if (getenv("MOE_ZERO_WEIGHTS")) {
        memset(w, 0, weight_bo_bytes());
        fprintf(stderr, "MoERuntimeLayer: WEIGHT BO ZEROED (packed-weight poison probe)\n");
    }
    bo_weight_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- router BO (arg-2) ----
    bo_router_ = std::make_unique<xrt::ext::bo>(dev, 0x3000 + 2048ull * 256ull * 2);
    uint8_t* r = static_cast<uint8_t*>(bo_router_->map());
    memset(r, 0, 0x3000 + 2048ull * 256ull * 2);
    npu_pack_moe_router_bo(r, mw_, layer);
    if (getenv("MOE_ZERO_ROUTER")) {
        memset(r + 0x3000, 0, 2048ull * 256ull * 2);   // zero the router, keep iln/paln/sg
        fprintf(stderr, "MoERuntimeLayer: ROUTER ZEROED (input-independence probe)\n");
    }
    bo_router_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- norms BO (arg-3) = the 5 MB linear-attn BO ----
    bo_norms_ = std::make_unique<xrt::ext::bo>(dev, 5242880);
    uint8_t* n = static_cast<uint8_t*>(bo_norms_->map());
    memset(n, 0, 5242880);
    npu_pack_moe_linear5_bo(n, mw_, layer);
    if (getenv("MOE_ZERO_NORMS_HEAD")) {
        memset(n, 0, 66048);   // conv1d + norm + a + dt (the SSM head)
        fprintf(stderr, "MoERuntimeLayer: NORMS HEAD ZEROED (SSM-param probe)\n");
    }
    // MOE_ZERO_NORMS: zero the WHOLE 5 MB norms BO, including ssm_out at [328192, ...).
    // ssm_out is the one region neither the data-path probe (act), the weights probe
    // (weight BO) nor the gate check (norms head) covers -- and it is read by the ELF at
    // exactly the offset where dump_bos coverage ended. 0 * inf = NaN, so if the NaN goes
    // away here, ssm_out held an inf.
    if (getenv("MOE_ZERO_NORMS")) {
        memset(n, 0, 5242880);
        fprintf(stderr, "MoERuntimeLayer: ENTIRE NORMS BO ZEROED (ssm_out probe)\n");
    }
    bo_norms_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- kv/state BO (arg-4) ----
    bo_kv_ = std::make_unique<xrt::ext::bo>(dev, 134217728);
    memset(bo_kv_->map(), 0, 134217728);
    bo_kv_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- logits BO ----
    bo_logits_ = std::make_unique<xrt::ext::bo>(dev, 1048576);
    memset(bo_logits_->map(), 0, 1048576);
    bo_logits_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- lm_head kernel + weight BO ----
    if (!lmhead_elf_path_.empty()) {
        std::vector<uint8_t> elfb;
        if (read_file(lmhead_elf_path_.c_str(), elfb)) {
            xrt::elf elf((const char*)elfb.data(), elfb.size());
            xrt::module mod(elf);
            kern_lmhead_ = std::make_unique<xrt::ext::kernel>(*hwctx_, mod, "MLIR_AIE");
            int n_tiles = mw_->lm_head_weight.ndim == 2 ? (int)mw_->lm_head_weight.shape[0] : 0;
            if (n_tiles > 0) {
                size_t lm_bytes = (size_t)n_tiles * 5120;
                bo_lmhead_w_ = std::make_unique<xrt::ext::bo>(dev, lm_bytes);
                memset(bo_lmhead_w_->map(), 0, lm_bytes);
                npu_pack_lmhead_bo(static_cast<uint8_t*>(bo_lmhead_w_->map()), mw_, &cfg_);
                bo_lmhead_w_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
                fprintf(stderr, "MoERuntimeLayer: lm_head packed (%d tiles)\n", n_tiles);
            }
        } else {
            fprintf(stderr, "MoERuntimeLayer: cannot read lm_head ELF %s\n", lmhead_elf_path_.c_str());
        }
    }

    fprintf(stderr, "MoERuntimeLayer: init OK (weight %zu B region-B base 0x%zx)\n",
            weight_bo_bytes(), region_b_base());
    if (getenv("MOE_DUMP_ADDRS")) {
        fprintf(stderr, "  BO device addrs: weight=0x%llx act=0x%llx router=0x%llx norms=0x%llx kv=0x%llx logits=0x%llx\n",
                (unsigned long long)bo_weight_->address(),
                (unsigned long long)bo_act_->address(),
                (unsigned long long)bo_router_->address(),
                (unsigned long long)bo_norms_->address(),
                (unsigned long long)bo_kv_->address(),
                (unsigned long long)bo_logits_->address());
        fprintf(stderr, "  desc targets: act=0x40000000 weight=0xe000000 kv=0x2000000 state=0xc0000000\n");
    }
    return true;
}

bool MoERuntimeLayerEngine::ensure_layer_kernel(int layer) {
    // The number in moe_layer_ctx<N>.elf is the MODEL LAYER INDEX (0..39), not a
    // context length — see the header. Because the weight BO holds exactly one
    // layer's weights, running any other layer's ELF would compute layer `layer`
    // with layer packed_layer_'s weights and report it as a result. Refuse.
    if (packed_layer_ >= 0 && layer != packed_layer_) {
        fprintf(stderr,
            "MoERuntimeLayer: REFUSING layer %d — this engine packed LAYER %d's weights "
            "(moe_layer_ctx%d.elf).\n"
            "  The weight/router/norms BOs hold one layer; running another layer's ELF mixes "
            "layer %d's program with layer %d's weights, which is wrong output, not an error.\n"
            "  Fix: init(..., layer=%d) to match, or call forward(%d).\n",
            layer, packed_layer_, packed_layer_, layer, packed_layer_, layer, packed_layer_);
        return false;
    }
    auto it = layer_kernels_.find(layer);
    if (it != layer_kernels_.end()) return true;
    char fname[512];
    snprintf(fname, sizeof(fname), "%s/moe_layer_ctx%d.elf", elf_dir_.c_str(), layer);
    std::vector<uint8_t> elfb;
    if (!read_file(fname, elfb)) {
        fprintf(stderr, "MoERuntimeLayer: missing layer ELF %s\n", fname);
        return false;
    }
    try {
        xrt::elf elf((const char*)elfb.data(), elfb.size());
        xrt::module mod(elf);
        layer_kernels_[layer] = std::make_unique<xrt::ext::kernel>(*hwctx_, mod, "MLIR_AIE");
    } catch (const std::exception& e) {
        fprintf(stderr, "MoERuntimeLayer: kernel build layer=%d failed: %s\n", layer, e.what());
        return false;
    }
    fprintf(stderr, "MoERuntimeLayer: layer kernel layer=%d ready\n", layer);
    return true;
}

bool MoERuntimeLayerEngine::embed(int token) {
    TensorDesc* emb = &mw_->embed_tokens;
    if (emb->ndim != 2 || token < 0 || token >= emb->shape[0]) return false;
    uint64_t off = (uint64_t)token * emb->shape[1] * 2;
    if (mw_->file_size < mw_->data_base + off + emb->shape[1] * 2) return false;
    memcpy(bo_act_->map(), (const uint8_t*)mw_->file_data + mw_->data_base + off,
           emb->shape[1] * 2);
    bo_act_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return true;
}

bool MoERuntimeLayerEngine::forward(int layer) {
    if (!ensure_layer_kernel(layer)) return false;
    if (getenv("MOE_ZERO_ACT")) {
        memset(bo_act_->map(), 0, 4096);
        bo_act_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        fprintf(stderr, "MoERuntimeLayer: ACT ZEROED (input-independence probe)\n");
    }
    // single-launch: layer + lm_head batched into ONE xrt::runlist submit.
    std::vector<xrt::run> runs;
    xrt::runlist rl(*hwctx_);
    {
        runs.emplace_back(*layer_kernels_[layer]);
        xrt::run& run = runs.back();
        uint32_t v0 = 3, v1 = 0, v2 = 0;
        run.set_arg(0, (const void*)&v0, sizeof(v0));
        run.set_arg(1, (const void*)&v1, sizeof(v1));
        run.set_arg(2, (const void*)&v2, sizeof(v2));
        // ARG ORDER -- and why NEITHER order here is a fix.
        //
        // History, because this comment previously claimed a root cause it had not earned:
        //   - the original order was idx3=weight, idx4=act.
        //   - swapping them removes the NaN: idx3=weight -> ALL NaN, idx3=act -> finite.
        //   - the mechanism is that this ELF reads its ACTIVATION from arg3, so with the
        //     weight BO there it reads packed quantized weight bytes as bf16. ~0.4% of
        //     arbitrary 16-bit patterns are bf16 NaN, one NaN in a matmul row NaNs the
        //     row, and the norm spreads it across the hidden vector.
        //
        // BUT THE SWAP IS NOT A FIX. Measured: zeroing ANY of the weight/router/norms BOs
        // leaves the output byte-identical, and the output is byte-identical for weight BO
        // sizes of 481935360, 536870912, 542113792 and 1073741824. A real MoE layer must
        // consume ~12 MB of expert weights per layer; this ELF consumes none of them in
        // ANY order. So it is not a complete MoE layer -- the expert FFN must be carried by
        // other kernels (FLM ships mm.xclbin and dequant_mm.xclbin separately). The finite
        // output below is therefore NOT a valid layer output and must not be read as one.
        //
        // FLM's real bindings for the 35B MoE were captured (run_qwen3_6_moe under
        // cap_interposer.so) and are recorded, including the fact that FLM uses DIFFERENT
        // arg orders for DIFFERENT kernels -- two kernels both carry ~512 MB weight BOs,
        // one at idx3 and one at idx4. See
        // benchmarks/RESULTS-moe35b-flm-real-arg-bindings-2026-09-16.md.
        // MOE_ARG_LEGACY_ORDER restores the original order for A/B only.
        if (getenv("MOE_ARG_LEGACY_ORDER")) {
            run.set_arg(3, (const xrt::bo&)*bo_weight_);
            run.set_arg(4, (const xrt::bo&)*bo_act_);
            fprintf(stderr, "MoERuntimeLayer: ARG ORDER = LEGACY (idx3=weight, idx4=act) -- NaNs\n");
        } else {
            run.set_arg(3, (const xrt::bo&)*bo_act_);      // this ELF reads the ACT from arg3
            run.set_arg(4, (const xrt::bo&)*bo_weight_);   // never consumed -- see above
        }
        run.set_arg(5, (const xrt::bo&)*bo_router_);
        run.set_arg(6, (const xrt::bo&)*bo_norms_);
        run.set_arg(7, (const xrt::bo&)*bo_kv_);
        rl.add(run);
    }
    if (kern_lmhead_ && bo_lmhead_w_) {
        runs.emplace_back(*kern_lmhead_);
        xrt::run& run = runs.back();
        uint32_t v0 = 3, v1 = 0, v2 = 0;
        run.set_arg(0, (const void*)&v0, sizeof(v0));
        run.set_arg(1, (const void*)&v1, sizeof(v1));
        run.set_arg(2, (const void*)&v2, sizeof(v2));
        // MoE lm_head ABI (decoded from gen_lm_head_seq): the weight BD is
        // DDR_PATCH arg_idx 2 = kernel slot 5. slot 3 = logits (output),
        // slot 4 = act (input), slot 6 = final norm (best-effort).
        run.set_arg(3, (const xrt::bo&)*bo_logits_);
        run.set_arg(4, (const xrt::bo&)*bo_act_);
        run.set_arg(5, (const xrt::bo&)*bo_lmhead_w_);
        run.set_arg(6, (const xrt::bo&)*bo_norms_);
        rl.add(run);
    }
    try {
        rl.execute();
        rl.wait();
    } catch (const std::exception& e) {
        fprintf(stderr, "MoERuntimeLayer: runlist FAILED: %s\n", e.what());
        return false;
    }
    if (getenv("NPU_RUNLIST_STATS"))
        fprintf(stderr, "[runlist] %zu runs batched -> 1 submit (layer=%d)\n", runs.size(), layer);
    return true;
}

bool MoERuntimeLayerEngine::get_logits(float* out, int vocab) {
    bo_logits_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
    const uint16_t* lg = (const uint16_t*)bo_logits_->map();
    for (int i = 0; i < vocab; i++) out[i] = bf16_to_f32(lg[i]);
    return true;
}

// ── Host lm_head for the 3-D / Q8_0 lm_head format ──────────────────────────
// Why this exists: init() derives the lm_head tile count as
//     n_tiles = (lm_head_weight.ndim == 2) ? shape[0] : 0
// and Qwen3.5-4B / Qwen3.6-35B-A3B load ndim == 3. n_tiles is therefore 0, the
// weight BO is never created, and forward()'s `if (kern_lmhead_ && bo_lmhead_w_)`
// never adds the lm_head to the runlist — so nothing writes bo_logits_, which is
// memset to zero at init. The run reports success and reads back all zeros.
// See benchmarks/RESULTS-lmhead-ndim-silent-skip-2026-09-16.md.
//
// The layout below is taken verbatim from the dequant the end-to-end engine
// already uses for these models (engine/npu/src/dequant_q4nx.cpp:
// dequant_q8_0_to_float_ex, TILE_ROWS=32 / TILE_COLS=256):
//   8704-byte row = 512 B of 256 bf16 scales, then 8192 B of signed int8
//   scale for (lr, col) = scales[(col/32)*32 + lr]
//   value             = values[lr*TILE_COLS + col]
//   row ir maps to    (tile_row = ir / n_tile_cols, tile_col = ir % n_tile_cols)
//   so output row is  tile_row*TILE_ROWS + lr  and output col is tile_col*TILE_COLS + col
// Checked against the probe: H=2048 -> n_tile_cols = 8 = shape[1], out_rows =
// shape[0]*32 = 248320 = vocab, out_cols = 2048 = H.
bool MoERuntimeLayerEngine::logits_host(float* out, int vocab) {
    if (!mw_ || !out || vocab <= 0) return false;
    TensorDesc* t = &mw_->lm_head_weight;
    if (t->ndim != 3) return false;                 // 2-D models keep the device path

    const int H  = cfg_.hidden_size;
    const int TC = 256, TR = 32;
    const long long ROW_BYTES = 8704;
    if (H <= 0 || H % TC != 0) return false;
    const int n_tile_cols = H / TC;
    const long long i8_rows  = (long long)t->shape[0] * n_tile_cols;
    const long long out_rows = (long long)t->shape[0] * TR;

    if (out_rows != (long long)vocab)
        fprintf(stderr, "MoERuntimeLayer: WARNING lm_head dequant yields %lld rows vs vocab %d\n",
                out_rows, vocab);
    if ((long long)t->data_size < i8_rows * ROW_BYTES) {
        fprintf(stderr, "MoERuntimeLayer: lm_head tensor is %llu B, need %lld B for %lld rows\n",
                (unsigned long long)t->data_size, i8_rows * ROW_BYTES, i8_rows);
        return false;
    }
    const uint8_t* data = (const uint8_t*)model_tensor_data(mw_, t);
    if (!data) return false;

    // hidden state = the first H bf16 of the act BO (it is 1 MB; only H*2 bytes are meaningful)
    bo_act_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
    const uint16_t* act = (const uint16_t*)bo_act_->map();
    std::vector<float> hid((size_t)H);
    for (int i = 0; i < H; i++) hid[(size_t)i] = bf16_to_f32(act[i]);

    for (int i = 0; i < vocab; i++) out[i] = 0.0f;

    // Stream one 8704-byte source row at a time: each yields TR output rows for one
    // tile_col slice of the hidden dim. No full dequantized copy (that would be 2 GB).
    for (long long ir = 0; ir < i8_rows; ir++) {
        const uint8_t* rd = data + ir * ROW_BYTES;
        const uint8_t* scales = rd;                       // 256 bf16, [0,512)
        const int8_t*  values = (const int8_t*)(rd + 512); // 8192 signed int8
        const long long tile_row = ir / n_tile_cols;
        const int tile_col = (int)(ir % n_tile_cols);
        const int hbase = tile_col * TC;

        for (int lr = 0; lr < TR; lr++) {
            const long long v = tile_row * TR + lr;
            if (v >= (long long)vocab) continue;
            float acc = 0.0f;
            const int8_t* vrow = values + (size_t)lr * TC;
            for (int col = 0; col < TC; col++) {
                uint16_t sb; memcpy(&sb, scales + ((size_t)(col / 32) * 32 + lr) * 2, 2);
                float s = bf16_to_f32(sb);
                if (!std::isfinite(s) || std::fabs(s) > 100.0f) s = 0.0f;  // as the reference dequant does
                acc += hid[(size_t)(hbase + col)] * ((float)vrow[col] * s);
            }
            out[v] = acc;
        }
    }
    return true;
}

bool MoERuntimeLayerEngine::dump_bos(const char* dir) {
    auto dump = [&](const char* name, xrt::ext::bo* bo, size_t n, size_t off) {
        if (!bo) return;
        try {
            bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE, n, off);
            char fn[512]; snprintf(fn, sizeof(fn), "%s/%s.bin", dir, name);
            FILE* f = fopen(fn, "wb");
            if (f) { fwrite((const uint8_t*)bo->map() + off, 1, n, f); fclose(f); }
            fprintf(stderr, "DUMPBO %s -> %s (%zu @%zu)\n", name, fn, n, off);
        } catch (...) {}
    };
    dump("act",      bo_act_.get(),    4096,      0);
    dump("router",   bo_router_.get(), 0x3000,    0);
    dump("norms",    bo_norms_.get(),  0x50200,   0);
    // 0x200000, not 0x100000: decode_txn reports the ELF's arg-4 lengths in 4-BYTE
    // WORDS, so `arg4 @49152 len=524288` is 2 MB of traffic. Dumping only the first
    // 1 MB showed "exactly half of every state entry non-finite", which is not what
    // overflow looks like -- it is what half a tensor looks like. See
    // RESULTS-moe-bo-nan-survey-2026-09-16.md.
    dump("kv",       bo_kv_.get(),     0x200000,  0);
    dump("weightA",  bo_weight_.get(), 0x100000,  0);          // region-A head
    dump("weightB",  bo_weight_.get(), 0x200000,  0x1bc00000); // region-B head
    dump("logits",   bo_logits_.get(), 0x100000,  0);
    return true;
}

bool MoERuntimeLayerEngine::dump_act(const char* path, size_t n) {
    bo_act_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fwrite(bo_act_->map(), 1, n, f);
    fclose(f);
    return true;
}
