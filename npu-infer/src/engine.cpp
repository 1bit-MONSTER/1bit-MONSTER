#include "engine.h"
#include "common.h"
#include "model.h"

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <chrono>
#include <sys/stat.h>
#include <random>

// ========= BF16 helpers =========
static inline float bf16_to_float_cpp(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
static inline uint16_t float_to_bf16_cpp(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return (uint16_t)((bits + rounding_bias) >> 16);
}

// ========= NpuBo =========
NpuBo::~NpuBo() {}
NpuBo::NpuBo(NpuBo&& other) noexcept 
    : bo(std::move(other.bo)), map(other.map), size(other.size), label(std::move(other.label)) {
    other.map = nullptr; other.size = 0;
}
NpuBo& NpuBo::operator=(NpuBo&& other) noexcept {
    bo = std::move(other.bo); map = other.map; size = other.size; label = std::move(other.label);
    other.map = nullptr; other.size = 0; return *this;
}
bool NpuBo::create(xrt::device& device, size_t sz, uint32_t group_id, const char* label_str) {
    try {
        // amdxdna binds each BO to the kernel argument slot via the BO group:
        // opcode=0, instr=1, ninstr=2, host buffers from slot 3 on. group 0 BOs
        // are silently ignored by kernels that declare non-zero groups (the ERT
        // command completes but the AIE never executes); large group-0 BOs can
        // even wedge the NPU (IO_PAGE_FAULTs). Use kernel.group_id(3+i).
        bo = std::make_unique<xrt::bo>(device, sz, xrt::bo::flags::host_only, group_id);
        map = (uint8_t*)bo->map(); size = sz;
        if (label_str) label = label_str;
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("BO create failed (%s): %s", label_str ? label_str : "?", e.what());
        return false;
    }
}
void NpuBo::sync_to_device(size_t offset, size_t sz) {
    if (sz == 0) sz = size - offset;
    bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, sz, offset);
}
void NpuBo::sync_from_device(size_t offset, size_t sz) {
    if (sz == 0) sz = size - offset;
    bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE, sz, offset);
}

// ========= WeightPacker =========
WeightPacker::WeightPacker(ModelWeights* mw, ModelConfig* config) : mw_(mw), config_(config) {}
WeightPacker::~WeightPacker() {}
// in_features per projection: attn q/k/v/o and FFN gate/up are [*, H]; the
// FFN down is [H, IM]. The dequant needs the true input dimension to split
// the Q4NX tile grid.
static int weight_in_features(const TensorDesc* desc, const ModelConfig* cfg) {
    const char* n = desc->name;
    if (strstr(n, "down_proj") && !strstr(n, "gate")) return (int)cfg->intermediate_size;
    return (int)cfg->hidden_size;
}
int WeightPacker::num_bos(const TensorDesc* desc) const { return npu_weight_num_blocks(desc, config_, weight_in_features(desc, config_)); }
size_t WeightPacker::bo_size(const TensorDesc* desc) const {
    int in = weight_in_features(desc, config_);
    // Wide weights (FFN down: in=3072) use full-width [256, in] blocks:
    // 256 * in * 2 bytes, rounded up. Narrow weights keep the 1 MB BO.
    if (in > (int)config_->npu_block_cols)
        return (size_t)config_->npu_block_rows * (size_t)in * 2;
    return config_->npu_weight_bo_size;
}
void WeightPacker::pack_block(uint8_t* buffer, const TensorDesc* desc, int block_idx) const {
    void* data = model_tensor_data(mw_, const_cast<TensorDesc*>(desc));
    npu_pack_weight_bo(buffer, data, desc, config_, block_idx, weight_in_features(desc, config_));
}
int WeightPacker::pack_to_bos(const TensorDesc* desc, NpuBo* bos, int max_bos, xrt::device& device, uint32_t group_id) const {
    int n = num_bos(desc); if (n > max_bos) n = max_bos;
    size_t bsize = bo_size(desc);
    for (int i = 0; i < n; i++) {
        char label[128]; snprintf(label, sizeof(label), "%s_b%d", desc->name, i);
        if (!bos[i].create(device, bsize, group_id, label)) return i;
        pack_block(bos[i].map, desc, i); bos[i].sync_to_device();
    }
    return n;
}

// ========= XclbinManager =========
XclbinManager::XclbinManager(xrt::device& device) : device_(device) {}
XclbinManager::~XclbinManager() {}
bool XclbinManager::load(XclbinType type) {
    if (type < 0 || type >= XCLBIN_COUNT) return false;
    Entry& e = entries_[type]; if (e.loaded) return true;
    std::string resolved;
    if (!xclbin_dir_.empty()) {
        resolved = xclbin_dir_ + "/" + XCLBIN_NAMES[type];
        // The 35B MoE dir names its dequant kernel dequant_mm.xclbin.
        if (type == XCLBIN_DEQUANT) {
            struct stat st;
            if (stat(resolved.c_str(), &st) != 0) {
                std::string alt = xclbin_dir_ + "/dequant_mm.xclbin";
                if (stat(alt.c_str(), &st) == 0) resolved = alt;
            }
        }
    } else {
        resolved = XCLBIN_PATHS[type];
    }
    const char* path = resolved.c_str();
    LOG_DEBUG("Loading %s ...", path);
    FILE* f = fopen(path, "rb"); if (!f) { LOG_ERROR("Cannot open %s", path); return false; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> raw_data(fsize);
    size_t br = fread(raw_data.data(), 1, fsize, f); (void)br; fclose(f);
    try {
        auto xclbin = std::make_unique<xrt::xclbin>(raw_data);
        device_.register_xclbin(*xclbin);
        auto kernel = std::make_unique<xrt::kernel>(device_, xclbin->get_uuid(), "MLIR_AIE");
        e.xclbin = std::move(xclbin); e.kernel = std::move(kernel);

        // Load the companion instruction stream (<xclbin>.bin). Without it the
        // ERT command completes but the AIE never executes (silent no-op).
        // Generate with tools/gen_mm_insts (FastFlowLM Gemm::generate_seq).
        std::string insts_path(path);
        size_t dot = insts_path.rfind('.');
        if (dot != std::string::npos) insts_path = insts_path.substr(0, dot);
        insts_path += ".bin";
        FILE* fi = fopen(insts_path.c_str(), "rb");
        if (fi) {
            fseek(fi, 0, SEEK_END); long isz = ftell(fi); fseek(fi, 0, SEEK_SET);
            if (isz > 0 && isz % 4 == 0) {
                std::vector<uint32_t> insts(isz / 4);
                size_t br = fread(insts.data(), 4, insts.size(), fi);
                if (br == insts.size()) {
                    auto insts_bo = std::make_unique<xrt::bo>(
                        device_, (size_t)isz, XCL_BO_FLAGS_CACHEABLE,
                        e.kernel->group_id(1));  // use-after-move fix:  local was moved into e.kernel
                    memcpy(insts_bo->map(), insts.data(), (size_t)isz);
                    insts_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, (size_t)isz, 0);
                    e.insts_bo = std::move(insts_bo);
                    e.ninstr = (uint32_t)insts.size();
                }
            }
            fclose(fi);
        }
        if (!e.insts_bo) {
            LOG_ERROR("No insts for %s — generate with tools/gen_mm_insts (kernel would be a silent no-op)", path);
        }
        e.loaded = true;
        LOG_INFO("Loaded %s (%u insts)", path, e.ninstr);
        return true;
    } catch (const std::exception& ex) {
        LOG_ERROR("Failed to load %s: %s", path, ex.what()); return false;
    }
}
xrt::kernel* XclbinManager::kernel(XclbinType type) {
    if (type < 0 || type >= XCLBIN_COUNT) return nullptr;
    return entries_[type].kernel.get();
}
xrt::bo* XclbinManager::insts_bo(XclbinType type) {
    if (type < 0 || type >= XCLBIN_COUNT) return nullptr;
    return entries_[type].insts_bo.get();
}
uint32_t XclbinManager::ninstr(XclbinType type) {
    if (type < 0 || type >= XCLBIN_COUNT) return 0;
    return entries_[type].ninstr;
}
xrt::bo* XclbinManager::insts_for(xrt::kernel* kern, uint32_t m, uint32_t k,
                                  uint32_t n, uint32_t woff,
                                  uint32_t* out_ninstr) {
    if (out_ninstr) *out_ninstr = 0;
    if (!kern) return nullptr;
    // Kernel-specific stream key: mm vs attn vs layer encode different
    // DMA/compute graphs. The previous shared key made the attn kernel
    // execute the MM stream (hang at layer 1; NPU_GEMM_FIX.md, #2006).
    const char* kern_name = "mm";
    {
        static const char* names[XCLBIN_COUNT] = {"mm", "attn", "layer", "dequant"};
        for (int i = 0; i < XCLBIN_COUNT; i++) {
            if (kern == entries_[i].kernel.get()) { kern_name = names[i]; break; }
        }
    }
    ShapeKey key{m, k, n, woff, std::string(kern_name)};
    auto it = shape_insts_.find(key);
    if (it != shape_insts_.end()) {
        if (out_ninstr) *out_ninstr = it->second.ninstr;
        return it->second.bo.get();
    }
    // Directory: $NPU_INSTS_DIR, else the companion <xclbin>.bin's dir.
    if (insts_dir_.empty()) {
        const char* dir = getenv("NPU_INSTS_DIR");
        if (dir && *dir) {
            insts_dir_ = dir;
        } else {
            std::string p = XCLBIN_PATHS[XCLBIN_MM];
            size_t slash = p.rfind('/');
            insts_dir_ = (slash == std::string::npos) ? "." : p.substr(0, slash);
        }
    }
    char fname[256];
    snprintf(fname, sizeof(fname), "%s/%s_%u_%u_%u_%u.bin",
             insts_dir_.c_str(), kern_name, m, k, n, woff);
    FILE* fi = fopen(fname, "rb");
    if (!fi) {
        LOG_ERROR("No per-shape insts %s — run tools/gen_mm_insts_batch "
                  "(issue #2006)", fname);
        return nullptr;
    }
    fseek(fi, 0, SEEK_END); long isz = ftell(fi); fseek(fi, 0, SEEK_SET);
    ShapeInsts si;
    if (isz > 0 && isz % 4 == 0) {
        std::vector<uint32_t> insts(isz / 4);
        size_t br = fread(insts.data(), 4, insts.size(), fi);
        if (br == insts.size()) {
            si.bo = std::make_unique<xrt::bo>(
                device_, (size_t)isz, XCL_BO_FLAGS_CACHEABLE, kern->group_id(1));
            memcpy(si.bo->map(), insts.data(), (size_t)isz);
            si.bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, (size_t)isz, 0);
            si.ninstr = (uint32_t)insts.size();
        }
    }
    fclose(fi);
    if (!si.bo) {
        LOG_ERROR("Failed to load %s", fname);
        return nullptr;
    }
    LOG_DEBUG("Loaded per-shape insts %s (%u words)", fname, si.ninstr);
    if (out_ninstr) *out_ninstr = si.ninstr;
    return shape_insts_.emplace(key, std::move(si)).first->second.bo.get();
}

// ========= NpuInferenceEngine =========
NpuInferenceEngine::NpuInferenceEngine() {}
NpuInferenceEngine::~NpuInferenceEngine() { model_free(model_); }

bool NpuInferenceEngine::pack_tensor_blocks(std::vector<NpuBo>& blocks, 
                                             const TensorDesc* desc,
                                             const char* label_prefix,
                                             uint32_t group_id) {
    int n = packer_->num_bos(desc);
    blocks.resize(n);
    for (int i = 0; i < n; i++) {
        char label[128];
        snprintf(label, sizeof(label), "%s_b%d", label_prefix, i);
        if (!blocks[i].create(*device_, packer_->bo_size(desc), group_id, label)) return false;
        packer_->pack_block(blocks[i].map, desc, i);
        blocks[i].sync_to_device();
    }
    return true;
}

bool NpuInferenceEngine::cache_all_weights() {
    LOG_INFO("=== Pre-packing all weights into BOs ===");
    auto t0 = std::chrono::steady_clock::now();
    
    weight_cache_master_.resize(config_.num_layers);
    for (int l = 0; l < config_.num_layers; l++) {
        weight_cache_master_[l] = std::make_unique<WeightCacheLayer>();
    }
    
    weight_cache_ptrs_.resize(config_.num_layers);
    for (int l = 0; l < config_.num_layers; l++)
        weight_cache_ptrs_[l] = weight_cache_master_[l].get();
    for (int c = 0; c < 3; c++)
        hwctx_[c].weight_cache_ptr = &weight_cache_ptrs_;
    
    for (int l = 0; l < config_.num_layers; l++) {
        LayerWeights* lw = &model_->layers[l];
        WeightCacheLayer* cache = weight_cache_master_[l].get();
        
        xrt::kernel* mm_kern = xclbins_->kernel(XCLBIN_MM);
        uint32_t g_w = mm_kern ? (uint32_t)mm_kern->group_id(5) : 0;
        if (!pack_tensor_blocks(cache->q_proj_blocks, &lw->q_proj_weight, "q_proj", g_w)) return false;
        if (!pack_tensor_blocks(cache->k_proj_blocks, &lw->k_proj_weight, "k_proj", g_w)) return false;
        if (!pack_tensor_blocks(cache->v_proj_blocks, &lw->v_proj_weight, "v_proj", g_w)) return false;
        if (!pack_tensor_blocks(cache->o_proj_blocks, &lw->o_proj_weight, "o_proj", g_w)) return false;
        if (!pack_tensor_blocks(cache->gate_proj_blocks, &lw->gate_proj_weight, "gate_proj", g_w)) return false;
        if (!pack_tensor_blocks(cache->up_proj_blocks, &lw->up_proj_weight, "up_proj", g_w)) return false;
        if (!pack_tensor_blocks(cache->down_proj_blocks, &lw->down_proj_weight, "down_proj", g_w)) return false;
        
        if (l % 10 == 0) LOG_DEBUG("  Layer %d weights cached", l);
    }
    
    TensorDesc* lm_head = &model_->lm_head_weight;
    {
        xrt::kernel* mm_kern = xclbins_->kernel(XCLBIN_MM);
        uint32_t g_w = mm_kern ? (uint32_t)mm_kern->group_id(5) : 0;
        if (!pack_tensor_blocks(lm_head_blocks_, lm_head, "lm_head", g_w)) return false;
    }
    
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    int total_bos = 0;
    for (auto& c : weight_cache_master_) {
        total_bos += c->q_proj_blocks.size() + c->k_proj_blocks.size() + c->v_proj_blocks.size()
                   + c->o_proj_blocks.size() + c->gate_proj_blocks.size() + c->up_proj_blocks.size()
                   + c->down_proj_blocks.size();
    }
    total_bos += lm_head_blocks_.size();
    
    LOG_INFO("All weights cached: %d layers, ~%d BOs in %.0f ms",
             config_.num_layers, total_bos, ms);
    return true;
}

bool NpuInferenceEngine::init(const char* model_path) {
    LOG_INFO("=== NPU Inference Engine Init ===");
    config_ = QWEN3_0_6B_CONFIG;
    
    model_ = model_load(model_path, config_);
    if (!model_) return false;
    // model_load derives num_layers/vocab/hidden from the metadata for
    // non-0.6B models — propagate the corrected config back.
    if (model_->config.num_layers != config_.num_layers ||
        model_->config.hidden_size != config_.hidden_size ||
        model_->config.vocab_size != config_.vocab_size) {
        config_ = model_->config;
        LOG_INFO("Engine config updated: %d layers, hidden %d, vocab %d",
                 config_.num_layers, config_.hidden_size, config_.vocab_size);
    }
    
    device_ = std::make_unique<xrt::device>();
    try { *device_ = xrt::device(0); }
    catch (...) {
        try { *device_ = xrt::device(1); }
        catch (const std::exception& e) { LOG_ERROR("Cannot open NPU: %s", e.what()); return false; }
    }
    
    xclbins_ = std::make_unique<XclbinManager>(*device_);
    // Per-model xclbin selection: NPU_XCLBIN_DIR wins (only if it actually
    // exists — a stale export (e.g. an old worktree's engine/npu/xclbins)
    // must not shadow the FLM path), else $FLM_XCLBIN_PATH/xclbins/<model>,
    // else the hardcoded default.
    if (const char* xd = getenv("NPU_XCLBIN_DIR")) {
        struct stat st;
        if (stat(xd, &st) == 0 && S_ISDIR(st.st_mode))
            xclbins_->set_xclbin_dir(xd);
        else
            LOG_WARNING("NPU_XCLBIN_DIR=%s does not exist — ignoring", xd);
    }
    if (xclbins_->xclbin_dir().empty()) {
        std::string mp = model_path;
        auto slash = mp.rfind('/');
        std::string model_name = (slash == std::string::npos) ? mp : mp.substr(slash + 1);
        if (model_name == "model.q4nx") {
            std::string dir = mp.substr(0, slash);
            auto s2 = dir.rfind('/');
            model_name = (s2 == std::string::npos) ? dir : dir.substr(s2 + 1);
        }
        const char* flm = getenv("FLM_XCLBIN_PATH");
        if (flm) {
            std::string d = std::string(flm) + "/xclbins/" + model_name;
            struct stat st;
            if (stat(d.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
                xclbins_->set_xclbin_dir(d);
        }
    }
    for (int i = 0; i < XCLBIN_COUNT; i++)
        if (!xclbins_->load((XclbinType)i)) return false;
    
    packer_ = std::make_unique<WeightPacker>(model_, &config_);
    
    for (int c = 0; c < 3; c++) {
        hwctx_[c].current_seq_len = 0;
        xrt::kernel* mm_kern = xclbins_->kernel(XCLBIN_MM);
        uint32_t g_act = mm_kern ? (uint32_t)mm_kern->group_id(3) : 0;
        uint32_t g_ws  = mm_kern ? (uint32_t)mm_kern->group_id(4) : 0;
        uint32_t g_kv  = mm_kern ? (uint32_t)mm_kern->group_id(7) : 0;
        if (!hwctx_[c].act_bo.create(*device_, config_.npu_activation_bo_size, g_act, "act")) return false;
        if (!hwctx_[c].act_workspace.create(*device_, 10485760, g_ws, "workspace")) return false;
        if (!hwctx_[c].kv_cache.create(*device_, config_.npu_kv_cache_bo_size, g_kv, "kv_cache")) return false;
        hwctx_[c].kv_cache.sync_to_device();
    }
    
    if (!cache_all_weights()) return false;
    
    int vocab_size = config_.vocab_size;
    lm_head_buffer_.resize(vocab_size);
    
    // ---- Runtime layer path (Round 36): NPU_RUNTIME_LAYERS=1 ----
    if (getenv("NPU_RUNTIME_LAYERS")) {
        const char* elf_dir = getenv("NPU_LAYER_ELF_DIR");
        const char* lmhead_elf = getenv("NPU_LMHEAD_ELF");
        runtime_layers_ = std::make_unique<RuntimeLayerEngine>();
        if (runtime_layers_->init(*device_, model_, config_,
                                  elf_dir ? elf_dir : "captures/txn-elfs",
                                  lmhead_elf ? lmhead_elf : "captures/txn-elfs/elf_0002_lmhead.bin")) {
            use_runtime_layers_ = true;
            rt_ctx_len_ = 0;
            LOG_INFO("Runtime layer path ENABLED (FastFlowLM ELF submission)");
            if (getenv("NPU_RUNTIME_SELFTEST")) {
                LOG_INFO("Runtime layer self-test: forward(1000) then forward(1001)");
                runtime_layers_->embed(1000);
                runtime_layers_->forward(1);
                runtime_layers_->dump_act(getenv("RT_ST_ACT1") ? getenv("RT_ST_ACT1") : "/tmp/st_act1.bin");
                runtime_layers_->embed(1001);
                runtime_layers_->forward(2);
                runtime_layers_->dump_act(getenv("RT_ST_ACT2") ? getenv("RT_ST_ACT2") : "/tmp/st_act2.bin");
                LOG_INFO("self-test dumps written");
            }
        } else {
            LOG_ERROR("Runtime layer path init FAILED — falling back to mm pipeline");
            runtime_layers_.reset();
        }
    }
    
    LOG_INFO("=== Init Complete ===");
    LOG_INFO("Model: %s (%d layers, %d hidden)",
             model_path, config_.num_layers, config_.hidden_size);
    return true;
}

// === Sequential GEMM ===
// Individual kernel call with wait. insts/ninstr come from
// XclbinManager::insts_for (per-shape, issue #2006); nullptr aborts the call
// (without insts the ERT command is a silent no-op).
static void run_gemm(xrt::kernel* kern, xrt::bo* insts, uint32_t ninstr,
                      xrt::bo& act, xrt::bo& ws,
                      xrt::bo& w1, xrt::bo& w2, xrt::bo& kv,
                      const char* tag = "") {
    if (!kern || !insts) return;
    try {
        auto r = (*kern)(
            (uint64_t)3,
            *insts,
            ninstr,
            act, ws, w1, w2, kv
        );
        r.wait();
    } catch (const std::exception& e) {
        LOG_ERROR("run_gemm FAILED %s: %s", tag, e.what());
        throw;
    }
}

// Blocked GEMM over one projection's weight BOs. The instruction stream is
// per-shape: out[256, N] = W[256, K] @ act[K, N] with K = hidden (q/k/v/o/
// gate/up) or intermediate (down) and N = token batch (engine: 128-padded
// decode). weight_offset is 0 — each block lives in its own BO.
static void run_blocked_gemm(XclbinManager* mgr, xrt::kernel* kern,
                              uint32_t k, uint32_t n,
                              xrt::bo& act, xrt::bo& ws,
                              xrt::bo& kv, std::vector<NpuBo>& weights) {
    const uint32_t M = 256;  // kernel block rows (NPU row grid)
    uint32_t ninstr = 0;
    xrt::bo* insts = mgr->insts_for(kern, M, k, n, 0, &ninstr);
    int bi = 0;
    for (auto& w : weights) {
        char tag[128];
        snprintf(tag, sizeof(tag), "k=%u n=%u block=%d", k, n, bi++);
        run_gemm(kern, insts, ninstr, act, ws, *w.bo, *w.bo, kv, tag);
    }
}

// === Layer MM pipeline ===
void NpuInferenceEngine::run_layer_mm(HwCtxState& ctx, int layer_idx) {
    WeightCacheLayer& wc = ctx.weight_cache(layer_idx);
    xrt::kernel* mm_kern = xclbins_->kernel(XCLBIN_MM);
    if (!mm_kern) return;
    xrt::bo& act = *ctx.act_bo.bo;
    xrt::bo& ws = *ctx.act_workspace.bo;
    xrt::bo& kv = *ctx.kv_cache.bo;
    const uint32_t N = 128;  // decode token batch (kernel N granularity)
    run_blocked_gemm(xclbins_.get(), mm_kern, config_.hidden_size, N, act, ws, kv, wc.q_proj_blocks);
    run_blocked_gemm(xclbins_.get(), mm_kern, config_.hidden_size, N, act, ws, kv, wc.k_proj_blocks);
    run_blocked_gemm(xclbins_.get(), mm_kern, config_.hidden_size, N, act, ws, kv, wc.v_proj_blocks);
}

// === Attention pipeline ===
void NpuInferenceEngine::run_layer_attn(HwCtxState& ctx, int layer_idx) {
    WeightCacheLayer& wc = ctx.weight_cache(layer_idx);
    xrt::kernel* attn_kern = xclbins_->kernel(XCLBIN_ATTN);
    if (!attn_kern) return;
    xrt::bo& act = *ctx.act_bo.bo;
    xrt::bo& ws = *ctx.act_workspace.bo;
    xrt::bo& kv = *ctx.kv_cache.bo;
    xrt::bo& w = (!wc.o_proj_blocks.empty()) ? *wc.o_proj_blocks[0].bo : act;
    uint32_t ninstr = 0;
    xrt::bo* insts = xclbins_->insts_for(attn_kern, 256, config_.hidden_size, 128, 0, &ninstr);
    run_gemm(attn_kern, insts, ninstr, act, ws, w, w, kv);
    if (!wc.o_proj_blocks.empty()) {
        run_blocked_gemm(xclbins_.get(), xclbins_->kernel(XCLBIN_MM), config_.hidden_size, 128, act, ws, kv, wc.o_proj_blocks);
    }
}

// === MLP pipeline ===
void NpuInferenceEngine::run_layer_mlp(HwCtxState& ctx, int layer_idx) {
    WeightCacheLayer& wc = ctx.weight_cache(layer_idx);
    xrt::kernel* mm_kern = xclbins_->kernel(XCLBIN_MM);
    if (!mm_kern) return;
    xrt::bo& act = *ctx.act_bo.bo;
    xrt::bo& ws = *ctx.act_workspace.bo;
    xrt::bo& kv = *ctx.kv_cache.bo;
    const uint32_t N = 128;
    run_blocked_gemm(xclbins_.get(), mm_kern, config_.hidden_size, N, act, ws, kv, wc.gate_proj_blocks);
    run_blocked_gemm(xclbins_.get(), mm_kern, config_.hidden_size, N, act, ws, kv, wc.up_proj_blocks);
    // down_proj: K = intermediate (3072) — the engine keeps [256, 1024]
    // blocks; a full [256, 3072] stream needs a shared-BO layout. For now
    // emit the K=intermediate shape (batch generator covers it) so a
    // correctly-laid-out weight BO feeds the right stream.
    run_blocked_gemm(xclbins_.get(), mm_kern, config_.intermediate_size, N, act, ws, kv, wc.down_proj_blocks);
}

// Decoder temperature: NPU_TEMPERATURE env (default 0 = greedy, which keeps
// the runtime-path byte-identity validations deterministic).
static float sampler_temperature() {
    const char* t = getenv("NPU_TEMPERATURE");
    return t ? (float)atof(t) : 0.0f;
}

// === Prefill ===
bool NpuInferenceEngine::run_prefill(const int* input_tokens, int num_input_tokens) {
    LOG_INFO("=== Prefill %d tokens ===", num_input_tokens);
    if (num_input_tokens < 1) return false;
    
    for (int t = 0; t < num_input_tokens; t++) {
        if (use_runtime_layers_) {
            // Runtime path: one layer-kernel run per layer + lm_head (byte-
            // verified vs the FastFlowLM runtime). ctx_len advances per token.
            if (!runtime_layers_->embed(input_tokens[t])) return false;
            rt_ctx_len_ = t + 1;
            if (!runtime_layers_->forward(rt_ctx_len_)) return false;
            // per-token logits dump for the runtime A/B (RT_LOGITS_PREFIX)
            if (const char* rp = getenv("RT_LOGITS_PREFIX")) {
                runtime_layers_->get_logits(lm_head_buffer_.data(), config_.vocab_size);
                char fn[512];
                snprintf(fn, sizeof(fn), "%s_%d.bin", rp, input_tokens[t]);
                FILE* f = fopen(fn, "wb");
                if (f) {
                    fwrite(lm_head_buffer_.data(), sizeof(float), config_.vocab_size, f);
                    fclose(f);
                }
            }
            if (t % 7 == 0) LOG_DEBUG("  Runtime layer %d/%d done (ctx=%d)",
                                      t, num_input_tokens, rt_ctx_len_);
            if (t == num_input_tokens - 1) {
                runtime_layers_->get_logits(lm_head_buffer_.data(), config_.vocab_size);
                rt_first_token_ = sample_token(lm_head_buffer_.data(), config_.vocab_size, sampler_temperature());
                LOG_DEBUG("  runtime prefill first token: %d", rt_first_token_);
            }
            continue;
        }
        embed_lookup(input_tokens[t], hwctx_[0].act_bo);
        
        for (int l = 0; l < config_.num_layers; l++) {
            run_layer_mm(hwctx_[0], l);
            run_layer_attn(hwctx_[1], l);
            run_layer_mlp(hwctx_[2], l);
            if (l % 7 == 0) LOG_DEBUG("  Layer %d done", l);
        }
    }
    
    LOG_INFO("Prefill complete");
    return true;
}

// === Decode ===
int NpuInferenceEngine::run_decode_step(int last_token) {
    if (use_runtime_layers_) {
        if (!runtime_layers_->embed(last_token)) return 0;
        rt_ctx_len_++;
        if (!runtime_layers_->forward(rt_ctx_len_)) return 0;
        if (const char* ksd = getenv("RT_DUMP_KV_STEP")) {
            char kf[256];
            snprintf(kf, sizeof(kf), "%s_ctx%d.bin", ksd, rt_ctx_len_);
            FILE* fk = fopen(kf, "wb");
            if (fk) { fwrite(runtime_layers_->map_kv(0), 1, 33554432, fk); fclose(fk); }
        }
        if (const char* asd = getenv("RT_DUMP_ACT_STEP")) {
            char af[256];
            snprintf(af, sizeof(af), "%s_ctx%d.bin", asd, rt_ctx_len_);
            runtime_layers_->dump_act(af);
        }
        runtime_layers_->get_logits(lm_head_buffer_.data(), config_.vocab_size);
        if (const char* lsd = getenv("RT_DUMP_LOGITS_STEP")) {
            char lf[256];
            snprintf(lf, sizeof(lf), "%s_ctx%d.bin", lsd, rt_ctx_len_);
            FILE* fl = fopen(lf, "wb");
            if (fl) {
                const uint16_t* lg16 = (const uint16_t*)lm_head_buffer_.data();
                // lm_head_buffer_ is float; write as bf16 for comparison
                FILE* fb = fopen(lf, "wb");
                if (fb) {
                    for (int i = 0; i < config_.vocab_size; i++) {
                        float v = lm_head_buffer_[i];
                        uint32_t bits; memcpy(&bits, &v, 4);
                        uint16_t bf = (uint16_t)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
                        fwrite(&bf, 2, 1, fb);
                    }
                    fclose(fb);
                }
                fclose(fl);
            }
        }
        return sample_token(lm_head_buffer_.data(), config_.vocab_size, sampler_temperature());
    }
    embed_lookup(last_token, hwctx_[0].act_bo);
    
    for (int l = 0; l < config_.num_layers; l++) {
        run_layer_mm(hwctx_[0], l);
        run_layer_attn(hwctx_[1], l);
        run_layer_mlp(hwctx_[2], l);
    }
    
    // LM head
    xrt::kernel* mm_kern = xclbins_->kernel(XCLBIN_MM);
    if (mm_kern && !lm_head_blocks_.empty()) {
        xrt::bo& act = *hwctx_[0].act_bo.bo;
        xrt::bo& ws = *hwctx_[0].act_workspace.bo;
        xrt::bo& kv = *hwctx_[0].kv_cache.bo;
        for (auto& w : lm_head_blocks_) {
            uint32_t ninstr = 0;
            xrt::bo* insts = xclbins_->insts_for(mm_kern, 256, config_.hidden_size, 128, 0, &ninstr);
            run_gemm(mm_kern, insts, ninstr, act, ws, *w.bo, *w.bo, kv);
        }
    }
    
    // Read logits (first vocab_size BF16 values from activation BO)
    hwctx_[0].act_bo.sync_from_device(0, config_.vocab_size * 2);
    uint16_t* logits_bf16 = (uint16_t*)hwctx_[0].act_bo.map;
    
    for (int i = 0; i < config_.vocab_size; i++) {
        lm_head_buffer_[i] = bf16_to_float_cpp(logits_bf16[i]);
    }
    
    return sample_token(lm_head_buffer_.data(), config_.vocab_size, sampler_temperature());
}

// === Sample ===// Real decoder sampling (Round 38): temperature softmax + optional top-k /
// top-p filtering, drawn from the engine's seeded RNG. temperature <= 0
// (or 1e-6-ish) falls back to greedy argmax — the default, which keeps the
// runtime-path byte-identity validations deterministic. Env knobs:
//   NPU_SEED         RNG seed (default 42)
//   NPU_TEMPERATURE  >0 enables sampling (softmax /T)
//   NPU_TOP_K        top-k filter (default 0 = off)
//   NPU_TOP_P        nucleus filter (default 1.0 = off)
int NpuInferenceEngine::sample_token(const float* logits, int vocab_size, float temperature) {
    // seed the RNG ONCE per run — reseeding before every draw would make
    // each token use the same first value of the stream (biased sampling)
    if (!rng_seeded_) {
        if (const char* s = getenv("NPU_SEED")) rng_.seed((uint64_t)strtoull(s, nullptr, 0));
        rng_seeded_ = true;
    }
    int top_k = getenv("NPU_TOP_K") ? atoi(getenv("NPU_TOP_K")) : 0;
    float top_p = getenv("NPU_TOP_P") ? (float)atof(getenv("NPU_TOP_P")) : 1.0f;
    if (temperature <= 0.0f) {
        // greedy argmax (default; matches the runtime harness GREEDY_NEXT)
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) { max_val = logits[i]; max_idx = i; }
        }
        return max_idx;
    }
    // temperature softmax over the top-k / top-p survivors
    std::vector<std::pair<float,int>> cand;
    cand.reserve(vocab_size);
    for (int i = 0; i < vocab_size; i++) cand.push_back({logits[i], i});
    std::sort(cand.begin(), cand.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (top_k > 0 && (size_t)top_k < cand.size()) cand.resize(top_k);
    float max_l = cand[0].first;
    double sum = 0.0;
    for (auto& c : cand) { c.first = expf((c.first - max_l) / temperature); sum += c.first; }
    if (top_p < 1.0f) {
        // nucleus: keep the smallest prefix of (sorted, now-scaled) candidates
        // whose cumulative probability >= top_p
        double acc = 0.0; size_t keep = cand.size();
        for (size_t i = 0; i < cand.size(); i++) {
            acc += cand[i].first / sum;
            if (acc >= top_p) { keep = i + 1; break; }
        }
        if (keep < cand.size()) cand.resize(keep);
        sum = 0.0; for (auto& c : cand) sum += c.first;
    }
    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng_);
    double acc = 0.0;
    for (auto& c : cand) { acc += c.first; if (r <= acc) return c.second; }
    return cand.back().second;
}

// === Embed lookup ===
void NpuInferenceEngine::embed_lookup(int token, NpuBo& dest) {
    TensorDesc* emb = &model_->embed_tokens;
    if (emb->ndim == 2 && token >= 0 && token < emb->shape[0]) {
        uint64_t offset = (uint64_t)token * emb->shape[1] * 2;
        size_t copy_size = emb->shape[1] * 2;
        if (copy_size > dest.size) copy_size = dest.size;
        memcpy(dest.map, model_->file_data + emb->data_offset + offset, copy_size);
        dest.sync_to_device(0, copy_size);
    }
}

// === Generate ===
int NpuInferenceEngine::generate(const int* input_tokens, int num_input_tokens,
                                  int* output_tokens, int max_output_tokens) {
    LOG_INFO("=== Generate ===");
    auto t_total = std::chrono::steady_clock::now();
    
    // Reset KV cache
    for (int c = 0; c < 3; c++) {
        memset(hwctx_[c].kv_cache.map, 0, hwctx_[c].kv_cache.size);
        hwctx_[c].kv_cache.sync_to_device();
        hwctx_[c].current_seq_len = 0;
    }
    
    if (!run_prefill(input_tokens, num_input_tokens)) {
        LOG_ERROR("Prefill failed");
        return 0;
    }

    // dump the prefill logits for comparison vs the runtime (NPU_LOGITS_DUMP)
    if (const char* ld = getenv("NPU_LOGITS_DUMP")) {
        FILE* f = fopen(ld, "wb");
        if (f && use_runtime_layers_) {
            runtime_layers_->get_logits(lm_head_buffer_.data(), config_.vocab_size);
            fwrite(lm_head_buffer_.data(), sizeof(float), config_.vocab_size, f);
            fprintf(stderr, "prefill logits dumped (%d floats)\n", config_.vocab_size);
        }
        if (f) fclose(f);
    }

    current_token_ = use_runtime_layers_ && rt_first_token_ >= 0
        ? rt_first_token_ : input_tokens[num_input_tokens - 1];
    int num_out = 0;
    
    for (int i = 0; i < max_output_tokens; i++) {
        current_token_ = run_decode_step(current_token_);
        output_tokens[num_out++] = current_token_;
        
        if (current_token_ == 0) break;
    }
    
    auto t_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_total).count();
    
    LOG_INFO("Generated %d tokens in %.0f ms (%.1f ms/tok)",
             num_out, total_ms, total_ms / (num_out > 0 ? num_out : 1));
    
    return num_out;
}
