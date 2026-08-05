// backend_onnx.cpp — ONNX Runtime + VitisAI EP backend for Strix Halo NPU.
//
// Runs a GGUF-exported LLM ONNX graph (see tools/gguf_to_onnx.py) on the
// XDNA 2 NPU via AMD's VitisAI execution provider. Pinned format (issue
// #1468, hardware-verified 2026-08-05):
//
//   - fp16 MatMuls with weights baked as initializers (runtime-weight inputs
//     are rejected by the EP's custom op; each weight set = own .onnx)
//   - fp16 boundary tensors (fp32/bf16/int8 boundaries rejected at runtime)
//   - static shapes only; KV cache = fixed [1, NKV, MAX, HD] buffer per layer
//     with a scalar `pos` input; positions > pos are masked to -inf
//   - RMSNorm MUST be the fp16 composed form (Cast-free Mul/ReduceMean/Pow):
//     the EP fuses both the fp32 composed form and the native RMSNormalization
//     op into its rmsnorm1pass superkernel, which HANGS the NPU
//     (ERT_CMD_STATE_TIMEOUT) — the fp16 composition stays on CPU.
//
// Graph I/O (per layer i):
//   in:  input_ids i64 [1,1], pos i64 [1], past_k{i} f16 [1,NKV,MAX,HD],
//        past_v{i} f16 [1,NKV,MAX,HD]
//   out: logits f32 [1,V], present_k{i}/present_v{i} f16 [1,NKV,MAX,HD]
//
// Runtime env (the EP needs all three or it silently runs CPU-only):
//   XILINX_XRT=/opt/xilinx/xrt
//   LD_LIBRARY_PATH += /opt/xilinx/xrt/lib and the libpython3.12 dir
//   (the EP embeds flexml's Python runtime) and the staged EP libs
//
// Env vars:
//   ONNX_MODEL_PATH     — path to .onnx model file
//   ONNX_NPU_DISABLE    — set to "1" to force CPU-only
//   ONNX_NPU_VERBOSE    — set to print model I/O and per-step timing

#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <algorithm>
#include <iterator>

#if __has_include(<onnxruntime_cxx_api.h>)
#define HAS_ORT 1
#include <onnxruntime_cxx_api.h>
#else
#define HAS_ORT 0
#endif

// ── ONNX Runtime Backend ────────────────────────────────────────────────────
#if HAS_ORT

struct OnnxNpuBackend : Backend {
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_opts_;
    std::vector<std::string> input_names_, output_names_;
    std::string model_path_;
    bool npu_available_ = false;
    bool verbose_ = false;

    // dims (from the <model>.dims.json sidecar written by gguf_to_onnx.py —
    // this ORT build returns empty shapes from GetShape())
    int n_layers_ = 0, n_kv_ = 0, hd_ = 0, max_seq_ = 0, vocab_ = 0;
    int pos_ = 0;

    // KV state: per layer a [1, NKV, MAX, HD] fp16 buffer (k then v)
    std::vector<std::vector<Ort::Float16_t>> kv_bufs_;
    std::vector<int64_t> kv_shape_;
    std::vector<float> logits_buf_;

    OnnxNpuBackend() {
        type = BackendType::ONNX_NPU;
        name = "ONNX NPU (VitisAI EP)";
    }

    ~OnnxNpuBackend() override { destroy(); }
    bool can_infer() const override { return initialized && session_ != nullptr; }

    static std::vector<int64_t> read_dims_json(const std::string& path) {
        // minimal JSON scalar read for the sidecar (no deps)
        std::vector<int64_t> out;
        FILE* f = fopen(path.c_str(), "r");
        if (!f) return out;
        std::string s;
        char buf[4096];
        size_t got;
        while ((got = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, got);
        fclose(f);
        const char* keys[] = {"n_layers", "n_kv_heads", "head_dim", "max_seq", "vocab"};
        for (auto k : keys) {
            auto p = s.find(k);
            if (p == std::string::npos) { out.clear(); return out; }
            auto c = s.find(':', p);
            out.push_back(atoll(s.c_str() + c + 1));
        }
        return out;
    }

    bool init(const ModelConfig& cfg_in, const std::string& /*weights_dir*/) override {
        this->cfg = cfg_in;
        verbose_ = (getenv("ONNX_NPU_VERBOSE") != nullptr);

        const char* mp = getenv("ONNX_MODEL_PATH");
        if (!mp || !mp[0]) {
            static const char* candidates[] = {
                "model.onnx", "models/model.onnx",
                "/opt/1bit/models/model.onnx", nullptr
            };
            for (int i = 0; candidates[i]; i++) {
                if (access(candidates[i], R_OK) == 0) { mp = candidates[i]; break; }
            }
        }
        if (!mp || !mp[0]) {
            fprintf(stderr, "ONNX_NPU: ONNX_MODEL_PATH not set and no model.onnx found\n");
            return false;
        }
        model_path_ = mp;
        printf("ONNX_NPU: loading %s\n", model_path_.c_str());

        // dims sidecar
        std::string side = model_path_;
        auto dot = side.find_last_of('.');
        if (dot != std::string::npos) side = side.substr(0, dot);
        side += ".dims.json";
        auto d = read_dims_json(side);
        if (d.size() == 5) {
            n_layers_ = (int)d[0]; n_kv_ = (int)d[1]; hd_ = (int)d[2];
            max_seq_ = (int)d[3]; vocab_ = (int)d[4];
            printf("ONNX_NPU: dims from %s — layers=%d kv_heads=%d hd=%d max_seq=%d vocab=%d\n",
                   side.c_str(), n_layers_, n_kv_, hd_, max_seq_, vocab_);
        } else {
            n_layers_ = cfg.num_layers; n_kv_ = cfg.num_kv_heads; hd_ = cfg.head_dim;
            max_seq_ = cfg.max_seq_len; vocab_ = cfg.vocab_size;
            fprintf(stderr, "ONNX_NPU: no dims sidecar (%s) — using ModelConfig "
                    "(layers=%d kv=%d hd=%d seq=%d vocab=%d); export must match!\n",
                    side.c_str(), n_layers_, n_kv_, hd_, max_seq_, vocab_);
        }
        kv_shape_ = {1, n_kv_, max_seq_, hd_};
        logits_buf_.resize(vocab_);

        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "1bit_onnx_npu");
            session_opts_ = std::make_unique<Ort::SessionOptions>();
            session_opts_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_opts_->SetIntraOpNumThreads(4);

            const char* disable_npu = getenv("ONNX_NPU_DISABLE");
            if (!disable_npu || strcmp(disable_npu, "1") != 0) {
                try {
                    session_opts_->AppendExecutionProvider("VitisAI");
                    // Startup EP assertion (#1468 item 3): the EP fails SILENTLY
                    // into CPU-only when its runtime libs/XRT are missing — hard
                    // error instead of a silently slow (or wrong) path.
                    bool ep_present = false;
                    for (auto& p : Ort::GetAvailableProviders())
                        if (p.find("VitisAI") != std::string::npos) ep_present = true;
                    if (!ep_present) {
                        fprintf(stderr,
                            "ONNX_NPU: VitisAI EP not registered — is the EP lib on "
                            "LD_LIBRARY_PATH? Set XILINX_XRT + libpython3.12 dir per "
                            "the header comment. Refusing to run CPU-only.\n");
                        return false;
                    }
                    npu_available_ = true;
                    printf("ONNX_NPU: VitisAI EP registered\n");
                } catch (const Ort::Exception& e) {
                    fprintf(stderr, "ONNX_NPU: VitisAI EP unavailable (%s). Set "
                            "XILINX_XRT + LD_LIBRARY_PATH per the header comment.\n", e.what());
                    return false;
                }
            }

            session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), *session_opts_);

            Ort::AllocatorWithDefaultOptions alloc;
            size_t n_in = session_->GetInputCount(), n_out = session_->GetOutputCount();
            printf("ONNX_NPU: model loaded — %zu inputs, %zu outputs\n", n_in, n_out);
            for (size_t i = 0; i < n_in; i++)
                input_names_.push_back(session_->GetInputNameAllocated(i, alloc).get());
            for (size_t i = 0; i < n_out; i++)
                output_names_.push_back(session_->GetOutputNameAllocated(i, alloc).get());
            if (verbose_) {
                for (auto& n : input_names_) printf("  in: %s\n", n.c_str());
                for (auto& n : output_names_) printf("  out: %s\n", n.c_str());
            }
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "ONNX_NPU: init failed: %s\n", e.what());
            return false;
        }

        reset();
        initialized = true;
        return true;
    }

    bool reset() override {
        pos_ = 0;
        kv_bufs_.assign((size_t)2 * n_layers_,
                        std::vector<Ort::Float16_t>((size_t)n_kv_ * max_seq_ * hd_, Ort::Float16_t(0.f)));
        return true;
    }

    // Fused forward: one token through the whole graph (KV-cache O(n) decode).
    int generate(int token_id) override {
        if (!initialized || !session_) return -1;
        if (pos_ >= max_seq_) {
            fprintf(stderr, "ONNX_NPU: context exhausted (max_seq=%d) — reset() needed\n", max_seq_);
            return -1;
        }
        try {
            auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> ids{token_id}, poss{pos_};
            std::vector<int64_t> id_sh{1, 1}, pos_sh{1};
            std::vector<Ort::Value> feeds;
            feeds.push_back(Ort::Value::CreateTensor<int64_t>(mem, ids.data(), 1, id_sh.data(), 2));
            feeds.push_back(Ort::Value::CreateTensor<int64_t>(mem, poss.data(), 1, pos_sh.data(), 1));
            for (int i = 0; i < 2 * n_layers_; i++)
                feeds.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                    mem, kv_bufs_[i].data(), kv_bufs_[i].size(), kv_shape_.data(), kv_shape_.size()));

            std::vector<const char*> in_ptrs, out_ptrs;
            for (auto& n : input_names_) in_ptrs.push_back(n.c_str());
            for (auto& n : output_names_) out_ptrs.push_back(n.c_str());

            auto t0 = verbose_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            auto outs = session_->Run(Ort::RunOptions{nullptr}, in_ptrs.data(), feeds.data(),
                                      feeds.size(), out_ptrs.data(), output_names_.size());
            if (verbose_) {
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                fprintf(stderr, "ONNX_NPU: step %d: %.2f ms\n", pos_, ms);
            }

            // logits
            auto* lg = outs[0].GetTensorMutableData<float>();
            size_t n = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
            if (n > logits_buf_.size()) logits_buf_.resize(n);
            memcpy(logits_buf_.data(), lg, n * sizeof(float));

            // present -> kv state
            for (int i = 0; i < 2 * n_layers_; i++)
                memcpy(kv_bufs_[i].data(), outs[1 + i].GetTensorMutableData<Ort::Float16_t>(),
                       (size_t)n_kv_ * max_seq_ * hd_ * 2);

            pos_++;
            int argmax = 0;
            float mx = logits_buf_[0];
            for (size_t i = 1; i < n; i++) {
                if (logits_buf_[i] > mx) { mx = logits_buf_[i]; argmax = (int)i; }
            }
            return argmax;
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "ONNX_NPU: inference failed: %s\n", e.what());
        }
        return -1;
    }

    bool forward(int /*token_id*/, float* /*hidden_out*/) override {
        fprintf(stderr, "ONNX_NPU: use generate() — fused forward only\n");
        return false;
    }

    bool lm_head(const float* /*hidden*/, float* /*logits*/, int* /*argmax*/) override {
        fprintf(stderr, "ONNX_NPU: use generate() — fused forward only\n");
        return false;
    }

    const float* last_logits() override {
        return logits_buf_.empty() ? nullptr : logits_buf_.data();
    }

    float benchmark(int tokens) override {
        if (!initialized) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 42;
        for (int i = 0; i < tokens; i++) {
            tok = generate(tok);
            if (tok < 0) break;
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }

    void destroy() override {
        session_.reset();
        session_opts_.reset();
        env_.reset();
        kv_bufs_.clear();
        initialized = false;
    }
};

#else // !HAS_ORT — stub

struct OnnxNpuBackend : Backend {
    OnnxNpuBackend() { type = BackendType::ONNX_NPU; name = "ONNX NPU (not available)"; }
    bool init(const ModelConfig&, const std::string&) override {
        fprintf(stderr, "ONNX_NPU: ONNX Runtime headers not found\n");
        return false;
    }
    bool can_infer() const override { return false; }
    bool reset() override { return false; }
    int  generate(int) override { return -1; }
    bool forward(int, float*) override { return false; }
    bool lm_head(const float*, float*, int*) override { return false; }
    float benchmark(int) override { return 0; }
    void destroy() override {}
};

#endif // HAS_ORT

extern "C" Backend* create_onnx_npu_backend() { return new OnnxNpuBackend(); }
