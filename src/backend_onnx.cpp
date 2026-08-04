// backend_onnx.cpp — ONNX Runtime + VitisAI EP backend for Strix Halo NPU.
//
// Uses AMD's official ONNX Runtime execution provider (VitisAI EP) to run
// INT8-quantized ONNX models on the XDNA 2 NPU. Falls back to CPU if NPU
// is unavailable. This is the supported, documented path — no reverse
// engineering needed.
//
// The model must be exported to ONNX with quantized ops (MatMulInteger,
// MatMulNBits, etc.) that the VitisAI EP can offload to the NPU.
//
// Env vars:
//   ONNX_MODEL_PATH    — path to .onnx model file
//   ONNX_NPU_CACHE_DIR — cache dir for compiled NPU artifacts
//   ONNX_NPU_DISABLE   — set to "1" to force CPU-only

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
#include <unistd.h>

#if __has_include(<onnxruntime_cxx_api.h>)
#define HAS_ORT 1
#include <onnxruntime_cxx_api.h>
#else
#define HAS_ORT 0
#endif

// ── Math helpers ────────────────────────────────────────────────────────────
static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// ── ONNX Runtime Backend ────────────────────────────────────────────────────
#if HAS_ORT

struct OnnxNpuBackend : Backend {
    // ONNX Runtime state
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_opts_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    
    std::string model_path_;
    std::string cache_dir_;
    bool npu_available_ = false;
    bool verbose_ = false;
    
    // Model dimensions
    int H = 0, NV = 0, max_seq_len = 4096;
    
    // State buffers
    std::vector<float> hidden;
    std::vector<float> logits_buf;
    std::vector<int64_t> input_ids; // accumulated token ids
    
    OnnxNpuBackend() { 
        type = BackendType::ONNX_NPU; 
        name = "ONNX NPU (VitisAI EP)"; 
    }
    
    ~OnnxNpuBackend() override { destroy(); }
    bool can_infer() const override { return initialized && session_ != nullptr; }

    bool init(const ModelConfig& cfg, const std::string& /*weights_dir*/) override {
        this->cfg = cfg;
        verbose_ = (getenv("ONNX_NPU_VERBOSE") != nullptr);
        
        const char* mp = getenv("ONNX_MODEL_PATH");
        if (!mp || !mp[0]) {
            // ponytail: try common paths; add explicit env var if needed
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
        
        cache_dir_ = getenv("ONNX_NPU_CACHE_DIR") ? getenv("ONNX_NPU_CACHE_DIR") : "/tmp/1bit_onnx_cache";
        
        printf("ONNX_NPU: loading %s\n", model_path_.c_str());
        
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "1bit_onnx_npu");
            session_opts_ = std::make_unique<Ort::SessionOptions>();
            session_opts_->SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_opts_->SetIntraOpNumThreads(4);
            
            // Configure VitisAI EP
            const char* disable_npu = getenv("ONNX_NPU_DISABLE");
            if (!disable_npu || strcmp(disable_npu, "1") != 0) {
                session_opts_->AddConfigEntry(
                    "ep.vitisaiexecutionprovider.cache_dir", cache_dir_.c_str());
                session_opts_->AddConfigEntry(
                    "ep.vitisaiexecutionprovider.cache_key", "1bit_model_v1");
                
                try {
                    // Register VitisAI EP by name — ORT auto-loads the provider .so
                    const char* ep_keys[] = {"config_file", "cache_dir", "cache_key"};
                    const char* ep_vals[] = {"/tmp/1bit_vaip.json", 
                        cache_dir_.c_str(), "1bit_model_v1"};
                    session_opts_->AppendExecutionProvider("VitisAIExecutionProvider", 
                        ep_keys, ep_vals, 3);
                    npu_available_ = true;
                    printf("ONNX_NPU: VitisAI EP registered — NPU enabled\n");
                } catch (const Ort::Exception& e) {
                    printf("ONNX_NPU: VitisAI EP unavailable (%s) — CPU only\n", e.what());
                }
            }
            
            // Create session
            session_ = std::make_unique<Ort::Session>(
                *env_, model_path_.c_str(), *session_opts_);
            
            // Inspect model inputs/outputs
            Ort::AllocatorWithDefaultOptions alloc;
            size_t n_in = session_->GetInputCount();
            size_t n_out = session_->GetOutputCount();
            
            printf("ONNX_NPU: model loaded — %zu inputs, %zu outputs\n", n_in, n_out);
            
            for (size_t i = 0; i < n_in; i++) {
                auto name = session_->GetInputNameAllocated(i, alloc);
                input_names_.push_back(name.get());
                auto info = session_->GetInputTypeInfo(i);
                auto tensor_info = info.GetTensorTypeAndShapeInfo();
                if (i == 0) input_shape_ = tensor_info.GetShape();
                if (verbose_) {
                    printf("  in[%zu]: %s shape=[", i, input_names_.back().c_str());
                    for (auto d : input_shape_) printf("%ld ", (long)d);
                    printf("]\n");
                }
            }
            
            for (size_t i = 0; i < n_out; i++) {
                auto name = session_->GetOutputNameAllocated(i, alloc);
                output_names_.push_back(name.get());
                auto info = session_->GetOutputTypeInfo(i);
                auto tensor_info = info.GetTensorTypeAndShapeInfo();
                if (i == 0) output_shape_ = tensor_info.GetShape();
                if (verbose_) {
                    printf("  out[%zu]: %s shape=[", i, output_names_.back().c_str());
                    for (auto d : output_shape_) printf("%ld ", (long)d);
                    printf("]\n");
                }
            }
            
            // Derive dimensions
            H = cfg.hidden_size > 0 ? cfg.hidden_size : 2048;
            NV = cfg.vocab_size > 0 ? cfg.vocab_size : 32000;
            max_seq_len = cfg.max_seq_len > 0 ? cfg.max_seq_len : 4096;
            
            hidden.resize(H);
            logits_buf.resize(NV);
            input_ids.reserve(max_seq_len);
            
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "ONNX_NPU: init failed: %s\n", e.what());
            return false;
        }
        
        initialized = true;
        return true;
    }

    bool reset() override {
        input_ids.clear();
        return true;
    }

    // Fused forward: append token → run ONNX session → extract logits → argmax
    int generate(int token_id) override {
        if (!initialized || !session_) return -1;
        
        input_ids.push_back(token_id);
        
        try {
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
            
            // Create input tensor: [1, seq_len] of int64 token ids
            std::vector<int64_t> shape = {1, (int64_t)input_ids.size()};
            Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, input_ids.data(), input_ids.size(), 
                shape.data(), shape.size());
            
            std::vector<const char*> in_names, out_names;
            for (auto& n : input_names_) in_names.push_back(n.c_str());
            for (auto& n : output_names_) out_names.push_back(n.c_str());
            
            auto outputs = session_->Run(Ort::RunOptions{nullptr},
                in_names.data(), &input_tensor, 1,
                out_names.data(), out_names.size());
            
            // Extract logits from first output
            if (!outputs.empty()) {
                auto& out = outputs[0];
                auto* data = out.GetTensorMutableData<float>();
                size_t n = out.GetTensorTypeAndShapeInfo().GetElementCount();
                
                if (logits_buf.size() < n) logits_buf.resize(n);
                memcpy(logits_buf.data(), data, n * sizeof(float));
                
                // Argmax
                int argmax = 0;
                float mx = logits_buf[0];
                for (size_t i = 1; i < n; i++) {
                    if (logits_buf[i] > mx) { mx = logits_buf[i]; argmax = (int)i; }
                }
                return argmax;
            }
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
        return logits_buf.empty() ? nullptr : logits_buf.data(); 
    }

    float benchmark(int tokens) override {
        if (!initialized) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 100;
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
