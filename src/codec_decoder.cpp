// codec_decoder.cpp — Audio codec decoder implementation
//
// Uses onnxruntime C API (OrtSession, OrtValue, etc.) when
// USE_ONNXRUNTIME is defined at build time. Falls back to a no-op
// stub that returns empty PCM — safe to ship in builds without
// onnxruntime, the caller handles the empty result as "TTS unavailable".
//
// Thread safety: load() is not thread-safe (call once at init). decode()
// is thread-safe — each call creates its own OrtIoBinding / OrtValues
// on the stack and does not mutate the session.

#include "codec_decoder.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// ── Debug logging ──────────────────────────────────────────────────────
#ifndef NDEBUG
#define CODEC_LOG(fmt, ...) fprintf(stderr, "[codec] " fmt "\n", ##__VA_ARGS__)
#else
#define CODEC_LOG(fmt, ...) ((void)0)
#endif

// ── ONNX Runtime path — only compiled when USE_ONNXRUNTIME is set ──────
#ifdef USE_ONNXRUNTIME

#include <onnxruntime_c_api.h>

// Forward declarations of ORT API functions (loaded dynamically or linked)
// We use the official OrtApi struct from onnxruntime_c_api.h directly.

struct CodecDecoderImpl {
    std::string model_path;
    const OrtApi* ort{nullptr};
    OrtEnv* env{nullptr};
    OrtSession* session{nullptr};
    OrtMemoryInfo* memory_info{nullptr};
    int64_t expected_samples_per_frame{0}; // samples per single code frame

    ~CodecDecoderImpl() {
        if (session)  ort->ReleaseSession(session);
        if (env)      ort->ReleaseEnv(env);
        if (memory_info) ort->ReleaseMemoryInfo(memory_info);
    }
};

bool CodecDecoder::load(const std::string& onnx_model_path) {
    impl_ = std::make_unique<CodecDecoderImpl>();
    impl_->model_path = onnx_model_path;

    // Get ORT API pointer
    impl_->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!impl_->ort) {
        CODEC_LOG("Failed to get ONNX Runtime API (ORT_API_VERSION=%d)", ORT_API_VERSION);
        impl_.reset();
        return false;
    }

    OrtStatus* status = nullptr;

    // Create environment
    status = impl_->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "codec_decoder", &impl_->env);
    if (status) {
        CODEC_LOG("CreateEnv failed: %s", impl_->ort->GetErrorMessage(status));
        impl_->ort->ReleaseStatus(status);
        impl_.reset();
        return false;
    }

    // Session options
    OrtSessionOptions* session_opts = nullptr;
    status = impl_->ort->CreateSessionOptions(&session_opts);
    if (status) {
        CODEC_LOG("CreateSessionOptions failed: %s", impl_->ort->GetErrorMessage(status));
        impl_->ort->ReleaseStatus(status);
        impl_.reset();
        return false;
    }

    // Enable CPU/CUDA/CoreML fallback: set intra-op threads
    impl_->ort->SetSessionGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL);

    // Try CUDA provider if available (not an error if unavailable)
    OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
    status = impl_->ort->CreateCUDAProviderOptions(&cuda_opts);
    if (status) {
        // CUDA not available at build time — fine, use CPU
        impl_->ort->ReleaseStatus(status);
    } else {
        // Try appending CUDA; if it fails, CPU is still the fallback
        OrtStatus* append_status = impl_->ort->SessionOptionsAppendExecutionProvider_CUDA_V2(session_opts, cuda_opts);
        if (append_status) {
            CODEC_LOG("CUDA provider not available, falling back to CPU: %s",
                      impl_->ort->GetErrorMessage(append_status));
            impl_->ort->ReleaseStatus(append_status);
        }
        impl_->ort->ReleaseCUDAProviderOptions(cuda_opts);
    }

    // Try CoreML provider on Apple platforms
#ifdef __APPLE__
    {
        OrtStatus* coreml_status = impl_->ort->SessionOptionsAppendExecutionProvider_CoreML(session_opts, 0);
        if (coreml_status) {
            CODEC_LOG("CoreML provider not available: %s",
                      impl_->ort->GetErrorMessage(coreml_status));
            impl_->ort->ReleaseStatus(coreml_status);
        }
    }
#endif

    // Create session from file
    status = impl_->ort->CreateSession(impl_->env, onnx_model_path.c_str(), session_opts, &impl_->session);
    impl_->ort->ReleaseSessionOptions(session_opts);
    if (status) {
        CODEC_LOG("CreateSession failed for '%s': %s", onnx_model_path.c_str(),
                  impl_->ort->GetErrorMessage(status));
        impl_->ort->ReleaseStatus(status);
        impl_.reset();
        return false;
    }

    // Get memory info for CPU
    status = impl_->ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &impl_->memory_info);
    if (status) {
        CODEC_LOG("CreateCpuMemoryInfo failed: %s", impl_->ort->GetErrorMessage(status));
        impl_->ort->ReleaseStatus(status);
        impl_.reset();
        return false;
    }

    // Probe the model: inspect output shape to determine expected samples per frame.
    // We do this by creating a dummy input with seq_len=1 and running the model,
    // then checking the output size. But to avoid inference just for metadata,
    // we read the output shape info.
    {
        OrtTypeInfo* type_info = nullptr;
        OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        size_t num_outputs = 0;
        status = impl_->ort->SessionGetOutputCount(impl_->session, &num_outputs);
        if (status) { impl_->ort->ReleaseStatus(status); goto probe_done; }

        if (num_outputs > 0) {
            OrtAllocator* allocator = nullptr;
            status = impl_->ort->GetAllocatorWithDefaultOptions(&allocator);
            if (status) { impl_->ort->ReleaseStatus(status); goto probe_done; }

            char* output_name = nullptr;
            status = impl_->ort->SessionGetOutputName(impl_->session, 0, allocator, &output_name);
            if (status) { impl_->ort->ReleaseStatus(status); goto probe_done; }

            status = impl_->ort->SessionGetOutputTypeInfo(impl_->session, 0, &type_info);
            if (status) { impl_->ort->ReleaseStatus(status); goto probe_done; }

            status = impl_->ort->CastTypeInfoToTensorInfo(type_info, &shape_info);
            if (status) { impl_->ort->ReleaseStatus(status); goto probe_done; }

            size_t num_dims = 0;
            status = impl_->ort->GetDimensionsCount(shape_info, &num_dims);
            if (status) { impl_->ort->ReleaseStatus(status); goto probe_done; }

            // Output shape: [1, output_samples] — dim=2, we want dim[1]
            if (num_dims >= 2) {
                int64_t dims[4];
                status = impl_->ort->GetDimensions(shape_info, dims, num_dims);
                if (status) {
                    impl_->ort->ReleaseStatus(status);
                } else if (num_dims >= 2 && dims[1] > 0) {
                    impl_->expected_samples_per_frame = dims[1];
                    CODEC_LOG("Model expects %lld samples per code frame",
                              (long long)impl_->expected_samples_per_frame);
                }
            }

            if (type_info) impl_->ort->ReleaseTypeInfo(type_info);
            if (output_name) allocator->Free(allocator, output_name);
        }
    }
probe_done:
    // If we couldn't probe, default to a reasonable estimate:
    // RVQ-VAE upsamples ~320x, so 1 token ≈ 320 samples at 24kHz
    if (impl_->expected_samples_per_frame <= 0) {
        impl_->expected_samples_per_frame = 320;
        CODEC_LOG("Could not probe output shape — using default %lld samples per frame",
                  (long long)impl_->expected_samples_per_frame);
    }

    CODEC_LOG("Loaded codec decoder from '%s'", onnx_model_path.c_str());
    return true;
}

bool CodecDecoder::is_loaded() const {
    return impl_ && impl_->session;
}

int CodecDecoder::expected_output_samples(int n_tokens) const {
    if (!impl_) return 0;
    return (int)(impl_->expected_samples_per_frame * n_tokens);
}

std::vector<float> CodecDecoder::decode(const int32_t* tokens, int n_tokens, const float* speaker_emb) {
    if (!impl_ || !impl_->session) {
        CODEC_LOG("decode() called but model is not loaded");
        return {};
    }

    OrtApi* ort = impl_->ort;
    OrtStatus* status = nullptr;

    // ── Input shapes ─────────────────────────────────────────────────
    // codec_tokens: [8, seq_len] int32
    // speaker_emb:  [512] float
    constexpr int64_t kNumCodebooks = 8;
    constexpr int64_t kSpeakerEmbDim = 512;

    int64_t token_shape[2] = {kNumCodebooks, n_tokens};
    int64_t speaker_shape[1] = {kSpeakerEmbDim};

    // ── Allocate OrtValues ───────────────────────────────────────────
    OrtValue* token_tensor = nullptr;
    OrtValue* speaker_tensor = nullptr;
    OrtValue* output_tensor = nullptr;

    // Codec tokens input
    status = ort->CreateTensorWithDataAsOpaqueDeleter(
        impl_->memory_info,
        const_cast<int32_t*>(tokens),          // input data (mutable per ORT C API)
        (size_t)(kNumCodebooks * n_tokens) * sizeof(int32_t),
        token_shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        &token_tensor);
    if (status) {
        CODEC_LOG("Failed to create token tensor: %s", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        return {};
    }

    // Speaker embedding input
    status = ort->CreateTensorWithDataAsOpaqueDeleter(
        impl_->memory_info,
        const_cast<float*>(speaker_emb),
        (size_t)kSpeakerEmbDim * sizeof(float),
        speaker_shape, 1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &speaker_tensor);
    if (status) {
        CODEC_LOG("Failed to create speaker tensor: %s", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        ort->ReleaseValue(token_tensor);
        return {};
    }

    // ── Prepare IO binding ───────────────────────────────────────────
    // Input names must match the ONNX model's expected input names.
    // RVQ-VAE convention: "codec_tokens" and "speaker_emb"
    const char* input_names[] = {"codec_tokens", "speaker_emb"};
    const OrtValue* input_values[] = {token_tensor, speaker_tensor};
    const char* output_name = "audio";

    // Run inference
    status = ort->Run(impl_->session,
                      nullptr,                // default run options
                      input_names,            // input names
                      input_values,           // input tensors
                      2,                      // num inputs
                      &output_name,           // output names
                      1,                      // num outputs
                      &output_tensor);
    if (status) {
        CODEC_LOG("Run() failed: %s", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        ort->ReleaseValue(token_tensor);
        ort->ReleaseValue(speaker_tensor);
        return {};
    }

    // ── Extract output ───────────────────────────────────────────────
    OrtTensorTypeAndShapeInfo* out_shape = nullptr;
    status = ort->GetTensorTypeAndShape(output_tensor, &out_shape);
    if (status) {
        CODEC_LOG("GetTensorTypeAndShape failed: %s", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        ort->ReleaseValue(token_tensor);
        ort->ReleaseValue(speaker_tensor);
        ort->ReleaseValue(output_tensor);
        return {};
    }

    size_t num_dims = 0;
    ort->GetDimensionsCount(out_shape, &num_dims);
    int64_t dims[4] = {0};
    if (num_dims > 0) ort->GetDimensions(out_shape, dims, num_dims);

    // Output is [1, output_samples] float
    size_t total_elements = 1;
    for (size_t i = 0; i < num_dims; i++) total_elements *= (size_t)dims[i];
    ort->ReleaseTypeInfo((OrtTypeInfo*)out_shape);

    float* output_data = nullptr;
    status = ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status || !output_data) {
        CODEC_LOG("GetTensorMutableData failed: %s", status ? ort->GetErrorMessage(status) : "null data");
        if (status) ort->ReleaseStatus(status);
        ort->ReleaseValue(token_tensor);
        ort->ReleaseValue(speaker_tensor);
        ort->ReleaseValue(output_tensor);
        return {};
    }

    // Copy output into a vector
    std::vector<float> result(output_data, output_data + total_elements);

    // Release resources
    ort->ReleaseValue(token_tensor);
    ort->ReleaseValue(speaker_tensor);
    ort->ReleaseValue(output_tensor);

    return result;
}

// ── Fallback path: ONNX Runtime not available ──────────────────────────
#else

struct CodecDecoderImpl {
    // Minimal stub — nothing to do
};

bool CodecDecoder::load(const std::string& onnx_model_path) {
    (void)onnx_model_path;
    CODEC_LOG("ONNX Runtime not available (USE_ONNXRUNTIME not defined) — load() is a no-op");
    impl_ = std::make_unique<CodecDecoderImpl>();
    return false; // not loaded
}

bool CodecDecoder::is_loaded() const {
    return false;
}

int CodecDecoder::expected_output_samples(int n_tokens) const {
    (void)n_tokens;
    return 0;
}

std::vector<float> CodecDecoder::decode(const int32_t* tokens, int n_tokens, const float* speaker_emb) {
    (void)tokens;
    (void)n_tokens;
    (void)speaker_emb;
    CODEC_LOG("ONNX Runtime not available — decode() returns empty");
    return {};
}

#endif // USE_ONNXRUNTIME

// ── Common: constructor / destructor / pcm_to_wav ──────────────────────

CodecDecoder::CodecDecoder() = default;
CodecDecoder::~CodecDecoder() = default;

std::string CodecDecoder::pcm_to_wav(const float* samples, size_t num_samples, int sample_rate) {
    if (!samples || num_samples == 0) return {};

    // Convert float PCM (-1..1) to s16le
    std::vector<int16_t> pcm(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        float clamped = samples[i];
        if (clamped < -1.0f) clamped = -1.0f;
        if (clamped > 1.0f) clamped = 1.0f;
        pcm[i] = (int16_t)(clamped * 32767.0f);
    }

    // Build WAV header
    const int channels = 1;
    const int bits_per_sample = 16;
    uint32_t data_size = (uint32_t)(pcm.size() * sizeof(int16_t));
    uint32_t byte_rate = (uint32_t)(sample_rate * channels * bits_per_sample / 8);
    uint16_t block_align = (uint16_t)(channels * bits_per_sample / 8);
    uint32_t riff_size = 36 + data_size;

    std::string out;
    out.reserve(44 + data_size);

    auto put_u32 = [&](uint32_t v) { out.append(reinterpret_cast<const char*>(&v), 4); };
    auto put_u16 = [&](uint16_t v) { out.append(reinterpret_cast<const char*>(&v), 2); };

    // RIFF header
    out += "RIFF"; put_u32(riff_size); out += "WAVE";

    // fmt chunk
    out += "fmt "; put_u32(16);           // chunk size
    put_u16(1);                            // PCM format
    put_u16((uint16_t)channels);           // mono
    put_u32((uint32_t)sample_rate);        // sample rate
    put_u32(byte_rate);                    // byte rate
    put_u16(block_align);                  // block align
    put_u16((uint16_t)bits_per_sample);    // bits per sample

    // data chunk
    out += "data"; put_u32(data_size);
    out.append(reinterpret_cast<const char*>(pcm.data()), data_size);

    return out;
}
