// diffusion_bridge.cpp — stable-diffusion.cpp integration (correct C API)
// Uses the struct-based C API from stable-diffusion.h

#include "diffusion_bridge.h"
#include "stable-diffusion.h"
#include "media_io.h"   // create_video_from_sd_images_to_vector (sd.cpp examples/common)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

// ─── Engine lifecycle ─────────────────────────────────────────────

DiffusionEngine::DiffusionEngine() = default;

DiffusionEngine::~DiffusionEngine() {
    unload_model();
}

bool DiffusionEngine::load_model(const std::string& model_path,
                                  const std::string& vae_path,
                                  const std::string& t5xxl_path,
                                  const std::string& clip_vision_path) {
    unload_model();
    if (model_path.empty()) return false;
    
    model_path_ = model_path;
    vae_path_ = vae_path;
    t5xxl_path_ = t5xxl_path;
    clip_vision_path_ = clip_vision_path;
    
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    
    params.model_path = model_path.c_str();
    params.vae_path = vae_path.empty() ? nullptr : vae_path.c_str();
    params.t5xxl_path = t5xxl_path.empty() ? nullptr : t5xxl_path.c_str();
    params.clip_vision_path = clip_vision_path.empty() ? nullptr
                                : clip_vision_path.c_str();
    params.n_threads = 8;
    params.wtype = SD_TYPE_F32;
    params.rng_type = CUDA_RNG;
    params.flash_attn = true;
    
    sd_ctx_ = new_sd_ctx(&params);
    if (!sd_ctx_) {
        fprintf(stderr, "diffusion: new_sd_ctx failed for %s\n",
                model_path.c_str());
        return false;
    }
    
    printf("diffusion: loaded %s\n", model_path.c_str());
    return true;
}

void DiffusionEngine::unload_model() {
    if (sd_ctx_) { ::free_sd_ctx(sd_ctx_); sd_ctx_ = nullptr; }
    if (up_ctx_) { ::free_upscaler_ctx(up_ctx_); up_ctx_ = nullptr; }
    model_path_.clear();
    vae_path_.clear();
    t5xxl_path_.clear();
    clip_vision_path_.clear();
}

bool DiffusionEngine::is_loaded() const { return sd_ctx_ != nullptr; }
bool DiffusionEngine::supports_video() const {
    return sd_ctx_ && sd_ctx_supports_video_generation(sd_ctx_);
}

// ─── Image generation ─────────────────────────────────────────────

DiffusionResult DiffusionEngine::txt2img(const DiffusionParams& params,
                                          DiffusionProgressFn progress) {
    if (!sd_ctx_) return {};
    auto t0 = std::chrono::steady_clock::now();
    
    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    
    gp.prompt = params.prompt.c_str();
    gp.negative_prompt = params.negative_prompt.empty() ? nullptr
                         : params.negative_prompt.c_str();
    gp.width = params.width;
    gp.height = params.height;
    gp.seed = params.seed;
    gp.batch_count = 1;
    
    sd_sample_params_t sp;
    sd_sample_params_init(&sp);
    sp.sample_steps = params.steps;
    sp.sample_method = EULER_A_SAMPLE_METHOD;
    sp.scheduler = KARRAS_SCHEDULER;
    sp.guidance.txt_cfg = params.cfg_scale;
    gp.sample_params = sp;
    
    // LoRAs
    std::vector<sd_lora_t> loras;
    for (size_t i = 0; i < params.lora_paths.size(); i++) {
        sd_lora_t l;
        l.path = params.lora_paths[i].c_str();
        l.multiplier = i < params.lora_weights.size() ?
                       params.lora_weights[i] : 1.0f;
        l.is_high_noise = false;
        loras.push_back(l);
    }
    gp.loras = loras.data();
    gp.lora_count = (uint32_t)loras.size();
    
    sd_image_t* images_out = nullptr;
    int num_images = 0;
    bool ok = generate_image(sd_ctx_, &gp, &images_out, &num_images);
    
    auto t1 = std::chrono::steady_clock::now();
    
    if (!ok || !images_out || num_images < 1) {
        fprintf(stderr, "diffusion: generate_image failed\n");
        return {};
    }
    
    DiffusionResult result;
    result.width = (int)images_out[0].width;
    result.height = (int)images_out[0].height;
    result.mime_type = "image/png";
    result.frames = 1;
    result.generation_time_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(t1 - t0).count();
    result.seed_used = gp.seed;
    
    // Copy image data
    size_t data_size = (size_t)images_out[0].width *
                       images_out[0].height *
                       images_out[0].channel;
    if (images_out[0].data && data_size > 0) {
        result.data.assign(images_out[0].data,
                           images_out[0].data + data_size);
    }
    
    free_sd_images(images_out, num_images);
    return result;
}

DiffusionResult DiffusionEngine::img2img(const DiffusionParams& params,
                                          DiffusionProgressFn progress) {
    if (!sd_ctx_) return {};
    auto t0 = std::chrono::steady_clock::now();
    
    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    
    gp.prompt = params.prompt.c_str();
    gp.negative_prompt = params.negative_prompt.empty() ? nullptr
                         : params.negative_prompt.c_str();
    gp.width = params.width;
    gp.height = params.height;
    gp.seed = params.seed;
    gp.strength = params.strength;
    gp.batch_count = 1;
    
    sd_sample_params_t sp;
    sd_sample_params_init(&sp);
    sp.sample_steps = params.steps;
    sp.sample_method = EULER_A_SAMPLE_METHOD;
    sp.scheduler = KARRAS_SCHEDULER;
    sp.guidance.txt_cfg = params.cfg_scale;
    gp.sample_params = sp;
    
    // Init image + optional mask (stbi-allocated; freed below).
    sd_image_t init{}, mask{};
    bool have_init = false, have_mask = false;
    if (!params.init_image_path.empty()) {
        have_init = load_sd_image_from_file(&init, params.init_image_path.c_str());
        if (have_init) {
            gp.init_image = init;
        } else {
            fprintf(stderr, "diffusion: failed to load init image %s\n",
                    params.init_image_path.c_str());
        }
    }
    if (!params.mask_image_path.empty()) {
        have_mask = load_sd_image_from_file(&mask, params.mask_image_path.c_str());
        if (have_mask) {
            gp.mask_image = mask;
        } else {
            fprintf(stderr, "diffusion: failed to load mask %s\n",
                    params.mask_image_path.c_str());
        }
    }
    
    sd_image_t* images_out = nullptr;
    int num_images = 0;
    bool ok = generate_image(sd_ctx_, &gp, &images_out, &num_images);
    auto t1 = std::chrono::steady_clock::now();
    
    if (have_init) std::free(init.data);
    if (have_mask) std::free(mask.data);
    
    if (!ok || !images_out || num_images < 1) return {};
    
    DiffusionResult result;
    result.width = (int)images_out[0].width;
    result.height = (int)images_out[0].height;
    result.mime_type = "image/png";
    result.generation_time_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(t1 - t0).count();
    
    size_t sz = (size_t)images_out[0].width * images_out[0].height * images_out[0].channel;
    if (images_out[0].data && sz > 0)
        result.data.assign(images_out[0].data, images_out[0].data + sz);
    
    free_sd_images(images_out, num_images);
    return result;
}

DiffusionResult DiffusionEngine::txt2vid(const DiffusionParams& params,
                                          DiffusionProgressFn progress) {
    return generate_video_impl(params, nullptr);
}

DiffusionResult DiffusionEngine::img2vid(const DiffusionParams& params,
                                          DiffusionProgressFn progress) {
    if (!sd_ctx_ || !supports_video()) return {};
    sd_image_t init{};
    bool have_init = false;
    if (!params.init_image_path.empty()) {
        have_init = load_sd_image_from_file(&init, params.init_image_path.c_str());
        if (!have_init) {
            fprintf(stderr, "diffusion: failed to load init image %s\n",
                    params.init_image_path.c_str());
        }
    }
    DiffusionResult r = generate_video_impl(params, have_init ? &init : nullptr);
    if (have_init) std::free(init.data);  // stbi-allocated
    return r;
}

DiffusionResult DiffusionEngine::generate_video_impl(const DiffusionParams& params,
                                                      const sd_image_t* init_image) {
    if (!sd_ctx_ || !supports_video()) return {};
    auto t0 = std::chrono::steady_clock::now();
    
    sd_vid_gen_params_t vp;
    sd_vid_gen_params_init(&vp);
    
    vp.prompt = params.prompt.c_str();
    vp.negative_prompt = params.negative_prompt.empty() ? nullptr
                         : params.negative_prompt.c_str();
    vp.width = params.width;
    vp.height = params.height;
    vp.seed = params.seed;
    vp.strength = params.strength;
    vp.video_frames = params.video_frames;
    vp.fps = params.video_fps;
    if (init_image) {
        vp.init_image = *init_image;
        vp.strength = params.strength;
    }
    
    sd_sample_params_t sp;
    sd_sample_params_init(&sp);
    sp.sample_steps = params.steps;
    sp.sample_method = EULER_A_SAMPLE_METHOD;
    sp.scheduler = KARRAS_SCHEDULER;
    sp.guidance.txt_cfg = params.cfg_scale;
    vp.sample_params = sp;
    
    // LoRAs (same pattern as txt2img)
    std::vector<sd_lora_t> loras;
    for (size_t i = 0; i < params.lora_paths.size(); i++) {
        sd_lora_t l;
        l.path = params.lora_paths[i].c_str();
        l.multiplier = i < params.lora_weights.size() ?
                       params.lora_weights[i] : 1.0f;
        l.is_high_noise = false;
        loras.push_back(l);
    }
    vp.loras = loras.data();
    vp.lora_count = (uint32_t)loras.size();
    
    sd_image_t* frames_out = nullptr;
    int num_frames = 0;
    sd_audio_t* audio_out = nullptr;
    bool ok = generate_video(sd_ctx_, &vp, &frames_out, &num_frames, &audio_out);
    auto t1 = std::chrono::steady_clock::now();
    
    if (!ok || !frames_out || num_frames < 1) {
        free_sd_audio(audio_out);
        fprintf(stderr, "diffusion: generate_video failed\n");
        return {};
    }
    
    DiffusionResult result;
    result.width = (int)frames_out[0].width;
    result.height = (int)frames_out[0].height;
    result.frames = num_frames;
    
    // Encode container exactly like sd.cpp's own server: webm/webp when the
    // submodule build has SD_USE_WEBM/WEBP, else MJPG AVI (video/x-msvideo).
    result.data = create_video_from_sd_images_to_vector(
        params.video_output_format, frames_out, num_frames,
        params.video_fps, params.output_quality, audio_out);
    free_sd_audio(audio_out);
    free_sd_images(frames_out, num_frames);
    
    if (result.data.empty()) {
        fprintf(stderr, "diffusion: video encode failed\n");
        return {};
    }
    
    // Sniff the actual container rather than trusting the requested format:
    // without SD_USE_WEBM/WEBP every request encodes as MJPG AVI, so a
    // "webm" request would otherwise come back mislabeled.
    result.mime_type = "video/x-msvideo";
    if (result.data.size() >= 4 &&
        memcmp(result.data.data(), "\x1A\x45\xDF\xA3", 4) == 0) {   // EBML magic
        result.mime_type = "video/webm";
    } else if (result.data.size() >= 12 &&
               memcmp(result.data.data(), "RIFF", 4) == 0 &&
               memcmp(result.data.data() + 8, "WEBP", 4) == 0) {
        result.mime_type = "image/webp";
    }
    result.generation_time_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(t1 - t0).count();
    result.seed_used = (int)vp.seed;
    return result;
}

// ─── Upscaling ────────────────────────────────────────────────────

bool DiffusionEngine::load_upscaler(const std::string& path) {
    if (up_ctx_) { free_upscaler_ctx(up_ctx_); up_ctx_ = nullptr; }
    up_ctx_ = ::new_upscaler_ctx(path.c_str(), false, 8, 0, nullptr, nullptr);
    return up_ctx_ != nullptr;
}

DiffusionResult DiffusionEngine::upscale(const uint8_t* rgb, int w, int h,
                                          int factor) {
    if (!up_ctx_) return {};
    sd_image_t input_img = {(uint32_t)w, (uint32_t)h, 3, (uint8_t*)rgb};
    sd_image_t* result_imgs = nullptr;
    int num_out = 0;
    bool ok = ::upscale(up_ctx_, input_img, (uint32_t)factor,
                        &result_imgs, &num_out);
    if (!ok || !result_imgs || num_out < 1) return {};
    
    DiffusionResult r;
    r.width = (int)result_imgs[0].width;
    r.height = (int)result_imgs[0].height;
    r.mime_type = "image/png";
    size_t sz = (size_t)result_imgs[0].width * result_imgs[0].height * result_imgs[0].channel;
    if (result_imgs[0].data && sz > 0)
        r.data.assign(result_imgs[0].data, result_imgs[0].data + sz);
    
    free_sd_images(result_imgs, num_out);
    return r;
}

// ─── LoRA ─────────────────────────────────────────────────────────

bool DiffusionEngine::load_lora(const std::string& path, float weight) {
    printf("diffusion: lora %s weight=%.2f\n", path.c_str(), weight);
    return true;
}

void DiffusionEngine::clear_loras() {}

// ─── Singleton ────────────────────────────────────────────────────

DiffusionEngine& diffusion_engine() {
    static DiffusionEngine engine;
    return engine;
}
