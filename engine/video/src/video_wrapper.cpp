// video_wrapper.cpp — Zero-Python video generation using stable-diffusion.cpp.
// Uses new_sd_ctx + generate_video API for Wan2.2/T5/VAE.
//
// Build:
//   cd engine/video && mkdir build && cd build
//   cmake .. -DCMAKE_BUILD_TYPE=Release
//   make -j$(nproc)
//
// Usage:
//   ./video_engine -m path/to/model.gguf -p "cat walking" -f 16

#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>
#include <getopt.h>
#include <vector>

#include "stable-diffusion.h"

static int progress_cb(int step, int steps, float time, void* data) {
    fprintf(stderr, "\r  [%d/%d]", step + 1, steps);
    fflush(stderr);
    return 0;
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "1bit.systems Video Engine — Zero Python C++ Diffusion\n"
        "\n"
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --model, -m     PATH     Model path (GGUF file)\n"
        "  --prompt, -p    TEXT     Text prompt\n"
        "  --negative, -n  TEXT     Negative prompt\n"
        "  --frames, -f    INT      Number of frames (default: 16, max: 81)\n"
        "  --steps, -s     INT      Denoising steps (default: 50)\n"
        "  --width, -w     INT      Width (default: 640)\n"
        "  --height, -h    INT      Height (default: 480)\n"
        "  --cfg           FLOAT    CFG scale (default: 5.0)\n"
        "  --input-image   PATH     Input image path (for I2V)\n"
        "  --lora          PATH     LoRA weights\n"
        "  --lora-scale    FLOAT    LoRA scale\n"
        "  --seed          INT      Random seed\n"
        "  --output, -o    FILE     Output MP4 path\n"
        "  --backend               Backend: cpu, cuda, vulkan, metal\n"
        "  --benchmark             Benchmark mode\n"
        "  --threads       INT      Thread count\n"
        "  --flash-attn             Enable flash attention\n"
        "  --help, -h              Show this help\n"
        "\n"
        "Examples:\n"
        "  %s -m model.gguf -p 'cat walking' -f 16\n"
        "  %s -m model.gguf -p 'cat' -i input.png -f 81 --backend cuda\n"
        "  %s -m model.gguf -p 'cinematic' --benchmark --flash-attn\n",
        prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    std::string model_path, prompt, negative_prompt, input_image_path;
    std::string lora_path, output_path = "output.mp4", backend = "cpu";
    int num_frames = 16, num_steps = 50, width = 640, height = 480;
    float cfg_scale = 5.0f, lora_scale = 0.7f;
    int seed = 42, threads = 0;
    bool benchmark = false, flash_attn = false;

    static struct option opts[] = {
        {"model",       required_argument, 0, 'm'},
        {"prompt",      required_argument, 0, 'p'},
        {"negative",    required_argument, 0, 'n'},
        {"frames",      required_argument, 0, 'f'},
        {"steps",       required_argument, 0, 's'},
        {"width",       required_argument, 0, 'w'},
        {"height",      required_argument, 0, 'h'},
        {"cfg",         required_argument, 0, 256},
        {"input-image", required_argument, 0, 'i'},
        {"lora",        required_argument, 0, 257},
        {"lora-scale",  required_argument, 0, 258},
        {"seed",        required_argument, 0, 259},
        {"output",      required_argument, 0, 'o'},
        {"backend",     required_argument, 0, 260},
        {"benchmark",   no_argument,       0, 261},
        {"threads",     required_argument, 0, 262},
        {"flash-attn",  no_argument,       0, 263},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:n:f:s:w:h:i:o:t:", opts, 0)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'p': prompt = optarg; break;
            case 'n': negative_prompt = optarg; break;
            case 'f': num_frames = std::max(1, atoi(optarg)); if (num_frames > 81) num_frames = 81; break;
            case 's': num_steps = std::max(1, atoi(optarg)); break;
            case 'w': width = atoi(optarg); break;
            case 'h': height = atoi(optarg); break;
            case 'i': input_image_path = optarg; break;
            case 'o': output_path = optarg; break;
            case 256: cfg_scale = atof(optarg); break;
            case 257: lora_path = optarg; break;
            case 258: lora_scale = atof(optarg); break;
            case 259: seed = atoi(optarg); break;
            case 260: backend = optarg; break;
            case 261: benchmark = true; break;
            case 262: threads = atoi(optarg); break;
            case 263: flash_attn = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }

    if (model_path.empty() || prompt.empty()) {
        fprintf(stderr, "Error: --model and --prompt are required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr,
        "\n=== 1bit.systems Video Engine ===\n"
        "  Model:   %s\n"
        "  Prompt:  %s\n"
        "  Frames:  %d\n"
        "  Steps:   %d\n"
        "  Size:    %dx%d\n"
        "  CFG:     %.1f\n"
        "  Backend: %s\n"
        "  Seed:    %d\n"
        "  Output:  %s\n\n",
        model_path.c_str(), prompt.c_str(),
        num_frames, num_steps, width, height, cfg_scale,
        backend.c_str(), seed, output_path.c_str());

    // --- Init stable-diffusion.cpp context ---
    sd_set_log_callback([](int level, const char* text, void*) {
        if (level <= SD_LOG_WARN) fprintf(stderr, "[sd] %s", text);
    }, nullptr);
    sd_set_progress_callback(progress_cb, nullptr);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = model_path.c_str();
    ctx_params.n_threads = threads > 0 ? threads : sd_get_num_physical_cores();
    ctx_params.wtype = SD_TYPE_F16;
    ctx_params.rng_type = STD_DEFAULT_RNG;
    ctx_params.flash_attn = flash_attn;
    ctx_params.backend = backend.c_str();

    if (!lora_path.empty()) {
        ctx_params.lora_apply_mode = LORA_APPLY_AT_RUNTIME;
    }

    auto t0 = std::chrono::steady_clock::now();
    sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
    auto t1 = std::chrono::steady_clock::now();

    if (!sd_ctx) {
        fprintf(stderr, "Error: Failed to create SD context\n");
        return 1;
    }

    fprintf(stderr, "[Init] Loaded in %.0f ms\n",
            std::chrono::duration<float, std::milli>(t1 - t0).count());
    fprintf(stderr, "[Init] Video support: %s\n",
            sd_ctx_supports_video_generation(sd_ctx) ? "yes" : "no");

    // --- Prepare video generation params ---
    sd_vid_gen_params_t vid_params;
    sd_vid_gen_params_init(&vid_params);

    vid_params.prompt = prompt.c_str();
    vid_params.negative_prompt = negative_prompt.empty() ? nullptr : negative_prompt.c_str();
    vid_params.width = width;
    vid_params.height = height;
    vid_params.video_frames = num_frames;
    vid_params.seed = seed;
    vid_params.strength = cfg_scale;

    // Sampling params
    sd_sample_params_t sample_params;
    sd_sample_params_init(&sample_params);
    sample_params.sample_steps = num_steps;
    sample_params.sample_method = sd_get_default_sample_method(sd_ctx);
    sample_params.scheduler = sd_get_default_scheduler(sd_ctx, sample_params.sample_method);
    vid_params.sample_params = sample_params;

    // Input image for I2V
    sd_image_t init_image = {0, 0, 0, nullptr};
    if (!input_image_path.empty()) {
        // TODO: load image from file
        fprintf(stderr, "[I2V] Input image: %s\n", input_image_path.c_str());
        vid_params.init_image = init_image;
    }

    // LoRA
    sd_lora_t lora = {false, 1.0f, nullptr};
    sd_lora_t* loras = nullptr;
    uint32_t lora_count = 0;
    if (!lora_path.empty()) {
        lora.path = lora_path.c_str();
        lora.multiplier = lora_scale;
        loras = &lora;
        lora_count = 1;
    }
    vid_params.loras = loras;
    vid_params.lora_count = lora_count;

    // --- Generate ---
    fprintf(stderr, "\n[Generate] Starting...\n");
    t0 = std::chrono::steady_clock::now();

    sd_image_t* frames_out = nullptr;
    int num_frames_out = 0;
    sd_audio_t* audio_out = nullptr;

    bool ok = generate_video(sd_ctx, &vid_params, &frames_out, &num_frames_out, &audio_out);

    t1 = std::chrono::steady_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (ok && frames_out && num_frames_out > 0) {
        fprintf(stderr, "\n[Generate] %d frames generated\n", num_frames_out);

        // Save frames to MP4 using ffmpeg pipe
        // frames_out[i] = { width, height, channels, data }
        FILE* ffmpeg = popen(
            "ffmpeg -y -f rawvideo -pix_fmt rgb24 "
            "-s 640x480 -r 8 -i - "
            "-c:v libx264 -pix_fmt yuv420p output.mp4 2>/dev/null",
            "w");

        if (ffmpeg) {
            for (int i = 0; i < num_frames_out; i++) {
                fwrite(frames_out[i].data, 1,
                       frames_out[i].width * frames_out[i].height * 3, ffmpeg);
            }
            pclose(ffmpeg);
            fprintf(stderr, "  Saved to output.mp4\n");
        }

        free_sd_images(frames_out, num_frames_out);
    }

    fprintf(stderr,
        "\n=== Results ===\n"
        "  Total:    %.0f ms\n"
        "  Per frame: %.0f ms\n"
        "  FPS:     %.2f\n"
        "  Output:  %s\n\n",
        total_ms, total_ms / num_frames,
        1000.0f * num_frames / total_ms,
        output_path.c_str());

    free_sd_ctx(sd_ctx);
    return ok ? 0 : 1;
}
