// video_main.cpp — Zero-Python C++ Video Diffusion Engine for 1bit.systems.
// Reuses NPU INT8 primitives when available, CPU OpenMP fallback.
//
// Build (CPU only, for development):
//   g++ -std=c++23 -O3 -march=native -fopenmp -o video_engine \
//       video_main.cpp -lm
//
// Build (NPU + XRT):
//   g++ -std=c++23 -O3 -march=native -fopenmp -DUSE_NPU -o video_engine \
//       video_main.cpp ../npu/src/npu_engine_universal.cpp \
//       -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl
//
// Usage:
//   ./video_engine --prompt "a cat walking" --frames 16 --steps 50
//   ./video_engine --prompt "cinematic dolly zoom" --model weights.bin --output out

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <getopt.h>

#include "video_model.h"
#include "video_dit.h"
#include "video_sampler.h"
#include "video_vae.h"

// --- T5 text encoder stub ---
// In production: use a C++ T5 implementation or pre-computed embeddings.
// For now: generate random text embeddings for pipeline testing.
struct TextEncoder {
    int dim = 4096;  // T5-encoder hidden size
    int max_len = 512;

    // Generate text embedding from prompt
    // In production: this calls a T5 encoder model
    void encode(const char* prompt, float* emb_out, int& out_len) {
        if (!prompt || strlen(prompt) == 0) {
            // Empty prompt → null embedding (all zeros)
            out_len = 1;
            memset(emb_out, 0, dim * sizeof(float));
            fprintf(stderr, "[Text] Empty prompt → null embedding\n");
            return;
        }

        // Stub: generate deterministic embedding from prompt hash
        // REPLACE with real T5 encoder in production
        unsigned long hash = 5381;
        for (const char* p = prompt; *p; p++) {
            hash = ((hash << 5) + hash) + (unsigned char)(*p);
        }

        out_len = 77; // CLIP-style fixed length
        for (int i = 0; i < out_len; i++) {
            for (int d = 0; d < dim; d++) {
                float val = (float)((hash + i * 2654435761u + d * 314159265u) % 1000) / 500.0f - 1.0f;
                emb_out[i * dim + d] = val * 0.1f;
            }
        }
        fprintf(stderr, "[Text] Encoded '%s' → %d tokens\n", prompt, out_len);
    }
};

// --- CLI ---
static void print_usage(const char* prog) {
    fprintf(stderr,
        "Zero-Python C++ Video Diffusion Engine — 1bit.systems\n"
        "\n"
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --prompt, -p   TEXT    Text prompt (default: 'a cat walking')\n"
        "  --frames, -f   INT     Number of frames (default: 16, max: 81)\n"
        "  --steps, -s    INT     Denoising steps (default: 50)\n"
        "  --width                 Output width (default: 640)\n"
        "  --height                Output height (default: 480)\n"
        "  --cfg           FLOAT   CFG scale (default: 5.0)\n"
        "  --model, -m     PATH    Model weights file (.bin)\n"
        "  --output, -o    FILE    Output path (default: 'output')\n"
        "  --seed          INT     Random seed\n"
        "  --benchmark             Run benchmark mode (no output, just timing)\n"
        "  --help, -h              Show this help\n"
        "\n"
        "Examples:\n"
        "  %s --prompt 'cat walking' --frames 16 --steps 50\n"
        "  %s --prompt 'dolly zoom' --model weights.bin --benchmark\n"
        "\n",
        prog, prog, prog);
}

int main(int argc, char** argv) {
    // Default config — small for quick test, override with --width --height
    std::string prompt = "a cat walking, cinematic lighting";
    std::string model_path;
    std::string output_path = "output";
    int num_frames = 16;
    int num_steps = 50;
    int width = 640;
    int height = 480;
    float cfg_scale = 5.0f;
    int seed = 42;
    bool benchmark_mode = false;

    // Parse args
    static struct option long_opts[] = {
        {"prompt",    required_argument, nullptr, 'p'},
        {"frames",    required_argument, nullptr, 'f'},
        {"steps",     required_argument, nullptr, 's'},
        {"width",     required_argument, nullptr, 'w'},
        {"height",    required_argument, nullptr, 'h'},
        {"cfg",       required_argument, nullptr, 'c'},
        {"model",     required_argument, nullptr, 'm'},
        {"output",    required_argument, nullptr, 'o'},
        {"seed",      required_argument, nullptr, 256},
        {"benchmark", no_argument,       nullptr, 257},
        {"help",      no_argument,       nullptr, 258},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "p:f:s:w:h:c:m:o:", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'p': prompt = optarg; break;
            case 'f': num_frames = atoi(optarg); break;
            case 's': num_steps = atoi(optarg); break;
            case 'w': width = atoi(optarg); break;
            case 'h': height = atoi(optarg); break;
            case 'c': cfg_scale = atof(optarg); break;
            case 'm': model_path = optarg; break;
            case 'o': output_path = optarg; break;
            case 256: seed = atoi(optarg); break;
            case 257: benchmark_mode = true; break;
            case 258: print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }

    if (num_frames < 1) num_frames = 1;
    if (num_frames > 81) num_frames = 81;
    if (num_steps < 1) num_steps = 1;
    srand(seed);

    fprintf(stderr,
        "\n=== Video Diffusion Engine — 1bit.systems ===\n"
        "  Prompt:  %s\n"
        "  Frames:  %d\n"
        "  Steps:   %d\n"
        "  Size:    %dx%d\n"
        "  CFG:     %.1f\n"
        "  Seed:    %d\n"
        "  Output:  %s\n"
        "\n",
        prompt.c_str(), num_frames, num_steps,
        width, height, cfg_scale, seed, output_path.c_str());

    // --- Init model config ---
    VideoModelConfig cfg;
    cfg.num_frames = num_frames;
    cfg.height = height;
    cfg.width = width;
    cfg.num_steps = num_steps;
    cfg.cfg_scale = cfg_scale;
    cfg.derive();
    
    fprintf(stderr, "[Config] H=%d NH=%d HD=%d IM=%d C=%d\n",
            cfg.H, cfg.NH, cfg.HD, cfg.IM, cfg.C);
    fprintf(stderr, "[Config] Latent: %dx%d patches = %d\n",
            cfg.latent_h, cfg.latent_w, cfg.num_patches);

    // --- Load weights ---
    std::vector<float> weights;
    if (!model_path.empty()) {
        if (!load_video_weights(model_path.c_str(), cfg, weights)) {
            fprintf(stderr, "[FATAL] Failed to load weights from %s\n", model_path.c_str());
            return 1;
        }
    } else {
        fprintf(stderr, "[Weights] No model specified — using random init\n");
        weights.resize(1024 * 1024); // 4MB placeholder
    }

    // --- Init DiT ---
    VideoDiT dit(cfg);
    if (!weights.empty()) {
        dit.load_weights(weights.data(), weights.size());
    }

    // --- Init text encoder ---
    TextEncoder text_encoder;
    text_encoder.dim = cfg.TH;
    std::vector<float> text_emb(cfg.TX * cfg.TH);
    int text_len = 0;
    text_encoder.encode(prompt.c_str(), text_emb.data(), text_len);

    // --- Init sampler ---
    FlowMatchSchedule fm_scheduler;
    fm_scheduler.prepare(num_steps);

    DenoisingPipeline pipeline(cfg);
    pipeline.is_flow_match = true;
    pipeline.scheduler = &fm_scheduler;

    // --- Full latent dimensions ---
    // Each frame: [C, latent_h, latent_w] → patched: [num_patches, C*PS*PS]
    // We process one frame at a time for memory efficiency
    int patch_dim = cfg.C * cfg.PS * cfg.PS;
    int latent_N = cfg.num_patches;
    int latent_total = latent_N * patch_dim;

    // --- Generate frames ---
    std::vector<float> all_frames(num_frames * cfg.C * (height/8) * (width/8));
    auto t_start = std::chrono::steady_clock::now();

    for (int f = 0; f < num_frames; f++) {
        fprintf(stderr, "\n[Frame %d/%d]\n", f + 1, num_frames);

        std::vector<float> latent(latent_total);
        auto t_frame_start = std::chrono::steady_clock::now();

        // Denoise
        pipeline.denoise(latent.data(), text_emb.data(), text_len, latent_total,
            [&](const float* noise, float t, const float* text, int tlen, float* pred) {
                // DiT forward pass
                dit.forward(noise, t, text, tlen, pred, cfg.num_patches);
            });

        auto t_frame_end = std::chrono::steady_clock::now();
        float frame_ms = std::chrono::duration<float, std::milli>(t_frame_end - t_frame_start).count();
        fprintf(stderr, "  [Frame %d] %.0f ms\n", f + 1, frame_ms);

        // Copy to all_frames (reshape from patches to [C, H, W])
        // Inverse of patch embedding: patches → spatial
        int lh = cfg.latent_h, lw = cfg.latent_w;
        for (int c = 0; c < cfg.C; c++) {
            for (int ph = 0; ph < lh; ph++) {
                for (int pw = 0; pw < lw; pw++) {
                    float val = 0.0f;
                    // Simple average over patch area
                    for (int dy = 0; dy < cfg.PS; dy++) {
                        for (int dx = 0; dx < cfg.PS; dx++) {
                            int patch_idx = ph * lw + pw;
                            int patch_chan = c * cfg.PS * cfg.PS + dy * cfg.PS + dx;
                            val += latent[patch_idx * patch_dim + patch_chan];
                        }
                    }
                    val /= (cfg.PS * cfg.PS);
                    int frame_chan = c;
                    all_frames[((f * cfg.C + frame_chan) * lh + ph) * lw + pw] = val;
                }
            }
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    fprintf(stderr,
        "\n=== Results ===\n"
        "  Total:   %.0f ms\n"
        "  Per frame: %.0f ms\n"
        "  FPS:     %.2f\n",
        total_ms, total_ms / num_frames, 1000.0f * num_frames / total_ms);

    if (!benchmark_mode) {
        // --- Decode and write output ---
        VideoDecoder decoder(cfg.C, height, width);
        decoder.decode_frames(all_frames.data(), num_frames, output_path.c_str());
    }

    fprintf(stderr, "\n✓ Done — %d frames generated\n", num_frames);
    return 0;
}
