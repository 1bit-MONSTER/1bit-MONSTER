// video_vae.h — Simplified VAE decoder for converting latents to frames.
// Wan2.2 uses a 16-channel VAE with 8× spatial compression.
// This is a CPU-based fallback — NPU-accelerated VAE is future work.

#pragma once
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <omp.h>

// --- Simple pixel output (bypass VAE for now) ---
// Given denoised latents [C, H, W], produce a viewable image
// by taking the first 3 channels and normalizing to [0,255].

struct VideoDecoder {
    int channels, height, width;
    std::vector<uint8_t> output; // RGB output

    VideoDecoder(int c, int h, int w) : channels(c), height(h), width(w) {
        output.resize(h * w * 3);
    }

    // Simple decode: take first 3 channels, normalize, write as PPM
    void decode(const float* latents, const char* filename, int frame_idx = 0) {
        int latent_h = height / 8;
        int latent_w = width / 8;
        int C = channels;

        // Simple upscale + channel select
        // For production: use actual VAE decoder (convolutional network)
        // For now: basic nearest-neighbor upscale of first 3 channels
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int ly = y / 8;
                int lx = x / 8;
                if (ly >= latent_h) ly = latent_h - 1;
                if (lx >= latent_w) lx = latent_w - 1;

                for (int c = 0; c < 3 && c < C; c++) {
                    float val = latents[(c * latent_h + ly) * latent_w + lx];
                    // Normalize to [0,1] then [0,255]
                    val = (val + 1.0f) * 0.5f;
                    if (val < 0) val = 0;
                    if (val > 1) val = 1;
                    output[(y * width + x) * 3 + c] = (uint8_t)(val * 255.0f);
                }
            }
        }

        // Write PPM
        char fname[256];
        if (frame_idx >= 0)
            snprintf(fname, sizeof(fname), "%s_frame_%04d.ppm", filename, frame_idx);
        else
            snprintf(fname, sizeof(fname), "%s.ppm", filename);

        FILE* f = fopen(fname, "wb");
        if (!f) {
            fprintf(stderr, "[VAE] Cannot write %s\n", fname);
            return;
        }
        fprintf(f, "P6\n%d %d\n255\n", width, height);
        fwrite(output.data(), 1, height * width * 3, f);
        fclose(f);
        fprintf(stderr, "[VAE] Wrote %s (%dx%d)\n", fname, width, height);
    }

    // Decode all frames and write as MP4 via pipe to ffmpeg
    void decode_frames(const float* all_latents, int num_frames,
                        const char* output_path) {
        for (int f = 0; f < num_frames; f++) {
            const float* frame_data = &all_latents[f * channels * (height/8) * (width/8)];
            decode(frame_data, output_path, f);
        }

        // Optionally combine with ffmpeg if available
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "ffmpeg -y -framerate 8 -i '%s_frame_%%04d.ppm' "
                 "-c:v libx264 -pix_fmt yuv420p '%s.mp4' 2>/dev/null",
                 output_path, output_path);
        int ret = system(cmd);
        if (ret == 0) {
            fprintf(stderr, "[VAE] Combined to %s.mp4\n", output_path);
        } else {
            fprintf(stderr, "[VAE] ffmpeg combine skipped (not installed or failed)\n");
        }
    }
};

// --- VAE decoder stub for future implementation ---
// The real VAE decoder is a convolutional network that converts
// 16-channel latents to 3-channel RGB at 8× resolution.
//
// Architecture:
//   Input: [16, H/8, W/8]
//   1. Conv2d 16→128, 3×3, pad=1
//   2. GroupNorm 32, SiLU
//   3. Conv2d 128→128, 3×3, pad=1
//   4. GroupNorm 32, SiLU
//   5. Nearest upsample 2×
//   6. Conv2d 128→64, 3×3, pad=1
//   7. GroupNorm 32, SiLU
//   8. Nearest upsample 2×
//   9. Conv2d 64→32, 3×3, pad=1
//   10. GroupNorm 32, SiLU
//   11. Nearest upsample 2×
//   12. Conv2d 32→3, 3×3, pad=1
//   Output: [3, H, W]
//
// This requires a full conv2d implementation on NPU or CPU.
// For now, the simple decode above demonstrates the pipeline.

struct FullVAEDecoder {
    // TODO: implement when conv2d xclbins are available for NPU
    // For CPU, use the existing OpenCV-style conv2d
};
