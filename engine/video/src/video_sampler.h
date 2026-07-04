// video_sampler.h — Diffusion samplers: DDIM and Flow Matching.
// Implements the denoising loop that converts noise to video latents.

#pragma once
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include "video_model.h"

// --- Noise scheduler base ---
struct NoiseSchedule {
    std::vector<float> alphas_cumprod; // ᾱ(t) for DDIM
    std::vector<float> sigmas;          // σ(t) for Flow Matching
    int num_steps = 50;

    virtual void prepare(int steps) = 0;
    virtual ~NoiseSchedule() = default;
};

// --- DDIM scheduler ---
// Standard DDIM: x_{t-1} = √ᾱ_{t-1} * x₀_pred + √(1-ᾱ_{t-1}) * ε_pred
struct DDIMSchedule : NoiseSchedule {
    float beta_start = 0.00085f;
    float beta_end = 0.012f;

    void prepare(int steps) override {
        num_steps = steps;
        alphas_cumprod.resize(steps + 1);
        for (int i = 0; i <= steps; i++) {
            float t = (float)i / steps;
            float beta = beta_start + t * (beta_end - beta_start);
            float alpha = 1.0f - beta;
            alphas_cumprod[i] = (i == 0) ? 1.0f : alphas_cumprod[i-1] * alpha;
        }
    }

    // Get predicted x₀ from noise prediction
    void predict_x0(const float* latent_t, const float* noise_pred,
                    float* x0_pred, int N, int t) {
        float sqrt_ac = sqrtf(alphas_cumprod[t]);
        float sqrt_one_minus_ac = sqrtf(1.0f - alphas_cumprod[t]);
        for (int i = 0; i < N; i++) {
            x0_pred[i] = (latent_t[i] - sqrt_one_minus_ac * noise_pred[i]) / sqrt_ac;
        }
    }

    // Step: get x_{t-1} from x_t and predicted x₀
    void step(const float* x_t, const float* x0_pred, float* x_t_minus_1,
              int N, int t) {
        float ac_t = alphas_cumprod[t];
        float ac_tm1 = (t > 0) ? alphas_cumprod[t-1] : 1.0f;
        float sqrt_ac_tm1 = sqrtf(ac_tm1);
        float sqrt_one_minus_ac_t = sqrtf(1.0f - ac_t);
        
        for (int i = 0; i < N; i++) {
            x_t_minus_1[i] = sqrt_ac_tm1 * x0_pred[i] +
                             sqrtf(1.0f - ac_tm1 - sqrt_one_minus_ac_t * sqrt_one_minus_ac_t) *
                             (x_t[i] - sqrt_ac_tm1 * x0_pred[i]) / sqrt_one_minus_ac_t;
        }
    }
};

// --- Flow Matching scheduler ---
// RF (Rectified Flow): dx/dt = v_θ(x_t, t)
// Euler step: x_{t-1} = x_t + (t_{t-1} - t_t) * v_θ(x_t, t)
struct FlowMatchSchedule : NoiseSchedule {
    void prepare(int steps) override {
        num_steps = steps;
        sigmas.resize(steps + 1);
        for (int i = 0; i <= steps; i++) {
            sigmas[i] = 1.0f - (float)i / steps;
        }
    }

    // Euler step: x_{t-1} = x_t + Δt * velocity
    void step(const float* x_t, const float* velocity, float* x_t_minus_1,
              int N, int t) {
        float dt = 1.0f / num_steps;
        for (int i = 0; i < N; i++) {
            x_t_minus_1[i] = x_t[i] + dt * velocity[i];
        }
    }
};

// --- Denoising pipeline ---
// Orchestrates the full denoising loop:
//   1. Sample random noise
//   2. For t = T-1 ... 0:
//      a. Run model (DiT) to get noise/velocity prediction
//      b. Step with scheduler
//   3. Return denoised latent

struct DenoisingPipeline {
    VideoModelConfig cfg;
    NoiseSchedule* scheduler = nullptr;
    bool is_flow_match = true;

    DenoisingPipeline(const VideoModelConfig& config) : cfg(config) {}

    // Initialize noise (gaussian latents for all frames)
    void init_noise(float* noise, int N) {
        // Simple rand-based gaussian (replace with proper RNG for production)
        for (int i = 0; i < N; i++) {
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            noise[i] = sqrtf(-2.0f * logf(u1 + 1e-8f)) * cosf(2.0f * M_PI * u2);
        }
    }

    // Add noise to latent at timestep t (for conditioning)
    void add_noise(const float* clean, float* noisy, int N, int t, const float* noise) {
        if (is_flow_match) {
            auto* fm = static_cast<FlowMatchSchedule*>(scheduler);
            float alpha = 1.0f - fm->sigmas[t];
            for (int i = 0; i < N; i++) {
                noisy[i] = alpha * clean[i] + (1.0f - alpha) * noise[i];
            }
        } else {
            auto* ddim = static_cast<DDIMSchedule*>(scheduler);
            float sqrt_ac = sqrtf(ddim->alphas_cumprod[t]);
            float sqrt_one_minus_ac = sqrtf(1.0f - ddim->alphas_cumprod[t]);
            for (int i = 0; i < N; i++) {
                noisy[i] = sqrt_ac * clean[i] + sqrt_one_minus_ac * noise[i];
            }
        }
    }

    // Full denoising loop.
    // model_predict: callback(latent, timestep, text_emb) -> prediction
    // Returns denoised latent in 'latent' buffer.
    template<typename F>
    void denoise(float* latent, const float* text_emb, int text_len,
                 int latent_N, F model_predict) {
        if (!scheduler) {
            fprintf(stderr, "[Denoise] No scheduler set\n");
            return;
        }

        // Initial noise
        std::vector<float> noise(latent_N);
        init_noise(noise.data(), latent_N);

        // Copy initial noise to latent
        memcpy(latent, noise.data(), latent_N * sizeof(float));

        // CFG: we need cond and uncond predictions
        std::vector<float> pred_cond(latent_N);
        std::vector<float> pred_uncond(latent_N);

        // Denoising loop
        for (int t = scheduler->num_steps - 1; t >= 0; t--) {
            float t_float = (float)t / scheduler->num_steps;
            fprintf(stderr, "\r  [Denoise] step %3d/%d (predicting...)", scheduler->num_steps - t, scheduler->num_steps);
            fflush(stderr);

            // Conditional prediction
            model_predict(latent, t_float, text_emb, text_len, pred_cond.data());

            // Unconditional prediction (empty prompt)
            // In production, use a learned null embedding
            model_predict(latent, t_float, nullptr, 0, pred_uncond.data());

            // Classifier-free guidance
            #pragma omp parallel for
            for (int i = 0; i < latent_N; i++) {
                pred_cond[i] = pred_uncond[i] + cfg.cfg_scale * (pred_cond[i] - pred_uncond[i]);
            }

            // Scheduler step
            if (is_flow_match) {
                auto* fm = static_cast<FlowMatchSchedule*>(scheduler);
                fm->step(latent, pred_cond.data(), latent, latent_N, t);
            } else {
                auto* ddim = static_cast<DDIMSchedule*>(scheduler);
                std::vector<float> x0_pred(latent_N);
                ddim->predict_x0(latent, pred_cond.data(), x0_pred.data(), latent_N, t);
                ddim->step(latent, x0_pred.data(), latent, latent_N, t);
            }

            if (t % 10 == 0) {
                fprintf(stderr, "\r  [Denoise] step %3d/%d", scheduler->num_steps - t, scheduler->num_steps);
            }
        }
        fprintf(stderr, "\r  [Denoise] step %3d/%d ✓\n", scheduler->num_steps, scheduler->num_steps);
    }
};
