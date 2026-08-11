// video-lora CLI — pure C++ Vulkan inference backend smoke test + runner.
//
// Usage:
//   video-lora-vk --selftest              run all ops vs CPU reference (CI)
//   video-lora-vk --device N              pick GPU device N
//   video-lora-vk --shaders DIR           .spv directory (default: ./shaders)
//
// The selftest is the verification gate: every Vulkan dispatch is checked
// against a naive CPU implementation with a 1e-3 tolerance, so a regression
// in either the shaders or the C++ plumbing fails loudly.

#include "vl_engine.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace video_lora;

// ── CPU references ──────────────────────────────────────────────────────────

static void cpu_conv2d(const Tensor& in, const Tensor& w, const Tensor& b,
                       Tensor& out, uint32_t pad) {
    uint32_t ic = in.c, oc = w.c, H = in.h, W = in.w, K = 3;
    out = Tensor{1, oc, H, W, std::vector<float>(size_t(oc) * H * W, 0.0f)};
    for (uint32_t oci = 0; oci < oc; oci++)
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++) {
                float sum = b.data[oci];
                for (uint32_t ici = 0; ici < ic; ici++)
                    for (uint32_t ky = 0; ky < K; ky++)
                        for (uint32_t kx = 0; kx < K; kx++) {
                            int iy = (int)y + (int)ky - (int)(K / 2);
                            int ix = (int)x + (int)kx - (int)(K / 2);
                            if (iy < 0 || iy >= (int)H || ix < 0 || ix >= (int)W) continue;
                            uint32_t widx = (oci * ic + ici) * K * K + ky * K + kx;
                            sum += in.data[(ici * H + iy) * W + ix] * w.data[widx];
                        }
                out.data[(oci * H + y) * W + x] = sum;
            }
}

static void cpu_group_norm(Tensor& t, uint32_t G, float eps, float gamma,
                           float beta) {
    uint32_t C = t.c, H = t.h, W = t.w, gc = C / G;
    size_t gsz = size_t(gc) * H * W;
    for (uint32_t n = 0; n < t.n; n++)
        for (uint32_t g = 0; g < G; g++) {
            float mean = 0, var = 0;
            for (size_t i = 0; i < gsz; i++)
                mean += t.data[(size_t(n) * C + (size_t)g * gc) * H * W + i];
            mean /= (float)gsz;
            for (size_t i = 0; i < gsz; i++) {
                float d = t.data[(size_t(n) * C + (size_t)g * gc) * H * W + i] - mean;
                var += d * d;
            }
            var /= (float)gsz;
            float inv = 1.0f / std::sqrt(var + eps);
            for (size_t i = 0; i < gsz; i++) {
                size_t idx = (size_t(n) * C + (size_t)g * gc) * H * W + i;
                uint32_t c = g * gc + (uint32_t)(i / (H * W)) % gc;
                (void)c;  // shader applies scalar gamma/beta for now
                t.data[idx] = (t.data[idx] - mean) * inv * gamma + beta;
            }
        }
}

static void cpu_silu(Tensor& t) {
    for (auto& x : t.data) x = x / (1.0f + std::exp(-x));
}

static void cpu_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                          Tensor& out, uint32_t heads) {
    uint32_t N = q.h, dim = q.c, hd = dim / heads;
    out = Tensor{1, dim, N, 1, std::vector<float>(size_t(dim) * N, 0.0f)};
    float scale = 1.0f / std::sqrt((double)hd);
    for (uint32_t pos = 0; pos < N; pos++)
        for (uint32_t h = 0; h < heads; h++) {
            std::vector<float> scores(N);
            for (uint32_t t = 0; t < N; t++) {
                float s = 0;
                for (uint32_t i = 0; i < hd; i++)
                    s += q.data[(pos * heads + h) * hd + i] *
                         k.data[(t * heads + h) * hd + i];
                scores[t] = s * scale;
            }
            float mx = scores[0];
            for (auto s : scores) mx = std::max(mx, s);
            float sum = 0;
            for (auto& s : scores) { s = std::exp(s - mx); sum += s; }
            for (uint32_t t = 0; t < N; t++)
                for (uint32_t i = 0; i < hd; i++)
                    out.data[(pos * heads + h) * hd + i] +=
                        (scores[t] / sum) * v.data[(t * heads + h) * hd + i];
        }
}

static void cpu_lora_merge(Tensor& base, const Tensor& a, const Tensor& b,
                           float alpha) {
    uint32_t in_dim = a.w, rank = a.h, out_dim = b.h;
    for (uint32_t i = 0; i < out_dim; i++)
        for (uint32_t j = 0; j < in_dim; j++) {
            float d = 0;
            for (uint32_t r = 0; r < rank; r++)
                d += b.data[i * rank + r] * a.data[r * in_dim + j];
            base.data[size_t(i) * in_dim + j] += d * alpha;
        }
}

// ── comparison helper ───────────────────────────────────────────────────────

static bool close(const Tensor& a, const Tensor& b, float tol = 1e-3f) {
    if (a.numel() != b.numel()) {
        std::printf("  FAIL: size %zu != %zu\n", a.numel(), b.numel());
        return false;
    }
    for (size_t i = 0; i < a.numel(); i++) {
        float diff = std::fabs(a.data[i] - b.data[i]);
        float scale = std::max(1.0f, std::fabs(b.data[i]));
        if (diff / scale > tol) {
            std::printf("  FAIL: [%zu] gpu=%f cpu=%f\n", i, a.data[i], b.data[i]);
            return false;
        }
    }
    return true;
}

static Tensor rand_tensor(uint32_t n, uint32_t c, uint32_t h, uint32_t w,
                          float scale = 1.0f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-scale, scale);
    Tensor t{n, c, h, w, std::vector<float>(size_t(n) * c * h * w)};
    for (auto& x : t.data) x = d(rng);
    return t;
}

// ── selftest ────────────────────────────────────────────────────────────────

static int selftest(VlEngine& eng) {
    int fails = 0;
    auto check = [&](const char* name, bool ok) {
        std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
        if (!ok) fails++;
    };

    // conv2d: 2ch in → 4ch out, 8×8
    {
        Tensor in = rand_tensor(1, 2, 8, 8);
        Tensor w = rand_tensor(1, 4, 2, 9, 0.2f);   // [oc][ic][3][3]
        Tensor b = rand_tensor(1, 1, 4, 1, 0.1f);
        Tensor gpu, cpu;
        cpu_conv2d(in, w, b, cpu, 1);
        check("conv2d", eng.conv2d(in, w, b, gpu, 1) && close(gpu, cpu));
    }

    // group_norm: 4ch, 2 groups, 4×4
    {
        Tensor gpu = rand_tensor(1, 4, 4, 4, 3.0f);
        Tensor cpu = gpu;
        cpu_group_norm(cpu, 2, 1e-5f, 1.0f, 0.0f);
        check("group_norm",
              eng.group_norm(gpu, 2, 1e-5f, 1.0f, 0.0f) && close(gpu, cpu, 5e-3f));
    }

    // silu
    {
        Tensor gpu = rand_tensor(1, 1, 16, 16, 4.0f);
        Tensor cpu = gpu;
        cpu_silu(cpu);
        check("silu", eng.silu(gpu) && close(gpu, cpu));
    }

    // elementwise add + scale
    {
        Tensor gpu = rand_tensor(1, 1, 8, 8);
        Tensor other = rand_tensor(1, 1, 8, 8);
        Tensor cpu = gpu;
        for (size_t i = 0; i < cpu.numel(); i++) cpu.data[i] += other.data[i];
        check("elementwise_add", eng.elementwise(gpu, other, 0) && close(gpu, cpu));
    }

    // attention: 4 heads × 64 head_dim (shader contract: head_dim == 64),
    // 16 positions
    {
        uint32_t heads = 4, hd = 64, N = 16;
        Tensor q = rand_tensor(1, heads * hd, N, 1);
        Tensor k = rand_tensor(1, heads * hd, N, 1);
        Tensor v = rand_tensor(1, heads * hd, N, 1);
        Tensor gpu, cpu;
        cpu_attention(q, k, v, cpu, heads);
        check("attention",
              eng.attention(q, k, v, gpu, heads) && close(gpu, cpu, 5e-3f));
    }

    // lora_merge: out_dim=16, in_dim=32, rank=4 (weights as h=rows, w=cols)
    {
        uint32_t in_dim = 32, out_dim = 16, rank = 4;
        Tensor base = rand_tensor(1, 1, out_dim, in_dim, 0.5f);
        Tensor a = rand_tensor(1, 1, rank, in_dim, 0.1f);    // [rank][in_dim]
        Tensor b = rand_tensor(1, 1, out_dim, rank, 0.1f);   // [out_dim][rank]
        Tensor cpu = base;
        cpu_lora_merge(cpu, a, b, 0.5f);
        check("lora_merge", eng.lora_merge(base, a, b, 0.5f) && close(base, cpu));
    }

    std::printf("selftest: %s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILED",
                fails);
    return fails == 0 ? 0 : 1;
}

// ── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string shader_dir = "shaders";
    int device = -1;
    bool run_selftest = false;

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--selftest")) run_selftest = true;
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc)
            device = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--shaders") && i + 1 < argc)
            shader_dir = argv[++i];
    }

    VlEngine eng;
    if (!eng.init(shader_dir, device)) {
        std::fprintf(stderr, "video-lora: engine init failed\n");
        return 1;
    }

    if (run_selftest) return selftest(eng);

    std::printf("video-lora Vulkan backend ready on %s\n",
                eng.device_name().c_str());
    std::printf("run --selftest to verify all ops against CPU reference\n");
    return 0;
}
