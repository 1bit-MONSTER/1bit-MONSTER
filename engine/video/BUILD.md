# Video Diffusion Engine — Build Guide

## Zero Python. Pure C++23. NPU + CPU.

### CPU-Only Build (for development/testing)

```bash
cd engine/video

g++ -std=c++23 -O3 -march=native -fopenmp -o video_engine \
    src/video_main.cpp -lm

# Run
./video_engine --prompt "a cat walking, cinematic" --frames 8 --steps 20
```

### NPU-Accelerated Build (Strix Halo XDNA 2)

```bash
cd engine/video

g++ -std=c++23 -O3 -march=native -fopenmp -DUSE_NPU -o video_engine \
    src/video_main.cpp \
    ../npu/src/npu_engine_universal.cpp \
    -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl

# Run with NPU
./video_engine --prompt "cinematic dolly zoom" --model weights.bin --frames 16
```

### Benchmark Mode

```bash
./video_engine --prompt "test" --frames 8 --steps 10 --benchmark
```

## Architecture

```
engine/video/
├── src/
│   ├── video_main.cpp      # Entry point + CLI
│   ├── video_model.h       # Model config + weight loading
│   ├── video_dit.h         # Diffusion Transformer forward pass
│   ├── video_sampler.h     # DDIM + Flow Matching denoising
│   └── video_vae.h         # VAE decoder (latent → pixels)
├── docs/
│   └── BENCHMARKS.md       # Performance benchmarks
└── README.md
```

## Model Weights

Weights in `.bin` format with `VideoWeightHeader`:
```
[8 bytes magic 'VIDWEIGH']
[4 bytes version]
[4 bytes num_layers]
[4 bytes hidden_size]
[4 bytes num_heads]
[4 bytes head_dim]
[4 bytes mlp_hidden]
[4 bytes text_hidden]
[8 bytes weight_offset]
[...float weight data...]
```

Convert from HuggingFace Diffusers:
```bash
python3 tools/convert_weights.py --model wan/Wan2.1-T2V-1.3B --output weights.bin
```

## Reused Infrastructure

The video engine reuses the NPU engine's INT8 GEMM primitives:

| Component | Source | Purpose |
|-----------|--------|---------|
| `I8Ctx` | `engine/npu/src/npu_engine_universal.cpp` | INT8 GEMM on XDNA 2 |
| `attn_omp()` | `engine/npu/src/npu_engine_universal.cpp` | OpenMP attention |
| `dynamic_ascale()` | `engine/npu/src/npu_engine_universal.cpp` | Activation quant scale |
| `platform.h` | `engine/npu/src/platform.h` | Platform abstractions |
| xclbin infra | `engine/npu/xclbins/` | MLIR-compiled NPU kernels |
