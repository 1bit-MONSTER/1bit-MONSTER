# Video Diffusion Engine — Benchmarks

## Performance (Strix Halo, Radeon 8060S)

### CPU Only (OpenMP, 16 threads)

| Model | Resolution | Frames | Steps | Total | Per Frame | Notes |
|-------|-----------|-------|-------|-------|-----------|-------|
| Wan2.2-1.3B DiT | 640×480 | 8 | 10 | TBD | TBD | Dev test |
| Wan2.2-1.3B DiT | 640×480 | 8 | 50 | TBD | TBD | Full quality |
| Wan2.2-1.3B DiT | 640×480 | 16 | 50 | TBD | TBD | Standard video |

### NPU (XDNA 2, INT8)

| Model | Resolution | Frames | Steps | Total | Per Frame | Notes |
|-------|-----------|-------|-------|-------|-----------|-------|
| Wan2.2-1.3B DiT | 640×480 | 8 | 50 | TBD | TBD | INT8 quantized |
| Wan2.2-1.3B DiT | 640×480 | 16 | 50 | TBD | TBD | Production target |

### Memory

| Config | Model RAM | Latent RAM | Total |
|--------|----------|-----------|-------|
| CPU (f32) | ~4.2 GB | ~50 MB | ~4.3 GB |
| NPU (INT8) | ~1.1 GB | ~50 MB | ~1.2 GB |

## Run Your Own

```bash
cd engine/video
g++ -std=c++23 -O3 -march=native -fopenmp -o video_engine src/video_main.cpp -lm
./video_engine --prompt "test" --frames 8 --steps 10 --benchmark
```
