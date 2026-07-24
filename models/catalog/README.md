# 1bit.systems Model Catalog — 24 Models (1BP)

All models available in **1BP format** — single-file, zero-config, memory-mappable.
Converted via C++ toolchain (`tools/gguf_to_onebp.cpp`), zero Python at runtime.

## Model Families

### Dense Transformer — 6
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Qwen3-0.6B | 0.6B | 356 MB | ZINC / NPU / HIP | qwen3 |
| Qwen3-4B | 4B | 2.2 GB | ZINC / NPU / HIP | qwen3 |
| Qwen3-8B | 8B | 4.1 GB | ZINC / NPU / HIP | qwen3 |
| Qwen2.5-0.5B | 0.5B | 328 MB | ZINC / NPU | qwen2 |
| TinyLlama-1.1B | 1.1B | 328 MB | ZINC / NPU | qwen2 (compat) |
| ZR1-1.5B | 1.5B | 781 MB | ZINC / NPU | qwen2 |

### Laguna (MoE, poolside) — 3
| Model | Params | 1BP Size | 1BP TQ2 Size | Backend | Architecture |
|-------|:------:|:--------:|:------------:|---------|:------------:|
| Laguna-S-2.1 | 48×256ex | 73.5 GB | 36.7 GB | ZINC / NPU / HIP | laguna (MoE) |
| Laguna-XS-2.1 | 40×256ex | 20.9 GB | 10.5 GB | ZINC / NPU / HIP | laguna (MoE) |
| Laguna-S-2.1-DFlash (draft) | 6L dense | 665 MB | 665 MB | ZINC / NPU / HIP | dflash |

### MoE — 1
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| ZAYA1-8B | 8.8B | 149 MB | ZINC | zaya ✅ |

### Mamba1 — 2
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| BlackMamba-1.5B | 1.5B | 970 MB | Mamba1 HIP (79.8 tok/s) | mamba |
| BlackMamba-2.8B | 2.8B | 1.8 GB | Mamba1 HIP (46.4 tok/s) | mamba |

### Mamba2-Hybrid — 3
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Zamba2-1.2B | 1.2B | 1.1 GB | ZINC / NPU | zamba2 |
| Zamba2-2.7B | 2.7B | 2.4 GB | ZINC / NPU | zamba2 |
| Zamba2-7B | 7B | 6.6 GB | ZINC / NPU | zamba2 |

### Mamba1 + Shared Attention — 1
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Zamba-7B-v1 | 7B | 4.3 GB | Mamba1 HIP | zamba |

### Ternary / 1-bit — 7
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Bonsai-1.7B-Q1 | 1.7B | 841 MB | HIP GPU | qwen3 (ternary) |
| Bonsai-4B-Q1 | 4B | 2.2 GB | HIP GPU | qwen3 (ternary) |
| Bonsai-8B-Q1 | 8B | 4.1 GB | HIP GPU | qwen3 (ternary) |
| Bonsai-27B-Q1 | 27B | 15 GB | HIP GPU | qwen3 (ternary) |
| Ternary-Bonsai-1.7B-Q2 | 1.7B | 841 MB | HIP GPU | qwen3 (ternary) |
| Ternary-Bonsai-4B-Q2 | 4B | 2.2 GB | HIP GPU | qwen3 (ternary) |
| Ternary-Bonsai-8B-Q2 | 8B | 4.1 GB | HIP GPU | qwen3 (ternary) |

### Vision-Language — 1
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Qwen2-VL-2B | 2B | 781 MB | ZINC (vision) | qwen2vl ✅ |

## Total: 24 models
All converted via C++ toolchain (`tools/gguf_to_onebp`).

## Conversion Pipeline (C++ only)
```bash
# Build converter
g++ -std=c++17 -O3 -mavx2 -I include -I src \
    tools/gguf_to_onebp.cpp src/gguf_reader.cpp src/gguf_zamba2_loader.cpp \
    -o build/gguf_to_onebp -lpthread

# Convert model
./build/gguf_to_onebp input.gguf output.1bp
```

## Adding a New Model
1. Get GGUF format model file
2. Convert: `./build/gguf_to_onebp model.gguf models/ModelName.1bp`
3. If new architecture: add to `include/rocm_cpp/bitnet_model.h` in `rcpp_arch_from_string()`
4. If new architecture: add to `include/onebp_format.h` in `OnebpArch` enum
5. Rebuild: `cmake --build build_cmake --target zaya_server`
6. Test: `./build/zaya_server --model models/ModelName.1bp --port 8088`
