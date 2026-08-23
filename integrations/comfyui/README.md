# 1bit-monster ComfyUI Integration

Custom nodes that expose the 1bit inference engine to ComfyUI workflows,
giving you NPU-accelerated LLM inference, image generation, vision-language
understanding, and audio synthesis — all from ComfyUI's visual graph editor.

## Installation

```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/1bit-MONSTER/1bit-MONSTER comfyui_1bit_systems
pip install httpx Pillow numpy

# Also install submodule dependencies
cd comfyui_1bit_systems
git submodule update --init --recursive
```

## Prerequisite: Start the 1bit Servers

Before using any node, start the backend servers (in separate terminals or
as systemd services):

```bash
# Start the LLM + VLM inference server
cd /path/to/1bit-monster
cmake -B build -G Ninja && cmake --build build --target unified_server -j$(nproc)
./build/unified_server -w /path/to/models/ -p 8088

# Start the image generation server (requires stable-diffusion.cpp)
# cmake -B build -DUSE_DIFFUSION=ON
# ./build/image_server -p 8089

# Start the audio + Jarvis server (ONNX Runtime optional for codec TTS)
# cmake -B build -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime  # optional, for voice cloning
# ./build/1bit jarvis --port 8090
```

## Available Nodes

### 1BP LLM Generate
Text generation using any 1BP model via the unified_server.
- Input: prompt text, system prompt, model name
- Parameters: max_tokens, temperature
- Output: generated text string

### 1BP VLM Analyze Image
Vision-language understanding. Send an image + question, get a description.
- Input: image from any ComfyUI image node, text prompt
- Parameters: model name (e.g. `zaya1-vl-8b`, `qwen2-vl-2b`)
- Output: text description

### 1BP Image Generate
Text-to-image generation via stable-diffusion.cpp backend.
- Input: text prompt, negative prompt; optional IMAGE for image-to-image
- Parameters: width, height, steps, CFG scale, seed, strength
- Supports: SD, SDXL, FLUX, Qwen-Image, Z-Image, and more
- Optional: LoRA path + strength
- Output: image tensor (compatible with ComfyUI image pipeline)

### 1BP Video Generate
Text-to-video / image-to-video generation (Wan, LTX, Hunyuan) via the stable-diffusion.cpp backend.
- Input: text prompt, negative prompt; optional first-frame IMAGE (image-to-video)
- Parameters: width, height, frames, fps, steps, CFG scale, seed, strength
- Optional: LoRA path + strength, output format (webm/avi)
- Output: video file saved to ComfyUI's output directory
- Container: WebM by default (in-browser preview); `avi` selects MJPG AVI

### 1BP Text-to-Speech
Synthesize speech from text.
- Input: text, voice name
- Output: audio waveform

### 1BP LoRA Loader
Hot-load a LoRA adapter into the inference engine.
- Input: LoRA GGUF path, target model name
- Output: status string

## Architecture

```
ComfyUI (Python nodes)
    │
    ├── HTTP POST /v1/chat/completions  ──►  unified_server (C++, port 8088)
    ├── HTTP POST /v1/images/generations ──►  image_server   (C++, port 8089)
    └── HTTP POST /v1/audio/speech      ──►  1bit jarvis  (C++, port 8090)
```

All heavy computation happens in the C++ backends (HIP CUDA Vulkan NPU CPU).
ComfyUI is just a thin UI layer — no Python inference overhead.

## NPU Acceleration

For NPU users (AMD Strix Halo):
- LLM inference automatically routes to NPU when available
- VLM image encoding uses NPU for image preprocessing
- Image generation uses GPU (diffusion is GPU-heavy)

## LoRA Workflow

```
[1BP LoRA Loader] ─► [1BP LLM Generate]
    │                          │
    │ loads adapter            │ uses adapted weights
    │ into server              │ for generation
    └──────────────────────────┘
```

Load a LoRA at the start of your workflow, then any LLM generation nodes
will use the adapted model. LoRA can be hot-swapped between generations
without restarting the server.
