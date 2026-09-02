# Unsloth Dynamic (UD) GGUFs

Unsloth Dynamic (UD) is Unsloth's selective-layer quantization format for GGUF models.
Important layers are upcast to 8/16-bit while less critical layers are aggressively quantized,
enabling massive models (671B+) to run on consumer hardware.

## Variants

| Variant | Bit-width | Description | Use Case |
|---|---|---|---|
| `UD-TQ1_0` | ~1-bit | Dynamic 1-bit, extreme compression | Minimal RAM (e.g., DeepSeek-V3.1 671B → ~192 GB) |
| `UD-IQ2_M` | ~2-bit | Importance-weighted 2-bit; slower conversion, may have better accuracy | Accuracy-maximizing 2-bit |
| `UD-Q2_K_XL` | ~2.7-bit | **Recommended** balance of size vs. accuracy; faster conversion than IQ2_M | General purpose (e.g., GLM-5.1 → ~220 GB) |
| `UD-Q3_K_XL` | ~3-bit | Dynamic 3-bit | Middle ground |
| `UD-Q4_K_XL` | ~4-bit | Dynamic 4-bit, highest quality | When RAM is plentiful |

## Download

```bash
# Dynamic 2-bit (recommended)
hf download unsloth/GLM-5.1-GGUF \
    --local-dir unsloth/GLM-5.1-GGUF \
    --include "*UD-Q2_K_XL*"

# Dynamic 1-bit
hf download unsloth/GLM-5.1-GGUF \
    --local-dir unsloth/GLM-5.1-GGUF \
    --include "*UD-TQ1_0*"
```

Python:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="unsloth/GLM-4.7-GGUF",
    local_dir="unsloth/GLM-4.7-GGUF",
    allow_patterns=["*UD-Q2_K_XL*"],
)
```

## Compatibility with 1bit.MONSTER

UD GGUFs are standard GGUF files, so the architecture is auto-detected from GGUF
metadata. But the quant *dtype* support is backend-specific — do not assume a
single UD file loads everywhere:

| Quant dtype | Project reader (`cpu_generic`, `gguf_to_onebp`) | llama.cpp path (`ggml_vulkan`, HRX) |
|---|---|---|
| `F16/BF16`, `Q4_0..Q8_K`, `Q2_K..Q8_K` | Supported | Supported |
| `UD-TQ1_0` / `TQ2_0` (~1-bit ternary) | **Supported** | Supported |
| `IQ` family (`IQ1_S/M`, `IQ2_XS/S`, `IQ3_S`, `IQ4_NL/XS`) | Not decoded | Supported |
| `UD-Q*_K_XL`, `IQ2_M` | Not understood | Not in the vendored llama.cpp snapshot |

Notes:
- The project's own reader (`src/gguf_reader.cpp`) decodes the block quants and
  ternary `TQ1_0`/`TQ2_0`. For any other dtype it returns a clear error (it
  never silently reads raw f32), so an unsupported model fails loudly at load.
- Route `IQ`-family files to the llama.cpp backend (`ggml_vulkan`, or the HRX
  bundle), which uses self-consistent current ggml dtype numbering. The router
  already prefers `hrx_gpu → ggml_vulkan → zinc → cpu_generic` for GGUFs.
- `UD-Q*_K_XL` and `IQ2_M` are not present in the vendored llama.cpp enum at
  all. Supporting them requires a newer llama.cpp snapshot (or a project-reader
  port); until then such files are refused with a clear error.


## Where to Find Them

All UD quantized models live on Hugging Face under the `unsloth/` org, in repos
named `*GGUF`. Filename pattern: `*UD-<variant>*`:

- `unsloth/GLM-5.1-GGUF` — `*UD-Q2_K_XL*`, `*UD-TQ1_0*`, `*UD-IQ2_M*`
- `unsloth/GLM-4.7-GGUF` — `*UD-Q2_K_XL*`, `*UD-TQ1_0*`
- `unsloth/gemma-4-26B-A4B-it-GGUF` — `*UD-Q4_K_XL*`
- `unsloth/DeepSeek-V3.1-GGUF` — `*UD-Q2_K_XL*`, `*UD-TQ1_0*`

## Running with llama.cpp

```bash
./llama-cli -m unsloth/.../UD-Q2_K_XL.gguf ...
```

## Reference

- [Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
