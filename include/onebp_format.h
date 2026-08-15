#ifndef ONEBP_FORMAT_H
#define ONEBP_FORMAT_H

/** 1BP — 1bit-systems binary package format.
 *
 *  Unified model format combining Q4NX NPU-tiled layout with self-contained
 *  metadata. One file, no external config.json needed. Memory-mappable.
 *
 *  Layout on disk:
 *    [Header: 256 bytes]
 *    [Tensor Index: variable]
 *    [Weight Data: tiled arrays]
 *
 *  Tile layout — Q4NX (header.quant == ONEBP_Q4NX), 32×256:
 *    Each tile = 32 rows × 256 cols of quantized data
 *    Per tile row:
 *      [0..511]:   256 BF16 scales (8 groups × 32 rows)
 *      [512..1023]: 256 BF16 zero_points (same layout)
 *      [1024..5119]: 4096 bytes packed INT4 (2 per byte)
 *    Total per tile: 5120 bytes
 *
 *  Tile layout — TQ2 (header.quant == ONEBP_TQ2), 32×256:
 *    Symmetric ternary: every value is exactly -scale, 0, or +scale — no
 *    zero_point needed (unlike Q4NX's asymmetric min/scale). One scale per
 *    32-element group.
 *    Per tile (block layout — scales for all 32 rows first, then codes):
 *      [0..511]:   256 BF16 scales (32 rows x 8 groups; scale[r][g] = r*8+g)
 *      [512..2559]: 2048 bytes packed 2-bit codes (4 per byte, LSB-first:
 *                  code = byte & 3, (byte>>2) & 3, (byte>>4) & 3, (byte>>6) & 3)
 *                  code 0 = -scale, 1 = 0, 2 = +scale, 3 = unused (encoder
 *                  never emits it; a decoder should treat it as 0)
 *    Total per tile: 2560 bytes (exactly half of
 *    Q4NX's 5120 — the real "1-bit-ish" storage win Q4NX doesn't give you).
 *    This is a generic symmetric-ternary quantizer (round to nearest of
 *    {-scale,0,scale} per group) — lossless when the source is already
 *    ternary-valued within each 32-group (as with BitNet/TriLM/Bonsai-style
 *    checkpoints), lossy-but-functional otherwise, same as Q4NX is lossy
 *    for any source that isn't already 4-bit-quantizable.
 *
 *  TQ1 (1.58-bit, base-3 packing): ceil(256/5) = 52 bytes per tile row
 *  for codes. Each byte packs 5 ternary values (3^5 = 243 < 256).
 *  The 256-column tile boundary is handled by padding the last group
 *  of 5 with zero codes.
 *
 *  TQ2NZ (header.quant == ONEBP_TQ2NZ), 32×256 — no-zero 2-bit:
 *  ROCmFPX-FP2-style S40 codebook {-4,-1,+1,+4} using ALL four 2-bit
 *  codes (TQ2 wastes code 3). Same 80-byte tile row as TQ2: [8 BF16
 *  scales][64 B packed codes]. code 0=-4s, 1=-1s, 2=+1s, 3=+4s.
 *  Scale = max|v|/4 per 32-group so the outer code covers the max.
 *  Same 2.50 bpw as TQ2/FP2; all-zero groups use scale=0 (code×0=0).
 *
 *  TQ2NZ_E4M3 (header.quant == ONEBP_TQ2NZ_E4M3), 32×256 — same S40
 *  codebook, but scales are 1-byte unsigned E4M3 (exp==0 → mant·2^-10,
 *  else (8+mant)·2^(exp-11); 127 finite values) instead of BF16 → tile
 *  row = [8 UE4M3 scales][64 B codes] = 72 B → 2.25 bpw. Scale picked by
 *  MSE search over the 127-value table starting at max|v|/4.
 *
 *  Tensor index entry (variable-length):
 *    [name_len:u32][name:str][ndim:u32][dims:u32 × ndim][offset:u64][bytes:u64]
 *
 *    v4 dedup alias: when `bytes` == 0, the entry shares data with an earlier
 *    tensor — `offset` holds that tensor's INDEX in the index (not a byte
 *    offset). dims/quant are still written normally; the loader resolves
 *    location + size from the aliased entry. Aliases always point backward
 *    to a real (bytes > 0) entry.
 *
 *    ndim=2: dims=[rows, cols] — a plain weight matrix. `bytes` is the
 *      tiled size of that one matrix.
 *    ndim=3: dims=[num_experts, rows, cols] — a stack of `num_experts`
 *      independently-tiled matrices (MoE expert weights: ffn_gate_up_exps,
 *      ffn_down_exps, etc.), laid out back-to-back with no padding between
 *      experts. `bytes` = num_experts × tiled_size(rows, cols). Each
 *      expert's slice uses the exact same 32×256 tile scheme as a regular
 *      2D tensor — there's no cross-expert structure to the quantization,
 *      it's just N matrices concatenated.
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

static constexpr uint32_t ONEBP_MAGIC        = 0x00504231;  // "1BP\0"
static constexpr uint32_t ONEBP_VERSION      = 4;  // v2: per-entry quant field (mixed-quant files);
                                                  // v3: rope_theta_f / rope_freq_base_swa_f hold
                                                  // RAW f32 bits (v1/v2: theta*1000 fixed-point,
                                                  // which overflows for theta > 4.29e6 — Granite's
                                                  // rope.freq_base 1e7 wrapped to garbage 1410065408)
                                                  // v4: dedup aliases — an index entry with
                                                  // bytes==0 is an alias whose offset field is the
                                                  // INDEX of an earlier tensor it shares data with

// ─── Quantization types ────────────────────────────────────────────
enum OnebpQuant : uint32_t {
    ONEBP_Q4NX = 0,   // 4-bit block quant, bf16 scales (Q4NX compatible)
    ONEBP_I8   = 1,   // INT8 per-tensor quant, f32 scales
    ONEBP_TQ1  = 2,   // Ternary TQ1 (1.58-bit)
    ONEBP_TQ2  = 3,   // Ternary TQ2
    ONEBP_F16  = 4,   // Float16 (no quant)
    ONEBP_F32  = 5,   // Float32 (no quant)
    ONEBP_TQ2NZ= 6,   // No-zero 2-bit S40 {-4,-1,+1,+4} (ROCmFPX-FP2-style)
    ONEBP_TQ2NZ_E4M3 = 7, // TQ2NZ with 1-byte UE4M3 scales (2.25 bpw)
    ONEBP_TQ2BS = 8,      // Block-scaled ternary, per-16 FP8 E4M3 scales (5 B/block)
};

// ─── Scale types ───────────────────────────────────────────────────
enum OnebpScale : uint32_t {
    ONEBP_SCALE_BF16 = 0,
    ONEBP_SCALE_F32  = 1,
    ONEBP_SCALE_F16  = 2,
};

// ─── Model architecture types ──────────────────────────────────────
enum OnebpArch : uint32_t {
    ONEBP_DENSE  = 0,  // Dense transformer (Qwen3, Llama, Phi, etc.)
    ONEBP_MOE    = 1,  // Mixture of Experts (generic)
    ONEBP_VISION = 2,  // Vision-language
    ONEBP_AUDIO  = 3,  // Audio/speech (Whisper, audio models)
    ONEBP_TERNARY= 4,  // Ternary/1-bit
    ONEBP_MAMBA  = 5,  // Mamba-style SSM
    ONEBP_LAGUNA = 6,  // Poolside Laguna — sigmoid-routed MoE, hybrid SWA/global attn, softplus gate
    
    // ═══ VLM architectures (added in Phase 1 expansion) ═══
    ONEBP_SMOLVLM   = 10, // SmolVLM (tiny VLM, SigLIP + LLM)
    ONEBP_LLAVA     = 11, // LLaVA-style VLM (CLIP + LLM)
    ONEBP_MOLMO     = 12, // Molmo (CLIP + OLMoE)
    ONEBP_OVIS      = 13, // Ovis VLM (SigLIP + Gemma)
    ONEBP_PALIGEMMA = 14, // PaliGemma (SigLIP + Gemma)
    ONEBP_FLORENCE  = 15, // Florence-2 (VL encoder-decoder)
    
    // ═══ Reasoning / MoE architectures ═══
    ONEBP_DEEPSEEK2    = 20, // DeepSeek v2/v3 MLA MoE
    ONEBP_PHI_MOE      = 21, // Phi-4 MoE
    ONEBP_DEEPSEEK_V4  = 22, // DeepSeek V4 Flash/Pro — mHC + CSA+HCA + FP4 MoE (284B/13B active)
    
    // ═══ Diffusion architectures ═══
    ONEBP_SD        = 30, // Stable Diffusion 1.x
    ONEBP_SDXL      = 31, // Stable Diffusion XL
    ONEBP_FLUX      = 32, // FLUX
    ONEBP_WAN       = 33, // Wan video diffusion
    ONEBP_HUNYUAN   = 34, // HunyuanVideo
    ONEBP_LTX       = 35, // LTX-Video
    
    // ═══ Audio / TTS / ASR architectures ═══
    ONEBP_WHISPER   = 40, // OpenAI Whisper (STT)
    ONEBP_AUDIO_CPP = 41, // Generic audio.cpp model

    // ═══ Moonshot Kimi Family ═══
    ONEBP_KIMI_K3   = 50, // Kimi K3 — 2.8T MoE, KDA + Gated MLA + LatentMoE + MXFP4
    ONEBP_MOONLIGHT = 51, // Moonlight-16B-A3B — Gated MLA MoE, base for Kimi-VL
    ONEBP_KIMI_VL   = 52, // Kimi-VL-A3B-Thinking — Moonlight + MoonViT vision encoder
};

// ─── Expert gating function types (Laguna/afmoe) ────────────────
enum OnebpExpertGating : uint32_t {
    ONEBP_EXPERT_GATING_SIGMOID  = 0,
    ONEBP_EXPERT_GATING_SOFTMAX  = 1,
    ONEBP_EXPERT_GATING_NONE     = 2,
};

// ─── Attention gate types (Laguna) ───────────────────────────────
enum OnebpAttnGate : uint32_t {
    ONEBP_ATTN_GATE_PER_HEAD    = 0,  // one scalar per head (XS.2 style)
    ONEBP_ATTN_GATE_PER_ELEMENT = 1,  // full per-element gate (M.1 style)
};

// ─── File header (256 bytes) ──────────────────────────────────────
#pragma pack(push, 1)
struct OnebpHeader {
    uint32_t magic;           // 0x00504231 = "1BP\0"
    uint32_t version;         // format version
    uint32_t arch;            // OnebpArch
    uint32_t quant;           // OnebpQuant
    uint32_t scale_type;      // OnebpScale
    int32_t  hidden_size;
    int32_t  num_layers;
    int32_t  num_attention_heads;
    int32_t  num_kv_heads;
    int32_t  head_dim;
    int32_t  intermediate_size;
    int32_t  vocab_size;
    int32_t  max_seq_len;
    uint32_t tile_rows;       // 32 (Q4NX compatible)
    uint32_t tile_cols;       // 256 (Q4NX compatible)
    uint32_t group_size;      // 32
    uint32_t has_q_norm;            // bool
    uint32_t has_k_norm;            // bool
    uint32_t has_bias;              // bool
    uint32_t rope_theta_f;          // rope_theta * 1000 (stored as fixed-point)
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t tensor_count;

    // ── Kimi/Moonshot/Laguna fields (reserved[0..51]) ─────────
    uint32_t num_experts;           // total routed experts
    uint32_t n_expert_used;         // experts used per token (top-k)
    uint32_t n_ff_exp;              // expert FFN intermediate size
    uint32_t n_ff_shexp;            // shared expert FFN intermediate size
    uint32_t n_layer_dense_lead;    // leading dense layers before MoE begins
    uint32_t sliding_window;        // SWA window size (0 = no SWA)
    uint32_t swa_period;            // SWA pattern period
    uint32_t expert_gating_func;    // OnebpExpertGating
    uint32_t expert_weights_norm;   // bool: sum-normalize expert weights after top-k
    uint32_t expert_weights_scale_f; // routed scaling factor * 1000 (fixed-point)
    uint32_t attn_gate_type;        // OnebpAttnGate (per-head vs per-element)
    uint32_t rope_freq_base_swa_f;  // SWA layer RoPE freq base * 1000
    uint32_t n_rot_swa;             // SWA layer RoPE dimension count
    uint32_t n_rot_full;            // FULL attention layer RoPE dim count (0 = use head_dim)
    uint8_t  reserved[44];          // remaining pad to 256 bytes
    char     model_tag[64];         // model identifier string
    
    // validity: core dims always required; attention heads are optional
    // (Mamba/MoE architectures have no attention, set NH=0, HD=0).
    bool valid() const {
        if (magic != ONEBP_MAGIC || version < 1 || version > ONEBP_VERSION) return false;
        if (hidden_size <= 0 || num_layers <= 0 || vocab_size <= 0) return false;
        if (num_attention_heads > 0 && head_dim <= 0) return false;
        if (head_dim > 0 && num_attention_heads <= 0) return false;
        if (intermediate_size <= 0 || tile_rows <= 0 || tile_cols <= 0 || group_size <= 0) return false;
        return true;
    }
    
    void init() {
        memset(this, 0, sizeof(*this));
        magic = ONEBP_MAGIC;
        version = ONEBP_VERSION;
        tile_rows = 32;
        tile_cols = 256;
        group_size = 32;
    }
    
    float rope_theta() const {
        if (version >= 3) { float t; memcpy(&t, &rope_theta_f, 4); return t; }
        return (float)rope_theta_f / 1000.0f;
    }
    void set_rope_theta(float v) { memcpy(&rope_theta_f, &v, 4); }  // v3: raw f32 bits
    float expert_weights_scale() const { return (float)expert_weights_scale_f / 1000.0f; }
    void set_expert_weights_scale(float v) { expert_weights_scale_f = (uint32_t)(v * 1000.0f); }
    float rope_freq_base_swa() const {
        if (version >= 3) { float t; memcpy(&t, &rope_freq_base_swa_f, 4); return t; }
        return (float)rope_freq_base_swa_f / 1000.0f;
    }
    void set_rope_freq_base_swa(float v) { memcpy(&rope_freq_base_swa_f, &v, 4); }  // v3: raw f32 bits
};
#pragma pack(pop)

// Verify header size at compile time
static_assert(sizeof(OnebpHeader) == 256, "OnebpHeader must be exactly 256 bytes");

// ─── Tensor index entry (on-disk) ─────────────────────────────────
// Variable-length: [name_len:u32][name:str][ndim:u32][shape:u32[]][offset:u64][bytes:u64]
// Not a fixed struct — written element by element in the converter.

// ─── UE4M3 scale codec (ROCmFPX-FP2-style, shared by converter + loader) ──
// exp==0 → mant·2^-10 ; else (8+mant)·2^(exp-11). 127 finite values (0x00..0x7e).
static inline float onebp_ue4m3_to_f32(uint8_t e) {
    if (e > 0x7e) return 0.0f;
    uint32_t exp = e >> 3, mant = e & 7;
    float v = exp == 0 ? (float)mant * 0.0009765625f   // 2^-10
                       : (8.0f + mant) * ldexpf(1.0f, (int)exp - 11);
    return v;
}

static inline uint8_t onebp_nearest_ue4m3(float target) {
    // binary search over the monotonic 127-value table
    if (target <= 0.0f) return 0;
    uint8_t lo = 0, hi = 126;
    while (lo < hi) {
        uint8_t mid = (lo + hi + 1) >> 1;
        if (onebp_ue4m3_to_f32(mid) <= target) lo = mid; else hi = mid - 1;
    }
    if (lo < 126 &&
        fabsf(onebp_ue4m3_to_f32(lo + 1) - target) < fabsf(onebp_ue4m3_to_f32(lo) - target))
        return lo + 1;
    return lo;
}

// ─── Compute tiled size for a weight matrix ──────────────────────
// Returns bytes needed after tiling rows×cols to tile_rows×tile_cols
static inline uint64_t onebp_tiled_size(
    uint32_t rows, uint32_t cols,
    uint32_t tile_rows, uint32_t tile_cols,
    uint32_t group_size, OnebpQuant quant
) {
    uint32_t num_tile_rows = (rows + tile_rows - 1) / tile_rows;
    uint32_t num_tile_cols = (cols + tile_cols - 1) / tile_cols;
    uint32_t groups_per_row = tile_cols / group_size;
    
    uint64_t tile_bytes;
    switch (quant) {
        case ONEBP_Q4NX:
            // scales (bf16 × groups × rows) + zero_points + packed indices
            tile_bytes = (uint64_t)tile_rows * groups_per_row * 2  // scales
                       + (uint64_t)tile_rows * groups_per_row * 2  // zero_points
                       + (uint64_t)tile_rows * tile_cols / 2;      // 4-bit packed
            break;
        case ONEBP_I8:
            tile_bytes = (uint64_t)tile_rows * tile_cols;  // 1 byte per element
            break;
        case ONEBP_TQ2:
        case ONEBP_TQ2NZ:
            // scales (bf16 x groups x rows) + 2-bit packed codes (4/byte)
            tile_bytes = (uint64_t)tile_rows * groups_per_row * 2   // scales
                       + (uint64_t)tile_rows * tile_cols / 4;       // 2-bit packed
            break;
        case ONEBP_TQ2NZ_E4M3:
            // scales (1-byte UE4M3 x groups x rows) + 2-bit packed codes
            tile_bytes = (uint64_t)tile_rows * groups_per_row * 1   // scales
                       + (uint64_t)tile_rows * tile_cols / 4;       // 2-bit packed
            break;
        case ONEBP_TQ1: {
            // 1.58-bit base-3 ternary: 5 codes per byte, ceil(tc/5) groups per row
            uint32_t tq1_grps = (tile_cols + 4) / 5;
            tile_bytes = (uint64_t)tile_rows * tq1_grps * 2  // bf16 scales
                       + (uint64_t)tile_rows * tq1_grps;      // packed codes
            break;
        }
        case ONEBP_F16:
            tile_bytes = (uint64_t)tile_rows * tile_cols * 2;
            break;
        default:
            tile_bytes = (uint64_t)tile_rows * tile_cols * 4;  // F32 fallback
    }
    
    return (uint64_t)num_tile_rows * num_tile_cols * tile_bytes;
}

#endif // ONEBP_FORMAT_H
