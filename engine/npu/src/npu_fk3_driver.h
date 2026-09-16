// npu_fk3_driver.h — fk-3 fused 0.6B layer: TWO launches replace the ~9-launch/layer
// per-op bf16 prefill path (npu_engine_universal.cpp ~4666-4862).
//
//   launch A : fused RMSNorm(input) + QKV          -> (M, NQKV) bf16, row-major
//   host     : fk3::rope_qk_bf16 (rotate Q and K in place)
//              + scatter K/V into the engine's KV cache (bKv)
//   launch B : attention + O-proj + FFN norm + GU + SiLU + D (+ both residuals)
//              -> (M, H) bf16 = the layer output
//
// Every weight is the engine's own, in the engine's own layout, dequantized by the
// engine's own bf16mm_dequant — so there is no numerics to reconcile, only wiring.
// The engine prepares Wqkv/Wo/Wgu/Wd[l] at npu_engine_universal.cpp ~4543-4571; the
// driver asks for the same six offsets and dequantizes into its own host buffers,
// with exactly two edits: Wgu is split into GATE then UP (the engine's order), and
// Wd gets an identity block appended (which is how residual 2 is fused).
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace fk3 {

// The packed Q4NX weight blob for one layer, as the engine's npu_bf16_pack_layer
// fills it. offs[0]=QKV offs[1]=? offs[2]=? offs[3]=W_O offs[4]=GU offs[5]=W_D,
// and every byte offset used below is offs[i] * 5120 (the engine's own expression).
struct WeightSource {
    const uint8_t* bo = nullptr;   // packed layer blob
    const int* offs = nullptr;     // the 6 offsets for this layer
    uint32_t byte_row = 5120;
};

// One fused layer. Not thread-safe: one instance drives one sequence on one device.
class FusedLayer {
public:
    FusedLayer();
    ~FusedLayer();

    // xclbinA/instsA = fused RMSNorm+QKV;  xclbinB/instsB = attention..D.
    // M is the token block (the kernel was built for it; the engine uses 128).
    bool init(int device_index,
              const char* xclbinA, const char* instsA,
              const char* xclbinB, const char* instsB,
              int M, int H, int NH, int NKV, int HD, int IM, int NC);

    // Dequantize layer l's four weights (via the engine's bf16mm_dequant*) and
    // upload them. Call once per layer before the first fk3_layer(l, ...).
    bool prepare_layer(int l, const WeightSource& src);

    // Run one layer for M tokens starting at absolute position pos0.
    //   x          (M, H)  f32, the layer input  -> becomes A rows 0..M-1
    //   gamma_in   (H)     f32, the input norm's weight      -> A row M
    //   gamma_ffn  (H)     f32, the FFN   norm's weight      -> A2 row M
    //   bKv        the engine's bf16 KV buffer; K/V are scattered into it in the
    //              region/slot layout the decode kernel reads (region = kvh<4?0:1,
    //              lh = kvh&3, slot = 4*HD) — matching qk_norm_pi exactly.
    //   out        (M, H)  f32, the layer output.
    // Returns false on any device error.
    bool run(int l, const float* x, const float* gamma_in, const float* gamma_ffn,
             int pos0, uint16_t* bKv, int kv_region, int v_add, float* out);

    int M() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

}  // namespace fk3
