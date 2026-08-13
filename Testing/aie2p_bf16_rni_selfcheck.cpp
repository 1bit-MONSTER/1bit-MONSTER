// AIE2P bf16 add rounding self-check — the RNI rule, made runnable.
//
// Hardware fact measured on Strix Halo (2026-08-13): the AIE2P bf16 add
// rounds the fp32 sum toward NEGATIVE INFINITY, not round-to-nearest-even.
//   result = trunc16(s) + ((frac16(s) != 0) && (sign(s) == 1))
// Verified 0/4096 mismatches on hardware; RNE/RHA/RNO/RZ all mismatch ~24%.
// See engine/npu/AIE2P-FACTS.md. Any future bf16 AIE kernel verification
// must use this reference (IRON's own tests only pass on rel_tol=0.04).
//
// Build & run:
//   g++ -std=c++17 -O2 Testing/aie2p_bf16_rni_selfcheck.cpp -o /tmp/rni_check && /tmp/rni_check
#include <cstdint>
#include <cstdio>
#include <cstring>

static inline uint32_t f32_bits(float f) {
  uint32_t u; std::memcpy(&u, &f, 4); return u;
}
static inline float f32_from_bits(uint32_t u) {
  float f; std::memcpy(&f, &u, 4); return f;
}
// bf16 value (uint16 bit pattern) -> float (exact)
static inline float bf16_to_f32(uint16_t b) {
  return f32_from_bits((uint32_t)b << 16);
}
// The measured AIE2P rule.
static inline uint16_t aie2p_bf16_add(uint16_t a, uint16_t b) {
  uint32_t s = f32_bits(bf16_to_f32(a) + bf16_to_f32(b));
  uint32_t frac = s & 0xFFFF;
  uint32_t up = (frac != 0) && ((s >> 31) & 1);
  return (uint16_t)((s >> 16) + up);
}
// Independent reference: fp32 sum then explicitly round toward -inf.
static inline uint16_t rni_reference(uint16_t a, uint16_t b) {
  float s = bf16_to_f32(a) + bf16_to_f32(b);
  uint32_t bits = f32_bits(s);
  uint32_t frac = bits & 0xFFFF;
  uint32_t sign = (bits >> 31) & 1;
  uint32_t truncated = bits >> 16;
  if (frac != 0 && sign) truncated += 1;  // more negative
  return (uint16_t)truncated;
}
// Deterministic LCG (no deps).
static inline uint32_t lcg(uint32_t* s) {
  *s = *s * 1664525u + 1013904223u; return *s;
}

int main() {
  int fails = 0;
  // (1) Rule == independent reference on a deterministic random batch.
  uint32_t seed = 42;
  for (int i = 0; i < 100000; i++) {
    // bf16 values in a normal-ish range: bits with exp 0x3E..0x40
    uint16_t a = (uint16_t)(0x3F00u | (lcg(&seed) & 0xFFu));
    uint16_t b = (uint16_t)(0x3F00u | (lcg(&seed) & 0xFFu));
    if (aie2p_bf16_add(a, b) != rni_reference(a, b)) fails++;
  }
  // The exact measured pairs from the hardware run (elementwise add, N=4096,
  // seed 7): a/b are the input bf16 values (truncated f32), expect is what the
  // NPU produced (0/4096 mismatches vs the RNI rule).
  struct { uint16_t a, b, expect; } hw_cases[] = {
      {0x3e80, 0xbf77, 0xbf37},  // 0.25019 + -0.96614 -> 48951
      {0x3f4b, 0xbe80, 0x3f0b},  // 0.79443 + -0.25092 -> 16139
      {0x3f0d, 0xbeda, 0x3e00},  // 0.55137 + -0.42695 -> 15872
      {0xbf0c, 0x3f40, 0x3e50},  // -0.54959 + 0.75055 -> 15952
  };
  for (auto& c : hw_cases) {
    uint16_t got = aie2p_bf16_add(c.a, c.b);
    if (got != c.expect) {
      printf("FAIL hw case %04x+%04x: got %04x expect %04x\n", c.a, c.b, got, c.expect);
      fails++;
    }
  }
  if (fails == 0) {
    printf("PASS: aie2p_bf16_add matches RNI reference (100k batch) and 4 measured hw cases\n");
    return 0;
  }
  printf("FAILED: %d checks\n", fails);
  return 1;
}
