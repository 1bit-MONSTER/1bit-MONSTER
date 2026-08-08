// Minimal ONNX weight extractor — protobuf wire format, zero dependencies.
// Reads .onnx files and extracts weight tensors into rcpp_bitnet_model_t.
//
// ONNX is protobuf. We decode the wire format directly (no libprotobuf needed)
// to extract tensor data from the GraphProto initializer list.
//
// Field numbers verified against onnx 1.22 (ground truth generated with the
// onnx python package and walked with a raw wire decoder):
//   ModelProto.graph = 7
//   GraphProto.node = 1, initializer = 5
//   NodeProto.input = 1, output = 2, name = 3, op_type = 4
//   TensorProto.dims = 1, data_type = 2, name = 8, raw_data = 9
//   (older onnx wrote raw_data at 10 — both accepted, disambiguated by
//   exact byte count vs dtype width; a legacy doc_string at 9 is short
//   and never matches).
//
// INT8/UINT8 weights with a DequantizeLinear scale are converted to the
// WMMA_I8 device layout (.h1b v5 equivalent): dequant → Hadamard-rotate
// each row → per-row INT8 requant with row scales, uploaded to the
// *_i8_dev / *_i8_scales_dev fields with weight_format = WMMA_I8.

#include "rocm_cpp/bitnet_model.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

#define HIP_CHECK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\n",_s,__FILE__,__LINE__); return RCPP_HIP_ERROR;}} while(0)

namespace {

// Minimal protobuf wire format decoder — handles the ONNX subset
struct PbReader {
    const uint8_t* data;
    size_t len;
    size_t pos = 0;

    PbReader(const uint8_t* d, size_t l) : data(d), len(l) {}

    bool ok() const { return pos < len; }

    // Read a varint (protobuf variable-length integer)
    uint64_t varint() {
        uint64_t val = 0;
        int shift = 0;
        while (pos < len) {
            uint8_t byte = data[pos++];
            val |= uint64_t(byte & 0x7F) << shift;
            // #1354: check the terminator BEFORE the overflow guard. A
            // 10-byte varint encoding a value >= 2^63 (e.g. 0x80..01 for
            // 2^63) ends with shift == 63 — the old check fired on the
            // following `shift += 7` and silently zeroed valid values.
            if (!(byte & 0x80)) return val;
            shift += 7;
            // fixes #1331 + #1354: only an 11th byte would shift >= 64 (UB) —
            // reject instead of computing it.
            if (shift > 63) return 0;
        }
        return val;
    }

    // Read a length-delimited field value (wire type 2)
    std::vector<uint8_t> bytes() {
        uint64_t sz = varint();
        if (sz > len || pos > len - sz) sz = len - pos;  // fixes #1326 overflow bypass
        std::vector<uint8_t> result(data + pos, data + pos + sz);
        pos += sz;
        return result;
    }

    // Read a fixed 32-bit value (wire type 5)
    uint32_t fixed32() {
        if (pos + 4 > len) return 0;
        uint32_t v;
        memcpy(&v, data + pos, 4);
        pos += 4;
        return v;
    }

    // Read a fixed 64-bit value (wire type 1)
    uint64_t fixed64() {
        if (pos + 8 > len) return 0;
        uint64_t v;
        memcpy(&v, data + pos, 8);
        pos += 8;
        return v;
    }

    // Skip current field — with bounds check to prevent OOB reads
    // from crafted/malformed ONNX files (issue #960).
    void skip_field(uint32_t wire_type) {
        if (wire_type == 0) { varint(); }
        else if (wire_type == 1) { if (pos + 8 > len) { pos = len; return; } pos += 8; }
        else if (wire_type == 2) { uint64_t sz = varint(); if (sz > len || pos > len - sz) sz = len - pos; pos += sz; }
        else if (wire_type == 5) { if (pos + 4 > len) { pos = len; return; } pos += 4; }
    }
};

// ONNX tensor data types
enum OnnxDataType {
    ONNX_FLOAT = 1,
    ONNX_UINT8 = 2,
    ONNX_INT8 = 3,
    ONNX_UINT16 = 4,
    ONNX_INT16 = 5,
    ONNX_INT32 = 6,
    ONNX_INT64 = 7,
    ONNX_FLOAT16 = 10,
    ONNX_DOUBLE = 11,
    ONNX_BFLOAT16 = 16,
};

struct OnnxTensor {
    std::string name;
    std::vector<int64_t> dims;
    int32_t data_type = 0;
    std::vector<float> float_data;  // dequantized
    std::vector<int8_t> i8_data;    // raw bytes for INT8/UINT8 tensors
};

struct OnnxNode {
    std::string op_type;                 // NodeProto field 4
    std::vector<std::string> inputs;     // NodeProto field 1
    std::vector<std::string> outputs;    // NodeProto field 2
};

// Recursively find all initializer tensors in an ONNX protobuf
static void find_initializers(PbReader& pb, std::vector<OnnxTensor>& tensors,
                              std::vector<OnnxNode>& nodes, int depth = 0) {
    if (depth > 20 || !pb.ok()) return;

    while (pb.ok()) {
        size_t field_start = pb.pos;
        if (field_start >= pb.len) break;

        uint8_t key_byte = pb.data[pb.pos++];
        uint32_t field_num = key_byte >> 3;
        uint32_t wire_type = key_byte & 0x7;

        if (field_num == 0) break; // shouldn't happen

        // ModelProto.graph = 7; GraphProto.node = 1, initializer = 5
        if (wire_type == 2) {
            auto content = pb.bytes();
            PbReader sub(content.data(), content.size());

            if (field_num == 7) {
                // ModelProto.graph — recurse to find initializers
                find_initializers(sub, tensors, nodes, depth + 1);
            } else if (field_num == 1) {
                // GraphProto.node (field 1) — a NodeProto: parse DQ/Q nodes
                OnnxNode nd;
                PbReader np(content.data(), content.size());
                while (np.ok()) {
                    size_t nk = np.pos;
                    if (nk >= content.size()) break;
                    uint8_t nkb = np.data[np.pos++];
                    uint32_t nf = nkb >> 3;
                    uint32_t nw = nkb & 0x7;
                    if (nf == 1 && nw == 2) { auto b = np.bytes(); nd.inputs.emplace_back((char*)b.data(), b.size()); }
                    else if (nf == 2 && nw == 2) { auto b = np.bytes(); nd.outputs.emplace_back((char*)b.data(), b.size()); }
                    else if (nf == 4 && nw == 2) { auto b = np.bytes(); nd.op_type.assign((char*)b.data(), b.size()); }
                    else np.skip_field(nw);
                }
                if (!nd.op_type.empty()) nodes.push_back(std::move(nd));
            } else if (field_num == 5) {
                // GraphProto.initializer (field 5) — a TensorProto — parse it
                OnnxTensor t;
                PbReader tp(content.data(), content.size());
                std::vector<uint8_t> raw9, raw10;  // raw_data candidates (1.22=9, legacy=10)
                while (tp.ok()) {
                    size_t tk = tp.pos;
                    if (tk >= content.size()) break;
                    uint8_t tkb = tp.data[tp.pos++];
                    uint32_t tf = tkb >> 3;
                    uint32_t tw = tkb & 0x7;

                    if (tf == 1 && tw == 2) { // dims (packed varint, field 1, wire type 2)
                        auto dim_bytes = tp.bytes();
                        PbReader dim_pb(dim_bytes.data(), dim_bytes.size());
                        while (dim_pb.ok()) t.dims.push_back((int64_t)dim_pb.varint());
                    } else if (tf == 1 && tw == 0) { // dims (non-packed varint, rare)
                        t.dims.push_back((int64_t)tp.varint());
                    } else if (tf == 2) { // data_type (int32, field 2)
                        t.data_type = (int32_t)tp.varint();
                    } else if (tf == 3) { // segment (field 3) — skip
                        auto seg = tp.bytes();
                    } else if (tf == 4) { // float_data (float, repeated, field 4)
                        if (tw == 2) { auto b = tp.bytes();
                            for (size_t i = 0; i + 4 <= b.size(); i += 4) {
                                float v; memcpy(&v, &b[i], 4); t.float_data.push_back(v);
                            }
                        }
                    } else if (tf == 5) { // int32_data (field 5)
                        if (tw == 2) { auto b = tp.bytes();
                            for (size_t i = 0; i + 4 <= b.size(); i += 4) {
                                int32_t v; memcpy(&v, &b[i], 4); t.float_data.push_back((float)v);
                            }
                        }
                    } else if (tf == 6) { // string_data (field 6) — skip
                        tp.skip_field(tw);
                    } else if (tf == 7) { // int64_data (field 7)
                        if (tw == 2) { auto b = tp.bytes();
                            for (size_t i = 0; i + 8 <= b.size(); i += 8) {
                                int64_t v; memcpy(&v, &b[i], 8); t.float_data.push_back((float)v);
                            }
                        }
                    } else if (tf == 8) { // name (string, field 8)
                        auto nm = tp.bytes();
                        if (!nm.empty()) t.name.assign((char*)nm.data(), nm.size());
                    } else if (tf == 9) { // raw_data (bytes, field 9 — onnx 1.22)
                        raw9 = tp.bytes();
                    } else if (tf == 10) { // double_data (field 10) OR legacy raw_data
                        raw10 = tp.bytes();
                    } else if (tf == 11 || tf == 12 || tf == 13 || tf == 14) {
                        // uint64_data / doc_string / external_data / data_location — skip
                        tp.skip_field(tw);
                    } else {
                        tp.skip_field(tw);
                    }
                }

                // Expand raw bytes into float_data (and i8_data for INT8/UINT8).
                // raw_data moved from field 10 → 9 in modern onnx; pick the
                // candidate whose byte count exactly matches dims × dtype width
                // (a legacy doc_string at 9 is short and never matches).
                if (!t.dims.empty() && !raw9.empty()) {
                    size_t elem = 1;
                    for (auto d : t.dims) elem *= (size_t)d;
                    size_t width = 4;
                    if (t.data_type == ONNX_FLOAT16 || t.data_type == ONNX_BFLOAT16) width = 2;
                    else if (t.data_type == ONNX_INT8 || t.data_type == ONNX_UINT8) width = 1;
                    if (t.data_type == ONNX_DOUBLE && !raw10.empty() &&
                        raw10.size() == elem * 8 && raw9.size() != elem * width) {
                        // double_data lives at 10; try that first for DOUBLE
                        for (size_t i = 0; i + 8 <= raw10.size(); i += 8) {
                            double v; memcpy(&v, &raw10[i], 8); t.float_data.push_back((float)v);
                        }
                    } else {
                        const std::vector<uint8_t>* raw = &raw9;
                        if (raw->size() != elem * width && raw10.size() == elem * width)
                            raw = &raw10;  // legacy raw_data at 10
                        if (raw->size() == elem * width) {
                            if (t.data_type == ONNX_FLOAT16) {
                                // Proper IEEE float16 → float32 (not bfloat16 shift trick)
                                t.float_data.resize(raw->size() / 2);
                                for (size_t i = 0; i + 2 <= raw->size(); i += 2) {
                                    uint16_t f16; memcpy(&f16, &(*raw)[i], 2);
                                    uint32_t s = (f16 >> 15) & 1, e = (f16 >> 10) & 0x1f, m = f16 & 0x3ff;
                                    float sign = s ? -1.0f : 1.0f;
                                    float v;
                                    if (e == 0)
                                        v = sign * (float)m * 5.9604644775390625e-08f;
                                    else if (e == 31)
                                        v = m ? NAN : sign * INFINITY;
                                    else
                                        v = sign * (1.0f + (float)m / 1024.0f) * powf(2.0f, (float)((int)e - 15));
                                    t.float_data[i / 2] = v;
                                }
                            } else if (t.data_type == ONNX_BFLOAT16) {
                                // bfloat16: upper 16 bits of float32
                                t.float_data.resize(raw->size() / 2);
                                for (size_t i = 0; i + 2 <= raw->size(); i += 2) {
                                    uint16_t bf16; memcpy(&bf16, &(*raw)[i], 2);
                                    uint32_t bits = (uint32_t)bf16 << 16;
                                    float v; memcpy(&v, &bits, 4);
                                    t.float_data[i / 2] = v;
                                }
                            } else if (t.data_type == ONNX_INT8 || t.data_type == ONNX_UINT8) {
                                // Keep raw bytes for the WMMA_I8 path; float_data
                                // is the unscaled expansion (QDQ scale applied at
                                // device upload via the DequantizeLinear map).
                                t.i8_data.assign((const int8_t*)raw->data(),
                                                 (const int8_t*)raw->data() + raw->size());
                                t.float_data.resize(raw->size());
                                for (size_t i = 0; i < raw->size(); i++)
                                    t.float_data[i] = (float)(int8_t)(*raw)[i];
                            } else {
                                // Default: F32 (4 bytes per value)
                                size_t n_floats = raw->size() / 4;
                                t.float_data.resize(n_floats);
                                for (size_t i = 0; i + 4 <= raw->size(); i += 4) {
                                    float v; memcpy(&v, &(*raw)[i], 4);
                                    t.float_data[i / 4] = v;
                                }
                            }
                        }
                    }
                }
                if (!t.name.empty() && !t.dims.empty() &&
                    (!t.float_data.empty() || !t.i8_data.empty())) {
                    tensors.push_back(std::move(t));
                }
            } else {
                // Skip other length-delimited fields
            }
        } else if (wire_type == 0) {
            pb.varint();
        } else if (wire_type == 1) {
            pb.pos += 8;
        } else if (wire_type == 5) {
            pb.pos += 4;
        } else {
            break;
        }
    }
}

// Hadamard + per-row requant, mirrors tools/hadamard_export.cpp exactly.
constexpr int HADAMARD_BLOCK = 128;

static void hadamard_rotate_row(float* row, int K) {
    for (int base = 0; base < K; base += HADAMARD_BLOCK) {
        float* blk = row + base;
        for (int stage = 0; (1 << stage) < HADAMARD_BLOCK; ++stage) {
            int dist = 1 << stage;
            for (int i = 0; i < HADAMARD_BLOCK; ++i) {
                if ((i & dist) == 0) {
                    int j = i | dist;
                    float a = blk[i], b = blk[j];
                    blk[i] = a + b;
                    blk[j] = a - b;
                }
            }
        }
        constexpr float inv_sqrt_b = 1.0f / 11.31370849898476f;
        for (int i = 0; i < HADAMARD_BLOCK; ++i)
            blk[i] *= inv_sqrt_b;
    }
}

static void quantize_row_i8(const float* row, int K,
                            std::vector<int8_t>& out_i8, float& out_scale) {
    float max_abs = 0.0f;
    for (int i = 0; i < K; ++i) {
        float a = fabsf(row[i]);
        if (a > max_abs) max_abs = a;
    }
    out_scale = max_abs / 127.0f;
    if (out_scale < 1e-10f) out_scale = 1e-10f;
    out_i8.resize(K);
    for (int i = 0; i < K; ++i) {
        float v = row[i] / out_scale;
        v = fminf(fmaxf(roundf(v), -128.0f), 127.0f);
        out_i8[i] = (int8_t)v;
    }
}

struct WmmaPacked {
    std::vector<int8_t> data;
    std::vector<float> scales;
};

// Dequantize (i8 - zp) x scale, Hadamard-rotate each row, re-quantize per-row.
// Produces exactly what .h1b v5 (RCPP_WEIGHT_FORMAT_WMMA_I8) files carry.
// Scale shapes accepted: scalar, [C] (per-col), [R] (per-row).
static bool make_wmma_i8(const OnnxTensor& t, const OnnxTensor& scale_t,
                         int8_t zp, WmmaPacked& out) {
    if (t.dims.size() != 2) return false;
    int64_t R = t.dims[0], C = t.dims[1];
    if (C <= 0 || C % HADAMARD_BLOCK != 0) return false;
    size_t n = (size_t)R * C;
    if (t.i8_data.size() != n) return false;
    const std::vector<float>& sv = scale_t.float_data;
    // ONNX weight QDQ convention is axis=0 → per-row ([R,C] weights). Prefer
    // per-row over per-col when both match (square matrices).
    bool per_row = (sv.size() == (size_t)R);
    bool per_col = (sv.size() == (size_t)C);
    if (sv.empty() || (!per_row && !per_col && sv.size() != 1)) return false;

    std::vector<float> row((size_t)C);
    out.data.resize(n);
    out.scales.resize((size_t)R);
    std::vector<int8_t> row_i8;
    for (int64_t r = 0; r < R; ++r) {
        float sr = per_row ? sv[(size_t)r] : sv[0];
        for (int64_t c = 0; c < C; ++c)
            row[(size_t)c] = (float)((int32_t)t.i8_data[(size_t)r * C + c] - zp)
                             * (per_row ? sr : (per_col ? sv[(size_t)c] : sr));
        hadamard_rotate_row(row.data(), (int)C);
        quantize_row_i8(row.data(), (int)C, row_i8, out.scales[(size_t)r]);
        memcpy(out.data.data() + (size_t)r * C, row_i8.data(), (size_t)C);
    }
    return true;
}

} // anonymous namespace

extern "C" {

rcpp_status_t rcpp_bitnet_load_onnx(const char* path, rcpp_bitnet_model_t* out_model) {
    if (!path || !out_model) return RCPP_INVALID_ARG;
    memset(out_model, 0, sizeof(*out_model));

    fprintf(stderr, "[onnx] Loading: %s\n", path);

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return RCPP_INVALID_ARG;
    size_t file_size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> file_data(file_size);
    f.read((char*)file_data.data(), file_size);
    if (!f) return RCPP_INVALID_ARG;

    // Parse protobuf to find initializer tensors + DequantizeLinear nodes
    PbReader pb(file_data.data(), file_size);
    std::vector<OnnxTensor> tensors;
    std::vector<OnnxNode> nodes;
    find_initializers(pb, tensors, nodes);

    fprintf(stderr, "[onnx] Found %zu tensors, %zu nodes\n", tensors.size(), nodes.size());

    // Build name → tensor lookup
    std::unordered_map<std::string, OnnxTensor*> name_map;
    for (auto& t : tensors) name_map[t.name] = &t;

    auto lookup = [&](const std::string& name) -> OnnxTensor* {
        auto it = name_map.find(name);
        return (it != name_map.end()) ? it->second : nullptr;
    };

    // DequantizeLinear map: weight initializer -> (scale tensor, zero point)
    std::unordered_map<std::string, const OnnxTensor*> dq_scale;
    std::unordered_map<std::string, int8_t> dq_zp;
    for (auto& nd : nodes) {
        if (nd.op_type != "DequantizeLinear" || nd.inputs.size() < 2) continue;
        OnnxTensor* w = lookup(nd.inputs[0]);
        OnnxTensor* s = lookup(nd.inputs[1]);
        if (!w || !s) continue;
        dq_scale[w->name] = s;
        if (nd.inputs.size() >= 3) {
            OnnxTensor* zp = lookup(nd.inputs[2]);
            if (zp && !zp->i8_data.empty()) dq_zp[w->name] = zp->i8_data[0];
        }
    }

    // Determine number of layers
    int n_layers = 0;
    for (auto& t : tensors) {
        int lidx = -1;
        if (sscanf(t.name.c_str(), "model.layers.%d.", &lidx) == 1 && lidx >= 0) {
            if (lidx + 1 > n_layers) n_layers = lidx + 1;
        }
    }
    if (n_layers == 0) {
        fprintf(stderr, "[onnx] ERROR: no layers found\n");
        return RCPP_INVALID_ARG;
    }

    // Determine model dimensions from tensor shapes
    auto* emb_t = lookup("model.embed_tokens.weight");
    if (!emb_t || emb_t->dims.size() < 2) {
        fprintf(stderr, "[onnx] ERROR: missing embed_tokens.weight\n");
        return RCPP_INVALID_ARG;
    }
    int hidden_size  = (int)emb_t->dims[1];
    int vocab_size   = (int)emb_t->dims[0];

    auto* gate0 = lookup("model.layers.0.mlp.gate_proj.weight");
    int intermediate_size = gate0 ? (int)gate0->dims[0] : hidden_size;

    int weight_dtype = emb_t->data_type;

    // Helper: allocate device memory and copy tensor data.
    // INT8/UINT8 tensors get their QDQ scale applied here (zero-point too).
    auto tensor_to_dev = [&](OnnxTensor* t, size_t* out_bytes = nullptr) -> void* {
        if (!t) {
            if (out_bytes) *out_bytes = 0;
            return nullptr;
        }
        size_t n_elems = t->float_data.size();
        void* dev_ptr = nullptr;
        size_t bytes = 0;

        if (t->data_type == ONNX_FLOAT16) {
            // Convert f32 back to f16 for device storage
            bytes = n_elems * sizeof(uint16_t);
            std::vector<uint16_t> f16_buf(n_elems);
            for (size_t i = 0; i < n_elems; i++) {
                float v = t->float_data[i];
                uint32_t f32_bits; memcpy(&f32_bits, &v, 4);
                uint32_t sign = (f32_bits >> 16) & 0x8000;
                int32_t exp = ((int32_t)(f32_bits >> 23) & 0xff) - 127 + 15;
                uint32_t mant = (f32_bits >> 13) & 0x3ff;
                if (exp <= 0) { f16_buf[i] = (uint16_t)sign; }
                else if (exp >= 31) { f16_buf[i] = (uint16_t)(sign | 0x7c00); }
                else { f16_buf[i] = (uint16_t)(sign | (exp << 10) | mant); }
            }
            if (hipMalloc(&dev_ptr, bytes) != hipSuccess) return nullptr;
            if (hipMemcpy(dev_ptr, f16_buf.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) { hipFree(dev_ptr); return nullptr; }
        } else if (t->data_type == ONNX_BFLOAT16) {
            // Norm weights: kernels read FP16 (h1b convention). Convert.
            bytes = (size_t)n_elems * sizeof(uint16_t);
            std::vector<uint16_t> f16_buf(n_elems);
            for (size_t i = 0; i < n_elems; i++) {
                float v = t->float_data[i];
                uint32_t f32_bits; memcpy(&f32_bits, &v, 4);
                uint32_t sign = (f32_bits >> 16) & 0x8000;
                int32_t exp = ((int32_t)(f32_bits >> 23) & 0xff) - 127 + 15;
                uint32_t mant = (f32_bits >> 13) & 0x3ff;
                if (exp <= 0) { f16_buf[i] = (uint16_t)sign; }
                else if (exp >= 31) { f16_buf[i] = (uint16_t)(sign | 0x7c00); }
                else { f16_buf[i] = (uint16_t)(sign | (exp << 10) | mant); }
            }
            if (hipMalloc(&dev_ptr, bytes) != hipSuccess) return nullptr;
            if (hipMemcpy(dev_ptr, f16_buf.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) { hipFree(dev_ptr); return nullptr; }
        } else if (t->data_type == ONNX_INT8 || t->data_type == ONNX_UINT8) {
            // Apply QDQ scale (previously hardcoded to 1.0 — wrong for
            // quantized norms/embeddings). Zero point defaults to 0.
            // Norms/embeddings: kernels read FP16 (h1b convention). Convert.
            bytes = (size_t)n_elems * sizeof(uint16_t);
            std::vector<uint16_t> f16_buf(n_elems);
            int8_t zp = 0;
            auto zit = dq_zp.find(t->name);
            if (zit != dq_zp.end()) zp = zit->second;
            auto sit = dq_scale.find(t->name);
            const OnnxTensor* st = sit != dq_scale.end() ? sit->second : nullptr;
            // Per-row scales (embeddings quantized along dim 0) or per-tensor.
            bool per_row = st && t->dims.size() == 2 &&
                           st->float_data.size() == (size_t)t->dims[0];
            size_t row_len = per_row ? (size_t)t->dims[1] : n_elems;
            for (size_t i = 0; i < n_elems; i++) {
                float s = per_row ? st->float_data[i / row_len]
                                  : (st && !st->float_data.empty() ? st->float_data[0] : 1.0f);
                float v = (float)((int32_t)t->i8_data[i] - zp) * s;
                uint32_t f32_bits; memcpy(&f32_bits, &v, 4);
                uint32_t sign = (f32_bits >> 16) & 0x8000;
                int32_t exp = ((int32_t)(f32_bits >> 23) & 0xff) - 127 + 15;
                uint32_t mant = (f32_bits >> 13) & 0x3ff;
                if (exp <= 0) { f16_buf[i] = (uint16_t)sign; }
                else if (exp >= 31) { f16_buf[i] = (uint16_t)(sign | 0x7c00); }
                else { f16_buf[i] = (uint16_t)(sign | (exp << 10) | mant); }
            }
            if (hipMalloc(&dev_ptr, bytes) != hipSuccess) return nullptr;
            if (hipMemcpy(dev_ptr, f16_buf.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) { hipFree(dev_ptr); return nullptr; }
        } else if (t->data_type == ONNX_FLOAT) {
            // Norm weights: kernels read FP16 (h1b convention). Convert.
            bytes = (size_t)n_elems * sizeof(uint16_t);
            std::vector<uint16_t> f16_buf(n_elems);
            for (size_t i = 0; i < n_elems; i++) {
                float v = t->float_data[i];
                uint32_t f32_bits; memcpy(&f32_bits, &v, 4);
                uint32_t sign = (f32_bits >> 16) & 0x8000;
                int32_t exp = ((int32_t)(f32_bits >> 23) & 0xff) - 127 + 15;
                uint32_t mant = (f32_bits >> 13) & 0x3ff;
                if (exp <= 0) { f16_buf[i] = (uint16_t)sign; }
                else if (exp >= 31) { f16_buf[i] = (uint16_t)(sign | 0x7c00); }
                else { f16_buf[i] = (uint16_t)(sign | (exp << 10) | mant); }
            }
            if (hipMalloc(&dev_ptr, bytes) != hipSuccess) return nullptr;
            if (hipMemcpy(dev_ptr, f16_buf.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) { hipFree(dev_ptr); return nullptr; }
        } else {
            // Fallback: F32 (4 bytes per value)
            bytes = (size_t)n_elems * sizeof(float);
            if (hipMalloc(&dev_ptr, bytes) != hipSuccess) return nullptr;
            if (hipMemcpy(dev_ptr, t->float_data.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) { hipFree(dev_ptr); return nullptr; }
        }

        if (out_bytes) *out_bytes = bytes;
        return dev_ptr;
    };

    // Allocate embedding and final norm
    out_model->embedding_dev = tensor_to_dev(emb_t);

    auto* norm_t = lookup("model.norm.weight");
    out_model->final_norm_weight_dev = tensor_to_dev(norm_t);

    // Allocate per-layer weights
    out_model->layers = (rcpp_bitnet_layer_t*)calloc(n_layers, sizeof(rcpp_bitnet_layer_t));
    if (!out_model->layers) return RCPP_INTERNAL;

    // Linear weights: WMMA_I8 path (i8_dev + per-row scales) when the tensor
    // is INT8 with a DQ scale and K % 128 == 0; legacy float path otherwise.
    auto linear_to_dev = [&](const char* name, void** packed_dev,
                             void** i8_dev, float** i8_scales_dev) -> bool {
        OnnxTensor* t = lookup(name);
        if (!t) return false;
        auto sit = dq_scale.find(t->name);
        if (sit != dq_scale.end()) {
            int8_t zp = 0;
            auto zit = dq_zp.find(t->name);
            if (zit != dq_zp.end()) zp = zit->second;
            WmmaPacked wp;
            if (make_wmma_i8(*t, *sit->second, zp, wp)) {
                size_t data_bytes = wp.data.size();
                if (hipMalloc(i8_dev, data_bytes) != hipSuccess) return false;
                if (hipMemcpy(*i8_dev, wp.data.data(), data_bytes, hipMemcpyHostToDevice) != hipSuccess) {
                    hipFree(*i8_dev); *i8_dev = nullptr; return false;
                }
                size_t scale_bytes = wp.scales.size() * sizeof(float);
                if (hipMalloc((void**)i8_scales_dev, scale_bytes) != hipSuccess) return false;
                if (hipMemcpy(*i8_scales_dev, wp.scales.data(), scale_bytes, hipMemcpyHostToDevice) != hipSuccess) {
                    hipFree(*i8_scales_dev); *i8_scales_dev = nullptr; return false;
                }
                return true;
            }
        }
        *packed_dev = tensor_to_dev(t);
        return false;
    };

    int n_i8_linears = 0, n_linears = 0;

    for (int i = 0; i < n_layers; i++) {
        char buf[256];
        auto& L = out_model->layers[i];

        snprintf(buf, sizeof(buf), "model.layers.%d.input_layernorm.weight", i);
        L.input_norm_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.post_attention_layernorm.weight", i);
        L.post_attn_norm_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_proj.bias", i);
        L.q_bias_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_proj.bias", i);
        L.k_bias_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.v_proj.bias", i);
        L.v_bias_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_norm.weight", i);
        L.attn_q_norm_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_norm.weight", i);
        L.attn_k_norm_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_proj.weight", i);
        n_linears++;
        if (linear_to_dev(buf, &L.q_packed_dev, &L.q_i8_dev, &L.q_i8_scales_dev)) n_i8_linears++;

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_proj.weight", i);
        n_linears++;
        if (linear_to_dev(buf, &L.k_packed_dev, &L.k_i8_dev, &L.k_i8_scales_dev)) n_i8_linears++;

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.v_proj.weight", i);
        n_linears++;
        if (linear_to_dev(buf, &L.v_packed_dev, &L.v_i8_dev, &L.v_i8_scales_dev)) n_i8_linears++;

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.o_proj.weight", i);
        n_linears++;
        if (linear_to_dev(buf, &L.o_packed_dev, &L.o_i8_dev, &L.o_i8_scales_dev)) n_i8_linears++;

        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.gate_proj.weight", i);
        n_linears++;
        if (linear_to_dev(buf, &L.gate_packed_dev, &L.gate_i8_dev, &L.gate_i8_scales_dev)) n_i8_linears++;

        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.up_proj.weight", i);
        n_linears++;
        if (linear_to_dev(buf, &L.up_packed_dev, &L.up_i8_dev, &L.up_i8_scales_dev)) n_i8_linears++;

        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.down_proj.weight", i);
        n_linears++;
        if (linear_to_dev(buf, &L.down_packed_dev, &L.down_i8_dev, &L.down_i8_scales_dev)) n_i8_linears++;
    }

    // Untied LM head (HF: lm_head.weight; gguf-converted: output.weight).
    // NULL stays tied-to-embedding.
    if (auto* lm = lookup("lm_head.weight")) {
        out_model->lm_head_dev = tensor_to_dev(lm);
    } else if (auto* lm = lookup("output.weight")) {
        out_model->lm_head_dev = tensor_to_dev(lm);
    }

    // Fill model metadata
    out_model->hidden_size       = hidden_size;
    out_model->intermediate_size = intermediate_size;
    out_model->num_layers        = n_layers;
    out_model->num_heads         = 0;      // caller should set from config
    out_model->num_kv_heads      = 0;      // caller should set from config
    out_model->vocab_size        = vocab_size;
    out_model->max_seq_len       = 2048;
    out_model->tie_embeddings    = 0;
    out_model->rope_theta        = 10000.0f;
    out_model->rms_norm_eps      = 1e-5f;
    out_model->format_version    = 0;
    out_model->flags             = 0;
    out_model->weight_format     = RCPP_WEIGHT_FORMAT_HALO_V2;
    out_model->is_qwen3          = 0;
    out_model->arch              = RCPP_ARCH_BITNET;

    if (n_i8_linears == n_linears && n_linears > 0) {
        out_model->weight_format = RCPP_WEIGHT_FORMAT_WMMA_I8;
        out_model->flags |= H1B_FLAG_HADAMARD_ROTATED;
    }

    fprintf(stderr, "[onnx] Model built: %d layers, hidden=%d, intermediate=%d, vocab=%d, format=%s\n",
            n_layers, hidden_size, intermediate_size, vocab_size,
            out_model->weight_format == RCPP_WEIGHT_FORMAT_WMMA_I8 ? "WMMA_I8" :
            (weight_dtype == ONNX_FLOAT16 ? "F16" : "F32"));

    return RCPP_OK;
}

} // extern "C"
