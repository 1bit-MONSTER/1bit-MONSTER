// src/model_registry.cpp — artifact-first registry implementation.
// See include/model_registry.h for the contract and the ADR references.
//
// Self-contained: std library only. The GGUF probe here is intentionally
// minimal (header + tensor dtype census + general.* strings) so this module
// stays linkable without HIP/XRT and can be unit-tested standalone. When the
// engine wires this in, ScanOptions::probe_headers can be swapped for
// read_gguf_metadata() from include/model_discovery.h.
#include "model_registry.h"
// OnebpHeader / OnebpQuant / OnebpArch. Dependency-light (std headers only),
// so including it keeps the registry free of HIP/XRT while avoiding a second
// copy of the native enums that could silently drift.
#include "onebp_format.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <system_error>
#include <unordered_map>

namespace fs = std::filesystem;

namespace onebit {
namespace {

// ── small string helpers ───────────────────────────────────────────────────
std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}
std::string normalize_token(std::string s) {
    s = lower(s);
    std::string out;
    bool dash = false;
    for (char c : s) {
        if (c == '_' || c == ' ' || c == '.' || c == '+' || c == '(' || c == ')' ||
            c == '[' || c == ']' || c == ',') {
            if (!out.empty() && !dash) { out += '-'; dash = true; }
        } else {
            out += c;
            dash = (c == '-');
        }
    }
    while (!out.empty() && (out.back() == '-' || out.back() == '.')) out.pop_back();
    // collapse doubled dashes
    std::string c2;
    for (char c : out) {
        if (c == '-' && !c2.empty() && c2.back() == '-') continue;
        c2 += c;
    }
    return c2;
}
std::string basename_of(const std::string& p) { return fs::path(p).filename().string(); }
std::string stem_of(const std::string& p) { return fs::path(p).stem().string(); }

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    o += buf;
                } else o += c;
        }
    }
    return o;
}

// ── SHA-256 (self-contained; used only with ScanOptions::digest) ───────────
struct Sha256 {
    uint32_t h[8];
    uint64_t len = 0;
    uint8_t buf[64];
    size_t buflen = 0;

    Sha256() {
        static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        memcpy(h, iv, sizeof h);
    }
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) |
                   ((uint32_t)p[i*4+2] << 8) | (uint32_t)p[i*4+3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    void update(const uint8_t* p, size_t n) {
        len += n;
        while (n) {
            size_t take = std::min(n, (size_t)64 - buflen);
            memcpy(buf + buflen, p, take);
            buflen += take; p += take; n -= take;
            if (buflen == 64) { block(buf); buflen = 0; }
        }
    }
    std::string hex() {
        uint64_t bits = len * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t z = 0;
        while (buflen != 56) update(&z, 1);
        uint8_t lb[8];
        for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8*i));
        update(lb, 8);
        static const char* hx = "0123456789abcdef";
        std::string o;
        o.reserve(64);
        for (int i = 0; i < 8; i++)
            for (int j = 3; j >= 0; j--) {
                uint8_t byte = (uint8_t)(h[i] >> (8*j));
                o += hx[byte >> 4]; o += hx[byte & 15];
            }
        return o;
    }
};

std::string sha256_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    Sha256 s;
    std::vector<char> buf(1 << 20);
    while (f) {
        f.read(buf.data(), (std::streamsize)buf.size());
        std::streamsize got = f.gcount();
        if (got <= 0) break;
        s.update((const uint8_t*)buf.data(), (size_t)got);
    }
    return s.hex();
}

// ── minimal JSON reader for recipe-keyed catalogs ──────────────────────────
// Just enough to walk `user_models.json`: a top-level object whose values are
// objects of scalar or string fields. Unknown shapes are skipped, not fatal —
// a catalog we cannot fully parse must never take the registry down with it.
class MiniJson {
public:
    explicit MiniJson(const std::string& s) : s_(s) {}

    using Entry = std::pair<std::string, std::map<std::string, std::string>>;
    bool top_object(std::vector<Entry>& out) {
        ws();
        if (!eat('{')) return false;
        ws();
        if (eat('}')) return true;
        while (true) {
            std::string key;
            if (!parse_string(key)) return false;
            ws();
            if (!eat(':')) return false;
            ws();
            if (peek() == '{') {
                std::map<std::string, std::string> kv;
                if (!flat_object(kv)) return false;
                out.emplace_back(key, std::move(kv));
            } else if (!skip_value()) {
                return false;
            }
            ws();
            if (eat(',')) { ws(); continue; }
            return eat('}');
        }
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) i_++;
    }
    bool eat(char c) { if (peek() == c) { i_++; return true; } return false; }

    bool parse_string(std::string& out) {
        ws();
        if (!eat('"')) return false;
        out.clear();
        while (i_ < s_.size()) {
            char c = s_[i_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (i_ >= s_.size()) return false;
                char e = s_[i_++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        if (i_ + 4 > s_.size()) return false;
                        unsigned cp = 0;
                        for (int k = 0; k < 4; k++) {
                            char h = s_[i_++];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                            else return false;
                        }
                        if (cp < 0x80) out += (char)cp;   // ASCII escapes only
                        break;
                    }
                    default: out += e;
                }
            } else {
                out += c;
            }
        }
        return false;
    }

    bool skip_value() {
        ws();
        char c = peek();
        if (c == '"') { std::string t; return parse_string(t); }
        if (c == '{' || c == '[') {
            char open = c, close = (c == '{') ? '}' : ']';
            int depth = 0;
            while (i_ < s_.size()) {
                char d = s_[i_];
                if (d == '"') { std::string t; if (!parse_string(t)) return false; continue; }
                if (d == open) depth++;
                else if (d == close) { depth--; if (depth == 0) { i_++; return true; } }
                i_++;
            }
            return false;
        }
        size_t start = i_;
        while (i_ < s_.size() && s_[i_] != ',' && s_[i_] != '}' && s_[i_] != ']') i_++;
        return i_ > start;
    }

    bool flat_object(std::map<std::string, std::string>& out) {
        ws();
        if (!eat('{')) return false;
        ws();
        if (eat('}')) return true;
        while (true) {
            std::string k;
            if (!parse_string(k)) return false;
            ws();
            if (!eat(':')) return false;
            ws();
            std::string v;
            if (peek() == '"') { if (!parse_string(v)) return false; }
            else if (!skip_value()) return false;
            out[k] = v;
            ws();
            if (eat(',')) { ws(); continue; }
            return eat('}');
        }
    }
};

// ── minimal GGUF header probe ──────────────────────────────────────────────
struct GgufProbe {
    bool ok = false;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    std::set<uint32_t> dtypes;
    std::string general_name;
    std::string architecture;
    // Declared architecture facts, keyed by SUFFIX so any model prefix works
    // (zaya.expert_count, qwen3.expert_count, ...). Needed for F12: comparing a
    // native header against a dims-identical sibling GGUF.
    int32_t declared_hidden = 0;
    int32_t declared_layers = 0;
    int32_t declared_experts = 0;
    int32_t declared_expert_used = 0;
    int32_t declared_n_ff_exp = 0;
};

bool ends_with(const std::string& s, const char* suf) {
    size_t l = strlen(suf);
    return s.size() >= l && s.compare(s.size() - l, l, suf) == 0;
}

class Cursor {
public:
    explicit Cursor(std::ifstream& f) : f_(f) {}
    bool bytes(void* out, size_t n) { f_.read((char*)out, (std::streamsize)n); return (size_t)f_.gcount() == n; }
    bool u32(uint32_t& v) { return bytes(&v, 4); }
    bool u64(uint64_t& v) { return bytes(&v, 8); }
    bool u16(uint16_t& v) { return bytes(&v, 2); }
    bool u8(uint8_t& v) { return bytes(&v, 1); }
    bool str(std::string& s) {
        uint64_t n = 0;
        if (!u64(n)) return false;
        if (n > (1u << 26)) return false;  // sanity: 64 MB strings are not metadata
        s.assign((size_t)n, '\0');
        return n == 0 ? true : bytes(&s[0], (size_t)n);
    }
private:
    std::ifstream& f_;
};

bool read_scalar_u64(Cursor& c, uint32_t t, uint64_t& out) {
    if (t == 0) { uint8_t v = 0; if (!c.u8(v)) return false; out = v; return true; }
    if (t == 2) { uint16_t v = 0; if (!c.u16(v)) return false; out = v; return true; }
    if (t == 4) { uint32_t v = 0; if (!c.u32(v)) return false; out = v; return true; }
    if (t == 5) { uint32_t v = 0; if (!c.u32(v)) return false; out = (uint64_t)(int32_t)v; return true; }
    if (t == 10) { uint64_t v = 0; if (!c.u64(v)) return false; out = v; return true; }
    return false;
}

size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;   // u8 i8 bool
        case 2: case 3: return 2;           // u16 i16
        case 4: case 5: case 6: return 4;   // u32 i32 f32
        case 10: case 11: case 12: return 8;// u64 i64 f64
        default: return 0;
    }
}

bool skip_value(Cursor& c, uint32_t t) {
    if (t == 8) { std::string s; return c.str(s); }
    if (t == 9) {                        // array: read element type + count, then skip
        uint32_t et = 0; uint64_t n = 0;
        if (!c.u32(et) || !c.u64(n)) return false;
        if (et == 8) {
            for (uint64_t i = 0; i < n; i++) { std::string s; if (!c.str(s)) return false; }
            return true;
        }
        size_t sz = scalar_size(et);
        if (!sz || n > (1ull << 40)) return false;
        std::vector<char> tmp((size_t)std::min<uint64_t>(sz * n, 1u << 20));
        if (tmp.empty()) return false;
        uint64_t left = sz * n;
        while (left) {
            size_t take = (size_t)std::min<uint64_t>(left, tmp.size());
            if (!c.bytes(tmp.data(), take)) return false;
            left -= take;
        }
        return true;
    }
    size_t sz = scalar_size(t);
    if (!sz) return false;
    std::vector<char> tmp(sz);
    return c.bytes(tmp.data(), sz);
}

GgufProbe probe_gguf(const std::string& path) {
    GgufProbe out;
    std::ifstream f(path, std::ios::binary);
    if (!f) return out;
    char magic[4];
    if (!f.read(magic, 4) || memcmp(magic, "GGUF", 4) != 0) return out;
    Cursor c(f);
    uint32_t ver = 0; uint64_t nt = 0, nk = 0;
    if (!c.u32(ver) || !c.u64(nt) || !c.u64(nk)) return out;
    out.version = ver;

    // KV section: we only need general.name / general.architecture, but we must
    // walk every pair to reach the tensor table.
    for (uint64_t i = 0; i < nk; i++) {
        std::string key;
        uint32_t t = 0;
        if (!c.str(key) || !c.u32(t)) return GgufProbe{};
        bool want = (key == "general.name" || key == "general.architecture");
        if (want && t == 8) {
            std::string v;
            if (!c.str(v)) return GgufProbe{};
            if (key == "general.name") out.general_name = v;
            else out.architecture = v;
            continue;
        }
        // Scalar architecture facts, matched by suffix so the model prefix
        // (zaya., qwen3., llama., ...) does not matter.
        bool want_num = (t == 0 || t == 2 || t == 4 || t == 5 || t == 10);
        if (want_num && (ends_with(key, "expert_count") || ends_with(key, "expert_used_count") ||
                         ends_with(key, "expert_feed_forward_length") ||
                         ends_with(key, "embedding_length") || ends_with(key, "block_count"))) {
            uint64_t n = 0;
            if (!read_scalar_u64(c, t, n)) return GgufProbe{};
            int32_t v = (int32_t)n;
            if (ends_with(key, "expert_used_count")) out.declared_expert_used = v;
            else if (ends_with(key, "expert_feed_forward_length")) out.declared_n_ff_exp = v;
            else if (ends_with(key, "expert_count")) out.declared_experts = v;
            else if (ends_with(key, "embedding_length")) out.declared_hidden = v;
            else if (ends_with(key, "block_count")) out.declared_layers = v;
            continue;
        }
        if (!skip_value(c, t)) return GgufProbe{};
    }

    // Tensor table: name, n_dims, dims[], dtype, offset
    for (uint64_t i = 0; i < nt; i++) {
        std::string name;
        uint32_t nd = 0, dtype = 0;
        if (!c.str(name) || !c.u32(nd)) return GgufProbe{};
        if (nd > 8) return GgufProbe{};
        for (uint32_t d = 0; d < nd; d++) { uint64_t dim = 0; if (!c.u64(dim)) return GgufProbe{}; }
        if (!c.u32(dtype)) return GgufProbe{};
        uint64_t off = 0;
        if (!c.u64(off)) return GgufProbe{};
        out.dtypes.insert(dtype);
    }
    out.tensor_count = nt;
    out.ok = true;
    return out;
}

// ── native container probe (ONEBP header / FLM-aie-rt Q4NX JSON) ────────
//
// Raised by @agent-ec855d (2026-09-10): a 46 GiB `.1bp` — the canonical native
// format — reported dtype_space=EMPTY, which is the same value as "not probed"
// and made the registry unable to describe the format it is named after.
//
// There are in fact TWO native layouts, and both self-describe:
//   1. ONEBP_HEADER — magic 0x00504231 ("1BP\0") then a 256-byte OnebpHeader
//      carrying version/arch/quant/scale/dims/tile geometry/model_tag.
//   2. Q4NX_JSON   — NO magic: u64 JSON-header-size at offset 0, then an
//      FLM/aie-rt JSON tensor manifest. Measured instance: hsz = 232415 in
//      research/ws12-hrx-loom/README.md; documented in
//      engine/fusion/cpu_q4nx_loader.h.
// Neither carries GGUF type ids, which is why EMPTY was wrong: the space is not
// empty, it is native.
struct NativeProbe {
    DtypeSpace space = DtypeSpace::EMPTY;
    uint32_t version = 0;
    bool header_valid = false;
    std::string quant, arch, tag;
    int32_t vocab = 0;
    int32_t num_experts = 0;
    int32_t top_k = 0;
    int32_t n_ff_exp = 0;
    int32_t n_ff_shexp = 0;
    int32_t hidden = 0;
    int32_t layers = 0;
    int32_t tensor_count = 0;
    int32_t heads = 0, kv_heads = 0, head_dim = 0, interm = 0;
    float rope_theta = 0.0f;
    bool expert_fields_absent = false;
    uint64_t json_bytes = 0;
    std::vector<std::string> json_dtypes;
};

const char* onebp_quant_name(uint32_t q) {
    switch (q) {
        case ONEBP_Q4NX: return "q4nx";
        case ONEBP_I8: return "i8";
        case ONEBP_TQ1: return "tq1";
        case ONEBP_TQ2: return "tq2";
        case ONEBP_F16: return "f16";
        case ONEBP_F32: return "f32";
        case ONEBP_TQ2NZ: return "tq2nz";
        case ONEBP_TQ2NZ_E4M3: return "tq2nz_e4m3";
        case ONEBP_TQ2BS: return "tq2bs";
        case ONEBP_Q4_ROCMFP4: return "q4_rocmfp4";
        case ONEBP_Q4_ROCMFP4_FAST: return "q4_rocmfp4_fast";
        default: return nullptr;
    }
}
const char* onebp_arch_name(uint32_t a) {
    switch (a) {
        case ONEBP_DENSE: return "dense";
        case ONEBP_MOE: return "moe";
        case ONEBP_VISION: return "vision";
        case ONEBP_AUDIO: return "audio";
        case ONEBP_TERNARY: return "ternary";
        case ONEBP_MAMBA: return "mamba";
        case ONEBP_LAGUNA: return "laguna";
        case ONEBP_SMOLVLM: return "smolvlm";
        case ONEBP_LLAVA: return "llava";
        case ONEBP_MOLMO: return "molmo";
        case ONEBP_OVIS: return "ovis";
        case ONEBP_PALIGEMMA: return "paligemma";
        case ONEBP_FLORENCE: return "florence";
        case ONEBP_DEEPSEEK2: return "deepseek2";
        case ONEBP_PHI_MOE: return "phi_moe";
        case ONEBP_DEEPSEEK_V4: return "deepseek_v4";
        case ONEBP_SD: return "sd";
        default: return nullptr;
    }
}

// Collect `"dtype"` values from a Q4NX JSON manifest. The manifest is a flat
// name -> {dtype, shape, offsets} map, so a text scan is enough and a malformed
// manifest must not be fatal.
std::vector<std::string> scan_json_dtypes(const std::string& txt) {
    std::vector<std::string> out;
    const std::string key = "\"dtype\"";
    size_t p = 0;
    while ((p = txt.find(key, p)) != std::string::npos) {
        p += key.size();
        while (p < txt.size() && (txt[p] == ' ' || txt[p] == '\t' || txt[p] == ':')) p++;
        if (p < txt.size() && txt[p] == '"') {
            p++;
            std::string v;
            while (p < txt.size() && txt[p] != '"') v += txt[p++];
            if (!v.empty() && std::find(out.begin(), out.end(), v) == out.end()) out.push_back(v);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

NativeProbe probe_native(const std::string& path) {
    NativeProbe np;
    std::ifstream f(path, std::ios::binary);
    if (!f) return np;
    uint32_t w0 = 0, w1 = 0;
    if (!f.read((char*)&w0, 4) || !f.read((char*)&w1, 4)) return np;

    if (w0 == ONEBP_MAGIC) {
        OnebpHeader hdr{};
        memcpy(&hdr.magic, &w0, 4);
        memcpy(&hdr.version, &w1, 4);
        if (!f.read((char*)&hdr.arch, (std::streamsize)(sizeof(OnebpHeader) - 8))) return np;
        np.space = DtypeSpace::ONEBP_HEADER;
        np.version = hdr.version;
        np.header_valid = hdr.valid();
        np.vocab = hdr.vocab_size;
        np.num_experts = (int32_t)hdr.num_experts;
        np.top_k = (int32_t)hdr.n_expert_used;
        np.n_ff_exp = (int32_t)hdr.n_ff_exp;
        np.n_ff_shexp = (int32_t)hdr.n_ff_shexp;
        np.hidden = hdr.hidden_size;
        np.layers = hdr.num_layers;
        np.tensor_count = (int32_t)hdr.tensor_count;
        np.heads = hdr.num_attention_heads;
        np.kv_heads = hdr.num_kv_heads;
        np.head_dim = hdr.head_dim;
        np.interm = hdr.intermediate_size;
        // rope_theta: version-dependent encoding (see the header comment).
        if (hdr.version >= 3) {
            float f = 0.0f;
            memcpy(&f, &hdr.rope_theta_f, 4);   // raw f32 bits
            np.rope_theta = f;
        } else {
            np.rope_theta = (float)hdr.rope_theta_f / 1000.0f;   // fixed point
        }
        // v1 predates the expert block: zeros there are ABSENT, not DENSE.
        np.expert_fields_absent = (hdr.version < 2);
        if (const char* q = onebp_quant_name(hdr.quant)) np.quant = q;
        if (const char* a = onebp_arch_name(hdr.arch)) np.arch = a;
        np.tag.assign(hdr.model_tag, strnlen(hdr.model_tag, sizeof hdr.model_tag));
        return np;
    }

    // Not 1BP: the Q4NX layout is [u64 json_size][json][payloads].
    uint64_t hsz = ((uint64_t)w1 << 32) | (uint64_t)w0;
    if (hsz > 16 && hsz <= (64ull << 20)) {
        std::vector<char> buf((size_t)hsz + 1, '\0');
        f.clear();
        f.seekg(8);
        if (f.read(buf.data(), (std::streamsize)hsz)) {
            size_t i = 0;
            while (i < (size_t)hsz && (buf[i] == ' ' || buf[i] == '\n' || buf[i] == '\r' || buf[i] == '\t')) i++;
            if (i < (size_t)hsz && buf[i] == '{') {
                np.space = DtypeSpace::Q4NX_JSON;
                np.json_bytes = hsz;
                np.json_dtypes = scan_json_dtypes(std::string(buf.data(), (size_t)hsz));
                return np;
            }
        }
    }
    np.space = DtypeSpace::UNRECOGNIZED_NATIVE;
    return np;
}

// ── shard detection:  foo-00001-of-00002.gguf ──────────────────────────────
struct ShardInfo {
    std::string base;       // "foo"
    uint16_t index = 0;
    uint16_t count = 1;
    bool is_shard = false;
};
ShardInfo parse_shard(const std::string& stem) {
    ShardInfo si;
    si.base = stem;
    // find last occurrence of "-NNNNN-of-NNNNN"
    for (size_t pos = stem.rfind("-of-"); pos != std::string::npos; pos = stem.rfind("-of-", pos - 1)) {
        if (pos < 6 || pos + 4 + 5 != stem.size()) continue;
        std::string a = stem.substr(pos - 5, 5), b = stem.substr(pos + 4, 5);
        bool da = true, db = true;
        for (char ch : a) if (!isdigit((unsigned char)ch)) da = false;
        for (char ch : b) if (!isdigit((unsigned char)ch)) db = false;
        if (!da || !db || stem[pos - 6] != '-') continue;
        si.base = stem.substr(0, pos - 6);
        si.index = (uint16_t)std::stoi(a);
        si.count = (uint16_t)std::stoi(b);
        si.is_shard = true;
        return si;
    }
    return si;
}

std::string pad_right(const std::string& s, size_t w) {
    if (s.size() >= w) return s;
    return s + std::string(w - s.size(), ' ');
}

}  // namespace

// ── enum printing ──────────────────────────────────────────────────────────
const char* to_string(Container c) {
    switch (c) {
        case Container::GGUF: return "gguf";
        case Container::ONEBP: return "onebp";
        case Container::SAFETENSORS: return "safetensors";
        case Container::MLX: return "mlx";
        case Container::RAW_BIN: return "raw_bin";
        default: return "unknown";
    }
}
const char* to_string(DtypeSpace d) {
    switch (d) {
        case DtypeSpace::EMPTY: return "empty";
        case DtypeSpace::GGML_MAINLINE: return "mainline";
        case DtypeSpace::ENGINE_TERNARY: return "engine-ternary";
        case DtypeSpace::HRX2_Q4NX: return "hrx2-q4nx";
        case DtypeSpace::AMBIGUOUS: return "ambiguous";
        case DtypeSpace::ONEBP_HEADER: return "onebp-header";
        case DtypeSpace::Q4NX_JSON: return "q4nx-json";
        case DtypeSpace::UNRECOGNIZED_NATIVE: return "unrecognized-native";
    }
    return "unknown";
}
const char* to_string(Capability c) {
    switch (c) {
        case Capability::NPU_Q4NX: return "NPU-Q4NX";
        case Capability::NPU_1BP: return "NPU-1BP";
        case Capability::HIP_1BP: return "HIP-1BP";
        case Capability::HIP_GGUF: return "HIP-GGUF";
        case Capability::RADV_GGUF: return "RADV-GGUF";
        case Capability::HRX2_GGUF_Q4NX: return "HRX2-GGUF-Q4NX";
        case Capability::HRX_GGUF: return "HRX-GGUF";
        case Capability::MLX_GPU: return "MLX-GPU";
        case Capability::CPU: return "CPU";
        default: return "UNKNOWN";
    }
}
std::optional<Capability> capability_from_string(const std::string& s) {
    static const Capability all[] = {
        Capability::NPU_Q4NX, Capability::NPU_1BP, Capability::HIP_1BP, Capability::HIP_GGUF,
        Capability::RADV_GGUF, Capability::HRX2_GGUF_Q4NX, Capability::HRX_GGUF,
        Capability::MLX_GPU, Capability::CPU};
    for (Capability c : all) if (s == to_string(c)) return c;
    return std::nullopt;
}

const char* to_string(LimitProvenance p) {
    switch (p) {
        case LimitProvenance::DECLARED: return "declared";
        case LimitProvenance::DOCUMENTED: return "documented";
        case LimitProvenance::MEASURED: return "measured";
    }
    return "unknown";
}
const char* to_string(Enforcement e) {
    switch (e) {
        case Enforcement::NONE: return "none (report only)";
        case Enforcement::ENGINE_SERVER: return "engine server only";
        case Enforcement::ENGINE_SERVER_AND_SHIM: return "engine server + shim";
    }
    return "unknown";
}

const CapabilityLimit* capability_limit(Capability c) {
    static const CapabilityLimit limits[] = {
        // b66 over-claims FLASH_ATTN_EXT above 2048 KV. MEASURED, not estimated
        // (@agent-ca60cf, issue #2145; independently hit by @agent-44437c in the
        // P2 HRX probe lane): KV 2048 DECODES; 2304 / 2560 / 3072 fail identically
        // with `unsupported HRX node 25: FLASH_ATTN_EXT` then `compute status: -1`
        // (an earlier pass saw 2021 pass and 2067/2151/2502/2931 fail). So 2048 is
        // an upper bound on a WORKING configuration, not a conservative estimate.
        //
        // Enforcement is mode-dependent and that distinction is the point: both
        // guards live in generate_completion(), so this bites on plain `unified`
        // and `zaya` and is INERT under `--lemonade`. Only the shim fail-close in
        // src/hrx_inprocess.cpp survives there, for callers driving that in-process
        // HRX instance. The registry reports; it does not stop anything.
        {Capability::HRX_GGUF, 2048,
         "HRX b66 over-claims FLASH_ATTN_EXT for KV>2048 (issue #2145, node 25); failure signature "
         "`unsupported HRX node 25: FLASH_ATTN_EXT` + `compute status: -1`",
         LimitProvenance::MEASURED, Enforcement::ENGINE_SERVER_AND_SHIM, "--lemonade", "hrx-b66"},
    };
    // Engine override wins: the configured bundle is a runtime fact.
    for (const auto& o : capability_limit_overrides()) {
        if (o.capability == c) {
            static CapabilityLimit ov;
            ov = CapabilityLimit{c, o.max_context_tokens,
                                 "runtime override from the configured bundle",
                                 LimitProvenance::MEASURED, Enforcement::ENGINE_SERVER_AND_SHIM,
                                 nullptr, nullptr};
            static std::string bundle_hold;
            bundle_hold = o.bundle;
            ov.bundle = bundle_hold.c_str();
            return o.max_context_tokens ? &ov : nullptr;
        }
    }
    for (const auto& l : limits) if (l.capability == c) return &l;
    return nullptr;
}

// Process-wide: the configured bundle is a process fact, set once at startup.
static std::vector<CapabilityLimitOverride> g_limit_overrides;

void set_capability_limit_override(Capability c, uint32_t max_context_tokens,
                                   std::string bundle) {
    for (auto& o : g_limit_overrides) {
        if (o.capability == c) { o.max_context_tokens = max_context_tokens; o.bundle = std::move(bundle); return; }
    }
    g_limit_overrides.push_back({c, max_context_tokens, std::move(bundle)});
}
void clear_capability_limit_overrides() { g_limit_overrides.clear(); }
std::vector<CapabilityLimitOverride> capability_limit_overrides() { return g_limit_overrides; }

// ── ModelArtifact ──────────────────────────────────────────────────────────
uint64_t ModelArtifact::total_bytes() const {
    uint64_t n = 0;
    for (const auto& f : files)
        if (!f.duplicate_copy) n += f.bytes;   // unique bytes only
    return n;
}
bool ModelArtifact::has(Capability c) const {
    return std::find(capabilities.begin(), capabilities.end(), c) != capabilities.end();
}
uint32_t ModelArtifact::max_context_for(Capability c) const {
    if (!has(c)) return 0;
    const CapabilityLimit* l = capability_limit(c);
    return l ? l->max_context_tokens : 0;
}
bool ModelArtifact::supports(Capability c, uint32_t context_tokens) const {
    if (!has(c)) return false;
    uint32_t lim = max_context_for(c);
    return lim == 0 || context_tokens <= lim;
}
std::string ModelArtifact::id_quality() const {
    // "" when the canonical id preserves the original spelling (modulo case and
    // separator normalization); "(renamed)" when the id differs from every alias.
    for (const auto& a : aliases) {
        if (normalize_token(a) == normalize_token(id)) return "";
    }
    return "(renamed)";
}

// ── scan ───────────────────────────────────────────────────────────────────
namespace {

struct Pending {
    std::string base;
    Container container = Container::UNKNOWN;
    std::string quant;
    std::vector<ArtifactFile> files;
    GgufProbe probe;
    bool q4nx_named = false;
    std::string config_dir;
    std::string display_name, architecture;
};

std::vector<Capability> derive_capabilities(Container c, DtypeSpace sp, bool q4nx_named) {
    std::vector<Capability> caps;
    switch (c) {
        case Container::ONEBP:
            // A native 1BP/Q4NX artifact is served by TWO FAMILIES, not one: the
            // NPU lane (native worker engine / npu_flm) AND the HIP-1BP GPU lane
            // (hip_1bp_gpu). @agent-ca60cf made HIP-1BP the DEFAULT for qwen35moe
            // 1BP on 2026-09-10, so collapsing onebp onto NPU-only makes that lane
            // UNREACHABLE and offers a qwen35moe 1BP file to an NPU path that does
            // not implement the architecture. The family cannot be derived from
            // `arch` either (F12: the 74B's v1 header omits the expert block).
            if (q4nx_named) caps = {Capability::NPU_Q4NX, Capability::HIP_1BP};
            else caps = {Capability::NPU_1BP, Capability::HIP_1BP};
            break;
        case Container::GGUF:
            if (sp == DtypeSpace::HRX2_Q4NX) caps = {Capability::HRX2_GGUF_Q4NX};
            else if (sp == DtypeSpace::ENGINE_TERNARY) caps = {Capability::HIP_GGUF, Capability::CPU};
            else caps = {Capability::RADV_GGUF, Capability::HRX_GGUF, Capability::HIP_GGUF, Capability::CPU};
            caps.push_back(Capability::CPU);
            break;
        case Container::MLX:
            caps = {Capability::MLX_GPU, Capability::CPU};
            break;
        case Container::SAFETENSORS:
            caps = {Capability::CPU};
            break;
        case Container::RAW_BIN:
            caps = {Capability::NPU_Q4NX, Capability::CPU};
            break;
        default:
            caps = {Capability::UNKNOWN};
    }
    std::sort(caps.begin(), caps.end());
    caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
    return caps;
}

std::string guess_quant(const std::string& base_lower, const GgufProbe& probe) {
    // ROCmFP4 first: its two variants (100 = Codebook10 dual UE4M3 scales,
    // 101 = single scale) are distinguishable ONLY by dtype id — a filename
    // says "rocmfp4" for both, so the census wins here.
    if (probe.ok) {
        if (probe.dtypes.count(101)) return "q4_0_rocmfp4_fast";   // GGUF_DTYPE_Q4_0_ROCMFP4_FAST
        if (probe.dtypes.count(100)) return "q4_0_rocmfp4";        // GGUF_DTYPE_Q4_0_ROCMFP4
    }
    // Order matters: more specific markers first, so `qwen35-1bp-fp4.1bp` reads
    // as fp4 rather than falling through to the generic "1bp".
    static const char* cands[] = {
        "q4nx", "rocmfp4", "fp4", "tq2nz", "tq2", "tq1", "q1_0",
        "iq2_xxs", "iq1_s", "iq3_xxs",
        "q4_k_m", "q4_k_s", "q4_k", "q5_k_m", "q5_0", "q6_k", "q8_0",
        "q4_0", "q3_k_m", "bf16", "f16", "f32", "1bp"};
    for (const char* q : cands) if (contains(base_lower, q)) return q;
    if (probe.ok) {
        // Fall back to the tensor census when the name says nothing. Ids from
        // include/gguf_reader.h; note 42 is overloaded across readers and is
        // only reached here when the name gave no hint.
        if (probe.dtypes.count(42)) return "type42";
        if (probe.dtypes.count(41)) return "q1_0";
        if (probe.dtypes.count(35)) return "tq2_0";
        if (probe.dtypes.count(34)) return "tq1_0";
        if (probe.dtypes.count(12)) return "q4_k";
        if (probe.dtypes.count(14)) return "q6_k";
        if (probe.dtypes.count(8)) return "q8_0";
    }
    return {};
}

std::string guess_lineage(const std::string& base_lower) {
    for (const char* tag : {"ft-merged", "merged", "lora", "qat", "preview", "fixed", "fresh", "eval"}) {
        if (contains(base_lower, tag)) return tag;
    }
    return {};
}

std::string canonical_id(const std::string& base, Container c, bool q4nx_in_gguf) {
    std::string b = normalize_token(base);
    auto strip = [&b](const char* suffix) {
        size_t l = strlen(suffix);
        if (b.size() > l && b.compare(b.size() - l, l, suffix) == 0) b.erase(b.size() - l);
    };
    // Only shed a container-ish suffix from the base when the tag reproduces it.
    // (Otherwise "zaya1-8b-ft-q4nx.gguf" would collapse to "zaya1-8b-ft.gguf" even
    //  though its tensors are NOT type 42 — the name lied.)
    switch (c) {
        case Container::GGUF:
            if (q4nx_in_gguf) strip("-q4nx"); else strip("-gguf");
            break;
        case Container::ONEBP:
            if (q4nx_in_gguf) strip("-q4nx"); else strip("-1bp");
            break;
        default: break;
    }
    std::string tag;
    switch (c) {
        case Container::GGUF: tag = q4nx_in_gguf ? "q4nx-gguf" : "gguf"; break;
        case Container::ONEBP: tag = q4nx_in_gguf ? "q4nx" : "1bp"; break;
        case Container::SAFETENSORS: tag = "safetensors"; break;
        case Container::MLX: tag = "mlx"; break;
        case Container::RAW_BIN: tag = "bin"; break;
        default: tag = "unknown";
    }
    return b.empty() ? ("artifact." + tag) : (b + "." + tag);
}

}  // namespace

ModelRegistry ModelRegistry::scan(const std::vector<std::string>& roots, const ScanOptions& opt) {
    ModelRegistry reg;
    reg.roots_ = roots;

    std::unordered_map<std::string, Pending> pending;   // key: container|base
    std::vector<std::string> htok_paths;
    std::map<std::string, std::set<std::string>> consumed_safetensors_dirs;

    auto add_file = [&](const std::string& path, uint64_t sz, Container c,
                        const std::string& base_key, const std::string& base,
                        const std::string& ext, const std::string& quant,
                        const GgufProbe& probe, bool q4nx_named,
                        uint16_t shard_index, uint16_t shard_count) {
        // Key on (container, EXTENSION, base): shards of one artifact share all
        // three, while same-stem/different-container siblings (foo.q4nx vs
        // foo.1bp — very common for native conversions) must stay separate.
        std::string k = to_string(c);
        k += '|'; k += ext;
        k += '|'; k += base_key;
        auto it = pending.find(k);
        if (it == pending.end()) {
            Pending p;
            p.base = base;
            p.container = c;
            p.quant = quant;
            p.probe = probe;
            p.q4nx_named = q4nx_named;
            it = pending.emplace(k, std::move(p)).first;
        }
        if (probe.ok && !it->second.probe.ok) { it->second.probe = probe; it->second.q4nx_named = q4nx_named; }
        ArtifactFile af;
        af.path = path;
        af.bytes = sz;
        af.shard_index = shard_index;
        af.shard_count = shard_count;
        it->second.files.push_back(std::move(af));
    };

    for (const auto& root : roots) {
        std::error_code ec;
        if (!fs::exists(root, ec)) continue;

        std::vector<std::pair<fs::directory_entry, size_t>> stack;
        stack.emplace_back(root, 0);
        while (!stack.empty()) {
            fs::directory_entry de = stack.back().first;
            size_t depth = stack.back().second;
            stack.pop_back();

            if (de.is_directory(ec)) {
                // ── directory-as-artifact: MLX / HF safetensors / RAW_BIN ──
                bool has_cfg = fs::exists(de.path() / "config.json", ec);
                bool has_st = false, has_bin_sentinel = false;
                std::vector<fs::directory_entry> children;
                for (const auto& e : fs::directory_iterator(de.path(), ec)) {
                    children.push_back(e);
                    std::string n = e.path().filename().string();
                    if (contains(n, ".safetensors")) has_st = true;
                    if (n == "model_embed_tokens_weight.bin") has_bin_sentinel = true;
                }
                if (has_cfg && has_st) {
                    std::string name = de.path().filename().string();
                    std::string cfg = (de.path() / "config.json").string();
                    std::ifstream cf(cfg);
                    std::string txt((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
                    bool affine = contains(lower(txt), "\"affine\"");
                    Pending p;
                    p.base = name;
                    p.container = affine ? Container::MLX : Container::SAFETENSORS;
                    p.q4nx_named = false;
                    p.config_dir = de.path().string();
                    for (const auto& e : children) {
                        if (!e.is_regular_file(ec)) continue;
                        std::string bn = e.path().filename().string();
                        if (!contains(bn, ".safetensors")) continue;
                        ArtifactFile af;
                        af.path = e.path().string();
                        af.bytes = (uint64_t)fs::file_size(e.path(), ec);
                        p.files.push_back(std::move(af));
                    }
                    if (!p.files.empty()) {
                        std::string k = std::string(to_string(p.container)) + "|dir:" + normalize_token(name);
                        pending[k] = std::move(p);
                        consumed_safetensors_dirs[de.path().string()] = {"*"};
                    }
                    continue;  // do not descend
                }
                if (has_bin_sentinel) {
                    Pending p;
                    p.base = de.path().filename().string();
                    p.container = Container::RAW_BIN;
                    p.quant = "raw_bin";
                    std::ifstream cf(de.path() / "config.json");
                    std::string txt((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
                    p.config_dir = de.path().string();
                    for (const auto& e : children) {
                        if (!e.is_regular_file(ec)) continue;
                        std::string ext = lower(e.path().extension().string());
                        if (ext != ".bin") continue;
                        ArtifactFile af;
                        af.path = e.path().string();
                        af.bytes = (uint64_t)fs::file_size(e.path(), ec);
                        p.files.push_back(std::move(af));
                    }
                    pending[std::string("raw_bin|dir:") + normalize_token(p.base)] = std::move(p);
                    continue;
                }
                if (opt.recurse && depth < opt.max_depth) {
                    for (const auto& e : children) stack.emplace_back(e, depth + 1);
                }
                continue;
            }
            if (!de.is_regular_file(ec)) continue;

            std::string path = de.path().string();
            std::string name = de.path().filename().string();
            std::string ext = lower(de.path().extension().string());
            uint64_t sz = (uint64_t)fs::file_size(de.path(), ec);
            std::string stem = de.path().stem().string();

            if (ext == ".htok") { htok_paths.push_back(path); continue; }
            if (ext == ".gguf" || ext == ".h1b" || ext == ".q4nx" || ext == ".1bp" ||
                ext == ".safetensors" || ext == ".bin") {
                Container c = Container::UNKNOWN;
                if (ext == ".gguf" || ext == ".h1b") c = Container::GGUF;
                else if (ext == ".q4nx" || ext == ".1bp") c = Container::ONEBP;
                else if (ext == ".safetensors") c = Container::SAFETENSORS;
                else continue;  // loose .bin: ignored unless part of a RAW_BIN dir

                std::string lower_name = lower(name);
                bool q4nx_named = contains(lower_name, "q4nx");
                GgufProbe probe;
                if (opt.probe_headers && c == Container::GGUF) probe = probe_gguf(path);

                ShardInfo sh = parse_shard(stem);
                std::string base_key = normalize_token(sh.base);
                std::string quant = guess_quant(lower_name, probe);
                // Native container: keep a SPECIFIC quant from the filename
                // (fp4 / tq2nz / q4nx / …) and only fall back to the generic
                // container default when the name says nothing. Overwriting
                // unconditionally made every .1bp read as "1bp" and threw away
                // exactly the marker a lane-selection decision needs.
                if (c == Container::ONEBP) {
                    if (q4nx_named) quant = "q4nx";
                    else if (quant.empty()) quant = "1bp";
                }
                add_file(path, sz, c, base_key, sh.base, ext, quant, probe, q4nx_named,
                         sh.index, sh.count);
            }
        }
    }

    // ── materialize ────────────────────────────────────────────────────────
    for (auto& kv : pending) {
        Pending& p = kv.second;
        std::sort(p.files.begin(), p.files.end(), [](const ArtifactFile& a, const ArtifactFile& b) {
            if (a.shard_index != b.shard_index) return a.shard_index < b.shard_index;
            return a.path < b.path;
        });
        ModelArtifact a;
        a.container = p.container;
        a.display_name = p.probe.general_name;
        a.architecture = p.probe.architecture;
        a.declared_hidden = p.probe.declared_hidden;
        a.declared_layers = p.probe.declared_layers;
        a.declared_experts = p.probe.declared_experts;
        a.tensor_count = (int32_t)p.probe.tensor_count;
        a.quantization = p.quant;
        a.lineage = guess_lineage(lower(p.base));
        a.config_dir = p.config_dir;
        for (uint32_t t : p.probe.dtypes) a.dtype_ids.push_back(t);
        a.has_dtype_42 = std::find(a.dtype_ids.begin(), a.dtype_ids.end(), 42u) != a.dtype_ids.end();

        // Two flags on purpose: the ID stays filename-derived so it is stable
        // across probe success/failure, while CAPABILITY follows the bytes.
        bool q4nx_in_gguf = false;   // filename-derived  -> id
        bool q4nx_bytes = false;     // bytes-derived     -> capability/quant
        if (p.container == Container::GGUF) {
            if (a.has_dtype_42 && p.q4nx_named) { a.dtype_space = DtypeSpace::HRX2_Q4NX; q4nx_in_gguf = true; }
            else if (a.has_dtype_42) a.dtype_space = DtypeSpace::ENGINE_TERNARY;
            else if (!p.probe.ok) a.dtype_space = DtypeSpace::AMBIGUOUS;
            else a.dtype_space = DtypeSpace::GGML_MAINLINE;
            if (p.q4nx_named && p.probe.ok && !a.has_dtype_42) a.q4nx_name_mismatch = true;
            q4nx_bytes = q4nx_in_gguf;
        } else if (p.container == Container::ONEBP) {
            q4nx_in_gguf = p.q4nx_named;
            q4nx_bytes = p.q4nx_named;
            if (opt.probe_headers && !p.files.empty()) {
                NativeProbe np = probe_native(p.files.front().path);
                a.dtype_space = np.space;
                a.native_version = np.version;
                a.native_json_bytes = np.json_bytes;
                a.native_dtypes = np.json_dtypes;
                a.native_vocab = np.vocab;
                a.native_num_experts = np.num_experts;
                a.native_top_k = np.top_k;
                a.native_n_ff_exp = np.n_ff_exp;
                a.native_n_ff_shexp = np.n_ff_shexp;
                a.native_tensor_count = np.tensor_count;
                a.tensor_count = np.tensor_count;
                a.native_rope_theta = np.rope_theta;
                a.expert_fields_absent = np.expert_fields_absent;
                a.declared_hidden = np.hidden;
                a.declared_layers = np.layers;
                a.declared_experts = np.num_experts;
                a.declared_heads = np.heads;
                a.declared_kv_heads = np.kv_heads;
                a.declared_head_dim = np.head_dim;
                a.declared_interm = np.interm;
                // Internal contradiction only: arch says MOE but no experts, or
                // arch says DENSE while this same header reports experts.
                // F12 (under-declaration) cannot be seen from ONE source and is
                // caught by the sibling cross-check in scan().
                if (np.space == DtypeSpace::ONEBP_HEADER &&
                    (np.arch == "dense" || np.arch == "moe")) {
                    if ((np.arch == "moe") != (np.num_experts > 0)) a.arch_suspect = true;
                }
                if (np.space == DtypeSpace::UNRECOGNIZED_NATIVE) a.native_name_mismatch = true;
                // For native containers the bytes are authoritative: the header
                // declares the quant and arch, so neither is guessed from the
                // filename (this also retires the F8 "filename lies" class).
                if (!np.quant.empty()) a.quantization = np.quant;
                if (!np.arch.empty()) a.architecture = np.arch;
                if (!np.tag.empty() && a.display_name.empty()) a.display_name = np.tag;
                if (np.quant == "q4nx") q4nx_bytes = true;
            }
        } else {
            a.dtype_space = DtypeSpace::EMPTY;
        }

        a.id = canonical_id(p.base, p.container, q4nx_in_gguf);
        // F11: a display name that claims a DIFFERENT quantization than the
        // tensor census. Metadata, not evidence — flagged so nobody routes on it.
        if (p.probe.ok && !a.quantization.empty()) {
            std::string dn = lower(a.display_name);
            if (!dn.empty()) {
                static const char* contradicting[] = {"awq", "gptq", "int4", "int8", "fp8", "bnb"};
                for (const char* q : contradicting) {
                    if (contains(dn, q) && !contains(lower(a.quantization), q)) {
                        a.display_name_suspect = true;
                        break;
                    }
                }
            }
        }
        a.capabilities = derive_capabilities(p.container, a.dtype_space, q4nx_bytes);
        for (const auto& f : p.files) a.aliases.push_back(basename_of(f.path));
        a.files = p.files;
        if (opt.digest) {
            for (auto& f : a.files) f.digest = sha256_file(f.path);
        }
        reg.artifacts_.push_back(std::move(a));
    }

    std::sort(reg.artifacts_.begin(), reg.artifacts_.end(),
              [](const ModelArtifact& a, const ModelArtifact& b) { return a.id < b.id; });

    // ── resolve .htok through the registry (inventory F6) ──────────────────
    // Two on-disk spellings exist: "<file>.htok" and "<file minus ext>.htok".
    std::set<std::string> claimed;
    for (auto& a : reg.artifacts_) {
        for (const auto& f : a.files) {
            for (const std::string& cand : {f.path + ".htok", stem_of(f.path) + ".htok"}) {
                std::error_code ec;
                if (fs::exists(cand, ec)) { a.tokenizer_path = cand; claimed.insert(cand); break; }
                // also try sibling-of-dir spelling
                fs::path sib = fs::path(f.path).parent_path() / (fs::path(f.path).stem().string() + ".htok");
                if (fs::exists(sib, ec)) { a.tokenizer_path = sib.string(); claimed.insert(sib.string()); break; }
            }
            if (!a.tokenizer_path.empty()) break;
        }
    }

    // ── F6 follow-up: collapse PROVEN duplicates, carrying the tokenizer ────
    // Only on --digest, because merging on a guess would invent identity. The
    // identity key is the content digest; when it is absent we only report size
    // twins (see RegistryReport::size_twin_groups) and merge nothing.
    if (opt.digest) {
        std::map<std::string, size_t> by_digest;
        std::vector<char> absorbed(reg.artifacts_.size(), 0);
        // The tokenizer is NOT a criterion: the whole point is to CARRY it onto
        // the survivor, so letting it pick the survivor would invert the fix (the
        // first version did exactly that and kept `zaya1-8b-fresh.q4nx` over the
        // ADR's canonical `zaya1-8b.q4nx`). Prefer the id closest to canonical:
        // one that matches an on-disk name, then the shorter (fewer qualifiers),
        // then lexicographic for determinism.
        auto better_survivor = [](const ModelArtifact& a, const ModelArtifact& b) {
            bool a_clean = a.id_quality().empty(), b_clean = b.id_quality().empty();
            if (a_clean != b_clean) return a_clean;
            if (a.id.size() != b.id.size()) return a.id.size() < b.id.size();
            return a.id < b.id;
        };
        for (size_t i = 0; i < reg.artifacts_.size(); ++i) {
            if (reg.artifacts_[i].files.size() != 1) continue;
            const std::string& dg = reg.artifacts_[i].files[0].digest;
            if (dg.empty()) continue;
            auto it = by_digest.find(dg);
            if (it == by_digest.end()) { by_digest[dg] = i; continue; }
            size_t keep = it->second, drop = i;
            if (better_survivor(reg.artifacts_[drop], reg.artifacts_[keep])) std::swap(keep, drop);
            ModelArtifact& K = reg.artifacts_[keep];
            const ModelArtifact D = reg.artifacts_[drop];   // copy: we mutate K only
            K.merged_ids.push_back(D.id);
            // Keep the absorbed files so their PATHS stay resolvable — a catalog or
            // a caller may still name the duplicate.
            for (auto f : D.files) { f.duplicate_copy = true; K.files.push_back(f); }
            for (const auto& al : D.aliases)
                if (std::find(K.aliases.begin(), K.aliases.end(), al) == K.aliases.end())
                    K.aliases.push_back(al);
            if (K.aliases.end() == std::find(K.aliases.begin(), K.aliases.end(), D.id))
                K.aliases.push_back(D.id);
            if (K.tokenizer_path.empty() && !D.tokenizer_path.empty()) {
                K.tokenizer_path = D.tokenizer_path;
                K.tokenizer_from_duplicate = true;
            }
            reg.reclaimed_bytes_ += D.total_bytes();
            reg.merged_artifacts_++;
            absorbed[drop] = 1;
            by_digest[dg] = keep;
        }
        std::vector<ModelArtifact> kept;
        kept.reserve(reg.artifacts_.size());
        for (size_t i = 0; i < reg.artifacts_.size(); ++i)
            if (!absorbed[i]) kept.push_back(std::move(reg.artifacts_[i]));
        reg.artifacts_.swap(kept);
    }

    for (const auto& h : htok_paths)
        if (!claimed.count(h)) reg.dangling_tokenizers_++;

    // ── F12: native header vs dims-identical sibling GGUF ──────────────────
    // A native header that declares DENSE with zero experts may simply be
    // SILENT, not correct: 0 is a meaningful value for both OnebpArch and
    // OnebpQuant, so an unwritten header reads as "dense Q4NX" and earns
    // NPU-Q4NX. The only way to catch it is a second source. We do not assert —
    // we compare declared dims and report the disagreement.
    for (auto& nat : reg.artifacts_) {
        if (nat.container != Container::ONEBP) continue;
        if (nat.dtype_space != DtypeSpace::ONEBP_HEADER) continue;
        if (nat.native_num_experts > 0) continue;         // header is explicit
        if (!nat.declared_hidden || !nat.declared_layers) continue;
        for (const auto& sib : reg.artifacts_) {
            if (sib.container != Container::GGUF) continue;
            if (sib.declared_experts <= 0) continue;
            if (sib.declared_hidden != nat.declared_hidden) continue;
            if (sib.declared_layers != nat.declared_layers) continue;
            // STRONG key first: identical tensor COUNT means the same file layout,
            // which expert packing would change. Dims alone are weak — two
            // same-base variants can share them (@agent-ec855d's refinement).
            bool tc_known = (nat.native_tensor_count > 0 && sib.tensor_count > 0);
            if (tc_known && nat.native_tensor_count != sib.tensor_count) continue;
            nat.experts_underdeclared = true;
            break;
        }
    }

    // ── F12b: declared geometry vs the file's own size (F32 hard bound) ─────
    // Independent of any sibling. See the field comment in the header for the
    // formula and for why F32 (not a chosen threshold) is the bound that matters.
    for (auto& a : reg.artifacts_) {
        if (a.dtype_space != DtypeSpace::ONEBP_HEADER) continue;
        if (a.architecture != "dense" || a.declared_experts > 0) continue;
        int32_t H = a.declared_hidden, L = a.declared_layers, V = a.native_vocab;
        if (H <= 0 || L <= 0 || V <= 0) continue;
        int32_t q_dim = a.declared_head_dim > 0 && a.declared_heads > 0
                            ? a.declared_heads * a.declared_head_dim
                            : H;
        int32_t kv_dim = a.declared_kv_heads > 0 && a.declared_head_dim > 0
                             ? a.declared_kv_heads * a.declared_head_dim
                             : H;
        int32_t I = a.declared_interm > 0 ? a.declared_interm : H;
        // UNTIED lm_head assumed: 2 * vocab * hidden. A HARD bound must never
        // under-estimate the declared capacity, and @agent-ec855d showed the tie
        // term is 11.4% of the base geometry - larger than the 10% slack - so
        // counting the embedding once would flag an honest untied dense F32
        // export of exactly this geometry. (Their check on the real file: the
        // sibling GGUF has token_embd.weight and ZERO output.weight / lm_head
        // tensors out of 1923, so THIS file is tied and the generous assumption
        // simply costs margin.)
        double params = 2.0 * (double)V * H + (double)L * ((double)H * q_dim +
                        (double)H * kv_dim + (double)kv_dim * H + (double)q_dim * H +
                        3.0 * (double)H * I);
        double f32_bytes = params * 4.0;
        double slack = 1.10;   // small extra tensors, headers, alignment
        if (f32_bytes > 0) {
            a.geometry_bound_ratio = (double)a.total_bytes() / f32_bytes;
            if (a.geometry_bound_ratio > slack) a.geometry_cannot_hold_file = true;
        }
    }

    // ── id collisions: two artifacts claiming the same canonical id ────────
    std::map<std::string, int> id_count;
    for (const auto& a : reg.artifacts_) id_count[a.id]++;
    // Disambiguate deterministically so `resolve_path` is never ambiguous.
    std::map<std::string, int> seen;
    for (auto& a : reg.artifacts_) {
        if (id_count[a.id] > 1) {
            int n = ++seen[a.id];
            if (n > 1) a.id += "-" + std::to_string(n);
        }
    }
    return reg;
}

// ── queries ────────────────────────────────────────────────────────────────
const ModelArtifact* ModelRegistry::find(const std::string& id_or_alias) const {
    for (const auto& a : artifacts_) {
        if (a.id == id_or_alias) return &a;
        for (const auto& al : a.aliases) if (al == id_or_alias) return &a;
    }
    return nullptr;
}

const ModelArtifact* ModelRegistry::resolve_path(const std::string& path) const {
    std::string want = fs::path(path).filename().string();
    for (const auto& a : artifacts_)
        for (const auto& f : a.files)
            if (f.path == path || basename_of(f.path) == want) return &a;
    for (const auto& a : artifacts_)
        if (a.id == path) return &a;
    return nullptr;
}

std::vector<const ModelArtifact*> ModelRegistry::with_capability(Capability c) const {
    std::vector<const ModelArtifact*> out;
    for (const auto& a : artifacts_) if (a.has(c)) out.push_back(&a);
    return out;
}

RouteDecision ModelRegistry::resolve(const RouteRequest& req) const {
    RouteDecision d;
    d.context_tokens = req.context_tokens;

    const ModelArtifact* a = resolve_path(req.target);
    if (!a) a = find(req.target);
    if (!a) {
        d.reason = "no artifact matches '" + req.target + "'";
        return d;
    }
    d.artifact = a;

    std::vector<Capability> order = req.prefer.empty() ? a->capabilities : req.prefer;
    for (Capability c : order) {
        if (!a->has(c)) {
            d.rejected.emplace_back(c, "not available on this artifact");
            continue;
        }
        if (req.context_tokens && !a->supports(c, req.context_tokens)) {
            uint32_t lim = a->max_context_for(c);
            d.rejected.emplace_back(c, "context " + std::to_string(req.context_tokens) +
                                           " exceeds limit " + std::to_string(lim) +
                                           (capability_limit(c) && capability_limit(c)->not_enforced_in
                                                ? std::string(" (and that limit is not enforced in ") +
                                                      capability_limit(c)->not_enforced_in + ")"
                                                : std::string()));
            continue;
        }
        d.resolved = true;
        d.chosen = c;
        d.limit_binding = (a->max_context_for(c) != 0);
        d.reason = "first capability in order that can serve";
        // Everything after the winner was not considered; say so rather than
        // leaving it silently absent.
        return d;
    }
    d.reason = "no capability can serve";
    return d;
}

std::vector<const ModelArtifact*> ModelRegistry::serve_at(Capability c,
                                                          uint32_t context_tokens) const {
    std::vector<const ModelArtifact*> out;
    for (const auto& a : artifacts_) if (a.supports(c, context_tokens)) out.push_back(&a);
    return out;
}

CatalogView ModelRegistry::attach_catalog(const std::string& json_path) {
    CatalogView view;
    view.path = json_path;
    // A catalog path can be wrong (missing, a directory, unreadable). None of
    // those may abort the registry: a view is optional, the artifacts are not.
    std::error_code ec;
    if (!fs::is_regular_file(json_path, ec)) return view;
    std::string txt;
    try {
        std::ifstream f(json_path, std::ios::binary);
        if (!f) return view;
        txt.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    } catch (const std::exception&) {
        return view;   // e.g. libstdc++ throws ios_failure on a directory read
    }

    std::vector<MiniJson::Entry> items;
    MiniJson p(txt);
    if (!p.top_object(items)) return view;
    view.parse_ok = true;

    for (auto& item : items) {
        CatalogEntry e;
        e.catalog_id = item.first;
        auto field = [&](const char* k) -> std::string {
            auto it = item.second.find(k);
            return it == item.second.end() ? std::string() : it->second;
        };
        e.checkpoint = field("checkpoint");
        e.recipe = field("recipe");
        e.source = field("source");
        view.entries.push_back(e);

        // A catalog may only ADD AN ALIAS to an artifact we already found. It
        // must never create one, or the catalog becomes a second source of
        // record (R1).
        ModelArtifact* hit = nullptr;
        if (!e.checkpoint.empty()) {
            for (auto& a : artifacts_) {
                for (const auto& af : a.files) {
                    if (af.path == e.checkpoint ||
                        basename_of(af.path) == basename_of(e.checkpoint)) {
                        hit = &a;
                        break;
                    }
                }
                if (hit) break;
            }
        }
        if (hit) {
            if (std::find(hit->catalog_ids.begin(), hit->catalog_ids.end(), e.catalog_id) ==
                hit->catalog_ids.end()) {
                hit->catalog_ids.push_back(e.catalog_id);
                hit->aliases.push_back(e.catalog_id);
            }
            view.resolved++;
        } else {
            view.unknown++;
        }
    }
    catalogs_.push_back(std::move(view));
    return catalogs_.back();
}

const ModelArtifact* ModelRegistry::prefer(const std::vector<const ModelArtifact*>& cands,
                                           Capability p) {
    for (const auto* a : cands) if (a && a->has(p)) return a;
    return cands.empty() ? nullptr : cands.front();
}

RegistryReport ModelRegistry::report() const {
    RegistryReport r;
    r.artifacts = artifacts_.size();
    r.total_bytes = 0;
    std::map<std::string, int> id_count;
    std::map<uint64_t, int> size_count;
    std::map<std::string, int> digest_count;
    for (const auto& a : artifacts_) {
        r.total_bytes += a.total_bytes();
        r.files += a.files.size();
        if (a.is_sharded()) r.sharded_artifacts++;
        id_count[normalize_token(a.id)]++;
        size_count[a.total_bytes()]++;
        for (const auto& f : a.files)
            if (!f.digest.empty()) digest_count[f.digest]++;
    }
    for (const auto& kv : id_count) if (kv.second > 1) r.duplicate_id_groups++;
    for (const auto& kv : size_count) if (kv.second > 1) r.size_twin_groups++;
    // reclaimable bytes are only *proven* when digests are on
    for (const auto& a : artifacts_) {
        if (a.files.size() != 1) continue;
        const auto& d = a.files[0].digest;
        if (!d.empty() && digest_count[d] > 1) r.duplicate_bytes += a.files[0].bytes;
    }
    r.dangling_tokenizers = dangling_tokenizers_;
    r.merged_artifacts = merged_artifacts_;
    r.reclaimed_bytes = reclaimed_bytes_;
    return r;
}

// ── rendering ──────────────────────────────────────────────────────────────
RegistryDelta ModelRegistry::diff(const ModelRegistry& prev) const {
    RegistryDelta d;
    auto key = [](const ModelArtifact& a) {
        return std::to_string(a.total_bytes()) + "/" + std::to_string(a.tensor_count) + "/" +
               std::to_string(a.files.size()) + "/" + a.quantization;
    };
    std::map<std::string, std::string> prev_by_id;
    for (const auto& a : prev.artifacts_) prev_by_id[a.id] = key(a);
    std::set<std::string> now;
    for (const auto& a : artifacts_) {
        now.insert(a.id);
        auto it = prev_by_id.find(a.id);
        if (it == prev_by_id.end()) d.added.push_back(a.id);
        else if (it->second != key(a)) d.changed.push_back(a.id);
    }
    for (const auto& kv : prev_by_id)
        if (!now.count(kv.first)) d.removed.push_back(kv.first);
    return d;
}

std::string ModelRegistry::to_table(uint32_t at_context) const {
    gate_context_ = at_context;
    std::ostringstream o;
    o << pad_right("id", 46) << " " << pad_right("container", 10) << " "
      << pad_right("dtype-space", 14) << " " << pad_right("GiB", 9) << " "
      << pad_right("caps", 6) << "capabilities\n";
    o << std::string(46 + 1 + 10 + 1 + 14 + 1 + 9 + 1 + 6, '-') << "\n";
    if (at_context)
        o << "! = cannot serve " << at_context << " context tokens\n";
    // Constraint footnotes: a limit is a REPORT, and its enforcement depends on
    // the serving mode. Rendering only "(<=2048)" invited reading it as a gate.
    for (const auto& l : {Capability::HRX_GGUF}) {
        const CapabilityLimit* lim = capability_limit(l);
        if (!lim) continue;
        o << "constraint " << to_string(l) << " <= " << lim->max_context_tokens
          << " ctx  [" << to_string(lim->provenance)
          << (lim->bundle ? std::string(", bundle ") + lim->bundle : std::string()) << "]"
          << "  enforced by: " << to_string(lim->enforced_by)
          << "  NOT enforced in: " << (lim->not_enforced_in ? lim->not_enforced_in : "-")
          << "  (this registry reports, it does not gate; the engine may override per bundle)\n";
    }
    for (const auto& a : artifacts_) {
        std::string caps;
        size_t disqualified = 0;
        for (size_t i = 0; i < a.capabilities.size(); i++) {
            if (i) caps += ",";
            caps += to_string(a.capabilities[i]);
            const CapabilityLimit* l = capability_limit(a.capabilities[i]);
            if (l) caps += "(<=" + std::to_string(l->max_context_tokens) + ")";
            if (at_context && !a.supports(a.capabilities[i], at_context)) {
                caps += "!";
                disqualified++;
            }
        }
        if (at_context && disqualified == a.capabilities.size()) caps += " [not-servable]";
        char nbuf[32];
        snprintf(nbuf, sizeof nbuf, "%.2f",
                 (double)a.total_bytes() / (1024.0 * 1024.0 * 1024.0));
        o << pad_right(a.id, 46) << " " << pad_right(to_string(a.container), 10) << " "
          << pad_right(to_string(a.dtype_space), 14) << " " << pad_right(nbuf, 9) << " "
          << pad_right(std::to_string(a.capabilities.size()), 6) << caps
          << (a.has_dtype_42 ? "  [type42!]" : "")
          << (a.q4nx_name_mismatch ? "  [name-says-q4nx-no-type42]" : "")
          << (a.native_name_mismatch ? "  [native-name-mismatch]" : "")
          << (a.arch_suspect ? "  [arch-suspect]" : "")
          << (a.experts_underdeclared ? "  [experts-underdeclared!]" : "")
          << (a.expert_fields_absent ? "  [expert-fields-absent-v1]" : "")
          << (a.geometry_cannot_hold_file ? "  [geometry-cannot-hold-file!]" : "")
          << (a.display_name_suspect ? "  [display-name-suspect]" : "")
          << (a.id_quality().empty() ? "" : "  " + a.id_quality()) << "\n";
    }
    return o.str();
}

std::string ModelRegistry::to_json() const {
    std::ostringstream o;
    o << "{\n  \"roots\": [";
    for (size_t i = 0; i < roots_.size(); i++) {
        if (i) o << ", ";
        o << "\"" << json_escape(roots_[i]) << "\"";
    }
    o << "],\n  \"artifacts\": [\n";
    for (size_t ai = 0; ai < artifacts_.size(); ai++) {
        const auto& a = artifacts_[ai];
        o << "    {\n";
        o << "      \"id\": \"" << json_escape(a.id) << "\",\n";
        o << "      \"aliases\": [";
        for (size_t i = 0; i < a.aliases.size(); i++) {
            if (i) o << ", ";
            o << "\"" << json_escape(a.aliases[i]) << "\"";
        }
        o << "],\n";
        o << "      \"container\": \"" << to_string(a.container) << "\",\n";
        o << "      \"dtype_space\": \"" << to_string(a.dtype_space) << "\",\n";
        o << "      \"dtype_ids\": [";
        for (size_t i = 0; i < a.dtype_ids.size(); i++) {
            if (i) o << ", ";
            o << a.dtype_ids[i];
        }
        o << "],\n";
        o << "      \"quantization\": \"" << json_escape(a.quantization) << "\",\n";
        o << "      \"architecture\": \"" << json_escape(a.architecture) << "\",\n";
        o << "      \"display_name\": \"" << json_escape(a.display_name) << "\",\n";
        o << "      \"display_name_suspect\": " << (a.display_name_suspect ? "true" : "false") << ",\n";
        // Source-agnostic declared architecture facts (GGUF metadata or native
        // header). Top level because they are not native-specific.
        o << "      \"declared_hidden\": " << a.declared_hidden
          << ", \"declared_layers\": " << a.declared_layers
          << ", \"declared_experts\": " << a.declared_experts << ",\n";
        o << "      \"lineage\": \"" << json_escape(a.lineage) << "\",\n";
        o << "      \"has_dtype_42\": " << (a.has_dtype_42 ? "true" : "false")
          << ", \"q4nx_name_mismatch\": " << (a.q4nx_name_mismatch ? "true" : "false") << ",\n";
        o << "      \"native\": {\"version\": " << a.native_version
          << ", \"vocab\": " << a.native_vocab
          << ", \"num_experts\": " << a.native_num_experts
          << ", \"top_k\": " << a.native_top_k
          << ", \"arch_suspect\": " << (a.arch_suspect ? "true" : "false")
          << ", \"experts_underdeclared\": " << (a.experts_underdeclared ? "true" : "false")
          << ", \"n_ff_exp\": " << a.native_n_ff_exp
          << ", \"n_ff_shexp\": " << a.native_n_ff_shexp
          << ", \"tensor_count\": " << a.native_tensor_count
          << ", \"rope_theta\": " << a.native_rope_theta
          << ", \"expert_fields_absent\": " << (a.expert_fields_absent ? "true" : "false")
          << ", \"geometry_cannot_hold_file\": " << (a.geometry_cannot_hold_file ? "true" : "false")
          << ", \"geometry_bound_ratio\": " << a.geometry_bound_ratio
          << ", \"geometry_bound_note\": \"file_bytes/(declared_params*4); untied lm_head assumed; flags above 1.10\""
          << ", \"json_bytes\": " << a.native_json_bytes
          << ", \"name_mismatch\": " << (a.native_name_mismatch ? "true" : "false")
          << ", \"dtypes\": [";
        for (size_t i = 0; i < a.native_dtypes.size(); i++) {
            if (i) o << ", ";
            o << "\"" << json_escape(a.native_dtypes[i]) << "\"";
        }
        o << "]},\n";
        o << "      \"tokenizer\": \"" << json_escape(a.tokenizer_path) << "\",\n";
        o << "      \"capabilities\": [";
        for (size_t i = 0; i < a.capabilities.size(); i++) {
            if (i) o << ", ";
            o << "\"" << to_string(a.capabilities[i]) << "\"";
        }
        o << "],\n";
        o << "      \"capability_limits\": {";
        bool first_lim = true;
        for (auto c : a.capabilities) {
            const CapabilityLimit* l = capability_limit(c);
            if (!l) continue;
            if (!first_lim) o << ", ";
            first_lim = false;
            o << "\"" << to_string(c) << "\": {\"max_context_tokens\": " << l->max_context_tokens
              << ", \"provenance\": \"" << to_string(l->provenance)
              << "\", \"enforced_by\": \"" << to_string(l->enforced_by)
              << "\", \"not_enforced_in\": \"" << (l->not_enforced_in ? l->not_enforced_in : "")
              << "\", \"bundle\": \"" << (l->bundle ? l->bundle : "")
              << "\"}";
        }
        o << "},\n";
        o << "      \"merged_ids\": [";
        for (size_t i = 0; i < a.merged_ids.size(); i++) {
            if (i) o << ", ";
            o << "\"" << json_escape(a.merged_ids[i]) << "\"";
        }
        o << "],\n";
        o << "      \"tokenizer_from_duplicate\": "
          << (a.tokenizer_from_duplicate ? "true" : "false") << ",\n";
        o << "      \"catalog_ids\": [";
        for (size_t i = 0; i < a.catalog_ids.size(); i++) {
            if (i) o << ", ";
            o << "\"" << json_escape(a.catalog_ids[i]) << "\"";
        }
        o << "],\n";
        o << "      \"total_bytes\": " << a.total_bytes() << ",\n";
        o << "      \"files\": [\n";
        for (size_t i = 0; i < a.files.size(); i++) {
            const auto& f = a.files[i];
            o << "        {\"path\": \"" << json_escape(f.path) << "\", \"bytes\": "
              << f.bytes << ", \"shard_index\": " << f.shard_index
              << ", \"shard_count\": " << f.shard_count;
            if (!f.digest.empty()) o << ", \"sha256\": \"" << f.digest << "\"";
            o << "}" << (i + 1 < a.files.size() ? "," : "") << "\n";
        }
        o << "      ]\n    }" << (ai + 1 < artifacts_.size() ? "," : "") << "\n";
    }
    o << "  ],\n  \"catalogs\": [";
    for (size_t ci = 0; ci < catalogs_.size(); ci++) {
        const auto& cv = catalogs_[ci];
        if (ci) o << ",\n";
        o << "\n    {\"path\": \"" << json_escape(cv.path) << "\", \"parse_ok\": "
          << (cv.parse_ok ? "true" : "false") << ", \"entries\": " << cv.entries.size()
          << ", \"resolved\": " << cv.resolved << ", \"unknown\": " << cv.unknown << "}";
    }
    o << "\n  ],\n  \"report\": {";
    RegistryReport r = report();
    o << "\"artifacts\": " << r.artifacts << ", \"files\": " << r.files
      << ", \"sharded\": " << r.sharded_artifacts
      << ", \"total_bytes\": " << r.total_bytes
      << ", \"duplicate_id_groups\": " << r.duplicate_id_groups
      << ", \"size_twin_groups\": " << r.size_twin_groups
      << ", \"duplicate_bytes\": " << r.duplicate_bytes
      << ", \"gate_context\": " << gate_context_
      << ", \"merged_artifacts\": " << r.merged_artifacts
      << ", \"reclaimed_bytes\": " << r.reclaimed_bytes
      << ", \"gate_enforces\": false}\n}\n";
    return o.str();
}

}  // namespace onebit
