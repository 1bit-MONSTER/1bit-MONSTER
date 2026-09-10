// src/model_registry.cpp — artifact-first registry implementation.
// See include/model_registry.h for the contract and the ADR references.
//
// Self-contained: std library only. The GGUF probe here is intentionally
// minimal (header + tensor dtype census + general.* strings) so this module
// stays linkable without HIP/XRT and can be unit-tested standalone. When the
// engine wires this in, ScanOptions::probe_headers can be swapped for
// read_gguf_metadata() from include/model_discovery.h.
#include "model_registry.h"

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

// ── minimal GGUF header probe ──────────────────────────────────────────────
struct GgufProbe {
    bool ok = false;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    std::set<uint32_t> dtypes;
    std::string general_name;
    std::string architecture;
};

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

// ── ModelArtifact ──────────────────────────────────────────────────────────
uint64_t ModelArtifact::total_bytes() const {
    uint64_t n = 0;
    for (const auto& f : files) n += f.bytes;
    return n;
}
bool ModelArtifact::has(Capability c) const {
    return std::find(capabilities.begin(), capabilities.end(), c) != capabilities.end();
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
            if (q4nx_named) caps = {Capability::NPU_Q4NX};
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
    static const char* cands[] = {"q4_k_m","q4_k_s","q4_k","q6_k","q8_0","q5_k_m","q5_0",
                                  "q4_0","q3_k_m","iq2_xxs","f16","bf16","f32",
                                  "q4nx","tq2","tq1","1bp","q1_0"};
    for (const char* q : cands) if (contains(base_lower, q)) return q;
    if (probe.ok) {
        if (probe.dtypes.count(42)) return "type42";
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
                        const std::string& quant, const GgufProbe& probe, bool q4nx_named,
                        uint16_t shard_index, uint16_t shard_count) {
        std::string k = to_string(c);
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
                if (c == Container::ONEBP && !q4nx_named) quant = "1bp";
                if (c == Container::ONEBP && q4nx_named) quant = "q4nx";
                add_file(path, sz, c, base_key, sh.base, quant, probe, q4nx_named,
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
        a.quantization = p.quant;
        a.lineage = guess_lineage(lower(p.base));
        a.config_dir = p.config_dir;
        for (uint32_t t : p.probe.dtypes) a.dtype_ids.push_back(t);
        a.has_dtype_42 = std::find(a.dtype_ids.begin(), a.dtype_ids.end(), 42u) != a.dtype_ids.end();

        bool q4nx_in_gguf = false;
        if (p.container == Container::GGUF) {
            if (a.has_dtype_42 && p.q4nx_named) { a.dtype_space = DtypeSpace::HRX2_Q4NX; q4nx_in_gguf = true; }
            else if (a.has_dtype_42) a.dtype_space = DtypeSpace::ENGINE_TERNARY;
            else if (!p.probe.ok) a.dtype_space = DtypeSpace::AMBIGUOUS;
            else a.dtype_space = DtypeSpace::GGML_MAINLINE;
            if (p.q4nx_named && p.probe.ok && !a.has_dtype_42) a.q4nx_name_mismatch = true;
        } else if (p.container == Container::ONEBP) {
            q4nx_in_gguf = p.q4nx_named;
        } else {
            a.dtype_space = DtypeSpace::EMPTY;
        }

        a.id = canonical_id(p.base, p.container, q4nx_in_gguf);
        a.capabilities = derive_capabilities(p.container, a.dtype_space, q4nx_in_gguf);
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

    for (const auto& h : htok_paths)
        if (!claimed.count(h)) reg.dangling_tokenizers_++;

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
    return r;
}

// ── rendering ──────────────────────────────────────────────────────────────
std::string ModelRegistry::to_table() const {
    std::ostringstream o;
    o << pad_right("id", 46) << " " << pad_right("container", 10) << " "
      << pad_right("dtype-space", 14) << " " << pad_right("GiB", 9) << " "
      << pad_right("caps", 6) << "capabilities\n";
    o << std::string(46 + 1 + 10 + 1 + 14 + 1 + 9 + 1 + 6, '-') << "\n";
    for (const auto& a : artifacts_) {
        std::string caps;
        for (size_t i = 0; i < a.capabilities.size(); i++) {
            if (i) caps += ",";
            caps += to_string(a.capabilities[i]);
        }
        char nbuf[32];
        snprintf(nbuf, sizeof nbuf, "%.2f",
                 (double)a.total_bytes() / (1024.0 * 1024.0 * 1024.0));
        o << pad_right(a.id, 46) << " " << pad_right(to_string(a.container), 10) << " "
          << pad_right(to_string(a.dtype_space), 14) << " " << pad_right(nbuf, 9) << " "
          << pad_right(std::to_string(a.capabilities.size()), 6) << caps
          << (a.has_dtype_42 ? "  [type42!]" : "")
          << (a.q4nx_name_mismatch ? "  [name-says-q4nx-no-type42]" : "")
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
        o << "      \"lineage\": \"" << json_escape(a.lineage) << "\",\n";
        o << "      \"has_dtype_42\": " << (a.has_dtype_42 ? "true" : "false")
          << ", \"q4nx_name_mismatch\": " << (a.q4nx_name_mismatch ? "true" : "false") << ",\n";
        o << "      \"tokenizer\": \"" << json_escape(a.tokenizer_path) << "\",\n";
        o << "      \"capabilities\": [";
        for (size_t i = 0; i < a.capabilities.size(); i++) {
            if (i) o << ", ";
            o << "\"" << to_string(a.capabilities[i]) << "\"";
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
    o << "  ],\n  \"report\": {";
    RegistryReport r = report();
    o << "\"artifacts\": " << r.artifacts << ", \"files\": " << r.files
      << ", \"sharded\": " << r.sharded_artifacts
      << ", \"total_bytes\": " << r.total_bytes
      << ", \"duplicate_id_groups\": " << r.duplicate_id_groups
      << ", \"size_twin_groups\": " << r.size_twin_groups
      << ", \"duplicate_bytes\": " << r.duplicate_bytes << "}\n}\n";
    return o.str();
}

}  // namespace onebit
