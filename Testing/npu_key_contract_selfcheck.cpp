// npu_key_contract_selfcheck.cpp — host-only checks for the two defects behind
// issue #2193's false "GEMM weights missing" report:
//
//   1. Q4nxReader::find_offset searched only the first 64 KB of the file, so
//      every artifact with a larger JSON header silently lost its later tensor
//      entries (zaya1-8b's header is 232,415 B; Gemma4-E4B-IT 97,256 B).
//   2. The parent backend verified one hardcoded dense key set. Zaya's worker
//      loads split-V attention (v_proj_current / v_proj_delayed) and MoE FFN
//      (mlp.experts.*), so the dense check reported a missing weight for a
//      model that serves.
//
// Build + run (no NPU, no device, no third-party package):
//   g++ -std=c++17 -Iinclude -Isrc -O2 Testing/npu_key_contract_selfcheck.cpp \
//       src/q4nx_reader.cpp -o /tmp/npu_key_contract_selfcheck
//   /tmp/npu_key_contract_selfcheck
#include "q4nx_reader.h"
#include "npu_key_contract.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

static int checks = 0, fails = 0;
#define CHECK(cond, msg) do { ++checks; if (!(cond)) { std::printf("  FAIL %s\n", msg); ++fails; } } while (0)

// Build a synthetic Q4NX in /tmp: {"num_hidden_layers":N,... "model_type":<mt>,
// "model.layers.L.<suffix>":{"shape":[1],"data_offsets":[0,2]}, ...}. The
// header is padded past `pad_to` bytes when asked, which is the regression
// input for the 64 KB window.
static std::string write_artifact(const char* path, const std::string& model_type,
                                  int layers, const std::vector<std::string>& suffixes,
                                  size_t pad_to = 0, const char* prefix = "model.layers.") {
    std::string h = "{\n";
    h += "  \"hidden_size\": 64,\n";
    h += "  \"num_hidden_layers\": " + std::to_string(layers) + ",\n";
    if (!model_type.empty()) h += "  \"model_type\": \"" + model_type + "\",\n";
    if (pad_to) {
        // A long unrelated string value — the header grows, the tensors below
        // move past the old 64 KB search window.
        h += "  \"pad\": \"" + std::string(pad_to, 'x') + "\",\n";
    }
    std::vector<std::string> keys;
    for (int l = 0; l < layers; l++)
        for (const auto& s : suffixes)
            keys.push_back(std::string(prefix) + std::to_string(l) + "." + s);
    for (size_t i = 0; i < keys.size(); i++) {
        h += "  \"" + keys[i] + "\": {\"dtype\": \"BF16\", \"shape\": [1, 1], "
             "\"data_offsets\": [" + std::to_string(i * 2) + ", " + std::to_string(i * 2 + 2) + "]}";
        if (i + 1 < keys.size()) h += ",";
        h += "\n";
    }
    h += "}\n";

    FILE* f = std::fopen(path, "wb");
    if (!f) return std::string();
    uint64_t hl = h.size();
    std::fwrite(&hl, 8, 1, f);
    std::fwrite(h.data(), 1, h.size(), f);
    const char two[2] = {0, 0};
    for (size_t i = 0; i < keys.size(); i++) std::fwrite(two, 1, 2, f);  // stub payload
    std::fclose(f);
    return h;
}

int main() {
    const char* dense_path = "/tmp/npu_key_contract_dense.q4nx";
    const char* big_path   = "/tmp/npu_key_contract_big.q4nx";
    const char* zaya_path  = "/tmp/npu_key_contract_zaya.q4nx";
    const char* odd_path   = "/tmp/npu_key_contract_odd.q4nx";

    // ── 1. the header window ──
    // 8 layers of dense suffixes sit well past 64 KB only when padded; with the
    // old window find_offset() saw none of them.
    {
        std::string hdr = write_artifact(big_path, "", 8, {"self_attn.q_proj.weight"}, 70000);
        CHECK(hdr.size() > 65536, "probe artifact header is larger than the old 64 KB window");
        Q4nxReader r;
        CHECK(r.open(big_path), "open large-header artifact");
        CHECK(r.find_offset("model.layers.0.self_attn.q_proj.weight") != 0,
              "layer 0 key found in a large header");
        CHECK(r.find_offset("model.layers.7.self_attn.q_proj.weight") != 0,
              "layer 7 key found in a large header (past the old 64 KB window)");
        CHECK(r.find_offset("model.layers.7.self_attn.nope.weight") == 0,
              "a key that is not in the file is still reported absent");
        r.close();
    }

    // ── 2. the contract: dense (no model_type declared) ──
    {
        write_artifact(dense_path, "", 2,
                       {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                        "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                        "mlp.down_proj.weight"});
        Q4nxReader r;
        CHECK(r.open(dense_path), "open dense artifact");
        CHECK(r.model_type().empty(), "dense artifact declares no model_type");
        NpuKeyContract c = npu_contract_for(r.model_type());
        CHECK(c.layout && std::strcmp(c.layout, "dense") == 0, "empty model_type -> dense contract");
        NpuKeyCheck ck = npu_verify_layer_keys(r, 2, c);
        CHECK(ck.status == NpuKeyCheck::Verified, "dense vocabulary verifies against the dense contract");
        r.close();
    }

    // ── 3. the contract: zaya declares its family, and its vocabulary passes ──
    {
        write_artifact(zaya_path, "zaya", 2,
                       {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight",
                        "self_attn.v_proj_current.weight", "self_attn.v_proj_delayed.weight",
                        "mlp.experts.gate_up_proj.weight", "mlp.experts.down_proj.weight"});
        Q4nxReader r;
        CHECK(r.open(zaya_path), "open zaya artifact");
        CHECK(r.model_type() == "zaya", "model_type() reads the declared family");
        NpuKeyContract c = npu_contract_for(r.model_type());
        CHECK(c.layout && std::strcmp(c.layout, "zaya (split-V attention, MoE FFN)") == 0,
              "model_type 'zaya' -> zaya contract");
        NpuKeyCheck ck = npu_verify_layer_keys(r, 2, c);
        CHECK(ck.status == NpuKeyCheck::Verified, "zaya vocabulary verifies against the zaya contract");
        // The old behaviour, for the record: the dense contract on this file
        // misses 'self_attn.v_proj.weight' at layer 0 — the false report.
        NpuKeyCheck dc = npu_verify_layer_keys(r, 2, npu_dense_contract());
        CHECK(dc.status == NpuKeyCheck::MissingKey && dc.miss_layer == 0 &&
              dc.miss_key == "model.layers.0.self_attn.v_proj.weight",
              "dense contract on a zaya vocabulary misses v_proj at layer 0 (the #2193 report)");
        r.close();
    }

    // ── 4. a declared family with no contract here is not silently dense-checked ──
    {
        write_artifact(odd_path, "somethingelse", 1, {"whatever.weight"});
        Q4nxReader r;
        CHECK(r.open(odd_path), "open unknown-family artifact");
        CHECK(r.model_type() == "somethingelse", "model_type() reads an unknown family");
        NpuKeyContract c = npu_contract_for(r.model_type());
        CHECK(c.layout == nullptr, "unknown family -> no contract (caller must say so)");
        r.close();
    }

    // ── 5. a missing key is named (layer and key), not just "missing" ──
    {
        write_artifact(dense_path, "", 3,
                       {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                        "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight"});
        Q4nxReader r;
        CHECK(r.open(dense_path), "reopen dense artifact without down_proj");
        NpuKeyCheck ck = npu_verify_layer_keys(r, 3, npu_dense_contract());
        CHECK(ck.status == NpuKeyCheck::MissingKey && ck.miss_layer == 0 &&
              ck.miss_key == "model.layers.0.mlp.down_proj.weight",
              "a missing key reports its layer and its name");
        r.close();
    }

    // ── 6. a vocabulary with none of the contract's names is not "missing" ──
    {
        // Measured on a shipped artifact: Qwen3.6-35B-A3B-NPU2 names its layers
        // `model.layer.N.linear_attn.ssm_*` — singular "layer", no attention
        // projections at all, so a dense-contract check can say nothing about it.
        write_artifact(odd_path, "", 4, {"linear_attn.ssm_a", "linear_attn.ssm_dt.bias"},
                       0, "model.layer.");
        Q4nxReader r;
        CHECK(r.open(odd_path), "open artifact with an unrelated vocabulary");
        NpuKeyCheck ck = npu_verify_layer_keys(r, 4, npu_dense_contract());
        CHECK(ck.status == NpuKeyCheck::UnknownVocabulary,
              "a vocabulary matching nothing at layer 0 is UnknownVocabulary, not MissingKey");
        NpuKeyCheck zero = npu_verify_layer_keys(r, 0, npu_dense_contract());
        CHECK(zero.status == NpuKeyCheck::UnknownVocabulary,
              "zero discovered layers verifies nothing (and must not claim success)");
        r.close();
    }

    std::printf("npu_key_contract_selfcheck: %d checks, %d fails\n", checks, fails);
    for (const char* p : {dense_path, big_path, zaya_path, odd_path}) ::unlink(p);
    return fails ? 1 : 0;
}
