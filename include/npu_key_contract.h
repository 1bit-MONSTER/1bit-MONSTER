#pragma once
// npu_key_contract.h — which per-layer GEMM tensors the NPU worker will load
// for an artifact, and the host-side verification of that contract.
//
// The parent backend used to assume exactly one layout: the dense SwiGLU
// transformer (`self_attn.q/k/v/o_proj` + `mlp.gate/up/down_proj`). Q4NX
// artifacts that are not dense declare their family in the JSON header
// ("model_type"), and their worker loads a different vocabulary. Zaya's
// attention splits V in two (`v_proj_current` / `v_proj_delayed`) and its FFN
// is a MoE (`mlp.experts.gate_up_proj` / `mlp.experts.down_proj`, plus the
// `mlp.gate.*` router). Checking dense names against such an artifact reports
// a missing weight — for a model the worker serves correctly (issue #2193,
// Defect 4; the names below are the ones engine/npu/src/zaya_decode.cpp reads).
//
// Header-only and device-free on purpose: the contract is verifiable against
// the artifact's own index, which is what
// Testing/npu_key_contract_selfcheck.cpp does.

#include "q4nx_reader.h"
#include <cstdio>
#include <string>
#include <vector>

struct NpuKeyContract {
    const char* layout = nullptr;        // named in the verification message
    std::vector<const char*> suffixes;   // per layer, after "model.layers.N."
};

// FLM's dense export: no model_type in the header.
inline NpuKeyContract npu_dense_contract() {
    return {"dense", {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }};
}

// Zaya: split-V attention + MoE FFN.
inline NpuKeyContract npu_zaya_contract() {
    return {"zaya (split-V attention, MoE FFN)", {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.v_proj_current.weight",
        "self_attn.v_proj_delayed.weight",
        "mlp.experts.gate_up_proj.weight",
        "mlp.experts.down_proj.weight",
    }};
}

// The contract for an artifact that declares `model_type`. An empty model_type
// is the dense case; a family with no contract here yields layout == nullptr,
// and the caller must say so rather than verify the wrong names.
inline NpuKeyContract npu_contract_for(const std::string& model_type) {
    if (model_type.empty()) return npu_dense_contract();
    if (model_type == "zaya") return npu_zaya_contract();
    return NpuKeyContract{};
}

struct NpuKeyCheck {
    enum Status {
        Verified,           // every contract key present in every layer
        MissingKey,         // the vocabulary matches the contract, but a key is absent
        UnknownVocabulary,  // this artifact's per-layer names are not the contract's at all
    };
    Status status = Verified;
    int miss_layer = -1;    // first layer with a missing key (Status::MissingKey)
    std::string miss_key;   // the key that was missing
};

// Verify `layers` layers against the contract.
//
// A vocabulary that matches NOTHING at layer 0 is reported as UnknownVocabulary,
// not as a missing weight: some shipped artifacts name their layers differently
// entirely — Qwen3.6-35B-A3B-NPU2 uses `model.layer.N.linear_attn.ssm_*`
// (singular `layer`, no attention projections), so "missing" would be a claim
// about a model the check cannot speak about. A contract that matches partially
// (3 of 7 keys) is a real miss and is reported with its layer and key name.
inline NpuKeyCheck npu_verify_layer_keys(const Q4nxReader& model, int layers,
                                         const NpuKeyContract& contract) {
    NpuKeyCheck r;
    if (layers <= 0) { r.status = NpuKeyCheck::UnknownVocabulary; return r; }

    auto key_at = [&](int l, const char* suffix, char* out, size_t n) {
        std::snprintf(out, n, "model.layers.%d.%s", l, suffix);
    };
    int hits_at_zero = 0;
    for (const char* suffix : contract.suffixes) {
        char key[256];
        key_at(0, suffix, key, sizeof(key));
        if (model.find_offset(key)) hits_at_zero++;
    }
    if (hits_at_zero == 0) { r.status = NpuKeyCheck::UnknownVocabulary; return r; }

    for (int l = 0; l < layers; l++) {
        for (const char* suffix : contract.suffixes) {
            char key[256];
            key_at(l, suffix, key, sizeof(key));
            if (!model.find_offset(key)) {
                r.status = NpuKeyCheck::MissingKey;
                r.miss_layer = l;
                r.miss_key = key;
                return r;
            }
        }
    }
    r.status = NpuKeyCheck::Verified;
    return r;
}
