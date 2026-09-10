// include/model_registry.h — artifact-first model registry (ADR R1/R3/R5).
//
// The registry's unit is an ARTIFACT, not a file and not a ModelConfig:
//
//   artifact { id, container, dtype_space, files[1..N shards], capabilities,
//              tokenizer, lineage, aliases }
//
// Why (see okf:references/engine-layering-and-lemonade-scope.md and
// okf:references/model-artifact-inventory.md):
//   R1  one registry of record; ModelConfig is a *view* of an artifact
//   R3  capability is per-ARTIFACT, not per-model (ZAYA1-74B ships .1bp + GGUF
//       twins; *-q4nx.gguf is a third class with its own capability)
//   R5  one canonical id per set of weights, with the on-disk names kept as
//       aliases; `.htok` tokenizers are resolved *through the registry*, never
//       by renaming next to the weights (the cache path is derived from the
//       artifact filename — see inventory F6)
//
// Self-contained on purpose: this header pulls in nothing from the engine, so
// (a) it compiles standalone and (b) the engine can later inject its own
// read_gguf_metadata via ScanOptions::probe instead of using the minimal
// built-in header probe in model_registry.cpp.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace onebit {

// ── Container: what the bytes on disk actually are ─────────────────────────
enum class Container {
    GGUF,          // GGUF v2/v3
    ONEBP,         // native .1bp / .q4nx container (magic df 8b 03 00), tile-fused
    SAFETENSORS,   // HF shard set (dir or file)
    MLX,           // MLX group-affine checkpoint dir
    RAW_BIN,       // loose .bin weight dir (engine native)
    UNKNOWN,
};

// ── Dtype space: whose numbering the tensor type ids are drawn from ────────
//
// CRITICAL (inventory F5): GGUF type id 42 means three different things.
//   * this engine  : GGUF_DTYPE_TQ2_0_G128, block {128 elements, 34 bytes}
//   * HRX2 ggml    : GGML_TYPE_Q4NX, 5120-byte tile (32 x 256)
//   * b66          : unknown entirely
// A reader that assumes the wrong space silently under/over-reads weights.
enum class DtypeSpace {
    EMPTY,             // no tensor census (native container, or not probed)
    GGML_MAINLINE,     // only mainline ids seen (Q4_K=12, Q6_K=14, F32=0, ...)
    ENGINE_TERNARY,    // id 42 present, read as TQ2_0_g128 / Q1_0 id 41
    HRX2_Q4NX,         // id 42 present, read as block_q4nx tiles
    AMBIGUOUS,         // id 42 present and we cannot tell which space
    // ── native containers (no GGUF type ids at all — but NOT "unknown") ──
    ONEBP_HEADER,      // magic 0x00504231 "1BP\0" -> OnebpHeader self-describes quant/arch
    Q4NX_JSON,         // u64 JSON-header-size, then an FLM/aie-rt JSON tensor manifest
    UNRECOGNIZED_NATIVE, // .1bp/.q4nx by name, but neither layout: the name lies
};

// ── Capability: what lane can actually serve this artifact ─────────────────
enum class Capability {
    NPU_Q4NX,       // native .q4nx / 1BP tiled -> XDNA2 (rt_HRX / npu_engine_*)
    NPU_1BP,        // native .1bp
    HIP_1BP,        // 1BP weights on the HIP lane
    HIP_GGUF,       // engine HIP backend
    RADV_GGUF,      // Vulkan/ZINC (commodity GGUF; only coopmat lane on gfx1151)
    HRX2_GGUF_Q4NX, // HRX2 dev build: GGUF container carrying block_q4nx (type 42)
    HRX_GGUF,       // released HRX bundle (b66-class; NO block_q4nx)
    MLX_GPU,        // LSE-style group-affine GPU lane
    CPU,            // generic CPU fallback
    UNKNOWN,
};

const char* to_string(Container c);
const char* to_string(DtypeSpace d);
const char* to_string(Capability c);
// Parse a capability name as printed by to_string(); empty optional if unknown.
std::optional<Capability> capability_from_string(const std::string& s);

// ── Capability constraints ─────────────────────────────────────────────────
//
// A capability is NOT a boolean. "This lane can serve this artifact" is only
// meaningful with a limit attached, because a lane that works at 1k context and
// fails at 3k is not the same capability.
//
// Source of the first entry: @agent-ca60cf (2026-09-10) — the shipped HRX b66
// bundle over-claims FLASH_ATTN_EXT for KV > 2048, so a fresh 2021-token prefill
// passes while 2067/2151/2502/2931 fail identically. Issue #2145. Without this
// constraint a route can hand b66 a longer blob and get silent failure.
// How the number was obtained. A declared limit can be wrong (F11/F12); a
// measured one is an upper bound on a configuration observed to work.
enum class LimitProvenance {
    DECLARED,     // from metadata
    DOCUMENTED,   // from a doc/manual
    MEASURED,     // from an experiment: N works, N+k fails identically
};
// Where the limit actually stops a call. NONE means this registry's report is
// the only thing standing between a caller and a failure.
enum class Enforcement {
    NONE,
    ENGINE_SERVER,             // engine HTTP server path only
    ENGINE_SERVER_AND_SHIM,    // plus the in-process shim fail-close
};

struct CapabilityLimit {
    Capability capability;
    uint32_t max_context_tokens;   // 0 = unconstrained
    const char* note;
    // ── Provenance and ENFORCEMENT SCOPE ─────────────────────────────────
    // A capability limit is not a gate. This registry only REPORTS; whether
    // anything actually stops the call depends on which serving mode is running.
    // Correction from @agent-ca60cf (2026-09-10): both HRX context guards live in
    // generate_completion(), so the b66 limit is enforced on plain `unified` and
    // `zaya` and is INERT under `--lemonade` (which returns from
    // unified_server.cpp:1222-1225 before any engine route exists). Only the
    // shim-level fail-close in src/hrx_inprocess.cpp survives there, and only for
    // callers driving that in-process HRX instance.
    LimitProvenance provenance;
    Enforcement enforced_by;
    const char* not_enforced_in;   // serving mode where the limit does NOT bite
};
const char* to_string(LimitProvenance p);
const char* to_string(Enforcement e);
// Returns nullptr when the capability is unconstrained (or unknown).
const CapabilityLimit* capability_limit(Capability c);

// ── Wire-field naming (JSON from `registry_scan --json` and the HTTP routes) ──
// One name per concept, mirroring the C++ members, so a consumer can be written
// once. Flagged by @agent-ec855d after finding three names for one idea:
//   top level : has_dtype_42, q4nx_name_mismatch, display_name_suspect,
//               arch_suspect(no — nested), native_name_mismatch(no — nested)
//   nested under "native": everything derived SOLELY from the native header —
//               version, vocab, num_experts, top_k, arch_suspect, json_bytes,
//               name_mismatch, dtypes, expert_fields_absent,
//               geometry_cannot_hold_file, geometry_bound_ratio, geometry_bound_note
// Inside the `native` object the `native_` prefix is dropped because the nesting
// already supplies the scope; at the top level it is kept because it does not.
// Rule: `native` = scope, not prefix. Members whose only source is the native
// header live there even when their C++ name has no prefix.

// ── Files ──────────────────────────────────────────────────────────────────
struct ArtifactFile {
    std::string path;              // absolute
    uint64_t bytes = 0;
    uint16_t shard_index = 0;      // 0 when not a shard set
    uint16_t shard_count = 1;
    std::string digest;            // empty unless ScanOptions::digest
};

// ── Artifact ───────────────────────────────────────────────────────────────
struct ModelArtifact {
    std::string id;                            // canonical, stable, unique
    std::vector<std::string> aliases;          // on-disk basenames, original spelling
    std::string display_name;                  // human name (GGUF general.name etc.)
    std::string architecture;                  // engine arch token / HF arch string
    std::string quantization;                  // q4_k_m, q4nx, tq2_g128, ...
    std::string lineage;                       // e.g. "ft-merged7" (inventory F2)

    Container container = Container::UNKNOWN;
    DtypeSpace dtype_space = DtypeSpace::EMPTY;
    std::vector<uint32_t> dtype_ids;           // sorted, deduped census
    bool has_dtype_42 = false;                 // the overloaded id (inventory F5)
    // Filename claims Q4NX but the tensor census has no type 42: the name lies.
    // Seen on zaya1-8b-ft-q4nx.gguf (2026-09-10) — a *q4nx*.gguf carrying
    // mainline quant types. Capability must follow the bytes, not the name.
    bool q4nx_name_mismatch = false;
    // ── native container probe (ONEBP_HEADER / Q4NX_JSON) ────────────────
    // The native formats are self-describing, so the registry can report an
    // authoritative quant/arch instead of guessing from the filename. Raised by
    // @agent-ec855d: a 46 GiB `.1bp` reporting dtype_space=EMPTY made the
    // canonical native format look like "nothing to say".
    uint32_t native_version = 0;               // OnebpHeader.version, or 0
    uint64_t native_json_bytes = 0;            // Q4NX_JSON manifest size
    std::vector<std::string> native_dtypes;    // dtype names found in the manifest
    bool native_name_mismatch = false;         // name says native, bytes say otherwise
    // F11 — GGUF `general.name` is NOT authoritative either. Official Qwen GGUF
    // repos report "Qwen3 4B Instruct Awq" and "Qwen3 30B Gptq Fp16" for plain
    // K-quants (found by @agent-ec855d, 2026-09-10). Reported, never used for a
    // lane decision — display_name is metadata, not evidence.
    bool display_name_suspect = false;
    // OnebpHeader.vocab_size. Verified against @agent-ca60cf's qwen35 reference:
    // 248320 declared == 248,320 f32 per position (993,280 B) observed. So the
    // per-position logits WIDTH every lane must produce is declared by the
    // artifact, and a width mismatch is checkable from the registry without
    // running anything. 0 when not a native header.
    int32_t native_vocab = 0;
    // OnebpHeader MoE fields. These are what actually distinguish two artifacts
    // of the same base model in different architectures — measured:
    // ZAYA1-74B-preview.1bp is DENSE (arch=0, num_experts=0, 1923 tensors,
    // 120 layers) while its sibling `...-a4b-...gguf` is the MoE variant.
    int32_t native_num_experts = 0;
    int32_t native_top_k = 0;
    int32_t native_n_ff_exp = 0;      // OnebpHeader.n_ff_exp (expert FFN width)
    int32_t native_n_ff_shexp = 0;    // OnebpHeader.n_ff_shexp (shared-expert width)
    int32_t native_tensor_count = 0;  // OnebpHeader.tensor_count — the STRONG key
    // rope_theta decoded under the header version's own rule. The encoding
    // CHANGED at v3 (@agent-ca60cf, 2026-09-10): v1/v2 store theta*1000 as
    // fixed point, v3+ store raw f32 bits, and the loader branches on version for
    // exactly that reason (include/onebp_format.h:247,256,262). Reporting the raw
    // field would be wrong by 1000x on v1/v2 files, so this is decoded, never raw.
    float native_rope_theta = 0.0f;
    // v1 is KNOWN not to carry the expert block: ZAYA1-74B-preview.1bp is header
    // v1, writes zero in every expert field, and is a 24-expert MoE by its own
    // tensor index and by its sibling GGUF. So for v1 those zeros mean ABSENT, not
    // DENSE — an absent field is not an assertion. (Mechanism from @agent-ca60cf:
    // the expert fields live in the "Kimi/Moonshot/Laguna" reserved block that
    // post-v1 versions added.) Positive control that v4 IS trustworthy:
    // qwen35-1bp-v2.1bp = v4, num_experts=256, n_expert_used=8, tensor_count=733.
    bool expert_fields_absent = false;
    // Declared dims/experts from whichever source this artifact came with (GGUF
    // metadata or the native header). Used to cross-check two artifacts of what
    // is supposed to be the same model — see experts_underdeclared below.
    int32_t declared_hidden = 0;
    int32_t declared_layers = 0;
    int32_t declared_experts = 0;
    // Attention geometry, needed for the F12b F32 bound below. Only the native
    // header carries these today (the GGUF probe does not read attention KVs).
    int32_t declared_heads = 0;
    int32_t declared_kv_heads = 0;
    int32_t declared_head_dim = 0;
    int32_t declared_interm = 0;
    // Tensor count from whichever source carries it (GGUF table or native
    // header). The STRONG identity key: identical counts mean identical layout,
    // which expert packing changes — dims alone do not discriminate.
    int32_t tensor_count = 0;
    // F12 — the MIRROR of F11. A native header can UNDER-DECLARE, and it does so
    // silently: ZAYA1-74B-preview.1bp writes arch=0 (DENSE) with every expert
    // field zero, while the dims-identical sibling ZAYA1PREVIEW-74B-A4B-Q4_K_M.gguf
    // in the same directory declares zaya.expert_count=24, expert_used_count=1.
    // Reported, never acted on: capability never keys on `arch` (see
    // derive_capabilities), so this is a warning to a consumer, not a route change.
    bool experts_underdeclared = false;
    // F12b — the declared geometry cannot account for the file, EVEN AT F32.
    //
    // Raised by @agent-ec855d after checking my `arch_suspect` rule against the
    // real file and finding a live FALSE NEGATIVE: arch_suspect fires only when
    // arch and the expert fields DISAGREE, and a producer that writes neither
    // leaves both at zero — exactly the case that motivated the flag. So it
    // needed a third source, and a hard one.
    //
    // The sound part is the F32 bound, not a heuristic threshold: no quantization
    // can make a file LARGER than F32 of the same weights. If the file exceeds
    // `params_from_declared_dims * 4 bytes` (plus slack), the header's own numbers
    // are impossible on their own terms and no sibling is required.
    //   dense params = vocab*hidden
    //                + layers * ( hidden*q_dim           // q  (heads*head_dim, GQA-aware)
    //                           + hidden*kv_dim + kv_dim*hidden   // k, v
    //                           + q_dim*hidden           // o
    //                           + 3*hidden*interm )      // SwiGLU MLP
    // ZAYA1-74B-preview.1bp: ~9.38B params -> 37.5 GB at F32, x1.10 slack = 41.3 GB,
    // file is 49.59 GB -> flagged. A legitimate dense F32 export lands at ratio
    // ~1.0 and is not flagged.
    //
    // Residual assumption, stated: intermediate_size must be the real FFN width,
    // and the model must be transformer-shaped. @agent-ec855d conceded the bound
    // is architecture-specific and cannot in principle be made general: a robust
    // version would need the real per-tensor shapes, and the 1BP header carries
    // only tensor_count. For a hybrid (zaya's ssm_conv1d, res_scale_*, routers,
    // extra ffn_gate path) the formula UNDERSTATES declared params, so the bound
    // is too tight there and a legitimate F32 hybrid export could cross it.
    //
    // The numbers that decide whether that matters, computed on the real file:
    //   ratio = total_bytes / (params*4) = 1.3218   (32.2% above the F32 bound)
    //   slack = 1.10
    //   -> only 22.2 points of headroom before the signal is lost.
    // So: it fires when a dense-declared header's own geometry is under 76% of
    // what the file needs at F32. Advisory only, never a route change.
    // Gated to arch == dense with zero experts.
    bool geometry_cannot_hold_file = false;
    // file_bytes / (declared_params*4); 0 when not computable. Exposed so a
    // consumer can judge the margin itself instead of trusting the boolean.
    double geometry_bound_ratio = 0.0;
    // `arch` is header metadata like everything else, so it is CROSS-CHECKED
    // rather than trusted: flagged when arch says DENSE but experts are present,
    // or arch says MOE with none. Raised by @agent-ec855d, who spotted the
    // dense-vs-"a4b" sibling and asked whether arch was a landmine.
    bool arch_suspect = false;

    std::vector<Capability> capabilities;
    std::vector<ArtifactFile> files;
    std::string tokenizer_path;                // resolved .htok, may be empty
    std::string config_dir;                    // dir holding config.json (MLX/HF)
    // Catalog ids (e.g. lemonade `pf-qwen3-30b-a3b-q4km`) that name this
    // artifact. A catalog is a VIEW: it may add aliases, never artifacts.
    std::vector<std::string> catalog_ids;

    uint64_t total_bytes() const;
    bool is_sharded() const { return files.size() > 1; }
    bool has(Capability c) const;
    // 0 = unconstrained. Only meaningful when has(c).
    uint32_t max_context_for(Capability c) const;
    // True when the artifact has the capability AND can take `context_tokens`.
    bool supports(Capability c, uint32_t context_tokens) const;
    // "  legacy-default" style suffix for the table; "" when unambiguous.
    std::string id_quality() const;
};

// ── Report ─────────────────────────────────────────────────────────────────
// ── Resolver ──────────────────────────────────────────────────────────────
struct RouteRequest {
    std::string target;             // canonical id, alias, or a path
    uint32_t context_tokens = 0;    // 0 = no context gate
    // Preferred capability order. EMPTY means "use the artifact's own order",
    // because the ORDER is policy and policy belongs to the caller (R6), not to
    // the registry. The registry only refuses what cannot work.
    std::vector<Capability> prefer;
};
struct RouteDecision {
    bool resolved = false;
    const ModelArtifact* artifact = nullptr;
    Capability chosen = Capability::UNKNOWN;
    uint32_t context_tokens = 0;
    bool limit_binding = false;      // chosen capability carries a constraint
    // Every capability that was NOT chosen, with the reason. Reported so a
    // consumer can see that failover was a decision and not an accident.
    std::vector<std::pair<Capability, std::string>> rejected;
    std::string reason;
};

// ── Catalog views ──────────────────────────────────────────────────────────
// A recipe-keyed catalog (lemonade `user_models.json`) is NOT a registry: it
// maps a catalog id to a checkpoint + recipe. Ingesting it as a view means the
// ids become aliases on the artifacts they point at, and checkpoints outside
// every scanned root are REPORTED, never silently added (R1).
struct CatalogEntry {
    std::string catalog_id;
    std::string checkpoint;
    std::string recipe;
    std::string source;
};
struct CatalogView {
    std::string path;
    std::vector<CatalogEntry> entries;
    size_t resolved = 0;   // catalog id attached to an artifact
    size_t unknown = 0;    // checkpoint not present in any scanned root
    bool parse_ok = false;
};

struct RegistryReport {
    size_t artifacts = 0;
    size_t files = 0;
    size_t sharded_artifacts = 0;
    size_t duplicate_id_groups = 0;     // >1 artifact whose canonical id collided
    size_t size_twin_groups = 0;        // same total_bytes, different id: suspects
    uint64_t total_bytes = 0;
    uint64_t duplicate_bytes = 0;       // bytes of proven-identical artifacts
    size_t dangling_tokenizers = 0;     // .htok whose artifact is absent
    size_t unreadable = 0;              // header probe failed
};

struct ScanOptions {
    bool digest = false;                // hash file contents (slow: 420 GB store)
    bool recurse = true;
    bool probe_headers = true;          // read GGUF header + dtype census
    size_t max_depth = 8;
};

// ── Registry ───────────────────────────────────────────────────────────────
class ModelRegistry {
public:
    static ModelRegistry scan(const std::vector<std::string>& roots,
                              const ScanOptions& opt = ScanOptions{});

    const std::vector<ModelArtifact>& artifacts() const { return artifacts_; }
    const std::vector<std::string>& roots() const { return roots_; }
    RegistryReport report() const;

    const ModelArtifact* find(const std::string& id_or_alias) const;
    // Resolve a path (or path substring) to its artifact, e.g. the acceptance
    // test's `zaya1-8b.q4nx`.
    const ModelArtifact* resolve_path(const std::string& path) const;
    std::vector<const ModelArtifact*> with_capability(Capability c) const;
    // Capability + context gate: the query a router should actually ask.
    std::vector<const ModelArtifact*> serve_at(Capability c, uint32_t context_tokens) const;

    // Ingest a recipe-keyed catalog as a view (see CatalogView).
    CatalogView attach_catalog(const std::string& json_path);
    const std::vector<CatalogView>& catalogs() const { return catalogs_; }

    // ── The resolver (ADR R1/R3/R6): id -> artifact -> capability -> route ────
    // Deliberately inside the registry, NOT in the HTTP layer: `--lemonade` hands
    // the whole HTTP surface to Lemonade's core, so a resolver living there would
    // be bypassed in that mode. `1bit registry` behaves identically everywhere.
    RouteDecision resolve(const RouteRequest& req) const;

    std::string to_table() const;      // human, stable ordering
    // Same table, but with capability constraints applied at `at_context` tokens:
    // a capability that cannot serve that context is marked so it is visible at
    // a glance. Raised by @agent-ec855d — `--at-context` without `--capability`
    // used to be silently ignored, i.e. indistinguishable from "not applied",
    // which is exactly the property this registry exists to eliminate.
    std::string to_table(uint32_t at_context) const;
    // State the gate explicitly for callers that never render the table (JSON).
    void set_gate_context(uint32_t c) const { gate_context_ = c; }
    std::string to_json() const;

    // Copy of `maybe`; if `prefer` is non-empty and present, returns that one.
    static const ModelArtifact* prefer(const std::vector<const ModelArtifact*>& cands,
                                       Capability prefer);

private:
    std::vector<ModelArtifact> artifacts_;
    std::vector<std::string> roots_;
    std::vector<CatalogView> catalogs_;
    size_t dangling_tokenizers_ = 0;   // .htok present with no artifact (inventory F6)
    // Context gate last applied by to_table(), so the JSON report can state it
    // rather than leave a caller guessing whether the gate was active.
    mutable uint32_t gate_context_ = 0;
};

}  // namespace onebit
