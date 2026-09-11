// model_registry_route.cpp — the capability -> backend translation (see header).
#include "model_registry_route.h"

namespace onebit {

// Default probe: UNKNOWN, i.e. do not filter. The bridge has no business guessing
// hardware facts, and an engine that has not installed a probe gets an unfiltered
// plan plus honest UNKNOWN markings.
static Availability default_probe(const std::string&) { return Availability::UNKNOWN; }
static AvailabilityProbe g_probe = default_probe;
void set_backend_availability_probe(AvailabilityProbe probe) {
    g_probe = probe ? probe : default_probe;
}
// Keyed by id: BackendType cannot express this table (npu_xrt vs npu_flm share it).
Availability backend_availability(const std::string& engine_id) { return g_probe(engine_id); }

bool backend_for(Capability c, BackendType& out_type, std::string& out_id,
                 std::string& out_constraint) {
    out_constraint.clear();
    switch (c) {
        case Capability::NPU_Q4NX:
        case Capability::NPU_1BP:
            // Type level is unambiguous: npu_flm registers as BackendType::NPU_XRT
            // (backend_manager.cpp:162) and the NPU_XRT factory prefers it
            // (:1768-1774). The ID is npu_flm, not npu_xrt — npu_xrt is registered
            // but never selected by model_router.cpp or dynamic_router.cpp.
            //
            // Precision added 2026-09-11 (goal mtvd3pmx), because the sentence
            // above is slightly stronger than the code: dynamic_router.cpp's
            // NPU_ONLY branch DOES name npu_xrt as an acceptable pick (:80, :143,
            // "first entry whose id is npu_flm or npu_xrt"). It is still not
            // reachable in practice, but for a better reason than registration
            // order — rank_backends() (backend_manager.cpp:481, end of discover())
            // ranks `functional` above `non-functional` before that strategy ever
            // sees the list, so NPU_ONLY returns whichever NPU lane actually
            // initialized and falls to npu_xrt only when npu_flm did not come up.
            // The two lanes are ~1000x apart (67.5 vs 0.06 tok/s), so a plan that
            // lands on npu_xrt reads as "served at ~0 tok/s" — not as a miss.
            out_type = BackendType::NPU_XRT;
            out_id = "npu_flm";
            // THE TRAP, from @agent-ca60cf: npu_flm is Q4NX-only. For GGUF/H1B its
            // token-level forward()/generate() are text-level stubs that return
            // false, and FLM init "succeeds" on ANY model tag and then loads FLM's
            // own q4nx model — the requested file is never served. That is
            // reachability-without-correctness: it looks available and answers with
            // a different model. The registry already prevents the hand-off
            // structurally (NPU_Q4NX is assigned only to ONEBP/RAW_BIN containers,
            // never to GGUF — see derive_capabilities), and this string states it so
            // a future change to that derivation cannot silently undo it.
            out_constraint =
                "native (ONEBP/RAW_BIN) containers only: npu_flm is Q4NX-only and would "
                "accept the init then serve its own q4nx model rather than the requested file";
            return true;
        case Capability::FUSED_GPU_NPU:
            // The router's FIRST preference in the ONEBP chain — the GPU+NPU fused
            // lane, opt-in per token via USE_NPU_FFN=1.
            out_type = BackendType::HIP_GPU;
            out_id = "fused_gpu_npu";
            out_constraint = "GPU+NPU fused lane; opt-in per token via USE_NPU_FFN=1 "
                             "(without it the fused backend is GPU-only attention+FFN)";
            return true;
        case Capability::VULKAN_1BP:
            // CONTAINER-DEPENDENT id, from ca60cf's review: the GGUF chain uses
            // ggml_vulkan, the 1BP chain uses vulkan_hpp_gpu. One rule for both
            // would leave 1BP artifacts with no Vulkan fallback.
            // TYPE CORRECTED from ca60cf's census: vulkan_hpp_gpu is REGISTERED as
            // BackendType::HIP_GPU (the factory comment says "factory dispatches from
            // HIP_GPU case"), not VULKAN — the only VULKAN-typed id is `vulkan_gpu`,
            // which neither router returns and this table does not use. Having the
            // type wrong would send a type-based dispatcher down the wrong branch.
            out_type = BackendType::HIP_GPU;
            out_id = "vulkan_hpp_gpu";
            return true;
        case Capability::HIP_1BP:
            out_type = BackendType::HIP_GPU;
            out_id = "hip_1bp_gpu";
            return true;
        case Capability::HIP_GGUF:
            out_type = BackendType::HIP_GPU;
            out_id = "hip_gpu";
            // CONDITIONAL, not a plain mapping: create_hip_backend() is the
            // ZAYA-shaped backend and cannot read Qwen/Llama blk.N models (it
            // zero-fills them and fails the coherence probe), which is why
            // Hip1bpBackendAdapter is registered after it. plan_route() applies the
            // gate; this text says why the target is not unconditional.
            out_constraint = "architecture-conditional: create_hip_backend() cannot read "
                             "blk.N-style models (it zero-fills and fails the coherence probe)";
            return true;
        case Capability::RADV_GGUF:
            // The router's GGUF/Vulkan lane is ggml_vulkan, not zinc_gpu: ZINC is a
            // separate Vulkan-IR runtime the router only reaches for specific
            // architectures.
            // TYPE CORRECTED from the census: ggml_vulkan is registered as
            // BackendType::HIP_GPU, NOT VULKAN. Its `has_vulkan()` predicate is what
            // makes it a Vulkan LANE while its type says HIP — which is exactly why
            // BackendType cannot be read as a capability family.
            out_type = BackendType::HIP_GPU;
            out_id = "ggml_vulkan";
            return true;
        case Capability::HRX_GGUF:
            out_type = BackendType::HRX_GPU;
            out_id = "hrx_gpu";
            // D5a: the context limit belongs to the configured BUNDLE, not the
            // artifact, and is enforced only on the engine server + the in-process
            // shim — NOT under --lemonade.
            out_constraint = "context limit is bundle-scoped (see capability_limit); "
                             "enforced by the engine server and the shim, NOT under --lemonade";
            return true;
        case Capability::HRX2_GGUF_Q4NX:
            // D5c: REFUSED, deliberately. dtype 42 means three different things to
            // three readers (F5), and no registered backend advertises the type-42
            // reader. Mapping this onto HRX_GPU would hand a *-q4nx.gguf to the
            // mainline path, where the mis-read produces GARBAGE rather than an
            // error. Refusing is strictly better than guessing.
            return false;
        case Capability::MLX_GPU:
            out_type = BackendType::LSE_GPU;
            out_id = "lse";
            return true;
        case Capability::CPU:
            out_type = BackendType::CPU_SCALAR;
            out_id = "cpu_generic";
            return true;
        case Capability::UNKNOWN:
        default:
            return false;
    }
}

Availability availability_for(bool available_is_a_predicate, bool available, bool functional) {
    // The general form, so no per-id exemption list is ever needed: a LITERAL
    // `available` is a declaration and yields UNKNOWN; only a `has_*()` predicate
    // earns PRESENT. `available && functional` is the minimum, not the target.
    if (!available_is_a_predicate) return Availability::UNKNOWN;
    if (!available) return Availability::ABSENT;
    if (!functional) return Availability::REGISTERED_DRY;
    return Availability::PRESENT;
}

const char* type_collapse_note(BackendType t) {
    switch (t) {
        case BackendType::HIP_GPU:
            return "COLLAPSED TYPE: HIP_GPU is a factory dispatch key with 7 ids "
                   "(fused_gpu_npu, ggml_vulkan, hip_1bp_gpu, hip_gpu, mamba1_gpu, "
                   "vulkan_hpp_gpu, zamba2_gpu) — routable only as (type,id)";
        case BackendType::NPU_XRT:
            return "COLLAPSED TYPE: NPU_XRT holds npu_flm (67.5 tok/s) and npu_xrt, and "
                   "the factory's fall-through for npu_xrt builds the legacy worker "
                   "backend the source rates at 0.06 tok/s — ~1000x apart, same type";
        case BackendType::GENERIC:
            return "COLLAPSED TYPE: GENERIC holds cpu_generic, laguna_gpu, nemotron_h_cpu "
                   "— a GPU backend typed GENERIC";
        case BackendType::ZINC_GPU:
            return "COLLAPSED TYPE: ZINC_GPU holds zamba2_vulkan and zinc_gpu";
        default:
            return nullptr;   // type maps to exactly one id in the census
    }
}

BackendRoute merge_router_and_registry(const BackendRoute& router, const RoutePlan& plan) {
    BackendRoute reg = to_backend_route(plan);
    if (reg.backend_ids_in_order.empty()) return router;   // registry has nothing to say

    // DOES THE REGISTRY HAVE A STATED REASON TO MOVE THE ROUTER'S HEAD?
    // My first version put the registry's order first UNCONDITIONALLY, which overrode the
    // router wherever the registry happened to name a lane — including where the registry
    // simply could not express the router's choice. On `Qwen3.6-35B-A3B-Q8_0` that moved
    // the head off `cpu_qwen3_5` (a class-specific CPU engine my vocabulary cannot name)
    // onto `ggml_vulkan`, demoting a lane for no reason I could defend. That is authority
    // taken rather than granted — the same fault as claiming unverified capabilities.
    //
    // So the registry moves the head only when it has EXCLUDED that lane for a STATED
    // reason. If the lane is a target, or is simply unknown to the registry (absent from
    // both the targets and every exclusion list), the router's head stands and the
    // registry only ADDS lanes. Where it has no measurement, it defers.
    auto id_of = [](Capability c, std::string& out) {
        BackendType t{};
        std::string cons;
        return backend_for(c, t, out, cons);
    };
    auto named_in = [&](const std::vector<std::pair<Capability, std::string>>& v,
                        const std::string& id) {
        for (const auto& kv : v) {
            std::string i;
            if (id_of(kv.first, i) && i == id) return true;
        }
        return false;
    };

    const std::string head = router.backend_ids_in_order.empty() ? std::string()
                                                                 : router.backend_ids_in_order[0];
    bool head_is_target = false;
    for (const auto& id : reg.backend_ids_in_order) if (id == head) { head_is_target = true; break; }
    bool head_excluded_with_reason =
        !head.empty() && !head_is_target &&
        (named_in(plan.refused, head) || named_in(plan.unavailable_here, head) ||
         named_in(plan.blocked, head) || named_in(plan.skipped_by_context, head) ||
         named_in(plan.conditional, head));

    BackendRoute out;
    if (head_excluded_with_reason) {
        out.backend_ids_in_order = reg.backend_ids_in_order;      // a correction: demote
        out.reason = reg.reason;
    } else {
        out.backend_ids_in_order = router.backend_ids_in_order;   // defer; registry adds only
        out.reason = "router head kept (the registry has no stated exclusion for it) | " + reg.reason;
    }
    size_t appended = 0;
    for (const auto& id : (head_excluded_with_reason ? router.backend_ids_in_order
                                                     : reg.backend_ids_in_order)) {
        bool present = false;
        for (const auto& have : out.backend_ids_in_order) if (have == id) { present = true; break; }
        if (!present) { out.backend_ids_in_order.push_back(id); appended++; }
    }
    if (appended)
        out.reason += " | " + std::to_string(appended) + " lane(s) added";
    return out;
}

const char* backend_evidence(Capability c) {
    switch (c) {
        // The two rows with box evidence, credited to @agent-ca60cf on strixhalo.
        case Capability::HRX_GGUF:
            return "BOX-VERIFIED (ca60cf: same-blob rc=0 decode, ctx-limit sweep, guard/re-import)";
        case Capability::HIP_1BP:
            return "BOX-VERIFIED (ca60cf: q35 chat served, 21.6 ms/token bench)";
        // Router-string rows: named by model_router.cpp, never exercised through the
        // engine's id by anyone who reported to me.
        case Capability::RADV_GGUF:
            return "ROUTER-STRING ONLY — a Vulkan/RADV lane at 359.54 tok/s was measured by "
                   "llama-bench INSIDE the fork build, which is outside the engine's id "
                   "system; that the engine id ggml_vulkan reaches it is UNVERIFIED";
        case Capability::VULKAN_1BP:
            return "ROUTER-STRING (backend_manager.cpp:126) — unverified through the engine";
        case Capability::FUSED_GPU_NPU:
            return "ROUTER-STRING — unverified through the engine; also opt-in via USE_NPU_FFN=1";
        case Capability::NPU_Q4NX:
        case Capability::NPU_1BP:
            return "ROUTER-STRING + BUILD-DEPENDENT (npu_flm is hardcoded available=true with "
                   "functional=false, so an available-only check emits it dry)";
        case Capability::HIP_GGUF:
            return "CODE-READING (tests/backends/backend_hip_adapter.cpp: create_hip_backend() "
                   "cannot read blk.N models — zero-fills then fails the coherence probe)";
        case Capability::MLX_GPU:
        case Capability::CPU:
            return "ROUTER-STRING — unverified through the engine";
        case Capability::HRX2_GGUF_Q4NX:
            return "REFUSED BY DESIGN (D5c: no registered backend advertises the type-42 reader)";
        default:
            return "UNKNOWN";
    }
}

RoutePlan plan_route(const ModelArtifact& a, uint32_t context_tokens,
                     const std::vector<Capability>& prefer) {
    RoutePlan plan;
    plan.artifact = &a;

    const std::vector<Capability>& order = prefer.empty() ? a.capabilities : prefer;
    for (Capability c : order) {
        if (!a.has(c)) continue;   // not this artifact's business; not a refusal
        if (context_tokens && !a.supports(c, context_tokens)) {
            uint32_t lim = a.max_context_for(c);
            std::string why = "context " + std::to_string(context_tokens) +
                              " exceeds limit " + std::to_string(lim);
            const CapabilityLimit* l = capability_limit(c);
            if (l && l->bundle) why += std::string(" (bundle ") + l->bundle + ")";
            if (l && l->not_enforced_in)
                why += std::string(" — and that limit is not enforced in ") + l->not_enforced_in;
            plan.skipped_by_context.emplace_back(c, why);
            continue;
        }
        BackendTarget t;
        t.capability = c;
        if (!backend_for(c, t.type, t.engine_id, t.constraint)) {
            plan.refused.emplace_back(c, "no registered backend advertises this capability");
            continue;
        }
        // THE BOX, NOT THE TABLE: intersect with the engine's own probe so a
        // hardware-blind registry cannot hand out a lane this machine lacks.
        // HIP-GGUF is emitted only when the artifact's BYTES say the engine's HIP
        // backend can read it: experts present (a MoE/zaya shape), or the metadata
        // name hints the zaya family. The experts test is bytes; the name test is a
        // hint and is labelled as one, because F11/F12 established that names lie.
        // HRX is constrained by the EMBEDDING's quant, not the file label
        // (@agent-44437c: GET_ROWS only fuses K-quant-class token embeddings; 0.6B-Q8_0
        // and 0.6B-Q4_K_M both fail closed while Coder-30B-A3B-Q4_K_M runs at 87.7
        // tok/s). K-quant class = ids 10..15; anything else is conditional rather than
        // handed over. The second measured case — a SMALL model whose token_embd is not
        // fused — is captured in the artifact (lm_head_fused) but NOT gated here, because
        // "small" would be a threshold I would be inventing rather than measuring.
        if (c == Capability::HRX_GGUF) {
            std::string arch_l = a.architecture;
            for (char& ch : arch_l) ch = (char)tolower((unsigned char)ch);
            if (arch_l.find("zaya") != std::string::npos) {
                plan.conditional.emplace_back(
                    c, "architecture '" + a.architecture + "' is not known to b66's "
                       "llama.cpp (measured: \"unknown model architecture: 'zaya'\" on 5 "
                       "files, incl. the 74B preview, whose prediction became a measurement "
                       "when 44437c copied it to strixhalo) — this belongs to the NPU/Zaya "
                       "lane, not HRX");
                continue;
            }
            // THE ABORT CLASS. Measured: non-K-quant token embedding on a qwen35moe arch
            // does not fail closed, it SIGABRTs (rc=134). Blocked rather than
            // conditional, because a caller cannot retry past an abort.
            if (a.tok_embd_dtype >= 0 && !(a.tok_embd_dtype >= 10 && a.tok_embd_dtype <= 15) &&
                arch_l.find("qwen35moe") != std::string::npos) {
                plan.blocked.emplace_back(
                    c, "KNOWN ABORT: measured SIGABRT (rc=134) on this class "
                       "(qwen35moe + non-K-quant token_embd) — worse than a refusal, the "
                       "abort takes the engine process with it");
                continue;
            }
        }
        // PRECEDENCE, because the REASON determines the fix: architecture first (b66
        // cannot read the arch at all, so the file belongs to another lane), then the
        // known-abort class (worse than a refusal), then embedding dtype, then the dense
        // MoE proxy. My first ordering ran dtype first and reported 'embedding dtype 1 is
        // not K-quant' for a zaya file whose actual blocker is the architecture — a
        // misleading reason on a correctly-excluded row, which is its own small version
        // of the blank-reads-as-confident problem.
        if (c == Capability::HRX_GGUF && a.tok_embd_dtype >= 0) {
            bool k_quant = (a.tok_embd_dtype >= 10 && a.tok_embd_dtype <= 15);
            if (!k_quant) {
                plan.conditional.emplace_back(
                    c, "token embedding dtype " + std::to_string(a.tok_embd_dtype) +
                       " is not K-quant class (10..15); b66 fails closed at GET_ROWS on "
                       "non-K-quant embeddings — measured on Qwen3-0.6B-Q8_0");
                continue;
            }
        }
        // HRX IS NECESSARY-BUT-NOT-SUFFICIENT ON EMBEDDING DTYPE ALONE, per
        // @agent-44437c's measured table against b66:
        //   Qwen3-0.6B-Q4_K_M (emb Q6_K) FAIL · 0.6B-fused (Q6_K) FAIL ·
        //   1.7B-Q4_K_M FAIL · 4B-Q4_K_M (fused=true) FAIL · 8B-Q4_K_M (fused=false) FAIL ·
        //   0.6B-Q8_0 (Q8_0) FAIL · Coder-30B-A3B-Q4_K_M (Q4_K) PASS at 87.72 t/s
        // So the `fused` flag cannot gate (4B fused=true fails, 8B fused=false fails) and
        // K-quant-class cannot gate either (Q4_K appears in both the failing 8B and the
        // passing 30B). The ONLY discriminator their measurements support is the MoE A3B
        // class, and on my store `declared_experts` separates the two perfectly:
        //   qwen3-30b-a3b-q4km experts=128 (their PASS class) vs 0 for every FAIL class.
        // Conservative until more classes are measured: HRX-GGUF is CONDITIONAL unless
        // the artifact is MoE. The engine fails closed either way, so this is about not
        // OVER-CLAIMING rather than about safety.
        // ARCH-LEVEL FACT, from 44437c's measurements: b66's llama.cpp does not know
        // the `zaya` architecture AT ALL ("unknown model architecture: 'zaya'"), so
        // every zaya artifact is mislabelled as HRX-servable. This is not a quant or
        // embedding property, and it caught a FALSE POSITIVE in my own MoE rule: the
        // 74B preview is MoE (experts=24) so the MoE-only rule kept it, while b66
        // cannot read its arch. Measured on 4 of 7 files in their batch.
        // THE MEASURED DISCRIMINATOR, and it cuts INSIDE the quant label (@agent-ca60cf,
        // from the HRX child's own log once they fixed the /dev/null redirect):
        //   unsupported HRX node 0: GET_ROWS ... inputs=[0:q6_K[1024,151936,1,1], ...]
        // — a file named Q4_K_M carrying a q6_K embedding. GET_ROWS accepts ONE embedding
        // dtype on this bundle, and the boundary sits within Q4_K_M: 30B Q4_K_M passes,
        // 0.6B Q4_K_M does not, because the small model keeps a higher-precision embedding.
        //
        // Verified against my own bytes on strixhalo — tok_embd_dtype separates their
        // measured outcomes exactly:
        //   0.6B-Q4_K_M / -fused / 1.7B / 4B   tok_embd=14 (Q6_K)  -> all measured FAIL
        //   35B-A3B-Q8_0                        tok_embd=8  (Q8_0)  -> measured ABORT
        //   Coder-30B-A3B-Q4_K_M                tok_embd=12 (Q4_K)  -> measured PASS 87.72 t/s
        //
        // NOTE ON THEIR WORDING, checked rather than assumed: they describe the rule as
        // "fused Q4_K", but the one PASSING file has lm_head_fused == false (it carries a
        // separate output.weight). Implementing "Q4_K AND fused" would have excluded the
        // only measured-passing artifact — so the predicate is the TENSOR DTYPE alone, and
        // `fused` is not part of it.
        //
        // This replaces the MoE-only proxy, which their measurements show mislabels both
        // directions: a dense file with a Q4_K embedding should pass, and a small MoE file
        // with a q6_K embedding should not.
        if (c == Capability::HRX_GGUF) {
            if (a.tok_embd_dtype < 0) {
                plan.conditional.emplace_back(
                    c, "token embedding dtype not readable; HRX acceptance requires a Q4_K "
                       "(12) token_embd.weight on this bundle — not verified, so not claimed");
                continue;
            }
            if (a.tok_embd_dtype != 12) {
                plan.conditional.emplace_back(
                    c, "token_embd.weight dtype " + std::to_string(a.tok_embd_dtype) +
                       " is not Q4_K (12) — GET_ROWS only accepts a Q4_K embedding on this "
                       "bundle, and the boundary cuts INSIDE the file quant label (0.6B "
                       "Q4_K_M carries q6_K and fails while 30B Q4_K_M passes)");
                continue;
            }
        }
        // The gate accepts EITHER a declared expert count (GGUF KV / OnebpHeader) OR the
        // tensor-name marker from a Q4NX_JSON manifest, because for that container the
        // former is structurally absent. Without this the registry was blind for native
        // .q4nx files and excluded the Zaya/CCA lane the router actually picks.
        if (c == Capability::HIP_GGUF && a.declared_experts <= 0 && !a.native_has_experts) {
            std::string arch_l = a.architecture;
            for (char& ch : arch_l) ch = (char)tolower((unsigned char)ch);
            if (arch_l.find("zaya") == std::string::npos &&
                arch_l.find("mamba") == std::string::npos) {
                plan.conditional.emplace_back(
                    c, "not architecture-matched for hip_gpu: no experts declared and no "
                       "zaya/mamba signature in the metadata (create_hip_backend() cannot "
                       "read blk.N-style models) — excluded rather than handed over");
                continue;
            }
        }
        Availability av = backend_availability(t.engine_id);
        if (av == Availability::REGISTERED_DRY) {
            plan.conditional.emplace_back(
                c, std::string("registered but DRY (available=true, functional=false until "
                               "init) for ") + t.engine_id + " — looks runnable while being dry");
            continue;
        }
        if (av == Availability::ABSENT) {
            plan.unavailable_here.emplace_back(
                c, std::string("capability present, HARDWARE ABSENT on this machine (") +
                       backend_name(t.type) + ", id " + t.engine_id + ") — not a constraint violation");
            continue;
        }
        // Carry the VERIFICATION STATE, not just the capability: PRESENT means the
        // probe confirmed it, UNKNOWN means nobody asked. Without this the two are
        // byte-identical in the output and an unverified plan reads as a verified one.
        t.availability = av;
        plan.targets.push_back(std::move(t));
    }

    // D5b: stated, never inferred from reachability.
    if (a.has(Capability::NPU_Q4NX)) {
        plan.quality_gate = QualityGate::KNOWN_DEGRADED;
        plan.quality_note = "NPU-Q4NX is reachable; the zaya cascade decode on it is not "
                           "numerically usable in-flow (#2114) — a separate decision from "
                           "whether it can serve";
    }
    return plan;
}

BackendRoute to_backend_route(const RoutePlan& plan) {
    BackendRoute out;
    for (const auto& t : plan.targets) out.backend_ids_in_order.push_back(t.engine_id);

    std::string why = plan.targets.empty() ? "no backend can serve this artifact"
                                           : "registry plan";
    // The FOURTH category, so it cannot be the silent one. Categories 1-3 are
    // appended below; this one means "not verified against this box", which must
    // never be indistinguishable from "verified".
    std::string unverified;
    for (const auto& t : plan.targets) {
        if (t.availability != Availability::PRESENT) {
            if (!unverified.empty()) unverified += ", ";
            unverified += to_string(t.capability);
        }
    }
    if (!unverified.empty())
        why += " | availability UNKNOWN: " + unverified +
               " (no probe installed or probe silent — NOT verified against this box)";
    if (!plan.refused.empty()) {
        why += " | refused:";
        for (const auto& r : plan.refused) why += " " + std::string(to_string(r.first));
    }
    if (!plan.blocked.empty()) {
        why += " | KNOWN-ABORT:";
        for (const auto& r : plan.blocked) why += " " + std::string(to_string(r.first));
    }
    if (!plan.conditional.empty()) {
        why += " | conditional:";
        for (const auto& r : plan.conditional) why += " " + std::string(to_string(r.first));
    }
    if (!plan.unavailable_here.empty()) {
        why += " | hardware absent:";
        for (const auto& r : plan.unavailable_here) why += " " + std::string(to_string(r.first));
    }
    if (!plan.skipped_by_context.empty()) {
        why += " | out of context:";
        for (const auto& r : plan.skipped_by_context) why += " " + std::string(to_string(r.first));
    }
    if (plan.quality_gate != QualityGate::NOT_EVALUATED)
        why += " | QUALITY GATE: reachable but flagged (see quality_note)";
    out.reason = why;
    return out;
}

BackendRoute select_route_with_registry(const ModelConfig& cfg,
                                        const std::string& model_path,
                                        const ModelRegistry* registry,
                                        uint32_t context_tokens) {
    // The router's shipped order is the baseline; a missing registry or an unknown
    // artifact is not an error, it is "the registry has nothing to say here" —
    // which is exactly why the fallback keeps the router route.
    BackendRoute router = select_backend_route(cfg);
    if (!registry || model_path.empty()) return router;
    const ModelArtifact* art = registry->resolve_path(model_path);
    if (!art && !cfg.model_name.empty()) art = registry->find(cfg.model_name);
    if (!art) return router;
    return merge_router_and_registry(router, plan_route(*art, context_tokens));
}

}  // namespace onebit
