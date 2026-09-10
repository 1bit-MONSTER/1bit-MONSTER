// tools/registry_scan.cpp — standalone driver for the artifact-first registry.
//
// Step 2 of goal mtvd3pmx ("one registry, one router"). Deliberately standalone
// (no engine link) so it can run anywhere and be verified before the registry is
// wired into the server. When wired in, `1bit unified` should report the same
// table; this CLI is the reference oracle for that.
//
// Usage:
//   registry_scan [options] <root>...
//     --json                 machine-readable output
//     --digest               SHA-256 every file (slow on a 420 GB store)
//     --no-probe             skip GGUF header/dtype census
//     --max-depth N          recursion depth (default 8)
//     --capability NAME      list only artifacts with this capability
//     --at-context N        apply capability constraints at N context tokens
//     --catalog PATH        ingest a recipe-keyed catalog as a VIEW
//     --catalog-default     same, at ~/.config/lemonade/user_models.json
//                            (NOTE: --catalog REQUIRES its path — an optional
//                             value silently swallows the first root)
//     --resolve PATH|ID      resolve one artifact (acceptance-test check)
//     --route ID|PATH        run the RESOLVER: id -> artifact -> capability
//     --prefer A,B,C         capability preference order for --route
//     --engine-limit CAP=TOKENS[:bundle]
//                            override a capability limit as the ENGINE sees it
//                            (the HRX limit belongs to the configured bundle)
//     --quiet                summary only
// FLAG CLASS CHECK: `tools/registry_flag_audit.py` enforces usage parity (every
// parsed flag documented) and no silent no-ops (each flag alone must change the
// output or exit non-zero). Four instances of that class were hand-patched before
// the check existed; run the audit after touching this file's flags.
#include "model_registry.h"

#include <chrono>
#include <cstdio>
#include <thread>
#include <cstring>
#include <string>
#include <vector>

using namespace onebit;

namespace {

void usage(const char* argv0) {
    fprintf(stderr,
            "usage: %s [--json] [--digest] [--no-probe] [--max-depth N]\n"
            "          [--capability NAME] [--at-context N]\n"
            "          [--catalog PATH | --catalog-default]\n"
            "          [--resolve PATH|ID] [--route ID|PATH]\n"
            "          [--prefer CAP,CAP] [--quiet] <root>...\n"
            "\n"
            "  --at-context N     apply capability constraints at N context tokens\n"
            "  --catalog PATH     ingest a recipe-keyed catalog as a VIEW\n"
            "  --catalog-default  same, at ~/.config/lemonade/user_models.json\n"
            "  --route ID|PATH    run the RESOLVER: id -> artifact -> capability\n"
            "  --prefer A,B,C     capability order for --route (default: the\n"
            "                     artifact's own order; order is policy, not registry)\n"
            "  --watch N          re-scan every N seconds and report the delta\n"
            "                     (--iterations K to stop after K passes; testing aid)\n"
            "  --engine-limit CAP=TOKENS[:bundle]\n"
            "                     override a capability limit as the ENGINE sees it\n"
            "                     (the HRX limit belongs to the configured bundle)\n"
            "\n"
            "capabilities: NPU-Q4NX NPU-1BP HIP-1BP HIP-GGUF RADV-GGUF\n"
            "              HRX2-GGUF-Q4NX HRX-GGUF MLX-GPU CPU\n",
            argv0);
}

void print_report(const RegistryReport& r) {
    printf("artifacts=%zu files=%zu sharded=%zu total=%.2f GiB\n",
           r.artifacts, r.files, r.sharded_artifacts,
           (double)r.total_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("duplicate_id_groups=%zu size_twin_groups=%zu dangling_tokenizers=%zu\n",
           r.duplicate_id_groups, r.size_twin_groups, r.dangling_tokenizers);
    if (r.duplicate_bytes)
        printf("proven_duplicate_bytes=%llu (%.2f GiB reclaimable)\n",
               (unsigned long long)r.duplicate_bytes,
               (double)r.duplicate_bytes / (1024.0 * 1024.0 * 1024.0));
    if (r.merged_artifacts)
        printf("merged_artifacts=%zu reclaimed=%.2f GiB (identical copies collapsed)\n",
               r.merged_artifacts, (double)r.reclaimed_bytes / (1024.0 * 1024.0 * 1024.0));
}

void describe(const ModelArtifact& a) {
    printf("id:           %s\n", a.id.c_str());
    printf("container:    %s\n", to_string(a.container));
    printf("dtype_space:  %s\n", to_string(a.dtype_space));
    printf("has_type42:   %s\n", a.has_dtype_42 ? "yes (OVERLOADED ID)" : "no");
    if (a.container == Container::ONEBP && a.dtype_space != DtypeSpace::ONEBP_HEADER) {
        // These zeros are NOT values. The container physically cannot carry an
        // OnebpHeader, and printing 0s made this indistinguishable from the v1
        // "expert block not carried" case that cost a day to untangle. Raised by
        // @agent-ca60cf against a q4nx-json file.
        printf("native:       n/a (no OnebpHeader in a %s container)", to_string(a.dtype_space));
        if (a.native_json_bytes) printf("  json_bytes=%llu", (unsigned long long)a.native_json_bytes);
        if (a.native_name_mismatch) printf("  [native-name-mismatch]");
        printf("\n");
    } else if (a.native_version || a.native_json_bytes || a.native_name_mismatch || !a.native_dtypes.empty()) {
        printf("native:       version=%u vocab=%d experts=%d top_k=%d tensors=%d rope_theta=%.1f",
               a.native_version, a.native_vocab, a.native_num_experts, a.native_top_k,
               a.native_tensor_count, a.native_rope_theta);
        if (a.expert_fields_absent) printf("  [v1: expert block not carried]");
        if (a.geometry_bound_ratio > 0)
            printf("\n              geometry bound: ratio=%.4f (file vs declared geometry at F32; >1.10 flags)"
                   "  %s", a.geometry_bound_ratio,
                   a.geometry_cannot_hold_file ? "[CANNOT HOLD]" : "ok");
        if (a.native_name_mismatch) printf("  [native-name-mismatch]");
        if (a.arch_suspect) printf("  [arch-suspect]");
        if (a.experts_underdeclared)
            printf("  [experts-underdeclared: a dims-identical sibling GGUF declares experts > 0]");
        printf("\n");
        if (!a.native_dtypes.empty()) {
            printf("native_types:");
            for (const auto& d : a.native_dtypes) printf(" %s", d.c_str());
            printf("\n");
        }
    }
    printf("quant:        %s\n", a.quantization.c_str());
    printf("arch:         %s\n", a.architecture.empty() ? "?" : a.architecture.c_str());
    printf("display_name: %s\n", a.display_name.empty() ? "-" : a.display_name.c_str());
    if (!a.lineage.empty()) printf("lineage:      %s\n", a.lineage.c_str());
    printf("tokenizer:    %s\n", a.tokenizer_path.empty() ? "(none)" : a.tokenizer_path.c_str());
    printf("bytes:        %llu (%.2f GiB) across %zu file(s)\n",
           (unsigned long long)a.total_bytes(),
           (double)a.total_bytes() / (1024.0 * 1024.0 * 1024.0), a.files.size());
    printf("capabilities:");
    for (auto c : a.capabilities) {
        printf(" %s", to_string(c));
        const CapabilityLimit* l = capability_limit(c);
        if (l) printf("(<=%u ctx)", l->max_context_tokens);
    }
    printf("\n");
    if (!a.catalog_ids.empty()) {
        printf("catalog_ids:");
        for (const auto& c : a.catalog_ids) printf(" %s", c.c_str());
        printf("\n");
    }
    if (!a.merged_ids.empty()) {
        printf("merged_ids: ");
        for (const auto& m : a.merged_ids) printf(" %s", m.c_str());
        printf("\n");
    }
    if (a.tokenizer_from_duplicate)
        printf("tokenizer_src: inherited from a proven-identical duplicate\n");
    printf("aliases:");
    for (const auto& al : a.aliases) printf(" %s", al.c_str());
    printf("\n");
    for (const auto& f : a.files) {
        printf("  file  %10llu  %s", (unsigned long long)f.bytes, f.path.c_str());
        if (f.shard_count > 1) printf("   [shard %u/%u]", f.shard_index, f.shard_count);
        if (!f.digest.empty()) printf("   sha256=%s", f.digest.c_str());
        printf("\n");
    }
}

}  // namespace

// Entry point, shared by the standalone `registry_scan` binary and by
// `1bit registry` (compiled into onebin with REGISTRY_SCAN_STANDALONE unset).
int registry_scan_main(int argc, char** argv) {
    ScanOptions opt;
    bool json = false, quiet = false, catalog_set = false;
    std::string resolve_arg, cap_arg, catalog_arg, route_arg, prefer_arg;
    std::vector<std::string> engine_limits;
    int watch_secs = 0, iterations = 0;
    uint32_t at_context = 0;
    std::vector<std::string> roots;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--json") json = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--digest") opt.digest = true;
        else if (a == "--no-probe") opt.probe_headers = false;
        else if (a == "--max-depth" && i + 1 < argc) opt.max_depth = (size_t)atoi(argv[++i]);
        else if (a == "--capability" && i + 1 < argc) cap_arg = argv[++i];
        else if (a == "--resolve" && i + 1 < argc) resolve_arg = argv[++i];
        else if (a == "--route" && i + 1 < argc) route_arg = argv[++i];
        else if (a == "--engine-limit" && i + 1 < argc) engine_limits.push_back(argv[++i]);
        else if (a == "--watch" && i + 1 < argc) watch_secs = atoi(argv[++i]);
        else if (a == "--iterations" && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (a == "--prefer" && i + 1 < argc) prefer_arg = argv[++i];
        else if (a == "--at-context" && i + 1 < argc) at_context = (uint32_t)atoi(argv[++i]);
        else if (a == "--catalog" && i + 1 < argc) { catalog_set = true; catalog_arg = argv[++i]; }
        else if (a == "--catalog-default") { catalog_set = true; }
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (!a.empty() && a[0] == '-') { usage(argv[0]); return 2; }
        else roots.push_back(a);
    }
    // A flag that parses and then does nothing is indistinguishable from one that
    // was never understood — the same hole as --at-context-without---capability,
    // which @agent-ec855d caught earlier. `--prefer` only has meaning relative to
    // a --route target, so refuse it instead of printing a table that silently
    // ignores the caller's stated order.
    if (!prefer_arg.empty() && route_arg.empty()) {
        fprintf(stderr,
                "registry_scan: --prefer only applies to --route (it orders capability "
                "selection for a target).\n"
                "  use: registry_scan --route <id|path> [--at-context N] [--prefer CAP,CAP] ...\n");
        return 2;
    }
    if (roots.empty()) { usage(argv[0]); return 2; }

    // Engine-scoped limits, applied BEFORE the scan so every artifact and every
    // query sees the bundle the engine actually configured.
    for (const auto& spec : engine_limits) {
        size_t eq = spec.find('=');
        if (eq == std::string::npos) {
            fprintf(stderr, "registry_scan: --engine-limit wants CAP=TOKENS[:bundle]\n");
            return 2;
        }
        std::string capname = spec.substr(0, eq), rest = spec.substr(eq + 1), bundle;
        size_t colon = rest.find(':');
        if (colon != std::string::npos) { bundle = rest.substr(colon + 1); rest = rest.substr(0, colon); }
        auto c = capability_from_string(capname);
        if (!c) { fprintf(stderr, "registry_scan: unknown capability '%s'\n", capname.c_str()); return 2; }
        set_capability_limit_override(*c, (uint32_t)atoi(rest.c_str()), bundle);
        fprintf(stderr, "[engine-limit] %s -> %s ctx (bundle %s)\n", capname.c_str(),
                rest.c_str(), bundle.empty() ? "(unspecified)" : bundle.c_str());
    }

    ModelRegistry reg = ModelRegistry::scan(roots, opt);
    reg.set_gate_context(at_context);

    if (catalog_set) {
        if (catalog_arg.empty()) {
            const char* home = getenv("HOME");
            catalog_arg = std::string(home ? home : "") + "/.config/lemonade/user_models.json";
        }
        CatalogView cv = reg.attach_catalog(catalog_arg);
        fprintf(stderr, "[catalog] %s — parse_ok=%s entries=%zu resolved=%zu unknown=%zu\n",
                cv.path.c_str(), cv.parse_ok ? "yes" : "no", cv.entries.size(),
                cv.resolved, cv.unknown);
        for (const auto& e : cv.entries) {
            if (!reg.find(e.catalog_id))
                fprintf(stderr, "  [catalog] unresolved: %s -> %s\n",
                        e.catalog_id.c_str(), e.checkpoint.c_str());
        }
    }

    if (!route_arg.empty()) {
        RouteRequest req;
        req.target = route_arg;
        req.context_tokens = at_context;
        if (!prefer_arg.empty()) {
            size_t p0 = 0;
            while (p0 <= prefer_arg.size()) {
                size_t pc = prefer_arg.find(',', p0);
                std::string tok = prefer_arg.substr(p0, pc == std::string::npos ? std::string::npos : pc - p0);
                if (!tok.empty()) {
                    auto c = capability_from_string(tok);
                    if (!c) { fprintf(stderr, "registry_scan: unknown capability '%s'\n", tok.c_str()); return 2; }
                    req.prefer.push_back(*c);
                }
                if (pc == std::string::npos) break;
                p0 = pc + 1;
            }
        }
        RouteDecision d = reg.resolve(req);
        printf("target:    %s\n", route_arg.c_str());
        if (at_context) printf("gate:      %u context tokens\n", at_context);
        if (!d.artifact) { printf("resolved:  NO -- %s\n", d.reason.c_str()); return 1; }
        printf("artifact:  %s\n", d.artifact->id.c_str());
        printf("resolved:  %s -> %s%s\n", d.resolved ? "YES" : "NO",
               onebit::to_string(d.chosen),
               d.limit_binding ? "  [constraint binding]" : "");
        printf("reason:    %s\n", d.reason.c_str());
        for (const auto& r : d.rejected)
            printf("  skipped: %-16s %s\n", onebit::to_string(r.first), r.second.c_str());
        return d.resolved ? 0 : 1;
    }

    if (!resolve_arg.empty()) {
        const ModelArtifact* a = reg.resolve_path(resolve_arg);
        if (!a) a = reg.find(resolve_arg);
        if (!a) {
            fprintf(stderr, "registry_scan: no artifact matches '%s'\n", resolve_arg.c_str());
            return 1;
        }
        describe(*a);
        return 0;
    }

    if (!cap_arg.empty()) {
        auto c = capability_from_string(cap_arg);
        if (!c) {
            fprintf(stderr, "registry_scan: unknown capability '%s'\n", cap_arg.c_str());
            return 2;
        }
        auto hits = at_context ? reg.serve_at(*c, at_context) : reg.with_capability(*c);
        for (const auto* a : hits) printf("%-46s %s\n", a->id.c_str(), to_string(a->container));
        const CapabilityLimit* l = capability_limit(*c);
        printf("-- %zu artifact(s) with capability %s", hits.size(), cap_arg.c_str());
        if (at_context) printf(" at %u context tokens", at_context);
        if (l) printf(" [limit %u]", l->max_context_tokens);
        printf("\n");
        return 0;
    }

    if (watch_secs > 0) {
        // R4: the autoload surface. Nothing here needs a hand-edited catalog — a
        // new file in a scanned root shows up as an addition on the next pass.
        int pass = 0;
        for (;;) {
            ++pass;
            printf("--- pass %d: %zu artifact(s)\n", pass, reg.artifacts().size());
            fflush(stdout);
            std::this_thread::sleep_for(std::chrono::seconds(watch_secs));
            ModelRegistry next = ModelRegistry::scan(roots, opt);
            RegistryDelta d = next.diff(reg);
            if (d.empty()) {
                printf("    (no change)\n");
            } else {
                for (const auto& x : d.added) printf("    + %s\n", x.c_str());
                for (const auto& x : d.removed) printf("    - %s\n", x.c_str());
                for (const auto& x : d.changed) printf("    ~ %s\n", x.c_str());
            }
            fflush(stdout);
            reg = std::move(next);
            if (iterations > 0 && pass >= iterations) break;
        }
        return 0;
    }

    if (json) { fputs(reg.to_json().c_str(), stdout); return 0; }
    if (!quiet) fputs(reg.to_table(at_context).c_str(), stdout);
    print_report(reg.report());
    return 0;
}

#ifdef REGISTRY_SCAN_STANDALONE
int main(int argc, char** argv) { return registry_scan_main(argc, argv); }
#endif
