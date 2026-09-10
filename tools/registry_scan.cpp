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
//     --resolve PATH|ID      resolve one artifact (acceptance-test check)
//     --quiet                summary only
#include "model_registry.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace onebit;

namespace {

void usage(const char* argv0) {
    fprintf(stderr,
            "usage: %s [--json] [--digest] [--no-probe] [--max-depth N]\n"
            "          [--capability NAME] [--resolve PATH|ID] [--quiet] <root>...\n"
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
}

void describe(const ModelArtifact& a) {
    printf("id:           %s\n", a.id.c_str());
    printf("container:    %s\n", to_string(a.container));
    printf("dtype_space:  %s\n", to_string(a.dtype_space));
    printf("has_type42:   %s\n", a.has_dtype_42 ? "yes (OVERLOADED ID)" : "no");
    printf("quant:        %s\n", a.quantization.c_str());
    printf("arch:         %s\n", a.architecture.empty() ? "?" : a.architecture.c_str());
    printf("display_name: %s\n", a.display_name.empty() ? "-" : a.display_name.c_str());
    if (!a.lineage.empty()) printf("lineage:      %s\n", a.lineage.c_str());
    printf("tokenizer:    %s\n", a.tokenizer_path.empty() ? "(none)" : a.tokenizer_path.c_str());
    printf("bytes:        %llu (%.2f GiB) across %zu file(s)\n",
           (unsigned long long)a.total_bytes(),
           (double)a.total_bytes() / (1024.0 * 1024.0 * 1024.0), a.files.size());
    printf("capabilities:");
    for (auto c : a.capabilities) printf(" %s", to_string(c));
    printf("\n");
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

int main(int argc, char** argv) {
    ScanOptions opt;
    bool json = false, quiet = false;
    std::string resolve_arg, cap_arg;
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
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (!a.empty() && a[0] == '-') { usage(argv[0]); return 2; }
        else roots.push_back(a);
    }
    if (roots.empty()) { usage(argv[0]); return 2; }

    ModelRegistry reg = ModelRegistry::scan(roots, opt);

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
        auto hits = reg.with_capability(*c);
        for (const auto* a : hits) printf("%-46s %s\n", a->id.c_str(), to_string(a->container));
        printf("-- %zu artifact(s) with capability %s\n", hits.size(), cap_arg.c_str());
        return 0;
    }

    if (json) { fputs(reg.to_json().c_str(), stdout); return 0; }
    if (!quiet) fputs(reg.to_table().c_str(), stdout);
    print_report(reg.report());
    return 0;
}
