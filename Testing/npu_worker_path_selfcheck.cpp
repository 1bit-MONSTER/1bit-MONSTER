// npu_worker_path_selfcheck.cpp — where the NPU worker is looked up.
//
// The native NPU lane fork/execs `npu_engine_universal`, a separate binary. It
// used to be resolved as `$NPU_ENGINE_BIN` else `./npu_engine_universal` — i.e.
// relative to the *current working directory* — so a service started from anywhere
// but the build directory silently had no NPU lane, and no released package ships
// the worker at all (issue #2360). The order is what this pins: an explicit
// override always wins, an installed/next-to-the-binary worker beats a stray file
// in the cwd, and the legacy cwd/build paths stay last so they cannot shadow a
// real install.
//
//   g++ -std=c++17 -Iinclude Testing/npu_worker_path_selfcheck.cpp -o /tmp/t && /tmp/t

#include "npu_worker_path.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>   // chmod
#include <unistd.h>     // access
#include <vector>

static int failures = 0;
static int checks = 0;

static void expect(bool ok, const std::string& what) {
    checks++;
    if (!ok) {
        fprintf(stderr, "  FAIL %s\n", what.c_str());
        failures++;
    }
}

static const std::string* find(const std::vector<std::string>& v, const std::string& s) {
    for (const auto& x : v)
        if (x == s) return &x;
    return nullptr;
}

static int index_of(const std::vector<std::string>& v, const std::string& s) {
    for (size_t i = 0; i < v.size(); i++)
        if (v[i] == s) return (int)i;
    return -1;
}

int main() {
    // 1. An explicit override is always first — never shadowed by anything.
    {
        auto c = npu_worker_candidates("/opt/mine/npu_engine_universal", "/app/bin");
        expect(!c.empty() && c[0] == "/opt/mine/npu_engine_universal",
               "NPU_ENGINE_BIN is the first candidate");
    }

    // 2. Without an override, the worker next to the executable comes first.
    {
        auto c = npu_worker_candidates(nullptr, "/app/bin");
        expect(!c.empty() && c[0] == "/app/bin/npu_engine_universal",
               "a worker next to the executable is tried before system paths");
        expect(index_of(c, "/app/bin/1bit-npu") > 0,
               "the legacy sidecar name next to the executable is also tried");
        expect(index_of(c, "/app/bin/npu_engine_universal") <
                   index_of(c, "/usr/lib/1bit/npu_engine_universal"),
               "…and before the system paths");
    }

    // 3. An empty exe_dir must not produce a bare "/npu_engine_universal".
    {
        auto c = npu_worker_candidates(nullptr, "");
        for (const auto& x : c)
            expect(x != "/npu_engine_universal" && !x.empty(),
                   "no empty or root-relative candidate when exe_dir is unknown");
        expect(!c.empty() && c[0] == "/usr/lib/1bit/npu_engine_universal",
               "system paths take over when exe_dir is unknown");
    }

    // 4. Every layout we ship is represented.
    {
        auto c = npu_worker_candidates(nullptr, "/app/bin");
        for (const char* want : {"/app/bin/npu_engine_universal",
                                 "/app/bin/1bit-npu",                     // sidecar next to the ELF
                                 "/usr/lib/1bit/npu_engine_universal",    // .deb / AppImage
                                 "/usr/lib/1bit/1bit-npu",
                                 "/usr/bin/npu_engine_universal",         // installed on PATH
                                 "/usr/bin/1bit-npu",                     // packaging/Makefile `stage`
                                 "/usr/local/bin/npu_engine_universal",
                                 "./npu_engine_universal",                // legacy cwd
                                 "./1bit-npu",
                                 "build/npu_engine_universal",
                                 "build/1bit-npu"})                       // legacy dev tree
            expect(find(c, want) != nullptr, std::string("candidate present: ") + want);
    }

    // 5. The cwd-relative legacy paths stay LAST: a stray ./npu_engine_universal
    //    must not shadow an installed worker.
    {
        auto c = npu_worker_candidates(nullptr, "/app/bin");
        const int cwd = index_of(c, "./npu_engine_universal");
        const int build = index_of(c, "build/npu_engine_universal");
        for (const char* installed : {"/usr/lib/1bit/npu_engine_universal",
                                      "/usr/bin/npu_engine_universal",
                                      "/usr/bin/1bit-npu",
                                      "/usr/local/bin/npu_engine_universal"})
            expect(cwd > index_of(c, installed), "cwd fallback ranks after installed paths");
        expect(build >= (int)c.size() - 2 && index_of(c, "build/1bit-npu") == (int)c.size() - 1,
               "the build/ paths are the last resort");
        expect(cwd < build, "cwd ranks before the legacy build/ path");
    }

    // 6. No duplicates — a repeated probe is a sign the list was assembled twice.
    {
        auto c = npu_worker_candidates("/opt/mine/npu_engine_universal", "/app/bin");
        for (size_t i = 0; i < c.size(); i++)
            for (size_t j = i + 1; j < c.size(); j++)
                expect(c[i] != c[j], "no duplicate candidate: " + c[i]);
    }

    // 7. resolve() honours an existing path (env override to a real temp file).
    {
        const std::string tmp = "/tmp/npu_worker_selfcheck_worker";
        FILE* f = fopen(tmp.c_str(), "w");
        expect(f != nullptr, "temp worker created");
        if (f) { fputs("#!/bin/sh\n", f); fclose(f); }
        chmod(tmp.c_str(), 0755);
        expect(npu_worker_resolve(tmp.c_str(), "/nonexistent") == tmp,
               "resolve() returns the override when it exists and is executable");
    }

    // 8. resolve() returns "" rather than guessing when nothing is found.
    {
        const std::string got = npu_worker_resolve("/nonexistent/dir/npu_engine_universal",
                                                   "/nonexistent/dir");
        bool any_fallback_exists = false;
        for (const char* p : {"/usr/lib/1bit/npu_engine_universal", "/usr/bin/npu_engine_universal",
                              "/usr/local/bin/npu_engine_universal", "./npu_engine_universal",
                              "build/npu_engine_universal"})
            if (access(p, X_OK) == 0) any_fallback_exists = true;
        if (!any_fallback_exists)
            expect(got.empty(), "resolve() returns an empty path when nothing is found");
        else
            expect(!got.empty(), "resolve() finds a fallback that exists on this box");
    }

    // 9. The failure line tells the user what was probed.
    {
        const std::string list = npu_worker_candidate_list(nullptr, "/app/bin");
        expect(list.find("/app/bin/npu_engine_universal") != std::string::npos &&
                   list.find("build/npu_engine_universal") != std::string::npos &&
                   list.find(", ") != std::string::npos,
               "candidate list is human-readable and complete");
    }

    if (failures) {
        fprintf(stderr, "npu worker path FAILED (%d/%d)\n", failures, checks);
        return 1;
    }
    printf("npu worker path OK (%d checks)\n", checks);
    return 0;
}
