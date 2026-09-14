#pragma once
// npu_worker_path.h — where the native NPU worker lives.
//
// The engine's NPU lane is a *separate executable*: src/backend_npu.cpp fork/execs
// `npu_engine_universal` and speaks the worker protocol to it (GEMM via
// pre-compiled xclbins, CPU fallback for RoPE/norm/residual — "Zero FLM
// dependency"). It is not part of the `1bit` ELF.
//
// Resolution used to be `$NPU_ENGINE_BIN` else `./npu_engine_universal`, i.e.
// relative to the *current working directory*, so a service started from anywhere
// but the build directory silently had no NPU lane. tools/capture_npu_acceptance.sh
// warns about exactly that, and no released package ships the worker at all
// (issue #2360). Resolve through every layout we actually produce:
//
//   1. $NPU_ENGINE_BIN                        explicit override — always first
//   2. <dir of this executable>/npu_engine_universal
//                                            tarball (bin/1bit + bin/worker) and a
//                                            source build (build/1bit + build/worker)
//   3. /usr/lib/1bit/npu_engine_universal     .deb / AppImage layout
//   4. /usr/bin/, /usr/local/bin/             installed on PATH
//   5. ./npu_engine_universal, build/npu_engine_universal
//                                            legacy cwd-relative use (a no-op when
//                                            the cwd is unrelated)
//
// Dependency-free, and split from the fork/exec site so the order itself is
// unit-testable — see Testing/npu_worker_path_selfcheck.cpp.

#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

// Directory holding the running executable, or "" when it cannot be determined.
inline std::string npu_worker_exe_dir() {
#if defined(__linux__)
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        const std::string p(buf);
        const auto slash = p.find_last_of('/');
        if (slash != std::string::npos) return p.substr(0, slash);
    }
#endif
    return "";
}

// Names the worker is installed/copied under. `npu_engine_universal` is the
// canonical one; `1bit-npu` is the legacy sidecar name — packaging/Makefile's
// `stage` target installs the same binary as `usr/bin/1bit-npu`, and it is
// argv[0]-independent (engine/npu/src/npu_engine_universal.cpp only uses argv[0]
// in a usage string), so either name drives the worker protocol.
inline const std::vector<const char*>& npu_worker_names() {
    static const std::vector<const char*> names = {"npu_engine_universal", "1bit-npu"};
    return names;
}

// Ordered candidate paths for the worker. `env_bin` is $NPU_ENGINE_BIN (may be
// null); `exe_dir` is npu_worker_exe_dir(). Pure: no filesystem access.
inline std::vector<std::string> npu_worker_candidates(const char* env_bin,
                                                      const std::string& exe_dir) {
    std::vector<std::string> c;
    if (env_bin && env_bin[0]) c.push_back(env_bin);
    std::vector<std::string> dirs;
    if (!exe_dir.empty()) dirs.push_back(exe_dir);
    dirs.push_back("/usr/lib/1bit");
    dirs.push_back("/usr/bin");
    dirs.push_back("/usr/local/bin");
    for (const auto& d : dirs)
        for (const char* n : npu_worker_names()) c.push_back(d + "/" + n);
    // Legacy cwd-relative use — kept last so a stray file in the working directory
    // can never shadow an installed worker.
    c.push_back("./npu_engine_universal");
    c.push_back("./1bit-npu");
    c.push_back("build/npu_engine_universal");
    c.push_back("build/1bit-npu");
    return c;
}

// First candidate that exists and is executable, or "" when none is.
// (POSIX-only like the lane it serves: src/backend_npu.cpp fork/execs the worker,
// so on a platform without unistd there is no worker to resolve.)
inline std::string npu_worker_resolve(const char* env_bin, const std::string& exe_dir) {
#if defined(__linux__)
    for (const auto& c : npu_worker_candidates(env_bin, exe_dir))
        if (access(c.c_str(), X_OK) == 0) return c;
#endif
    (void)env_bin;
    (void)exe_dir;
    return "";
}

// Human-readable "what was probed" line for failure paths ("a, b, c").
inline std::string npu_worker_candidate_list(const char* env_bin, const std::string& exe_dir) {
    std::string out;
    for (const auto& c : npu_worker_candidates(env_bin, exe_dir)) {
        if (!out.empty()) out += ", ";
        out += c;
    }
    return out;
}
