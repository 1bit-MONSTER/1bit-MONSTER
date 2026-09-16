// npu_paths_selfcheck.cpp — host-only checks for NPU path resolution.
//
// The case that motivated it (measured 2026-09-14): this box's ~/.bashrc
// exported NPU_XCLBIN_DIR=/home/bcloud/1bit-MONSTER-pi/engine/npu/xclbins, a
// tree that no longer exists. npu_xclbin_dir() returned it verbatim, the Zaya
// worker could not find insts_i8_MOE_GU_zaya_m16.txt, and `zaya1-8b.q4nx`
// served nothing on the engine face — with every log line naming a path that
// looked deliberate. An override that does not exist is not an override.
//
//   g++ -std=c++17 -Iinclude -Isrc -O2 Testing/npu_paths_selfcheck.cpp -o /tmp/npu_paths_selfcheck
#include "../engine/npu/src/npu_paths.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

static int checks = 0, fails = 0;
#define CHECK(cond, msg) do { ++checks; if (!(cond)) { std::printf("  FAIL %s\n", msg); ++fails; } } while (0)

int main() {
    const char* real_dir = "/tmp/npu_paths_selfcheck_dir";
    ::mkdir(real_dir, 0755);
    const char* ghost = "/tmp/npu_paths_selfcheck_ghost/engine/npu/xclbins";

    // 1. a valid override is honoured, verbatim
    ::setenv("NPU_XCLBIN_DIR", real_dir, 1);
    CHECK(npu_xclbin_dir() == std::string(real_dir), "an existing NPU_XCLBIN_DIR is used");
    // trailing-slash form is the same directory and must still be honoured
    ::setenv("NPU_XCLBIN_DIR", (std::string(real_dir) + "/").c_str(), 1);
    CHECK(npu_xclbin_dir() == std::string(real_dir) + "/", "an existing dir with a trailing slash is used");

    // 2. an override naming a path this machine does not have is refused, and
    //    the resolver falls back deterministically: a repo checkout in the
    //    current directory wins, otherwise the installed layout is named.
    ::setenv("NPU_XCLBIN_DIR", ghost, 1);
    const char* dev_root = "/tmp/npu_paths_selfcheck_dev";
    ::mkdir(dev_root, 0755);
    ::mkdir((std::string(dev_root) + "/engine").c_str(), 0755);
    ::mkdir((std::string(dev_root) + "/engine/npu").c_str(), 0755);
    ::mkdir((std::string(dev_root) + "/engine/npu/xclbins").c_str(), 0755);

    CHECK(::chdir(dev_root) == 0, "chdir into the synthetic checkout");
    CHECK(npu_xclbin_dir() == std::string("engine/npu/xclbins"),
          "with a repo checkout in CWD, the invalid override falls back to engine/npu/xclbins");

    CHECK(::chdir("/tmp") == 0, "chdir to a directory with no repo checkout");
    std::string got = npu_xclbin_dir();
    CHECK(got != std::string(ghost), "a non-existent NPU_XCLBIN_DIR is never returned");
    CHECK(got.find("1bit-monster/xclbins") != std::string::npos,
          "with no repo checkout in CWD, the invalid override falls back to the install path");

    // 3. the same rule for the insts override
    ::setenv("NPU_XCLBIN_DIR", real_dir, 1);
    ::setenv("NPU_INSTS_DIR", ghost, 1);
    CHECK(npu_insts_dir() != std::string(ghost), "a non-existent NPU_INSTS_DIR is not returned");
    ::setenv("NPU_INSTS_DIR", real_dir, 1);
    CHECK(npu_insts_dir() == std::string(real_dir), "an existing NPU_INSTS_DIR is used");

    // 4. unset: the install path, not an empty string
    ::unsetenv("NPU_XCLBIN_DIR");
    CHECK(!npu_xclbin_dir().empty(), "with no override the install path is returned");

    std::printf("npu_paths_selfcheck: %d checks, %d fails\n", checks, fails);
    ::rmdir(real_dir);
    return fails ? 1 : 0;
}
