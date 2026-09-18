// tests/prism/test_prism_failclosed.cpp — the R15 fail-closed invariant, executable.
//
// A folded Prism pack carries __onebp_ext_prism_transform; an unfolded pack does not.
// The loader must report this so a backend that cannot apply the Hadamard transform can
// refuse instead of serving folded weights as plain. Checks each real .1bp against the
// expected flag, and that a backend refusing on the flag would accept the unfolded packs.
//
// Build: g++ -O2 -std=c++17 -I include -I src tests/prism/test_prism_failclosed.cpp \
//          src/onebp_model.cpp -o /tmp/fc
// Run:   /tmp/fc <model.1bp> <expected 0|1>

#include "onebp_loader.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model.1bp> <expected 0|1>\n", argv[0]); return 2; }
    OnebpModel m;
    if (!m.load(argv[1])) { std::fprintf(stderr, "load failed: %s\n", argv[1]); return 1; }
    const bool got = m.has_prism_transform();
    const bool want = std::atoi(argv[2]) != 0;
    std::printf("  %-40s has_prism_transform=%d expected=%d  %s\n",
                argv[1], (int)got, (int)want, got == want ? "PASS" : "FAIL");
    return got == want ? 0 : 1;
}
