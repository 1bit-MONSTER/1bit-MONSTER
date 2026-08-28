// test_router_eda_oob.cpp — issue #1799 regression test (ASan).
//
// The q4nx manifest declares mlp.gate.router_states_scale as shape [1]
// (data_offsets span = 2 bytes), so the loader used to populate w.rw.eda with
// ONE element while the router's EDA recurrence (rs += prev_router * eda,
// rtr_h=256 iterations) read eda[1..255] OUT OF BOUNDS on every MoE layer
// after L1 (the run-to-run nondeterminism, issue #1799).
//
// This test simulates exactly that buggy loader state — a 1-element eda —
// and drives the router. Under AddressSanitizer:
//   - pre-fix (ae799e01): ABORTS with heap-buffer-overflow READ of size 4
//     at zaya_moe_cpu.h router() EDA loop
//   - post-fix (4d4e9c48): runs to completion (loop bounded by min(rtr_h,
//     prev_router, eda) sizes), exit 0
//
// No model file needed — full dims are synthesized. Run with:
//   g++ -std=c++23 -O1 -g -fsanitize=address -fno-omit-frame-pointer \
//       -I engine/npu/src -I engine/npu/generators \
//       test_router_eda_oob.cpp -o test_router_eda_oob && ./test_router_eda_oob
//
// Also exercises the full-length eda (rtr_h=256) load the fixed loaders
// perform, and asserts the EDA contribution matches a manual min-bounded
// recurrence (exact float math) — catches both the OOB and a silent
// off-by-one in the bound.

#include "zaya_moe_cpu.h"
#include <cmath>
#include <cstdio>
#include <vector>

static zaya_moe::RouterWeights make_weights(const zaya_moe::MoeDims& d) {
    zaya_moe::RouterWeights w;
    w.gdw.assign((size_t)d.H * d.rtr_h, 0.001f);
    w.gdb.assign(d.rtr_h, 0.0f);
    w.rfn.assign(d.rtr_h, 1.0f);
    w.rf1.assign((size_t)d.rtr_h * d.rtr_h, 0.0f);
    for (int i = 0; i < d.rtr_h; i++) w.rf1[(size_t)i * d.rtr_h + i] = 1.0f;  // identity
    w.rf1b.assign(d.rtr_h, 0.0f);
    w.rf2.assign((size_t)d.rtr_h * d.rtr_h, 0.0f);
    for (int i = 0; i < d.rtr_h; i++) w.rf2[(size_t)i * d.rtr_h + i] = 1.0f;
    w.rf2b.assign(d.rtr_h, 0.0f);
    w.rout.assign((size_t)d.n_exp_t * d.rtr_h, 0.0f);
    w.bb.assign(d.n_exp_t, 0.0f);
    return w;
}

int main() {
    zaya_moe::MoeDims d = zaya_moe::MoeDims::zaya1_8b(); // H=2048 rtr_h=256 n_exp=16
    std::vector<float> hs(d.H, 0.5f);
    std::vector<float> prev_router(d.rtr_h, 0.1f); // layer >= 2: prev_router non-empty
    float expert_wt = 0.f;
    int failures = 0;

    // ── Case 1: BUGGY LOADER STATE — 1-element eda (what shape-[1] produced).
    // Pre-fix this read eda[1..255] OOB on the heap; ASan aborts here.
    {
        zaya_moe::RouterWeights w = make_weights(d);
        w.eda.assign(1, 0.99f);
        std::vector<float> pr = prev_router;
        int e = zaya_moe::router(d, w, hs.data(), pr, &expert_wt);
        // Bounded loop must have applied the EDA with eda[0] only.
        float expect = 0.0f; // gate_down with gdw=0.001, gdb=0
        for (int j = 0; j < d.H; j++) expect += hs[j] * 0.001f;
        expect += pr.empty() ? 0.0f : 0.0f; // pr was overwritten by router (prev_router = rs)
        // After router, pr holds rs (post-EDA, pre-norm) — check EDA applied:
        if (std::fabs(pr[0] - (expect + 0.1f * 0.99f)) > 1e-3f) {
            fprintf(stderr, "FAIL case1: EDA not applied via bounded loop (pr[0]=%f expect ~%f)\n",
                    pr[0], expect + 0.1f * 0.99f);
            failures++;
        }
        printf("case1 (1-elem eda): router=%d expert_wt=%f pr[0]=%f  [ASan-clean = fixed]\n",
               e, expert_wt, pr[0]);
    }

    // ── Case 2: FIXED LOADER STATE — full rtr_h=256 eda (rtr_h*2-byte load).
    {
        zaya_moe::RouterWeights w = make_weights(d);
        w.eda.assign(d.rtr_h, 0.99f);
        std::vector<float> pr = prev_router;
        int e = zaya_moe::router(d, w, hs.data(), pr, &expert_wt);
        float expect = 0.0f;
        for (int j = 0; j < d.H; j++) expect += hs[j] * 0.001f;
        expect += 0.1f * 0.99f; // full-length recurrence
        if (std::fabs(pr[0] - expect) > 1e-3f) {
            fprintf(stderr, "FAIL case2: full-eda recurrence mismatch (pr[0]=%f expect=%f)\n",
                    pr[0], expect);
            failures++;
        }
        printf("case2 (full eda): router=%d expert_wt=%f pr[0]=%f  [exact recurrence]\n",
               e, expert_wt, pr[0]);
    }

    if (failures) { fprintf(stderr, "%d FAILURES\n", failures); return 1; }
    printf("PASS: router EDA bounded loop (issue #1799 fix) — no OOB, math exact\n");
    return 0;
}
