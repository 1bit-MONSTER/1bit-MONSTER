// test_runtime_layer.cpp — drive RuntimeLayerEngine (the FastFlowLM runtime
// submission path wired into npu-infer) with tokens 1000/1001/1002 and dump
// act/logits for byte-comparison against the runtime's captures.
//
// Build: g++ -O2 -std=c++17 -fpermissive tests/test_runtime_layer.cpp src/runtime_layer.cpp \
//        src/model.c -Iinclude -lxrt_coreutil -lxrt_core -o /tmp/txn_decode/test_runtime_layer
// Run:   /tmp/txn_decode/test_runtime_layer <model_dir> [dump_prefix]
//        - RT_ONLY_FWD1: stop after forward 1
//        - RT_FWD3: also run forward 3 (token 1002, ctx=3)
//        - CAP_DIR: directory with actpost_002/004/006 captures (default /tmp/capF2)
//        - RT_ELF_DIR: layer ELF dir (default captures/txn-elfs)
#include "runtime_layer.h"
#include "model.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <glob.h>
#include <xrt/xrt_device.h>

static std::string glob_first(const char* pattern) {
    glob_t g; std::string out;
    if (glob(pattern, 0, nullptr, &g) == 0 && g.gl_pathc > 0) out = g.gl_pathv[0];
    globfree(&g);
    return out;
}

static void cmp_file(const char* a, const char* b, const char* what) {
    FILE* fa = fopen(a, "rb"); FILE* fb = fopen(b, "rb");
    if (!fa || !fb) {
        fprintf(stderr, "  %s: missing file (%s / %s)\n", what, a, b);
        if (fa) fclose(fa); if (fb) fclose(fb);
        return;
    }
    fseek(fa, 0, SEEK_END); long sa = ftell(fa); fseek(fa, 0, SEEK_SET);
    fseek(fb, 0, SEEK_END); long sb = ftell(fb); fseek(fb, 0, SEEK_SET);
    long n = sa < sb ? sa : sb;
    std::vector<char> va(n), vb(n);
    fread(va.data(), 1, n, fa); fread(vb.data(), 1, n, fb);
    fclose(fa); fclose(fb);
    bool same = (sa == sb) && (va == vb);
    fprintf(stderr, "  %s: %s\n", what, same ? "BYTE-IDENTICAL" : "DIFFERS");
}

int main(int argc, char** argv) {
    const char* model_dir = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    std::string prefix = (argc > 2) ? argv[2] : "/tmp/txn_decode/rt";
    const char* cap_dir = getenv("CAP_DIR") ? getenv("CAP_DIR") : "/tmp/capF2";
    const char* elfdir = getenv("RT_ELF_DIR") ? getenv("RT_ELF_DIR") : "captures/txn-elfs";

    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    std::string model_file = std::string(model_dir) + "/model.q4nx";
    ModelWeights* mw = model_load(model_file.c_str(), cfg);
    if (!mw) { fprintf(stderr, "model_load failed\n"); return 1; }

    xrt::device dev(0);
    RuntimeLayerEngine rt;
    if (!rt.init(dev, mw, cfg, elfdir,
                 "captures/txn-elfs/elf_0002_lmhead.bin")) {
        fprintf(stderr, "RuntimeLayerEngine init failed\n");
        return 1;
    }

    // ---- tokens from RT_TOKENS file (default 1000,1001,1002) ----
    std::vector<int> toks = {1000, 1001, 1002};
    if (const char* tf = getenv("RT_TOKENS")) {
        FILE* ft = fopen(tf, "r");
        if (ft) { toks.clear(); int t; while (fscanf(ft, "%d", &t) == 1) toks.push_back(t); fclose(ft); }
    }
    // ---- forward 1: token toks[0], ctx=1 ----
    if (!rt.embed(toks[0])) { fprintf(stderr, "embed(%d) failed\n", toks[0]); return 1; }
    if (!rt.forward(1)) { fprintf(stderr, "forward(1) failed\n"); return 1; }
    std::string a1 = prefix + "_act_fwd1.bin";
    std::string l1 = prefix + "_logits_fwd1.bin";
    rt.dump_act(a1.c_str());
    rt.dump_logits(l1.c_str(), cfg.vocab_size);
    fprintf(stderr, "fwd1: act -> %s logits -> %s\n", a1.c_str(), l1.c_str());
    if (getenv("RT_ONLY_FWD1")) return 0;

    // ---- forward 2: token toks[1], ctx=2 ----
    if (toks.size() < 2) return 0;
    if (getenv("RT_TEST_GETLOGITS")) {
        float tmp[151936];
        rt.get_logits(tmp, 151936);  // like the engine's prefill
    }
    if (!rt.embed(toks[1])) { fprintf(stderr, "embed(%d) failed\n", toks[1]); return 1; }
    if (!rt.forward(2)) { fprintf(stderr, "forward(2) failed\n"); return 1; }
    std::string a2 = prefix + "_act_fwd2.bin";
    std::string l2 = prefix + "_logits_fwd2.bin";
    rt.dump_act(a2.c_str());
    rt.dump_logits(l2.c_str(), cfg.vocab_size);
    fprintf(stderr, "fwd2: act -> %s logits -> %s\n", a2.c_str(), l2.c_str());

    // ---- forward 3: token 1002, ctx=3 (optional) ----
    if (getenv("RT_FWD3")) {
        if (toks.size() < 3) return 0;
        if (!rt.embed(toks[2])) { fprintf(stderr, "embed(%d) failed\n", toks[2]); return 1; }
        if (!rt.forward(3)) { fprintf(stderr, "forward(3) failed\n"); return 1; }
        rt.dump_act((prefix + "_act_fwd3.bin").c_str());
        rt.dump_logits((prefix + "_logits_fwd3.bin").c_str(), cfg.vocab_size);
        fprintf(stderr, "fwd3: dumped\n");
    }

    // ---- compare vs runtime captures ----
    std::string cap_act1 = glob_first((cap_dir + std::string("/actpost_002_*.bin")).c_str());
    std::string cap_act2 = glob_first((cap_dir + std::string("/actpost_004_*.bin")).c_str());
    std::string cap_act3 = glob_first((cap_dir + std::string("/actpost_006_*.bin")).c_str());
    std::string cap_lg1 = "/tmp/txn_decode/logits_1000.bin";
    std::string cap_lg2 = "/tmp/txn_decode/logits_1001.bin";
    if (!cap_act1.empty()) cmp_file(a1.c_str(), cap_act1.c_str(), "fwd1 act");
    cmp_file(l1.c_str(), cap_lg1.c_str(), "fwd1 logits");
    if (!cap_act2.empty()) cmp_file(a2.c_str(), cap_act2.c_str(), "fwd2 act");
    cmp_file(l2.c_str(), cap_lg2.c_str(), "fwd2 logits");
    if (!cap_act3.empty() && getenv("RT_FWD3"))
        cmp_file((prefix + "_act_fwd3.bin").c_str(), cap_act3.c_str(), "fwd3 act");

    fprintf(stderr, "DONE\n");
    return 0;
}
