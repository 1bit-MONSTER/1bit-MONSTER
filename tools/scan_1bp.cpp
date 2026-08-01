// scan_1bp.c — quick tensor-name scanner using the CURRENT OnebpModel loader.
// Usage: scan_1bp model.1bp [expected_tensor_name]
// Exits 1 if the file won't load; prints tensor count and whether the
// expected tensor (default token_embd.weight) is present.
#include <cstdio>
#include <cstring>
#include <string>
#include "onebp_loader.cpp"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.1bp [name]\n", argv[0]); return 2; }
    const char* want = argc > 2 ? argv[2] : "token_embd.weight";
    OnebpModel m;
    if (!m.open(argv[1])) { printf("NOOPEN %s\n", argv[1]); return 1; }
    int n = m.tensor_count();
    bool found = false;
    for (int i = 0; i < n; i++) {
        auto* t = m.tensor(i);
        if (t && t->name == want) { found = true; break; }
    }
    printf("%s tensors=%d %s=%s\n", argv[1], n, want, found ? "YES" : "NO");
    return found ? 0 : 1;
}
