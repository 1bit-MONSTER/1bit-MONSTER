#include <cstdio>
#include <cstring>
#include "rocm_cpp/bitnet_model.h"
int main() {
    char buf[512];
    while (fgets(buf, sizeof buf, stdin)) {
        buf[strcspn(buf, "\n")] = 0;
        std::printf("%d\n", (int)rcpp_arch_from_string(buf));
    }
    return 0;
}
