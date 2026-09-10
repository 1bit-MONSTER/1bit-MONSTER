#include <cstdio>
#include <vector>
#include <fstream>
#include <xrt/xrt_device.h>
#include <xrt/experimental/xrt_xclbin.h>
int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3.6-35B-A3B-NPU2/layer.xclbin";
    std::ifstream f(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    xrt::xclbin xclbin(raw);
    auto ks = xclbin.get_kernels();
    printf("kernels: %zu\n", ks.size());
    for (auto& k : ks) {
        printf("  kernel: %s\n", k.get_name().c_str());
        auto args = k.get_args();
        printf("  args: %zu\n", args.size());
        for (auto& a : args) {
            printf("    arg: name=%s index=%zu\n", a.get_name().c_str());
        }
    }
    return 0;
}
