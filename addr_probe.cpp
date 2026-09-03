#include <cstdio>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/experimental/xrt_ext.h>
int main() {
    xrt::device dev(0);
    std::vector<xrt::ext::bo> bos;
    // sizes in the runtime's allocation order (from moe-cap4 EXTBO manifest)
    unsigned long sizes[] = {
        536870912,536870912,536870912,536870912,536870912,536870912,536870912,
        536870912,536870912,536870912,536870912,536870912,536870912,536870912,536870912,
        134217728,134217728,134217728,134217728,134217728,134217728,134217728,134217728,
        1048576,1048576,1048576,1048576,1048576,1048576,1048576,1048576,
        2097152,2097152,2097152,2097152,2097152,2097152,2097152,2097152,
        5242880,5242880,5242880,5242880,5242880,5242880,5242880,5242880,
    };
    int n = sizeof(sizes)/sizeof(sizes[0]);
    for (int i = 0; i < n; i++) {
        try {
            xrt::ext::bo b(dev, sizes[i]);
            unsigned long long a = b.address();
            printf("BO %2d size=%-10lu addr=0x%llx (%llu MB)\n", i, sizes[i], a, a>>20);
            bos.push_back(std::move(b));
        } catch (const std::exception& e) {
            printf("BO %2d alloc failed: %s\n", i, e.what());
            break;
        }
    }
    return 0;
}
