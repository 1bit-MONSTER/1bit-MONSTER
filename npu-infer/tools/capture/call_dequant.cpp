#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cstring>
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "buffer.hpp"
#include "typedef.hpp"
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: call_dequant <tile.bin> <columns> <out.bin>\n"); return 1; }
    int columns = atoi(argv[2]);
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> tile((std::istreambuf_iterator<char>(f)), {});
    bytes q4nx(tile.size());
    memcpy(q4nx.data(), tile.data(), tile.size());
    int n = 32 * columns * 4;  // f32 bytes
    bytes weight(n);
    // the host lib's q4nx_dequantize<float>: (weight bytes, q4nx bytes, columns)
    Q4NX::q4nx_dequantize<float>(weight, q4nx, columns);
    FILE* fo = fopen(argv[3], "wb");
    fwrite(weight.data(), 1, n, fo);
    fclose(fo);
    fprintf(stderr, "dequantized %d f32 values columns=%d -> %s\n", 32*columns, columns, argv[3]);
    return 0;
}
