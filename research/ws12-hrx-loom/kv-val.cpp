#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
int main(int argc, char ** argv) {
    gguf_init_params p = { false, nullptr };
    gguf_context * c = gguf_init_from_file(argv[1], p);
    int n = (int) gguf_get_n_kv(c);
    for (int i = 0; i < n; ++i) {
        const char * k = gguf_get_key(c, i);
        enum gguf_type t = gguf_get_kv_type(c, i);
        printf("%s=%d:", k, (int) t);
        switch (t) {
            case GGUF_TYPE_UINT32: printf("%u", gguf_get_val_u32(c, i)); break;
            case GGUF_TYPE_INT32: printf("%d", gguf_get_val_i32(c, i)); break;
            case GGUF_TYPE_FLOAT32: printf("%f", gguf_get_val_f32(c, i)); break;
            case GGUF_TYPE_UINT64: printf("%llu", (unsigned long long) gguf_get_val_u64(c, i)); break;
            case GGUF_TYPE_INT64: printf("%lld", (long long) gguf_get_val_i64(c, i)); break;
            case GGUF_TYPE_FLOAT64: printf("%f", gguf_get_val_f64(c, i)); break;
            case GGUF_TYPE_BOOL: printf("%d", gguf_get_val_bool(c, i)); break;
            case GGUF_TYPE_STRING: printf("%s", gguf_get_val_str(c, i)); break;
            default: {
                int cnt = (int) gguf_get_arr_n(c, i);
                printf("arr[%d]", cnt);
                enum gguf_type at = gguf_get_arr_type(c, i);
                const void * d = (at == GGUF_TYPE_STRING || at == GGUF_TYPE_ARRAY) ? nullptr : gguf_get_arr_data(c, i);
                if (at == GGUF_TYPE_STRING) {
                    for (int j = 0; j < cnt; ++j) printf(" %s", gguf_get_arr_str(c, i, j));
                } else if (at == GGUF_TYPE_INT32) {
                    const int32_t * p = (const int32_t *) d;
                    for (int j = 0; j < cnt; ++j) printf(" %d", p[j]);
                } else if (at == GGUF_TYPE_UINT32) {
                    const uint32_t * p = (const uint32_t *) d;
                    for (int j = 0; j < cnt; ++j) printf(" %u", p[j]);
                } else if (at == GGUF_TYPE_FLOAT32) {
                    const float * p = (const float *) d;
                    for (int j = 0; j < cnt; ++j) printf(" %g", p[j]);
                } else if (at == GGUF_TYPE_INT64) {
                    const int64_t * p = (const int64_t *) d;
                    for (int j = 0; j < cnt; ++j) printf(" %lld", (long long) p[j]);
                } else if (at == GGUF_TYPE_UINT64) {
                    const uint64_t * p = (const uint64_t *) d;
                    for (int j = 0; j < cnt; ++j) printf(" %llu", (unsigned long long) p[j]);
                } else if (at == GGUF_TYPE_BOOL) {
                    const bool * p = (const bool *) d;
                    for (int j = 0; j < cnt; ++j) printf(" %d", (int) p[j]);
                }
            }
        }
        printf("\n");
    }
    return 0;
}
