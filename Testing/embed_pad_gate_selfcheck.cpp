// embed_pad_gate_selfcheck.cpp — the padded-vocab embedding gate.
//
// Some 1BP artifacts declare the checkpoint's *padded* vocab in the header while
// carrying the unpadded embedding table: the published v1 ZAYA1-8B upload declares
// 262272 and holds 262147 rows (125 padding rows), which is the file that printed
//
//   zaya_init_onebp: model embed size 536877056 != expected 537133056
//     (cfg H=2048, vocab=262272)
//   Refusing to load — would produce silent garbage.
//
// and then fell through to the dense GPU lane, which cannot run Zaya at all. The
// table defines the real (tied) vocab, so the shortfall is padding and can be
// adopted; anything else must still be refused, because the engine uploads
// vocab*H elements and padding past a real shortfall would read the host buffer
// out of bounds.
//
//   g++ -std=c++17 -Iinclude Testing/embed_pad_gate_selfcheck.cpp -o /tmp/t && /tmp/t

#include "onebp_format.h"

#include <cstdio>
#include <cstdlib>

static int failures = 0;
static int checks = 0;

static void expect(long got, long want, const char* what) {
    checks++;
    if (got != want) {
        fprintf(stderr, "  FAIL %s: got %ld, want %ld\n", what, got, want);
        failures++;
    }
}

int main() {
    const int H = 2048, V = 262272;
    const size_t elems = (size_t)V * H;

    // 1. Exact match — the common case, no adjustment.
    expect(onebp_embed_padding_rows(elems, V, H), 0, "exact match");

    // 2. The published-artifact case: 262147 rows against a declared 262272.
    expect(onebp_embed_padding_rows((size_t)262147 * H, V, H), 125, "v1 ZAYA1-8B padded vocab");

    // 3. The same numbers as the failing log line, so a regression re-reads them.
    checks++;
    if ((size_t)262147 * H != 536877056 || (size_t)262272 * H != 537133056) {
        fprintf(stderr, "  FAIL log-line arithmetic: %zu / %zu\n",
                (size_t)262147 * H, (size_t)262272 * H);
        failures++;
    }

    // 4. A file that already declares the unpadded vocab (post-#1521 conversions).
    expect(onebp_embed_padding_rows((size_t)262147 * H, 262147, H), 0, "self-consistent unpadded");

    // 5. A table *larger* than declared is a wrong file, not padding.
    expect(onebp_embed_padding_rows((size_t)(V + 1) * H, V, H), -1, "bigger table than declared");

    // 6. Not a whole number of rows — corrupt or mis-sized.
    expect(onebp_embed_padding_rows(elems - 1, V, H), -1, "non-multiple of H");

    // 7. Shortfall boundary: ONEBP_EMBED_MAX_PAD_ROWS is accepted, one more is not.
    expect(onebp_embed_padding_rows((size_t)(V - 4096) * H, V, H), 4096, "max padding accepted");
    expect(onebp_embed_padding_rows((size_t)(V - 4097) * H, V, H), -1, "beyond max padding refused");

    // 8. Degenerate inputs.
    expect(onebp_embed_padding_rows(elems, 0, H), -1, "vocab 0");
    expect(onebp_embed_padding_rows(elems, V, 0), -1, "hidden 0");
    expect(onebp_embed_padding_rows(0, V, H), -1, "empty table");

    // 9. Adopting the rows must give a table that exactly fills the adopted vocab —
    //    this is what the engine replaces eng.vocab with before uploading.
    {
        const size_t rows = ((size_t)262147 * H) / H;
        expect((long)rows, 262147, "adopted vocab == table rows");
        expect((long)(rows * H), (long)((size_t)262147 * H), "adopted vocab*H == table size");
    }

    if (failures) {
        fprintf(stderr, "embed pad gate FAILED (%d/%d)\n", failures, checks);
        return 1;
    }
    printf("embed pad gate OK (%d checks)\n", checks);
    return 0;
}
