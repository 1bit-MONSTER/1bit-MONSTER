// detokenize.cpp — ID→text for NPU engine. Reads comma-sep IDs from stdin.
// g++ -std=c++17 -O3 -o detokenize detokenize.cpp
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

constexpr int kMaxVocab    = 200000;
constexpr int kMaxTokenLen = 256;

static char   vocab[kMaxVocab][kMaxTokenLen]{};
static int    vocab_size = 0;

// ── GPT-2 byte-level BPE decode ────────────────────────────────────────────────
// tokenizer.json stores vocabulary strings in GPT-2's *byte-level* form: every raw
// byte is mapped to a printable Unicode codepoint (space -> U+0120 'G', newline ->
// U+010A 'C', and any byte >= 0x80 to a codepoint at 256+n). Printing those strings
// verbatim is why output looked like "H<accent>O" instead of "H2O" with a subscript.
// Reversing the mapping recovers the original bytes, which are already valid UTF-8.
static int g_rev[512];   // codepoint -> original byte, -1 when unmapped

static void init_rev() {
    static bool printable[256];
    for (int b = 0; b < 256; b++) printable[b] = false;
    for (int b = 33;  b <= 126; b++) printable[b] = true;
    for (int b = 161; b <= 172; b++) printable[b] = true;
    for (int b = 174; b <= 255; b++) printable[b] = true;
    for (int i = 0; i < 512; i++) g_rev[i] = -1;
    for (int b = 0; b < 256; b++) if (printable[b]) g_rev[b] = b;
    int n = 0;
    for (int b = 0; b < 256; b++) if (!printable[b]) { int cp = 256 + n++; if (cp < 512) g_rev[cp] = b; }
}

static void emit_decoded(const char *s) {
    const unsigned char *p = reinterpret_cast<const unsigned char *>(s);
    while (*p) {
        int cp = -1, len = 1;
        if (*p < 0x80)                                          { cp = *p; len = 1; }
        else if ((*p & 0xE0) == 0xC0 && p[1])                    { cp = ((*p & 0x1F) << 6)  |  (p[1] & 0x3F); len = 2; }
        else if ((*p & 0xF0) == 0xE0 && p[1] && p[2])            { cp = ((*p & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); len = 3; }
        else if ((*p & 0xF8) == 0xF0 && p[1] && p[2] && p[3])    { cp = ((*p & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F); len = 4; }
        if (cp >= 0 && cp < 512 && g_rev[cp] >= 0) std::fputc(g_rev[cp], stdout);
        else                                       std::fwrite(p, 1, len, stdout);
        p += len;
    }
}

static void load(const char *path) {
    FILE *f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    char line[512];
    bool in_vocab = false;
    while (std::fgets(line, sizeof(line), f)) {
        if (!in_vocab) {
            if (std::strstr(line, "\"vocab\"")) in_vocab = true;
            continue;
        }
        if (std::strstr(line, "}") && !std::strstr(line, "\"")) break;
        char *tok = std::strchr(line, '"');
        if (!tok) continue;
        tok++;
        char *te = std::strchr(tok, '"');
        if (!te) continue;
        int tlen = static_cast<int>(te - tok);
        if (tlen >= kMaxTokenLen) tlen = kMaxTokenLen - 1;
        char *idp = std::strchr(te + 1, ':');
        if (!idp) continue;
        int id = static_cast<int>(std::strtol(idp + 1, nullptr, 10));
        if (id < 0 || id >= kMaxVocab) continue;
        std::memcpy(vocab[id], tok, tlen);
        vocab[id][tlen] = 0;
        if (id >= vocab_size) vocab_size = id + 1;
    }
    // Merge added_tokens array
    std::rewind(f);
    in_vocab = false;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strstr(line, "\"added_tokens\"")) { in_vocab = true; continue; }
        if (!in_vocab) continue;
        if (std::strstr(line, "]")) break;
        char *idp = std::strstr(line, "\"id\"");
        char *ctp = std::strstr(line, "\"content\"");
        if (!idp || !ctp) continue;
        int id = static_cast<int>(std::strtol(std::strchr(idp, ':') + 1, nullptr, 10));
        char *cs = std::strchr(ctp, '"') + 1;
        char *ce = std::strchr(cs, '"');
        if (id < 0 || id >= kMaxVocab || !ce) continue;
        int clen = static_cast<int>(ce - cs);
        if (clen >= kMaxTokenLen) clen = kMaxTokenLen - 1;
        std::memcpy(vocab[id], cs, clen);
        vocab[id][clen] = 0;
        if (id >= vocab_size) vocab_size = id + 1;
    }
    std::fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: detokenize <tokenizer.json>\n");
        return 1;
    }
    load(argv[1]);
    init_rev();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    char line[65536];
    while (std::fgets(line, sizeof(line), stdin)) {
        char *p = line;
        while (*p) {
            while (*p && !std::isdigit(static_cast<unsigned char>(*p)) && *p != '-') p++;
            if (!*p) break;
            int id = static_cast<int>(std::strtol(p, &p, 10));
            if (id >= 0 && id < vocab_size && vocab[id][0])
                emit_decoded(vocab[id]);
            if (*p == ',') p++;
        }
    }
    std::printf("\n");
    return 0;
}
