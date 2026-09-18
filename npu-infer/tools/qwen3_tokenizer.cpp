// qwen3_tokenizer.cpp — implementation of the Qwen3 byte-level BPE tokenizer.
// See qwen3_tokenizer.h. Validated against the `tokenizers` python package
// (HF tokenizer.json semantics) on plain-text prompts.
#include "qwen3_tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ===================== minimal JSON parser =====================
namespace tokjson {

static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static bool parse_string(const char*& p, std::string& out) {
    // p points at opening quote
    if (*p != '"') return false;
    p++;
    out.clear();
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case '"': out += '"'; p++; break;
                case '\\': out += '\\'; p++; break;
                case '/': out += '/'; p++; break;
                case 'b': out += '\b'; p++; break;
                case 'f': out += '\f'; p++; break;
                case 'n': out += '\n'; p++; break;
                case 'r': out += '\r'; p++; break;
                case 't': out += '\t'; p++; break;
                case 'u': {
                    // \uXXXX (surrogate pairs handled by caller via raw pass-through)
                    char hex[5] = {0};
                    if (!isxdigit((unsigned char)p[1]) || !isxdigit((unsigned char)p[2]) ||
                        !isxdigit((unsigned char)p[3]) || !isxdigit((unsigned char)p[4]))
                        return false;
                    memcpy(hex, p + 1, 4);
                    p += 5;
                    unsigned cp = (unsigned)strtoul(hex, nullptr, 16);
                    if (cp >= 0xD800 && cp <= 0xDBFF && p[1] == '\\' && p[2] == 'u') {
                        // high surrogate: expect low surrogate
                        char hex2[5] = {0};
                        memcpy(hex2, p + 3, 4);
                        unsigned lo = (unsigned)strtoul(hex2, nullptr, 16);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            p += 6;
                        }
                    }
                    // encode cp as UTF-8
                    if (cp < 0x80) out += (char)cp;
                    else if (cp < 0x800) {
                        out += (char)(0xC0 | (cp >> 6));
                        out += (char)(0x80 | (cp & 0x3F));
                    } else if (cp < 0x10000) {
                        out += (char)(0xE0 | (cp >> 12));
                        out += (char)(0x80 | ((cp >> 6) & 0x3F));
                        out += (char)(0x80 | (cp & 0x3F));
                    } else {
                        out += (char)(0xF0 | (cp >> 18));
                        out += (char)(0x80 | ((cp >> 12) & 0x3F));
                        out += (char)(0x80 | ((cp >> 6) & 0x3F));
                        out += (char)(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: return false;
            }
        } else {
            out += *p++;
        }
    }
    if (*p != '"') return false;
    p++;
    return true;
}

static bool parse_value(const char*& p, Value& out);

static bool parse_object(const char*& p, Value& out) {
    p = skip_ws(p + 1);  // past '{'
    out.type = Value::OBJ;
    while (*p && *p != '}') {
        p = skip_ws(p);
        std::string key;
        if (!parse_string(p, key)) return false;
        p = skip_ws(p);
        if (*p != ':') return false;
        p = skip_ws(p + 1);
        Value v;
        if (!parse_value(p, v)) return false;
        out.obj.emplace_back(std::move(key), std::move(v));
        p = skip_ws(p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;
        return false;
    }
    if (*p != '}') return false;
    p++;
    return true;
}

static bool parse_array(const char*& p, Value& out) {
    p = skip_ws(p + 1);  // past '['
    out.type = Value::ARR;
    while (*p && *p != ']') {
        p = skip_ws(p);
        Value v;
        if (!parse_value(p, v)) return false;
        out.arr.push_back(std::move(v));
        p = skip_ws(p);
        if (*p == ',') { p++; continue; }
        if (*p == ']') break;
        return false;
    }
    if (*p != ']') return false;
    p++;
    return true;
}

static bool parse_value(const char*& p, Value& out) {
    p = skip_ws(p);
    if (!*p) return false;
    if (*p == '{') return parse_object(p, out);
    if (*p == '[') return parse_array(p, out);
    if (*p == '"') {
        out.type = Value::STR;
        return parse_string(p, out.str);
    }
    if (strncmp(p, "true", 4) == 0) { out.type = Value::BOOL; out.b = true; p += 4; return true; }
    if (strncmp(p, "false", 5) == 0) { out.type = Value::BOOL; out.b = false; p += 5; return true; }
    if (strncmp(p, "null", 4) == 0) { out.type = Value::NUL; p += 4; return true; }
    // number
    char* end = nullptr;
    out.num = strtod(p, &end);
    if (end == p) return false;
    out.type = Value::NUM;
    p = end;
    return true;
}

bool parse(const std::string& src, Value& out) {
    const char* p = src.c_str();
    return parse_value(p, out);
}

const Value* find(const Value& obj, const std::string& key) {
    if (obj.type != Value::OBJ) return nullptr;
    for (const auto& kv : obj.obj)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

}  // namespace tokjson

// ===================== byte<->unicode tables =====================
void Qwen3Tokenizer::build_byte_tables() {
    byte_to_char_.assign(256, 0);
    // GPT-2 / tiktoken bytes_to_unicode: printable ASCII + Latin-1 keep
    // their codepoint; the rest are assigned U+0100+n in order.
    std::vector<int> keep;
    for (int b = '!'; b <= '~'; b++) keep.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) keep.push_back(b);  // ¡..¬
    for (int b = 0xAE; b <= 0xFF; b++) keep.push_back(b);  // ®..ÿ
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool in_keep = false;
        for (int k : keep) if (k == b) { in_keep = true; break; }
        uint32_t cp = in_keep ? (uint32_t)b : (uint32_t)(256 + (n++));
        byte_to_char_[b] = cp;
        char_to_byte_[cp] = b;
    }
}

static void utf8_append(std::string& s, uint32_t cp) {
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
}

// map UTF-8 bytes of `utf8` to the byte-level unicode-char string
static std::string bytes_to_chars(const std::string& utf8,
                                  const std::vector<uint32_t>& b2c) {
    std::string out;
    for (unsigned char b : utf8) utf8_append(out, b2c[b]);
    return out;
}

// decode a byte-level unicode-char string back to raw bytes; characters not
// in the byte table (literal text of special/added tokens) pass through as
// UTF-8.
static std::string chars_to_bytes(const std::string& chars,
                                  const std::unordered_map<uint32_t, int>& c2b) {
    std::string out;
    for (size_t i = 0; i < chars.size();) {
        unsigned char c = (unsigned char)chars[i];
        uint32_t cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
        else { out += chars[i]; i++; continue; }
        if (i + len > chars.size()) { out += chars[i]; i++; continue; }
        for (int j = 1; j < len; j++) cp = (cp << 6) | ((unsigned char)chars[i + j] & 0x3F);
        auto it = c2b.find(cp);
        if (it != c2b.end()) out += (char)it->second;
        else {
            // literal pass-through (specials / added tokens contain ASCII)
            for (int j = 0; j < len; j++) out += chars[i + j];
        }
        i += len;
    }
    return out;
}

// ===================== GPT-2 regex pre-tokenization =====================
static const char* GPT2_PATTERN =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

struct RegexSplitter {
    pcre2_code* re = nullptr;
    bool ok = false;
    RegexSplitter(const std::string& pattern) {
        int err = 0;
        PCRE2_SIZE erroff = 0;
        re = pcre2_compile((PCRE2_SPTR)pattern.c_str(), pattern.size(),
                           PCRE2_UTF | PCRE2_UCP, &err, &erroff, nullptr);
        if (!re) {
            fprintf(stderr, "tokenizer: pcre2 compile error %d at %zu — "
                            "falling back to the GPT-2 pattern\n", err, (size_t)erroff);
            re = pcre2_compile((PCRE2_SPTR)GPT2_PATTERN, strlen(GPT2_PATTERN),
                               PCRE2_UTF | PCRE2_UCP, &err, &erroff, nullptr);
        }
        ok = (re != nullptr);
    }
    ~RegexSplitter() { if (re) pcre2_code_free(re); }

    // Returns the UTF-8 pieces the pattern matches; any text between matches
    // is preserved as its own piece so nothing is dropped.
    std::vector<std::string> split(const std::string& text) {
        std::vector<std::string> pieces;
        if (!ok) { pieces.push_back(text); return pieces; }
        pcre2_match_data* md = pcre2_match_data_create_from_pattern(re, nullptr);
        PCRE2_SIZE off = 0;
        size_t last_end = 0;
        while (off < text.size()) {
            int rc = pcre2_match(re, (PCRE2_SPTR)text.data(), text.size(), off, 0, md, nullptr);
            if (rc < 0) break;  // PCRE2_ERROR_NOMATCH
            PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
            PCRE2_SIZE start = ov[0], end = ov[1];
            if (end == start) { off = start + 1; continue; }  // empty match guard
            if (start > last_end)
                pieces.push_back(text.substr(last_end, start - last_end));
            pieces.push_back(text.substr(start, end - start));
            last_end = end;
            off = end;
        }
        if (last_end < text.size()) pieces.push_back(text.substr(last_end));
        pcre2_match_data_free(md);
        return pieces;
    }
};

// ===================== loader =====================
bool Qwen3Tokenizer::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "tokenizer: cannot open %s\n", path.c_str()); return false; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string src((size_t)sz, '\0');
    size_t rd = fread(&src[0], 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) return false;

    build_byte_tables();

    tokjson::Value root;
    if (!tokjson::parse(src, root)) {
        fprintf(stderr, "tokenizer: JSON parse failed for %s\n", path.c_str());
        return false;
    }
    const tokjson::Value* model = tokjson::find(root, "model");
    if (!model) { fprintf(stderr, "tokenizer: no model section\n"); return false; }

    // ---- vocab ----
    const tokjson::Value* vocab = tokjson::find(*model, "vocab");
    if (!vocab || vocab->type != tokjson::Value::OBJ) {
        fprintf(stderr, "tokenizer: no vocab\n"); return false;
    }
    int max_id = -1;
    for (const auto& kv : vocab->obj) {
        int id = (int)kv.second.num;
        vocab_[kv.first] = id;
        if (id > max_id) max_id = id;
    }

    // ---- added tokens (specials for the pre-split; all map id->text) ----
    if (const tokjson::Value* added = tokjson::find(root, "added_tokens")) {
        if (added->type == tokjson::Value::ARR) {
            for (const auto& at : added->arr) {
                const tokjson::Value* content = tokjson::find(at, "content");
                const tokjson::Value* idv = tokjson::find(at, "id");
                const tokjson::Value* spv = tokjson::find(at, "special");
                if (!content || !idv) continue;
                int id = (int)idv->num;
                vocab_[content->str] = id;
                if (id > max_id) max_id = id;
                bool special = spv && spv->type == tokjson::Value::BOOL && spv->b;
                if (special) specials_.emplace_back(content->str, id);
            }
        }
    }
    // sort specials longest-first so <|im_start|> style prefixes split right
    std::sort(specials_.begin(), specials_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    // ---- merges -> ranks (index in the list = rank; lower = earlier) ----
    const tokjson::Value* merges = tokjson::find(*model, "merges");
    if (merges && merges->type == tokjson::Value::ARR) {
        for (size_t i = 0; i < merges->arr.size(); i++) {
            const auto& pair = merges->arr[i];
            if (pair.type != tokjson::Value::ARR || pair.arr.size() < 2) continue;
            merge_ranks_[pair.arr[0].str + "\x1f" + pair.arr[1].str] = (int)i;
        }
    }

    // ---- dense id->text (for decode) ----
    id_to_text_.assign((size_t)max_id + 1, "");
    for (const auto& kv : vocab_) {
        int id = kv.second;
        if (id >= 0 && id <= max_id) id_to_text_[id] = kv.first;
    }

    loaded_ = true;
    fprintf(stderr, "tokenizer: loaded %s — vocab %zu, merges %zu, specials %zu\n",
            path.c_str(), vocab_.size(), merge_ranks_.size(), specials_.size());
    return true;
}

// ===================== encode =====================
static std::vector<int> split_specials(const Qwen3Tokenizer& /*unused*/) { return {}; }

std::vector<int> Qwen3Tokenizer::bpe_piece(const std::string& chars) const {
    // chars = byte-level unicode-char string of one regex piece
    std::vector<std::string> word;
    for (size_t i = 0; i < chars.size();) {
        unsigned char c = (unsigned char)chars[i];
        int len = (c < 0x80) ? 1 : ((c >> 5) == 0x6 ? 2 : ((c >> 4) == 0xE ? 3 : 4));
        word.push_back(chars.substr(i, (size_t)len));
        i += (size_t)len;
    }
    // merge loop: repeatedly find the lowest-rank adjacent pair
    while (word.size() > 1) {
        int best_rank = -1;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < word.size(); i++) {
            std::string key = word[i] + "\x1f" + word[i + 1];
            auto it = merge_ranks_.find(key);
            if (it != merge_ranks_.end() && (best_rank < 0 || it->second < best_rank)) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_rank < 0) break;
        word[best_i] += word[best_i + 1];
        word.erase(word.begin() + (long)best_i + 1);
    }
    std::vector<int> ids;
    for (const auto& w : word) {
        auto it = vocab_.find(w);
        if (it != vocab_.end()) ids.push_back(it->second);
        // full byte coverage: every byte-level char string is in the vocab
    }
    return ids;
}

std::vector<int> Qwen3Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    if (text.empty()) return ids;

    // Segment the text at special-token strings (whole-token semantics, like
    // tokenizers' split_special_tokens). Plain runs between specials go
    // through regex pre-split + byte-level BPE.
    std::vector<std::pair<std::string, bool>> segs;  // (text, is_special)
    size_t pos = 0;
    while (pos < text.size()) {
        // earliest special occurrence at/after pos
        size_t best = std::string::npos;
        std::string bs;
        for (const auto& sp : specials_) {
            size_t at = text.find(sp.first, pos);
            if (at != std::string::npos && (best == std::string::npos || at < best)) {
                best = at;
                bs = sp.first;
            }
        }
        if (best == std::string::npos) {
            if (pos < text.size()) segs.emplace_back(text.substr(pos), false);
            break;
        }
        if (best > pos) segs.emplace_back(text.substr(pos, best - pos), false);
        segs.emplace_back(bs, true);
        pos = best + bs.size();
    }

    RegexSplitter splitter(GPT2_PATTERN);
    for (const auto& seg : segs) {
        if (seg.first.empty()) continue;
        if (seg.second) {
            auto it = vocab_.find(seg.first);
            if (it != vocab_.end()) ids.push_back(it->second);
            continue;
        }
        for (const auto& piece : splitter.split(seg.first)) {
            if (piece.empty()) continue;
            std::string chars = bytes_to_chars(piece, byte_to_char_);
            std::vector<int> pids = bpe_piece(chars);
            ids.insert(ids.end(), pids.begin(), pids.end());
        }
    }
    return ids;
}

// ===================== decode =====================
std::string Qwen3Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || (size_t)id >= id_to_text_.size()) continue;
        const std::string& t = id_to_text_[id];
        if (t.empty()) continue;
        out += chars_to_bytes(t, char_to_byte_);
    }
    return out;
}

bool Qwen3Tokenizer::is_eos(int id) {
    return id == 151643 /* <|endoftext|> */ || id == 151645 /* <|im_end|> */;
}

std::string Qwen3Tokenizer::chat_prompt(const std::string& system,
                                        const std::string& user) {
    return "<|im_start|>system\n" + system + "<|im_end|>\n"
           "<|im_start|>user\n" + user + "<|im_end|>\n"
           "<|im_start|>assistant\n";
}
