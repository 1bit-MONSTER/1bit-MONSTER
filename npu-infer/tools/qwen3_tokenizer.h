// qwen3_tokenizer.h — minimal HF-format byte-level BPE tokenizer for Qwen3.
//
// Reads tokenizer.json (HF format) directly and implements the byte-level
// BPE used by Qwen3/Qwen2.5-family models to match transformers/tokenizers:
//   NFC input (assumed; caller normalizes if needed) ->
//   special-token pre-split (<|im_start|> etc. stay whole tokens) ->
//   GPT-2-style regex pre-tokenization (PCRE2, \p{L}/\p{N} unicode classes) ->
//   byte<->unicode mapping (tiktoken table) ->
//   rank-ordered BPE merges.
//
// Decode inverts byte<->unicode so the engine can print real text from the
// sampled token stream (round 37 remaining item: "real-prompt tokenizer
// wiring for coherent engine generation").
//
// Note: engine output vocab is 151936 but the tokenizer covers 151643 vocab
// + 26 added tokens (ids 0..151668); ids above that (unreachable by the
// tokenizer) decode to empty text.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class Qwen3Tokenizer {
public:
    bool load(const std::string& tokenizer_json_path);
    bool loaded() const { return loaded_; }

    /// Encode UTF-8 text to token ids. Special tokens (<|im_start|>, ...)
    /// map to their single ids; everything else goes through byte-level BPE.
    std::vector<int> encode(const std::string& text) const;

    /// Decode token ids to UTF-8 text (byte-level inverse; special/added
    /// tokens and raw ASCII pass through literally).
    std::string decode(const std::vector<int>& ids) const;

    /// True for the Qwen3 end-of-sequence ids (<|endoftext|>, <|im_end|>).
    static bool is_eos(int id);

    /// Standard Qwen3 chat scaffold used by main.cpp for chat-mode prompts.
    static std::string chat_prompt(const std::string& system,
                                   const std::string& user);

    int vocab_size() const { return (int)vocab_.size(); }

private:
    // token string -> id (vocab + added tokens)
    std::unordered_map<std::string, int> vocab_;
    // id -> token string (dense over 0..max_id); "" for gaps
    std::vector<std::string> id_to_text_;
    // special added tokens, longest first, for the pre-split
    std::vector<std::pair<std::string, int>> specials_;
    // byte <-> unicode (tiktoken byte-level tables)
    std::vector<uint32_t> byte_to_char_;   // 256 entries
    std::unordered_map<uint32_t, int> char_to_byte_;
    // merge ranks: pair key (a SEP b) -> rank
    std::unordered_map<std::string, int> merge_ranks_;
    bool loaded_ = false;

    // BPE over one pre-tokenized piece (already byte->unicode mapped).
    std::vector<int> bpe_piece(const std::string& chars) const;
    void build_byte_tables();
};

// ---- minimal JSON DOM (subset sufficient for tokenizer.json) ----
namespace tokjson {
struct Value;
using Object = std::vector<std::pair<std::string, Value>>;
struct Value {
    enum Type { NUL, BOOL, NUM, STR, OBJ, ARR } type = NUL;
    bool b = false;
    double num = 0;
    std::string str;
    Object obj;
    std::vector<Value> arr;
};
bool parse(const std::string& src, Value& out);
const Value* find(const Value& obj, const std::string& key);
}  // namespace tokjson
