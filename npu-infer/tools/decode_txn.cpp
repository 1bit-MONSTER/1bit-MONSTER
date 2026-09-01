// decode_txn.cpp — decode the FastFlowLM runtime's per-layer TXN instruction
// streams and their weight block descriptors (issues #2006/#2015).
//
// Generates the runtime's ACTUAL control code with the same generators the
// runtime links against (libqwen3_npu.so / libgemm.so / libdequant.so):
//   layer TXN   : qwen3_npu_sequence::gen_layer_seq(seq, L)
//   lm_head TXN : qwen3_npu_sequence::gen_lm_head_seq(seq)
//   mha TXN     : qwen3_npu_sequence::gen_mha_engine_seq(seq, L0, L1)
//   dequant TXN : Dequant::generate_dequant_q80_packed_in_q4nx_seq /
//                 generate_dequant_q4_1_seq (per projection block, modes 0..7)
// and decodes each into structured JSON: TXN header, every command
// (BLOCKWRITE / DDR_PATCH / WRITE / MASKWRITE / TCT) with its DMA BD fields,
// plus the BD -> (arg_idx, arg_offset) patch table — the WEIGHT-BD map that
// says which kernel-arg BO (and byte offset inside it) each BD's DMA touches.
//
// The patch table is the point: the mm kernel's weight slot is arg 5
// (kernel ABI: opcode=0, instr_bo=1, ninstr=2, act=3, ws=4, w1=5, w2=6,
// kv=7). Every DDR_PATCH with arg_idx 5/6 names a weight BD: its
// buffer_length / buffer_offset / D0/D1/D2 dims + strides describe EXACTLY
// how the runtime lays the weights out in the weight BO — the layout the
// hand-rolled npu-infer packer (npu_pack_weight_bo) must reproduce instead
// of guessing, and the layout the captured B0 (npu-infer/captures/) embodies.
//
// Also decodes raw insts .bin files given as extra args (e.g. the hand-rolled
// per-shape streams from gen_mm_insts_batch) so they can be diffed against
// the runtime's own TXNs (closing the insts/layout mismatch, #2015).
//
// Build (same libs as tools/gen_mm_insts_batch):
//   g++ -O2 -std=c++17 -include climits -mavx2 decode_txn.cpp -o decode_txn \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt -lqwen3_npu -lgemm -lmha -lq4_npu_eXpress \
//     -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt
// Run:
//   ./decode_txn <config.json> <outdir> [n_tokens=128] [insts.bin ...]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <nlohmann/json.hpp>
#include "npu_utils/npu_instr_utils.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "lm_config.hpp"
#include "models/qwen3/qwen3_npu_sequence.hpp"

// ===========================================================================
// Raw-word command decoder — self-contained, mirrors npu_cmd_*.hpp layouts.
// ===========================================================================

struct Cmd {
    std::string type;      // BLOCKWRITE | MT_BLOCKWRITE | DDR_PATCH | WRITE | MASKWRITE | TCT | PREEMPT | UNKNOWN
    uint32_t word;         // sequence word index of the op header
    uint32_t n_words;
    // location (most cmds)
    uint32_t row = 0, col = 0;
    // BLOCKWRITE
    uint32_t bd_id = 0;
    uint32_t buffer_length = 0, buffer_offset = 0;
    bool packet_enable = false, linear = false;
    uint32_t packet_id = 0, packet_type = 0, out_of_order_id = 0;
    uint32_t dim0_size = 0, dim0_stride = 0;
    uint32_t dim1_size = 0, dim1_stride = 0;
    uint32_t dim2_stride = 0, cache_flag = 0;
    uint32_t iter_size = 0, iter_stride = 0;
    uint32_t next_bd_id = 0, valid_bd = 0;
    // MT_BLOCKWRITE (memtile BD-register write, 6-word op=1 variant)
    uint32_t mt_addr = 0, mt_value = 0;
    // DDR_PATCH
    uint32_t arg_idx = 0, arg_offset = 0;
    // WRITE
    uint32_t reg_addr = 0, value = 0;
    bool push_queue = false, issue_token = false;
    uint32_t channel_id = 0, repeat_count = 0;
    // MASKWRITE
    uint32_t addr = 0, mask = 0;
    // TCT
    uint32_t wait_row = 0, wait_col = 0, wait_channel = 0;
    bool wait_mm2s = false;
    // PREEMPT
    uint32_t preemption_level = 0;
    // DMA direction derived from the paired queue WRITE (0x1d204/0x1d20c =
    // S2MM ch0/1 = tile->BO, 0x1d214/0x1d21c = MM2S ch0/1 = BO->tile).
    bool dir_known = false;
    bool dir_mm2s = false;   // true: BO -> tile (source read); false: tile -> BO (dest write)
};

// op headers (op_headers enum, npu_cmd.hpp)
enum { OP_WRITE = 0x00, OP_BLOCKWRITE = 0x01, OP_MASKWRITE = 0x03,
       OP_PREEMPT = 0x06, OP_TCT = 0x80, OP_DDR_PATCH = 0x81 };

static const char* op_name(uint32_t h) {
    switch (h) {
        case OP_WRITE: return "WRITE";
        case OP_BLOCKWRITE: return "BLOCKWRITE";
        case OP_MASKWRITE: return "MASKWRITE";
        case OP_PREEMPT: return "PREEMPT";
        case OP_TCT: return "TCT";
        case OP_DDR_PATCH: return "DDR_PATCH";
        default: return "UNKNOWN";
    }
}

// A TXN header looks like (rows<<24)|(devGen<<16)|(minor<<8)|major at w0,
// (memTileRows<<8)|numCols at w1, op_count at w2, total bytes at w3.
// FastFlowLM's generators emit headerless op runs (e.g. the layer TXN's
// phase-1 memtile setup) and embed the real TXN header mid-stream — so the
// parser must detect headers anywhere, not just at word 0.
// The shipped .bin insts (mm.bin etc.) use the aiebu TXN header instead:
// [magic 0x535f544e ("NTS_" le)][version 0xd00][op_count][total_bytes].
static bool looks_like_header(const std::vector<uint32_t>& w, size_t i) {
    if (i + 4 > w.size()) return false;
    if (w[i] == 0x535f544e) {   // aiebu TXN magic
        uint32_t op_count = w[i+2], total_bytes = w[i+3];
        if (op_count == 0 || op_count > 100000) return false;
        if (total_bytes == 0 || total_bytes % 4 != 0) return false;
        return total_bytes <= (uint32_t)((w.size() - i) * 4);
    }
    uint32_t w0 = w[i];
    uint32_t rows = (w0 >> 24) & 0xFF, gen = (w0 >> 16) & 0xFF,
             minor = (w0 >> 8) & 0xFF, major = w0 & 0xFF;
    if (rows < 1 || rows > 8 || gen < 1 || gen > 8 || minor > 2 || major > 1) return false;
    uint32_t cols = w[i+1] & 0xFF, mtrows = (w[i+1] >> 8) & 0xFF;
    if (cols < 1 || cols > 16 || mtrows > 8) return false;
    uint32_t op_count = w[i+2], total_bytes = w[i+3];
    if (op_count == 0 || op_count > 100000) return false;
    if (total_bytes == 0 || total_bytes % 4 != 0) return false;
    // The claimed byte count must fit within the remaining buffer.
    uint32_t span = (uint32_t)((w.size() - i) * 4);
    return total_bytes <= span;
}

// Decode the whole word array into commands, starting at `start`, detecting
// embedded TXN headers as phase boundaries.
struct DecodeResult {
    std::vector<Cmd> cmds;
    std::vector<size_t> header_positions;  // word indices of detected headers
};
static DecodeResult decode_words(const std::vector<uint32_t>& w, size_t start) {
    DecodeResult res;
    std::vector<Cmd>& cmds = res.cmds;
    size_t i = start;
    while (i < w.size()) {
        if (looks_like_header(w, i)) {
            res.header_positions.push_back(i);
            Cmd c;
            c.word = (uint32_t)i;
            c.type = "TXN_HEADER";
            c.n_words = 4;
            c.buffer_length = w[i+2];   // op count claimed
            c.buffer_offset = w[i+3];   // total bytes claimed
            c.row = (w[i] >> 24) & 0xFF;
            c.col = w[i+1] & 0xFF;
            cmds.push_back(std::move(c));
            i += 4;
            continue;
        }
        uint32_t h = w[i] & 0xFF;
        Cmd c;
        c.word = (uint32_t)i;
        c.type = op_name(h);
        if (h == OP_BLOCKWRITE && i + 12 <= w.size()) {
            // Two variants share op header 1: the shim-DMA BD write (12 words,
            // op_size word = 48) and the memtile BD-register write (6 words,
            // op_size word = 24). Disambiguate on the op_size field.
            if (w[i+1] == 24) {
                c.n_words = 6;
                c.type = "MT_BLOCKWRITE";
                c.mt_addr = w[i+4];
                c.mt_value = w[i+5];
                i += 6;
                cmds.push_back(std::move(c));
                continue;
            }
            c.n_words = 12;
            c.col = (w[i+2] >> 25) & 0x7F;
            c.row = (w[i+2] >> 20) & 0x1F;
            c.bd_id = (w[i+2] >> 5) & 0xF;
            c.buffer_length = w[i+4];
            c.buffer_offset = w[i+5];
            c.packet_enable = (w[i+6] >> 30) & 1;
            c.out_of_order_id = (w[i+6] >> 24) & 0x3F;
            c.packet_id = (w[i+6] >> 19) & 0x1F;
            c.packet_type = (w[i+6] >> 16) & 0x7;
            c.linear = (w[i+7] == 0);
            c.dim0_size = (w[i+7] >> 20) & 0x3FF;
            c.dim0_stride = ((w[i+7] & 0xFFFFF) + 1);
            c.dim1_size = (w[i+8] >> 20) & 0x3FF;
            c.dim1_stride = ((w[i+8] & 0xFFFFF) + 1);
            c.cache_flag = (w[i+9] >> 24) & 0xF;
            c.dim2_stride = ((w[i+9] & 0xFFFFF) + 1);
            c.iter_size = ((w[i+10] >> 20) & 0x3FF) + 1;
            c.iter_stride = ((w[i+10] & 0xFFFFF) + 1);
            c.next_bd_id = (w[i+11] >> 27) & 0xF;
            c.valid_bd = (w[i+11] >> 25) & 1;
            i += 12;
        } else if (h == OP_DDR_PATCH && i + 12 <= w.size()) {
            c.n_words = 12;
            c.col = (w[i+6] >> 25) & 0x7F;
            c.row = (w[i+6] >> 20) & 0x1F;
            c.bd_id = ((w[i+6] - 0x04) >> 5) & 0x1F;
            c.arg_idx = w[i+8];
            c.arg_offset = w[i+10];
            i += 12;
        } else if (h == OP_WRITE && i + 6 <= w.size()) {
            c.n_words = 6;
            c.col = (w[i+2] >> 25) & 0x7F;
            c.row = (w[i+2] >> 20) & 0x1F;
            c.reg_addr = w[i+2] & 0xFFFFF;
            c.value = w[i+4];
            c.push_queue = ((c.reg_addr & 0x1FE00) == 0x1d200);
            if (c.push_queue) {
                c.channel_id = (w[i+2] >> 3) & 1;
                c.repeat_count = (w[i+4] >> 16) & 0xFF;
                c.issue_token = (w[i+4] >> 31) & 1;
                c.bd_id = w[i+4] & 0xF;
            }
            i += 6;
        } else if (h == OP_MASKWRITE && i + 7 <= w.size()) {
            c.n_words = 7;
            c.col = (w[i+2] >> 25) & 0x7F;
            c.row = (w[i+2] >> 20) & 0x1F;
            c.addr = w[i+2] & 0xFFFFF;
            c.value = w[i+4];
            c.mask = w[i+5];
            c.channel_id = (w[i+2] >> 3) & 1;
            c.issue_token = ((w[i+2] & 0x10) != 0);
            i += 7;
        } else if (h == OP_TCT && i + 4 <= w.size()) {
            c.n_words = 4;
            c.wait_mm2s = (w[i+2] & 1) != 0;
            c.wait_row = (w[i+2] >> 8) & 0xFF;
            c.wait_col = (w[i+2] >> 16) & 0xFF;
            c.wait_channel = (w[i+3] >> 24) & 0xFF;
            i += 4;
        } else if (h == OP_PREEMPT && i + 1 <= w.size()) {
            c.n_words = 1;
            c.preemption_level = (w[i] >> 8) & 0x3;
            i += 1;
        } else {
            // Unknown op: try to resync at the next known op header.
            c.n_words = 1;
            c.type = "UNKNOWN";
            i += 1;
        }
        cmds.push_back(std::move(c));
    }
    return res;
}

static nlohmann::json cmd_to_json(const Cmd& c, const std::vector<uint32_t>& w) {
    nlohmann::json j = nlohmann::json::object();
    j["type"] = c.type;
    j["word"] = c.word;
    j["n_words"] = c.n_words;
    j["row"] = c.row;
    j["col"] = c.col;
    if (c.type == "BLOCKWRITE") {
        j["bd_id"] = c.bd_id;
        j["buffer_length"] = c.buffer_length;
        j["buffer_offset"] = c.buffer_offset;
        j["linear"] = c.linear;
        j["packet_enable"] = c.packet_enable;
        if (c.packet_enable) {
            j["packet_id"] = c.packet_id;
            j["packet_type"] = c.packet_type;
            j["out_of_order_id"] = c.out_of_order_id;
        }
        j["dim0_size"] = c.dim0_size;
        j["dim0_stride"] = c.dim0_stride;
        j["dim1_size"] = c.dim1_size;
        j["dim1_stride"] = c.dim1_stride;
        j["dim2_stride"] = c.dim2_stride;
        j["iter_size"] = c.iter_size;
        j["iter_stride"] = c.iter_stride;
        j["next_bd_id"] = c.next_bd_id;
        j["valid_bd"] = c.valid_bd;
        j["cache_flag"] = c.cache_flag;
        if (c.dir_known) j["direction"] = c.dir_mm2s ? "MM2S" : "S2MM";
    } else if (c.type == "TXN_HEADER") {
        j["rows"] = c.row;
        j["cols"] = c.col;
        j["claimed_op_count"] = c.buffer_length;
        j["claimed_bytes"] = c.buffer_offset;
    } else if (c.type == "MT_BLOCKWRITE") {
        char buf[16];
        snprintf(buf, sizeof(buf), "0x%08x", c.mt_addr);
        j["mt_addr"] = buf;
        j["mt_value"] = c.mt_value;
    } else if (c.type == "DDR_PATCH") {
        j["bd_id"] = c.bd_id;
        j["arg_idx"] = c.arg_idx;
        j["arg_offset"] = c.arg_offset;
    } else if (c.type == "WRITE") {
        j["reg_addr"] = c.reg_addr;
        j["value"] = c.value;
        if (c.push_queue) {
            j["push_queue"] = true;
            j["channel_id"] = c.channel_id;
            j["repeat_count"] = c.repeat_count;
            j["issue_token"] = c.issue_token;
            j["bd_id"] = c.bd_id;
        }
    } else if (c.type == "MASKWRITE") {
        j["addr"] = c.addr;
        j["value"] = c.value;
        j["mask"] = c.mask;
        j["issue_token"] = c.issue_token;
        j["channel_id"] = c.channel_id;
    } else if (c.type == "TCT") {
        j["wait_row"] = c.wait_row;
        j["wait_col"] = c.wait_col;
        j["wait_channel"] = c.wait_channel;
        j["direction"] = c.wait_mm2s ? "MM2S" : "S2MM";
    } else if (c.type == "PREEMPT") {
        j["preemption_level"] = c.preemption_level;
    }
    nlohmann::json raw = nlohmann::json::array();
    for (uint32_t k = 0; k < c.n_words; k++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "0x%08x", w[c.word + k]);
        raw.push_back(buf);
    }
    j["raw"] = raw;
    return j;
}

// Build the BD -> (arg_idx, arg_offset) patch table by pairing each
// DDR_PATCH with the most recent BLOCKWRITE on the same (col,row,bd_id).
struct Patch {
    uint32_t col, row, bd_id;
    uint32_t arg_idx, arg_offset;
    uint32_t buffer_length, buffer_offset;
    bool linear;
    uint32_t dim0_size, dim0_stride, dim1_size, dim1_stride, dim2_stride;
    uint32_t iter_size, iter_stride;
    uint32_t word;
    bool dir_known;
    bool dir_mm2s;
};
static std::vector<Patch> build_patches(const std::vector<Cmd>& cmds) {
    std::map<uint64_t, const Cmd*> bd_map;  // (col<<16)|(row<<8)|bd_id -> BLOCKWRITE
    std::vector<Patch> patches;
    for (auto& c : cmds) {
        if (c.type == "BLOCKWRITE") {
            uint64_t key = ((uint64_t)c.col << 16) | ((uint64_t)c.row << 8) | c.bd_id;
            bd_map[key] = &c;
        } else if (c.type == "DDR_PATCH") {
            uint64_t key = ((uint64_t)c.col << 16) | ((uint64_t)c.row << 8) | (c.bd_id & 0xF);
            auto it = bd_map.find(key);
            Patch p;
            p.col = c.col; p.row = c.row; p.bd_id = c.bd_id & 0xF;
            p.arg_idx = c.arg_idx; p.arg_offset = c.arg_offset;
            p.word = c.word;
            p.dir_known = false;
            p.dir_mm2s = false;
            if (it != bd_map.end()) {
                const Cmd* b = it->second;
                p.buffer_length = b->buffer_length;
                p.buffer_offset = b->buffer_offset;
                p.linear = b->linear;
                p.dim0_size = b->dim0_size; p.dim0_stride = b->dim0_stride;
                p.dim1_size = b->dim1_size; p.dim1_stride = b->dim1_stride;
                p.dim2_stride = b->dim2_stride;
                p.iter_size = b->iter_size; p.iter_stride = b->iter_stride;
                p.dir_known = b->dir_known;
                p.dir_mm2s = b->dir_mm2s;
            } else {
                p.buffer_length = 0; p.buffer_offset = 0;
                p.linear = false;
                p.dim0_size = p.dim0_stride = p.dim1_size = p.dim1_stride = p.dim2_stride = 0;
                p.iter_size = p.iter_stride = 0;
            }
            patches.push_back(p);
        }
    }
    return patches;
}

// Derive each BLOCKWRITE BD's DMA direction from the queue WRITE that pushes
// it (reg 0x1d204/0x1d20c = S2MM ch0/1, 0x1d214/0x1d21c = MM2S ch0/1; the
// write's value carries the bd_id). S2MM = tile -> BO (destination write),
// MM2S = BO -> tile (source read).
static void apply_directions(std::vector<Cmd>& cmds) {
    std::map<uint64_t, bool> dir_map;  // (col<<16)|(row<<8)|bd_id -> mm2s
    for (auto& c : cmds) {
        if (c.type == "WRITE" && c.push_queue) {
            uint64_t key = ((uint64_t)c.col << 16) | ((uint64_t)c.row << 8) | (c.value & 0xF);
            dir_map[key] = (c.reg_addr & 0x10) != 0;
        }
    }
    for (auto& c : cmds) {
        if (c.type == "BLOCKWRITE") {
            uint64_t key = ((uint64_t)c.col << 16) | ((uint64_t)c.row << 8) | c.bd_id;
            auto it = dir_map.find(key);
            if (it != dir_map.end()) {
                c.dir_known = true;
                c.dir_mm2s = it->second;
            }
        }
    }
}

// Decode one TXN (raw words) into a JSON doc with commands + patch table.
static nlohmann::json decode_txn(const std::vector<uint32_t>& w,
                                 const std::string& name,
                                 const std::string& generator) {
    nlohmann::json doc = nlohmann::json::object();
    doc["name"] = name;
    doc["generator"] = generator;
    doc["n_words"] = w.size();
    nlohmann::json hdr = nlohmann::json::object();
    if (w.size() >= 4) {
        hdr["dev_major"] = (w[0] >> 0) & 0xFF;
        hdr["dev_minor"] = (w[0] >> 8) & 0xFF;
        hdr["dev_gen"] = (w[0] >> 16) & 0xFF;
        hdr["rows"] = (w[0] >> 24) & 0xFF;
        hdr["cols"] = (w[1] >> 0) & 0xFF;
        hdr["mem_tile_rows"] = (w[1] >> 8) & 0xFF;
        hdr["cmd_count"] = w[2];
        hdr["lines_bytes"] = w[3];
    }
    doc["header"] = hdr;
    // Parse from word 0: the layer/lm_head/mha/dequant generators emit
    // headerless op runs (sometimes with the real TXN header embedded
    // mid-stream); only .bin files produced by cmds2seq carry a leading
    // header, which the parser detects and records as TXN_HEADER.
    DecodeResult dr = decode_words(w, 0);
    std::vector<Cmd>& cmds = dr.cmds;
    if (!dr.header_positions.empty()) {
        nlohmann::json hp = nlohmann::json::array();
        for (auto p : dr.header_positions) hp.push_back(p);
        doc["embedded_headers"] = hp;
    }
    apply_directions(cmds);
    nlohmann::json carr = nlohmann::json::array();
    for (auto& c : cmds) carr.push_back(cmd_to_json(c, w));
    doc["commands"] = carr;
    std::vector<Patch> patches = build_patches(cmds);
    nlohmann::json parr = nlohmann::json::array();
    for (auto& p : patches) {
        nlohmann::json j = nlohmann::json::object();
        j["word"] = p.word;
        j["bd_id"] = p.bd_id;
        j["row"] = p.row;
        j["col"] = p.col;
        j["arg_idx"] = p.arg_idx;
        j["arg_offset"] = p.arg_offset;
        j["buffer_length"] = p.buffer_length;
        j["buffer_offset"] = p.buffer_offset;
        j["linear"] = p.linear;
        j["dim0_size"] = p.dim0_size;
        j["dim0_stride"] = p.dim0_stride;
        j["dim1_size"] = p.dim1_size;
        j["dim1_stride"] = p.dim1_stride;
        j["dim2_stride"] = p.dim2_stride;
        j["iter_size"] = p.iter_size;
        j["iter_stride"] = p.iter_stride;
        if (p.dir_known) j["direction"] = p.dir_mm2s ? "MM2S" : "S2MM";
        parr.push_back(j);
    }
    doc["patches"] = parr;
    // op histogram
    std::map<std::string, int> hist;
    for (auto& c : cmds) hist[c.type]++;
    doc["op_histogram"] = hist;
    return doc;
}

static bool read_bin(const std::string& path, std::vector<uint32_t>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % 4 != 0) { fclose(f); return false; }
    out.resize(sz / 4);
    size_t br = fread(out.data(), 4, out.size(), f);
    fclose(f);
    return br == out.size();
}

static void print_summary(const std::string& label, const nlohmann::json& doc) {
    printf("\n===== %s =====\n", label.c_str());
    printf("generator : %s\n", doc["generator"].get<std::string>().c_str());
    printf("words     : %u\n", doc["n_words"].get<uint32_t>());
    printf("cmd count : %s\n", doc["header"]["cmd_count"].dump().c_str());
    std::map<std::string, int> hist = doc["op_histogram"].get<std::map<std::string, int>>();
    for (auto& [k, v] : hist) printf("  %-12s %d\n", k.c_str(), v);
    if (doc.contains("patches") && doc["patches"].is_array()) {
        printf("patches (BD -> arg_idx@offset, len, dims):\n");
        for (auto& p : doc["patches"]) {
            printf("  arg%-2u @%-8u BD(%u,%u,%u) len=%-7u off=%-6u linear=%d D0=%u/%u D1=%u/%u D2s=%u iter=%u/%u %s\n",
                   p["arg_idx"].get<uint32_t>(), p["arg_offset"].get<uint32_t>(),
                   p["row"].get<uint32_t>(), p["col"].get<uint32_t>(), p["bd_id"].get<uint32_t>(),
                   p["buffer_length"].get<uint32_t>(), p["buffer_offset"].get<uint32_t>(),
                   p["linear"].get<bool>() ? 1 : 0,
                   p["dim0_size"].get<uint32_t>(), p["dim0_stride"].get<uint32_t>(),
                   p["dim1_size"].get<uint32_t>(), p["dim1_stride"].get<uint32_t>(),
                   p["dim2_stride"].get<uint32_t>(),
                   p["iter_size"].get<uint32_t>(), p["iter_stride"].get<uint32_t>(),
                   (p.contains("direction") ? p["direction"].get<std::string>().c_str() : "?"));
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: decode_txn <config.json> <outdir> [n_tokens=128] [insts.bin ...]\n"
                "  Generates + decodes the FastFlowLM runtime's layer TXN (L=1 and L=n_tokens),\n"
                "  lm_head TXN, mha TXN, and per-projection dequant TXNs (q80+q4_1, modes 0..7)\n"
                "  into <outdir>/*.json; decodes any extra .bin insts the same way.\n");
        return 1;
    }
    const char* cfg_path = argv[1];
    std::string outdir = argv[2];
    uint32_t n_tokens = (argc > 3) ? (uint32_t)strtoul(argv[3], nullptr, 10) : 128;

    LM_Config config;
    std::ifstream f(cfg_path);
    if (!f.is_open()) { fprintf(stderr, "cannot open %s\n", cfg_path); return 1; }
    f >> config._json_config;

    uint32_t hidden = config.get("hidden_size", 1024u);
    uint32_t interm = config.get("intermediate_size", 3072u);
    uint32_t vocab  = config.get("vocab_size", 151936u);
    printf("config: hidden=%u interm=%u vocab=%u\n", hidden, interm, vocab);

    qwen3_npu_sequence qseq(config, 4096);

    // ---- layer TXN for decode (L=1) and L=n_tokens ----
    for (uint32_t L : {1u, n_tokens}) {
        npu_sequence seq(device_npu2);
        qseq.gen_layer_seq(&seq, L);
        seq.cmds2seq();
        auto [ptr, nw] = seq.dump();
        std::vector<uint32_t> w(ptr, ptr + nw);
        std::string name = "layer_L" + std::to_string(L);
        nlohmann::json doc = decode_txn(w, name, "qwen3_npu_sequence::gen_layer_seq(" + std::to_string(L) + ")");
        std::string path = outdir + "/" + name + ".json";
        std::ofstream of(path); of << doc.dump(2); of.close();
        print_summary(name, doc);
    }

    // ---- lm_head TXN ----
    {
        // gen_lm_head_seq is exported from libqwen3_npu.so but NOT declared in
        // the shipped qwen3_npu_sequence.hpp — resolve the ABI entry directly:
        //   _ZN18qwen3_npu_sequence15gen_lm_head_seqEP12npu_sequence
        typedef void (*gen_lm_head_t)(qwen3_npu_sequence*, npu_sequence*);
        static gen_lm_head_t gen_lm_head = nullptr;
        if (!gen_lm_head) {
            gen_lm_head = (gen_lm_head_t)dlsym(RTLD_DEFAULT,
                "_ZN18qwen3_npu_sequence15gen_lm_head_seqEP12npu_sequence");
            if (!gen_lm_head) {
                fprintf(stderr, "dlsym gen_lm_head_seq failed: %s\n", dlerror());
                return 1;
            }
        }
        npu_sequence seq(device_npu2);
        gen_lm_head(&qseq, &seq);
        seq.cmds2seq();
        auto [ptr, nw] = seq.dump();
        std::vector<uint32_t> w(ptr, ptr + nw);
        nlohmann::json doc = decode_txn(w, "lm_head", "qwen3_npu_sequence::gen_lm_head_seq()");
        std::ofstream of(outdir + "/lm_head.json"); of << doc.dump(2); of.close();
        print_summary("lm_head", doc);
    }

    // ---- mha engine TXN (decode window L=1..n_tokens) ----
    {
        npu_sequence seq(device_npu2);
        qseq.gen_mha_engine_seq(&seq, 0, n_tokens);
        seq.cmds2seq();
        auto [ptr, nw] = seq.dump();
        std::vector<uint32_t> w(ptr, ptr + nw);
        std::string name = "mha_L0_" + std::to_string(n_tokens);
        nlohmann::json doc = decode_txn(w, name, "qwen3_npu_sequence::gen_mha_engine_seq(0," + std::to_string(n_tokens) + ")");
        std::ofstream of(outdir + "/" + name + ".json"); of << doc.dump(2); of.close();
        print_summary(name, doc);
    }

    // ---- dequant TXNs: q80-packed-in-q4nx + q4_1, modes 0..7, q_proj block ----
    Dequant deq(config);
    struct { const char* fn; uint32_t din, dout, woff; const char* label; } projs[] = {
        {"q_proj", hidden, 256, 0, "q_proj_b0"},
        {"gate_proj", hidden, 256, 0, "gate_proj_b0"},
        {"down_proj", interm, 256, 0, "down_proj_b0"},
    };
    for (auto& pr : projs) {
        for (int mode = 0; mode < 8; mode++) {
            npu_sequence seq(device_npu2);
            deq.generate_dequant_q80_packed_in_q4nx_seq(&seq, pr.din, pr.dout, pr.woff, mode);
            seq.cmds2seq();
            auto [ptr, nw] = seq.dump();
            std::vector<uint32_t> w(ptr, ptr + nw);
            std::string name = std::string("dequant_q80_") + pr.label + "_m" + std::to_string(mode);
            nlohmann::json doc = decode_txn(w, name,
                "Dequant::generate_dequant_q80_packed_in_q4nx_seq(din=" + std::to_string(pr.din) +
                ",dout=" + std::to_string(pr.dout) + ",woff=" + std::to_string(pr.woff) + ",mode=" + std::to_string(mode) + ")");
            std::ofstream of(outdir + "/" + name + ".json"); of << doc.dump(2); of.close();
            print_summary(name, doc);
        }
        for (int mode = 0; mode < 8; mode++) {
            npu_sequence seq(device_npu2);
            deq.generate_dequant_q4_1_seq(&seq, pr.din, pr.dout, pr.woff, mode);
            seq.cmds2seq();
            auto [ptr, nw] = seq.dump();
            std::vector<uint32_t> w(ptr, ptr + nw);
            std::string name = std::string("dequant_q41_") + pr.label + "_m" + std::to_string(mode);
            nlohmann::json doc = decode_txn(w, name,
                "Dequant::generate_dequant_q4_1_seq(din=" + std::to_string(pr.din) +
                ",dout=" + std::to_string(pr.dout) + ",woff=" + std::to_string(pr.woff) + ",mode=" + std::to_string(mode) + ")");
            std::ofstream of(outdir + "/" + name + ".json"); of << doc.dump(2); of.close();
            print_summary(name, doc);
        }
    }

    // ---- decode extra insts .bin files (hand-rolled per-shape streams) ----
    for (int a = 4; a < argc; a++) {
        std::vector<uint32_t> w;
        if (!read_bin(argv[a], w)) {
            fprintf(stderr, "cannot read insts file %s\n", argv[a]);
            continue;
        }
        std::string base = argv[a];
        size_t slash = base.rfind('/');
        if (slash != std::string::npos) base = base.substr(slash + 1);
        size_t dot = base.rfind('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        nlohmann::json doc = decode_txn(w, base, "file:" + std::string(argv[a]));
        std::ofstream of(outdir + "/" + base + ".json"); of << doc.dump(2); of.close();
        print_summary(base, doc);
    }

    printf("\nwrote JSON decodes to %s/\n", outdir.c_str());
    return 0;
}
