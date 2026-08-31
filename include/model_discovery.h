#pragma once
#ifndef MODEL_DISCOVERY_H
#define MODEL_DISCOVERY_H

#include "common.h"
#include <string>
#include <vector>
#include <cstdint>

// Discover models available in a directory.
std::vector<ModelConfig> discover_models(const std::string& dir);

// Read metadata for a single model FILE (any supported format), dispatching
// on the extension. Used for direct `-m <path>` selection (issue #1958) so an
// absolute/relative model path is honored instead of being treated as a
// registry name. Returns false (and leaves cfg untouched) for unknown formats.
bool read_model_file_metadata(const std::string& path, ModelConfig& cfg);

// Read GGUF header (model config + tensor metadata).
bool read_gguf_header(const std::string& path, ModelConfig& cfg);

// Read a GGUF tensor's data into a float vector (dequantized).
// Returns true if the tensor was found and read.
// Tensor names are llama.cpp-style: "token_embd.weight", "blk.0.attn_q.weight", etc.
int read_gguf_vocab(const std::string& path);
bool read_gguf_tensor(const std::string& gguf_path, const std::string& tensor_name,
                      std::vector<float>& output, size_t* out_n = nullptr);

#endif
