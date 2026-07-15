#ifndef MODEL_DISCOVERY_H
#define MODEL_DISCOVERY_H

#include "common.h"
#include <string>
#include <vector>

// Discover models available in a directory. Scans for .gguf, .h1b, .safetensors files
// and reads their headers to populate ModelConfig. Returns discovered models.
std::vector<ModelConfig> discover_models(const std::string& dir);

// Quick check: read a GGUF file header and populate ModelConfig without loading weights.
// Returns true if the file was read successfully.
bool read_gguf_header(const std::string& path, ModelConfig& cfg);

#endif
