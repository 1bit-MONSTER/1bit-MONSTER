// hrx_decode_engine.h — DecodeEngine adapter over hrx::Inprocess (the real
// engine in-process HRX path with load_session_mem, 0a54070c). Gives the
// PhaseRouter a real decode engine: import the shared-memory state fd and
// continue decode on the HRX custom-kernel device.
#pragma once

#include "router/phase_router.h"
#include "hrx_inprocess.h"

#include <cstdio>
#include <memory>
#include <string>

namespace engine {

class HrxDecodeEngine : public DecodeEngine {
public:
    HrxDecodeEngine(std::shared_ptr<hrx::Inprocess> inprocess, const std::string& model_path,
                    int n_gpu_layers, uint32_t ctx_size,
                    const std::string& device_pin = "HRX0")
        : hrx_(std::move(inprocess)), model_(model_path),
          ngl_(n_gpu_layers), ctx_(ctx_size), devpin_(device_pin) {}

    /// Initialize the in-process engine (device per policy: Vulkan0 for the
    /// stock Q4_K class, HRX0 for moat Q4NX/zaya) + load the model. Must be
    /// called before decode(). Returns true on success.
    bool init() {
        if (!hrx_) return false;
        hrx_->set_device_pin(devpin_);
        if (!hrx_->init()) { fprintf(stderr, "[hrxdec] Inprocess::init failed\n"); return false; }
        if (!hrx_->load_model(model_, ngl_, ctx_)) {
            fprintf(stderr, "[hrxdec] load_model failed\n");
            return false;
        }
        return true;
    }

    const char* id() const override { return "hrx"; }

    std::string decode(int state_fd, long n_tokens, int max_tokens) override {
        if (!hrx_ || state_fd < 0) return "";
        // Zero-copy state import: the fd is a memfd/dma-buf carrying the
        // session-v9 state; Inprocess::load_session_mem mmaps it and calls
        // llama_state_set_data (no file, no host copy) - 0a54070c. It also
        // exposes the resume token (last stored input), which the state is
        // positioned after - decoding it again at pos reproduces the exact
        // continuation (rt_session semantics, fork-verified 203926057).
        long imported = hrx_->load_session_mem(state_fd);
        if (imported < 0) {
            fprintf(stderr, "[hrxdec] load_session_mem failed (fd %d)\n", state_fd);
            return "";
        }
        int resume = hrx_->resume_token();
        fprintf(stderr, "[hrxdec] imported %ld tokens from shared state (resume=%d)\n",
                imported, resume);
        // Greedy continuation: first decode the resume token at the imported
        // position (produces the logits for the true next token), then argmax
        // loop via the engine generate().
        std::string out;
        int tok = resume >= 0 ? resume : 0;
        (void)n_tokens;
        for (int i = 0; i < max_tokens; i++) {
            int next = hrx_->generate(tok);
            if (next < 0) break;
            char b[32];
            int l = snprintf(b, sizeof b, "%d ", next);
            out.append(b, (size_t)l);
            tok = next;
        }
        return out;
    }

private:
    std::shared_ptr<hrx::Inprocess> hrx_;
    std::string model_;
    int ngl_ = -1;
    uint32_t ctx_ = 0;
    std::string devpin_ = "HRX0";
};

}  // namespace engine
