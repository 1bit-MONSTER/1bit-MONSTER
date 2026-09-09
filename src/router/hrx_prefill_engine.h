// hrx_prefill_engine.h — HrxPrefillEngine: a LIVE PrefillEngine adapter over
// the real Inprocess seam (goal mtsy05dx single-api-router). Replaces the
// file->memfd standin: loads the model on an Inprocess context, tokenizes the
// prompt with the bundle vocab, decodes it on-device (HRX0), and exports the
// resulting llama_state into a fresh memfd via Inprocess::export_session_mem.
// The decode engine then imports that fd zero-copy (load_session_mem) and
// continues. Full pipeline: real prefill context -> memfd -> decode context,
// no files, no host-side copies of the handoff buffer.
#pragma once

#include "hrx_inprocess.h"
#include "phase_router.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

namespace engine {

class HrxPrefillEngine final : public PrefillEngine {
public:
    HrxPrefillEngine(std::shared_ptr<hrx::Inprocess> inprocess,
                     std::string model_path, int n_gpu_layers, uint32_t ctx_size,
                     const std::string& device_pin = "HRX0")
        : hrx_(std::move(inprocess)), model_path_(std::move(model_path)),
          n_gpu_layers_(n_gpu_layers), ctx_size_(ctx_size), devpin_(device_pin) {}

    bool init() {
        if (!hrx_) return false;
        if (!hrx_->has_model()) {
            hrx_->set_device_pin(devpin_);
            if (!hrx_->init()) { fprintf(stderr, "[hrxpre] init FAILED\n"); return false; }
            if (!hrx_->load_model(model_path_, n_gpu_layers_, ctx_size_)) {
                fprintf(stderr, "[hrxpre] load_model FAILED\n"); return false;
            }
        }
        fprintf(stderr, "[hrxpre] model ready (vocab=%d n_embd=%d)\n",
                hrx_->vocab_size(), hrx_->n_embd());
        return true;
    }

    PrefillResult prefill(const std::string& prompt) override {
        PrefillResult r;
        if (!hrx_ || !hrx_->has_model()) return r;
        // Tokenize with the bundle vocab (real llama_tokenize, add_special).
        std::vector<int32_t> toks(4096);
        int np = hrx_->tokenize(prompt, toks.data(), (int32_t)toks.size());
        if (np < 0) {
            fprintf(stderr, "[hrxpre] tokenize failed (%d)\n", np);
            return r;
        }
        toks.resize((size_t)np);
        fprintf(stderr, "[hrxpre] prompt tokens: %d\n", np);
        // Live prefill: decode each prompt token at its position on HRX0.
        // Inprocess::generate feeds one token per llama_decode; the context
        // tracks pos/resume_token/n_session for the export.
        hrx_->reset();
        for (int i = 0; i < np; i++) {
            int nx = hrx_->generate(toks[(size_t)i]);
            if (nx < 0) { fprintf(stderr, "[hrxpre] prefill decode failed at %d\n", i); return r; }
        }
        // Export the live state into a fresh memfd (zero-copy handoff).
        int fd = (int)syscall(319, "hrxpre-state", 0);
        if (fd < 0) { perror("memfd"); return r; }
        if (hrx_->export_session_mem(fd) < 0) {
            fprintf(stderr, "[hrxpre] export FAILED\n");
            close(fd);
            return r;
        }
        lseek(fd, 0, SEEK_SET);
        r.state_fd = fd;
        r.n_tokens = np;
        r.ok = true;
        fprintf(stderr, "[hrxpre] state exported: fd %d, %ld tokens\n", fd, np);
        return r;
    }

    const char* id() const override { return "hrx"; }

private:
    std::shared_ptr<hrx::Inprocess> hrx_;
    std::string model_path_;
    int n_gpu_layers_;
    uint32_t ctx_size_;
    std::string devpin_ = "HRX0";
};

}  // namespace engine
