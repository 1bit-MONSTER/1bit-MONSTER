// dynamic_router.cpp — Per-token dynamic router for GPU ↔ NPU inference.

#include "dynamic_router.h"
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <cmath>

DynamicRouter::~DynamicRouter() {
    // Router does NOT own backends — caller manages lifetime
}

void DynamicRouter::add_backend(const std::string& id, std::shared_ptr<Backend> backend, Strategy strategy) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& e : entries_)
        if (e.id == id) { e.backend = backend; e.stats = BackendStats{}; return; }
    entries_.push_back({id, backend, std::make_shared<std::mutex>(), BackendStats{id}, {}});
    strategy_ = strategy;
    printf("[router] added backend '%s' (%s)\n", id.c_str(), strategy_name(strategy));
}

void DynamicRouter::remove_backend(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx_);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
        [&](auto& e) { return e.id == id; }), entries_.end());
}

void DynamicRouter::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    entries_.clear();
}

bool DynamicRouter::reset_all() {
    std::lock_guard<std::mutex> lock(mtx_);
    bool ok = true;
    try {
        for (auto& e : entries_) {
            if (e.backend && e.stats.alive) {
                // #1345: hold the entry's compute lock so an in-flight
                // generate()/forward() on this backend isn't racing reset().
                std::lock_guard<std::mutex> cl(*e.compute_mtx);
                try { ok = e.backend->reset() && ok; }
                catch (...) { e.stats.alive = false; fprintf(stderr, "[router] reset failed: %s\n", e.id.c_str()); }
            }
        }
    } catch (const std::exception& ex) {
        fprintf(stderr, "[router] reset_all exception: %s\n", ex.what());
        return false;
    } catch (...) {
        fprintf(stderr, "[router] reset_all unknown exception\n");
        return false;
    }
    return ok;
}

void DynamicRouter::set_strategy(Strategy s) {
    std::lock_guard<std::mutex> lock(mtx_);
    strategy_ = s;
    printf("[router] strategy → %s\n", strategy_name(s));
}

DynamicRouter::Strategy DynamicRouter::strategy() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return strategy_;
}

// ── Backend selection ──
DynamicRouter::BackendEntry* DynamicRouter::pick_backend() {
    if (entries_.empty()) return nullptr;
    if (entries_.size() == 1) return &entries_[0];

    switch (strategy_) {
        case Strategy::GPU_ONLY: {
            for (auto& e : entries_)
                if (e.id == "hip_gpu" || e.id == "zinc_gpu") return &e;
            return &entries_[0];
        }
        case Strategy::NPU_ONLY: {
            for (auto& e : entries_)
                if (e.id == "npu_flm" || e.id == "npu_xrt") return &e;
            return &entries_[0];
        }
        case Strategy::GPU_BACKFILL: {
            // GPU gets ~80% of tokens, NPU gets ~20%
            // Simple deterministic round-based: 4 GPU, 1 NPU, repeat
            int cycle = round_robin_counter_++ % 5;
            if (cycle < 4) {
                for (auto& e : entries_)
                    if (e.id.find("gpu") != std::string::npos || e.id.find("hip") != std::string::npos)
                        return &e;
            } else {
                for (auto& e : entries_)
                    if (e.id.find("npu") != std::string::npos)
                        return &e;
            }
            return &entries_[0];
        }
        case Strategy::NPU_BACKFILL: {
            // NPU gets ~80% of tokens, GPU gets ~20%
            int cycle = round_robin_counter_++ % 5;
            if (cycle < 4) {
                for (auto& e : entries_)
                    if (e.id.find("npu") != std::string::npos) return &e;
            } else {
                for (auto& e : entries_)
                    if (e.id.find("gpu") != std::string::npos || e.id.find("hip") != std::string::npos)
                        return &e;
            }
            return &entries_[0];
        }
        case Strategy::FASTEST:
        default: {
            // Pick the backend with lowest avg latency in the sliding window
            BackendEntry* best = nullptr;
            double best_avg = 1e30;
            for (auto& e : entries_) {
                if (!e.stats.alive || !e.backend) continue;
                double avg = window_avg(&e);
                if (avg < best_avg) { best_avg = avg; best = &e; }
            }
            if (best) return best;
            // Fallback: first alive
            for (auto& e : entries_)
                if (e.stats.alive && e.backend) return &e;
            return entries_.empty() ? nullptr : &entries_[0];
        }
    }
}

DynamicRouter::BackendEntry* DynamicRouter::pick_backend_excluding(const std::string& exclude_id) {
    if (entries_.empty()) return nullptr;
    if (entries_.size() == 1)
        return (entries_[0].id == exclude_id) ? nullptr : &entries_[0];

    // Same strategy, but never return the excluded (just-failed) backend.
    switch (strategy_) {
        case Strategy::GPU_ONLY:
            for (auto& e : entries_)
                if ((e.id == "hip_gpu" || e.id == "zinc_gpu") && e.id != exclude_id) return &e;
            break;
        case Strategy::NPU_ONLY:
            for (auto& e : entries_)
                if ((e.id == "npu_flm" || e.id == "npu_xrt") && e.id != exclude_id) return &e;
            break;
        case Strategy::GPU_BACKFILL: {
            int cycle = round_robin_counter_++ % 5;
            if (cycle < 4) {
                for (auto& e : entries_)
                    if ((e.id.find("gpu") != std::string::npos || e.id.find("hip") != std::string::npos) && e.id != exclude_id) return &e;
            } else {
                for (auto& e : entries_)
                    if (e.id.find("npu") != std::string::npos && e.id != exclude_id) return &e;
            }
            break;
        }
        case Strategy::NPU_BACKFILL: {
            int cycle = round_robin_counter_++ % 5;
            if (cycle < 4) {
                for (auto& e : entries_)
                    if (e.id.find("npu") != std::string::npos && e.id != exclude_id) return &e;
            } else {
                for (auto& e : entries_)
                    if ((e.id.find("gpu") != std::string::npos || e.id.find("hip") != std::string::npos) && e.id != exclude_id) return &e;
            }
            break;
        }
        case Strategy::FASTEST:
        default: {
            BackendEntry* best = nullptr;
            double best_avg = 1e30;
            for (auto& e : entries_) {
                if (!e.stats.alive || !e.backend || e.id == exclude_id) continue;
                double avg = window_avg(&e);
                if (avg < best_avg) { best_avg = avg; best = &e; }
            }
            if (best) return best;
            for (auto& e : entries_)
                if (e.stats.alive && e.backend && e.id != exclude_id) return &e;
            break;
        }
    }
    // If only the excluded backend remains, return it last-chance (caller
    // aborts if it still fails) rather than nullptr-ing spurious failures.
    return entries_.empty() ? nullptr : &entries_[0];
}

// ── Inference ──
int DynamicRouter::generate(int token_id) {
    return generate_with_failover(token_id, 0);
}

int DynamicRouter::generate_with_failover(int token_id, int depth) {
    std::shared_ptr<Backend> backend;
    std::shared_ptr<std::mutex> compute_mtx;
    std::string id;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        BackendEntry* entry = (depth == 0) ? pick_backend() : pick_backend_excluding(last_failed_id_);
        if (!entry || !entry->backend) return -1;
        backend = entry->backend;            // snapshot to keep alive outside lock
        compute_mtx = entry->compute_mtx;    // #1345: serialize per backend instance
        id = entry->id;                      // #1348: re-resolve by id after unlock
    }

    std::lock_guard<std::mutex> compute_lock(*compute_mtx);
    auto t0 = std::chrono::steady_clock::now();
    int result = -1;
    try {
        result = backend->generate(token_id);  // fixes #1315: no router lock held during inference
    } catch (const std::exception& e) {
        fprintf(stderr, "[router] backend %s threw in generate() (%s) — failing over\n", id.c_str(), e.what());
        result = -1;
    } catch (...) {
        fprintf(stderr, "[router] backend %s threw an unknown exception in generate() — failing over\n", id.c_str());
        result = -1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    {
        std::lock_guard<std::mutex> lock(mtx_);
        // #1348: entry may have been removed or the vector reallocated while
        // the lock was released — never dereference the stale pointer, look
        // the entry up by id (tolerating removal).
        for (auto& e : entries_) {
            if (e.id == id) { record_latency(&e, ms, result >= 0); break; }
        }
    }

    // Decode-time failover: a backend whose generate() fails (e.g. HRX
    // GET_ROWS fail-closed, or HIP/DynamicRouter incompatibility) must not
    // stall the request. Retry once with a different backend, then give up.
    if (result < 0 && depth == 0 && entries_.size() > 1) {
        last_failed_id_ = id;
        fprintf(stderr, "[router] backend %s failed at decode — retrying on a different backend\n", id.c_str());
        int retry = generate_with_failover(token_id, 1);
        last_failed_id_.clear();
        return retry;
    }
    return result;
}

bool DynamicRouter::forward(int token_id, float* hidden_out) {
    std::shared_ptr<Backend> backend;
    std::shared_ptr<std::mutex> compute_mtx;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto* entry = pick_backend();
        if (!entry || !entry->backend) return false;
        backend = entry->backend;
        compute_mtx = entry->compute_mtx;  // #1345
    }
    std::lock_guard<std::mutex> compute_lock(*compute_mtx);
    return backend->forward(token_id, hidden_out);
}

bool DynamicRouter::lm_head(const float* hidden, float* logits, int* argmax) {
    std::shared_ptr<Backend> backend;
    std::shared_ptr<std::mutex> compute_mtx;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto* entry = pick_backend();
        if (!entry || !entry->backend) return false;
        backend = entry->backend;
        compute_mtx = entry->compute_mtx;  // #1345
    }
    std::lock_guard<std::mutex> compute_lock(*compute_mtx);
    return backend->lm_head(hidden, logits, argmax);
}

// ── Stats ──
std::vector<DynamicRouter::BackendStats> DynamicRouter::stats() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<BackendStats> result;
    for (auto& e : entries_) result.push_back(e.stats);
    return result;
}

void DynamicRouter::report() const {
    auto s = stats();
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║        Dynamic Router — Stats            ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("  Strategy: %s\n\n", strategy_name(strategy_));
    for (auto& st : s) {
        printf("  %-12s  %5.0f ms/tok  %5.1f tok/s  %5lld total  %s\n",
               st.id.c_str(), st.avg_ms_per_tok, st.tok_s,
               st.total_tokens, st.alive ? "✅" : "❌");
    }
    printf("\n");
}

void DynamicRouter::record_latency(BackendEntry* entry, double ms, bool success) {
    entry->stats.total_tokens++;
    if (!success) { entry->stats.failed_tokens++; return; }

    entry->recent_latencies.push_back(ms);
    if ((int)entry->recent_latencies.size() > WINDOW_SIZE)
        entry->recent_latencies.erase(entry->recent_latencies.begin());

    double avg = window_avg(entry);
    entry->stats.avg_ms_per_tok = avg;
    entry->stats.tok_s = avg > 0 ? 1000.0 / avg : 0;
}

double DynamicRouter::window_avg(const BackendEntry* entry) const {
    if (entry->recent_latencies.empty()) return 1e30;
    double sum = std::accumulate(entry->recent_latencies.begin(),
                                  entry->recent_latencies.end(), 0.0);
    return sum / entry->recent_latencies.size();
}

// ── Strategy name helpers ──
const char* strategy_name(DynamicRouter::Strategy s) {
    switch (s) {
        case DynamicRouter::Strategy::FASTEST:      return "FASTEST";
        case DynamicRouter::Strategy::GPU_BACKFILL: return "GPU_BACKFILL (80/20)";
        case DynamicRouter::Strategy::NPU_BACKFILL: return "NPU_BACKFILL (80/20)";
        case DynamicRouter::Strategy::GPU_ONLY:     return "GPU_ONLY";
        case DynamicRouter::Strategy::NPU_ONLY:     return "NPU_ONLY";
        default: return "UNKNOWN";
    }
}

DynamicRouter::Strategy strategy_from_string(const std::string& s) {
    if (s == "fastest") return DynamicRouter::Strategy::FASTEST;
    if (s == "gpu_backfill" || s == "gpu") return DynamicRouter::Strategy::GPU_BACKFILL;
    if (s == "npu_backfill" || s == "npu") return DynamicRouter::Strategy::NPU_BACKFILL;
    if (s == "gpu_only") return DynamicRouter::Strategy::GPU_ONLY;
    if (s == "npu_only") return DynamicRouter::Strategy::NPU_ONLY;
    return DynamicRouter::Strategy::FASTEST;
}
