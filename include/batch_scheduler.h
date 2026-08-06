#pragma once
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <future>
#include <atomic>
#include <chrono>
#include <cstdio>

struct BatchRequest {
    std::vector<int> prompt_tokens;
    int max_tokens;
    float temperature;
    int top_k;
    int eos_id;
    std::promise<std::vector<int>> result;
};

struct BatchSlot {
    enum State { IDLE, PREFILLING, DECODING, DONE };
    State state = IDLE;
    std::vector<int> prompt_tokens;
    std::vector<int> output_tokens;
    int max_tokens = 0;
    float temperature = 0.0f;
    int top_k = 0;
    int eos_id = 0;
    int prefill_pos = 0;
    int last_token = 0;
    std::promise<std::vector<int>> result;
};

class BatchScheduler {
public:
    using GenerateBatchFn = std::function<std::vector<int>(
        const std::vector<std::pair<int,int>>&)>;
    using GenerateFn = std::function<int(int)>;
    using ResetSlotFn = std::function<bool(int)>;

    BatchScheduler(int max_slots, GenerateBatchFn gen_batch,
                   GenerateFn gen_single, ResetSlotFn reset_slot)
        : max_slots_(max_slots), gen_batch_(std::move(gen_batch)),
          gen_single_(std::move(gen_single)),
          reset_slot_(std::move(reset_slot)),
          slots_(max_slots), running_(true) {
        decode_thread_ = std::thread(&BatchScheduler::run, this);
    }

    ~BatchScheduler() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            running_ = false;
        }
        cv_.notify_all();
        if (decode_thread_.joinable()) decode_thread_.join();
    }

    std::future<std::vector<int>> submit(std::vector<int> prompt,
                                          int max_tokens, float temperature,
                                          int top_k, int eos_id) {
        BatchRequest req;
        req.prompt_tokens = std::move(prompt);
        req.max_tokens = max_tokens;
        req.temperature = temperature;
        req.top_k = top_k;
        req.eos_id = eos_id;
        auto fut = req.result.get_future();
        {
            std::lock_guard<std::mutex> lk(mu_);
            pending_.push(std::move(req));
        }
        cv_.notify_one();
        return fut;
    }

    int active_slots() const { return active_count_.load(); }

private:
    int max_slots_;
    GenerateBatchFn gen_batch_;
    GenerateFn gen_single_;
    ResetSlotFn reset_slot_;
    std::vector<BatchSlot> slots_;
    std::queue<BatchRequest> pending_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::thread decode_thread_;
    std::atomic<bool> running_{true};
    std::atomic<int> active_count_{0};

    void admit_pending() {
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < max_slots_ && !pending_.empty(); i++) {
            if (slots_[i].state != BatchSlot::IDLE) continue;
            auto& req = pending_.front();
            slots_[i].state = BatchSlot::PREFILLING;
            slots_[i].prompt_tokens = std::move(req.prompt_tokens);
            slots_[i].max_tokens = req.max_tokens;
            slots_[i].temperature = req.temperature;
            slots_[i].top_k = req.top_k;
            slots_[i].eos_id = req.eos_id;
            slots_[i].output_tokens.clear();
            slots_[i].prefill_pos = 0;
            slots_[i].result = std::move(req.result);
            pending_.pop();
            reset_slot_(i);
        }
    }

    bool has_active_slots() const {
        for (int i = 0; i < max_slots_; i++)
            if (slots_[i].state != BatchSlot::IDLE) return true;
        return false;
    }

    bool has_pending() {
        std::lock_guard<std::mutex> lk(mu_);
        return !pending_.empty();
    }

    void run() {
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]{ return !running_ || !pending_.empty() || has_active_slots(); });
                if (!running_ && pending_.empty() && !has_active_slots()) return;
            }

            admit_pending();

            int active = 0;
            for (int i = 0; i < max_slots_; i++) {
                if (slots_[i].state == BatchSlot::PREFILLING) {
                    auto& s = slots_[i];
                    size_t start = (s.prompt_tokens.size() > 0 && s.prompt_tokens[0] == s.eos_id) ? 1 : 0;
                    for (size_t j = start; j + 1 < s.prompt_tokens.size(); j++) {
                        gen_single_(s.prompt_tokens[j]);
                    }
                    s.last_token = s.prompt_tokens.empty() ? s.eos_id : s.prompt_tokens.back();
                    s.state = BatchSlot::DECODING;
                }
                if (slots_[i].state == BatchSlot::DECODING) active++;
            }
            active_count_.store(active);

            if (active == 0) continue;

            std::vector<std::pair<int,int>> batch;
            std::vector<int> slot_indices;
            for (int i = 0; i < max_slots_; i++) {
                if (slots_[i].state == BatchSlot::DECODING) {
                    batch.push_back({i, slots_[i].last_token});
                    slot_indices.push_back(i);
                }
            }

            std::vector<int> results;
            if (batch.size() == 1) {
                results.push_back(gen_single_(batch[0].second));
            } else {
                results = gen_batch_(batch);
            }

            for (size_t j = 0; j < slot_indices.size(); j++) {
                auto& s = slots_[slot_indices[j]];
                int tok = results[j];
                s.output_tokens.push_back(tok);
                s.last_token = tok;

                bool done = (tok == s.eos_id) ||
                            ((int)s.output_tokens.size() >= s.max_tokens);
                if (done) {
                    s.result.set_value(std::move(s.output_tokens));
                    s.state = BatchSlot::IDLE;
                }
            }

            active = 0;
            for (int i = 0; i < max_slots_; i++)
                if (slots_[i].state == BatchSlot::DECODING) active++;
            active_count_.store(active);
        }
    }
};
