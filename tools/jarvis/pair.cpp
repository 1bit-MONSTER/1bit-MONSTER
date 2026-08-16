// pair.cpp — QR-code zero-trust device pairing (see pair.h).
#include "pair.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <random>
#include <vector>

namespace jarvis {

// ── Terminal QR ────────────────────────────────────────────────────────────
// qrencode is a tiny, near-universal package (libqrencode) already in the
// install script deps. Rendering in-process would mean embedding a full
// QR encoder + Reed-Solomon — not worth it.
bool print_qr(const std::string& url) {
    // URL goes via stdin — never interpolated into a shell command.
    // 1>&2 routes qrencode's QR output to stderr alongside the banner;
    // 2>/dev/null keeps qrencode's own errors silent. Order matters.
    FILE* pipe = popen("qrencode -t ANSIUTF8 1>&2 2>/dev/null", "w");
    if (!pipe) return false;
    fwrite(url.data(), 1, url.size(), pipe);
    if (pclose(pipe) == 0) return true;  // exit 0 == qrencode rendered
    fprintf(stderr, "Pairing URL (install qrencode for a scannable QR): %s\n", url.c_str());
    return false;
}

// ── PairingManager ─────────────────────────────────────────────────────────
namespace {
constexpr double kDefaultTtl = 300.0;   // 5 min
double kNow() { return (double)std::chrono::duration_cast<std::chrono::seconds>(
    std::chrono::system_clock::now().time_since_epoch()).count(); }

// Alphabet without confusables (I/L/O/0/1): ~37 bits for 8 chars.
const char kAlphabet[] = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";
}

struct PairingManager::Impl {
    std::mutex mtx;
    double ttl = kDefaultTtl;
    std::map<std::string, double> codes;          // code -> expires_at
    std::map<std::string, bool> used;             // code -> consumed
    std::map<std::string, std::vector<double>> starts;  // ip -> timestamps
    std::map<std::string, std::vector<double>> claims;  // ip -> timestamps

    void purge_locked() {
        double now = kNow();
        for (auto it = codes.begin(); it != codes.end();) {
            if (it->second < now) it = codes.erase(it);
            else ++it;
        }
    }

    std::string gen_code() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        std::uniform_int_distribution<int> dist(0, (int)std::strlen(kAlphabet) - 1);
        std::string c;
        for (int i = 0; i < 8; i++) c += kAlphabet[dist(rng)];
        return c;
    }

    bool rate_locked(std::map<std::string, std::vector<double>>& bucket,
                     const std::string& ip, int max_per_min) {
        double now = kNow();
        auto& v = bucket[ip];
        v.erase(std::remove_if(v.begin(), v.end(),
                               [now](double t) { return t < now - 60.0; }), v.end());
        if ((int)v.size() >= max_per_min) return true;
        v.push_back(now);
        return false;
    }
};

void PairingManager::ImplDeleter::operator()(Impl* p) const { delete p; }

PairingManager::PairingManager() : impl_(new Impl) {}
PairingManager::~PairingManager() = default;

void PairingManager::set_ttl_seconds_for_test(double ttl) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->ttl = ttl > 0 ? ttl : kDefaultTtl;
}

bool PairingManager::start_allowed(const std::string& ip) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    return !impl_->rate_locked(impl_->starts, ip, 3);
}

bool PairingManager::claim_allowed(const std::string& ip) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    return !impl_->rate_locked(impl_->claims, ip, 10);
}

std::pair<std::string, double> PairingManager::start() {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->purge_locked();
    std::string code;
    do { code = impl_->gen_code(); } while (impl_->codes.count(code));
    double expires = kNow() + impl_->ttl;
    impl_->codes[code] = expires;
    impl_->used[code] = false;
    return {code, expires};
}

std::string PairingManager::claim(const std::string& code, std::string* owner_id_out) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->purge_locked();
    auto it = impl_->codes.find(code);
    if (it == impl_->codes.end()) return "not_found";
    if (kNow() > it->second) { impl_->codes.erase(it); impl_->used.erase(code); return "expired"; }
    if (impl_->used[code]) return "already_used";
    impl_->used[code] = true;  // consume — atomic under mtx, single winner
    impl_->codes.erase(it);
    if (owner_id_out) *owner_id_out = "pair:" + code;
    return "ok";
}

} // namespace jarvis
