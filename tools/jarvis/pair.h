// pair.h — QR-code zero-trust device pairing for the Jarvis server.
//
// Bootstrap protocol: the operator asks the server for a pairing code
// (POST /v1/pair/start, or it's printed at startup). The server renders
// a QR encoding the claim URL http://<lan-ip>:<port>/v1/pair/claim?code=X
// on the operator's console. Scanning it with a phone opens the claim URL,
// which consumes the code ONCE (single-use, 5 min TTL) and mints a
// per-device API key. The QR is the out-of-band human-verified credential:
// possession of the screen + one-time code == trust. Everything after the
// bootstrap is standard Bearer-key auth (see auth.h).
//
// Zero-trust properties:
//   - single-use codes (atomic consume under mutex)
//   - short TTL (default 300 s) — a leaked code is useless in 5 minutes
//   - per-IP rate limits on both start (3/min) and claim (10/min)
//   - per-device keys, revocable via /v1/api-key/revoke
//   - no implicit trust: the server binds loopback by default
#pragma once

#include <memory>
#include <string>
#include <utility>

namespace jarvis {

// Generates a QR code for `url` on the terminal (via `qrencode`, if
// installed). Prints the URL as fallback text otherwise. Writes to stderr
// so it never pollutes stdout JSON. Returns true if a QR was rendered.
bool print_qr(const std::string& url);

class PairingManager {
public:
    PairingManager();
    ~PairingManager();

    // Allocate a fresh one-time code with a 5-minute TTL.
    // Returns {code, expires_at_unix}.
    std::pair<std::string, double> start();

    // Consume `code` (single-use). Sets `owner_id_out` on success.
    // Returns "ok", "not_found", "already_used", or "expired".
    std::string claim(const std::string& code, std::string* owner_id_out);

    bool start_allowed(const std::string& ip);   // 3 starts / min / IP
    bool claim_allowed(const std::string& ip);   // 10 claims / min / IP

    // Internal: TTL override for tests (seconds). 0 = default.
    void set_ttl_seconds_for_test(double ttl);

private:
    struct Impl;
    struct ImplDeleter { void operator()(Impl*) const; };
    std::unique_ptr<Impl, ImplDeleter> impl_;
};

} // namespace jarvis
