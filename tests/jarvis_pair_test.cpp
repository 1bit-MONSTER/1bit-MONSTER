// jarvis_pair_test.cpp — QR zero-trust pairing: single-use codes, TTL,
// rate limits. Run: build/jarvis_pair_test
#include "../tools/jarvis/pair.h"
#include <cassert>
#include <cstdio>
#include <chrono>
#include <string>
#include <thread>

using namespace jarvis;

int main() {
    PairingManager pm;

    // Single use: first claim wins, second is rejected.
    auto [code, expires] = pm.start();
    assert(code.size() == 8 && expires > 0);
    std::string owner;
    assert(pm.claim(code, &owner) == "ok" && !owner.empty());
    assert(pm.claim(code, &owner) == "already_used");

    // Unknown code.
    assert(pm.claim("ZZZZZZZZ", &owner) == "not_found");

    // Expiry: TTL of 1s — wait it out.
    pm.set_ttl_seconds_for_test(1.0);
    auto [code2, expires2] = pm.start();
    (void)expires2;
    {
        using namespace std::chrono;
        std::this_thread::sleep_for(milliseconds(1100));
    }
    assert(pm.claim(code2, &owner) == "expired");

    // Rate limits: >10 claims/min from one IP blocked.
    pm.set_ttl_seconds_for_test(300.0);
    for (int i = 0; i < 10; i++) {
        auto [c, e] = pm.start();
        (void)e;
        assert(pm.claim(c, &owner) == "ok");
    }
    assert(!pm.claim_allowed("10.0.0.1"));

    std::printf("PASS jarvis_pair_test\n");
    return 0;
}
