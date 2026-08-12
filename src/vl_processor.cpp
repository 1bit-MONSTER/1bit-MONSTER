// vl_processor.cpp — stb_image implementation + ViT bbox-aware resize.
//
// This file:
//   1. Provides the single TU for stb_image.h's implementation
//   2. Bbox-aware resize: VL models like Qwen2-VL support dynamic resolution
//      by dividing the image into a grid of patches. This computes the
//      optimal crop-and-resize to minimize padding.
//   3. Image download from URL (curl subprocess or direct read)
//
// Upstream tracking: additive file, no existing file modified.

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"

#include "vl_processor.h"

#include <cstdio>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <arpa/inet.h>
#endif
#include <cstring>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <httplib.h>

// ── Bbox-aware resize for dynamic-resolution VLMs ──
// Qwen2-VL divides the image into a grid of patch-sized tiles
// (patch_size=14). The vision encoder processes (grid_h * grid_w) patches.
// This function resizes the image so the longest side fits within
// (max_patches * patch_size) while preserving aspect ratio, then pads
// to exact patch grid alignment.
//
// Returns processed pixels (resized + padded + normalized).
std::vector<float> vl_resize_bbox(const float* src, int sw, int sh,
                                   int patch_size, int max_patches,
                                   const float mean[3], const float std[3],
                                   int* out_w, int* out_h) {
    // Compute target size: fit longest side to max_patches * patch_size
    int max_dim = max_patches * patch_size;
    float scale;
    int tw, th;
    if (sw >= sh) {
        scale = (float)max_dim / sw;
        tw = max_dim;
        th = (int)(sh * scale + 0.5f);
        // Round th down to nearest multiple of patch_size
        th = (th / patch_size) * patch_size;
        if (th < patch_size) th = patch_size;
    } else {
        scale = (float)max_dim / sh;
        th = max_dim;
        tw = (int)(sw * scale + 0.5f);
        tw = (tw / patch_size) * patch_size;
        if (tw < patch_size) tw = patch_size;
    }

    if (out_w) *out_w = tw;
    if (out_h) *out_h = th;

    // First do a plain resize to (tw, th)
    std::vector<float> resized = vl_resize_normalize(src, sw, sh, tw, th, mean, std);

    return resized;
}

// ── Download image from URL to memory buffer ──
// In-process HTTP GET via cpp-httplib (no curl subprocess).
// SSRF hardening (issues #1278, #1291): every hop — the original URL AND each
// redirect target — is scheme-checked and DNS-resolved with private/loopback/
// link-local addresses rejected. Redirects are followed manually (max 5) so a
// 302 can't smuggle a request to an internal host past the guard. Response
// size is capped (32 MB) to bound memory.
// Returns empty vector on failure.

// Resolve `host` and reject private/loopback/link-local/unresolvable targets.
static bool vl_host_is_blocked(const std::string& host) {
    struct addrinfo hints{}, * res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0)
        return true;  // unresolvable — refuse rather than let the client guess
    bool blocked = false;
    for (auto* ai = res; ai; ai = ai->ai_next) {
        bool priv = false;
        if (ai->ai_family == AF_INET) {
            const unsigned char* b = (const unsigned char*)&((struct sockaddr_in*)ai->ai_addr)->sin_addr;
            priv = b[0] == 127 || b[0] == 10 || (b[0] == 172 && b[1] >= 16 && b[1] <= 31) ||
                   (b[0] == 192 && b[1] == 168) || (b[0] == 169 && b[1] == 254) || b[0] == 0;
        } else if (ai->ai_family == AF_INET6) {
            const unsigned char* s6 = (const unsigned char*)&((struct sockaddr_in6*)ai->ai_addr)->sin6_addr;
            priv = s6[0] == 0xFE && (s6[1] & 0xC0) == 0x80;  // link-local fe80::/10
            bool loop = true;
            for (int i = 0; i < 15; i++) if (s6[i] != 0) loop = false;
            if (loop && s6[15] == 1) priv = true;  // ::1
        } else {
            priv = true;  // non-IP family
        }
        if (priv) { blocked = true; break; }
    }
    freeaddrinfo(res);
    return blocked;
}

// Split "scheme://host[:port]/path" into parts. Returns false on malformed input.
static bool vl_split_url(const std::string& url, std::string& scheme,
                         std::string& host, int& port, std::string& path) {
    auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) return false;
    scheme = url.substr(0, scheme_end);
    if (scheme != "http" && scheme != "https") return false;
    size_t hs = scheme_end + 3;
    size_t he = url.find('/', hs);
    if (he == std::string::npos) { he = url.size(); path = "/"; }
    else path = url.substr(he);
    std::string authority = url.substr(hs, he - hs);
    if (authority.empty()) return false;
    if (authority[0] == '[') {  // IPv6 literal
        auto br = authority.find(']');
        if (br == std::string::npos) return false;
        host = authority.substr(1, br - 1);
        if (br + 1 < authority.size() && authority[br + 1] == ':')
            port = atoi(authority.c_str() + br + 2);
    } else {
        auto colon = authority.rfind(':');
        if (colon != std::string::npos && authority.find(':') == colon) {
            host = authority.substr(0, colon);
            port = atoi(authority.c_str() + colon + 1);
        } else {
            host = authority;  // bare host (or IPv6 without brackets — reject)
            if (host.find(':') != std::string::npos) return false;
        }
    }
    if (host.empty()) return false;
    if (port <= 0) port = (scheme == "https") ? 443 : 80;
    return true;
}

std::vector<unsigned char> vl_download_image(const std::string& url, int timeout_sec) {
    if (timeout_sec <= 0) timeout_sec = 10;
    const size_t MAX_BODY = 32ull * 1024 * 1024;  // 32 MB cap (issue #1296)
    const int MAX_HOPS = 5;

    std::string cur = url;
    for (int hop = 0; hop < MAX_HOPS; hop++) {
        std::string scheme, host;
        int port = 0;
        std::string path;
        if (!vl_split_url(cur, scheme, host, port, path)) {
            fprintf(stderr, "[vl] ERROR: unsupported/malformed URL: '%s'\n", cur.c_str());
            return {};
        }
        if (vl_host_is_blocked(host)) {
            fprintf(stderr, "[vl] ERROR: blocked internal/private host: '%s'\n", host.c_str());
            return {};
        }

        httplib::Client cli(scheme + "://" + host + ":" + std::to_string(port));
        cli.set_follow_location(false);
        cli.set_connection_timeout(timeout_sec, 0);
        cli.set_read_timeout(timeout_sec, 0);

        std::vector<unsigned char> data;
        data.reserve(1 << 20);
        bool too_big = false;
        auto res = cli.Get(path, [&](const char* buf, size_t n) -> bool {
            if (data.size() + n > MAX_BODY) { too_big = true; return false; }
            data.insert(data.end(), buf, buf + n);
            return true;
        });

        if (too_big) {
            fprintf(stderr, "[vl] ERROR: response too large (>%zu MB): '%s'\n", MAX_BODY >> 20, cur.c_str());
            return {};
        }
        if (!res) {
            fprintf(stderr, "[vl] ERROR: request failed for '%s': %s\n", cur.c_str(), httplib::to_string(res.error()).c_str());
            return {};
        }

        int status = res->status;
        if (status == 301 || status == 302 || status == 303 || status == 307 || status == 308) {
            std::string loc = res->get_header_value("Location");
            if (loc.empty()) {
                fprintf(stderr, "[vl] ERROR: redirect without Location from '%s'\n", cur.c_str());
                return {};
            }
            // Resolve relative Location against the current URL.
            if (loc.find("://") == std::string::npos) {
                if (!loc.empty() && loc[0] == '/') {
                    loc = scheme + "://" + host + ":" + std::to_string(port) + loc;
                } else {
                    // relative-path redirect (rare) — resolve against current dir
                    std::string base = path;
                    auto slash = base.rfind('/');
                    base = (slash == std::string::npos) ? "/" : base.substr(0, slash + 1);
                    loc = scheme + "://" + host + ":" + std::to_string(port) + base + loc;
                }
            }
            cur = loc;
            continue;  // next hop: scheme + host re-validated
        }
        if (status != 200) {
            fprintf(stderr, "[vl] ERROR: HTTP %d from '%s'\n", status, cur.c_str());
            return {};
        }
        if (data.empty()) {
            fprintf(stderr, "[vl] ERROR: empty response from '%s'\n", cur.c_str());
            return {};
        }
        return data;
    }
    fprintf(stderr, "[vl] ERROR: too many redirects (>%d) for '%s'\n", MAX_HOPS, url.c_str());
    return {};
}

// ── Detect if a string is a data URL (base64-encoded image) ──
bool vl_is_data_url(const std::string& url) {
    return url.find("data:image/") == 0;
}

// ── Decode base64 data URL to raw bytes ──
std::vector<unsigned char> vl_decode_base64_image(const std::string& data_url) {
    auto comma = data_url.find(',');
    if (comma == std::string::npos) return {};
    std::string b64 = data_url.substr(comma + 1);

    // Cap encoded size (issue #1296); reject invalid chars instead of
    // silently decoding them as 0xFF garbage (issue #1299).
    const size_t MAX_B64 = 32ull * 1024 * 1024;  // 32 MB encoded -> ~24 MB raw
    if (b64.size() > MAX_B64) {
        fprintf(stderr, "[vl] ERROR: base64 payload too large (>%zu MB)\n", MAX_B64 >> 20);
        return {};
    }

    static const char b64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    signed char b64_rev[256];
    for (int i = 0; i < 256; i++) b64_rev[i] = -1;
    for (int i = 0; i < 64; i++) b64_rev[(unsigned char)b64_chars[i]] = i;

    std::string clean;
    clean.reserve(b64.size());
    for (char c : b64) {
        if (isspace((unsigned char)c)) continue;
        if (c != '=' && b64_rev[(unsigned char)c] < 0) {
            fprintf(stderr, "[vl] ERROR: invalid base64 character '%c' in data URL\n", c);
            return {};
        }
        clean += c;
    }

    size_t len = clean.size();
    if (len == 0) return {};

    size_t out_len = len / 4 * 3;
    if (len > 0 && clean[len-1] == '=') out_len--;
    if (len > 1 && clean[len-2] == '=') out_len--;

    std::vector<unsigned char> out(out_len);
    size_t pos = 0;
    for (size_t i = 0; i < len; i += 4) {
        unsigned char s[4] = {0, 0, 0, 0};
        for (int j = 0; j < 4 && i + j < len; j++) {
            char c = clean[i + j];
            if (c == '=') s[j] = 0;
            else s[j] = (unsigned char)b64_rev[(unsigned char)c];
        }
        if (pos < out_len) out[pos++] = (s[0] << 2) | (s[1] >> 4);
        if (pos < out_len) out[pos++] = (s[1] << 4) | (s[2] >> 2);
        if (pos < out_len) out[pos++] = (s[2] << 6) | s[3];
    }
    return out;
}
