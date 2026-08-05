// Standalone check for the video-encode path in src/diffusion_bridge.cpp:
// generate_video() -> create_video_from_sd_images_to_vector("avi", ...).
// Fails if the AVI/MJPG container isn't produced from RGB frames.
//
// Build & run:
//   g++ -std=c++17 -O1 \
//     -I third_party/stable-diffusion.cpp/examples/common \
//     -I third_party/stable-diffusion.cpp/thirdparty \
//     -I third_party/stable-diffusion.cpp/include \
//     tests/vid_encoder_check.cpp \
//     third_party/stable-diffusion.cpp/examples/common/media_io.cpp \
//     third_party/stable-diffusion.cpp/examples/common/log.cpp \
//     -o /tmp/vid_encoder_check && /tmp/vid_encoder_check
#include "media_io.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cassert>

static bool has(const std::vector<uint8_t>& v, const char* fourcc) {
    for (size_t i = 0; i + 4 <= v.size(); i++)
        if (memcmp(v.data() + i, fourcc, 4) == 0) return true;
    return false;
}

int main() {
    const int W = 8, H = 8, N = 3;
    std::vector<sd_image_t> frames(N);
    std::vector<std::vector<uint8_t>> pixels(N);
    for (int i = 0; i < N; i++) {
        pixels[i].assign((size_t)W * H * 3, (uint8_t)(i * 80)); // gray ramp per frame
        frames[i] = {(uint32_t)W, (uint32_t)H, 3, pixels[i].data()};
    }
    auto avi = create_video_from_sd_images_to_vector("avi", frames.data(), N, 16, 90, nullptr);
    assert(!avi.empty());
    assert(memcmp(avi.data(), "RIFF", 4) == 0);
    assert(memcmp(avi.data() + 8, "AVI ", 4) == 0);
    assert(has(avi, "MJPG"));

#ifdef SD_USE_WEBM
    // WebM build: real EBML container, playable in browsers/ComfyUI.
    auto webm = create_video_from_sd_images_to_vector("webm", frames.data(), N, 16, 90, nullptr);
    assert(!webm.empty());
    assert(memcmp(webm.data(), "\x1A\x45\xDF\xA3", 4) == 0);
    printf("ok: avi=%zu bytes MJPG; webm=%zu bytes EBML container\n", avi.size(), webm.size());
#else
    // No webm in this build: a webm request must degrade to AVI bytes, not
    // crash — the bridge sniffs the container, so the mime stays honest.
    auto webm = create_video_from_sd_images_to_vector("webm", frames.data(), N, 16, 90, nullptr);
    assert(!webm.empty());
    assert(memcmp(webm.data(), "RIFF", 4) == 0 && has(webm, "MJPG"));
    printf("ok: avi=%zu bytes, 3 frames 8x8 MJPG container; webm degrades to AVI\n", avi.size());
#endif
    return 0;
}
