#include "../tools/jarvis/voice_session.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace jarvis;

static void assert_state(SessionState got, SessionState want, const char* what) {
    if (got != want) { std::printf("FAIL %s: got %d want %d\n", what, (int)got, (int)want); std::exit(1); }
}

static double rms(const std::vector<int16_t>& v) {
    if (v.empty()) return 0.0;
    double sum = 0.0;
    for (int16_t s : v) sum += (double)s * s;
    return std::sqrt(sum / v.size());
}

int main() {
    // 1 second of 16k sine = "speech" (RMS >> 0.01 threshold)
    std::vector<int16_t> speech(16000);
    for (int i = 0; i < 16000; ++i) speech[i] = (int16_t)(12000 * std::sin(2 * 3.14159 * 440 * i / 16000.0));
    // 1 s of silence (well past min_silence_ms=500)
    std::vector<int16_t> silence(16000, 0);

    VoiceSession s;
    std::vector<SessionState> states;
    int utterances = 0;
    std::vector<std::vector<int16_t>> got_audio;
    s.set_callbacks(
        [&](SessionState st) { states.push_back(st); },
        [&](const std::vector<int16_t>& pcm) { utterances++; got_audio.push_back(pcm); },
        [&](const std::string&) {});
    s.start();
    assert_state(s.state(), SessionState::Listening, "start -> Listening");

    // speech then silence -> exactly one utterance, state Processing
    s.feed(speech.data(), speech.size());
    s.feed(silence.data(), silence.size());
    assert_state(s.state(), SessionState::Processing, "utterance -> Processing");
    assert(utterances == 1);
    // (a) payload is the VAD-purified speech segment: non-empty, significant energy
    assert(!got_audio[0].empty());
    assert(rms(got_audio[0]) > 1000.0);

    // (b) set_speaking(false) directly returns to Listening, VAD re-armed
    s.set_speaking(true);
    assert_state(s.state(), SessionState::Speaking, "set_speaking(true) -> Speaking");
    s.set_speaking(false);
    assert_state(s.state(), SessionState::Listening, "set_speaking(false) -> Listening");

    // fresh utterance -> Processing, then speaking -> tick -> back to listening
    s.feed(speech.data(), speech.size());
    s.feed(silence.data(), silence.size());
    assert(utterances == 2);
    assert_state(s.state(), SessionState::Processing, "utterance2 -> Processing");
    s.set_speaking(true);
    assert_state(s.state(), SessionState::Speaking, "set_speaking -> Speaking");
    s.tick(200);  // > 100 ms quiet timeout
    assert_state(s.state(), SessionState::Listening, "tick -> Listening");

    // (c) regression: short burst after the transition must NOT include stale
    // pre-transition audio; utterance 3 is fresh VAD output.
    std::vector<int16_t> burst(4800);  // 0.3 s of speech
    for (int i = 0; i < 4800; ++i) burst[i] = (int16_t)(12000 * std::sin(2 * 3.14159 * 440 * i / 16000.0));
    s.feed(burst.data(), burst.size());
    s.feed(silence.data(), silence.size());
    assert_state(s.state(), SessionState::Processing, "burst utterance -> Processing");
    assert(utterances == 3);
    assert(rms(got_audio[2]) > 1000.0);                      // real speech, not silence
    assert(got_audio[2].size() < 20000);                     // stale pre-roll would push it past this

    // stop drops audio, returns Idle
    s.feed(speech.data(), speech.size());
    s.stop();
    assert_state(s.state(), SessionState::Idle, "stop -> Idle");
    assert(utterances == 3);

    std::printf("PASS voice_session_test (%d utterances, %zu states)\n", utterances, states.size());
    return 0;
}
