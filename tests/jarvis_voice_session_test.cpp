#include "../tools/jarvis/voice_session.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace jarvis;

static void assert_state(SessionState got, SessionState want, const char* what) {
    if (got != want) { std::printf("FAIL %s: got %d want %d\n", what, (int)got, (int)want); std::exit(1); }
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
    assert(!got_audio[0].empty());

    // speaking -> tick -> back to listening (VAD re-arm)
    s.set_speaking(true);
    assert_state(s.state(), SessionState::Speaking, "set_speaking -> Speaking");
    s.tick(200);  // > 100 ms quiet timeout
    assert_state(s.state(), SessionState::Listening, "tick -> Listening");

    // stop drops audio, returns Idle
    s.feed(speech.data(), speech.size());
    s.stop();
    assert_state(s.state(), SessionState::Idle, "stop -> Idle");
    assert(utterances == 1);

    std::printf("PASS voice_session_test (%d utterances, %zu states)\n", utterances, states.size());
    return 0;
}
