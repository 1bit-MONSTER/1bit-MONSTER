#include "voice_session.h"
#include "vad.h"
#include <algorithm>

namespace jarvis {

struct VoiceSession::Impl {
    VAD vad{VADConfig{}};
    SessionState st = SessionState::Idle;
    std::vector<float> pcm_f32;
    std::vector<int16_t> pending;   // pcm16 since utterance start
    int speaking_ms = 0;
    StateCallback on_state;
    UtteranceCallback on_utterance;
    ErrorCallback on_error;

    void set(SessionState next) {
        if (st == next) return;
        st = next;
        if (on_state) on_state(st);
    }
};

VoiceSession::VoiceSession() : impl_(new Impl) {}
VoiceSession::~VoiceSession() = default;

void VoiceSession::set_callbacks(StateCallback s, UtteranceCallback u, ErrorCallback e) {
    impl_->on_state = std::move(s);
    impl_->on_utterance = std::move(u);
    impl_->on_error = std::move(e);
}

SessionState VoiceSession::state() const { return impl_->st; }

void VoiceSession::start() {
    impl_->vad.reset();
    impl_->pending.clear();
    impl_->speaking_ms = 0;
    impl_->set(SessionState::Listening);
}

void VoiceSession::stop() {
    impl_->vad.reset();
    impl_->pending.clear();
    impl_->speaking_ms = 0;
    impl_->set(SessionState::Idle);
}

void VoiceSession::feed(const int16_t* pcm16, size_t n_samples) {
    if (impl_->st == SessionState::Idle || n_samples == 0) return;
    impl_->pcm_f32.resize(n_samples);
    for (size_t i = 0; i < n_samples; ++i) impl_->pcm_f32[i] = pcm16[i] / 32768.0f;
    impl_->vad.process(impl_->pcm_f32.data(), (int)n_samples);
    if (impl_->vad.is_speaking()) {
        impl_->pending.insert(impl_->pending.end(), pcm16, pcm16 + n_samples);
    }
    auto utt = impl_->vad.get_last_utterance();
    if (!utt.empty() && impl_->st == SessionState::Listening) {
        impl_->set(SessionState::Processing);
        if (impl_->on_utterance) impl_->on_utterance(impl_->pending);
        impl_->pending.clear();
        impl_->vad.reset();
    }
}

void VoiceSession::set_speaking(bool speaking) {
    if (speaking && impl_->st == SessionState::Processing) {
        impl_->speaking_ms = 0;
        impl_->set(SessionState::Speaking);
    } else if (!speaking && impl_->st == SessionState::Speaking) {
        impl_->set(SessionState::Listening);
        impl_->vad.reset();
    }
}

void VoiceSession::tick(int ms_elapsed) {
    if (impl_->st != SessionState::Speaking) return;
    impl_->speaking_ms += ms_elapsed;
    if (impl_->speaking_ms > 100) {  // quiet timeout after speech playback
        impl_->set(SessionState::Listening);
        impl_->vad.reset();
    }
}

} // namespace jarvis
