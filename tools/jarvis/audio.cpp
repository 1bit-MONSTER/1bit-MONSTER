// audio.cpp — arecord/aplay child-process audio. See audio.h.

#include "jarvis/audio.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

namespace jarvis {

namespace {

// s16le interleaved mono -> f32
void s16_to_f32(const int16_t* in, int n, float* out) {
    for (int i = 0; i < n; i++) out[i] = (float)in[i] / 32768.0f;
}

// s16 mono f32 -> s16
void f32_to_s16(const float* in, int n, int16_t* out) {
    for (int i = 0; i < n; i++) {
        float v = in[i] * 32768.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        out[i] = (int16_t)v;
    }
}

}  // namespace

// ── Capture ────────────────────────────────────────────────────────────

struct Capture::Impl {
    pid_t pid = -1;
    int fd = -1;                     // arecord stdout (raw s16)
    std::thread reader;
    std::atomic<bool> running{false};
    int sample_rate = 16000;
    SamplesCb cb;

    void read_loop() {
        std::vector<int16_t> raw(4096);
        std::vector<float> f32(4096);
        while (running) {
            ssize_t got = ::read(fd, raw.data(), raw.size() * sizeof(int16_t));
            if (got <= 0) break;
            int n = (int)(got / sizeof(int16_t));
            s16_to_f32(raw.data(), n, f32.data());
            cb(f32.data(), n);
        }
    }
};

bool Capture::start(int sample_rate_hz, const std::string& device, SamplesCb cb) {
    stop();
    auto* im = new Impl;
    impl_ = im;
    im->sample_rate = sample_rate_hz;
    im->cb = std::move(cb);

    int pipefd[2];
    if (pipe(pipefd) != 0) { delete im; impl_ = nullptr; return false; }

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); delete im; impl_ = nullptr; return false; }

    if (pid == 0) {
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[0]); close(pipefd[1]);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        execlp("arecord", "arecord", "-q", "-t", "raw", "-f", "S16_LE",
               "-r", std::to_string(sample_rate_hz).c_str(), "-c", "1",
               "-D", device.c_str(), (char*)nullptr);
        _exit(127);
    }

    close(pipefd[1]);
    im->pid = pid;
    im->fd = pipefd[0];
    im->running = true;
    im->reader = std::thread([im] { im->read_loop(); });
    return true;
}

void Capture::stop() {
    if (!impl_) return;
    impl_->running = false;
    if (impl_->pid > 0) {
        kill(impl_->pid, SIGTERM);
        int status = 0;
        for (int i = 0; i < 50 && waitpid(impl_->pid, &status, WNOHANG) == 0; i++)
            usleep(20000);
        if (waitpid(impl_->pid, &status, WNOHANG) == 0) kill(impl_->pid, SIGKILL);
        waitpid(impl_->pid, &status, 0);
    }
    if (impl_->fd >= 0) close(impl_->fd);
    if (impl_->reader.joinable()) impl_->reader.join();
    delete impl_;
    impl_ = nullptr;
}

// ── Playback ───────────────────────────────────────────────────────────

struct Playback::Impl {
    pid_t pid = -1;
    int fd = -1;  // aplay stdin
};

bool Playback::start(int sample_rate_hz) {
    stop();
    auto* im = new Impl;
    impl_ = im;

    int pipefd[2];
    if (pipe(pipefd) != 0) { delete im; impl_ = nullptr; return false; }

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); delete im; impl_ = nullptr; return false; }

    if (pid == 0) {
        dup2(pipefd[0], STDIN_FILENO);
        close(pipefd[0]); close(pipefd[1]);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        execlp("aplay", "aplay", "-q", "-t", "raw", "-f", "S16_LE",
               "-r", std::to_string(sample_rate_hz).c_str(), "-c", "1",
               (char*)nullptr);
        _exit(127);
    }

    close(pipefd[0]);
    im->pid = pid;
    im->fd = pipefd[1];
    return true;
}

void Playback::write(const float* samples, int n) {
    if (!impl_ || n <= 0) return;
    std::vector<int16_t> s16((size_t)n);
    f32_to_s16(samples, n, s16.data());
    const char* p = (const char*)s16.data();
    size_t left = s16.size() * sizeof(int16_t);
    while (left > 0) {
        ssize_t w = ::write(impl_->fd, p, left);
        if (w <= 0) return;  // aplay died (SIGPIPE ignored at startup)
        p += w;
        left -= (size_t)w;
    }
}

void Playback::stop() {
    if (!impl_) return;
    if (impl_->pid > 0) {
        close(impl_->fd);
        impl_->fd = -1;
        int status = 0;
        for (int i = 0; i < 100 && waitpid(impl_->pid, &status, WNOHANG) == 0; i++)
            usleep(20000);
        if (waitpid(impl_->pid, &status, WNOHANG) == 0) kill(impl_->pid, SIGKILL);
        waitpid(impl_->pid, &status, 0);
        impl_->pid = -1;
    }
    delete impl_;
    impl_ = nullptr;
}

}  // namespace jarvis
