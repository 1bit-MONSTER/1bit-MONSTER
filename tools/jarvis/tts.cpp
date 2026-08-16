// tts.cpp — Piper subprocess TTS (bidirectional pipe, like the old tts.cpp
// but without the agent crust). See tts.h.

#include "jarvis/tts.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>

namespace jarvis {

bool TTS::load(const std::string& piper_bin, const std::string& model_path) {
    piper_bin_ = piper_bin.empty() ? "piper" : piper_bin;
    model_path_ = model_path;
    loaded_ = !model_path_.empty();
    return loaded_;
}

std::vector<float> TTS::synth(const std::string& text, int& out_sample_rate) {
    out_sample_rate = 22050;
    if (!loaded_ || text.empty()) return {};

    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) return {};

    pid_t pid = fork();
    if (pid < 0) { close(in_pipe[0]); close(in_pipe[1]); close(out_pipe[0]); close(out_pipe[1]); return {}; }

    if (pid == 0) {
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        execlp(piper_bin_.c_str(), piper_bin_.c_str(),
               "--model", model_path_.c_str(), "--output-raw", (char*)nullptr);
        _exit(127);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);

    // Write text, close stdin so piper flushes and exits.
    const char* p = text.data();
    size_t left = text.size();
    while (left > 0) {
        ssize_t w = write(in_pipe[1], p, left);
        if (w <= 0) break;
        p += w;
        left -= (size_t)w;
    }
    close(in_pipe[1]);

    // Read raw s16le PCM (22050 Hz mono) to EOF.
    std::vector<int16_t> raw;
    int16_t buf[8192];
    ssize_t got;
    while ((got = read(out_pipe[0], buf, sizeof(buf))) > 0)
        raw.insert(raw.end(), buf, buf + got / (ssize_t)sizeof(int16_t));
    close(out_pipe[0]);

    int status = 0;
    waitpid(pid, &status, 0);

    if (raw.empty()) return {};
    std::vector<float> f32(raw.size());
    for (size_t i = 0; i < raw.size(); i++) f32[i] = (float)raw[i] / 32768.0f;
    return f32;
}

}  // namespace jarvis
