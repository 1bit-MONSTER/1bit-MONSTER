#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>

#define MAX_HEADERS 65536

static pid_t worker_pid = 0;
static int worker_stdin = -1;

// ── Drain stdout until READY (separate thread, prevents pipe deadlock) ──
struct DrainArg { int fd; volatile int* ready; };
static void* drain_stdout(void* arg) {
    struct DrainArg* da = (struct DrainArg*)arg;
    char buf[4096];
    FILE* wout = fdopen(da->fd, "r");
    if (!wout) return NULL;
    while (fgets(buf, sizeof(buf), wout)) {
        if (strcmp(buf, "READY\n") == 0) { *da->ready = 1; break; }
    }
    fclose(wout);
    return NULL;
}

// ── Spawn NPU worker ──
static int spawn_worker(void) {
    const char* model = getenv("NPU_MODEL_PATH");
    const char* tag = getenv("NPU_MODEL_TAG") ?: "unknown";
    const char* bin = getenv("NPU_ENGINE_BIN");
    char default_bin[1024];
    if (!bin) {
        snprintf(default_bin, sizeof(default_bin), "%s/1bit-systems/engine/npu/build/npu_engine_universal", getenv("HOME"));
        bin = default_bin;
    }
    int to_c[2], from_c[2];
    if (pipe(to_c) < 0 || pipe(from_c) < 0) return -1;
    worker_pid = fork();
    if (worker_pid < 0) return -1;
    if (worker_pid == 0) {
        close(to_c[1]); dup2(to_c[0], 0); close(to_c[0]);
        close(from_c[0]); dup2(from_c[1], 1); close(from_c[1]);
        execlp(bin, bin, model, "--model-tag", tag, "--worker", NULL);
        _exit(1);
    }
    close(to_c[0]); close(from_c[1]);
    worker_stdin = to_c[1];
    // Drain stdout in a separate process to prevent pipe deadlock
    // Reads raw bytes and scans for "READY\n" (avoids fdopen buffering issues)
    int drain_pid = fork();
    if (drain_pid == 0) {
        char dbuf[4096];
        size_t got = 0;
        while (1) {
            ssize_t n = read(from_c[0], dbuf + got, sizeof(dbuf) - got - 1);
            if (n <= 0) _exit(1);
            got += n;
            dbuf[got] = 0;
            if (strstr(dbuf, "READY\n")) _exit(0);
            // Keep buffer from growing unbounded — shift if full
            if (got > sizeof(dbuf) - 256) {
                char* p = strrchr(dbuf, '\n');
                if (p) { got -= (p + 1 - dbuf); memmove(dbuf, p + 1, got); }
                else { got = 0; }
            }
        }
    }
    close(from_c[0]);
    int wstatus;
    waitpid(drain_pid, &wstatus, 0);
    if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) {
        fprintf(stderr, "Worker not ready\n");
        kill(worker_pid, SIGTERM);
        return -1;
    }
    fprintf(stderr, "Worker ready (PID %d)\n", worker_pid);
    return 0;
}

// ── HTTP response helpers ──
static void send_json(int fd, const char* json) {
    char hdr[4096];
    int n = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n%s", strlen(json), json);
    write(fd, hdr, n);
    close(fd);
}

static void send_error(int fd, int c, const char* m) {
    char buf[256]; snprintf(buf, sizeof(buf), "{\"error\":\"%s\"}", m);
    send_json(fd, buf);
}

// ── Client handler ──
static void* handle_client(void* arg) {
    int fd = (int)(long)arg;
    char buf[MAX_HEADERS];
    ssize_t n = read(fd, buf, sizeof(buf)-1);
    if (n <= 0) { close(fd); return NULL; }
    buf[n] = 0;
    char method[16], path[256];
    if (sscanf(buf, "%15s %255s", method, path) < 2) { close(fd); return NULL; }
    if (strcmp(method, "GET") == 0 && (strcmp(path, "/v1/models") == 0 || strcmp(path, "/health") == 0)) {
        char _b[256]; snprintf(_b, sizeof(_b), "{\"status\":\"ok\",\"worker_pid\":%d}", worker_pid); send_json(fd, _b);
        return NULL;
    }
    // Chat endpoint — returns worker status (full inference needs transformer code)
    if (strcmp(method, "POST") == 0 && strstr(path, "chat/completions")) {
        char resp[4096];
        snprintf(resp, sizeof(resp),
            "{\"id\":\"1\",\"object\":\"chat.completion\","
            "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
            "\"content\":\"NPU worker alive (PID %d)\"},\"finish_reason\":\"stop\"}]}",
            worker_pid);
        send_json(fd, resp);
        return NULL;
    }
    send_error(fd, 404, "not found");
    return NULL;
}

int main(int argc, char** argv) {
    int port = argc > 1 ? atoi(argv[1]) : 9091;
    if (!getenv("NPU_MODEL_PATH")) { fprintf(stderr, "Set NPU_MODEL_PATH\n"); return 1; }
    signal(SIGCHLD, SIG_IGN); signal(SIGPIPE, SIG_IGN);
    if (spawn_worker() < 0) { fprintf(stderr, "Worker failed\n"); return 1; }

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(port), .sin_addr = { INADDR_ANY } };
    if (bind(fd, (struct sockaddr*)&a, sizeof(a)) < 0 || listen(fd, 16) < 0) { perror("bind"); return 1; }
    fprintf(stderr, "NPU chat: http://127.0.0.1:%d (PID %d)\n", port, worker_pid);
    fprintf(stderr, "Zero FLM dependency.\n");

    while (1) {
        struct sockaddr_in ca; socklen_t cl = sizeof(ca);
        int cfd = accept(fd, (struct sockaddr*)&ca, &cl);
        if (cfd < 0) break;
        pthread_t t; pthread_create(&t, NULL, handle_client, (void*)(long)cfd);
        pthread_detach(t);
    }
    close(fd);
    return 0;
}
