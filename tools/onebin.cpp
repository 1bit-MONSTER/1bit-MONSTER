// onebin.cpp — the single binary. Everything in one ELF, busybox-style.
//
// ONE BINARY. Every entry point dispatches on argv[0] (symlink names) or a
// subcommand, so `1bit zaya`, `1bit unified`, `1bit router`, `1bit lemonade`,
// and the legacy names `zaya_server` / `unified_server` / `unified_router`
// (symlinks to this binary) all work without changing a single exec/pkill
// site in the tools.
//
// Subcommands:
//   zaya              → zaya_server (HIP/NPU/CPU multi-backend server)
//   unified           → unified_server (multi-backend + embedded Lemonade core)
//   router            → unified_router (NPU/GPU policy routing proxy)
//   lemonade          → unified_server --lemonade (Lemonade's full server)
//   jarvis, voice     → jarvis_app (clean-slate voice assistant, pure C++)
//   vision, vl        → vision_server (vision-language server)
//   zuna              → zuna_port

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int zaya_server_main(int argc, char** argv);
int unified_server_main(int argc, char** argv);
int unified_router_main(int argc, char *argv[]);
int onebitd_main(int argc, char *argv[]);
int onebit_main(int argc, char *argv[]);
int jarvis_app_main(int argc, char** argv);
int vision_server_main(int argc, char** argv);
int zuna_main(int argc, char** argv);

static std::string prog_name(const char* argv0) {
    std::string p = argv0 ? argv0 : "1bit";
    auto slash = p.find_last_of('/');
    if (slash != std::string::npos) p = p.substr(slash + 1);
    return p;
}

static int print_usage() {
    std::fprintf(stderr,
        "usage: 1bit <subcommand> [args...]\n\n"
        "subcommands:\n"
        "  zaya              multi-backend inference server\n"
        "  unified           multi-backend server + embedded Lemonade core\n"
        "  router            NPU/GPU policy routing proxy\n"
        "  lemonade          Lemonade-compatible server\n"
        "  jarvis|voice|tts  voice pipeline server\n"
        "  vision|vl         vision-language server\n"
        "  zuna              zuna_port\n");
    return 1;
}

int main(int argc, char** argv) {
    // ── argv[0] dispatch (symlink names) ──
    std::string prog = prog_name(argv[0]);
    if (prog == "zaya_server")    return zaya_server_main(argc, argv);
    if (prog == "unified_server") return unified_server_main(argc, argv);
    if (prog == "unified_router") return unified_router_main(argc, argv);
    if (prog == "onebitd")        return onebitd_main(argc, argv);
    if (prog == "jarvis_server")  return jarvis_app_main(argc, argv);  // legacy symlink → JARVIS v2
    if (prog == "vision_server")  return vision_server_main(argc, argv);

    // ── Subcommand dispatch ──
    if (argc > 1) {
        std::string cmd = argv[1];
        if (cmd == "zaya" || cmd == "zaya-server" || cmd == "server") {
            return zaya_server_main(argc - 1, argv + 1);
        }
        if (cmd == "unified" || cmd == "unified-server") {
            return unified_server_main(argc - 1, argv + 1);
        }
        if (cmd == "router" || cmd == "unified-router") {
            return unified_router_main(argc - 1, argv + 1);
        }
        if (cmd == "lemonade" || cmd == "lemond") {
            // Inject --lemonade; unified_server_main hands off to Lemonade's
            // server core with the rest of the args.
            static std::vector<char*> args;  // kept alive for the call
            args.clear();
            args.push_back(argv[0]);
            args.push_back(const_cast<char*>("--lemonade"));
            for (int i = 2; i < argc; ++i) args.push_back(argv[i]);
            return unified_server_main(static_cast<int>(args.size()), args.data());
        }
        if (cmd == "onebitd" || cmd == "daemon") {
            return onebitd_main(argc - 1, argv + 1);
        }
        if (cmd == "jarvis" || cmd == "voice") {
            return jarvis_app_main(argc - 1, argv + 1);
        }
        if (cmd == "vision" || cmd == "vl") {
            return vision_server_main(argc - 1, argv + 1);
        }
        if (cmd == "zuna") {
            return zuna_main(argc - 1, argv + 1);
        }
        if (cmd == "-h" || cmd == "--help" || cmd == "help") {
            return print_usage();
        }
    }
    return print_usage();
}
