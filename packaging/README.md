# Packaging — 1bit.MONSTER v2026.08.04

**One binary. 47 1BP models. Auto-detect.** Zero Python. Zero pip. No Docker required.
The HTTP server speaks OpenAI-compatible JSON — Ollama, Open WebUI, LangChain, anything that hits `/v1/chat/completions` just works.

| Format | Status | Command |
|--------|--------|---------|
| **Website downloads** | ✅ [1bit.monster/downloads](https://1bit.monster/1bit-downloads.html) | tarball, `.deb`, AppImage — hosted on the site, updated by `make package-site` |
| **One-liner install** | ✅ | `curl -sL https://1bit.monster/install.sh \| bash` |
| **Debian (.deb)** | ✅ | `sudo dpkg -i 1bit-monster_*_amd64.deb` (download from the website) |
| **AppImage** | ✅ | `chmod +x 1bit-monster-*.AppImage && ./1bit-monster-*.AppImage` (download from the website) |
| **Binary tarball** | ✅ | `make package-tarball` — the website hosts the `.tar.xz` build |
| **GitHub Releases** | 📋 attached when a `v*` tag is pushed | `gh release download` |
| **Docker** | ✅ Dockerfile ready | `docker run 1bit-monster/npu` |
| **Ollama** | ✅ Modelfile | `ollama create qwen3-npu -f Modelfile` |
| **OpenAI SDK** | ✅ Drop-in | `client = OpenAI(base_url="http://localhost:8081/v1")` |
| **Open WebUI** | ✅ Compatible | Point `OPENAI_API_BASE` at the NPU server |
| **LangChain** | ✅ Compatible | `ChatOpenAI(openai_api_base="http://localhost:8081/v1")` |
| **Arch (AUR)** | 📋 PKGBUILD ready | `yay -S 1bit-monster-bin` |
| **Homebrew** | 📋 Formula ready | `brew install 1bit-monster` |
| **Snap** | 📋 snapcraft.yaml ready | `snap install 1bit-monster` |

### Model coverage

Auto-detects **19 model architectures** from GGUF/1BP headers, **47 1BP models** — Qwen2/3/3.5, Llama 3.1/3.2, Mistral/Pixtral, Gemma 3/4, Falcon, DeepSeek V2/V3/R1, Zaya1 MoE, BlackMamba, Zamba/Zamba2, Kimi (Gated MLA MoE), and more. Per-model support matrix and performance data: [`docs/wiki/models.md`](../docs/wiki/models.md).

### Client Compatibility (same HTTP API, no SDK needed)

| Client | Integration | Effort |
|--------|-----------|--------|
| **Ollama** | `ollama create qwen3-npu -f Modelfile` | 1 command |
| **OpenAI Python** | `OpenAI(base_url="http://localhost:8081/v1")` | 1 line |
| **OpenAI JS** | `new OpenAI({baseURL: "http://localhost:8081/v1"})` | 1 line |
| **Open WebUI** | Set `OPENAI_API_BASE` env var | 1 env var |
| **LangChain** | `ChatOpenAI(openai_api_base=...)` | 1 param |
| **LlamaIndex** | `OpenAI(api_base=...)` | 1 param |
| **curl** | `curl -d '{"messages":[...]}' localhost:8081/v1/chat/completions` | 0 deps |
| **Anything with HTTP** | POST JSON → get JSON back | Universal |

## Included in every package

| Binary | Purpose | Size |
|--------|---------|------|
| `1bit` | Single ELF — every server + CLI (zaya_server, unified_server, unified_router, jarvis_server, vision_server, onebit, onebitd, 1bit-server; legacy names are symlinks, argv[0] dispatch) | ~67 MB raw / ~64 MB stripped |
| `1bit-npu` | CLI inference engine (47 1BP models, auto-detect; NPU engine sidecar, needs XRT) | ~2.1 MB |
| `video_lora_vk_cli` | Video-LoRA Vulkan CLI (dev tool, optional sidecar) | — |

## Build them yourself

```bash
# Binary tarball
make package-tarball

# Debian package
make package-deb

# Website packages — sync tarball (.tar.xz) + .deb + AppImage into
# site/downloads/ and regenerate SHA256SUMS + manifest.json (deployed to
# https://1bit.monster/1bit-downloads.html by deploy.yml)
make package-site

# Docker image
docker build -t 1bit-monster/npu:2026.08.04 -f packaging/docker/Dockerfile .
docker run --device /dev/accel/accel0 -p 8081:8081 1bit-monster/npu:2026.08.04

# Snap
make package-snap

# RPM (needs rpmbuild: Fedora `dnf install rpm-build`, Ubuntu `apt install rpm`)
make package-rpm

# Flatpak (needs flatpak + flatpak-builder; installs the Freedesktop runtime --user)
make package-flatpak

# AUR
cd packaging/aur && makepkg -si
```
