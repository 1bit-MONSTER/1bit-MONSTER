# Zaya AI Co-Host — OBS Plugin

## Quick Start
1. Make sure jarvis_server is running: `./build/jarvis_server`
2. In OBS, add a new Browser Source
3. URL: `http://localhost:8080/obs-plugin/?voice=zaya_default&persona=Zaya`
4. Width: 1920, Height: 1080
5. Done! Your AI co-host is live.

## URL Parameters
- `server` — WebSocket server (default: localhost:8082)
- `voice` — voice pack name
- `persona` — persona name (default: Zaya)
- `volume` — audio volume 0.0-1.0 (default: 1.0)

## Files
- `index.html` — Single-file OBS Browser Source (777 lines, 24KB)
- `README.md` — This file

## Build
No build step. Just serve via jarvis_server at /obs-plugin/.
