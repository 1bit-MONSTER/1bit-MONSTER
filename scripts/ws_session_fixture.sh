#!/usr/bin/env bash
# Generates a "speech" fixture as PCM16 @ 16k: 1 s of 440 Hz tone
# followed by 1 s of silence (the VAD completes an utterance only after
# ~500 ms of trailing silence, so a pure tone would never fire one).
# Uses ffmpeg when available; falls back to a stdlib python3 sine
# generator (some dev boxes lack an ffmpeg binary).
set -euo pipefail
OUT="${1:-/tmp/jarvis_fixture.pcm16}"

if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -loglevel error -f lavfi -i "sine=frequency=440:duration=1" -af apad -t 2 -ar 16000 -ac 1 -f s16le "$OUT"
else
    python3 - "$OUT" <<'PY'
import math, struct, sys
out = sys.argv[1]
n_tone, n_total = 16000, 32000
with open(out, "wb") as f:
    for i in range(n_total):
        if i < n_tone:
            v = int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / 16000))
        else:
            v = 0
        f.write(struct.pack("<h", v))
PY
fi
echo "$OUT"
