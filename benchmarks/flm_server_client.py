#!/usr/bin/env python3
"""FLM server client: one chat completion, timed. Prints JSON with wall/ttft/decode."""
import json, sys, time, urllib.request

port, model, prompt, maxtok = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": maxtok, "temperature": 0, "stream": False}).encode()
t0 = time.time()
try:
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                              data=body, headers={"Content-Type": "application/json"}), timeout=600)
    j = json.load(r)
    wall = time.time() - t0
    u = j.get("usage", {}) or {}
    ct = u.get("completion_tokens") or 0
    ttft = u.get("prefill_duration_ttft") or u.get("ttft") or None
    dec = (ct - 1) / (wall - ttft) if (ttft and ct > 1 and wall > ttft) else None
    print(json.dumps({"port": port, "model": model, "wall": round(wall, 3), "completion_tokens": ct,
                      "ttft": ttft, "decode_tps": round(dec, 2) if dec else None,
                      "usage_keys": sorted(u.keys()),
                      "usage": {k: u[k] for k in sorted(u) if isinstance(u[k], (int, float))}}))
except Exception as e:
    print(json.dumps({"port": port, "model": model, "error": repr(e)}))
