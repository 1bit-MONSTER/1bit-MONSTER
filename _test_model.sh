#!/bin/bash
cd /home/bcloud/1bit-systems
./build/zaya_server --model models/ZAYA1-8B.h1b --port 18094 &
PID=$!
sleep 5
for i in $(seq 1 30); do
  MODELS=$(curl -s --max-time 2 http://localhost:18094/v1/models 2>/dev/null)
  if echo "$MODELS" | grep -q 'ZAYA'; then
    echo "=== MODEL LOADED after ${i} checks ==="
    echo "=== SENDING INFERENCE REQUEST ==="
    curl -s --max-time 60 -X POST http://localhost:18094/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"ZAYA1-8B","messages":[{"role":"user","content":"Hello"}],"max_tokens":5}' 2>&1
    echo ""
    echo "=== DONE ==="
    break
  fi
  sleep 2
done
kill $PID 2>/dev/null
wait $PID 2>/dev/null
