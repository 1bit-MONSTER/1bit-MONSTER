#!/bin/bash
# Build the HRX verification harnesses against the local hrx-v2 fork build.
set -e
FORK=/home/bcloud/hrx-ws/hrx-v2-src
g++ -O2 -o dump_hrx_logits dump_hrx_logits.cpp \
  -I"$FORK/include" -I"$FORK/ggml/include" \
  -L"$FORK/build/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$FORK/build/bin"
g++ -O2 -o gtok_hrx gtok_hrx.cpp \
  -I"$FORK/include" -I"$FORK/ggml/include" \
  -L"$FORK/build/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$FORK/build/bin"
echo "built: $(pwd)/dump_hrx_logits $(pwd)/gtok_hrx"
