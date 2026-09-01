#!/bin/bash
# run_real_runtime.sh — run the REAL FastFlowLM Qwen3-0.6B model on the NPU.
# Works on kernel 7.2.0-perfopt (identity IOMMU + SVA) with the pristine
# amdxdna driver. Verified 2026-08-31: prefill 27.8 tok/s, decode 96.6 tok/s,
# zero IO_PAGE_FAULTs / TDRs.
set -e
FLM=/home/bcloud/1bit-MONSTER/third_party/FastFlowLM
cd "$FLM/src/build/test/qwen3_npu"
export LD_LIBRARY_PATH="$FLM/src/lib/xrt:$FLM/src/build/tokenizers-cpp:$FLM/src/build/tokenizers-cpp/sentencepiece/src"
exec ./test_qwen3_npu --model qwen3:0.6b -s 1
