#!/usr/bin/env python3
"""A/B test: NPU MoE FFN (moe_ffn_npu) vs CPU MoE FFN (moe_ffn_cpu) in the engine.

Drives npu_engine_universal in worker mode (op=32 fused decode step) with
NPU_MOE=1 vs NPU_MOE=0 on the same prompt and compares the greedy token
sequences. NPU is an int8 pipeline (probe: sim-vs-f32 corr 0.91), so a few
divergences are expected; majority agreement + no errors = pass.

Measured on Qwen3.6-35B-A3B (2026-08-04): NPU 193 s/tok vs CPU 165 s/tok — both
dominated by per-token expert dequant, which is why NPU MoE is opt-in.
"""
import struct
import subprocess
import sys
import os

MODEL = "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx"
ENGINE = "/home/bcloud/projects/1bit-systems/engine/npu/build-fresh/npu_engine_universal"
PROMPT = [151644, 872, 198, 13048, 151645, 198, 151644, 77091, 198]  # default seq
STEPS = 4

XCLBIN_DIR = "/home/bcloud/projects/1bit-systems/engine/npu/xclbins"  # overrides stale shell NPU_XCLBIN_DIR


def run(npu_moe: int, timeout=1800):
    env = dict(os.environ)
    env["NPU_MOE"] = str(npu_moe)
    env["NPU_XCLBIN_DIR"] = XCLBIN_DIR
    p = subprocess.Popen([ENGINE, MODEL, "--worker"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    # consume READY handshake
    ready = p.stdout.readline()
    assert ready.strip() == b"READY", f"no READY handshake: {ready!r}"
    while True:  # skip WORKER_READY on stderr, non-blocking
        line = p.stderr.readline()
        if b"WORKER_READY" in line:
            break

    def send(op, layer, batch, in_dim, floats):
        p.stdin.write(struct.pack("<4I", op, layer, batch, in_dim))
        if floats:
            p.stdin.write(struct.pack(f"<{len(floats)}f", *floats))
        p.stdin.flush()
        hdr = p.stdout.read(8)
        assert len(hdr) == 8, f"short response header: {hdr!r}"
        ok, out_dim = struct.unpack("<2I", hdr)
        assert ok == 0, f"op {op} failed (ok={ok})"
        if out_dim == 0:
            return []
        out = struct.unpack(f"<{out_dim}f", p.stdout.read(out_dim * 4))
        return list(out)

    toks = []
    for i, t in enumerate(PROMPT + [0] * (STEPS - len(PROMPT))):
        # first STEPS tokens: feed prompt tokens, emit STEPS next-tokens
        # (no op=31 reset: its batch=0 header fails the worker's input
        # validation, and the fused KV state is fresh on the first op=32 anyway)
        out = send(32, 0, 1, 1, [float(t)])
        toks.append(int(round(out[0])))
    p.stdin.write(struct.pack("<4I", 0, 0, 0, 0))  # QUIT
    p.stdin.flush()
    p.wait(timeout=30)
    return toks


def main():
    print(f"=== A/B: NPU MoE vs CPU MoE ({STEPS} steps) ===", flush=True)
    cpu = run(0)
    print(f"CPU MoE: {cpu}", flush=True)
    npu = run(1)
    print(f"NPU MoE: {npu}", flush=True)
    match = sum(1 for a, b in zip(cpu, npu) if a == b)
    print(f"match: {match}/{STEPS}", flush=True)
    if match >= STEPS // 2:
        print("PASS — NPU MoE path agrees with CPU path", flush=True)
        return 0
    print("FAIL — NPU MoE diverges from CPU MoE", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
