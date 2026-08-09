#!/usr/bin/env python3
"""MoE expert worker ops (40/41) test + latency benchmark.

Requires a MoE q4nx model (Qwen3.6-35B-A3B-NPU2) and the qwen3_6_moe_35b
engine variant. Payload layout for op=40/41:
  header {op, layer, 1, in_dim}, batch*in_dim floats, u32 k, k*u32 expert ids
Responses: op=40 -> k*2*IM_EXP floats; op=41 -> H floats.
"""
import struct, subprocess, sys, os, select, time

H, IM_EXP = 2048, 512          # Qwen3.6-35B-A3B dims
N_EXPERTS, TOP_K = 256, 8

class Streams:
    def __init__(self, p):
        self.p = p
        self.out = b""
        self.err = b""
        self.out_fd = p.stdout.fileno()
        self.err_fd = p.stderr.fileno()

    def pump(self, timeout):
        end = time.time() + timeout
        got = False
        while time.time() < end and not got:
            r, _, _ = select.select([self.out_fd, self.err_fd], [], [], 0.5)
            for fd in r:
                d = os.read(fd, 4096)
                if not d:
                    raise EOFError(f"pipe closed; err_tail={self.err[-300:]!r}")
                got = True
                if fd == self.out_fd:
                    self.out += d
                else:
                    self.err += d
            if self.p.poll() is not None:
                raise SystemExit(f"engine exited rc={self.p.returncode} err_tail={self.err[-500:]!r}")
        return got

    def wait_ready(self, timeout=600):
        while b"READY\n" not in self.out:
            if not self.pump(timeout):
                raise TimeoutError("no READY")
        self.out = self.out.split(b"READY\n", 1)[1]
        print("READY")

    def call(self, op, layer, k, vec, in_dim, timeout=120):
        ids = struct.pack(f"<{k}I", *range(k))
        self.p.stdin.write(struct.pack("<4I", op, layer, 1, in_dim) + vec + struct.pack("<I", k) + ids)
        self.p.stdin.flush()
        t0 = time.time()
        end = time.time() + timeout
        while len(self.out) < 8:
            if not self.pump(timeout):
                raise TimeoutError("no response header")
        status, od = struct.unpack("<2I", self.out[:8])
        self.out = self.out[8:]
        n = od * 4
        while len(self.out) < n:
            if not self.pump(timeout):
                raise TimeoutError("short response body")
        data, self.out = self.out[:n], self.out[n:]
        return status, od, struct.unpack(f"<{od}f", data), time.time() - t0

def main():
    engine, model = sys.argv[1], sys.argv[2]
    p = subprocess.Popen([engine, model, "--worker"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    s = Streams(p)
    s.wait_ready()

    rng = [0.1 * (i % 7) - 0.3 for i in range(H)]
    x = struct.pack(f"<{H}f", *rng)

    # op=40: GU for k experts (k=1,2,4,8), cold cache first, then warm
    for k in (1, 2, 4, 8):
        st, od, vals, dt = s.call(40, 0, k, x, H)
        assert st == 0, f"op40 k={k}: status {st}"
        assert od == k * 2 * IM_EXP, f"op40 k={k}: od {od}"
        assert any(abs(v) > 0 for v in vals), f"op40 k={k}: all zeros"
        ts = []
        for _ in range(3):
            st, od, vals, dt = s.call(40, 0, k, x, H)
            assert st == 0
            ts.append(dt)
        print(f"op40 GU k={k:2d}: {min(ts)*1000:7.1f} ms/call (warm)")

    # op=41: D with su[k*IM_EXP] (caller-side SiLU*gate*prob inputs)
    k = 8
    su = struct.pack(f"<{k*IM_EXP}f", *[0.05 * (i % 11) - 0.25 for i in range(k * IM_EXP)])
    st, od, vals, dt = s.call(41, 0, k, su, k * IM_EXP)
    assert st == 0, f"op41: status {st}"
    assert od == H, f"op41: od {od}"
    assert any(abs(v) > 0 for v in vals), "op41 all zeros"
    ts = []
    for _ in range(3):
        st, od, vals, dt = s.call(41, 0, k, su, k * IM_EXP)
        assert st == 0
        ts.append(dt)
    print(f"op41 D  k=8: {min(ts)*1000:7.1f} ms/call (warm)")

    # sanity: different expert sets must give different outputs
    st1, od1, v1, _ = s.call(40, 0, 1, x, H)
    st2, od2, v2, _ = s.call(40, 0, 1, x, H)  # expert id 0 again (cache warm)
    assert v1 == v2, "same expert, different output (cache inconsistency)"
    print("cache determinism ok")
    ids = struct.pack("<I", 1)
    s.p.stdin.write(struct.pack("<4I", 40, 0, 1, H) + x + struct.pack("<I", 1) + ids)
    s.p.stdin.flush()
    while len(s.out) < 8:
        s.pump(30)
    status, od = struct.unpack("<2I", s.out[:8])
    s.out = s.out[8:]
    n = od * 4
    while len(s.out) < n:
        s.pump(30)
    v3, s.out = struct.unpack(f"<{od}f", s.out[:n]), s.out[n:]
    assert v1 != v3, "expert 0 and expert 1 outputs identical"
    print("different experts -> different outputs ok")

    # QUIT
    s.p.stdin.write(struct.pack("<4I", 0, 0, 0, 0)); s.p.stdin.flush()
    s.p.wait(timeout=10)
    print("QUIT ok")

if __name__ == "__main__":
    main()
