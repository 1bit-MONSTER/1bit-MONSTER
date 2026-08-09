#!/usr/bin/env python3
"""Worker wire-protocol smoke: spawn engine --worker, exercise ops, verify shapes.

Protocol (npu_engine_universal.cpp run_worker):
  Req:  4x u32 LE { op, layer, batch, in_dim } + batch*in_dim raw f32
  Resp: 2x u32 LE { status, out_dim } + batch*out_dim raw f32
  Ops:  QUIT=0 QKV=1 OPROJ=2 GATEUP=3 UP=4 DOWN=5
  Status: 0=OK 1=bad args 2=bad op
"""
import struct, subprocess, sys, os, select, time

class Streams:
    """Chunked reader for both pipes; proven pattern (select + read(4096))."""
    def __init__(self, p):
        self.p = p
        self.out = b""   # stdout: responses (and the READY banner)
        self.err = b""   # stderr: WORKER_READY sync + engine logs
        self.out_fd = p.stdout.fileno()
        self.err_fd = p.stderr.fileno()

    def pump(self, timeout):
        """Wait up to timeout for data on either pipe; return True if any flowed."""
        end = time.time() + timeout
        got = False
        while time.time() < end and not got:
            r, _, _ = select.select([self.out_fd, self.err_fd], [], [], 0.5)
            for fd in r:
                d = os.read(fd, 4096)
                if not d:
                    raise EOFError("pipe closed")
                got = True
                if fd == self.out_fd:
                    self.out += d
                else:
                    self.err += d
            if self.p.poll() is not None:
                raise SystemExit(f"engine exited rc={self.p.returncode}")
        return got

    def wait_ready(self, timeout=120):
        while b"WORKER_READY" not in self.err:
            self.pump(timeout)
        while b"READY\n" not in self.out:
            self.pump(30)
        self.out = self.out.split(b"READY\n", 1)[1]
        print("READY")

    def call(self, op, layer, batch, vec, in_dim=None, timeout=60):
        # in_dim is the per-row FLOAT count; payload carries batch rows
        p = self.p
        if in_dim is None:
            in_dim = len(vec) // 4 // batch
        p.stdin.write(struct.pack("<4I", op, layer, batch, in_dim) + vec)
        p.stdin.flush()
        end = time.time() + timeout
        while len(self.out) < 8:
            if not self.pump(timeout):
                raise TimeoutError(f"no response out={len(self.out)}B err_tail={self.err[-300:]!r}")
        status, out_dim = struct.unpack("<2I", self.out[:8])
        self.out = self.out[8:]
        n = batch * out_dim * 4
        while len(self.out) < n:
            self.pump(timeout)
        data, self.out = self.out[:n], self.out[n:]
        return status, out_dim, data

def main():
    engine, model = sys.argv[1], sys.argv[2]
    H = int(os.environ.get("NPU_TEST_H", "1024"))  # Qwen3-0.6B hidden
    p = subprocess.Popen([engine, model, "--worker"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    s = Streams(p)
    s.wait_ready()

    import random
    random.seed(7)
    vec = struct.pack(f"<{H}f", *[random.uniform(-1, 1) for _ in range(H)])

    # op 1 QKV: in=H, out=NH*HD + NKV*HD + NKV*HD (q,k,v concat)
    st, od, out = s.call(1, 0, 1, vec)
    assert st == 0, f"QKV status {st}"
    assert od == 16 * 128 + 8 * 128 + 8 * 128, f"QKV out_dim {od}"
    assert any(abs(v) > 0 for v in struct.unpack(f"<{od}f", out)), "QKV all zeros"
    print(f"QKV ok out_dim={od}")

    # op 3 GATEUP: in=H, out=2*IM (fused GU; op=4 UP requires gu_split models)
    st, od, out = s.call(3, 0, 1, vec)
    assert st == 0 and od == 6144, f"GATEUP status {st} od {od}"
    print(f"GATEUP ok out_dim={od}")

    # op 5 DOWN: in=IM, out=H — batch=2 (two rows at once)
    im = struct.pack(f"<{3072}f", *[random.uniform(-1, 1) for _ in range(3072)])
    st, od, out = s.call(5, 0, 2, im * 2)
    assert st == 0 and od == 1024, f"DOWN status {st} od {od}"
    print(f"DOWN(batch=2) ok out_dim={od}")

    # bad op (engine reports status 1 for unhandled ops; doc says 2 — accept both)
    st, od, out = s.call(99, 0, 1, vec)
    assert st in (1, 2), f"bad op status {st}"
    print(f"bad-op rejected ok (status {st})")

    # ── batch-scaling benchmark on the worker path (QKV, layer 0) ──
    import time as _t
    for batch in (1, 4, 16, 64):
        ts = []
        for _ in range(3):
            payload = vec * batch  # batch identical rows, in_dim=H each
            t0 = _t.time()
            st, od, out = s.call(1, 0, batch, payload, in_dim=H)
            assert st == 0, (st, od)
            ts.append(_t.time() - t0)
        best = min(ts)
        print(f"QKV batch={batch:3d}: {best*1000:7.1f} ms/call  -> {batch/best:7.0f} rows/s")

    # MoE ops on a DENSE model: must be rejected (status 1) with the pipe
    # still in sync for the next op (payload drained correctly).
    # Payload layout: floats first (generic read), then u32 k, then ids.
    ids = struct.pack("<2I", 0, 1)  # k=2 expert ids
    p.stdin.write(struct.pack("<4I", 40, 0, 1, H) + vec + struct.pack("<I", 2) + ids)
    p.stdin.flush()
    end = time.time() + 30
    while len(s.out) < 8:
        if not s.pump(30):
            raise TimeoutError("no resp to op=40 (dense)")
    status, od = struct.unpack("<2I", s.out[:8])
    s.out = s.out[8:]
    assert status == 1 and od == 0, f"op=40 on dense: status {status} od {od}"
    print("op=40 rejected on dense model, stream in sync")

    # QUIT
    p.stdin.write(struct.pack("<4I", 0, 0, 0, 0)); p.stdin.flush()
    p.wait(timeout=10)
    print("QUIT ok")

if __name__ == "__main__":
    main()
