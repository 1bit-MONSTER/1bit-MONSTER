#!/usr/bin/env python3
"""Compare the 32-token prefill logits: Q4NX-HRX vs F32-CPU vs Q4NX-twin-CPU."""
import numpy as np
import glob
import os

BINS = {
    "q4nx_hrx (dump32_logits.bin)": "/tmp/dump32_logits.bin",
    "old 'F32 ref' (f32_32.bin)": "/tmp/f32_32.bin",
    "f32 regenerated (current build)": "/tmp/f32_32_regen.bin",
}


def load(p):
    return np.fromfile(p, dtype=np.float32)


def report(name, a, b):
    c = np.corrcoef(a, b)[0, 1]
    md = np.abs(a - b).max()
    me = np.abs(a - b).mean()
    print(f"  {name:44s} corr={c:.10f}  maxdiff={md:.3e}  meandiff={me:.3e}")


def main():
    files = dict(BINS)
    twin = "/tmp/twin32.bin"
    files["q4nx_twin (twin32.bin)"] = twin
    for p in glob.glob("/tmp/dump32*.bin"):
        files.setdefault(f"q4nx_hrx ({p.split('/')[-1]})", p)
    if not glob.glob(twin):
        print(f"missing {twin} — run: /tmp/dump32 /home/bcloud/zaya-q4nx-f32twin.gguf 0 {twin}")
        return 1
    # tolerate missing stale references (e.g. /tmp/f32_32.bin from old rounds)
    data = {}
    for k, v in list(files.items()):
        if os.path.exists(v):
            data[k] = load(v)
        else:
            print(f"skip missing {v}")
    files = {k: v for k, v in files.items() if k in data}
    print("=== all-vs-all (32-token logits) ===")
    keys = list(data)
    for i, k in enumerate(keys):
        for j in range(i + 1, len(keys)):
            report(f"{k} vs {keys[j]}", data[k], data[keys[j]])
    print("\n=== target: corr(q4nx_hrx, twin) should be 1.0 if execution is exact ===")
    report("q4nx_hrx vs twin", data["q4nx_hrx (dump32_logits.bin)"], data["q4nx_twin (twin32.bin)"])
    return 0


if __name__ == "__main__":
    main()
