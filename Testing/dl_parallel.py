#!/usr/bin/env python3
"""Parallel HTTP downloader for a single HuggingFace file (range requests).
Usage: dl_parallel.py <url> <outfile> [nchunks]
"""
import os, sys, threading, time, urllib.request

url, out = sys.argv[1], sys.argv[2]
nchunks = int(sys.argv[3]) if len(sys.argv) > 3 else 8

# HEAD for size
req = urllib.request.Request(url, method="HEAD", headers={"User-Agent":"census"})
with urllib.request.urlopen(req, timeout=30) as r:
    total = int(r.headers.get("Content-Length", 0))
print(f"total: {total/1e9:.1f} GB, {nchunks} chunks", flush=True)

chunk = total // nchunks
state = [0]*nchunks
lock = threading.Lock()
start = time.time()

def fetch(i):
    lo = i*chunk
    hi = total-1 if i == nchunks-1 else (i+1)*chunk - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}", "User-Agent":"census"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with open(f"{out}.part{i}", "wb") as f:
        f.write(data)
    state[i] = len(data)
    done = sum(state)
    pct = 100*done/total
    rate = done/(time.time()-start)/1e6
    print(f"  chunk {i}: {len(data)/1e6:.0f}MB  total {pct:.1f}%  {rate:.1f} MB/s", flush=True)

threads = [threading.Thread(target=fetch, args=(i,)) for i in range(nchunks)]
for t in threads: t.start()
for t in threads: t.join()

# assemble
with open(out, "wb") as f:
    for i in range(nchunks):
        with open(f"{out}.part{i}", "rb") as p:
            f.write(p.read())
        os.remove(f"{out}.part{i}")
print(f"DONE {out} ({total/1e9:.1f} GB)", flush=True)
