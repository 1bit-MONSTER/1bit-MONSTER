#!/usr/bin/env python3
"""Write a minimal registry fixture that the flag audit's behavioural half can act on.

WHY THIS EXISTS: `registry_flag_audit.py` splits by cost — A (usage parity) and C (cross-TU
references) are source-only, but B (behavioural no-op) and D (repeat safety) need a BUILT
binary AND a fixture with at least one recognized artifact. @agent-ec855d established that
`registry_scan` compiles standalone with plain clang++ and no HIP/XRT, which makes B CI-
feasible on a stock runner; the missing piece was the fixture. Making it a versioned tool
rather than an inline heredoc means the same bytes are used in CI and locally, so a CI-only
failure is real rather than an environment difference.

The audit already guards the degenerate case (a fixture with ZERO artifacts makes B report
phantom failures), so this generator's one job is to be non-empty and to exercise the code
paths B and D touch: a GGUF with a dtype census, a native-container file, a tokenizer cache
in the second on-disk spelling, and a byte-identical duplicate pair.

Usage: registry_fixture.py <dir>
Exit 0 on success; the audit is what interprets the result.
"""
import os
import struct
import sys


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _gguf(path: str, name: str, arch: str, dtypes) -> None:
    """A GGUF header with a tensor table and no payload — enough for the dtype census."""
    out = bytearray(b"GGUF" + struct.pack("<I", 3))
    out += struct.pack("<Q", len(dtypes)) + struct.pack("<Q", 2)
    out += _s("general.name") + struct.pack("<I", 8) + _s(name)
    out += _s("general.architecture") + struct.pack("<I", 8) + _s(arch)
    for i, dt in enumerate(dtypes):
        out += _s("blk.%d.weight" % i)
        out += struct.pack("<I", 2) + struct.pack("<Q", 256) + struct.pack("<Q", 256)
        out += struct.pack("<I", dt) + struct.pack("<Q", i * 1024)
    out += b"\x00" * 2048
    with open(path, "wb") as fh:
        fh.write(bytes(out))


def _gguf_complete(path: str, name: str, arch: str = "qwen3") -> None:
    """A COMPLETE (non-truncated) GGUF whose `general.name` EQUALS its basename.

    Why this exists (@agent-ec855d found the gap): `registry_diff` prints `same-file` and
    `id-divergent`. Every population measured so far has same-file == id-divergent, i.e. every
    legacy entry carries a DIFFERENT canonical id — so the branch that reports a difference was
    never exercised, and a regression of `id-divergent` to a constant 0 would have gone unnoticed
    by every fixture and every store available. This artifact closes that: the registry id for a
    GGUF is its basename, so naming `general.name` exactly that makes the two counters disagree.

    It must be COMPLETE. The other fixtures here are metadata-only and the strict engine reader
    refuses them as truncated ("GGUF truncated: 'blk.2.weight' needs 53760 bytes ... but file is
    2318 bytes"), while the registry's metadata reader accepts them — which is why a flat scan
    reports them as invisible. An artifact meant to be SEEN by the legacy scan has to carry its
    tensor bytes.
    """
    dims = [2, 2]
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    out = bytearray(b"GGUF" + struct.pack("<I", 3))
    out += struct.pack("<Q", 1) + struct.pack("<Q", 2)          # 1 tensor, 2 KVs
    out += _s("general.name") + struct.pack("<I", 8) + _s(name)
    out += _s("general.architecture") + struct.pack("<I", 8) + _s(arch)
    out += _s("blk.0.weight") + struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    out += struct.pack("<I", 0) + struct.pack("<Q", 0)          # GGML_TYPE_F32, offset 0
    out += b"\x00" * ((32 - len(out) % 32) % 32)                 # align to 32
    out += payload
    with open(path, "wb") as fh:
        fh.write(bytes(out))

def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: registry_fixture.py <dir>\n")
        return 2
    root = sys.argv[1]
    os.makedirs(os.path.join(root, "nested"), exist_ok=True)

    # Include token_embd.weight explicitly: the HRX gate reads THAT tensor's dtype, so a
    # fixture without it would never exercise the gate.
    _gguf(os.path.join(root, "fixture-main.gguf"), "Audit Fixture", "qwen3", [0, 12, 14])
    _gguf(os.path.join(root, "nested", "fixture-nested.gguf"), "Nested Fixture", "zaya", [0, 12])

    # The counters' DISAGREEING case: a complete GGUF named so that its legacy id (general.name)
    # equals its registry id (the basename). Without this, same-file == id-divergent in every
    # population and the differing branch is dead code that no fixture can reach.
    _gguf_complete(os.path.join(root, "audit-match.gguf"), "audit-match.gguf")
    # ...and its TWIN, the case where the ids MUST diverge. Both are needed: a fixture with only
    # the matching case still reads id-divergent=0, so a regression of that counter to a constant
    # 0 stays invisible. With both, the expected pair is (same-file=2, id-divergent=1) — non-zero,
    # so the counter has to actually count for the fixture to agree. That is the negative-control
    # rule applied to the counter itself, not just to the tool.
    _gguf_complete(os.path.join(root, "audit-divergent.gguf"), "Something Else Entirely")

    # A native container: magic 1BP\0 + a version, padded to a plausible header size.
    with open(os.path.join(root, "fixture-native.1bp"), "wb") as fh:
        fh.write(struct.pack("<II", 0x00504231, 4) + b"\x00" * 248)

    # A tokenizer cache in the second on-disk spelling (`<stem>.htok`), which is the one the
    # resolution pass has to find.
    with open(os.path.join(root, "fixture-main.htok"), "wb") as fh:
        fh.write(b"fixture-tokenizer")

    # A byte-identical duplicate pair, so `--digest` has something to collapse.
    with open(os.path.join(root, "dup-a.q4nx"), "wb") as fh:
        fh.write(b"\xdf\x8b\x03\x00" + b"D" * 4096)
    with open(os.path.join(root, "dup-b.q4nx"), "wb") as fh:
        fh.write(b"\xdf\x8b\x03\x00" + b"D" * 4096)
    with open(os.path.join(root, "dup-b.htok"), "wb") as fh:
        fh.write(b"fixture-tokenizer")

    print("registry fixture written to %s" % root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
