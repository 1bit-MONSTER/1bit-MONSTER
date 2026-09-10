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
