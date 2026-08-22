#!/usr/bin/env python3
"""Regression test: GGUFTensor.unpack() must not permute non-square tensors.

Issue #1760 claimed `unpack()` passes `self.shape[0]` as `columns` to
`unpack_q4_0/1` but should pass `self.shape[-1]`. That claim is backwards:

  * gguf-py's GGUFReader stores `ReaderTensor.shape = dims` (the raw GGUF
    file dims, e.g. [in, out] for a linear weight) while the numpy `data`
    array is reshaped to `reversed(dims)` (e.g. [out, in]).  Therefore
    `shape[0] == data.shape[-1]` is a tautology, and
    `gguf.quantize`/`dequantize` operate along the *last* numpy dim
    (`quant_shape_to_byte_shape` divides `shape[-1]` by the block size).

  * `unpack_q4_0/1(data, columns)` reshapes the per-block scales/values so
    that the result has shape (rows, columns).  For the result to reproduce
    `gguf.dequantize(data)` (and the layout `gguf.quantize` produced),
    `columns` must equal the numpy `data.shape[-1]` — which is exactly
    `self.shape[0]`.

  * Passing `self.shape[-1]` instead yields a transposed (scrambled) result
    for non-square tensors — verified empirically below (the "wrong" case
    reconstructs a (in, out) matrix from a (out, in) tensor, corr ~0).

Run:  python3 tests/test_gguf_tensor_nonsquare.py
Deps: gguf-py, numpy, torch (same as the converter's venv).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from gguf import GGUFWriter, GGUFReader, quantize, dequantize
from gguf.constants import GGMLQuantizationType

from q4nx.gguf_tensor import GGUFTensor

torch.set_num_threads(1)


def build_nonsquare_gguf(path, logical_shape, seed=3):
    """Write a small non-square Q8_0 GGUF using the standard gguf-py writer
    (file dims = reversed numpy shape, exactly like llama.cpp converters)."""
    out, in_ = logical_shape
    assert in_ % 32 == 0
    rng = np.random.default_rng(seed)
    w = (rng.normal(size=logical_shape) * 0.02).astype(np.float32)
    q = quantize(w, GGMLQuantizationType.Q8_0)
    writer = GGUFWriter(path, "llama")
    writer.add_architecture()
    writer.add_block_count(1)
    writer.add_embedding_length(in_)
    writer.add_feed_forward_length(2 * in_)
    writer.add_head_count(8)
    writer.add_head_count_kv(2)
    writer.add_tensor("blk.0.attn_q.weight", q, raw_dtype=GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return w


def reconstruct(d, m, qw, block_size=32):
    d = d.repeat_interleave(block_size, dim=1)
    m = m.repeat_interleave(block_size, dim=1)
    return (d * qw + m).numpy()


def main():
    failures = 0

    # Zaya-like non-square shapes: (out, in)
    for (out, in_) in [(1024, 2048), (2048, 1024)]:
        path = f"/tmp/gguf_nonsquare_{out}_{in_}.gguf"
        w_orig = build_nonsquare_gguf(path, (out, in_))

        reader = GGUFReader(path)
        t = reader.tensors[0]
        # Exactly what ModelConverter._read_gguf_tensors() builds:
        gt = GGUFTensor(
            name=t.name,
            shape=tuple(t.shape.tolist()),
            data=t.data,
            tensor_type=t.tensor_type,
        )
        print(f"--- {gt.name}: ReaderTensor.shape={gt.shape} data.shape={gt.data.shape}")

        # Ground truth: what gguf.quantize over the last dim produced.
        ref = dequantize(
            quantize(w_orig, GGMLQuantizationType.Q4_1), GGMLQuantizationType.Q4_1
        )

        # CURRENT code path (columns = shape[0]).
        d, m, qw = GGUFTensor.unpack_q4_1(
            quantize(w_orig, GGMLQuantizationType.Q4_1).copy(), gt.shape[0]
        )
        rec = reconstruct(d, m, qw)
        ok_shape = tuple(qw.shape) == (out, in_)
        ok_vals = np.allclose(rec, ref, atol=1e-3)
        corr = np.corrcoef(rec.ravel(), w_orig.ravel())[0, 1]
        print(f"  columns=shape[0]={gt.shape[0]} (current): "
              f"shape {tuple(qw.shape)} ok={ok_shape}, matches dequantize={ok_vals}, "
              f"corr vs original={corr:.4f}")
        assert ok_shape, f"wrong unpack shape {tuple(qw.shape)} != {(out, in_)}"
        assert ok_vals, "unpack(shape[0]) does not reproduce gguf.quantize+dequantize"
        assert corr > 0.99, "unpack(shape[0]) does not round-trip the original weights"

        # The issue #1760 proposed fix (columns = shape[-1]) — document that it
        # is WRONG: it transposes/scrambles non-square tensors.  The unpacked
        # matrix comes out as (in, out) instead of (out, in): the same values
        # reinterpreted with the rows/columns swapped, which `_pack_q4nx`
        # (rows, cols = qw.shape, cols == the in-dim) then packs as if the
        # columns were the in-features — i.e. a scrambled Q4NX.
        d2, m2, qw2 = GGUFTensor.unpack_q4_1(
            quantize(w_orig, GGMLQuantizationType.Q4_1).copy(), gt.shape[-1]
        )
        rec2 = reconstruct(d2, m2, qw2)
        wrong_shape = tuple(qw2.shape) == (in_, out)
        # The exact scramble the issue describes: with `columns` = the out dim,
        # blocks are re-grouped (block count per row = out/32 instead of in/32),
        # so each output row pulls half-rows from consecutive matrix rows.
        C = gt.shape[-1]
        cols_per_row = C // 32
        interleaved = np.empty_like(rec2)
        for r2 in range(rec2.shape[0]):
            for c in range(rec2.shape[1]):
                bi = r2 * cols_per_row + c // 32
                src_row = bi // (in_ // 32)
                src_col = (bi % (in_ // 32)) * 32 + (c % 32)
                interleaved[r2, c] = ref[src_row, src_col]
        matches_interleave = np.allclose(rec2, interleaved, atol=1e-3)
        matches_interleave = np.allclose(rec2, interleaved, atol=1e-3)
        matches_ref = (rec2.shape == ref.shape) and np.allclose(rec2, ref, atol=1e-3)
        print(f"  columns=shape[-1]={gt.shape[-1]} (issue #1760 fix): "
              f"shape {tuple(qw2.shape)} (transposed={wrong_shape}), "
              f"== column-pair-interleave={matches_interleave}, "
              f"== reference={matches_ref}")
        assert wrong_shape, "shape[-1] should yield the (in, out) shape"
        assert matches_interleave and not matches_ref, (
            "shape[-1] produces exactly the column-pair-interleaved scramble the "
            "issue describes (w[r][c] -> qs[2r + c//in][c%in]); _pack_q4nx would "
            "emit a scrambled tile. columns=shape[0] is the correct value."
        )
        print("  OK (documents that the #1760 fix is the scramble, not a fix)")

    print("\nPASS: unpack() with columns=shape[0] is correct for non-square tensors.")
    return failures


if __name__ == "__main__":
    sys.exit(main())
