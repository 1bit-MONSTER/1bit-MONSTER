"""Vendored copy of prism-ml/Ternary-Bonsai-2-27B-mlx-2bit runtime/codec.py.

Source: https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/blob/main/runtime/codec.py
License: Apache-2.0 (see the model card). Copied verbatim 2026-09-18 so the
oracle does not depend on a downloaded pack; do not edit — re-vendor instead.
"""

"""Lossless PQ2_0/PTQ1_0 to MLX affine 2-bit block transcoding.

Preserves folded weights. Runtime Hadamard metadata remains mandatory; this
module alone does not produce a loadable or validated model.
"""

import numpy as np


def transcode(raw: bytes, shape: tuple[int, int], source: str):
    """Return packed uint32 weights and FP16 affine scales/biases, group size 128."""
    rows, width = shape
    if rows <= 0 or width <= 0 or width % 128:
        raise ValueError(
            "Expected positive [output, input] shape with input divisible by 128"
        )
    sizes = {"PQ2_0": 34, "PTQ1_0": 28}
    if source not in sizes:
        raise ValueError(f"Unsupported format: {source}")
    blocks = rows * width // 128
    if len(raw) != blocks * sizes[source]:
        raise ValueError("Raw byte length does not match shape and format")
    data = np.frombuffer(raw, dtype=np.uint8).reshape(blocks, sizes[source])
    scale_bytes = data[:, :2] if source == "PQ2_0" else data[:, 26:28]
    scales = scale_bytes.copy().view("<f2").reshape(rows, width // 128)
    if not np.isfinite(scales).all():
        raise ValueError("Non-finite quantization scale")
    if source == "PQ2_0":
        words = data[:, 2:].copy().view("<u4").reshape(rows, width // 16)
        return words, np.ascontiguousarray(scales), np.ascontiguousarray(-scales)
    else:
        pieces = []
        # GGML PTQ stages consume 16 bytes, then 8, then the two-byte tail.
        for lo, hi, count in [(0, 16, 5), (16, 24, 5), (24, 26, 4)]:
            packed = data[:, lo:hi].astype(np.uint16)
            for trit in range(count):
                remainder = (packed * (3**trit)) & 255
                pieces.append(((remainder * 3) >> 8).astype(np.uint8))
        codes = np.concatenate(pieces, axis=1)
    words = np.bitwise_or.reduce(
        codes.astype(np.uint32).reshape(rows, width // 16, 16)
        << (2 * np.arange(16, dtype=np.uint32)),
        axis=-1,
    ).astype("<u4")
    return (
        np.ascontiguousarray(words),
        np.ascontiguousarray(scales),
        np.ascontiguousarray(-scales),
    )


def unpack(weight, scales, biases):
    """Independent affine reconstruction in float32 for packing validation."""
    rows, words = weight.shape
    values = np.empty((rows, words * 16), dtype=np.float32)
    for lane in range(16):
        values[:, lane::16] = (weight >> (2 * lane)) & 3
    groups = values.reshape(rows, -1, 128)
    return (
        groups * scales.astype(np.float32)[..., None]
        + biases.astype(np.float32)[..., None]
    ).reshape(rows, -1)
