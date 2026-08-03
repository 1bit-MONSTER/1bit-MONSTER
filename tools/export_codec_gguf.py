#! /usr/bin/env python3
"""export_codec_gguf.py — export the zaya_audio RVQ-VAE codec weights to GGUF.

Phase 1 deliverable for issue #1368: a C++ codec decoder for embedded /
llama.cpp-style deployment.  This script converts a trained codec checkpoint
(or a freshly-initialised codec, ``--random``) into the GGUF file the C++
``ZayaCodecDecoder`` (src/zaya_codec.cpp) loads.

The decoder weights written are exactly the ones used by the ONNX decoder
export (zaya_audio/export_onnx.py): codebook embeddings, post_vq, dec_proj,
the 5 FiLM-conditioned decoder blocks, and post_conv.  Structural
hyperparameters are stored as ``zaya_codec.*`` KV metadata (the C++ side
derives channel counts from tensor shapes, so only the time-domain structure
needs explicit KVs: decoder strides + output paddings).

Usage
-----
.. code-block:: bash

    # Real checkpoint (train_codec.py format: {"model_state_dict": ...})
    python3 tools/export_codec_gguf.py \\
        --checkpoint models/codec_final.pt --out models/codec.gguf

    # Untrained codec — for the C++ self-check without a checkpoint
    python3 tools/export_codec_gguf.py --random --out /tmp/codec_test.gguf

Sidecars written next to the GGUF (for the C++ self-check):
``<out>.tokens.bin`` (int32, 8×T), ``<out>.speaker_emb.bin`` (f32),
``<out>.ref_output.bin`` (f32 reference PCM from PyTorch).

Config knobs (decoder_strides / decoder_output_paddings / ...) default to
``AudioCodecConfig``; override any field with ``--config config.json``.
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ─── GGUF v3 writer (repo convention: tools/whisper_ggml_to_gguf.cpp,
#     tools/convert_gguf_to_h1b.cpp — F32 tensors, 32-byte aligned) ─────────


def _write_string(f, s: str) -> None:
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def _write_kv(f, key: str, value) -> None:
    """value: int (u32), str, or list[str] (string array)."""
    _write_string(f, key)
    if isinstance(value, int):
        f.write(struct.pack("<I", 4))  # GGUF_TYPE_UINT32
        f.write(struct.pack("<I", value))
    elif isinstance(value, str):
        f.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING
        _write_string(f, value)
    elif isinstance(value, list):
        f.write(struct.pack("<I", 9))  # GGUF_TYPE_ARRAY
        f.write(struct.pack("<I", 8))  # element type: string
        f.write(struct.pack("<Q", len(value)))
        for v in value:
            _write_string(f, v)
    else:
        raise TypeError(f"unsupported KV value type for {key}")


def write_gguf(path: str, kv: List[Tuple[str, object]],
               tensors: List[Tuple[str, List[int], bytes]]) -> None:
    """Write a GGUF v3 file.

    tensors: (name, gguf_shape, f32 bytes) — gguf_shape is the PyTorch dim
    order reversed (shape[0] = fastest-varying, matching GgufReader).
    """
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(kv)))
        for key, value in kv:
            _write_kv(f, key, value)
        offset = 0
        for name, shape, data in tensors:
            _write_string(f, name)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", 0))  # GGUF_TYPE_F32
            f.write(struct.pack("<Q", offset))
            offset += len(data)
        # 32-byte alignment before the data section (repo convention)
        rem = f.tell() % 32
        if rem:
            f.write(b"\0" * (32 - rem))
        for _, _, data in tensors:
            f.write(data)


# ─── Model loading ──────────────────────────────────────────────────────────


def _load_codec(args) -> "torch.nn.Module":
    import torch
    from zaya_audio.codec import RVQVAE
    from zaya_audio.config import AudioCodecConfig, DEFAULT_CONFIG

    config = DEFAULT_CONFIG
    if args.config:
        overrides = json.loads(Path(args.config).read_text())
        config = AudioCodecConfig(**{**config.__dict__, **overrides})

    codec = RVQVAE(config)
    if not args.random:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        if not isinstance(state, dict) or not any(
            isinstance(v, torch.Tensor) for v in state.values()
        ):
            sys.exit(f"{args.checkpoint}: not a checkpoint/state dict")
        own = codec.state_dict()
        filtered = {k: v for k, v in state.items() if k in own}
        if not filtered:
            sys.exit(f"{args.checkpoint}: no matching codec keys found")
        missing = set(own) - set(filtered)
        if missing:
            print(f"WARNING: {len(missing)} codec keys absent from checkpoint "
                  f"(e.g. {sorted(missing)[0]})", file=sys.stderr)
        codec.load_state_dict(filtered, strict=True)
    codec.float().eval()
    return codec


def _tensor_bytes(t: "torch.Tensor") -> bytes:
    return t.detach().contiguous().cpu().numpy().astype("<f4").tobytes()


def _collect_tensors(codec) -> List[Tuple[str, List[int], bytes]]:
    """(name, gguf_shape, f32 bytes) for every decoder weight."""
    import torch

    sd = codec.state_dict()
    out: List[Tuple[str, List[int], bytes]] = []

    def add(key: str) -> None:
        t = sd[key]
        assert t.dtype == torch.float32, f"{key} not float32 after .float()"
        gguf_shape = [int(d) for d in reversed(t.shape)]
        out.append((key, gguf_shape, _tensor_bytes(t)))

    add("res_vq.embed")
    add("post_vq.weight"); add("post_vq.bias")
    add("dec_proj.weight"); add("dec_proj.bias")
    n_blocks = len(codec.config.decoder_strides)
    for i in range(n_blocks):
        p = f"decoder.{i}"
        add(f"{p}.film.proj.weight"); add(f"{p}.film.proj.bias")
        add(f"{p}.conv_transpose.weight"); add(f"{p}.conv_transpose.bias")
        for j in range(codec.config.n_res_blocks):
            rp = f"{p}.res_blocks.{j}.net."
            add(f"{rp}0.weight"); add(f"{rp}0.bias")
            add(f"{rp}2.weight"); add(f"{rp}2.bias")
    add("post_conv.weight"); add("post_conv.bias")
    return out


def _reference_decode(codec, tokens, speaker_emb) -> "torch.Tensor":
    """PyTorch reference: tokens (n_codebooks, T) + emb (speaker_dim) -> PCM.

    Mirrors DecoderONNX in zaya_audio/export_onnx.py exactly.
    """
    import torch

    cfg = codec.config
    with torch.no_grad():
        z_q = torch.zeros(1, cfg.code_dim, tokens.shape[1])
        for cb in range(cfg.n_codebooks):
            embed = codec.res_vq.embed[cb]  # (K, code_dim)
            idx = tokens[cb].clamp(min=0, max=cfg.codebook_size - 1)
            z_q += torch.nn.functional.embedding(idx, embed).permute(1, 0)
        z = codec.post_vq(z_q)
        audio = codec.decode(z, speaker_emb.unsqueeze(0))  # (1, 1, T_out)
    return audio[0, 0]


def _make_inputs(codec, seed: int, frames: int):
    import torch

    gen = torch.Generator().manual_seed(seed)
    cfg = codec.config
    tokens = torch.randint(0, cfg.codebook_size, (cfg.n_codebooks, frames),
                           generator=gen)
    speaker_emb = torch.randn(cfg.speaker_dim, generator=gen)
    return tokens, speaker_emb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", metavar="PT",
                     help="trained RVQVAE checkpoint (.pt, train_codec.py format)")
    src.add_argument("--random", action="store_true",
                     help="export a randomly-initialised codec (no checkpoint; "
                          "for the C++ self-check)")
    ap.add_argument("--out", default="codec.gguf", metavar="GGUF")
    ap.add_argument("--config", metavar="JSON",
                    help="AudioCodecConfig field overrides (e.g. decoder_strides)")
    ap.add_argument("--frames", type=int, default=56,
                    help="latent frames for the reference sidecar (default 56 = 3 s)")
    ap.add_argument("--seed", type=int, default=1368)
    args = ap.parse_args()

    codec = _load_codec(args)
    cfg = codec.config

    # ── GGUF KVs ────────────────────────────────────────────────────────
    kv: List[Tuple[str, object]] = [
        ("general.architecture", "zaya_codec"),
        ("zaya_codec.sample_rate", cfg.sample_rate),
        ("zaya_codec.n_codebooks", cfg.n_codebooks),
        ("zaya_codec.codebook_size", cfg.codebook_size),
        ("zaya_codec.code_dim", cfg.code_dim),
        ("zaya_codec.latent_dim", cfg.latent_dim),
        ("zaya_codec.speaker_dim", cfg.speaker_dim),
        ("zaya_codec.n_res_blocks", cfg.n_res_blocks),
        ("zaya_codec.decoder_strides",
         [str(s) for s in cfg.decoder_strides]),
        ("zaya_codec.decoder_output_paddings",
         [str(b.conv_transpose.output_padding[0]) for b in codec.decoder]),
    ]
    tensors = _collect_tensors(codec)
    # Output paddings are hardcoded per block in codec.py (RVQVAE.__init__),
    # not in the config — read them from the modules (ground truth).
    output_paddings = [str(b.conv_transpose.output_padding[0]) for b in codec.decoder]
    write_gguf(args.out, kv, tensors)
    total_mb = sum(len(d) for _, _, d in tensors) / 1e6
    print(f"wrote {args.out}: {len(tensors)} tensors, {total_mb:.2f} MB, "
          f"arch=zaya_codec")

    # ── Reference sidecars for the C++ self-check ───────────────────────
    import torch

    tokens, speaker_emb = _make_inputs(codec, args.seed, args.frames)
    audio = _reference_decode(codec, tokens, speaker_emb)
    out_base = Path(args.out)
    (out_base.with_suffix(".tokens.bin")).write_bytes(
        tokens.numpy().astype("<i4").tobytes())
    (out_base.with_suffix(".speaker_emb.bin")).write_bytes(
        speaker_emb.numpy().astype("<f4").tobytes())
    (out_base.with_suffix(".ref_output.bin")).write_bytes(
        audio.numpy().astype("<f4").tobytes())
    print(f"reference: {args.frames} frames -> {audio.numel()} samples "
          f"({audio.numel() / cfg.sample_rate:.3f} s @ {cfg.sample_rate} Hz)")


if __name__ == "__main__":
    main()
