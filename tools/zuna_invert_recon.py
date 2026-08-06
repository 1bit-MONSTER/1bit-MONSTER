#!/usr/bin/env python3
"""Invert the zuna_preprocess tokenization + normalization: recon tokens -> waveform.

recon.bin holds [S, nfine] reconstructed tokens from `1bit zuna`. There are exactly
(orig n_chans)*(n_tc) tokens (in channel-major order, matching chop 'B'). This
denormalizes (the common-average-ref + robust z-score zuna_preprocess applied) and
reshapes back to [n_chans, n_tc*nfine], trimming padding.

Usage: zuna_invert_recon.py <recon.bin> <orig_eeg.npy> <meta.json> <out.npy>
"""
import sys, json, os, numpy as np


def main():
    recon_bin, orig_npy, meta_path, out_npy = sys.argv[1:5]
    tokens = np.fromfile(recon_bin, dtype=np.float32)
    meta = json.load(open(meta_path))
    nfine = meta["nfine"]
    S = meta["S"]; n_chans = meta["n_chans"]
    assert S % n_chans == 0, f"S={S} not divisible by chans={n_chans}"
    n_tc = S // n_chans
    tokens = tokens.reshape(n_chans, n_tc, nfine)      # channel-major (chop 'B')
    recon = tokens.reshape(n_chans, n_tc * nfine)      # [n_chans, n_tc*nfine]

    # denormalize using the same stats zuna_preprocess applied per channel
    raw = np.load(orig_npy).astype(np.float64)
    raw_mean = np.mean(raw, axis=1, keepdims=True)
    raw_centered = raw - raw_mean
    sd = np.std(raw_centered, axis=1, keepdims=True) + 1e-12
    recon = recon * sd + raw_mean

    n_samples = meta["n_samples"]
    recon = recon[:, :n_samples]
    np.save(out_npy, recon)
    print(f"wrote {out_npy} [{recon.shape[0]}x{recon.shape[1]}]")


if __name__ == "__main__":
    main()
