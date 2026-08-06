#!/usr/bin/env python3
"""ZUNA preprocessing bridge: raw EEG -> tokens.bin + tok_idx.bin for `1bit zuna`.

Mirrors the reference tokenization (chop_and_reshape_signals + discretize_chan_pos)
so the C++ port receives exactly the input contract it was verified against.

Input: a raw EEG recording laid out as [n_channels, n_samples] float32/float64.
Channel 3D scalp positions (meters) are required to build the 4D RoPE tok_idx.

Usage:
  zuna_preprocess.py <eeg.npy|eeg.f32> <chan_pos.npy> <out_dir> [sampling_hz]
                        [--nfine 32] [--num_bins 100] [--rate target_hz] [--filter]
  out_dir: writes tokens.bin (fp32 [S,32]), tok_idx.bin (int32 [S,4]), and meta.json

If --rate is given, signals are resampled to target_hz (linear-rate preserves
channel count; use a multiple of nfine ideally 8x). If --filter, a 0.5 Hz highpass +
50/60 Hz notch is applied (lightweight, matches reference intent). For production
quality conditioning use MRI/BIDS EDF and the reference `zuna` preprocessing package;
this bridge is the minimal path from raw waveform to `1bit zuna`.

`1bit zuna <weights_dir> tokens.bin tok_idx.bin recon.bin enc.bin <seed> z_true.bin`
"""
import sys, os, json, argparse
import numpy as np


def discretize_chan_pos(chan_pos, xyz_extremes=(-0.13, 0.13), num_bins=100):
    """Reference discretize_chan_pos: [N,3] meters -> [N,3] int bin indices."""
    xyz_min = np.asarray(xyz_extremes[0], dtype=np.float64)
    xyz_max = np.asarray(xyz_extremes[1], dtype=np.float64)
    norm = (chan_pos - xyz_min) / (xyz_max - xyz_min)
    disc = (norm * num_bins).astype(np.int64)
    return np.clip(disc, 0, num_bins - 1)


def chop_and_reshape_signals(eeg_signal, chan_pos, chan_pos_discrete, tf=32, use_coarse_time="B"):
    """Reference chop_and_reshape_signals (use_coarse_time='B', default).
    eeg_signal [N,total]; returns tokens [S,tf], chan_pos_disc [S,3], t_coarse [S,1], seqlen.
    """
    num_chans, num_tpts = eeg_signal.shape
    assert num_tpts % tf == 0, f"{num_tpts} not divisible by tf={tf}"
    tc = num_tpts // tf
    seqlen = num_chans * tc
    eeg_reshaped = eeg_signal.reshape(num_chans, tc, tf).reshape(seqlen, tf)
    # 'B': repeat_interleave channels together: ch1 all its tc, then ch2 ...
    cp_disc = np.repeat(chan_pos_discrete, tc, axis=0)          # [S,3]
    t_coarse = np.tile(np.arange(tc, dtype=np.int64), num_chans).reshape(seqlen, 1)
    return eeg_reshaped, cp_disc, t_coarse, seqlen


def resample(x, src_rate, dst_rate):
    """Linear resample along last axis. Keeps sample count proportional: bad for
    discrete filters; adequate for feeding a model that resamples internally."""
    if src_rate == dst_rate:
        return x
    n_in, n_out = x.shape[1], int(round(x.shape[1] * dst_rate / src_rate))
    src_idx = np.linspace(0, n_in - 1, n_out)
    out = np.empty((x.shape[0], n_out), dtype=x.dtype)
    for ch in range(x.shape[0]):
        out[ch] = np.interp(src_idx, np.arange(n_in), x[ch])
    return out


def highpass_notch(x, rate, hp=0.5):
    """Very light IIR-ish highpass + 50/60Hz notch via FFT (no scipy dep)."""
    n = x.shape[1]
    X = np.fft.rfft(x, axis=1)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    # highpass
    X[:, freqs < hp] = 0
    # notch 50/60 + harmonics within Nyquist
    for f0 in (50, 60):
        for k in range(1, int(rate / 2 / f0) + 1):
            fc = f0 * k
            if fc >= rate / 2:
                break
            band = np.abs(freqs - fc) <= 1.0
            X[:, band] = 0
    return np.fft.irfft(X, n=n, axis=1).astype(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eeg", help="[N,n_samples] float32/64 .npy or raw .f32")
    ap.add_argument("chan_pos", help="[N,3] float64 channel xyz (meters) .npy")
    ap.add_argument("out_dir")
    ap.add_argument("rate", type=float, nargs="?", default=256.0)
    ap.add_argument("--nfine", type=int, default=32, help="fine time pts per token (32)")
    ap.add_argument("--num_bins", type=int, default=100)
    ap.add_argument("--target-rate", type=float, default=0.0, help="resample to this Hz")
    ap.add_argument("--filter", action="store_true", help="apply light highpass+notch")
    a = ap.parse_args()

    if a.eeg.endswith(".npy"):
        eeg = np.load(a.eeg)
    else:
        eeg = np.fromfile(a.eeg, dtype=np.float32)
        raise SystemExit("raw .f32: must know channel count; pass an .npy with [N,nsamp] instead")
    eeg = eeg.astype(np.float64)
    if eeg.ndim != 2:
        raise SystemExit(f"expected 2D [channels,samples], got {eeg.shape}")
    chan_pos = np.load(a.chan_pos).astype(np.float64)
    if chan_pos.shape[0] != eeg.shape[0]:
        raise SystemExit(f"chan_pos rows {chan_pos.shape[0]} != channels {eeg.shape[0]}")

    n_chans, n_pts = eeg.shape
    rate = a.rate
    if a.target_rate:
        eeg = resample(eeg, rate, a.target_rate)
        rate = a.target_rate
        n_pts = eeg.shape[1]
    if a.filter:
        eeg = highpass_notch(eeg, rate)
    # center + scale (common-average reference + robust z) — keep simple, optional
    eeg = eeg - np.mean(eeg, axis=1, keepdims=True)
    sd = np.std(eeg, axis=1, keepdims=True) + 1e-12
    eeg = eeg / sd

    # pad to multiple of nfine
    tf = a.nfine
    pad = (tf - (n_pts % tf)) % tf
    if pad:
        eeg = np.pad(eeg, ((0, 0), (0, pad)))
    cp_disc = discretize_chan_pos(chan_pos, num_bins=a.num_bins)
    tokens, cp_disc_r, t_coarse, S = chop_and_reshape_signals(eeg, chan_pos, cp_disc, tf=tf)

    os.makedirs(a.out_dir, exist_ok=True)
    tokens.astype(np.float32).tofile(os.path.join(a.out_dir, "tokens.bin"))
    tok_idx = np.concatenate([cp_disc_r.astype(np.int64), t_coarse], axis=1).astype(np.int32)
    tok_idx.tofile(os.path.join(a.out_dir, "tok_idx.bin"))
    json.dump({"S": S, "n_chans": n_chans, "n_samples": n_pts, "rate": rate,
               "nfine": tf, "num_bins": a.num_bins,
               "tok_idx_cols": ["x", "y", "z", "tc"]},
              open(os.path.join(a.out_dir, "meta.json"), "w"), indent=2)
    print(f"tokens[{S},{tf}] tok_idx[{S},4] max_tok={tok_idx.max()} -> {a.out_dir}")


if __name__ == "__main__":
    main()
