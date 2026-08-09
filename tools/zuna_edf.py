#!/usr/bin/env python3
"""EDF/EDF+/BDF -> ZUNA tokens bridge (numpy-only, no mne/scipy).

Reads a real medical EEG recording (EDF, EDF+, or BDF), maps its channel
labels onto standard 10-20 scalp positions, conditions the signal (FFT
resample + highpass/notch), and writes the exact same contract as
zuna_preprocess.py: tokens.bin [S,32], tok_idx.bin [S,4], meta.json, and
raw.npy (the post-conditioning array).

raw.npy is the invert contract: zuna_invert_recon.py recomputes the
denormalization stats from the raw file it is given, so raw.npy MUST be the
exact post-resample/post-filter array (pre-centering, pre-pad) or the
reconstruction comes out with wrong amplitude/offset. Do not change that.

Usage:
  zuna_edf.py <rec.edf|rec.bdf> <out_dir> [rate] [--filter]
              [--chan-pos chan_pos.npy] [--bids <root>] [--selfcheck]

  --filter     apply 0.5 Hz highpass + 50/60 Hz notch (reuses preprocess)
  --chan-pos   [N,3] xyz meters; overrides the embedded 10-20 table
               (TUH/OpenNeuro recordings have non-10-20 channels; unknown
               labels fail loudly instead of misplacing electrodes)
  --bids       pick the first participants/*/eeg/*_eeg.edf under <root>
  --selfcheck  write synthetic EDF/BDF/EDF+, read back, assert, and run the
               tokenize->invert round-trip (recon = tokens)

Writes: tokens.bin (fp32 [S,32]), tok_idx.bin (int32 [S,4]), meta.json
(+ "source":"edf", "research_only":true), raw.npy ([N, n_conditioned] f64).
"""
import sys, os, json, argparse, struct
import numpy as np
from zuna_preprocess import discretize_chan_pos, chop_and_reshape_signals, highpass_notch

# Standard 10-20 scalp positions (MNE spherical_1020.tsv, unit sphere),
# scaled to meters with head radius 0.1 m (matches discretize ±0.13 box).
_R = 0.1
TEN_TWENTY_XYZ = {
    "Fp1": (-0.2939, 0.9045, 0.3090), "Fp2": (0.2939, 0.9045, 0.3090),
    "Fpz": (0.0000, 0.9511, 0.3090), "F3": (-0.4591, 0.5800, 0.6730),
    "F4": (0.4591, 0.5800, 0.6730), "Fz": (0.0000, 0.5878, 0.8090),
    "F7": (-0.7695, 0.5590, 0.3090), "F8": (0.7695, 0.5590, 0.3090),
    "C3": (-0.5878, 0.0000, 0.8090), "C4": (0.5878, 0.0000, 0.8090),
    "Cz": (0.0000, 0.0000, 1.0000), "T7": (-0.9511, 0.0000, 0.3090),
    "T8": (0.9511, 0.0000, 0.3090), "P3": (-0.4591, -0.5800, 0.6730),
    "P4": (0.4591, -0.5800, 0.6730), "Pz": (0.0000, -0.5878, 0.8090),
    "P7": (-0.7695, -0.5590, 0.3090), "P8": (0.7695, -0.5590, 0.3090),
    "O1": (-0.2939, -0.9045, 0.3090), "O2": (0.2939, -0.9045, 0.3090),
    "Oz": (0.0000, -0.9511, 0.3090),
}
for _k, _v in list(TEN_TWENTY_XYZ.items()):
    TEN_TWENTY_XYZ[_k] = tuple(c * _R for c in _v)
# TUH uses the older 10-20 names T3/T4/T5/T6 for T7/T8/P7/P8.
TEN_TWENTY_XYZ["T3"] = TEN_TWENTY_XYZ["T7"]
TEN_TWENTY_XYZ["T4"] = TEN_TWENTY_XYZ["T8"]
TEN_TWENTY_XYZ["T5"] = TEN_TWENTY_XYZ["P7"]
TEN_TWENTY_XYZ["T6"] = TEN_TWENTY_XYZ["P8"]


def norm_label(raw):
    """'EEG Fp1-Ref' / 'Fp1-Avg' / 'FP1' -> 'Fp1'; '' if not EEG-ish."""
    s = raw.strip().replace(" ", "")
    s = s.replace("EEG", "", 1) if s.startswith("EEG") else s
    for suf in ("-Ref", "-Avg", "-LE", "-A1", "-A2", "-M1", "-M2", "-REF"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    if not s or s.lower().startswith(("eog", "ecg", "ekg", "emg")):
        return ""
    return s[0].upper() + s[1:].lower()


def read_edf(path):
    """Return (labels, data[n_chans, n_samples], is_bdf, rate).

    Reads real EDF/EDF+/BDF files via pyedflib. The manual parser this
    replaces used a self-consistent custom dialect (writer and reader agreed
    with each other but not with any real EDF), which rejected every real
    file. pyedflib handles the standard spec layout, TUH's non-standard
    header dialect (numeric fields shifted from spec positions, 16-byte
    label runs), EDF+ annotation channels, and BDF 24-bit sign extension.
    Annotation channels are excluded; data is scaled digital -> physical.
    """
    try:
        import pyedflib
    except ImportError:
        raise ImportError(
            "zuna_edf.py needs pyedflib to read real EDF files; "
            "install it with: pip install pyedflib")
    with open(path, "rb") as f:
        head = f.read(48)
    is_bdf = head[:8].decode("ascii", "ignore").strip().startswith("BIOSEMI") \
        or head[44:48] == b"24BIT"
    f = pyedflib.EdfReader(path)
    rate = 0.0
    try:
        all_labels = f.getSignalLabels()
        signals = []
        for c in range(f.signals_in_file):
            nsamp = f.getNSamples()[c]
            if nsamp <= 0:
                continue
            signals.append((all_labels[c], f.readSignal(c, 0, nsamp)))
            if rate == 0.0:
                rate = float(f.getSampleFrequencies()[c])
    finally:
        f.close()
    if not signals:
        raise ValueError(f"{path}: no signal channels found")
    labels = [l for l, _ in signals]
    n = min(len(d) for _, d in signals)
    data = np.stack([d[:n] for _, d in signals]).astype(np.float64)
    return labels, data, is_bdf, rate



def phys_scale(data, dig_min, dig_max, phys_min, phys_max):
    if dig_max == dig_min:
        return data
    return (data - dig_min) * (phys_max - phys_min) / (dig_max - dig_min) + phys_min


def fft_resample(x, src_rate, dst_rate):
    """Anti-aliased FFT resample along last axis (scipy.signal.resample equiv)."""
    if src_rate == dst_rate:
        return x
    n_in, n_out = x.shape[1], int(round(x.shape[1] * dst_rate / src_rate))
    X = np.fft.rfft(x, axis=1)
    if n_out > n_in:
        X = np.pad(X, ((0, 0), (0, n_out // 2 + 1 - X.shape[1])))
    else:
        X = X[:, : n_out // 2 + 1]
    return np.fft.irfft(X, n=n_out, axis=1) * (n_out / n_in)


def find_bids_edf(root):
    for p in sorted(os.path.join(dp, fn)
                    for dp, _, fns in os.walk(root) for fn in fns):
        if p.endswith("_eeg.edf"):
            return p
    raise SystemExit(f"--bids: no *_eeg.edf under {root}")


def label_to_pos(labels, chan_pos_override=None):
    if chan_pos_override is not None:
        pos = np.load(chan_pos_override).astype(np.float64)
        if pos.shape[0] != len(labels):
            raise SystemExit(f"--chan-pos rows {pos.shape[0]} != channels {len(labels)}")
        return pos
    pos = np.zeros((len(labels), 3))
    for i, l in enumerate(labels):
        key = norm_label(l)
        if key not in TEN_TWENTY_XYZ:
            raise SystemExit(
                f'channel "{l}" (normalized "{key}") not in 10-20 table; '
                "provide --chan-pos file.npy or rename the channel")
        pos[i] = TEN_TWENTY_XYZ[key]
    return pos


def condition(eeg, rate, target_rate, do_filter):
    """Resample + filter -> the array invert will denormalize FROM. This exact
    array is saved as raw.npy (pre-centering, pre-pad) — the invert contract."""
    if target_rate and target_rate != rate:
        eeg = fft_resample(eeg, rate, target_rate)
        rate = target_rate
    if do_filter:
        eeg = highpass_notch(eeg, rate)
    return eeg, rate


def tokenize(eeg, chan_pos, out_dir, rate, tf=32, num_bins=100):
    """Center/scale + pad + chop (mirrors zuna_preprocess). Returns meta."""
    n_chans, n_pts = eeg.shape
    centered = eeg - np.mean(eeg, axis=1, keepdims=True)
    sd = np.std(centered, axis=1, keepdims=True) + 1e-12
    eeg_n = centered / sd
    pad = (tf - (n_pts % tf)) % tf
    if pad:
        eeg_n = np.pad(eeg_n, ((0, 0), (0, pad)))
    cp_disc = discretize_chan_pos(chan_pos, num_bins=num_bins)
    tokens, cp_disc_r, t_coarse, S = chop_and_reshape_signals(eeg_n, chan_pos, cp_disc, tf=tf)
    os.makedirs(out_dir, exist_ok=True)
    tokens.astype(np.float32).tofile(os.path.join(out_dir, "tokens.bin"))
    tok_idx = np.concatenate([cp_disc_r.astype(np.int64), t_coarse], axis=1).astype(np.int32)
    tok_idx.tofile(os.path.join(out_dir, "tok_idx.bin"))
    meta = {"S": S, "n_chans": n_chans, "n_samples": n_pts, "rate": rate,
            "nfine": tf, "num_bins": num_bins,
            "tok_idx_cols": ["x", "y", "z", "tc"],
            "source": "edf", "research_only": True}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    print(f"tokens[{S},{tf}] tok_idx[{S},4] max_tok={tok_idx.max()} -> {out_dir}")
    return meta


# ---------------------------------------------------------------- selfcheck

def _write_synth_edf(path, labels, data, rate, is_bdf=False, annotate=False):
    """Minimal EDF/BDF writer via pyedflib (1 record, 1 s, int16 / 24-bit LE).

    (Replaced the hand-rolled header writer, which emitted a non-standard
    dialect that pyedflib rejects. For EDF+, pyedflib manages the annotation
    channel internally; writeAnnotation() feeds it.)"""
    import datetime
    import pyedflib
    n_ch, _ = data.shape
    if is_bdf:
        file_type = pyedflib.FILETYPE_BDF
    elif annotate:
        file_type = pyedflib.FILETYPE_EDFPLUS
    else:
        file_type = pyedflib.FILETYPE_EDF
    f = pyedflib.EdfWriter(path, n_ch, file_type=file_type)
    try:
        ch_info = []
        for c in range(n_ch):
            ch_info.append({"label": labels[c], "dimension": "uV",
                            "sample_frequency": rate,
                            "physical_min": -1000.0, "physical_max": 1000.0,
                            "digital_min": -32768, "digital_max": 32767,
                            "transducer": "", "prefilter": ""})
        f.setSignalHeaders(ch_info)
        f.setHeader({"technician": "synth", "recording_additional": "selfchk",
                     "patientname": "synth", "patient_additional": "",
                     "patientcode": "synth", "equipment": "", "admincode": "",
                     "sex": "", "birthdate": datetime.datetime(1970, 1, 1),
                     "startdate": datetime.datetime(1970, 1, 1)})
        f.setStartdatetime(datetime.datetime(1970, 1, 1))
        f.writeSamples([data[c].astype(np.float64) for c in range(n_ch)])
        if annotate:
            f.writeAnnotation(0, 0, "synth event")
    finally:
        f.close()



def _selfcheck():
    import tempfile, subprocess, shutil
    t = tempfile.mkdtemp(prefix="zuna_edf_sc_")
    try:
        rate = 256
        t8 = np.linspace(0, 1, rate, endpoint=False)
        data = np.zeros((8, rate))
        data[0] = np.sin(2 * np.pi * 10 * t8) * 500  # 10 Hz, 500 uV
        labels = ["Fp1", "Fp2", "C3", "C4", "Cz", "Pz", "O1", "O2"]
        # 1) EDF round-trip
        p = os.path.join(t, "synth.edf")
        _write_synth_edf(p, labels, data, rate)
        lbl, d, is_bdf, r = read_edf(p)
        assert r == rate, f"rate {r}"
        assert lbl == labels, f"labels {lbl} != {labels}"
        assert d.shape == (8, rate), f"shape {d.shape}"
        assert np.allclose(d[0], data[0], atol=1.0), "sine amplitude lost"
        assert not is_bdf
        # 2) BDF 24-bit sign extension (negative values)
        pb = os.path.join(t, "synth.bdf")
        _write_synth_edf(pb, labels, data, rate, is_bdf=True)
        lbl, db, is_bdf, r = read_edf(pb)
        assert is_bdf and np.allclose(db[0], data[0], atol=1.0), "BDF 24-bit broken"
        # 3) EDF+ annotation skip
        pa = os.path.join(t, "synth_plus.edf")
        _write_synth_edf(pa, labels, data, rate, annotate=True)
        lbl, da, _, _ = read_edf(pa)
        assert lbl == labels and np.allclose(da[0], data[0], atol=1.0), "EDF+ annotation skip broken"
        # 4) full tokenize -> invert round-trip (recon = tokens)
        out = os.path.join(t, "tok")
        os.makedirs(out, exist_ok=True)
        eeg, r = condition(d, rate, rate, do_filter=True)
        np.save(os.path.join(out, "raw.npy"), eeg)  # exact post-conditioning array
        pos = label_to_pos(labels)
        meta = tokenize(eeg, pos, out, rate)
        shutil.copyfile(os.path.join(out, "tokens.bin"), os.path.join(out, "recon.bin"))
        rp = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__),
                            "zuna_invert_recon.py"), os.path.join(out, "recon.bin"),
                            os.path.join(out, "raw.npy"), os.path.join(out, "meta.json"),
                            os.path.join(out, "out.npy")], capture_output=True, text=True)
        assert rp.returncode == 0, rp.stderr
        recon = np.load(os.path.join(out, "out.npy"))
        assert recon.shape == (8, rate), f"recon shape {recon.shape}"
        assert np.all(np.isfinite(recon)), "non-finite recon"
        print("SELFCHECK PASS: EDF/BDF/EDF+ round-trip + tokenize->invert [8x%d]" % rate)
    finally:
        shutil.rmtree(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="rec.edf / rec.bdf")
    ap.add_argument("out_dir", nargs="?", help="output dir (tokens.bin, tok_idx.bin, meta.json, raw.npy)")
    ap.add_argument("rate", type=float, nargs="?", default=256.0)
    ap.add_argument("--filter", action="store_true")
    ap.add_argument("--chan-pos")
    ap.add_argument("--bids")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        _selfcheck()
        return
    if a.bids:
        a.input = find_bids_edf(a.bids)
    if not a.input or not a.out_dir:
        ap.error("input and out_dir required (or --selfcheck / --bids)")
    labels, eeg, is_bdf, src_rate = read_edf(a.input)
    if src_rate <= 0:
        raise SystemExit(f"{a.input}: cannot determine sample rate from header")
    if a.chan_pos is None:
        # Clinical recordings carry reference (A1/A2), EKG, photic and other
        # non-scalp channels; keep only channels that map onto the 10-20 table.
        keep = [i for i, l in enumerate(labels) if norm_label(l) in TEN_TWENTY_XYZ]
        dropped = [labels[i] for i in range(len(labels)) if i not in keep]
        if not keep:
            raise SystemExit(f"{a.input}: no channels map to the 10-20 table; "
                             "provide --chan-pos file.npy")
        if dropped:
            print(f"note: dropping non-10-20 channels: {dropped}", file=sys.stderr)
        labels = [labels[i] for i in keep]
        eeg = eeg[keep]
    eeg, rate = condition(eeg, src_rate, a.rate, a.filter)
    pos = label_to_pos(labels, a.chan_pos)
    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "raw.npy"), eeg)  # invert contract — see docstring
    tokenize(eeg, pos, a.out_dir, rate)
    print(f"{a.input}: {len(labels)} ch ({'BDF' if is_bdf else 'EDF'}), "
          f"{rate} Hz, research-only (imputed reconstruction)")


if __name__ == "__main__":
    main()
