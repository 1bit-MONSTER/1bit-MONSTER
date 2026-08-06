# EEG / ZUNA Research — Datasets & Landscape

Status: **initial recon** (branch `feat/eeg-zuna-research`, 2026-08)
Questions this answers: what ZUNA is, what it was trained on, where EEG datasets
live (HF / Zyphra / elsewhere), and what it would take to run it in 1bit.

---

## 1. What ZUNA actually is

**ZUNA1.1** (380M params) = masked **diffusion autoencoder** for scalp-EEG, Apache-2.0, bf16.
Not a language model, not autoregressive. It reconstructs/denoises/upsamples EEG channels:

- **Denoise** existing channels
- **Reconstruct** missing/dropped channels
- **Upsample** sparse montages — predict signals at electrode positions never recorded

Key trick: tokens are 0.125 s EEG snippets (32 samples @ 256 Hz), position-encoded with
**4D rotary embedding over (x, y, z, t)** — the electrode's 3D scalp coordinate + time index.
Channel-agnostic: works on any montage (4-ch Muse → 256-ch cap) with no retraining.

| Fact | Value |
|---|---|
| Params / dtype | 380M, bf16 (`dim` 1024, 16 enc + 16 dec layers, `head_dim` 64) |
| Token | 32 samples = 0.125 s @ 256 Hz |
| Positional encoding | 4D RoPE (x, y, z, t), θ = 10⁴ |
| Objective | Rectified-flow diffusion (Euler, 50 steps, noise scale 0.1) |
| Training | 580k steps, ~3.5M channel-hours public EEG (v1.1); 2M (v1) |
| License / distro | Apache-2.0 · HF `Zyphra/ZUNA1.1` · `pip install zuna` |
| VRAM | <1 GB — runs on consumer GPU, Mac, CPU |
| Papers | v1: arXiv:2602.18478 · v1.1: arXiv:2607.27308 |

**v1.1 changes over v1:** variable-length input (0.5–30 s), 8 dropout schemes in a 3-stage
curriculum (whole-channel, full-time, channel-time, random-uniform + layout variants),
per-channel-per-second quality scoring (recovers partially-noisy channels), two filter
variants per recording (0.1–45 Hz bandpass; 0.01 Hz highpass + notch), ~1.5M more
channel-hours + more training steps.

**Baseline it beats:** MNE spherical-spline interpolation (NMSE; gap widens at higher
dropout). Evaluated on ANPHY-Sleep (83 ch), Berlin BCI III Dataset V (32 ch), BCI2000
motor-imagery (109 subj, 64 ch), 255-ch AAD dataset — plus TUH/OpenNeuro held-out.

**⚠ Medical caveat:** reconstructions are imputed/plausible, not ground truth. Research use only.

---

## 2. Training data — where the 208 datasets came from

Paper (v1, §II) states the corpus explicitly:

> "We aggregated data from **two major open EEG repositories**: (i) the **Temple
> University Hospital (TUH) EEG Corpus**, and (ii) a large collection of publicly
> available datasets hosted on **OpenNeuro**."

Final corpus: **208 datasets, 24,823,808 non-overlapping 5 s epochs, ~2M channel-hours**.
Channels 2–256 per recording (mean 45, median 22). Only datasets/channels with resolvable
3D scalp coordinates were kept (~5.8 channels/recording dropped, mostly EOG/aux).
Preprocessing: resample to 256 Hz, 0.5 Hz high-pass, common-average reference, adaptive
notch (50/60 Hz detection), flat/clipping/artifact rejection, per-recording z-score.
v1.1 grows this to ~3.5M channel-hours via quality-aware loading + pre-epoched datasets.

**The full 208-dataset list is NOT published** in either paper — the exact TUH+OpenNeuro
subset is internal (Zyphra trains from a private v7 mmap corpus; repo defaults reference
`/data/groups/bci/datasets/v7_train/` + Backblaze B2). Rebuilding the corpus means
re-collecting TUH + OpenNeuro ourselves.

### Zyphra's HF org — datasets
**None for EEG.** `Zyphra/` on HF hosts only text corpora (`Zyda`, `Zyda-2`, `dclm-dedup`).
No training-data release for ZUNA. (Model weights + inference/preprocessing code are open;
the corpus is not.)

---

## 3. Where to get EEG data (public landscape)

### Canonical clinical/research repositories
| Source | What | Access |
|---|---|---|
| **TUH EEG Corpus** | Largest clinical corpus (~25k+ recordings, seizure/epilepsy-heavy, 250 Hz+) | Registration + license (free for research); `smam/tuh-eeg` on HF is a small mirror |
| **OpenNeuro** | ~500 neuro datasets incl. many EEG studies (BIDS) | Open, `openneuro` CLI / S3 |
| **PhysioNet** | Sleep-EDF, CHB-MIT seizure, EEGMMIDB motor imagery, etc. | Open |
| **MOABB** | Aggregator: 40+ BCI datasets (BCI2000, BNCI Horizon, ERP, SSVEP, motor imagery) | Python API, downloads from original hosts |
| **BNCI Horizon 2020** | BCI benchmark datasets | Open (mirrored on HF: `Kkuntal990/bnci-*`) |
| **ANPHY-Sleep** | High-density (83 ch) polysomnography | Open |

### HuggingFace (already-windowed, ready-to-train)
- `braindecode/` org — pre-windowed classic benchmarks: `bcic2a`, `physionet`, `chbmit`,
  `isruc-sleep`, `faced`, `mdd_mumtaz2016`, `bcic2020-3`, `arithmetic_zyma2019` (+ `example_dataset*`)
- Sleep: `WymanYYY/Sleep-EDF-V2`, `carmencraciun/PhysioNet-sleep-edfx`, `iatosh/SleepEDF-V-Dataset`
- BCI/other: `introvoyz041/moabb`, `smam/tuh-eeg`, `Haitao999/things-eeg` (image-decoding), `NeuroBench/thor_eeg_mi`

### Benchmarks for evaluating EEG foundation models
- **OmniEEG-Bench** (arXiv:2606.00815) — 6 task families, unified eval
- **EEG-FM-Bench** (arXiv:2508.17742, github `xw1216/EEG-FM-Bench`) — 14 datasets, 10 paradigms, PEFT sweeps
- **NeuroAtlas** (arXiv:2605.14698, KU Leuven + MIT) — clinical EEG + BCI benchmarking
- **OpenEEGBench** (`braindecode/OpenEEGBench`) — 12 datasets, one-call `benchmark()`, hosted on HF

### Other EEG foundation models (for context)
BEND/BENDR, LaBraM, EEGPT, BrainWave, SleepFM, BioT (Yang et al.), NeuralBench (Banville et al.).
Most are discriminative encoders for classification — ZUNA is unique in being a
**generative reconstruction** model (diffusion).

---

## 4. Can 1bit run it? (TL;DR from earlier analysis)

**No — and not worth forcing.** 1BP conversion is a *format* question, but:

1. **True 1-bit (TQ1/TQ2 ternary) is destructive for dense weights** — repo-measured:
   Qwen3-0.6B dense→TQ2 = PPL 2.6e8 vs 21.8 fp16. ZUNA is dense bf16. Dead on arrival.
2. **Even Q4NX is blocked:** no GGUF exists (no llama.cpp arch for 4D-RoPE continuous
   signals, no vocab), `gguf_to_onebp` can't ingest safetensors→GGUF without a new arch,
   and the engine has no ZUNA arch (19 supported, none matches).
3. **No diffusion runtime:** 1bit runs single-pass decode pipelines. ZUNA inference = 50
   iterative rectified-flow steps + channel masking logic — a diffusion loop the engine
   doesn't have.

**Realistic lane if we ever want ZUNA in 1bit:** a dedicated `zuna` subcommand with a
diffusion sampling loop, 4D RoPE attention (engine has 2D RoPE kernels — 4D needs new
kernel work), and a converter. The linear/GEMV cores would reuse existing kernels.
This is a multi-week project, not a conversion exercise. The 380M weights are small
(<1 GB bf16) — quantization buys little; the problem is the runtime, not the size.

---

## 5. Recommended next steps

1. **Pick a goal.** (a) just run ZUNA1.1 on a local GPU for EEG denoising (pip package
   works today — nothing to build), (b) reproduce/improve the corpus (TUH + OpenNeuro
   collection, ~2–3.5M channel-hours is a serious data-engineering project), or
   (c) port ZUNA into 1bit (diffusion loop + 4D RoPE — multi-week).
2. **If (b):** start with MOABB + braindecode HF windows as a fast pilot corpus (they're
   preprocessed), then add TUH (registration) and OpenNeuro via BIDS. ZUNA's own
   preprocessing pipeline is open source (`src/zuna/preprocessing/`) — reuse it.
3. **If (c):** prototype in Python first (validate rectified-flow + 4D RoPE numerics
   against `zuna` pip package), then spec the C++ arch.

---

## Sources
- ZUNA v1 paper: arXiv:2602.18478 (data section: TUH + OpenNeuro, 208 datasets, 2M ch-hours)
- ZUNA1.1 paper: arXiv:2607.27308 (3.5M ch-hours, 8 dropout schemes, 580k steps)
- Model cards: HF `Zyphra/ZUNA`, `Zyphra/ZUNA1.1` · repo: `github.com/Zyphra/zuna` (Apache-2.0, `pip install zuna`)
- Zyphra blog: zyphra.com/our-work/zuna · Zyphra Cloud EEG playground (free inference, no code)
- Repo docs: `docs/model-families/zyphra.md`, `models/catalog/README.md`, `docs/wiki/models.md` (1BP quant policy)
- Benchmarks: OmniEEG-Bench · EEG-FM-Bench · NeuroAtlas · OpenEEGBench (links in §3)
