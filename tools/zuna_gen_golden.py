#!/usr/bin/env python3
"""ZUNA golden-trace generator + weight exporter for the C++ port parity harness.

Generates:
  <out>/tokens.bin       fp32 [S,32]  tokenized toy EEG input
  <out>/tok_idx.bin      int32 [S,4]  (x,y,z,tc) positions
  <out>/z_true.bin       fp32 [S,32]  exact initial diffusion noise used by the
                                      reference (captured in-sample, so RNG matches)
  <out>/enc_out_ref.npy  reference encoder latent  [1,S,32]
  <out>/recon_ref.npy    reference 50-step reconstruction  [1,S,32]
  <out>/weights.bin      raw fp32 row-major weights (all 639 tensors)
  <out>/weights.json     manifest {name,shape,offset,bytes}

Usage: zuna_gen_golden.py <hf_repo> <ckpt_dir> <out_dir> [seed]
  hf_repo   e.g. Zyphra/ZUNA1.1 (for config.json)
  ckpt_dir  local dir containing model-00001-of-00001.safetensors
  out_dir   where to write binaries/numpy
  seed      default 0

Requires: torch, numpy, safetensors, the `zuna` package (pip install zuna),
and network access for config.json.
"""
import json, sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../tools")  # noqa
from zuna.inference.AY2l.lingua.lingua.args import dataclass_from_dict            # noqa: E402
from zuna.inference.AY2l.lingua.apps.AY2latent_bci.transformer import EncoderDecoder, DecoderTransformerArgs  # noqa: E402
from zuna.inference.AY2l.lingua.apps.AY2latent_bci.eeg_data import chop_and_reshape_signals, discretize_chan_pos  # noqa: E402
import urllib.request  # noqa: E402


def main():
    repo, ck, OUT, seed = sys.argv[1], sys.argv[2], sys.argv[3], (int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(urllib.request.urlopen(f"https://huggingface.co/{repo}/raw/main/config.json"))
    args = dataclass_from_dict(DecoderTransformerArgs, cfg["model"])
    model = EncoderDecoder(args).to("cpu"); model.eval()
    from safetensors.torch import load_file
    sd = load_file(os.path.join(ck, "model-00001-of-00001.safetensors"))
    model.load_state_dict({k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}, strict=True)

    # Toy EEG: 8ch x 256 samples (1 s @ 256 Hz), sparse features on 2 channels.
    fs, NC = 256, 8
    t = np.arange(0, 1.0, 1/fs)
    eeg = np.zeros((NC, len(t)))
    eeg[0] = 50e-6*np.sin(2*np.pi*10.0*t)
    eeg[1] = 20e-6*np.cos(2*np.pi*25.0*t)
    eeg_sig = torch.from_numpy(eeg)
    tf = args.num_fine_time_pts
    angles = np.linspace(0, 2*np.pi, NC, endpoint=False)
    cp = torch.from_numpy(np.stack([0.1*np.sin(angles), 0.1*np.cos(angles), 0.01*np.ones(NC)], 1).astype(np.float64))
    cpsi = discretize_chan_pos(cp, (-0.13, 0.13), 100)
    nc, _, cpdr, _, tcr, sql, _ = chop_and_reshape_signals(eeg_sig.double(), chan_pos=cp,
                                                           chan_pos_discrete=cpsi, tf=tf, use_coarse_time="B")
    S = nc.shape[0]
    tokens = nc.float().unsqueeze(0)                      # [1,S,32]
    tok_idx = torch.cat([cpdr.unsqueeze(0).float(), tcr.unsqueeze(0).float()], 2).long()  # [1,S,4]
    seq_lens = torch.tensor([S], dtype=torch.int64)

    with torch.no_grad():
        enc_out, tok_idx_reg, _ = model.encoder(token_values=tokens, seq_lens=seq_lens, tok_idx=tok_idx, mask=None)
        # Replicate sample() exactly (so the captured z matches the reconstruction RNG)
        z = model.global_sigma * torch.randn_like(tokens)
        recon = z.clone(); dt = 0.02
        for i in range(50, 0, -1):
            tm = torch.tensor([[[i*0.02]]])
            vc, _ = model.decoder(tokens=recon.unsqueeze(1), cross_attended=enc_out, timeD=tm,
                                 seq_lens=seq_lens, cross_seq_lens=seq_lens, tok_idx=tok_idx, cross_tok_idx=tok_idx)
            recon = recon - dt*vc.squeeze(1)

    tokens.numpy()[0].astype(np.float32).tofile(os.path.join(OUT, "tokens.bin"))
    tok_idx.numpy()[0].astype(np.int32).tofile(os.path.join(OUT, "tok_idx.bin"))
    z.numpy()[0].astype(np.float32).tofile(os.path.join(OUT, "z_true.bin"))
    np.save(os.path.join(OUT, "enc_out_ref.npy"), enc_out.float().numpy())
    np.save(os.path.join(OUT, "recon_ref.npy"), recon.float().numpy())

    # Export weights: all tensors flattened fp32 + manifest
    with open(os.path.join(OUT, "weights.bin"), "wb") as f, open(os.path.join(OUT, "weights.json"), "w") as mj:
        man, off = [], 0
        for k in sorted(sd):
            t = sd[k].float().contiguous(); b = t.numpy().tobytes()
            f.write(b)
            man.append({"name": k[len("model."):] if k.startswith("model.") else k,
                        "shape": list(t.shape), "offset": off, "bytes": len(b)})
            off += len(b)
        json.dump(man, mj)
    print(f"golden S={S} enc_max={enc_out.abs().max().item():.3f} recon_max={recon.abs().max().item():.5f} weights={off/1e6:.0f}MB")


if __name__ == "__main__":
    main()
