#!/usr/bin/env python3
"""train_eagle3.py — self-contained Eagle3 draft trainer for Qwen3-0.6B.

Trains the 1-layer speculative-decoding draft that engine/npu/src/npu_engine_spec.hip
loads, and writes the checkpoint in EXACTLY that binary layout:

  16 float32 tensors, concatenated in this order (from the C++ rv() calls):
    et[V*H]  fc[H*5H]  hn[H]  iln[H]
    qpk[(NH*HD)*(2H)]  kpk[(NKV*HD)*(2H)]  vpk[(NKV*HD)*(2H)]  opk[H*(NH*HD)]
    qn[HD]  kn[HD]  pan[H]
    gpk[IM*H]  upk[IM*H]  dpk[H*IM]  fn_[H]
    lmk[V*H]

The draft forward mirrors `draft_forward` in npu_engine_spec.hip:
  h  = fc(trunk[5H])                 # trunk = 5 target-layer hidden states
  e  = embed(token)
  x  = [rmsnorm(h, hn) | rmsnorm(e, iln)]            # concat, 2H
  q  = qP@x, k = kP@x, v = vP@x                      # per-head RMS norm + RoPE
  attn (1/sqrt(128), GQA 16→8) → oP → +fc_out (pre-norm residual)
  post-attn norm → SwiGLU FFN → residual → final norm → lm_head

Teacher forcing: 7-token windows (ttt_length=7), window-relative RoPE positions
0..6, one draft KV cache (capacity 7) per window — matches deployed block
behavior (draft KV is "block_size positions" in the C++).

LM head = target embedding (tied, frozen) — same as the target's
tie_word_embeddings, so the draft's argmax converges toward the target's.

Usage:
  train-venv/bin/python train_eagle3.py --data train_data/gsm8k_train.jsonl \
      --epochs 1 --max-samples 300 --out checkpoints/eagle3_draft_v3.bin
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

H, NH, NKV, HD, IM = 1024, 16, 8, 128, 3072
NTL = 5                      # target layers used for trunk features
TRUNK_LAYERS = [1, 6, 12, 18, 24]
TTT = 7                      # block/window length == draft KV capacity
EPS = 1e-6
GQA = NH // NKV


def rms_norm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rms_norm_perhead(x, w):
    """x: [NH*HD] flat, w: [HD] — norm over each head, weight broadcast per head."""
    xr = x.view(-1, HD)
    return (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + EPS) * w).reshape(-1)


def make_rope(pos, npos, hd=HD, theta=1_000_000.0):
    """cos/sin tables, matching the C++ RopeTables (theta=1e6, d in [0, hd/2))."""
    d2 = hd // 2
    f = 1.0 / theta ** (torch.arange(d2, dtype=torch.float32) / d2)
    a = torch.arange(npos, dtype=torch.float32).unsqueeze(1) * f.unsqueeze(0)
    return torch.cos(a).float(), torch.sin(a).float()


def rope_apply(x, pos, cos_t, sin_t):
    """x: [NH*HD] flat; rotate each head's [:, :d2]/[:, d2:] by cos/sin[pos]."""
    xr = x.view(-1, HD)
    d2 = HD // 2
    a, b = xr[:, :d2], xr[:, d2:]
    c, s = cos_t[pos], sin_t[pos]
    out = torch.empty_like(xr)
    out[:, :d2] = a * c - b * s
    out[:, d2:] = b * c + a * s
    return out.reshape(-1)


class Eagle3Draft(nn.Module):
    """1 hidden layer — parameter layout mirrors the .bin tensor order."""

    def __init__(self, embed):
        super().__init__()
        self.embed = embed                 # [V, H] target embedding, frozen, shared as LM head
        self.fc = nn.Linear(NTL * H, H, bias=False)
        self.hn = nn.Parameter(torch.ones(H))
        self.iln = nn.Parameter(torch.ones(H))
        self.qP = nn.Linear(2 * H, NH * HD, bias=False)
        self.kP = nn.Linear(2 * H, NKV * HD, bias=False)
        self.vP = nn.Linear(2 * H, NKV * HD, bias=False)
        self.oP = nn.Linear(NH * HD, H, bias=False)
        self.qn = nn.Parameter(torch.ones(HD))
        self.kn = nn.Parameter(torch.ones(HD))
        self.pan = nn.Parameter(torch.ones(H))
        self.gate = nn.Linear(H, IM, bias=False)
        self.up = nn.Linear(H, IM, bias=False)
        self.down = nn.Linear(IM, H, bias=False)
        self.fn = nn.Parameter(torch.ones(H))
        with torch.no_grad():
            for m in (self.fc, self.qP, self.kP, self.vP, self.oP, self.gate, self.up, self.down):
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.embed.requires_grad_(False)

    def step(self, trunk, tok, pos, cos_t, sin_t):
        """One teacher-forced step. trunk: [5H], tok: scalar int. Returns (hidden [H], k, v)."""
        h = self.fc(trunk)
        e = self.embed[tok]
        fc_out = h
        x = torch.cat([rms_norm(h, self.hn), rms_norm(e, self.iln)])   # [2H]
        q = rope_apply(rms_norm_perhead(self.qP(x), self.qn), pos, cos_t, sin_t).view(NH, HD)
        k = rope_apply(rms_norm_perhead(self.kP(x), self.kn), pos, cos_t, sin_t).view(NKV, HD)
        v = self.vP(x).view(NKV, HD)
        self._ks.append(k)              # cache holds entries 0..pos (self-attention incl. current)
        self._vs.append(v)
        # attention: GQA — kv head j serves query heads {GQA*j .. GQA*j+GQA-1}
        nkv = pos + 1
        kc = torch.stack(self._ks)   # [nkv, NKV, HD]
        vc = torch.stack(self._vs)
        qr = q.view(NKV, GQA, HD)        # [NKV, GQA, HD]
        scores = torch.einsum('jgh,pjh->jgp', qr, kc) * (HD ** -0.5)    # [NKV, GQA, nkv]
        attn = torch.softmax(scores, dim=-1)
        o = torch.einsum('jgp,pjh->jgh', attn, vc).reshape(NH * HD)     # [NH*HD]
        h = self.oP(o) + fc_out
        res = h
        h = rms_norm(h, self.pan)
        g, u = self.gate(h), self.up(h)
        h = res + self.down(F.silu(g) * u)
        return rms_norm(h, self.fn), k, v                              # [H], k, v

    def window(self, trunk, toks, w0, cos_t, sin_t):
        """Teacher-force a 7-token window; returns (hidden [K,H], targets [K])."""
        self._ks, self._vs = [], []
        hs, tgts = [], []
        for k in range(min(TTT, len(toks) - 1 - w0)):
            i = w0 + k
            h, kk, vv = self.step(trunk[i], toks[i], k, cos_t, sin_t)
            hs.append(h)
            tgts.append(toks[i + 1])
        return torch.stack(hs), torch.tensor(tgts)


def build_checkpoint(draft, vocab, out_path):
    """Write the 16 f32 tensors in the exact npu_engine_spec.hip order."""
    with torch.no_grad():
        w = lambda t: t.detach().float().reshape(-1).numpy()
        parts = [
            w(draft.embed), w(draft.fc.weight), w(draft.hn), w(draft.iln),
            w(draft.qP.weight), w(draft.kP.weight), w(draft.vP.weight), w(draft.oP.weight),
            w(draft.qn), w(draft.kn), w(draft.pan),
            w(draft.gate.weight), w(draft.up.weight), w(draft.down.weight), w(draft.fn),
            w(draft.embed),   # lmk = tied embedding
        ]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    buf = b"".join(p.tobytes() for p in parts)
    with open(out_path, "wb") as f:
        f.write(buf)
    return len(buf)


def expected_bin_size(vocab):
    n = (vocab * H * 2                      # et + lmk
         + H * NTL * H                       # fc
         + 2 * H                             # hn + iln
         + (NH * HD) * (2 * H)               # qpk
         + (NKV * HD) * (2 * H) * 2          # kpk + vpk
         + H * (NH * HD)                     # opk
         + 2 * HD                            # qn + kn
         + H                                 # pan
         + IM * H * 2                        # gpk + upk
         + H * IM                            # dpk
         + H)                                # fn_
    return n * 4


def load_samples(path, tokenizer, max_len=512, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            assert isinstance(d["turns"], list) and len(d["turns"]) == 2, d
            rows.append(d["turns"])
            if limit and len(rows) >= limit:
                break
    out = []
    for q, a in rows:
        ids = tokenizer(f"{q}\n\n{a}", add_special_tokens=True).input_ids
        if len(ids) < 8 or len(ids) > max_len:
            continue
        out.append(ids)
    return out


@torch.no_grad()
def trunk_features(model, ids, device):
    """Target forward → trunk hidden states [L, 5H] (post-residual at TRUNK_LAYERS)."""
    out = model(torch.tensor([ids], device=device), output_hidden_states=True)
    hs = [out.hidden_states[l + 1][0].to(torch.float32) for l in TRUNK_LAYERS]
    return torch.cat(hs, dim=-1)          # [L, 5H]


def acceptance_proxy(draft, model, tokenizer, trunk_layers, ids, cos_t, sin_t, device):
    """% of positions where draft argmax == target argmax (same criterion as C++)."""
    with torch.no_grad():
        out = model(torch.tensor([ids], device=device), output_hidden_states=True)
        target = torch.argmax(out.logits[0, :-1], dim=-1)                 # [L-1]
        trunk = torch.cat([out.hidden_states[l + 1][0].to(torch.float32) for l in trunk_layers], dim=-1)
        agree = total = 0
        for w0 in range(0, len(ids) - 1, TTT):
            hs, _ = draft.window(trunk, ids, w0, cos_t, sin_t)
            logits = (hs @ draft.embed.t()).float()
            pred = torch.argmax(logits, dim=-1)
            off = w0
            agree += int((pred == target[off:off + len(pred)]).sum())
            total += len(pred)
        return agree / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_data/gsm8k_train.jsonl")
    ap.add_argument("--out", default="checkpoints/eagle3_draft_v3.bin")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--eval-jsonl", default="train_data/eval/perfectblend.jsonl")
    ap.add_argument("--eval-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"
    t0 = time.time()
    print("Loading Qwen3-0.6B (target) + tokenizer...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    model.eval()
    V = model.config.vocab_size
    assert V == 151936, f"vocab mismatch: {V}"
    embed = model.model.embed_tokens.weight.detach().float()
    print(f"  target loaded in {time.time() - t0:.1f}s, vocab={V}")

    draft = Eagle3Draft(embed)
    n_train = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"Draft params (trainable): {n_train/1e6:.1f}M")

    cos_t, sin_t = make_rope(0, TTT)
    samples = load_samples(args.data, tok, args.max_len, args.max_samples)
    print(f"Dataset: {len(samples)} samples from {args.data}")
    if not samples:
        sys.exit("ERROR: no training samples — dataset format or tokenizer issue")
    lens = [len(s) for s in samples]
    print(f"  token lengths: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.0f}")

    opt = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    autocast = torch.autocast(device_type="cpu", dtype=torch.bfloat16)  # config precision="bf16"
    losses = []
    t_train = time.time()
    for ep in range(args.epochs):
        for si, ids in enumerate(samples):
            trunk = trunk_features(model, ids, device)                      # [L, 5H]
            n_pos = 0
            loss_sum = torch.zeros(())
            with autocast:
                for w0 in range(0, len(ids) - 1, TTT):                          # non-overlapping windows
                    hs, tgts = draft.window(trunk, ids, w0, cos_t, sin_t)       # [K, H], [K]
                    logits = hs @ embed.t()                                     # [K, V] (tied, frozen)
                    loss_sum = loss_sum + F.cross_entropy(logits.float(), tgts)
                    n_pos += len(tgts)
            loss = loss_sum / n_pos
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if (si + 1) % 25 == 0 or si == 0:
                el = time.time() - t_train
                sps = (si + 1) / el
                print(f"  ep{ep} s{si+1}/{len(samples)} loss={loss.item():.2f} "
                      f"avg(last25)={sum(losses[-25:])/len(losses[-25:]):.2f} "
                      f"({sps:.2f} samp/s, {el/(si+1):.2f}s/samp)")

    elapsed = time.time() - t_train
    print(f"\nTraining: {len(samples)} samples in {elapsed:.0f}s "
          f"({elapsed/len(samples):.2f}s/sample → full 7473-sample epoch ≈ {elapsed/len(samples)*7473/60:.0f} min)")
    print(f"Loss trajectory: first={losses[0]:.2f} last={losses[-1]:.2f} min={min(losses):.2f}")

    # Acceptance proxy on eval samples
    print(f"\nAcceptance proxy (draft argmax == target argmax, greedy) on {args.eval_jsonl}:")
    with open(args.eval_jsonl) as f:
        ev_rows = [json.loads(l) for l in f if l.strip()][: args.eval_samples]
    accs = []
    for d in ev_rows:
        ids = tok(d["turns"][0], add_special_tokens=True).input_ids
        if len(ids) < 8:
            continue
        a = acceptance_proxy(draft, model, tok, TRUNK_LAYERS, ids, cos_t, sin_t, device)
        accs.append(a)
        print(f"  {a*100:5.1f}%  n_tok={len(ids)}")
    if accs:
        print(f"  mean acceptance proxy: {sum(accs)/len(accs)*100:.1f}%")

    size = build_checkpoint(draft, V, args.out)
    exp = expected_bin_size(V)
    print(f"\nCheckpoint: {args.out} ({size/1e9:.2f} GB, expected {exp/1e9:.2f} GB) "
          f"{'OK' if size == exp else 'SIZE MISMATCH'}")
    sys.exit(0 if size == exp else 1)


if __name__ == "__main__":
    main()
