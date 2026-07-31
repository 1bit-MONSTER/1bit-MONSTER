#!/usr/bin/env python3
"""vl_logit_check_ref.py — compare 1BP Q4NX decoder logits vs BF16 reference.

Replays the token ids the C++ harness used (tools/vl_logit_check.cpp) through
the checkpoint's own Qwen3 text model in torch and compares next-token logits:
cosine, top-5 overlap, argmax agreement. Exit 0 = match.

Usage:
  vl_logit_check <model.1bp> <tok.htok> <prompt> <logits.bin> <ids.txt>
  vl_logit_check_ref.py <ckpt_dir> <logits.bin> <ids.txt>
"""
import argparse, json, struct, sys
import numpy as np
import torch

CKPT = '/home/bcloud/checkpoints/Mage-VL'


def load_shard_tensors(fns):
    out = {}
    for fn in fns:
        with open(fn, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(n))
            data = f.read()
        for name, e in hdr.items():
            if name == '__metadata__':
                continue
            o0, o1 = e['data_offsets']
            raw = data[o0:o1]
            if e['dtype'] == 'BF16':
                u16 = np.frombuffer(raw, dtype='<u2').reshape(e['shape'])
                arr = (u16.astype(np.uint32) << 16).view(np.float32)
            elif e['dtype'] == 'F32':
                arr = np.frombuffer(raw, dtype='<f4').reshape(e['shape']).copy()
            else:
                raise ValueError(e['dtype'])
            out[name] = arr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('logits_bin')
    ap.add_argument('ids_txt')
    args = ap.parse_args()

    cpp_logits = np.fromfile(args.logits_bin, dtype='<f4')
    ids = [int(x) for x in open(args.ids_txt)]

    sys.path.insert(0, args.ckpt)
    import modeling_mage_vl as mm
    cfg = json.load(open(f'{args.ckpt}/config.json'))
    tcfg = mm.MageVLConfig(**cfg).text_config

    from transformers import Qwen3Config, Qwen3ForCausalLM
    qcfg = Qwen3Config(
        vocab_size=tcfg.vocab_size,
        hidden_size=tcfg.hidden_size,
        intermediate_size=tcfg.intermediate_size,
        num_hidden_layers=tcfg.num_hidden_layers,
        num_attention_heads=tcfg.num_attention_heads,
        num_key_value_heads=tcfg.num_key_value_heads,
        head_dim=tcfg.head_dim,
        max_position_embeddings=tcfg.max_position_embeddings,
        rope_theta=1e6,
        rms_norm_eps=tcfg.rms_norm_eps,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(qcfg)

    sd = load_shard_tensors([f'{args.ckpt}/model-00001-of-00002.safetensors',
                             f'{args.ckpt}/model-00002-of-00002.safetensors'])
    state = {}
    for k, v in sd.items():
        if k == 'lm_head.weight':
            state[k] = torch.tensor(v)
        elif k.startswith('model.language_model.'):
            state['model.' + k[len('model.language_model.'):]] = torch.tensor(v)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        out = model(torch.tensor([ids]))
        ref_logits = out.logits[0, -1].numpy()

    print(f'cpp: {cpp_logits.shape}  torch: {ref_logits.shape} '
          f'(prompt {len(ids)} tokens)')
    # normalize for cosine (logit scales may differ from temperature)
    a = cpp_logits - cpp_logits.mean()
    b = ref_logits - ref_logits.mean()
    cos = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f'logit cosine (mean-centered): {cos:.6f}')
    scale = np.linalg.norm(ref_logits) / np.linalg.norm(cpp_logits)
    print(f'scale ref/cpp: {scale:.4f}')
    top5_cpp = np.argsort(cpp_logits)[-5:][::-1]
    top5_ref = np.argsort(ref_logits)[-5:][::-1]
    print('top5 cpp: ', top5_cpp.tolist())
    print('top5 ref: ', top5_ref.tolist())
    overlap = len(set(top5_cpp) & set(top5_ref))
    argmax_match = top5_cpp[0] == top5_ref[0]
    print(f'top5 overlap: {overlap}/5, argmax match: {argmax_match}')
    ok = cos > 0.95 and argmax_match
    print('MATCH' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
