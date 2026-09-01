#!/usr/bin/env python3
"""zaya_to_gguf.py — convert a zaya .q4nx container to a llama.cpp GGUF with
canonical ZAYA arch names (llama.cpp PR #23112 mapping, 8B-era).

Modes:
  --f32   all tensors dequantized: I8 tiles -> F32, BF16 kept as BF16
          (CPU validation of the arch/graph).
  --q4nx  I8 tiles -> GGML_TYPE_Q4NX (42), embedding dequantized to BF16,
          BF16 kept (HRX lane; expert MUL_MAT_ID Q4NX dispatch is a
          follow-up milestone).

Tensor logical shapes are derived from the ARCHITECTURE + data byte sizes
(the container JSON "shape" field is unreliable — the engine notes fc1 is
declared [256, 2048] but holds 256x256 BF16).
"""
import json, struct, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.home()) + "/hrx-ws/hrx-v2-src/gguf-py")
from gguf import GGUFWriter, LlamaHfVocab, SpecialVocab, TokenType

H = 2048; NC = 40; NQ = 8; NKV = 2; HD = 128
QD = NQ * HD; KD = NKV * HD
N_FF = 2048; N_EXP = 16; RTR_H = 256; VOCAB = 262272; D_CONV = 2

def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

def dequant_tiles(blob, rows, k, n_tc):
    Wd = np.zeros((rows, k), dtype=np.float32)
    n_tiles = len(blob) // 5120
    for t in range(n_tiles):
        tr, tc = t // n_tc, t % n_tc
        base = blob[t*5120:(t+1)*5120]
        scales = bf16_to_f32(np.frombuffer(base[0:512], dtype=np.uint16)).reshape(32, 8)
        packed = base[1024:]
        for r in range(32):
            lane = r//16; lr = r%16; bi = lr//2; nib = r%2
            row = tr*32 + r
            colbytes = packed[lane*2048 + np.arange(256)*8 + bi]
            qv = np.where(nib == 0, colbytes & 0x0F, (colbytes >> 4) & 0x0F).astype(np.int32)
            v = np.where(qv < 8, qv, qv-16).astype(np.float32)
            srow = scales[r]
            Wd[row, tc*256:(tc+1)*256] = v * srow.repeat(32)
    return Wd

def load_container(path):
    with open(path, 'rb') as f:
        (jl,) = struct.unpack('<Q', f.read(8))
        js = f.read(jl).decode()
    obj = json.loads(js)
    df = 8 + jl
    tensors = {k: v for k, v in obj.items() if isinstance(v, dict) and 'shape' in v}
    return obj, df, tensors

def main():
    args = sys.argv[1:]
    mode = 'q4nx'
    if '--f32' in args:
        mode = 'f32'
        args.remove('--f32')
    if len(args) < 2:
        print(__doc__)
        return 1
    model_path, out_path = args[0], args[1]
    tok_path = args[2] if len(args) > 2 else '/tmp/zaya-tok/tokenizer.json'

    obj, df, tensors = load_container(model_path)
    f = open(model_path, 'rb')

    w = GGUFWriter(out_path, 'zaya')
    w.add_string("general.architecture", "zaya")
    w.add_string("general.name", Path(model_path).stem)
    w.add_uint32("general.file_type", 2 if mode == 'q4nx' else 1)
    w.add_uint32("zaya.block_count", NC)
    w.add_uint32("zaya.embedding_length", H)
    w.add_uint32("zaya.feed_forward_length", N_FF)
    w.add_uint32("zaya.expert_feed_forward_length", RTR_H)
    w.add_uint32("zaya.attention.head_count", NQ)
    w.add_uint32("zaya.attention.head_count_kv", NKV)
    w.add_uint32("zaya.attention.key_length", HD)
    w.add_uint32("zaya.attention.value_length", HD)
    w.add_float32("zaya.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_uint32("zaya.ssm.conv_kernel", D_CONV)
    w.add_uint32("zaya.expert_count", N_EXP)
    w.add_uint32("zaya.expert_used_count", 1)
    w.add_float32("zaya.rope.freq_base", 5000000.0)
    w.add_uint32("zaya.context_length", 256)
    w.add_uint32("zaya.vocab_size", VOCAB)

    # tokenizer
    vocab = LlamaHfVocab(Path(tok_path).parent)
    tokens, scores, toktypes = [], [], []
    for text, score, toktype in vocab.all_tokens():
        tokens.append(text); scores.append(score); toktypes.append(toktype)
    while len(tokens) < VOCAB:
        tokens.append(bytes(f"[PAD{len(tokens)}]", encoding="utf-8"))
        scores.append(-1000.0)
        toktypes.append(TokenType.UNUSED)
    if len(tokens) > VOCAB:
        raise RuntimeError(f"tokenizer vocab {len(tokens)} > model vocab {VOCAB}")
    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(toktypes)
    SpecialVocab(Path(tok_path).parent, n_vocab=len(tokens)).add_to_gguf(w)

    def add_bf16(name, shape, bytes_):
        # shape is the GGUF (file) dims == ggml ne reversed: (ne_last .. ne_0)
        # Keep the small BF16 tensors (scales/biases/norms/conv) as F32 in
        # BOTH modes: the CPU backend cannot binary-op f32 activations with
        # bf16 weights, and these tensors are tiny.
        arr = bf16_to_f32(np.frombuffer(bytes_, dtype=np.uint16)).reshape(-1)
        arr = arr.reshape(shape[::-1])
        w.add_tensor(name, arr, raw_dtype=0)

    def add_i8(name, logical_shape, bytes_):
        rows, cols = logical_shape
        n_tc = cols // 256
        n_tiles = (rows // 32) * n_tc
        assert len(bytes_) == n_tiles * 5120, f"{name}: {len(bytes_)} vs {n_tiles*5120}"
        if mode == 'q4nx':
            w.add_tensor(name, np.frombuffer(bytes_, dtype=np.uint8), raw_shape=[n_tiles, 5120], raw_dtype=42)
        else:
            w.add_tensor(name, dequant_tiles(np.frombuffer(bytes_, dtype=np.uint8), rows, cols, n_tc), raw_dtype=0)

    def add_i8_3d(name, n_ff, n_exp, bytes_):
        rows = n_exp * n_ff; cols = H
        n_tc = cols // 256
        n_tiles = (rows // 32) * n_tc
        assert len(bytes_) == n_tiles * 5120, f"{name}: {len(bytes_)} vs {n_tiles*5120}"
        if mode == 'q4nx':
            # 3-D Q4NX: [n_exp, tiles_per_expert, 5120 bytes/tile] -> ggml ne
            # [8192, tiles_per_expert, n_expert] (tiles per expert contiguous).
            # quant_shape_from_byte_shape turns the trailing 5120 bytes into
            # 8192 elements, so file dims are [n_exp, tpe, 8192] -> ne
            # [8192, tpe, n_exp].
            tpe = n_tiles // n_exp
            assert n_tiles % n_exp == 0, f"{name}: {n_tiles} % {n_exp}"
            w.add_tensor(name, np.frombuffer(bytes_, dtype=np.uint8), raw_shape=[n_exp, tpe, 5120], raw_dtype=42)
        else:
            f32 = dequant_tiles(np.frombuffer(bytes_, dtype=np.uint8), rows, cols, n_tc)
            w.add_tensor(name, f32.reshape(n_exp, n_ff, H), raw_dtype=0)

    def get_off(name):
        return tuple(tensors[name]["data_offsets"])

    def read_bf16(name):
        o0, o1 = get_off(name)
        f.seek(df + o0); return f.read(o1 - o0)

    def read_i8(name):
        o0, o1 = get_off(name)
        f.seek(df + o0); return f.read(o1 - o0)

    emb_bytes = read_i8("model.embed_tokens.weight")
    if mode == 'q4nx':
        # embedding dequantized to F32 [VOCAB, H] (GGUF dims [VOCAB, H] ->
        # ggml ne [H, VOCAB]). F32 keeps the CPU/HRX binary ops and the tied
        # output matmul working (no BF16 matmul path in the backends).
        f32 = dequant_tiles(np.frombuffer(emb_bytes, dtype=np.uint8), VOCAB, H, H // 256)
        w.add_tensor("token_embd.weight", f32.reshape(VOCAB, H), raw_dtype=0)
    else:
        add_i8("token_embd.weight", (VOCAB, H), emb_bytes)

    add_bf16("output_norm.weight", (H,), read_bf16("model.norm.weight"))
    if "model.input_hidden_states_scale" in tensors:
        add_bf16("input_hidden_states_scale.weight", (H,), read_bf16("model.input_hidden_states_scale"))
        add_bf16("input_hidden_states_scale.bias", (H,), read_bf16("model.input_hidden_states_bias"))

    for l in range(NC):
        p = f"model.layers.{l}."
        # Zaya 8B: EVERY layer runs CCA attention AND MoE (HF ZayaDecoderLayer):
        #   residual = h
        #   h = input_layernorm(residual); attn = CCA(h)
        #   residual = (attn+hs_b)*hs_s + (residual+res_b)*res_s          (post_attention)
        #   h = post_attention_layernorm(residual); moe = MoE(h)
        #   h = (moe+hs_b)*hs_s + (residual+res_b)*res_s                  (post_mlp)
        add_bf16(f"blk.{l}.attn_norm.weight", (H,), read_bf16(f"{p}input_layernorm.weight"))
        add_bf16(f"blk.{l}.post_attn_norm.weight", (H,), read_bf16(f"{p}post_attention_layernorm.weight"))
        for pair, gname in [("post_attention_residual_scale", "res_scale_hs"),
                            ("post_mlp_residual_scale", "res_scale_hs_mlp")]:
            base = f"{p}{pair}."
            add_bf16(f"blk.{l}.{gname}.weight", (H,), read_bf16(f"{base}hidden_states_scale"))
            add_bf16(f"blk.{l}.{gname}.bias", (H,), read_bf16(f"{base}hidden_states_bias"))
        for pair, gname in [("post_attention_residual_scale", "res_scale_res"),
                            ("post_mlp_residual_scale", "res_scale_res_mlp")]:
            base = f"{p}{pair}."
            add_bf16(f"blk.{l}.{gname}.weight", (H,), read_bf16(f"{base}residual_scale"))
            add_bf16(f"blk.{l}.{gname}.bias", (H,), read_bf16(f"{base}residual_bias"))

        sa = f"{p}self_attn."
        add_i8(f"blk.{l}.attn_q.weight", (QD, H), read_i8(f"{sa}q_proj.weight"))
        add_i8(f"blk.{l}.attn_k.weight", (KD, H), read_i8(f"{sa}k_proj.weight"))
        add_i8(f"blk.{l}.cca_val_proj1.weight", (KD // 2, H), read_i8(f"{sa}v_proj_current.weight"))
        add_i8(f"blk.{l}.cca_val_proj2.weight", (KD // 2, H), read_i8(f"{sa}v_proj_delayed.weight"))
        add_i8(f"blk.{l}.attn_output.weight", (H, QD), read_i8(f"{sa}o_proj.weight"))
        add_bf16(f"blk.{l}.ssm_conv1d.weight", (D_CONV, QD + KD), read_bf16(f"{sa}conv_qk_depthwise.weight"))
        add_bf16(f"blk.{l}.ssm_conv1d.bias", (QD + KD,), read_bf16(f"{sa}conv_qk_depthwise.bias"))
        add_bf16(f"blk.{l}.cca_conv_grp.weight", (D_CONV, (QD + KD) // (NQ + NKV), QD + KD), read_bf16(f"{sa}conv_qk_grouped.weight"))
        add_bf16(f"blk.{l}.cca_conv_grp.bias", (QD + KD,), read_bf16(f"{sa}conv_qk_grouped.bias"))
        add_bf16(f"blk.{l}.cca_k_scale.weight", (NKV,), read_bf16(f"{sa}qk_norm.temp"))

        mg = f"{p}mlp.gate."
        add_bf16(f"blk.{l}.ffn_gate_inp.weight", (H, RTR_H), read_bf16(f"{mg}down_proj.weight"))
        add_bf16(f"blk.{l}.ffn_gate_inp.bias", (RTR_H,), read_bf16(f"{mg}down_proj.bias"))
        add_bf16(f"blk.{l}.ffn_norm.weight", (RTR_H,), read_bf16(f"{mg}router_mlp.norm.weight"))
        add_bf16(f"blk.{l}.ffn_gate.weight", (RTR_H, RTR_H), read_bf16(f"{mg}router_mlp.fc1.weight"))
        add_bf16(f"blk.{l}.ffn_gate.bias", (RTR_H,), read_bf16(f"{mg}router_mlp.fc1.bias"))
        add_bf16(f"blk.{l}.zaya_router_mlp2.weight", (RTR_H, RTR_H), read_bf16(f"{mg}router_mlp.fc2.weight"))
        add_bf16(f"blk.{l}.zaya_router_mlp2.bias", (RTR_H,), read_bf16(f"{mg}router_mlp.fc2.bias"))
        add_bf16(f"blk.{l}.zaya_router_mlp4.weight", (RTR_H, N_EXP + 1), read_bf16(f"{mg}router_mlp.out_proj.weight"))
        add_bf16(f"blk.{l}.zaya_router_biases.weight", (N_EXP + 1,), read_bf16(f"{mg}balancing_biases"))
        if f"{mg}router_states_scale" in tensors:
            add_bf16(f"blk.{l}.zaya_router_eda.weight", (RTR_H,), read_bf16(f"{mg}router_states_scale"))
        add_i8_3d(f"blk.{l}.ffn_gate_up_exps.weight", 2 * N_FF, N_EXP, read_i8(f"{p}mlp.experts.gate_up_proj.weight"))
        add_i8_3d(f"blk.{l}.ffn_down_exps.weight", N_FF, N_EXP, read_i8(f"{p}mlp.experts.down_proj.weight"))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} ({mode})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
