#!/usr/bin/env python3
"""convert_qwen36_moe_q4nx.py — Convert Qwen3.6-35B-A3B (qwen35moe) GGUF → NPU-engine q4nx.

The FLM converter has no qwen35moe support; this produces the exact layouts the
engine's MoE path reads (model_config.h + npu_engine_universal.cpp, #1699):

  Dense attention (Q8_0 tiled, 8704 B/row, dequant_q8_0_to_float_ex):
    512 B BF16 scales (32 rows x 8 groups, flat g*32+r) + 8192 B int8 (row-major).
  MoE experts (Q4NX 1BP, 5120 B/row, dequant_1bp):
    512 B BF16 scales (flat r*8+g) + 512 B BF16 zps + 4096 B nibbles
    (byte (r*256+col)/2, low nibble = even col). Experts stacked into a 3D
    tensor [experts*tile_rows, col_blocks, tile_bytes] = [4096, 8, 5120].
  Router (moe_router.weight): BF16 [H, N_EXPERTS] stride-8 interleave
    flat[(i%8)*65536 + j*256 + i/8].
  Norms / embeddings / GDN extras: raw BF16 / F32.
  lm_head: FLM group-major Q4_1 pack (dequant_i8_to_float_ex).

Run:  venv/bin/python tools/convert_qwen36_moe_q4nx.py model.gguf out_dir
"""
import sys, os
import numpy as np
import torch
from safetensors.torch import save_file
from gguf import GGUFReader, dequantize

H = 2048
IM_EXP = 512
N_EXPERTS = 256
NV = 248320


def bf16_u8(w):
    """float32 [n] -> raw BF16 bytes as np.uint8."""
    u16 = (w.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return np.ascontiguousarray(u16).view(np.uint8)


def pack_q8_tiled(w):
    """[R, C] f32 -> Q8_0 tiled rows (8704 B/row): 512 B BF16 scales (flat
    g*32+r) + 8192 B int8 row-major. Tile grid 32x256."""
    R, C = w.shape
    ntc = C // 256
    ntr = R // 32
    rows = ntr * ntc
    out = np.zeros(rows * 8704, np.uint8)
    tiles = w.reshape(ntr, 32, ntc, 256).transpose(0, 2, 1, 3)  # [ntr, ntc, 32, 256]
    tiles = tiles.reshape(-1, 32, 8, 32)  # [rows, 32, 8 groups, 32]
    amax = np.abs(tiles).max(axis=3)  # [rows, 32, 8]
    amax = np.where(amax < 1e-12, 1.0, amax)
    scales = amax / 127.0  # [rows, 32, 8]
    q = np.clip(np.round(tiles / scales[:, :, :, None]).astype(np.int32), -127, 127)
    q8 = q.astype(np.int8).reshape(rows, 32, 256)
    # scales flat g*32+r: transpose [rows, 32, 8] -> [rows, 8, 32] -> flat
    sc_flat = np.ascontiguousarray(scales.transpose(0, 2, 1)).reshape(rows, 256)
    out2 = out.reshape(rows, 8704)
    out2[:, :512] = bf16_u8(sc_flat.reshape(-1)).reshape(rows, 512)
    out2[:, 512:] = q8.view(np.uint8).reshape(rows, 8192)
    return out2.reshape(-1), (rows, 8704)


def pack_q4nx_1bp(w):
    """[R, C] f32 -> Q4NX 1BP tile rows (5120 B/row): row-major BF16 scales/zps
    (flat r*8+g) + 4096 B nibbles (byte (r*256+col)/2, low = even col).
    value = v*s + z, v in [0,15]."""
    R, C = w.shape
    ntc = C // 256
    ntr = R // 32
    rows = ntr * ntc
    out = np.zeros(rows * 5120, np.uint8)
    tiles = w.reshape(ntr, 32, ntc, 256).transpose(0, 2, 1, 3).reshape(-1, 32, 8, 32)
    lo = tiles.min(axis=3)
    hi = tiles.max(axis=3)
    rng = hi - lo
    rng = np.where(rng < 1e-12, 1.0, rng)
    s = rng / 15.0
    z = np.where(rng < 1e-12, hi, lo)
    v = np.clip(np.round((tiles - z[:, :, :, None]) / s[:, :, :, None]).astype(np.int32), 0, 15)
    qd = v.astype(np.uint8).reshape(rows, 32, 256)
    packed = (qd[:, :, 0::2] & 0x0F) | ((qd[:, :, 1::2] & 0x0F) << 4)
    # scales/zps flat r*8+g
    sc_flat = np.ascontiguousarray(s.reshape(rows, 32, 8).reshape(rows, 256))
    zp_flat = np.ascontiguousarray(z.reshape(rows, 32, 8).reshape(rows, 256))
    out2 = out.reshape(rows, 5120)
    out2[:, :512] = bf16_u8(sc_flat.reshape(-1)).reshape(rows, 512)
    out2[:, 512:1024] = bf16_u8(zp_flat.reshape(-1)).reshape(rows, 512)
    out2[:, 1024:] = packed.view(np.uint8).reshape(rows, 4096)
    return out2.reshape(-1), (rows, 5120)


def pack_experts(w, n_experts, im_exp):
    """[H, im_exp, n_experts] f32 -> 3D Q4NX 1BP [experts*tile_rows, col_blocks,
    tile_bytes] = [n_experts*(im_exp/32)*(H/256), H/256, 5120]."""
    Hh, ie, ne = w.shape
    assert ne == n_experts and ie == im_exp and Hh == H, "expert shape %s" % (str(w.shape),)
    ntc = H // 256
    # expert-major stack: [n, im_exp, H] -> [n*im_exp, H] tiles identically to
    # per-expert pack_q4nx_1bp calls (each expert contributes 16 row-tiles).
    w2 = np.ascontiguousarray(w.transpose(2, 1, 0).reshape(ne * ie, H))
    flat, _ = pack_q4nx_1bp(w2)
    # shape[0] = experts x row-tiles (the engine multiplies shape[0]*shape[1]
    # to get the tile-row count); total flat = shape[0]*shape[1]*5120
    return flat, (ne * (ie // 32), ntc, 5120)


_perm_cache = {}

def permute_to(w, target):
    """Permute dequantized 3D tensor to (H, im_exp, n) regardless of the
    gguf lib's dequant output dimension order. The dequant shape is stable
    within a process, so cache the permutation."""
    key = (w.shape, target)
    if key in _perm_cache:
        return w.transpose(_perm_cache[key])
    import itertools
    for perm in itertools.permutations(range(3)):
        if tuple(w.transpose(perm).shape) == target:
            _perm_cache[key] = perm
            return w.transpose(perm)
    raise ValueError(f"cannot permute {w.shape} to {target}")


def pack_router(w):
    """[H, N_EXPERTS] f32 -> BF16 stride-8 interleave:
    flat[(i%8)*65536 + j*256 + i/8] = w[i, j]."""
    bf = (w.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    out = np.zeros(H * N_EXPERTS, np.uint16)
    for i in range(H):
        for j in range(N_EXPERTS):
            out[(i % 8) * 65536 + j * 256 + i // 8] = bf[i, j]
    return np.ascontiguousarray(out).view(np.uint8)


def pack_lm_head_q4(w):
    """FLM group-major Q4_1 pack (dequant_i8_to_float_ex)."""
    saved = sys.path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'FLM_Q4NX_Converter'))
    try:
        from gguf import quantize, GGMLQuantizationType
        from q4nx.gguf_tensor import GGUFTensor
        from q4nx.models.llama import Llama
        wq = quantize(w.astype(np.float32), GGMLQuantizationType.Q4_1).copy()
        d, m, qs = GGUFTensor.unpack_q4_1(wq, w.shape[1])
        conv = Llama.__new__(Llama)
        conv.row_block_size = 32; conv.col_block_size = 256
        conv.parallel_size = 16; conv.keep_block_in_2D = False
        packed = conv._pack_q4nx(d, m, qs)
        return np.ascontiguousarray(packed.numpy()).view(np.uint8).reshape(-1)
    finally:
        sys.path = saved


def main():
    gguf_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    g = GGUFReader(gguf_path)
    tensors = {t.name: t for t in g.tensors}

    def deq(name):
        return np.asarray(dequantize(tensors[name].data, tensors[name].tensor_type))

    q4 = {}
    emb = deq('token_embd.weight')
    q4['model.embed_tokens.weight'] = torch.from_numpy(emb.astype(np.float32)).contiguous().to(torch.bfloat16)
    q4['model.norm.weight'] = torch.from_numpy(
        deq('output_norm.weight').reshape(-1).astype(np.float32)).contiguous().to(torch.bfloat16)
    print("packing lm_head ...", flush=True)
    lm = np.ascontiguousarray(deq('output.weight'))
    lmflat = pack_lm_head_q4(lm)
    ntr = (NV // 32) * (H // 256)
    q4['lm_head.weight'] = torch.from_numpy(lmflat.reshape(ntr, 5120).copy())

    is_gdn = [f'blk.{l}.attn_qkv.weight' in tensors for l in range(40)]
    print("GDN layers:", sum(is_gdn), "full-attn:", sum(not x for x in is_gdn), flush=True)

    for l in range(40):
        pref = f'model.layer.{l}.'
        print(f"layer {l} ({'GDN' if is_gdn[l] else 'std'})", flush=True)
        q4[pref + 'input_layernorm.weight'] = torch.from_numpy(
            deq(f'blk.{l}.attn_norm.weight').reshape(-1).astype(np.float32)).contiguous().to(torch.bfloat16)
        q4[pref + 'post_attention_layernorm.weight'] = torch.from_numpy(
            deq(f'blk.{l}.post_attention_norm.weight').reshape(-1).astype(np.float32)).contiguous().to(torch.bfloat16)
        if is_gdn[l]:
            qkv = np.ascontiguousarray(deq(f'blk.{l}.attn_qkv.weight'))
            data, shp = pack_q8_tiled(qkv.T)
            q4[pref + 'linear_attn.qkv_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
            zg = np.ascontiguousarray(deq(f'blk.{l}.attn_gate.weight'))
            data, shp = pack_q8_tiled(zg.T)
            q4[pref + 'self_attn.gate_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
            so = np.ascontiguousarray(deq(f'blk.{l}.ssm_out.weight'))
            data, shp = pack_q8_tiled(so.T)
            q4[pref + 'linear_attn.ssm_out_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
            q4[pref + 'linear_attn.ssm_alpha_proj.weight'] = torch.from_numpy(
                deq(f'blk.{l}.ssm_alpha.weight').reshape(-1).astype(np.float32)).contiguous().to(torch.bfloat16)
            q4[pref + 'linear_attn.ssm_beta_proj.weight'] = torch.from_numpy(
                deq(f'blk.{l}.ssm_beta.weight').reshape(-1).astype(np.float32)).contiguous().to(torch.bfloat16)
            conv1d = np.ascontiguousarray(deq(f'blk.{l}.ssm_conv1d.weight'))
            q4[pref + 'linear_attn.ssm_conv1d.weight'] = torch.from_numpy(np.ascontiguousarray(
                conv1d.reshape(-1).astype(np.float32))).contiguous().to(torch.bfloat16)
            q4[pref + 'linear_attn.ssm_a'] = torch.from_numpy(np.ascontiguousarray(
                deq(f'blk.{l}.ssm_a').astype(np.float32)).copy())
            q4[pref + 'linear_attn.ssm_dt.bias'] = torch.from_numpy(np.ascontiguousarray(
                deq(f'blk.{l}.ssm_dt.bias').astype(np.float32)).copy())
            q4[pref + 'linear_attn.ssm_norm.weight'] = torch.from_numpy(
                deq(f'blk.{l}.ssm_norm.weight').reshape(-1).astype(np.float32)).contiguous().to(torch.bfloat16)
        else:
            aq = np.ascontiguousarray(deq(f'blk.{l}.attn_q.weight'))
            data, shp = pack_q8_tiled(aq.T)
            q4[pref + 'self_attn.q_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
            ak = np.ascontiguousarray(deq(f'blk.{l}.attn_k.weight'))
            data, shp = pack_q8_tiled(ak.T)
            q4[pref + 'self_attn.k_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
            av = np.ascontiguousarray(deq(f'blk.{l}.attn_v.weight'))
            data, shp = pack_q8_tiled(av.T)
            q4[pref + 'self_attn.v_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
            ao = np.ascontiguousarray(deq(f'blk.{l}.attn_output.weight'))
            data, shp = pack_q8_tiled(ao.T)
            q4[pref + 'self_attn.o_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
            q4[pref + 'self_attn.q_norm.weight'] = torch.from_numpy(
                deq(f'blk.{l}.attn_q_norm.weight').reshape(-1).astype(np.float32)).contiguous().to(torch.bfloat16)
            q4[pref + 'self_attn.k_norm.weight'] = torch.from_numpy(
                deq(f'blk.{l}.attn_k_norm.weight').reshape(-1).astype(np.float32)).contiguous().to(torch.bfloat16)
        # MoE experts
        ge = permute_to(deq(f'blk.{l}.ffn_gate_exps.weight'), (H, IM_EXP, N_EXPERTS))
        data, shp = pack_experts(ge, N_EXPERTS, IM_EXP)
        q4[pref + 'mlp.gate_exps_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
        ue = permute_to(deq(f'blk.{l}.ffn_up_exps.weight'), (H, IM_EXP, N_EXPERTS))
        data, shp = pack_experts(ue, N_EXPERTS, IM_EXP)
        q4[pref + 'mlp.up_exps_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
        de_ = permute_to(deq(f'blk.{l}.ffn_down_exps.weight'), (H, IM_EXP, N_EXPERTS))
        data, shp = pack_experts(de_, N_EXPERTS, IM_EXP)
        q4[pref + 'mlp.down_exps_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
        # router
        rt = np.ascontiguousarray(deq(f'blk.{l}.ffn_gate_inp.weight'))
        if rt.shape == (N_EXPERTS, H):
            rt = rt.T  # dequantize returns [n, H]; engine wants [H, n]
        q4[pref + 'moe_router.weight'] = torch.from_numpy(rt.astype(np.float32)).contiguous().to(torch.bfloat16)
        # shared experts
        sg = np.ascontiguousarray(deq(f'blk.{l}.ffn_gate_shexp.weight'))
        data, shp = pack_q4nx_1bp(sg)  # deq [im_exp, H] -> [R=im_exp, C=H]? need [H out, im_exp in] -> pack expects [R, C] weights [out, in]
        q4[pref + 'mlp.share_gate_exps_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
        su = np.ascontiguousarray(deq(f'blk.{l}.ffn_up_shexp.weight'))
        data, shp = pack_q4nx_1bp(su)
        q4[pref + 'mlp.share_up_exps_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
        sd = np.ascontiguousarray(deq(f'blk.{l}.ffn_down_shexp.weight'))
        data, shp = pack_q4nx_1bp(sd)
        q4[pref + 'mlp.share_down_exps_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
        q4[pref + 'mlp.share_down_exps_proj.weight'] = torch.from_numpy(data.reshape(shp).copy())
        sgi = deq(f'blk.{l}.ffn_gate_inp_shexp.weight')
        q4[pref + 'shared_expert_gate.weight'] = torch.from_numpy(np.ascontiguousarray(
            sgi.reshape(-1).astype(np.float32))).contiguous().to(torch.bfloat16)

    print(f"saving to {out_dir}/model.q4nx ...", flush=True)
    save_file(q4, os.path.join(out_dir, 'model.q4nx'))
    print("done", flush=True)


if __name__ == '__main__':
    main()
