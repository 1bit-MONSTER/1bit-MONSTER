import os
import numpy as np, json, struct, time

data = open('/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx','rb').read()
hdr_len = struct.unpack('<Q', data[:8])[0]
meta = json.loads(data[8:8+hdr_len])
dbase = 8 + hdr_len

def tensor(name):
    t = meta[name]
    return np.frombuffer(data, dtype=np.uint8, count=t['data_offsets'][1]-t['data_offsets'][0],
                         offset=dbase+t['data_offsets'][0]), t

def bf16_to_f32(buf):
    u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
    return ((u << 16).astype(np.uint32)).view(np.float32)

# vectorized dequant of a projection: tiles [i8_rows,5120] -> f32 [log_rows, in_features]
def dequant_proj(name, in_features):
    qd, t = tensor(name)
    n_tile_cols = in_features // 256
    i8_rows = t['shape'][0]
    log_rows = i8_rows * 8192 // in_features
    n_tr = log_rows // 32
    tiles = qd.reshape(i8_rows, 5120)
    W = np.zeros((log_rows, in_features), dtype=np.float32)
    for tr in range(n_tr):
        for tc in range(n_tile_cols):
            row = tiles[tr*n_tile_cols+tc]
            s = bf16_to_f32(row[0:512].tobytes()).reshape(8, 32)   # [g, lr] (s[g*32+lr])
            z = bf16_to_f32(row[512:1024].tobytes()).reshape(8, 32)
            packed = row[1024:5120]
            import os
            if os.environ.get("SCALE_IDX") == "row":
                # row-major: s[lr*8+g] -> s.reshape(32,8)[lr,g]
                s = s.reshape(32, 8); z = z.reshape(32, 8)
                S = np.repeat(s[:, :, None], 32, axis=2).reshape(32, 256)
                Z = np.repeat(z[:, :, None], 32, axis=2).reshape(32, 256)
            else:
                S = np.repeat(s.T[:, :, None], 32, axis=2).reshape(32, 256)
                Z = np.repeat(z.T[:, :, None], 32, axis=2).reshape(32, 256)
            q = np.zeros((32, 256), dtype=np.int16)
            for lr in range(32):
                lane = lr//16; bi = (lr%16)//2; nib = lr%2
                q[lr] = (packed[lane*2048 + np.arange(256)*8 + bi] >> (4*nib)) & 0xF
            import os
            fm = os.environ.get("FM", "0")
            fm = os.environ.get("LF", "0")
            if fm == "1":   W[tr*32:(tr+1)*32, tc*256:(tc+1)*256] = q.astype(np.float32)*S - Z
            elif fm == "2": W[tr*32:(tr+1)*32, tc*256:(tc+1)*256] = (q.astype(np.float32) - Z)*S
            else:           W[tr*32:(tr+1)*32, tc*256:(tc+1)*256] = q.astype(np.float32)*S + Z
    if os.environ.get("BF16W") == "1":
        bits = W.view(np.uint32); r = ((bits >> 16) & 1) + 0x7FFF
        W = (((bits + r) >> 16) << 16).astype(np.uint32).view(np.float32)
    return W

t0 = time.time()
emb, emb_t = tensor('model.embed_tokens.weight')
emb_f = bf16_to_f32(emb.tobytes()).reshape(151936, 1024)
tok = 1000
x = emb_f[tok].copy()
eps = 1e-6
NLAY = int(os.environ.get("NLAY", "28"))
for l in range(NLAY):
    iln, _ = tensor(f'model.layers.{l}.input_layernorm.weight')
    iln_f = bf16_to_f32(iln.tobytes())
    x = x / np.sqrt((x**2).mean() + eps) * iln_f
    q = dequant_proj(f'model.layers.{l}.self_attn.q_proj.weight', 1024)
    k = dequant_proj(f'model.layers.{l}.self_attn.k_proj.weight', 1024)
    v = dequant_proj(f'model.layers.{l}.self_attn.v_proj.weight', 1024)
    o = dequant_proj(f'model.layers.{l}.self_attn.o_proj.weight', 2048)
    qq = (q @ x).reshape(16, 128); kk = (k @ x).reshape(8, 128); vv = (v @ x).reshape(8, 128)
    np.save('/tmp/txn_decode/my_k%d.npy' % l, kk.flatten())
    np.save('/tmp/txn_decode/my_v%d.npy' % l, vv.flatten())
    # qwen3: q_norm / k_norm RMS before attention
    qn, _ = tensor(f'model.layers.{l}.self_attn.q_norm.weight')
    kn, _ = tensor(f'model.layers.{l}.self_attn.k_norm.weight')
    qn_f = bf16_to_f32(qn.tobytes()); kn_f = bf16_to_f32(kn.tobytes())
    qq = qq / np.sqrt((qq**2).mean(axis=1, keepdims=True) + eps) * qn_f[None, :]
    kk = kk / np.sqrt((kk**2).mean(axis=1, keepdims=True) + eps) * kn_f[None, :]
    out = np.zeros(2048, dtype=np.float32)
    for h in range(16):
        out[h*128:(h+1)*128] = vv[h//2]
    o_out = o @ out
    if l == 0: print(f"  L0 v-std={vv.std():.3f} v-range=[{vv.min():.2f},{vv.max():.2f}] o@out std={o_out.std():.3f} range=[{o_out.min():.2f},{o_out.max():.2f}]", flush=True)
    x = x + o_out
    if l in (0, 1): print(f"  L{l} after attn: std={x.std():.3f} range=[{x.min():.2f},{x.max():.2f}]", flush=True)
    if l == 27: print(f"  L27 after attn: std={x.std():.3f} range=[{x.min():.2f},{x.max():.2f}]", flush=True)
    if os.environ.get("BF16X") == "1":
        bits = x.view(np.uint32); r = ((bits >> 16) & 1) + 0x7FFF
        x = (((bits + r) >> 16) << 16).astype(np.uint32).view(np.float32)
    paln, _ = tensor(f'model.layers.{l}.post_attention_layernorm.weight')
    paln_f = bf16_to_f32(paln.tobytes())
    x = x / np.sqrt((x**2).mean() + eps) * paln_f
    gate = dequant_proj(f'model.layers.{l}.mlp.gate_proj.weight', 1024)
    up = dequant_proj(f'model.layers.{l}.mlp.up_proj.weight', 1024)
    down = dequant_proj(f'model.layers.{l}.mlp.down_proj.weight', 3072)
    g = gate @ x; u = up @ x
    silu = g / (1 + np.exp(-g))
    x = x + down @ (silu * u)
    if l in (0, 1): print(f"  L{l} RMSd-x std={x.std():.3f} paln-std={paln_f.std():.3f} gateW-std={gate.std():.3f} gate-std={g.std():.3f} up-std={u.std():.3f} silu*up-std={(silu*u).std():.3f}", flush=True)
    if l in (0, 1): print(f"  L{l} gate std={g.std():.3f} up std={u.std():.3f} silu*up std={(silu*u).std():.3f} after-MLP std={x.std():.3f}", flush=True)
    if l == 27: print(f"  L27 gate std={g.std():.3f} up std={u.std():.3f} silu*up std={(silu*u).std():.3f} after-MLP std={x.std():.3f}", flush=True)
    print(f"layer {l}: x mean={x.mean():.4f} std={x.std():.4f} range=[{x.min():.2f},{x.max():.2f}]", flush=True)

fn, _ = tensor('model.norm.weight')
fn_f = bf16_to_f32(fn.tobytes())
x = x / np.sqrt((x**2).mean() + eps) * fn_f
np.save('/tmp/txn_decode/cpu_ref_hidden.npy', x)

# lm_head dequant [151936, 1024] + matmul
if os.environ.get("NLAY") is not None and int(os.environ.get("NLAY")) < 28:
    # short-run: compare the hidden after NLAY layers via the lm_head (with final norm)
    fn2, _ = tensor('model.norm.weight')
    fn2_f = bf16_to_f32(fn2.tobytes())
    x2 = x / np.sqrt((x**2).mean() + eps) * fn2_f
    lm_qd, lm_t = tensor('lm_head.weight')
    lmW = np.zeros((151936, 1024), dtype=np.float32)
    tiles = lm_qd.reshape(18992, 5120)
    for tr in range(4748):
        for tc in range(4):
            row = tiles[tr*4+tc]
            s = bf16_to_f32(row[0:512].tobytes()).reshape(8, 32)
            z = bf16_to_f32(row[512:1024].tobytes()).reshape(8, 32)
            packed = row[1024:5120]
            S = np.repeat(s.T[:, :, None], 32, axis=2).reshape(32, 256)
            Z = np.repeat(z.T[:, :, None], 32, axis=2).reshape(32, 256)
            q = np.zeros((32, 256), dtype=np.int16)
            for lr in range(32):
                lane = lr//16; bi = (lr%16)//2; nib = lr%2
                q[lr] = (packed[lane*2048 + np.arange(256)*8 + bi] >> (4*nib)) & 0xF
            lmW[tr*32:(tr+1)*32, tc*256:(tc+1)*256] = q.astype(np.float32)*S + Z
    logits = lmW @ x2
    np.save('/tmp/txn_decode/cpu_ref_logits.npy', logits)
    print(f"NLAY short run done, argmax: {logits.argmax()}", flush=True)
    import sys; sys.exit(0)
lm_qd, lm_t = tensor('lm_head.weight')
lmW = np.zeros((151936, 1024), dtype=np.float32)
tiles = lm_qd.reshape(18992, 5120)
for tr in range(4748):
    for tc in range(4):
        row = tiles[tr*4+tc]
        s = bf16_to_f32(row[0:512].tobytes()).reshape(8, 32)
        z = bf16_to_f32(row[512:1024].tobytes()).reshape(8, 32)
        packed = row[1024:5120]
        S = np.repeat(s.T[:, :, None], 32, axis=2).reshape(32, 256)
        Z = np.repeat(z.T[:, :, None], 32, axis=2).reshape(32, 256)
        q = np.zeros((32, 256), dtype=np.int16)
        for lr in range(32):
            lane = lr//16; bi = (lr%16)//2; nib = lr%2
            q[lr] = (packed[lane*2048 + np.arange(256)*8 + bi] >> (4*nib)) & 0xF
        import os
        fm = os.environ.get("LMF", "0")
        if fm == "1":   lmW[tr*32:(tr+1)*32, tc*256:(tc+1)*256] = q.astype(np.float32)*S - Z
        elif fm == "2": lmW[tr*32:(tr+1)*32, tc*256:(tc+1)*256] = (q.astype(np.float32) - Z)*S
        else:           lmW[tr*32:(tr+1)*32, tc*256:(tc+1)*256] = q.astype(np.float32)*S + Z
if os.environ.get("BF16W") == "1":
    bits = lmW.view(np.uint32); r = ((bits >> 16) & 1) + 0x7FFF
    lmW = (((bits + r) >> 16) << 16).astype(np.uint32).view(np.float32)
logits = lmW @ x
np.save('/tmp/txn_decode/cpu_ref_logits.npy', logits)
print("CPU ref saved. argmax:", logits.argmax(), "top3:", np.argsort(logits)[-3:])
print("logits[0:8]:", logits[:8])
