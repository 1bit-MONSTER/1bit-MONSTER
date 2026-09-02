## 35B weight reorder — exact puzzle defined (Round 37 continued)

### Confirmed facts
- Weight-BO OFFSETS are fully decoded from the desc (tools/dump_moe_layout.cpp):
  descriptor rule = 8 words, OFF at word 8. Layer-0: up_exps@0,
  gate_exps@0x9400000 (148 MiB), down_exps@0x12800000 (296 MiB),
  share_*@444-445 MiB, qkv@467 MiB, gate_proj@454 MiB.
- The runtime's expert rows are 4736 B, NOT the file's 5120-B q4nx tiles:
  up_exps file [4096,8,5120] = 167,772,160 B -> runtime region 155,189,248 B
  (= 32768 rows x 4736). 4736 = 4096 data + 512 scales + 128 zps.
- The reorder_cpy (the crashing function) overreads for small BF16/F32
  tensors: its chunk size for dtype=8 is 9216 B (18*512 = the linear-attn
  addr_qk stride) > the tensor sizes (2048/4096 B) -> heap overread -> the
  positional SIGSEGV (buffer near a page boundary). The size arg comes from
  the tensor SHAPE (not the data offsets), so model-file patches (zero-size
  or shape) do not dodge it; gdb len-skip lets the load proceed but the
  small-tensor outputs are garbage and the load state corrupts later.
- The zps delta: the q4nx tile stores 256 bf16 zps (512 B); the runtime's
  4736-row has only 128 B of zps (64 bf16 or 128 i8). The dequant_mm kernel
  presumably uses a shared/truncated zero-point scheme.

### Next step (the only remaining puzzle)
Disassemble the I8 path of qwen3_6_reorder_cpy (the elementsize dispatch:
dtype&1->10/18, dtype&2->9/17, dtype&4->+1, then x512) to extract the exact
5120->4736 transformation (which zps survive + where the scales/zps land),
OR capture the reordered expert rows from a partial load (gdb-skip the small
tensor crashes + grab the 512 MB weight BOs before the load aborts).

### Cleanup note
/tmp model copies (23 GB x2) removed; the patched dir /tmp/moe-p +
xclbins/moe-p symlinks remain for the next capture attempt.

## UPDATE — reorder captured + decoded (shape-0 patch + interposer)

The shape-0 model-file patch (zero the small BF16/F32 tensor SHAPES so the
reorder's size arg = 0 -> loop skipped) lets the runtime's load_weights
COMPLETE (40 layers, 36 GB of captures). The 512 MB weight BOs are the
ground truth:

- The runtime's expert rows are 4736 B and equal the FILE's raw bytes at
  strided offsets: row0 = share_up @ file-rel 1113288 (its last 824 B +
  the next tensor's start), the up_exps rows at file-rel 3912 + 4736k
  (k=0..7 then a 4-row break), gate_exps interleaved. The BO layout =
  the layer tensors' data re-sliced at 4736-byte boundaries in a strided
  interleave (NOT a per-tile permutation; the 4736 = 4096 data + 512
  scales + 128 zps with NO value transform — byte-identical slices).
- The capture set is at ~/.cache/moe-cap (50 x 512 MB weight BOs + ELFs).

Remaining: extract the reorder_cpy's chunk-offset rule (the stride math
from the disassembly, or a full row->file map) so the engine's packer
reproduces the interleave; then the weight BOs + forward loop complete the
35B engine path.
