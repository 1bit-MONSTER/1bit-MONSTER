"""Qwen3-0.6B contract — master dimension constants for full-layer decode dataflow."""
from dataclasses import dataclass

# --- Model dimensions ---
HIDDEN_DIM = 1024
INTERMEDIATE_DIM = 3072
HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS  # 2

# --- AIE tile grid (same as 8B: 16 tiles in 4×4, columns 2-5, rows 2-5) ---
MAIN_COLUMNS = (2, 3, 4, 5)
MAIN_ROWS = (2, 3, 4, 5)
EDGE_COLUMNS = (0, 1, 6, 7)
ROWS_PER_COLUMN = len(MAIN_ROWS)
M_PER_TILE = 32
OUTPUT_BLOCK_ROWS = len(MAIN_COLUMNS) * len(MAIN_ROWS) * M_PER_TILE  # 512

# --- Chunk / patch geometry (hardware invariants) ---
ACT_SLICE_BF16 = 256
K_CHUNK = 256
GROUP_SIZE = 32
RECORD_DWORDS = 17
RECORD_PAYLOAD_DWORDS = RECORD_DWORDS - 1
PATCHES_PER_COLUMN = 2
ROWS_PER_PATCH = 2
CHUNK_BF16 = 2560

# --- Phase definitions (Q, K, V, O, UP, GATE, DOWN) ---
PHASE_NAMES = ("Q", "K", "V", "O", "UP", "GATE", "DOWN")
PHASE_INPUT_DIMS  = (1024, 1024, 1024, 1024, 1024, 1024, 3072)
PHASE_OUTPUT_DIMS = (2048, 1024, 1024, 1024, 3072, 3072, 1024)
#    Q: 16 heads × 128 dim = 2048
#    K:  8 heads × 128 dim = 1024
#    V:  8 heads × 128 dim = 1024
#    O: HIDDEN_DIM = 1024
#    UP: INTERMEDIATE_DIM = 3072
#    GATE: INTERMEDIATE_DIM = 3072
#    DOWN: HIDDEN_DIM = 1024

PHASE_BLOCKS = tuple(output_dim // OUTPUT_BLOCK_ROWS for output_dim in PHASE_OUTPUT_DIMS)
# → (4, 2, 2, 2, 6, 6, 2)

PHASE_CHUNKS = tuple(input_dim // K_CHUNK for input_dim in PHASE_INPUT_DIMS)
# → (4, 4, 4, 4, 4, 4, 12)

NUM_PHASES = len(PHASE_NAMES)
TOTAL_LOGICAL_BLOCKS = sum(PHASE_BLOCKS)  # 24

PATCH_BF16_BY_PHASE = tuple(
    ROWS_PER_PATCH * PHASE_CHUNKS[phase] * CHUNK_BF16 for phase in range(NUM_PHASES)
)
PHASE_PATCH_COUNTS = tuple(
    PHASE_BLOCKS[phase] * len(MAIN_COLUMNS) * PATCHES_PER_COLUMN
    for phase in range(NUM_PHASES)
)
# → (32, 16, 16, 16, 48, 48, 16)  total=192 patches

PHASE_WEIGHT_BF16 = tuple(
    PHASE_PATCH_COUNTS[phase] * PATCH_BF16_BY_PHASE[phase]
    for phase in range(NUM_PHASES)
)
TOTAL_PATCHES = sum(PHASE_PATCH_COUNTS)  # 192
TOTAL_WEIGHT_BF16 = sum(PHASE_WEIGHT_BF16)
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2

# --- Attention ---
ATTENTION_BF16 = NUM_Q_HEADS * HEAD_DIM   # 2048
O_CHUNKS = ATTENTION_BF16 // ACT_SLICE_BF16  # 8
SWIGLU_SLICES = INTERMEDIATE_DIM // OUTPUT_BLOCK_ROWS  # 6

# --- Packet / compact / replay constants ---
C1R2_PACKET_DWORDS = 1 + HIDDEN_DIM // 2        # 513
C1R2_QKV_REPLAYS = PHASE_BLOCKS[0] + PHASE_BLOCKS[1] + PHASE_BLOCKS[2]   # 8
C1R2_UPGATE_REPLAYS = PHASE_BLOCKS[4] + PHASE_BLOCKS[5]                   # 12
C1R2_FINAL_REPLAYS = 1
C6R2_INPUT_DWORDS = 512
C6R2_HALF_DWORDS = C6R2_INPUT_DWORDS // 2
COMPACT_PACKET_DWORDS = 1 + len(MAIN_COLUMNS) * ROWS_PER_COLUMN * RECORD_PAYLOAD_DWORDS  # 257
ATTENTION_PACKET_DWORDS = ATTENTION_BF16 // 2   # 1024
DOWN_PACKET_DWORDS = INTERMEDIATE_DIM // 2       # 1536
SHAPE_WINDOW_DWORDS = (NUM_Q_HEADS // 4) * HEAD_DIM // 2  # 256
SHAPE_CARRIER_DWORDS = (0x100 + 0x40) // 4       # 80 (kept for compatibility)

# --- Weight chunk base addresses (per-tile chunks, accumulated across phases) ---
# Q: 4 blocks × 4 chunks = 16
# K: 2 blocks × 4 chunks = 8
# V: 2 blocks × 4 chunks = 8
# O: 2 blocks × 4 chunks = 8
# UP: 6 blocks × 4 chunks = 24
# GATE: 6 blocks × 4 chunks = 24
# DOWN: 2 blocks × 12 chunks = 24
kQWeightChunkBase = 0
kKWeightChunkBase = 16
kVWeightChunkBase = kKWeightChunkBase + 8       # 24
kFullLayerOWeightChunkBase = kVWeightChunkBase + 8   # 32
kFullLayerUpGateWeightChunkBase = kFullLayerOWeightChunkBase + 8  # 40
kFullLayerDownWeightChunkBase = kFullLayerUpGateWeightChunkBase + 48  # 88

# --- Schedule validation ---
@dataclass(frozen=True)
class PhaseSpec:
    name: str; input_dim: int; output_dim: int; blocks: int; chunks: int; patches: int

PHASE_SPECS = tuple(
    PhaseSpec(name=PHASE_NAMES[idx], input_dim=PHASE_INPUT_DIMS[idx],
              output_dim=PHASE_OUTPUT_DIMS[idx], blocks=PHASE_BLOCKS[idx],
              chunks=PHASE_CHUNKS[idx], patches=PHASE_PATCH_COUNTS[idx])
    for idx in range(NUM_PHASES)
)

def main16_tile_order():
    return tuple((c, r) for c in MAIN_COLUMNS for r in MAIN_ROWS)

def validate_contract():
    errors = []
    if PHASE_NAMES != ("Q", "K", "V", "O", "UP", "GATE", "DOWN"):
        errors.append(f"phase order: {PHASE_NAMES}")
    if PHASE_BLOCKS != (4, 2, 2, 2, 6, 6, 2):
        errors.append(f"phase blocks: {PHASE_BLOCKS}")
    if PHASE_CHUNKS != (4, 4, 4, 4, 4, 4, 12):
        errors.append(f"phase chunks: {PHASE_CHUNKS}")
    if TOTAL_PATCHES != 192:
        errors.append(f"patches: {TOTAL_PATCHES}")
    if C1R2_QKV_REPLAYS != 8:
        errors.append(f"qkv replays: {C1R2_QKV_REPLAYS}")
    if C1R2_UPGATE_REPLAYS != 12:
        errors.append(f"upgate replays: {C1R2_UPGATE_REPLAYS}")
    if O_CHUNKS != 8:
        errors.append(f"o chunks: {O_CHUNKS}")
    if SWIGLU_SLICES != 6:
        errors.append(f"swiglu slices: {SWIGLU_SLICES}")
    if C1R2_PACKET_DWORDS != 513:
        errors.append(f"c1r2 packet: {C1R2_PACKET_DWORDS}")
    if COMPACT_PACKET_DWORDS != 257:
        errors.append(f"compact packet: {COMPACT_PACKET_DWORDS}")
    if ATTENTION_PACKET_DWORDS != 1024:
        errors.append(f"attention packet: {ATTENTION_PACKET_DWORDS}")
    if DOWN_PACKET_DWORDS != 1536:
        errors.append(f"down packet: {DOWN_PACKET_DWORDS}")
    return errors

def summary_lines():
    return [
        "Qwen3-0.6B dataflow contract:",
        f"  hidden={HIDDEN_DIM} intermediate={INTERMEDIATE_DIM}",
        f"  Q_heads={NUM_Q_HEADS} KV_heads={NUM_KV_HEADS} HD={HEAD_DIM} GQA={GQA_RATIO}",
        f"  phases={','.join(PHASE_NAMES)}",
        f"  blocks={PHASE_BLOCKS} chunks={PHASE_CHUNKS}",
        f"  patches={TOTAL_PATCHES} total_blocks={TOTAL_LOGICAL_BLOCKS}",
        f"  attention_bf16={ATTENTION_BF16} attention_packet={ATTENTION_PACKET_DWORDS}",
        f"  c1r2_packet={C1R2_PACKET_DWORDS} down_packet={DOWN_PACKET_DWORDS}",
        f"  o_chunks={O_CHUNKS} swiglu_slices={SWIGLU_SLICES}",
    ]
