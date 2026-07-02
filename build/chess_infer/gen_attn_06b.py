"""Generate edge_attention MLIR for Qwen3-0.6B (16 Q heads, 8 KV heads, GQA=2)."""

from __future__ import annotations

from pathlib import Path

# --- Qwen3-0.6B window: 4 Q heads + 2 KV heads per window (4 windows total) ---
# 16 Q heads / 4 windows = 4 Q heads per window
# 8 KV heads / 4 windows = 2 KV heads per window (= GQA=2)
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 128
CONTEXT = 16
WINDOW_DWORDS = HEADS * HEAD_DIM // 2       # 256
K_WINDOW_DWORDS = KV_HEADS * CONTEXT * HEAD_DIM // 2  # 2048
V_WINDOW_DWORDS = K_WINDOW_DWORDS            # 2048
OUTPUT_DWORDS = HEADS * HEAD_DIM // 2        # 256
ACCUM_LANES = HEADS * HEAD_DIM              # 512
SCALAR_DWORDS = HEADS * 2                    # 8
SHAPE_CARRIER_DWORDS = (HEADS * CONTEXT // 2) + HEADS * 2  # 32 + 8 = 40

from mlir_utils import flow, npu_address_patch, npu_push_queue, npu_sync, npu_writebd

CASE_NAME = "qwen3-06b-edge-attention"
PROFILE_CASE_NAME = "qwen3-06b-edge-attention-profile"
EDGE_ATTENTION_OBJECT = "edge_attention_06b.o"
EDGE_ATTENTION_PROFILE_OBJECT = "edge_attention_06b_profile.o"
BLOCKS = 2
TAIL_TOKENS = 7
INPUT_DWORDS = WINDOW_DWORDS + BLOCKS * K_WINDOW_DWORDS + BLOCKS * V_WINDOW_DWORDS


def _attention_tile() -> str:
    return f"""
    %att_q = aie.buffer(%att) {{sym_name = "att_q"}} : memref<{WINDOW_DWORDS}xi32>
    %att_k0 = aie.buffer(%att) {{sym_name = "att_k0"}} : memref<{K_WINDOW_DWORDS}xi32>
    %att_k1 = aie.buffer(%att) {{sym_name = "att_k1"}} : memref<{K_WINDOW_DWORDS}xi32>
    %att_v0 = aie.buffer(%att) {{sym_name = "att_v0"}} : memref<{V_WINDOW_DWORDS}xi32>
    %att_v1 = aie.buffer(%att) {{sym_name = "att_v1"}} : memref<{V_WINDOW_DWORDS}xi32>
    %att_carrier = aie.buffer(%att) {{sym_name = "att_carrier"}} : memref<{SHAPE_CARRIER_DWORDS}xi32>
    %att_accum = aie.buffer(%att) {{sym_name = "att_accum"}} : memref<{ACCUM_LANES}xi32>
    %att_state = aie.buffer(%att) {{sym_name = "att_state"}} : memref<{SCALAR_DWORDS}xi32>
    %att_output = aie.buffer(%att) {{sym_name = "att_output"}} : memref<{OUTPUT_DWORDS}xi32>
    %att_input_empty = aie.lock(%att, 0) {{init = 5 : i32, sym_name = "att_input_empty"}}
    %att_input_full = aie.lock(%att, 1) {{init = 0 : i32, sym_name = "att_input_full"}}
    %att_output_empty = aie.lock(%att, 2) {{init = 1 : i32, sym_name = "att_output_empty"}}
    %att_output_full = aie.lock(%att, 3) {{init = 0 : i32, sym_name = "att_output_full"}}

    %att_core = aie.core(%att) {{
      %window_i32 = arith.constant 0 : i32
      %block0_i32 = arith.constant 0 : i32
      %block1_i32 = arith.constant 1 : i32
      %blocks_i32 = arith.constant {BLOCKS} : i32
      %tail_tokens_i32 = arith.constant {TAIL_TOKENS} : i32
      %q_dwords_i32 = arith.constant {WINDOW_DWORDS} : i32
      %k_dwords_i32 = arith.constant {K_WINDOW_DWORDS} : i32
      %v_dwords_i32 = arith.constant {V_WINDOW_DWORDS} : i32
      %out_dwords_i32 = arith.constant {OUTPUT_DWORDS} : i32
      %carrier_dwords_i32 = arith.constant {SHAPE_CARRIER_DWORDS} : i32
      %accum_lanes_i32 = arith.constant {ACCUM_LANES} : i32
      %state_dwords_i32 = arith.constant {SCALAR_DWORDS} : i32

      aie.use_lock(%att_input_full, AcquireGreaterEqual, 5)
      func.call @qwen3_attention_bf16_init_accum(%att_accum, %att_state, %accum_lanes_i32, %state_dwords_i32)
        : (memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32) -> ()

      func.call @qwen3_attention_bf16_make_carrier_masked(%att_q, %att_k0, %att_carrier, %window_i32, %block0_i32, %blocks_i32, %tail_tokens_i32, %q_dwords_i32, %k_dwords_i32, %carrier_dwords_i32)
        : (memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32, i32, i32, i32, i32) -> ()
      func.call @qwen3_attention_bf16_accum_block(%att_v0, %att_carrier, %att_accum, %att_state, %block0_i32, %v_dwords_i32, %carrier_dwords_i32, %accum_lanes_i32, %state_dwords_i32)
        : (memref<{V_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32, i32, i32, i32) -> ()

      func.call @qwen3_attention_bf16_make_carrier_masked(%att_q, %att_k1, %att_carrier, %window_i32, %block1_i32, %blocks_i32, %tail_tokens_i32, %q_dwords_i32, %k_dwords_i32, %carrier_dwords_i32)
        : (memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32, i32, i32, i32, i32) -> ()
      func.call @qwen3_attention_bf16_accum_block(%att_v1, %att_carrier, %att_accum, %att_state, %block1_i32, %v_dwords_i32, %carrier_dwords_i32, %accum_lanes_i32, %state_dwords_i32)
        : (memref<{V_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32, i32, i32, i32) -> ()

      aie.use_lock(%att_output_empty, AcquireGreaterEqual, 1)
      func.call @qwen3_attention_bf16_finish_accum(%att_accum, %att_state, %att_output, %out_dwords_i32, %accum_lanes_i32, %state_dwords_i32)
        : (memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32, i32, i32) -> ()
      aie.use_lock(%att_output_full, Release, 1)
      aie.use_lock(%att_input_empty, Release, 5)
      aie.end
    }}

    %att_mem = aie.mem(%att) {{
      %input_dma = aie.dma_start(S2MM, 0, ^q_in, ^out_start)
    ^q_in:
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_q : memref<{WINDOW_DWORDS}xi32>, 0, {WINDOW_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^k0_in
    ^k0_in:
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_k0 : memref<{K_WINDOW_DWORDS}xi32>, 0, {K_WINDOW_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^k1_in
    ^k1_in:
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_k1 : memref<{K_WINDOW_DWORDS}xi32>, 0, {K_WINDOW_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^v0_in
    ^v0_in:
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_v0 : memref<{V_WINDOW_DWORDS}xi32>, 0, {V_WINDOW_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 4 : i32}}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^v1_in
    ^v1_in:
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_v1 : memref<{V_WINDOW_DWORDS}xi32>, 0, {V_WINDOW_DWORDS}) {{bd_id = 4 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^q_in

    ^out_start:
      %output_dma = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%att_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_output : memref<{OUTPUT_DWORDS}xi32>, 0, {OUTPUT_DWORDS}) {{bd_id = 5 : i32}}
      aie.use_lock(%att_output_empty, Release, 1)
      aie.next_bd ^out
    ^end:
      aie.end
    }}
"""


def _runtime_sequence() -> str:
    return "\n".join(
        (
            f"    aie.runtime_sequence(%input: memref<{INPUT_DWORDS}xi32>, "
            f"%output: memref<{OUTPUT_DWORDS}xi32>) {{",
            npu_writebd(0, 0, OUTPUT_DWORDS, 0),
            npu_address_patch(0, 0, 1, 0),
            npu_push_queue(0, "S2MM", 0, 0),
            npu_writebd(0, 1, INPUT_DWORDS, 0),
            npu_address_patch(0, 1, 0, 0),
            npu_push_queue(0, "MM2S", 0, 1),
            npu_sync(0, 0),
            "    }",
        )
    )


def generate_mlir(profile_cycles: bool = False) -> str:
    experiment_dir = Path(__file__).parent.parent.resolve()
    case_name = PROFILE_CASE_NAME if profile_cycles else CASE_NAME
    object_name = EDGE_ATTENTION_PROFILE_OBJECT if profile_cycles else EDGE_ATTENTION_OBJECT
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %att = aie.tile(0, 2)

    // case marker {case_name}
{flow("shim", 0, "att", 0)}
{flow("att", 0, "shim", 0)}

    func.func private @qwen3_attention_bf16_make_carrier_masked(memref<{WINDOW_DWORDS}xi32>, memref<{K_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, i32, i32, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/{object_name}"}}
    func.func private @qwen3_attention_bf16_init_accum(memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/{object_name}"}}
    func.func private @qwen3_attention_bf16_accum_block(memref<{V_WINDOW_DWORDS}xi32>, memref<{SHAPE_CARRIER_DWORDS}xi32>, memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, i32, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/{object_name}"}}
    func.func private @qwen3_attention_bf16_finish_accum(memref<{ACCUM_LANES}xi32>, memref<{SCALAR_DWORDS}xi32>, memref<{OUTPUT_DWORDS}xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/{object_name}"}}

{_attention_tile()}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str, profile_cycles: bool = False) -> list[str]:
    case_name = PROFILE_CASE_NAME if profile_cycles else CASE_NAME
    required = (
        f"case marker {case_name}",
        "qwen3_attention_bf16_make_carrier_masked",
        "qwen3_attention_bf16_init_accum",
        "qwen3_attention_bf16_accum_block",
        "qwen3_attention_bf16_finish_accum",
        f"memref<{INPUT_DWORDS}xi32>",
        "aie.runtime_sequence(%input",
    )
    return [f"missing edge-attention microbench marker: {marker}" for marker in required if marker not in mlir]
