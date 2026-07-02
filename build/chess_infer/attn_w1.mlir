module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %att = aie.tile(0, 2)

    // case marker qwen3-06b-edge-attention
    aie.flow(%shim, DMA : 0, %att, DMA : 0)
    aie.flow(%att, DMA : 0, %shim, DMA : 0)

    func.func private @qwen3_attention_bf16_make_carrier_masked(memref<256xi32>, memref<2048xi32>, memref<40xi32>, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "edge_attention_06b.o"}
    func.func private @qwen3_attention_bf16_init_accum(memref<512xi32>, memref<8xi32>, i32, i32) attributes {link_with = "edge_attention_06b.o"}
    func.func private @qwen3_attention_bf16_accum_block(memref<2048xi32>, memref<40xi32>, memref<512xi32>, memref<8xi32>, i32, i32, i32, i32, i32) attributes {link_with = "edge_attention_06b.o"}
    func.func private @qwen3_attention_bf16_finish_accum(memref<512xi32>, memref<8xi32>, memref<256xi32>, i32, i32, i32) attributes {link_with = "edge_attention_06b.o"}


    %att_q = aie.buffer(%att) {sym_name = "att_q"} : memref<256xi32>
    %att_k0 = aie.buffer(%att) {sym_name = "att_k0"} : memref<2048xi32>
    %att_k1 = aie.buffer(%att) {sym_name = "att_k1"} : memref<2048xi32>
    %att_v0 = aie.buffer(%att) {sym_name = "att_v0"} : memref<2048xi32>
    %att_v1 = aie.buffer(%att) {sym_name = "att_v1"} : memref<2048xi32>
    %att_carrier = aie.buffer(%att) {sym_name = "att_carrier"} : memref<40xi32>
    %att_accum = aie.buffer(%att) {sym_name = "att_accum"} : memref<512xi32>
    %att_state = aie.buffer(%att) {sym_name = "att_state"} : memref<8xi32>
    %att_output = aie.buffer(%att) {sym_name = "att_output"} : memref<256xi32>
    %att_input_empty = aie.lock(%att, 0) {init = 5 : i32, sym_name = "att_input_empty"}
    %att_input_full = aie.lock(%att, 1) {init = 0 : i32, sym_name = "att_input_full"}
    %att_output_empty = aie.lock(%att, 2) {init = 1 : i32, sym_name = "att_output_empty"}
    %att_output_full = aie.lock(%att, 3) {init = 0 : i32, sym_name = "att_output_full"}

    %att_core = aie.core(%att) {
      %window_i32 = arith.constant 1 : i32
      %block0_i32 = arith.constant 1 : i32
      %block1_i32 = arith.constant 1 : i32
      %blocks_i32 = arith.constant 2 : i32
      %tail_tokens_i32 = arith.constant 7 : i32
      %q_dwords_i32 = arith.constant 256 : i32
      %k_dwords_i32 = arith.constant 2048 : i32
      %v_dwords_i32 = arith.constant 2048 : i32
      %out_dwords_i32 = arith.constant 256 : i32
      %carrier_dwords_i32 = arith.constant 40 : i32
      %accum_lanes_i32 = arith.constant 512 : i32
      %state_dwords_i32 = arith.constant 8 : i32

      aie.use_lock(%att_input_full, AcquireGreaterEqual, 5)
      func.call @qwen3_attention_bf16_init_accum(%att_accum, %att_state, %accum_lanes_i32, %state_dwords_i32)
        : (memref<512xi32>, memref<8xi32>, i32, i32) -> ()

      func.call @qwen3_attention_bf16_make_carrier_masked(%att_q, %att_k0, %att_carrier, %window_i32, %block0_i32, %blocks_i32, %tail_tokens_i32, %q_dwords_i32, %k_dwords_i32, %carrier_dwords_i32)
        : (memref<256xi32>, memref<2048xi32>, memref<40xi32>, i32, i32, i32, i32, i32, i32, i32) -> ()
      func.call @qwen3_attention_bf16_accum_block(%att_v0, %att_carrier, %att_accum, %att_state, %block0_i32, %v_dwords_i32, %carrier_dwords_i32, %accum_lanes_i32, %state_dwords_i32)
        : (memref<2048xi32>, memref<40xi32>, memref<512xi32>, memref<8xi32>, i32, i32, i32, i32, i32) -> ()

      func.call @qwen3_attention_bf16_make_carrier_masked(%att_q, %att_k1, %att_carrier, %window_i32, %block1_i32, %blocks_i32, %tail_tokens_i32, %q_dwords_i32, %k_dwords_i32, %carrier_dwords_i32)
        : (memref<256xi32>, memref<2048xi32>, memref<40xi32>, i32, i32, i32, i32, i32, i32, i32) -> ()
      func.call @qwen3_attention_bf16_accum_block(%att_v1, %att_carrier, %att_accum, %att_state, %block1_i32, %v_dwords_i32, %carrier_dwords_i32, %accum_lanes_i32, %state_dwords_i32)
        : (memref<2048xi32>, memref<40xi32>, memref<512xi32>, memref<8xi32>, i32, i32, i32, i32, i32) -> ()

      aie.use_lock(%att_output_empty, AcquireGreaterEqual, 1)
      func.call @qwen3_attention_bf16_finish_accum(%att_accum, %att_state, %att_output, %out_dwords_i32, %accum_lanes_i32, %state_dwords_i32)
        : (memref<512xi32>, memref<8xi32>, memref<256xi32>, i32, i32, i32) -> ()
      aie.use_lock(%att_output_full, Release, 1)
      aie.use_lock(%att_input_empty, Release, 5)
      aie.end
    }

    %att_mem = aie.mem(%att) {
      %input_dma = aie.dma_start(S2MM, 0, ^q_in, ^out_start)
    ^q_in:
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_q : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^k0_in
    ^k0_in:
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_k0 : memref<2048xi32>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^k1_in
    ^k1_in:
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_k1 : memref<2048xi32>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^v0_in
    ^v0_in:
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_v0 : memref<2048xi32>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^v1_in
    ^v1_in:
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_v1 : memref<2048xi32>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^q_in

    ^out_start:
      %output_dma = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%att_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_output : memref<256xi32>, 0, 256) {bd_id = 5 : i32}
      aie.use_lock(%att_output_empty, Release, 1)
      aie.next_bd ^out
    ^end:
      aie.end
    }

    aie.runtime_sequence(%input: memref<8448xi32>, %output: memref<256xi32>) {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 256 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 8448 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
