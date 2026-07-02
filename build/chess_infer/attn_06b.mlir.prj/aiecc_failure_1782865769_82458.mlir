module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    func.func private @qwen3_attention_bf16_make_carrier_masked(memref<1024xi32>, memref<8192xi32>, memref<160xi32>, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "edge_attention_06b.o"}
    func.func private @qwen3_attention_bf16_init_accum(memref<2048xi32>, memref<32xi32>, i32, i32) attributes {link_with = "edge_attention_06b.o"}
    func.func private @qwen3_attention_bf16_accum_block(memref<8192xi32>, memref<160xi32>, memref<2048xi32>, memref<32xi32>, i32, i32, i32, i32, i32) attributes {link_with = "edge_attention_06b.o"}
    func.func private @qwen3_attention_bf16_finish_accum(memref<2048xi32>, memref<32xi32>, memref<1024xi32>, i32, i32, i32) attributes {link_with = "edge_attention_06b.o"}
    %att_q = aie.buffer(%tile_0_2) {sym_name = "att_q"} : memref<1024xi32> 
    %att_k0 = aie.buffer(%tile_0_2) {sym_name = "att_k0"} : memref<8192xi32> 
    %att_k1 = aie.buffer(%tile_0_2) {sym_name = "att_k1"} : memref<8192xi32> 
    %att_v0 = aie.buffer(%tile_0_2) {sym_name = "att_v0"} : memref<8192xi32> 
    %att_v1 = aie.buffer(%tile_0_2) {sym_name = "att_v1"} : memref<8192xi32> 
    %att_carrier = aie.buffer(%tile_0_2) {sym_name = "att_carrier"} : memref<160xi32> 
    %att_accum = aie.buffer(%tile_0_2) {sym_name = "att_accum"} : memref<2048xi32> 
    %att_state = aie.buffer(%tile_0_2) {sym_name = "att_state"} : memref<32xi32> 
    %att_output = aie.buffer(%tile_0_2) {sym_name = "att_output"} : memref<1024xi32> 
    %att_input_empty = aie.lock(%tile_0_2, 0) {init = 5 : i32, sym_name = "att_input_empty"}
    %att_input_full = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "att_input_full"}
    %att_output_empty = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "att_output_empty"}
    %att_output_full = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "att_output_full"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_0 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c7_i32 = arith.constant 7 : i32
      %c1024_i32 = arith.constant 1024 : i32
      %c8192_i32 = arith.constant 8192 : i32
      %c8192_i32_1 = arith.constant 8192 : i32
      %c1024_i32_2 = arith.constant 1024 : i32
      %c160_i32 = arith.constant 160 : i32
      %c2048_i32 = arith.constant 2048 : i32
      %c32_i32 = arith.constant 32 : i32
      aie.use_lock(%att_input_full, AcquireGreaterEqual, 5)
      func.call @qwen3_attention_bf16_init_accum(%att_accum, %att_state, %c2048_i32, %c32_i32) : (memref<2048xi32>, memref<32xi32>, i32, i32) -> ()
      func.call @qwen3_attention_bf16_make_carrier_masked(%att_q, %att_k0, %att_carrier, %c0_i32, %c0_i32_0, %c2_i32, %c7_i32, %c1024_i32, %c8192_i32, %c160_i32) : (memref<1024xi32>, memref<8192xi32>, memref<160xi32>, i32, i32, i32, i32, i32, i32, i32) -> ()
      func.call @qwen3_attention_bf16_accum_block(%att_v0, %att_carrier, %att_accum, %att_state, %c0_i32_0, %c8192_i32_1, %c160_i32, %c2048_i32, %c32_i32) : (memref<8192xi32>, memref<160xi32>, memref<2048xi32>, memref<32xi32>, i32, i32, i32, i32, i32) -> ()
      func.call @qwen3_attention_bf16_make_carrier_masked(%att_q, %att_k1, %att_carrier, %c0_i32, %c1_i32, %c2_i32, %c7_i32, %c1024_i32, %c8192_i32, %c160_i32) : (memref<1024xi32>, memref<8192xi32>, memref<160xi32>, i32, i32, i32, i32, i32, i32, i32) -> ()
      func.call @qwen3_attention_bf16_accum_block(%att_v1, %att_carrier, %att_accum, %att_state, %c1_i32, %c8192_i32_1, %c160_i32, %c2048_i32, %c32_i32) : (memref<8192xi32>, memref<160xi32>, memref<2048xi32>, memref<32xi32>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%att_output_empty, AcquireGreaterEqual, 1)
      func.call @qwen3_attention_bf16_finish_accum(%att_accum, %att_state, %att_output, %c1024_i32_2, %c2048_i32, %c32_i32) : (memref<2048xi32>, memref<32xi32>, memref<1024xi32>, i32, i32, i32) -> ()
      aie.use_lock(%att_output_full, Release, 1)
      aie.use_lock(%att_input_empty, Release, 5)
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb6)
    ^bb1:  // 2 preds: ^bb0, ^bb5
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_q : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_k0 : memref<8192xi32>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_k1 : memref<8192xi32>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_v0 : memref<8192xi32>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%att_input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_v1 : memref<8192xi32>, 0, 8192) {bd_id = 4 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%att_input_full, Release, 1)
      aie.next_bd ^bb1
    ^bb6:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%att_output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%att_output : memref<1024xi32>, 0, 1024) {bd_id = 5 : i32}
      aie.use_lock(%att_output_empty, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<33792xi32>, %arg1: memref<1024xi32>) {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 1024 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 33792 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
