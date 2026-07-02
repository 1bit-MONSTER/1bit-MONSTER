module {
  aie.device(npu2) {
    %shim = aie.tile(2, 0)
    %main = aie.tile(2, 2)

    // case marker qwen3-kernel-main16-q4nx
    aie.flow(%shim, DMA : 0, %main, DMA : 0)
    aie.flow(%shim, DMA : 1, %main, DMA : 1)
    aie.flow(%main, DMA : 1, %shim, DMA : 0)

    func.func private @q4nx_main16_layer_scheduler(memref<2560xbf16>, memref<2560xbf16>, memref<128xi32>, memref<128xi32>, memref<128xi32>, memref<128xi32>, i32, i32, i32, i32) attributes {link_with = "/home/bcloud/torch2aie/examples/qwen3-decode-layer/main_projection_q4nx_fast.o"}


    %main_accum = aie.buffer(%main) {address = 8192 : i32, sym_name = "main_accum"} : memref<32xi32>
    %main_wt_ping = aie.buffer(%main) {address = 10240 : i32, sym_name = "main_wt_ping"} : memref<2560xbf16>
    %main_record_ping = aie.buffer(%main) {address = 15388 : i32, sym_name = "main_record_ping"} : memref<128xi32>
    %main_wt_pong = aie.buffer(%main) {address = 16384 : i32, sym_name = "main_wt_pong"} : memref<2560xbf16>
    %main_record_pong = aie.buffer(%main) {address = 21532 : i32, sym_name = "main_record_pong"} : memref<128xi32>
    %main_chunk_ping = aie.buffer(%main) {address = 32768 : i32, sym_name = "main_chunk_ping"} : memref<128xi32>
    %main_chunk_pong = aie.buffer(%main) {address = 49152 : i32, sym_name = "main_chunk_pong"} : memref<128xi32>

    %main_chunk_empty = aie.lock(%main, 0) {init = 2 : i32, sym_name = "main_chunk_empty"}
    %main_chunk_full = aie.lock(%main, 1) {init = 0 : i32, sym_name = "main_chunk_full"}
    %main_wt_empty = aie.lock(%main, 2) {init = 2 : i32, sym_name = "main_wt_empty"}
    %main_wt_full = aie.lock(%main, 3) {init = 0 : i32, sym_name = "main_wt_full"}
    %main_records_empty = aie.lock(%main, 4) {init = 2 : i32, sym_name = "main_records_empty"}
    %main_records_full = aie.lock(%main, 5) {init = 0 : i32, sym_name = "main_records_full"}

    %main_core = aie.core(%main) {

      %m_i32 = arith.constant 32 : i32
      %group_i32 = arith.constant 0 : i32
      %row_i32 = arith.constant 0 : i32
      %main16_phase_limit_i32 = arith.constant 1 : i32
      func.call @q4nx_main16_layer_scheduler(%main_wt_ping, %main_wt_pong, %main_chunk_ping, %main_chunk_pong, %main_record_ping, %main_record_pong, %group_i32, %row_i32, %m_i32, %main16_phase_limit_i32)
        : (memref<2560xbf16>, memref<2560xbf16>, memref<128xi32>, memref<128xi32>, memref<128xi32>, memref<128xi32>, i32, i32, i32, i32) -> ()
      aie.end
    }

    %main_mem = aie.mem(%main) {
      %chunk_dma = aie.dma_start(S2MM, 0, ^chunk_ping, ^wt_start)
    ^chunk_ping:
      aie.use_lock(%main_chunk_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%main_chunk_ping : memref<128xi32>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%main_chunk_full, Release, 1)
      aie.next_bd ^chunk_pong
    ^chunk_pong:
      aie.use_lock(%main_chunk_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%main_chunk_pong : memref<128xi32>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%main_chunk_full, Release, 1)
      aie.next_bd ^chunk_ping

    ^wt_start:
      %wt_dma = aie.dma_start(S2MM, 1, ^wt_ping, ^record_start)
    ^wt_ping:
      aie.use_lock(%main_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%main_wt_ping : memref<2560xbf16>, 0, 2560) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%main_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%main_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%main_wt_pong : memref<2560xbf16>, 0, 2560) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%main_wt_full, Release, 1)
      aie.next_bd ^wt_ping

    ^record_start:
      %record_dma = aie.dma_start(MM2S, 1, ^record_ping, ^end)
    ^record_ping:
      aie.use_lock(%main_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%main_record_ping : memref<128xi32>, 0, 17) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%main_records_empty, Release, 1)
      aie.next_bd ^record_pong
    ^record_pong:
      aie.use_lock(%main_records_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%main_record_pong : memref<128xi32>, 0, 17) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%main_records_empty, Release, 1)
      aie.next_bd ^record_ping
    ^end:
      aie.end
    }

    aie.runtime_sequence(%activation: memref<16384xi32>, %weights: memref<163840xi32>, %records: memref<136xi32>) {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 136 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67227652 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(2, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 16384 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67227684 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.writebd {bd_id = 2 : i32, buffer_length = 163840 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67227716 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 1) {bd_id = 2 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 2 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
