module {
  aie.device(npu2) {
    func.func private @zero_kernel_bf16(memref<128x128xbf16>) attributes {link_with = "mm_bf16_direct.o"}
    func.func private @matmul_vectorized_bf16(memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) attributes {link_with = "mm_bf16_direct.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %C_out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_out_cons_prod_lock_0"}
    %C_out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_out_cons_cons_lock_0"}
    %C_l1_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, sym_name = "C_l1_cons_buff_0"} : memref<128x128xbf16> 
    %C_l1_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32, sym_name = "C_l1_cons_prod_lock_0"}
    %C_l1_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "C_l1_cons_cons_lock_0"}
    %C_l1_buff_0 = aie.buffer(%tile_0_2) {address = 2560 : i32, sym_name = "C_l1_buff_0"} : memref<128x128xbf16> 
    %C_l1_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "C_l1_prod_lock_0"}
    %C_l1_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_l1_cons_lock_0"}
    %B_l1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 35328 : i32, sym_name = "B_l1_cons_buff_0"} : memref<64x128xbf16> 
    %B_l1_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "B_l1_cons_prod_lock_0"}
    %B_l1_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_l1_cons_cons_lock_0"}
    %B_in_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 163840 : i32, sym_name = "B_in_cons_buff_0"} : memref<64x128xbf16> 
    %B_in_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 1 : i32, sym_name = "B_in_cons_prod_lock_0"}
    %B_in_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "B_in_cons_cons_lock_0"}
    %B_in_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_in_prod_lock_0"}
    %B_in_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_in_cons_lock_0"}
    %A_l1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 51712 : i32, sym_name = "A_l1_cons_buff_0"} : memref<32x64xbf16> 
    %A_l1_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "A_l1_cons_prod_lock_0"}
    %A_l1_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_l1_cons_cons_lock_0"}
    %A_in_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "A_in_cons_buff_0"} : memref<128x512xbf16> 
    %A_in_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 1 : i32, sym_name = "A_in_cons_prod_lock_0"}
    %A_in_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "A_in_cons_cons_lock_0"}
    %A_in_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_in_prod_lock_0"}
    %A_in_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_in_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c64 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%C_l1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_l1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_l1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%A_l1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%A_l1_cons_buff_0, %B_l1_cons_buff_0, %C_l1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_l1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%A_l1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%A_l1_cons_buff_0, %B_l1_cons_buff_0, %C_l1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_l1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%A_l1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%A_l1_cons_buff_0, %B_l1_cons_buff_0, %C_l1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_l1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%A_l1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%A_l1_cons_buff_0, %B_l1_cons_buff_0, %C_l1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_l1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%B_l1_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c1 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_l1_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2560 : i32}
    aie.runtime_sequence @seq(%arg0: memref<1048576xbf16>, %arg1: memref<1048576xbf16>, %arg2: memref<1048576xbf16>) {
      %0 = aiex.dma_configure_task_for @A_in_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_in_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @C_out_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
    }
    aie.shim_dma_allocation @A_in_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%A_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_in_cons_buff_0 : memref<128x512xbf16>, 0, 65536) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%A_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_in_cons_buff_0 : memref<128x512xbf16>, 0, 65536) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_in_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%B_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_in_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%B_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_in_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_in_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 2, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_l1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_l1_cons_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%C_l1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(MM2S, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%C_l1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_l1_cons_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_l1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%A_l1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_l1_cons_buff_0 : memref<32x64xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_l1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%B_l1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_l1_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_l1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%C_l1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_l1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%C_l1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    aie.shim_dma_allocation @B_in_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @C_out_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
