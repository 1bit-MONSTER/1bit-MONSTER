module {
  aie.device(npu2) {
    func.func private @matmul_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
    func.func private @zero_i16(memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %c_s_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "c_s_cons_prod_lock_0"}
    %c_s_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "c_s_cons_cons_lock_0"}
    %c_c_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "c_c_cons_buff_0"} : memref<32x128xi16> 
    %c_c_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32, sym_name = "c_c_cons_prod_lock_0"}
    %c_c_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "c_c_cons_cons_lock_0"}
    %c_c_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "c_c_buff_0"} : memref<32x128xi16> 
    %c_c_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "c_c_prod_lock_0"}
    %c_c_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "c_c_cons_lock_0"}
    %b_c_cons_buff_0 = aie.buffer(%tile_0_2) {address = 9216 : i32, sym_name = "b_c_cons_buff_0"} : memref<64x128xi8> 
    %b_c_cons_buff_1 = aie.buffer(%tile_0_2) {address = 17408 : i32, sym_name = "b_c_cons_buff_1"} : memref<64x128xi8> 
    %b_c_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "b_c_cons_prod_lock_0"}
    %b_c_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "b_c_cons_cons_lock_0"}
    %b_s_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 8192 : i32, sym_name = "b_s_cons_buff_0"} : memref<64x128xi8> 
    %b_s_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 16384 : i32, sym_name = "b_s_cons_buff_1"} : memref<64x128xi8> 
    %b_s_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "b_s_cons_prod_lock_0"}
    %b_s_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "b_s_cons_cons_lock_0"}
    %b_s_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "b_s_prod_lock_0"}
    %b_s_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "b_s_cons_lock_0"}
    %a_c_cons_buff_0 = aie.buffer(%tile_0_2) {address = 25600 : i32, sym_name = "a_c_cons_buff_0"} : memref<32x64xi8> 
    %a_c_cons_buff_1 = aie.buffer(%tile_0_2) {address = 27648 : i32, sym_name = "a_c_cons_buff_1"} : memref<32x64xi8> 
    %a_c_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "a_c_cons_prod_lock_0"}
    %a_c_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "a_c_cons_cons_lock_0"}
    %a_s_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 24576 : i32, sym_name = "a_s_cons_buff_0"} : memref<32x64xi8> 
    %a_s_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 26624 : i32, sym_name = "a_s_cons_buff_1"} : memref<32x64xi8> 
    %a_s_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "a_s_cons_prod_lock_0"}
    %a_s_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "a_s_cons_cons_lock_0"}
    %a_s_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "a_s_prod_lock_0"}
    %a_s_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "a_s_cons_lock_0"}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "/home/bcloud/npu-sandbox/npu-infer/build/int8/minimal_i8.mlir.prj/main_core_0_2.elf", link_files = ["mm_32x64x128.o"]}
    aie.runtime_sequence(%arg0: memref<2048xi8>, %arg1: memref<8192xi8>, %arg2: memref<4096xi16>) {
      %0 = aiex.dma_configure_task_for @a_s_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xi8>, 0, 0, [<size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %1 = aiex.dma_configure_task_for @b_s_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xi8>, 0, 0, [<size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %2 = aiex.dma_configure_task_for @c_s_shim_alloc {
        aie.dma_bd(%arg2 : memref<4096xi16>, 0, 0, [<size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 0>, <size = 0, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
    }
    aie.shim_dma_allocation @a_s_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%a_s_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%a_s_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%a_s_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%a_s_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%a_s_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%a_s_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%a_s_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%a_s_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%a_s_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%a_s_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%a_s_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%a_s_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%b_s_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%b_s_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%b_s_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%b_s_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%b_s_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%b_s_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%b_s_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%b_s_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%b_s_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%b_s_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%b_s_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%b_s_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%c_c_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%c_c_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%c_c_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%c_c_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%c_c_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%c_c_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%a_c_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%a_c_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%a_c_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%a_c_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%a_c_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%a_c_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%b_c_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%b_c_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%b_c_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%b_c_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%b_c_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%b_c_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%c_c_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%c_c_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%c_c_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @b_s_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @c_s_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
      aie.connect<South : 7, North : 5>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<North : 1, DMA : 2>
      aie.connect<DMA : 2, South : 2>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, South : 1>
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  }
}
