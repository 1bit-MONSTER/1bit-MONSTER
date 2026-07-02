module {
  aie.device(npu2) {
    func.func private @zero_kernel_bf16(memref<128x128xbf16>) attributes {link_with = "mm_128x64x128.o"}
    func.func private @matmul_vectorized_different_datatypes(memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) attributes {link_with = "mm_128x64x128.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_4_1 = aie.tile(4, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_5_1 = aie.tile(5, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_6_1 = aie.tile(6, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_7_1 = aie.tile(7, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_4_2 = aie.tile(4, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_5_2 = aie.tile(5, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_6_2 = aie.tile(6, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_7_2 = aie.tile(7, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_3_3 = aie.tile(3, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_4_3 = aie.tile(4, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_5_3 = aie.tile(5, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_6_3 = aie.tile(6, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_7_3 = aie.tile(7, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_3_4 = aie.tile(3, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_4_4 = aie.tile(4, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_5_4 = aie.tile(5, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_6_4 = aie.tile(6, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_7_4 = aie.tile(7, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_2_5 = aie.tile(2, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_3_5 = aie.tile(3, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_4_5 = aie.tile(4, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_5_5 = aie.tile(5, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_6_5 = aie.tile(6, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_7_5 = aie.tile(7, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %C_L2L3_7_cons_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "C_L2L3_7_cons_prod_lock_0"}
    %C_L2L3_7_cons_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "C_L2L3_7_cons_cons_lock_0"}
    %C_L2L3_7_buff_0 = aie.buffer(%mem_tile_7_1) {address = 0 : i32, sym_name = "C_L2L3_7_buff_0"} : memref<512x128xbf16> 
    %C_L2L3_7_buff_1 = aie.buffer(%mem_tile_7_1) {address = 131072 : i32, sym_name = "C_L2L3_7_buff_1"} : memref<512x128xbf16> 
    %C_L2L3_7_prod_lock_0 = aie.lock(%mem_tile_7_1, 2) {init = 2 : i32, sym_name = "C_L2L3_7_prod_lock_0"}
    %C_L2L3_7_cons_lock_0 = aie.lock(%mem_tile_7_1, 3) {init = 0 : i32, sym_name = "C_L2L3_7_cons_lock_0"}
    %C_L2L3_7_prod_lock_1 = aie.lock(%mem_tile_7_1, 4) {init = 2 : i32, sym_name = "C_L2L3_7_prod_lock_1"}
    %C_L2L3_7_cons_lock_1 = aie.lock(%mem_tile_7_1, 5) {init = 0 : i32, sym_name = "C_L2L3_7_cons_lock_1"}
    %C_L2L3_7_prod_lock_2 = aie.lock(%mem_tile_7_1, 6) {init = 2 : i32, sym_name = "C_L2L3_7_prod_lock_2"}
    %C_L2L3_7_cons_lock_2 = aie.lock(%mem_tile_7_1, 7) {init = 0 : i32, sym_name = "C_L2L3_7_cons_lock_2"}
    %C_L2L3_7_prod_lock_3 = aie.lock(%mem_tile_7_1, 8) {init = 2 : i32, sym_name = "C_L2L3_7_prod_lock_3"}
    %C_L2L3_7_cons_lock_3 = aie.lock(%mem_tile_7_1, 9) {init = 0 : i32, sym_name = "C_L2L3_7_cons_lock_3"}
    %C_L1L2_7_3_buff_0 = aie.buffer(%tile_7_5) {address = 3328 : i32, sym_name = "C_L1L2_7_3_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_7_3_prod_lock_0 = aie.lock(%tile_7_5, 4) {init = 1 : i32, sym_name = "C_L1L2_7_3_prod_lock_0"}
    %C_L1L2_7_3_cons_lock_0 = aie.lock(%tile_7_5, 5) {init = 0 : i32, sym_name = "C_L1L2_7_3_cons_lock_0"}
    %C_L1L2_7_2_buff_0 = aie.buffer(%tile_7_4) {address = 3328 : i32, sym_name = "C_L1L2_7_2_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_7_2_prod_lock_0 = aie.lock(%tile_7_4, 4) {init = 1 : i32, sym_name = "C_L1L2_7_2_prod_lock_0"}
    %C_L1L2_7_2_cons_lock_0 = aie.lock(%tile_7_4, 5) {init = 0 : i32, sym_name = "C_L1L2_7_2_cons_lock_0"}
    %C_L1L2_7_1_buff_0 = aie.buffer(%tile_7_3) {address = 3328 : i32, sym_name = "C_L1L2_7_1_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_7_1_prod_lock_0 = aie.lock(%tile_7_3, 4) {init = 1 : i32, sym_name = "C_L1L2_7_1_prod_lock_0"}
    %C_L1L2_7_1_cons_lock_0 = aie.lock(%tile_7_3, 5) {init = 0 : i32, sym_name = "C_L1L2_7_1_cons_lock_0"}
    %C_L1L2_7_0_buff_0 = aie.buffer(%tile_7_2) {address = 3328 : i32, sym_name = "C_L1L2_7_0_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_7_0_prod_lock_0 = aie.lock(%tile_7_2, 4) {init = 1 : i32, sym_name = "C_L1L2_7_0_prod_lock_0"}
    %C_L1L2_7_0_cons_lock_0 = aie.lock(%tile_7_2, 5) {init = 0 : i32, sym_name = "C_L1L2_7_0_cons_lock_0"}
    %C_L2L3_6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 4) {init = 0 : i32, sym_name = "C_L2L3_6_cons_prod_lock_0"}
    %C_L2L3_6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 5) {init = 0 : i32, sym_name = "C_L2L3_6_cons_cons_lock_0"}
    %C_L2L3_6_buff_0 = aie.buffer(%mem_tile_6_1) {address = 0 : i32, sym_name = "C_L2L3_6_buff_0"} : memref<512x128xbf16> 
    %C_L2L3_6_buff_1 = aie.buffer(%mem_tile_7_1) {address = 262144 : i32, sym_name = "C_L2L3_6_buff_1"} : memref<512x128xbf16> 
    %C_L2L3_6_prod_lock_0 = aie.lock(%mem_tile_6_1, 4) {init = 2 : i32, sym_name = "C_L2L3_6_prod_lock_0"}
    %C_L2L3_6_cons_lock_0 = aie.lock(%mem_tile_6_1, 5) {init = 0 : i32, sym_name = "C_L2L3_6_cons_lock_0"}
    %C_L2L3_6_prod_lock_1 = aie.lock(%mem_tile_6_1, 6) {init = 2 : i32, sym_name = "C_L2L3_6_prod_lock_1"}
    %C_L2L3_6_cons_lock_1 = aie.lock(%mem_tile_6_1, 7) {init = 0 : i32, sym_name = "C_L2L3_6_cons_lock_1"}
    %C_L2L3_6_prod_lock_2 = aie.lock(%mem_tile_6_1, 8) {init = 2 : i32, sym_name = "C_L2L3_6_prod_lock_2"}
    %C_L2L3_6_cons_lock_2 = aie.lock(%mem_tile_6_1, 9) {init = 0 : i32, sym_name = "C_L2L3_6_cons_lock_2"}
    %C_L2L3_6_prod_lock_3 = aie.lock(%mem_tile_6_1, 10) {init = 2 : i32, sym_name = "C_L2L3_6_prod_lock_3"}
    %C_L2L3_6_cons_lock_3 = aie.lock(%mem_tile_6_1, 11) {init = 0 : i32, sym_name = "C_L2L3_6_cons_lock_3"}
    %C_L1L2_6_3_buff_0 = aie.buffer(%tile_6_5) {address = 3328 : i32, sym_name = "C_L1L2_6_3_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_6_3_prod_lock_0 = aie.lock(%tile_6_5, 4) {init = 1 : i32, sym_name = "C_L1L2_6_3_prod_lock_0"}
    %C_L1L2_6_3_cons_lock_0 = aie.lock(%tile_6_5, 5) {init = 0 : i32, sym_name = "C_L1L2_6_3_cons_lock_0"}
    %C_L1L2_6_2_buff_0 = aie.buffer(%tile_6_4) {address = 3328 : i32, sym_name = "C_L1L2_6_2_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_6_2_prod_lock_0 = aie.lock(%tile_6_4, 4) {init = 1 : i32, sym_name = "C_L1L2_6_2_prod_lock_0"}
    %C_L1L2_6_2_cons_lock_0 = aie.lock(%tile_6_4, 5) {init = 0 : i32, sym_name = "C_L1L2_6_2_cons_lock_0"}
    %C_L1L2_6_1_buff_0 = aie.buffer(%tile_6_3) {address = 3328 : i32, sym_name = "C_L1L2_6_1_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_6_1_prod_lock_0 = aie.lock(%tile_6_3, 4) {init = 1 : i32, sym_name = "C_L1L2_6_1_prod_lock_0"}
    %C_L1L2_6_1_cons_lock_0 = aie.lock(%tile_6_3, 5) {init = 0 : i32, sym_name = "C_L1L2_6_1_cons_lock_0"}
    %C_L1L2_6_0_buff_0 = aie.buffer(%tile_6_2) {address = 3328 : i32, sym_name = "C_L1L2_6_0_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_6_0_prod_lock_0 = aie.lock(%tile_6_2, 4) {init = 1 : i32, sym_name = "C_L1L2_6_0_prod_lock_0"}
    %C_L1L2_6_0_cons_lock_0 = aie.lock(%tile_6_2, 5) {init = 0 : i32, sym_name = "C_L1L2_6_0_cons_lock_0"}
    %C_L2L3_5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "C_L2L3_5_cons_prod_lock_0"}
    %C_L2L3_5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "C_L2L3_5_cons_cons_lock_0"}
    %C_L2L3_5_buff_0 = aie.buffer(%mem_tile_5_1) {address = 0 : i32, sym_name = "C_L2L3_5_buff_0"} : memref<512x128xbf16> 
    %C_L2L3_5_buff_1 = aie.buffer(%mem_tile_5_1) {address = 131072 : i32, sym_name = "C_L2L3_5_buff_1"} : memref<512x128xbf16> 
    %C_L2L3_5_prod_lock_0 = aie.lock(%mem_tile_5_1, 2) {init = 2 : i32, sym_name = "C_L2L3_5_prod_lock_0"}
    %C_L2L3_5_cons_lock_0 = aie.lock(%mem_tile_5_1, 3) {init = 0 : i32, sym_name = "C_L2L3_5_cons_lock_0"}
    %C_L2L3_5_prod_lock_1 = aie.lock(%mem_tile_5_1, 4) {init = 2 : i32, sym_name = "C_L2L3_5_prod_lock_1"}
    %C_L2L3_5_cons_lock_1 = aie.lock(%mem_tile_5_1, 5) {init = 0 : i32, sym_name = "C_L2L3_5_cons_lock_1"}
    %C_L2L3_5_prod_lock_2 = aie.lock(%mem_tile_5_1, 6) {init = 2 : i32, sym_name = "C_L2L3_5_prod_lock_2"}
    %C_L2L3_5_cons_lock_2 = aie.lock(%mem_tile_5_1, 7) {init = 0 : i32, sym_name = "C_L2L3_5_cons_lock_2"}
    %C_L2L3_5_prod_lock_3 = aie.lock(%mem_tile_5_1, 8) {init = 2 : i32, sym_name = "C_L2L3_5_prod_lock_3"}
    %C_L2L3_5_cons_lock_3 = aie.lock(%mem_tile_5_1, 9) {init = 0 : i32, sym_name = "C_L2L3_5_cons_lock_3"}
    %C_L1L2_5_3_buff_0 = aie.buffer(%tile_5_5) {address = 3328 : i32, sym_name = "C_L1L2_5_3_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_5_3_prod_lock_0 = aie.lock(%tile_5_5, 4) {init = 1 : i32, sym_name = "C_L1L2_5_3_prod_lock_0"}
    %C_L1L2_5_3_cons_lock_0 = aie.lock(%tile_5_5, 5) {init = 0 : i32, sym_name = "C_L1L2_5_3_cons_lock_0"}
    %C_L1L2_5_2_buff_0 = aie.buffer(%tile_5_4) {address = 3328 : i32, sym_name = "C_L1L2_5_2_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_5_2_prod_lock_0 = aie.lock(%tile_5_4, 4) {init = 1 : i32, sym_name = "C_L1L2_5_2_prod_lock_0"}
    %C_L1L2_5_2_cons_lock_0 = aie.lock(%tile_5_4, 5) {init = 0 : i32, sym_name = "C_L1L2_5_2_cons_lock_0"}
    %C_L1L2_5_1_buff_0 = aie.buffer(%tile_5_3) {address = 3328 : i32, sym_name = "C_L1L2_5_1_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_5_1_prod_lock_0 = aie.lock(%tile_5_3, 4) {init = 1 : i32, sym_name = "C_L1L2_5_1_prod_lock_0"}
    %C_L1L2_5_1_cons_lock_0 = aie.lock(%tile_5_3, 5) {init = 0 : i32, sym_name = "C_L1L2_5_1_cons_lock_0"}
    %C_L1L2_5_0_buff_0 = aie.buffer(%tile_5_2) {address = 3328 : i32, sym_name = "C_L1L2_5_0_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_5_0_prod_lock_0 = aie.lock(%tile_5_2, 4) {init = 1 : i32, sym_name = "C_L1L2_5_0_prod_lock_0"}
    %C_L1L2_5_0_cons_lock_0 = aie.lock(%tile_5_2, 5) {init = 0 : i32, sym_name = "C_L1L2_5_0_cons_lock_0"}
    %C_L2L3_4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 4) {init = 0 : i32, sym_name = "C_L2L3_4_cons_prod_lock_0"}
    %C_L2L3_4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 5) {init = 0 : i32, sym_name = "C_L2L3_4_cons_cons_lock_0"}
    %C_L2L3_4_buff_0 = aie.buffer(%mem_tile_4_1) {address = 0 : i32, sym_name = "C_L2L3_4_buff_0"} : memref<512x128xbf16> 
    %C_L2L3_4_buff_1 = aie.buffer(%mem_tile_5_1) {address = 262144 : i32, sym_name = "C_L2L3_4_buff_1"} : memref<512x128xbf16> 
    %C_L2L3_4_prod_lock_0 = aie.lock(%mem_tile_4_1, 4) {init = 2 : i32, sym_name = "C_L2L3_4_prod_lock_0"}
    %C_L2L3_4_cons_lock_0 = aie.lock(%mem_tile_4_1, 5) {init = 0 : i32, sym_name = "C_L2L3_4_cons_lock_0"}
    %C_L2L3_4_prod_lock_1 = aie.lock(%mem_tile_4_1, 6) {init = 2 : i32, sym_name = "C_L2L3_4_prod_lock_1"}
    %C_L2L3_4_cons_lock_1 = aie.lock(%mem_tile_4_1, 7) {init = 0 : i32, sym_name = "C_L2L3_4_cons_lock_1"}
    %C_L2L3_4_prod_lock_2 = aie.lock(%mem_tile_4_1, 8) {init = 2 : i32, sym_name = "C_L2L3_4_prod_lock_2"}
    %C_L2L3_4_cons_lock_2 = aie.lock(%mem_tile_4_1, 9) {init = 0 : i32, sym_name = "C_L2L3_4_cons_lock_2"}
    %C_L2L3_4_prod_lock_3 = aie.lock(%mem_tile_4_1, 10) {init = 2 : i32, sym_name = "C_L2L3_4_prod_lock_3"}
    %C_L2L3_4_cons_lock_3 = aie.lock(%mem_tile_4_1, 11) {init = 0 : i32, sym_name = "C_L2L3_4_cons_lock_3"}
    %C_L1L2_4_3_buff_0 = aie.buffer(%tile_4_5) {address = 3328 : i32, sym_name = "C_L1L2_4_3_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_4_3_prod_lock_0 = aie.lock(%tile_4_5, 4) {init = 1 : i32, sym_name = "C_L1L2_4_3_prod_lock_0"}
    %C_L1L2_4_3_cons_lock_0 = aie.lock(%tile_4_5, 5) {init = 0 : i32, sym_name = "C_L1L2_4_3_cons_lock_0"}
    %C_L1L2_4_2_buff_0 = aie.buffer(%tile_4_4) {address = 3328 : i32, sym_name = "C_L1L2_4_2_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_4_2_prod_lock_0 = aie.lock(%tile_4_4, 4) {init = 1 : i32, sym_name = "C_L1L2_4_2_prod_lock_0"}
    %C_L1L2_4_2_cons_lock_0 = aie.lock(%tile_4_4, 5) {init = 0 : i32, sym_name = "C_L1L2_4_2_cons_lock_0"}
    %C_L1L2_4_1_buff_0 = aie.buffer(%tile_4_3) {address = 3328 : i32, sym_name = "C_L1L2_4_1_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_4_1_prod_lock_0 = aie.lock(%tile_4_3, 4) {init = 1 : i32, sym_name = "C_L1L2_4_1_prod_lock_0"}
    %C_L1L2_4_1_cons_lock_0 = aie.lock(%tile_4_3, 5) {init = 0 : i32, sym_name = "C_L1L2_4_1_cons_lock_0"}
    %C_L1L2_4_0_buff_0 = aie.buffer(%tile_4_2) {address = 3328 : i32, sym_name = "C_L1L2_4_0_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_4_0_prod_lock_0 = aie.lock(%tile_4_2, 4) {init = 1 : i32, sym_name = "C_L1L2_4_0_prod_lock_0"}
    %C_L1L2_4_0_cons_lock_0 = aie.lock(%tile_4_2, 5) {init = 0 : i32, sym_name = "C_L1L2_4_0_cons_lock_0"}
    %C_L2L3_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "C_L2L3_3_cons_prod_lock_0"}
    %C_L2L3_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "C_L2L3_3_cons_cons_lock_0"}
    %C_L2L3_3_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, sym_name = "C_L2L3_3_buff_0"} : memref<512x128xbf16> 
    %C_L2L3_3_buff_1 = aie.buffer(%mem_tile_3_1) {address = 131072 : i32, sym_name = "C_L2L3_3_buff_1"} : memref<512x128xbf16> 
    %C_L2L3_3_prod_lock_0 = aie.lock(%mem_tile_3_1, 2) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_0"}
    %C_L2L3_3_cons_lock_0 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_0"}
    %C_L2L3_3_prod_lock_1 = aie.lock(%mem_tile_3_1, 4) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_1"}
    %C_L2L3_3_cons_lock_1 = aie.lock(%mem_tile_3_1, 5) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_1"}
    %C_L2L3_3_prod_lock_2 = aie.lock(%mem_tile_3_1, 6) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_2"}
    %C_L2L3_3_cons_lock_2 = aie.lock(%mem_tile_3_1, 7) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_2"}
    %C_L2L3_3_prod_lock_3 = aie.lock(%mem_tile_3_1, 8) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_3"}
    %C_L2L3_3_cons_lock_3 = aie.lock(%mem_tile_3_1, 9) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_3"}
    %C_L1L2_3_3_buff_0 = aie.buffer(%tile_3_5) {address = 3328 : i32, sym_name = "C_L1L2_3_3_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_3_3_prod_lock_0 = aie.lock(%tile_3_5, 4) {init = 1 : i32, sym_name = "C_L1L2_3_3_prod_lock_0"}
    %C_L1L2_3_3_cons_lock_0 = aie.lock(%tile_3_5, 5) {init = 0 : i32, sym_name = "C_L1L2_3_3_cons_lock_0"}
    %C_L1L2_3_2_buff_0 = aie.buffer(%tile_3_4) {address = 3328 : i32, sym_name = "C_L1L2_3_2_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_3_2_prod_lock_0 = aie.lock(%tile_3_4, 4) {init = 1 : i32, sym_name = "C_L1L2_3_2_prod_lock_0"}
    %C_L1L2_3_2_cons_lock_0 = aie.lock(%tile_3_4, 5) {init = 0 : i32, sym_name = "C_L1L2_3_2_cons_lock_0"}
    %C_L1L2_3_1_buff_0 = aie.buffer(%tile_3_3) {address = 3328 : i32, sym_name = "C_L1L2_3_1_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_3_1_prod_lock_0 = aie.lock(%tile_3_3, 4) {init = 1 : i32, sym_name = "C_L1L2_3_1_prod_lock_0"}
    %C_L1L2_3_1_cons_lock_0 = aie.lock(%tile_3_3, 5) {init = 0 : i32, sym_name = "C_L1L2_3_1_cons_lock_0"}
    %C_L1L2_3_0_buff_0 = aie.buffer(%tile_3_2) {address = 3328 : i32, sym_name = "C_L1L2_3_0_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_3_0_prod_lock_0 = aie.lock(%tile_3_2, 4) {init = 1 : i32, sym_name = "C_L1L2_3_0_prod_lock_0"}
    %C_L1L2_3_0_cons_lock_0 = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "C_L1L2_3_0_cons_lock_0"}
    %C_L2L3_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "C_L2L3_2_cons_prod_lock_0"}
    %C_L2L3_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "C_L2L3_2_cons_cons_lock_0"}
    %C_L2L3_2_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "C_L2L3_2_buff_0"} : memref<512x128xbf16> 
    %C_L2L3_2_buff_1 = aie.buffer(%mem_tile_3_1) {address = 262144 : i32, sym_name = "C_L2L3_2_buff_1"} : memref<512x128xbf16> 
    %C_L2L3_2_prod_lock_0 = aie.lock(%mem_tile_2_1, 4) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_0"}
    %C_L2L3_2_cons_lock_0 = aie.lock(%mem_tile_2_1, 5) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_0"}
    %C_L2L3_2_prod_lock_1 = aie.lock(%mem_tile_2_1, 6) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_1"}
    %C_L2L3_2_cons_lock_1 = aie.lock(%mem_tile_2_1, 7) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_1"}
    %C_L2L3_2_prod_lock_2 = aie.lock(%mem_tile_2_1, 8) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_2"}
    %C_L2L3_2_cons_lock_2 = aie.lock(%mem_tile_2_1, 9) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_2"}
    %C_L2L3_2_prod_lock_3 = aie.lock(%mem_tile_2_1, 10) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_3"}
    %C_L2L3_2_cons_lock_3 = aie.lock(%mem_tile_2_1, 11) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_3"}
    %C_L1L2_2_3_buff_0 = aie.buffer(%tile_2_5) {address = 3328 : i32, sym_name = "C_L1L2_2_3_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_2_3_prod_lock_0 = aie.lock(%tile_2_5, 4) {init = 1 : i32, sym_name = "C_L1L2_2_3_prod_lock_0"}
    %C_L1L2_2_3_cons_lock_0 = aie.lock(%tile_2_5, 5) {init = 0 : i32, sym_name = "C_L1L2_2_3_cons_lock_0"}
    %C_L1L2_2_2_buff_0 = aie.buffer(%tile_2_4) {address = 3328 : i32, sym_name = "C_L1L2_2_2_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_2_2_prod_lock_0 = aie.lock(%tile_2_4, 4) {init = 1 : i32, sym_name = "C_L1L2_2_2_prod_lock_0"}
    %C_L1L2_2_2_cons_lock_0 = aie.lock(%tile_2_4, 5) {init = 0 : i32, sym_name = "C_L1L2_2_2_cons_lock_0"}
    %C_L1L2_2_1_buff_0 = aie.buffer(%tile_2_3) {address = 3328 : i32, sym_name = "C_L1L2_2_1_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_2_1_prod_lock_0 = aie.lock(%tile_2_3, 4) {init = 1 : i32, sym_name = "C_L1L2_2_1_prod_lock_0"}
    %C_L1L2_2_1_cons_lock_0 = aie.lock(%tile_2_3, 5) {init = 0 : i32, sym_name = "C_L1L2_2_1_cons_lock_0"}
    %C_L1L2_2_0_buff_0 = aie.buffer(%tile_2_2) {address = 3328 : i32, sym_name = "C_L1L2_2_0_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_2_0_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 1 : i32, sym_name = "C_L1L2_2_0_prod_lock_0"}
    %C_L1L2_2_0_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "C_L1L2_2_0_cons_lock_0"}
    %C_L2L3_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "C_L2L3_1_cons_prod_lock_0"}
    %C_L2L3_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "C_L2L3_1_cons_cons_lock_0"}
    %C_L2L3_1_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "C_L2L3_1_buff_0"} : memref<512x128xbf16> 
    %C_L2L3_1_buff_1 = aie.buffer(%mem_tile_1_1) {address = 131072 : i32, sym_name = "C_L2L3_1_buff_1"} : memref<512x128xbf16> 
    %C_L2L3_1_prod_lock_0 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_0"}
    %C_L2L3_1_cons_lock_0 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_0"}
    %C_L2L3_1_prod_lock_1 = aie.lock(%mem_tile_1_1, 4) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_1"}
    %C_L2L3_1_cons_lock_1 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_1"}
    %C_L2L3_1_prod_lock_2 = aie.lock(%mem_tile_1_1, 6) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_2"}
    %C_L2L3_1_cons_lock_2 = aie.lock(%mem_tile_1_1, 7) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_2"}
    %C_L2L3_1_prod_lock_3 = aie.lock(%mem_tile_1_1, 8) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_3"}
    %C_L2L3_1_cons_lock_3 = aie.lock(%mem_tile_1_1, 9) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_3"}
    %C_L1L2_1_3_buff_0 = aie.buffer(%tile_1_5) {address = 3328 : i32, sym_name = "C_L1L2_1_3_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_1_3_prod_lock_0 = aie.lock(%tile_1_5, 4) {init = 1 : i32, sym_name = "C_L1L2_1_3_prod_lock_0"}
    %C_L1L2_1_3_cons_lock_0 = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "C_L1L2_1_3_cons_lock_0"}
    %C_L1L2_1_2_buff_0 = aie.buffer(%tile_1_4) {address = 3328 : i32, sym_name = "C_L1L2_1_2_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_1_2_prod_lock_0 = aie.lock(%tile_1_4, 4) {init = 1 : i32, sym_name = "C_L1L2_1_2_prod_lock_0"}
    %C_L1L2_1_2_cons_lock_0 = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "C_L1L2_1_2_cons_lock_0"}
    %C_L1L2_1_1_buff_0 = aie.buffer(%tile_1_3) {address = 3328 : i32, sym_name = "C_L1L2_1_1_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_1_1_prod_lock_0 = aie.lock(%tile_1_3, 4) {init = 1 : i32, sym_name = "C_L1L2_1_1_prod_lock_0"}
    %C_L1L2_1_1_cons_lock_0 = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "C_L1L2_1_1_cons_lock_0"}
    %C_L1L2_1_0_buff_0 = aie.buffer(%tile_1_2) {address = 3328 : i32, sym_name = "C_L1L2_1_0_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_1_0_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 1 : i32, sym_name = "C_L1L2_1_0_prod_lock_0"}
    %C_L1L2_1_0_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "C_L1L2_1_0_cons_lock_0"}
    %C_L2L3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L2L3_0_cons_prod_lock_0"}
    %C_L2L3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L2L3_0_cons_cons_lock_0"}
    %C_L2L3_0_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "C_L2L3_0_buff_0"} : memref<512x128xbf16> 
    %C_L2L3_0_buff_1 = aie.buffer(%mem_tile_1_1) {address = 262144 : i32, sym_name = "C_L2L3_0_buff_1"} : memref<512x128xbf16> 
    %C_L2L3_0_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_0"}
    %C_L2L3_0_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_0"}
    %C_L2L3_0_prod_lock_1 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_1"}
    %C_L2L3_0_cons_lock_1 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_1"}
    %C_L2L3_0_prod_lock_2 = aie.lock(%mem_tile_0_1, 8) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_2"}
    %C_L2L3_0_cons_lock_2 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_2"}
    %C_L2L3_0_prod_lock_3 = aie.lock(%mem_tile_0_1, 10) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_3"}
    %C_L2L3_0_cons_lock_3 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_3"}
    %C_L1L2_0_3_buff_0 = aie.buffer(%tile_0_5) {address = 3328 : i32, sym_name = "C_L1L2_0_3_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_0_3_prod_lock_0 = aie.lock(%tile_0_5, 4) {init = 1 : i32, sym_name = "C_L1L2_0_3_prod_lock_0"}
    %C_L1L2_0_3_cons_lock_0 = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "C_L1L2_0_3_cons_lock_0"}
    %C_L1L2_0_2_buff_0 = aie.buffer(%tile_0_4) {address = 3328 : i32, sym_name = "C_L1L2_0_2_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_0_2_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 1 : i32, sym_name = "C_L1L2_0_2_prod_lock_0"}
    %C_L1L2_0_2_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "C_L1L2_0_2_cons_lock_0"}
    %C_L1L2_0_1_buff_0 = aie.buffer(%tile_0_3) {address = 3328 : i32, sym_name = "C_L1L2_0_1_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_0_1_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 1 : i32, sym_name = "C_L1L2_0_1_prod_lock_0"}
    %C_L1L2_0_1_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "C_L1L2_0_1_cons_lock_0"}
    %C_L1L2_0_0_buff_0 = aie.buffer(%tile_0_2) {address = 3328 : i32, sym_name = "C_L1L2_0_0_buff_0"} : memref<128x128xbf16> 
    %C_L1L2_0_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "C_L1L2_0_0_prod_lock_0"}
    %C_L1L2_0_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_L1L2_0_0_cons_lock_0"}
    %B_L2L1_7_0_cons_buff_0 = aie.buffer(%tile_7_2) {address = 36096 : i32, sym_name = "B_L2L1_7_0_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_7_0_cons_buff_1 = aie.buffer(%tile_7_2) {address = 45312 : i32, sym_name = "B_L2L1_7_0_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_7_0_cons_prod_lock_0 = aie.lock(%tile_7_2, 2) {init = 2 : i32, sym_name = "B_L2L1_7_0_cons_prod_lock_0"}
    %B_L2L1_7_0_cons_cons_lock_0 = aie.lock(%tile_7_2, 3) {init = 0 : i32, sym_name = "B_L2L1_7_0_cons_cons_lock_0"}
    %B_L2L1_7_1_cons_buff_0 = aie.buffer(%tile_7_3) {address = 36096 : i32, sym_name = "B_L2L1_7_1_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_7_1_cons_buff_1 = aie.buffer(%tile_7_3) {address = 45312 : i32, sym_name = "B_L2L1_7_1_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_7_1_cons_prod_lock_0 = aie.lock(%tile_7_3, 2) {init = 2 : i32, sym_name = "B_L2L1_7_1_cons_prod_lock_0"}
    %B_L2L1_7_1_cons_cons_lock_0 = aie.lock(%tile_7_3, 3) {init = 0 : i32, sym_name = "B_L2L1_7_1_cons_cons_lock_0"}
    %B_L2L1_7_2_cons_buff_0 = aie.buffer(%tile_7_4) {address = 36096 : i32, sym_name = "B_L2L1_7_2_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_7_2_cons_buff_1 = aie.buffer(%tile_7_4) {address = 45312 : i32, sym_name = "B_L2L1_7_2_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_7_2_cons_prod_lock_0 = aie.lock(%tile_7_4, 2) {init = 2 : i32, sym_name = "B_L2L1_7_2_cons_prod_lock_0"}
    %B_L2L1_7_2_cons_cons_lock_0 = aie.lock(%tile_7_4, 3) {init = 0 : i32, sym_name = "B_L2L1_7_2_cons_cons_lock_0"}
    %B_L2L1_7_3_cons_buff_0 = aie.buffer(%tile_7_5) {address = 36096 : i32, sym_name = "B_L2L1_7_3_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_7_3_cons_buff_1 = aie.buffer(%tile_7_5) {address = 45312 : i32, sym_name = "B_L2L1_7_3_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_7_3_cons_prod_lock_0 = aie.lock(%tile_7_5, 2) {init = 2 : i32, sym_name = "B_L2L1_7_3_cons_prod_lock_0"}
    %B_L2L1_7_3_cons_cons_lock_0 = aie.lock(%tile_7_5, 3) {init = 0 : i32, sym_name = "B_L2L1_7_3_cons_cons_lock_0"}
    %B_L3L2_7_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 393216 : i32, sym_name = "B_L3L2_7_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_7_cons_buff_1 = aie.buffer(%mem_tile_7_1) {address = 402432 : i32, sym_name = "B_L3L2_7_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_7_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 0) {init = 2 : i32, sym_name = "B_L3L2_7_cons_prod_lock_0"}
    %B_L3L2_7_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 1) {init = 0 : i32, sym_name = "B_L3L2_7_cons_cons_lock_0"}
    %B_L3L2_7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "B_L3L2_7_prod_lock_0"}
    %B_L3L2_7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "B_L3L2_7_cons_lock_0"}
    %B_L2L1_6_0_cons_buff_0 = aie.buffer(%tile_6_2) {address = 36096 : i32, sym_name = "B_L2L1_6_0_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_6_0_cons_buff_1 = aie.buffer(%tile_6_2) {address = 45312 : i32, sym_name = "B_L2L1_6_0_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_6_0_cons_prod_lock_0 = aie.lock(%tile_6_2, 2) {init = 2 : i32, sym_name = "B_L2L1_6_0_cons_prod_lock_0"}
    %B_L2L1_6_0_cons_cons_lock_0 = aie.lock(%tile_6_2, 3) {init = 0 : i32, sym_name = "B_L2L1_6_0_cons_cons_lock_0"}
    %B_L2L1_6_1_cons_buff_0 = aie.buffer(%tile_6_3) {address = 36096 : i32, sym_name = "B_L2L1_6_1_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_6_1_cons_buff_1 = aie.buffer(%tile_6_3) {address = 45312 : i32, sym_name = "B_L2L1_6_1_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_6_1_cons_prod_lock_0 = aie.lock(%tile_6_3, 2) {init = 2 : i32, sym_name = "B_L2L1_6_1_cons_prod_lock_0"}
    %B_L2L1_6_1_cons_cons_lock_0 = aie.lock(%tile_6_3, 3) {init = 0 : i32, sym_name = "B_L2L1_6_1_cons_cons_lock_0"}
    %B_L2L1_6_2_cons_buff_0 = aie.buffer(%tile_6_4) {address = 36096 : i32, sym_name = "B_L2L1_6_2_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_6_2_cons_buff_1 = aie.buffer(%tile_6_4) {address = 45312 : i32, sym_name = "B_L2L1_6_2_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_6_2_cons_prod_lock_0 = aie.lock(%tile_6_4, 2) {init = 2 : i32, sym_name = "B_L2L1_6_2_cons_prod_lock_0"}
    %B_L2L1_6_2_cons_cons_lock_0 = aie.lock(%tile_6_4, 3) {init = 0 : i32, sym_name = "B_L2L1_6_2_cons_cons_lock_0"}
    %B_L2L1_6_3_cons_buff_0 = aie.buffer(%tile_6_5) {address = 36096 : i32, sym_name = "B_L2L1_6_3_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_6_3_cons_buff_1 = aie.buffer(%tile_6_5) {address = 45312 : i32, sym_name = "B_L2L1_6_3_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_6_3_cons_prod_lock_0 = aie.lock(%tile_6_5, 2) {init = 2 : i32, sym_name = "B_L2L1_6_3_cons_prod_lock_0"}
    %B_L2L1_6_3_cons_cons_lock_0 = aie.lock(%tile_6_5, 3) {init = 0 : i32, sym_name = "B_L2L1_6_3_cons_cons_lock_0"}
    %B_L3L2_6_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 393216 : i32, sym_name = "B_L3L2_6_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_6_cons_buff_1 = aie.buffer(%mem_tile_6_1) {address = 402432 : i32, sym_name = "B_L3L2_6_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_6_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 2) {init = 2 : i32, sym_name = "B_L3L2_6_cons_prod_lock_0"}
    %B_L3L2_6_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 3) {init = 0 : i32, sym_name = "B_L3L2_6_cons_cons_lock_0"}
    %B_L3L2_6_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "B_L3L2_6_prod_lock_0"}
    %B_L3L2_6_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "B_L3L2_6_cons_lock_0"}
    %B_L2L1_5_0_cons_buff_0 = aie.buffer(%tile_5_2) {address = 36096 : i32, sym_name = "B_L2L1_5_0_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_5_0_cons_buff_1 = aie.buffer(%tile_5_2) {address = 45312 : i32, sym_name = "B_L2L1_5_0_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_5_0_cons_prod_lock_0 = aie.lock(%tile_5_2, 2) {init = 2 : i32, sym_name = "B_L2L1_5_0_cons_prod_lock_0"}
    %B_L2L1_5_0_cons_cons_lock_0 = aie.lock(%tile_5_2, 3) {init = 0 : i32, sym_name = "B_L2L1_5_0_cons_cons_lock_0"}
    %B_L2L1_5_1_cons_buff_0 = aie.buffer(%tile_5_3) {address = 36096 : i32, sym_name = "B_L2L1_5_1_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_5_1_cons_buff_1 = aie.buffer(%tile_5_3) {address = 45312 : i32, sym_name = "B_L2L1_5_1_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_5_1_cons_prod_lock_0 = aie.lock(%tile_5_3, 2) {init = 2 : i32, sym_name = "B_L2L1_5_1_cons_prod_lock_0"}
    %B_L2L1_5_1_cons_cons_lock_0 = aie.lock(%tile_5_3, 3) {init = 0 : i32, sym_name = "B_L2L1_5_1_cons_cons_lock_0"}
    %B_L2L1_5_2_cons_buff_0 = aie.buffer(%tile_5_4) {address = 36096 : i32, sym_name = "B_L2L1_5_2_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_5_2_cons_buff_1 = aie.buffer(%tile_5_4) {address = 45312 : i32, sym_name = "B_L2L1_5_2_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_5_2_cons_prod_lock_0 = aie.lock(%tile_5_4, 2) {init = 2 : i32, sym_name = "B_L2L1_5_2_cons_prod_lock_0"}
    %B_L2L1_5_2_cons_cons_lock_0 = aie.lock(%tile_5_4, 3) {init = 0 : i32, sym_name = "B_L2L1_5_2_cons_cons_lock_0"}
    %B_L2L1_5_3_cons_buff_0 = aie.buffer(%tile_5_5) {address = 36096 : i32, sym_name = "B_L2L1_5_3_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_5_3_cons_buff_1 = aie.buffer(%tile_5_5) {address = 45312 : i32, sym_name = "B_L2L1_5_3_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_5_3_cons_prod_lock_0 = aie.lock(%tile_5_5, 2) {init = 2 : i32, sym_name = "B_L2L1_5_3_cons_prod_lock_0"}
    %B_L2L1_5_3_cons_cons_lock_0 = aie.lock(%tile_5_5, 3) {init = 0 : i32, sym_name = "B_L2L1_5_3_cons_cons_lock_0"}
    %B_L3L2_5_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 393216 : i32, sym_name = "B_L3L2_5_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_5_cons_buff_1 = aie.buffer(%mem_tile_5_1) {address = 402432 : i32, sym_name = "B_L3L2_5_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_5_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 0) {init = 2 : i32, sym_name = "B_L3L2_5_cons_prod_lock_0"}
    %B_L3L2_5_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 1) {init = 0 : i32, sym_name = "B_L3L2_5_cons_cons_lock_0"}
    %B_L3L2_5_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "B_L3L2_5_prod_lock_0"}
    %B_L3L2_5_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "B_L3L2_5_cons_lock_0"}
    %B_L2L1_4_0_cons_buff_0 = aie.buffer(%tile_4_2) {address = 36096 : i32, sym_name = "B_L2L1_4_0_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_4_0_cons_buff_1 = aie.buffer(%tile_4_2) {address = 45312 : i32, sym_name = "B_L2L1_4_0_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_4_0_cons_prod_lock_0 = aie.lock(%tile_4_2, 2) {init = 2 : i32, sym_name = "B_L2L1_4_0_cons_prod_lock_0"}
    %B_L2L1_4_0_cons_cons_lock_0 = aie.lock(%tile_4_2, 3) {init = 0 : i32, sym_name = "B_L2L1_4_0_cons_cons_lock_0"}
    %B_L2L1_4_1_cons_buff_0 = aie.buffer(%tile_4_3) {address = 36096 : i32, sym_name = "B_L2L1_4_1_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_4_1_cons_buff_1 = aie.buffer(%tile_4_3) {address = 45312 : i32, sym_name = "B_L2L1_4_1_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_4_1_cons_prod_lock_0 = aie.lock(%tile_4_3, 2) {init = 2 : i32, sym_name = "B_L2L1_4_1_cons_prod_lock_0"}
    %B_L2L1_4_1_cons_cons_lock_0 = aie.lock(%tile_4_3, 3) {init = 0 : i32, sym_name = "B_L2L1_4_1_cons_cons_lock_0"}
    %B_L2L1_4_2_cons_buff_0 = aie.buffer(%tile_4_4) {address = 36096 : i32, sym_name = "B_L2L1_4_2_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_4_2_cons_buff_1 = aie.buffer(%tile_4_4) {address = 45312 : i32, sym_name = "B_L2L1_4_2_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_4_2_cons_prod_lock_0 = aie.lock(%tile_4_4, 2) {init = 2 : i32, sym_name = "B_L2L1_4_2_cons_prod_lock_0"}
    %B_L2L1_4_2_cons_cons_lock_0 = aie.lock(%tile_4_4, 3) {init = 0 : i32, sym_name = "B_L2L1_4_2_cons_cons_lock_0"}
    %B_L2L1_4_3_cons_buff_0 = aie.buffer(%tile_4_5) {address = 36096 : i32, sym_name = "B_L2L1_4_3_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_4_3_cons_buff_1 = aie.buffer(%tile_4_5) {address = 45312 : i32, sym_name = "B_L2L1_4_3_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_4_3_cons_prod_lock_0 = aie.lock(%tile_4_5, 2) {init = 2 : i32, sym_name = "B_L2L1_4_3_cons_prod_lock_0"}
    %B_L2L1_4_3_cons_cons_lock_0 = aie.lock(%tile_4_5, 3) {init = 0 : i32, sym_name = "B_L2L1_4_3_cons_cons_lock_0"}
    %B_L3L2_4_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 393216 : i32, sym_name = "B_L3L2_4_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_4_cons_buff_1 = aie.buffer(%mem_tile_4_1) {address = 402432 : i32, sym_name = "B_L3L2_4_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_4_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 2) {init = 2 : i32, sym_name = "B_L3L2_4_cons_prod_lock_0"}
    %B_L3L2_4_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 3) {init = 0 : i32, sym_name = "B_L3L2_4_cons_cons_lock_0"}
    %B_L3L2_4_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "B_L3L2_4_prod_lock_0"}
    %B_L3L2_4_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "B_L3L2_4_cons_lock_0"}
    %B_L2L1_3_0_cons_buff_0 = aie.buffer(%tile_3_2) {address = 36096 : i32, sym_name = "B_L2L1_3_0_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_3_0_cons_buff_1 = aie.buffer(%tile_3_2) {address = 45312 : i32, sym_name = "B_L2L1_3_0_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_3_0_cons_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "B_L2L1_3_0_cons_prod_lock_0"}
    %B_L2L1_3_0_cons_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "B_L2L1_3_0_cons_cons_lock_0"}
    %B_L2L1_3_1_cons_buff_0 = aie.buffer(%tile_3_3) {address = 36096 : i32, sym_name = "B_L2L1_3_1_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_3_1_cons_buff_1 = aie.buffer(%tile_3_3) {address = 45312 : i32, sym_name = "B_L2L1_3_1_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_3_1_cons_prod_lock_0 = aie.lock(%tile_3_3, 2) {init = 2 : i32, sym_name = "B_L2L1_3_1_cons_prod_lock_0"}
    %B_L2L1_3_1_cons_cons_lock_0 = aie.lock(%tile_3_3, 3) {init = 0 : i32, sym_name = "B_L2L1_3_1_cons_cons_lock_0"}
    %B_L2L1_3_2_cons_buff_0 = aie.buffer(%tile_3_4) {address = 36096 : i32, sym_name = "B_L2L1_3_2_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_3_2_cons_buff_1 = aie.buffer(%tile_3_4) {address = 45312 : i32, sym_name = "B_L2L1_3_2_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_3_2_cons_prod_lock_0 = aie.lock(%tile_3_4, 2) {init = 2 : i32, sym_name = "B_L2L1_3_2_cons_prod_lock_0"}
    %B_L2L1_3_2_cons_cons_lock_0 = aie.lock(%tile_3_4, 3) {init = 0 : i32, sym_name = "B_L2L1_3_2_cons_cons_lock_0"}
    %B_L2L1_3_3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 36096 : i32, sym_name = "B_L2L1_3_3_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_3_3_cons_buff_1 = aie.buffer(%tile_3_5) {address = 45312 : i32, sym_name = "B_L2L1_3_3_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_3_3_cons_prod_lock_0 = aie.lock(%tile_3_5, 2) {init = 2 : i32, sym_name = "B_L2L1_3_3_cons_prod_lock_0"}
    %B_L2L1_3_3_cons_cons_lock_0 = aie.lock(%tile_3_5, 3) {init = 0 : i32, sym_name = "B_L2L1_3_3_cons_cons_lock_0"}
    %B_L3L2_3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 393216 : i32, sym_name = "B_L3L2_3_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 402432 : i32, sym_name = "B_L3L2_3_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "B_L3L2_3_cons_prod_lock_0"}
    %B_L3L2_3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "B_L3L2_3_cons_cons_lock_0"}
    %B_L3L2_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "B_L3L2_3_prod_lock_0"}
    %B_L3L2_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "B_L3L2_3_cons_lock_0"}
    %B_L2L1_2_0_cons_buff_0 = aie.buffer(%tile_2_2) {address = 36096 : i32, sym_name = "B_L2L1_2_0_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_2_0_cons_buff_1 = aie.buffer(%tile_2_2) {address = 45312 : i32, sym_name = "B_L2L1_2_0_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_2_0_cons_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "B_L2L1_2_0_cons_prod_lock_0"}
    %B_L2L1_2_0_cons_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "B_L2L1_2_0_cons_cons_lock_0"}
    %B_L2L1_2_1_cons_buff_0 = aie.buffer(%tile_2_3) {address = 36096 : i32, sym_name = "B_L2L1_2_1_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_2_1_cons_buff_1 = aie.buffer(%tile_2_3) {address = 45312 : i32, sym_name = "B_L2L1_2_1_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_2_1_cons_prod_lock_0 = aie.lock(%tile_2_3, 2) {init = 2 : i32, sym_name = "B_L2L1_2_1_cons_prod_lock_0"}
    %B_L2L1_2_1_cons_cons_lock_0 = aie.lock(%tile_2_3, 3) {init = 0 : i32, sym_name = "B_L2L1_2_1_cons_cons_lock_0"}
    %B_L2L1_2_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 36096 : i32, sym_name = "B_L2L1_2_2_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_2_2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 45312 : i32, sym_name = "B_L2L1_2_2_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_2_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 2) {init = 2 : i32, sym_name = "B_L2L1_2_2_cons_prod_lock_0"}
    %B_L2L1_2_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 3) {init = 0 : i32, sym_name = "B_L2L1_2_2_cons_cons_lock_0"}
    %B_L2L1_2_3_cons_buff_0 = aie.buffer(%tile_2_5) {address = 36096 : i32, sym_name = "B_L2L1_2_3_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_2_3_cons_buff_1 = aie.buffer(%tile_2_5) {address = 45312 : i32, sym_name = "B_L2L1_2_3_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_2_3_cons_prod_lock_0 = aie.lock(%tile_2_5, 2) {init = 2 : i32, sym_name = "B_L2L1_2_3_cons_prod_lock_0"}
    %B_L2L1_2_3_cons_cons_lock_0 = aie.lock(%tile_2_5, 3) {init = 0 : i32, sym_name = "B_L2L1_2_3_cons_cons_lock_0"}
    %B_L3L2_2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 393216 : i32, sym_name = "B_L3L2_2_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 402432 : i32, sym_name = "B_L3L2_2_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 2) {init = 2 : i32, sym_name = "B_L3L2_2_cons_prod_lock_0"}
    %B_L3L2_2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "B_L3L2_2_cons_cons_lock_0"}
    %B_L3L2_2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "B_L3L2_2_prod_lock_0"}
    %B_L3L2_2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "B_L3L2_2_cons_lock_0"}
    %B_L2L1_1_0_cons_buff_0 = aie.buffer(%tile_1_2) {address = 36096 : i32, sym_name = "B_L2L1_1_0_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_1_0_cons_buff_1 = aie.buffer(%tile_1_2) {address = 45312 : i32, sym_name = "B_L2L1_1_0_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_1_0_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "B_L2L1_1_0_cons_prod_lock_0"}
    %B_L2L1_1_0_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "B_L2L1_1_0_cons_cons_lock_0"}
    %B_L2L1_1_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 36096 : i32, sym_name = "B_L2L1_1_1_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_1_1_cons_buff_1 = aie.buffer(%tile_1_3) {address = 45312 : i32, sym_name = "B_L2L1_1_1_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_1_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "B_L2L1_1_1_cons_prod_lock_0"}
    %B_L2L1_1_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "B_L2L1_1_1_cons_cons_lock_0"}
    %B_L2L1_1_2_cons_buff_0 = aie.buffer(%tile_1_4) {address = 36096 : i32, sym_name = "B_L2L1_1_2_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_1_2_cons_buff_1 = aie.buffer(%tile_1_4) {address = 45312 : i32, sym_name = "B_L2L1_1_2_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_1_2_cons_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "B_L2L1_1_2_cons_prod_lock_0"}
    %B_L2L1_1_2_cons_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "B_L2L1_1_2_cons_cons_lock_0"}
    %B_L2L1_1_3_cons_buff_0 = aie.buffer(%tile_1_5) {address = 36096 : i32, sym_name = "B_L2L1_1_3_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_1_3_cons_buff_1 = aie.buffer(%tile_1_5) {address = 45312 : i32, sym_name = "B_L2L1_1_3_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_1_3_cons_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 2 : i32, sym_name = "B_L2L1_1_3_cons_prod_lock_0"}
    %B_L2L1_1_3_cons_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "B_L2L1_1_3_cons_cons_lock_0"}
    %B_L3L2_1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 393216 : i32, sym_name = "B_L3L2_1_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_1_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 402432 : i32, sym_name = "B_L3L2_1_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "B_L3L2_1_cons_prod_lock_0"}
    %B_L3L2_1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "B_L3L2_1_cons_cons_lock_0"}
    %B_L3L2_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "B_L3L2_1_prod_lock_0"}
    %B_L3L2_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "B_L3L2_1_cons_lock_0"}
    %B_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 36096 : i32, sym_name = "B_L2L1_0_0_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 45312 : i32, sym_name = "B_L2L1_0_0_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "B_L2L1_0_0_cons_prod_lock_0"}
    %B_L2L1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_L2L1_0_0_cons_cons_lock_0"}
    %B_L2L1_0_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 36096 : i32, sym_name = "B_L2L1_0_1_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_0_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 45312 : i32, sym_name = "B_L2L1_0_1_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_0_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "B_L2L1_0_1_cons_prod_lock_0"}
    %B_L2L1_0_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "B_L2L1_0_1_cons_cons_lock_0"}
    %B_L2L1_0_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 36096 : i32, sym_name = "B_L2L1_0_2_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_0_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 45312 : i32, sym_name = "B_L2L1_0_2_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_0_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "B_L2L1_0_2_cons_prod_lock_0"}
    %B_L2L1_0_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "B_L2L1_0_2_cons_cons_lock_0"}
    %B_L2L1_0_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 36096 : i32, sym_name = "B_L2L1_0_3_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_0_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 45312 : i32, sym_name = "B_L2L1_0_3_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L2L1_0_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "B_L2L1_0_3_cons_prod_lock_0"}
    %B_L2L1_0_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "B_L2L1_0_3_cons_cons_lock_0"}
    %B_L3L2_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 393216 : i32, sym_name = "B_L3L2_0_cons_buff_0"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 402432 : i32, sym_name = "B_L3L2_0_cons_buff_1"} : memref<64x16x!aiex.bfp<"v8bfp16ebs8">> 
    %B_L3L2_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "B_L3L2_0_cons_prod_lock_0"}
    %B_L3L2_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "B_L3L2_0_cons_cons_lock_0"}
    %B_L3L2_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_L3L2_0_prod_lock_0"}
    %B_L3L2_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_L3L2_0_cons_lock_0"}
    %A_L2L1_3_0_cons_buff_0 = aie.buffer(%tile_0_5) {address = 54528 : i32, sym_name = "A_L2L1_3_0_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_3_0_cons_buff_1 = aie.buffer(%tile_0_5) {address = 58624 : i32, sym_name = "A_L2L1_3_0_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_3_0_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "A_L2L1_3_0_cons_prod_lock_0"}
    %A_L2L1_3_0_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "A_L2L1_3_0_cons_cons_lock_0"}
    %A_L2L1_3_1_cons_buff_0 = aie.buffer(%tile_1_5) {address = 54528 : i32, sym_name = "A_L2L1_3_1_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_3_1_cons_buff_1 = aie.buffer(%tile_1_5) {address = 58624 : i32, sym_name = "A_L2L1_3_1_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_3_1_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "A_L2L1_3_1_cons_prod_lock_0"}
    %A_L2L1_3_1_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "A_L2L1_3_1_cons_cons_lock_0"}
    %A_L2L1_3_2_cons_buff_0 = aie.buffer(%tile_2_5) {address = 54528 : i32, sym_name = "A_L2L1_3_2_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_3_2_cons_buff_1 = aie.buffer(%tile_2_5) {address = 58624 : i32, sym_name = "A_L2L1_3_2_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_3_2_cons_prod_lock_0 = aie.lock(%tile_2_5, 0) {init = 2 : i32, sym_name = "A_L2L1_3_2_cons_prod_lock_0"}
    %A_L2L1_3_2_cons_cons_lock_0 = aie.lock(%tile_2_5, 1) {init = 0 : i32, sym_name = "A_L2L1_3_2_cons_cons_lock_0"}
    %A_L2L1_3_3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 54528 : i32, sym_name = "A_L2L1_3_3_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_3_3_cons_buff_1 = aie.buffer(%tile_3_5) {address = 58624 : i32, sym_name = "A_L2L1_3_3_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_3_3_cons_prod_lock_0 = aie.lock(%tile_3_5, 0) {init = 2 : i32, sym_name = "A_L2L1_3_3_cons_prod_lock_0"}
    %A_L2L1_3_3_cons_cons_lock_0 = aie.lock(%tile_3_5, 1) {init = 0 : i32, sym_name = "A_L2L1_3_3_cons_cons_lock_0"}
    %A_L2L1_3_4_cons_buff_0 = aie.buffer(%tile_4_5) {address = 54528 : i32, sym_name = "A_L2L1_3_4_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_3_4_cons_buff_1 = aie.buffer(%tile_4_5) {address = 58624 : i32, sym_name = "A_L2L1_3_4_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_3_4_cons_prod_lock_0 = aie.lock(%tile_4_5, 0) {init = 2 : i32, sym_name = "A_L2L1_3_4_cons_prod_lock_0"}
    %A_L2L1_3_4_cons_cons_lock_0 = aie.lock(%tile_4_5, 1) {init = 0 : i32, sym_name = "A_L2L1_3_4_cons_cons_lock_0"}
    %A_L2L1_3_5_cons_buff_0 = aie.buffer(%tile_5_5) {address = 54528 : i32, sym_name = "A_L2L1_3_5_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_3_5_cons_buff_1 = aie.buffer(%tile_5_5) {address = 58624 : i32, sym_name = "A_L2L1_3_5_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_3_5_cons_prod_lock_0 = aie.lock(%tile_5_5, 0) {init = 2 : i32, sym_name = "A_L2L1_3_5_cons_prod_lock_0"}
    %A_L2L1_3_5_cons_cons_lock_0 = aie.lock(%tile_5_5, 1) {init = 0 : i32, sym_name = "A_L2L1_3_5_cons_cons_lock_0"}
    %A_L2L1_3_6_cons_buff_0 = aie.buffer(%tile_6_5) {address = 54528 : i32, sym_name = "A_L2L1_3_6_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_3_6_cons_buff_1 = aie.buffer(%tile_6_5) {address = 58624 : i32, sym_name = "A_L2L1_3_6_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_3_6_cons_prod_lock_0 = aie.lock(%tile_6_5, 0) {init = 2 : i32, sym_name = "A_L2L1_3_6_cons_prod_lock_0"}
    %A_L2L1_3_6_cons_cons_lock_0 = aie.lock(%tile_6_5, 1) {init = 0 : i32, sym_name = "A_L2L1_3_6_cons_cons_lock_0"}
    %A_L2L1_3_7_cons_buff_0 = aie.buffer(%tile_7_5) {address = 54528 : i32, sym_name = "A_L2L1_3_7_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_3_7_cons_buff_1 = aie.buffer(%tile_7_5) {address = 58624 : i32, sym_name = "A_L2L1_3_7_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_3_7_cons_prod_lock_0 = aie.lock(%tile_7_5, 0) {init = 2 : i32, sym_name = "A_L2L1_3_7_cons_prod_lock_0"}
    %A_L2L1_3_7_cons_cons_lock_0 = aie.lock(%tile_7_5, 1) {init = 0 : i32, sym_name = "A_L2L1_3_7_cons_cons_lock_0"}
    %A_L3L2_3_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 131072 : i32, sym_name = "A_L3L2_3_cons_buff_0"} : memref<128x512xbf16> 
    %A_L3L2_3_cons_buff_1 = aie.buffer(%mem_tile_6_1) {address = 262144 : i32, sym_name = "A_L3L2_3_cons_buff_1"} : memref<128x512xbf16> 
    %A_L3L2_3_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 0) {init = 2 : i32, sym_name = "A_L3L2_3_cons_prod_lock_0"}
    %A_L3L2_3_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 1) {init = 0 : i32, sym_name = "A_L3L2_3_cons_cons_lock_0"}
    %A_L3L2_3_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "A_L3L2_3_prod_lock_0"}
    %A_L3L2_3_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "A_L3L2_3_cons_lock_0"}
    %A_L2L1_2_0_cons_buff_0 = aie.buffer(%tile_0_4) {address = 54528 : i32, sym_name = "A_L2L1_2_0_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_2_0_cons_buff_1 = aie.buffer(%tile_0_4) {address = 58624 : i32, sym_name = "A_L2L1_2_0_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_2_0_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "A_L2L1_2_0_cons_prod_lock_0"}
    %A_L2L1_2_0_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "A_L2L1_2_0_cons_cons_lock_0"}
    %A_L2L1_2_1_cons_buff_0 = aie.buffer(%tile_1_4) {address = 54528 : i32, sym_name = "A_L2L1_2_1_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_2_1_cons_buff_1 = aie.buffer(%tile_1_4) {address = 58624 : i32, sym_name = "A_L2L1_2_1_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_2_1_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "A_L2L1_2_1_cons_prod_lock_0"}
    %A_L2L1_2_1_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "A_L2L1_2_1_cons_cons_lock_0"}
    %A_L2L1_2_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 54528 : i32, sym_name = "A_L2L1_2_2_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_2_2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 58624 : i32, sym_name = "A_L2L1_2_2_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_2_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 0) {init = 2 : i32, sym_name = "A_L2L1_2_2_cons_prod_lock_0"}
    %A_L2L1_2_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 1) {init = 0 : i32, sym_name = "A_L2L1_2_2_cons_cons_lock_0"}
    %A_L2L1_2_3_cons_buff_0 = aie.buffer(%tile_3_4) {address = 54528 : i32, sym_name = "A_L2L1_2_3_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_2_3_cons_buff_1 = aie.buffer(%tile_3_4) {address = 58624 : i32, sym_name = "A_L2L1_2_3_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_2_3_cons_prod_lock_0 = aie.lock(%tile_3_4, 0) {init = 2 : i32, sym_name = "A_L2L1_2_3_cons_prod_lock_0"}
    %A_L2L1_2_3_cons_cons_lock_0 = aie.lock(%tile_3_4, 1) {init = 0 : i32, sym_name = "A_L2L1_2_3_cons_cons_lock_0"}
    %A_L2L1_2_4_cons_buff_0 = aie.buffer(%tile_4_4) {address = 54528 : i32, sym_name = "A_L2L1_2_4_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_2_4_cons_buff_1 = aie.buffer(%tile_4_4) {address = 58624 : i32, sym_name = "A_L2L1_2_4_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_2_4_cons_prod_lock_0 = aie.lock(%tile_4_4, 0) {init = 2 : i32, sym_name = "A_L2L1_2_4_cons_prod_lock_0"}
    %A_L2L1_2_4_cons_cons_lock_0 = aie.lock(%tile_4_4, 1) {init = 0 : i32, sym_name = "A_L2L1_2_4_cons_cons_lock_0"}
    %A_L2L1_2_5_cons_buff_0 = aie.buffer(%tile_5_4) {address = 54528 : i32, sym_name = "A_L2L1_2_5_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_2_5_cons_buff_1 = aie.buffer(%tile_5_4) {address = 58624 : i32, sym_name = "A_L2L1_2_5_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_2_5_cons_prod_lock_0 = aie.lock(%tile_5_4, 0) {init = 2 : i32, sym_name = "A_L2L1_2_5_cons_prod_lock_0"}
    %A_L2L1_2_5_cons_cons_lock_0 = aie.lock(%tile_5_4, 1) {init = 0 : i32, sym_name = "A_L2L1_2_5_cons_cons_lock_0"}
    %A_L2L1_2_6_cons_buff_0 = aie.buffer(%tile_6_4) {address = 54528 : i32, sym_name = "A_L2L1_2_6_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_2_6_cons_buff_1 = aie.buffer(%tile_6_4) {address = 58624 : i32, sym_name = "A_L2L1_2_6_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_2_6_cons_prod_lock_0 = aie.lock(%tile_6_4, 0) {init = 2 : i32, sym_name = "A_L2L1_2_6_cons_prod_lock_0"}
    %A_L2L1_2_6_cons_cons_lock_0 = aie.lock(%tile_6_4, 1) {init = 0 : i32, sym_name = "A_L2L1_2_6_cons_cons_lock_0"}
    %A_L2L1_2_7_cons_buff_0 = aie.buffer(%tile_7_4) {address = 54528 : i32, sym_name = "A_L2L1_2_7_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_2_7_cons_buff_1 = aie.buffer(%tile_7_4) {address = 58624 : i32, sym_name = "A_L2L1_2_7_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_2_7_cons_prod_lock_0 = aie.lock(%tile_7_4, 0) {init = 2 : i32, sym_name = "A_L2L1_2_7_cons_prod_lock_0"}
    %A_L2L1_2_7_cons_cons_lock_0 = aie.lock(%tile_7_4, 1) {init = 0 : i32, sym_name = "A_L2L1_2_7_cons_cons_lock_0"}
    %A_L3L2_2_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 131072 : i32, sym_name = "A_L3L2_2_cons_buff_0"} : memref<128x512xbf16> 
    %A_L3L2_2_cons_buff_1 = aie.buffer(%mem_tile_4_1) {address = 262144 : i32, sym_name = "A_L3L2_2_cons_buff_1"} : memref<128x512xbf16> 
    %A_L3L2_2_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 0) {init = 2 : i32, sym_name = "A_L3L2_2_cons_prod_lock_0"}
    %A_L3L2_2_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 1) {init = 0 : i32, sym_name = "A_L3L2_2_cons_cons_lock_0"}
    %A_L3L2_2_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "A_L3L2_2_prod_lock_0"}
    %A_L3L2_2_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "A_L3L2_2_cons_lock_0"}
    %A_L2L1_1_0_cons_buff_0 = aie.buffer(%tile_0_3) {address = 54528 : i32, sym_name = "A_L2L1_1_0_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_1_0_cons_buff_1 = aie.buffer(%tile_0_3) {address = 58624 : i32, sym_name = "A_L2L1_1_0_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_1_0_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "A_L2L1_1_0_cons_prod_lock_0"}
    %A_L2L1_1_0_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "A_L2L1_1_0_cons_cons_lock_0"}
    %A_L2L1_1_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 54528 : i32, sym_name = "A_L2L1_1_1_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_1_1_cons_buff_1 = aie.buffer(%tile_1_3) {address = 58624 : i32, sym_name = "A_L2L1_1_1_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_1_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "A_L2L1_1_1_cons_prod_lock_0"}
    %A_L2L1_1_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "A_L2L1_1_1_cons_cons_lock_0"}
    %A_L2L1_1_2_cons_buff_0 = aie.buffer(%tile_2_3) {address = 54528 : i32, sym_name = "A_L2L1_1_2_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_1_2_cons_buff_1 = aie.buffer(%tile_2_3) {address = 58624 : i32, sym_name = "A_L2L1_1_2_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_1_2_cons_prod_lock_0 = aie.lock(%tile_2_3, 0) {init = 2 : i32, sym_name = "A_L2L1_1_2_cons_prod_lock_0"}
    %A_L2L1_1_2_cons_cons_lock_0 = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "A_L2L1_1_2_cons_cons_lock_0"}
    %A_L2L1_1_3_cons_buff_0 = aie.buffer(%tile_3_3) {address = 54528 : i32, sym_name = "A_L2L1_1_3_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_1_3_cons_buff_1 = aie.buffer(%tile_3_3) {address = 58624 : i32, sym_name = "A_L2L1_1_3_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_1_3_cons_prod_lock_0 = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "A_L2L1_1_3_cons_prod_lock_0"}
    %A_L2L1_1_3_cons_cons_lock_0 = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "A_L2L1_1_3_cons_cons_lock_0"}
    %A_L2L1_1_4_cons_buff_0 = aie.buffer(%tile_4_3) {address = 54528 : i32, sym_name = "A_L2L1_1_4_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_1_4_cons_buff_1 = aie.buffer(%tile_4_3) {address = 58624 : i32, sym_name = "A_L2L1_1_4_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_1_4_cons_prod_lock_0 = aie.lock(%tile_4_3, 0) {init = 2 : i32, sym_name = "A_L2L1_1_4_cons_prod_lock_0"}
    %A_L2L1_1_4_cons_cons_lock_0 = aie.lock(%tile_4_3, 1) {init = 0 : i32, sym_name = "A_L2L1_1_4_cons_cons_lock_0"}
    %A_L2L1_1_5_cons_buff_0 = aie.buffer(%tile_5_3) {address = 54528 : i32, sym_name = "A_L2L1_1_5_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_1_5_cons_buff_1 = aie.buffer(%tile_5_3) {address = 58624 : i32, sym_name = "A_L2L1_1_5_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_1_5_cons_prod_lock_0 = aie.lock(%tile_5_3, 0) {init = 2 : i32, sym_name = "A_L2L1_1_5_cons_prod_lock_0"}
    %A_L2L1_1_5_cons_cons_lock_0 = aie.lock(%tile_5_3, 1) {init = 0 : i32, sym_name = "A_L2L1_1_5_cons_cons_lock_0"}
    %A_L2L1_1_6_cons_buff_0 = aie.buffer(%tile_6_3) {address = 54528 : i32, sym_name = "A_L2L1_1_6_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_1_6_cons_buff_1 = aie.buffer(%tile_6_3) {address = 58624 : i32, sym_name = "A_L2L1_1_6_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_1_6_cons_prod_lock_0 = aie.lock(%tile_6_3, 0) {init = 2 : i32, sym_name = "A_L2L1_1_6_cons_prod_lock_0"}
    %A_L2L1_1_6_cons_cons_lock_0 = aie.lock(%tile_6_3, 1) {init = 0 : i32, sym_name = "A_L2L1_1_6_cons_cons_lock_0"}
    %A_L2L1_1_7_cons_buff_0 = aie.buffer(%tile_7_3) {address = 54528 : i32, sym_name = "A_L2L1_1_7_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_1_7_cons_buff_1 = aie.buffer(%tile_7_3) {address = 58624 : i32, sym_name = "A_L2L1_1_7_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_1_7_cons_prod_lock_0 = aie.lock(%tile_7_3, 0) {init = 2 : i32, sym_name = "A_L2L1_1_7_cons_prod_lock_0"}
    %A_L2L1_1_7_cons_cons_lock_0 = aie.lock(%tile_7_3, 1) {init = 0 : i32, sym_name = "A_L2L1_1_7_cons_cons_lock_0"}
    %A_L3L2_1_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 131072 : i32, sym_name = "A_L3L2_1_cons_buff_0"} : memref<128x512xbf16> 
    %A_L3L2_1_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 262144 : i32, sym_name = "A_L3L2_1_cons_buff_1"} : memref<128x512xbf16> 
    %A_L3L2_1_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "A_L3L2_1_cons_prod_lock_0"}
    %A_L3L2_1_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "A_L3L2_1_cons_cons_lock_0"}
    %A_L3L2_1_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "A_L3L2_1_prod_lock_0"}
    %A_L3L2_1_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "A_L3L2_1_cons_lock_0"}
    %A_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 54528 : i32, sym_name = "A_L2L1_0_0_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 58624 : i32, sym_name = "A_L2L1_0_0_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_0_cons_prod_lock_0"}
    %A_L2L1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_0_cons_cons_lock_0"}
    %A_L2L1_0_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 54528 : i32, sym_name = "A_L2L1_0_1_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_0_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 58624 : i32, sym_name = "A_L2L1_0_1_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_0_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_1_cons_prod_lock_0"}
    %A_L2L1_0_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_1_cons_cons_lock_0"}
    %A_L2L1_0_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 54528 : i32, sym_name = "A_L2L1_0_2_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_0_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 58624 : i32, sym_name = "A_L2L1_0_2_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_0_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_2_cons_prod_lock_0"}
    %A_L2L1_0_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_2_cons_cons_lock_0"}
    %A_L2L1_0_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 54528 : i32, sym_name = "A_L2L1_0_3_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_0_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 58624 : i32, sym_name = "A_L2L1_0_3_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_0_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_3_cons_prod_lock_0"}
    %A_L2L1_0_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_3_cons_cons_lock_0"}
    %A_L2L1_0_4_cons_buff_0 = aie.buffer(%tile_4_2) {address = 54528 : i32, sym_name = "A_L2L1_0_4_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_0_4_cons_buff_1 = aie.buffer(%tile_4_2) {address = 58624 : i32, sym_name = "A_L2L1_0_4_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_0_4_cons_prod_lock_0 = aie.lock(%tile_4_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_4_cons_prod_lock_0"}
    %A_L2L1_0_4_cons_cons_lock_0 = aie.lock(%tile_4_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_4_cons_cons_lock_0"}
    %A_L2L1_0_5_cons_buff_0 = aie.buffer(%tile_5_2) {address = 54528 : i32, sym_name = "A_L2L1_0_5_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_0_5_cons_buff_1 = aie.buffer(%tile_5_2) {address = 58624 : i32, sym_name = "A_L2L1_0_5_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_0_5_cons_prod_lock_0 = aie.lock(%tile_5_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_5_cons_prod_lock_0"}
    %A_L2L1_0_5_cons_cons_lock_0 = aie.lock(%tile_5_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_5_cons_cons_lock_0"}
    %A_L2L1_0_6_cons_buff_0 = aie.buffer(%tile_6_2) {address = 54528 : i32, sym_name = "A_L2L1_0_6_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_0_6_cons_buff_1 = aie.buffer(%tile_6_2) {address = 58624 : i32, sym_name = "A_L2L1_0_6_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_0_6_cons_prod_lock_0 = aie.lock(%tile_6_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_6_cons_prod_lock_0"}
    %A_L2L1_0_6_cons_cons_lock_0 = aie.lock(%tile_6_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_6_cons_cons_lock_0"}
    %A_L2L1_0_7_cons_buff_0 = aie.buffer(%tile_7_2) {address = 54528 : i32, sym_name = "A_L2L1_0_7_cons_buff_0"} : memref<32x64xbf16> 
    %A_L2L1_0_7_cons_buff_1 = aie.buffer(%tile_7_2) {address = 58624 : i32, sym_name = "A_L2L1_0_7_cons_buff_1"} : memref<32x64xbf16> 
    %A_L2L1_0_7_cons_prod_lock_0 = aie.lock(%tile_7_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_7_cons_prod_lock_0"}
    %A_L2L1_0_7_cons_cons_lock_0 = aie.lock(%tile_7_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_7_cons_cons_lock_0"}
    %A_L3L2_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, sym_name = "A_L3L2_0_cons_buff_0"} : memref<128x512xbf16> 
    %A_L3L2_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 262144 : i32, sym_name = "A_L3L2_0_cons_buff_1"} : memref<128x512xbf16> 
    %A_L3L2_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "A_L3L2_0_cons_prod_lock_0"}
    %A_L3L2_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "A_L3L2_0_cons_cons_lock_0"}
    %A_L3L2_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L2_0_prod_lock_0"}
    %A_L3L2_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L2_0_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 4)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_7_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_6_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_5_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_4_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_3_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_2_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_1_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 4)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_7_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_6_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_5_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_4_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %mem_tile_4_1, DMA : 4)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_7_4, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_6_4, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_5_4, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_4_4, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_3_4, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_2_4, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_1_4, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_0_4, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 4)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_7_5, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_6_5, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_5_5, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_4_5, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_3_5, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_2_5, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_1_5, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_0_5, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 5)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_2, DMA : 1)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %mem_tile_2_1, DMA : 5)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_2_5, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_2_2, DMA : 1)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_4, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_3, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_2, DMA : 1)
    aie.flow(%shim_noc_tile_4_0, DMA : 1, %mem_tile_4_1, DMA : 5)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_4_5, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_4_4, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_4_3, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 2, %tile_4_2, DMA : 1)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_5_5, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_5_4, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_5_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_5_2, DMA : 1)
    aie.flow(%shim_noc_tile_6_0, DMA : 1, %mem_tile_6_1, DMA : 5)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_6_5, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_6_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_6_3, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_6_2, DMA : 1)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_7_5, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_7_4, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_7_3, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_7_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_0_1, DMA : 3)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 1)
    aie.flow(%tile_1_3, DMA : 0, %mem_tile_1_1, DMA : 2)
    aie.flow(%tile_1_4, DMA : 0, %mem_tile_1_1, DMA : 3)
    aie.flow(%tile_1_5, DMA : 0, %mem_tile_1_1, DMA : 4)
    aie.flow(%mem_tile_1_1, DMA : 1, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%tile_2_3, DMA : 0, %mem_tile_2_1, DMA : 1)
    aie.flow(%tile_2_4, DMA : 0, %mem_tile_2_1, DMA : 2)
    aie.flow(%tile_2_5, DMA : 0, %mem_tile_2_1, DMA : 3)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 1)
    aie.flow(%tile_3_3, DMA : 0, %mem_tile_3_1, DMA : 2)
    aie.flow(%tile_3_4, DMA : 0, %mem_tile_3_1, DMA : 3)
    aie.flow(%tile_3_5, DMA : 0, %mem_tile_3_1, DMA : 4)
    aie.flow(%mem_tile_3_1, DMA : 1, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%tile_4_2, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%tile_4_3, DMA : 0, %mem_tile_4_1, DMA : 1)
    aie.flow(%tile_4_4, DMA : 0, %mem_tile_4_1, DMA : 2)
    aie.flow(%tile_4_5, DMA : 0, %mem_tile_4_1, DMA : 3)
    aie.flow(%mem_tile_4_1, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%tile_5_2, DMA : 0, %mem_tile_5_1, DMA : 1)
    aie.flow(%tile_5_3, DMA : 0, %mem_tile_5_1, DMA : 2)
    aie.flow(%tile_5_4, DMA : 0, %mem_tile_5_1, DMA : 3)
    aie.flow(%tile_5_5, DMA : 0, %mem_tile_5_1, DMA : 4)
    aie.flow(%mem_tile_5_1, DMA : 1, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%tile_6_2, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%tile_6_3, DMA : 0, %mem_tile_6_1, DMA : 1)
    aie.flow(%tile_6_4, DMA : 0, %mem_tile_6_1, DMA : 2)
    aie.flow(%tile_6_5, DMA : 0, %mem_tile_6_1, DMA : 3)
    aie.flow(%mem_tile_6_1, DMA : 0, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%tile_7_2, DMA : 0, %mem_tile_7_1, DMA : 1)
    aie.flow(%tile_7_3, DMA : 0, %mem_tile_7_1, DMA : 2)
    aie.flow(%tile_7_4, DMA : 0, %mem_tile_7_1, DMA : 3)
    aie.flow(%tile_7_5, DMA : 0, %mem_tile_7_1, DMA : 4)
    aie.flow(%mem_tile_7_1, DMA : 1, %shim_noc_tile_7_0, DMA : 0)
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 62720 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_0_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_0_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_0_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_0_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_0_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous0[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_0_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_0_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_0_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_0_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_0_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous0[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous0[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous1 = aie.buffer(%tile_1_2) {address = 62720 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
    %core_1_2 = aie.core(%tile_1_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous1[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_1_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_1_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_1_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_1_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_1_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous1[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_1_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_1_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_1_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_1_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_1_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous1[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous1[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous2 = aie.buffer(%tile_2_2) {address = 62720 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
    %core_2_2 = aie.core(%tile_2_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous2[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_2_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_2_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_2_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_2_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_2_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous2[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_2_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_2_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_2_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_2_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_2_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous2[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous2[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous3 = aie.buffer(%tile_3_2) {address = 62720 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
    %core_3_2 = aie.core(%tile_3_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous3[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_3_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_3_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_3_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_3_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_3_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous3[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_3_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_3_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_3_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_3_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_3_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous3[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous3[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous4 = aie.buffer(%tile_4_2) {address = 62720 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
    %core_4_2 = aie.core(%tile_4_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous4[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_4_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_4_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_4_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_4_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_4_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_0_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous4[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_4_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_4_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_4_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_4_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_4_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_0_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous4[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous4[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous5 = aie.buffer(%tile_5_2) {address = 62720 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
    %core_5_2 = aie.core(%tile_5_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous5[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_5_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_5_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_5_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_5_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_5_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_0_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous5[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_5_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_5_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_5_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_5_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_5_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_0_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous5[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous5[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous6 = aie.buffer(%tile_6_2) {address = 62720 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
    %core_6_2 = aie.core(%tile_6_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous6[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_6_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_6_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_6_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_6_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_6_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_0_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous6[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_6_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_6_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_6_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_6_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_6_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_0_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous6[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous6[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous7 = aie.buffer(%tile_7_2) {address = 62720 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
    %core_7_2 = aie.core(%tile_7_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous7[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_7_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_7_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_7_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_7_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_7_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_0_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous7[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_7_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_7_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_7_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_7_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_7_0_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_0_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous7[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous7[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous8 = aie.buffer(%tile_0_3) {address = 62720 : i32, sym_name = "_anonymous8"} : memref<3xi32> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous8[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_0_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_0_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous8[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_0_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous8[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_0_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous8[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_0_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous8[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous8[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous8[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_0_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_0_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous8[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_0_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous8[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_0_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous8[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_0_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous8[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous8[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous8[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous9 = aie.buffer(%tile_1_3) {address = 62720 : i32, sym_name = "_anonymous9"} : memref<3xi32> 
    %core_1_3 = aie.core(%tile_1_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous9[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_1_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_1_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous9[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_1_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous9[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_1_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous9[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_1_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous9[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous9[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous9[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_1_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_1_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous9[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_1_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous9[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_1_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous9[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_1_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous9[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous9[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous9[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous10 = aie.buffer(%tile_2_3) {address = 62720 : i32, sym_name = "_anonymous10"} : memref<3xi32> 
    %core_2_3 = aie.core(%tile_2_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous10[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_2_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_2_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous10[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_2_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous10[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_2_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous10[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_2_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous10[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous10[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous10[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous10[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_2_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_2_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous10[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_2_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous10[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_2_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous10[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_2_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous10[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous10[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous10[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous10[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous11 = aie.buffer(%tile_3_3) {address = 62720 : i32, sym_name = "_anonymous11"} : memref<3xi32> 
    %core_3_3 = aie.core(%tile_3_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous11[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_3_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_3_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous11[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_3_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous11[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_3_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous11[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_3_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous11[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous11[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous11[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous11[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_3_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_3_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous11[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_3_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous11[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_3_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous11[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_3_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous11[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous11[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous11[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous11[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous12 = aie.buffer(%tile_4_3) {address = 62720 : i32, sym_name = "_anonymous12"} : memref<3xi32> 
    %core_4_3 = aie.core(%tile_4_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous12[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_4_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_4_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_4_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_4_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_4_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous12[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_4_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_4_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous12[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_4_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_4_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous12[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_4_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_4_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous12[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_1_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous12[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_4_1_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous12[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_4_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_4_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_4_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_4_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous12[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_4_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_4_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous12[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_4_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_4_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous12[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_4_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_4_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous12[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_1_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous12[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_4_1_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous12[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous13 = aie.buffer(%tile_5_3) {address = 62720 : i32, sym_name = "_anonymous13"} : memref<3xi32> 
    %core_5_3 = aie.core(%tile_5_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous13[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_5_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_5_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_5_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_5_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_5_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous13[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_5_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_5_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous13[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_5_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_5_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous13[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_5_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_5_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous13[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_1_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous13[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_5_1_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous13[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_5_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_5_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_5_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_5_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous13[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_5_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_5_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous13[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_5_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_5_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous13[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_5_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_5_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous13[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_1_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous13[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_5_1_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous13[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous14 = aie.buffer(%tile_6_3) {address = 62720 : i32, sym_name = "_anonymous14"} : memref<3xi32> 
    %core_6_3 = aie.core(%tile_6_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous14[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous14[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous14[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_6_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_6_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_6_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_6_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_6_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous14[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_6_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_6_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous14[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_6_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_6_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous14[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_6_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_6_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous14[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_1_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous14[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_6_1_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous14[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous14[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_6_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_6_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_6_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_6_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous14[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_6_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_6_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous14[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_6_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_6_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous14[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_6_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_6_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous14[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_1_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous14[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_6_1_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous14[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous14[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous15 = aie.buffer(%tile_7_3) {address = 62720 : i32, sym_name = "_anonymous15"} : memref<3xi32> 
    %core_7_3 = aie.core(%tile_7_3) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous15[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous15[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous15[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_7_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_7_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_7_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_7_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_7_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous15[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_7_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_7_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous15[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_7_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_7_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous15[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_7_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_7_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous15[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_1_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous15[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_7_1_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous15[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous15[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_7_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_7_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_7_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_7_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous15[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_7_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_7_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous15[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_7_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_7_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous15[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_7_1_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_1_7_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous15[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_1_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous15[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_7_1_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous15[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous15[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous16 = aie.buffer(%tile_0_4) {address = 62720 : i32, sym_name = "_anonymous16"} : memref<3xi32> 
    %core_0_4 = aie.core(%tile_0_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous16[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous16[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous16[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_0_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_0_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous16[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_0_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous16[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_0_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous16[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_0_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous16[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous16[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous16[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous16[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_0_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_0_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous16[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_0_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous16[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_0_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous16[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_0_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous16[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous16[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous16[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous16[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous17 = aie.buffer(%tile_1_4) {address = 62720 : i32, sym_name = "_anonymous17"} : memref<3xi32> 
    %core_1_4 = aie.core(%tile_1_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous17[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous17[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous17[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_1_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous17[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_1_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous17[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_1_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous17[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_1_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous17[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_1_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous17[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous17[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous17[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous17[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous17[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_1_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous17[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_1_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous17[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_1_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous17[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_1_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous17[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_1_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous17[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous17[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous17[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous17[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous17[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous17[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous18 = aie.buffer(%tile_2_4) {address = 62720 : i32, sym_name = "_anonymous18"} : memref<3xi32> 
    %core_2_4 = aie.core(%tile_2_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous18[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous18[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous18[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_2_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous18[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_2_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous18[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_2_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous18[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_2_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous18[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_2_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous18[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous18[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous18[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous18[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous18[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_2_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous18[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_2_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous18[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_2_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous18[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_2_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous18[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_2_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous18[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous18[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous18[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous18[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous18[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous18[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous19 = aie.buffer(%tile_3_4) {address = 62720 : i32, sym_name = "_anonymous19"} : memref<3xi32> 
    %core_3_4 = aie.core(%tile_3_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous19[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous19[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous19[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_3_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous19[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_3_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous19[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_3_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous19[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_3_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous19[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_3_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous19[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous19[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous19[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous19[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous19[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_3_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous19[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_3_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous19[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_3_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous19[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_3_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous19[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_3_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous19[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous19[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous19[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous19[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous19[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous19[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous20 = aie.buffer(%tile_4_4) {address = 62720 : i32, sym_name = "_anonymous20"} : memref<3xi32> 
    %core_4_4 = aie.core(%tile_4_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous20[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous20[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous20[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_4_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_4_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_4_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_4_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_4_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous20[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_4_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_4_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous20[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_4_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_4_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous20[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_4_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_4_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous20[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_2_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous20[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_4_2_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous20[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous20[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_4_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_4_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_4_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_4_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous20[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_4_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_4_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous20[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_4_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_4_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous20[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_4_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_4_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous20[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous20[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_2_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous20[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous20[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_4_2_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous20[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous20[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous21 = aie.buffer(%tile_5_4) {address = 62720 : i32, sym_name = "_anonymous21"} : memref<3xi32> 
    %core_5_4 = aie.core(%tile_5_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous21[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous21[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous21[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_5_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_5_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_5_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous21[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_5_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_5_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous21[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_5_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_5_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous21[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_5_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_5_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous21[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_5_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_5_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous21[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_2_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous21[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous21[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_5_2_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous21[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous21[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_5_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_5_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous21[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_5_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_5_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous21[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_5_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_5_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous21[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_5_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_5_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous21[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_5_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_5_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous21[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous21[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_2_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous21[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous21[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_5_2_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous21[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous21[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous22 = aie.buffer(%tile_6_4) {address = 62720 : i32, sym_name = "_anonymous22"} : memref<3xi32> 
    %core_6_4 = aie.core(%tile_6_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous22[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous22[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous22[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_6_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_6_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_6_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous22[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_6_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_6_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous22[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_6_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_6_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous22[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_6_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_6_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous22[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_6_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_6_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous22[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_2_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous22[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous22[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_6_2_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous22[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous22[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_6_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_6_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous22[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_6_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_6_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous22[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_6_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_6_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous22[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_6_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_6_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous22[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_6_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_6_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous22[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous22[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_2_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous22[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous22[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_6_2_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous22[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous22[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous23 = aie.buffer(%tile_7_4) {address = 62720 : i32, sym_name = "_anonymous23"} : memref<3xi32> 
    %core_7_4 = aie.core(%tile_7_4) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous23[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous23[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous23[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_7_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_7_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_7_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous23[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_7_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_7_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous23[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_7_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_7_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous23[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_7_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_7_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous23[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_7_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_7_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous23[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_2_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous23[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous23[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_7_2_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous23[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous23[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_7_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_7_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous23[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_7_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_7_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous23[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_7_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_7_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous23[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_7_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_7_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous23[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_7_2_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_2_7_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous23[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous23[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_2_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous23[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous23[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_7_2_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous23[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous23[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous24 = aie.buffer(%tile_0_5) {address = 62720 : i32, sym_name = "_anonymous24"} : memref<3xi32> 
    %core_0_5 = aie.core(%tile_0_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous24[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous24[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous24[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_0_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous24[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_0_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous24[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_0_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous24[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_0_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous24[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_0_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous24[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous24[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous24[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous24[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous24[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_0_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous24[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_0_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous24[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_0_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous24[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_0_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous24[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_0_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_0_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous24[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous24[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous24[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous24[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous24[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous24[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous25 = aie.buffer(%tile_1_5) {address = 62720 : i32, sym_name = "_anonymous25"} : memref<3xi32> 
    %core_1_5 = aie.core(%tile_1_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous25[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous25[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous25[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_1_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous25[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_1_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous25[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_1_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous25[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_1_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous25[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_1_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous25[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous25[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous25[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous25[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous25[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_1_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous25[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_1_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous25[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_1_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous25[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_1_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous25[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_1_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_1_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous25[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous25[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous25[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous25[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous25[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous25[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous26 = aie.buffer(%tile_2_5) {address = 62720 : i32, sym_name = "_anonymous26"} : memref<3xi32> 
    %core_2_5 = aie.core(%tile_2_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous26[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous26[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous26[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_2_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous26[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_2_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous26[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_2_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous26[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_2_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous26[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_2_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous26[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous26[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous26[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous26[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous26[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_2_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous26[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_2_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous26[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_2_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous26[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_2_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous26[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_2_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_2_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous26[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous26[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous26[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous26[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous26[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous26[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous27 = aie.buffer(%tile_3_5) {address = 62720 : i32, sym_name = "_anonymous27"} : memref<3xi32> 
    %core_3_5 = aie.core(%tile_3_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous27[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous27[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous27[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_3_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous27[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_3_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous27[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_3_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous27[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_3_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous27[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_3_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous27[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous27[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous27[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous27[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous27[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_3_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous27[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_3_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous27[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_3_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous27[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_3_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous27[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_3_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_3_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous27[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous27[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous27[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous27[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous27[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous27[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous28 = aie.buffer(%tile_4_5) {address = 62720 : i32, sym_name = "_anonymous28"} : memref<3xi32> 
    %core_4_5 = aie.core(%tile_4_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous28[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous28[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous28[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_4_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_4_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_4_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous28[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_4_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_4_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous28[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_4_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_4_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous28[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_4_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_4_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous28[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_4_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_4_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous28[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_3_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous28[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous28[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_4_3_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous28[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous28[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_4_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_4_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous28[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_4_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_4_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous28[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_4_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_4_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous28[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_4_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_4_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous28[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_4_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_4_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_4_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous28[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous28[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_3_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous28[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous28[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_4_3_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous28[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous28[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous29 = aie.buffer(%tile_5_5) {address = 62720 : i32, sym_name = "_anonymous29"} : memref<3xi32> 
    %core_5_5 = aie.core(%tile_5_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous29[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous29[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous29[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_5_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_5_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_5_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous29[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_5_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_5_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous29[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_5_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_5_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous29[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_5_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_5_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous29[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_5_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_5_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous29[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_3_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous29[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous29[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_5_3_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous29[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous29[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_5_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_5_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous29[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_5_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_5_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous29[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_5_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_5_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous29[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_5_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_5_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous29[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_5_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_5_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_5_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous29[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous29[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_3_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous29[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous29[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_5_3_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous29[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous29[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous30 = aie.buffer(%tile_6_5) {address = 62720 : i32, sym_name = "_anonymous30"} : memref<3xi32> 
    %core_6_5 = aie.core(%tile_6_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous30[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous30[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous30[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_6_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_6_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_6_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous30[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_6_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_6_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous30[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_6_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_6_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous30[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_6_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_6_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous30[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_6_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_6_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous30[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_3_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous30[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous30[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_6_3_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous30[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous30[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_6_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_6_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous30[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_6_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_6_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous30[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_6_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_6_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous30[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_6_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_6_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous30[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_6_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_6_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_6_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous30[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous30[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_3_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous30[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous30[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_6_3_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous30[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous30[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous31 = aie.buffer(%tile_7_5) {address = 62720 : i32, sym_name = "_anonymous31"} : memref<3xi32> 
    %core_7_5 = aie.core(%tile_7_5) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous31[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous31[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous31[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_7_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_7_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_7_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous31[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb8(%7: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%A_L2L1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb12(%11: memref<32x64xbf16>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_vectorized_different_datatypes(%11, %7, %C_L1L2_7_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_7_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous31[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %17 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb16(%20: memref<32x64xbf16>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_vectorized_different_datatypes(%20, %7, %C_L1L2_7_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_7_cons_prod_lock_0, Release, 1)
      %21 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous31[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %26 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.index_cast %27 : index to i32
      cf.switch %28 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb20(%29: memref<32x64xbf16>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_vectorized_different_datatypes(%29, %7, %C_L1L2_7_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_7_cons_prod_lock_0, Release, 1)
      %30 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c2_i32 : i32
      %33 = arith.subi %31, %c2_i32 : i32
      %34 = arith.select %32, %33, %31 : i32
      memref.store %34, %_anonymous31[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %35 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %36 : index to i32
      cf.switch %37 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb24(%38: memref<32x64xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_vectorized_different_datatypes(%38, %7, %C_L1L2_7_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_7_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous31[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_3_cons_prod_lock_0, Release, 1)
      %44 = memref.load %_anonymous31[%c1] : memref<3xi32>
      %45 = arith.addi %44, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.subi %45, %c2_i32 : i32
      %48 = arith.select %46, %47, %45 : i32
      memref.store %48, %_anonymous31[%c1] : memref<3xi32>
      %49 = arith.addi %2, %c1 : index
      cf.br ^bb3(%49 : index)
    ^bb25:  // pred: ^bb3
      aie.use_lock(%C_L1L2_7_3_cons_lock_0, Release, 1)
      %50 = memref.load %_anonymous31[%c0] : memref<3xi32>
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.cmpi sge, %51, %c1_i32 : i32
      %53 = arith.select %52, %50, %51 : i32
      memref.store %53, %_anonymous31[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%C_L1L2_7_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_7_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous31[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>)
    ^bb31(%59: memref<64x16x!aiex.bfp<"v8bfp16ebs8">>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%A_L2L1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %60 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %61 : index to i32
      cf.switch %62 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb35(%63: memref<32x64xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_vectorized_different_datatypes(%63, %59, %C_L1L2_7_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_7_cons_prod_lock_0, Release, 1)
      %64 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %65 = arith.addi %64, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c2_i32 : i32
      %67 = arith.subi %65, %c2_i32 : i32
      %68 = arith.select %66, %67, %65 : i32
      memref.store %68, %_anonymous31[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb38,
        0: ^bb36,
        1: ^bb37
      ]
    ^bb36:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb39(%72: memref<32x64xbf16>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_vectorized_different_datatypes(%72, %59, %C_L1L2_7_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_7_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous31[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %78 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.index_cast %79 : index to i32
      cf.switch %80 : i32, [
        default: ^bb42,
        0: ^bb40,
        1: ^bb41
      ]
    ^bb40:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb43(%81: memref<32x64xbf16>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_vectorized_different_datatypes(%81, %59, %C_L1L2_7_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_7_cons_prod_lock_0, Release, 1)
      %82 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c2_i32 : i32
      %85 = arith.subi %83, %c2_i32 : i32
      %86 = arith.select %84, %85, %83 : i32
      memref.store %86, %_anonymous31[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %87 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %88 = arith.index_cast %87 : i32 to index
      %89 = arith.index_cast %88 : index to i32
      cf.switch %89 : i32, [
        default: ^bb46,
        0: ^bb44,
        1: ^bb45
      ]
    ^bb44:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_7_cons_buff_1 : memref<32x64xbf16>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>)
    ^bb47(%90: memref<32x64xbf16>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_vectorized_different_datatypes(%90, %59, %C_L1L2_7_3_buff_0) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
      aie.use_lock(%A_L2L1_3_7_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous31[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous31[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_3_cons_prod_lock_0, Release, 1)
      %96 = memref.load %_anonymous31[%c1] : memref<3xi32>
      %97 = arith.addi %96, %c1_i32 : i32
      %98 = arith.cmpi sge, %97, %c2_i32 : i32
      %99 = arith.subi %97, %c2_i32 : i32
      %100 = arith.select %98, %99, %97 : i32
      memref.store %100, %_anonymous31[%c1] : memref<3xi32>
      %101 = arith.addi %54, %c1 : index
      cf.br ^bb26(%101 : index)
    ^bb48:  // pred: ^bb26
      aie.use_lock(%C_L1L2_7_3_cons_lock_0, Release, 1)
      %102 = memref.load %_anonymous31[%c0] : memref<3xi32>
      %103 = arith.addi %102, %c1_i32 : i32
      %104 = arith.cmpi sge, %103, %c1_i32 : i32
      %105 = arith.select %104, %102, %103 : i32
      memref.store %105, %_anonymous31[%c0] : memref<3xi32>
      %106 = arith.addi %0, %c1 : index
      cf.br ^bb1(%106 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<1048576xbf16>, %arg1: memref<131072x!aiex.bfp<"v8bfp16ebs8">>, %arg2: memref<1048576xbf16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @A_L3L2_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 131072, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @A_L3L2_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 262144, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @A_L3L2_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 393216, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 16384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 32768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 49152, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 65536, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 81920, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 98304, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 114688, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 0, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 128, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 256, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 384, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 512, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 640, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 768, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 896, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 524288, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @A_L3L2_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 655360, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @A_L3L2_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 786432, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @A_L3L2_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 917504, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 16384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 32768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 49152, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 65536, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 81920, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 98304, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<131072x!aiex.bfp<"v8bfp16ebs8">>, 114688, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524288, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524416, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524544, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524672, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524800, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524928, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 525056, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 525184, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%39)
    }
    aie.shim_dma_allocation @A_L3L2_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 4, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 5, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 0, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 1, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 2, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 0, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_0_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_0_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_1_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_1_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_2_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_2_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_3_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_3_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_4_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_4_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_5_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_5_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_6_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_6_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_7_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_7_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_0_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_0_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_1_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 4, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 5, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 0, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 1, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_2_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_2_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 2, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_2_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_2_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_2_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_2_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 0, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_2_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_2_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_2_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_2_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_2_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_2_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_0_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_0_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_1_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_1_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_2_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_2_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_3_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_3_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_4_3 = aie.mem(%tile_4_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_4_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_4_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_4_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_4_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_4_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_4_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_4_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_4_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_5_3 = aie.mem(%tile_5_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_5_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_5_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_5_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_5_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_5_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_5_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_5_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_5_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_6_3 = aie.mem(%tile_6_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_6_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_6_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_6_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_6_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_6_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_6_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_6_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_6_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_7_3 = aie.mem(%tile_7_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_7_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_7_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_7_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_7_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_7_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_7_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_7_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_7_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_2_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(S2MM, 4, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 5, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 0, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_4_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_4_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 1, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_4_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_4_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 2, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_4_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_4_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_4_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_4_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_4_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 0, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_4_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_4_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_4_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_4_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_4_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_4_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_4_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_4_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_0_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_0_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_1_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_1_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_2_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_2_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_3_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_3_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_4_4 = aie.mem(%tile_4_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_4_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_4_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_4_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_4_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_4_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_4_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_4_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_4_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_5_4 = aie.mem(%tile_5_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_5_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_5_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_5_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_5_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_5_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_5_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_5_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_5_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_6_4 = aie.mem(%tile_6_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_6_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_6_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_6_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_6_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_6_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_6_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_6_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_6_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_7_4 = aie.mem(%tile_7_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_7_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_7_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_7_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_2_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_7_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_7_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_2_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_7_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_7_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_7_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_3_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(S2MM, 4, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 5, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 0, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_6_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_6_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 1, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_6_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_6_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 2, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_6_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_6_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_6_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_6_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_6_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 0, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_6_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_6_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_6_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_6_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_6_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_6_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_6_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_6_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_0_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_0_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_1_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_1_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_2_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_2_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_3_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_3_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_4_5 = aie.mem(%tile_4_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_4_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_4_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_4_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_4_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_4_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_4_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_4_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_4_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_5_5 = aie.mem(%tile_5_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_5_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_5_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_5_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_5_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_5_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_5_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_5_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_5_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_6_5 = aie.mem(%tile_6_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_6_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_6_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_6_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_6_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_6_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_6_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_6_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_6_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_7_5 = aie.mem(%tile_7_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_7_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_7_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_7_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_7_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_7_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_7_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_7_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_7_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L2_1_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L2L3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L2L3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%C_L2L3_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%C_L2L3_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 3, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_1_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_2, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_1_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_2, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 4, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_1_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_3, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_1_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_3, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 1, ^bb19, ^bb27)
    ^bb19:  // 2 preds: ^bb18, ^bb26
      aie.use_lock(%C_L2L3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L2L3_1_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%C_L2L3_1_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      aie.use_lock(%C_L2L3_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%C_L2L3_1_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_2, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_1_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 35 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb27:  // pred: ^bb18
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_2_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L2_3_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L2L3_3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L2L3_3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%C_L2L3_3_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%C_L2L3_3_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 3, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_3_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_2, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_3_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_2, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 4, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_3_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_3, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_3_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_3, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 1, ^bb19, ^bb27)
    ^bb19:  // 2 preds: ^bb18, ^bb26
      aie.use_lock(%C_L2L3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_3_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_1, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L2L3_3_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%C_L2L3_3_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      aie.use_lock(%C_L2L3_3_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%C_L2L3_3_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_2, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_3_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 35 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb27:  // pred: ^bb18
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_4_shim_alloc(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L2_5_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L2L3_5_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L2L3_5_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%C_L2L3_5_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%C_L2L3_5_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 3, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_5_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_2, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_5_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_2, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 4, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_5_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_3, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_5_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_5_cons_lock_3, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 1, ^bb19, ^bb27)
    ^bb19:  // 2 preds: ^bb18, ^bb26
      aie.use_lock(%C_L2L3_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_5_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_1, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L2L3_5_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%C_L2L3_5_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      aie.use_lock(%C_L2L3_5_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%C_L2L3_5_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_2, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_5_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_5_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 35 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_5_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb27:  // pred: ^bb18
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_6_shim_alloc(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L2_7_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_L2L3_7_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_L2L3_7_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%C_L2L3_7_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%C_L2L3_7_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 3, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_7_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_2, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_7_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_2, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 4, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_7_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_3, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_7_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_7_cons_lock_3, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 1, ^bb19, ^bb27)
    ^bb19:  // 2 preds: ^bb18, ^bb26
      aie.use_lock(%C_L2L3_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_7_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_1, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%C_L2L3_7_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%C_L2L3_7_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      aie.use_lock(%C_L2L3_7_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%C_L2L3_7_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_2, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_7_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_7_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 35 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_7_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb27:  // pred: ^bb18
      aie.end
    }
    aie.shim_dma_allocation @C_L2L3_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_1_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_2_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_3_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_4_shim_alloc(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_5_shim_alloc(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_6_shim_alloc(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_7_shim_alloc(%shim_noc_tile_7_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_4_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_4_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_5_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_5_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_6_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_6_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_7_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_7_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
