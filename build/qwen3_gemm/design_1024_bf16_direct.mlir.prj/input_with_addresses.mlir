module {
  aie.device(npu2) {
    func.func private @zero_kernel_bf16(memref<128x128xbf16>) attributes {link_with = "mm_bf16_direct.o"}
    func.func private @matmul_vectorized_bf16(memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) attributes {link_with = "mm_bf16_direct.o"}
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
    %CL3_7_cons_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "CL3_7_cons_prod_lock_0"}
    %CL3_7_cons_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "CL3_7_cons_cons_lock_0"}
    %CL3_7_buff_0 = aie.buffer(%mem_tile_7_1) {address = 0 : i32, sym_name = "CL3_7_buff_0"} : memref<512x128xbf16> 
    %CL3_7_buff_1 = aie.buffer(%mem_tile_7_1) {address = 131072 : i32, sym_name = "CL3_7_buff_1"} : memref<512x128xbf16> 
    %CL3_7_prod_lock_0 = aie.lock(%mem_tile_7_1, 2) {init = 2 : i32, sym_name = "CL3_7_prod_lock_0"}
    %CL3_7_cons_lock_0 = aie.lock(%mem_tile_7_1, 3) {init = 0 : i32, sym_name = "CL3_7_cons_lock_0"}
    %CL3_7_prod_lock_1 = aie.lock(%mem_tile_7_1, 4) {init = 2 : i32, sym_name = "CL3_7_prod_lock_1"}
    %CL3_7_cons_lock_1 = aie.lock(%mem_tile_7_1, 5) {init = 0 : i32, sym_name = "CL3_7_cons_lock_1"}
    %CL3_7_prod_lock_2 = aie.lock(%mem_tile_7_1, 6) {init = 2 : i32, sym_name = "CL3_7_prod_lock_2"}
    %CL3_7_cons_lock_2 = aie.lock(%mem_tile_7_1, 7) {init = 0 : i32, sym_name = "CL3_7_cons_lock_2"}
    %CL3_7_prod_lock_3 = aie.lock(%mem_tile_7_1, 8) {init = 2 : i32, sym_name = "CL3_7_prod_lock_3"}
    %CL3_7_cons_lock_3 = aie.lock(%mem_tile_7_1, 9) {init = 0 : i32, sym_name = "CL3_7_cons_lock_3"}
    %CL1_7_3_buff_0 = aie.buffer(%tile_7_5) {address = 2816 : i32, sym_name = "CL1_7_3_buff_0"} : memref<128x128xbf16> 
    %CL1_7_3_prod_lock_0 = aie.lock(%tile_7_5, 4) {init = 1 : i32, sym_name = "CL1_7_3_prod_lock_0"}
    %CL1_7_3_cons_lock_0 = aie.lock(%tile_7_5, 5) {init = 0 : i32, sym_name = "CL1_7_3_cons_lock_0"}
    %CL1_7_2_buff_0 = aie.buffer(%tile_7_4) {address = 2816 : i32, sym_name = "CL1_7_2_buff_0"} : memref<128x128xbf16> 
    %CL1_7_2_prod_lock_0 = aie.lock(%tile_7_4, 4) {init = 1 : i32, sym_name = "CL1_7_2_prod_lock_0"}
    %CL1_7_2_cons_lock_0 = aie.lock(%tile_7_4, 5) {init = 0 : i32, sym_name = "CL1_7_2_cons_lock_0"}
    %CL1_7_1_buff_0 = aie.buffer(%tile_7_3) {address = 2816 : i32, sym_name = "CL1_7_1_buff_0"} : memref<128x128xbf16> 
    %CL1_7_1_prod_lock_0 = aie.lock(%tile_7_3, 4) {init = 1 : i32, sym_name = "CL1_7_1_prod_lock_0"}
    %CL1_7_1_cons_lock_0 = aie.lock(%tile_7_3, 5) {init = 0 : i32, sym_name = "CL1_7_1_cons_lock_0"}
    %CL1_7_0_buff_0 = aie.buffer(%tile_7_2) {address = 2816 : i32, sym_name = "CL1_7_0_buff_0"} : memref<128x128xbf16> 
    %CL1_7_0_prod_lock_0 = aie.lock(%tile_7_2, 4) {init = 1 : i32, sym_name = "CL1_7_0_prod_lock_0"}
    %CL1_7_0_cons_lock_0 = aie.lock(%tile_7_2, 5) {init = 0 : i32, sym_name = "CL1_7_0_cons_lock_0"}
    %CL3_6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 4) {init = 0 : i32, sym_name = "CL3_6_cons_prod_lock_0"}
    %CL3_6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 5) {init = 0 : i32, sym_name = "CL3_6_cons_cons_lock_0"}
    %CL3_6_buff_0 = aie.buffer(%mem_tile_6_1) {address = 0 : i32, sym_name = "CL3_6_buff_0"} : memref<512x128xbf16> 
    %CL3_6_buff_1 = aie.buffer(%mem_tile_7_1) {address = 262144 : i32, sym_name = "CL3_6_buff_1"} : memref<512x128xbf16> 
    %CL3_6_prod_lock_0 = aie.lock(%mem_tile_6_1, 4) {init = 2 : i32, sym_name = "CL3_6_prod_lock_0"}
    %CL3_6_cons_lock_0 = aie.lock(%mem_tile_6_1, 5) {init = 0 : i32, sym_name = "CL3_6_cons_lock_0"}
    %CL3_6_prod_lock_1 = aie.lock(%mem_tile_6_1, 6) {init = 2 : i32, sym_name = "CL3_6_prod_lock_1"}
    %CL3_6_cons_lock_1 = aie.lock(%mem_tile_6_1, 7) {init = 0 : i32, sym_name = "CL3_6_cons_lock_1"}
    %CL3_6_prod_lock_2 = aie.lock(%mem_tile_6_1, 8) {init = 2 : i32, sym_name = "CL3_6_prod_lock_2"}
    %CL3_6_cons_lock_2 = aie.lock(%mem_tile_6_1, 9) {init = 0 : i32, sym_name = "CL3_6_cons_lock_2"}
    %CL3_6_prod_lock_3 = aie.lock(%mem_tile_6_1, 10) {init = 2 : i32, sym_name = "CL3_6_prod_lock_3"}
    %CL3_6_cons_lock_3 = aie.lock(%mem_tile_6_1, 11) {init = 0 : i32, sym_name = "CL3_6_cons_lock_3"}
    %CL1_6_3_buff_0 = aie.buffer(%tile_6_5) {address = 2816 : i32, sym_name = "CL1_6_3_buff_0"} : memref<128x128xbf16> 
    %CL1_6_3_prod_lock_0 = aie.lock(%tile_6_5, 4) {init = 1 : i32, sym_name = "CL1_6_3_prod_lock_0"}
    %CL1_6_3_cons_lock_0 = aie.lock(%tile_6_5, 5) {init = 0 : i32, sym_name = "CL1_6_3_cons_lock_0"}
    %CL1_6_2_buff_0 = aie.buffer(%tile_6_4) {address = 2816 : i32, sym_name = "CL1_6_2_buff_0"} : memref<128x128xbf16> 
    %CL1_6_2_prod_lock_0 = aie.lock(%tile_6_4, 4) {init = 1 : i32, sym_name = "CL1_6_2_prod_lock_0"}
    %CL1_6_2_cons_lock_0 = aie.lock(%tile_6_4, 5) {init = 0 : i32, sym_name = "CL1_6_2_cons_lock_0"}
    %CL1_6_1_buff_0 = aie.buffer(%tile_6_3) {address = 2816 : i32, sym_name = "CL1_6_1_buff_0"} : memref<128x128xbf16> 
    %CL1_6_1_prod_lock_0 = aie.lock(%tile_6_3, 4) {init = 1 : i32, sym_name = "CL1_6_1_prod_lock_0"}
    %CL1_6_1_cons_lock_0 = aie.lock(%tile_6_3, 5) {init = 0 : i32, sym_name = "CL1_6_1_cons_lock_0"}
    %CL1_6_0_buff_0 = aie.buffer(%tile_6_2) {address = 2816 : i32, sym_name = "CL1_6_0_buff_0"} : memref<128x128xbf16> 
    %CL1_6_0_prod_lock_0 = aie.lock(%tile_6_2, 4) {init = 1 : i32, sym_name = "CL1_6_0_prod_lock_0"}
    %CL1_6_0_cons_lock_0 = aie.lock(%tile_6_2, 5) {init = 0 : i32, sym_name = "CL1_6_0_cons_lock_0"}
    %CL3_5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "CL3_5_cons_prod_lock_0"}
    %CL3_5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "CL3_5_cons_cons_lock_0"}
    %CL3_5_buff_0 = aie.buffer(%mem_tile_5_1) {address = 0 : i32, sym_name = "CL3_5_buff_0"} : memref<512x128xbf16> 
    %CL3_5_buff_1 = aie.buffer(%mem_tile_5_1) {address = 131072 : i32, sym_name = "CL3_5_buff_1"} : memref<512x128xbf16> 
    %CL3_5_prod_lock_0 = aie.lock(%mem_tile_5_1, 2) {init = 2 : i32, sym_name = "CL3_5_prod_lock_0"}
    %CL3_5_cons_lock_0 = aie.lock(%mem_tile_5_1, 3) {init = 0 : i32, sym_name = "CL3_5_cons_lock_0"}
    %CL3_5_prod_lock_1 = aie.lock(%mem_tile_5_1, 4) {init = 2 : i32, sym_name = "CL3_5_prod_lock_1"}
    %CL3_5_cons_lock_1 = aie.lock(%mem_tile_5_1, 5) {init = 0 : i32, sym_name = "CL3_5_cons_lock_1"}
    %CL3_5_prod_lock_2 = aie.lock(%mem_tile_5_1, 6) {init = 2 : i32, sym_name = "CL3_5_prod_lock_2"}
    %CL3_5_cons_lock_2 = aie.lock(%mem_tile_5_1, 7) {init = 0 : i32, sym_name = "CL3_5_cons_lock_2"}
    %CL3_5_prod_lock_3 = aie.lock(%mem_tile_5_1, 8) {init = 2 : i32, sym_name = "CL3_5_prod_lock_3"}
    %CL3_5_cons_lock_3 = aie.lock(%mem_tile_5_1, 9) {init = 0 : i32, sym_name = "CL3_5_cons_lock_3"}
    %CL1_5_3_buff_0 = aie.buffer(%tile_5_5) {address = 2816 : i32, sym_name = "CL1_5_3_buff_0"} : memref<128x128xbf16> 
    %CL1_5_3_prod_lock_0 = aie.lock(%tile_5_5, 4) {init = 1 : i32, sym_name = "CL1_5_3_prod_lock_0"}
    %CL1_5_3_cons_lock_0 = aie.lock(%tile_5_5, 5) {init = 0 : i32, sym_name = "CL1_5_3_cons_lock_0"}
    %CL1_5_2_buff_0 = aie.buffer(%tile_5_4) {address = 2816 : i32, sym_name = "CL1_5_2_buff_0"} : memref<128x128xbf16> 
    %CL1_5_2_prod_lock_0 = aie.lock(%tile_5_4, 4) {init = 1 : i32, sym_name = "CL1_5_2_prod_lock_0"}
    %CL1_5_2_cons_lock_0 = aie.lock(%tile_5_4, 5) {init = 0 : i32, sym_name = "CL1_5_2_cons_lock_0"}
    %CL1_5_1_buff_0 = aie.buffer(%tile_5_3) {address = 2816 : i32, sym_name = "CL1_5_1_buff_0"} : memref<128x128xbf16> 
    %CL1_5_1_prod_lock_0 = aie.lock(%tile_5_3, 4) {init = 1 : i32, sym_name = "CL1_5_1_prod_lock_0"}
    %CL1_5_1_cons_lock_0 = aie.lock(%tile_5_3, 5) {init = 0 : i32, sym_name = "CL1_5_1_cons_lock_0"}
    %CL1_5_0_buff_0 = aie.buffer(%tile_5_2) {address = 2816 : i32, sym_name = "CL1_5_0_buff_0"} : memref<128x128xbf16> 
    %CL1_5_0_prod_lock_0 = aie.lock(%tile_5_2, 4) {init = 1 : i32, sym_name = "CL1_5_0_prod_lock_0"}
    %CL1_5_0_cons_lock_0 = aie.lock(%tile_5_2, 5) {init = 0 : i32, sym_name = "CL1_5_0_cons_lock_0"}
    %CL3_4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 4) {init = 0 : i32, sym_name = "CL3_4_cons_prod_lock_0"}
    %CL3_4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 5) {init = 0 : i32, sym_name = "CL3_4_cons_cons_lock_0"}
    %CL3_4_buff_0 = aie.buffer(%mem_tile_4_1) {address = 0 : i32, sym_name = "CL3_4_buff_0"} : memref<512x128xbf16> 
    %CL3_4_buff_1 = aie.buffer(%mem_tile_5_1) {address = 262144 : i32, sym_name = "CL3_4_buff_1"} : memref<512x128xbf16> 
    %CL3_4_prod_lock_0 = aie.lock(%mem_tile_4_1, 4) {init = 2 : i32, sym_name = "CL3_4_prod_lock_0"}
    %CL3_4_cons_lock_0 = aie.lock(%mem_tile_4_1, 5) {init = 0 : i32, sym_name = "CL3_4_cons_lock_0"}
    %CL3_4_prod_lock_1 = aie.lock(%mem_tile_4_1, 6) {init = 2 : i32, sym_name = "CL3_4_prod_lock_1"}
    %CL3_4_cons_lock_1 = aie.lock(%mem_tile_4_1, 7) {init = 0 : i32, sym_name = "CL3_4_cons_lock_1"}
    %CL3_4_prod_lock_2 = aie.lock(%mem_tile_4_1, 8) {init = 2 : i32, sym_name = "CL3_4_prod_lock_2"}
    %CL3_4_cons_lock_2 = aie.lock(%mem_tile_4_1, 9) {init = 0 : i32, sym_name = "CL3_4_cons_lock_2"}
    %CL3_4_prod_lock_3 = aie.lock(%mem_tile_4_1, 10) {init = 2 : i32, sym_name = "CL3_4_prod_lock_3"}
    %CL3_4_cons_lock_3 = aie.lock(%mem_tile_4_1, 11) {init = 0 : i32, sym_name = "CL3_4_cons_lock_3"}
    %CL1_4_3_buff_0 = aie.buffer(%tile_4_5) {address = 2816 : i32, sym_name = "CL1_4_3_buff_0"} : memref<128x128xbf16> 
    %CL1_4_3_prod_lock_0 = aie.lock(%tile_4_5, 4) {init = 1 : i32, sym_name = "CL1_4_3_prod_lock_0"}
    %CL1_4_3_cons_lock_0 = aie.lock(%tile_4_5, 5) {init = 0 : i32, sym_name = "CL1_4_3_cons_lock_0"}
    %CL1_4_2_buff_0 = aie.buffer(%tile_4_4) {address = 2816 : i32, sym_name = "CL1_4_2_buff_0"} : memref<128x128xbf16> 
    %CL1_4_2_prod_lock_0 = aie.lock(%tile_4_4, 4) {init = 1 : i32, sym_name = "CL1_4_2_prod_lock_0"}
    %CL1_4_2_cons_lock_0 = aie.lock(%tile_4_4, 5) {init = 0 : i32, sym_name = "CL1_4_2_cons_lock_0"}
    %CL1_4_1_buff_0 = aie.buffer(%tile_4_3) {address = 2816 : i32, sym_name = "CL1_4_1_buff_0"} : memref<128x128xbf16> 
    %CL1_4_1_prod_lock_0 = aie.lock(%tile_4_3, 4) {init = 1 : i32, sym_name = "CL1_4_1_prod_lock_0"}
    %CL1_4_1_cons_lock_0 = aie.lock(%tile_4_3, 5) {init = 0 : i32, sym_name = "CL1_4_1_cons_lock_0"}
    %CL1_4_0_buff_0 = aie.buffer(%tile_4_2) {address = 2816 : i32, sym_name = "CL1_4_0_buff_0"} : memref<128x128xbf16> 
    %CL1_4_0_prod_lock_0 = aie.lock(%tile_4_2, 4) {init = 1 : i32, sym_name = "CL1_4_0_prod_lock_0"}
    %CL1_4_0_cons_lock_0 = aie.lock(%tile_4_2, 5) {init = 0 : i32, sym_name = "CL1_4_0_cons_lock_0"}
    %CL3_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "CL3_3_cons_prod_lock_0"}
    %CL3_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "CL3_3_cons_cons_lock_0"}
    %CL3_3_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, sym_name = "CL3_3_buff_0"} : memref<512x128xbf16> 
    %CL3_3_buff_1 = aie.buffer(%mem_tile_3_1) {address = 131072 : i32, sym_name = "CL3_3_buff_1"} : memref<512x128xbf16> 
    %CL3_3_prod_lock_0 = aie.lock(%mem_tile_3_1, 2) {init = 2 : i32, sym_name = "CL3_3_prod_lock_0"}
    %CL3_3_cons_lock_0 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "CL3_3_cons_lock_0"}
    %CL3_3_prod_lock_1 = aie.lock(%mem_tile_3_1, 4) {init = 2 : i32, sym_name = "CL3_3_prod_lock_1"}
    %CL3_3_cons_lock_1 = aie.lock(%mem_tile_3_1, 5) {init = 0 : i32, sym_name = "CL3_3_cons_lock_1"}
    %CL3_3_prod_lock_2 = aie.lock(%mem_tile_3_1, 6) {init = 2 : i32, sym_name = "CL3_3_prod_lock_2"}
    %CL3_3_cons_lock_2 = aie.lock(%mem_tile_3_1, 7) {init = 0 : i32, sym_name = "CL3_3_cons_lock_2"}
    %CL3_3_prod_lock_3 = aie.lock(%mem_tile_3_1, 8) {init = 2 : i32, sym_name = "CL3_3_prod_lock_3"}
    %CL3_3_cons_lock_3 = aie.lock(%mem_tile_3_1, 9) {init = 0 : i32, sym_name = "CL3_3_cons_lock_3"}
    %CL1_3_3_buff_0 = aie.buffer(%tile_3_5) {address = 2816 : i32, sym_name = "CL1_3_3_buff_0"} : memref<128x128xbf16> 
    %CL1_3_3_prod_lock_0 = aie.lock(%tile_3_5, 4) {init = 1 : i32, sym_name = "CL1_3_3_prod_lock_0"}
    %CL1_3_3_cons_lock_0 = aie.lock(%tile_3_5, 5) {init = 0 : i32, sym_name = "CL1_3_3_cons_lock_0"}
    %CL1_3_2_buff_0 = aie.buffer(%tile_3_4) {address = 2816 : i32, sym_name = "CL1_3_2_buff_0"} : memref<128x128xbf16> 
    %CL1_3_2_prod_lock_0 = aie.lock(%tile_3_4, 4) {init = 1 : i32, sym_name = "CL1_3_2_prod_lock_0"}
    %CL1_3_2_cons_lock_0 = aie.lock(%tile_3_4, 5) {init = 0 : i32, sym_name = "CL1_3_2_cons_lock_0"}
    %CL1_3_1_buff_0 = aie.buffer(%tile_3_3) {address = 2816 : i32, sym_name = "CL1_3_1_buff_0"} : memref<128x128xbf16> 
    %CL1_3_1_prod_lock_0 = aie.lock(%tile_3_3, 4) {init = 1 : i32, sym_name = "CL1_3_1_prod_lock_0"}
    %CL1_3_1_cons_lock_0 = aie.lock(%tile_3_3, 5) {init = 0 : i32, sym_name = "CL1_3_1_cons_lock_0"}
    %CL1_3_0_buff_0 = aie.buffer(%tile_3_2) {address = 2816 : i32, sym_name = "CL1_3_0_buff_0"} : memref<128x128xbf16> 
    %CL1_3_0_prod_lock_0 = aie.lock(%tile_3_2, 4) {init = 1 : i32, sym_name = "CL1_3_0_prod_lock_0"}
    %CL1_3_0_cons_lock_0 = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "CL1_3_0_cons_lock_0"}
    %CL3_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "CL3_2_cons_prod_lock_0"}
    %CL3_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "CL3_2_cons_cons_lock_0"}
    %CL3_2_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "CL3_2_buff_0"} : memref<512x128xbf16> 
    %CL3_2_buff_1 = aie.buffer(%mem_tile_3_1) {address = 262144 : i32, sym_name = "CL3_2_buff_1"} : memref<512x128xbf16> 
    %CL3_2_prod_lock_0 = aie.lock(%mem_tile_2_1, 4) {init = 2 : i32, sym_name = "CL3_2_prod_lock_0"}
    %CL3_2_cons_lock_0 = aie.lock(%mem_tile_2_1, 5) {init = 0 : i32, sym_name = "CL3_2_cons_lock_0"}
    %CL3_2_prod_lock_1 = aie.lock(%mem_tile_2_1, 6) {init = 2 : i32, sym_name = "CL3_2_prod_lock_1"}
    %CL3_2_cons_lock_1 = aie.lock(%mem_tile_2_1, 7) {init = 0 : i32, sym_name = "CL3_2_cons_lock_1"}
    %CL3_2_prod_lock_2 = aie.lock(%mem_tile_2_1, 8) {init = 2 : i32, sym_name = "CL3_2_prod_lock_2"}
    %CL3_2_cons_lock_2 = aie.lock(%mem_tile_2_1, 9) {init = 0 : i32, sym_name = "CL3_2_cons_lock_2"}
    %CL3_2_prod_lock_3 = aie.lock(%mem_tile_2_1, 10) {init = 2 : i32, sym_name = "CL3_2_prod_lock_3"}
    %CL3_2_cons_lock_3 = aie.lock(%mem_tile_2_1, 11) {init = 0 : i32, sym_name = "CL3_2_cons_lock_3"}
    %CL1_2_3_buff_0 = aie.buffer(%tile_2_5) {address = 2816 : i32, sym_name = "CL1_2_3_buff_0"} : memref<128x128xbf16> 
    %CL1_2_3_prod_lock_0 = aie.lock(%tile_2_5, 4) {init = 1 : i32, sym_name = "CL1_2_3_prod_lock_0"}
    %CL1_2_3_cons_lock_0 = aie.lock(%tile_2_5, 5) {init = 0 : i32, sym_name = "CL1_2_3_cons_lock_0"}
    %CL1_2_2_buff_0 = aie.buffer(%tile_2_4) {address = 2816 : i32, sym_name = "CL1_2_2_buff_0"} : memref<128x128xbf16> 
    %CL1_2_2_prod_lock_0 = aie.lock(%tile_2_4, 4) {init = 1 : i32, sym_name = "CL1_2_2_prod_lock_0"}
    %CL1_2_2_cons_lock_0 = aie.lock(%tile_2_4, 5) {init = 0 : i32, sym_name = "CL1_2_2_cons_lock_0"}
    %CL1_2_1_buff_0 = aie.buffer(%tile_2_3) {address = 2816 : i32, sym_name = "CL1_2_1_buff_0"} : memref<128x128xbf16> 
    %CL1_2_1_prod_lock_0 = aie.lock(%tile_2_3, 4) {init = 1 : i32, sym_name = "CL1_2_1_prod_lock_0"}
    %CL1_2_1_cons_lock_0 = aie.lock(%tile_2_3, 5) {init = 0 : i32, sym_name = "CL1_2_1_cons_lock_0"}
    %CL1_2_0_buff_0 = aie.buffer(%tile_2_2) {address = 2816 : i32, sym_name = "CL1_2_0_buff_0"} : memref<128x128xbf16> 
    %CL1_2_0_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 1 : i32, sym_name = "CL1_2_0_prod_lock_0"}
    %CL1_2_0_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "CL1_2_0_cons_lock_0"}
    %CL3_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "CL3_1_cons_prod_lock_0"}
    %CL3_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "CL3_1_cons_cons_lock_0"}
    %CL3_1_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "CL3_1_buff_0"} : memref<512x128xbf16> 
    %CL3_1_buff_1 = aie.buffer(%mem_tile_1_1) {address = 131072 : i32, sym_name = "CL3_1_buff_1"} : memref<512x128xbf16> 
    %CL3_1_prod_lock_0 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "CL3_1_prod_lock_0"}
    %CL3_1_cons_lock_0 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "CL3_1_cons_lock_0"}
    %CL3_1_prod_lock_1 = aie.lock(%mem_tile_1_1, 4) {init = 2 : i32, sym_name = "CL3_1_prod_lock_1"}
    %CL3_1_cons_lock_1 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "CL3_1_cons_lock_1"}
    %CL3_1_prod_lock_2 = aie.lock(%mem_tile_1_1, 6) {init = 2 : i32, sym_name = "CL3_1_prod_lock_2"}
    %CL3_1_cons_lock_2 = aie.lock(%mem_tile_1_1, 7) {init = 0 : i32, sym_name = "CL3_1_cons_lock_2"}
    %CL3_1_prod_lock_3 = aie.lock(%mem_tile_1_1, 8) {init = 2 : i32, sym_name = "CL3_1_prod_lock_3"}
    %CL3_1_cons_lock_3 = aie.lock(%mem_tile_1_1, 9) {init = 0 : i32, sym_name = "CL3_1_cons_lock_3"}
    %CL1_1_3_buff_0 = aie.buffer(%tile_1_5) {address = 2816 : i32, sym_name = "CL1_1_3_buff_0"} : memref<128x128xbf16> 
    %CL1_1_3_prod_lock_0 = aie.lock(%tile_1_5, 4) {init = 1 : i32, sym_name = "CL1_1_3_prod_lock_0"}
    %CL1_1_3_cons_lock_0 = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "CL1_1_3_cons_lock_0"}
    %CL1_1_2_buff_0 = aie.buffer(%tile_1_4) {address = 2816 : i32, sym_name = "CL1_1_2_buff_0"} : memref<128x128xbf16> 
    %CL1_1_2_prod_lock_0 = aie.lock(%tile_1_4, 4) {init = 1 : i32, sym_name = "CL1_1_2_prod_lock_0"}
    %CL1_1_2_cons_lock_0 = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "CL1_1_2_cons_lock_0"}
    %CL1_1_1_buff_0 = aie.buffer(%tile_1_3) {address = 2816 : i32, sym_name = "CL1_1_1_buff_0"} : memref<128x128xbf16> 
    %CL1_1_1_prod_lock_0 = aie.lock(%tile_1_3, 4) {init = 1 : i32, sym_name = "CL1_1_1_prod_lock_0"}
    %CL1_1_1_cons_lock_0 = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "CL1_1_1_cons_lock_0"}
    %CL1_1_0_buff_0 = aie.buffer(%tile_1_2) {address = 2816 : i32, sym_name = "CL1_1_0_buff_0"} : memref<128x128xbf16> 
    %CL1_1_0_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 1 : i32, sym_name = "CL1_1_0_prod_lock_0"}
    %CL1_1_0_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "CL1_1_0_cons_lock_0"}
    %CL3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "CL3_0_cons_prod_lock_0"}
    %CL3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "CL3_0_cons_cons_lock_0"}
    %CL3_0_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "CL3_0_buff_0"} : memref<512x128xbf16> 
    %CL3_0_buff_1 = aie.buffer(%mem_tile_1_1) {address = 262144 : i32, sym_name = "CL3_0_buff_1"} : memref<512x128xbf16> 
    %CL3_0_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "CL3_0_prod_lock_0"}
    %CL3_0_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "CL3_0_cons_lock_0"}
    %CL3_0_prod_lock_1 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "CL3_0_prod_lock_1"}
    %CL3_0_cons_lock_1 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "CL3_0_cons_lock_1"}
    %CL3_0_prod_lock_2 = aie.lock(%mem_tile_0_1, 8) {init = 2 : i32, sym_name = "CL3_0_prod_lock_2"}
    %CL3_0_cons_lock_2 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "CL3_0_cons_lock_2"}
    %CL3_0_prod_lock_3 = aie.lock(%mem_tile_0_1, 10) {init = 2 : i32, sym_name = "CL3_0_prod_lock_3"}
    %CL3_0_cons_lock_3 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32, sym_name = "CL3_0_cons_lock_3"}
    %CL1_0_3_buff_0 = aie.buffer(%tile_0_5) {address = 2816 : i32, sym_name = "CL1_0_3_buff_0"} : memref<128x128xbf16> 
    %CL1_0_3_prod_lock_0 = aie.lock(%tile_0_5, 4) {init = 1 : i32, sym_name = "CL1_0_3_prod_lock_0"}
    %CL1_0_3_cons_lock_0 = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "CL1_0_3_cons_lock_0"}
    %CL1_0_2_buff_0 = aie.buffer(%tile_0_4) {address = 2816 : i32, sym_name = "CL1_0_2_buff_0"} : memref<128x128xbf16> 
    %CL1_0_2_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 1 : i32, sym_name = "CL1_0_2_prod_lock_0"}
    %CL1_0_2_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "CL1_0_2_cons_lock_0"}
    %CL1_0_1_buff_0 = aie.buffer(%tile_0_3) {address = 2816 : i32, sym_name = "CL1_0_1_buff_0"} : memref<128x128xbf16> 
    %CL1_0_1_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 1 : i32, sym_name = "CL1_0_1_prod_lock_0"}
    %CL1_0_1_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "CL1_0_1_cons_lock_0"}
    %CL1_0_0_buff_0 = aie.buffer(%tile_0_2) {address = 2816 : i32, sym_name = "CL1_0_0_buff_0"} : memref<128x128xbf16> 
    %CL1_0_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "CL1_0_0_prod_lock_0"}
    %CL1_0_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "CL1_0_0_cons_lock_0"}
    %BL1_7_0_cons_buff_0 = aie.buffer(%tile_7_2) {address = 35584 : i32, sym_name = "BL1_7_0_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_7_0_cons_prod_lock_0 = aie.lock(%tile_7_2, 2) {init = 1 : i32, sym_name = "BL1_7_0_cons_prod_lock_0"}
    %BL1_7_0_cons_cons_lock_0 = aie.lock(%tile_7_2, 3) {init = 0 : i32, sym_name = "BL1_7_0_cons_cons_lock_0"}
    %BL1_7_1_cons_buff_0 = aie.buffer(%tile_7_3) {address = 35584 : i32, sym_name = "BL1_7_1_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_7_1_cons_prod_lock_0 = aie.lock(%tile_7_3, 2) {init = 1 : i32, sym_name = "BL1_7_1_cons_prod_lock_0"}
    %BL1_7_1_cons_cons_lock_0 = aie.lock(%tile_7_3, 3) {init = 0 : i32, sym_name = "BL1_7_1_cons_cons_lock_0"}
    %BL1_7_2_cons_buff_0 = aie.buffer(%tile_7_4) {address = 35584 : i32, sym_name = "BL1_7_2_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_7_2_cons_prod_lock_0 = aie.lock(%tile_7_4, 2) {init = 1 : i32, sym_name = "BL1_7_2_cons_prod_lock_0"}
    %BL1_7_2_cons_cons_lock_0 = aie.lock(%tile_7_4, 3) {init = 0 : i32, sym_name = "BL1_7_2_cons_cons_lock_0"}
    %BL1_7_3_cons_buff_0 = aie.buffer(%tile_7_5) {address = 35584 : i32, sym_name = "BL1_7_3_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_7_3_cons_prod_lock_0 = aie.lock(%tile_7_5, 2) {init = 1 : i32, sym_name = "BL1_7_3_cons_prod_lock_0"}
    %BL1_7_3_cons_cons_lock_0 = aie.lock(%tile_7_5, 3) {init = 0 : i32, sym_name = "BL1_7_3_cons_cons_lock_0"}
    %BL3_7_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 393216 : i32, sym_name = "BL3_7_cons_buff_0"} : memref<64x128xbf16> 
    %BL3_7_cons_buff_1 = aie.buffer(%mem_tile_7_1) {address = 409600 : i32, sym_name = "BL3_7_cons_buff_1"} : memref<64x128xbf16> 
    %BL3_7_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 0) {init = 2 : i32, sym_name = "BL3_7_cons_prod_lock_0"}
    %BL3_7_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 1) {init = 0 : i32, sym_name = "BL3_7_cons_cons_lock_0"}
    %BL3_7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "BL3_7_prod_lock_0"}
    %BL3_7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "BL3_7_cons_lock_0"}
    %BL1_6_0_cons_buff_0 = aie.buffer(%tile_6_2) {address = 35584 : i32, sym_name = "BL1_6_0_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_6_0_cons_prod_lock_0 = aie.lock(%tile_6_2, 2) {init = 1 : i32, sym_name = "BL1_6_0_cons_prod_lock_0"}
    %BL1_6_0_cons_cons_lock_0 = aie.lock(%tile_6_2, 3) {init = 0 : i32, sym_name = "BL1_6_0_cons_cons_lock_0"}
    %BL1_6_1_cons_buff_0 = aie.buffer(%tile_6_3) {address = 35584 : i32, sym_name = "BL1_6_1_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_6_1_cons_prod_lock_0 = aie.lock(%tile_6_3, 2) {init = 1 : i32, sym_name = "BL1_6_1_cons_prod_lock_0"}
    %BL1_6_1_cons_cons_lock_0 = aie.lock(%tile_6_3, 3) {init = 0 : i32, sym_name = "BL1_6_1_cons_cons_lock_0"}
    %BL1_6_2_cons_buff_0 = aie.buffer(%tile_6_4) {address = 35584 : i32, sym_name = "BL1_6_2_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_6_2_cons_prod_lock_0 = aie.lock(%tile_6_4, 2) {init = 1 : i32, sym_name = "BL1_6_2_cons_prod_lock_0"}
    %BL1_6_2_cons_cons_lock_0 = aie.lock(%tile_6_4, 3) {init = 0 : i32, sym_name = "BL1_6_2_cons_cons_lock_0"}
    %BL1_6_3_cons_buff_0 = aie.buffer(%tile_6_5) {address = 35584 : i32, sym_name = "BL1_6_3_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_6_3_cons_prod_lock_0 = aie.lock(%tile_6_5, 2) {init = 1 : i32, sym_name = "BL1_6_3_cons_prod_lock_0"}
    %BL1_6_3_cons_cons_lock_0 = aie.lock(%tile_6_5, 3) {init = 0 : i32, sym_name = "BL1_6_3_cons_cons_lock_0"}
    %BL3_6_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 393216 : i32, sym_name = "BL3_6_cons_buff_0"} : memref<64x128xbf16> 
    %BL3_6_cons_buff_1 = aie.buffer(%mem_tile_6_1) {address = 409600 : i32, sym_name = "BL3_6_cons_buff_1"} : memref<64x128xbf16> 
    %BL3_6_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 2) {init = 2 : i32, sym_name = "BL3_6_cons_prod_lock_0"}
    %BL3_6_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 3) {init = 0 : i32, sym_name = "BL3_6_cons_cons_lock_0"}
    %BL3_6_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "BL3_6_prod_lock_0"}
    %BL3_6_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "BL3_6_cons_lock_0"}
    %BL1_5_0_cons_buff_0 = aie.buffer(%tile_5_2) {address = 35584 : i32, sym_name = "BL1_5_0_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_5_0_cons_prod_lock_0 = aie.lock(%tile_5_2, 2) {init = 1 : i32, sym_name = "BL1_5_0_cons_prod_lock_0"}
    %BL1_5_0_cons_cons_lock_0 = aie.lock(%tile_5_2, 3) {init = 0 : i32, sym_name = "BL1_5_0_cons_cons_lock_0"}
    %BL1_5_1_cons_buff_0 = aie.buffer(%tile_5_3) {address = 35584 : i32, sym_name = "BL1_5_1_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_5_1_cons_prod_lock_0 = aie.lock(%tile_5_3, 2) {init = 1 : i32, sym_name = "BL1_5_1_cons_prod_lock_0"}
    %BL1_5_1_cons_cons_lock_0 = aie.lock(%tile_5_3, 3) {init = 0 : i32, sym_name = "BL1_5_1_cons_cons_lock_0"}
    %BL1_5_2_cons_buff_0 = aie.buffer(%tile_5_4) {address = 35584 : i32, sym_name = "BL1_5_2_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_5_2_cons_prod_lock_0 = aie.lock(%tile_5_4, 2) {init = 1 : i32, sym_name = "BL1_5_2_cons_prod_lock_0"}
    %BL1_5_2_cons_cons_lock_0 = aie.lock(%tile_5_4, 3) {init = 0 : i32, sym_name = "BL1_5_2_cons_cons_lock_0"}
    %BL1_5_3_cons_buff_0 = aie.buffer(%tile_5_5) {address = 35584 : i32, sym_name = "BL1_5_3_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_5_3_cons_prod_lock_0 = aie.lock(%tile_5_5, 2) {init = 1 : i32, sym_name = "BL1_5_3_cons_prod_lock_0"}
    %BL1_5_3_cons_cons_lock_0 = aie.lock(%tile_5_5, 3) {init = 0 : i32, sym_name = "BL1_5_3_cons_cons_lock_0"}
    %BL3_5_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 393216 : i32, sym_name = "BL3_5_cons_buff_0"} : memref<64x128xbf16> 
    %BL3_5_cons_buff_1 = aie.buffer(%mem_tile_5_1) {address = 409600 : i32, sym_name = "BL3_5_cons_buff_1"} : memref<64x128xbf16> 
    %BL3_5_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 0) {init = 2 : i32, sym_name = "BL3_5_cons_prod_lock_0"}
    %BL3_5_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 1) {init = 0 : i32, sym_name = "BL3_5_cons_cons_lock_0"}
    %BL3_5_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "BL3_5_prod_lock_0"}
    %BL3_5_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "BL3_5_cons_lock_0"}
    %BL1_4_0_cons_buff_0 = aie.buffer(%tile_4_2) {address = 35584 : i32, sym_name = "BL1_4_0_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_4_0_cons_prod_lock_0 = aie.lock(%tile_4_2, 2) {init = 1 : i32, sym_name = "BL1_4_0_cons_prod_lock_0"}
    %BL1_4_0_cons_cons_lock_0 = aie.lock(%tile_4_2, 3) {init = 0 : i32, sym_name = "BL1_4_0_cons_cons_lock_0"}
    %BL1_4_1_cons_buff_0 = aie.buffer(%tile_4_3) {address = 35584 : i32, sym_name = "BL1_4_1_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_4_1_cons_prod_lock_0 = aie.lock(%tile_4_3, 2) {init = 1 : i32, sym_name = "BL1_4_1_cons_prod_lock_0"}
    %BL1_4_1_cons_cons_lock_0 = aie.lock(%tile_4_3, 3) {init = 0 : i32, sym_name = "BL1_4_1_cons_cons_lock_0"}
    %BL1_4_2_cons_buff_0 = aie.buffer(%tile_4_4) {address = 35584 : i32, sym_name = "BL1_4_2_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_4_2_cons_prod_lock_0 = aie.lock(%tile_4_4, 2) {init = 1 : i32, sym_name = "BL1_4_2_cons_prod_lock_0"}
    %BL1_4_2_cons_cons_lock_0 = aie.lock(%tile_4_4, 3) {init = 0 : i32, sym_name = "BL1_4_2_cons_cons_lock_0"}
    %BL1_4_3_cons_buff_0 = aie.buffer(%tile_4_5) {address = 35584 : i32, sym_name = "BL1_4_3_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_4_3_cons_prod_lock_0 = aie.lock(%tile_4_5, 2) {init = 1 : i32, sym_name = "BL1_4_3_cons_prod_lock_0"}
    %BL1_4_3_cons_cons_lock_0 = aie.lock(%tile_4_5, 3) {init = 0 : i32, sym_name = "BL1_4_3_cons_cons_lock_0"}
    %BL3_4_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 393216 : i32, sym_name = "BL3_4_cons_buff_0"} : memref<64x128xbf16> 
    %BL3_4_cons_buff_1 = aie.buffer(%mem_tile_4_1) {address = 409600 : i32, sym_name = "BL3_4_cons_buff_1"} : memref<64x128xbf16> 
    %BL3_4_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 2) {init = 2 : i32, sym_name = "BL3_4_cons_prod_lock_0"}
    %BL3_4_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 3) {init = 0 : i32, sym_name = "BL3_4_cons_cons_lock_0"}
    %BL3_4_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "BL3_4_prod_lock_0"}
    %BL3_4_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "BL3_4_cons_lock_0"}
    %BL1_3_0_cons_buff_0 = aie.buffer(%tile_3_2) {address = 35584 : i32, sym_name = "BL1_3_0_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_3_0_cons_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 1 : i32, sym_name = "BL1_3_0_cons_prod_lock_0"}
    %BL1_3_0_cons_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "BL1_3_0_cons_cons_lock_0"}
    %BL1_3_1_cons_buff_0 = aie.buffer(%tile_3_3) {address = 35584 : i32, sym_name = "BL1_3_1_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_3_1_cons_prod_lock_0 = aie.lock(%tile_3_3, 2) {init = 1 : i32, sym_name = "BL1_3_1_cons_prod_lock_0"}
    %BL1_3_1_cons_cons_lock_0 = aie.lock(%tile_3_3, 3) {init = 0 : i32, sym_name = "BL1_3_1_cons_cons_lock_0"}
    %BL1_3_2_cons_buff_0 = aie.buffer(%tile_3_4) {address = 35584 : i32, sym_name = "BL1_3_2_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_3_2_cons_prod_lock_0 = aie.lock(%tile_3_4, 2) {init = 1 : i32, sym_name = "BL1_3_2_cons_prod_lock_0"}
    %BL1_3_2_cons_cons_lock_0 = aie.lock(%tile_3_4, 3) {init = 0 : i32, sym_name = "BL1_3_2_cons_cons_lock_0"}
    %BL1_3_3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 35584 : i32, sym_name = "BL1_3_3_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_3_3_cons_prod_lock_0 = aie.lock(%tile_3_5, 2) {init = 1 : i32, sym_name = "BL1_3_3_cons_prod_lock_0"}
    %BL1_3_3_cons_cons_lock_0 = aie.lock(%tile_3_5, 3) {init = 0 : i32, sym_name = "BL1_3_3_cons_cons_lock_0"}
    %BL3_3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 393216 : i32, sym_name = "BL3_3_cons_buff_0"} : memref<64x128xbf16> 
    %BL3_3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 409600 : i32, sym_name = "BL3_3_cons_buff_1"} : memref<64x128xbf16> 
    %BL3_3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "BL3_3_cons_prod_lock_0"}
    %BL3_3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "BL3_3_cons_cons_lock_0"}
    %BL3_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "BL3_3_prod_lock_0"}
    %BL3_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "BL3_3_cons_lock_0"}
    %BL1_2_0_cons_buff_0 = aie.buffer(%tile_2_2) {address = 35584 : i32, sym_name = "BL1_2_0_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_2_0_cons_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 1 : i32, sym_name = "BL1_2_0_cons_prod_lock_0"}
    %BL1_2_0_cons_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "BL1_2_0_cons_cons_lock_0"}
    %BL1_2_1_cons_buff_0 = aie.buffer(%tile_2_3) {address = 35584 : i32, sym_name = "BL1_2_1_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_2_1_cons_prod_lock_0 = aie.lock(%tile_2_3, 2) {init = 1 : i32, sym_name = "BL1_2_1_cons_prod_lock_0"}
    %BL1_2_1_cons_cons_lock_0 = aie.lock(%tile_2_3, 3) {init = 0 : i32, sym_name = "BL1_2_1_cons_cons_lock_0"}
    %BL1_2_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 35584 : i32, sym_name = "BL1_2_2_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_2_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 2) {init = 1 : i32, sym_name = "BL1_2_2_cons_prod_lock_0"}
    %BL1_2_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 3) {init = 0 : i32, sym_name = "BL1_2_2_cons_cons_lock_0"}
    %BL1_2_3_cons_buff_0 = aie.buffer(%tile_2_5) {address = 35584 : i32, sym_name = "BL1_2_3_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_2_3_cons_prod_lock_0 = aie.lock(%tile_2_5, 2) {init = 1 : i32, sym_name = "BL1_2_3_cons_prod_lock_0"}
    %BL1_2_3_cons_cons_lock_0 = aie.lock(%tile_2_5, 3) {init = 0 : i32, sym_name = "BL1_2_3_cons_cons_lock_0"}
    %BL3_2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 393216 : i32, sym_name = "BL3_2_cons_buff_0"} : memref<64x128xbf16> 
    %BL3_2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 409600 : i32, sym_name = "BL3_2_cons_buff_1"} : memref<64x128xbf16> 
    %BL3_2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 2) {init = 2 : i32, sym_name = "BL3_2_cons_prod_lock_0"}
    %BL3_2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "BL3_2_cons_cons_lock_0"}
    %BL3_2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "BL3_2_prod_lock_0"}
    %BL3_2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "BL3_2_cons_lock_0"}
    %BL1_1_0_cons_buff_0 = aie.buffer(%tile_1_2) {address = 35584 : i32, sym_name = "BL1_1_0_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_1_0_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 1 : i32, sym_name = "BL1_1_0_cons_prod_lock_0"}
    %BL1_1_0_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "BL1_1_0_cons_cons_lock_0"}
    %BL1_1_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 35584 : i32, sym_name = "BL1_1_1_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_1_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 1 : i32, sym_name = "BL1_1_1_cons_prod_lock_0"}
    %BL1_1_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "BL1_1_1_cons_cons_lock_0"}
    %BL1_1_2_cons_buff_0 = aie.buffer(%tile_1_4) {address = 35584 : i32, sym_name = "BL1_1_2_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_1_2_cons_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 1 : i32, sym_name = "BL1_1_2_cons_prod_lock_0"}
    %BL1_1_2_cons_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "BL1_1_2_cons_cons_lock_0"}
    %BL1_1_3_cons_buff_0 = aie.buffer(%tile_1_5) {address = 35584 : i32, sym_name = "BL1_1_3_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_1_3_cons_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 1 : i32, sym_name = "BL1_1_3_cons_prod_lock_0"}
    %BL1_1_3_cons_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "BL1_1_3_cons_cons_lock_0"}
    %BL3_1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 393216 : i32, sym_name = "BL3_1_cons_buff_0"} : memref<64x128xbf16> 
    %BL3_1_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 409600 : i32, sym_name = "BL3_1_cons_buff_1"} : memref<64x128xbf16> 
    %BL3_1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "BL3_1_cons_prod_lock_0"}
    %BL3_1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "BL3_1_cons_cons_lock_0"}
    %BL3_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "BL3_1_prod_lock_0"}
    %BL3_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "BL3_1_cons_lock_0"}
    %BL1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 35584 : i32, sym_name = "BL1_0_0_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "BL1_0_0_cons_prod_lock_0"}
    %BL1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "BL1_0_0_cons_cons_lock_0"}
    %BL1_0_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 35584 : i32, sym_name = "BL1_0_1_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_0_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 1 : i32, sym_name = "BL1_0_1_cons_prod_lock_0"}
    %BL1_0_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "BL1_0_1_cons_cons_lock_0"}
    %BL1_0_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 35584 : i32, sym_name = "BL1_0_2_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_0_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 1 : i32, sym_name = "BL1_0_2_cons_prod_lock_0"}
    %BL1_0_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "BL1_0_2_cons_cons_lock_0"}
    %BL1_0_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 35584 : i32, sym_name = "BL1_0_3_cons_buff_0"} : memref<64x128xbf16> 
    %BL1_0_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 1 : i32, sym_name = "BL1_0_3_cons_prod_lock_0"}
    %BL1_0_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "BL1_0_3_cons_cons_lock_0"}
    %BL3_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 393216 : i32, sym_name = "BL3_0_cons_buff_0"} : memref<64x128xbf16> 
    %BL3_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 409600 : i32, sym_name = "BL3_0_cons_buff_1"} : memref<64x128xbf16> 
    %BL3_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "BL3_0_cons_prod_lock_0"}
    %BL3_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "BL3_0_cons_cons_lock_0"}
    %BL3_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "BL3_0_prod_lock_0"}
    %BL3_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "BL3_0_cons_lock_0"}
    %AL1_3_0_cons_buff_0 = aie.buffer(%tile_0_5) {address = 51968 : i32, sym_name = "AL1_3_0_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_3_0_cons_buff_1 = aie.buffer(%tile_0_5) {address = 56064 : i32, sym_name = "AL1_3_0_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_3_0_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "AL1_3_0_cons_prod_lock_0"}
    %AL1_3_0_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "AL1_3_0_cons_cons_lock_0"}
    %AL1_3_1_cons_buff_0 = aie.buffer(%tile_1_5) {address = 51968 : i32, sym_name = "AL1_3_1_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_3_1_cons_buff_1 = aie.buffer(%tile_1_5) {address = 56064 : i32, sym_name = "AL1_3_1_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_3_1_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "AL1_3_1_cons_prod_lock_0"}
    %AL1_3_1_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "AL1_3_1_cons_cons_lock_0"}
    %AL1_3_2_cons_buff_0 = aie.buffer(%tile_2_5) {address = 51968 : i32, sym_name = "AL1_3_2_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_3_2_cons_buff_1 = aie.buffer(%tile_2_5) {address = 56064 : i32, sym_name = "AL1_3_2_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_3_2_cons_prod_lock_0 = aie.lock(%tile_2_5, 0) {init = 2 : i32, sym_name = "AL1_3_2_cons_prod_lock_0"}
    %AL1_3_2_cons_cons_lock_0 = aie.lock(%tile_2_5, 1) {init = 0 : i32, sym_name = "AL1_3_2_cons_cons_lock_0"}
    %AL1_3_3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 51968 : i32, sym_name = "AL1_3_3_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_3_3_cons_buff_1 = aie.buffer(%tile_3_5) {address = 56064 : i32, sym_name = "AL1_3_3_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_3_3_cons_prod_lock_0 = aie.lock(%tile_3_5, 0) {init = 2 : i32, sym_name = "AL1_3_3_cons_prod_lock_0"}
    %AL1_3_3_cons_cons_lock_0 = aie.lock(%tile_3_5, 1) {init = 0 : i32, sym_name = "AL1_3_3_cons_cons_lock_0"}
    %AL1_3_4_cons_buff_0 = aie.buffer(%tile_4_5) {address = 51968 : i32, sym_name = "AL1_3_4_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_3_4_cons_buff_1 = aie.buffer(%tile_4_5) {address = 56064 : i32, sym_name = "AL1_3_4_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_3_4_cons_prod_lock_0 = aie.lock(%tile_4_5, 0) {init = 2 : i32, sym_name = "AL1_3_4_cons_prod_lock_0"}
    %AL1_3_4_cons_cons_lock_0 = aie.lock(%tile_4_5, 1) {init = 0 : i32, sym_name = "AL1_3_4_cons_cons_lock_0"}
    %AL1_3_5_cons_buff_0 = aie.buffer(%tile_5_5) {address = 51968 : i32, sym_name = "AL1_3_5_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_3_5_cons_buff_1 = aie.buffer(%tile_5_5) {address = 56064 : i32, sym_name = "AL1_3_5_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_3_5_cons_prod_lock_0 = aie.lock(%tile_5_5, 0) {init = 2 : i32, sym_name = "AL1_3_5_cons_prod_lock_0"}
    %AL1_3_5_cons_cons_lock_0 = aie.lock(%tile_5_5, 1) {init = 0 : i32, sym_name = "AL1_3_5_cons_cons_lock_0"}
    %AL1_3_6_cons_buff_0 = aie.buffer(%tile_6_5) {address = 51968 : i32, sym_name = "AL1_3_6_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_3_6_cons_buff_1 = aie.buffer(%tile_6_5) {address = 56064 : i32, sym_name = "AL1_3_6_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_3_6_cons_prod_lock_0 = aie.lock(%tile_6_5, 0) {init = 2 : i32, sym_name = "AL1_3_6_cons_prod_lock_0"}
    %AL1_3_6_cons_cons_lock_0 = aie.lock(%tile_6_5, 1) {init = 0 : i32, sym_name = "AL1_3_6_cons_cons_lock_0"}
    %AL1_3_7_cons_buff_0 = aie.buffer(%tile_7_5) {address = 51968 : i32, sym_name = "AL1_3_7_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_3_7_cons_buff_1 = aie.buffer(%tile_7_5) {address = 56064 : i32, sym_name = "AL1_3_7_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_3_7_cons_prod_lock_0 = aie.lock(%tile_7_5, 0) {init = 2 : i32, sym_name = "AL1_3_7_cons_prod_lock_0"}
    %AL1_3_7_cons_cons_lock_0 = aie.lock(%tile_7_5, 1) {init = 0 : i32, sym_name = "AL1_3_7_cons_cons_lock_0"}
    %AL3_3_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 131072 : i32, sym_name = "AL3_3_cons_buff_0"} : memref<128x512xbf16> 
    %AL3_3_cons_buff_1 = aie.buffer(%mem_tile_6_1) {address = 262144 : i32, sym_name = "AL3_3_cons_buff_1"} : memref<128x512xbf16> 
    %AL3_3_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 0) {init = 2 : i32, sym_name = "AL3_3_cons_prod_lock_0"}
    %AL3_3_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 1) {init = 0 : i32, sym_name = "AL3_3_cons_cons_lock_0"}
    %AL3_3_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "AL3_3_prod_lock_0"}
    %AL3_3_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "AL3_3_cons_lock_0"}
    %AL1_2_0_cons_buff_0 = aie.buffer(%tile_0_4) {address = 51968 : i32, sym_name = "AL1_2_0_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_2_0_cons_buff_1 = aie.buffer(%tile_0_4) {address = 56064 : i32, sym_name = "AL1_2_0_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_2_0_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "AL1_2_0_cons_prod_lock_0"}
    %AL1_2_0_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "AL1_2_0_cons_cons_lock_0"}
    %AL1_2_1_cons_buff_0 = aie.buffer(%tile_1_4) {address = 51968 : i32, sym_name = "AL1_2_1_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_2_1_cons_buff_1 = aie.buffer(%tile_1_4) {address = 56064 : i32, sym_name = "AL1_2_1_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_2_1_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "AL1_2_1_cons_prod_lock_0"}
    %AL1_2_1_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "AL1_2_1_cons_cons_lock_0"}
    %AL1_2_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 51968 : i32, sym_name = "AL1_2_2_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_2_2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 56064 : i32, sym_name = "AL1_2_2_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_2_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 0) {init = 2 : i32, sym_name = "AL1_2_2_cons_prod_lock_0"}
    %AL1_2_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 1) {init = 0 : i32, sym_name = "AL1_2_2_cons_cons_lock_0"}
    %AL1_2_3_cons_buff_0 = aie.buffer(%tile_3_4) {address = 51968 : i32, sym_name = "AL1_2_3_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_2_3_cons_buff_1 = aie.buffer(%tile_3_4) {address = 56064 : i32, sym_name = "AL1_2_3_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_2_3_cons_prod_lock_0 = aie.lock(%tile_3_4, 0) {init = 2 : i32, sym_name = "AL1_2_3_cons_prod_lock_0"}
    %AL1_2_3_cons_cons_lock_0 = aie.lock(%tile_3_4, 1) {init = 0 : i32, sym_name = "AL1_2_3_cons_cons_lock_0"}
    %AL1_2_4_cons_buff_0 = aie.buffer(%tile_4_4) {address = 51968 : i32, sym_name = "AL1_2_4_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_2_4_cons_buff_1 = aie.buffer(%tile_4_4) {address = 56064 : i32, sym_name = "AL1_2_4_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_2_4_cons_prod_lock_0 = aie.lock(%tile_4_4, 0) {init = 2 : i32, sym_name = "AL1_2_4_cons_prod_lock_0"}
    %AL1_2_4_cons_cons_lock_0 = aie.lock(%tile_4_4, 1) {init = 0 : i32, sym_name = "AL1_2_4_cons_cons_lock_0"}
    %AL1_2_5_cons_buff_0 = aie.buffer(%tile_5_4) {address = 51968 : i32, sym_name = "AL1_2_5_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_2_5_cons_buff_1 = aie.buffer(%tile_5_4) {address = 56064 : i32, sym_name = "AL1_2_5_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_2_5_cons_prod_lock_0 = aie.lock(%tile_5_4, 0) {init = 2 : i32, sym_name = "AL1_2_5_cons_prod_lock_0"}
    %AL1_2_5_cons_cons_lock_0 = aie.lock(%tile_5_4, 1) {init = 0 : i32, sym_name = "AL1_2_5_cons_cons_lock_0"}
    %AL1_2_6_cons_buff_0 = aie.buffer(%tile_6_4) {address = 51968 : i32, sym_name = "AL1_2_6_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_2_6_cons_buff_1 = aie.buffer(%tile_6_4) {address = 56064 : i32, sym_name = "AL1_2_6_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_2_6_cons_prod_lock_0 = aie.lock(%tile_6_4, 0) {init = 2 : i32, sym_name = "AL1_2_6_cons_prod_lock_0"}
    %AL1_2_6_cons_cons_lock_0 = aie.lock(%tile_6_4, 1) {init = 0 : i32, sym_name = "AL1_2_6_cons_cons_lock_0"}
    %AL1_2_7_cons_buff_0 = aie.buffer(%tile_7_4) {address = 51968 : i32, sym_name = "AL1_2_7_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_2_7_cons_buff_1 = aie.buffer(%tile_7_4) {address = 56064 : i32, sym_name = "AL1_2_7_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_2_7_cons_prod_lock_0 = aie.lock(%tile_7_4, 0) {init = 2 : i32, sym_name = "AL1_2_7_cons_prod_lock_0"}
    %AL1_2_7_cons_cons_lock_0 = aie.lock(%tile_7_4, 1) {init = 0 : i32, sym_name = "AL1_2_7_cons_cons_lock_0"}
    %AL3_2_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 131072 : i32, sym_name = "AL3_2_cons_buff_0"} : memref<128x512xbf16> 
    %AL3_2_cons_buff_1 = aie.buffer(%mem_tile_4_1) {address = 262144 : i32, sym_name = "AL3_2_cons_buff_1"} : memref<128x512xbf16> 
    %AL3_2_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 0) {init = 2 : i32, sym_name = "AL3_2_cons_prod_lock_0"}
    %AL3_2_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 1) {init = 0 : i32, sym_name = "AL3_2_cons_cons_lock_0"}
    %AL3_2_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "AL3_2_prod_lock_0"}
    %AL3_2_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "AL3_2_cons_lock_0"}
    %AL1_1_0_cons_buff_0 = aie.buffer(%tile_0_3) {address = 51968 : i32, sym_name = "AL1_1_0_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_1_0_cons_buff_1 = aie.buffer(%tile_0_3) {address = 56064 : i32, sym_name = "AL1_1_0_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_1_0_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "AL1_1_0_cons_prod_lock_0"}
    %AL1_1_0_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "AL1_1_0_cons_cons_lock_0"}
    %AL1_1_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 51968 : i32, sym_name = "AL1_1_1_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_1_1_cons_buff_1 = aie.buffer(%tile_1_3) {address = 56064 : i32, sym_name = "AL1_1_1_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_1_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "AL1_1_1_cons_prod_lock_0"}
    %AL1_1_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "AL1_1_1_cons_cons_lock_0"}
    %AL1_1_2_cons_buff_0 = aie.buffer(%tile_2_3) {address = 51968 : i32, sym_name = "AL1_1_2_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_1_2_cons_buff_1 = aie.buffer(%tile_2_3) {address = 56064 : i32, sym_name = "AL1_1_2_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_1_2_cons_prod_lock_0 = aie.lock(%tile_2_3, 0) {init = 2 : i32, sym_name = "AL1_1_2_cons_prod_lock_0"}
    %AL1_1_2_cons_cons_lock_0 = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "AL1_1_2_cons_cons_lock_0"}
    %AL1_1_3_cons_buff_0 = aie.buffer(%tile_3_3) {address = 51968 : i32, sym_name = "AL1_1_3_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_1_3_cons_buff_1 = aie.buffer(%tile_3_3) {address = 56064 : i32, sym_name = "AL1_1_3_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_1_3_cons_prod_lock_0 = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "AL1_1_3_cons_prod_lock_0"}
    %AL1_1_3_cons_cons_lock_0 = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "AL1_1_3_cons_cons_lock_0"}
    %AL1_1_4_cons_buff_0 = aie.buffer(%tile_4_3) {address = 51968 : i32, sym_name = "AL1_1_4_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_1_4_cons_buff_1 = aie.buffer(%tile_4_3) {address = 56064 : i32, sym_name = "AL1_1_4_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_1_4_cons_prod_lock_0 = aie.lock(%tile_4_3, 0) {init = 2 : i32, sym_name = "AL1_1_4_cons_prod_lock_0"}
    %AL1_1_4_cons_cons_lock_0 = aie.lock(%tile_4_3, 1) {init = 0 : i32, sym_name = "AL1_1_4_cons_cons_lock_0"}
    %AL1_1_5_cons_buff_0 = aie.buffer(%tile_5_3) {address = 51968 : i32, sym_name = "AL1_1_5_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_1_5_cons_buff_1 = aie.buffer(%tile_5_3) {address = 56064 : i32, sym_name = "AL1_1_5_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_1_5_cons_prod_lock_0 = aie.lock(%tile_5_3, 0) {init = 2 : i32, sym_name = "AL1_1_5_cons_prod_lock_0"}
    %AL1_1_5_cons_cons_lock_0 = aie.lock(%tile_5_3, 1) {init = 0 : i32, sym_name = "AL1_1_5_cons_cons_lock_0"}
    %AL1_1_6_cons_buff_0 = aie.buffer(%tile_6_3) {address = 51968 : i32, sym_name = "AL1_1_6_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_1_6_cons_buff_1 = aie.buffer(%tile_6_3) {address = 56064 : i32, sym_name = "AL1_1_6_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_1_6_cons_prod_lock_0 = aie.lock(%tile_6_3, 0) {init = 2 : i32, sym_name = "AL1_1_6_cons_prod_lock_0"}
    %AL1_1_6_cons_cons_lock_0 = aie.lock(%tile_6_3, 1) {init = 0 : i32, sym_name = "AL1_1_6_cons_cons_lock_0"}
    %AL1_1_7_cons_buff_0 = aie.buffer(%tile_7_3) {address = 51968 : i32, sym_name = "AL1_1_7_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_1_7_cons_buff_1 = aie.buffer(%tile_7_3) {address = 56064 : i32, sym_name = "AL1_1_7_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_1_7_cons_prod_lock_0 = aie.lock(%tile_7_3, 0) {init = 2 : i32, sym_name = "AL1_1_7_cons_prod_lock_0"}
    %AL1_1_7_cons_cons_lock_0 = aie.lock(%tile_7_3, 1) {init = 0 : i32, sym_name = "AL1_1_7_cons_cons_lock_0"}
    %AL3_1_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 131072 : i32, sym_name = "AL3_1_cons_buff_0"} : memref<128x512xbf16> 
    %AL3_1_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 262144 : i32, sym_name = "AL3_1_cons_buff_1"} : memref<128x512xbf16> 
    %AL3_1_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "AL3_1_cons_prod_lock_0"}
    %AL3_1_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "AL3_1_cons_cons_lock_0"}
    %AL3_1_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "AL3_1_prod_lock_0"}
    %AL3_1_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "AL3_1_cons_lock_0"}
    %AL1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 51968 : i32, sym_name = "AL1_0_0_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 56064 : i32, sym_name = "AL1_0_0_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "AL1_0_0_cons_prod_lock_0"}
    %AL1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "AL1_0_0_cons_cons_lock_0"}
    %AL1_0_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 51968 : i32, sym_name = "AL1_0_1_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_0_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 56064 : i32, sym_name = "AL1_0_1_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_0_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "AL1_0_1_cons_prod_lock_0"}
    %AL1_0_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "AL1_0_1_cons_cons_lock_0"}
    %AL1_0_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 51968 : i32, sym_name = "AL1_0_2_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_0_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 56064 : i32, sym_name = "AL1_0_2_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_0_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "AL1_0_2_cons_prod_lock_0"}
    %AL1_0_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "AL1_0_2_cons_cons_lock_0"}
    %AL1_0_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 51968 : i32, sym_name = "AL1_0_3_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_0_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 56064 : i32, sym_name = "AL1_0_3_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_0_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "AL1_0_3_cons_prod_lock_0"}
    %AL1_0_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "AL1_0_3_cons_cons_lock_0"}
    %AL1_0_4_cons_buff_0 = aie.buffer(%tile_4_2) {address = 51968 : i32, sym_name = "AL1_0_4_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_0_4_cons_buff_1 = aie.buffer(%tile_4_2) {address = 56064 : i32, sym_name = "AL1_0_4_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_0_4_cons_prod_lock_0 = aie.lock(%tile_4_2, 0) {init = 2 : i32, sym_name = "AL1_0_4_cons_prod_lock_0"}
    %AL1_0_4_cons_cons_lock_0 = aie.lock(%tile_4_2, 1) {init = 0 : i32, sym_name = "AL1_0_4_cons_cons_lock_0"}
    %AL1_0_5_cons_buff_0 = aie.buffer(%tile_5_2) {address = 51968 : i32, sym_name = "AL1_0_5_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_0_5_cons_buff_1 = aie.buffer(%tile_5_2) {address = 56064 : i32, sym_name = "AL1_0_5_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_0_5_cons_prod_lock_0 = aie.lock(%tile_5_2, 0) {init = 2 : i32, sym_name = "AL1_0_5_cons_prod_lock_0"}
    %AL1_0_5_cons_cons_lock_0 = aie.lock(%tile_5_2, 1) {init = 0 : i32, sym_name = "AL1_0_5_cons_cons_lock_0"}
    %AL1_0_6_cons_buff_0 = aie.buffer(%tile_6_2) {address = 51968 : i32, sym_name = "AL1_0_6_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_0_6_cons_buff_1 = aie.buffer(%tile_6_2) {address = 56064 : i32, sym_name = "AL1_0_6_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_0_6_cons_prod_lock_0 = aie.lock(%tile_6_2, 0) {init = 2 : i32, sym_name = "AL1_0_6_cons_prod_lock_0"}
    %AL1_0_6_cons_cons_lock_0 = aie.lock(%tile_6_2, 1) {init = 0 : i32, sym_name = "AL1_0_6_cons_cons_lock_0"}
    %AL1_0_7_cons_buff_0 = aie.buffer(%tile_7_2) {address = 51968 : i32, sym_name = "AL1_0_7_cons_buff_0"} : memref<32x64xbf16> 
    %AL1_0_7_cons_buff_1 = aie.buffer(%tile_7_2) {address = 56064 : i32, sym_name = "AL1_0_7_cons_buff_1"} : memref<32x64xbf16> 
    %AL1_0_7_cons_prod_lock_0 = aie.lock(%tile_7_2, 0) {init = 2 : i32, sym_name = "AL1_0_7_cons_prod_lock_0"}
    %AL1_0_7_cons_cons_lock_0 = aie.lock(%tile_7_2, 1) {init = 0 : i32, sym_name = "AL1_0_7_cons_cons_lock_0"}
    %AL3_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, sym_name = "AL3_0_cons_buff_0"} : memref<128x512xbf16> 
    %AL3_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 262144 : i32, sym_name = "AL3_0_cons_buff_1"} : memref<128x512xbf16> 
    %AL3_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "AL3_0_cons_prod_lock_0"}
    %AL3_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "AL3_0_cons_cons_lock_0"}
    %AL3_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "AL3_0_prod_lock_0"}
    %AL3_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "AL3_0_cons_lock_0"}
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
    %core_0_2 = aie.core(%tile_0_2) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_0_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_0_cons_buff_0, %BL1_0_0_cons_buff_0, %CL1_0_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_0_cons_buff_1, %BL1_0_0_cons_buff_0, %CL1_0_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_0_cons_buff_0, %BL1_0_0_cons_buff_0, %CL1_0_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_0_cons_buff_1, %BL1_0_0_cons_buff_0, %CL1_0_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_0_cons_buff_0, %BL1_0_0_cons_buff_0, %CL1_0_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_0_cons_buff_1, %BL1_0_0_cons_buff_0, %CL1_0_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_0_cons_buff_0, %BL1_0_0_cons_buff_0, %CL1_0_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_0_cons_buff_1, %BL1_0_0_cons_buff_0, %CL1_0_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_0_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_0_0_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_1_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_1_cons_buff_0, %BL1_1_0_cons_buff_0, %CL1_1_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_1_cons_buff_1, %BL1_1_0_cons_buff_0, %CL1_1_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_1_cons_buff_0, %BL1_1_0_cons_buff_0, %CL1_1_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_1_cons_buff_1, %BL1_1_0_cons_buff_0, %CL1_1_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_1_cons_buff_0, %BL1_1_0_cons_buff_0, %CL1_1_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_1_cons_buff_1, %BL1_1_0_cons_buff_0, %CL1_1_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_1_cons_buff_0, %BL1_1_0_cons_buff_0, %CL1_1_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_1_cons_buff_1, %BL1_1_0_cons_buff_0, %CL1_1_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_0_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_1_0_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_2_2 = aie.core(%tile_2_2) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_2_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_2_cons_buff_0, %BL1_2_0_cons_buff_0, %CL1_2_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_2_cons_buff_1, %BL1_2_0_cons_buff_0, %CL1_2_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_2_cons_buff_0, %BL1_2_0_cons_buff_0, %CL1_2_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_2_cons_buff_1, %BL1_2_0_cons_buff_0, %CL1_2_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_2_cons_buff_0, %BL1_2_0_cons_buff_0, %CL1_2_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_2_cons_buff_1, %BL1_2_0_cons_buff_0, %CL1_2_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_2_cons_buff_0, %BL1_2_0_cons_buff_0, %CL1_2_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_2_cons_buff_1, %BL1_2_0_cons_buff_0, %CL1_2_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_0_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_2_0_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_3_2 = aie.core(%tile_3_2) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_3_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_3_cons_buff_0, %BL1_3_0_cons_buff_0, %CL1_3_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_3_cons_buff_1, %BL1_3_0_cons_buff_0, %CL1_3_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_3_cons_buff_0, %BL1_3_0_cons_buff_0, %CL1_3_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_3_cons_buff_1, %BL1_3_0_cons_buff_0, %CL1_3_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_3_cons_buff_0, %BL1_3_0_cons_buff_0, %CL1_3_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_3_cons_buff_1, %BL1_3_0_cons_buff_0, %CL1_3_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_3_cons_buff_0, %BL1_3_0_cons_buff_0, %CL1_3_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_3_cons_buff_1, %BL1_3_0_cons_buff_0, %CL1_3_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_0_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_3_0_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_4_2 = aie.core(%tile_4_2) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_4_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_4_cons_buff_0, %BL1_4_0_cons_buff_0, %CL1_4_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_4_cons_buff_1, %BL1_4_0_cons_buff_0, %CL1_4_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_4_cons_buff_0, %BL1_4_0_cons_buff_0, %CL1_4_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_4_cons_buff_1, %BL1_4_0_cons_buff_0, %CL1_4_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_4_cons_buff_0, %BL1_4_0_cons_buff_0, %CL1_4_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_4_cons_buff_1, %BL1_4_0_cons_buff_0, %CL1_4_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_4_cons_buff_0, %BL1_4_0_cons_buff_0, %CL1_4_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_4_cons_buff_1, %BL1_4_0_cons_buff_0, %CL1_4_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_0_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_4_0_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_5_2 = aie.core(%tile_5_2) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_5_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_5_cons_buff_0, %BL1_5_0_cons_buff_0, %CL1_5_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_5_cons_buff_1, %BL1_5_0_cons_buff_0, %CL1_5_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_5_cons_buff_0, %BL1_5_0_cons_buff_0, %CL1_5_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_5_cons_buff_1, %BL1_5_0_cons_buff_0, %CL1_5_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_5_cons_buff_0, %BL1_5_0_cons_buff_0, %CL1_5_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_5_cons_buff_1, %BL1_5_0_cons_buff_0, %CL1_5_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_5_cons_buff_0, %BL1_5_0_cons_buff_0, %CL1_5_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_5_cons_buff_1, %BL1_5_0_cons_buff_0, %CL1_5_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_0_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_5_0_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_6_2 = aie.core(%tile_6_2) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_6_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_6_cons_buff_0, %BL1_6_0_cons_buff_0, %CL1_6_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_6_cons_buff_1, %BL1_6_0_cons_buff_0, %CL1_6_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_6_cons_buff_0, %BL1_6_0_cons_buff_0, %CL1_6_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_6_cons_buff_1, %BL1_6_0_cons_buff_0, %CL1_6_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_6_cons_buff_0, %BL1_6_0_cons_buff_0, %CL1_6_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_6_cons_buff_1, %BL1_6_0_cons_buff_0, %CL1_6_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_6_cons_buff_0, %BL1_6_0_cons_buff_0, %CL1_6_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_6_cons_buff_1, %BL1_6_0_cons_buff_0, %CL1_6_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_0_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_6_0_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_7_2 = aie.core(%tile_7_2) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_7_0_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_7_cons_buff_0, %BL1_7_0_cons_buff_0, %CL1_7_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_7_cons_buff_1, %BL1_7_0_cons_buff_0, %CL1_7_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_7_cons_buff_0, %BL1_7_0_cons_buff_0, %CL1_7_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_7_cons_buff_1, %BL1_7_0_cons_buff_0, %CL1_7_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_7_cons_buff_0, %BL1_7_0_cons_buff_0, %CL1_7_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_7_cons_buff_1, %BL1_7_0_cons_buff_0, %CL1_7_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_7_cons_buff_0, %BL1_7_0_cons_buff_0, %CL1_7_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_0_7_cons_buff_1, %BL1_7_0_cons_buff_0, %CL1_7_0_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_0_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_0_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_7_0_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_0_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_0_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_0_cons_buff_0, %BL1_0_1_cons_buff_0, %CL1_0_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_0_cons_buff_1, %BL1_0_1_cons_buff_0, %CL1_0_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_0_cons_buff_0, %BL1_0_1_cons_buff_0, %CL1_0_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_0_cons_buff_1, %BL1_0_1_cons_buff_0, %CL1_0_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_0_cons_buff_0, %BL1_0_1_cons_buff_0, %CL1_0_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_0_cons_buff_1, %BL1_0_1_cons_buff_0, %CL1_0_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_0_cons_buff_0, %BL1_0_1_cons_buff_0, %CL1_0_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_0_cons_buff_1, %BL1_0_1_cons_buff_0, %CL1_0_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_1_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_0_1_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_1_3 = aie.core(%tile_1_3) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_1_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_1_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_1_cons_buff_0, %BL1_1_1_cons_buff_0, %CL1_1_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_1_cons_buff_1, %BL1_1_1_cons_buff_0, %CL1_1_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_1_cons_buff_0, %BL1_1_1_cons_buff_0, %CL1_1_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_1_cons_buff_1, %BL1_1_1_cons_buff_0, %CL1_1_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_1_cons_buff_0, %BL1_1_1_cons_buff_0, %CL1_1_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_1_cons_buff_1, %BL1_1_1_cons_buff_0, %CL1_1_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_1_cons_buff_0, %BL1_1_1_cons_buff_0, %CL1_1_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_1_cons_buff_1, %BL1_1_1_cons_buff_0, %CL1_1_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_1_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_1_1_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_2_3 = aie.core(%tile_2_3) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_2_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_2_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_2_cons_buff_0, %BL1_2_1_cons_buff_0, %CL1_2_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_2_cons_buff_1, %BL1_2_1_cons_buff_0, %CL1_2_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_2_cons_buff_0, %BL1_2_1_cons_buff_0, %CL1_2_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_2_cons_buff_1, %BL1_2_1_cons_buff_0, %CL1_2_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_2_cons_buff_0, %BL1_2_1_cons_buff_0, %CL1_2_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_2_cons_buff_1, %BL1_2_1_cons_buff_0, %CL1_2_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_2_cons_buff_0, %BL1_2_1_cons_buff_0, %CL1_2_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_2_cons_buff_1, %BL1_2_1_cons_buff_0, %CL1_2_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_1_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_2_1_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_3_3 = aie.core(%tile_3_3) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_3_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_3_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_3_cons_buff_0, %BL1_3_1_cons_buff_0, %CL1_3_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_3_cons_buff_1, %BL1_3_1_cons_buff_0, %CL1_3_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_3_cons_buff_0, %BL1_3_1_cons_buff_0, %CL1_3_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_3_cons_buff_1, %BL1_3_1_cons_buff_0, %CL1_3_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_3_cons_buff_0, %BL1_3_1_cons_buff_0, %CL1_3_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_3_cons_buff_1, %BL1_3_1_cons_buff_0, %CL1_3_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_3_cons_buff_0, %BL1_3_1_cons_buff_0, %CL1_3_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_3_cons_buff_1, %BL1_3_1_cons_buff_0, %CL1_3_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_1_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_3_1_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_4_3 = aie.core(%tile_4_3) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_4_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_4_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_4_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_4_cons_buff_0, %BL1_4_1_cons_buff_0, %CL1_4_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_4_cons_buff_1, %BL1_4_1_cons_buff_0, %CL1_4_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_4_cons_buff_0, %BL1_4_1_cons_buff_0, %CL1_4_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_4_cons_buff_1, %BL1_4_1_cons_buff_0, %CL1_4_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_4_cons_buff_0, %BL1_4_1_cons_buff_0, %CL1_4_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_4_cons_buff_1, %BL1_4_1_cons_buff_0, %CL1_4_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_4_cons_buff_0, %BL1_4_1_cons_buff_0, %CL1_4_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_4_cons_buff_1, %BL1_4_1_cons_buff_0, %CL1_4_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_1_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_4_1_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_5_3 = aie.core(%tile_5_3) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_5_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_5_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_5_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_5_cons_buff_0, %BL1_5_1_cons_buff_0, %CL1_5_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_5_cons_buff_1, %BL1_5_1_cons_buff_0, %CL1_5_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_5_cons_buff_0, %BL1_5_1_cons_buff_0, %CL1_5_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_5_cons_buff_1, %BL1_5_1_cons_buff_0, %CL1_5_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_5_cons_buff_0, %BL1_5_1_cons_buff_0, %CL1_5_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_5_cons_buff_1, %BL1_5_1_cons_buff_0, %CL1_5_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_5_cons_buff_0, %BL1_5_1_cons_buff_0, %CL1_5_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_5_cons_buff_1, %BL1_5_1_cons_buff_0, %CL1_5_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_1_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_5_1_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_6_3 = aie.core(%tile_6_3) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_6_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_6_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_6_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_6_cons_buff_0, %BL1_6_1_cons_buff_0, %CL1_6_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_6_cons_buff_1, %BL1_6_1_cons_buff_0, %CL1_6_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_6_cons_buff_0, %BL1_6_1_cons_buff_0, %CL1_6_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_6_cons_buff_1, %BL1_6_1_cons_buff_0, %CL1_6_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_6_cons_buff_0, %BL1_6_1_cons_buff_0, %CL1_6_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_6_cons_buff_1, %BL1_6_1_cons_buff_0, %CL1_6_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_6_cons_buff_0, %BL1_6_1_cons_buff_0, %CL1_6_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_6_cons_buff_1, %BL1_6_1_cons_buff_0, %CL1_6_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_1_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_6_1_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_7_3 = aie.core(%tile_7_3) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_7_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_7_1_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_7_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_7_cons_buff_0, %BL1_7_1_cons_buff_0, %CL1_7_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_7_cons_buff_1, %BL1_7_1_cons_buff_0, %CL1_7_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_7_cons_buff_0, %BL1_7_1_cons_buff_0, %CL1_7_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_7_cons_buff_1, %BL1_7_1_cons_buff_0, %CL1_7_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_7_cons_buff_0, %BL1_7_1_cons_buff_0, %CL1_7_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_7_cons_buff_1, %BL1_7_1_cons_buff_0, %CL1_7_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_7_cons_buff_0, %BL1_7_1_cons_buff_0, %CL1_7_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_1_7_cons_buff_1, %BL1_7_1_cons_buff_0, %CL1_7_1_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_1_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_7_1_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_0_4 = aie.core(%tile_0_4) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_0_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_0_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_0_cons_buff_0, %BL1_0_2_cons_buff_0, %CL1_0_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_0_cons_buff_1, %BL1_0_2_cons_buff_0, %CL1_0_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_0_cons_buff_0, %BL1_0_2_cons_buff_0, %CL1_0_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_0_cons_buff_1, %BL1_0_2_cons_buff_0, %CL1_0_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_0_cons_buff_0, %BL1_0_2_cons_buff_0, %CL1_0_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_0_cons_buff_1, %BL1_0_2_cons_buff_0, %CL1_0_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_0_cons_buff_0, %BL1_0_2_cons_buff_0, %CL1_0_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_0_cons_buff_1, %BL1_0_2_cons_buff_0, %CL1_0_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_2_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_0_2_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_1_4 = aie.core(%tile_1_4) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_1_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_1_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_1_cons_buff_0, %BL1_1_2_cons_buff_0, %CL1_1_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_1_cons_buff_1, %BL1_1_2_cons_buff_0, %CL1_1_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_1_cons_buff_0, %BL1_1_2_cons_buff_0, %CL1_1_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_1_cons_buff_1, %BL1_1_2_cons_buff_0, %CL1_1_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_1_cons_buff_0, %BL1_1_2_cons_buff_0, %CL1_1_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_1_cons_buff_1, %BL1_1_2_cons_buff_0, %CL1_1_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_1_cons_buff_0, %BL1_1_2_cons_buff_0, %CL1_1_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_1_cons_buff_1, %BL1_1_2_cons_buff_0, %CL1_1_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_2_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_1_2_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_2_4 = aie.core(%tile_2_4) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_2_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_2_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_2_cons_buff_0, %BL1_2_2_cons_buff_0, %CL1_2_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_2_cons_buff_1, %BL1_2_2_cons_buff_0, %CL1_2_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_2_cons_buff_0, %BL1_2_2_cons_buff_0, %CL1_2_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_2_cons_buff_1, %BL1_2_2_cons_buff_0, %CL1_2_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_2_cons_buff_0, %BL1_2_2_cons_buff_0, %CL1_2_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_2_cons_buff_1, %BL1_2_2_cons_buff_0, %CL1_2_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_2_cons_buff_0, %BL1_2_2_cons_buff_0, %CL1_2_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_2_cons_buff_1, %BL1_2_2_cons_buff_0, %CL1_2_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_2_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_2_2_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_3_4 = aie.core(%tile_3_4) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_3_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_3_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_3_cons_buff_0, %BL1_3_2_cons_buff_0, %CL1_3_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_3_cons_buff_1, %BL1_3_2_cons_buff_0, %CL1_3_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_3_cons_buff_0, %BL1_3_2_cons_buff_0, %CL1_3_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_3_cons_buff_1, %BL1_3_2_cons_buff_0, %CL1_3_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_3_cons_buff_0, %BL1_3_2_cons_buff_0, %CL1_3_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_3_cons_buff_1, %BL1_3_2_cons_buff_0, %CL1_3_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_3_cons_buff_0, %BL1_3_2_cons_buff_0, %CL1_3_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_3_cons_buff_1, %BL1_3_2_cons_buff_0, %CL1_3_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_2_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_3_2_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_4_4 = aie.core(%tile_4_4) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_4_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_4_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_4_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_4_cons_buff_0, %BL1_4_2_cons_buff_0, %CL1_4_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_4_cons_buff_1, %BL1_4_2_cons_buff_0, %CL1_4_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_4_cons_buff_0, %BL1_4_2_cons_buff_0, %CL1_4_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_4_cons_buff_1, %BL1_4_2_cons_buff_0, %CL1_4_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_4_cons_buff_0, %BL1_4_2_cons_buff_0, %CL1_4_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_4_cons_buff_1, %BL1_4_2_cons_buff_0, %CL1_4_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_4_cons_buff_0, %BL1_4_2_cons_buff_0, %CL1_4_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_4_cons_buff_1, %BL1_4_2_cons_buff_0, %CL1_4_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_2_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_4_2_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_5_4 = aie.core(%tile_5_4) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_5_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_5_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_5_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_5_cons_buff_0, %BL1_5_2_cons_buff_0, %CL1_5_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_5_cons_buff_1, %BL1_5_2_cons_buff_0, %CL1_5_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_5_cons_buff_0, %BL1_5_2_cons_buff_0, %CL1_5_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_5_cons_buff_1, %BL1_5_2_cons_buff_0, %CL1_5_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_5_cons_buff_0, %BL1_5_2_cons_buff_0, %CL1_5_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_5_cons_buff_1, %BL1_5_2_cons_buff_0, %CL1_5_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_5_cons_buff_0, %BL1_5_2_cons_buff_0, %CL1_5_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_5_cons_buff_1, %BL1_5_2_cons_buff_0, %CL1_5_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_2_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_5_2_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_6_4 = aie.core(%tile_6_4) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_6_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_6_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_6_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_6_cons_buff_0, %BL1_6_2_cons_buff_0, %CL1_6_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_6_cons_buff_1, %BL1_6_2_cons_buff_0, %CL1_6_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_6_cons_buff_0, %BL1_6_2_cons_buff_0, %CL1_6_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_6_cons_buff_1, %BL1_6_2_cons_buff_0, %CL1_6_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_6_cons_buff_0, %BL1_6_2_cons_buff_0, %CL1_6_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_6_cons_buff_1, %BL1_6_2_cons_buff_0, %CL1_6_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_6_cons_buff_0, %BL1_6_2_cons_buff_0, %CL1_6_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_6_cons_buff_1, %BL1_6_2_cons_buff_0, %CL1_6_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_2_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_6_2_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_7_4 = aie.core(%tile_7_4) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_7_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_7_2_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_7_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_7_cons_buff_0, %BL1_7_2_cons_buff_0, %CL1_7_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_7_cons_buff_1, %BL1_7_2_cons_buff_0, %CL1_7_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_7_cons_buff_0, %BL1_7_2_cons_buff_0, %CL1_7_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_7_cons_buff_1, %BL1_7_2_cons_buff_0, %CL1_7_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_7_cons_buff_0, %BL1_7_2_cons_buff_0, %CL1_7_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_7_cons_buff_1, %BL1_7_2_cons_buff_0, %CL1_7_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_7_cons_buff_0, %BL1_7_2_cons_buff_0, %CL1_7_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_2_7_cons_buff_1, %BL1_7_2_cons_buff_0, %CL1_7_2_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_2_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_7_2_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_0_5 = aie.core(%tile_0_5) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_0_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_0_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_0_cons_buff_0, %BL1_0_3_cons_buff_0, %CL1_0_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_0_cons_buff_1, %BL1_0_3_cons_buff_0, %CL1_0_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_0_cons_buff_0, %BL1_0_3_cons_buff_0, %CL1_0_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_0_cons_buff_1, %BL1_0_3_cons_buff_0, %CL1_0_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_0_cons_buff_0, %BL1_0_3_cons_buff_0, %CL1_0_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_0_cons_buff_1, %BL1_0_3_cons_buff_0, %CL1_0_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_0_cons_buff_0, %BL1_0_3_cons_buff_0, %CL1_0_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_0_cons_buff_1, %BL1_0_3_cons_buff_0, %CL1_0_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_0_3_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_0_3_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_1_5 = aie.core(%tile_1_5) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_1_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_1_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_1_cons_buff_0, %BL1_1_3_cons_buff_0, %CL1_1_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_1_cons_buff_1, %BL1_1_3_cons_buff_0, %CL1_1_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_1_cons_buff_0, %BL1_1_3_cons_buff_0, %CL1_1_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_1_cons_buff_1, %BL1_1_3_cons_buff_0, %CL1_1_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_1_cons_buff_0, %BL1_1_3_cons_buff_0, %CL1_1_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_1_cons_buff_1, %BL1_1_3_cons_buff_0, %CL1_1_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_1_cons_buff_0, %BL1_1_3_cons_buff_0, %CL1_1_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_1_cons_buff_1, %BL1_1_3_cons_buff_0, %CL1_1_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_1_3_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_1_3_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_2_5 = aie.core(%tile_2_5) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_2_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_2_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_2_cons_buff_0, %BL1_2_3_cons_buff_0, %CL1_2_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_2_cons_buff_1, %BL1_2_3_cons_buff_0, %CL1_2_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_2_cons_buff_0, %BL1_2_3_cons_buff_0, %CL1_2_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_2_cons_buff_1, %BL1_2_3_cons_buff_0, %CL1_2_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_2_cons_buff_0, %BL1_2_3_cons_buff_0, %CL1_2_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_2_cons_buff_1, %BL1_2_3_cons_buff_0, %CL1_2_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_2_cons_buff_0, %BL1_2_3_cons_buff_0, %CL1_2_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_2_cons_buff_1, %BL1_2_3_cons_buff_0, %CL1_2_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_2_3_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_2_3_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_3_5 = aie.core(%tile_3_5) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_3_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_3_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_3_cons_buff_0, %BL1_3_3_cons_buff_0, %CL1_3_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_3_cons_buff_1, %BL1_3_3_cons_buff_0, %CL1_3_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_3_cons_buff_0, %BL1_3_3_cons_buff_0, %CL1_3_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_3_cons_buff_1, %BL1_3_3_cons_buff_0, %CL1_3_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_3_cons_buff_0, %BL1_3_3_cons_buff_0, %CL1_3_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_3_cons_buff_1, %BL1_3_3_cons_buff_0, %CL1_3_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_3_cons_buff_0, %BL1_3_3_cons_buff_0, %CL1_3_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_3_cons_buff_1, %BL1_3_3_cons_buff_0, %CL1_3_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_3_3_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_3_3_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_4_5 = aie.core(%tile_4_5) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_4_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_4_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_4_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_4_cons_buff_0, %BL1_4_3_cons_buff_0, %CL1_4_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_4_cons_buff_1, %BL1_4_3_cons_buff_0, %CL1_4_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_4_cons_buff_0, %BL1_4_3_cons_buff_0, %CL1_4_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_4_cons_buff_1, %BL1_4_3_cons_buff_0, %CL1_4_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_4_cons_buff_0, %BL1_4_3_cons_buff_0, %CL1_4_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_4_cons_buff_1, %BL1_4_3_cons_buff_0, %CL1_4_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_4_cons_buff_0, %BL1_4_3_cons_buff_0, %CL1_4_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_4_cons_buff_1, %BL1_4_3_cons_buff_0, %CL1_4_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_4_3_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_4_3_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_5_5 = aie.core(%tile_5_5) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_5_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_5_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_5_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_5_cons_buff_0, %BL1_5_3_cons_buff_0, %CL1_5_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_5_cons_buff_1, %BL1_5_3_cons_buff_0, %CL1_5_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_5_cons_buff_0, %BL1_5_3_cons_buff_0, %CL1_5_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_5_cons_buff_1, %BL1_5_3_cons_buff_0, %CL1_5_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_5_cons_buff_0, %BL1_5_3_cons_buff_0, %CL1_5_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_5_cons_buff_1, %BL1_5_3_cons_buff_0, %CL1_5_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_5_cons_buff_0, %BL1_5_3_cons_buff_0, %CL1_5_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_5_cons_buff_1, %BL1_5_3_cons_buff_0, %CL1_5_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_5_3_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_5_3_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_6_5 = aie.core(%tile_6_5) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_6_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_6_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_6_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_6_cons_buff_0, %BL1_6_3_cons_buff_0, %CL1_6_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_6_cons_buff_1, %BL1_6_3_cons_buff_0, %CL1_6_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_6_cons_buff_0, %BL1_6_3_cons_buff_0, %CL1_6_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_6_cons_buff_1, %BL1_6_3_cons_buff_0, %CL1_6_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_6_cons_buff_0, %BL1_6_3_cons_buff_0, %CL1_6_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_6_cons_buff_1, %BL1_6_3_cons_buff_0, %CL1_6_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_6_cons_buff_0, %BL1_6_3_cons_buff_0, %CL1_6_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_6_cons_buff_1, %BL1_6_3_cons_buff_0, %CL1_6_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_6_3_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_6_3_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    %core_7_5 = aie.core(%tile_7_5) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%CL1_7_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_kernel_bf16(%CL1_7_3_buff_0) : (memref<128x128xbf16>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%BL1_7_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_7_cons_buff_0, %BL1_7_3_cons_buff_0, %CL1_7_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_7_cons_buff_1, %BL1_7_3_cons_buff_0, %CL1_7_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_7_cons_buff_0, %BL1_7_3_cons_buff_0, %CL1_7_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_7_cons_buff_1, %BL1_7_3_cons_buff_0, %CL1_7_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%AL1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_7_cons_buff_0, %BL1_7_3_cons_buff_0, %CL1_7_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_7_cons_buff_1, %BL1_7_3_cons_buff_0, %CL1_7_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_7_cons_buff_0, %BL1_7_3_cons_buff_0, %CL1_7_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%AL1_3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_vectorized_bf16(%AL1_3_7_cons_buff_1, %BL1_7_3_cons_buff_0, %CL1_7_3_buff_0) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
      aie.use_lock(%AL1_3_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%BL1_7_3_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%CL1_7_3_cons_lock_0, Release, 1)
      %7 = arith.addi %2, %c1 : index
      cf.br ^bb3(%7 : index)
    ^bb8:  // pred: ^bb3
      %8 = arith.addi %0, %c1 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_bf16_direct.o"], stack_size = 2816 : i32}
    aie.runtime_sequence @seq(%arg0: memref<1048576xbf16>, %arg1: memref<1048576xbf16>, %arg2: memref<1048576xbf16>) {
      %0 = aiex.dma_configure_task_for @AL3_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @AL3_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 131072, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @AL3_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 262144, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @AL3_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 393216, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @BL3_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @BL3_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 128, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @BL3_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 256, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @BL3_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 384, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @BL3_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 512, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @BL3_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 640, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @BL3_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 768, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @BL3_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 896, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @CL3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 0, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @CL3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 128, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @CL3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 256, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @CL3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 384, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @CL3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 512, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @CL3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 640, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @CL3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 768, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @CL3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 896, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @AL3_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 524288, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @AL3_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 655360, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @AL3_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 786432, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @AL3_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 917504, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @BL3_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @BL3_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 128, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @BL3_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 256, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @BL3_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 384, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @BL3_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 512, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @BL3_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 640, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @BL3_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 768, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @BL3_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 896, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @CL3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524288, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @CL3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524416, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @CL3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524544, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @CL3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524672, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @CL3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524800, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @CL3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524928, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @CL3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 525056, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @CL3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 525184, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%39)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      aiex.dma_await_task(%32)
      aiex.dma_await_task(%33)
      aiex.dma_await_task(%34)
      aiex.dma_await_task(%35)
      aiex.dma_await_task(%36)
      aiex.dma_await_task(%37)
      aiex.dma_await_task(%38)
      aiex.dma_await_task(%39)
      aiex.dma_free_task(%20)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%23)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%25)
      aiex.dma_free_task(%26)
      aiex.dma_free_task(%27)
      aiex.dma_free_task(%28)
      aiex.dma_free_task(%29)
      aiex.dma_free_task(%30)
      aiex.dma_free_task(%31)
    }
    aie.shim_dma_allocation @AL3_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 4, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_0_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_0_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AL3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_0_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%AL3_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AL3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_0_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%AL3_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 5, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BL3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_0_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BL3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BL3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_0_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BL3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BL3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_0_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BL3_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BL3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_0_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL3_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 0, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%CL3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CL3_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%CL3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CL3_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 1, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%CL3_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%CL3_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%CL3_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%CL3_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 2, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%CL3_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%CL3_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%CL3_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%CL3_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%CL3_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%CL3_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%CL3_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%CL3_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 0, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%CL3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%CL3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%CL3_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%CL3_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%CL3_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%CL3_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%CL3_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%CL3_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%CL3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%CL3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%CL3_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%CL3_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%CL3_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%CL3_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%CL3_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_0_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%CL3_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_0_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_0_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_0_0_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_0_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_1_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_1_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_1_0_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_1_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_2_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_2_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_2_0_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_2_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_3_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_3_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_3_0_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_3_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_0_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_4_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_0_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_0_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_4_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_0_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_4_0_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_4_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_4_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_4_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_0_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_5_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_0_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_0_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_5_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_0_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_5_0_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_5_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_5_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_5_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_0_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_6_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_0_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_0_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_6_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_0_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_6_0_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_6_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_6_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_6_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_0_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_7_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_0_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_0_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_0_7_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_0_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_7_0_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_7_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_7_0_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_7_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @AL3_1_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 4, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_1_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_1_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AL3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_1_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%AL3_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AL3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_1_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%AL3_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 5, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BL3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_2_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BL3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BL3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_2_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BL3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BL3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_2_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BL3_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BL3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_2_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL3_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 0, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%CL3_2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CL3_2_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%CL3_2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CL3_2_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 1, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%CL3_2_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%CL3_2_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%CL3_2_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%CL3_2_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 2, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%CL3_2_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%CL3_2_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%CL3_2_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%CL3_2_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%CL3_2_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%CL3_2_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%CL3_2_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%CL3_2_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 0, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%CL3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%CL3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%CL3_2_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%CL3_2_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%CL3_2_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%CL3_2_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%CL3_2_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%CL3_2_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%CL3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%CL3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%CL3_2_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%CL3_2_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%CL3_2_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%CL3_2_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%CL3_2_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_2_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%CL3_2_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_0_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_0_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_0_1_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_0_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_0_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_0_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_1_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_1_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_1_1_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_1_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_1_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_1_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_2_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_2_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_2_1_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_2_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_2_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_2_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_3_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_3_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_3_1_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_3_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_4_3 = aie.mem(%tile_4_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_4_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_4_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_4_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_4_1_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_4_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_4_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_4_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_4_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_5_3 = aie.mem(%tile_5_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_5_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_5_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_5_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_5_1_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_5_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_5_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_5_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_5_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_6_3 = aie.mem(%tile_6_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_6_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_6_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_6_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_6_1_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_6_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_6_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_6_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_6_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_7_3 = aie.mem(%tile_7_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_7_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_1_7_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_7_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_7_1_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_7_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_7_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_7_1_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_7_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @AL3_2_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(S2MM, 4, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_2_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_2_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AL3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_2_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%AL3_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AL3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_2_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%AL3_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 5, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BL3_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_4_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BL3_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BL3_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_4_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BL3_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BL3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_4_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BL3_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BL3_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_4_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL3_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 0, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%CL3_4_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CL3_4_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%CL3_4_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CL3_4_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 1, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%CL3_4_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%CL3_4_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%CL3_4_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%CL3_4_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 2, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%CL3_4_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%CL3_4_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%CL3_4_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%CL3_4_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%CL3_4_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%CL3_4_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%CL3_4_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%CL3_4_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 0, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%CL3_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%CL3_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%CL3_4_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%CL3_4_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%CL3_4_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%CL3_4_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%CL3_4_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%CL3_4_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%CL3_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%CL3_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%CL3_4_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%CL3_4_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%CL3_4_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%CL3_4_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%CL3_4_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_4_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%CL3_4_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_0_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_0_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_0_2_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_0_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_0_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_0_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_1_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_1_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_1_2_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_1_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_1_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_1_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_2_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_2_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_2_2_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_2_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_2_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_2_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_3_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_3_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_3_2_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_3_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_4_4 = aie.mem(%tile_4_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_4_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_4_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_4_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_4_2_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_4_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_4_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_4_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_4_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_5_4 = aie.mem(%tile_5_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_5_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_5_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_5_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_5_2_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_5_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_5_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_5_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_5_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_6_4 = aie.mem(%tile_6_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_6_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_6_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_6_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_6_2_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_6_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_6_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_6_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_6_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_7_4 = aie.mem(%tile_7_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_7_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_2_7_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_7_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_7_2_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_7_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_7_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_7_2_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_7_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @AL3_3_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(S2MM, 4, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_3_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_3_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AL3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_3_cons_buff_0 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%AL3_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AL3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL3_3_cons_buff_1 : memref<128x512xbf16>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%AL3_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 5, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BL3_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_6_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BL3_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BL3_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_6_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BL3_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BL3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_6_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BL3_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BL3_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_6_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL3_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 0, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%CL3_6_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CL3_6_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%CL3_6_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CL3_6_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 1, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%CL3_6_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%CL3_6_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%CL3_6_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%CL3_6_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 2, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%CL3_6_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%CL3_6_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%CL3_6_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%CL3_6_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%CL3_6_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%CL3_6_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%CL3_6_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%CL3_6_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 0, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%CL3_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%CL3_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%CL3_6_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%CL3_6_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%CL3_6_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%CL3_6_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%CL3_6_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%CL3_6_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%CL3_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%CL3_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%CL3_6_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%CL3_6_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%CL3_6_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%CL3_6_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%CL3_6_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_6_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%CL3_6_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_0_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_0_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_0_3_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_0_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_0_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_0_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_1_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_1_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_1_3_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_1_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_1_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_1_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_2_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_2_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_2_3_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_2_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_2_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_2_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_3_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_3_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_3_3_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_3_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_4_5 = aie.mem(%tile_4_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_3_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_4_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_3_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_3_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_4_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_3_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_4_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_4_3_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_4_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_4_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_4_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_4_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_5_5 = aie.mem(%tile_5_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_3_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_5_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_3_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_3_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_5_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_3_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_5_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_5_3_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_5_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_5_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_5_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_5_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_6_5 = aie.mem(%tile_6_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_3_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_6_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_3_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_3_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_6_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_3_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_6_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_6_3_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_6_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_6_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_6_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_6_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    %mem_7_5 = aie.mem(%tile_7_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AL1_3_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_7_cons_buff_0 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AL1_3_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AL1_3_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AL1_3_7_cons_buff_1 : memref<32x64xbf16>, 0, 2048, [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AL1_3_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%BL1_7_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL1_7_3_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL1_7_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%CL1_7_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL1_7_3_buff_0 : memref<128x128xbf16>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%CL1_7_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @BL3_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @BL3_1_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%BL3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_1_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%BL3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%BL3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_1_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%BL3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BL3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_1_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BL3_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BL3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_1_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL3_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%CL3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%CL3_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%CL3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%CL3_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%CL3_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CL3_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%CL3_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CL3_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 3, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%CL3_1_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%CL3_1_cons_lock_2, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%CL3_1_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%CL3_1_cons_lock_2, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 4, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%CL3_1_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%CL3_1_cons_lock_3, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%CL3_1_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%CL3_1_cons_lock_3, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 1, ^bb19, ^bb27)
    ^bb19:  // 2 preds: ^bb18, ^bb26
      aie.use_lock(%CL3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%CL3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%CL3_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%CL3_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%CL3_1_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%CL3_1_prod_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%CL3_1_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%CL3_1_prod_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%CL3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%CL3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      aie.use_lock(%CL3_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%CL3_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%CL3_1_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%CL3_1_prod_lock_2, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%CL3_1_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_1_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 35 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%CL3_1_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb27:  // pred: ^bb18
      aie.end
    }
    aie.shim_dma_allocation @BL3_2_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @BL3_3_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%BL3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_3_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%BL3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%BL3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_3_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%BL3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BL3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_3_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BL3_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BL3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_3_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL3_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%CL3_3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%CL3_3_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%CL3_3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%CL3_3_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%CL3_3_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CL3_3_cons_lock_1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%CL3_3_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CL3_3_cons_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 3, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%CL3_3_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%CL3_3_cons_lock_2, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%CL3_3_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%CL3_3_cons_lock_2, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 4, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%CL3_3_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%CL3_3_cons_lock_3, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%CL3_3_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%CL3_3_cons_lock_3, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 1, ^bb19, ^bb27)
    ^bb19:  // 2 preds: ^bb18, ^bb26
      aie.use_lock(%CL3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%CL3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%CL3_3_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%CL3_3_prod_lock_1, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%CL3_3_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%CL3_3_prod_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%CL3_3_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%CL3_3_prod_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%CL3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%CL3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      aie.use_lock(%CL3_3_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%CL3_3_prod_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%CL3_3_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%CL3_3_prod_lock_2, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%CL3_3_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_3_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 35 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%CL3_3_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb27:  // pred: ^bb18
      aie.end
    }
    aie.shim_dma_allocation @BL3_4_shim_alloc(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @BL3_5_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%BL3_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_5_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%BL3_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%BL3_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_5_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%BL3_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BL3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_5_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BL3_5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BL3_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_5_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL3_5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%CL3_5_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%CL3_5_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%CL3_5_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%CL3_5_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%CL3_5_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CL3_5_cons_lock_1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%CL3_5_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CL3_5_cons_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 3, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%CL3_5_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%CL3_5_cons_lock_2, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%CL3_5_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%CL3_5_cons_lock_2, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 4, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%CL3_5_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%CL3_5_cons_lock_3, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%CL3_5_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%CL3_5_cons_lock_3, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 1, ^bb19, ^bb27)
    ^bb19:  // 2 preds: ^bb18, ^bb26
      aie.use_lock(%CL3_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%CL3_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%CL3_5_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%CL3_5_prod_lock_1, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%CL3_5_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%CL3_5_prod_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%CL3_5_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%CL3_5_prod_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%CL3_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%CL3_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      aie.use_lock(%CL3_5_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%CL3_5_prod_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%CL3_5_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%CL3_5_prod_lock_2, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%CL3_5_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_5_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 35 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%CL3_5_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb27:  // pred: ^bb18
      aie.end
    }
    aie.shim_dma_allocation @BL3_6_shim_alloc(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @BL3_7_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%BL3_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_7_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%BL3_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%BL3_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_7_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%BL3_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BL3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_7_cons_buff_0 : memref<64x128xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BL3_7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BL3_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BL3_7_cons_buff_1 : memref<64x128xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BL3_7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%CL3_7_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_0 : memref<512x128xbf16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%CL3_7_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%CL3_7_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_1 : memref<512x128xbf16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%CL3_7_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%CL3_7_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_0 : memref<512x128xbf16>, 16384, 16384) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CL3_7_cons_lock_1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%CL3_7_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_1 : memref<512x128xbf16>, 16384, 16384) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CL3_7_cons_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 3, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%CL3_7_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_0 : memref<512x128xbf16>, 32768, 16384) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%CL3_7_cons_lock_2, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%CL3_7_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_1 : memref<512x128xbf16>, 32768, 16384) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%CL3_7_cons_lock_2, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 4, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%CL3_7_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_0 : memref<512x128xbf16>, 49152, 16384) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%CL3_7_cons_lock_3, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%CL3_7_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_1 : memref<512x128xbf16>, 49152, 16384) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%CL3_7_cons_lock_3, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 1, ^bb19, ^bb27)
    ^bb19:  // 2 preds: ^bb18, ^bb26
      aie.use_lock(%CL3_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_0 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%CL3_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%CL3_7_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_0 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%CL3_7_prod_lock_1, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%CL3_7_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_0 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%CL3_7_prod_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%CL3_7_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_0 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%CL3_7_prod_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%CL3_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_1 : memref<512x128xbf16>, 0, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%CL3_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      aie.use_lock(%CL3_7_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_1 : memref<512x128xbf16>, 16384, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.use_lock(%CL3_7_prod_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%CL3_7_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_1 : memref<512x128xbf16>, 32768, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.use_lock(%CL3_7_prod_lock_2, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%CL3_7_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%CL3_7_buff_1 : memref<512x128xbf16>, 49152, 16384, [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>]) {bd_id = 35 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%CL3_7_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb27:  // pred: ^bb18
      aie.end
    }
    aie.shim_dma_allocation @CL3_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @CL3_1_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @CL3_2_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @CL3_3_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @CL3_4_shim_alloc(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @CL3_5_shim_alloc(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @CL3_6_shim_alloc(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @CL3_7_shim_alloc(%shim_noc_tile_7_0, S2MM, 0)
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
