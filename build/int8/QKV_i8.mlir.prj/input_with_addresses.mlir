module {
  aie.device(npu2) {
    func.func private @zero_i16(memref<128x128xi16>) attributes {link_with = "mm_128x64x128.o"}
    func.func private @matmul_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) attributes {link_with = "mm_128x64x128.o"}
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
    %C_L2L3_7_cons_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "C_L2L3_7_cons_prod_lock_0"}
    %C_L2L3_7_cons_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "C_L2L3_7_cons_cons_lock_0"}
    %C_L1L2_7_0_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 0 : i32, sym_name = "C_L1L2_7_0_cons_buff_0"} : memref<128x128xi16> 
    %C_L1L2_7_0_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 2) {init = 1 : i32, sym_name = "C_L1L2_7_0_cons_prod_lock_0"}
    %C_L1L2_7_0_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 3) {init = 0 : i32, sym_name = "C_L1L2_7_0_cons_cons_lock_0"}
    %C_L1L2_7_0_buff_0 = aie.buffer(%tile_7_2) {address = 3328 : i32, sym_name = "C_L1L2_7_0_buff_0"} : memref<128x128xi16> 
    %C_L1L2_7_0_prod_lock_0 = aie.lock(%tile_7_2, 4) {init = 1 : i32, sym_name = "C_L1L2_7_0_prod_lock_0"}
    %C_L1L2_7_0_cons_lock_0 = aie.lock(%tile_7_2, 5) {init = 0 : i32, sym_name = "C_L1L2_7_0_cons_lock_0"}
    %C_L2L3_6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "C_L2L3_6_cons_prod_lock_0"}
    %C_L2L3_6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "C_L2L3_6_cons_cons_lock_0"}
    %C_L1L2_6_0_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 0 : i32, sym_name = "C_L1L2_6_0_cons_buff_0"} : memref<128x128xi16> 
    %C_L1L2_6_0_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 2) {init = 1 : i32, sym_name = "C_L1L2_6_0_cons_prod_lock_0"}
    %C_L1L2_6_0_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 3) {init = 0 : i32, sym_name = "C_L1L2_6_0_cons_cons_lock_0"}
    %C_L1L2_6_0_buff_0 = aie.buffer(%tile_6_2) {address = 3328 : i32, sym_name = "C_L1L2_6_0_buff_0"} : memref<128x128xi16> 
    %C_L1L2_6_0_prod_lock_0 = aie.lock(%tile_6_2, 4) {init = 1 : i32, sym_name = "C_L1L2_6_0_prod_lock_0"}
    %C_L1L2_6_0_cons_lock_0 = aie.lock(%tile_6_2, 5) {init = 0 : i32, sym_name = "C_L1L2_6_0_cons_lock_0"}
    %C_L2L3_5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "C_L2L3_5_cons_prod_lock_0"}
    %C_L2L3_5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "C_L2L3_5_cons_cons_lock_0"}
    %C_L1L2_5_0_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 0 : i32, sym_name = "C_L1L2_5_0_cons_buff_0"} : memref<128x128xi16> 
    %C_L1L2_5_0_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 2) {init = 1 : i32, sym_name = "C_L1L2_5_0_cons_prod_lock_0"}
    %C_L1L2_5_0_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 3) {init = 0 : i32, sym_name = "C_L1L2_5_0_cons_cons_lock_0"}
    %C_L1L2_5_0_buff_0 = aie.buffer(%tile_5_2) {address = 3328 : i32, sym_name = "C_L1L2_5_0_buff_0"} : memref<128x128xi16> 
    %C_L1L2_5_0_prod_lock_0 = aie.lock(%tile_5_2, 4) {init = 1 : i32, sym_name = "C_L1L2_5_0_prod_lock_0"}
    %C_L1L2_5_0_cons_lock_0 = aie.lock(%tile_5_2, 5) {init = 0 : i32, sym_name = "C_L1L2_5_0_cons_lock_0"}
    %C_L2L3_4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "C_L2L3_4_cons_prod_lock_0"}
    %C_L2L3_4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "C_L2L3_4_cons_cons_lock_0"}
    %C_L1L2_4_0_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 0 : i32, sym_name = "C_L1L2_4_0_cons_buff_0"} : memref<128x128xi16> 
    %C_L1L2_4_0_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 2) {init = 1 : i32, sym_name = "C_L1L2_4_0_cons_prod_lock_0"}
    %C_L1L2_4_0_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 3) {init = 0 : i32, sym_name = "C_L1L2_4_0_cons_cons_lock_0"}
    %C_L1L2_4_0_buff_0 = aie.buffer(%tile_4_2) {address = 3328 : i32, sym_name = "C_L1L2_4_0_buff_0"} : memref<128x128xi16> 
    %C_L1L2_4_0_prod_lock_0 = aie.lock(%tile_4_2, 4) {init = 1 : i32, sym_name = "C_L1L2_4_0_prod_lock_0"}
    %C_L1L2_4_0_cons_lock_0 = aie.lock(%tile_4_2, 5) {init = 0 : i32, sym_name = "C_L1L2_4_0_cons_lock_0"}
    %C_L2L3_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "C_L2L3_3_cons_prod_lock_0"}
    %C_L2L3_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "C_L2L3_3_cons_cons_lock_0"}
    %C_L1L2_3_0_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, sym_name = "C_L1L2_3_0_cons_buff_0"} : memref<128x128xi16> 
    %C_L1L2_3_0_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 2) {init = 1 : i32, sym_name = "C_L1L2_3_0_cons_prod_lock_0"}
    %C_L1L2_3_0_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "C_L1L2_3_0_cons_cons_lock_0"}
    %C_L1L2_3_0_buff_0 = aie.buffer(%tile_3_2) {address = 3328 : i32, sym_name = "C_L1L2_3_0_buff_0"} : memref<128x128xi16> 
    %C_L1L2_3_0_prod_lock_0 = aie.lock(%tile_3_2, 4) {init = 1 : i32, sym_name = "C_L1L2_3_0_prod_lock_0"}
    %C_L1L2_3_0_cons_lock_0 = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "C_L1L2_3_0_cons_lock_0"}
    %C_L2L3_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "C_L2L3_2_cons_prod_lock_0"}
    %C_L2L3_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "C_L2L3_2_cons_cons_lock_0"}
    %C_L1L2_2_0_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "C_L1L2_2_0_cons_buff_0"} : memref<128x128xi16> 
    %C_L1L2_2_0_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 2) {init = 1 : i32, sym_name = "C_L1L2_2_0_cons_prod_lock_0"}
    %C_L1L2_2_0_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "C_L1L2_2_0_cons_cons_lock_0"}
    %C_L1L2_2_0_buff_0 = aie.buffer(%tile_2_2) {address = 3328 : i32, sym_name = "C_L1L2_2_0_buff_0"} : memref<128x128xi16> 
    %C_L1L2_2_0_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 1 : i32, sym_name = "C_L1L2_2_0_prod_lock_0"}
    %C_L1L2_2_0_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "C_L1L2_2_0_cons_lock_0"}
    %C_L2L3_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "C_L2L3_1_cons_prod_lock_0"}
    %C_L2L3_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "C_L2L3_1_cons_cons_lock_0"}
    %C_L1L2_1_0_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "C_L1L2_1_0_cons_buff_0"} : memref<128x128xi16> 
    %C_L1L2_1_0_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 2) {init = 1 : i32, sym_name = "C_L1L2_1_0_cons_prod_lock_0"}
    %C_L1L2_1_0_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "C_L1L2_1_0_cons_cons_lock_0"}
    %C_L1L2_1_0_buff_0 = aie.buffer(%tile_1_2) {address = 3328 : i32, sym_name = "C_L1L2_1_0_buff_0"} : memref<128x128xi16> 
    %C_L1L2_1_0_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 1 : i32, sym_name = "C_L1L2_1_0_prod_lock_0"}
    %C_L1L2_1_0_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "C_L1L2_1_0_cons_lock_0"}
    %C_L2L3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L2L3_0_cons_prod_lock_0"}
    %C_L2L3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L2L3_0_cons_cons_lock_0"}
    %C_L1L2_0_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, sym_name = "C_L1L2_0_0_cons_buff_0"} : memref<128x128xi16> 
    %C_L1L2_0_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32, sym_name = "C_L1L2_0_0_cons_prod_lock_0"}
    %C_L1L2_0_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "C_L1L2_0_0_cons_cons_lock_0"}
    %C_L1L2_0_0_buff_0 = aie.buffer(%tile_0_2) {address = 3328 : i32, sym_name = "C_L1L2_0_0_buff_0"} : memref<128x128xi16> 
    %C_L1L2_0_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "C_L1L2_0_0_prod_lock_0"}
    %C_L1L2_0_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_L1L2_0_0_cons_lock_0"}
    %B_L2L1_7_cons_buff_0 = aie.buffer(%tile_7_2) {address = 36096 : i32, sym_name = "B_L2L1_7_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_7_cons_buff_1 = aie.buffer(%tile_7_2) {address = 44288 : i32, sym_name = "B_L2L1_7_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_7_cons_prod_lock_0 = aie.lock(%tile_7_2, 2) {init = 2 : i32, sym_name = "B_L2L1_7_cons_prod_lock_0"}
    %B_L2L1_7_cons_cons_lock_0 = aie.lock(%tile_7_2, 3) {init = 0 : i32, sym_name = "B_L2L1_7_cons_cons_lock_0"}
    %B_L3L2_7_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 32768 : i32, sym_name = "B_L3L2_7_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_7_cons_buff_1 = aie.buffer(%mem_tile_7_1) {address = 40960 : i32, sym_name = "B_L3L2_7_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_7_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 0) {init = 2 : i32, sym_name = "B_L3L2_7_cons_prod_lock_0"}
    %B_L3L2_7_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 1) {init = 0 : i32, sym_name = "B_L3L2_7_cons_cons_lock_0"}
    %B_L3L2_7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "B_L3L2_7_prod_lock_0"}
    %B_L3L2_7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "B_L3L2_7_cons_lock_0"}
    %B_L2L1_6_cons_buff_0 = aie.buffer(%tile_6_2) {address = 36096 : i32, sym_name = "B_L2L1_6_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_6_cons_buff_1 = aie.buffer(%tile_6_2) {address = 44288 : i32, sym_name = "B_L2L1_6_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_6_cons_prod_lock_0 = aie.lock(%tile_6_2, 2) {init = 2 : i32, sym_name = "B_L2L1_6_cons_prod_lock_0"}
    %B_L2L1_6_cons_cons_lock_0 = aie.lock(%tile_6_2, 3) {init = 0 : i32, sym_name = "B_L2L1_6_cons_cons_lock_0"}
    %B_L3L2_6_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 32768 : i32, sym_name = "B_L3L2_6_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_6_cons_buff_1 = aie.buffer(%mem_tile_6_1) {address = 40960 : i32, sym_name = "B_L3L2_6_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_6_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 0) {init = 2 : i32, sym_name = "B_L3L2_6_cons_prod_lock_0"}
    %B_L3L2_6_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 1) {init = 0 : i32, sym_name = "B_L3L2_6_cons_cons_lock_0"}
    %B_L3L2_6_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "B_L3L2_6_prod_lock_0"}
    %B_L3L2_6_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "B_L3L2_6_cons_lock_0"}
    %B_L2L1_5_cons_buff_0 = aie.buffer(%tile_5_2) {address = 36096 : i32, sym_name = "B_L2L1_5_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_5_cons_buff_1 = aie.buffer(%tile_5_2) {address = 44288 : i32, sym_name = "B_L2L1_5_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_5_cons_prod_lock_0 = aie.lock(%tile_5_2, 2) {init = 2 : i32, sym_name = "B_L2L1_5_cons_prod_lock_0"}
    %B_L2L1_5_cons_cons_lock_0 = aie.lock(%tile_5_2, 3) {init = 0 : i32, sym_name = "B_L2L1_5_cons_cons_lock_0"}
    %B_L3L2_5_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 32768 : i32, sym_name = "B_L3L2_5_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_5_cons_buff_1 = aie.buffer(%mem_tile_5_1) {address = 40960 : i32, sym_name = "B_L3L2_5_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_5_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 0) {init = 2 : i32, sym_name = "B_L3L2_5_cons_prod_lock_0"}
    %B_L3L2_5_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 1) {init = 0 : i32, sym_name = "B_L3L2_5_cons_cons_lock_0"}
    %B_L3L2_5_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "B_L3L2_5_prod_lock_0"}
    %B_L3L2_5_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "B_L3L2_5_cons_lock_0"}
    %B_L2L1_4_cons_buff_0 = aie.buffer(%tile_4_2) {address = 36096 : i32, sym_name = "B_L2L1_4_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_4_cons_buff_1 = aie.buffer(%tile_4_2) {address = 44288 : i32, sym_name = "B_L2L1_4_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_4_cons_prod_lock_0 = aie.lock(%tile_4_2, 2) {init = 2 : i32, sym_name = "B_L2L1_4_cons_prod_lock_0"}
    %B_L2L1_4_cons_cons_lock_0 = aie.lock(%tile_4_2, 3) {init = 0 : i32, sym_name = "B_L2L1_4_cons_cons_lock_0"}
    %B_L3L2_4_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 32768 : i32, sym_name = "B_L3L2_4_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_4_cons_buff_1 = aie.buffer(%mem_tile_4_1) {address = 40960 : i32, sym_name = "B_L3L2_4_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_4_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 0) {init = 2 : i32, sym_name = "B_L3L2_4_cons_prod_lock_0"}
    %B_L3L2_4_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 1) {init = 0 : i32, sym_name = "B_L3L2_4_cons_cons_lock_0"}
    %B_L3L2_4_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "B_L3L2_4_prod_lock_0"}
    %B_L3L2_4_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "B_L3L2_4_cons_lock_0"}
    %B_L2L1_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 36096 : i32, sym_name = "B_L2L1_3_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 44288 : i32, sym_name = "B_L2L1_3_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "B_L2L1_3_cons_prod_lock_0"}
    %B_L2L1_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "B_L2L1_3_cons_cons_lock_0"}
    %B_L3L2_3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 32768 : i32, sym_name = "B_L3L2_3_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 40960 : i32, sym_name = "B_L3L2_3_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "B_L3L2_3_cons_prod_lock_0"}
    %B_L3L2_3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "B_L3L2_3_cons_cons_lock_0"}
    %B_L3L2_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "B_L3L2_3_prod_lock_0"}
    %B_L3L2_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "B_L3L2_3_cons_lock_0"}
    %B_L2L1_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 36096 : i32, sym_name = "B_L2L1_2_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 44288 : i32, sym_name = "B_L2L1_2_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "B_L2L1_2_cons_prod_lock_0"}
    %B_L2L1_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "B_L2L1_2_cons_cons_lock_0"}
    %B_L3L2_2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 32768 : i32, sym_name = "B_L3L2_2_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 40960 : i32, sym_name = "B_L3L2_2_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "B_L3L2_2_cons_prod_lock_0"}
    %B_L3L2_2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "B_L3L2_2_cons_cons_lock_0"}
    %B_L3L2_2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "B_L3L2_2_prod_lock_0"}
    %B_L3L2_2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "B_L3L2_2_cons_lock_0"}
    %B_L2L1_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 36096 : i32, sym_name = "B_L2L1_1_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 44288 : i32, sym_name = "B_L2L1_1_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "B_L2L1_1_cons_prod_lock_0"}
    %B_L2L1_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "B_L2L1_1_cons_cons_lock_0"}
    %B_L3L2_1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 32768 : i32, sym_name = "B_L3L2_1_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_1_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 40960 : i32, sym_name = "B_L3L2_1_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "B_L3L2_1_cons_prod_lock_0"}
    %B_L3L2_1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "B_L3L2_1_cons_cons_lock_0"}
    %B_L3L2_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "B_L3L2_1_prod_lock_0"}
    %B_L3L2_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "B_L3L2_1_cons_lock_0"}
    %B_L2L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 36096 : i32, sym_name = "B_L2L1_0_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 44288 : i32, sym_name = "B_L2L1_0_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "B_L2L1_0_cons_prod_lock_0"}
    %B_L2L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_L2L1_0_cons_cons_lock_0"}
    %B_L3L2_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 163840 : i32, sym_name = "B_L3L2_0_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 172032 : i32, sym_name = "B_L3L2_0_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "B_L3L2_0_cons_prod_lock_0"}
    %B_L3L2_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "B_L3L2_0_cons_cons_lock_0"}
    %B_L3L2_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_L3L2_0_prod_lock_0"}
    %B_L3L2_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_L3L2_0_cons_lock_0"}
    %A_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 52480 : i32, sym_name = "A_L2L1_0_0_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 54528 : i32, sym_name = "A_L2L1_0_0_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_0_cons_prod_lock_0"}
    %A_L2L1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_0_cons_cons_lock_0"}
    %A_L2L1_0_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 52480 : i32, sym_name = "A_L2L1_0_1_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 54528 : i32, sym_name = "A_L2L1_0_1_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_1_cons_prod_lock_0"}
    %A_L2L1_0_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_1_cons_cons_lock_0"}
    %A_L2L1_0_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 52480 : i32, sym_name = "A_L2L1_0_2_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 54528 : i32, sym_name = "A_L2L1_0_2_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_2_cons_prod_lock_0"}
    %A_L2L1_0_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_2_cons_cons_lock_0"}
    %A_L2L1_0_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 52480 : i32, sym_name = "A_L2L1_0_3_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 54528 : i32, sym_name = "A_L2L1_0_3_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_3_cons_prod_lock_0"}
    %A_L2L1_0_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_3_cons_cons_lock_0"}
    %A_L2L1_0_4_cons_buff_0 = aie.buffer(%tile_4_2) {address = 52480 : i32, sym_name = "A_L2L1_0_4_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_4_cons_buff_1 = aie.buffer(%tile_4_2) {address = 54528 : i32, sym_name = "A_L2L1_0_4_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_4_cons_prod_lock_0 = aie.lock(%tile_4_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_4_cons_prod_lock_0"}
    %A_L2L1_0_4_cons_cons_lock_0 = aie.lock(%tile_4_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_4_cons_cons_lock_0"}
    %A_L2L1_0_5_cons_buff_0 = aie.buffer(%tile_5_2) {address = 52480 : i32, sym_name = "A_L2L1_0_5_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_5_cons_buff_1 = aie.buffer(%tile_5_2) {address = 54528 : i32, sym_name = "A_L2L1_0_5_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_5_cons_prod_lock_0 = aie.lock(%tile_5_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_5_cons_prod_lock_0"}
    %A_L2L1_0_5_cons_cons_lock_0 = aie.lock(%tile_5_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_5_cons_cons_lock_0"}
    %A_L2L1_0_6_cons_buff_0 = aie.buffer(%tile_6_2) {address = 52480 : i32, sym_name = "A_L2L1_0_6_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_6_cons_buff_1 = aie.buffer(%tile_6_2) {address = 54528 : i32, sym_name = "A_L2L1_0_6_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_6_cons_prod_lock_0 = aie.lock(%tile_6_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_6_cons_prod_lock_0"}
    %A_L2L1_0_6_cons_cons_lock_0 = aie.lock(%tile_6_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_6_cons_cons_lock_0"}
    %A_L2L1_0_7_cons_buff_0 = aie.buffer(%tile_7_2) {address = 52480 : i32, sym_name = "A_L2L1_0_7_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_7_cons_buff_1 = aie.buffer(%tile_7_2) {address = 54528 : i32, sym_name = "A_L2L1_0_7_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_7_cons_prod_lock_0 = aie.lock(%tile_7_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_7_cons_prod_lock_0"}
    %A_L2L1_0_7_cons_cons_lock_0 = aie.lock(%tile_7_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_7_cons_cons_lock_0"}
    %A_L3L2_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "A_L3L2_0_cons_buff_0"} : memref<128x512xi8> 
    %A_L3L2_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, sym_name = "A_L3L2_0_cons_buff_1"} : memref<128x512xi8> 
    %A_L3L2_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "A_L3L2_0_cons_prod_lock_0"}
    %A_L3L2_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "A_L3L2_0_cons_cons_lock_0"}
    %A_L3L2_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L2_0_prod_lock_0"}
    %A_L3L2_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L2_0_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_7_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_6_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_5_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_4_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_2, DMA : 1)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_2, DMA : 1)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_2, DMA : 1)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_4_2, DMA : 1)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_5_2, DMA : 1)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_6_2, DMA : 1)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_7_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%tile_4_2, DMA : 0, %mem_tile_4_1, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 1, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%tile_5_2, DMA : 0, %mem_tile_5_1, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 1, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%tile_6_2, DMA : 0, %mem_tile_6_1, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%tile_7_2, DMA : 0, %mem_tile_7_1, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %shim_noc_tile_7_0, DMA : 0)
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 56576 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb94
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb95
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
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
      cf.br ^bb12(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb12(%11: memref<32x64xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb16(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb16(%20: memref<32x64xi8>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_i8_i16(%20, %7, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb20(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb20(%29: memref<32x64xi8>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_i8_i16(%29, %7, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb24(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb24(%38: memref<32x64xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_i8_i16(%38, %7, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
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
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb31(%59: memref<64x128xi8>):  // 3 preds: ^bb28, ^bb29, ^bb30
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
      cf.br ^bb35(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb35(%63: memref<32x64xi8>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_i8_i16(%63, %59, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb39(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb39(%72: memref<32x64xi8>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_i8_i16(%72, %59, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb43(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb43(%81: memref<32x64xi8>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_i8_i16(%81, %59, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb47(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb47(%90: memref<32x64xi8>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_i8_i16(%90, %59, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
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
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb49(%c0 : index)
    ^bb49(%106: index):  // 2 preds: ^bb48, ^bb70
      %107 = arith.cmpi slt, %106, %c16 : index
      cf.cond_br %107, ^bb50, ^bb71
    ^bb50:  // pred: ^bb49
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb53,
        0: ^bb51,
        1: ^bb52
      ]
    ^bb51:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb52:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb53:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb54(%111: memref<64x128xi8>):  // 3 preds: ^bb51, ^bb52, ^bb53
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %112 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %113 = arith.index_cast %112 : i32 to index
      %114 = arith.index_cast %113 : index to i32
      cf.switch %114 : i32, [
        default: ^bb57,
        0: ^bb55,
        1: ^bb56
      ]
    ^bb55:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb56:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb57:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb58(%115: memref<32x64xi8>):  // 3 preds: ^bb55, ^bb56, ^bb57
      func.call @matmul_i8_i16(%115, %111, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %116 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %117 = arith.addi %116, %c1_i32 : i32
      %118 = arith.cmpi sge, %117, %c2_i32 : i32
      %119 = arith.subi %117, %c2_i32 : i32
      %120 = arith.select %118, %119, %117 : i32
      memref.store %120, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %121 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %122 = arith.index_cast %121 : i32 to index
      %123 = arith.index_cast %122 : index to i32
      cf.switch %123 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb62(%124: memref<32x64xi8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @matmul_i8_i16(%124, %111, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %125 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %126 = arith.addi %125, %c1_i32 : i32
      %127 = arith.cmpi sge, %126, %c2_i32 : i32
      %128 = arith.subi %126, %c2_i32 : i32
      %129 = arith.select %127, %128, %126 : i32
      memref.store %129, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %130 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %131 = arith.index_cast %130 : i32 to index
      %132 = arith.index_cast %131 : index to i32
      cf.switch %132 : i32, [
        default: ^bb65,
        0: ^bb63,
        1: ^bb64
      ]
    ^bb63:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb64:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb66(%133: memref<32x64xi8>):  // 3 preds: ^bb63, ^bb64, ^bb65
      func.call @matmul_i8_i16(%133, %111, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %134 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %135 = arith.addi %134, %c1_i32 : i32
      %136 = arith.cmpi sge, %135, %c2_i32 : i32
      %137 = arith.subi %135, %c2_i32 : i32
      %138 = arith.select %136, %137, %135 : i32
      memref.store %138, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %139 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %140 = arith.index_cast %139 : i32 to index
      %141 = arith.index_cast %140 : index to i32
      cf.switch %141 : i32, [
        default: ^bb69,
        0: ^bb67,
        1: ^bb68
      ]
    ^bb67:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb68:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb69:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb70(%142: memref<32x64xi8>):  // 3 preds: ^bb67, ^bb68, ^bb69
      func.call @matmul_i8_i16(%142, %111, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %143 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %144 = arith.addi %143, %c1_i32 : i32
      %145 = arith.cmpi sge, %144, %c2_i32 : i32
      %146 = arith.subi %144, %c2_i32 : i32
      %147 = arith.select %145, %146, %144 : i32
      memref.store %147, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %148 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c2_i32 : i32
      %151 = arith.subi %149, %c2_i32 : i32
      %152 = arith.select %150, %151, %149 : i32
      memref.store %152, %_anonymous0[%c1] : memref<3xi32>
      %153 = arith.addi %106, %c1 : index
      cf.br ^bb49(%153 : index)
    ^bb71:  // pred: ^bb49
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %154 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %155 = arith.addi %154, %c1_i32 : i32
      %156 = arith.cmpi sge, %155, %c1_i32 : i32
      %157 = arith.select %156, %154, %155 : i32
      memref.store %157, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb72(%c0 : index)
    ^bb72(%158: index):  // 2 preds: ^bb71, ^bb93
      %159 = arith.cmpi slt, %158, %c16 : index
      cf.cond_br %159, ^bb73, ^bb94
    ^bb73:  // pred: ^bb72
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %160 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %161 = arith.index_cast %160 : i32 to index
      %162 = arith.index_cast %161 : index to i32
      cf.switch %162 : i32, [
        default: ^bb76,
        0: ^bb74,
        1: ^bb75
      ]
    ^bb74:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb75:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb76:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb77(%163: memref<64x128xi8>):  // 3 preds: ^bb74, ^bb75, ^bb76
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %164 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %165 = arith.index_cast %164 : i32 to index
      %166 = arith.index_cast %165 : index to i32
      cf.switch %166 : i32, [
        default: ^bb80,
        0: ^bb78,
        1: ^bb79
      ]
    ^bb78:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb79:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb80:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb81(%167: memref<32x64xi8>):  // 3 preds: ^bb78, ^bb79, ^bb80
      func.call @matmul_i8_i16(%167, %163, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %168 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %169 = arith.addi %168, %c1_i32 : i32
      %170 = arith.cmpi sge, %169, %c2_i32 : i32
      %171 = arith.subi %169, %c2_i32 : i32
      %172 = arith.select %170, %171, %169 : i32
      memref.store %172, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %173 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %174 = arith.index_cast %173 : i32 to index
      %175 = arith.index_cast %174 : index to i32
      cf.switch %175 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb85(%176: memref<32x64xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      func.call @matmul_i8_i16(%176, %163, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %177 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %178 = arith.addi %177, %c1_i32 : i32
      %179 = arith.cmpi sge, %178, %c2_i32 : i32
      %180 = arith.subi %178, %c2_i32 : i32
      %181 = arith.select %179, %180, %178 : i32
      memref.store %181, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %182 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %183 = arith.index_cast %182 : i32 to index
      %184 = arith.index_cast %183 : index to i32
      cf.switch %184 : i32, [
        default: ^bb88,
        0: ^bb86,
        1: ^bb87
      ]
    ^bb86:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb87:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb88:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb89(%185: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%185, %163, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %186 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %187 = arith.addi %186, %c1_i32 : i32
      %188 = arith.cmpi sge, %187, %c2_i32 : i32
      %189 = arith.subi %187, %c2_i32 : i32
      %190 = arith.select %188, %189, %187 : i32
      memref.store %190, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %191 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %192 = arith.index_cast %191 : i32 to index
      %193 = arith.index_cast %192 : index to i32
      cf.switch %193 : i32, [
        default: ^bb92,
        0: ^bb90,
        1: ^bb91
      ]
    ^bb90:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb91:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb92:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb93(%194: memref<32x64xi8>):  // 3 preds: ^bb90, ^bb91, ^bb92
      func.call @matmul_i8_i16(%194, %163, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %195 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %196 = arith.addi %195, %c1_i32 : i32
      %197 = arith.cmpi sge, %196, %c2_i32 : i32
      %198 = arith.subi %196, %c2_i32 : i32
      %199 = arith.select %197, %198, %196 : i32
      memref.store %199, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %200 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %201 = arith.addi %200, %c1_i32 : i32
      %202 = arith.cmpi sge, %201, %c2_i32 : i32
      %203 = arith.subi %201, %c2_i32 : i32
      %204 = arith.select %202, %203, %201 : i32
      memref.store %204, %_anonymous0[%c1] : memref<3xi32>
      %205 = arith.addi %158, %c1 : index
      cf.br ^bb72(%205 : index)
    ^bb94:  // pred: ^bb72
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %206 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %207 = arith.addi %206, %c1_i32 : i32
      %208 = arith.cmpi sge, %207, %c1_i32 : i32
      %209 = arith.select %208, %206, %207 : i32
      memref.store %209, %_anonymous0[%c0] : memref<3xi32>
      %210 = arith.addi %0, %c1 : index
      cf.br ^bb1(%210 : index)
    ^bb95:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous1 = aie.buffer(%tile_1_2) {address = 56576 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb94
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb95
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
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
      cf.br ^bb12(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb12(%11: memref<32x64xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb16(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb16(%20: memref<32x64xi8>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_i8_i16(%20, %7, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb20(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb20(%29: memref<32x64xi8>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_i8_i16(%29, %7, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb24(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb24(%38: memref<32x64xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_i8_i16(%38, %7, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
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
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb31(%59: memref<64x128xi8>):  // 3 preds: ^bb28, ^bb29, ^bb30
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
      cf.br ^bb35(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb35(%63: memref<32x64xi8>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_i8_i16(%63, %59, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb39(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb39(%72: memref<32x64xi8>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_i8_i16(%72, %59, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb43(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb43(%81: memref<32x64xi8>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_i8_i16(%81, %59, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb47(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb47(%90: memref<32x64xi8>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_i8_i16(%90, %59, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
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
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb49(%c0 : index)
    ^bb49(%106: index):  // 2 preds: ^bb48, ^bb70
      %107 = arith.cmpi slt, %106, %c16 : index
      cf.cond_br %107, ^bb50, ^bb71
    ^bb50:  // pred: ^bb49
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb53,
        0: ^bb51,
        1: ^bb52
      ]
    ^bb51:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb52:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb53:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb54(%111: memref<64x128xi8>):  // 3 preds: ^bb51, ^bb52, ^bb53
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %112 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %113 = arith.index_cast %112 : i32 to index
      %114 = arith.index_cast %113 : index to i32
      cf.switch %114 : i32, [
        default: ^bb57,
        0: ^bb55,
        1: ^bb56
      ]
    ^bb55:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb56:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb57:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb58(%115: memref<32x64xi8>):  // 3 preds: ^bb55, ^bb56, ^bb57
      func.call @matmul_i8_i16(%115, %111, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %116 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %117 = arith.addi %116, %c1_i32 : i32
      %118 = arith.cmpi sge, %117, %c2_i32 : i32
      %119 = arith.subi %117, %c2_i32 : i32
      %120 = arith.select %118, %119, %117 : i32
      memref.store %120, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %121 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %122 = arith.index_cast %121 : i32 to index
      %123 = arith.index_cast %122 : index to i32
      cf.switch %123 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb62(%124: memref<32x64xi8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @matmul_i8_i16(%124, %111, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %125 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %126 = arith.addi %125, %c1_i32 : i32
      %127 = arith.cmpi sge, %126, %c2_i32 : i32
      %128 = arith.subi %126, %c2_i32 : i32
      %129 = arith.select %127, %128, %126 : i32
      memref.store %129, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %130 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %131 = arith.index_cast %130 : i32 to index
      %132 = arith.index_cast %131 : index to i32
      cf.switch %132 : i32, [
        default: ^bb65,
        0: ^bb63,
        1: ^bb64
      ]
    ^bb63:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb64:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb66(%133: memref<32x64xi8>):  // 3 preds: ^bb63, ^bb64, ^bb65
      func.call @matmul_i8_i16(%133, %111, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %134 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %135 = arith.addi %134, %c1_i32 : i32
      %136 = arith.cmpi sge, %135, %c2_i32 : i32
      %137 = arith.subi %135, %c2_i32 : i32
      %138 = arith.select %136, %137, %135 : i32
      memref.store %138, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %139 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %140 = arith.index_cast %139 : i32 to index
      %141 = arith.index_cast %140 : index to i32
      cf.switch %141 : i32, [
        default: ^bb69,
        0: ^bb67,
        1: ^bb68
      ]
    ^bb67:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb68:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb69:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb70(%142: memref<32x64xi8>):  // 3 preds: ^bb67, ^bb68, ^bb69
      func.call @matmul_i8_i16(%142, %111, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %143 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %144 = arith.addi %143, %c1_i32 : i32
      %145 = arith.cmpi sge, %144, %c2_i32 : i32
      %146 = arith.subi %144, %c2_i32 : i32
      %147 = arith.select %145, %146, %144 : i32
      memref.store %147, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %148 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c2_i32 : i32
      %151 = arith.subi %149, %c2_i32 : i32
      %152 = arith.select %150, %151, %149 : i32
      memref.store %152, %_anonymous1[%c1] : memref<3xi32>
      %153 = arith.addi %106, %c1 : index
      cf.br ^bb49(%153 : index)
    ^bb71:  // pred: ^bb49
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %154 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %155 = arith.addi %154, %c1_i32 : i32
      %156 = arith.cmpi sge, %155, %c1_i32 : i32
      %157 = arith.select %156, %154, %155 : i32
      memref.store %157, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb72(%c0 : index)
    ^bb72(%158: index):  // 2 preds: ^bb71, ^bb93
      %159 = arith.cmpi slt, %158, %c16 : index
      cf.cond_br %159, ^bb73, ^bb94
    ^bb73:  // pred: ^bb72
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %160 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %161 = arith.index_cast %160 : i32 to index
      %162 = arith.index_cast %161 : index to i32
      cf.switch %162 : i32, [
        default: ^bb76,
        0: ^bb74,
        1: ^bb75
      ]
    ^bb74:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb75:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb76:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb77(%163: memref<64x128xi8>):  // 3 preds: ^bb74, ^bb75, ^bb76
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %164 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %165 = arith.index_cast %164 : i32 to index
      %166 = arith.index_cast %165 : index to i32
      cf.switch %166 : i32, [
        default: ^bb80,
        0: ^bb78,
        1: ^bb79
      ]
    ^bb78:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb79:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb80:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb81(%167: memref<32x64xi8>):  // 3 preds: ^bb78, ^bb79, ^bb80
      func.call @matmul_i8_i16(%167, %163, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %168 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %169 = arith.addi %168, %c1_i32 : i32
      %170 = arith.cmpi sge, %169, %c2_i32 : i32
      %171 = arith.subi %169, %c2_i32 : i32
      %172 = arith.select %170, %171, %169 : i32
      memref.store %172, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %173 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %174 = arith.index_cast %173 : i32 to index
      %175 = arith.index_cast %174 : index to i32
      cf.switch %175 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb85(%176: memref<32x64xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      func.call @matmul_i8_i16(%176, %163, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %177 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %178 = arith.addi %177, %c1_i32 : i32
      %179 = arith.cmpi sge, %178, %c2_i32 : i32
      %180 = arith.subi %178, %c2_i32 : i32
      %181 = arith.select %179, %180, %178 : i32
      memref.store %181, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %182 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %183 = arith.index_cast %182 : i32 to index
      %184 = arith.index_cast %183 : index to i32
      cf.switch %184 : i32, [
        default: ^bb88,
        0: ^bb86,
        1: ^bb87
      ]
    ^bb86:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb87:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb88:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb89(%185: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%185, %163, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %186 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %187 = arith.addi %186, %c1_i32 : i32
      %188 = arith.cmpi sge, %187, %c2_i32 : i32
      %189 = arith.subi %187, %c2_i32 : i32
      %190 = arith.select %188, %189, %187 : i32
      memref.store %190, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %191 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %192 = arith.index_cast %191 : i32 to index
      %193 = arith.index_cast %192 : index to i32
      cf.switch %193 : i32, [
        default: ^bb92,
        0: ^bb90,
        1: ^bb91
      ]
    ^bb90:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb91:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb92:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb93(%194: memref<32x64xi8>):  // 3 preds: ^bb90, ^bb91, ^bb92
      func.call @matmul_i8_i16(%194, %163, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %195 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %196 = arith.addi %195, %c1_i32 : i32
      %197 = arith.cmpi sge, %196, %c2_i32 : i32
      %198 = arith.subi %196, %c2_i32 : i32
      %199 = arith.select %197, %198, %196 : i32
      memref.store %199, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %200 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %201 = arith.addi %200, %c1_i32 : i32
      %202 = arith.cmpi sge, %201, %c2_i32 : i32
      %203 = arith.subi %201, %c2_i32 : i32
      %204 = arith.select %202, %203, %201 : i32
      memref.store %204, %_anonymous1[%c1] : memref<3xi32>
      %205 = arith.addi %158, %c1 : index
      cf.br ^bb72(%205 : index)
    ^bb94:  // pred: ^bb72
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %206 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %207 = arith.addi %206, %c1_i32 : i32
      %208 = arith.cmpi sge, %207, %c1_i32 : i32
      %209 = arith.select %208, %206, %207 : i32
      memref.store %209, %_anonymous1[%c0] : memref<3xi32>
      %210 = arith.addi %0, %c1 : index
      cf.br ^bb1(%210 : index)
    ^bb95:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous2 = aie.buffer(%tile_2_2) {address = 56576 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb94
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb95
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
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
      cf.br ^bb12(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb12(%11: memref<32x64xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb16(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb16(%20: memref<32x64xi8>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_i8_i16(%20, %7, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb20(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb20(%29: memref<32x64xi8>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_i8_i16(%29, %7, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb24(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb24(%38: memref<32x64xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_i8_i16(%38, %7, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
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
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb31(%59: memref<64x128xi8>):  // 3 preds: ^bb28, ^bb29, ^bb30
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
      cf.br ^bb35(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb35(%63: memref<32x64xi8>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_i8_i16(%63, %59, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb39(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb39(%72: memref<32x64xi8>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_i8_i16(%72, %59, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb43(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb43(%81: memref<32x64xi8>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_i8_i16(%81, %59, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb47(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb47(%90: memref<32x64xi8>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_i8_i16(%90, %59, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
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
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb49(%c0 : index)
    ^bb49(%106: index):  // 2 preds: ^bb48, ^bb70
      %107 = arith.cmpi slt, %106, %c16 : index
      cf.cond_br %107, ^bb50, ^bb71
    ^bb50:  // pred: ^bb49
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb53,
        0: ^bb51,
        1: ^bb52
      ]
    ^bb51:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb52:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb53:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb54(%111: memref<64x128xi8>):  // 3 preds: ^bb51, ^bb52, ^bb53
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %112 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %113 = arith.index_cast %112 : i32 to index
      %114 = arith.index_cast %113 : index to i32
      cf.switch %114 : i32, [
        default: ^bb57,
        0: ^bb55,
        1: ^bb56
      ]
    ^bb55:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb56:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb57:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb58(%115: memref<32x64xi8>):  // 3 preds: ^bb55, ^bb56, ^bb57
      func.call @matmul_i8_i16(%115, %111, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %116 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %117 = arith.addi %116, %c1_i32 : i32
      %118 = arith.cmpi sge, %117, %c2_i32 : i32
      %119 = arith.subi %117, %c2_i32 : i32
      %120 = arith.select %118, %119, %117 : i32
      memref.store %120, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %121 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %122 = arith.index_cast %121 : i32 to index
      %123 = arith.index_cast %122 : index to i32
      cf.switch %123 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb62(%124: memref<32x64xi8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @matmul_i8_i16(%124, %111, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %125 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %126 = arith.addi %125, %c1_i32 : i32
      %127 = arith.cmpi sge, %126, %c2_i32 : i32
      %128 = arith.subi %126, %c2_i32 : i32
      %129 = arith.select %127, %128, %126 : i32
      memref.store %129, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %130 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %131 = arith.index_cast %130 : i32 to index
      %132 = arith.index_cast %131 : index to i32
      cf.switch %132 : i32, [
        default: ^bb65,
        0: ^bb63,
        1: ^bb64
      ]
    ^bb63:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb64:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb66(%133: memref<32x64xi8>):  // 3 preds: ^bb63, ^bb64, ^bb65
      func.call @matmul_i8_i16(%133, %111, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %134 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %135 = arith.addi %134, %c1_i32 : i32
      %136 = arith.cmpi sge, %135, %c2_i32 : i32
      %137 = arith.subi %135, %c2_i32 : i32
      %138 = arith.select %136, %137, %135 : i32
      memref.store %138, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %139 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %140 = arith.index_cast %139 : i32 to index
      %141 = arith.index_cast %140 : index to i32
      cf.switch %141 : i32, [
        default: ^bb69,
        0: ^bb67,
        1: ^bb68
      ]
    ^bb67:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb68:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb69:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb70(%142: memref<32x64xi8>):  // 3 preds: ^bb67, ^bb68, ^bb69
      func.call @matmul_i8_i16(%142, %111, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %143 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %144 = arith.addi %143, %c1_i32 : i32
      %145 = arith.cmpi sge, %144, %c2_i32 : i32
      %146 = arith.subi %144, %c2_i32 : i32
      %147 = arith.select %145, %146, %144 : i32
      memref.store %147, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %148 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c2_i32 : i32
      %151 = arith.subi %149, %c2_i32 : i32
      %152 = arith.select %150, %151, %149 : i32
      memref.store %152, %_anonymous2[%c1] : memref<3xi32>
      %153 = arith.addi %106, %c1 : index
      cf.br ^bb49(%153 : index)
    ^bb71:  // pred: ^bb49
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %154 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %155 = arith.addi %154, %c1_i32 : i32
      %156 = arith.cmpi sge, %155, %c1_i32 : i32
      %157 = arith.select %156, %154, %155 : i32
      memref.store %157, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb72(%c0 : index)
    ^bb72(%158: index):  // 2 preds: ^bb71, ^bb93
      %159 = arith.cmpi slt, %158, %c16 : index
      cf.cond_br %159, ^bb73, ^bb94
    ^bb73:  // pred: ^bb72
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %160 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %161 = arith.index_cast %160 : i32 to index
      %162 = arith.index_cast %161 : index to i32
      cf.switch %162 : i32, [
        default: ^bb76,
        0: ^bb74,
        1: ^bb75
      ]
    ^bb74:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb75:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb76:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb77(%163: memref<64x128xi8>):  // 3 preds: ^bb74, ^bb75, ^bb76
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %164 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %165 = arith.index_cast %164 : i32 to index
      %166 = arith.index_cast %165 : index to i32
      cf.switch %166 : i32, [
        default: ^bb80,
        0: ^bb78,
        1: ^bb79
      ]
    ^bb78:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb79:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb80:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb81(%167: memref<32x64xi8>):  // 3 preds: ^bb78, ^bb79, ^bb80
      func.call @matmul_i8_i16(%167, %163, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %168 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %169 = arith.addi %168, %c1_i32 : i32
      %170 = arith.cmpi sge, %169, %c2_i32 : i32
      %171 = arith.subi %169, %c2_i32 : i32
      %172 = arith.select %170, %171, %169 : i32
      memref.store %172, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %173 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %174 = arith.index_cast %173 : i32 to index
      %175 = arith.index_cast %174 : index to i32
      cf.switch %175 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb85(%176: memref<32x64xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      func.call @matmul_i8_i16(%176, %163, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %177 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %178 = arith.addi %177, %c1_i32 : i32
      %179 = arith.cmpi sge, %178, %c2_i32 : i32
      %180 = arith.subi %178, %c2_i32 : i32
      %181 = arith.select %179, %180, %178 : i32
      memref.store %181, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %182 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %183 = arith.index_cast %182 : i32 to index
      %184 = arith.index_cast %183 : index to i32
      cf.switch %184 : i32, [
        default: ^bb88,
        0: ^bb86,
        1: ^bb87
      ]
    ^bb86:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb87:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb88:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb89(%185: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%185, %163, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %186 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %187 = arith.addi %186, %c1_i32 : i32
      %188 = arith.cmpi sge, %187, %c2_i32 : i32
      %189 = arith.subi %187, %c2_i32 : i32
      %190 = arith.select %188, %189, %187 : i32
      memref.store %190, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %191 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %192 = arith.index_cast %191 : i32 to index
      %193 = arith.index_cast %192 : index to i32
      cf.switch %193 : i32, [
        default: ^bb92,
        0: ^bb90,
        1: ^bb91
      ]
    ^bb90:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb91:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb92:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb93(%194: memref<32x64xi8>):  // 3 preds: ^bb90, ^bb91, ^bb92
      func.call @matmul_i8_i16(%194, %163, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %195 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %196 = arith.addi %195, %c1_i32 : i32
      %197 = arith.cmpi sge, %196, %c2_i32 : i32
      %198 = arith.subi %196, %c2_i32 : i32
      %199 = arith.select %197, %198, %196 : i32
      memref.store %199, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %200 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %201 = arith.addi %200, %c1_i32 : i32
      %202 = arith.cmpi sge, %201, %c2_i32 : i32
      %203 = arith.subi %201, %c2_i32 : i32
      %204 = arith.select %202, %203, %201 : i32
      memref.store %204, %_anonymous2[%c1] : memref<3xi32>
      %205 = arith.addi %158, %c1 : index
      cf.br ^bb72(%205 : index)
    ^bb94:  // pred: ^bb72
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %206 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %207 = arith.addi %206, %c1_i32 : i32
      %208 = arith.cmpi sge, %207, %c1_i32 : i32
      %209 = arith.select %208, %206, %207 : i32
      memref.store %209, %_anonymous2[%c0] : memref<3xi32>
      %210 = arith.addi %0, %c1 : index
      cf.br ^bb1(%210 : index)
    ^bb95:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous3 = aie.buffer(%tile_3_2) {address = 56576 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb94
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb95
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
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
      cf.br ^bb12(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb12(%11: memref<32x64xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb16(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb16(%20: memref<32x64xi8>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_i8_i16(%20, %7, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb20(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb20(%29: memref<32x64xi8>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_i8_i16(%29, %7, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb24(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb24(%38: memref<32x64xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_i8_i16(%38, %7, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
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
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb31(%59: memref<64x128xi8>):  // 3 preds: ^bb28, ^bb29, ^bb30
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
      cf.br ^bb35(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb35(%63: memref<32x64xi8>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_i8_i16(%63, %59, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb39(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb39(%72: memref<32x64xi8>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_i8_i16(%72, %59, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb43(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb43(%81: memref<32x64xi8>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_i8_i16(%81, %59, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb47(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb47(%90: memref<32x64xi8>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_i8_i16(%90, %59, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
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
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb49(%c0 : index)
    ^bb49(%106: index):  // 2 preds: ^bb48, ^bb70
      %107 = arith.cmpi slt, %106, %c16 : index
      cf.cond_br %107, ^bb50, ^bb71
    ^bb50:  // pred: ^bb49
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb53,
        0: ^bb51,
        1: ^bb52
      ]
    ^bb51:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb52:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb53:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb54(%111: memref<64x128xi8>):  // 3 preds: ^bb51, ^bb52, ^bb53
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %112 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %113 = arith.index_cast %112 : i32 to index
      %114 = arith.index_cast %113 : index to i32
      cf.switch %114 : i32, [
        default: ^bb57,
        0: ^bb55,
        1: ^bb56
      ]
    ^bb55:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb56:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb57:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb58(%115: memref<32x64xi8>):  // 3 preds: ^bb55, ^bb56, ^bb57
      func.call @matmul_i8_i16(%115, %111, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %116 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %117 = arith.addi %116, %c1_i32 : i32
      %118 = arith.cmpi sge, %117, %c2_i32 : i32
      %119 = arith.subi %117, %c2_i32 : i32
      %120 = arith.select %118, %119, %117 : i32
      memref.store %120, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %121 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %122 = arith.index_cast %121 : i32 to index
      %123 = arith.index_cast %122 : index to i32
      cf.switch %123 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb62(%124: memref<32x64xi8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @matmul_i8_i16(%124, %111, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %125 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %126 = arith.addi %125, %c1_i32 : i32
      %127 = arith.cmpi sge, %126, %c2_i32 : i32
      %128 = arith.subi %126, %c2_i32 : i32
      %129 = arith.select %127, %128, %126 : i32
      memref.store %129, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %130 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %131 = arith.index_cast %130 : i32 to index
      %132 = arith.index_cast %131 : index to i32
      cf.switch %132 : i32, [
        default: ^bb65,
        0: ^bb63,
        1: ^bb64
      ]
    ^bb63:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb64:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb66(%133: memref<32x64xi8>):  // 3 preds: ^bb63, ^bb64, ^bb65
      func.call @matmul_i8_i16(%133, %111, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %134 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %135 = arith.addi %134, %c1_i32 : i32
      %136 = arith.cmpi sge, %135, %c2_i32 : i32
      %137 = arith.subi %135, %c2_i32 : i32
      %138 = arith.select %136, %137, %135 : i32
      memref.store %138, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %139 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %140 = arith.index_cast %139 : i32 to index
      %141 = arith.index_cast %140 : index to i32
      cf.switch %141 : i32, [
        default: ^bb69,
        0: ^bb67,
        1: ^bb68
      ]
    ^bb67:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb68:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb69:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb70(%142: memref<32x64xi8>):  // 3 preds: ^bb67, ^bb68, ^bb69
      func.call @matmul_i8_i16(%142, %111, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %143 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %144 = arith.addi %143, %c1_i32 : i32
      %145 = arith.cmpi sge, %144, %c2_i32 : i32
      %146 = arith.subi %144, %c2_i32 : i32
      %147 = arith.select %145, %146, %144 : i32
      memref.store %147, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %148 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c2_i32 : i32
      %151 = arith.subi %149, %c2_i32 : i32
      %152 = arith.select %150, %151, %149 : i32
      memref.store %152, %_anonymous3[%c1] : memref<3xi32>
      %153 = arith.addi %106, %c1 : index
      cf.br ^bb49(%153 : index)
    ^bb71:  // pred: ^bb49
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %154 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %155 = arith.addi %154, %c1_i32 : i32
      %156 = arith.cmpi sge, %155, %c1_i32 : i32
      %157 = arith.select %156, %154, %155 : i32
      memref.store %157, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb72(%c0 : index)
    ^bb72(%158: index):  // 2 preds: ^bb71, ^bb93
      %159 = arith.cmpi slt, %158, %c16 : index
      cf.cond_br %159, ^bb73, ^bb94
    ^bb73:  // pred: ^bb72
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %160 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %161 = arith.index_cast %160 : i32 to index
      %162 = arith.index_cast %161 : index to i32
      cf.switch %162 : i32, [
        default: ^bb76,
        0: ^bb74,
        1: ^bb75
      ]
    ^bb74:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb75:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb76:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb77(%163: memref<64x128xi8>):  // 3 preds: ^bb74, ^bb75, ^bb76
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %164 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %165 = arith.index_cast %164 : i32 to index
      %166 = arith.index_cast %165 : index to i32
      cf.switch %166 : i32, [
        default: ^bb80,
        0: ^bb78,
        1: ^bb79
      ]
    ^bb78:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb79:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb80:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb81(%167: memref<32x64xi8>):  // 3 preds: ^bb78, ^bb79, ^bb80
      func.call @matmul_i8_i16(%167, %163, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %168 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %169 = arith.addi %168, %c1_i32 : i32
      %170 = arith.cmpi sge, %169, %c2_i32 : i32
      %171 = arith.subi %169, %c2_i32 : i32
      %172 = arith.select %170, %171, %169 : i32
      memref.store %172, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %173 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %174 = arith.index_cast %173 : i32 to index
      %175 = arith.index_cast %174 : index to i32
      cf.switch %175 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb85(%176: memref<32x64xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      func.call @matmul_i8_i16(%176, %163, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %177 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %178 = arith.addi %177, %c1_i32 : i32
      %179 = arith.cmpi sge, %178, %c2_i32 : i32
      %180 = arith.subi %178, %c2_i32 : i32
      %181 = arith.select %179, %180, %178 : i32
      memref.store %181, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %182 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %183 = arith.index_cast %182 : i32 to index
      %184 = arith.index_cast %183 : index to i32
      cf.switch %184 : i32, [
        default: ^bb88,
        0: ^bb86,
        1: ^bb87
      ]
    ^bb86:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb87:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb88:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb89(%185: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%185, %163, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %186 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %187 = arith.addi %186, %c1_i32 : i32
      %188 = arith.cmpi sge, %187, %c2_i32 : i32
      %189 = arith.subi %187, %c2_i32 : i32
      %190 = arith.select %188, %189, %187 : i32
      memref.store %190, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %191 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %192 = arith.index_cast %191 : i32 to index
      %193 = arith.index_cast %192 : index to i32
      cf.switch %193 : i32, [
        default: ^bb92,
        0: ^bb90,
        1: ^bb91
      ]
    ^bb90:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb91:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb92:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb93(%194: memref<32x64xi8>):  // 3 preds: ^bb90, ^bb91, ^bb92
      func.call @matmul_i8_i16(%194, %163, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %195 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %196 = arith.addi %195, %c1_i32 : i32
      %197 = arith.cmpi sge, %196, %c2_i32 : i32
      %198 = arith.subi %196, %c2_i32 : i32
      %199 = arith.select %197, %198, %196 : i32
      memref.store %199, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %200 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %201 = arith.addi %200, %c1_i32 : i32
      %202 = arith.cmpi sge, %201, %c2_i32 : i32
      %203 = arith.subi %201, %c2_i32 : i32
      %204 = arith.select %202, %203, %201 : i32
      memref.store %204, %_anonymous3[%c1] : memref<3xi32>
      %205 = arith.addi %158, %c1 : index
      cf.br ^bb72(%205 : index)
    ^bb94:  // pred: ^bb72
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %206 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %207 = arith.addi %206, %c1_i32 : i32
      %208 = arith.cmpi sge, %207, %c1_i32 : i32
      %209 = arith.select %208, %206, %207 : i32
      memref.store %209, %_anonymous3[%c0] : memref<3xi32>
      %210 = arith.addi %0, %c1 : index
      cf.br ^bb1(%210 : index)
    ^bb95:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous4 = aie.buffer(%tile_4_2) {address = 56576 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb94
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb95
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
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
      cf.br ^bb12(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb12(%11: memref<32x64xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb16(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb16(%20: memref<32x64xi8>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_i8_i16(%20, %7, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb20(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb20(%29: memref<32x64xi8>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_i8_i16(%29, %7, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb24(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb24(%38: memref<32x64xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_i8_i16(%38, %7, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
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
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb31(%59: memref<64x128xi8>):  // 3 preds: ^bb28, ^bb29, ^bb30
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
      cf.br ^bb35(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb35(%63: memref<32x64xi8>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_i8_i16(%63, %59, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb39(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb39(%72: memref<32x64xi8>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_i8_i16(%72, %59, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb43(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb43(%81: memref<32x64xi8>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_i8_i16(%81, %59, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb47(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb47(%90: memref<32x64xi8>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_i8_i16(%90, %59, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
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
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb49(%c0 : index)
    ^bb49(%106: index):  // 2 preds: ^bb48, ^bb70
      %107 = arith.cmpi slt, %106, %c16 : index
      cf.cond_br %107, ^bb50, ^bb71
    ^bb50:  // pred: ^bb49
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb53,
        0: ^bb51,
        1: ^bb52
      ]
    ^bb51:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb52:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb53:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb54(%111: memref<64x128xi8>):  // 3 preds: ^bb51, ^bb52, ^bb53
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %112 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %113 = arith.index_cast %112 : i32 to index
      %114 = arith.index_cast %113 : index to i32
      cf.switch %114 : i32, [
        default: ^bb57,
        0: ^bb55,
        1: ^bb56
      ]
    ^bb55:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb56:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb57:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb58(%115: memref<32x64xi8>):  // 3 preds: ^bb55, ^bb56, ^bb57
      func.call @matmul_i8_i16(%115, %111, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %116 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %117 = arith.addi %116, %c1_i32 : i32
      %118 = arith.cmpi sge, %117, %c2_i32 : i32
      %119 = arith.subi %117, %c2_i32 : i32
      %120 = arith.select %118, %119, %117 : i32
      memref.store %120, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %121 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %122 = arith.index_cast %121 : i32 to index
      %123 = arith.index_cast %122 : index to i32
      cf.switch %123 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb62(%124: memref<32x64xi8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @matmul_i8_i16(%124, %111, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %125 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %126 = arith.addi %125, %c1_i32 : i32
      %127 = arith.cmpi sge, %126, %c2_i32 : i32
      %128 = arith.subi %126, %c2_i32 : i32
      %129 = arith.select %127, %128, %126 : i32
      memref.store %129, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %130 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %131 = arith.index_cast %130 : i32 to index
      %132 = arith.index_cast %131 : index to i32
      cf.switch %132 : i32, [
        default: ^bb65,
        0: ^bb63,
        1: ^bb64
      ]
    ^bb63:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb64:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb66(%133: memref<32x64xi8>):  // 3 preds: ^bb63, ^bb64, ^bb65
      func.call @matmul_i8_i16(%133, %111, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %134 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %135 = arith.addi %134, %c1_i32 : i32
      %136 = arith.cmpi sge, %135, %c2_i32 : i32
      %137 = arith.subi %135, %c2_i32 : i32
      %138 = arith.select %136, %137, %135 : i32
      memref.store %138, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %139 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %140 = arith.index_cast %139 : i32 to index
      %141 = arith.index_cast %140 : index to i32
      cf.switch %141 : i32, [
        default: ^bb69,
        0: ^bb67,
        1: ^bb68
      ]
    ^bb67:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb68:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb69:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb70(%142: memref<32x64xi8>):  // 3 preds: ^bb67, ^bb68, ^bb69
      func.call @matmul_i8_i16(%142, %111, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %143 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %144 = arith.addi %143, %c1_i32 : i32
      %145 = arith.cmpi sge, %144, %c2_i32 : i32
      %146 = arith.subi %144, %c2_i32 : i32
      %147 = arith.select %145, %146, %144 : i32
      memref.store %147, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %148 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c2_i32 : i32
      %151 = arith.subi %149, %c2_i32 : i32
      %152 = arith.select %150, %151, %149 : i32
      memref.store %152, %_anonymous4[%c1] : memref<3xi32>
      %153 = arith.addi %106, %c1 : index
      cf.br ^bb49(%153 : index)
    ^bb71:  // pred: ^bb49
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %154 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %155 = arith.addi %154, %c1_i32 : i32
      %156 = arith.cmpi sge, %155, %c1_i32 : i32
      %157 = arith.select %156, %154, %155 : i32
      memref.store %157, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb72(%c0 : index)
    ^bb72(%158: index):  // 2 preds: ^bb71, ^bb93
      %159 = arith.cmpi slt, %158, %c16 : index
      cf.cond_br %159, ^bb73, ^bb94
    ^bb73:  // pred: ^bb72
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %160 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %161 = arith.index_cast %160 : i32 to index
      %162 = arith.index_cast %161 : index to i32
      cf.switch %162 : i32, [
        default: ^bb76,
        0: ^bb74,
        1: ^bb75
      ]
    ^bb74:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb75:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb76:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb77(%163: memref<64x128xi8>):  // 3 preds: ^bb74, ^bb75, ^bb76
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %164 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %165 = arith.index_cast %164 : i32 to index
      %166 = arith.index_cast %165 : index to i32
      cf.switch %166 : i32, [
        default: ^bb80,
        0: ^bb78,
        1: ^bb79
      ]
    ^bb78:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb79:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb80:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb81(%167: memref<32x64xi8>):  // 3 preds: ^bb78, ^bb79, ^bb80
      func.call @matmul_i8_i16(%167, %163, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %168 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %169 = arith.addi %168, %c1_i32 : i32
      %170 = arith.cmpi sge, %169, %c2_i32 : i32
      %171 = arith.subi %169, %c2_i32 : i32
      %172 = arith.select %170, %171, %169 : i32
      memref.store %172, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %173 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %174 = arith.index_cast %173 : i32 to index
      %175 = arith.index_cast %174 : index to i32
      cf.switch %175 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb85(%176: memref<32x64xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      func.call @matmul_i8_i16(%176, %163, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %177 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %178 = arith.addi %177, %c1_i32 : i32
      %179 = arith.cmpi sge, %178, %c2_i32 : i32
      %180 = arith.subi %178, %c2_i32 : i32
      %181 = arith.select %179, %180, %178 : i32
      memref.store %181, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %182 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %183 = arith.index_cast %182 : i32 to index
      %184 = arith.index_cast %183 : index to i32
      cf.switch %184 : i32, [
        default: ^bb88,
        0: ^bb86,
        1: ^bb87
      ]
    ^bb86:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb87:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb88:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb89(%185: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%185, %163, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %186 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %187 = arith.addi %186, %c1_i32 : i32
      %188 = arith.cmpi sge, %187, %c2_i32 : i32
      %189 = arith.subi %187, %c2_i32 : i32
      %190 = arith.select %188, %189, %187 : i32
      memref.store %190, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %191 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %192 = arith.index_cast %191 : i32 to index
      %193 = arith.index_cast %192 : index to i32
      cf.switch %193 : i32, [
        default: ^bb92,
        0: ^bb90,
        1: ^bb91
      ]
    ^bb90:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb91:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb92:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb93(%194: memref<32x64xi8>):  // 3 preds: ^bb90, ^bb91, ^bb92
      func.call @matmul_i8_i16(%194, %163, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %195 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %196 = arith.addi %195, %c1_i32 : i32
      %197 = arith.cmpi sge, %196, %c2_i32 : i32
      %198 = arith.subi %196, %c2_i32 : i32
      %199 = arith.select %197, %198, %196 : i32
      memref.store %199, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %200 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %201 = arith.addi %200, %c1_i32 : i32
      %202 = arith.cmpi sge, %201, %c2_i32 : i32
      %203 = arith.subi %201, %c2_i32 : i32
      %204 = arith.select %202, %203, %201 : i32
      memref.store %204, %_anonymous4[%c1] : memref<3xi32>
      %205 = arith.addi %158, %c1 : index
      cf.br ^bb72(%205 : index)
    ^bb94:  // pred: ^bb72
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %206 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %207 = arith.addi %206, %c1_i32 : i32
      %208 = arith.cmpi sge, %207, %c1_i32 : i32
      %209 = arith.select %208, %206, %207 : i32
      memref.store %209, %_anonymous4[%c0] : memref<3xi32>
      %210 = arith.addi %0, %c1 : index
      cf.br ^bb1(%210 : index)
    ^bb95:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous5 = aie.buffer(%tile_5_2) {address = 56576 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb94
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb95
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
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
      cf.br ^bb12(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb12(%11: memref<32x64xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb16(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb16(%20: memref<32x64xi8>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_i8_i16(%20, %7, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb20(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb20(%29: memref<32x64xi8>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_i8_i16(%29, %7, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb24(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb24(%38: memref<32x64xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_i8_i16(%38, %7, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
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
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb31(%59: memref<64x128xi8>):  // 3 preds: ^bb28, ^bb29, ^bb30
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
      cf.br ^bb35(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb35(%63: memref<32x64xi8>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_i8_i16(%63, %59, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb39(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb39(%72: memref<32x64xi8>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_i8_i16(%72, %59, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb43(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb43(%81: memref<32x64xi8>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_i8_i16(%81, %59, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb47(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb47(%90: memref<32x64xi8>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_i8_i16(%90, %59, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
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
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb49(%c0 : index)
    ^bb49(%106: index):  // 2 preds: ^bb48, ^bb70
      %107 = arith.cmpi slt, %106, %c16 : index
      cf.cond_br %107, ^bb50, ^bb71
    ^bb50:  // pred: ^bb49
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb53,
        0: ^bb51,
        1: ^bb52
      ]
    ^bb51:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb52:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb53:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb54(%111: memref<64x128xi8>):  // 3 preds: ^bb51, ^bb52, ^bb53
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %112 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %113 = arith.index_cast %112 : i32 to index
      %114 = arith.index_cast %113 : index to i32
      cf.switch %114 : i32, [
        default: ^bb57,
        0: ^bb55,
        1: ^bb56
      ]
    ^bb55:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb56:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb57:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb58(%115: memref<32x64xi8>):  // 3 preds: ^bb55, ^bb56, ^bb57
      func.call @matmul_i8_i16(%115, %111, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %116 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %117 = arith.addi %116, %c1_i32 : i32
      %118 = arith.cmpi sge, %117, %c2_i32 : i32
      %119 = arith.subi %117, %c2_i32 : i32
      %120 = arith.select %118, %119, %117 : i32
      memref.store %120, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %121 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %122 = arith.index_cast %121 : i32 to index
      %123 = arith.index_cast %122 : index to i32
      cf.switch %123 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb62(%124: memref<32x64xi8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @matmul_i8_i16(%124, %111, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %125 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %126 = arith.addi %125, %c1_i32 : i32
      %127 = arith.cmpi sge, %126, %c2_i32 : i32
      %128 = arith.subi %126, %c2_i32 : i32
      %129 = arith.select %127, %128, %126 : i32
      memref.store %129, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %130 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %131 = arith.index_cast %130 : i32 to index
      %132 = arith.index_cast %131 : index to i32
      cf.switch %132 : i32, [
        default: ^bb65,
        0: ^bb63,
        1: ^bb64
      ]
    ^bb63:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb64:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb66(%133: memref<32x64xi8>):  // 3 preds: ^bb63, ^bb64, ^bb65
      func.call @matmul_i8_i16(%133, %111, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %134 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %135 = arith.addi %134, %c1_i32 : i32
      %136 = arith.cmpi sge, %135, %c2_i32 : i32
      %137 = arith.subi %135, %c2_i32 : i32
      %138 = arith.select %136, %137, %135 : i32
      memref.store %138, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %139 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %140 = arith.index_cast %139 : i32 to index
      %141 = arith.index_cast %140 : index to i32
      cf.switch %141 : i32, [
        default: ^bb69,
        0: ^bb67,
        1: ^bb68
      ]
    ^bb67:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb68:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb69:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb70(%142: memref<32x64xi8>):  // 3 preds: ^bb67, ^bb68, ^bb69
      func.call @matmul_i8_i16(%142, %111, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %143 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %144 = arith.addi %143, %c1_i32 : i32
      %145 = arith.cmpi sge, %144, %c2_i32 : i32
      %146 = arith.subi %144, %c2_i32 : i32
      %147 = arith.select %145, %146, %144 : i32
      memref.store %147, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %148 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c2_i32 : i32
      %151 = arith.subi %149, %c2_i32 : i32
      %152 = arith.select %150, %151, %149 : i32
      memref.store %152, %_anonymous5[%c1] : memref<3xi32>
      %153 = arith.addi %106, %c1 : index
      cf.br ^bb49(%153 : index)
    ^bb71:  // pred: ^bb49
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %154 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %155 = arith.addi %154, %c1_i32 : i32
      %156 = arith.cmpi sge, %155, %c1_i32 : i32
      %157 = arith.select %156, %154, %155 : i32
      memref.store %157, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb72(%c0 : index)
    ^bb72(%158: index):  // 2 preds: ^bb71, ^bb93
      %159 = arith.cmpi slt, %158, %c16 : index
      cf.cond_br %159, ^bb73, ^bb94
    ^bb73:  // pred: ^bb72
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %160 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %161 = arith.index_cast %160 : i32 to index
      %162 = arith.index_cast %161 : index to i32
      cf.switch %162 : i32, [
        default: ^bb76,
        0: ^bb74,
        1: ^bb75
      ]
    ^bb74:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb75:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb76:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb77(%163: memref<64x128xi8>):  // 3 preds: ^bb74, ^bb75, ^bb76
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %164 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %165 = arith.index_cast %164 : i32 to index
      %166 = arith.index_cast %165 : index to i32
      cf.switch %166 : i32, [
        default: ^bb80,
        0: ^bb78,
        1: ^bb79
      ]
    ^bb78:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb79:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb80:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb81(%167: memref<32x64xi8>):  // 3 preds: ^bb78, ^bb79, ^bb80
      func.call @matmul_i8_i16(%167, %163, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %168 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %169 = arith.addi %168, %c1_i32 : i32
      %170 = arith.cmpi sge, %169, %c2_i32 : i32
      %171 = arith.subi %169, %c2_i32 : i32
      %172 = arith.select %170, %171, %169 : i32
      memref.store %172, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %173 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %174 = arith.index_cast %173 : i32 to index
      %175 = arith.index_cast %174 : index to i32
      cf.switch %175 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb85(%176: memref<32x64xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      func.call @matmul_i8_i16(%176, %163, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %177 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %178 = arith.addi %177, %c1_i32 : i32
      %179 = arith.cmpi sge, %178, %c2_i32 : i32
      %180 = arith.subi %178, %c2_i32 : i32
      %181 = arith.select %179, %180, %178 : i32
      memref.store %181, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %182 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %183 = arith.index_cast %182 : i32 to index
      %184 = arith.index_cast %183 : index to i32
      cf.switch %184 : i32, [
        default: ^bb88,
        0: ^bb86,
        1: ^bb87
      ]
    ^bb86:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb87:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb88:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb89(%185: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%185, %163, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %186 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %187 = arith.addi %186, %c1_i32 : i32
      %188 = arith.cmpi sge, %187, %c2_i32 : i32
      %189 = arith.subi %187, %c2_i32 : i32
      %190 = arith.select %188, %189, %187 : i32
      memref.store %190, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %191 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %192 = arith.index_cast %191 : i32 to index
      %193 = arith.index_cast %192 : index to i32
      cf.switch %193 : i32, [
        default: ^bb92,
        0: ^bb90,
        1: ^bb91
      ]
    ^bb90:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb91:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb92:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb93(%194: memref<32x64xi8>):  // 3 preds: ^bb90, ^bb91, ^bb92
      func.call @matmul_i8_i16(%194, %163, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %195 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %196 = arith.addi %195, %c1_i32 : i32
      %197 = arith.cmpi sge, %196, %c2_i32 : i32
      %198 = arith.subi %196, %c2_i32 : i32
      %199 = arith.select %197, %198, %196 : i32
      memref.store %199, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %200 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %201 = arith.addi %200, %c1_i32 : i32
      %202 = arith.cmpi sge, %201, %c2_i32 : i32
      %203 = arith.subi %201, %c2_i32 : i32
      %204 = arith.select %202, %203, %201 : i32
      memref.store %204, %_anonymous5[%c1] : memref<3xi32>
      %205 = arith.addi %158, %c1 : index
      cf.br ^bb72(%205 : index)
    ^bb94:  // pred: ^bb72
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %206 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %207 = arith.addi %206, %c1_i32 : i32
      %208 = arith.cmpi sge, %207, %c1_i32 : i32
      %209 = arith.select %208, %206, %207 : i32
      memref.store %209, %_anonymous5[%c0] : memref<3xi32>
      %210 = arith.addi %0, %c1 : index
      cf.br ^bb1(%210 : index)
    ^bb95:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous6 = aie.buffer(%tile_6_2) {address = 56576 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb94
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb95
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
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
      cf.br ^bb12(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb12(%11: memref<32x64xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb16(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb16(%20: memref<32x64xi8>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_i8_i16(%20, %7, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb20(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb20(%29: memref<32x64xi8>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_i8_i16(%29, %7, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb24(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb24(%38: memref<32x64xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_i8_i16(%38, %7, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
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
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb31(%59: memref<64x128xi8>):  // 3 preds: ^bb28, ^bb29, ^bb30
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
      cf.br ^bb35(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb35(%63: memref<32x64xi8>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_i8_i16(%63, %59, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb39(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb39(%72: memref<32x64xi8>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_i8_i16(%72, %59, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb43(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb43(%81: memref<32x64xi8>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_i8_i16(%81, %59, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb47(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb47(%90: memref<32x64xi8>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_i8_i16(%90, %59, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
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
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb49(%c0 : index)
    ^bb49(%106: index):  // 2 preds: ^bb48, ^bb70
      %107 = arith.cmpi slt, %106, %c16 : index
      cf.cond_br %107, ^bb50, ^bb71
    ^bb50:  // pred: ^bb49
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb53,
        0: ^bb51,
        1: ^bb52
      ]
    ^bb51:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb52:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb53:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb54(%111: memref<64x128xi8>):  // 3 preds: ^bb51, ^bb52, ^bb53
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %112 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %113 = arith.index_cast %112 : i32 to index
      %114 = arith.index_cast %113 : index to i32
      cf.switch %114 : i32, [
        default: ^bb57,
        0: ^bb55,
        1: ^bb56
      ]
    ^bb55:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb56:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb57:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb58(%115: memref<32x64xi8>):  // 3 preds: ^bb55, ^bb56, ^bb57
      func.call @matmul_i8_i16(%115, %111, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %116 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %117 = arith.addi %116, %c1_i32 : i32
      %118 = arith.cmpi sge, %117, %c2_i32 : i32
      %119 = arith.subi %117, %c2_i32 : i32
      %120 = arith.select %118, %119, %117 : i32
      memref.store %120, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %121 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %122 = arith.index_cast %121 : i32 to index
      %123 = arith.index_cast %122 : index to i32
      cf.switch %123 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb62(%124: memref<32x64xi8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @matmul_i8_i16(%124, %111, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %125 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %126 = arith.addi %125, %c1_i32 : i32
      %127 = arith.cmpi sge, %126, %c2_i32 : i32
      %128 = arith.subi %126, %c2_i32 : i32
      %129 = arith.select %127, %128, %126 : i32
      memref.store %129, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %130 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %131 = arith.index_cast %130 : i32 to index
      %132 = arith.index_cast %131 : index to i32
      cf.switch %132 : i32, [
        default: ^bb65,
        0: ^bb63,
        1: ^bb64
      ]
    ^bb63:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb64:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb66(%133: memref<32x64xi8>):  // 3 preds: ^bb63, ^bb64, ^bb65
      func.call @matmul_i8_i16(%133, %111, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %134 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %135 = arith.addi %134, %c1_i32 : i32
      %136 = arith.cmpi sge, %135, %c2_i32 : i32
      %137 = arith.subi %135, %c2_i32 : i32
      %138 = arith.select %136, %137, %135 : i32
      memref.store %138, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %139 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %140 = arith.index_cast %139 : i32 to index
      %141 = arith.index_cast %140 : index to i32
      cf.switch %141 : i32, [
        default: ^bb69,
        0: ^bb67,
        1: ^bb68
      ]
    ^bb67:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb68:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb69:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb70(%142: memref<32x64xi8>):  // 3 preds: ^bb67, ^bb68, ^bb69
      func.call @matmul_i8_i16(%142, %111, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %143 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %144 = arith.addi %143, %c1_i32 : i32
      %145 = arith.cmpi sge, %144, %c2_i32 : i32
      %146 = arith.subi %144, %c2_i32 : i32
      %147 = arith.select %145, %146, %144 : i32
      memref.store %147, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %148 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c2_i32 : i32
      %151 = arith.subi %149, %c2_i32 : i32
      %152 = arith.select %150, %151, %149 : i32
      memref.store %152, %_anonymous6[%c1] : memref<3xi32>
      %153 = arith.addi %106, %c1 : index
      cf.br ^bb49(%153 : index)
    ^bb71:  // pred: ^bb49
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %154 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %155 = arith.addi %154, %c1_i32 : i32
      %156 = arith.cmpi sge, %155, %c1_i32 : i32
      %157 = arith.select %156, %154, %155 : i32
      memref.store %157, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb72(%c0 : index)
    ^bb72(%158: index):  // 2 preds: ^bb71, ^bb93
      %159 = arith.cmpi slt, %158, %c16 : index
      cf.cond_br %159, ^bb73, ^bb94
    ^bb73:  // pred: ^bb72
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %160 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %161 = arith.index_cast %160 : i32 to index
      %162 = arith.index_cast %161 : index to i32
      cf.switch %162 : i32, [
        default: ^bb76,
        0: ^bb74,
        1: ^bb75
      ]
    ^bb74:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb75:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb76:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb77(%163: memref<64x128xi8>):  // 3 preds: ^bb74, ^bb75, ^bb76
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %164 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %165 = arith.index_cast %164 : i32 to index
      %166 = arith.index_cast %165 : index to i32
      cf.switch %166 : i32, [
        default: ^bb80,
        0: ^bb78,
        1: ^bb79
      ]
    ^bb78:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb79:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb80:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb81(%167: memref<32x64xi8>):  // 3 preds: ^bb78, ^bb79, ^bb80
      func.call @matmul_i8_i16(%167, %163, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %168 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %169 = arith.addi %168, %c1_i32 : i32
      %170 = arith.cmpi sge, %169, %c2_i32 : i32
      %171 = arith.subi %169, %c2_i32 : i32
      %172 = arith.select %170, %171, %169 : i32
      memref.store %172, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %173 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %174 = arith.index_cast %173 : i32 to index
      %175 = arith.index_cast %174 : index to i32
      cf.switch %175 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb85(%176: memref<32x64xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      func.call @matmul_i8_i16(%176, %163, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %177 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %178 = arith.addi %177, %c1_i32 : i32
      %179 = arith.cmpi sge, %178, %c2_i32 : i32
      %180 = arith.subi %178, %c2_i32 : i32
      %181 = arith.select %179, %180, %178 : i32
      memref.store %181, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %182 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %183 = arith.index_cast %182 : i32 to index
      %184 = arith.index_cast %183 : index to i32
      cf.switch %184 : i32, [
        default: ^bb88,
        0: ^bb86,
        1: ^bb87
      ]
    ^bb86:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb87:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb88:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb89(%185: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%185, %163, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %186 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %187 = arith.addi %186, %c1_i32 : i32
      %188 = arith.cmpi sge, %187, %c2_i32 : i32
      %189 = arith.subi %187, %c2_i32 : i32
      %190 = arith.select %188, %189, %187 : i32
      memref.store %190, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %191 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %192 = arith.index_cast %191 : i32 to index
      %193 = arith.index_cast %192 : index to i32
      cf.switch %193 : i32, [
        default: ^bb92,
        0: ^bb90,
        1: ^bb91
      ]
    ^bb90:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb91:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb92:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb93(%194: memref<32x64xi8>):  // 3 preds: ^bb90, ^bb91, ^bb92
      func.call @matmul_i8_i16(%194, %163, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %195 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %196 = arith.addi %195, %c1_i32 : i32
      %197 = arith.cmpi sge, %196, %c2_i32 : i32
      %198 = arith.subi %196, %c2_i32 : i32
      %199 = arith.select %197, %198, %196 : i32
      memref.store %199, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %200 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %201 = arith.addi %200, %c1_i32 : i32
      %202 = arith.cmpi sge, %201, %c2_i32 : i32
      %203 = arith.subi %201, %c2_i32 : i32
      %204 = arith.select %202, %203, %201 : i32
      memref.store %204, %_anonymous6[%c1] : memref<3xi32>
      %205 = arith.addi %158, %c1 : index
      cf.br ^bb72(%205 : index)
    ^bb94:  // pred: ^bb72
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %206 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %207 = arith.addi %206, %c1_i32 : i32
      %208 = arith.cmpi sge, %207, %c1_i32 : i32
      %209 = arith.select %208, %206, %207 : i32
      memref.store %209, %_anonymous6[%c0] : memref<3xi32>
      %210 = arith.addi %0, %c1 : index
      cf.br ^bb1(%210 : index)
    ^bb95:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    %_anonymous7 = aie.buffer(%tile_7_2) {address = 56576 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb94
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb95
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb24
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb25
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
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
      cf.br ^bb12(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb12(%11: memref<32x64xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb16(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb16(%20: memref<32x64xi8>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @matmul_i8_i16(%20, %7, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb20(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb20(%29: memref<32x64xi8>):  // 3 preds: ^bb17, ^bb18, ^bb19
      func.call @matmul_i8_i16(%29, %7, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb24(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb24(%38: memref<32x64xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @matmul_i8_i16(%38, %7, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %39 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %40 = arith.addi %39, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.subi %40, %c2_i32 : i32
      %43 = arith.select %41, %42, %40 : i32
      memref.store %43, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
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
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb26(%c0 : index)
    ^bb26(%54: index):  // 2 preds: ^bb25, ^bb47
      %55 = arith.cmpi slt, %54, %c16 : index
      cf.cond_br %55, ^bb27, ^bb48
    ^bb27:  // pred: ^bb26
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %56 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.index_cast %57 : index to i32
      cf.switch %58 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb31(%59: memref<64x128xi8>):  // 3 preds: ^bb28, ^bb29, ^bb30
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
      cf.br ^bb35(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb35(%63: memref<32x64xi8>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @matmul_i8_i16(%63, %59, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb39(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb37:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb38:  // pred: ^bb35
      cf.br ^bb39(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb39(%72: memref<32x64xi8>):  // 3 preds: ^bb36, ^bb37, ^bb38
      func.call @matmul_i8_i16(%72, %59, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb43(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb42:  // pred: ^bb39
      cf.br ^bb43(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb43(%81: memref<32x64xi8>):  // 3 preds: ^bb40, ^bb41, ^bb42
      func.call @matmul_i8_i16(%81, %59, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
      cf.br ^bb47(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb45:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb46:  // pred: ^bb43
      cf.br ^bb47(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb47(%90: memref<32x64xi8>):  // 3 preds: ^bb44, ^bb45, ^bb46
      func.call @matmul_i8_i16(%90, %59, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %91 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %92 = arith.addi %91, %c1_i32 : i32
      %93 = arith.cmpi sge, %92, %c2_i32 : i32
      %94 = arith.subi %92, %c2_i32 : i32
      %95 = arith.select %93, %94, %92 : i32
      memref.store %95, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
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
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb49(%c0 : index)
    ^bb49(%106: index):  // 2 preds: ^bb48, ^bb70
      %107 = arith.cmpi slt, %106, %c16 : index
      cf.cond_br %107, ^bb50, ^bb71
    ^bb50:  // pred: ^bb49
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb53,
        0: ^bb51,
        1: ^bb52
      ]
    ^bb51:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb52:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb53:  // pred: ^bb50
      cf.br ^bb54(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb54(%111: memref<64x128xi8>):  // 3 preds: ^bb51, ^bb52, ^bb53
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %112 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %113 = arith.index_cast %112 : i32 to index
      %114 = arith.index_cast %113 : index to i32
      cf.switch %114 : i32, [
        default: ^bb57,
        0: ^bb55,
        1: ^bb56
      ]
    ^bb55:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb56:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb57:  // pred: ^bb54
      cf.br ^bb58(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb58(%115: memref<32x64xi8>):  // 3 preds: ^bb55, ^bb56, ^bb57
      func.call @matmul_i8_i16(%115, %111, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %116 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %117 = arith.addi %116, %c1_i32 : i32
      %118 = arith.cmpi sge, %117, %c2_i32 : i32
      %119 = arith.subi %117, %c2_i32 : i32
      %120 = arith.select %118, %119, %117 : i32
      memref.store %120, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %121 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %122 = arith.index_cast %121 : i32 to index
      %123 = arith.index_cast %122 : index to i32
      cf.switch %123 : i32, [
        default: ^bb61,
        0: ^bb59,
        1: ^bb60
      ]
    ^bb59:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb60:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb61:  // pred: ^bb58
      cf.br ^bb62(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb62(%124: memref<32x64xi8>):  // 3 preds: ^bb59, ^bb60, ^bb61
      func.call @matmul_i8_i16(%124, %111, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %125 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %126 = arith.addi %125, %c1_i32 : i32
      %127 = arith.cmpi sge, %126, %c2_i32 : i32
      %128 = arith.subi %126, %c2_i32 : i32
      %129 = arith.select %127, %128, %126 : i32
      memref.store %129, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %130 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %131 = arith.index_cast %130 : i32 to index
      %132 = arith.index_cast %131 : index to i32
      cf.switch %132 : i32, [
        default: ^bb65,
        0: ^bb63,
        1: ^bb64
      ]
    ^bb63:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb64:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb62
      cf.br ^bb66(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb66(%133: memref<32x64xi8>):  // 3 preds: ^bb63, ^bb64, ^bb65
      func.call @matmul_i8_i16(%133, %111, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %134 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %135 = arith.addi %134, %c1_i32 : i32
      %136 = arith.cmpi sge, %135, %c2_i32 : i32
      %137 = arith.subi %135, %c2_i32 : i32
      %138 = arith.select %136, %137, %135 : i32
      memref.store %138, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %139 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %140 = arith.index_cast %139 : i32 to index
      %141 = arith.index_cast %140 : index to i32
      cf.switch %141 : i32, [
        default: ^bb69,
        0: ^bb67,
        1: ^bb68
      ]
    ^bb67:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb68:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb69:  // pred: ^bb66
      cf.br ^bb70(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb70(%142: memref<32x64xi8>):  // 3 preds: ^bb67, ^bb68, ^bb69
      func.call @matmul_i8_i16(%142, %111, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %143 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %144 = arith.addi %143, %c1_i32 : i32
      %145 = arith.cmpi sge, %144, %c2_i32 : i32
      %146 = arith.subi %144, %c2_i32 : i32
      %147 = arith.select %145, %146, %144 : i32
      memref.store %147, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %148 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c2_i32 : i32
      %151 = arith.subi %149, %c2_i32 : i32
      %152 = arith.select %150, %151, %149 : i32
      memref.store %152, %_anonymous7[%c1] : memref<3xi32>
      %153 = arith.addi %106, %c1 : index
      cf.br ^bb49(%153 : index)
    ^bb71:  // pred: ^bb49
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %154 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %155 = arith.addi %154, %c1_i32 : i32
      %156 = arith.cmpi sge, %155, %c1_i32 : i32
      %157 = arith.select %156, %154, %155 : i32
      memref.store %157, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb72(%c0 : index)
    ^bb72(%158: index):  // 2 preds: ^bb71, ^bb93
      %159 = arith.cmpi slt, %158, %c16 : index
      cf.cond_br %159, ^bb73, ^bb94
    ^bb73:  // pred: ^bb72
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %160 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %161 = arith.index_cast %160 : i32 to index
      %162 = arith.index_cast %161 : index to i32
      cf.switch %162 : i32, [
        default: ^bb76,
        0: ^bb74,
        1: ^bb75
      ]
    ^bb74:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb75:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb76:  // pred: ^bb73
      cf.br ^bb77(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb77(%163: memref<64x128xi8>):  // 3 preds: ^bb74, ^bb75, ^bb76
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %164 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %165 = arith.index_cast %164 : i32 to index
      %166 = arith.index_cast %165 : index to i32
      cf.switch %166 : i32, [
        default: ^bb80,
        0: ^bb78,
        1: ^bb79
      ]
    ^bb78:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb79:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb80:  // pred: ^bb77
      cf.br ^bb81(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb81(%167: memref<32x64xi8>):  // 3 preds: ^bb78, ^bb79, ^bb80
      func.call @matmul_i8_i16(%167, %163, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %168 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %169 = arith.addi %168, %c1_i32 : i32
      %170 = arith.cmpi sge, %169, %c2_i32 : i32
      %171 = arith.subi %169, %c2_i32 : i32
      %172 = arith.select %170, %171, %169 : i32
      memref.store %172, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %173 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %174 = arith.index_cast %173 : i32 to index
      %175 = arith.index_cast %174 : index to i32
      cf.switch %175 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb85(%176: memref<32x64xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      func.call @matmul_i8_i16(%176, %163, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %177 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %178 = arith.addi %177, %c1_i32 : i32
      %179 = arith.cmpi sge, %178, %c2_i32 : i32
      %180 = arith.subi %178, %c2_i32 : i32
      %181 = arith.select %179, %180, %178 : i32
      memref.store %181, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %182 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %183 = arith.index_cast %182 : i32 to index
      %184 = arith.index_cast %183 : index to i32
      cf.switch %184 : i32, [
        default: ^bb88,
        0: ^bb86,
        1: ^bb87
      ]
    ^bb86:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb87:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb88:  // pred: ^bb85
      cf.br ^bb89(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb89(%185: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%185, %163, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %186 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %187 = arith.addi %186, %c1_i32 : i32
      %188 = arith.cmpi sge, %187, %c2_i32 : i32
      %189 = arith.subi %187, %c2_i32 : i32
      %190 = arith.select %188, %189, %187 : i32
      memref.store %190, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %191 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %192 = arith.index_cast %191 : i32 to index
      %193 = arith.index_cast %192 : index to i32
      cf.switch %193 : i32, [
        default: ^bb92,
        0: ^bb90,
        1: ^bb91
      ]
    ^bb90:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb91:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb92:  // pred: ^bb89
      cf.br ^bb93(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb93(%194: memref<32x64xi8>):  // 3 preds: ^bb90, ^bb91, ^bb92
      func.call @matmul_i8_i16(%194, %163, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %195 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %196 = arith.addi %195, %c1_i32 : i32
      %197 = arith.cmpi sge, %196, %c2_i32 : i32
      %198 = arith.subi %196, %c2_i32 : i32
      %199 = arith.select %197, %198, %196 : i32
      memref.store %199, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %200 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %201 = arith.addi %200, %c1_i32 : i32
      %202 = arith.cmpi sge, %201, %c2_i32 : i32
      %203 = arith.subi %201, %c2_i32 : i32
      %204 = arith.select %202, %203, %201 : i32
      memref.store %204, %_anonymous7[%c1] : memref<3xi32>
      %205 = arith.addi %158, %c1 : index
      cf.br ^bb72(%205 : index)
    ^bb94:  // pred: ^bb72
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %206 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %207 = arith.addi %206, %c1_i32 : i32
      %208 = arith.cmpi sge, %207, %c1_i32 : i32
      %209 = arith.select %208, %206, %207 : i32
      memref.store %209, %_anonymous7[%c0] : memref<3xi32>
      %210 = arith.addi %0, %c1 : index
      cf.br ^bb1(%210 : index)
    ^bb95:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"], stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<131072xi8>, %arg1: memref<4194304xi8>, %arg2: memref<524288xi16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 128, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 256, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 512, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 640, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 896, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 1048576, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 1179648, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 1310720, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 1441792, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 1572864, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 1703936, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 1835008, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 1966080, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 1024, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 1152, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 1280, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 1408, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 1536, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 1664, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 1792, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 1920, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 2097152, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 2228224, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 2359296, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 2490368, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 2621440, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 2752512, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 2883584, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 3014656, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 2048, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 2176, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 2304, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 2432, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 2560, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 2688, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 2816, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 2944, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 3145728, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 3276800, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 3407872, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 3538944, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 3670016, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 3801088, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 3932160, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 4063232, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 3072, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 3200, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 3328, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 3456, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 3584, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 3712, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 3840, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xi16>, 3968, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%67)
      aiex.dma_await_task(%9)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%16)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_await_task(%26)
      aiex.dma_await_task(%27)
      aiex.dma_await_task(%28)
      aiex.dma_await_task(%29)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%31)
      aiex.dma_await_task(%32)
      aiex.dma_await_task(%33)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%20)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%23)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%25)
      aiex.dma_await_task(%43)
      aiex.dma_await_task(%44)
      aiex.dma_await_task(%45)
      aiex.dma_await_task(%46)
      aiex.dma_await_task(%47)
      aiex.dma_await_task(%48)
      aiex.dma_await_task(%49)
      aiex.dma_await_task(%50)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%39)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%41)
      aiex.dma_free_task(%42)
      aiex.dma_await_task(%60)
      aiex.dma_await_task(%61)
      aiex.dma_await_task(%62)
      aiex.dma_await_task(%63)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%65)
      aiex.dma_await_task(%66)
      aiex.dma_await_task(%67)
      aiex.dma_free_task(%51)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%55)
      aiex.dma_free_task(%56)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%58)
      aiex.dma_free_task(%59)
    }
    aie.shim_dma_allocation @A_L3L2_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<128x512xi8>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<128x512xi8>, 0, 65536, [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<128x512xi8>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<128x512xi8>, 0, 65536, [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%C_L1L2_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_0_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%C_L1L2_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_0_cons_buff_0 : memref<128x128xi16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_0_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_0_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_0_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_0_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_0_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_0_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_0_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_0_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_0_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, Release, 1)
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
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_0_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_0_cons_buff_0 : memref<128x128xi16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L1L2_1_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_2_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_0_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_0_cons_buff_0 : memref<128x128xi16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L1L2_2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_3_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_0_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_0_cons_buff_0 : memref<128x128xi16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L1L2_3_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_4_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_4_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_0_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_0_cons_buff_0 : memref<128x128xi16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L1L2_4_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_5_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_5_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_0_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_0_cons_buff_0 : memref<128x128xi16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L1L2_5_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_6_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_6_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_0_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_0_cons_buff_0 : memref<128x128xi16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L1L2_6_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @B_L3L2_7_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L3L2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_7_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L2_7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_0_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_0_cons_buff_0 : memref<128x128xi16>, 0, 16384, [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%C_L1L2_7_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
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
