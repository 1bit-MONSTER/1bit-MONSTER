module {
  aie.device(npu2) {
    func.func private @zero_i16(memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
    func.func private @matmul_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
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
    %C_L1L2_7_0_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 0 : i32, sym_name = "C_L1L2_7_0_cons_buff_0"} : memref<32x128xi16> 
    %C_L1L2_7_0_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 2) {init = 1 : i32, sym_name = "C_L1L2_7_0_cons_prod_lock_0"}
    %C_L1L2_7_0_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 3) {init = 0 : i32, sym_name = "C_L1L2_7_0_cons_cons_lock_0"}
    %C_L1L2_7_0_buff_0 = aie.buffer(%tile_7_2) {address = 3328 : i32, sym_name = "C_L1L2_7_0_buff_0"} : memref<32x128xi16> 
    %C_L1L2_7_0_prod_lock_0 = aie.lock(%tile_7_2, 4) {init = 1 : i32, sym_name = "C_L1L2_7_0_prod_lock_0"}
    %C_L1L2_7_0_cons_lock_0 = aie.lock(%tile_7_2, 5) {init = 0 : i32, sym_name = "C_L1L2_7_0_cons_lock_0"}
    %C_L2L3_6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "C_L2L3_6_cons_prod_lock_0"}
    %C_L2L3_6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "C_L2L3_6_cons_cons_lock_0"}
    %C_L1L2_6_0_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 0 : i32, sym_name = "C_L1L2_6_0_cons_buff_0"} : memref<32x128xi16> 
    %C_L1L2_6_0_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 2) {init = 1 : i32, sym_name = "C_L1L2_6_0_cons_prod_lock_0"}
    %C_L1L2_6_0_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 3) {init = 0 : i32, sym_name = "C_L1L2_6_0_cons_cons_lock_0"}
    %C_L1L2_6_0_buff_0 = aie.buffer(%tile_6_2) {address = 3328 : i32, sym_name = "C_L1L2_6_0_buff_0"} : memref<32x128xi16> 
    %C_L1L2_6_0_prod_lock_0 = aie.lock(%tile_6_2, 4) {init = 1 : i32, sym_name = "C_L1L2_6_0_prod_lock_0"}
    %C_L1L2_6_0_cons_lock_0 = aie.lock(%tile_6_2, 5) {init = 0 : i32, sym_name = "C_L1L2_6_0_cons_lock_0"}
    %C_L2L3_5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "C_L2L3_5_cons_prod_lock_0"}
    %C_L2L3_5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "C_L2L3_5_cons_cons_lock_0"}
    %C_L1L2_5_0_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 0 : i32, sym_name = "C_L1L2_5_0_cons_buff_0"} : memref<32x128xi16> 
    %C_L1L2_5_0_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 2) {init = 1 : i32, sym_name = "C_L1L2_5_0_cons_prod_lock_0"}
    %C_L1L2_5_0_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 3) {init = 0 : i32, sym_name = "C_L1L2_5_0_cons_cons_lock_0"}
    %C_L1L2_5_0_buff_0 = aie.buffer(%tile_5_2) {address = 3328 : i32, sym_name = "C_L1L2_5_0_buff_0"} : memref<32x128xi16> 
    %C_L1L2_5_0_prod_lock_0 = aie.lock(%tile_5_2, 4) {init = 1 : i32, sym_name = "C_L1L2_5_0_prod_lock_0"}
    %C_L1L2_5_0_cons_lock_0 = aie.lock(%tile_5_2, 5) {init = 0 : i32, sym_name = "C_L1L2_5_0_cons_lock_0"}
    %C_L2L3_4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "C_L2L3_4_cons_prod_lock_0"}
    %C_L2L3_4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "C_L2L3_4_cons_cons_lock_0"}
    %C_L1L2_4_0_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 0 : i32, sym_name = "C_L1L2_4_0_cons_buff_0"} : memref<32x128xi16> 
    %C_L1L2_4_0_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 2) {init = 1 : i32, sym_name = "C_L1L2_4_0_cons_prod_lock_0"}
    %C_L1L2_4_0_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 3) {init = 0 : i32, sym_name = "C_L1L2_4_0_cons_cons_lock_0"}
    %C_L1L2_4_0_buff_0 = aie.buffer(%tile_4_2) {address = 3328 : i32, sym_name = "C_L1L2_4_0_buff_0"} : memref<32x128xi16> 
    %C_L1L2_4_0_prod_lock_0 = aie.lock(%tile_4_2, 4) {init = 1 : i32, sym_name = "C_L1L2_4_0_prod_lock_0"}
    %C_L1L2_4_0_cons_lock_0 = aie.lock(%tile_4_2, 5) {init = 0 : i32, sym_name = "C_L1L2_4_0_cons_lock_0"}
    %C_L2L3_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "C_L2L3_3_cons_prod_lock_0"}
    %C_L2L3_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "C_L2L3_3_cons_cons_lock_0"}
    %C_L1L2_3_0_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, sym_name = "C_L1L2_3_0_cons_buff_0"} : memref<32x128xi16> 
    %C_L1L2_3_0_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 2) {init = 1 : i32, sym_name = "C_L1L2_3_0_cons_prod_lock_0"}
    %C_L1L2_3_0_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "C_L1L2_3_0_cons_cons_lock_0"}
    %C_L1L2_3_0_buff_0 = aie.buffer(%tile_3_2) {address = 3328 : i32, sym_name = "C_L1L2_3_0_buff_0"} : memref<32x128xi16> 
    %C_L1L2_3_0_prod_lock_0 = aie.lock(%tile_3_2, 4) {init = 1 : i32, sym_name = "C_L1L2_3_0_prod_lock_0"}
    %C_L1L2_3_0_cons_lock_0 = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "C_L1L2_3_0_cons_lock_0"}
    %C_L2L3_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "C_L2L3_2_cons_prod_lock_0"}
    %C_L2L3_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "C_L2L3_2_cons_cons_lock_0"}
    %C_L1L2_2_0_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "C_L1L2_2_0_cons_buff_0"} : memref<32x128xi16> 
    %C_L1L2_2_0_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 2) {init = 1 : i32, sym_name = "C_L1L2_2_0_cons_prod_lock_0"}
    %C_L1L2_2_0_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "C_L1L2_2_0_cons_cons_lock_0"}
    %C_L1L2_2_0_buff_0 = aie.buffer(%tile_2_2) {address = 3328 : i32, sym_name = "C_L1L2_2_0_buff_0"} : memref<32x128xi16> 
    %C_L1L2_2_0_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 1 : i32, sym_name = "C_L1L2_2_0_prod_lock_0"}
    %C_L1L2_2_0_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "C_L1L2_2_0_cons_lock_0"}
    %C_L2L3_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "C_L2L3_1_cons_prod_lock_0"}
    %C_L2L3_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "C_L2L3_1_cons_cons_lock_0"}
    %C_L1L2_1_0_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "C_L1L2_1_0_cons_buff_0"} : memref<32x128xi16> 
    %C_L1L2_1_0_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 2) {init = 1 : i32, sym_name = "C_L1L2_1_0_cons_prod_lock_0"}
    %C_L1L2_1_0_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "C_L1L2_1_0_cons_cons_lock_0"}
    %C_L1L2_1_0_buff_0 = aie.buffer(%tile_1_2) {address = 3328 : i32, sym_name = "C_L1L2_1_0_buff_0"} : memref<32x128xi16> 
    %C_L1L2_1_0_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 1 : i32, sym_name = "C_L1L2_1_0_prod_lock_0"}
    %C_L1L2_1_0_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "C_L1L2_1_0_cons_lock_0"}
    %C_L2L3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L2L3_0_cons_prod_lock_0"}
    %C_L2L3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L2L3_0_cons_cons_lock_0"}
    %C_L1L2_0_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 32768 : i32, sym_name = "C_L1L2_0_0_cons_buff_0"} : memref<32x128xi16> 
    %C_L1L2_0_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32, sym_name = "C_L1L2_0_0_cons_prod_lock_0"}
    %C_L1L2_0_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "C_L1L2_0_0_cons_cons_lock_0"}
    %C_L1L2_0_0_buff_0 = aie.buffer(%tile_0_2) {address = 3328 : i32, sym_name = "C_L1L2_0_0_buff_0"} : memref<32x128xi16> 
    %C_L1L2_0_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "C_L1L2_0_0_prod_lock_0"}
    %C_L1L2_0_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_L1L2_0_0_cons_lock_0"}
    %B_L2L1_7_cons_buff_0 = aie.buffer(%tile_7_2) {address = 11520 : i32, sym_name = "B_L2L1_7_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_7_cons_buff_1 = aie.buffer(%tile_7_2) {address = 19712 : i32, sym_name = "B_L2L1_7_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_7_cons_prod_lock_0 = aie.lock(%tile_7_2, 2) {init = 2 : i32, sym_name = "B_L2L1_7_cons_prod_lock_0"}
    %B_L2L1_7_cons_cons_lock_0 = aie.lock(%tile_7_2, 3) {init = 0 : i32, sym_name = "B_L2L1_7_cons_cons_lock_0"}
    %B_L3L2_7_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 8192 : i32, sym_name = "B_L3L2_7_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_7_cons_buff_1 = aie.buffer(%mem_tile_7_1) {address = 16384 : i32, sym_name = "B_L3L2_7_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_7_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 0) {init = 2 : i32, sym_name = "B_L3L2_7_cons_prod_lock_0"}
    %B_L3L2_7_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 1) {init = 0 : i32, sym_name = "B_L3L2_7_cons_cons_lock_0"}
    %B_L3L2_7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "B_L3L2_7_prod_lock_0"}
    %B_L3L2_7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "B_L3L2_7_cons_lock_0"}
    %B_L2L1_6_cons_buff_0 = aie.buffer(%tile_6_2) {address = 11520 : i32, sym_name = "B_L2L1_6_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_6_cons_buff_1 = aie.buffer(%tile_6_2) {address = 19712 : i32, sym_name = "B_L2L1_6_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_6_cons_prod_lock_0 = aie.lock(%tile_6_2, 2) {init = 2 : i32, sym_name = "B_L2L1_6_cons_prod_lock_0"}
    %B_L2L1_6_cons_cons_lock_0 = aie.lock(%tile_6_2, 3) {init = 0 : i32, sym_name = "B_L2L1_6_cons_cons_lock_0"}
    %B_L3L2_6_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 8192 : i32, sym_name = "B_L3L2_6_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_6_cons_buff_1 = aie.buffer(%mem_tile_6_1) {address = 16384 : i32, sym_name = "B_L3L2_6_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_6_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 0) {init = 2 : i32, sym_name = "B_L3L2_6_cons_prod_lock_0"}
    %B_L3L2_6_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 1) {init = 0 : i32, sym_name = "B_L3L2_6_cons_cons_lock_0"}
    %B_L3L2_6_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "B_L3L2_6_prod_lock_0"}
    %B_L3L2_6_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "B_L3L2_6_cons_lock_0"}
    %B_L2L1_5_cons_buff_0 = aie.buffer(%tile_5_2) {address = 11520 : i32, sym_name = "B_L2L1_5_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_5_cons_buff_1 = aie.buffer(%tile_5_2) {address = 19712 : i32, sym_name = "B_L2L1_5_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_5_cons_prod_lock_0 = aie.lock(%tile_5_2, 2) {init = 2 : i32, sym_name = "B_L2L1_5_cons_prod_lock_0"}
    %B_L2L1_5_cons_cons_lock_0 = aie.lock(%tile_5_2, 3) {init = 0 : i32, sym_name = "B_L2L1_5_cons_cons_lock_0"}
    %B_L3L2_5_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 8192 : i32, sym_name = "B_L3L2_5_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_5_cons_buff_1 = aie.buffer(%mem_tile_5_1) {address = 16384 : i32, sym_name = "B_L3L2_5_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_5_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 0) {init = 2 : i32, sym_name = "B_L3L2_5_cons_prod_lock_0"}
    %B_L3L2_5_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 1) {init = 0 : i32, sym_name = "B_L3L2_5_cons_cons_lock_0"}
    %B_L3L2_5_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "B_L3L2_5_prod_lock_0"}
    %B_L3L2_5_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "B_L3L2_5_cons_lock_0"}
    %B_L2L1_4_cons_buff_0 = aie.buffer(%tile_4_2) {address = 11520 : i32, sym_name = "B_L2L1_4_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_4_cons_buff_1 = aie.buffer(%tile_4_2) {address = 19712 : i32, sym_name = "B_L2L1_4_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_4_cons_prod_lock_0 = aie.lock(%tile_4_2, 2) {init = 2 : i32, sym_name = "B_L2L1_4_cons_prod_lock_0"}
    %B_L2L1_4_cons_cons_lock_0 = aie.lock(%tile_4_2, 3) {init = 0 : i32, sym_name = "B_L2L1_4_cons_cons_lock_0"}
    %B_L3L2_4_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 8192 : i32, sym_name = "B_L3L2_4_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_4_cons_buff_1 = aie.buffer(%mem_tile_4_1) {address = 16384 : i32, sym_name = "B_L3L2_4_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_4_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 0) {init = 2 : i32, sym_name = "B_L3L2_4_cons_prod_lock_0"}
    %B_L3L2_4_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 1) {init = 0 : i32, sym_name = "B_L3L2_4_cons_cons_lock_0"}
    %B_L3L2_4_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "B_L3L2_4_prod_lock_0"}
    %B_L3L2_4_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "B_L3L2_4_cons_lock_0"}
    %B_L2L1_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 11520 : i32, sym_name = "B_L2L1_3_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 19712 : i32, sym_name = "B_L2L1_3_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "B_L2L1_3_cons_prod_lock_0"}
    %B_L2L1_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "B_L2L1_3_cons_cons_lock_0"}
    %B_L3L2_3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 8192 : i32, sym_name = "B_L3L2_3_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 16384 : i32, sym_name = "B_L3L2_3_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "B_L3L2_3_cons_prod_lock_0"}
    %B_L3L2_3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "B_L3L2_3_cons_cons_lock_0"}
    %B_L3L2_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "B_L3L2_3_prod_lock_0"}
    %B_L3L2_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "B_L3L2_3_cons_lock_0"}
    %B_L2L1_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 11520 : i32, sym_name = "B_L2L1_2_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 19712 : i32, sym_name = "B_L2L1_2_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "B_L2L1_2_cons_prod_lock_0"}
    %B_L2L1_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "B_L2L1_2_cons_cons_lock_0"}
    %B_L3L2_2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 8192 : i32, sym_name = "B_L3L2_2_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 16384 : i32, sym_name = "B_L3L2_2_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "B_L3L2_2_cons_prod_lock_0"}
    %B_L3L2_2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "B_L3L2_2_cons_cons_lock_0"}
    %B_L3L2_2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "B_L3L2_2_prod_lock_0"}
    %B_L3L2_2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "B_L3L2_2_cons_lock_0"}
    %B_L2L1_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 11520 : i32, sym_name = "B_L2L1_1_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 19712 : i32, sym_name = "B_L2L1_1_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "B_L2L1_1_cons_prod_lock_0"}
    %B_L2L1_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "B_L2L1_1_cons_cons_lock_0"}
    %B_L3L2_1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 8192 : i32, sym_name = "B_L3L2_1_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_1_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 16384 : i32, sym_name = "B_L3L2_1_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "B_L3L2_1_cons_prod_lock_0"}
    %B_L3L2_1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "B_L3L2_1_cons_cons_lock_0"}
    %B_L3L2_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "B_L3L2_1_prod_lock_0"}
    %B_L3L2_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "B_L3L2_1_cons_lock_0"}
    %B_L2L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 11520 : i32, sym_name = "B_L2L1_0_cons_buff_0"} : memref<64x128xi8> 
    %B_L2L1_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 19712 : i32, sym_name = "B_L2L1_0_cons_buff_1"} : memref<64x128xi8> 
    %B_L2L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "B_L2L1_0_cons_prod_lock_0"}
    %B_L2L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_L2L1_0_cons_cons_lock_0"}
    %B_L3L2_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 40960 : i32, sym_name = "B_L3L2_0_cons_buff_0"} : memref<64x128xi8> 
    %B_L3L2_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 49152 : i32, sym_name = "B_L3L2_0_cons_buff_1"} : memref<64x128xi8> 
    %B_L3L2_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "B_L3L2_0_cons_prod_lock_0"}
    %B_L3L2_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "B_L3L2_0_cons_cons_lock_0"}
    %B_L3L2_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_L3L2_0_prod_lock_0"}
    %B_L3L2_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_L3L2_0_cons_lock_0"}
    %A_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 27904 : i32, sym_name = "A_L2L1_0_0_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 29952 : i32, sym_name = "A_L2L1_0_0_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_0_cons_prod_lock_0"}
    %A_L2L1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_0_cons_cons_lock_0"}
    %A_L2L1_0_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 27904 : i32, sym_name = "A_L2L1_0_1_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 29952 : i32, sym_name = "A_L2L1_0_1_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_1_cons_prod_lock_0"}
    %A_L2L1_0_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_1_cons_cons_lock_0"}
    %A_L2L1_0_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 27904 : i32, sym_name = "A_L2L1_0_2_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 29952 : i32, sym_name = "A_L2L1_0_2_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_2_cons_prod_lock_0"}
    %A_L2L1_0_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_2_cons_cons_lock_0"}
    %A_L2L1_0_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 27904 : i32, sym_name = "A_L2L1_0_3_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 29952 : i32, sym_name = "A_L2L1_0_3_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_3_cons_prod_lock_0"}
    %A_L2L1_0_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_3_cons_cons_lock_0"}
    %A_L2L1_0_4_cons_buff_0 = aie.buffer(%tile_4_2) {address = 27904 : i32, sym_name = "A_L2L1_0_4_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_4_cons_buff_1 = aie.buffer(%tile_4_2) {address = 29952 : i32, sym_name = "A_L2L1_0_4_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_4_cons_prod_lock_0 = aie.lock(%tile_4_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_4_cons_prod_lock_0"}
    %A_L2L1_0_4_cons_cons_lock_0 = aie.lock(%tile_4_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_4_cons_cons_lock_0"}
    %A_L2L1_0_5_cons_buff_0 = aie.buffer(%tile_5_2) {address = 27904 : i32, sym_name = "A_L2L1_0_5_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_5_cons_buff_1 = aie.buffer(%tile_5_2) {address = 29952 : i32, sym_name = "A_L2L1_0_5_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_5_cons_prod_lock_0 = aie.lock(%tile_5_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_5_cons_prod_lock_0"}
    %A_L2L1_0_5_cons_cons_lock_0 = aie.lock(%tile_5_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_5_cons_cons_lock_0"}
    %A_L2L1_0_6_cons_buff_0 = aie.buffer(%tile_6_2) {address = 27904 : i32, sym_name = "A_L2L1_0_6_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_6_cons_buff_1 = aie.buffer(%tile_6_2) {address = 29952 : i32, sym_name = "A_L2L1_0_6_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_6_cons_prod_lock_0 = aie.lock(%tile_6_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_6_cons_prod_lock_0"}
    %A_L2L1_0_6_cons_cons_lock_0 = aie.lock(%tile_6_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_6_cons_cons_lock_0"}
    %A_L2L1_0_7_cons_buff_0 = aie.buffer(%tile_7_2) {address = 27904 : i32, sym_name = "A_L2L1_0_7_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_7_cons_buff_1 = aie.buffer(%tile_7_2) {address = 29952 : i32, sym_name = "A_L2L1_0_7_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_7_cons_prod_lock_0 = aie.lock(%tile_7_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_7_cons_prod_lock_0"}
    %A_L2L1_0_7_cons_cons_lock_0 = aie.lock(%tile_7_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_7_cons_cons_lock_0"}
    %A_L3L2_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "A_L3L2_0_cons_buff_0"} : memref<32x512xi8> 
    %A_L3L2_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 16384 : i32, sym_name = "A_L3L2_0_cons_buff_1"} : memref<32x512xi8> 
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
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 32000 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb266
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb267
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
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
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous0[%c1] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb19(%32: memref<64x128xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb23(%36: memref<32x64xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_i8_i16(%36, %32, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous0[%c1] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb30(%57: memref<64x128xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb34(%61: memref<32x64xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_i8_i16(%61, %57, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous0[%c1] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb41(%82: memref<64x128xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb45(%86: memref<32x64xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_i8_i16(%86, %82, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous0[%c1] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb47(%c0 : index)
    ^bb47(%102: index):  // 2 preds: ^bb46, ^bb56
      %103 = arith.cmpi slt, %102, %c16 : index
      cf.cond_br %103, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %104 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %105 = arith.index_cast %104 : i32 to index
      %106 = arith.index_cast %105 : index to i32
      cf.switch %106 : i32, [
        default: ^bb51,
        0: ^bb49,
        1: ^bb50
      ]
    ^bb49:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb50:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb51:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb52(%107: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb55,
        0: ^bb53,
        1: ^bb54
      ]
    ^bb53:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb54:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb55:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb56(%111: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_i8_i16(%111, %107, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %112 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %113 = arith.addi %112, %c1_i32 : i32
      %114 = arith.cmpi sge, %113, %c2_i32 : i32
      %115 = arith.subi %113, %c2_i32 : i32
      %116 = arith.select %114, %115, %113 : i32
      memref.store %116, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %117 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %118 = arith.addi %117, %c1_i32 : i32
      %119 = arith.cmpi sge, %118, %c2_i32 : i32
      %120 = arith.subi %118, %c2_i32 : i32
      %121 = arith.select %119, %120, %118 : i32
      memref.store %121, %_anonymous0[%c1] : memref<3xi32>
      %122 = arith.addi %102, %c1 : index
      cf.br ^bb47(%122 : index)
    ^bb57:  // pred: ^bb47
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %123 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c1_i32 : i32
      %126 = arith.select %125, %123, %124 : i32
      memref.store %126, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb58(%c0 : index)
    ^bb58(%127: index):  // 2 preds: ^bb57, ^bb67
      %128 = arith.cmpi slt, %127, %c16 : index
      cf.cond_br %128, ^bb59, ^bb68
    ^bb59:  // pred: ^bb58
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %129 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %130 = arith.index_cast %129 : i32 to index
      %131 = arith.index_cast %130 : index to i32
      cf.switch %131 : i32, [
        default: ^bb62,
        0: ^bb60,
        1: ^bb61
      ]
    ^bb60:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb61:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb62:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb63(%132: memref<64x128xi8>):  // 3 preds: ^bb60, ^bb61, ^bb62
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %133 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %134 = arith.index_cast %133 : i32 to index
      %135 = arith.index_cast %134 : index to i32
      cf.switch %135 : i32, [
        default: ^bb66,
        0: ^bb64,
        1: ^bb65
      ]
    ^bb64:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb66:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb67(%136: memref<32x64xi8>):  // 3 preds: ^bb64, ^bb65, ^bb66
      func.call @matmul_i8_i16(%136, %132, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %137 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %138 = arith.addi %137, %c1_i32 : i32
      %139 = arith.cmpi sge, %138, %c2_i32 : i32
      %140 = arith.subi %138, %c2_i32 : i32
      %141 = arith.select %139, %140, %138 : i32
      memref.store %141, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %142 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %143 = arith.addi %142, %c1_i32 : i32
      %144 = arith.cmpi sge, %143, %c2_i32 : i32
      %145 = arith.subi %143, %c2_i32 : i32
      %146 = arith.select %144, %145, %143 : i32
      memref.store %146, %_anonymous0[%c1] : memref<3xi32>
      %147 = arith.addi %127, %c1 : index
      cf.br ^bb58(%147 : index)
    ^bb68:  // pred: ^bb58
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %148 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c1_i32 : i32
      %151 = arith.select %150, %148, %149 : i32
      memref.store %151, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb69(%c0 : index)
    ^bb69(%152: index):  // 2 preds: ^bb68, ^bb78
      %153 = arith.cmpi slt, %152, %c16 : index
      cf.cond_br %153, ^bb70, ^bb79
    ^bb70:  // pred: ^bb69
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %154 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %155 = arith.index_cast %154 : i32 to index
      %156 = arith.index_cast %155 : index to i32
      cf.switch %156 : i32, [
        default: ^bb73,
        0: ^bb71,
        1: ^bb72
      ]
    ^bb71:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb72:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb73:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb74(%157: memref<64x128xi8>):  // 3 preds: ^bb71, ^bb72, ^bb73
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %158 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %159 = arith.index_cast %158 : i32 to index
      %160 = arith.index_cast %159 : index to i32
      cf.switch %160 : i32, [
        default: ^bb77,
        0: ^bb75,
        1: ^bb76
      ]
    ^bb75:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb76:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb77:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb78(%161: memref<32x64xi8>):  // 3 preds: ^bb75, ^bb76, ^bb77
      func.call @matmul_i8_i16(%161, %157, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %162 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %163 = arith.addi %162, %c1_i32 : i32
      %164 = arith.cmpi sge, %163, %c2_i32 : i32
      %165 = arith.subi %163, %c2_i32 : i32
      %166 = arith.select %164, %165, %163 : i32
      memref.store %166, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %167 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %168 = arith.addi %167, %c1_i32 : i32
      %169 = arith.cmpi sge, %168, %c2_i32 : i32
      %170 = arith.subi %168, %c2_i32 : i32
      %171 = arith.select %169, %170, %168 : i32
      memref.store %171, %_anonymous0[%c1] : memref<3xi32>
      %172 = arith.addi %152, %c1 : index
      cf.br ^bb69(%172 : index)
    ^bb79:  // pred: ^bb69
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %173 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %174 = arith.addi %173, %c1_i32 : i32
      %175 = arith.cmpi sge, %174, %c1_i32 : i32
      %176 = arith.select %175, %173, %174 : i32
      memref.store %176, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb80(%c0 : index)
    ^bb80(%177: index):  // 2 preds: ^bb79, ^bb89
      %178 = arith.cmpi slt, %177, %c16 : index
      cf.cond_br %178, ^bb81, ^bb90
    ^bb81:  // pred: ^bb80
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %179 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %180 = arith.index_cast %179 : i32 to index
      %181 = arith.index_cast %180 : index to i32
      cf.switch %181 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb85(%182: memref<64x128xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %183 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %184 = arith.index_cast %183 : i32 to index
      %185 = arith.index_cast %184 : index to i32
      cf.switch %185 : i32, [
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
    ^bb89(%186: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%186, %182, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %187 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %188 = arith.addi %187, %c1_i32 : i32
      %189 = arith.cmpi sge, %188, %c2_i32 : i32
      %190 = arith.subi %188, %c2_i32 : i32
      %191 = arith.select %189, %190, %188 : i32
      memref.store %191, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %192 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %193 = arith.addi %192, %c1_i32 : i32
      %194 = arith.cmpi sge, %193, %c2_i32 : i32
      %195 = arith.subi %193, %c2_i32 : i32
      %196 = arith.select %194, %195, %193 : i32
      memref.store %196, %_anonymous0[%c1] : memref<3xi32>
      %197 = arith.addi %177, %c1 : index
      cf.br ^bb80(%197 : index)
    ^bb90:  // pred: ^bb80
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %198 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %199 = arith.addi %198, %c1_i32 : i32
      %200 = arith.cmpi sge, %199, %c1_i32 : i32
      %201 = arith.select %200, %198, %199 : i32
      memref.store %201, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb91(%c0 : index)
    ^bb91(%202: index):  // 2 preds: ^bb90, ^bb100
      %203 = arith.cmpi slt, %202, %c16 : index
      cf.cond_br %203, ^bb92, ^bb101
    ^bb92:  // pred: ^bb91
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %204 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %205 = arith.index_cast %204 : i32 to index
      %206 = arith.index_cast %205 : index to i32
      cf.switch %206 : i32, [
        default: ^bb95,
        0: ^bb93,
        1: ^bb94
      ]
    ^bb93:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb94:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb95:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb96(%207: memref<64x128xi8>):  // 3 preds: ^bb93, ^bb94, ^bb95
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %208 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %209 = arith.index_cast %208 : i32 to index
      %210 = arith.index_cast %209 : index to i32
      cf.switch %210 : i32, [
        default: ^bb99,
        0: ^bb97,
        1: ^bb98
      ]
    ^bb97:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb98:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb99:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb100(%211: memref<32x64xi8>):  // 3 preds: ^bb97, ^bb98, ^bb99
      func.call @matmul_i8_i16(%211, %207, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %212 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %213 = arith.addi %212, %c1_i32 : i32
      %214 = arith.cmpi sge, %213, %c2_i32 : i32
      %215 = arith.subi %213, %c2_i32 : i32
      %216 = arith.select %214, %215, %213 : i32
      memref.store %216, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %217 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %218 = arith.addi %217, %c1_i32 : i32
      %219 = arith.cmpi sge, %218, %c2_i32 : i32
      %220 = arith.subi %218, %c2_i32 : i32
      %221 = arith.select %219, %220, %218 : i32
      memref.store %221, %_anonymous0[%c1] : memref<3xi32>
      %222 = arith.addi %202, %c1 : index
      cf.br ^bb91(%222 : index)
    ^bb101:  // pred: ^bb91
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %223 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %224 = arith.addi %223, %c1_i32 : i32
      %225 = arith.cmpi sge, %224, %c1_i32 : i32
      %226 = arith.select %225, %223, %224 : i32
      memref.store %226, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb102(%c0 : index)
    ^bb102(%227: index):  // 2 preds: ^bb101, ^bb111
      %228 = arith.cmpi slt, %227, %c16 : index
      cf.cond_br %228, ^bb103, ^bb112
    ^bb103:  // pred: ^bb102
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %229 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %230 = arith.index_cast %229 : i32 to index
      %231 = arith.index_cast %230 : index to i32
      cf.switch %231 : i32, [
        default: ^bb106,
        0: ^bb104,
        1: ^bb105
      ]
    ^bb104:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb105:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb106:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb107(%232: memref<64x128xi8>):  // 3 preds: ^bb104, ^bb105, ^bb106
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %233 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %234 = arith.index_cast %233 : i32 to index
      %235 = arith.index_cast %234 : index to i32
      cf.switch %235 : i32, [
        default: ^bb110,
        0: ^bb108,
        1: ^bb109
      ]
    ^bb108:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb109:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb110:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb111(%236: memref<32x64xi8>):  // 3 preds: ^bb108, ^bb109, ^bb110
      func.call @matmul_i8_i16(%236, %232, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %237 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %238 = arith.addi %237, %c1_i32 : i32
      %239 = arith.cmpi sge, %238, %c2_i32 : i32
      %240 = arith.subi %238, %c2_i32 : i32
      %241 = arith.select %239, %240, %238 : i32
      memref.store %241, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %242 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %243 = arith.addi %242, %c1_i32 : i32
      %244 = arith.cmpi sge, %243, %c2_i32 : i32
      %245 = arith.subi %243, %c2_i32 : i32
      %246 = arith.select %244, %245, %243 : i32
      memref.store %246, %_anonymous0[%c1] : memref<3xi32>
      %247 = arith.addi %227, %c1 : index
      cf.br ^bb102(%247 : index)
    ^bb112:  // pred: ^bb102
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %248 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %249 = arith.addi %248, %c1_i32 : i32
      %250 = arith.cmpi sge, %249, %c1_i32 : i32
      %251 = arith.select %250, %248, %249 : i32
      memref.store %251, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb113(%c0 : index)
    ^bb113(%252: index):  // 2 preds: ^bb112, ^bb122
      %253 = arith.cmpi slt, %252, %c16 : index
      cf.cond_br %253, ^bb114, ^bb123
    ^bb114:  // pred: ^bb113
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %254 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %255 = arith.index_cast %254 : i32 to index
      %256 = arith.index_cast %255 : index to i32
      cf.switch %256 : i32, [
        default: ^bb117,
        0: ^bb115,
        1: ^bb116
      ]
    ^bb115:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb116:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb117:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb118(%257: memref<64x128xi8>):  // 3 preds: ^bb115, ^bb116, ^bb117
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %258 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %259 = arith.index_cast %258 : i32 to index
      %260 = arith.index_cast %259 : index to i32
      cf.switch %260 : i32, [
        default: ^bb121,
        0: ^bb119,
        1: ^bb120
      ]
    ^bb119:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb120:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb121:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb122(%261: memref<32x64xi8>):  // 3 preds: ^bb119, ^bb120, ^bb121
      func.call @matmul_i8_i16(%261, %257, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %262 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %263 = arith.addi %262, %c1_i32 : i32
      %264 = arith.cmpi sge, %263, %c2_i32 : i32
      %265 = arith.subi %263, %c2_i32 : i32
      %266 = arith.select %264, %265, %263 : i32
      memref.store %266, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %267 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %268 = arith.addi %267, %c1_i32 : i32
      %269 = arith.cmpi sge, %268, %c2_i32 : i32
      %270 = arith.subi %268, %c2_i32 : i32
      %271 = arith.select %269, %270, %268 : i32
      memref.store %271, %_anonymous0[%c1] : memref<3xi32>
      %272 = arith.addi %252, %c1 : index
      cf.br ^bb113(%272 : index)
    ^bb123:  // pred: ^bb113
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %273 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %274 = arith.addi %273, %c1_i32 : i32
      %275 = arith.cmpi sge, %274, %c1_i32 : i32
      %276 = arith.select %275, %273, %274 : i32
      memref.store %276, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb124(%c0 : index)
    ^bb124(%277: index):  // 2 preds: ^bb123, ^bb133
      %278 = arith.cmpi slt, %277, %c16 : index
      cf.cond_br %278, ^bb125, ^bb134
    ^bb125:  // pred: ^bb124
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %279 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %280 = arith.index_cast %279 : i32 to index
      %281 = arith.index_cast %280 : index to i32
      cf.switch %281 : i32, [
        default: ^bb128,
        0: ^bb126,
        1: ^bb127
      ]
    ^bb126:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb127:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb128:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb129(%282: memref<64x128xi8>):  // 3 preds: ^bb126, ^bb127, ^bb128
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %283 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %284 = arith.index_cast %283 : i32 to index
      %285 = arith.index_cast %284 : index to i32
      cf.switch %285 : i32, [
        default: ^bb132,
        0: ^bb130,
        1: ^bb131
      ]
    ^bb130:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb131:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb132:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb133(%286: memref<32x64xi8>):  // 3 preds: ^bb130, ^bb131, ^bb132
      func.call @matmul_i8_i16(%286, %282, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %287 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %288 = arith.addi %287, %c1_i32 : i32
      %289 = arith.cmpi sge, %288, %c2_i32 : i32
      %290 = arith.subi %288, %c2_i32 : i32
      %291 = arith.select %289, %290, %288 : i32
      memref.store %291, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %292 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %293 = arith.addi %292, %c1_i32 : i32
      %294 = arith.cmpi sge, %293, %c2_i32 : i32
      %295 = arith.subi %293, %c2_i32 : i32
      %296 = arith.select %294, %295, %293 : i32
      memref.store %296, %_anonymous0[%c1] : memref<3xi32>
      %297 = arith.addi %277, %c1 : index
      cf.br ^bb124(%297 : index)
    ^bb134:  // pred: ^bb124
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %298 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %299 = arith.addi %298, %c1_i32 : i32
      %300 = arith.cmpi sge, %299, %c1_i32 : i32
      %301 = arith.select %300, %298, %299 : i32
      memref.store %301, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb135(%c0 : index)
    ^bb135(%302: index):  // 2 preds: ^bb134, ^bb144
      %303 = arith.cmpi slt, %302, %c16 : index
      cf.cond_br %303, ^bb136, ^bb145
    ^bb136:  // pred: ^bb135
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %304 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %305 = arith.index_cast %304 : i32 to index
      %306 = arith.index_cast %305 : index to i32
      cf.switch %306 : i32, [
        default: ^bb139,
        0: ^bb137,
        1: ^bb138
      ]
    ^bb137:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb138:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb139:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb140(%307: memref<64x128xi8>):  // 3 preds: ^bb137, ^bb138, ^bb139
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %308 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %309 = arith.index_cast %308 : i32 to index
      %310 = arith.index_cast %309 : index to i32
      cf.switch %310 : i32, [
        default: ^bb143,
        0: ^bb141,
        1: ^bb142
      ]
    ^bb141:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb142:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb143:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb144(%311: memref<32x64xi8>):  // 3 preds: ^bb141, ^bb142, ^bb143
      func.call @matmul_i8_i16(%311, %307, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %312 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %313 = arith.addi %312, %c1_i32 : i32
      %314 = arith.cmpi sge, %313, %c2_i32 : i32
      %315 = arith.subi %313, %c2_i32 : i32
      %316 = arith.select %314, %315, %313 : i32
      memref.store %316, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %317 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %318 = arith.addi %317, %c1_i32 : i32
      %319 = arith.cmpi sge, %318, %c2_i32 : i32
      %320 = arith.subi %318, %c2_i32 : i32
      %321 = arith.select %319, %320, %318 : i32
      memref.store %321, %_anonymous0[%c1] : memref<3xi32>
      %322 = arith.addi %302, %c1 : index
      cf.br ^bb135(%322 : index)
    ^bb145:  // pred: ^bb135
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %323 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %324 = arith.addi %323, %c1_i32 : i32
      %325 = arith.cmpi sge, %324, %c1_i32 : i32
      %326 = arith.select %325, %323, %324 : i32
      memref.store %326, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb146(%c0 : index)
    ^bb146(%327: index):  // 2 preds: ^bb145, ^bb155
      %328 = arith.cmpi slt, %327, %c16 : index
      cf.cond_br %328, ^bb147, ^bb156
    ^bb147:  // pred: ^bb146
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %329 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %330 = arith.index_cast %329 : i32 to index
      %331 = arith.index_cast %330 : index to i32
      cf.switch %331 : i32, [
        default: ^bb150,
        0: ^bb148,
        1: ^bb149
      ]
    ^bb148:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb149:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb150:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb151(%332: memref<64x128xi8>):  // 3 preds: ^bb148, ^bb149, ^bb150
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %333 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %334 = arith.index_cast %333 : i32 to index
      %335 = arith.index_cast %334 : index to i32
      cf.switch %335 : i32, [
        default: ^bb154,
        0: ^bb152,
        1: ^bb153
      ]
    ^bb152:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb153:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb154:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb155(%336: memref<32x64xi8>):  // 3 preds: ^bb152, ^bb153, ^bb154
      func.call @matmul_i8_i16(%336, %332, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %337 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %338 = arith.addi %337, %c1_i32 : i32
      %339 = arith.cmpi sge, %338, %c2_i32 : i32
      %340 = arith.subi %338, %c2_i32 : i32
      %341 = arith.select %339, %340, %338 : i32
      memref.store %341, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %342 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %343 = arith.addi %342, %c1_i32 : i32
      %344 = arith.cmpi sge, %343, %c2_i32 : i32
      %345 = arith.subi %343, %c2_i32 : i32
      %346 = arith.select %344, %345, %343 : i32
      memref.store %346, %_anonymous0[%c1] : memref<3xi32>
      %347 = arith.addi %327, %c1 : index
      cf.br ^bb146(%347 : index)
    ^bb156:  // pred: ^bb146
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %348 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %349 = arith.addi %348, %c1_i32 : i32
      %350 = arith.cmpi sge, %349, %c1_i32 : i32
      %351 = arith.select %350, %348, %349 : i32
      memref.store %351, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb157(%c0 : index)
    ^bb157(%352: index):  // 2 preds: ^bb156, ^bb166
      %353 = arith.cmpi slt, %352, %c16 : index
      cf.cond_br %353, ^bb158, ^bb167
    ^bb158:  // pred: ^bb157
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %354 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %355 = arith.index_cast %354 : i32 to index
      %356 = arith.index_cast %355 : index to i32
      cf.switch %356 : i32, [
        default: ^bb161,
        0: ^bb159,
        1: ^bb160
      ]
    ^bb159:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb160:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb161:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb162(%357: memref<64x128xi8>):  // 3 preds: ^bb159, ^bb160, ^bb161
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %358 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %359 = arith.index_cast %358 : i32 to index
      %360 = arith.index_cast %359 : index to i32
      cf.switch %360 : i32, [
        default: ^bb165,
        0: ^bb163,
        1: ^bb164
      ]
    ^bb163:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb164:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb165:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb166(%361: memref<32x64xi8>):  // 3 preds: ^bb163, ^bb164, ^bb165
      func.call @matmul_i8_i16(%361, %357, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %362 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %363 = arith.addi %362, %c1_i32 : i32
      %364 = arith.cmpi sge, %363, %c2_i32 : i32
      %365 = arith.subi %363, %c2_i32 : i32
      %366 = arith.select %364, %365, %363 : i32
      memref.store %366, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %367 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %368 = arith.addi %367, %c1_i32 : i32
      %369 = arith.cmpi sge, %368, %c2_i32 : i32
      %370 = arith.subi %368, %c2_i32 : i32
      %371 = arith.select %369, %370, %368 : i32
      memref.store %371, %_anonymous0[%c1] : memref<3xi32>
      %372 = arith.addi %352, %c1 : index
      cf.br ^bb157(%372 : index)
    ^bb167:  // pred: ^bb157
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %373 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %374 = arith.addi %373, %c1_i32 : i32
      %375 = arith.cmpi sge, %374, %c1_i32 : i32
      %376 = arith.select %375, %373, %374 : i32
      memref.store %376, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb168(%c0 : index)
    ^bb168(%377: index):  // 2 preds: ^bb167, ^bb177
      %378 = arith.cmpi slt, %377, %c16 : index
      cf.cond_br %378, ^bb169, ^bb178
    ^bb169:  // pred: ^bb168
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %379 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %380 = arith.index_cast %379 : i32 to index
      %381 = arith.index_cast %380 : index to i32
      cf.switch %381 : i32, [
        default: ^bb172,
        0: ^bb170,
        1: ^bb171
      ]
    ^bb170:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb171:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb172:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb173(%382: memref<64x128xi8>):  // 3 preds: ^bb170, ^bb171, ^bb172
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %383 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %384 = arith.index_cast %383 : i32 to index
      %385 = arith.index_cast %384 : index to i32
      cf.switch %385 : i32, [
        default: ^bb176,
        0: ^bb174,
        1: ^bb175
      ]
    ^bb174:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb175:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb176:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb177(%386: memref<32x64xi8>):  // 3 preds: ^bb174, ^bb175, ^bb176
      func.call @matmul_i8_i16(%386, %382, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %387 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %388 = arith.addi %387, %c1_i32 : i32
      %389 = arith.cmpi sge, %388, %c2_i32 : i32
      %390 = arith.subi %388, %c2_i32 : i32
      %391 = arith.select %389, %390, %388 : i32
      memref.store %391, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %392 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %393 = arith.addi %392, %c1_i32 : i32
      %394 = arith.cmpi sge, %393, %c2_i32 : i32
      %395 = arith.subi %393, %c2_i32 : i32
      %396 = arith.select %394, %395, %393 : i32
      memref.store %396, %_anonymous0[%c1] : memref<3xi32>
      %397 = arith.addi %377, %c1 : index
      cf.br ^bb168(%397 : index)
    ^bb178:  // pred: ^bb168
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %398 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %399 = arith.addi %398, %c1_i32 : i32
      %400 = arith.cmpi sge, %399, %c1_i32 : i32
      %401 = arith.select %400, %398, %399 : i32
      memref.store %401, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb179(%c0 : index)
    ^bb179(%402: index):  // 2 preds: ^bb178, ^bb188
      %403 = arith.cmpi slt, %402, %c16 : index
      cf.cond_br %403, ^bb180, ^bb189
    ^bb180:  // pred: ^bb179
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %404 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %405 = arith.index_cast %404 : i32 to index
      %406 = arith.index_cast %405 : index to i32
      cf.switch %406 : i32, [
        default: ^bb183,
        0: ^bb181,
        1: ^bb182
      ]
    ^bb181:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb182:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb183:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb184(%407: memref<64x128xi8>):  // 3 preds: ^bb181, ^bb182, ^bb183
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %408 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %409 = arith.index_cast %408 : i32 to index
      %410 = arith.index_cast %409 : index to i32
      cf.switch %410 : i32, [
        default: ^bb187,
        0: ^bb185,
        1: ^bb186
      ]
    ^bb185:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb186:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb187:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb188(%411: memref<32x64xi8>):  // 3 preds: ^bb185, ^bb186, ^bb187
      func.call @matmul_i8_i16(%411, %407, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %412 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %413 = arith.addi %412, %c1_i32 : i32
      %414 = arith.cmpi sge, %413, %c2_i32 : i32
      %415 = arith.subi %413, %c2_i32 : i32
      %416 = arith.select %414, %415, %413 : i32
      memref.store %416, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %417 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %418 = arith.addi %417, %c1_i32 : i32
      %419 = arith.cmpi sge, %418, %c2_i32 : i32
      %420 = arith.subi %418, %c2_i32 : i32
      %421 = arith.select %419, %420, %418 : i32
      memref.store %421, %_anonymous0[%c1] : memref<3xi32>
      %422 = arith.addi %402, %c1 : index
      cf.br ^bb179(%422 : index)
    ^bb189:  // pred: ^bb179
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %423 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %424 = arith.addi %423, %c1_i32 : i32
      %425 = arith.cmpi sge, %424, %c1_i32 : i32
      %426 = arith.select %425, %423, %424 : i32
      memref.store %426, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb190(%c0 : index)
    ^bb190(%427: index):  // 2 preds: ^bb189, ^bb199
      %428 = arith.cmpi slt, %427, %c16 : index
      cf.cond_br %428, ^bb191, ^bb200
    ^bb191:  // pred: ^bb190
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %429 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %430 = arith.index_cast %429 : i32 to index
      %431 = arith.index_cast %430 : index to i32
      cf.switch %431 : i32, [
        default: ^bb194,
        0: ^bb192,
        1: ^bb193
      ]
    ^bb192:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb193:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb194:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb195(%432: memref<64x128xi8>):  // 3 preds: ^bb192, ^bb193, ^bb194
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %433 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %434 = arith.index_cast %433 : i32 to index
      %435 = arith.index_cast %434 : index to i32
      cf.switch %435 : i32, [
        default: ^bb198,
        0: ^bb196,
        1: ^bb197
      ]
    ^bb196:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb197:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb198:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb199(%436: memref<32x64xi8>):  // 3 preds: ^bb196, ^bb197, ^bb198
      func.call @matmul_i8_i16(%436, %432, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %437 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %438 = arith.addi %437, %c1_i32 : i32
      %439 = arith.cmpi sge, %438, %c2_i32 : i32
      %440 = arith.subi %438, %c2_i32 : i32
      %441 = arith.select %439, %440, %438 : i32
      memref.store %441, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %442 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %443 = arith.addi %442, %c1_i32 : i32
      %444 = arith.cmpi sge, %443, %c2_i32 : i32
      %445 = arith.subi %443, %c2_i32 : i32
      %446 = arith.select %444, %445, %443 : i32
      memref.store %446, %_anonymous0[%c1] : memref<3xi32>
      %447 = arith.addi %427, %c1 : index
      cf.br ^bb190(%447 : index)
    ^bb200:  // pred: ^bb190
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %448 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %449 = arith.addi %448, %c1_i32 : i32
      %450 = arith.cmpi sge, %449, %c1_i32 : i32
      %451 = arith.select %450, %448, %449 : i32
      memref.store %451, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb201(%c0 : index)
    ^bb201(%452: index):  // 2 preds: ^bb200, ^bb210
      %453 = arith.cmpi slt, %452, %c16 : index
      cf.cond_br %453, ^bb202, ^bb211
    ^bb202:  // pred: ^bb201
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %454 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %455 = arith.index_cast %454 : i32 to index
      %456 = arith.index_cast %455 : index to i32
      cf.switch %456 : i32, [
        default: ^bb205,
        0: ^bb203,
        1: ^bb204
      ]
    ^bb203:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb204:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb205:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb206(%457: memref<64x128xi8>):  // 3 preds: ^bb203, ^bb204, ^bb205
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %458 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %459 = arith.index_cast %458 : i32 to index
      %460 = arith.index_cast %459 : index to i32
      cf.switch %460 : i32, [
        default: ^bb209,
        0: ^bb207,
        1: ^bb208
      ]
    ^bb207:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb208:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb209:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb210(%461: memref<32x64xi8>):  // 3 preds: ^bb207, ^bb208, ^bb209
      func.call @matmul_i8_i16(%461, %457, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %462 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %463 = arith.addi %462, %c1_i32 : i32
      %464 = arith.cmpi sge, %463, %c2_i32 : i32
      %465 = arith.subi %463, %c2_i32 : i32
      %466 = arith.select %464, %465, %463 : i32
      memref.store %466, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %467 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %468 = arith.addi %467, %c1_i32 : i32
      %469 = arith.cmpi sge, %468, %c2_i32 : i32
      %470 = arith.subi %468, %c2_i32 : i32
      %471 = arith.select %469, %470, %468 : i32
      memref.store %471, %_anonymous0[%c1] : memref<3xi32>
      %472 = arith.addi %452, %c1 : index
      cf.br ^bb201(%472 : index)
    ^bb211:  // pred: ^bb201
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %473 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %474 = arith.addi %473, %c1_i32 : i32
      %475 = arith.cmpi sge, %474, %c1_i32 : i32
      %476 = arith.select %475, %473, %474 : i32
      memref.store %476, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb212(%c0 : index)
    ^bb212(%477: index):  // 2 preds: ^bb211, ^bb221
      %478 = arith.cmpi slt, %477, %c16 : index
      cf.cond_br %478, ^bb213, ^bb222
    ^bb213:  // pred: ^bb212
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %479 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %480 = arith.index_cast %479 : i32 to index
      %481 = arith.index_cast %480 : index to i32
      cf.switch %481 : i32, [
        default: ^bb216,
        0: ^bb214,
        1: ^bb215
      ]
    ^bb214:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb215:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb216:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb217(%482: memref<64x128xi8>):  // 3 preds: ^bb214, ^bb215, ^bb216
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %483 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %484 = arith.index_cast %483 : i32 to index
      %485 = arith.index_cast %484 : index to i32
      cf.switch %485 : i32, [
        default: ^bb220,
        0: ^bb218,
        1: ^bb219
      ]
    ^bb218:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb219:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb220:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb221(%486: memref<32x64xi8>):  // 3 preds: ^bb218, ^bb219, ^bb220
      func.call @matmul_i8_i16(%486, %482, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %487 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %488 = arith.addi %487, %c1_i32 : i32
      %489 = arith.cmpi sge, %488, %c2_i32 : i32
      %490 = arith.subi %488, %c2_i32 : i32
      %491 = arith.select %489, %490, %488 : i32
      memref.store %491, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %492 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %493 = arith.addi %492, %c1_i32 : i32
      %494 = arith.cmpi sge, %493, %c2_i32 : i32
      %495 = arith.subi %493, %c2_i32 : i32
      %496 = arith.select %494, %495, %493 : i32
      memref.store %496, %_anonymous0[%c1] : memref<3xi32>
      %497 = arith.addi %477, %c1 : index
      cf.br ^bb212(%497 : index)
    ^bb222:  // pred: ^bb212
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %498 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %499 = arith.addi %498, %c1_i32 : i32
      %500 = arith.cmpi sge, %499, %c1_i32 : i32
      %501 = arith.select %500, %498, %499 : i32
      memref.store %501, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb223(%c0 : index)
    ^bb223(%502: index):  // 2 preds: ^bb222, ^bb232
      %503 = arith.cmpi slt, %502, %c16 : index
      cf.cond_br %503, ^bb224, ^bb233
    ^bb224:  // pred: ^bb223
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %504 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %505 = arith.index_cast %504 : i32 to index
      %506 = arith.index_cast %505 : index to i32
      cf.switch %506 : i32, [
        default: ^bb227,
        0: ^bb225,
        1: ^bb226
      ]
    ^bb225:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb226:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb227:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb228(%507: memref<64x128xi8>):  // 3 preds: ^bb225, ^bb226, ^bb227
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %508 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %509 = arith.index_cast %508 : i32 to index
      %510 = arith.index_cast %509 : index to i32
      cf.switch %510 : i32, [
        default: ^bb231,
        0: ^bb229,
        1: ^bb230
      ]
    ^bb229:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb230:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb231:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb232(%511: memref<32x64xi8>):  // 3 preds: ^bb229, ^bb230, ^bb231
      func.call @matmul_i8_i16(%511, %507, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %512 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %513 = arith.addi %512, %c1_i32 : i32
      %514 = arith.cmpi sge, %513, %c2_i32 : i32
      %515 = arith.subi %513, %c2_i32 : i32
      %516 = arith.select %514, %515, %513 : i32
      memref.store %516, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %517 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %518 = arith.addi %517, %c1_i32 : i32
      %519 = arith.cmpi sge, %518, %c2_i32 : i32
      %520 = arith.subi %518, %c2_i32 : i32
      %521 = arith.select %519, %520, %518 : i32
      memref.store %521, %_anonymous0[%c1] : memref<3xi32>
      %522 = arith.addi %502, %c1 : index
      cf.br ^bb223(%522 : index)
    ^bb233:  // pred: ^bb223
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %523 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %524 = arith.addi %523, %c1_i32 : i32
      %525 = arith.cmpi sge, %524, %c1_i32 : i32
      %526 = arith.select %525, %523, %524 : i32
      memref.store %526, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb234(%c0 : index)
    ^bb234(%527: index):  // 2 preds: ^bb233, ^bb243
      %528 = arith.cmpi slt, %527, %c16 : index
      cf.cond_br %528, ^bb235, ^bb244
    ^bb235:  // pred: ^bb234
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %529 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %530 = arith.index_cast %529 : i32 to index
      %531 = arith.index_cast %530 : index to i32
      cf.switch %531 : i32, [
        default: ^bb238,
        0: ^bb236,
        1: ^bb237
      ]
    ^bb236:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb237:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb238:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb239(%532: memref<64x128xi8>):  // 3 preds: ^bb236, ^bb237, ^bb238
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %533 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %534 = arith.index_cast %533 : i32 to index
      %535 = arith.index_cast %534 : index to i32
      cf.switch %535 : i32, [
        default: ^bb242,
        0: ^bb240,
        1: ^bb241
      ]
    ^bb240:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb241:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb242:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb243(%536: memref<32x64xi8>):  // 3 preds: ^bb240, ^bb241, ^bb242
      func.call @matmul_i8_i16(%536, %532, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %537 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %538 = arith.addi %537, %c1_i32 : i32
      %539 = arith.cmpi sge, %538, %c2_i32 : i32
      %540 = arith.subi %538, %c2_i32 : i32
      %541 = arith.select %539, %540, %538 : i32
      memref.store %541, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %542 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %543 = arith.addi %542, %c1_i32 : i32
      %544 = arith.cmpi sge, %543, %c2_i32 : i32
      %545 = arith.subi %543, %c2_i32 : i32
      %546 = arith.select %544, %545, %543 : i32
      memref.store %546, %_anonymous0[%c1] : memref<3xi32>
      %547 = arith.addi %527, %c1 : index
      cf.br ^bb234(%547 : index)
    ^bb244:  // pred: ^bb234
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %548 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %549 = arith.addi %548, %c1_i32 : i32
      %550 = arith.cmpi sge, %549, %c1_i32 : i32
      %551 = arith.select %550, %548, %549 : i32
      memref.store %551, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb245(%c0 : index)
    ^bb245(%552: index):  // 2 preds: ^bb244, ^bb254
      %553 = arith.cmpi slt, %552, %c16 : index
      cf.cond_br %553, ^bb246, ^bb255
    ^bb246:  // pred: ^bb245
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %554 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %555 = arith.index_cast %554 : i32 to index
      %556 = arith.index_cast %555 : index to i32
      cf.switch %556 : i32, [
        default: ^bb249,
        0: ^bb247,
        1: ^bb248
      ]
    ^bb247:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb248:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb249:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb250(%557: memref<64x128xi8>):  // 3 preds: ^bb247, ^bb248, ^bb249
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %558 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %559 = arith.index_cast %558 : i32 to index
      %560 = arith.index_cast %559 : index to i32
      cf.switch %560 : i32, [
        default: ^bb253,
        0: ^bb251,
        1: ^bb252
      ]
    ^bb251:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb252:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb253:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb254(%561: memref<32x64xi8>):  // 3 preds: ^bb251, ^bb252, ^bb253
      func.call @matmul_i8_i16(%561, %557, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %562 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %563 = arith.addi %562, %c1_i32 : i32
      %564 = arith.cmpi sge, %563, %c2_i32 : i32
      %565 = arith.subi %563, %c2_i32 : i32
      %566 = arith.select %564, %565, %563 : i32
      memref.store %566, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %567 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %568 = arith.addi %567, %c1_i32 : i32
      %569 = arith.cmpi sge, %568, %c2_i32 : i32
      %570 = arith.subi %568, %c2_i32 : i32
      %571 = arith.select %569, %570, %568 : i32
      memref.store %571, %_anonymous0[%c1] : memref<3xi32>
      %572 = arith.addi %552, %c1 : index
      cf.br ^bb245(%572 : index)
    ^bb255:  // pred: ^bb245
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %573 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %574 = arith.addi %573, %c1_i32 : i32
      %575 = arith.cmpi sge, %574, %c1_i32 : i32
      %576 = arith.select %575, %573, %574 : i32
      memref.store %576, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb256(%c0 : index)
    ^bb256(%577: index):  // 2 preds: ^bb255, ^bb265
      %578 = arith.cmpi slt, %577, %c16 : index
      cf.cond_br %578, ^bb257, ^bb266
    ^bb257:  // pred: ^bb256
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %579 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %580 = arith.index_cast %579 : i32 to index
      %581 = arith.index_cast %580 : index to i32
      cf.switch %581 : i32, [
        default: ^bb260,
        0: ^bb258,
        1: ^bb259
      ]
    ^bb258:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb259:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb260:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb261(%582: memref<64x128xi8>):  // 3 preds: ^bb258, ^bb259, ^bb260
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %583 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %584 = arith.index_cast %583 : i32 to index
      %585 = arith.index_cast %584 : index to i32
      cf.switch %585 : i32, [
        default: ^bb264,
        0: ^bb262,
        1: ^bb263
      ]
    ^bb262:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb263:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb264:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb265(%586: memref<32x64xi8>):  // 3 preds: ^bb262, ^bb263, ^bb264
      func.call @matmul_i8_i16(%586, %582, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %587 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %588 = arith.addi %587, %c1_i32 : i32
      %589 = arith.cmpi sge, %588, %c2_i32 : i32
      %590 = arith.subi %588, %c2_i32 : i32
      %591 = arith.select %589, %590, %588 : i32
      memref.store %591, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %592 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %593 = arith.addi %592, %c1_i32 : i32
      %594 = arith.cmpi sge, %593, %c2_i32 : i32
      %595 = arith.subi %593, %c2_i32 : i32
      %596 = arith.select %594, %595, %593 : i32
      memref.store %596, %_anonymous0[%c1] : memref<3xi32>
      %597 = arith.addi %577, %c1 : index
      cf.br ^bb256(%597 : index)
    ^bb266:  // pred: ^bb256
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %598 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %599 = arith.addi %598, %c1_i32 : i32
      %600 = arith.cmpi sge, %599, %c1_i32 : i32
      %601 = arith.select %600, %598, %599 : i32
      memref.store %601, %_anonymous0[%c0] : memref<3xi32>
      %602 = arith.addi %0, %c1 : index
      cf.br ^bb1(%602 : index)
    ^bb267:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous1 = aie.buffer(%tile_1_2) {address = 32000 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb266
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb267
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
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
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous1[%c1] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb19(%32: memref<64x128xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb23(%36: memref<32x64xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_i8_i16(%36, %32, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous1[%c1] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb30(%57: memref<64x128xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb34(%61: memref<32x64xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_i8_i16(%61, %57, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous1[%c1] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb41(%82: memref<64x128xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb45(%86: memref<32x64xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_i8_i16(%86, %82, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous1[%c1] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb47(%c0 : index)
    ^bb47(%102: index):  // 2 preds: ^bb46, ^bb56
      %103 = arith.cmpi slt, %102, %c16 : index
      cf.cond_br %103, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %104 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %105 = arith.index_cast %104 : i32 to index
      %106 = arith.index_cast %105 : index to i32
      cf.switch %106 : i32, [
        default: ^bb51,
        0: ^bb49,
        1: ^bb50
      ]
    ^bb49:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb50:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb51:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb52(%107: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb55,
        0: ^bb53,
        1: ^bb54
      ]
    ^bb53:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb54:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb55:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb56(%111: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_i8_i16(%111, %107, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %112 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %113 = arith.addi %112, %c1_i32 : i32
      %114 = arith.cmpi sge, %113, %c2_i32 : i32
      %115 = arith.subi %113, %c2_i32 : i32
      %116 = arith.select %114, %115, %113 : i32
      memref.store %116, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %117 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %118 = arith.addi %117, %c1_i32 : i32
      %119 = arith.cmpi sge, %118, %c2_i32 : i32
      %120 = arith.subi %118, %c2_i32 : i32
      %121 = arith.select %119, %120, %118 : i32
      memref.store %121, %_anonymous1[%c1] : memref<3xi32>
      %122 = arith.addi %102, %c1 : index
      cf.br ^bb47(%122 : index)
    ^bb57:  // pred: ^bb47
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %123 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c1_i32 : i32
      %126 = arith.select %125, %123, %124 : i32
      memref.store %126, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb58(%c0 : index)
    ^bb58(%127: index):  // 2 preds: ^bb57, ^bb67
      %128 = arith.cmpi slt, %127, %c16 : index
      cf.cond_br %128, ^bb59, ^bb68
    ^bb59:  // pred: ^bb58
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %129 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %130 = arith.index_cast %129 : i32 to index
      %131 = arith.index_cast %130 : index to i32
      cf.switch %131 : i32, [
        default: ^bb62,
        0: ^bb60,
        1: ^bb61
      ]
    ^bb60:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb61:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb62:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb63(%132: memref<64x128xi8>):  // 3 preds: ^bb60, ^bb61, ^bb62
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %133 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %134 = arith.index_cast %133 : i32 to index
      %135 = arith.index_cast %134 : index to i32
      cf.switch %135 : i32, [
        default: ^bb66,
        0: ^bb64,
        1: ^bb65
      ]
    ^bb64:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb66:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb67(%136: memref<32x64xi8>):  // 3 preds: ^bb64, ^bb65, ^bb66
      func.call @matmul_i8_i16(%136, %132, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %137 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %138 = arith.addi %137, %c1_i32 : i32
      %139 = arith.cmpi sge, %138, %c2_i32 : i32
      %140 = arith.subi %138, %c2_i32 : i32
      %141 = arith.select %139, %140, %138 : i32
      memref.store %141, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %142 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %143 = arith.addi %142, %c1_i32 : i32
      %144 = arith.cmpi sge, %143, %c2_i32 : i32
      %145 = arith.subi %143, %c2_i32 : i32
      %146 = arith.select %144, %145, %143 : i32
      memref.store %146, %_anonymous1[%c1] : memref<3xi32>
      %147 = arith.addi %127, %c1 : index
      cf.br ^bb58(%147 : index)
    ^bb68:  // pred: ^bb58
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %148 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c1_i32 : i32
      %151 = arith.select %150, %148, %149 : i32
      memref.store %151, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb69(%c0 : index)
    ^bb69(%152: index):  // 2 preds: ^bb68, ^bb78
      %153 = arith.cmpi slt, %152, %c16 : index
      cf.cond_br %153, ^bb70, ^bb79
    ^bb70:  // pred: ^bb69
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %154 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %155 = arith.index_cast %154 : i32 to index
      %156 = arith.index_cast %155 : index to i32
      cf.switch %156 : i32, [
        default: ^bb73,
        0: ^bb71,
        1: ^bb72
      ]
    ^bb71:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb72:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb73:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb74(%157: memref<64x128xi8>):  // 3 preds: ^bb71, ^bb72, ^bb73
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %158 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %159 = arith.index_cast %158 : i32 to index
      %160 = arith.index_cast %159 : index to i32
      cf.switch %160 : i32, [
        default: ^bb77,
        0: ^bb75,
        1: ^bb76
      ]
    ^bb75:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb76:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb77:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb78(%161: memref<32x64xi8>):  // 3 preds: ^bb75, ^bb76, ^bb77
      func.call @matmul_i8_i16(%161, %157, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %162 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %163 = arith.addi %162, %c1_i32 : i32
      %164 = arith.cmpi sge, %163, %c2_i32 : i32
      %165 = arith.subi %163, %c2_i32 : i32
      %166 = arith.select %164, %165, %163 : i32
      memref.store %166, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %167 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %168 = arith.addi %167, %c1_i32 : i32
      %169 = arith.cmpi sge, %168, %c2_i32 : i32
      %170 = arith.subi %168, %c2_i32 : i32
      %171 = arith.select %169, %170, %168 : i32
      memref.store %171, %_anonymous1[%c1] : memref<3xi32>
      %172 = arith.addi %152, %c1 : index
      cf.br ^bb69(%172 : index)
    ^bb79:  // pred: ^bb69
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %173 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %174 = arith.addi %173, %c1_i32 : i32
      %175 = arith.cmpi sge, %174, %c1_i32 : i32
      %176 = arith.select %175, %173, %174 : i32
      memref.store %176, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb80(%c0 : index)
    ^bb80(%177: index):  // 2 preds: ^bb79, ^bb89
      %178 = arith.cmpi slt, %177, %c16 : index
      cf.cond_br %178, ^bb81, ^bb90
    ^bb81:  // pred: ^bb80
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %179 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %180 = arith.index_cast %179 : i32 to index
      %181 = arith.index_cast %180 : index to i32
      cf.switch %181 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb85(%182: memref<64x128xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %183 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %184 = arith.index_cast %183 : i32 to index
      %185 = arith.index_cast %184 : index to i32
      cf.switch %185 : i32, [
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
    ^bb89(%186: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%186, %182, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %187 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %188 = arith.addi %187, %c1_i32 : i32
      %189 = arith.cmpi sge, %188, %c2_i32 : i32
      %190 = arith.subi %188, %c2_i32 : i32
      %191 = arith.select %189, %190, %188 : i32
      memref.store %191, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %192 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %193 = arith.addi %192, %c1_i32 : i32
      %194 = arith.cmpi sge, %193, %c2_i32 : i32
      %195 = arith.subi %193, %c2_i32 : i32
      %196 = arith.select %194, %195, %193 : i32
      memref.store %196, %_anonymous1[%c1] : memref<3xi32>
      %197 = arith.addi %177, %c1 : index
      cf.br ^bb80(%197 : index)
    ^bb90:  // pred: ^bb80
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %198 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %199 = arith.addi %198, %c1_i32 : i32
      %200 = arith.cmpi sge, %199, %c1_i32 : i32
      %201 = arith.select %200, %198, %199 : i32
      memref.store %201, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb91(%c0 : index)
    ^bb91(%202: index):  // 2 preds: ^bb90, ^bb100
      %203 = arith.cmpi slt, %202, %c16 : index
      cf.cond_br %203, ^bb92, ^bb101
    ^bb92:  // pred: ^bb91
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %204 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %205 = arith.index_cast %204 : i32 to index
      %206 = arith.index_cast %205 : index to i32
      cf.switch %206 : i32, [
        default: ^bb95,
        0: ^bb93,
        1: ^bb94
      ]
    ^bb93:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb94:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb95:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb96(%207: memref<64x128xi8>):  // 3 preds: ^bb93, ^bb94, ^bb95
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %208 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %209 = arith.index_cast %208 : i32 to index
      %210 = arith.index_cast %209 : index to i32
      cf.switch %210 : i32, [
        default: ^bb99,
        0: ^bb97,
        1: ^bb98
      ]
    ^bb97:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb98:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb99:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb100(%211: memref<32x64xi8>):  // 3 preds: ^bb97, ^bb98, ^bb99
      func.call @matmul_i8_i16(%211, %207, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %212 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %213 = arith.addi %212, %c1_i32 : i32
      %214 = arith.cmpi sge, %213, %c2_i32 : i32
      %215 = arith.subi %213, %c2_i32 : i32
      %216 = arith.select %214, %215, %213 : i32
      memref.store %216, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %217 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %218 = arith.addi %217, %c1_i32 : i32
      %219 = arith.cmpi sge, %218, %c2_i32 : i32
      %220 = arith.subi %218, %c2_i32 : i32
      %221 = arith.select %219, %220, %218 : i32
      memref.store %221, %_anonymous1[%c1] : memref<3xi32>
      %222 = arith.addi %202, %c1 : index
      cf.br ^bb91(%222 : index)
    ^bb101:  // pred: ^bb91
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %223 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %224 = arith.addi %223, %c1_i32 : i32
      %225 = arith.cmpi sge, %224, %c1_i32 : i32
      %226 = arith.select %225, %223, %224 : i32
      memref.store %226, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb102(%c0 : index)
    ^bb102(%227: index):  // 2 preds: ^bb101, ^bb111
      %228 = arith.cmpi slt, %227, %c16 : index
      cf.cond_br %228, ^bb103, ^bb112
    ^bb103:  // pred: ^bb102
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %229 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %230 = arith.index_cast %229 : i32 to index
      %231 = arith.index_cast %230 : index to i32
      cf.switch %231 : i32, [
        default: ^bb106,
        0: ^bb104,
        1: ^bb105
      ]
    ^bb104:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb105:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb106:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb107(%232: memref<64x128xi8>):  // 3 preds: ^bb104, ^bb105, ^bb106
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %233 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %234 = arith.index_cast %233 : i32 to index
      %235 = arith.index_cast %234 : index to i32
      cf.switch %235 : i32, [
        default: ^bb110,
        0: ^bb108,
        1: ^bb109
      ]
    ^bb108:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb109:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb110:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb111(%236: memref<32x64xi8>):  // 3 preds: ^bb108, ^bb109, ^bb110
      func.call @matmul_i8_i16(%236, %232, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %237 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %238 = arith.addi %237, %c1_i32 : i32
      %239 = arith.cmpi sge, %238, %c2_i32 : i32
      %240 = arith.subi %238, %c2_i32 : i32
      %241 = arith.select %239, %240, %238 : i32
      memref.store %241, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %242 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %243 = arith.addi %242, %c1_i32 : i32
      %244 = arith.cmpi sge, %243, %c2_i32 : i32
      %245 = arith.subi %243, %c2_i32 : i32
      %246 = arith.select %244, %245, %243 : i32
      memref.store %246, %_anonymous1[%c1] : memref<3xi32>
      %247 = arith.addi %227, %c1 : index
      cf.br ^bb102(%247 : index)
    ^bb112:  // pred: ^bb102
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %248 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %249 = arith.addi %248, %c1_i32 : i32
      %250 = arith.cmpi sge, %249, %c1_i32 : i32
      %251 = arith.select %250, %248, %249 : i32
      memref.store %251, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb113(%c0 : index)
    ^bb113(%252: index):  // 2 preds: ^bb112, ^bb122
      %253 = arith.cmpi slt, %252, %c16 : index
      cf.cond_br %253, ^bb114, ^bb123
    ^bb114:  // pred: ^bb113
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %254 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %255 = arith.index_cast %254 : i32 to index
      %256 = arith.index_cast %255 : index to i32
      cf.switch %256 : i32, [
        default: ^bb117,
        0: ^bb115,
        1: ^bb116
      ]
    ^bb115:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb116:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb117:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb118(%257: memref<64x128xi8>):  // 3 preds: ^bb115, ^bb116, ^bb117
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %258 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %259 = arith.index_cast %258 : i32 to index
      %260 = arith.index_cast %259 : index to i32
      cf.switch %260 : i32, [
        default: ^bb121,
        0: ^bb119,
        1: ^bb120
      ]
    ^bb119:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb120:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb121:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb122(%261: memref<32x64xi8>):  // 3 preds: ^bb119, ^bb120, ^bb121
      func.call @matmul_i8_i16(%261, %257, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %262 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %263 = arith.addi %262, %c1_i32 : i32
      %264 = arith.cmpi sge, %263, %c2_i32 : i32
      %265 = arith.subi %263, %c2_i32 : i32
      %266 = arith.select %264, %265, %263 : i32
      memref.store %266, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %267 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %268 = arith.addi %267, %c1_i32 : i32
      %269 = arith.cmpi sge, %268, %c2_i32 : i32
      %270 = arith.subi %268, %c2_i32 : i32
      %271 = arith.select %269, %270, %268 : i32
      memref.store %271, %_anonymous1[%c1] : memref<3xi32>
      %272 = arith.addi %252, %c1 : index
      cf.br ^bb113(%272 : index)
    ^bb123:  // pred: ^bb113
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %273 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %274 = arith.addi %273, %c1_i32 : i32
      %275 = arith.cmpi sge, %274, %c1_i32 : i32
      %276 = arith.select %275, %273, %274 : i32
      memref.store %276, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb124(%c0 : index)
    ^bb124(%277: index):  // 2 preds: ^bb123, ^bb133
      %278 = arith.cmpi slt, %277, %c16 : index
      cf.cond_br %278, ^bb125, ^bb134
    ^bb125:  // pred: ^bb124
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %279 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %280 = arith.index_cast %279 : i32 to index
      %281 = arith.index_cast %280 : index to i32
      cf.switch %281 : i32, [
        default: ^bb128,
        0: ^bb126,
        1: ^bb127
      ]
    ^bb126:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb127:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb128:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb129(%282: memref<64x128xi8>):  // 3 preds: ^bb126, ^bb127, ^bb128
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %283 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %284 = arith.index_cast %283 : i32 to index
      %285 = arith.index_cast %284 : index to i32
      cf.switch %285 : i32, [
        default: ^bb132,
        0: ^bb130,
        1: ^bb131
      ]
    ^bb130:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb131:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb132:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb133(%286: memref<32x64xi8>):  // 3 preds: ^bb130, ^bb131, ^bb132
      func.call @matmul_i8_i16(%286, %282, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %287 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %288 = arith.addi %287, %c1_i32 : i32
      %289 = arith.cmpi sge, %288, %c2_i32 : i32
      %290 = arith.subi %288, %c2_i32 : i32
      %291 = arith.select %289, %290, %288 : i32
      memref.store %291, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %292 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %293 = arith.addi %292, %c1_i32 : i32
      %294 = arith.cmpi sge, %293, %c2_i32 : i32
      %295 = arith.subi %293, %c2_i32 : i32
      %296 = arith.select %294, %295, %293 : i32
      memref.store %296, %_anonymous1[%c1] : memref<3xi32>
      %297 = arith.addi %277, %c1 : index
      cf.br ^bb124(%297 : index)
    ^bb134:  // pred: ^bb124
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %298 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %299 = arith.addi %298, %c1_i32 : i32
      %300 = arith.cmpi sge, %299, %c1_i32 : i32
      %301 = arith.select %300, %298, %299 : i32
      memref.store %301, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb135(%c0 : index)
    ^bb135(%302: index):  // 2 preds: ^bb134, ^bb144
      %303 = arith.cmpi slt, %302, %c16 : index
      cf.cond_br %303, ^bb136, ^bb145
    ^bb136:  // pred: ^bb135
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %304 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %305 = arith.index_cast %304 : i32 to index
      %306 = arith.index_cast %305 : index to i32
      cf.switch %306 : i32, [
        default: ^bb139,
        0: ^bb137,
        1: ^bb138
      ]
    ^bb137:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb138:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb139:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb140(%307: memref<64x128xi8>):  // 3 preds: ^bb137, ^bb138, ^bb139
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %308 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %309 = arith.index_cast %308 : i32 to index
      %310 = arith.index_cast %309 : index to i32
      cf.switch %310 : i32, [
        default: ^bb143,
        0: ^bb141,
        1: ^bb142
      ]
    ^bb141:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb142:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb143:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb144(%311: memref<32x64xi8>):  // 3 preds: ^bb141, ^bb142, ^bb143
      func.call @matmul_i8_i16(%311, %307, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %312 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %313 = arith.addi %312, %c1_i32 : i32
      %314 = arith.cmpi sge, %313, %c2_i32 : i32
      %315 = arith.subi %313, %c2_i32 : i32
      %316 = arith.select %314, %315, %313 : i32
      memref.store %316, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %317 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %318 = arith.addi %317, %c1_i32 : i32
      %319 = arith.cmpi sge, %318, %c2_i32 : i32
      %320 = arith.subi %318, %c2_i32 : i32
      %321 = arith.select %319, %320, %318 : i32
      memref.store %321, %_anonymous1[%c1] : memref<3xi32>
      %322 = arith.addi %302, %c1 : index
      cf.br ^bb135(%322 : index)
    ^bb145:  // pred: ^bb135
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %323 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %324 = arith.addi %323, %c1_i32 : i32
      %325 = arith.cmpi sge, %324, %c1_i32 : i32
      %326 = arith.select %325, %323, %324 : i32
      memref.store %326, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb146(%c0 : index)
    ^bb146(%327: index):  // 2 preds: ^bb145, ^bb155
      %328 = arith.cmpi slt, %327, %c16 : index
      cf.cond_br %328, ^bb147, ^bb156
    ^bb147:  // pred: ^bb146
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %329 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %330 = arith.index_cast %329 : i32 to index
      %331 = arith.index_cast %330 : index to i32
      cf.switch %331 : i32, [
        default: ^bb150,
        0: ^bb148,
        1: ^bb149
      ]
    ^bb148:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb149:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb150:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb151(%332: memref<64x128xi8>):  // 3 preds: ^bb148, ^bb149, ^bb150
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %333 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %334 = arith.index_cast %333 : i32 to index
      %335 = arith.index_cast %334 : index to i32
      cf.switch %335 : i32, [
        default: ^bb154,
        0: ^bb152,
        1: ^bb153
      ]
    ^bb152:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb153:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb154:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb155(%336: memref<32x64xi8>):  // 3 preds: ^bb152, ^bb153, ^bb154
      func.call @matmul_i8_i16(%336, %332, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %337 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %338 = arith.addi %337, %c1_i32 : i32
      %339 = arith.cmpi sge, %338, %c2_i32 : i32
      %340 = arith.subi %338, %c2_i32 : i32
      %341 = arith.select %339, %340, %338 : i32
      memref.store %341, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %342 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %343 = arith.addi %342, %c1_i32 : i32
      %344 = arith.cmpi sge, %343, %c2_i32 : i32
      %345 = arith.subi %343, %c2_i32 : i32
      %346 = arith.select %344, %345, %343 : i32
      memref.store %346, %_anonymous1[%c1] : memref<3xi32>
      %347 = arith.addi %327, %c1 : index
      cf.br ^bb146(%347 : index)
    ^bb156:  // pred: ^bb146
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %348 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %349 = arith.addi %348, %c1_i32 : i32
      %350 = arith.cmpi sge, %349, %c1_i32 : i32
      %351 = arith.select %350, %348, %349 : i32
      memref.store %351, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb157(%c0 : index)
    ^bb157(%352: index):  // 2 preds: ^bb156, ^bb166
      %353 = arith.cmpi slt, %352, %c16 : index
      cf.cond_br %353, ^bb158, ^bb167
    ^bb158:  // pred: ^bb157
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %354 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %355 = arith.index_cast %354 : i32 to index
      %356 = arith.index_cast %355 : index to i32
      cf.switch %356 : i32, [
        default: ^bb161,
        0: ^bb159,
        1: ^bb160
      ]
    ^bb159:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb160:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb161:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb162(%357: memref<64x128xi8>):  // 3 preds: ^bb159, ^bb160, ^bb161
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %358 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %359 = arith.index_cast %358 : i32 to index
      %360 = arith.index_cast %359 : index to i32
      cf.switch %360 : i32, [
        default: ^bb165,
        0: ^bb163,
        1: ^bb164
      ]
    ^bb163:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb164:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb165:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb166(%361: memref<32x64xi8>):  // 3 preds: ^bb163, ^bb164, ^bb165
      func.call @matmul_i8_i16(%361, %357, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %362 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %363 = arith.addi %362, %c1_i32 : i32
      %364 = arith.cmpi sge, %363, %c2_i32 : i32
      %365 = arith.subi %363, %c2_i32 : i32
      %366 = arith.select %364, %365, %363 : i32
      memref.store %366, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %367 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %368 = arith.addi %367, %c1_i32 : i32
      %369 = arith.cmpi sge, %368, %c2_i32 : i32
      %370 = arith.subi %368, %c2_i32 : i32
      %371 = arith.select %369, %370, %368 : i32
      memref.store %371, %_anonymous1[%c1] : memref<3xi32>
      %372 = arith.addi %352, %c1 : index
      cf.br ^bb157(%372 : index)
    ^bb167:  // pred: ^bb157
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %373 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %374 = arith.addi %373, %c1_i32 : i32
      %375 = arith.cmpi sge, %374, %c1_i32 : i32
      %376 = arith.select %375, %373, %374 : i32
      memref.store %376, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb168(%c0 : index)
    ^bb168(%377: index):  // 2 preds: ^bb167, ^bb177
      %378 = arith.cmpi slt, %377, %c16 : index
      cf.cond_br %378, ^bb169, ^bb178
    ^bb169:  // pred: ^bb168
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %379 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %380 = arith.index_cast %379 : i32 to index
      %381 = arith.index_cast %380 : index to i32
      cf.switch %381 : i32, [
        default: ^bb172,
        0: ^bb170,
        1: ^bb171
      ]
    ^bb170:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb171:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb172:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb173(%382: memref<64x128xi8>):  // 3 preds: ^bb170, ^bb171, ^bb172
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %383 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %384 = arith.index_cast %383 : i32 to index
      %385 = arith.index_cast %384 : index to i32
      cf.switch %385 : i32, [
        default: ^bb176,
        0: ^bb174,
        1: ^bb175
      ]
    ^bb174:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb175:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb176:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb177(%386: memref<32x64xi8>):  // 3 preds: ^bb174, ^bb175, ^bb176
      func.call @matmul_i8_i16(%386, %382, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %387 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %388 = arith.addi %387, %c1_i32 : i32
      %389 = arith.cmpi sge, %388, %c2_i32 : i32
      %390 = arith.subi %388, %c2_i32 : i32
      %391 = arith.select %389, %390, %388 : i32
      memref.store %391, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %392 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %393 = arith.addi %392, %c1_i32 : i32
      %394 = arith.cmpi sge, %393, %c2_i32 : i32
      %395 = arith.subi %393, %c2_i32 : i32
      %396 = arith.select %394, %395, %393 : i32
      memref.store %396, %_anonymous1[%c1] : memref<3xi32>
      %397 = arith.addi %377, %c1 : index
      cf.br ^bb168(%397 : index)
    ^bb178:  // pred: ^bb168
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %398 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %399 = arith.addi %398, %c1_i32 : i32
      %400 = arith.cmpi sge, %399, %c1_i32 : i32
      %401 = arith.select %400, %398, %399 : i32
      memref.store %401, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb179(%c0 : index)
    ^bb179(%402: index):  // 2 preds: ^bb178, ^bb188
      %403 = arith.cmpi slt, %402, %c16 : index
      cf.cond_br %403, ^bb180, ^bb189
    ^bb180:  // pred: ^bb179
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %404 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %405 = arith.index_cast %404 : i32 to index
      %406 = arith.index_cast %405 : index to i32
      cf.switch %406 : i32, [
        default: ^bb183,
        0: ^bb181,
        1: ^bb182
      ]
    ^bb181:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb182:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb183:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb184(%407: memref<64x128xi8>):  // 3 preds: ^bb181, ^bb182, ^bb183
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %408 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %409 = arith.index_cast %408 : i32 to index
      %410 = arith.index_cast %409 : index to i32
      cf.switch %410 : i32, [
        default: ^bb187,
        0: ^bb185,
        1: ^bb186
      ]
    ^bb185:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb186:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb187:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb188(%411: memref<32x64xi8>):  // 3 preds: ^bb185, ^bb186, ^bb187
      func.call @matmul_i8_i16(%411, %407, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %412 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %413 = arith.addi %412, %c1_i32 : i32
      %414 = arith.cmpi sge, %413, %c2_i32 : i32
      %415 = arith.subi %413, %c2_i32 : i32
      %416 = arith.select %414, %415, %413 : i32
      memref.store %416, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %417 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %418 = arith.addi %417, %c1_i32 : i32
      %419 = arith.cmpi sge, %418, %c2_i32 : i32
      %420 = arith.subi %418, %c2_i32 : i32
      %421 = arith.select %419, %420, %418 : i32
      memref.store %421, %_anonymous1[%c1] : memref<3xi32>
      %422 = arith.addi %402, %c1 : index
      cf.br ^bb179(%422 : index)
    ^bb189:  // pred: ^bb179
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %423 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %424 = arith.addi %423, %c1_i32 : i32
      %425 = arith.cmpi sge, %424, %c1_i32 : i32
      %426 = arith.select %425, %423, %424 : i32
      memref.store %426, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb190(%c0 : index)
    ^bb190(%427: index):  // 2 preds: ^bb189, ^bb199
      %428 = arith.cmpi slt, %427, %c16 : index
      cf.cond_br %428, ^bb191, ^bb200
    ^bb191:  // pred: ^bb190
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %429 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %430 = arith.index_cast %429 : i32 to index
      %431 = arith.index_cast %430 : index to i32
      cf.switch %431 : i32, [
        default: ^bb194,
        0: ^bb192,
        1: ^bb193
      ]
    ^bb192:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb193:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb194:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb195(%432: memref<64x128xi8>):  // 3 preds: ^bb192, ^bb193, ^bb194
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %433 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %434 = arith.index_cast %433 : i32 to index
      %435 = arith.index_cast %434 : index to i32
      cf.switch %435 : i32, [
        default: ^bb198,
        0: ^bb196,
        1: ^bb197
      ]
    ^bb196:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb197:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb198:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb199(%436: memref<32x64xi8>):  // 3 preds: ^bb196, ^bb197, ^bb198
      func.call @matmul_i8_i16(%436, %432, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %437 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %438 = arith.addi %437, %c1_i32 : i32
      %439 = arith.cmpi sge, %438, %c2_i32 : i32
      %440 = arith.subi %438, %c2_i32 : i32
      %441 = arith.select %439, %440, %438 : i32
      memref.store %441, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %442 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %443 = arith.addi %442, %c1_i32 : i32
      %444 = arith.cmpi sge, %443, %c2_i32 : i32
      %445 = arith.subi %443, %c2_i32 : i32
      %446 = arith.select %444, %445, %443 : i32
      memref.store %446, %_anonymous1[%c1] : memref<3xi32>
      %447 = arith.addi %427, %c1 : index
      cf.br ^bb190(%447 : index)
    ^bb200:  // pred: ^bb190
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %448 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %449 = arith.addi %448, %c1_i32 : i32
      %450 = arith.cmpi sge, %449, %c1_i32 : i32
      %451 = arith.select %450, %448, %449 : i32
      memref.store %451, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb201(%c0 : index)
    ^bb201(%452: index):  // 2 preds: ^bb200, ^bb210
      %453 = arith.cmpi slt, %452, %c16 : index
      cf.cond_br %453, ^bb202, ^bb211
    ^bb202:  // pred: ^bb201
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %454 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %455 = arith.index_cast %454 : i32 to index
      %456 = arith.index_cast %455 : index to i32
      cf.switch %456 : i32, [
        default: ^bb205,
        0: ^bb203,
        1: ^bb204
      ]
    ^bb203:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb204:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb205:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb206(%457: memref<64x128xi8>):  // 3 preds: ^bb203, ^bb204, ^bb205
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %458 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %459 = arith.index_cast %458 : i32 to index
      %460 = arith.index_cast %459 : index to i32
      cf.switch %460 : i32, [
        default: ^bb209,
        0: ^bb207,
        1: ^bb208
      ]
    ^bb207:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb208:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb209:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb210(%461: memref<32x64xi8>):  // 3 preds: ^bb207, ^bb208, ^bb209
      func.call @matmul_i8_i16(%461, %457, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %462 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %463 = arith.addi %462, %c1_i32 : i32
      %464 = arith.cmpi sge, %463, %c2_i32 : i32
      %465 = arith.subi %463, %c2_i32 : i32
      %466 = arith.select %464, %465, %463 : i32
      memref.store %466, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %467 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %468 = arith.addi %467, %c1_i32 : i32
      %469 = arith.cmpi sge, %468, %c2_i32 : i32
      %470 = arith.subi %468, %c2_i32 : i32
      %471 = arith.select %469, %470, %468 : i32
      memref.store %471, %_anonymous1[%c1] : memref<3xi32>
      %472 = arith.addi %452, %c1 : index
      cf.br ^bb201(%472 : index)
    ^bb211:  // pred: ^bb201
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %473 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %474 = arith.addi %473, %c1_i32 : i32
      %475 = arith.cmpi sge, %474, %c1_i32 : i32
      %476 = arith.select %475, %473, %474 : i32
      memref.store %476, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb212(%c0 : index)
    ^bb212(%477: index):  // 2 preds: ^bb211, ^bb221
      %478 = arith.cmpi slt, %477, %c16 : index
      cf.cond_br %478, ^bb213, ^bb222
    ^bb213:  // pred: ^bb212
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %479 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %480 = arith.index_cast %479 : i32 to index
      %481 = arith.index_cast %480 : index to i32
      cf.switch %481 : i32, [
        default: ^bb216,
        0: ^bb214,
        1: ^bb215
      ]
    ^bb214:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb215:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb216:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb217(%482: memref<64x128xi8>):  // 3 preds: ^bb214, ^bb215, ^bb216
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %483 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %484 = arith.index_cast %483 : i32 to index
      %485 = arith.index_cast %484 : index to i32
      cf.switch %485 : i32, [
        default: ^bb220,
        0: ^bb218,
        1: ^bb219
      ]
    ^bb218:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb219:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb220:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb221(%486: memref<32x64xi8>):  // 3 preds: ^bb218, ^bb219, ^bb220
      func.call @matmul_i8_i16(%486, %482, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %487 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %488 = arith.addi %487, %c1_i32 : i32
      %489 = arith.cmpi sge, %488, %c2_i32 : i32
      %490 = arith.subi %488, %c2_i32 : i32
      %491 = arith.select %489, %490, %488 : i32
      memref.store %491, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %492 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %493 = arith.addi %492, %c1_i32 : i32
      %494 = arith.cmpi sge, %493, %c2_i32 : i32
      %495 = arith.subi %493, %c2_i32 : i32
      %496 = arith.select %494, %495, %493 : i32
      memref.store %496, %_anonymous1[%c1] : memref<3xi32>
      %497 = arith.addi %477, %c1 : index
      cf.br ^bb212(%497 : index)
    ^bb222:  // pred: ^bb212
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %498 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %499 = arith.addi %498, %c1_i32 : i32
      %500 = arith.cmpi sge, %499, %c1_i32 : i32
      %501 = arith.select %500, %498, %499 : i32
      memref.store %501, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb223(%c0 : index)
    ^bb223(%502: index):  // 2 preds: ^bb222, ^bb232
      %503 = arith.cmpi slt, %502, %c16 : index
      cf.cond_br %503, ^bb224, ^bb233
    ^bb224:  // pred: ^bb223
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %504 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %505 = arith.index_cast %504 : i32 to index
      %506 = arith.index_cast %505 : index to i32
      cf.switch %506 : i32, [
        default: ^bb227,
        0: ^bb225,
        1: ^bb226
      ]
    ^bb225:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb226:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb227:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb228(%507: memref<64x128xi8>):  // 3 preds: ^bb225, ^bb226, ^bb227
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %508 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %509 = arith.index_cast %508 : i32 to index
      %510 = arith.index_cast %509 : index to i32
      cf.switch %510 : i32, [
        default: ^bb231,
        0: ^bb229,
        1: ^bb230
      ]
    ^bb229:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb230:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb231:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb232(%511: memref<32x64xi8>):  // 3 preds: ^bb229, ^bb230, ^bb231
      func.call @matmul_i8_i16(%511, %507, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %512 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %513 = arith.addi %512, %c1_i32 : i32
      %514 = arith.cmpi sge, %513, %c2_i32 : i32
      %515 = arith.subi %513, %c2_i32 : i32
      %516 = arith.select %514, %515, %513 : i32
      memref.store %516, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %517 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %518 = arith.addi %517, %c1_i32 : i32
      %519 = arith.cmpi sge, %518, %c2_i32 : i32
      %520 = arith.subi %518, %c2_i32 : i32
      %521 = arith.select %519, %520, %518 : i32
      memref.store %521, %_anonymous1[%c1] : memref<3xi32>
      %522 = arith.addi %502, %c1 : index
      cf.br ^bb223(%522 : index)
    ^bb233:  // pred: ^bb223
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %523 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %524 = arith.addi %523, %c1_i32 : i32
      %525 = arith.cmpi sge, %524, %c1_i32 : i32
      %526 = arith.select %525, %523, %524 : i32
      memref.store %526, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb234(%c0 : index)
    ^bb234(%527: index):  // 2 preds: ^bb233, ^bb243
      %528 = arith.cmpi slt, %527, %c16 : index
      cf.cond_br %528, ^bb235, ^bb244
    ^bb235:  // pred: ^bb234
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %529 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %530 = arith.index_cast %529 : i32 to index
      %531 = arith.index_cast %530 : index to i32
      cf.switch %531 : i32, [
        default: ^bb238,
        0: ^bb236,
        1: ^bb237
      ]
    ^bb236:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb237:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb238:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb239(%532: memref<64x128xi8>):  // 3 preds: ^bb236, ^bb237, ^bb238
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %533 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %534 = arith.index_cast %533 : i32 to index
      %535 = arith.index_cast %534 : index to i32
      cf.switch %535 : i32, [
        default: ^bb242,
        0: ^bb240,
        1: ^bb241
      ]
    ^bb240:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb241:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb242:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb243(%536: memref<32x64xi8>):  // 3 preds: ^bb240, ^bb241, ^bb242
      func.call @matmul_i8_i16(%536, %532, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %537 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %538 = arith.addi %537, %c1_i32 : i32
      %539 = arith.cmpi sge, %538, %c2_i32 : i32
      %540 = arith.subi %538, %c2_i32 : i32
      %541 = arith.select %539, %540, %538 : i32
      memref.store %541, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %542 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %543 = arith.addi %542, %c1_i32 : i32
      %544 = arith.cmpi sge, %543, %c2_i32 : i32
      %545 = arith.subi %543, %c2_i32 : i32
      %546 = arith.select %544, %545, %543 : i32
      memref.store %546, %_anonymous1[%c1] : memref<3xi32>
      %547 = arith.addi %527, %c1 : index
      cf.br ^bb234(%547 : index)
    ^bb244:  // pred: ^bb234
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %548 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %549 = arith.addi %548, %c1_i32 : i32
      %550 = arith.cmpi sge, %549, %c1_i32 : i32
      %551 = arith.select %550, %548, %549 : i32
      memref.store %551, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb245(%c0 : index)
    ^bb245(%552: index):  // 2 preds: ^bb244, ^bb254
      %553 = arith.cmpi slt, %552, %c16 : index
      cf.cond_br %553, ^bb246, ^bb255
    ^bb246:  // pred: ^bb245
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %554 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %555 = arith.index_cast %554 : i32 to index
      %556 = arith.index_cast %555 : index to i32
      cf.switch %556 : i32, [
        default: ^bb249,
        0: ^bb247,
        1: ^bb248
      ]
    ^bb247:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb248:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb249:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb250(%557: memref<64x128xi8>):  // 3 preds: ^bb247, ^bb248, ^bb249
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %558 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %559 = arith.index_cast %558 : i32 to index
      %560 = arith.index_cast %559 : index to i32
      cf.switch %560 : i32, [
        default: ^bb253,
        0: ^bb251,
        1: ^bb252
      ]
    ^bb251:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb252:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb253:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb254(%561: memref<32x64xi8>):  // 3 preds: ^bb251, ^bb252, ^bb253
      func.call @matmul_i8_i16(%561, %557, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %562 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %563 = arith.addi %562, %c1_i32 : i32
      %564 = arith.cmpi sge, %563, %c2_i32 : i32
      %565 = arith.subi %563, %c2_i32 : i32
      %566 = arith.select %564, %565, %563 : i32
      memref.store %566, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %567 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %568 = arith.addi %567, %c1_i32 : i32
      %569 = arith.cmpi sge, %568, %c2_i32 : i32
      %570 = arith.subi %568, %c2_i32 : i32
      %571 = arith.select %569, %570, %568 : i32
      memref.store %571, %_anonymous1[%c1] : memref<3xi32>
      %572 = arith.addi %552, %c1 : index
      cf.br ^bb245(%572 : index)
    ^bb255:  // pred: ^bb245
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %573 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %574 = arith.addi %573, %c1_i32 : i32
      %575 = arith.cmpi sge, %574, %c1_i32 : i32
      %576 = arith.select %575, %573, %574 : i32
      memref.store %576, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb256(%c0 : index)
    ^bb256(%577: index):  // 2 preds: ^bb255, ^bb265
      %578 = arith.cmpi slt, %577, %c16 : index
      cf.cond_br %578, ^bb257, ^bb266
    ^bb257:  // pred: ^bb256
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %579 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %580 = arith.index_cast %579 : i32 to index
      %581 = arith.index_cast %580 : index to i32
      cf.switch %581 : i32, [
        default: ^bb260,
        0: ^bb258,
        1: ^bb259
      ]
    ^bb258:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb259:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb260:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb261(%582: memref<64x128xi8>):  // 3 preds: ^bb258, ^bb259, ^bb260
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %583 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %584 = arith.index_cast %583 : i32 to index
      %585 = arith.index_cast %584 : index to i32
      cf.switch %585 : i32, [
        default: ^bb264,
        0: ^bb262,
        1: ^bb263
      ]
    ^bb262:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb263:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb264:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb265(%586: memref<32x64xi8>):  // 3 preds: ^bb262, ^bb263, ^bb264
      func.call @matmul_i8_i16(%586, %582, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %587 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %588 = arith.addi %587, %c1_i32 : i32
      %589 = arith.cmpi sge, %588, %c2_i32 : i32
      %590 = arith.subi %588, %c2_i32 : i32
      %591 = arith.select %589, %590, %588 : i32
      memref.store %591, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %592 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %593 = arith.addi %592, %c1_i32 : i32
      %594 = arith.cmpi sge, %593, %c2_i32 : i32
      %595 = arith.subi %593, %c2_i32 : i32
      %596 = arith.select %594, %595, %593 : i32
      memref.store %596, %_anonymous1[%c1] : memref<3xi32>
      %597 = arith.addi %577, %c1 : index
      cf.br ^bb256(%597 : index)
    ^bb266:  // pred: ^bb256
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %598 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %599 = arith.addi %598, %c1_i32 : i32
      %600 = arith.cmpi sge, %599, %c1_i32 : i32
      %601 = arith.select %600, %598, %599 : i32
      memref.store %601, %_anonymous1[%c0] : memref<3xi32>
      %602 = arith.addi %0, %c1 : index
      cf.br ^bb1(%602 : index)
    ^bb267:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous2 = aie.buffer(%tile_2_2) {address = 32000 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb266
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb267
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
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
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous2[%c1] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb19(%32: memref<64x128xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb23(%36: memref<32x64xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_i8_i16(%36, %32, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous2[%c1] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb30(%57: memref<64x128xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb34(%61: memref<32x64xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_i8_i16(%61, %57, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous2[%c1] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb41(%82: memref<64x128xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb45(%86: memref<32x64xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_i8_i16(%86, %82, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous2[%c1] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb47(%c0 : index)
    ^bb47(%102: index):  // 2 preds: ^bb46, ^bb56
      %103 = arith.cmpi slt, %102, %c16 : index
      cf.cond_br %103, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %104 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %105 = arith.index_cast %104 : i32 to index
      %106 = arith.index_cast %105 : index to i32
      cf.switch %106 : i32, [
        default: ^bb51,
        0: ^bb49,
        1: ^bb50
      ]
    ^bb49:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb50:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb51:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb52(%107: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb55,
        0: ^bb53,
        1: ^bb54
      ]
    ^bb53:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb54:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb55:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb56(%111: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_i8_i16(%111, %107, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %112 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %113 = arith.addi %112, %c1_i32 : i32
      %114 = arith.cmpi sge, %113, %c2_i32 : i32
      %115 = arith.subi %113, %c2_i32 : i32
      %116 = arith.select %114, %115, %113 : i32
      memref.store %116, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %117 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %118 = arith.addi %117, %c1_i32 : i32
      %119 = arith.cmpi sge, %118, %c2_i32 : i32
      %120 = arith.subi %118, %c2_i32 : i32
      %121 = arith.select %119, %120, %118 : i32
      memref.store %121, %_anonymous2[%c1] : memref<3xi32>
      %122 = arith.addi %102, %c1 : index
      cf.br ^bb47(%122 : index)
    ^bb57:  // pred: ^bb47
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %123 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c1_i32 : i32
      %126 = arith.select %125, %123, %124 : i32
      memref.store %126, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb58(%c0 : index)
    ^bb58(%127: index):  // 2 preds: ^bb57, ^bb67
      %128 = arith.cmpi slt, %127, %c16 : index
      cf.cond_br %128, ^bb59, ^bb68
    ^bb59:  // pred: ^bb58
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %129 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %130 = arith.index_cast %129 : i32 to index
      %131 = arith.index_cast %130 : index to i32
      cf.switch %131 : i32, [
        default: ^bb62,
        0: ^bb60,
        1: ^bb61
      ]
    ^bb60:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb61:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb62:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb63(%132: memref<64x128xi8>):  // 3 preds: ^bb60, ^bb61, ^bb62
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %133 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %134 = arith.index_cast %133 : i32 to index
      %135 = arith.index_cast %134 : index to i32
      cf.switch %135 : i32, [
        default: ^bb66,
        0: ^bb64,
        1: ^bb65
      ]
    ^bb64:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb66:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb67(%136: memref<32x64xi8>):  // 3 preds: ^bb64, ^bb65, ^bb66
      func.call @matmul_i8_i16(%136, %132, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %137 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %138 = arith.addi %137, %c1_i32 : i32
      %139 = arith.cmpi sge, %138, %c2_i32 : i32
      %140 = arith.subi %138, %c2_i32 : i32
      %141 = arith.select %139, %140, %138 : i32
      memref.store %141, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %142 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %143 = arith.addi %142, %c1_i32 : i32
      %144 = arith.cmpi sge, %143, %c2_i32 : i32
      %145 = arith.subi %143, %c2_i32 : i32
      %146 = arith.select %144, %145, %143 : i32
      memref.store %146, %_anonymous2[%c1] : memref<3xi32>
      %147 = arith.addi %127, %c1 : index
      cf.br ^bb58(%147 : index)
    ^bb68:  // pred: ^bb58
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %148 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c1_i32 : i32
      %151 = arith.select %150, %148, %149 : i32
      memref.store %151, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb69(%c0 : index)
    ^bb69(%152: index):  // 2 preds: ^bb68, ^bb78
      %153 = arith.cmpi slt, %152, %c16 : index
      cf.cond_br %153, ^bb70, ^bb79
    ^bb70:  // pred: ^bb69
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %154 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %155 = arith.index_cast %154 : i32 to index
      %156 = arith.index_cast %155 : index to i32
      cf.switch %156 : i32, [
        default: ^bb73,
        0: ^bb71,
        1: ^bb72
      ]
    ^bb71:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb72:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb73:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb74(%157: memref<64x128xi8>):  // 3 preds: ^bb71, ^bb72, ^bb73
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %158 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %159 = arith.index_cast %158 : i32 to index
      %160 = arith.index_cast %159 : index to i32
      cf.switch %160 : i32, [
        default: ^bb77,
        0: ^bb75,
        1: ^bb76
      ]
    ^bb75:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb76:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb77:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb78(%161: memref<32x64xi8>):  // 3 preds: ^bb75, ^bb76, ^bb77
      func.call @matmul_i8_i16(%161, %157, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %162 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %163 = arith.addi %162, %c1_i32 : i32
      %164 = arith.cmpi sge, %163, %c2_i32 : i32
      %165 = arith.subi %163, %c2_i32 : i32
      %166 = arith.select %164, %165, %163 : i32
      memref.store %166, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %167 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %168 = arith.addi %167, %c1_i32 : i32
      %169 = arith.cmpi sge, %168, %c2_i32 : i32
      %170 = arith.subi %168, %c2_i32 : i32
      %171 = arith.select %169, %170, %168 : i32
      memref.store %171, %_anonymous2[%c1] : memref<3xi32>
      %172 = arith.addi %152, %c1 : index
      cf.br ^bb69(%172 : index)
    ^bb79:  // pred: ^bb69
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %173 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %174 = arith.addi %173, %c1_i32 : i32
      %175 = arith.cmpi sge, %174, %c1_i32 : i32
      %176 = arith.select %175, %173, %174 : i32
      memref.store %176, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb80(%c0 : index)
    ^bb80(%177: index):  // 2 preds: ^bb79, ^bb89
      %178 = arith.cmpi slt, %177, %c16 : index
      cf.cond_br %178, ^bb81, ^bb90
    ^bb81:  // pred: ^bb80
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %179 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %180 = arith.index_cast %179 : i32 to index
      %181 = arith.index_cast %180 : index to i32
      cf.switch %181 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb85(%182: memref<64x128xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %183 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %184 = arith.index_cast %183 : i32 to index
      %185 = arith.index_cast %184 : index to i32
      cf.switch %185 : i32, [
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
    ^bb89(%186: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%186, %182, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %187 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %188 = arith.addi %187, %c1_i32 : i32
      %189 = arith.cmpi sge, %188, %c2_i32 : i32
      %190 = arith.subi %188, %c2_i32 : i32
      %191 = arith.select %189, %190, %188 : i32
      memref.store %191, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %192 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %193 = arith.addi %192, %c1_i32 : i32
      %194 = arith.cmpi sge, %193, %c2_i32 : i32
      %195 = arith.subi %193, %c2_i32 : i32
      %196 = arith.select %194, %195, %193 : i32
      memref.store %196, %_anonymous2[%c1] : memref<3xi32>
      %197 = arith.addi %177, %c1 : index
      cf.br ^bb80(%197 : index)
    ^bb90:  // pred: ^bb80
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %198 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %199 = arith.addi %198, %c1_i32 : i32
      %200 = arith.cmpi sge, %199, %c1_i32 : i32
      %201 = arith.select %200, %198, %199 : i32
      memref.store %201, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb91(%c0 : index)
    ^bb91(%202: index):  // 2 preds: ^bb90, ^bb100
      %203 = arith.cmpi slt, %202, %c16 : index
      cf.cond_br %203, ^bb92, ^bb101
    ^bb92:  // pred: ^bb91
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %204 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %205 = arith.index_cast %204 : i32 to index
      %206 = arith.index_cast %205 : index to i32
      cf.switch %206 : i32, [
        default: ^bb95,
        0: ^bb93,
        1: ^bb94
      ]
    ^bb93:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb94:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb95:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb96(%207: memref<64x128xi8>):  // 3 preds: ^bb93, ^bb94, ^bb95
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %208 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %209 = arith.index_cast %208 : i32 to index
      %210 = arith.index_cast %209 : index to i32
      cf.switch %210 : i32, [
        default: ^bb99,
        0: ^bb97,
        1: ^bb98
      ]
    ^bb97:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb98:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb99:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb100(%211: memref<32x64xi8>):  // 3 preds: ^bb97, ^bb98, ^bb99
      func.call @matmul_i8_i16(%211, %207, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %212 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %213 = arith.addi %212, %c1_i32 : i32
      %214 = arith.cmpi sge, %213, %c2_i32 : i32
      %215 = arith.subi %213, %c2_i32 : i32
      %216 = arith.select %214, %215, %213 : i32
      memref.store %216, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %217 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %218 = arith.addi %217, %c1_i32 : i32
      %219 = arith.cmpi sge, %218, %c2_i32 : i32
      %220 = arith.subi %218, %c2_i32 : i32
      %221 = arith.select %219, %220, %218 : i32
      memref.store %221, %_anonymous2[%c1] : memref<3xi32>
      %222 = arith.addi %202, %c1 : index
      cf.br ^bb91(%222 : index)
    ^bb101:  // pred: ^bb91
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %223 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %224 = arith.addi %223, %c1_i32 : i32
      %225 = arith.cmpi sge, %224, %c1_i32 : i32
      %226 = arith.select %225, %223, %224 : i32
      memref.store %226, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb102(%c0 : index)
    ^bb102(%227: index):  // 2 preds: ^bb101, ^bb111
      %228 = arith.cmpi slt, %227, %c16 : index
      cf.cond_br %228, ^bb103, ^bb112
    ^bb103:  // pred: ^bb102
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %229 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %230 = arith.index_cast %229 : i32 to index
      %231 = arith.index_cast %230 : index to i32
      cf.switch %231 : i32, [
        default: ^bb106,
        0: ^bb104,
        1: ^bb105
      ]
    ^bb104:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb105:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb106:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb107(%232: memref<64x128xi8>):  // 3 preds: ^bb104, ^bb105, ^bb106
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %233 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %234 = arith.index_cast %233 : i32 to index
      %235 = arith.index_cast %234 : index to i32
      cf.switch %235 : i32, [
        default: ^bb110,
        0: ^bb108,
        1: ^bb109
      ]
    ^bb108:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb109:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb110:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb111(%236: memref<32x64xi8>):  // 3 preds: ^bb108, ^bb109, ^bb110
      func.call @matmul_i8_i16(%236, %232, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %237 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %238 = arith.addi %237, %c1_i32 : i32
      %239 = arith.cmpi sge, %238, %c2_i32 : i32
      %240 = arith.subi %238, %c2_i32 : i32
      %241 = arith.select %239, %240, %238 : i32
      memref.store %241, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %242 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %243 = arith.addi %242, %c1_i32 : i32
      %244 = arith.cmpi sge, %243, %c2_i32 : i32
      %245 = arith.subi %243, %c2_i32 : i32
      %246 = arith.select %244, %245, %243 : i32
      memref.store %246, %_anonymous2[%c1] : memref<3xi32>
      %247 = arith.addi %227, %c1 : index
      cf.br ^bb102(%247 : index)
    ^bb112:  // pred: ^bb102
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %248 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %249 = arith.addi %248, %c1_i32 : i32
      %250 = arith.cmpi sge, %249, %c1_i32 : i32
      %251 = arith.select %250, %248, %249 : i32
      memref.store %251, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb113(%c0 : index)
    ^bb113(%252: index):  // 2 preds: ^bb112, ^bb122
      %253 = arith.cmpi slt, %252, %c16 : index
      cf.cond_br %253, ^bb114, ^bb123
    ^bb114:  // pred: ^bb113
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %254 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %255 = arith.index_cast %254 : i32 to index
      %256 = arith.index_cast %255 : index to i32
      cf.switch %256 : i32, [
        default: ^bb117,
        0: ^bb115,
        1: ^bb116
      ]
    ^bb115:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb116:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb117:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb118(%257: memref<64x128xi8>):  // 3 preds: ^bb115, ^bb116, ^bb117
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %258 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %259 = arith.index_cast %258 : i32 to index
      %260 = arith.index_cast %259 : index to i32
      cf.switch %260 : i32, [
        default: ^bb121,
        0: ^bb119,
        1: ^bb120
      ]
    ^bb119:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb120:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb121:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb122(%261: memref<32x64xi8>):  // 3 preds: ^bb119, ^bb120, ^bb121
      func.call @matmul_i8_i16(%261, %257, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %262 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %263 = arith.addi %262, %c1_i32 : i32
      %264 = arith.cmpi sge, %263, %c2_i32 : i32
      %265 = arith.subi %263, %c2_i32 : i32
      %266 = arith.select %264, %265, %263 : i32
      memref.store %266, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %267 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %268 = arith.addi %267, %c1_i32 : i32
      %269 = arith.cmpi sge, %268, %c2_i32 : i32
      %270 = arith.subi %268, %c2_i32 : i32
      %271 = arith.select %269, %270, %268 : i32
      memref.store %271, %_anonymous2[%c1] : memref<3xi32>
      %272 = arith.addi %252, %c1 : index
      cf.br ^bb113(%272 : index)
    ^bb123:  // pred: ^bb113
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %273 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %274 = arith.addi %273, %c1_i32 : i32
      %275 = arith.cmpi sge, %274, %c1_i32 : i32
      %276 = arith.select %275, %273, %274 : i32
      memref.store %276, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb124(%c0 : index)
    ^bb124(%277: index):  // 2 preds: ^bb123, ^bb133
      %278 = arith.cmpi slt, %277, %c16 : index
      cf.cond_br %278, ^bb125, ^bb134
    ^bb125:  // pred: ^bb124
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %279 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %280 = arith.index_cast %279 : i32 to index
      %281 = arith.index_cast %280 : index to i32
      cf.switch %281 : i32, [
        default: ^bb128,
        0: ^bb126,
        1: ^bb127
      ]
    ^bb126:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb127:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb128:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb129(%282: memref<64x128xi8>):  // 3 preds: ^bb126, ^bb127, ^bb128
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %283 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %284 = arith.index_cast %283 : i32 to index
      %285 = arith.index_cast %284 : index to i32
      cf.switch %285 : i32, [
        default: ^bb132,
        0: ^bb130,
        1: ^bb131
      ]
    ^bb130:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb131:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb132:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb133(%286: memref<32x64xi8>):  // 3 preds: ^bb130, ^bb131, ^bb132
      func.call @matmul_i8_i16(%286, %282, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %287 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %288 = arith.addi %287, %c1_i32 : i32
      %289 = arith.cmpi sge, %288, %c2_i32 : i32
      %290 = arith.subi %288, %c2_i32 : i32
      %291 = arith.select %289, %290, %288 : i32
      memref.store %291, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %292 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %293 = arith.addi %292, %c1_i32 : i32
      %294 = arith.cmpi sge, %293, %c2_i32 : i32
      %295 = arith.subi %293, %c2_i32 : i32
      %296 = arith.select %294, %295, %293 : i32
      memref.store %296, %_anonymous2[%c1] : memref<3xi32>
      %297 = arith.addi %277, %c1 : index
      cf.br ^bb124(%297 : index)
    ^bb134:  // pred: ^bb124
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %298 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %299 = arith.addi %298, %c1_i32 : i32
      %300 = arith.cmpi sge, %299, %c1_i32 : i32
      %301 = arith.select %300, %298, %299 : i32
      memref.store %301, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb135(%c0 : index)
    ^bb135(%302: index):  // 2 preds: ^bb134, ^bb144
      %303 = arith.cmpi slt, %302, %c16 : index
      cf.cond_br %303, ^bb136, ^bb145
    ^bb136:  // pred: ^bb135
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %304 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %305 = arith.index_cast %304 : i32 to index
      %306 = arith.index_cast %305 : index to i32
      cf.switch %306 : i32, [
        default: ^bb139,
        0: ^bb137,
        1: ^bb138
      ]
    ^bb137:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb138:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb139:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb140(%307: memref<64x128xi8>):  // 3 preds: ^bb137, ^bb138, ^bb139
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %308 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %309 = arith.index_cast %308 : i32 to index
      %310 = arith.index_cast %309 : index to i32
      cf.switch %310 : i32, [
        default: ^bb143,
        0: ^bb141,
        1: ^bb142
      ]
    ^bb141:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb142:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb143:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb144(%311: memref<32x64xi8>):  // 3 preds: ^bb141, ^bb142, ^bb143
      func.call @matmul_i8_i16(%311, %307, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %312 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %313 = arith.addi %312, %c1_i32 : i32
      %314 = arith.cmpi sge, %313, %c2_i32 : i32
      %315 = arith.subi %313, %c2_i32 : i32
      %316 = arith.select %314, %315, %313 : i32
      memref.store %316, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %317 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %318 = arith.addi %317, %c1_i32 : i32
      %319 = arith.cmpi sge, %318, %c2_i32 : i32
      %320 = arith.subi %318, %c2_i32 : i32
      %321 = arith.select %319, %320, %318 : i32
      memref.store %321, %_anonymous2[%c1] : memref<3xi32>
      %322 = arith.addi %302, %c1 : index
      cf.br ^bb135(%322 : index)
    ^bb145:  // pred: ^bb135
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %323 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %324 = arith.addi %323, %c1_i32 : i32
      %325 = arith.cmpi sge, %324, %c1_i32 : i32
      %326 = arith.select %325, %323, %324 : i32
      memref.store %326, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb146(%c0 : index)
    ^bb146(%327: index):  // 2 preds: ^bb145, ^bb155
      %328 = arith.cmpi slt, %327, %c16 : index
      cf.cond_br %328, ^bb147, ^bb156
    ^bb147:  // pred: ^bb146
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %329 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %330 = arith.index_cast %329 : i32 to index
      %331 = arith.index_cast %330 : index to i32
      cf.switch %331 : i32, [
        default: ^bb150,
        0: ^bb148,
        1: ^bb149
      ]
    ^bb148:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb149:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb150:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb151(%332: memref<64x128xi8>):  // 3 preds: ^bb148, ^bb149, ^bb150
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %333 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %334 = arith.index_cast %333 : i32 to index
      %335 = arith.index_cast %334 : index to i32
      cf.switch %335 : i32, [
        default: ^bb154,
        0: ^bb152,
        1: ^bb153
      ]
    ^bb152:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb153:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb154:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb155(%336: memref<32x64xi8>):  // 3 preds: ^bb152, ^bb153, ^bb154
      func.call @matmul_i8_i16(%336, %332, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %337 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %338 = arith.addi %337, %c1_i32 : i32
      %339 = arith.cmpi sge, %338, %c2_i32 : i32
      %340 = arith.subi %338, %c2_i32 : i32
      %341 = arith.select %339, %340, %338 : i32
      memref.store %341, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %342 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %343 = arith.addi %342, %c1_i32 : i32
      %344 = arith.cmpi sge, %343, %c2_i32 : i32
      %345 = arith.subi %343, %c2_i32 : i32
      %346 = arith.select %344, %345, %343 : i32
      memref.store %346, %_anonymous2[%c1] : memref<3xi32>
      %347 = arith.addi %327, %c1 : index
      cf.br ^bb146(%347 : index)
    ^bb156:  // pred: ^bb146
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %348 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %349 = arith.addi %348, %c1_i32 : i32
      %350 = arith.cmpi sge, %349, %c1_i32 : i32
      %351 = arith.select %350, %348, %349 : i32
      memref.store %351, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb157(%c0 : index)
    ^bb157(%352: index):  // 2 preds: ^bb156, ^bb166
      %353 = arith.cmpi slt, %352, %c16 : index
      cf.cond_br %353, ^bb158, ^bb167
    ^bb158:  // pred: ^bb157
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %354 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %355 = arith.index_cast %354 : i32 to index
      %356 = arith.index_cast %355 : index to i32
      cf.switch %356 : i32, [
        default: ^bb161,
        0: ^bb159,
        1: ^bb160
      ]
    ^bb159:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb160:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb161:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb162(%357: memref<64x128xi8>):  // 3 preds: ^bb159, ^bb160, ^bb161
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %358 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %359 = arith.index_cast %358 : i32 to index
      %360 = arith.index_cast %359 : index to i32
      cf.switch %360 : i32, [
        default: ^bb165,
        0: ^bb163,
        1: ^bb164
      ]
    ^bb163:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb164:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb165:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb166(%361: memref<32x64xi8>):  // 3 preds: ^bb163, ^bb164, ^bb165
      func.call @matmul_i8_i16(%361, %357, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %362 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %363 = arith.addi %362, %c1_i32 : i32
      %364 = arith.cmpi sge, %363, %c2_i32 : i32
      %365 = arith.subi %363, %c2_i32 : i32
      %366 = arith.select %364, %365, %363 : i32
      memref.store %366, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %367 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %368 = arith.addi %367, %c1_i32 : i32
      %369 = arith.cmpi sge, %368, %c2_i32 : i32
      %370 = arith.subi %368, %c2_i32 : i32
      %371 = arith.select %369, %370, %368 : i32
      memref.store %371, %_anonymous2[%c1] : memref<3xi32>
      %372 = arith.addi %352, %c1 : index
      cf.br ^bb157(%372 : index)
    ^bb167:  // pred: ^bb157
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %373 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %374 = arith.addi %373, %c1_i32 : i32
      %375 = arith.cmpi sge, %374, %c1_i32 : i32
      %376 = arith.select %375, %373, %374 : i32
      memref.store %376, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb168(%c0 : index)
    ^bb168(%377: index):  // 2 preds: ^bb167, ^bb177
      %378 = arith.cmpi slt, %377, %c16 : index
      cf.cond_br %378, ^bb169, ^bb178
    ^bb169:  // pred: ^bb168
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %379 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %380 = arith.index_cast %379 : i32 to index
      %381 = arith.index_cast %380 : index to i32
      cf.switch %381 : i32, [
        default: ^bb172,
        0: ^bb170,
        1: ^bb171
      ]
    ^bb170:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb171:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb172:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb173(%382: memref<64x128xi8>):  // 3 preds: ^bb170, ^bb171, ^bb172
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %383 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %384 = arith.index_cast %383 : i32 to index
      %385 = arith.index_cast %384 : index to i32
      cf.switch %385 : i32, [
        default: ^bb176,
        0: ^bb174,
        1: ^bb175
      ]
    ^bb174:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb175:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb176:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb177(%386: memref<32x64xi8>):  // 3 preds: ^bb174, ^bb175, ^bb176
      func.call @matmul_i8_i16(%386, %382, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %387 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %388 = arith.addi %387, %c1_i32 : i32
      %389 = arith.cmpi sge, %388, %c2_i32 : i32
      %390 = arith.subi %388, %c2_i32 : i32
      %391 = arith.select %389, %390, %388 : i32
      memref.store %391, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %392 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %393 = arith.addi %392, %c1_i32 : i32
      %394 = arith.cmpi sge, %393, %c2_i32 : i32
      %395 = arith.subi %393, %c2_i32 : i32
      %396 = arith.select %394, %395, %393 : i32
      memref.store %396, %_anonymous2[%c1] : memref<3xi32>
      %397 = arith.addi %377, %c1 : index
      cf.br ^bb168(%397 : index)
    ^bb178:  // pred: ^bb168
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %398 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %399 = arith.addi %398, %c1_i32 : i32
      %400 = arith.cmpi sge, %399, %c1_i32 : i32
      %401 = arith.select %400, %398, %399 : i32
      memref.store %401, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb179(%c0 : index)
    ^bb179(%402: index):  // 2 preds: ^bb178, ^bb188
      %403 = arith.cmpi slt, %402, %c16 : index
      cf.cond_br %403, ^bb180, ^bb189
    ^bb180:  // pred: ^bb179
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %404 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %405 = arith.index_cast %404 : i32 to index
      %406 = arith.index_cast %405 : index to i32
      cf.switch %406 : i32, [
        default: ^bb183,
        0: ^bb181,
        1: ^bb182
      ]
    ^bb181:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb182:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb183:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb184(%407: memref<64x128xi8>):  // 3 preds: ^bb181, ^bb182, ^bb183
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %408 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %409 = arith.index_cast %408 : i32 to index
      %410 = arith.index_cast %409 : index to i32
      cf.switch %410 : i32, [
        default: ^bb187,
        0: ^bb185,
        1: ^bb186
      ]
    ^bb185:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb186:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb187:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb188(%411: memref<32x64xi8>):  // 3 preds: ^bb185, ^bb186, ^bb187
      func.call @matmul_i8_i16(%411, %407, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %412 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %413 = arith.addi %412, %c1_i32 : i32
      %414 = arith.cmpi sge, %413, %c2_i32 : i32
      %415 = arith.subi %413, %c2_i32 : i32
      %416 = arith.select %414, %415, %413 : i32
      memref.store %416, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %417 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %418 = arith.addi %417, %c1_i32 : i32
      %419 = arith.cmpi sge, %418, %c2_i32 : i32
      %420 = arith.subi %418, %c2_i32 : i32
      %421 = arith.select %419, %420, %418 : i32
      memref.store %421, %_anonymous2[%c1] : memref<3xi32>
      %422 = arith.addi %402, %c1 : index
      cf.br ^bb179(%422 : index)
    ^bb189:  // pred: ^bb179
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %423 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %424 = arith.addi %423, %c1_i32 : i32
      %425 = arith.cmpi sge, %424, %c1_i32 : i32
      %426 = arith.select %425, %423, %424 : i32
      memref.store %426, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb190(%c0 : index)
    ^bb190(%427: index):  // 2 preds: ^bb189, ^bb199
      %428 = arith.cmpi slt, %427, %c16 : index
      cf.cond_br %428, ^bb191, ^bb200
    ^bb191:  // pred: ^bb190
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %429 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %430 = arith.index_cast %429 : i32 to index
      %431 = arith.index_cast %430 : index to i32
      cf.switch %431 : i32, [
        default: ^bb194,
        0: ^bb192,
        1: ^bb193
      ]
    ^bb192:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb193:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb194:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb195(%432: memref<64x128xi8>):  // 3 preds: ^bb192, ^bb193, ^bb194
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %433 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %434 = arith.index_cast %433 : i32 to index
      %435 = arith.index_cast %434 : index to i32
      cf.switch %435 : i32, [
        default: ^bb198,
        0: ^bb196,
        1: ^bb197
      ]
    ^bb196:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb197:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb198:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb199(%436: memref<32x64xi8>):  // 3 preds: ^bb196, ^bb197, ^bb198
      func.call @matmul_i8_i16(%436, %432, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %437 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %438 = arith.addi %437, %c1_i32 : i32
      %439 = arith.cmpi sge, %438, %c2_i32 : i32
      %440 = arith.subi %438, %c2_i32 : i32
      %441 = arith.select %439, %440, %438 : i32
      memref.store %441, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %442 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %443 = arith.addi %442, %c1_i32 : i32
      %444 = arith.cmpi sge, %443, %c2_i32 : i32
      %445 = arith.subi %443, %c2_i32 : i32
      %446 = arith.select %444, %445, %443 : i32
      memref.store %446, %_anonymous2[%c1] : memref<3xi32>
      %447 = arith.addi %427, %c1 : index
      cf.br ^bb190(%447 : index)
    ^bb200:  // pred: ^bb190
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %448 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %449 = arith.addi %448, %c1_i32 : i32
      %450 = arith.cmpi sge, %449, %c1_i32 : i32
      %451 = arith.select %450, %448, %449 : i32
      memref.store %451, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb201(%c0 : index)
    ^bb201(%452: index):  // 2 preds: ^bb200, ^bb210
      %453 = arith.cmpi slt, %452, %c16 : index
      cf.cond_br %453, ^bb202, ^bb211
    ^bb202:  // pred: ^bb201
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %454 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %455 = arith.index_cast %454 : i32 to index
      %456 = arith.index_cast %455 : index to i32
      cf.switch %456 : i32, [
        default: ^bb205,
        0: ^bb203,
        1: ^bb204
      ]
    ^bb203:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb204:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb205:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb206(%457: memref<64x128xi8>):  // 3 preds: ^bb203, ^bb204, ^bb205
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %458 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %459 = arith.index_cast %458 : i32 to index
      %460 = arith.index_cast %459 : index to i32
      cf.switch %460 : i32, [
        default: ^bb209,
        0: ^bb207,
        1: ^bb208
      ]
    ^bb207:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb208:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb209:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb210(%461: memref<32x64xi8>):  // 3 preds: ^bb207, ^bb208, ^bb209
      func.call @matmul_i8_i16(%461, %457, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %462 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %463 = arith.addi %462, %c1_i32 : i32
      %464 = arith.cmpi sge, %463, %c2_i32 : i32
      %465 = arith.subi %463, %c2_i32 : i32
      %466 = arith.select %464, %465, %463 : i32
      memref.store %466, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %467 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %468 = arith.addi %467, %c1_i32 : i32
      %469 = arith.cmpi sge, %468, %c2_i32 : i32
      %470 = arith.subi %468, %c2_i32 : i32
      %471 = arith.select %469, %470, %468 : i32
      memref.store %471, %_anonymous2[%c1] : memref<3xi32>
      %472 = arith.addi %452, %c1 : index
      cf.br ^bb201(%472 : index)
    ^bb211:  // pred: ^bb201
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %473 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %474 = arith.addi %473, %c1_i32 : i32
      %475 = arith.cmpi sge, %474, %c1_i32 : i32
      %476 = arith.select %475, %473, %474 : i32
      memref.store %476, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb212(%c0 : index)
    ^bb212(%477: index):  // 2 preds: ^bb211, ^bb221
      %478 = arith.cmpi slt, %477, %c16 : index
      cf.cond_br %478, ^bb213, ^bb222
    ^bb213:  // pred: ^bb212
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %479 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %480 = arith.index_cast %479 : i32 to index
      %481 = arith.index_cast %480 : index to i32
      cf.switch %481 : i32, [
        default: ^bb216,
        0: ^bb214,
        1: ^bb215
      ]
    ^bb214:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb215:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb216:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb217(%482: memref<64x128xi8>):  // 3 preds: ^bb214, ^bb215, ^bb216
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %483 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %484 = arith.index_cast %483 : i32 to index
      %485 = arith.index_cast %484 : index to i32
      cf.switch %485 : i32, [
        default: ^bb220,
        0: ^bb218,
        1: ^bb219
      ]
    ^bb218:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb219:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb220:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb221(%486: memref<32x64xi8>):  // 3 preds: ^bb218, ^bb219, ^bb220
      func.call @matmul_i8_i16(%486, %482, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %487 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %488 = arith.addi %487, %c1_i32 : i32
      %489 = arith.cmpi sge, %488, %c2_i32 : i32
      %490 = arith.subi %488, %c2_i32 : i32
      %491 = arith.select %489, %490, %488 : i32
      memref.store %491, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %492 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %493 = arith.addi %492, %c1_i32 : i32
      %494 = arith.cmpi sge, %493, %c2_i32 : i32
      %495 = arith.subi %493, %c2_i32 : i32
      %496 = arith.select %494, %495, %493 : i32
      memref.store %496, %_anonymous2[%c1] : memref<3xi32>
      %497 = arith.addi %477, %c1 : index
      cf.br ^bb212(%497 : index)
    ^bb222:  // pred: ^bb212
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %498 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %499 = arith.addi %498, %c1_i32 : i32
      %500 = arith.cmpi sge, %499, %c1_i32 : i32
      %501 = arith.select %500, %498, %499 : i32
      memref.store %501, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb223(%c0 : index)
    ^bb223(%502: index):  // 2 preds: ^bb222, ^bb232
      %503 = arith.cmpi slt, %502, %c16 : index
      cf.cond_br %503, ^bb224, ^bb233
    ^bb224:  // pred: ^bb223
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %504 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %505 = arith.index_cast %504 : i32 to index
      %506 = arith.index_cast %505 : index to i32
      cf.switch %506 : i32, [
        default: ^bb227,
        0: ^bb225,
        1: ^bb226
      ]
    ^bb225:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb226:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb227:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb228(%507: memref<64x128xi8>):  // 3 preds: ^bb225, ^bb226, ^bb227
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %508 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %509 = arith.index_cast %508 : i32 to index
      %510 = arith.index_cast %509 : index to i32
      cf.switch %510 : i32, [
        default: ^bb231,
        0: ^bb229,
        1: ^bb230
      ]
    ^bb229:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb230:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb231:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb232(%511: memref<32x64xi8>):  // 3 preds: ^bb229, ^bb230, ^bb231
      func.call @matmul_i8_i16(%511, %507, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %512 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %513 = arith.addi %512, %c1_i32 : i32
      %514 = arith.cmpi sge, %513, %c2_i32 : i32
      %515 = arith.subi %513, %c2_i32 : i32
      %516 = arith.select %514, %515, %513 : i32
      memref.store %516, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %517 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %518 = arith.addi %517, %c1_i32 : i32
      %519 = arith.cmpi sge, %518, %c2_i32 : i32
      %520 = arith.subi %518, %c2_i32 : i32
      %521 = arith.select %519, %520, %518 : i32
      memref.store %521, %_anonymous2[%c1] : memref<3xi32>
      %522 = arith.addi %502, %c1 : index
      cf.br ^bb223(%522 : index)
    ^bb233:  // pred: ^bb223
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %523 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %524 = arith.addi %523, %c1_i32 : i32
      %525 = arith.cmpi sge, %524, %c1_i32 : i32
      %526 = arith.select %525, %523, %524 : i32
      memref.store %526, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb234(%c0 : index)
    ^bb234(%527: index):  // 2 preds: ^bb233, ^bb243
      %528 = arith.cmpi slt, %527, %c16 : index
      cf.cond_br %528, ^bb235, ^bb244
    ^bb235:  // pred: ^bb234
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %529 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %530 = arith.index_cast %529 : i32 to index
      %531 = arith.index_cast %530 : index to i32
      cf.switch %531 : i32, [
        default: ^bb238,
        0: ^bb236,
        1: ^bb237
      ]
    ^bb236:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb237:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb238:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb239(%532: memref<64x128xi8>):  // 3 preds: ^bb236, ^bb237, ^bb238
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %533 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %534 = arith.index_cast %533 : i32 to index
      %535 = arith.index_cast %534 : index to i32
      cf.switch %535 : i32, [
        default: ^bb242,
        0: ^bb240,
        1: ^bb241
      ]
    ^bb240:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb241:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb242:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb243(%536: memref<32x64xi8>):  // 3 preds: ^bb240, ^bb241, ^bb242
      func.call @matmul_i8_i16(%536, %532, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %537 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %538 = arith.addi %537, %c1_i32 : i32
      %539 = arith.cmpi sge, %538, %c2_i32 : i32
      %540 = arith.subi %538, %c2_i32 : i32
      %541 = arith.select %539, %540, %538 : i32
      memref.store %541, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %542 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %543 = arith.addi %542, %c1_i32 : i32
      %544 = arith.cmpi sge, %543, %c2_i32 : i32
      %545 = arith.subi %543, %c2_i32 : i32
      %546 = arith.select %544, %545, %543 : i32
      memref.store %546, %_anonymous2[%c1] : memref<3xi32>
      %547 = arith.addi %527, %c1 : index
      cf.br ^bb234(%547 : index)
    ^bb244:  // pred: ^bb234
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %548 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %549 = arith.addi %548, %c1_i32 : i32
      %550 = arith.cmpi sge, %549, %c1_i32 : i32
      %551 = arith.select %550, %548, %549 : i32
      memref.store %551, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb245(%c0 : index)
    ^bb245(%552: index):  // 2 preds: ^bb244, ^bb254
      %553 = arith.cmpi slt, %552, %c16 : index
      cf.cond_br %553, ^bb246, ^bb255
    ^bb246:  // pred: ^bb245
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %554 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %555 = arith.index_cast %554 : i32 to index
      %556 = arith.index_cast %555 : index to i32
      cf.switch %556 : i32, [
        default: ^bb249,
        0: ^bb247,
        1: ^bb248
      ]
    ^bb247:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb248:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb249:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb250(%557: memref<64x128xi8>):  // 3 preds: ^bb247, ^bb248, ^bb249
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %558 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %559 = arith.index_cast %558 : i32 to index
      %560 = arith.index_cast %559 : index to i32
      cf.switch %560 : i32, [
        default: ^bb253,
        0: ^bb251,
        1: ^bb252
      ]
    ^bb251:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb252:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb253:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb254(%561: memref<32x64xi8>):  // 3 preds: ^bb251, ^bb252, ^bb253
      func.call @matmul_i8_i16(%561, %557, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %562 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %563 = arith.addi %562, %c1_i32 : i32
      %564 = arith.cmpi sge, %563, %c2_i32 : i32
      %565 = arith.subi %563, %c2_i32 : i32
      %566 = arith.select %564, %565, %563 : i32
      memref.store %566, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %567 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %568 = arith.addi %567, %c1_i32 : i32
      %569 = arith.cmpi sge, %568, %c2_i32 : i32
      %570 = arith.subi %568, %c2_i32 : i32
      %571 = arith.select %569, %570, %568 : i32
      memref.store %571, %_anonymous2[%c1] : memref<3xi32>
      %572 = arith.addi %552, %c1 : index
      cf.br ^bb245(%572 : index)
    ^bb255:  // pred: ^bb245
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %573 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %574 = arith.addi %573, %c1_i32 : i32
      %575 = arith.cmpi sge, %574, %c1_i32 : i32
      %576 = arith.select %575, %573, %574 : i32
      memref.store %576, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb256(%c0 : index)
    ^bb256(%577: index):  // 2 preds: ^bb255, ^bb265
      %578 = arith.cmpi slt, %577, %c16 : index
      cf.cond_br %578, ^bb257, ^bb266
    ^bb257:  // pred: ^bb256
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %579 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %580 = arith.index_cast %579 : i32 to index
      %581 = arith.index_cast %580 : index to i32
      cf.switch %581 : i32, [
        default: ^bb260,
        0: ^bb258,
        1: ^bb259
      ]
    ^bb258:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb259:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb260:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb261(%582: memref<64x128xi8>):  // 3 preds: ^bb258, ^bb259, ^bb260
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %583 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %584 = arith.index_cast %583 : i32 to index
      %585 = arith.index_cast %584 : index to i32
      cf.switch %585 : i32, [
        default: ^bb264,
        0: ^bb262,
        1: ^bb263
      ]
    ^bb262:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb263:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb264:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb265(%586: memref<32x64xi8>):  // 3 preds: ^bb262, ^bb263, ^bb264
      func.call @matmul_i8_i16(%586, %582, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %587 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %588 = arith.addi %587, %c1_i32 : i32
      %589 = arith.cmpi sge, %588, %c2_i32 : i32
      %590 = arith.subi %588, %c2_i32 : i32
      %591 = arith.select %589, %590, %588 : i32
      memref.store %591, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %592 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %593 = arith.addi %592, %c1_i32 : i32
      %594 = arith.cmpi sge, %593, %c2_i32 : i32
      %595 = arith.subi %593, %c2_i32 : i32
      %596 = arith.select %594, %595, %593 : i32
      memref.store %596, %_anonymous2[%c1] : memref<3xi32>
      %597 = arith.addi %577, %c1 : index
      cf.br ^bb256(%597 : index)
    ^bb266:  // pred: ^bb256
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %598 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %599 = arith.addi %598, %c1_i32 : i32
      %600 = arith.cmpi sge, %599, %c1_i32 : i32
      %601 = arith.select %600, %598, %599 : i32
      memref.store %601, %_anonymous2[%c0] : memref<3xi32>
      %602 = arith.addi %0, %c1 : index
      cf.br ^bb1(%602 : index)
    ^bb267:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous3 = aie.buffer(%tile_3_2) {address = 32000 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb266
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb267
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
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
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous3[%c1] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb19(%32: memref<64x128xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb23(%36: memref<32x64xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_i8_i16(%36, %32, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous3[%c1] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb30(%57: memref<64x128xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb34(%61: memref<32x64xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_i8_i16(%61, %57, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous3[%c1] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb41(%82: memref<64x128xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb45(%86: memref<32x64xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_i8_i16(%86, %82, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous3[%c1] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb47(%c0 : index)
    ^bb47(%102: index):  // 2 preds: ^bb46, ^bb56
      %103 = arith.cmpi slt, %102, %c16 : index
      cf.cond_br %103, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %104 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %105 = arith.index_cast %104 : i32 to index
      %106 = arith.index_cast %105 : index to i32
      cf.switch %106 : i32, [
        default: ^bb51,
        0: ^bb49,
        1: ^bb50
      ]
    ^bb49:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb50:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb51:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb52(%107: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb55,
        0: ^bb53,
        1: ^bb54
      ]
    ^bb53:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb54:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb55:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb56(%111: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_i8_i16(%111, %107, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %112 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %113 = arith.addi %112, %c1_i32 : i32
      %114 = arith.cmpi sge, %113, %c2_i32 : i32
      %115 = arith.subi %113, %c2_i32 : i32
      %116 = arith.select %114, %115, %113 : i32
      memref.store %116, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %117 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %118 = arith.addi %117, %c1_i32 : i32
      %119 = arith.cmpi sge, %118, %c2_i32 : i32
      %120 = arith.subi %118, %c2_i32 : i32
      %121 = arith.select %119, %120, %118 : i32
      memref.store %121, %_anonymous3[%c1] : memref<3xi32>
      %122 = arith.addi %102, %c1 : index
      cf.br ^bb47(%122 : index)
    ^bb57:  // pred: ^bb47
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %123 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c1_i32 : i32
      %126 = arith.select %125, %123, %124 : i32
      memref.store %126, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb58(%c0 : index)
    ^bb58(%127: index):  // 2 preds: ^bb57, ^bb67
      %128 = arith.cmpi slt, %127, %c16 : index
      cf.cond_br %128, ^bb59, ^bb68
    ^bb59:  // pred: ^bb58
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %129 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %130 = arith.index_cast %129 : i32 to index
      %131 = arith.index_cast %130 : index to i32
      cf.switch %131 : i32, [
        default: ^bb62,
        0: ^bb60,
        1: ^bb61
      ]
    ^bb60:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb61:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb62:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb63(%132: memref<64x128xi8>):  // 3 preds: ^bb60, ^bb61, ^bb62
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %133 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %134 = arith.index_cast %133 : i32 to index
      %135 = arith.index_cast %134 : index to i32
      cf.switch %135 : i32, [
        default: ^bb66,
        0: ^bb64,
        1: ^bb65
      ]
    ^bb64:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb66:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb67(%136: memref<32x64xi8>):  // 3 preds: ^bb64, ^bb65, ^bb66
      func.call @matmul_i8_i16(%136, %132, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %137 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %138 = arith.addi %137, %c1_i32 : i32
      %139 = arith.cmpi sge, %138, %c2_i32 : i32
      %140 = arith.subi %138, %c2_i32 : i32
      %141 = arith.select %139, %140, %138 : i32
      memref.store %141, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %142 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %143 = arith.addi %142, %c1_i32 : i32
      %144 = arith.cmpi sge, %143, %c2_i32 : i32
      %145 = arith.subi %143, %c2_i32 : i32
      %146 = arith.select %144, %145, %143 : i32
      memref.store %146, %_anonymous3[%c1] : memref<3xi32>
      %147 = arith.addi %127, %c1 : index
      cf.br ^bb58(%147 : index)
    ^bb68:  // pred: ^bb58
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %148 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c1_i32 : i32
      %151 = arith.select %150, %148, %149 : i32
      memref.store %151, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb69(%c0 : index)
    ^bb69(%152: index):  // 2 preds: ^bb68, ^bb78
      %153 = arith.cmpi slt, %152, %c16 : index
      cf.cond_br %153, ^bb70, ^bb79
    ^bb70:  // pred: ^bb69
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %154 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %155 = arith.index_cast %154 : i32 to index
      %156 = arith.index_cast %155 : index to i32
      cf.switch %156 : i32, [
        default: ^bb73,
        0: ^bb71,
        1: ^bb72
      ]
    ^bb71:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb72:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb73:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb74(%157: memref<64x128xi8>):  // 3 preds: ^bb71, ^bb72, ^bb73
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %158 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %159 = arith.index_cast %158 : i32 to index
      %160 = arith.index_cast %159 : index to i32
      cf.switch %160 : i32, [
        default: ^bb77,
        0: ^bb75,
        1: ^bb76
      ]
    ^bb75:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb76:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb77:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb78(%161: memref<32x64xi8>):  // 3 preds: ^bb75, ^bb76, ^bb77
      func.call @matmul_i8_i16(%161, %157, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %162 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %163 = arith.addi %162, %c1_i32 : i32
      %164 = arith.cmpi sge, %163, %c2_i32 : i32
      %165 = arith.subi %163, %c2_i32 : i32
      %166 = arith.select %164, %165, %163 : i32
      memref.store %166, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %167 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %168 = arith.addi %167, %c1_i32 : i32
      %169 = arith.cmpi sge, %168, %c2_i32 : i32
      %170 = arith.subi %168, %c2_i32 : i32
      %171 = arith.select %169, %170, %168 : i32
      memref.store %171, %_anonymous3[%c1] : memref<3xi32>
      %172 = arith.addi %152, %c1 : index
      cf.br ^bb69(%172 : index)
    ^bb79:  // pred: ^bb69
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %173 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %174 = arith.addi %173, %c1_i32 : i32
      %175 = arith.cmpi sge, %174, %c1_i32 : i32
      %176 = arith.select %175, %173, %174 : i32
      memref.store %176, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb80(%c0 : index)
    ^bb80(%177: index):  // 2 preds: ^bb79, ^bb89
      %178 = arith.cmpi slt, %177, %c16 : index
      cf.cond_br %178, ^bb81, ^bb90
    ^bb81:  // pred: ^bb80
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %179 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %180 = arith.index_cast %179 : i32 to index
      %181 = arith.index_cast %180 : index to i32
      cf.switch %181 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb85(%182: memref<64x128xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %183 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %184 = arith.index_cast %183 : i32 to index
      %185 = arith.index_cast %184 : index to i32
      cf.switch %185 : i32, [
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
    ^bb89(%186: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%186, %182, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %187 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %188 = arith.addi %187, %c1_i32 : i32
      %189 = arith.cmpi sge, %188, %c2_i32 : i32
      %190 = arith.subi %188, %c2_i32 : i32
      %191 = arith.select %189, %190, %188 : i32
      memref.store %191, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %192 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %193 = arith.addi %192, %c1_i32 : i32
      %194 = arith.cmpi sge, %193, %c2_i32 : i32
      %195 = arith.subi %193, %c2_i32 : i32
      %196 = arith.select %194, %195, %193 : i32
      memref.store %196, %_anonymous3[%c1] : memref<3xi32>
      %197 = arith.addi %177, %c1 : index
      cf.br ^bb80(%197 : index)
    ^bb90:  // pred: ^bb80
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %198 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %199 = arith.addi %198, %c1_i32 : i32
      %200 = arith.cmpi sge, %199, %c1_i32 : i32
      %201 = arith.select %200, %198, %199 : i32
      memref.store %201, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb91(%c0 : index)
    ^bb91(%202: index):  // 2 preds: ^bb90, ^bb100
      %203 = arith.cmpi slt, %202, %c16 : index
      cf.cond_br %203, ^bb92, ^bb101
    ^bb92:  // pred: ^bb91
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %204 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %205 = arith.index_cast %204 : i32 to index
      %206 = arith.index_cast %205 : index to i32
      cf.switch %206 : i32, [
        default: ^bb95,
        0: ^bb93,
        1: ^bb94
      ]
    ^bb93:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb94:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb95:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb96(%207: memref<64x128xi8>):  // 3 preds: ^bb93, ^bb94, ^bb95
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %208 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %209 = arith.index_cast %208 : i32 to index
      %210 = arith.index_cast %209 : index to i32
      cf.switch %210 : i32, [
        default: ^bb99,
        0: ^bb97,
        1: ^bb98
      ]
    ^bb97:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb98:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb99:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb100(%211: memref<32x64xi8>):  // 3 preds: ^bb97, ^bb98, ^bb99
      func.call @matmul_i8_i16(%211, %207, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %212 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %213 = arith.addi %212, %c1_i32 : i32
      %214 = arith.cmpi sge, %213, %c2_i32 : i32
      %215 = arith.subi %213, %c2_i32 : i32
      %216 = arith.select %214, %215, %213 : i32
      memref.store %216, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %217 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %218 = arith.addi %217, %c1_i32 : i32
      %219 = arith.cmpi sge, %218, %c2_i32 : i32
      %220 = arith.subi %218, %c2_i32 : i32
      %221 = arith.select %219, %220, %218 : i32
      memref.store %221, %_anonymous3[%c1] : memref<3xi32>
      %222 = arith.addi %202, %c1 : index
      cf.br ^bb91(%222 : index)
    ^bb101:  // pred: ^bb91
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %223 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %224 = arith.addi %223, %c1_i32 : i32
      %225 = arith.cmpi sge, %224, %c1_i32 : i32
      %226 = arith.select %225, %223, %224 : i32
      memref.store %226, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb102(%c0 : index)
    ^bb102(%227: index):  // 2 preds: ^bb101, ^bb111
      %228 = arith.cmpi slt, %227, %c16 : index
      cf.cond_br %228, ^bb103, ^bb112
    ^bb103:  // pred: ^bb102
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %229 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %230 = arith.index_cast %229 : i32 to index
      %231 = arith.index_cast %230 : index to i32
      cf.switch %231 : i32, [
        default: ^bb106,
        0: ^bb104,
        1: ^bb105
      ]
    ^bb104:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb105:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb106:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb107(%232: memref<64x128xi8>):  // 3 preds: ^bb104, ^bb105, ^bb106
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %233 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %234 = arith.index_cast %233 : i32 to index
      %235 = arith.index_cast %234 : index to i32
      cf.switch %235 : i32, [
        default: ^bb110,
        0: ^bb108,
        1: ^bb109
      ]
    ^bb108:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb109:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb110:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb111(%236: memref<32x64xi8>):  // 3 preds: ^bb108, ^bb109, ^bb110
      func.call @matmul_i8_i16(%236, %232, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %237 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %238 = arith.addi %237, %c1_i32 : i32
      %239 = arith.cmpi sge, %238, %c2_i32 : i32
      %240 = arith.subi %238, %c2_i32 : i32
      %241 = arith.select %239, %240, %238 : i32
      memref.store %241, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %242 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %243 = arith.addi %242, %c1_i32 : i32
      %244 = arith.cmpi sge, %243, %c2_i32 : i32
      %245 = arith.subi %243, %c2_i32 : i32
      %246 = arith.select %244, %245, %243 : i32
      memref.store %246, %_anonymous3[%c1] : memref<3xi32>
      %247 = arith.addi %227, %c1 : index
      cf.br ^bb102(%247 : index)
    ^bb112:  // pred: ^bb102
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %248 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %249 = arith.addi %248, %c1_i32 : i32
      %250 = arith.cmpi sge, %249, %c1_i32 : i32
      %251 = arith.select %250, %248, %249 : i32
      memref.store %251, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb113(%c0 : index)
    ^bb113(%252: index):  // 2 preds: ^bb112, ^bb122
      %253 = arith.cmpi slt, %252, %c16 : index
      cf.cond_br %253, ^bb114, ^bb123
    ^bb114:  // pred: ^bb113
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %254 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %255 = arith.index_cast %254 : i32 to index
      %256 = arith.index_cast %255 : index to i32
      cf.switch %256 : i32, [
        default: ^bb117,
        0: ^bb115,
        1: ^bb116
      ]
    ^bb115:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb116:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb117:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb118(%257: memref<64x128xi8>):  // 3 preds: ^bb115, ^bb116, ^bb117
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %258 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %259 = arith.index_cast %258 : i32 to index
      %260 = arith.index_cast %259 : index to i32
      cf.switch %260 : i32, [
        default: ^bb121,
        0: ^bb119,
        1: ^bb120
      ]
    ^bb119:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb120:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb121:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb122(%261: memref<32x64xi8>):  // 3 preds: ^bb119, ^bb120, ^bb121
      func.call @matmul_i8_i16(%261, %257, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %262 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %263 = arith.addi %262, %c1_i32 : i32
      %264 = arith.cmpi sge, %263, %c2_i32 : i32
      %265 = arith.subi %263, %c2_i32 : i32
      %266 = arith.select %264, %265, %263 : i32
      memref.store %266, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %267 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %268 = arith.addi %267, %c1_i32 : i32
      %269 = arith.cmpi sge, %268, %c2_i32 : i32
      %270 = arith.subi %268, %c2_i32 : i32
      %271 = arith.select %269, %270, %268 : i32
      memref.store %271, %_anonymous3[%c1] : memref<3xi32>
      %272 = arith.addi %252, %c1 : index
      cf.br ^bb113(%272 : index)
    ^bb123:  // pred: ^bb113
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %273 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %274 = arith.addi %273, %c1_i32 : i32
      %275 = arith.cmpi sge, %274, %c1_i32 : i32
      %276 = arith.select %275, %273, %274 : i32
      memref.store %276, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb124(%c0 : index)
    ^bb124(%277: index):  // 2 preds: ^bb123, ^bb133
      %278 = arith.cmpi slt, %277, %c16 : index
      cf.cond_br %278, ^bb125, ^bb134
    ^bb125:  // pred: ^bb124
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %279 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %280 = arith.index_cast %279 : i32 to index
      %281 = arith.index_cast %280 : index to i32
      cf.switch %281 : i32, [
        default: ^bb128,
        0: ^bb126,
        1: ^bb127
      ]
    ^bb126:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb127:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb128:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb129(%282: memref<64x128xi8>):  // 3 preds: ^bb126, ^bb127, ^bb128
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %283 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %284 = arith.index_cast %283 : i32 to index
      %285 = arith.index_cast %284 : index to i32
      cf.switch %285 : i32, [
        default: ^bb132,
        0: ^bb130,
        1: ^bb131
      ]
    ^bb130:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb131:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb132:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb133(%286: memref<32x64xi8>):  // 3 preds: ^bb130, ^bb131, ^bb132
      func.call @matmul_i8_i16(%286, %282, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %287 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %288 = arith.addi %287, %c1_i32 : i32
      %289 = arith.cmpi sge, %288, %c2_i32 : i32
      %290 = arith.subi %288, %c2_i32 : i32
      %291 = arith.select %289, %290, %288 : i32
      memref.store %291, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %292 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %293 = arith.addi %292, %c1_i32 : i32
      %294 = arith.cmpi sge, %293, %c2_i32 : i32
      %295 = arith.subi %293, %c2_i32 : i32
      %296 = arith.select %294, %295, %293 : i32
      memref.store %296, %_anonymous3[%c1] : memref<3xi32>
      %297 = arith.addi %277, %c1 : index
      cf.br ^bb124(%297 : index)
    ^bb134:  // pred: ^bb124
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %298 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %299 = arith.addi %298, %c1_i32 : i32
      %300 = arith.cmpi sge, %299, %c1_i32 : i32
      %301 = arith.select %300, %298, %299 : i32
      memref.store %301, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb135(%c0 : index)
    ^bb135(%302: index):  // 2 preds: ^bb134, ^bb144
      %303 = arith.cmpi slt, %302, %c16 : index
      cf.cond_br %303, ^bb136, ^bb145
    ^bb136:  // pred: ^bb135
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %304 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %305 = arith.index_cast %304 : i32 to index
      %306 = arith.index_cast %305 : index to i32
      cf.switch %306 : i32, [
        default: ^bb139,
        0: ^bb137,
        1: ^bb138
      ]
    ^bb137:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb138:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb139:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb140(%307: memref<64x128xi8>):  // 3 preds: ^bb137, ^bb138, ^bb139
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %308 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %309 = arith.index_cast %308 : i32 to index
      %310 = arith.index_cast %309 : index to i32
      cf.switch %310 : i32, [
        default: ^bb143,
        0: ^bb141,
        1: ^bb142
      ]
    ^bb141:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb142:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb143:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb144(%311: memref<32x64xi8>):  // 3 preds: ^bb141, ^bb142, ^bb143
      func.call @matmul_i8_i16(%311, %307, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %312 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %313 = arith.addi %312, %c1_i32 : i32
      %314 = arith.cmpi sge, %313, %c2_i32 : i32
      %315 = arith.subi %313, %c2_i32 : i32
      %316 = arith.select %314, %315, %313 : i32
      memref.store %316, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %317 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %318 = arith.addi %317, %c1_i32 : i32
      %319 = arith.cmpi sge, %318, %c2_i32 : i32
      %320 = arith.subi %318, %c2_i32 : i32
      %321 = arith.select %319, %320, %318 : i32
      memref.store %321, %_anonymous3[%c1] : memref<3xi32>
      %322 = arith.addi %302, %c1 : index
      cf.br ^bb135(%322 : index)
    ^bb145:  // pred: ^bb135
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %323 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %324 = arith.addi %323, %c1_i32 : i32
      %325 = arith.cmpi sge, %324, %c1_i32 : i32
      %326 = arith.select %325, %323, %324 : i32
      memref.store %326, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb146(%c0 : index)
    ^bb146(%327: index):  // 2 preds: ^bb145, ^bb155
      %328 = arith.cmpi slt, %327, %c16 : index
      cf.cond_br %328, ^bb147, ^bb156
    ^bb147:  // pred: ^bb146
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %329 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %330 = arith.index_cast %329 : i32 to index
      %331 = arith.index_cast %330 : index to i32
      cf.switch %331 : i32, [
        default: ^bb150,
        0: ^bb148,
        1: ^bb149
      ]
    ^bb148:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb149:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb150:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb151(%332: memref<64x128xi8>):  // 3 preds: ^bb148, ^bb149, ^bb150
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %333 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %334 = arith.index_cast %333 : i32 to index
      %335 = arith.index_cast %334 : index to i32
      cf.switch %335 : i32, [
        default: ^bb154,
        0: ^bb152,
        1: ^bb153
      ]
    ^bb152:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb153:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb154:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb155(%336: memref<32x64xi8>):  // 3 preds: ^bb152, ^bb153, ^bb154
      func.call @matmul_i8_i16(%336, %332, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %337 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %338 = arith.addi %337, %c1_i32 : i32
      %339 = arith.cmpi sge, %338, %c2_i32 : i32
      %340 = arith.subi %338, %c2_i32 : i32
      %341 = arith.select %339, %340, %338 : i32
      memref.store %341, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %342 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %343 = arith.addi %342, %c1_i32 : i32
      %344 = arith.cmpi sge, %343, %c2_i32 : i32
      %345 = arith.subi %343, %c2_i32 : i32
      %346 = arith.select %344, %345, %343 : i32
      memref.store %346, %_anonymous3[%c1] : memref<3xi32>
      %347 = arith.addi %327, %c1 : index
      cf.br ^bb146(%347 : index)
    ^bb156:  // pred: ^bb146
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %348 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %349 = arith.addi %348, %c1_i32 : i32
      %350 = arith.cmpi sge, %349, %c1_i32 : i32
      %351 = arith.select %350, %348, %349 : i32
      memref.store %351, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb157(%c0 : index)
    ^bb157(%352: index):  // 2 preds: ^bb156, ^bb166
      %353 = arith.cmpi slt, %352, %c16 : index
      cf.cond_br %353, ^bb158, ^bb167
    ^bb158:  // pred: ^bb157
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %354 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %355 = arith.index_cast %354 : i32 to index
      %356 = arith.index_cast %355 : index to i32
      cf.switch %356 : i32, [
        default: ^bb161,
        0: ^bb159,
        1: ^bb160
      ]
    ^bb159:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb160:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb161:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb162(%357: memref<64x128xi8>):  // 3 preds: ^bb159, ^bb160, ^bb161
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %358 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %359 = arith.index_cast %358 : i32 to index
      %360 = arith.index_cast %359 : index to i32
      cf.switch %360 : i32, [
        default: ^bb165,
        0: ^bb163,
        1: ^bb164
      ]
    ^bb163:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb164:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb165:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb166(%361: memref<32x64xi8>):  // 3 preds: ^bb163, ^bb164, ^bb165
      func.call @matmul_i8_i16(%361, %357, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %362 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %363 = arith.addi %362, %c1_i32 : i32
      %364 = arith.cmpi sge, %363, %c2_i32 : i32
      %365 = arith.subi %363, %c2_i32 : i32
      %366 = arith.select %364, %365, %363 : i32
      memref.store %366, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %367 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %368 = arith.addi %367, %c1_i32 : i32
      %369 = arith.cmpi sge, %368, %c2_i32 : i32
      %370 = arith.subi %368, %c2_i32 : i32
      %371 = arith.select %369, %370, %368 : i32
      memref.store %371, %_anonymous3[%c1] : memref<3xi32>
      %372 = arith.addi %352, %c1 : index
      cf.br ^bb157(%372 : index)
    ^bb167:  // pred: ^bb157
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %373 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %374 = arith.addi %373, %c1_i32 : i32
      %375 = arith.cmpi sge, %374, %c1_i32 : i32
      %376 = arith.select %375, %373, %374 : i32
      memref.store %376, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb168(%c0 : index)
    ^bb168(%377: index):  // 2 preds: ^bb167, ^bb177
      %378 = arith.cmpi slt, %377, %c16 : index
      cf.cond_br %378, ^bb169, ^bb178
    ^bb169:  // pred: ^bb168
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %379 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %380 = arith.index_cast %379 : i32 to index
      %381 = arith.index_cast %380 : index to i32
      cf.switch %381 : i32, [
        default: ^bb172,
        0: ^bb170,
        1: ^bb171
      ]
    ^bb170:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb171:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb172:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb173(%382: memref<64x128xi8>):  // 3 preds: ^bb170, ^bb171, ^bb172
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %383 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %384 = arith.index_cast %383 : i32 to index
      %385 = arith.index_cast %384 : index to i32
      cf.switch %385 : i32, [
        default: ^bb176,
        0: ^bb174,
        1: ^bb175
      ]
    ^bb174:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb175:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb176:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb177(%386: memref<32x64xi8>):  // 3 preds: ^bb174, ^bb175, ^bb176
      func.call @matmul_i8_i16(%386, %382, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %387 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %388 = arith.addi %387, %c1_i32 : i32
      %389 = arith.cmpi sge, %388, %c2_i32 : i32
      %390 = arith.subi %388, %c2_i32 : i32
      %391 = arith.select %389, %390, %388 : i32
      memref.store %391, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %392 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %393 = arith.addi %392, %c1_i32 : i32
      %394 = arith.cmpi sge, %393, %c2_i32 : i32
      %395 = arith.subi %393, %c2_i32 : i32
      %396 = arith.select %394, %395, %393 : i32
      memref.store %396, %_anonymous3[%c1] : memref<3xi32>
      %397 = arith.addi %377, %c1 : index
      cf.br ^bb168(%397 : index)
    ^bb178:  // pred: ^bb168
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %398 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %399 = arith.addi %398, %c1_i32 : i32
      %400 = arith.cmpi sge, %399, %c1_i32 : i32
      %401 = arith.select %400, %398, %399 : i32
      memref.store %401, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb179(%c0 : index)
    ^bb179(%402: index):  // 2 preds: ^bb178, ^bb188
      %403 = arith.cmpi slt, %402, %c16 : index
      cf.cond_br %403, ^bb180, ^bb189
    ^bb180:  // pred: ^bb179
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %404 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %405 = arith.index_cast %404 : i32 to index
      %406 = arith.index_cast %405 : index to i32
      cf.switch %406 : i32, [
        default: ^bb183,
        0: ^bb181,
        1: ^bb182
      ]
    ^bb181:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb182:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb183:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb184(%407: memref<64x128xi8>):  // 3 preds: ^bb181, ^bb182, ^bb183
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %408 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %409 = arith.index_cast %408 : i32 to index
      %410 = arith.index_cast %409 : index to i32
      cf.switch %410 : i32, [
        default: ^bb187,
        0: ^bb185,
        1: ^bb186
      ]
    ^bb185:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb186:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb187:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb188(%411: memref<32x64xi8>):  // 3 preds: ^bb185, ^bb186, ^bb187
      func.call @matmul_i8_i16(%411, %407, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %412 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %413 = arith.addi %412, %c1_i32 : i32
      %414 = arith.cmpi sge, %413, %c2_i32 : i32
      %415 = arith.subi %413, %c2_i32 : i32
      %416 = arith.select %414, %415, %413 : i32
      memref.store %416, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %417 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %418 = arith.addi %417, %c1_i32 : i32
      %419 = arith.cmpi sge, %418, %c2_i32 : i32
      %420 = arith.subi %418, %c2_i32 : i32
      %421 = arith.select %419, %420, %418 : i32
      memref.store %421, %_anonymous3[%c1] : memref<3xi32>
      %422 = arith.addi %402, %c1 : index
      cf.br ^bb179(%422 : index)
    ^bb189:  // pred: ^bb179
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %423 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %424 = arith.addi %423, %c1_i32 : i32
      %425 = arith.cmpi sge, %424, %c1_i32 : i32
      %426 = arith.select %425, %423, %424 : i32
      memref.store %426, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb190(%c0 : index)
    ^bb190(%427: index):  // 2 preds: ^bb189, ^bb199
      %428 = arith.cmpi slt, %427, %c16 : index
      cf.cond_br %428, ^bb191, ^bb200
    ^bb191:  // pred: ^bb190
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %429 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %430 = arith.index_cast %429 : i32 to index
      %431 = arith.index_cast %430 : index to i32
      cf.switch %431 : i32, [
        default: ^bb194,
        0: ^bb192,
        1: ^bb193
      ]
    ^bb192:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb193:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb194:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb195(%432: memref<64x128xi8>):  // 3 preds: ^bb192, ^bb193, ^bb194
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %433 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %434 = arith.index_cast %433 : i32 to index
      %435 = arith.index_cast %434 : index to i32
      cf.switch %435 : i32, [
        default: ^bb198,
        0: ^bb196,
        1: ^bb197
      ]
    ^bb196:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb197:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb198:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb199(%436: memref<32x64xi8>):  // 3 preds: ^bb196, ^bb197, ^bb198
      func.call @matmul_i8_i16(%436, %432, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %437 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %438 = arith.addi %437, %c1_i32 : i32
      %439 = arith.cmpi sge, %438, %c2_i32 : i32
      %440 = arith.subi %438, %c2_i32 : i32
      %441 = arith.select %439, %440, %438 : i32
      memref.store %441, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %442 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %443 = arith.addi %442, %c1_i32 : i32
      %444 = arith.cmpi sge, %443, %c2_i32 : i32
      %445 = arith.subi %443, %c2_i32 : i32
      %446 = arith.select %444, %445, %443 : i32
      memref.store %446, %_anonymous3[%c1] : memref<3xi32>
      %447 = arith.addi %427, %c1 : index
      cf.br ^bb190(%447 : index)
    ^bb200:  // pred: ^bb190
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %448 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %449 = arith.addi %448, %c1_i32 : i32
      %450 = arith.cmpi sge, %449, %c1_i32 : i32
      %451 = arith.select %450, %448, %449 : i32
      memref.store %451, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb201(%c0 : index)
    ^bb201(%452: index):  // 2 preds: ^bb200, ^bb210
      %453 = arith.cmpi slt, %452, %c16 : index
      cf.cond_br %453, ^bb202, ^bb211
    ^bb202:  // pred: ^bb201
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %454 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %455 = arith.index_cast %454 : i32 to index
      %456 = arith.index_cast %455 : index to i32
      cf.switch %456 : i32, [
        default: ^bb205,
        0: ^bb203,
        1: ^bb204
      ]
    ^bb203:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb204:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb205:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb206(%457: memref<64x128xi8>):  // 3 preds: ^bb203, ^bb204, ^bb205
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %458 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %459 = arith.index_cast %458 : i32 to index
      %460 = arith.index_cast %459 : index to i32
      cf.switch %460 : i32, [
        default: ^bb209,
        0: ^bb207,
        1: ^bb208
      ]
    ^bb207:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb208:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb209:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb210(%461: memref<32x64xi8>):  // 3 preds: ^bb207, ^bb208, ^bb209
      func.call @matmul_i8_i16(%461, %457, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %462 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %463 = arith.addi %462, %c1_i32 : i32
      %464 = arith.cmpi sge, %463, %c2_i32 : i32
      %465 = arith.subi %463, %c2_i32 : i32
      %466 = arith.select %464, %465, %463 : i32
      memref.store %466, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %467 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %468 = arith.addi %467, %c1_i32 : i32
      %469 = arith.cmpi sge, %468, %c2_i32 : i32
      %470 = arith.subi %468, %c2_i32 : i32
      %471 = arith.select %469, %470, %468 : i32
      memref.store %471, %_anonymous3[%c1] : memref<3xi32>
      %472 = arith.addi %452, %c1 : index
      cf.br ^bb201(%472 : index)
    ^bb211:  // pred: ^bb201
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %473 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %474 = arith.addi %473, %c1_i32 : i32
      %475 = arith.cmpi sge, %474, %c1_i32 : i32
      %476 = arith.select %475, %473, %474 : i32
      memref.store %476, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb212(%c0 : index)
    ^bb212(%477: index):  // 2 preds: ^bb211, ^bb221
      %478 = arith.cmpi slt, %477, %c16 : index
      cf.cond_br %478, ^bb213, ^bb222
    ^bb213:  // pred: ^bb212
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %479 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %480 = arith.index_cast %479 : i32 to index
      %481 = arith.index_cast %480 : index to i32
      cf.switch %481 : i32, [
        default: ^bb216,
        0: ^bb214,
        1: ^bb215
      ]
    ^bb214:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb215:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb216:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb217(%482: memref<64x128xi8>):  // 3 preds: ^bb214, ^bb215, ^bb216
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %483 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %484 = arith.index_cast %483 : i32 to index
      %485 = arith.index_cast %484 : index to i32
      cf.switch %485 : i32, [
        default: ^bb220,
        0: ^bb218,
        1: ^bb219
      ]
    ^bb218:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb219:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb220:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb221(%486: memref<32x64xi8>):  // 3 preds: ^bb218, ^bb219, ^bb220
      func.call @matmul_i8_i16(%486, %482, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %487 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %488 = arith.addi %487, %c1_i32 : i32
      %489 = arith.cmpi sge, %488, %c2_i32 : i32
      %490 = arith.subi %488, %c2_i32 : i32
      %491 = arith.select %489, %490, %488 : i32
      memref.store %491, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %492 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %493 = arith.addi %492, %c1_i32 : i32
      %494 = arith.cmpi sge, %493, %c2_i32 : i32
      %495 = arith.subi %493, %c2_i32 : i32
      %496 = arith.select %494, %495, %493 : i32
      memref.store %496, %_anonymous3[%c1] : memref<3xi32>
      %497 = arith.addi %477, %c1 : index
      cf.br ^bb212(%497 : index)
    ^bb222:  // pred: ^bb212
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %498 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %499 = arith.addi %498, %c1_i32 : i32
      %500 = arith.cmpi sge, %499, %c1_i32 : i32
      %501 = arith.select %500, %498, %499 : i32
      memref.store %501, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb223(%c0 : index)
    ^bb223(%502: index):  // 2 preds: ^bb222, ^bb232
      %503 = arith.cmpi slt, %502, %c16 : index
      cf.cond_br %503, ^bb224, ^bb233
    ^bb224:  // pred: ^bb223
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %504 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %505 = arith.index_cast %504 : i32 to index
      %506 = arith.index_cast %505 : index to i32
      cf.switch %506 : i32, [
        default: ^bb227,
        0: ^bb225,
        1: ^bb226
      ]
    ^bb225:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb226:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb227:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb228(%507: memref<64x128xi8>):  // 3 preds: ^bb225, ^bb226, ^bb227
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %508 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %509 = arith.index_cast %508 : i32 to index
      %510 = arith.index_cast %509 : index to i32
      cf.switch %510 : i32, [
        default: ^bb231,
        0: ^bb229,
        1: ^bb230
      ]
    ^bb229:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb230:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb231:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb232(%511: memref<32x64xi8>):  // 3 preds: ^bb229, ^bb230, ^bb231
      func.call @matmul_i8_i16(%511, %507, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %512 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %513 = arith.addi %512, %c1_i32 : i32
      %514 = arith.cmpi sge, %513, %c2_i32 : i32
      %515 = arith.subi %513, %c2_i32 : i32
      %516 = arith.select %514, %515, %513 : i32
      memref.store %516, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %517 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %518 = arith.addi %517, %c1_i32 : i32
      %519 = arith.cmpi sge, %518, %c2_i32 : i32
      %520 = arith.subi %518, %c2_i32 : i32
      %521 = arith.select %519, %520, %518 : i32
      memref.store %521, %_anonymous3[%c1] : memref<3xi32>
      %522 = arith.addi %502, %c1 : index
      cf.br ^bb223(%522 : index)
    ^bb233:  // pred: ^bb223
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %523 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %524 = arith.addi %523, %c1_i32 : i32
      %525 = arith.cmpi sge, %524, %c1_i32 : i32
      %526 = arith.select %525, %523, %524 : i32
      memref.store %526, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb234(%c0 : index)
    ^bb234(%527: index):  // 2 preds: ^bb233, ^bb243
      %528 = arith.cmpi slt, %527, %c16 : index
      cf.cond_br %528, ^bb235, ^bb244
    ^bb235:  // pred: ^bb234
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %529 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %530 = arith.index_cast %529 : i32 to index
      %531 = arith.index_cast %530 : index to i32
      cf.switch %531 : i32, [
        default: ^bb238,
        0: ^bb236,
        1: ^bb237
      ]
    ^bb236:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb237:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb238:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb239(%532: memref<64x128xi8>):  // 3 preds: ^bb236, ^bb237, ^bb238
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %533 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %534 = arith.index_cast %533 : i32 to index
      %535 = arith.index_cast %534 : index to i32
      cf.switch %535 : i32, [
        default: ^bb242,
        0: ^bb240,
        1: ^bb241
      ]
    ^bb240:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb241:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb242:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb243(%536: memref<32x64xi8>):  // 3 preds: ^bb240, ^bb241, ^bb242
      func.call @matmul_i8_i16(%536, %532, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %537 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %538 = arith.addi %537, %c1_i32 : i32
      %539 = arith.cmpi sge, %538, %c2_i32 : i32
      %540 = arith.subi %538, %c2_i32 : i32
      %541 = arith.select %539, %540, %538 : i32
      memref.store %541, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %542 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %543 = arith.addi %542, %c1_i32 : i32
      %544 = arith.cmpi sge, %543, %c2_i32 : i32
      %545 = arith.subi %543, %c2_i32 : i32
      %546 = arith.select %544, %545, %543 : i32
      memref.store %546, %_anonymous3[%c1] : memref<3xi32>
      %547 = arith.addi %527, %c1 : index
      cf.br ^bb234(%547 : index)
    ^bb244:  // pred: ^bb234
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %548 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %549 = arith.addi %548, %c1_i32 : i32
      %550 = arith.cmpi sge, %549, %c1_i32 : i32
      %551 = arith.select %550, %548, %549 : i32
      memref.store %551, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb245(%c0 : index)
    ^bb245(%552: index):  // 2 preds: ^bb244, ^bb254
      %553 = arith.cmpi slt, %552, %c16 : index
      cf.cond_br %553, ^bb246, ^bb255
    ^bb246:  // pred: ^bb245
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %554 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %555 = arith.index_cast %554 : i32 to index
      %556 = arith.index_cast %555 : index to i32
      cf.switch %556 : i32, [
        default: ^bb249,
        0: ^bb247,
        1: ^bb248
      ]
    ^bb247:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb248:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb249:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb250(%557: memref<64x128xi8>):  // 3 preds: ^bb247, ^bb248, ^bb249
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %558 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %559 = arith.index_cast %558 : i32 to index
      %560 = arith.index_cast %559 : index to i32
      cf.switch %560 : i32, [
        default: ^bb253,
        0: ^bb251,
        1: ^bb252
      ]
    ^bb251:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb252:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb253:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb254(%561: memref<32x64xi8>):  // 3 preds: ^bb251, ^bb252, ^bb253
      func.call @matmul_i8_i16(%561, %557, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %562 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %563 = arith.addi %562, %c1_i32 : i32
      %564 = arith.cmpi sge, %563, %c2_i32 : i32
      %565 = arith.subi %563, %c2_i32 : i32
      %566 = arith.select %564, %565, %563 : i32
      memref.store %566, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %567 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %568 = arith.addi %567, %c1_i32 : i32
      %569 = arith.cmpi sge, %568, %c2_i32 : i32
      %570 = arith.subi %568, %c2_i32 : i32
      %571 = arith.select %569, %570, %568 : i32
      memref.store %571, %_anonymous3[%c1] : memref<3xi32>
      %572 = arith.addi %552, %c1 : index
      cf.br ^bb245(%572 : index)
    ^bb255:  // pred: ^bb245
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %573 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %574 = arith.addi %573, %c1_i32 : i32
      %575 = arith.cmpi sge, %574, %c1_i32 : i32
      %576 = arith.select %575, %573, %574 : i32
      memref.store %576, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb256(%c0 : index)
    ^bb256(%577: index):  // 2 preds: ^bb255, ^bb265
      %578 = arith.cmpi slt, %577, %c16 : index
      cf.cond_br %578, ^bb257, ^bb266
    ^bb257:  // pred: ^bb256
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %579 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %580 = arith.index_cast %579 : i32 to index
      %581 = arith.index_cast %580 : index to i32
      cf.switch %581 : i32, [
        default: ^bb260,
        0: ^bb258,
        1: ^bb259
      ]
    ^bb258:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb259:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb260:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb261(%582: memref<64x128xi8>):  // 3 preds: ^bb258, ^bb259, ^bb260
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %583 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %584 = arith.index_cast %583 : i32 to index
      %585 = arith.index_cast %584 : index to i32
      cf.switch %585 : i32, [
        default: ^bb264,
        0: ^bb262,
        1: ^bb263
      ]
    ^bb262:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb263:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb264:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb265(%586: memref<32x64xi8>):  // 3 preds: ^bb262, ^bb263, ^bb264
      func.call @matmul_i8_i16(%586, %582, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %587 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %588 = arith.addi %587, %c1_i32 : i32
      %589 = arith.cmpi sge, %588, %c2_i32 : i32
      %590 = arith.subi %588, %c2_i32 : i32
      %591 = arith.select %589, %590, %588 : i32
      memref.store %591, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %592 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %593 = arith.addi %592, %c1_i32 : i32
      %594 = arith.cmpi sge, %593, %c2_i32 : i32
      %595 = arith.subi %593, %c2_i32 : i32
      %596 = arith.select %594, %595, %593 : i32
      memref.store %596, %_anonymous3[%c1] : memref<3xi32>
      %597 = arith.addi %577, %c1 : index
      cf.br ^bb256(%597 : index)
    ^bb266:  // pred: ^bb256
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %598 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %599 = arith.addi %598, %c1_i32 : i32
      %600 = arith.cmpi sge, %599, %c1_i32 : i32
      %601 = arith.select %600, %598, %599 : i32
      memref.store %601, %_anonymous3[%c0] : memref<3xi32>
      %602 = arith.addi %0, %c1 : index
      cf.br ^bb1(%602 : index)
    ^bb267:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous4 = aie.buffer(%tile_4_2) {address = 32000 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb266
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb267
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
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
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous4[%c1] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb19(%32: memref<64x128xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb23(%36: memref<32x64xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_i8_i16(%36, %32, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous4[%c1] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb30(%57: memref<64x128xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb34(%61: memref<32x64xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_i8_i16(%61, %57, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous4[%c1] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb41(%82: memref<64x128xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb45(%86: memref<32x64xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_i8_i16(%86, %82, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous4[%c1] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb47(%c0 : index)
    ^bb47(%102: index):  // 2 preds: ^bb46, ^bb56
      %103 = arith.cmpi slt, %102, %c16 : index
      cf.cond_br %103, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %104 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %105 = arith.index_cast %104 : i32 to index
      %106 = arith.index_cast %105 : index to i32
      cf.switch %106 : i32, [
        default: ^bb51,
        0: ^bb49,
        1: ^bb50
      ]
    ^bb49:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb50:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb51:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb52(%107: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb55,
        0: ^bb53,
        1: ^bb54
      ]
    ^bb53:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb54:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb55:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb56(%111: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_i8_i16(%111, %107, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %112 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %113 = arith.addi %112, %c1_i32 : i32
      %114 = arith.cmpi sge, %113, %c2_i32 : i32
      %115 = arith.subi %113, %c2_i32 : i32
      %116 = arith.select %114, %115, %113 : i32
      memref.store %116, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %117 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %118 = arith.addi %117, %c1_i32 : i32
      %119 = arith.cmpi sge, %118, %c2_i32 : i32
      %120 = arith.subi %118, %c2_i32 : i32
      %121 = arith.select %119, %120, %118 : i32
      memref.store %121, %_anonymous4[%c1] : memref<3xi32>
      %122 = arith.addi %102, %c1 : index
      cf.br ^bb47(%122 : index)
    ^bb57:  // pred: ^bb47
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %123 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c1_i32 : i32
      %126 = arith.select %125, %123, %124 : i32
      memref.store %126, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb58(%c0 : index)
    ^bb58(%127: index):  // 2 preds: ^bb57, ^bb67
      %128 = arith.cmpi slt, %127, %c16 : index
      cf.cond_br %128, ^bb59, ^bb68
    ^bb59:  // pred: ^bb58
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %129 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %130 = arith.index_cast %129 : i32 to index
      %131 = arith.index_cast %130 : index to i32
      cf.switch %131 : i32, [
        default: ^bb62,
        0: ^bb60,
        1: ^bb61
      ]
    ^bb60:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb61:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb62:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb63(%132: memref<64x128xi8>):  // 3 preds: ^bb60, ^bb61, ^bb62
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %133 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %134 = arith.index_cast %133 : i32 to index
      %135 = arith.index_cast %134 : index to i32
      cf.switch %135 : i32, [
        default: ^bb66,
        0: ^bb64,
        1: ^bb65
      ]
    ^bb64:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb66:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb67(%136: memref<32x64xi8>):  // 3 preds: ^bb64, ^bb65, ^bb66
      func.call @matmul_i8_i16(%136, %132, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %137 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %138 = arith.addi %137, %c1_i32 : i32
      %139 = arith.cmpi sge, %138, %c2_i32 : i32
      %140 = arith.subi %138, %c2_i32 : i32
      %141 = arith.select %139, %140, %138 : i32
      memref.store %141, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %142 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %143 = arith.addi %142, %c1_i32 : i32
      %144 = arith.cmpi sge, %143, %c2_i32 : i32
      %145 = arith.subi %143, %c2_i32 : i32
      %146 = arith.select %144, %145, %143 : i32
      memref.store %146, %_anonymous4[%c1] : memref<3xi32>
      %147 = arith.addi %127, %c1 : index
      cf.br ^bb58(%147 : index)
    ^bb68:  // pred: ^bb58
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %148 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c1_i32 : i32
      %151 = arith.select %150, %148, %149 : i32
      memref.store %151, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb69(%c0 : index)
    ^bb69(%152: index):  // 2 preds: ^bb68, ^bb78
      %153 = arith.cmpi slt, %152, %c16 : index
      cf.cond_br %153, ^bb70, ^bb79
    ^bb70:  // pred: ^bb69
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %154 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %155 = arith.index_cast %154 : i32 to index
      %156 = arith.index_cast %155 : index to i32
      cf.switch %156 : i32, [
        default: ^bb73,
        0: ^bb71,
        1: ^bb72
      ]
    ^bb71:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb72:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb73:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb74(%157: memref<64x128xi8>):  // 3 preds: ^bb71, ^bb72, ^bb73
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %158 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %159 = arith.index_cast %158 : i32 to index
      %160 = arith.index_cast %159 : index to i32
      cf.switch %160 : i32, [
        default: ^bb77,
        0: ^bb75,
        1: ^bb76
      ]
    ^bb75:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb76:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb77:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb78(%161: memref<32x64xi8>):  // 3 preds: ^bb75, ^bb76, ^bb77
      func.call @matmul_i8_i16(%161, %157, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %162 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %163 = arith.addi %162, %c1_i32 : i32
      %164 = arith.cmpi sge, %163, %c2_i32 : i32
      %165 = arith.subi %163, %c2_i32 : i32
      %166 = arith.select %164, %165, %163 : i32
      memref.store %166, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %167 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %168 = arith.addi %167, %c1_i32 : i32
      %169 = arith.cmpi sge, %168, %c2_i32 : i32
      %170 = arith.subi %168, %c2_i32 : i32
      %171 = arith.select %169, %170, %168 : i32
      memref.store %171, %_anonymous4[%c1] : memref<3xi32>
      %172 = arith.addi %152, %c1 : index
      cf.br ^bb69(%172 : index)
    ^bb79:  // pred: ^bb69
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %173 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %174 = arith.addi %173, %c1_i32 : i32
      %175 = arith.cmpi sge, %174, %c1_i32 : i32
      %176 = arith.select %175, %173, %174 : i32
      memref.store %176, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb80(%c0 : index)
    ^bb80(%177: index):  // 2 preds: ^bb79, ^bb89
      %178 = arith.cmpi slt, %177, %c16 : index
      cf.cond_br %178, ^bb81, ^bb90
    ^bb81:  // pred: ^bb80
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %179 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %180 = arith.index_cast %179 : i32 to index
      %181 = arith.index_cast %180 : index to i32
      cf.switch %181 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb85(%182: memref<64x128xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %183 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %184 = arith.index_cast %183 : i32 to index
      %185 = arith.index_cast %184 : index to i32
      cf.switch %185 : i32, [
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
    ^bb89(%186: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%186, %182, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %187 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %188 = arith.addi %187, %c1_i32 : i32
      %189 = arith.cmpi sge, %188, %c2_i32 : i32
      %190 = arith.subi %188, %c2_i32 : i32
      %191 = arith.select %189, %190, %188 : i32
      memref.store %191, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %192 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %193 = arith.addi %192, %c1_i32 : i32
      %194 = arith.cmpi sge, %193, %c2_i32 : i32
      %195 = arith.subi %193, %c2_i32 : i32
      %196 = arith.select %194, %195, %193 : i32
      memref.store %196, %_anonymous4[%c1] : memref<3xi32>
      %197 = arith.addi %177, %c1 : index
      cf.br ^bb80(%197 : index)
    ^bb90:  // pred: ^bb80
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %198 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %199 = arith.addi %198, %c1_i32 : i32
      %200 = arith.cmpi sge, %199, %c1_i32 : i32
      %201 = arith.select %200, %198, %199 : i32
      memref.store %201, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb91(%c0 : index)
    ^bb91(%202: index):  // 2 preds: ^bb90, ^bb100
      %203 = arith.cmpi slt, %202, %c16 : index
      cf.cond_br %203, ^bb92, ^bb101
    ^bb92:  // pred: ^bb91
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %204 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %205 = arith.index_cast %204 : i32 to index
      %206 = arith.index_cast %205 : index to i32
      cf.switch %206 : i32, [
        default: ^bb95,
        0: ^bb93,
        1: ^bb94
      ]
    ^bb93:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb94:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb95:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb96(%207: memref<64x128xi8>):  // 3 preds: ^bb93, ^bb94, ^bb95
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %208 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %209 = arith.index_cast %208 : i32 to index
      %210 = arith.index_cast %209 : index to i32
      cf.switch %210 : i32, [
        default: ^bb99,
        0: ^bb97,
        1: ^bb98
      ]
    ^bb97:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb98:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb99:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb100(%211: memref<32x64xi8>):  // 3 preds: ^bb97, ^bb98, ^bb99
      func.call @matmul_i8_i16(%211, %207, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %212 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %213 = arith.addi %212, %c1_i32 : i32
      %214 = arith.cmpi sge, %213, %c2_i32 : i32
      %215 = arith.subi %213, %c2_i32 : i32
      %216 = arith.select %214, %215, %213 : i32
      memref.store %216, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %217 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %218 = arith.addi %217, %c1_i32 : i32
      %219 = arith.cmpi sge, %218, %c2_i32 : i32
      %220 = arith.subi %218, %c2_i32 : i32
      %221 = arith.select %219, %220, %218 : i32
      memref.store %221, %_anonymous4[%c1] : memref<3xi32>
      %222 = arith.addi %202, %c1 : index
      cf.br ^bb91(%222 : index)
    ^bb101:  // pred: ^bb91
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %223 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %224 = arith.addi %223, %c1_i32 : i32
      %225 = arith.cmpi sge, %224, %c1_i32 : i32
      %226 = arith.select %225, %223, %224 : i32
      memref.store %226, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb102(%c0 : index)
    ^bb102(%227: index):  // 2 preds: ^bb101, ^bb111
      %228 = arith.cmpi slt, %227, %c16 : index
      cf.cond_br %228, ^bb103, ^bb112
    ^bb103:  // pred: ^bb102
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %229 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %230 = arith.index_cast %229 : i32 to index
      %231 = arith.index_cast %230 : index to i32
      cf.switch %231 : i32, [
        default: ^bb106,
        0: ^bb104,
        1: ^bb105
      ]
    ^bb104:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb105:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb106:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb107(%232: memref<64x128xi8>):  // 3 preds: ^bb104, ^bb105, ^bb106
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %233 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %234 = arith.index_cast %233 : i32 to index
      %235 = arith.index_cast %234 : index to i32
      cf.switch %235 : i32, [
        default: ^bb110,
        0: ^bb108,
        1: ^bb109
      ]
    ^bb108:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb109:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb110:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb111(%236: memref<32x64xi8>):  // 3 preds: ^bb108, ^bb109, ^bb110
      func.call @matmul_i8_i16(%236, %232, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %237 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %238 = arith.addi %237, %c1_i32 : i32
      %239 = arith.cmpi sge, %238, %c2_i32 : i32
      %240 = arith.subi %238, %c2_i32 : i32
      %241 = arith.select %239, %240, %238 : i32
      memref.store %241, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %242 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %243 = arith.addi %242, %c1_i32 : i32
      %244 = arith.cmpi sge, %243, %c2_i32 : i32
      %245 = arith.subi %243, %c2_i32 : i32
      %246 = arith.select %244, %245, %243 : i32
      memref.store %246, %_anonymous4[%c1] : memref<3xi32>
      %247 = arith.addi %227, %c1 : index
      cf.br ^bb102(%247 : index)
    ^bb112:  // pred: ^bb102
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %248 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %249 = arith.addi %248, %c1_i32 : i32
      %250 = arith.cmpi sge, %249, %c1_i32 : i32
      %251 = arith.select %250, %248, %249 : i32
      memref.store %251, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb113(%c0 : index)
    ^bb113(%252: index):  // 2 preds: ^bb112, ^bb122
      %253 = arith.cmpi slt, %252, %c16 : index
      cf.cond_br %253, ^bb114, ^bb123
    ^bb114:  // pred: ^bb113
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %254 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %255 = arith.index_cast %254 : i32 to index
      %256 = arith.index_cast %255 : index to i32
      cf.switch %256 : i32, [
        default: ^bb117,
        0: ^bb115,
        1: ^bb116
      ]
    ^bb115:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb116:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb117:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb118(%257: memref<64x128xi8>):  // 3 preds: ^bb115, ^bb116, ^bb117
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %258 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %259 = arith.index_cast %258 : i32 to index
      %260 = arith.index_cast %259 : index to i32
      cf.switch %260 : i32, [
        default: ^bb121,
        0: ^bb119,
        1: ^bb120
      ]
    ^bb119:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb120:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb121:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb122(%261: memref<32x64xi8>):  // 3 preds: ^bb119, ^bb120, ^bb121
      func.call @matmul_i8_i16(%261, %257, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %262 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %263 = arith.addi %262, %c1_i32 : i32
      %264 = arith.cmpi sge, %263, %c2_i32 : i32
      %265 = arith.subi %263, %c2_i32 : i32
      %266 = arith.select %264, %265, %263 : i32
      memref.store %266, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %267 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %268 = arith.addi %267, %c1_i32 : i32
      %269 = arith.cmpi sge, %268, %c2_i32 : i32
      %270 = arith.subi %268, %c2_i32 : i32
      %271 = arith.select %269, %270, %268 : i32
      memref.store %271, %_anonymous4[%c1] : memref<3xi32>
      %272 = arith.addi %252, %c1 : index
      cf.br ^bb113(%272 : index)
    ^bb123:  // pred: ^bb113
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %273 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %274 = arith.addi %273, %c1_i32 : i32
      %275 = arith.cmpi sge, %274, %c1_i32 : i32
      %276 = arith.select %275, %273, %274 : i32
      memref.store %276, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb124(%c0 : index)
    ^bb124(%277: index):  // 2 preds: ^bb123, ^bb133
      %278 = arith.cmpi slt, %277, %c16 : index
      cf.cond_br %278, ^bb125, ^bb134
    ^bb125:  // pred: ^bb124
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %279 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %280 = arith.index_cast %279 : i32 to index
      %281 = arith.index_cast %280 : index to i32
      cf.switch %281 : i32, [
        default: ^bb128,
        0: ^bb126,
        1: ^bb127
      ]
    ^bb126:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb127:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb128:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb129(%282: memref<64x128xi8>):  // 3 preds: ^bb126, ^bb127, ^bb128
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %283 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %284 = arith.index_cast %283 : i32 to index
      %285 = arith.index_cast %284 : index to i32
      cf.switch %285 : i32, [
        default: ^bb132,
        0: ^bb130,
        1: ^bb131
      ]
    ^bb130:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb131:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb132:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb133(%286: memref<32x64xi8>):  // 3 preds: ^bb130, ^bb131, ^bb132
      func.call @matmul_i8_i16(%286, %282, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %287 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %288 = arith.addi %287, %c1_i32 : i32
      %289 = arith.cmpi sge, %288, %c2_i32 : i32
      %290 = arith.subi %288, %c2_i32 : i32
      %291 = arith.select %289, %290, %288 : i32
      memref.store %291, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %292 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %293 = arith.addi %292, %c1_i32 : i32
      %294 = arith.cmpi sge, %293, %c2_i32 : i32
      %295 = arith.subi %293, %c2_i32 : i32
      %296 = arith.select %294, %295, %293 : i32
      memref.store %296, %_anonymous4[%c1] : memref<3xi32>
      %297 = arith.addi %277, %c1 : index
      cf.br ^bb124(%297 : index)
    ^bb134:  // pred: ^bb124
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %298 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %299 = arith.addi %298, %c1_i32 : i32
      %300 = arith.cmpi sge, %299, %c1_i32 : i32
      %301 = arith.select %300, %298, %299 : i32
      memref.store %301, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb135(%c0 : index)
    ^bb135(%302: index):  // 2 preds: ^bb134, ^bb144
      %303 = arith.cmpi slt, %302, %c16 : index
      cf.cond_br %303, ^bb136, ^bb145
    ^bb136:  // pred: ^bb135
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %304 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %305 = arith.index_cast %304 : i32 to index
      %306 = arith.index_cast %305 : index to i32
      cf.switch %306 : i32, [
        default: ^bb139,
        0: ^bb137,
        1: ^bb138
      ]
    ^bb137:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb138:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb139:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb140(%307: memref<64x128xi8>):  // 3 preds: ^bb137, ^bb138, ^bb139
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %308 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %309 = arith.index_cast %308 : i32 to index
      %310 = arith.index_cast %309 : index to i32
      cf.switch %310 : i32, [
        default: ^bb143,
        0: ^bb141,
        1: ^bb142
      ]
    ^bb141:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb142:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb143:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb144(%311: memref<32x64xi8>):  // 3 preds: ^bb141, ^bb142, ^bb143
      func.call @matmul_i8_i16(%311, %307, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %312 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %313 = arith.addi %312, %c1_i32 : i32
      %314 = arith.cmpi sge, %313, %c2_i32 : i32
      %315 = arith.subi %313, %c2_i32 : i32
      %316 = arith.select %314, %315, %313 : i32
      memref.store %316, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %317 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %318 = arith.addi %317, %c1_i32 : i32
      %319 = arith.cmpi sge, %318, %c2_i32 : i32
      %320 = arith.subi %318, %c2_i32 : i32
      %321 = arith.select %319, %320, %318 : i32
      memref.store %321, %_anonymous4[%c1] : memref<3xi32>
      %322 = arith.addi %302, %c1 : index
      cf.br ^bb135(%322 : index)
    ^bb145:  // pred: ^bb135
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %323 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %324 = arith.addi %323, %c1_i32 : i32
      %325 = arith.cmpi sge, %324, %c1_i32 : i32
      %326 = arith.select %325, %323, %324 : i32
      memref.store %326, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb146(%c0 : index)
    ^bb146(%327: index):  // 2 preds: ^bb145, ^bb155
      %328 = arith.cmpi slt, %327, %c16 : index
      cf.cond_br %328, ^bb147, ^bb156
    ^bb147:  // pred: ^bb146
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %329 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %330 = arith.index_cast %329 : i32 to index
      %331 = arith.index_cast %330 : index to i32
      cf.switch %331 : i32, [
        default: ^bb150,
        0: ^bb148,
        1: ^bb149
      ]
    ^bb148:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb149:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb150:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb151(%332: memref<64x128xi8>):  // 3 preds: ^bb148, ^bb149, ^bb150
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %333 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %334 = arith.index_cast %333 : i32 to index
      %335 = arith.index_cast %334 : index to i32
      cf.switch %335 : i32, [
        default: ^bb154,
        0: ^bb152,
        1: ^bb153
      ]
    ^bb152:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb153:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb154:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb155(%336: memref<32x64xi8>):  // 3 preds: ^bb152, ^bb153, ^bb154
      func.call @matmul_i8_i16(%336, %332, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %337 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %338 = arith.addi %337, %c1_i32 : i32
      %339 = arith.cmpi sge, %338, %c2_i32 : i32
      %340 = arith.subi %338, %c2_i32 : i32
      %341 = arith.select %339, %340, %338 : i32
      memref.store %341, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %342 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %343 = arith.addi %342, %c1_i32 : i32
      %344 = arith.cmpi sge, %343, %c2_i32 : i32
      %345 = arith.subi %343, %c2_i32 : i32
      %346 = arith.select %344, %345, %343 : i32
      memref.store %346, %_anonymous4[%c1] : memref<3xi32>
      %347 = arith.addi %327, %c1 : index
      cf.br ^bb146(%347 : index)
    ^bb156:  // pred: ^bb146
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %348 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %349 = arith.addi %348, %c1_i32 : i32
      %350 = arith.cmpi sge, %349, %c1_i32 : i32
      %351 = arith.select %350, %348, %349 : i32
      memref.store %351, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb157(%c0 : index)
    ^bb157(%352: index):  // 2 preds: ^bb156, ^bb166
      %353 = arith.cmpi slt, %352, %c16 : index
      cf.cond_br %353, ^bb158, ^bb167
    ^bb158:  // pred: ^bb157
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %354 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %355 = arith.index_cast %354 : i32 to index
      %356 = arith.index_cast %355 : index to i32
      cf.switch %356 : i32, [
        default: ^bb161,
        0: ^bb159,
        1: ^bb160
      ]
    ^bb159:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb160:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb161:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb162(%357: memref<64x128xi8>):  // 3 preds: ^bb159, ^bb160, ^bb161
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %358 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %359 = arith.index_cast %358 : i32 to index
      %360 = arith.index_cast %359 : index to i32
      cf.switch %360 : i32, [
        default: ^bb165,
        0: ^bb163,
        1: ^bb164
      ]
    ^bb163:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb164:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb165:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb166(%361: memref<32x64xi8>):  // 3 preds: ^bb163, ^bb164, ^bb165
      func.call @matmul_i8_i16(%361, %357, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %362 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %363 = arith.addi %362, %c1_i32 : i32
      %364 = arith.cmpi sge, %363, %c2_i32 : i32
      %365 = arith.subi %363, %c2_i32 : i32
      %366 = arith.select %364, %365, %363 : i32
      memref.store %366, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %367 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %368 = arith.addi %367, %c1_i32 : i32
      %369 = arith.cmpi sge, %368, %c2_i32 : i32
      %370 = arith.subi %368, %c2_i32 : i32
      %371 = arith.select %369, %370, %368 : i32
      memref.store %371, %_anonymous4[%c1] : memref<3xi32>
      %372 = arith.addi %352, %c1 : index
      cf.br ^bb157(%372 : index)
    ^bb167:  // pred: ^bb157
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %373 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %374 = arith.addi %373, %c1_i32 : i32
      %375 = arith.cmpi sge, %374, %c1_i32 : i32
      %376 = arith.select %375, %373, %374 : i32
      memref.store %376, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb168(%c0 : index)
    ^bb168(%377: index):  // 2 preds: ^bb167, ^bb177
      %378 = arith.cmpi slt, %377, %c16 : index
      cf.cond_br %378, ^bb169, ^bb178
    ^bb169:  // pred: ^bb168
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %379 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %380 = arith.index_cast %379 : i32 to index
      %381 = arith.index_cast %380 : index to i32
      cf.switch %381 : i32, [
        default: ^bb172,
        0: ^bb170,
        1: ^bb171
      ]
    ^bb170:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb171:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb172:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb173(%382: memref<64x128xi8>):  // 3 preds: ^bb170, ^bb171, ^bb172
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %383 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %384 = arith.index_cast %383 : i32 to index
      %385 = arith.index_cast %384 : index to i32
      cf.switch %385 : i32, [
        default: ^bb176,
        0: ^bb174,
        1: ^bb175
      ]
    ^bb174:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb175:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb176:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb177(%386: memref<32x64xi8>):  // 3 preds: ^bb174, ^bb175, ^bb176
      func.call @matmul_i8_i16(%386, %382, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %387 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %388 = arith.addi %387, %c1_i32 : i32
      %389 = arith.cmpi sge, %388, %c2_i32 : i32
      %390 = arith.subi %388, %c2_i32 : i32
      %391 = arith.select %389, %390, %388 : i32
      memref.store %391, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %392 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %393 = arith.addi %392, %c1_i32 : i32
      %394 = arith.cmpi sge, %393, %c2_i32 : i32
      %395 = arith.subi %393, %c2_i32 : i32
      %396 = arith.select %394, %395, %393 : i32
      memref.store %396, %_anonymous4[%c1] : memref<3xi32>
      %397 = arith.addi %377, %c1 : index
      cf.br ^bb168(%397 : index)
    ^bb178:  // pred: ^bb168
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %398 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %399 = arith.addi %398, %c1_i32 : i32
      %400 = arith.cmpi sge, %399, %c1_i32 : i32
      %401 = arith.select %400, %398, %399 : i32
      memref.store %401, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb179(%c0 : index)
    ^bb179(%402: index):  // 2 preds: ^bb178, ^bb188
      %403 = arith.cmpi slt, %402, %c16 : index
      cf.cond_br %403, ^bb180, ^bb189
    ^bb180:  // pred: ^bb179
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %404 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %405 = arith.index_cast %404 : i32 to index
      %406 = arith.index_cast %405 : index to i32
      cf.switch %406 : i32, [
        default: ^bb183,
        0: ^bb181,
        1: ^bb182
      ]
    ^bb181:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb182:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb183:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb184(%407: memref<64x128xi8>):  // 3 preds: ^bb181, ^bb182, ^bb183
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %408 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %409 = arith.index_cast %408 : i32 to index
      %410 = arith.index_cast %409 : index to i32
      cf.switch %410 : i32, [
        default: ^bb187,
        0: ^bb185,
        1: ^bb186
      ]
    ^bb185:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb186:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb187:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb188(%411: memref<32x64xi8>):  // 3 preds: ^bb185, ^bb186, ^bb187
      func.call @matmul_i8_i16(%411, %407, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %412 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %413 = arith.addi %412, %c1_i32 : i32
      %414 = arith.cmpi sge, %413, %c2_i32 : i32
      %415 = arith.subi %413, %c2_i32 : i32
      %416 = arith.select %414, %415, %413 : i32
      memref.store %416, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %417 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %418 = arith.addi %417, %c1_i32 : i32
      %419 = arith.cmpi sge, %418, %c2_i32 : i32
      %420 = arith.subi %418, %c2_i32 : i32
      %421 = arith.select %419, %420, %418 : i32
      memref.store %421, %_anonymous4[%c1] : memref<3xi32>
      %422 = arith.addi %402, %c1 : index
      cf.br ^bb179(%422 : index)
    ^bb189:  // pred: ^bb179
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %423 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %424 = arith.addi %423, %c1_i32 : i32
      %425 = arith.cmpi sge, %424, %c1_i32 : i32
      %426 = arith.select %425, %423, %424 : i32
      memref.store %426, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb190(%c0 : index)
    ^bb190(%427: index):  // 2 preds: ^bb189, ^bb199
      %428 = arith.cmpi slt, %427, %c16 : index
      cf.cond_br %428, ^bb191, ^bb200
    ^bb191:  // pred: ^bb190
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %429 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %430 = arith.index_cast %429 : i32 to index
      %431 = arith.index_cast %430 : index to i32
      cf.switch %431 : i32, [
        default: ^bb194,
        0: ^bb192,
        1: ^bb193
      ]
    ^bb192:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb193:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb194:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb195(%432: memref<64x128xi8>):  // 3 preds: ^bb192, ^bb193, ^bb194
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %433 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %434 = arith.index_cast %433 : i32 to index
      %435 = arith.index_cast %434 : index to i32
      cf.switch %435 : i32, [
        default: ^bb198,
        0: ^bb196,
        1: ^bb197
      ]
    ^bb196:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb197:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb198:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb199(%436: memref<32x64xi8>):  // 3 preds: ^bb196, ^bb197, ^bb198
      func.call @matmul_i8_i16(%436, %432, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %437 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %438 = arith.addi %437, %c1_i32 : i32
      %439 = arith.cmpi sge, %438, %c2_i32 : i32
      %440 = arith.subi %438, %c2_i32 : i32
      %441 = arith.select %439, %440, %438 : i32
      memref.store %441, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %442 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %443 = arith.addi %442, %c1_i32 : i32
      %444 = arith.cmpi sge, %443, %c2_i32 : i32
      %445 = arith.subi %443, %c2_i32 : i32
      %446 = arith.select %444, %445, %443 : i32
      memref.store %446, %_anonymous4[%c1] : memref<3xi32>
      %447 = arith.addi %427, %c1 : index
      cf.br ^bb190(%447 : index)
    ^bb200:  // pred: ^bb190
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %448 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %449 = arith.addi %448, %c1_i32 : i32
      %450 = arith.cmpi sge, %449, %c1_i32 : i32
      %451 = arith.select %450, %448, %449 : i32
      memref.store %451, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb201(%c0 : index)
    ^bb201(%452: index):  // 2 preds: ^bb200, ^bb210
      %453 = arith.cmpi slt, %452, %c16 : index
      cf.cond_br %453, ^bb202, ^bb211
    ^bb202:  // pred: ^bb201
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %454 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %455 = arith.index_cast %454 : i32 to index
      %456 = arith.index_cast %455 : index to i32
      cf.switch %456 : i32, [
        default: ^bb205,
        0: ^bb203,
        1: ^bb204
      ]
    ^bb203:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb204:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb205:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb206(%457: memref<64x128xi8>):  // 3 preds: ^bb203, ^bb204, ^bb205
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %458 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %459 = arith.index_cast %458 : i32 to index
      %460 = arith.index_cast %459 : index to i32
      cf.switch %460 : i32, [
        default: ^bb209,
        0: ^bb207,
        1: ^bb208
      ]
    ^bb207:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb208:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb209:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb210(%461: memref<32x64xi8>):  // 3 preds: ^bb207, ^bb208, ^bb209
      func.call @matmul_i8_i16(%461, %457, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %462 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %463 = arith.addi %462, %c1_i32 : i32
      %464 = arith.cmpi sge, %463, %c2_i32 : i32
      %465 = arith.subi %463, %c2_i32 : i32
      %466 = arith.select %464, %465, %463 : i32
      memref.store %466, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %467 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %468 = arith.addi %467, %c1_i32 : i32
      %469 = arith.cmpi sge, %468, %c2_i32 : i32
      %470 = arith.subi %468, %c2_i32 : i32
      %471 = arith.select %469, %470, %468 : i32
      memref.store %471, %_anonymous4[%c1] : memref<3xi32>
      %472 = arith.addi %452, %c1 : index
      cf.br ^bb201(%472 : index)
    ^bb211:  // pred: ^bb201
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %473 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %474 = arith.addi %473, %c1_i32 : i32
      %475 = arith.cmpi sge, %474, %c1_i32 : i32
      %476 = arith.select %475, %473, %474 : i32
      memref.store %476, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb212(%c0 : index)
    ^bb212(%477: index):  // 2 preds: ^bb211, ^bb221
      %478 = arith.cmpi slt, %477, %c16 : index
      cf.cond_br %478, ^bb213, ^bb222
    ^bb213:  // pred: ^bb212
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %479 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %480 = arith.index_cast %479 : i32 to index
      %481 = arith.index_cast %480 : index to i32
      cf.switch %481 : i32, [
        default: ^bb216,
        0: ^bb214,
        1: ^bb215
      ]
    ^bb214:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb215:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb216:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb217(%482: memref<64x128xi8>):  // 3 preds: ^bb214, ^bb215, ^bb216
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %483 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %484 = arith.index_cast %483 : i32 to index
      %485 = arith.index_cast %484 : index to i32
      cf.switch %485 : i32, [
        default: ^bb220,
        0: ^bb218,
        1: ^bb219
      ]
    ^bb218:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb219:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb220:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb221(%486: memref<32x64xi8>):  // 3 preds: ^bb218, ^bb219, ^bb220
      func.call @matmul_i8_i16(%486, %482, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %487 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %488 = arith.addi %487, %c1_i32 : i32
      %489 = arith.cmpi sge, %488, %c2_i32 : i32
      %490 = arith.subi %488, %c2_i32 : i32
      %491 = arith.select %489, %490, %488 : i32
      memref.store %491, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %492 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %493 = arith.addi %492, %c1_i32 : i32
      %494 = arith.cmpi sge, %493, %c2_i32 : i32
      %495 = arith.subi %493, %c2_i32 : i32
      %496 = arith.select %494, %495, %493 : i32
      memref.store %496, %_anonymous4[%c1] : memref<3xi32>
      %497 = arith.addi %477, %c1 : index
      cf.br ^bb212(%497 : index)
    ^bb222:  // pred: ^bb212
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %498 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %499 = arith.addi %498, %c1_i32 : i32
      %500 = arith.cmpi sge, %499, %c1_i32 : i32
      %501 = arith.select %500, %498, %499 : i32
      memref.store %501, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb223(%c0 : index)
    ^bb223(%502: index):  // 2 preds: ^bb222, ^bb232
      %503 = arith.cmpi slt, %502, %c16 : index
      cf.cond_br %503, ^bb224, ^bb233
    ^bb224:  // pred: ^bb223
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %504 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %505 = arith.index_cast %504 : i32 to index
      %506 = arith.index_cast %505 : index to i32
      cf.switch %506 : i32, [
        default: ^bb227,
        0: ^bb225,
        1: ^bb226
      ]
    ^bb225:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb226:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb227:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb228(%507: memref<64x128xi8>):  // 3 preds: ^bb225, ^bb226, ^bb227
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %508 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %509 = arith.index_cast %508 : i32 to index
      %510 = arith.index_cast %509 : index to i32
      cf.switch %510 : i32, [
        default: ^bb231,
        0: ^bb229,
        1: ^bb230
      ]
    ^bb229:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb230:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb231:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb232(%511: memref<32x64xi8>):  // 3 preds: ^bb229, ^bb230, ^bb231
      func.call @matmul_i8_i16(%511, %507, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %512 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %513 = arith.addi %512, %c1_i32 : i32
      %514 = arith.cmpi sge, %513, %c2_i32 : i32
      %515 = arith.subi %513, %c2_i32 : i32
      %516 = arith.select %514, %515, %513 : i32
      memref.store %516, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %517 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %518 = arith.addi %517, %c1_i32 : i32
      %519 = arith.cmpi sge, %518, %c2_i32 : i32
      %520 = arith.subi %518, %c2_i32 : i32
      %521 = arith.select %519, %520, %518 : i32
      memref.store %521, %_anonymous4[%c1] : memref<3xi32>
      %522 = arith.addi %502, %c1 : index
      cf.br ^bb223(%522 : index)
    ^bb233:  // pred: ^bb223
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %523 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %524 = arith.addi %523, %c1_i32 : i32
      %525 = arith.cmpi sge, %524, %c1_i32 : i32
      %526 = arith.select %525, %523, %524 : i32
      memref.store %526, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb234(%c0 : index)
    ^bb234(%527: index):  // 2 preds: ^bb233, ^bb243
      %528 = arith.cmpi slt, %527, %c16 : index
      cf.cond_br %528, ^bb235, ^bb244
    ^bb235:  // pred: ^bb234
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %529 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %530 = arith.index_cast %529 : i32 to index
      %531 = arith.index_cast %530 : index to i32
      cf.switch %531 : i32, [
        default: ^bb238,
        0: ^bb236,
        1: ^bb237
      ]
    ^bb236:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb237:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb238:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb239(%532: memref<64x128xi8>):  // 3 preds: ^bb236, ^bb237, ^bb238
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %533 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %534 = arith.index_cast %533 : i32 to index
      %535 = arith.index_cast %534 : index to i32
      cf.switch %535 : i32, [
        default: ^bb242,
        0: ^bb240,
        1: ^bb241
      ]
    ^bb240:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb241:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb242:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb243(%536: memref<32x64xi8>):  // 3 preds: ^bb240, ^bb241, ^bb242
      func.call @matmul_i8_i16(%536, %532, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %537 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %538 = arith.addi %537, %c1_i32 : i32
      %539 = arith.cmpi sge, %538, %c2_i32 : i32
      %540 = arith.subi %538, %c2_i32 : i32
      %541 = arith.select %539, %540, %538 : i32
      memref.store %541, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %542 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %543 = arith.addi %542, %c1_i32 : i32
      %544 = arith.cmpi sge, %543, %c2_i32 : i32
      %545 = arith.subi %543, %c2_i32 : i32
      %546 = arith.select %544, %545, %543 : i32
      memref.store %546, %_anonymous4[%c1] : memref<3xi32>
      %547 = arith.addi %527, %c1 : index
      cf.br ^bb234(%547 : index)
    ^bb244:  // pred: ^bb234
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %548 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %549 = arith.addi %548, %c1_i32 : i32
      %550 = arith.cmpi sge, %549, %c1_i32 : i32
      %551 = arith.select %550, %548, %549 : i32
      memref.store %551, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb245(%c0 : index)
    ^bb245(%552: index):  // 2 preds: ^bb244, ^bb254
      %553 = arith.cmpi slt, %552, %c16 : index
      cf.cond_br %553, ^bb246, ^bb255
    ^bb246:  // pred: ^bb245
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %554 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %555 = arith.index_cast %554 : i32 to index
      %556 = arith.index_cast %555 : index to i32
      cf.switch %556 : i32, [
        default: ^bb249,
        0: ^bb247,
        1: ^bb248
      ]
    ^bb247:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb248:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb249:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb250(%557: memref<64x128xi8>):  // 3 preds: ^bb247, ^bb248, ^bb249
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %558 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %559 = arith.index_cast %558 : i32 to index
      %560 = arith.index_cast %559 : index to i32
      cf.switch %560 : i32, [
        default: ^bb253,
        0: ^bb251,
        1: ^bb252
      ]
    ^bb251:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb252:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb253:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb254(%561: memref<32x64xi8>):  // 3 preds: ^bb251, ^bb252, ^bb253
      func.call @matmul_i8_i16(%561, %557, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %562 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %563 = arith.addi %562, %c1_i32 : i32
      %564 = arith.cmpi sge, %563, %c2_i32 : i32
      %565 = arith.subi %563, %c2_i32 : i32
      %566 = arith.select %564, %565, %563 : i32
      memref.store %566, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %567 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %568 = arith.addi %567, %c1_i32 : i32
      %569 = arith.cmpi sge, %568, %c2_i32 : i32
      %570 = arith.subi %568, %c2_i32 : i32
      %571 = arith.select %569, %570, %568 : i32
      memref.store %571, %_anonymous4[%c1] : memref<3xi32>
      %572 = arith.addi %552, %c1 : index
      cf.br ^bb245(%572 : index)
    ^bb255:  // pred: ^bb245
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %573 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %574 = arith.addi %573, %c1_i32 : i32
      %575 = arith.cmpi sge, %574, %c1_i32 : i32
      %576 = arith.select %575, %573, %574 : i32
      memref.store %576, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb256(%c0 : index)
    ^bb256(%577: index):  // 2 preds: ^bb255, ^bb265
      %578 = arith.cmpi slt, %577, %c16 : index
      cf.cond_br %578, ^bb257, ^bb266
    ^bb257:  // pred: ^bb256
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %579 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %580 = arith.index_cast %579 : i32 to index
      %581 = arith.index_cast %580 : index to i32
      cf.switch %581 : i32, [
        default: ^bb260,
        0: ^bb258,
        1: ^bb259
      ]
    ^bb258:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb259:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb260:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb261(%582: memref<64x128xi8>):  // 3 preds: ^bb258, ^bb259, ^bb260
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %583 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %584 = arith.index_cast %583 : i32 to index
      %585 = arith.index_cast %584 : index to i32
      cf.switch %585 : i32, [
        default: ^bb264,
        0: ^bb262,
        1: ^bb263
      ]
    ^bb262:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb263:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb264:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb265(%586: memref<32x64xi8>):  // 3 preds: ^bb262, ^bb263, ^bb264
      func.call @matmul_i8_i16(%586, %582, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %587 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %588 = arith.addi %587, %c1_i32 : i32
      %589 = arith.cmpi sge, %588, %c2_i32 : i32
      %590 = arith.subi %588, %c2_i32 : i32
      %591 = arith.select %589, %590, %588 : i32
      memref.store %591, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %592 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %593 = arith.addi %592, %c1_i32 : i32
      %594 = arith.cmpi sge, %593, %c2_i32 : i32
      %595 = arith.subi %593, %c2_i32 : i32
      %596 = arith.select %594, %595, %593 : i32
      memref.store %596, %_anonymous4[%c1] : memref<3xi32>
      %597 = arith.addi %577, %c1 : index
      cf.br ^bb256(%597 : index)
    ^bb266:  // pred: ^bb256
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %598 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %599 = arith.addi %598, %c1_i32 : i32
      %600 = arith.cmpi sge, %599, %c1_i32 : i32
      %601 = arith.select %600, %598, %599 : i32
      memref.store %601, %_anonymous4[%c0] : memref<3xi32>
      %602 = arith.addi %0, %c1 : index
      cf.br ^bb1(%602 : index)
    ^bb267:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous5 = aie.buffer(%tile_5_2) {address = 32000 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb266
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb267
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
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
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous5[%c1] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb19(%32: memref<64x128xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb23(%36: memref<32x64xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_i8_i16(%36, %32, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous5[%c1] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb30(%57: memref<64x128xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb34(%61: memref<32x64xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_i8_i16(%61, %57, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous5[%c1] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb41(%82: memref<64x128xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb45(%86: memref<32x64xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_i8_i16(%86, %82, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous5[%c1] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb47(%c0 : index)
    ^bb47(%102: index):  // 2 preds: ^bb46, ^bb56
      %103 = arith.cmpi slt, %102, %c16 : index
      cf.cond_br %103, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %104 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %105 = arith.index_cast %104 : i32 to index
      %106 = arith.index_cast %105 : index to i32
      cf.switch %106 : i32, [
        default: ^bb51,
        0: ^bb49,
        1: ^bb50
      ]
    ^bb49:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb50:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb51:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb52(%107: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb55,
        0: ^bb53,
        1: ^bb54
      ]
    ^bb53:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb54:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb55:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb56(%111: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_i8_i16(%111, %107, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %112 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %113 = arith.addi %112, %c1_i32 : i32
      %114 = arith.cmpi sge, %113, %c2_i32 : i32
      %115 = arith.subi %113, %c2_i32 : i32
      %116 = arith.select %114, %115, %113 : i32
      memref.store %116, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %117 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %118 = arith.addi %117, %c1_i32 : i32
      %119 = arith.cmpi sge, %118, %c2_i32 : i32
      %120 = arith.subi %118, %c2_i32 : i32
      %121 = arith.select %119, %120, %118 : i32
      memref.store %121, %_anonymous5[%c1] : memref<3xi32>
      %122 = arith.addi %102, %c1 : index
      cf.br ^bb47(%122 : index)
    ^bb57:  // pred: ^bb47
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %123 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c1_i32 : i32
      %126 = arith.select %125, %123, %124 : i32
      memref.store %126, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb58(%c0 : index)
    ^bb58(%127: index):  // 2 preds: ^bb57, ^bb67
      %128 = arith.cmpi slt, %127, %c16 : index
      cf.cond_br %128, ^bb59, ^bb68
    ^bb59:  // pred: ^bb58
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %129 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %130 = arith.index_cast %129 : i32 to index
      %131 = arith.index_cast %130 : index to i32
      cf.switch %131 : i32, [
        default: ^bb62,
        0: ^bb60,
        1: ^bb61
      ]
    ^bb60:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb61:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb62:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb63(%132: memref<64x128xi8>):  // 3 preds: ^bb60, ^bb61, ^bb62
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %133 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %134 = arith.index_cast %133 : i32 to index
      %135 = arith.index_cast %134 : index to i32
      cf.switch %135 : i32, [
        default: ^bb66,
        0: ^bb64,
        1: ^bb65
      ]
    ^bb64:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb66:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb67(%136: memref<32x64xi8>):  // 3 preds: ^bb64, ^bb65, ^bb66
      func.call @matmul_i8_i16(%136, %132, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %137 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %138 = arith.addi %137, %c1_i32 : i32
      %139 = arith.cmpi sge, %138, %c2_i32 : i32
      %140 = arith.subi %138, %c2_i32 : i32
      %141 = arith.select %139, %140, %138 : i32
      memref.store %141, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %142 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %143 = arith.addi %142, %c1_i32 : i32
      %144 = arith.cmpi sge, %143, %c2_i32 : i32
      %145 = arith.subi %143, %c2_i32 : i32
      %146 = arith.select %144, %145, %143 : i32
      memref.store %146, %_anonymous5[%c1] : memref<3xi32>
      %147 = arith.addi %127, %c1 : index
      cf.br ^bb58(%147 : index)
    ^bb68:  // pred: ^bb58
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %148 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c1_i32 : i32
      %151 = arith.select %150, %148, %149 : i32
      memref.store %151, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb69(%c0 : index)
    ^bb69(%152: index):  // 2 preds: ^bb68, ^bb78
      %153 = arith.cmpi slt, %152, %c16 : index
      cf.cond_br %153, ^bb70, ^bb79
    ^bb70:  // pred: ^bb69
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %154 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %155 = arith.index_cast %154 : i32 to index
      %156 = arith.index_cast %155 : index to i32
      cf.switch %156 : i32, [
        default: ^bb73,
        0: ^bb71,
        1: ^bb72
      ]
    ^bb71:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb72:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb73:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb74(%157: memref<64x128xi8>):  // 3 preds: ^bb71, ^bb72, ^bb73
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %158 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %159 = arith.index_cast %158 : i32 to index
      %160 = arith.index_cast %159 : index to i32
      cf.switch %160 : i32, [
        default: ^bb77,
        0: ^bb75,
        1: ^bb76
      ]
    ^bb75:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb76:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb77:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb78(%161: memref<32x64xi8>):  // 3 preds: ^bb75, ^bb76, ^bb77
      func.call @matmul_i8_i16(%161, %157, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %162 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %163 = arith.addi %162, %c1_i32 : i32
      %164 = arith.cmpi sge, %163, %c2_i32 : i32
      %165 = arith.subi %163, %c2_i32 : i32
      %166 = arith.select %164, %165, %163 : i32
      memref.store %166, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %167 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %168 = arith.addi %167, %c1_i32 : i32
      %169 = arith.cmpi sge, %168, %c2_i32 : i32
      %170 = arith.subi %168, %c2_i32 : i32
      %171 = arith.select %169, %170, %168 : i32
      memref.store %171, %_anonymous5[%c1] : memref<3xi32>
      %172 = arith.addi %152, %c1 : index
      cf.br ^bb69(%172 : index)
    ^bb79:  // pred: ^bb69
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %173 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %174 = arith.addi %173, %c1_i32 : i32
      %175 = arith.cmpi sge, %174, %c1_i32 : i32
      %176 = arith.select %175, %173, %174 : i32
      memref.store %176, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb80(%c0 : index)
    ^bb80(%177: index):  // 2 preds: ^bb79, ^bb89
      %178 = arith.cmpi slt, %177, %c16 : index
      cf.cond_br %178, ^bb81, ^bb90
    ^bb81:  // pred: ^bb80
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %179 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %180 = arith.index_cast %179 : i32 to index
      %181 = arith.index_cast %180 : index to i32
      cf.switch %181 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb85(%182: memref<64x128xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %183 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %184 = arith.index_cast %183 : i32 to index
      %185 = arith.index_cast %184 : index to i32
      cf.switch %185 : i32, [
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
    ^bb89(%186: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%186, %182, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %187 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %188 = arith.addi %187, %c1_i32 : i32
      %189 = arith.cmpi sge, %188, %c2_i32 : i32
      %190 = arith.subi %188, %c2_i32 : i32
      %191 = arith.select %189, %190, %188 : i32
      memref.store %191, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %192 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %193 = arith.addi %192, %c1_i32 : i32
      %194 = arith.cmpi sge, %193, %c2_i32 : i32
      %195 = arith.subi %193, %c2_i32 : i32
      %196 = arith.select %194, %195, %193 : i32
      memref.store %196, %_anonymous5[%c1] : memref<3xi32>
      %197 = arith.addi %177, %c1 : index
      cf.br ^bb80(%197 : index)
    ^bb90:  // pred: ^bb80
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %198 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %199 = arith.addi %198, %c1_i32 : i32
      %200 = arith.cmpi sge, %199, %c1_i32 : i32
      %201 = arith.select %200, %198, %199 : i32
      memref.store %201, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb91(%c0 : index)
    ^bb91(%202: index):  // 2 preds: ^bb90, ^bb100
      %203 = arith.cmpi slt, %202, %c16 : index
      cf.cond_br %203, ^bb92, ^bb101
    ^bb92:  // pred: ^bb91
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %204 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %205 = arith.index_cast %204 : i32 to index
      %206 = arith.index_cast %205 : index to i32
      cf.switch %206 : i32, [
        default: ^bb95,
        0: ^bb93,
        1: ^bb94
      ]
    ^bb93:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb94:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb95:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb96(%207: memref<64x128xi8>):  // 3 preds: ^bb93, ^bb94, ^bb95
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %208 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %209 = arith.index_cast %208 : i32 to index
      %210 = arith.index_cast %209 : index to i32
      cf.switch %210 : i32, [
        default: ^bb99,
        0: ^bb97,
        1: ^bb98
      ]
    ^bb97:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb98:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb99:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb100(%211: memref<32x64xi8>):  // 3 preds: ^bb97, ^bb98, ^bb99
      func.call @matmul_i8_i16(%211, %207, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %212 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %213 = arith.addi %212, %c1_i32 : i32
      %214 = arith.cmpi sge, %213, %c2_i32 : i32
      %215 = arith.subi %213, %c2_i32 : i32
      %216 = arith.select %214, %215, %213 : i32
      memref.store %216, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %217 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %218 = arith.addi %217, %c1_i32 : i32
      %219 = arith.cmpi sge, %218, %c2_i32 : i32
      %220 = arith.subi %218, %c2_i32 : i32
      %221 = arith.select %219, %220, %218 : i32
      memref.store %221, %_anonymous5[%c1] : memref<3xi32>
      %222 = arith.addi %202, %c1 : index
      cf.br ^bb91(%222 : index)
    ^bb101:  // pred: ^bb91
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %223 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %224 = arith.addi %223, %c1_i32 : i32
      %225 = arith.cmpi sge, %224, %c1_i32 : i32
      %226 = arith.select %225, %223, %224 : i32
      memref.store %226, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb102(%c0 : index)
    ^bb102(%227: index):  // 2 preds: ^bb101, ^bb111
      %228 = arith.cmpi slt, %227, %c16 : index
      cf.cond_br %228, ^bb103, ^bb112
    ^bb103:  // pred: ^bb102
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %229 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %230 = arith.index_cast %229 : i32 to index
      %231 = arith.index_cast %230 : index to i32
      cf.switch %231 : i32, [
        default: ^bb106,
        0: ^bb104,
        1: ^bb105
      ]
    ^bb104:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb105:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb106:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb107(%232: memref<64x128xi8>):  // 3 preds: ^bb104, ^bb105, ^bb106
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %233 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %234 = arith.index_cast %233 : i32 to index
      %235 = arith.index_cast %234 : index to i32
      cf.switch %235 : i32, [
        default: ^bb110,
        0: ^bb108,
        1: ^bb109
      ]
    ^bb108:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb109:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb110:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb111(%236: memref<32x64xi8>):  // 3 preds: ^bb108, ^bb109, ^bb110
      func.call @matmul_i8_i16(%236, %232, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %237 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %238 = arith.addi %237, %c1_i32 : i32
      %239 = arith.cmpi sge, %238, %c2_i32 : i32
      %240 = arith.subi %238, %c2_i32 : i32
      %241 = arith.select %239, %240, %238 : i32
      memref.store %241, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %242 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %243 = arith.addi %242, %c1_i32 : i32
      %244 = arith.cmpi sge, %243, %c2_i32 : i32
      %245 = arith.subi %243, %c2_i32 : i32
      %246 = arith.select %244, %245, %243 : i32
      memref.store %246, %_anonymous5[%c1] : memref<3xi32>
      %247 = arith.addi %227, %c1 : index
      cf.br ^bb102(%247 : index)
    ^bb112:  // pred: ^bb102
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %248 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %249 = arith.addi %248, %c1_i32 : i32
      %250 = arith.cmpi sge, %249, %c1_i32 : i32
      %251 = arith.select %250, %248, %249 : i32
      memref.store %251, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb113(%c0 : index)
    ^bb113(%252: index):  // 2 preds: ^bb112, ^bb122
      %253 = arith.cmpi slt, %252, %c16 : index
      cf.cond_br %253, ^bb114, ^bb123
    ^bb114:  // pred: ^bb113
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %254 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %255 = arith.index_cast %254 : i32 to index
      %256 = arith.index_cast %255 : index to i32
      cf.switch %256 : i32, [
        default: ^bb117,
        0: ^bb115,
        1: ^bb116
      ]
    ^bb115:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb116:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb117:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb118(%257: memref<64x128xi8>):  // 3 preds: ^bb115, ^bb116, ^bb117
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %258 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %259 = arith.index_cast %258 : i32 to index
      %260 = arith.index_cast %259 : index to i32
      cf.switch %260 : i32, [
        default: ^bb121,
        0: ^bb119,
        1: ^bb120
      ]
    ^bb119:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb120:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb121:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb122(%261: memref<32x64xi8>):  // 3 preds: ^bb119, ^bb120, ^bb121
      func.call @matmul_i8_i16(%261, %257, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %262 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %263 = arith.addi %262, %c1_i32 : i32
      %264 = arith.cmpi sge, %263, %c2_i32 : i32
      %265 = arith.subi %263, %c2_i32 : i32
      %266 = arith.select %264, %265, %263 : i32
      memref.store %266, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %267 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %268 = arith.addi %267, %c1_i32 : i32
      %269 = arith.cmpi sge, %268, %c2_i32 : i32
      %270 = arith.subi %268, %c2_i32 : i32
      %271 = arith.select %269, %270, %268 : i32
      memref.store %271, %_anonymous5[%c1] : memref<3xi32>
      %272 = arith.addi %252, %c1 : index
      cf.br ^bb113(%272 : index)
    ^bb123:  // pred: ^bb113
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %273 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %274 = arith.addi %273, %c1_i32 : i32
      %275 = arith.cmpi sge, %274, %c1_i32 : i32
      %276 = arith.select %275, %273, %274 : i32
      memref.store %276, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb124(%c0 : index)
    ^bb124(%277: index):  // 2 preds: ^bb123, ^bb133
      %278 = arith.cmpi slt, %277, %c16 : index
      cf.cond_br %278, ^bb125, ^bb134
    ^bb125:  // pred: ^bb124
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %279 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %280 = arith.index_cast %279 : i32 to index
      %281 = arith.index_cast %280 : index to i32
      cf.switch %281 : i32, [
        default: ^bb128,
        0: ^bb126,
        1: ^bb127
      ]
    ^bb126:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb127:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb128:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb129(%282: memref<64x128xi8>):  // 3 preds: ^bb126, ^bb127, ^bb128
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %283 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %284 = arith.index_cast %283 : i32 to index
      %285 = arith.index_cast %284 : index to i32
      cf.switch %285 : i32, [
        default: ^bb132,
        0: ^bb130,
        1: ^bb131
      ]
    ^bb130:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb131:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb132:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb133(%286: memref<32x64xi8>):  // 3 preds: ^bb130, ^bb131, ^bb132
      func.call @matmul_i8_i16(%286, %282, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %287 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %288 = arith.addi %287, %c1_i32 : i32
      %289 = arith.cmpi sge, %288, %c2_i32 : i32
      %290 = arith.subi %288, %c2_i32 : i32
      %291 = arith.select %289, %290, %288 : i32
      memref.store %291, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %292 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %293 = arith.addi %292, %c1_i32 : i32
      %294 = arith.cmpi sge, %293, %c2_i32 : i32
      %295 = arith.subi %293, %c2_i32 : i32
      %296 = arith.select %294, %295, %293 : i32
      memref.store %296, %_anonymous5[%c1] : memref<3xi32>
      %297 = arith.addi %277, %c1 : index
      cf.br ^bb124(%297 : index)
    ^bb134:  // pred: ^bb124
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %298 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %299 = arith.addi %298, %c1_i32 : i32
      %300 = arith.cmpi sge, %299, %c1_i32 : i32
      %301 = arith.select %300, %298, %299 : i32
      memref.store %301, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb135(%c0 : index)
    ^bb135(%302: index):  // 2 preds: ^bb134, ^bb144
      %303 = arith.cmpi slt, %302, %c16 : index
      cf.cond_br %303, ^bb136, ^bb145
    ^bb136:  // pred: ^bb135
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %304 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %305 = arith.index_cast %304 : i32 to index
      %306 = arith.index_cast %305 : index to i32
      cf.switch %306 : i32, [
        default: ^bb139,
        0: ^bb137,
        1: ^bb138
      ]
    ^bb137:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb138:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb139:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb140(%307: memref<64x128xi8>):  // 3 preds: ^bb137, ^bb138, ^bb139
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %308 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %309 = arith.index_cast %308 : i32 to index
      %310 = arith.index_cast %309 : index to i32
      cf.switch %310 : i32, [
        default: ^bb143,
        0: ^bb141,
        1: ^bb142
      ]
    ^bb141:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb142:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb143:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb144(%311: memref<32x64xi8>):  // 3 preds: ^bb141, ^bb142, ^bb143
      func.call @matmul_i8_i16(%311, %307, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %312 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %313 = arith.addi %312, %c1_i32 : i32
      %314 = arith.cmpi sge, %313, %c2_i32 : i32
      %315 = arith.subi %313, %c2_i32 : i32
      %316 = arith.select %314, %315, %313 : i32
      memref.store %316, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %317 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %318 = arith.addi %317, %c1_i32 : i32
      %319 = arith.cmpi sge, %318, %c2_i32 : i32
      %320 = arith.subi %318, %c2_i32 : i32
      %321 = arith.select %319, %320, %318 : i32
      memref.store %321, %_anonymous5[%c1] : memref<3xi32>
      %322 = arith.addi %302, %c1 : index
      cf.br ^bb135(%322 : index)
    ^bb145:  // pred: ^bb135
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %323 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %324 = arith.addi %323, %c1_i32 : i32
      %325 = arith.cmpi sge, %324, %c1_i32 : i32
      %326 = arith.select %325, %323, %324 : i32
      memref.store %326, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb146(%c0 : index)
    ^bb146(%327: index):  // 2 preds: ^bb145, ^bb155
      %328 = arith.cmpi slt, %327, %c16 : index
      cf.cond_br %328, ^bb147, ^bb156
    ^bb147:  // pred: ^bb146
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %329 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %330 = arith.index_cast %329 : i32 to index
      %331 = arith.index_cast %330 : index to i32
      cf.switch %331 : i32, [
        default: ^bb150,
        0: ^bb148,
        1: ^bb149
      ]
    ^bb148:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb149:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb150:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb151(%332: memref<64x128xi8>):  // 3 preds: ^bb148, ^bb149, ^bb150
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %333 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %334 = arith.index_cast %333 : i32 to index
      %335 = arith.index_cast %334 : index to i32
      cf.switch %335 : i32, [
        default: ^bb154,
        0: ^bb152,
        1: ^bb153
      ]
    ^bb152:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb153:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb154:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb155(%336: memref<32x64xi8>):  // 3 preds: ^bb152, ^bb153, ^bb154
      func.call @matmul_i8_i16(%336, %332, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %337 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %338 = arith.addi %337, %c1_i32 : i32
      %339 = arith.cmpi sge, %338, %c2_i32 : i32
      %340 = arith.subi %338, %c2_i32 : i32
      %341 = arith.select %339, %340, %338 : i32
      memref.store %341, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %342 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %343 = arith.addi %342, %c1_i32 : i32
      %344 = arith.cmpi sge, %343, %c2_i32 : i32
      %345 = arith.subi %343, %c2_i32 : i32
      %346 = arith.select %344, %345, %343 : i32
      memref.store %346, %_anonymous5[%c1] : memref<3xi32>
      %347 = arith.addi %327, %c1 : index
      cf.br ^bb146(%347 : index)
    ^bb156:  // pred: ^bb146
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %348 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %349 = arith.addi %348, %c1_i32 : i32
      %350 = arith.cmpi sge, %349, %c1_i32 : i32
      %351 = arith.select %350, %348, %349 : i32
      memref.store %351, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb157(%c0 : index)
    ^bb157(%352: index):  // 2 preds: ^bb156, ^bb166
      %353 = arith.cmpi slt, %352, %c16 : index
      cf.cond_br %353, ^bb158, ^bb167
    ^bb158:  // pred: ^bb157
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %354 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %355 = arith.index_cast %354 : i32 to index
      %356 = arith.index_cast %355 : index to i32
      cf.switch %356 : i32, [
        default: ^bb161,
        0: ^bb159,
        1: ^bb160
      ]
    ^bb159:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb160:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb161:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb162(%357: memref<64x128xi8>):  // 3 preds: ^bb159, ^bb160, ^bb161
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %358 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %359 = arith.index_cast %358 : i32 to index
      %360 = arith.index_cast %359 : index to i32
      cf.switch %360 : i32, [
        default: ^bb165,
        0: ^bb163,
        1: ^bb164
      ]
    ^bb163:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb164:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb165:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb166(%361: memref<32x64xi8>):  // 3 preds: ^bb163, ^bb164, ^bb165
      func.call @matmul_i8_i16(%361, %357, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %362 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %363 = arith.addi %362, %c1_i32 : i32
      %364 = arith.cmpi sge, %363, %c2_i32 : i32
      %365 = arith.subi %363, %c2_i32 : i32
      %366 = arith.select %364, %365, %363 : i32
      memref.store %366, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %367 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %368 = arith.addi %367, %c1_i32 : i32
      %369 = arith.cmpi sge, %368, %c2_i32 : i32
      %370 = arith.subi %368, %c2_i32 : i32
      %371 = arith.select %369, %370, %368 : i32
      memref.store %371, %_anonymous5[%c1] : memref<3xi32>
      %372 = arith.addi %352, %c1 : index
      cf.br ^bb157(%372 : index)
    ^bb167:  // pred: ^bb157
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %373 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %374 = arith.addi %373, %c1_i32 : i32
      %375 = arith.cmpi sge, %374, %c1_i32 : i32
      %376 = arith.select %375, %373, %374 : i32
      memref.store %376, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb168(%c0 : index)
    ^bb168(%377: index):  // 2 preds: ^bb167, ^bb177
      %378 = arith.cmpi slt, %377, %c16 : index
      cf.cond_br %378, ^bb169, ^bb178
    ^bb169:  // pred: ^bb168
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %379 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %380 = arith.index_cast %379 : i32 to index
      %381 = arith.index_cast %380 : index to i32
      cf.switch %381 : i32, [
        default: ^bb172,
        0: ^bb170,
        1: ^bb171
      ]
    ^bb170:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb171:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb172:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb173(%382: memref<64x128xi8>):  // 3 preds: ^bb170, ^bb171, ^bb172
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %383 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %384 = arith.index_cast %383 : i32 to index
      %385 = arith.index_cast %384 : index to i32
      cf.switch %385 : i32, [
        default: ^bb176,
        0: ^bb174,
        1: ^bb175
      ]
    ^bb174:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb175:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb176:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb177(%386: memref<32x64xi8>):  // 3 preds: ^bb174, ^bb175, ^bb176
      func.call @matmul_i8_i16(%386, %382, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %387 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %388 = arith.addi %387, %c1_i32 : i32
      %389 = arith.cmpi sge, %388, %c2_i32 : i32
      %390 = arith.subi %388, %c2_i32 : i32
      %391 = arith.select %389, %390, %388 : i32
      memref.store %391, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %392 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %393 = arith.addi %392, %c1_i32 : i32
      %394 = arith.cmpi sge, %393, %c2_i32 : i32
      %395 = arith.subi %393, %c2_i32 : i32
      %396 = arith.select %394, %395, %393 : i32
      memref.store %396, %_anonymous5[%c1] : memref<3xi32>
      %397 = arith.addi %377, %c1 : index
      cf.br ^bb168(%397 : index)
    ^bb178:  // pred: ^bb168
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %398 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %399 = arith.addi %398, %c1_i32 : i32
      %400 = arith.cmpi sge, %399, %c1_i32 : i32
      %401 = arith.select %400, %398, %399 : i32
      memref.store %401, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb179(%c0 : index)
    ^bb179(%402: index):  // 2 preds: ^bb178, ^bb188
      %403 = arith.cmpi slt, %402, %c16 : index
      cf.cond_br %403, ^bb180, ^bb189
    ^bb180:  // pred: ^bb179
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %404 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %405 = arith.index_cast %404 : i32 to index
      %406 = arith.index_cast %405 : index to i32
      cf.switch %406 : i32, [
        default: ^bb183,
        0: ^bb181,
        1: ^bb182
      ]
    ^bb181:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb182:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb183:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb184(%407: memref<64x128xi8>):  // 3 preds: ^bb181, ^bb182, ^bb183
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %408 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %409 = arith.index_cast %408 : i32 to index
      %410 = arith.index_cast %409 : index to i32
      cf.switch %410 : i32, [
        default: ^bb187,
        0: ^bb185,
        1: ^bb186
      ]
    ^bb185:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb186:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb187:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb188(%411: memref<32x64xi8>):  // 3 preds: ^bb185, ^bb186, ^bb187
      func.call @matmul_i8_i16(%411, %407, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %412 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %413 = arith.addi %412, %c1_i32 : i32
      %414 = arith.cmpi sge, %413, %c2_i32 : i32
      %415 = arith.subi %413, %c2_i32 : i32
      %416 = arith.select %414, %415, %413 : i32
      memref.store %416, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %417 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %418 = arith.addi %417, %c1_i32 : i32
      %419 = arith.cmpi sge, %418, %c2_i32 : i32
      %420 = arith.subi %418, %c2_i32 : i32
      %421 = arith.select %419, %420, %418 : i32
      memref.store %421, %_anonymous5[%c1] : memref<3xi32>
      %422 = arith.addi %402, %c1 : index
      cf.br ^bb179(%422 : index)
    ^bb189:  // pred: ^bb179
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %423 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %424 = arith.addi %423, %c1_i32 : i32
      %425 = arith.cmpi sge, %424, %c1_i32 : i32
      %426 = arith.select %425, %423, %424 : i32
      memref.store %426, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb190(%c0 : index)
    ^bb190(%427: index):  // 2 preds: ^bb189, ^bb199
      %428 = arith.cmpi slt, %427, %c16 : index
      cf.cond_br %428, ^bb191, ^bb200
    ^bb191:  // pred: ^bb190
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %429 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %430 = arith.index_cast %429 : i32 to index
      %431 = arith.index_cast %430 : index to i32
      cf.switch %431 : i32, [
        default: ^bb194,
        0: ^bb192,
        1: ^bb193
      ]
    ^bb192:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb193:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb194:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb195(%432: memref<64x128xi8>):  // 3 preds: ^bb192, ^bb193, ^bb194
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %433 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %434 = arith.index_cast %433 : i32 to index
      %435 = arith.index_cast %434 : index to i32
      cf.switch %435 : i32, [
        default: ^bb198,
        0: ^bb196,
        1: ^bb197
      ]
    ^bb196:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb197:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb198:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb199(%436: memref<32x64xi8>):  // 3 preds: ^bb196, ^bb197, ^bb198
      func.call @matmul_i8_i16(%436, %432, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %437 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %438 = arith.addi %437, %c1_i32 : i32
      %439 = arith.cmpi sge, %438, %c2_i32 : i32
      %440 = arith.subi %438, %c2_i32 : i32
      %441 = arith.select %439, %440, %438 : i32
      memref.store %441, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %442 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %443 = arith.addi %442, %c1_i32 : i32
      %444 = arith.cmpi sge, %443, %c2_i32 : i32
      %445 = arith.subi %443, %c2_i32 : i32
      %446 = arith.select %444, %445, %443 : i32
      memref.store %446, %_anonymous5[%c1] : memref<3xi32>
      %447 = arith.addi %427, %c1 : index
      cf.br ^bb190(%447 : index)
    ^bb200:  // pred: ^bb190
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %448 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %449 = arith.addi %448, %c1_i32 : i32
      %450 = arith.cmpi sge, %449, %c1_i32 : i32
      %451 = arith.select %450, %448, %449 : i32
      memref.store %451, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb201(%c0 : index)
    ^bb201(%452: index):  // 2 preds: ^bb200, ^bb210
      %453 = arith.cmpi slt, %452, %c16 : index
      cf.cond_br %453, ^bb202, ^bb211
    ^bb202:  // pred: ^bb201
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %454 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %455 = arith.index_cast %454 : i32 to index
      %456 = arith.index_cast %455 : index to i32
      cf.switch %456 : i32, [
        default: ^bb205,
        0: ^bb203,
        1: ^bb204
      ]
    ^bb203:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb204:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb205:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb206(%457: memref<64x128xi8>):  // 3 preds: ^bb203, ^bb204, ^bb205
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %458 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %459 = arith.index_cast %458 : i32 to index
      %460 = arith.index_cast %459 : index to i32
      cf.switch %460 : i32, [
        default: ^bb209,
        0: ^bb207,
        1: ^bb208
      ]
    ^bb207:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb208:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb209:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb210(%461: memref<32x64xi8>):  // 3 preds: ^bb207, ^bb208, ^bb209
      func.call @matmul_i8_i16(%461, %457, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %462 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %463 = arith.addi %462, %c1_i32 : i32
      %464 = arith.cmpi sge, %463, %c2_i32 : i32
      %465 = arith.subi %463, %c2_i32 : i32
      %466 = arith.select %464, %465, %463 : i32
      memref.store %466, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %467 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %468 = arith.addi %467, %c1_i32 : i32
      %469 = arith.cmpi sge, %468, %c2_i32 : i32
      %470 = arith.subi %468, %c2_i32 : i32
      %471 = arith.select %469, %470, %468 : i32
      memref.store %471, %_anonymous5[%c1] : memref<3xi32>
      %472 = arith.addi %452, %c1 : index
      cf.br ^bb201(%472 : index)
    ^bb211:  // pred: ^bb201
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %473 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %474 = arith.addi %473, %c1_i32 : i32
      %475 = arith.cmpi sge, %474, %c1_i32 : i32
      %476 = arith.select %475, %473, %474 : i32
      memref.store %476, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb212(%c0 : index)
    ^bb212(%477: index):  // 2 preds: ^bb211, ^bb221
      %478 = arith.cmpi slt, %477, %c16 : index
      cf.cond_br %478, ^bb213, ^bb222
    ^bb213:  // pred: ^bb212
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %479 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %480 = arith.index_cast %479 : i32 to index
      %481 = arith.index_cast %480 : index to i32
      cf.switch %481 : i32, [
        default: ^bb216,
        0: ^bb214,
        1: ^bb215
      ]
    ^bb214:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb215:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb216:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb217(%482: memref<64x128xi8>):  // 3 preds: ^bb214, ^bb215, ^bb216
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %483 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %484 = arith.index_cast %483 : i32 to index
      %485 = arith.index_cast %484 : index to i32
      cf.switch %485 : i32, [
        default: ^bb220,
        0: ^bb218,
        1: ^bb219
      ]
    ^bb218:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb219:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb220:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb221(%486: memref<32x64xi8>):  // 3 preds: ^bb218, ^bb219, ^bb220
      func.call @matmul_i8_i16(%486, %482, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %487 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %488 = arith.addi %487, %c1_i32 : i32
      %489 = arith.cmpi sge, %488, %c2_i32 : i32
      %490 = arith.subi %488, %c2_i32 : i32
      %491 = arith.select %489, %490, %488 : i32
      memref.store %491, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %492 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %493 = arith.addi %492, %c1_i32 : i32
      %494 = arith.cmpi sge, %493, %c2_i32 : i32
      %495 = arith.subi %493, %c2_i32 : i32
      %496 = arith.select %494, %495, %493 : i32
      memref.store %496, %_anonymous5[%c1] : memref<3xi32>
      %497 = arith.addi %477, %c1 : index
      cf.br ^bb212(%497 : index)
    ^bb222:  // pred: ^bb212
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %498 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %499 = arith.addi %498, %c1_i32 : i32
      %500 = arith.cmpi sge, %499, %c1_i32 : i32
      %501 = arith.select %500, %498, %499 : i32
      memref.store %501, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb223(%c0 : index)
    ^bb223(%502: index):  // 2 preds: ^bb222, ^bb232
      %503 = arith.cmpi slt, %502, %c16 : index
      cf.cond_br %503, ^bb224, ^bb233
    ^bb224:  // pred: ^bb223
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %504 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %505 = arith.index_cast %504 : i32 to index
      %506 = arith.index_cast %505 : index to i32
      cf.switch %506 : i32, [
        default: ^bb227,
        0: ^bb225,
        1: ^bb226
      ]
    ^bb225:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb226:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb227:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb228(%507: memref<64x128xi8>):  // 3 preds: ^bb225, ^bb226, ^bb227
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %508 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %509 = arith.index_cast %508 : i32 to index
      %510 = arith.index_cast %509 : index to i32
      cf.switch %510 : i32, [
        default: ^bb231,
        0: ^bb229,
        1: ^bb230
      ]
    ^bb229:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb230:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb231:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb232(%511: memref<32x64xi8>):  // 3 preds: ^bb229, ^bb230, ^bb231
      func.call @matmul_i8_i16(%511, %507, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %512 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %513 = arith.addi %512, %c1_i32 : i32
      %514 = arith.cmpi sge, %513, %c2_i32 : i32
      %515 = arith.subi %513, %c2_i32 : i32
      %516 = arith.select %514, %515, %513 : i32
      memref.store %516, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %517 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %518 = arith.addi %517, %c1_i32 : i32
      %519 = arith.cmpi sge, %518, %c2_i32 : i32
      %520 = arith.subi %518, %c2_i32 : i32
      %521 = arith.select %519, %520, %518 : i32
      memref.store %521, %_anonymous5[%c1] : memref<3xi32>
      %522 = arith.addi %502, %c1 : index
      cf.br ^bb223(%522 : index)
    ^bb233:  // pred: ^bb223
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %523 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %524 = arith.addi %523, %c1_i32 : i32
      %525 = arith.cmpi sge, %524, %c1_i32 : i32
      %526 = arith.select %525, %523, %524 : i32
      memref.store %526, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb234(%c0 : index)
    ^bb234(%527: index):  // 2 preds: ^bb233, ^bb243
      %528 = arith.cmpi slt, %527, %c16 : index
      cf.cond_br %528, ^bb235, ^bb244
    ^bb235:  // pred: ^bb234
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %529 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %530 = arith.index_cast %529 : i32 to index
      %531 = arith.index_cast %530 : index to i32
      cf.switch %531 : i32, [
        default: ^bb238,
        0: ^bb236,
        1: ^bb237
      ]
    ^bb236:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb237:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb238:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb239(%532: memref<64x128xi8>):  // 3 preds: ^bb236, ^bb237, ^bb238
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %533 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %534 = arith.index_cast %533 : i32 to index
      %535 = arith.index_cast %534 : index to i32
      cf.switch %535 : i32, [
        default: ^bb242,
        0: ^bb240,
        1: ^bb241
      ]
    ^bb240:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb241:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb242:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb243(%536: memref<32x64xi8>):  // 3 preds: ^bb240, ^bb241, ^bb242
      func.call @matmul_i8_i16(%536, %532, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %537 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %538 = arith.addi %537, %c1_i32 : i32
      %539 = arith.cmpi sge, %538, %c2_i32 : i32
      %540 = arith.subi %538, %c2_i32 : i32
      %541 = arith.select %539, %540, %538 : i32
      memref.store %541, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %542 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %543 = arith.addi %542, %c1_i32 : i32
      %544 = arith.cmpi sge, %543, %c2_i32 : i32
      %545 = arith.subi %543, %c2_i32 : i32
      %546 = arith.select %544, %545, %543 : i32
      memref.store %546, %_anonymous5[%c1] : memref<3xi32>
      %547 = arith.addi %527, %c1 : index
      cf.br ^bb234(%547 : index)
    ^bb244:  // pred: ^bb234
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %548 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %549 = arith.addi %548, %c1_i32 : i32
      %550 = arith.cmpi sge, %549, %c1_i32 : i32
      %551 = arith.select %550, %548, %549 : i32
      memref.store %551, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb245(%c0 : index)
    ^bb245(%552: index):  // 2 preds: ^bb244, ^bb254
      %553 = arith.cmpi slt, %552, %c16 : index
      cf.cond_br %553, ^bb246, ^bb255
    ^bb246:  // pred: ^bb245
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %554 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %555 = arith.index_cast %554 : i32 to index
      %556 = arith.index_cast %555 : index to i32
      cf.switch %556 : i32, [
        default: ^bb249,
        0: ^bb247,
        1: ^bb248
      ]
    ^bb247:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb248:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb249:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb250(%557: memref<64x128xi8>):  // 3 preds: ^bb247, ^bb248, ^bb249
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %558 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %559 = arith.index_cast %558 : i32 to index
      %560 = arith.index_cast %559 : index to i32
      cf.switch %560 : i32, [
        default: ^bb253,
        0: ^bb251,
        1: ^bb252
      ]
    ^bb251:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb252:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb253:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb254(%561: memref<32x64xi8>):  // 3 preds: ^bb251, ^bb252, ^bb253
      func.call @matmul_i8_i16(%561, %557, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %562 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %563 = arith.addi %562, %c1_i32 : i32
      %564 = arith.cmpi sge, %563, %c2_i32 : i32
      %565 = arith.subi %563, %c2_i32 : i32
      %566 = arith.select %564, %565, %563 : i32
      memref.store %566, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %567 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %568 = arith.addi %567, %c1_i32 : i32
      %569 = arith.cmpi sge, %568, %c2_i32 : i32
      %570 = arith.subi %568, %c2_i32 : i32
      %571 = arith.select %569, %570, %568 : i32
      memref.store %571, %_anonymous5[%c1] : memref<3xi32>
      %572 = arith.addi %552, %c1 : index
      cf.br ^bb245(%572 : index)
    ^bb255:  // pred: ^bb245
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %573 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %574 = arith.addi %573, %c1_i32 : i32
      %575 = arith.cmpi sge, %574, %c1_i32 : i32
      %576 = arith.select %575, %573, %574 : i32
      memref.store %576, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb256(%c0 : index)
    ^bb256(%577: index):  // 2 preds: ^bb255, ^bb265
      %578 = arith.cmpi slt, %577, %c16 : index
      cf.cond_br %578, ^bb257, ^bb266
    ^bb257:  // pred: ^bb256
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %579 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %580 = arith.index_cast %579 : i32 to index
      %581 = arith.index_cast %580 : index to i32
      cf.switch %581 : i32, [
        default: ^bb260,
        0: ^bb258,
        1: ^bb259
      ]
    ^bb258:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb259:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb260:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb261(%582: memref<64x128xi8>):  // 3 preds: ^bb258, ^bb259, ^bb260
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %583 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %584 = arith.index_cast %583 : i32 to index
      %585 = arith.index_cast %584 : index to i32
      cf.switch %585 : i32, [
        default: ^bb264,
        0: ^bb262,
        1: ^bb263
      ]
    ^bb262:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb263:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb264:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb265(%586: memref<32x64xi8>):  // 3 preds: ^bb262, ^bb263, ^bb264
      func.call @matmul_i8_i16(%586, %582, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %587 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %588 = arith.addi %587, %c1_i32 : i32
      %589 = arith.cmpi sge, %588, %c2_i32 : i32
      %590 = arith.subi %588, %c2_i32 : i32
      %591 = arith.select %589, %590, %588 : i32
      memref.store %591, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %592 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %593 = arith.addi %592, %c1_i32 : i32
      %594 = arith.cmpi sge, %593, %c2_i32 : i32
      %595 = arith.subi %593, %c2_i32 : i32
      %596 = arith.select %594, %595, %593 : i32
      memref.store %596, %_anonymous5[%c1] : memref<3xi32>
      %597 = arith.addi %577, %c1 : index
      cf.br ^bb256(%597 : index)
    ^bb266:  // pred: ^bb256
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %598 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %599 = arith.addi %598, %c1_i32 : i32
      %600 = arith.cmpi sge, %599, %c1_i32 : i32
      %601 = arith.select %600, %598, %599 : i32
      memref.store %601, %_anonymous5[%c0] : memref<3xi32>
      %602 = arith.addi %0, %c1 : index
      cf.br ^bb1(%602 : index)
    ^bb267:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous6 = aie.buffer(%tile_6_2) {address = 32000 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb266
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb267
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
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
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous6[%c1] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb19(%32: memref<64x128xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb23(%36: memref<32x64xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_i8_i16(%36, %32, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous6[%c1] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb30(%57: memref<64x128xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb34(%61: memref<32x64xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_i8_i16(%61, %57, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous6[%c1] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb41(%82: memref<64x128xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb45(%86: memref<32x64xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_i8_i16(%86, %82, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous6[%c1] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb47(%c0 : index)
    ^bb47(%102: index):  // 2 preds: ^bb46, ^bb56
      %103 = arith.cmpi slt, %102, %c16 : index
      cf.cond_br %103, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %104 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %105 = arith.index_cast %104 : i32 to index
      %106 = arith.index_cast %105 : index to i32
      cf.switch %106 : i32, [
        default: ^bb51,
        0: ^bb49,
        1: ^bb50
      ]
    ^bb49:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb50:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb51:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb52(%107: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb55,
        0: ^bb53,
        1: ^bb54
      ]
    ^bb53:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb54:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb55:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb56(%111: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_i8_i16(%111, %107, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %112 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %113 = arith.addi %112, %c1_i32 : i32
      %114 = arith.cmpi sge, %113, %c2_i32 : i32
      %115 = arith.subi %113, %c2_i32 : i32
      %116 = arith.select %114, %115, %113 : i32
      memref.store %116, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %117 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %118 = arith.addi %117, %c1_i32 : i32
      %119 = arith.cmpi sge, %118, %c2_i32 : i32
      %120 = arith.subi %118, %c2_i32 : i32
      %121 = arith.select %119, %120, %118 : i32
      memref.store %121, %_anonymous6[%c1] : memref<3xi32>
      %122 = arith.addi %102, %c1 : index
      cf.br ^bb47(%122 : index)
    ^bb57:  // pred: ^bb47
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %123 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c1_i32 : i32
      %126 = arith.select %125, %123, %124 : i32
      memref.store %126, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb58(%c0 : index)
    ^bb58(%127: index):  // 2 preds: ^bb57, ^bb67
      %128 = arith.cmpi slt, %127, %c16 : index
      cf.cond_br %128, ^bb59, ^bb68
    ^bb59:  // pred: ^bb58
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %129 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %130 = arith.index_cast %129 : i32 to index
      %131 = arith.index_cast %130 : index to i32
      cf.switch %131 : i32, [
        default: ^bb62,
        0: ^bb60,
        1: ^bb61
      ]
    ^bb60:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb61:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb62:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb63(%132: memref<64x128xi8>):  // 3 preds: ^bb60, ^bb61, ^bb62
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %133 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %134 = arith.index_cast %133 : i32 to index
      %135 = arith.index_cast %134 : index to i32
      cf.switch %135 : i32, [
        default: ^bb66,
        0: ^bb64,
        1: ^bb65
      ]
    ^bb64:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb66:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb67(%136: memref<32x64xi8>):  // 3 preds: ^bb64, ^bb65, ^bb66
      func.call @matmul_i8_i16(%136, %132, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %137 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %138 = arith.addi %137, %c1_i32 : i32
      %139 = arith.cmpi sge, %138, %c2_i32 : i32
      %140 = arith.subi %138, %c2_i32 : i32
      %141 = arith.select %139, %140, %138 : i32
      memref.store %141, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %142 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %143 = arith.addi %142, %c1_i32 : i32
      %144 = arith.cmpi sge, %143, %c2_i32 : i32
      %145 = arith.subi %143, %c2_i32 : i32
      %146 = arith.select %144, %145, %143 : i32
      memref.store %146, %_anonymous6[%c1] : memref<3xi32>
      %147 = arith.addi %127, %c1 : index
      cf.br ^bb58(%147 : index)
    ^bb68:  // pred: ^bb58
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %148 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c1_i32 : i32
      %151 = arith.select %150, %148, %149 : i32
      memref.store %151, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb69(%c0 : index)
    ^bb69(%152: index):  // 2 preds: ^bb68, ^bb78
      %153 = arith.cmpi slt, %152, %c16 : index
      cf.cond_br %153, ^bb70, ^bb79
    ^bb70:  // pred: ^bb69
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %154 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %155 = arith.index_cast %154 : i32 to index
      %156 = arith.index_cast %155 : index to i32
      cf.switch %156 : i32, [
        default: ^bb73,
        0: ^bb71,
        1: ^bb72
      ]
    ^bb71:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb72:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb73:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb74(%157: memref<64x128xi8>):  // 3 preds: ^bb71, ^bb72, ^bb73
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %158 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %159 = arith.index_cast %158 : i32 to index
      %160 = arith.index_cast %159 : index to i32
      cf.switch %160 : i32, [
        default: ^bb77,
        0: ^bb75,
        1: ^bb76
      ]
    ^bb75:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb76:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb77:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb78(%161: memref<32x64xi8>):  // 3 preds: ^bb75, ^bb76, ^bb77
      func.call @matmul_i8_i16(%161, %157, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %162 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %163 = arith.addi %162, %c1_i32 : i32
      %164 = arith.cmpi sge, %163, %c2_i32 : i32
      %165 = arith.subi %163, %c2_i32 : i32
      %166 = arith.select %164, %165, %163 : i32
      memref.store %166, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %167 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %168 = arith.addi %167, %c1_i32 : i32
      %169 = arith.cmpi sge, %168, %c2_i32 : i32
      %170 = arith.subi %168, %c2_i32 : i32
      %171 = arith.select %169, %170, %168 : i32
      memref.store %171, %_anonymous6[%c1] : memref<3xi32>
      %172 = arith.addi %152, %c1 : index
      cf.br ^bb69(%172 : index)
    ^bb79:  // pred: ^bb69
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %173 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %174 = arith.addi %173, %c1_i32 : i32
      %175 = arith.cmpi sge, %174, %c1_i32 : i32
      %176 = arith.select %175, %173, %174 : i32
      memref.store %176, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb80(%c0 : index)
    ^bb80(%177: index):  // 2 preds: ^bb79, ^bb89
      %178 = arith.cmpi slt, %177, %c16 : index
      cf.cond_br %178, ^bb81, ^bb90
    ^bb81:  // pred: ^bb80
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %179 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %180 = arith.index_cast %179 : i32 to index
      %181 = arith.index_cast %180 : index to i32
      cf.switch %181 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb85(%182: memref<64x128xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %183 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %184 = arith.index_cast %183 : i32 to index
      %185 = arith.index_cast %184 : index to i32
      cf.switch %185 : i32, [
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
    ^bb89(%186: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%186, %182, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %187 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %188 = arith.addi %187, %c1_i32 : i32
      %189 = arith.cmpi sge, %188, %c2_i32 : i32
      %190 = arith.subi %188, %c2_i32 : i32
      %191 = arith.select %189, %190, %188 : i32
      memref.store %191, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %192 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %193 = arith.addi %192, %c1_i32 : i32
      %194 = arith.cmpi sge, %193, %c2_i32 : i32
      %195 = arith.subi %193, %c2_i32 : i32
      %196 = arith.select %194, %195, %193 : i32
      memref.store %196, %_anonymous6[%c1] : memref<3xi32>
      %197 = arith.addi %177, %c1 : index
      cf.br ^bb80(%197 : index)
    ^bb90:  // pred: ^bb80
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %198 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %199 = arith.addi %198, %c1_i32 : i32
      %200 = arith.cmpi sge, %199, %c1_i32 : i32
      %201 = arith.select %200, %198, %199 : i32
      memref.store %201, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb91(%c0 : index)
    ^bb91(%202: index):  // 2 preds: ^bb90, ^bb100
      %203 = arith.cmpi slt, %202, %c16 : index
      cf.cond_br %203, ^bb92, ^bb101
    ^bb92:  // pred: ^bb91
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %204 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %205 = arith.index_cast %204 : i32 to index
      %206 = arith.index_cast %205 : index to i32
      cf.switch %206 : i32, [
        default: ^bb95,
        0: ^bb93,
        1: ^bb94
      ]
    ^bb93:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb94:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb95:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb96(%207: memref<64x128xi8>):  // 3 preds: ^bb93, ^bb94, ^bb95
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %208 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %209 = arith.index_cast %208 : i32 to index
      %210 = arith.index_cast %209 : index to i32
      cf.switch %210 : i32, [
        default: ^bb99,
        0: ^bb97,
        1: ^bb98
      ]
    ^bb97:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb98:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb99:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb100(%211: memref<32x64xi8>):  // 3 preds: ^bb97, ^bb98, ^bb99
      func.call @matmul_i8_i16(%211, %207, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %212 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %213 = arith.addi %212, %c1_i32 : i32
      %214 = arith.cmpi sge, %213, %c2_i32 : i32
      %215 = arith.subi %213, %c2_i32 : i32
      %216 = arith.select %214, %215, %213 : i32
      memref.store %216, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %217 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %218 = arith.addi %217, %c1_i32 : i32
      %219 = arith.cmpi sge, %218, %c2_i32 : i32
      %220 = arith.subi %218, %c2_i32 : i32
      %221 = arith.select %219, %220, %218 : i32
      memref.store %221, %_anonymous6[%c1] : memref<3xi32>
      %222 = arith.addi %202, %c1 : index
      cf.br ^bb91(%222 : index)
    ^bb101:  // pred: ^bb91
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %223 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %224 = arith.addi %223, %c1_i32 : i32
      %225 = arith.cmpi sge, %224, %c1_i32 : i32
      %226 = arith.select %225, %223, %224 : i32
      memref.store %226, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb102(%c0 : index)
    ^bb102(%227: index):  // 2 preds: ^bb101, ^bb111
      %228 = arith.cmpi slt, %227, %c16 : index
      cf.cond_br %228, ^bb103, ^bb112
    ^bb103:  // pred: ^bb102
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %229 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %230 = arith.index_cast %229 : i32 to index
      %231 = arith.index_cast %230 : index to i32
      cf.switch %231 : i32, [
        default: ^bb106,
        0: ^bb104,
        1: ^bb105
      ]
    ^bb104:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb105:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb106:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb107(%232: memref<64x128xi8>):  // 3 preds: ^bb104, ^bb105, ^bb106
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %233 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %234 = arith.index_cast %233 : i32 to index
      %235 = arith.index_cast %234 : index to i32
      cf.switch %235 : i32, [
        default: ^bb110,
        0: ^bb108,
        1: ^bb109
      ]
    ^bb108:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb109:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb110:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb111(%236: memref<32x64xi8>):  // 3 preds: ^bb108, ^bb109, ^bb110
      func.call @matmul_i8_i16(%236, %232, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %237 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %238 = arith.addi %237, %c1_i32 : i32
      %239 = arith.cmpi sge, %238, %c2_i32 : i32
      %240 = arith.subi %238, %c2_i32 : i32
      %241 = arith.select %239, %240, %238 : i32
      memref.store %241, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %242 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %243 = arith.addi %242, %c1_i32 : i32
      %244 = arith.cmpi sge, %243, %c2_i32 : i32
      %245 = arith.subi %243, %c2_i32 : i32
      %246 = arith.select %244, %245, %243 : i32
      memref.store %246, %_anonymous6[%c1] : memref<3xi32>
      %247 = arith.addi %227, %c1 : index
      cf.br ^bb102(%247 : index)
    ^bb112:  // pred: ^bb102
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %248 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %249 = arith.addi %248, %c1_i32 : i32
      %250 = arith.cmpi sge, %249, %c1_i32 : i32
      %251 = arith.select %250, %248, %249 : i32
      memref.store %251, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb113(%c0 : index)
    ^bb113(%252: index):  // 2 preds: ^bb112, ^bb122
      %253 = arith.cmpi slt, %252, %c16 : index
      cf.cond_br %253, ^bb114, ^bb123
    ^bb114:  // pred: ^bb113
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %254 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %255 = arith.index_cast %254 : i32 to index
      %256 = arith.index_cast %255 : index to i32
      cf.switch %256 : i32, [
        default: ^bb117,
        0: ^bb115,
        1: ^bb116
      ]
    ^bb115:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb116:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb117:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb118(%257: memref<64x128xi8>):  // 3 preds: ^bb115, ^bb116, ^bb117
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %258 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %259 = arith.index_cast %258 : i32 to index
      %260 = arith.index_cast %259 : index to i32
      cf.switch %260 : i32, [
        default: ^bb121,
        0: ^bb119,
        1: ^bb120
      ]
    ^bb119:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb120:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb121:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb122(%261: memref<32x64xi8>):  // 3 preds: ^bb119, ^bb120, ^bb121
      func.call @matmul_i8_i16(%261, %257, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %262 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %263 = arith.addi %262, %c1_i32 : i32
      %264 = arith.cmpi sge, %263, %c2_i32 : i32
      %265 = arith.subi %263, %c2_i32 : i32
      %266 = arith.select %264, %265, %263 : i32
      memref.store %266, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %267 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %268 = arith.addi %267, %c1_i32 : i32
      %269 = arith.cmpi sge, %268, %c2_i32 : i32
      %270 = arith.subi %268, %c2_i32 : i32
      %271 = arith.select %269, %270, %268 : i32
      memref.store %271, %_anonymous6[%c1] : memref<3xi32>
      %272 = arith.addi %252, %c1 : index
      cf.br ^bb113(%272 : index)
    ^bb123:  // pred: ^bb113
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %273 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %274 = arith.addi %273, %c1_i32 : i32
      %275 = arith.cmpi sge, %274, %c1_i32 : i32
      %276 = arith.select %275, %273, %274 : i32
      memref.store %276, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb124(%c0 : index)
    ^bb124(%277: index):  // 2 preds: ^bb123, ^bb133
      %278 = arith.cmpi slt, %277, %c16 : index
      cf.cond_br %278, ^bb125, ^bb134
    ^bb125:  // pred: ^bb124
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %279 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %280 = arith.index_cast %279 : i32 to index
      %281 = arith.index_cast %280 : index to i32
      cf.switch %281 : i32, [
        default: ^bb128,
        0: ^bb126,
        1: ^bb127
      ]
    ^bb126:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb127:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb128:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb129(%282: memref<64x128xi8>):  // 3 preds: ^bb126, ^bb127, ^bb128
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %283 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %284 = arith.index_cast %283 : i32 to index
      %285 = arith.index_cast %284 : index to i32
      cf.switch %285 : i32, [
        default: ^bb132,
        0: ^bb130,
        1: ^bb131
      ]
    ^bb130:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb131:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb132:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb133(%286: memref<32x64xi8>):  // 3 preds: ^bb130, ^bb131, ^bb132
      func.call @matmul_i8_i16(%286, %282, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %287 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %288 = arith.addi %287, %c1_i32 : i32
      %289 = arith.cmpi sge, %288, %c2_i32 : i32
      %290 = arith.subi %288, %c2_i32 : i32
      %291 = arith.select %289, %290, %288 : i32
      memref.store %291, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %292 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %293 = arith.addi %292, %c1_i32 : i32
      %294 = arith.cmpi sge, %293, %c2_i32 : i32
      %295 = arith.subi %293, %c2_i32 : i32
      %296 = arith.select %294, %295, %293 : i32
      memref.store %296, %_anonymous6[%c1] : memref<3xi32>
      %297 = arith.addi %277, %c1 : index
      cf.br ^bb124(%297 : index)
    ^bb134:  // pred: ^bb124
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %298 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %299 = arith.addi %298, %c1_i32 : i32
      %300 = arith.cmpi sge, %299, %c1_i32 : i32
      %301 = arith.select %300, %298, %299 : i32
      memref.store %301, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb135(%c0 : index)
    ^bb135(%302: index):  // 2 preds: ^bb134, ^bb144
      %303 = arith.cmpi slt, %302, %c16 : index
      cf.cond_br %303, ^bb136, ^bb145
    ^bb136:  // pred: ^bb135
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %304 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %305 = arith.index_cast %304 : i32 to index
      %306 = arith.index_cast %305 : index to i32
      cf.switch %306 : i32, [
        default: ^bb139,
        0: ^bb137,
        1: ^bb138
      ]
    ^bb137:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb138:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb139:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb140(%307: memref<64x128xi8>):  // 3 preds: ^bb137, ^bb138, ^bb139
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %308 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %309 = arith.index_cast %308 : i32 to index
      %310 = arith.index_cast %309 : index to i32
      cf.switch %310 : i32, [
        default: ^bb143,
        0: ^bb141,
        1: ^bb142
      ]
    ^bb141:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb142:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb143:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb144(%311: memref<32x64xi8>):  // 3 preds: ^bb141, ^bb142, ^bb143
      func.call @matmul_i8_i16(%311, %307, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %312 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %313 = arith.addi %312, %c1_i32 : i32
      %314 = arith.cmpi sge, %313, %c2_i32 : i32
      %315 = arith.subi %313, %c2_i32 : i32
      %316 = arith.select %314, %315, %313 : i32
      memref.store %316, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %317 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %318 = arith.addi %317, %c1_i32 : i32
      %319 = arith.cmpi sge, %318, %c2_i32 : i32
      %320 = arith.subi %318, %c2_i32 : i32
      %321 = arith.select %319, %320, %318 : i32
      memref.store %321, %_anonymous6[%c1] : memref<3xi32>
      %322 = arith.addi %302, %c1 : index
      cf.br ^bb135(%322 : index)
    ^bb145:  // pred: ^bb135
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %323 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %324 = arith.addi %323, %c1_i32 : i32
      %325 = arith.cmpi sge, %324, %c1_i32 : i32
      %326 = arith.select %325, %323, %324 : i32
      memref.store %326, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb146(%c0 : index)
    ^bb146(%327: index):  // 2 preds: ^bb145, ^bb155
      %328 = arith.cmpi slt, %327, %c16 : index
      cf.cond_br %328, ^bb147, ^bb156
    ^bb147:  // pred: ^bb146
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %329 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %330 = arith.index_cast %329 : i32 to index
      %331 = arith.index_cast %330 : index to i32
      cf.switch %331 : i32, [
        default: ^bb150,
        0: ^bb148,
        1: ^bb149
      ]
    ^bb148:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb149:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb150:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb151(%332: memref<64x128xi8>):  // 3 preds: ^bb148, ^bb149, ^bb150
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %333 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %334 = arith.index_cast %333 : i32 to index
      %335 = arith.index_cast %334 : index to i32
      cf.switch %335 : i32, [
        default: ^bb154,
        0: ^bb152,
        1: ^bb153
      ]
    ^bb152:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb153:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb154:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb155(%336: memref<32x64xi8>):  // 3 preds: ^bb152, ^bb153, ^bb154
      func.call @matmul_i8_i16(%336, %332, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %337 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %338 = arith.addi %337, %c1_i32 : i32
      %339 = arith.cmpi sge, %338, %c2_i32 : i32
      %340 = arith.subi %338, %c2_i32 : i32
      %341 = arith.select %339, %340, %338 : i32
      memref.store %341, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %342 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %343 = arith.addi %342, %c1_i32 : i32
      %344 = arith.cmpi sge, %343, %c2_i32 : i32
      %345 = arith.subi %343, %c2_i32 : i32
      %346 = arith.select %344, %345, %343 : i32
      memref.store %346, %_anonymous6[%c1] : memref<3xi32>
      %347 = arith.addi %327, %c1 : index
      cf.br ^bb146(%347 : index)
    ^bb156:  // pred: ^bb146
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %348 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %349 = arith.addi %348, %c1_i32 : i32
      %350 = arith.cmpi sge, %349, %c1_i32 : i32
      %351 = arith.select %350, %348, %349 : i32
      memref.store %351, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb157(%c0 : index)
    ^bb157(%352: index):  // 2 preds: ^bb156, ^bb166
      %353 = arith.cmpi slt, %352, %c16 : index
      cf.cond_br %353, ^bb158, ^bb167
    ^bb158:  // pred: ^bb157
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %354 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %355 = arith.index_cast %354 : i32 to index
      %356 = arith.index_cast %355 : index to i32
      cf.switch %356 : i32, [
        default: ^bb161,
        0: ^bb159,
        1: ^bb160
      ]
    ^bb159:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb160:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb161:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb162(%357: memref<64x128xi8>):  // 3 preds: ^bb159, ^bb160, ^bb161
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %358 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %359 = arith.index_cast %358 : i32 to index
      %360 = arith.index_cast %359 : index to i32
      cf.switch %360 : i32, [
        default: ^bb165,
        0: ^bb163,
        1: ^bb164
      ]
    ^bb163:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb164:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb165:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb166(%361: memref<32x64xi8>):  // 3 preds: ^bb163, ^bb164, ^bb165
      func.call @matmul_i8_i16(%361, %357, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %362 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %363 = arith.addi %362, %c1_i32 : i32
      %364 = arith.cmpi sge, %363, %c2_i32 : i32
      %365 = arith.subi %363, %c2_i32 : i32
      %366 = arith.select %364, %365, %363 : i32
      memref.store %366, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %367 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %368 = arith.addi %367, %c1_i32 : i32
      %369 = arith.cmpi sge, %368, %c2_i32 : i32
      %370 = arith.subi %368, %c2_i32 : i32
      %371 = arith.select %369, %370, %368 : i32
      memref.store %371, %_anonymous6[%c1] : memref<3xi32>
      %372 = arith.addi %352, %c1 : index
      cf.br ^bb157(%372 : index)
    ^bb167:  // pred: ^bb157
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %373 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %374 = arith.addi %373, %c1_i32 : i32
      %375 = arith.cmpi sge, %374, %c1_i32 : i32
      %376 = arith.select %375, %373, %374 : i32
      memref.store %376, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb168(%c0 : index)
    ^bb168(%377: index):  // 2 preds: ^bb167, ^bb177
      %378 = arith.cmpi slt, %377, %c16 : index
      cf.cond_br %378, ^bb169, ^bb178
    ^bb169:  // pred: ^bb168
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %379 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %380 = arith.index_cast %379 : i32 to index
      %381 = arith.index_cast %380 : index to i32
      cf.switch %381 : i32, [
        default: ^bb172,
        0: ^bb170,
        1: ^bb171
      ]
    ^bb170:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb171:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb172:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb173(%382: memref<64x128xi8>):  // 3 preds: ^bb170, ^bb171, ^bb172
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %383 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %384 = arith.index_cast %383 : i32 to index
      %385 = arith.index_cast %384 : index to i32
      cf.switch %385 : i32, [
        default: ^bb176,
        0: ^bb174,
        1: ^bb175
      ]
    ^bb174:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb175:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb176:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb177(%386: memref<32x64xi8>):  // 3 preds: ^bb174, ^bb175, ^bb176
      func.call @matmul_i8_i16(%386, %382, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %387 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %388 = arith.addi %387, %c1_i32 : i32
      %389 = arith.cmpi sge, %388, %c2_i32 : i32
      %390 = arith.subi %388, %c2_i32 : i32
      %391 = arith.select %389, %390, %388 : i32
      memref.store %391, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %392 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %393 = arith.addi %392, %c1_i32 : i32
      %394 = arith.cmpi sge, %393, %c2_i32 : i32
      %395 = arith.subi %393, %c2_i32 : i32
      %396 = arith.select %394, %395, %393 : i32
      memref.store %396, %_anonymous6[%c1] : memref<3xi32>
      %397 = arith.addi %377, %c1 : index
      cf.br ^bb168(%397 : index)
    ^bb178:  // pred: ^bb168
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %398 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %399 = arith.addi %398, %c1_i32 : i32
      %400 = arith.cmpi sge, %399, %c1_i32 : i32
      %401 = arith.select %400, %398, %399 : i32
      memref.store %401, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb179(%c0 : index)
    ^bb179(%402: index):  // 2 preds: ^bb178, ^bb188
      %403 = arith.cmpi slt, %402, %c16 : index
      cf.cond_br %403, ^bb180, ^bb189
    ^bb180:  // pred: ^bb179
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %404 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %405 = arith.index_cast %404 : i32 to index
      %406 = arith.index_cast %405 : index to i32
      cf.switch %406 : i32, [
        default: ^bb183,
        0: ^bb181,
        1: ^bb182
      ]
    ^bb181:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb182:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb183:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb184(%407: memref<64x128xi8>):  // 3 preds: ^bb181, ^bb182, ^bb183
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %408 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %409 = arith.index_cast %408 : i32 to index
      %410 = arith.index_cast %409 : index to i32
      cf.switch %410 : i32, [
        default: ^bb187,
        0: ^bb185,
        1: ^bb186
      ]
    ^bb185:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb186:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb187:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb188(%411: memref<32x64xi8>):  // 3 preds: ^bb185, ^bb186, ^bb187
      func.call @matmul_i8_i16(%411, %407, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %412 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %413 = arith.addi %412, %c1_i32 : i32
      %414 = arith.cmpi sge, %413, %c2_i32 : i32
      %415 = arith.subi %413, %c2_i32 : i32
      %416 = arith.select %414, %415, %413 : i32
      memref.store %416, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %417 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %418 = arith.addi %417, %c1_i32 : i32
      %419 = arith.cmpi sge, %418, %c2_i32 : i32
      %420 = arith.subi %418, %c2_i32 : i32
      %421 = arith.select %419, %420, %418 : i32
      memref.store %421, %_anonymous6[%c1] : memref<3xi32>
      %422 = arith.addi %402, %c1 : index
      cf.br ^bb179(%422 : index)
    ^bb189:  // pred: ^bb179
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %423 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %424 = arith.addi %423, %c1_i32 : i32
      %425 = arith.cmpi sge, %424, %c1_i32 : i32
      %426 = arith.select %425, %423, %424 : i32
      memref.store %426, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb190(%c0 : index)
    ^bb190(%427: index):  // 2 preds: ^bb189, ^bb199
      %428 = arith.cmpi slt, %427, %c16 : index
      cf.cond_br %428, ^bb191, ^bb200
    ^bb191:  // pred: ^bb190
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %429 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %430 = arith.index_cast %429 : i32 to index
      %431 = arith.index_cast %430 : index to i32
      cf.switch %431 : i32, [
        default: ^bb194,
        0: ^bb192,
        1: ^bb193
      ]
    ^bb192:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb193:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb194:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb195(%432: memref<64x128xi8>):  // 3 preds: ^bb192, ^bb193, ^bb194
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %433 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %434 = arith.index_cast %433 : i32 to index
      %435 = arith.index_cast %434 : index to i32
      cf.switch %435 : i32, [
        default: ^bb198,
        0: ^bb196,
        1: ^bb197
      ]
    ^bb196:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb197:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb198:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb199(%436: memref<32x64xi8>):  // 3 preds: ^bb196, ^bb197, ^bb198
      func.call @matmul_i8_i16(%436, %432, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %437 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %438 = arith.addi %437, %c1_i32 : i32
      %439 = arith.cmpi sge, %438, %c2_i32 : i32
      %440 = arith.subi %438, %c2_i32 : i32
      %441 = arith.select %439, %440, %438 : i32
      memref.store %441, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %442 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %443 = arith.addi %442, %c1_i32 : i32
      %444 = arith.cmpi sge, %443, %c2_i32 : i32
      %445 = arith.subi %443, %c2_i32 : i32
      %446 = arith.select %444, %445, %443 : i32
      memref.store %446, %_anonymous6[%c1] : memref<3xi32>
      %447 = arith.addi %427, %c1 : index
      cf.br ^bb190(%447 : index)
    ^bb200:  // pred: ^bb190
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %448 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %449 = arith.addi %448, %c1_i32 : i32
      %450 = arith.cmpi sge, %449, %c1_i32 : i32
      %451 = arith.select %450, %448, %449 : i32
      memref.store %451, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb201(%c0 : index)
    ^bb201(%452: index):  // 2 preds: ^bb200, ^bb210
      %453 = arith.cmpi slt, %452, %c16 : index
      cf.cond_br %453, ^bb202, ^bb211
    ^bb202:  // pred: ^bb201
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %454 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %455 = arith.index_cast %454 : i32 to index
      %456 = arith.index_cast %455 : index to i32
      cf.switch %456 : i32, [
        default: ^bb205,
        0: ^bb203,
        1: ^bb204
      ]
    ^bb203:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb204:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb205:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb206(%457: memref<64x128xi8>):  // 3 preds: ^bb203, ^bb204, ^bb205
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %458 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %459 = arith.index_cast %458 : i32 to index
      %460 = arith.index_cast %459 : index to i32
      cf.switch %460 : i32, [
        default: ^bb209,
        0: ^bb207,
        1: ^bb208
      ]
    ^bb207:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb208:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb209:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb210(%461: memref<32x64xi8>):  // 3 preds: ^bb207, ^bb208, ^bb209
      func.call @matmul_i8_i16(%461, %457, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %462 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %463 = arith.addi %462, %c1_i32 : i32
      %464 = arith.cmpi sge, %463, %c2_i32 : i32
      %465 = arith.subi %463, %c2_i32 : i32
      %466 = arith.select %464, %465, %463 : i32
      memref.store %466, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %467 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %468 = arith.addi %467, %c1_i32 : i32
      %469 = arith.cmpi sge, %468, %c2_i32 : i32
      %470 = arith.subi %468, %c2_i32 : i32
      %471 = arith.select %469, %470, %468 : i32
      memref.store %471, %_anonymous6[%c1] : memref<3xi32>
      %472 = arith.addi %452, %c1 : index
      cf.br ^bb201(%472 : index)
    ^bb211:  // pred: ^bb201
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %473 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %474 = arith.addi %473, %c1_i32 : i32
      %475 = arith.cmpi sge, %474, %c1_i32 : i32
      %476 = arith.select %475, %473, %474 : i32
      memref.store %476, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb212(%c0 : index)
    ^bb212(%477: index):  // 2 preds: ^bb211, ^bb221
      %478 = arith.cmpi slt, %477, %c16 : index
      cf.cond_br %478, ^bb213, ^bb222
    ^bb213:  // pred: ^bb212
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %479 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %480 = arith.index_cast %479 : i32 to index
      %481 = arith.index_cast %480 : index to i32
      cf.switch %481 : i32, [
        default: ^bb216,
        0: ^bb214,
        1: ^bb215
      ]
    ^bb214:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb215:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb216:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb217(%482: memref<64x128xi8>):  // 3 preds: ^bb214, ^bb215, ^bb216
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %483 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %484 = arith.index_cast %483 : i32 to index
      %485 = arith.index_cast %484 : index to i32
      cf.switch %485 : i32, [
        default: ^bb220,
        0: ^bb218,
        1: ^bb219
      ]
    ^bb218:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb219:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb220:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb221(%486: memref<32x64xi8>):  // 3 preds: ^bb218, ^bb219, ^bb220
      func.call @matmul_i8_i16(%486, %482, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %487 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %488 = arith.addi %487, %c1_i32 : i32
      %489 = arith.cmpi sge, %488, %c2_i32 : i32
      %490 = arith.subi %488, %c2_i32 : i32
      %491 = arith.select %489, %490, %488 : i32
      memref.store %491, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %492 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %493 = arith.addi %492, %c1_i32 : i32
      %494 = arith.cmpi sge, %493, %c2_i32 : i32
      %495 = arith.subi %493, %c2_i32 : i32
      %496 = arith.select %494, %495, %493 : i32
      memref.store %496, %_anonymous6[%c1] : memref<3xi32>
      %497 = arith.addi %477, %c1 : index
      cf.br ^bb212(%497 : index)
    ^bb222:  // pred: ^bb212
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %498 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %499 = arith.addi %498, %c1_i32 : i32
      %500 = arith.cmpi sge, %499, %c1_i32 : i32
      %501 = arith.select %500, %498, %499 : i32
      memref.store %501, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb223(%c0 : index)
    ^bb223(%502: index):  // 2 preds: ^bb222, ^bb232
      %503 = arith.cmpi slt, %502, %c16 : index
      cf.cond_br %503, ^bb224, ^bb233
    ^bb224:  // pred: ^bb223
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %504 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %505 = arith.index_cast %504 : i32 to index
      %506 = arith.index_cast %505 : index to i32
      cf.switch %506 : i32, [
        default: ^bb227,
        0: ^bb225,
        1: ^bb226
      ]
    ^bb225:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb226:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb227:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb228(%507: memref<64x128xi8>):  // 3 preds: ^bb225, ^bb226, ^bb227
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %508 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %509 = arith.index_cast %508 : i32 to index
      %510 = arith.index_cast %509 : index to i32
      cf.switch %510 : i32, [
        default: ^bb231,
        0: ^bb229,
        1: ^bb230
      ]
    ^bb229:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb230:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb231:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb232(%511: memref<32x64xi8>):  // 3 preds: ^bb229, ^bb230, ^bb231
      func.call @matmul_i8_i16(%511, %507, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %512 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %513 = arith.addi %512, %c1_i32 : i32
      %514 = arith.cmpi sge, %513, %c2_i32 : i32
      %515 = arith.subi %513, %c2_i32 : i32
      %516 = arith.select %514, %515, %513 : i32
      memref.store %516, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %517 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %518 = arith.addi %517, %c1_i32 : i32
      %519 = arith.cmpi sge, %518, %c2_i32 : i32
      %520 = arith.subi %518, %c2_i32 : i32
      %521 = arith.select %519, %520, %518 : i32
      memref.store %521, %_anonymous6[%c1] : memref<3xi32>
      %522 = arith.addi %502, %c1 : index
      cf.br ^bb223(%522 : index)
    ^bb233:  // pred: ^bb223
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %523 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %524 = arith.addi %523, %c1_i32 : i32
      %525 = arith.cmpi sge, %524, %c1_i32 : i32
      %526 = arith.select %525, %523, %524 : i32
      memref.store %526, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb234(%c0 : index)
    ^bb234(%527: index):  // 2 preds: ^bb233, ^bb243
      %528 = arith.cmpi slt, %527, %c16 : index
      cf.cond_br %528, ^bb235, ^bb244
    ^bb235:  // pred: ^bb234
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %529 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %530 = arith.index_cast %529 : i32 to index
      %531 = arith.index_cast %530 : index to i32
      cf.switch %531 : i32, [
        default: ^bb238,
        0: ^bb236,
        1: ^bb237
      ]
    ^bb236:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb237:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb238:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb239(%532: memref<64x128xi8>):  // 3 preds: ^bb236, ^bb237, ^bb238
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %533 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %534 = arith.index_cast %533 : i32 to index
      %535 = arith.index_cast %534 : index to i32
      cf.switch %535 : i32, [
        default: ^bb242,
        0: ^bb240,
        1: ^bb241
      ]
    ^bb240:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb241:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb242:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb243(%536: memref<32x64xi8>):  // 3 preds: ^bb240, ^bb241, ^bb242
      func.call @matmul_i8_i16(%536, %532, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %537 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %538 = arith.addi %537, %c1_i32 : i32
      %539 = arith.cmpi sge, %538, %c2_i32 : i32
      %540 = arith.subi %538, %c2_i32 : i32
      %541 = arith.select %539, %540, %538 : i32
      memref.store %541, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %542 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %543 = arith.addi %542, %c1_i32 : i32
      %544 = arith.cmpi sge, %543, %c2_i32 : i32
      %545 = arith.subi %543, %c2_i32 : i32
      %546 = arith.select %544, %545, %543 : i32
      memref.store %546, %_anonymous6[%c1] : memref<3xi32>
      %547 = arith.addi %527, %c1 : index
      cf.br ^bb234(%547 : index)
    ^bb244:  // pred: ^bb234
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %548 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %549 = arith.addi %548, %c1_i32 : i32
      %550 = arith.cmpi sge, %549, %c1_i32 : i32
      %551 = arith.select %550, %548, %549 : i32
      memref.store %551, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb245(%c0 : index)
    ^bb245(%552: index):  // 2 preds: ^bb244, ^bb254
      %553 = arith.cmpi slt, %552, %c16 : index
      cf.cond_br %553, ^bb246, ^bb255
    ^bb246:  // pred: ^bb245
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %554 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %555 = arith.index_cast %554 : i32 to index
      %556 = arith.index_cast %555 : index to i32
      cf.switch %556 : i32, [
        default: ^bb249,
        0: ^bb247,
        1: ^bb248
      ]
    ^bb247:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb248:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb249:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb250(%557: memref<64x128xi8>):  // 3 preds: ^bb247, ^bb248, ^bb249
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %558 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %559 = arith.index_cast %558 : i32 to index
      %560 = arith.index_cast %559 : index to i32
      cf.switch %560 : i32, [
        default: ^bb253,
        0: ^bb251,
        1: ^bb252
      ]
    ^bb251:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb252:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb253:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb254(%561: memref<32x64xi8>):  // 3 preds: ^bb251, ^bb252, ^bb253
      func.call @matmul_i8_i16(%561, %557, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %562 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %563 = arith.addi %562, %c1_i32 : i32
      %564 = arith.cmpi sge, %563, %c2_i32 : i32
      %565 = arith.subi %563, %c2_i32 : i32
      %566 = arith.select %564, %565, %563 : i32
      memref.store %566, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %567 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %568 = arith.addi %567, %c1_i32 : i32
      %569 = arith.cmpi sge, %568, %c2_i32 : i32
      %570 = arith.subi %568, %c2_i32 : i32
      %571 = arith.select %569, %570, %568 : i32
      memref.store %571, %_anonymous6[%c1] : memref<3xi32>
      %572 = arith.addi %552, %c1 : index
      cf.br ^bb245(%572 : index)
    ^bb255:  // pred: ^bb245
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %573 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %574 = arith.addi %573, %c1_i32 : i32
      %575 = arith.cmpi sge, %574, %c1_i32 : i32
      %576 = arith.select %575, %573, %574 : i32
      memref.store %576, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb256(%c0 : index)
    ^bb256(%577: index):  // 2 preds: ^bb255, ^bb265
      %578 = arith.cmpi slt, %577, %c16 : index
      cf.cond_br %578, ^bb257, ^bb266
    ^bb257:  // pred: ^bb256
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %579 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %580 = arith.index_cast %579 : i32 to index
      %581 = arith.index_cast %580 : index to i32
      cf.switch %581 : i32, [
        default: ^bb260,
        0: ^bb258,
        1: ^bb259
      ]
    ^bb258:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb259:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb260:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb261(%582: memref<64x128xi8>):  // 3 preds: ^bb258, ^bb259, ^bb260
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %583 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %584 = arith.index_cast %583 : i32 to index
      %585 = arith.index_cast %584 : index to i32
      cf.switch %585 : i32, [
        default: ^bb264,
        0: ^bb262,
        1: ^bb263
      ]
    ^bb262:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb263:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb264:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb265(%586: memref<32x64xi8>):  // 3 preds: ^bb262, ^bb263, ^bb264
      func.call @matmul_i8_i16(%586, %582, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %587 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %588 = arith.addi %587, %c1_i32 : i32
      %589 = arith.cmpi sge, %588, %c2_i32 : i32
      %590 = arith.subi %588, %c2_i32 : i32
      %591 = arith.select %589, %590, %588 : i32
      memref.store %591, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %592 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %593 = arith.addi %592, %c1_i32 : i32
      %594 = arith.cmpi sge, %593, %c2_i32 : i32
      %595 = arith.subi %593, %c2_i32 : i32
      %596 = arith.select %594, %595, %593 : i32
      memref.store %596, %_anonymous6[%c1] : memref<3xi32>
      %597 = arith.addi %577, %c1 : index
      cf.br ^bb256(%597 : index)
    ^bb266:  // pred: ^bb256
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %598 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %599 = arith.addi %598, %c1_i32 : i32
      %600 = arith.cmpi sge, %599, %c1_i32 : i32
      %601 = arith.select %600, %598, %599 : i32
      memref.store %601, %_anonymous6[%c0] : memref<3xi32>
      %602 = arith.addi %0, %c1 : index
      cf.br ^bb1(%602 : index)
    ^bb267:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous7 = aie.buffer(%tile_7_2) {address = 32000 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb266
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb267
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
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
      func.call @matmul_i8_i16(%11, %7, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous7[%c1] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb19(%32: memref<64x128xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb23(%36: memref<32x64xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_i8_i16(%36, %32, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous7[%c1] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb30(%57: memref<64x128xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb34(%61: memref<32x64xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_i8_i16(%61, %57, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous7[%c1] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb41(%82: memref<64x128xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb45(%86: memref<32x64xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_i8_i16(%86, %82, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous7[%c1] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb47(%c0 : index)
    ^bb47(%102: index):  // 2 preds: ^bb46, ^bb56
      %103 = arith.cmpi slt, %102, %c16 : index
      cf.cond_br %103, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %104 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %105 = arith.index_cast %104 : i32 to index
      %106 = arith.index_cast %105 : index to i32
      cf.switch %106 : i32, [
        default: ^bb51,
        0: ^bb49,
        1: ^bb50
      ]
    ^bb49:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb50:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb51:  // pred: ^bb48
      cf.br ^bb52(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb52(%107: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %108 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %109 = arith.index_cast %108 : i32 to index
      %110 = arith.index_cast %109 : index to i32
      cf.switch %110 : i32, [
        default: ^bb55,
        0: ^bb53,
        1: ^bb54
      ]
    ^bb53:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb54:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb55:  // pred: ^bb52
      cf.br ^bb56(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb56(%111: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_i8_i16(%111, %107, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %112 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %113 = arith.addi %112, %c1_i32 : i32
      %114 = arith.cmpi sge, %113, %c2_i32 : i32
      %115 = arith.subi %113, %c2_i32 : i32
      %116 = arith.select %114, %115, %113 : i32
      memref.store %116, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %117 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %118 = arith.addi %117, %c1_i32 : i32
      %119 = arith.cmpi sge, %118, %c2_i32 : i32
      %120 = arith.subi %118, %c2_i32 : i32
      %121 = arith.select %119, %120, %118 : i32
      memref.store %121, %_anonymous7[%c1] : memref<3xi32>
      %122 = arith.addi %102, %c1 : index
      cf.br ^bb47(%122 : index)
    ^bb57:  // pred: ^bb47
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %123 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %124 = arith.addi %123, %c1_i32 : i32
      %125 = arith.cmpi sge, %124, %c1_i32 : i32
      %126 = arith.select %125, %123, %124 : i32
      memref.store %126, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb58(%c0 : index)
    ^bb58(%127: index):  // 2 preds: ^bb57, ^bb67
      %128 = arith.cmpi slt, %127, %c16 : index
      cf.cond_br %128, ^bb59, ^bb68
    ^bb59:  // pred: ^bb58
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %129 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %130 = arith.index_cast %129 : i32 to index
      %131 = arith.index_cast %130 : index to i32
      cf.switch %131 : i32, [
        default: ^bb62,
        0: ^bb60,
        1: ^bb61
      ]
    ^bb60:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb61:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb62:  // pred: ^bb59
      cf.br ^bb63(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb63(%132: memref<64x128xi8>):  // 3 preds: ^bb60, ^bb61, ^bb62
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %133 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %134 = arith.index_cast %133 : i32 to index
      %135 = arith.index_cast %134 : index to i32
      cf.switch %135 : i32, [
        default: ^bb66,
        0: ^bb64,
        1: ^bb65
      ]
    ^bb64:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb65:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb66:  // pred: ^bb63
      cf.br ^bb67(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb67(%136: memref<32x64xi8>):  // 3 preds: ^bb64, ^bb65, ^bb66
      func.call @matmul_i8_i16(%136, %132, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %137 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %138 = arith.addi %137, %c1_i32 : i32
      %139 = arith.cmpi sge, %138, %c2_i32 : i32
      %140 = arith.subi %138, %c2_i32 : i32
      %141 = arith.select %139, %140, %138 : i32
      memref.store %141, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %142 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %143 = arith.addi %142, %c1_i32 : i32
      %144 = arith.cmpi sge, %143, %c2_i32 : i32
      %145 = arith.subi %143, %c2_i32 : i32
      %146 = arith.select %144, %145, %143 : i32
      memref.store %146, %_anonymous7[%c1] : memref<3xi32>
      %147 = arith.addi %127, %c1 : index
      cf.br ^bb58(%147 : index)
    ^bb68:  // pred: ^bb58
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %148 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %149 = arith.addi %148, %c1_i32 : i32
      %150 = arith.cmpi sge, %149, %c1_i32 : i32
      %151 = arith.select %150, %148, %149 : i32
      memref.store %151, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb69(%c0 : index)
    ^bb69(%152: index):  // 2 preds: ^bb68, ^bb78
      %153 = arith.cmpi slt, %152, %c16 : index
      cf.cond_br %153, ^bb70, ^bb79
    ^bb70:  // pred: ^bb69
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %154 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %155 = arith.index_cast %154 : i32 to index
      %156 = arith.index_cast %155 : index to i32
      cf.switch %156 : i32, [
        default: ^bb73,
        0: ^bb71,
        1: ^bb72
      ]
    ^bb71:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb72:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb73:  // pred: ^bb70
      cf.br ^bb74(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb74(%157: memref<64x128xi8>):  // 3 preds: ^bb71, ^bb72, ^bb73
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %158 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %159 = arith.index_cast %158 : i32 to index
      %160 = arith.index_cast %159 : index to i32
      cf.switch %160 : i32, [
        default: ^bb77,
        0: ^bb75,
        1: ^bb76
      ]
    ^bb75:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb76:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb77:  // pred: ^bb74
      cf.br ^bb78(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb78(%161: memref<32x64xi8>):  // 3 preds: ^bb75, ^bb76, ^bb77
      func.call @matmul_i8_i16(%161, %157, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %162 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %163 = arith.addi %162, %c1_i32 : i32
      %164 = arith.cmpi sge, %163, %c2_i32 : i32
      %165 = arith.subi %163, %c2_i32 : i32
      %166 = arith.select %164, %165, %163 : i32
      memref.store %166, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %167 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %168 = arith.addi %167, %c1_i32 : i32
      %169 = arith.cmpi sge, %168, %c2_i32 : i32
      %170 = arith.subi %168, %c2_i32 : i32
      %171 = arith.select %169, %170, %168 : i32
      memref.store %171, %_anonymous7[%c1] : memref<3xi32>
      %172 = arith.addi %152, %c1 : index
      cf.br ^bb69(%172 : index)
    ^bb79:  // pred: ^bb69
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %173 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %174 = arith.addi %173, %c1_i32 : i32
      %175 = arith.cmpi sge, %174, %c1_i32 : i32
      %176 = arith.select %175, %173, %174 : i32
      memref.store %176, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb80(%c0 : index)
    ^bb80(%177: index):  // 2 preds: ^bb79, ^bb89
      %178 = arith.cmpi slt, %177, %c16 : index
      cf.cond_br %178, ^bb81, ^bb90
    ^bb81:  // pred: ^bb80
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %179 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %180 = arith.index_cast %179 : i32 to index
      %181 = arith.index_cast %180 : index to i32
      cf.switch %181 : i32, [
        default: ^bb84,
        0: ^bb82,
        1: ^bb83
      ]
    ^bb82:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb83:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb84:  // pred: ^bb81
      cf.br ^bb85(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb85(%182: memref<64x128xi8>):  // 3 preds: ^bb82, ^bb83, ^bb84
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %183 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %184 = arith.index_cast %183 : i32 to index
      %185 = arith.index_cast %184 : index to i32
      cf.switch %185 : i32, [
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
    ^bb89(%186: memref<32x64xi8>):  // 3 preds: ^bb86, ^bb87, ^bb88
      func.call @matmul_i8_i16(%186, %182, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %187 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %188 = arith.addi %187, %c1_i32 : i32
      %189 = arith.cmpi sge, %188, %c2_i32 : i32
      %190 = arith.subi %188, %c2_i32 : i32
      %191 = arith.select %189, %190, %188 : i32
      memref.store %191, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %192 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %193 = arith.addi %192, %c1_i32 : i32
      %194 = arith.cmpi sge, %193, %c2_i32 : i32
      %195 = arith.subi %193, %c2_i32 : i32
      %196 = arith.select %194, %195, %193 : i32
      memref.store %196, %_anonymous7[%c1] : memref<3xi32>
      %197 = arith.addi %177, %c1 : index
      cf.br ^bb80(%197 : index)
    ^bb90:  // pred: ^bb80
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %198 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %199 = arith.addi %198, %c1_i32 : i32
      %200 = arith.cmpi sge, %199, %c1_i32 : i32
      %201 = arith.select %200, %198, %199 : i32
      memref.store %201, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb91(%c0 : index)
    ^bb91(%202: index):  // 2 preds: ^bb90, ^bb100
      %203 = arith.cmpi slt, %202, %c16 : index
      cf.cond_br %203, ^bb92, ^bb101
    ^bb92:  // pred: ^bb91
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %204 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %205 = arith.index_cast %204 : i32 to index
      %206 = arith.index_cast %205 : index to i32
      cf.switch %206 : i32, [
        default: ^bb95,
        0: ^bb93,
        1: ^bb94
      ]
    ^bb93:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb94:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb95:  // pred: ^bb92
      cf.br ^bb96(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb96(%207: memref<64x128xi8>):  // 3 preds: ^bb93, ^bb94, ^bb95
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %208 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %209 = arith.index_cast %208 : i32 to index
      %210 = arith.index_cast %209 : index to i32
      cf.switch %210 : i32, [
        default: ^bb99,
        0: ^bb97,
        1: ^bb98
      ]
    ^bb97:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb98:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb99:  // pred: ^bb96
      cf.br ^bb100(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb100(%211: memref<32x64xi8>):  // 3 preds: ^bb97, ^bb98, ^bb99
      func.call @matmul_i8_i16(%211, %207, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %212 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %213 = arith.addi %212, %c1_i32 : i32
      %214 = arith.cmpi sge, %213, %c2_i32 : i32
      %215 = arith.subi %213, %c2_i32 : i32
      %216 = arith.select %214, %215, %213 : i32
      memref.store %216, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %217 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %218 = arith.addi %217, %c1_i32 : i32
      %219 = arith.cmpi sge, %218, %c2_i32 : i32
      %220 = arith.subi %218, %c2_i32 : i32
      %221 = arith.select %219, %220, %218 : i32
      memref.store %221, %_anonymous7[%c1] : memref<3xi32>
      %222 = arith.addi %202, %c1 : index
      cf.br ^bb91(%222 : index)
    ^bb101:  // pred: ^bb91
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %223 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %224 = arith.addi %223, %c1_i32 : i32
      %225 = arith.cmpi sge, %224, %c1_i32 : i32
      %226 = arith.select %225, %223, %224 : i32
      memref.store %226, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb102(%c0 : index)
    ^bb102(%227: index):  // 2 preds: ^bb101, ^bb111
      %228 = arith.cmpi slt, %227, %c16 : index
      cf.cond_br %228, ^bb103, ^bb112
    ^bb103:  // pred: ^bb102
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %229 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %230 = arith.index_cast %229 : i32 to index
      %231 = arith.index_cast %230 : index to i32
      cf.switch %231 : i32, [
        default: ^bb106,
        0: ^bb104,
        1: ^bb105
      ]
    ^bb104:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb105:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb106:  // pred: ^bb103
      cf.br ^bb107(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb107(%232: memref<64x128xi8>):  // 3 preds: ^bb104, ^bb105, ^bb106
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %233 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %234 = arith.index_cast %233 : i32 to index
      %235 = arith.index_cast %234 : index to i32
      cf.switch %235 : i32, [
        default: ^bb110,
        0: ^bb108,
        1: ^bb109
      ]
    ^bb108:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb109:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb110:  // pred: ^bb107
      cf.br ^bb111(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb111(%236: memref<32x64xi8>):  // 3 preds: ^bb108, ^bb109, ^bb110
      func.call @matmul_i8_i16(%236, %232, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %237 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %238 = arith.addi %237, %c1_i32 : i32
      %239 = arith.cmpi sge, %238, %c2_i32 : i32
      %240 = arith.subi %238, %c2_i32 : i32
      %241 = arith.select %239, %240, %238 : i32
      memref.store %241, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %242 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %243 = arith.addi %242, %c1_i32 : i32
      %244 = arith.cmpi sge, %243, %c2_i32 : i32
      %245 = arith.subi %243, %c2_i32 : i32
      %246 = arith.select %244, %245, %243 : i32
      memref.store %246, %_anonymous7[%c1] : memref<3xi32>
      %247 = arith.addi %227, %c1 : index
      cf.br ^bb102(%247 : index)
    ^bb112:  // pred: ^bb102
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %248 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %249 = arith.addi %248, %c1_i32 : i32
      %250 = arith.cmpi sge, %249, %c1_i32 : i32
      %251 = arith.select %250, %248, %249 : i32
      memref.store %251, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb113(%c0 : index)
    ^bb113(%252: index):  // 2 preds: ^bb112, ^bb122
      %253 = arith.cmpi slt, %252, %c16 : index
      cf.cond_br %253, ^bb114, ^bb123
    ^bb114:  // pred: ^bb113
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %254 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %255 = arith.index_cast %254 : i32 to index
      %256 = arith.index_cast %255 : index to i32
      cf.switch %256 : i32, [
        default: ^bb117,
        0: ^bb115,
        1: ^bb116
      ]
    ^bb115:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb116:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb117:  // pred: ^bb114
      cf.br ^bb118(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb118(%257: memref<64x128xi8>):  // 3 preds: ^bb115, ^bb116, ^bb117
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %258 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %259 = arith.index_cast %258 : i32 to index
      %260 = arith.index_cast %259 : index to i32
      cf.switch %260 : i32, [
        default: ^bb121,
        0: ^bb119,
        1: ^bb120
      ]
    ^bb119:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb120:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb121:  // pred: ^bb118
      cf.br ^bb122(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb122(%261: memref<32x64xi8>):  // 3 preds: ^bb119, ^bb120, ^bb121
      func.call @matmul_i8_i16(%261, %257, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %262 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %263 = arith.addi %262, %c1_i32 : i32
      %264 = arith.cmpi sge, %263, %c2_i32 : i32
      %265 = arith.subi %263, %c2_i32 : i32
      %266 = arith.select %264, %265, %263 : i32
      memref.store %266, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %267 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %268 = arith.addi %267, %c1_i32 : i32
      %269 = arith.cmpi sge, %268, %c2_i32 : i32
      %270 = arith.subi %268, %c2_i32 : i32
      %271 = arith.select %269, %270, %268 : i32
      memref.store %271, %_anonymous7[%c1] : memref<3xi32>
      %272 = arith.addi %252, %c1 : index
      cf.br ^bb113(%272 : index)
    ^bb123:  // pred: ^bb113
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %273 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %274 = arith.addi %273, %c1_i32 : i32
      %275 = arith.cmpi sge, %274, %c1_i32 : i32
      %276 = arith.select %275, %273, %274 : i32
      memref.store %276, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb124(%c0 : index)
    ^bb124(%277: index):  // 2 preds: ^bb123, ^bb133
      %278 = arith.cmpi slt, %277, %c16 : index
      cf.cond_br %278, ^bb125, ^bb134
    ^bb125:  // pred: ^bb124
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %279 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %280 = arith.index_cast %279 : i32 to index
      %281 = arith.index_cast %280 : index to i32
      cf.switch %281 : i32, [
        default: ^bb128,
        0: ^bb126,
        1: ^bb127
      ]
    ^bb126:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb127:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb128:  // pred: ^bb125
      cf.br ^bb129(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb129(%282: memref<64x128xi8>):  // 3 preds: ^bb126, ^bb127, ^bb128
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %283 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %284 = arith.index_cast %283 : i32 to index
      %285 = arith.index_cast %284 : index to i32
      cf.switch %285 : i32, [
        default: ^bb132,
        0: ^bb130,
        1: ^bb131
      ]
    ^bb130:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb131:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb132:  // pred: ^bb129
      cf.br ^bb133(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb133(%286: memref<32x64xi8>):  // 3 preds: ^bb130, ^bb131, ^bb132
      func.call @matmul_i8_i16(%286, %282, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %287 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %288 = arith.addi %287, %c1_i32 : i32
      %289 = arith.cmpi sge, %288, %c2_i32 : i32
      %290 = arith.subi %288, %c2_i32 : i32
      %291 = arith.select %289, %290, %288 : i32
      memref.store %291, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %292 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %293 = arith.addi %292, %c1_i32 : i32
      %294 = arith.cmpi sge, %293, %c2_i32 : i32
      %295 = arith.subi %293, %c2_i32 : i32
      %296 = arith.select %294, %295, %293 : i32
      memref.store %296, %_anonymous7[%c1] : memref<3xi32>
      %297 = arith.addi %277, %c1 : index
      cf.br ^bb124(%297 : index)
    ^bb134:  // pred: ^bb124
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %298 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %299 = arith.addi %298, %c1_i32 : i32
      %300 = arith.cmpi sge, %299, %c1_i32 : i32
      %301 = arith.select %300, %298, %299 : i32
      memref.store %301, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb135(%c0 : index)
    ^bb135(%302: index):  // 2 preds: ^bb134, ^bb144
      %303 = arith.cmpi slt, %302, %c16 : index
      cf.cond_br %303, ^bb136, ^bb145
    ^bb136:  // pred: ^bb135
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %304 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %305 = arith.index_cast %304 : i32 to index
      %306 = arith.index_cast %305 : index to i32
      cf.switch %306 : i32, [
        default: ^bb139,
        0: ^bb137,
        1: ^bb138
      ]
    ^bb137:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb138:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb139:  // pred: ^bb136
      cf.br ^bb140(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb140(%307: memref<64x128xi8>):  // 3 preds: ^bb137, ^bb138, ^bb139
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %308 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %309 = arith.index_cast %308 : i32 to index
      %310 = arith.index_cast %309 : index to i32
      cf.switch %310 : i32, [
        default: ^bb143,
        0: ^bb141,
        1: ^bb142
      ]
    ^bb141:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb142:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb143:  // pred: ^bb140
      cf.br ^bb144(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb144(%311: memref<32x64xi8>):  // 3 preds: ^bb141, ^bb142, ^bb143
      func.call @matmul_i8_i16(%311, %307, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %312 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %313 = arith.addi %312, %c1_i32 : i32
      %314 = arith.cmpi sge, %313, %c2_i32 : i32
      %315 = arith.subi %313, %c2_i32 : i32
      %316 = arith.select %314, %315, %313 : i32
      memref.store %316, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %317 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %318 = arith.addi %317, %c1_i32 : i32
      %319 = arith.cmpi sge, %318, %c2_i32 : i32
      %320 = arith.subi %318, %c2_i32 : i32
      %321 = arith.select %319, %320, %318 : i32
      memref.store %321, %_anonymous7[%c1] : memref<3xi32>
      %322 = arith.addi %302, %c1 : index
      cf.br ^bb135(%322 : index)
    ^bb145:  // pred: ^bb135
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %323 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %324 = arith.addi %323, %c1_i32 : i32
      %325 = arith.cmpi sge, %324, %c1_i32 : i32
      %326 = arith.select %325, %323, %324 : i32
      memref.store %326, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb146(%c0 : index)
    ^bb146(%327: index):  // 2 preds: ^bb145, ^bb155
      %328 = arith.cmpi slt, %327, %c16 : index
      cf.cond_br %328, ^bb147, ^bb156
    ^bb147:  // pred: ^bb146
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %329 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %330 = arith.index_cast %329 : i32 to index
      %331 = arith.index_cast %330 : index to i32
      cf.switch %331 : i32, [
        default: ^bb150,
        0: ^bb148,
        1: ^bb149
      ]
    ^bb148:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb149:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb150:  // pred: ^bb147
      cf.br ^bb151(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb151(%332: memref<64x128xi8>):  // 3 preds: ^bb148, ^bb149, ^bb150
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %333 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %334 = arith.index_cast %333 : i32 to index
      %335 = arith.index_cast %334 : index to i32
      cf.switch %335 : i32, [
        default: ^bb154,
        0: ^bb152,
        1: ^bb153
      ]
    ^bb152:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb153:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb154:  // pred: ^bb151
      cf.br ^bb155(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb155(%336: memref<32x64xi8>):  // 3 preds: ^bb152, ^bb153, ^bb154
      func.call @matmul_i8_i16(%336, %332, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %337 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %338 = arith.addi %337, %c1_i32 : i32
      %339 = arith.cmpi sge, %338, %c2_i32 : i32
      %340 = arith.subi %338, %c2_i32 : i32
      %341 = arith.select %339, %340, %338 : i32
      memref.store %341, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %342 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %343 = arith.addi %342, %c1_i32 : i32
      %344 = arith.cmpi sge, %343, %c2_i32 : i32
      %345 = arith.subi %343, %c2_i32 : i32
      %346 = arith.select %344, %345, %343 : i32
      memref.store %346, %_anonymous7[%c1] : memref<3xi32>
      %347 = arith.addi %327, %c1 : index
      cf.br ^bb146(%347 : index)
    ^bb156:  // pred: ^bb146
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %348 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %349 = arith.addi %348, %c1_i32 : i32
      %350 = arith.cmpi sge, %349, %c1_i32 : i32
      %351 = arith.select %350, %348, %349 : i32
      memref.store %351, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb157(%c0 : index)
    ^bb157(%352: index):  // 2 preds: ^bb156, ^bb166
      %353 = arith.cmpi slt, %352, %c16 : index
      cf.cond_br %353, ^bb158, ^bb167
    ^bb158:  // pred: ^bb157
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %354 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %355 = arith.index_cast %354 : i32 to index
      %356 = arith.index_cast %355 : index to i32
      cf.switch %356 : i32, [
        default: ^bb161,
        0: ^bb159,
        1: ^bb160
      ]
    ^bb159:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb160:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb161:  // pred: ^bb158
      cf.br ^bb162(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb162(%357: memref<64x128xi8>):  // 3 preds: ^bb159, ^bb160, ^bb161
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %358 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %359 = arith.index_cast %358 : i32 to index
      %360 = arith.index_cast %359 : index to i32
      cf.switch %360 : i32, [
        default: ^bb165,
        0: ^bb163,
        1: ^bb164
      ]
    ^bb163:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb164:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb165:  // pred: ^bb162
      cf.br ^bb166(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb166(%361: memref<32x64xi8>):  // 3 preds: ^bb163, ^bb164, ^bb165
      func.call @matmul_i8_i16(%361, %357, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %362 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %363 = arith.addi %362, %c1_i32 : i32
      %364 = arith.cmpi sge, %363, %c2_i32 : i32
      %365 = arith.subi %363, %c2_i32 : i32
      %366 = arith.select %364, %365, %363 : i32
      memref.store %366, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %367 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %368 = arith.addi %367, %c1_i32 : i32
      %369 = arith.cmpi sge, %368, %c2_i32 : i32
      %370 = arith.subi %368, %c2_i32 : i32
      %371 = arith.select %369, %370, %368 : i32
      memref.store %371, %_anonymous7[%c1] : memref<3xi32>
      %372 = arith.addi %352, %c1 : index
      cf.br ^bb157(%372 : index)
    ^bb167:  // pred: ^bb157
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %373 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %374 = arith.addi %373, %c1_i32 : i32
      %375 = arith.cmpi sge, %374, %c1_i32 : i32
      %376 = arith.select %375, %373, %374 : i32
      memref.store %376, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb168(%c0 : index)
    ^bb168(%377: index):  // 2 preds: ^bb167, ^bb177
      %378 = arith.cmpi slt, %377, %c16 : index
      cf.cond_br %378, ^bb169, ^bb178
    ^bb169:  // pred: ^bb168
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %379 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %380 = arith.index_cast %379 : i32 to index
      %381 = arith.index_cast %380 : index to i32
      cf.switch %381 : i32, [
        default: ^bb172,
        0: ^bb170,
        1: ^bb171
      ]
    ^bb170:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb171:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb172:  // pred: ^bb169
      cf.br ^bb173(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb173(%382: memref<64x128xi8>):  // 3 preds: ^bb170, ^bb171, ^bb172
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %383 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %384 = arith.index_cast %383 : i32 to index
      %385 = arith.index_cast %384 : index to i32
      cf.switch %385 : i32, [
        default: ^bb176,
        0: ^bb174,
        1: ^bb175
      ]
    ^bb174:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb175:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb176:  // pred: ^bb173
      cf.br ^bb177(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb177(%386: memref<32x64xi8>):  // 3 preds: ^bb174, ^bb175, ^bb176
      func.call @matmul_i8_i16(%386, %382, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %387 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %388 = arith.addi %387, %c1_i32 : i32
      %389 = arith.cmpi sge, %388, %c2_i32 : i32
      %390 = arith.subi %388, %c2_i32 : i32
      %391 = arith.select %389, %390, %388 : i32
      memref.store %391, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %392 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %393 = arith.addi %392, %c1_i32 : i32
      %394 = arith.cmpi sge, %393, %c2_i32 : i32
      %395 = arith.subi %393, %c2_i32 : i32
      %396 = arith.select %394, %395, %393 : i32
      memref.store %396, %_anonymous7[%c1] : memref<3xi32>
      %397 = arith.addi %377, %c1 : index
      cf.br ^bb168(%397 : index)
    ^bb178:  // pred: ^bb168
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %398 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %399 = arith.addi %398, %c1_i32 : i32
      %400 = arith.cmpi sge, %399, %c1_i32 : i32
      %401 = arith.select %400, %398, %399 : i32
      memref.store %401, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb179(%c0 : index)
    ^bb179(%402: index):  // 2 preds: ^bb178, ^bb188
      %403 = arith.cmpi slt, %402, %c16 : index
      cf.cond_br %403, ^bb180, ^bb189
    ^bb180:  // pred: ^bb179
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %404 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %405 = arith.index_cast %404 : i32 to index
      %406 = arith.index_cast %405 : index to i32
      cf.switch %406 : i32, [
        default: ^bb183,
        0: ^bb181,
        1: ^bb182
      ]
    ^bb181:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb182:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb183:  // pred: ^bb180
      cf.br ^bb184(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb184(%407: memref<64x128xi8>):  // 3 preds: ^bb181, ^bb182, ^bb183
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %408 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %409 = arith.index_cast %408 : i32 to index
      %410 = arith.index_cast %409 : index to i32
      cf.switch %410 : i32, [
        default: ^bb187,
        0: ^bb185,
        1: ^bb186
      ]
    ^bb185:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb186:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb187:  // pred: ^bb184
      cf.br ^bb188(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb188(%411: memref<32x64xi8>):  // 3 preds: ^bb185, ^bb186, ^bb187
      func.call @matmul_i8_i16(%411, %407, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %412 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %413 = arith.addi %412, %c1_i32 : i32
      %414 = arith.cmpi sge, %413, %c2_i32 : i32
      %415 = arith.subi %413, %c2_i32 : i32
      %416 = arith.select %414, %415, %413 : i32
      memref.store %416, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %417 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %418 = arith.addi %417, %c1_i32 : i32
      %419 = arith.cmpi sge, %418, %c2_i32 : i32
      %420 = arith.subi %418, %c2_i32 : i32
      %421 = arith.select %419, %420, %418 : i32
      memref.store %421, %_anonymous7[%c1] : memref<3xi32>
      %422 = arith.addi %402, %c1 : index
      cf.br ^bb179(%422 : index)
    ^bb189:  // pred: ^bb179
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %423 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %424 = arith.addi %423, %c1_i32 : i32
      %425 = arith.cmpi sge, %424, %c1_i32 : i32
      %426 = arith.select %425, %423, %424 : i32
      memref.store %426, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb190(%c0 : index)
    ^bb190(%427: index):  // 2 preds: ^bb189, ^bb199
      %428 = arith.cmpi slt, %427, %c16 : index
      cf.cond_br %428, ^bb191, ^bb200
    ^bb191:  // pred: ^bb190
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %429 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %430 = arith.index_cast %429 : i32 to index
      %431 = arith.index_cast %430 : index to i32
      cf.switch %431 : i32, [
        default: ^bb194,
        0: ^bb192,
        1: ^bb193
      ]
    ^bb192:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb193:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb194:  // pred: ^bb191
      cf.br ^bb195(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb195(%432: memref<64x128xi8>):  // 3 preds: ^bb192, ^bb193, ^bb194
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %433 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %434 = arith.index_cast %433 : i32 to index
      %435 = arith.index_cast %434 : index to i32
      cf.switch %435 : i32, [
        default: ^bb198,
        0: ^bb196,
        1: ^bb197
      ]
    ^bb196:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb197:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb198:  // pred: ^bb195
      cf.br ^bb199(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb199(%436: memref<32x64xi8>):  // 3 preds: ^bb196, ^bb197, ^bb198
      func.call @matmul_i8_i16(%436, %432, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %437 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %438 = arith.addi %437, %c1_i32 : i32
      %439 = arith.cmpi sge, %438, %c2_i32 : i32
      %440 = arith.subi %438, %c2_i32 : i32
      %441 = arith.select %439, %440, %438 : i32
      memref.store %441, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %442 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %443 = arith.addi %442, %c1_i32 : i32
      %444 = arith.cmpi sge, %443, %c2_i32 : i32
      %445 = arith.subi %443, %c2_i32 : i32
      %446 = arith.select %444, %445, %443 : i32
      memref.store %446, %_anonymous7[%c1] : memref<3xi32>
      %447 = arith.addi %427, %c1 : index
      cf.br ^bb190(%447 : index)
    ^bb200:  // pred: ^bb190
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %448 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %449 = arith.addi %448, %c1_i32 : i32
      %450 = arith.cmpi sge, %449, %c1_i32 : i32
      %451 = arith.select %450, %448, %449 : i32
      memref.store %451, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb201(%c0 : index)
    ^bb201(%452: index):  // 2 preds: ^bb200, ^bb210
      %453 = arith.cmpi slt, %452, %c16 : index
      cf.cond_br %453, ^bb202, ^bb211
    ^bb202:  // pred: ^bb201
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %454 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %455 = arith.index_cast %454 : i32 to index
      %456 = arith.index_cast %455 : index to i32
      cf.switch %456 : i32, [
        default: ^bb205,
        0: ^bb203,
        1: ^bb204
      ]
    ^bb203:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb204:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb205:  // pred: ^bb202
      cf.br ^bb206(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb206(%457: memref<64x128xi8>):  // 3 preds: ^bb203, ^bb204, ^bb205
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %458 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %459 = arith.index_cast %458 : i32 to index
      %460 = arith.index_cast %459 : index to i32
      cf.switch %460 : i32, [
        default: ^bb209,
        0: ^bb207,
        1: ^bb208
      ]
    ^bb207:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb208:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb209:  // pred: ^bb206
      cf.br ^bb210(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb210(%461: memref<32x64xi8>):  // 3 preds: ^bb207, ^bb208, ^bb209
      func.call @matmul_i8_i16(%461, %457, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %462 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %463 = arith.addi %462, %c1_i32 : i32
      %464 = arith.cmpi sge, %463, %c2_i32 : i32
      %465 = arith.subi %463, %c2_i32 : i32
      %466 = arith.select %464, %465, %463 : i32
      memref.store %466, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %467 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %468 = arith.addi %467, %c1_i32 : i32
      %469 = arith.cmpi sge, %468, %c2_i32 : i32
      %470 = arith.subi %468, %c2_i32 : i32
      %471 = arith.select %469, %470, %468 : i32
      memref.store %471, %_anonymous7[%c1] : memref<3xi32>
      %472 = arith.addi %452, %c1 : index
      cf.br ^bb201(%472 : index)
    ^bb211:  // pred: ^bb201
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %473 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %474 = arith.addi %473, %c1_i32 : i32
      %475 = arith.cmpi sge, %474, %c1_i32 : i32
      %476 = arith.select %475, %473, %474 : i32
      memref.store %476, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb212(%c0 : index)
    ^bb212(%477: index):  // 2 preds: ^bb211, ^bb221
      %478 = arith.cmpi slt, %477, %c16 : index
      cf.cond_br %478, ^bb213, ^bb222
    ^bb213:  // pred: ^bb212
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %479 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %480 = arith.index_cast %479 : i32 to index
      %481 = arith.index_cast %480 : index to i32
      cf.switch %481 : i32, [
        default: ^bb216,
        0: ^bb214,
        1: ^bb215
      ]
    ^bb214:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb215:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb216:  // pred: ^bb213
      cf.br ^bb217(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb217(%482: memref<64x128xi8>):  // 3 preds: ^bb214, ^bb215, ^bb216
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %483 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %484 = arith.index_cast %483 : i32 to index
      %485 = arith.index_cast %484 : index to i32
      cf.switch %485 : i32, [
        default: ^bb220,
        0: ^bb218,
        1: ^bb219
      ]
    ^bb218:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb219:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb220:  // pred: ^bb217
      cf.br ^bb221(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb221(%486: memref<32x64xi8>):  // 3 preds: ^bb218, ^bb219, ^bb220
      func.call @matmul_i8_i16(%486, %482, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %487 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %488 = arith.addi %487, %c1_i32 : i32
      %489 = arith.cmpi sge, %488, %c2_i32 : i32
      %490 = arith.subi %488, %c2_i32 : i32
      %491 = arith.select %489, %490, %488 : i32
      memref.store %491, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %492 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %493 = arith.addi %492, %c1_i32 : i32
      %494 = arith.cmpi sge, %493, %c2_i32 : i32
      %495 = arith.subi %493, %c2_i32 : i32
      %496 = arith.select %494, %495, %493 : i32
      memref.store %496, %_anonymous7[%c1] : memref<3xi32>
      %497 = arith.addi %477, %c1 : index
      cf.br ^bb212(%497 : index)
    ^bb222:  // pred: ^bb212
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %498 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %499 = arith.addi %498, %c1_i32 : i32
      %500 = arith.cmpi sge, %499, %c1_i32 : i32
      %501 = arith.select %500, %498, %499 : i32
      memref.store %501, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb223(%c0 : index)
    ^bb223(%502: index):  // 2 preds: ^bb222, ^bb232
      %503 = arith.cmpi slt, %502, %c16 : index
      cf.cond_br %503, ^bb224, ^bb233
    ^bb224:  // pred: ^bb223
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %504 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %505 = arith.index_cast %504 : i32 to index
      %506 = arith.index_cast %505 : index to i32
      cf.switch %506 : i32, [
        default: ^bb227,
        0: ^bb225,
        1: ^bb226
      ]
    ^bb225:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb226:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb227:  // pred: ^bb224
      cf.br ^bb228(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb228(%507: memref<64x128xi8>):  // 3 preds: ^bb225, ^bb226, ^bb227
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %508 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %509 = arith.index_cast %508 : i32 to index
      %510 = arith.index_cast %509 : index to i32
      cf.switch %510 : i32, [
        default: ^bb231,
        0: ^bb229,
        1: ^bb230
      ]
    ^bb229:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb230:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb231:  // pred: ^bb228
      cf.br ^bb232(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb232(%511: memref<32x64xi8>):  // 3 preds: ^bb229, ^bb230, ^bb231
      func.call @matmul_i8_i16(%511, %507, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %512 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %513 = arith.addi %512, %c1_i32 : i32
      %514 = arith.cmpi sge, %513, %c2_i32 : i32
      %515 = arith.subi %513, %c2_i32 : i32
      %516 = arith.select %514, %515, %513 : i32
      memref.store %516, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %517 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %518 = arith.addi %517, %c1_i32 : i32
      %519 = arith.cmpi sge, %518, %c2_i32 : i32
      %520 = arith.subi %518, %c2_i32 : i32
      %521 = arith.select %519, %520, %518 : i32
      memref.store %521, %_anonymous7[%c1] : memref<3xi32>
      %522 = arith.addi %502, %c1 : index
      cf.br ^bb223(%522 : index)
    ^bb233:  // pred: ^bb223
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %523 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %524 = arith.addi %523, %c1_i32 : i32
      %525 = arith.cmpi sge, %524, %c1_i32 : i32
      %526 = arith.select %525, %523, %524 : i32
      memref.store %526, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb234(%c0 : index)
    ^bb234(%527: index):  // 2 preds: ^bb233, ^bb243
      %528 = arith.cmpi slt, %527, %c16 : index
      cf.cond_br %528, ^bb235, ^bb244
    ^bb235:  // pred: ^bb234
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %529 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %530 = arith.index_cast %529 : i32 to index
      %531 = arith.index_cast %530 : index to i32
      cf.switch %531 : i32, [
        default: ^bb238,
        0: ^bb236,
        1: ^bb237
      ]
    ^bb236:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb237:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb238:  // pred: ^bb235
      cf.br ^bb239(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb239(%532: memref<64x128xi8>):  // 3 preds: ^bb236, ^bb237, ^bb238
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %533 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %534 = arith.index_cast %533 : i32 to index
      %535 = arith.index_cast %534 : index to i32
      cf.switch %535 : i32, [
        default: ^bb242,
        0: ^bb240,
        1: ^bb241
      ]
    ^bb240:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb241:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb242:  // pred: ^bb239
      cf.br ^bb243(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb243(%536: memref<32x64xi8>):  // 3 preds: ^bb240, ^bb241, ^bb242
      func.call @matmul_i8_i16(%536, %532, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %537 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %538 = arith.addi %537, %c1_i32 : i32
      %539 = arith.cmpi sge, %538, %c2_i32 : i32
      %540 = arith.subi %538, %c2_i32 : i32
      %541 = arith.select %539, %540, %538 : i32
      memref.store %541, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %542 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %543 = arith.addi %542, %c1_i32 : i32
      %544 = arith.cmpi sge, %543, %c2_i32 : i32
      %545 = arith.subi %543, %c2_i32 : i32
      %546 = arith.select %544, %545, %543 : i32
      memref.store %546, %_anonymous7[%c1] : memref<3xi32>
      %547 = arith.addi %527, %c1 : index
      cf.br ^bb234(%547 : index)
    ^bb244:  // pred: ^bb234
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %548 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %549 = arith.addi %548, %c1_i32 : i32
      %550 = arith.cmpi sge, %549, %c1_i32 : i32
      %551 = arith.select %550, %548, %549 : i32
      memref.store %551, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb245(%c0 : index)
    ^bb245(%552: index):  // 2 preds: ^bb244, ^bb254
      %553 = arith.cmpi slt, %552, %c16 : index
      cf.cond_br %553, ^bb246, ^bb255
    ^bb246:  // pred: ^bb245
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %554 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %555 = arith.index_cast %554 : i32 to index
      %556 = arith.index_cast %555 : index to i32
      cf.switch %556 : i32, [
        default: ^bb249,
        0: ^bb247,
        1: ^bb248
      ]
    ^bb247:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb248:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb249:  // pred: ^bb246
      cf.br ^bb250(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb250(%557: memref<64x128xi8>):  // 3 preds: ^bb247, ^bb248, ^bb249
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %558 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %559 = arith.index_cast %558 : i32 to index
      %560 = arith.index_cast %559 : index to i32
      cf.switch %560 : i32, [
        default: ^bb253,
        0: ^bb251,
        1: ^bb252
      ]
    ^bb251:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb252:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb253:  // pred: ^bb250
      cf.br ^bb254(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb254(%561: memref<32x64xi8>):  // 3 preds: ^bb251, ^bb252, ^bb253
      func.call @matmul_i8_i16(%561, %557, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %562 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %563 = arith.addi %562, %c1_i32 : i32
      %564 = arith.cmpi sge, %563, %c2_i32 : i32
      %565 = arith.subi %563, %c2_i32 : i32
      %566 = arith.select %564, %565, %563 : i32
      memref.store %566, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %567 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %568 = arith.addi %567, %c1_i32 : i32
      %569 = arith.cmpi sge, %568, %c2_i32 : i32
      %570 = arith.subi %568, %c2_i32 : i32
      %571 = arith.select %569, %570, %568 : i32
      memref.store %571, %_anonymous7[%c1] : memref<3xi32>
      %572 = arith.addi %552, %c1 : index
      cf.br ^bb245(%572 : index)
    ^bb255:  // pred: ^bb245
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %573 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %574 = arith.addi %573, %c1_i32 : i32
      %575 = arith.cmpi sge, %574, %c1_i32 : i32
      %576 = arith.select %575, %573, %574 : i32
      memref.store %576, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb256(%c0 : index)
    ^bb256(%577: index):  // 2 preds: ^bb255, ^bb265
      %578 = arith.cmpi slt, %577, %c16 : index
      cf.cond_br %578, ^bb257, ^bb266
    ^bb257:  // pred: ^bb256
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %579 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %580 = arith.index_cast %579 : i32 to index
      %581 = arith.index_cast %580 : index to i32
      cf.switch %581 : i32, [
        default: ^bb260,
        0: ^bb258,
        1: ^bb259
      ]
    ^bb258:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb259:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb260:  // pred: ^bb257
      cf.br ^bb261(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb261(%582: memref<64x128xi8>):  // 3 preds: ^bb258, ^bb259, ^bb260
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %583 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %584 = arith.index_cast %583 : i32 to index
      %585 = arith.index_cast %584 : index to i32
      cf.switch %585 : i32, [
        default: ^bb264,
        0: ^bb262,
        1: ^bb263
      ]
    ^bb262:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb263:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb264:  // pred: ^bb261
      cf.br ^bb265(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb265(%586: memref<32x64xi8>):  // 3 preds: ^bb262, ^bb263, ^bb264
      func.call @matmul_i8_i16(%586, %582, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %587 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %588 = arith.addi %587, %c1_i32 : i32
      %589 = arith.cmpi sge, %588, %c2_i32 : i32
      %590 = arith.subi %588, %c2_i32 : i32
      %591 = arith.select %589, %590, %588 : i32
      memref.store %591, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %592 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %593 = arith.addi %592, %c1_i32 : i32
      %594 = arith.cmpi sge, %593, %c2_i32 : i32
      %595 = arith.subi %593, %c2_i32 : i32
      %596 = arith.select %594, %595, %593 : i32
      memref.store %596, %_anonymous7[%c1] : memref<3xi32>
      %597 = arith.addi %577, %c1 : index
      cf.br ^bb256(%597 : index)
    ^bb266:  // pred: ^bb256
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %598 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %599 = arith.addi %598, %c1_i32 : i32
      %600 = arith.cmpi sge, %599, %c1_i32 : i32
      %601 = arith.select %600, %598, %599 : i32
      memref.store %601, %_anonymous7[%c0] : memref<3xi32>
      %602 = arith.addi %0, %c1 : index
      cf.br ^bb1(%602 : index)
    ^bb267:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<131072xi8>, %arg1: memref<6291456xi8>, %arg2: memref<786432xi16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 128, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 256, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 384, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 512, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 640, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
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
      %17 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1048576, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1179648, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1310720, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1441792, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1572864, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1703936, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1835008, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1966080, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1024, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1152, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1280, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1408, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1792, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1920, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
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
      %34 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2097152, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2228224, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2359296, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2490368, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2621440, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2752512, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2883584, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3014656, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2048, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2176, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2560, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2688, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2816, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2944, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
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
      %51 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3145728, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3276800, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3407872, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3538944, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3670016, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3801088, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3932160, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4063232, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3072, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3200, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3328, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3456, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3584, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3712, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3840, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3968, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%67)
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
      %68 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4194304, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4325376, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4456448, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4587520, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4718592, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4849664, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4980736, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5111808, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4096, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4224, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4352, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4480, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4608, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4736, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4864, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4992, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%84)
      aiex.dma_await_task(%77)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%80)
      aiex.dma_await_task(%81)
      aiex.dma_await_task(%82)
      aiex.dma_await_task(%83)
      aiex.dma_await_task(%84)
      aiex.dma_free_task(%68)
      aiex.dma_free_task(%69)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%71)
      aiex.dma_free_task(%72)
      aiex.dma_free_task(%73)
      aiex.dma_free_task(%74)
      aiex.dma_free_task(%75)
      aiex.dma_free_task(%76)
      %85 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5242880, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5373952, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5505024, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5636096, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5767168, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5898240, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6029312, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6160384, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5120, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5248, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5376, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5504, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5632, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5760, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5888, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 6016, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%101)
      aiex.dma_await_task(%94)
      aiex.dma_await_task(%95)
      aiex.dma_await_task(%96)
      aiex.dma_await_task(%97)
      aiex.dma_await_task(%98)
      aiex.dma_await_task(%99)
      aiex.dma_await_task(%100)
      aiex.dma_await_task(%101)
      aiex.dma_free_task(%85)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%87)
      aiex.dma_free_task(%88)
      aiex.dma_free_task(%89)
      aiex.dma_free_task(%90)
      aiex.dma_free_task(%91)
      aiex.dma_free_task(%92)
      aiex.dma_free_task(%93)
      %102 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 196608, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 196736, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 196864, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 196992, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197120, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197248, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197376, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197504, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%118)
      aiex.dma_await_task(%111)
      aiex.dma_await_task(%112)
      aiex.dma_await_task(%113)
      aiex.dma_await_task(%114)
      aiex.dma_await_task(%115)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%117)
      aiex.dma_await_task(%118)
      aiex.dma_free_task(%102)
      aiex.dma_free_task(%103)
      aiex.dma_free_task(%104)
      aiex.dma_free_task(%105)
      aiex.dma_free_task(%106)
      aiex.dma_free_task(%107)
      aiex.dma_free_task(%108)
      aiex.dma_free_task(%109)
      aiex.dma_free_task(%110)
      %119 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1048576, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1179648, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1310720, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1441792, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1572864, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1703936, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1835008, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1966080, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197632, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197760, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197888, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198016, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198144, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198272, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198400, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198528, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%135)
      aiex.dma_await_task(%128)
      aiex.dma_await_task(%129)
      aiex.dma_await_task(%130)
      aiex.dma_await_task(%131)
      aiex.dma_await_task(%132)
      aiex.dma_await_task(%133)
      aiex.dma_await_task(%134)
      aiex.dma_await_task(%135)
      aiex.dma_free_task(%119)
      aiex.dma_free_task(%120)
      aiex.dma_free_task(%121)
      aiex.dma_free_task(%122)
      aiex.dma_free_task(%123)
      aiex.dma_free_task(%124)
      aiex.dma_free_task(%125)
      aiex.dma_free_task(%126)
      aiex.dma_free_task(%127)
      %136 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%136)
      %137 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2097152, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2228224, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2359296, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%139)
      %140 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2490368, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2621440, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2752512, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%142)
      %143 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2883584, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%143)
      %144 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3014656, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198656, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198784, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%146)
      %147 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198912, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199040, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199168, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%149)
      %150 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199296, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199424, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199552, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%152)
      aiex.dma_await_task(%145)
      aiex.dma_await_task(%146)
      aiex.dma_await_task(%147)
      aiex.dma_await_task(%148)
      aiex.dma_await_task(%149)
      aiex.dma_await_task(%150)
      aiex.dma_await_task(%151)
      aiex.dma_await_task(%152)
      aiex.dma_free_task(%136)
      aiex.dma_free_task(%137)
      aiex.dma_free_task(%138)
      aiex.dma_free_task(%139)
      aiex.dma_free_task(%140)
      aiex.dma_free_task(%141)
      aiex.dma_free_task(%142)
      aiex.dma_free_task(%143)
      aiex.dma_free_task(%144)
      %153 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3145728, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3276800, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%155)
      %156 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3407872, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%156)
      %157 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3538944, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3670016, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3801088, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%159)
      %160 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3932160, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%160)
      %161 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4063232, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%161)
      %162 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199680, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%162)
      %163 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199808, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%163)
      %164 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199936, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%164)
      %165 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200064, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%165)
      %166 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200192, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%166)
      %167 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200320, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%167)
      %168 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200448, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%168)
      %169 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200576, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%169)
      aiex.dma_await_task(%162)
      aiex.dma_await_task(%163)
      aiex.dma_await_task(%164)
      aiex.dma_await_task(%165)
      aiex.dma_await_task(%166)
      aiex.dma_await_task(%167)
      aiex.dma_await_task(%168)
      aiex.dma_await_task(%169)
      aiex.dma_free_task(%153)
      aiex.dma_free_task(%154)
      aiex.dma_free_task(%155)
      aiex.dma_free_task(%156)
      aiex.dma_free_task(%157)
      aiex.dma_free_task(%158)
      aiex.dma_free_task(%159)
      aiex.dma_free_task(%160)
      aiex.dma_free_task(%161)
      %170 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%170)
      %171 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4194304, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%171)
      %172 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4325376, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%172)
      %173 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4456448, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%173)
      %174 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4587520, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%174)
      %175 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4718592, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%175)
      %176 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4849664, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%176)
      %177 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4980736, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%177)
      %178 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5111808, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%178)
      %179 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200704, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%179)
      %180 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200832, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%180)
      %181 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200960, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%181)
      %182 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201088, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%182)
      %183 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201216, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%183)
      %184 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201344, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%184)
      %185 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201472, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%185)
      %186 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201600, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%186)
      aiex.dma_await_task(%179)
      aiex.dma_await_task(%180)
      aiex.dma_await_task(%181)
      aiex.dma_await_task(%182)
      aiex.dma_await_task(%183)
      aiex.dma_await_task(%184)
      aiex.dma_await_task(%185)
      aiex.dma_await_task(%186)
      aiex.dma_free_task(%170)
      aiex.dma_free_task(%171)
      aiex.dma_free_task(%172)
      aiex.dma_free_task(%173)
      aiex.dma_free_task(%174)
      aiex.dma_free_task(%175)
      aiex.dma_free_task(%176)
      aiex.dma_free_task(%177)
      aiex.dma_free_task(%178)
      %187 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%187)
      %188 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5242880, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%188)
      %189 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5373952, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%189)
      %190 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5505024, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%190)
      %191 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5636096, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%191)
      %192 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5767168, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%192)
      %193 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5898240, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%193)
      %194 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6029312, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%194)
      %195 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6160384, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%195)
      %196 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201728, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%196)
      %197 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201856, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%197)
      %198 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201984, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%198)
      %199 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 202112, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%199)
      %200 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 202240, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%200)
      %201 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 202368, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%201)
      %202 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 202496, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%202)
      %203 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 202624, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%203)
      aiex.dma_await_task(%196)
      aiex.dma_await_task(%197)
      aiex.dma_await_task(%198)
      aiex.dma_await_task(%199)
      aiex.dma_await_task(%200)
      aiex.dma_await_task(%201)
      aiex.dma_await_task(%202)
      aiex.dma_await_task(%203)
      aiex.dma_free_task(%187)
      aiex.dma_free_task(%188)
      aiex.dma_free_task(%189)
      aiex.dma_free_task(%190)
      aiex.dma_free_task(%191)
      aiex.dma_free_task(%192)
      aiex.dma_free_task(%193)
      aiex.dma_free_task(%194)
      aiex.dma_free_task(%195)
      %204 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%204)
      %205 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%205)
      %206 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%206)
      %207 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%207)
      %208 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%208)
      %209 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%209)
      %210 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%210)
      %211 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%211)
      %212 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%212)
      %213 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393216, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%213)
      %214 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393344, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%214)
      %215 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393472, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%215)
      %216 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393600, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%216)
      %217 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393728, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%217)
      %218 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393856, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%218)
      %219 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393984, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%219)
      %220 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394112, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%220)
      aiex.dma_await_task(%213)
      aiex.dma_await_task(%214)
      aiex.dma_await_task(%215)
      aiex.dma_await_task(%216)
      aiex.dma_await_task(%217)
      aiex.dma_await_task(%218)
      aiex.dma_await_task(%219)
      aiex.dma_await_task(%220)
      aiex.dma_free_task(%204)
      aiex.dma_free_task(%205)
      aiex.dma_free_task(%206)
      aiex.dma_free_task(%207)
      aiex.dma_free_task(%208)
      aiex.dma_free_task(%209)
      aiex.dma_free_task(%210)
      aiex.dma_free_task(%211)
      aiex.dma_free_task(%212)
      %221 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%221)
      %222 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1048576, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%222)
      %223 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1179648, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%223)
      %224 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1310720, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%224)
      %225 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1441792, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%225)
      %226 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1572864, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%226)
      %227 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1703936, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%227)
      %228 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1835008, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%228)
      %229 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1966080, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%229)
      %230 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394240, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%230)
      %231 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394368, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%231)
      %232 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394496, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%232)
      %233 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394624, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%233)
      %234 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394752, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%234)
      %235 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394880, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%235)
      %236 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395008, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%236)
      %237 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395136, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%237)
      aiex.dma_await_task(%230)
      aiex.dma_await_task(%231)
      aiex.dma_await_task(%232)
      aiex.dma_await_task(%233)
      aiex.dma_await_task(%234)
      aiex.dma_await_task(%235)
      aiex.dma_await_task(%236)
      aiex.dma_await_task(%237)
      aiex.dma_free_task(%221)
      aiex.dma_free_task(%222)
      aiex.dma_free_task(%223)
      aiex.dma_free_task(%224)
      aiex.dma_free_task(%225)
      aiex.dma_free_task(%226)
      aiex.dma_free_task(%227)
      aiex.dma_free_task(%228)
      aiex.dma_free_task(%229)
      %238 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%238)
      %239 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2097152, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%239)
      %240 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2228224, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%240)
      %241 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2359296, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%241)
      %242 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2490368, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%242)
      %243 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2621440, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%243)
      %244 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2752512, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%244)
      %245 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2883584, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%245)
      %246 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3014656, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%246)
      %247 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395264, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%247)
      %248 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395392, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%248)
      %249 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395520, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%249)
      %250 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395648, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%250)
      %251 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395776, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%251)
      %252 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395904, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%252)
      %253 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396032, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%253)
      %254 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396160, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%254)
      aiex.dma_await_task(%247)
      aiex.dma_await_task(%248)
      aiex.dma_await_task(%249)
      aiex.dma_await_task(%250)
      aiex.dma_await_task(%251)
      aiex.dma_await_task(%252)
      aiex.dma_await_task(%253)
      aiex.dma_await_task(%254)
      aiex.dma_free_task(%238)
      aiex.dma_free_task(%239)
      aiex.dma_free_task(%240)
      aiex.dma_free_task(%241)
      aiex.dma_free_task(%242)
      aiex.dma_free_task(%243)
      aiex.dma_free_task(%244)
      aiex.dma_free_task(%245)
      aiex.dma_free_task(%246)
      %255 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%255)
      %256 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3145728, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%256)
      %257 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3276800, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%257)
      %258 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3407872, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%258)
      %259 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3538944, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%259)
      %260 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3670016, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%260)
      %261 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3801088, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%261)
      %262 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3932160, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%262)
      %263 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4063232, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%263)
      %264 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396288, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%264)
      %265 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396416, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%265)
      %266 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396544, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%266)
      %267 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396672, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%267)
      %268 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396800, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%268)
      %269 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396928, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%269)
      %270 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397056, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%270)
      %271 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397184, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%271)
      aiex.dma_await_task(%264)
      aiex.dma_await_task(%265)
      aiex.dma_await_task(%266)
      aiex.dma_await_task(%267)
      aiex.dma_await_task(%268)
      aiex.dma_await_task(%269)
      aiex.dma_await_task(%270)
      aiex.dma_await_task(%271)
      aiex.dma_free_task(%255)
      aiex.dma_free_task(%256)
      aiex.dma_free_task(%257)
      aiex.dma_free_task(%258)
      aiex.dma_free_task(%259)
      aiex.dma_free_task(%260)
      aiex.dma_free_task(%261)
      aiex.dma_free_task(%262)
      aiex.dma_free_task(%263)
      %272 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%272)
      %273 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4194304, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%273)
      %274 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4325376, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%274)
      %275 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4456448, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%275)
      %276 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4587520, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%276)
      %277 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4718592, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%277)
      %278 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4849664, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%278)
      %279 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4980736, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%279)
      %280 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5111808, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%280)
      %281 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397312, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%281)
      %282 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397440, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%282)
      %283 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397568, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%283)
      %284 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397696, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%284)
      %285 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397824, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%285)
      %286 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397952, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%286)
      %287 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398080, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%287)
      %288 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398208, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%288)
      aiex.dma_await_task(%281)
      aiex.dma_await_task(%282)
      aiex.dma_await_task(%283)
      aiex.dma_await_task(%284)
      aiex.dma_await_task(%285)
      aiex.dma_await_task(%286)
      aiex.dma_await_task(%287)
      aiex.dma_await_task(%288)
      aiex.dma_free_task(%272)
      aiex.dma_free_task(%273)
      aiex.dma_free_task(%274)
      aiex.dma_free_task(%275)
      aiex.dma_free_task(%276)
      aiex.dma_free_task(%277)
      aiex.dma_free_task(%278)
      aiex.dma_free_task(%279)
      aiex.dma_free_task(%280)
      %289 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%289)
      %290 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5242880, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%290)
      %291 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5373952, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%291)
      %292 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5505024, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%292)
      %293 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5636096, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%293)
      %294 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5767168, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%294)
      %295 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5898240, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%295)
      %296 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6029312, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%296)
      %297 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6160384, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%297)
      %298 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398336, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%298)
      %299 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398464, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%299)
      %300 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398592, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%300)
      %301 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398720, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%301)
      %302 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398848, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%302)
      %303 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398976, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%303)
      %304 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 399104, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%304)
      %305 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 399232, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%305)
      aiex.dma_await_task(%298)
      aiex.dma_await_task(%299)
      aiex.dma_await_task(%300)
      aiex.dma_await_task(%301)
      aiex.dma_await_task(%302)
      aiex.dma_await_task(%303)
      aiex.dma_await_task(%304)
      aiex.dma_await_task(%305)
      aiex.dma_free_task(%289)
      aiex.dma_free_task(%290)
      aiex.dma_free_task(%291)
      aiex.dma_free_task(%292)
      aiex.dma_free_task(%293)
      aiex.dma_free_task(%294)
      aiex.dma_free_task(%295)
      aiex.dma_free_task(%296)
      aiex.dma_free_task(%297)
      %306 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%306)
      %307 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%307)
      %308 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%308)
      %309 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%309)
      %310 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%310)
      %311 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%311)
      %312 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%312)
      %313 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%313)
      %314 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%314)
      %315 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 589824, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%315)
      %316 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 589952, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%316)
      %317 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590080, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%317)
      %318 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590208, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%318)
      %319 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590336, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%319)
      %320 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590464, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%320)
      %321 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590592, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%321)
      %322 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590720, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%322)
      aiex.dma_await_task(%315)
      aiex.dma_await_task(%316)
      aiex.dma_await_task(%317)
      aiex.dma_await_task(%318)
      aiex.dma_await_task(%319)
      aiex.dma_await_task(%320)
      aiex.dma_await_task(%321)
      aiex.dma_await_task(%322)
      aiex.dma_free_task(%306)
      aiex.dma_free_task(%307)
      aiex.dma_free_task(%308)
      aiex.dma_free_task(%309)
      aiex.dma_free_task(%310)
      aiex.dma_free_task(%311)
      aiex.dma_free_task(%312)
      aiex.dma_free_task(%313)
      aiex.dma_free_task(%314)
      %323 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%323)
      %324 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1048576, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%324)
      %325 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1179648, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%325)
      %326 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1310720, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%326)
      %327 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1441792, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%327)
      %328 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1572864, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%328)
      %329 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1703936, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%329)
      %330 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1835008, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%330)
      %331 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1966080, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%331)
      %332 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590848, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%332)
      %333 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590976, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%333)
      %334 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591104, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%334)
      %335 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591232, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%335)
      %336 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591360, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%336)
      %337 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591488, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%337)
      %338 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591616, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%338)
      %339 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591744, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%339)
      aiex.dma_await_task(%332)
      aiex.dma_await_task(%333)
      aiex.dma_await_task(%334)
      aiex.dma_await_task(%335)
      aiex.dma_await_task(%336)
      aiex.dma_await_task(%337)
      aiex.dma_await_task(%338)
      aiex.dma_await_task(%339)
      aiex.dma_free_task(%323)
      aiex.dma_free_task(%324)
      aiex.dma_free_task(%325)
      aiex.dma_free_task(%326)
      aiex.dma_free_task(%327)
      aiex.dma_free_task(%328)
      aiex.dma_free_task(%329)
      aiex.dma_free_task(%330)
      aiex.dma_free_task(%331)
      %340 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%340)
      %341 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2097152, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%341)
      %342 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2228224, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%342)
      %343 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2359296, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%343)
      %344 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2490368, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%344)
      %345 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2621440, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%345)
      %346 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2752512, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%346)
      %347 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2883584, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%347)
      %348 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3014656, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%348)
      %349 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591872, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%349)
      %350 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592000, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%350)
      %351 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592128, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%351)
      %352 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592256, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%352)
      %353 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592384, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%353)
      %354 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592512, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%354)
      %355 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592640, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%355)
      %356 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%356)
      aiex.dma_await_task(%349)
      aiex.dma_await_task(%350)
      aiex.dma_await_task(%351)
      aiex.dma_await_task(%352)
      aiex.dma_await_task(%353)
      aiex.dma_await_task(%354)
      aiex.dma_await_task(%355)
      aiex.dma_await_task(%356)
      aiex.dma_free_task(%340)
      aiex.dma_free_task(%341)
      aiex.dma_free_task(%342)
      aiex.dma_free_task(%343)
      aiex.dma_free_task(%344)
      aiex.dma_free_task(%345)
      aiex.dma_free_task(%346)
      aiex.dma_free_task(%347)
      aiex.dma_free_task(%348)
      %357 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%357)
      %358 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3145728, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%358)
      %359 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3276800, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%359)
      %360 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3407872, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%360)
      %361 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3538944, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%361)
      %362 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3670016, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%362)
      %363 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3801088, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%363)
      %364 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3932160, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%364)
      %365 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4063232, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%365)
      %366 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%366)
      %367 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593024, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%367)
      %368 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593152, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%368)
      %369 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593280, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%369)
      %370 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593408, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%370)
      %371 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%371)
      %372 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%372)
      %373 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593792, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%373)
      aiex.dma_await_task(%366)
      aiex.dma_await_task(%367)
      aiex.dma_await_task(%368)
      aiex.dma_await_task(%369)
      aiex.dma_await_task(%370)
      aiex.dma_await_task(%371)
      aiex.dma_await_task(%372)
      aiex.dma_await_task(%373)
      aiex.dma_free_task(%357)
      aiex.dma_free_task(%358)
      aiex.dma_free_task(%359)
      aiex.dma_free_task(%360)
      aiex.dma_free_task(%361)
      aiex.dma_free_task(%362)
      aiex.dma_free_task(%363)
      aiex.dma_free_task(%364)
      aiex.dma_free_task(%365)
      %374 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%374)
      %375 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4194304, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%375)
      %376 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4325376, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%376)
      %377 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4456448, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%377)
      %378 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4587520, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%378)
      %379 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4718592, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%379)
      %380 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4849664, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%380)
      %381 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4980736, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%381)
      %382 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5111808, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%382)
      %383 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593920, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%383)
      %384 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594048, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%384)
      %385 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594176, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%385)
      %386 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%386)
      %387 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%387)
      %388 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594560, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%388)
      %389 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594688, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%389)
      %390 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594816, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%390)
      aiex.dma_await_task(%383)
      aiex.dma_await_task(%384)
      aiex.dma_await_task(%385)
      aiex.dma_await_task(%386)
      aiex.dma_await_task(%387)
      aiex.dma_await_task(%388)
      aiex.dma_await_task(%389)
      aiex.dma_await_task(%390)
      aiex.dma_free_task(%374)
      aiex.dma_free_task(%375)
      aiex.dma_free_task(%376)
      aiex.dma_free_task(%377)
      aiex.dma_free_task(%378)
      aiex.dma_free_task(%379)
      aiex.dma_free_task(%380)
      aiex.dma_free_task(%381)
      aiex.dma_free_task(%382)
      %391 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%391)
      %392 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5242880, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%392)
      %393 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5373952, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%393)
      %394 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5505024, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%394)
      %395 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5636096, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%395)
      %396 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5767168, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%396)
      %397 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5898240, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%397)
      %398 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6029312, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%398)
      %399 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6160384, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%399)
      %400 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594944, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%400)
      %401 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595072, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%401)
      %402 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595200, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%402)
      %403 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595328, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%403)
      %404 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595456, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%404)
      %405 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595584, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%405)
      %406 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595712, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%406)
      %407 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595840, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%407)
      aiex.dma_await_task(%400)
      aiex.dma_await_task(%401)
      aiex.dma_await_task(%402)
      aiex.dma_await_task(%403)
      aiex.dma_await_task(%404)
      aiex.dma_await_task(%405)
      aiex.dma_await_task(%406)
      aiex.dma_await_task(%407)
      aiex.dma_free_task(%391)
      aiex.dma_free_task(%392)
      aiex.dma_free_task(%393)
      aiex.dma_free_task(%394)
      aiex.dma_free_task(%395)
      aiex.dma_free_task(%396)
      aiex.dma_free_task(%397)
      aiex.dma_free_task(%398)
      aiex.dma_free_task(%399)
    }
    aie.shim_dma_allocation @A_L3L2_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 512>, <size = 512, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 512>, <size = 64, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 512>, <size = 64, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
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
      aie.dma_bd(%C_L1L2_0_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%C_L1L2_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L1L2_0_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%C_L1L2_0_0_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%C_L1L2_1_0_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%C_L1L2_2_0_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%C_L1L2_3_0_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%C_L1L2_4_0_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%C_L1L2_5_0_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%C_L1L2_6_0_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>, 0, 2048, [<size = 32, stride = 64>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%C_L1L2_7_0_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
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
      aie.dma_bd(%C_L1L2_1_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 25 : i32, next_bd_id = 25 : i32}
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
      aie.dma_bd(%C_L1L2_2_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 25 : i32, next_bd_id = 25 : i32}
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
      aie.dma_bd(%C_L1L2_3_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 25 : i32, next_bd_id = 25 : i32}
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
      aie.dma_bd(%C_L1L2_4_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_4_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 25 : i32, next_bd_id = 25 : i32}
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
      aie.dma_bd(%C_L1L2_5_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_5_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 25 : i32, next_bd_id = 25 : i32}
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
      aie.dma_bd(%C_L1L2_6_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_6_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 25 : i32, next_bd_id = 25 : i32}
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
      aie.dma_bd(%C_L1L2_7_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%C_L1L2_7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%C_L1L2_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_7_0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 25 : i32, next_bd_id = 25 : i32}
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
