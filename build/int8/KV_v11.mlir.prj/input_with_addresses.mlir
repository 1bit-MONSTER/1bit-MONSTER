module {
  aie.device(npu2) {
    func.func private @zero_scalar_i16(memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
    func.func private @matmul_scalar_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
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
      %c8 = arith.constant 8 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb58
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb59
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb10(%9: memref<64x128xi8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %9, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous0[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb17(%c0 : index)
    ^bb17(%30: index):  // 2 preds: ^bb16, ^bb29
      %31 = arith.cmpi slt, %30, %c2 : index
      cf.cond_br %31, ^bb18, ^bb30
    ^bb18:  // pred: ^bb17
      cf.br ^bb19(%c0 : index)
    ^bb19(%32: index):  // 2 preds: ^bb18, ^bb28
      %33 = arith.cmpi slt, %32, %c8 : index
      cf.cond_br %33, ^bb20, ^bb29
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %34 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.index_cast %35 : index to i32
      cf.switch %36 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb24(%37: memref<64x128xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb27,
        0: ^bb25,
        1: ^bb26
      ]
    ^bb25:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb26:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb27:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb28(%41: memref<32x64xi8>):  // 3 preds: ^bb25, ^bb26, ^bb27
      func.call @matmul_scalar_i8_i16(%41, %37, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %47 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %48 = arith.addi %47, %c1_i32 : i32
      %49 = arith.cmpi sge, %48, %c2_i32 : i32
      %50 = arith.subi %48, %c2_i32 : i32
      %51 = arith.select %49, %50, %48 : i32
      memref.store %51, %_anonymous0[%c1] : memref<3xi32>
      %52 = arith.addi %32, %c1 : index
      cf.br ^bb19(%52 : index)
    ^bb29:  // pred: ^bb19
      %53 = arith.addi %30, %c1 : index
      cf.br ^bb17(%53 : index)
    ^bb30:  // pred: ^bb17
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %54 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %55 = arith.addi %54, %c1_i32 : i32
      %56 = arith.cmpi sge, %55, %c1_i32 : i32
      %57 = arith.select %56, %54, %55 : i32
      memref.store %57, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb31(%c0 : index)
    ^bb31(%58: index):  // 2 preds: ^bb30, ^bb43
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb32, ^bb44
    ^bb32:  // pred: ^bb31
      cf.br ^bb33(%c0 : index)
    ^bb33(%60: index):  // 2 preds: ^bb32, ^bb42
      %61 = arith.cmpi slt, %60, %c8 : index
      cf.cond_br %61, ^bb34, ^bb43
    ^bb34:  // pred: ^bb33
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %62 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %63 = arith.index_cast %62 : i32 to index
      %64 = arith.index_cast %63 : index to i32
      cf.switch %64 : i32, [
        default: ^bb37,
        0: ^bb35,
        1: ^bb36
      ]
    ^bb35:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb36:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_0_cons_buff_1 : memref<64x128xi8>)
    ^bb37:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_0_cons_buff_0 : memref<64x128xi8>)
    ^bb38(%65: memref<64x128xi8>):  // 3 preds: ^bb35, ^bb36, ^bb37
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %66 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %67 = arith.index_cast %66 : i32 to index
      %68 = arith.index_cast %67 : index to i32
      cf.switch %68 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>)
    ^bb42(%69: memref<32x64xi8>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @matmul_scalar_i8_i16(%69, %65, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %70 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %71 = arith.addi %70, %c1_i32 : i32
      %72 = arith.cmpi sge, %71, %c2_i32 : i32
      %73 = arith.subi %71, %c2_i32 : i32
      %74 = arith.select %72, %73, %71 : i32
      memref.store %74, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %75 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %76 = arith.addi %75, %c1_i32 : i32
      %77 = arith.cmpi sge, %76, %c2_i32 : i32
      %78 = arith.subi %76, %c2_i32 : i32
      %79 = arith.select %77, %78, %76 : i32
      memref.store %79, %_anonymous0[%c1] : memref<3xi32>
      %80 = arith.addi %60, %c1 : index
      cf.br ^bb33(%80 : index)
    ^bb43:  // pred: ^bb33
      %81 = arith.addi %58, %c1 : index
      cf.br ^bb31(%81 : index)
    ^bb44:  // pred: ^bb31
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %82 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c1_i32 : i32
      %85 = arith.select %84, %82, %83 : i32
      memref.store %85, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_0_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb45(%c0 : index)
    ^bb45(%86: index):  // 2 preds: ^bb44, ^bb57
      %87 = arith.cmpi slt, %86, %c2 : index
      cf.cond_br %87, ^bb46, ^bb58
    ^bb46:  // pred: ^bb45
      cf.br ^bb47(%c0 : index)
    ^bb47(%88: index):  // 2 preds: ^bb46, ^bb56
      %89 = arith.cmpi slt, %88, %c8 : index
      cf.cond_br %89, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %90 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %91 = arith.index_cast %90 : i32 to index
      %92 = arith.index_cast %91 : index to i32
      cf.switch %92 : i32, [
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
    ^bb52(%93: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %94 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %95 = arith.index_cast %94 : i32 to index
      %96 = arith.index_cast %95 : index to i32
      cf.switch %96 : i32, [
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
    ^bb56(%97: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_scalar_i8_i16(%97, %93, %C_L1L2_0_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous0[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_cons_prod_lock_0, Release, 1)
      %103 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %104 = arith.addi %103, %c1_i32 : i32
      %105 = arith.cmpi sge, %104, %c2_i32 : i32
      %106 = arith.subi %104, %c2_i32 : i32
      %107 = arith.select %105, %106, %104 : i32
      memref.store %107, %_anonymous0[%c1] : memref<3xi32>
      %108 = arith.addi %88, %c1 : index
      cf.br ^bb47(%108 : index)
    ^bb57:  // pred: ^bb47
      %109 = arith.addi %86, %c1 : index
      cf.br ^bb45(%109 : index)
    ^bb58:  // pred: ^bb45
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %110 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %111 = arith.addi %110, %c1_i32 : i32
      %112 = arith.cmpi sge, %111, %c1_i32 : i32
      %113 = arith.select %112, %110, %111 : i32
      memref.store %113, %_anonymous0[%c0] : memref<3xi32>
      %114 = arith.addi %0, %c1 : index
      cf.br ^bb1(%114 : index)
    ^bb59:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous1 = aie.buffer(%tile_1_2) {address = 32000 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
    %core_1_2 = aie.core(%tile_1_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c8 = arith.constant 8 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb58
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb59
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb10(%9: memref<64x128xi8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %9, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous1[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb17(%c0 : index)
    ^bb17(%30: index):  // 2 preds: ^bb16, ^bb29
      %31 = arith.cmpi slt, %30, %c2 : index
      cf.cond_br %31, ^bb18, ^bb30
    ^bb18:  // pred: ^bb17
      cf.br ^bb19(%c0 : index)
    ^bb19(%32: index):  // 2 preds: ^bb18, ^bb28
      %33 = arith.cmpi slt, %32, %c8 : index
      cf.cond_br %33, ^bb20, ^bb29
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %34 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.index_cast %35 : index to i32
      cf.switch %36 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb24(%37: memref<64x128xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb27,
        0: ^bb25,
        1: ^bb26
      ]
    ^bb25:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb26:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb27:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb28(%41: memref<32x64xi8>):  // 3 preds: ^bb25, ^bb26, ^bb27
      func.call @matmul_scalar_i8_i16(%41, %37, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %47 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %48 = arith.addi %47, %c1_i32 : i32
      %49 = arith.cmpi sge, %48, %c2_i32 : i32
      %50 = arith.subi %48, %c2_i32 : i32
      %51 = arith.select %49, %50, %48 : i32
      memref.store %51, %_anonymous1[%c1] : memref<3xi32>
      %52 = arith.addi %32, %c1 : index
      cf.br ^bb19(%52 : index)
    ^bb29:  // pred: ^bb19
      %53 = arith.addi %30, %c1 : index
      cf.br ^bb17(%53 : index)
    ^bb30:  // pred: ^bb17
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %54 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %55 = arith.addi %54, %c1_i32 : i32
      %56 = arith.cmpi sge, %55, %c1_i32 : i32
      %57 = arith.select %56, %54, %55 : i32
      memref.store %57, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb31(%c0 : index)
    ^bb31(%58: index):  // 2 preds: ^bb30, ^bb43
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb32, ^bb44
    ^bb32:  // pred: ^bb31
      cf.br ^bb33(%c0 : index)
    ^bb33(%60: index):  // 2 preds: ^bb32, ^bb42
      %61 = arith.cmpi slt, %60, %c8 : index
      cf.cond_br %61, ^bb34, ^bb43
    ^bb34:  // pred: ^bb33
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %62 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %63 = arith.index_cast %62 : i32 to index
      %64 = arith.index_cast %63 : index to i32
      cf.switch %64 : i32, [
        default: ^bb37,
        0: ^bb35,
        1: ^bb36
      ]
    ^bb35:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb36:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_1_cons_buff_1 : memref<64x128xi8>)
    ^bb37:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_1_cons_buff_0 : memref<64x128xi8>)
    ^bb38(%65: memref<64x128xi8>):  // 3 preds: ^bb35, ^bb36, ^bb37
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %66 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %67 = arith.index_cast %66 : i32 to index
      %68 = arith.index_cast %67 : index to i32
      cf.switch %68 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>)
    ^bb42(%69: memref<32x64xi8>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @matmul_scalar_i8_i16(%69, %65, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %70 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %71 = arith.addi %70, %c1_i32 : i32
      %72 = arith.cmpi sge, %71, %c2_i32 : i32
      %73 = arith.subi %71, %c2_i32 : i32
      %74 = arith.select %72, %73, %71 : i32
      memref.store %74, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %75 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %76 = arith.addi %75, %c1_i32 : i32
      %77 = arith.cmpi sge, %76, %c2_i32 : i32
      %78 = arith.subi %76, %c2_i32 : i32
      %79 = arith.select %77, %78, %76 : i32
      memref.store %79, %_anonymous1[%c1] : memref<3xi32>
      %80 = arith.addi %60, %c1 : index
      cf.br ^bb33(%80 : index)
    ^bb43:  // pred: ^bb33
      %81 = arith.addi %58, %c1 : index
      cf.br ^bb31(%81 : index)
    ^bb44:  // pred: ^bb31
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %82 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c1_i32 : i32
      %85 = arith.select %84, %82, %83 : i32
      memref.store %85, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_1_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb45(%c0 : index)
    ^bb45(%86: index):  // 2 preds: ^bb44, ^bb57
      %87 = arith.cmpi slt, %86, %c2 : index
      cf.cond_br %87, ^bb46, ^bb58
    ^bb46:  // pred: ^bb45
      cf.br ^bb47(%c0 : index)
    ^bb47(%88: index):  // 2 preds: ^bb46, ^bb56
      %89 = arith.cmpi slt, %88, %c8 : index
      cf.cond_br %89, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %90 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %91 = arith.index_cast %90 : i32 to index
      %92 = arith.index_cast %91 : index to i32
      cf.switch %92 : i32, [
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
    ^bb52(%93: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %94 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %95 = arith.index_cast %94 : i32 to index
      %96 = arith.index_cast %95 : index to i32
      cf.switch %96 : i32, [
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
    ^bb56(%97: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_scalar_i8_i16(%97, %93, %C_L1L2_1_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous1[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_cons_prod_lock_0, Release, 1)
      %103 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %104 = arith.addi %103, %c1_i32 : i32
      %105 = arith.cmpi sge, %104, %c2_i32 : i32
      %106 = arith.subi %104, %c2_i32 : i32
      %107 = arith.select %105, %106, %104 : i32
      memref.store %107, %_anonymous1[%c1] : memref<3xi32>
      %108 = arith.addi %88, %c1 : index
      cf.br ^bb47(%108 : index)
    ^bb57:  // pred: ^bb47
      %109 = arith.addi %86, %c1 : index
      cf.br ^bb45(%109 : index)
    ^bb58:  // pred: ^bb45
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %110 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %111 = arith.addi %110, %c1_i32 : i32
      %112 = arith.cmpi sge, %111, %c1_i32 : i32
      %113 = arith.select %112, %110, %111 : i32
      memref.store %113, %_anonymous1[%c0] : memref<3xi32>
      %114 = arith.addi %0, %c1 : index
      cf.br ^bb1(%114 : index)
    ^bb59:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous2 = aie.buffer(%tile_2_2) {address = 32000 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
    %core_2_2 = aie.core(%tile_2_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c8 = arith.constant 8 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb58
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb59
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb10(%9: memref<64x128xi8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %9, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous2[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb17(%c0 : index)
    ^bb17(%30: index):  // 2 preds: ^bb16, ^bb29
      %31 = arith.cmpi slt, %30, %c2 : index
      cf.cond_br %31, ^bb18, ^bb30
    ^bb18:  // pred: ^bb17
      cf.br ^bb19(%c0 : index)
    ^bb19(%32: index):  // 2 preds: ^bb18, ^bb28
      %33 = arith.cmpi slt, %32, %c8 : index
      cf.cond_br %33, ^bb20, ^bb29
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %34 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.index_cast %35 : index to i32
      cf.switch %36 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb24(%37: memref<64x128xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb27,
        0: ^bb25,
        1: ^bb26
      ]
    ^bb25:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb26:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb27:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb28(%41: memref<32x64xi8>):  // 3 preds: ^bb25, ^bb26, ^bb27
      func.call @matmul_scalar_i8_i16(%41, %37, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %47 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %48 = arith.addi %47, %c1_i32 : i32
      %49 = arith.cmpi sge, %48, %c2_i32 : i32
      %50 = arith.subi %48, %c2_i32 : i32
      %51 = arith.select %49, %50, %48 : i32
      memref.store %51, %_anonymous2[%c1] : memref<3xi32>
      %52 = arith.addi %32, %c1 : index
      cf.br ^bb19(%52 : index)
    ^bb29:  // pred: ^bb19
      %53 = arith.addi %30, %c1 : index
      cf.br ^bb17(%53 : index)
    ^bb30:  // pred: ^bb17
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %54 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %55 = arith.addi %54, %c1_i32 : i32
      %56 = arith.cmpi sge, %55, %c1_i32 : i32
      %57 = arith.select %56, %54, %55 : i32
      memref.store %57, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb31(%c0 : index)
    ^bb31(%58: index):  // 2 preds: ^bb30, ^bb43
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb32, ^bb44
    ^bb32:  // pred: ^bb31
      cf.br ^bb33(%c0 : index)
    ^bb33(%60: index):  // 2 preds: ^bb32, ^bb42
      %61 = arith.cmpi slt, %60, %c8 : index
      cf.cond_br %61, ^bb34, ^bb43
    ^bb34:  // pred: ^bb33
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %62 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %63 = arith.index_cast %62 : i32 to index
      %64 = arith.index_cast %63 : index to i32
      cf.switch %64 : i32, [
        default: ^bb37,
        0: ^bb35,
        1: ^bb36
      ]
    ^bb35:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb36:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_2_cons_buff_1 : memref<64x128xi8>)
    ^bb37:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_2_cons_buff_0 : memref<64x128xi8>)
    ^bb38(%65: memref<64x128xi8>):  // 3 preds: ^bb35, ^bb36, ^bb37
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %66 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %67 = arith.index_cast %66 : i32 to index
      %68 = arith.index_cast %67 : index to i32
      cf.switch %68 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>)
    ^bb42(%69: memref<32x64xi8>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @matmul_scalar_i8_i16(%69, %65, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %70 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %71 = arith.addi %70, %c1_i32 : i32
      %72 = arith.cmpi sge, %71, %c2_i32 : i32
      %73 = arith.subi %71, %c2_i32 : i32
      %74 = arith.select %72, %73, %71 : i32
      memref.store %74, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %75 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %76 = arith.addi %75, %c1_i32 : i32
      %77 = arith.cmpi sge, %76, %c2_i32 : i32
      %78 = arith.subi %76, %c2_i32 : i32
      %79 = arith.select %77, %78, %76 : i32
      memref.store %79, %_anonymous2[%c1] : memref<3xi32>
      %80 = arith.addi %60, %c1 : index
      cf.br ^bb33(%80 : index)
    ^bb43:  // pred: ^bb33
      %81 = arith.addi %58, %c1 : index
      cf.br ^bb31(%81 : index)
    ^bb44:  // pred: ^bb31
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %82 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c1_i32 : i32
      %85 = arith.select %84, %82, %83 : i32
      memref.store %85, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_2_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb45(%c0 : index)
    ^bb45(%86: index):  // 2 preds: ^bb44, ^bb57
      %87 = arith.cmpi slt, %86, %c2 : index
      cf.cond_br %87, ^bb46, ^bb58
    ^bb46:  // pred: ^bb45
      cf.br ^bb47(%c0 : index)
    ^bb47(%88: index):  // 2 preds: ^bb46, ^bb56
      %89 = arith.cmpi slt, %88, %c8 : index
      cf.cond_br %89, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %90 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %91 = arith.index_cast %90 : i32 to index
      %92 = arith.index_cast %91 : index to i32
      cf.switch %92 : i32, [
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
    ^bb52(%93: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %94 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %95 = arith.index_cast %94 : i32 to index
      %96 = arith.index_cast %95 : index to i32
      cf.switch %96 : i32, [
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
    ^bb56(%97: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_scalar_i8_i16(%97, %93, %C_L1L2_2_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous2[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_cons_prod_lock_0, Release, 1)
      %103 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %104 = arith.addi %103, %c1_i32 : i32
      %105 = arith.cmpi sge, %104, %c2_i32 : i32
      %106 = arith.subi %104, %c2_i32 : i32
      %107 = arith.select %105, %106, %104 : i32
      memref.store %107, %_anonymous2[%c1] : memref<3xi32>
      %108 = arith.addi %88, %c1 : index
      cf.br ^bb47(%108 : index)
    ^bb57:  // pred: ^bb47
      %109 = arith.addi %86, %c1 : index
      cf.br ^bb45(%109 : index)
    ^bb58:  // pred: ^bb45
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %110 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %111 = arith.addi %110, %c1_i32 : i32
      %112 = arith.cmpi sge, %111, %c1_i32 : i32
      %113 = arith.select %112, %110, %111 : i32
      memref.store %113, %_anonymous2[%c0] : memref<3xi32>
      %114 = arith.addi %0, %c1 : index
      cf.br ^bb1(%114 : index)
    ^bb59:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous3 = aie.buffer(%tile_3_2) {address = 32000 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
    %core_3_2 = aie.core(%tile_3_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c8 = arith.constant 8 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb58
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb59
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb10(%9: memref<64x128xi8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %9, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous3[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb17(%c0 : index)
    ^bb17(%30: index):  // 2 preds: ^bb16, ^bb29
      %31 = arith.cmpi slt, %30, %c2 : index
      cf.cond_br %31, ^bb18, ^bb30
    ^bb18:  // pred: ^bb17
      cf.br ^bb19(%c0 : index)
    ^bb19(%32: index):  // 2 preds: ^bb18, ^bb28
      %33 = arith.cmpi slt, %32, %c8 : index
      cf.cond_br %33, ^bb20, ^bb29
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %34 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.index_cast %35 : index to i32
      cf.switch %36 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb24(%37: memref<64x128xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb27,
        0: ^bb25,
        1: ^bb26
      ]
    ^bb25:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb26:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb27:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb28(%41: memref<32x64xi8>):  // 3 preds: ^bb25, ^bb26, ^bb27
      func.call @matmul_scalar_i8_i16(%41, %37, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %47 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %48 = arith.addi %47, %c1_i32 : i32
      %49 = arith.cmpi sge, %48, %c2_i32 : i32
      %50 = arith.subi %48, %c2_i32 : i32
      %51 = arith.select %49, %50, %48 : i32
      memref.store %51, %_anonymous3[%c1] : memref<3xi32>
      %52 = arith.addi %32, %c1 : index
      cf.br ^bb19(%52 : index)
    ^bb29:  // pred: ^bb19
      %53 = arith.addi %30, %c1 : index
      cf.br ^bb17(%53 : index)
    ^bb30:  // pred: ^bb17
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %54 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %55 = arith.addi %54, %c1_i32 : i32
      %56 = arith.cmpi sge, %55, %c1_i32 : i32
      %57 = arith.select %56, %54, %55 : i32
      memref.store %57, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb31(%c0 : index)
    ^bb31(%58: index):  // 2 preds: ^bb30, ^bb43
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb32, ^bb44
    ^bb32:  // pred: ^bb31
      cf.br ^bb33(%c0 : index)
    ^bb33(%60: index):  // 2 preds: ^bb32, ^bb42
      %61 = arith.cmpi slt, %60, %c8 : index
      cf.cond_br %61, ^bb34, ^bb43
    ^bb34:  // pred: ^bb33
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %62 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %63 = arith.index_cast %62 : i32 to index
      %64 = arith.index_cast %63 : index to i32
      cf.switch %64 : i32, [
        default: ^bb37,
        0: ^bb35,
        1: ^bb36
      ]
    ^bb35:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb36:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_3_cons_buff_1 : memref<64x128xi8>)
    ^bb37:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_3_cons_buff_0 : memref<64x128xi8>)
    ^bb38(%65: memref<64x128xi8>):  // 3 preds: ^bb35, ^bb36, ^bb37
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %66 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %67 = arith.index_cast %66 : i32 to index
      %68 = arith.index_cast %67 : index to i32
      cf.switch %68 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>)
    ^bb42(%69: memref<32x64xi8>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @matmul_scalar_i8_i16(%69, %65, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %70 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %71 = arith.addi %70, %c1_i32 : i32
      %72 = arith.cmpi sge, %71, %c2_i32 : i32
      %73 = arith.subi %71, %c2_i32 : i32
      %74 = arith.select %72, %73, %71 : i32
      memref.store %74, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %75 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %76 = arith.addi %75, %c1_i32 : i32
      %77 = arith.cmpi sge, %76, %c2_i32 : i32
      %78 = arith.subi %76, %c2_i32 : i32
      %79 = arith.select %77, %78, %76 : i32
      memref.store %79, %_anonymous3[%c1] : memref<3xi32>
      %80 = arith.addi %60, %c1 : index
      cf.br ^bb33(%80 : index)
    ^bb43:  // pred: ^bb33
      %81 = arith.addi %58, %c1 : index
      cf.br ^bb31(%81 : index)
    ^bb44:  // pred: ^bb31
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %82 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c1_i32 : i32
      %85 = arith.select %84, %82, %83 : i32
      memref.store %85, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_3_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb45(%c0 : index)
    ^bb45(%86: index):  // 2 preds: ^bb44, ^bb57
      %87 = arith.cmpi slt, %86, %c2 : index
      cf.cond_br %87, ^bb46, ^bb58
    ^bb46:  // pred: ^bb45
      cf.br ^bb47(%c0 : index)
    ^bb47(%88: index):  // 2 preds: ^bb46, ^bb56
      %89 = arith.cmpi slt, %88, %c8 : index
      cf.cond_br %89, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %90 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %91 = arith.index_cast %90 : i32 to index
      %92 = arith.index_cast %91 : index to i32
      cf.switch %92 : i32, [
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
    ^bb52(%93: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %94 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %95 = arith.index_cast %94 : i32 to index
      %96 = arith.index_cast %95 : index to i32
      cf.switch %96 : i32, [
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
    ^bb56(%97: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_scalar_i8_i16(%97, %93, %C_L1L2_3_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous3[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_cons_prod_lock_0, Release, 1)
      %103 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %104 = arith.addi %103, %c1_i32 : i32
      %105 = arith.cmpi sge, %104, %c2_i32 : i32
      %106 = arith.subi %104, %c2_i32 : i32
      %107 = arith.select %105, %106, %104 : i32
      memref.store %107, %_anonymous3[%c1] : memref<3xi32>
      %108 = arith.addi %88, %c1 : index
      cf.br ^bb47(%108 : index)
    ^bb57:  // pred: ^bb47
      %109 = arith.addi %86, %c1 : index
      cf.br ^bb45(%109 : index)
    ^bb58:  // pred: ^bb45
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %110 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %111 = arith.addi %110, %c1_i32 : i32
      %112 = arith.cmpi sge, %111, %c1_i32 : i32
      %113 = arith.select %112, %110, %111 : i32
      memref.store %113, %_anonymous3[%c0] : memref<3xi32>
      %114 = arith.addi %0, %c1 : index
      cf.br ^bb1(%114 : index)
    ^bb59:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous4 = aie.buffer(%tile_4_2) {address = 32000 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
    %core_4_2 = aie.core(%tile_4_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c8 = arith.constant 8 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb58
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb59
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb10(%9: memref<64x128xi8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %9, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous4[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb17(%c0 : index)
    ^bb17(%30: index):  // 2 preds: ^bb16, ^bb29
      %31 = arith.cmpi slt, %30, %c2 : index
      cf.cond_br %31, ^bb18, ^bb30
    ^bb18:  // pred: ^bb17
      cf.br ^bb19(%c0 : index)
    ^bb19(%32: index):  // 2 preds: ^bb18, ^bb28
      %33 = arith.cmpi slt, %32, %c8 : index
      cf.cond_br %33, ^bb20, ^bb29
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %34 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.index_cast %35 : index to i32
      cf.switch %36 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb24(%37: memref<64x128xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb27,
        0: ^bb25,
        1: ^bb26
      ]
    ^bb25:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb26:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb27:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb28(%41: memref<32x64xi8>):  // 3 preds: ^bb25, ^bb26, ^bb27
      func.call @matmul_scalar_i8_i16(%41, %37, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %47 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %48 = arith.addi %47, %c1_i32 : i32
      %49 = arith.cmpi sge, %48, %c2_i32 : i32
      %50 = arith.subi %48, %c2_i32 : i32
      %51 = arith.select %49, %50, %48 : i32
      memref.store %51, %_anonymous4[%c1] : memref<3xi32>
      %52 = arith.addi %32, %c1 : index
      cf.br ^bb19(%52 : index)
    ^bb29:  // pred: ^bb19
      %53 = arith.addi %30, %c1 : index
      cf.br ^bb17(%53 : index)
    ^bb30:  // pred: ^bb17
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %54 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %55 = arith.addi %54, %c1_i32 : i32
      %56 = arith.cmpi sge, %55, %c1_i32 : i32
      %57 = arith.select %56, %54, %55 : i32
      memref.store %57, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb31(%c0 : index)
    ^bb31(%58: index):  // 2 preds: ^bb30, ^bb43
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb32, ^bb44
    ^bb32:  // pred: ^bb31
      cf.br ^bb33(%c0 : index)
    ^bb33(%60: index):  // 2 preds: ^bb32, ^bb42
      %61 = arith.cmpi slt, %60, %c8 : index
      cf.cond_br %61, ^bb34, ^bb43
    ^bb34:  // pred: ^bb33
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %62 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %63 = arith.index_cast %62 : i32 to index
      %64 = arith.index_cast %63 : index to i32
      cf.switch %64 : i32, [
        default: ^bb37,
        0: ^bb35,
        1: ^bb36
      ]
    ^bb35:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb36:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_4_cons_buff_1 : memref<64x128xi8>)
    ^bb37:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_4_cons_buff_0 : memref<64x128xi8>)
    ^bb38(%65: memref<64x128xi8>):  // 3 preds: ^bb35, ^bb36, ^bb37
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %66 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %67 = arith.index_cast %66 : i32 to index
      %68 = arith.index_cast %67 : index to i32
      cf.switch %68 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>)
    ^bb42(%69: memref<32x64xi8>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @matmul_scalar_i8_i16(%69, %65, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %70 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %71 = arith.addi %70, %c1_i32 : i32
      %72 = arith.cmpi sge, %71, %c2_i32 : i32
      %73 = arith.subi %71, %c2_i32 : i32
      %74 = arith.select %72, %73, %71 : i32
      memref.store %74, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %75 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %76 = arith.addi %75, %c1_i32 : i32
      %77 = arith.cmpi sge, %76, %c2_i32 : i32
      %78 = arith.subi %76, %c2_i32 : i32
      %79 = arith.select %77, %78, %76 : i32
      memref.store %79, %_anonymous4[%c1] : memref<3xi32>
      %80 = arith.addi %60, %c1 : index
      cf.br ^bb33(%80 : index)
    ^bb43:  // pred: ^bb33
      %81 = arith.addi %58, %c1 : index
      cf.br ^bb31(%81 : index)
    ^bb44:  // pred: ^bb31
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %82 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c1_i32 : i32
      %85 = arith.select %84, %82, %83 : i32
      memref.store %85, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_4_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb45(%c0 : index)
    ^bb45(%86: index):  // 2 preds: ^bb44, ^bb57
      %87 = arith.cmpi slt, %86, %c2 : index
      cf.cond_br %87, ^bb46, ^bb58
    ^bb46:  // pred: ^bb45
      cf.br ^bb47(%c0 : index)
    ^bb47(%88: index):  // 2 preds: ^bb46, ^bb56
      %89 = arith.cmpi slt, %88, %c8 : index
      cf.cond_br %89, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %90 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %91 = arith.index_cast %90 : i32 to index
      %92 = arith.index_cast %91 : index to i32
      cf.switch %92 : i32, [
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
    ^bb52(%93: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %94 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %95 = arith.index_cast %94 : i32 to index
      %96 = arith.index_cast %95 : index to i32
      cf.switch %96 : i32, [
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
    ^bb56(%97: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_scalar_i8_i16(%97, %93, %C_L1L2_4_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous4[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_4_cons_prod_lock_0, Release, 1)
      %103 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %104 = arith.addi %103, %c1_i32 : i32
      %105 = arith.cmpi sge, %104, %c2_i32 : i32
      %106 = arith.subi %104, %c2_i32 : i32
      %107 = arith.select %105, %106, %104 : i32
      memref.store %107, %_anonymous4[%c1] : memref<3xi32>
      %108 = arith.addi %88, %c1 : index
      cf.br ^bb47(%108 : index)
    ^bb57:  // pred: ^bb47
      %109 = arith.addi %86, %c1 : index
      cf.br ^bb45(%109 : index)
    ^bb58:  // pred: ^bb45
      aie.use_lock(%C_L1L2_4_0_cons_lock_0, Release, 1)
      %110 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %111 = arith.addi %110, %c1_i32 : i32
      %112 = arith.cmpi sge, %111, %c1_i32 : i32
      %113 = arith.select %112, %110, %111 : i32
      memref.store %113, %_anonymous4[%c0] : memref<3xi32>
      %114 = arith.addi %0, %c1 : index
      cf.br ^bb1(%114 : index)
    ^bb59:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous5 = aie.buffer(%tile_5_2) {address = 32000 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
    %core_5_2 = aie.core(%tile_5_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c8 = arith.constant 8 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb58
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb59
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb10(%9: memref<64x128xi8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %9, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous5[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb17(%c0 : index)
    ^bb17(%30: index):  // 2 preds: ^bb16, ^bb29
      %31 = arith.cmpi slt, %30, %c2 : index
      cf.cond_br %31, ^bb18, ^bb30
    ^bb18:  // pred: ^bb17
      cf.br ^bb19(%c0 : index)
    ^bb19(%32: index):  // 2 preds: ^bb18, ^bb28
      %33 = arith.cmpi slt, %32, %c8 : index
      cf.cond_br %33, ^bb20, ^bb29
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %34 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.index_cast %35 : index to i32
      cf.switch %36 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb24(%37: memref<64x128xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb27,
        0: ^bb25,
        1: ^bb26
      ]
    ^bb25:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb26:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb27:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb28(%41: memref<32x64xi8>):  // 3 preds: ^bb25, ^bb26, ^bb27
      func.call @matmul_scalar_i8_i16(%41, %37, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %47 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %48 = arith.addi %47, %c1_i32 : i32
      %49 = arith.cmpi sge, %48, %c2_i32 : i32
      %50 = arith.subi %48, %c2_i32 : i32
      %51 = arith.select %49, %50, %48 : i32
      memref.store %51, %_anonymous5[%c1] : memref<3xi32>
      %52 = arith.addi %32, %c1 : index
      cf.br ^bb19(%52 : index)
    ^bb29:  // pred: ^bb19
      %53 = arith.addi %30, %c1 : index
      cf.br ^bb17(%53 : index)
    ^bb30:  // pred: ^bb17
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %54 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %55 = arith.addi %54, %c1_i32 : i32
      %56 = arith.cmpi sge, %55, %c1_i32 : i32
      %57 = arith.select %56, %54, %55 : i32
      memref.store %57, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb31(%c0 : index)
    ^bb31(%58: index):  // 2 preds: ^bb30, ^bb43
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb32, ^bb44
    ^bb32:  // pred: ^bb31
      cf.br ^bb33(%c0 : index)
    ^bb33(%60: index):  // 2 preds: ^bb32, ^bb42
      %61 = arith.cmpi slt, %60, %c8 : index
      cf.cond_br %61, ^bb34, ^bb43
    ^bb34:  // pred: ^bb33
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %62 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %63 = arith.index_cast %62 : i32 to index
      %64 = arith.index_cast %63 : index to i32
      cf.switch %64 : i32, [
        default: ^bb37,
        0: ^bb35,
        1: ^bb36
      ]
    ^bb35:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb36:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_5_cons_buff_1 : memref<64x128xi8>)
    ^bb37:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_5_cons_buff_0 : memref<64x128xi8>)
    ^bb38(%65: memref<64x128xi8>):  // 3 preds: ^bb35, ^bb36, ^bb37
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %66 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %67 = arith.index_cast %66 : i32 to index
      %68 = arith.index_cast %67 : index to i32
      cf.switch %68 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>)
    ^bb42(%69: memref<32x64xi8>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @matmul_scalar_i8_i16(%69, %65, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %70 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %71 = arith.addi %70, %c1_i32 : i32
      %72 = arith.cmpi sge, %71, %c2_i32 : i32
      %73 = arith.subi %71, %c2_i32 : i32
      %74 = arith.select %72, %73, %71 : i32
      memref.store %74, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %75 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %76 = arith.addi %75, %c1_i32 : i32
      %77 = arith.cmpi sge, %76, %c2_i32 : i32
      %78 = arith.subi %76, %c2_i32 : i32
      %79 = arith.select %77, %78, %76 : i32
      memref.store %79, %_anonymous5[%c1] : memref<3xi32>
      %80 = arith.addi %60, %c1 : index
      cf.br ^bb33(%80 : index)
    ^bb43:  // pred: ^bb33
      %81 = arith.addi %58, %c1 : index
      cf.br ^bb31(%81 : index)
    ^bb44:  // pred: ^bb31
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %82 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c1_i32 : i32
      %85 = arith.select %84, %82, %83 : i32
      memref.store %85, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_5_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb45(%c0 : index)
    ^bb45(%86: index):  // 2 preds: ^bb44, ^bb57
      %87 = arith.cmpi slt, %86, %c2 : index
      cf.cond_br %87, ^bb46, ^bb58
    ^bb46:  // pred: ^bb45
      cf.br ^bb47(%c0 : index)
    ^bb47(%88: index):  // 2 preds: ^bb46, ^bb56
      %89 = arith.cmpi slt, %88, %c8 : index
      cf.cond_br %89, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %90 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %91 = arith.index_cast %90 : i32 to index
      %92 = arith.index_cast %91 : index to i32
      cf.switch %92 : i32, [
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
    ^bb52(%93: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %94 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %95 = arith.index_cast %94 : i32 to index
      %96 = arith.index_cast %95 : index to i32
      cf.switch %96 : i32, [
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
    ^bb56(%97: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_scalar_i8_i16(%97, %93, %C_L1L2_5_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous5[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_5_cons_prod_lock_0, Release, 1)
      %103 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %104 = arith.addi %103, %c1_i32 : i32
      %105 = arith.cmpi sge, %104, %c2_i32 : i32
      %106 = arith.subi %104, %c2_i32 : i32
      %107 = arith.select %105, %106, %104 : i32
      memref.store %107, %_anonymous5[%c1] : memref<3xi32>
      %108 = arith.addi %88, %c1 : index
      cf.br ^bb47(%108 : index)
    ^bb57:  // pred: ^bb47
      %109 = arith.addi %86, %c1 : index
      cf.br ^bb45(%109 : index)
    ^bb58:  // pred: ^bb45
      aie.use_lock(%C_L1L2_5_0_cons_lock_0, Release, 1)
      %110 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %111 = arith.addi %110, %c1_i32 : i32
      %112 = arith.cmpi sge, %111, %c1_i32 : i32
      %113 = arith.select %112, %110, %111 : i32
      memref.store %113, %_anonymous5[%c0] : memref<3xi32>
      %114 = arith.addi %0, %c1 : index
      cf.br ^bb1(%114 : index)
    ^bb59:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous6 = aie.buffer(%tile_6_2) {address = 32000 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
    %core_6_2 = aie.core(%tile_6_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c8 = arith.constant 8 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb58
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb59
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb10(%9: memref<64x128xi8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %9, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous6[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb17(%c0 : index)
    ^bb17(%30: index):  // 2 preds: ^bb16, ^bb29
      %31 = arith.cmpi slt, %30, %c2 : index
      cf.cond_br %31, ^bb18, ^bb30
    ^bb18:  // pred: ^bb17
      cf.br ^bb19(%c0 : index)
    ^bb19(%32: index):  // 2 preds: ^bb18, ^bb28
      %33 = arith.cmpi slt, %32, %c8 : index
      cf.cond_br %33, ^bb20, ^bb29
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %34 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.index_cast %35 : index to i32
      cf.switch %36 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb24(%37: memref<64x128xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb27,
        0: ^bb25,
        1: ^bb26
      ]
    ^bb25:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb26:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb27:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb28(%41: memref<32x64xi8>):  // 3 preds: ^bb25, ^bb26, ^bb27
      func.call @matmul_scalar_i8_i16(%41, %37, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %47 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %48 = arith.addi %47, %c1_i32 : i32
      %49 = arith.cmpi sge, %48, %c2_i32 : i32
      %50 = arith.subi %48, %c2_i32 : i32
      %51 = arith.select %49, %50, %48 : i32
      memref.store %51, %_anonymous6[%c1] : memref<3xi32>
      %52 = arith.addi %32, %c1 : index
      cf.br ^bb19(%52 : index)
    ^bb29:  // pred: ^bb19
      %53 = arith.addi %30, %c1 : index
      cf.br ^bb17(%53 : index)
    ^bb30:  // pred: ^bb17
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %54 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %55 = arith.addi %54, %c1_i32 : i32
      %56 = arith.cmpi sge, %55, %c1_i32 : i32
      %57 = arith.select %56, %54, %55 : i32
      memref.store %57, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb31(%c0 : index)
    ^bb31(%58: index):  // 2 preds: ^bb30, ^bb43
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb32, ^bb44
    ^bb32:  // pred: ^bb31
      cf.br ^bb33(%c0 : index)
    ^bb33(%60: index):  // 2 preds: ^bb32, ^bb42
      %61 = arith.cmpi slt, %60, %c8 : index
      cf.cond_br %61, ^bb34, ^bb43
    ^bb34:  // pred: ^bb33
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %62 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %63 = arith.index_cast %62 : i32 to index
      %64 = arith.index_cast %63 : index to i32
      cf.switch %64 : i32, [
        default: ^bb37,
        0: ^bb35,
        1: ^bb36
      ]
    ^bb35:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb36:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_6_cons_buff_1 : memref<64x128xi8>)
    ^bb37:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_6_cons_buff_0 : memref<64x128xi8>)
    ^bb38(%65: memref<64x128xi8>):  // 3 preds: ^bb35, ^bb36, ^bb37
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %66 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %67 = arith.index_cast %66 : i32 to index
      %68 = arith.index_cast %67 : index to i32
      cf.switch %68 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>)
    ^bb42(%69: memref<32x64xi8>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @matmul_scalar_i8_i16(%69, %65, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %70 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %71 = arith.addi %70, %c1_i32 : i32
      %72 = arith.cmpi sge, %71, %c2_i32 : i32
      %73 = arith.subi %71, %c2_i32 : i32
      %74 = arith.select %72, %73, %71 : i32
      memref.store %74, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %75 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %76 = arith.addi %75, %c1_i32 : i32
      %77 = arith.cmpi sge, %76, %c2_i32 : i32
      %78 = arith.subi %76, %c2_i32 : i32
      %79 = arith.select %77, %78, %76 : i32
      memref.store %79, %_anonymous6[%c1] : memref<3xi32>
      %80 = arith.addi %60, %c1 : index
      cf.br ^bb33(%80 : index)
    ^bb43:  // pred: ^bb33
      %81 = arith.addi %58, %c1 : index
      cf.br ^bb31(%81 : index)
    ^bb44:  // pred: ^bb31
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %82 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c1_i32 : i32
      %85 = arith.select %84, %82, %83 : i32
      memref.store %85, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_6_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb45(%c0 : index)
    ^bb45(%86: index):  // 2 preds: ^bb44, ^bb57
      %87 = arith.cmpi slt, %86, %c2 : index
      cf.cond_br %87, ^bb46, ^bb58
    ^bb46:  // pred: ^bb45
      cf.br ^bb47(%c0 : index)
    ^bb47(%88: index):  // 2 preds: ^bb46, ^bb56
      %89 = arith.cmpi slt, %88, %c8 : index
      cf.cond_br %89, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %90 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %91 = arith.index_cast %90 : i32 to index
      %92 = arith.index_cast %91 : index to i32
      cf.switch %92 : i32, [
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
    ^bb52(%93: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %94 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %95 = arith.index_cast %94 : i32 to index
      %96 = arith.index_cast %95 : index to i32
      cf.switch %96 : i32, [
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
    ^bb56(%97: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_scalar_i8_i16(%97, %93, %C_L1L2_6_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous6[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_6_cons_prod_lock_0, Release, 1)
      %103 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %104 = arith.addi %103, %c1_i32 : i32
      %105 = arith.cmpi sge, %104, %c2_i32 : i32
      %106 = arith.subi %104, %c2_i32 : i32
      %107 = arith.select %105, %106, %104 : i32
      memref.store %107, %_anonymous6[%c1] : memref<3xi32>
      %108 = arith.addi %88, %c1 : index
      cf.br ^bb47(%108 : index)
    ^bb57:  // pred: ^bb47
      %109 = arith.addi %86, %c1 : index
      cf.br ^bb45(%109 : index)
    ^bb58:  // pred: ^bb45
      aie.use_lock(%C_L1L2_6_0_cons_lock_0, Release, 1)
      %110 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %111 = arith.addi %110, %c1_i32 : i32
      %112 = arith.cmpi sge, %111, %c1_i32 : i32
      %113 = arith.select %112, %110, %111 : i32
      memref.store %113, %_anonymous6[%c0] : memref<3xi32>
      %114 = arith.addi %0, %c1 : index
      cf.br ^bb1(%114 : index)
    ^bb59:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    %_anonymous7 = aie.buffer(%tile_7_2) {address = 32000 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
    %core_7_2 = aie.core(%tile_7_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c8 = arith.constant 8 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb58
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb59
    ^bb2:  // pred: ^bb1
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c8 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb10(%9: memref<64x128xi8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %9, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous7[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb17(%c0 : index)
    ^bb17(%30: index):  // 2 preds: ^bb16, ^bb29
      %31 = arith.cmpi slt, %30, %c2 : index
      cf.cond_br %31, ^bb18, ^bb30
    ^bb18:  // pred: ^bb17
      cf.br ^bb19(%c0 : index)
    ^bb19(%32: index):  // 2 preds: ^bb18, ^bb28
      %33 = arith.cmpi slt, %32, %c8 : index
      cf.cond_br %33, ^bb20, ^bb29
    ^bb20:  // pred: ^bb19
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %34 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.index_cast %35 : index to i32
      cf.switch %36 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb24(%37: memref<64x128xi8>):  // 3 preds: ^bb21, ^bb22, ^bb23
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %38 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.index_cast %39 : index to i32
      cf.switch %40 : i32, [
        default: ^bb27,
        0: ^bb25,
        1: ^bb26
      ]
    ^bb25:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb26:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb27:  // pred: ^bb24
      cf.br ^bb28(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb28(%41: memref<32x64xi8>):  // 3 preds: ^bb25, ^bb26, ^bb27
      func.call @matmul_scalar_i8_i16(%41, %37, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %47 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %48 = arith.addi %47, %c1_i32 : i32
      %49 = arith.cmpi sge, %48, %c2_i32 : i32
      %50 = arith.subi %48, %c2_i32 : i32
      %51 = arith.select %49, %50, %48 : i32
      memref.store %51, %_anonymous7[%c1] : memref<3xi32>
      %52 = arith.addi %32, %c1 : index
      cf.br ^bb19(%52 : index)
    ^bb29:  // pred: ^bb19
      %53 = arith.addi %30, %c1 : index
      cf.br ^bb17(%53 : index)
    ^bb30:  // pred: ^bb17
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %54 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %55 = arith.addi %54, %c1_i32 : i32
      %56 = arith.cmpi sge, %55, %c1_i32 : i32
      %57 = arith.select %56, %54, %55 : i32
      memref.store %57, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb31(%c0 : index)
    ^bb31(%58: index):  // 2 preds: ^bb30, ^bb43
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb32, ^bb44
    ^bb32:  // pred: ^bb31
      cf.br ^bb33(%c0 : index)
    ^bb33(%60: index):  // 2 preds: ^bb32, ^bb42
      %61 = arith.cmpi slt, %60, %c8 : index
      cf.cond_br %61, ^bb34, ^bb43
    ^bb34:  // pred: ^bb33
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %62 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %63 = arith.index_cast %62 : i32 to index
      %64 = arith.index_cast %63 : index to i32
      cf.switch %64 : i32, [
        default: ^bb37,
        0: ^bb35,
        1: ^bb36
      ]
    ^bb35:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb36:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_7_cons_buff_1 : memref<64x128xi8>)
    ^bb37:  // pred: ^bb34
      cf.br ^bb38(%B_L2L1_7_cons_buff_0 : memref<64x128xi8>)
    ^bb38(%65: memref<64x128xi8>):  // 3 preds: ^bb35, ^bb36, ^bb37
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %66 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %67 = arith.index_cast %66 : i32 to index
      %68 = arith.index_cast %67 : index to i32
      cf.switch %68 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>)
    ^bb42(%69: memref<32x64xi8>):  // 3 preds: ^bb39, ^bb40, ^bb41
      func.call @matmul_scalar_i8_i16(%69, %65, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %70 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %71 = arith.addi %70, %c1_i32 : i32
      %72 = arith.cmpi sge, %71, %c2_i32 : i32
      %73 = arith.subi %71, %c2_i32 : i32
      %74 = arith.select %72, %73, %71 : i32
      memref.store %74, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %75 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %76 = arith.addi %75, %c1_i32 : i32
      %77 = arith.cmpi sge, %76, %c2_i32 : i32
      %78 = arith.subi %76, %c2_i32 : i32
      %79 = arith.select %77, %78, %76 : i32
      memref.store %79, %_anonymous7[%c1] : memref<3xi32>
      %80 = arith.addi %60, %c1 : index
      cf.br ^bb33(%80 : index)
    ^bb43:  // pred: ^bb33
      %81 = arith.addi %58, %c1 : index
      cf.br ^bb31(%81 : index)
    ^bb44:  // pred: ^bb31
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %82 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %83 = arith.addi %82, %c1_i32 : i32
      %84 = arith.cmpi sge, %83, %c1_i32 : i32
      %85 = arith.select %84, %82, %83 : i32
      memref.store %85, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%C_L1L2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%C_L1L2_7_0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb45(%c0 : index)
    ^bb45(%86: index):  // 2 preds: ^bb44, ^bb57
      %87 = arith.cmpi slt, %86, %c2 : index
      cf.cond_br %87, ^bb46, ^bb58
    ^bb46:  // pred: ^bb45
      cf.br ^bb47(%c0 : index)
    ^bb47(%88: index):  // 2 preds: ^bb46, ^bb56
      %89 = arith.cmpi slt, %88, %c8 : index
      cf.cond_br %89, ^bb48, ^bb57
    ^bb48:  // pred: ^bb47
      aie.use_lock(%B_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %90 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %91 = arith.index_cast %90 : i32 to index
      %92 = arith.index_cast %91 : index to i32
      cf.switch %92 : i32, [
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
    ^bb52(%93: memref<64x128xi8>):  // 3 preds: ^bb49, ^bb50, ^bb51
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %94 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %95 = arith.index_cast %94 : i32 to index
      %96 = arith.index_cast %95 : index to i32
      cf.switch %96 : i32, [
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
    ^bb56(%97: memref<32x64xi8>):  // 3 preds: ^bb53, ^bb54, ^bb55
      func.call @matmul_scalar_i8_i16(%97, %93, %C_L1L2_7_0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, Release, 1)
      %98 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c2_i32 : i32
      %101 = arith.subi %99, %c2_i32 : i32
      %102 = arith.select %100, %101, %99 : i32
      memref.store %102, %_anonymous7[%c2] : memref<3xi32>
      aie.use_lock(%B_L2L1_7_cons_prod_lock_0, Release, 1)
      %103 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %104 = arith.addi %103, %c1_i32 : i32
      %105 = arith.cmpi sge, %104, %c2_i32 : i32
      %106 = arith.subi %104, %c2_i32 : i32
      %107 = arith.select %105, %106, %104 : i32
      memref.store %107, %_anonymous7[%c1] : memref<3xi32>
      %108 = arith.addi %88, %c1 : index
      cf.br ^bb47(%108 : index)
    ^bb57:  // pred: ^bb47
      %109 = arith.addi %86, %c1 : index
      cf.br ^bb45(%109 : index)
    ^bb58:  // pred: ^bb45
      aie.use_lock(%C_L1L2_7_0_cons_lock_0, Release, 1)
      %110 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %111 = arith.addi %110, %c1_i32 : i32
      %112 = arith.cmpi sge, %111, %c1_i32 : i32
      %113 = arith.select %112, %110, %111 : i32
      memref.store %113, %_anonymous7[%c0] : memref<3xi32>
      %114 = arith.addi %0, %c1 : index
      cf.br ^bb1(%114 : index)
    ^bb59:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"], stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<131072xi8>, %arg1: memref<1048576xi8>, %arg2: memref<131072xi16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 128, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 256, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 384, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 512, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 640, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 32768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 32896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33024, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33152, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33280, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33408, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65792, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65920, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66048, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66176, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @B_L3L2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @B_L3L2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @B_L3L2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @B_L3L2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98560, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98688, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @C_L2L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98816, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @C_L2L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98944, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @C_L2L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 99072, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @C_L2L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 99200, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
      aiex.dma_await_task(%26)
      aiex.dma_await_task(%27)
      aiex.dma_await_task(%28)
      aiex.dma_await_task(%29)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%31)
      aiex.dma_await_task(%32)
      aiex.dma_await_task(%33)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
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
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<32x512xi8>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<32x512xi8>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 2 : i32}
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
      aie.dma_bd(%A_L2L1_0_0_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_0_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%A_L2L1_0_1_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_1_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%A_L2L1_0_2_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_2_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%A_L2L1_0_3_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_3_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%A_L2L1_0_4_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_4_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%A_L2L1_0_5_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_5_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%A_L2L1_0_6_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_6_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
      aie.dma_bd(%A_L2L1_0_7_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_7_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
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
