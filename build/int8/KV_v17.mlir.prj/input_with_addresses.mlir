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
    %CS7_cons_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 4) {init = 0 : i32, sym_name = "CS7_cons_prod_lock_0"}
    %CS7_cons_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 5) {init = 0 : i32, sym_name = "CS7_cons_cons_lock_0"}
    %CC7_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 32768 : i32, sym_name = "CC7_cons_buff_0"} : memref<32x128xi16> 
    %CC7_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 4) {init = 1 : i32, sym_name = "CC7_cons_prod_lock_0"}
    %CC7_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 5) {init = 0 : i32, sym_name = "CC7_cons_cons_lock_0"}
    %CC7_buff_0 = aie.buffer(%tile_7_2) {address = 1024 : i32, sym_name = "CC7_buff_0"} : memref<32x128xi16> 
    %CC7_prod_lock_0 = aie.lock(%tile_7_2, 4) {init = 1 : i32, sym_name = "CC7_prod_lock_0"}
    %CC7_cons_lock_0 = aie.lock(%tile_7_2, 5) {init = 0 : i32, sym_name = "CC7_cons_lock_0"}
    %BC7_cons_buff_0 = aie.buffer(%tile_7_2) {address = 9216 : i32, sym_name = "BC7_cons_buff_0"} : memref<64x128xi8> 
    %BC7_cons_buff_1 = aie.buffer(%tile_7_2) {address = 17408 : i32, sym_name = "BC7_cons_buff_1"} : memref<64x128xi8> 
    %BC7_cons_prod_lock_0 = aie.lock(%tile_7_2, 2) {init = 2 : i32, sym_name = "BC7_cons_prod_lock_0"}
    %BC7_cons_cons_lock_0 = aie.lock(%tile_7_2, 3) {init = 0 : i32, sym_name = "BC7_cons_cons_lock_0"}
    %BS7_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 40960 : i32, sym_name = "BS7_cons_buff_0"} : memref<64x128xi8> 
    %BS7_cons_buff_1 = aie.buffer(%mem_tile_7_1) {address = 49152 : i32, sym_name = "BS7_cons_buff_1"} : memref<64x128xi8> 
    %BS7_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 2) {init = 2 : i32, sym_name = "BS7_cons_prod_lock_0"}
    %BS7_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 3) {init = 0 : i32, sym_name = "BS7_cons_cons_lock_0"}
    %BS7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "BS7_prod_lock_0"}
    %BS7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "BS7_cons_lock_0"}
    %AC7_cons_buff_0 = aie.buffer(%tile_7_2) {address = 25600 : i32, sym_name = "AC7_cons_buff_0"} : memref<32x64xi8> 
    %AC7_cons_buff_1 = aie.buffer(%tile_7_2) {address = 27648 : i32, sym_name = "AC7_cons_buff_1"} : memref<32x64xi8> 
    %AC7_cons_prod_lock_0 = aie.lock(%tile_7_2, 0) {init = 2 : i32, sym_name = "AC7_cons_prod_lock_0"}
    %AC7_cons_cons_lock_0 = aie.lock(%tile_7_2, 1) {init = 0 : i32, sym_name = "AC7_cons_cons_lock_0"}
    %AS7_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 0 : i32, sym_name = "AS7_cons_buff_0"} : memref<32x512xi8> 
    %AS7_cons_buff_1 = aie.buffer(%mem_tile_7_1) {address = 16384 : i32, sym_name = "AS7_cons_buff_1"} : memref<32x512xi8> 
    %AS7_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 0) {init = 2 : i32, sym_name = "AS7_cons_prod_lock_0"}
    %AS7_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 1) {init = 0 : i32, sym_name = "AS7_cons_cons_lock_0"}
    %AS7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "AS7_prod_lock_0"}
    %AS7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "AS7_cons_lock_0"}
    %CS6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 4) {init = 0 : i32, sym_name = "CS6_cons_prod_lock_0"}
    %CS6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 5) {init = 0 : i32, sym_name = "CS6_cons_cons_lock_0"}
    %CC6_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 32768 : i32, sym_name = "CC6_cons_buff_0"} : memref<32x128xi16> 
    %CC6_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 4) {init = 1 : i32, sym_name = "CC6_cons_prod_lock_0"}
    %CC6_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 5) {init = 0 : i32, sym_name = "CC6_cons_cons_lock_0"}
    %CC6_buff_0 = aie.buffer(%tile_6_2) {address = 1024 : i32, sym_name = "CC6_buff_0"} : memref<32x128xi16> 
    %CC6_prod_lock_0 = aie.lock(%tile_6_2, 4) {init = 1 : i32, sym_name = "CC6_prod_lock_0"}
    %CC6_cons_lock_0 = aie.lock(%tile_6_2, 5) {init = 0 : i32, sym_name = "CC6_cons_lock_0"}
    %BC6_cons_buff_0 = aie.buffer(%tile_6_2) {address = 9216 : i32, sym_name = "BC6_cons_buff_0"} : memref<64x128xi8> 
    %BC6_cons_buff_1 = aie.buffer(%tile_6_2) {address = 17408 : i32, sym_name = "BC6_cons_buff_1"} : memref<64x128xi8> 
    %BC6_cons_prod_lock_0 = aie.lock(%tile_6_2, 2) {init = 2 : i32, sym_name = "BC6_cons_prod_lock_0"}
    %BC6_cons_cons_lock_0 = aie.lock(%tile_6_2, 3) {init = 0 : i32, sym_name = "BC6_cons_cons_lock_0"}
    %BS6_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 40960 : i32, sym_name = "BS6_cons_buff_0"} : memref<64x128xi8> 
    %BS6_cons_buff_1 = aie.buffer(%mem_tile_6_1) {address = 49152 : i32, sym_name = "BS6_cons_buff_1"} : memref<64x128xi8> 
    %BS6_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 2) {init = 2 : i32, sym_name = "BS6_cons_prod_lock_0"}
    %BS6_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 3) {init = 0 : i32, sym_name = "BS6_cons_cons_lock_0"}
    %BS6_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "BS6_prod_lock_0"}
    %BS6_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "BS6_cons_lock_0"}
    %AC6_cons_buff_0 = aie.buffer(%tile_6_2) {address = 25600 : i32, sym_name = "AC6_cons_buff_0"} : memref<32x64xi8> 
    %AC6_cons_buff_1 = aie.buffer(%tile_6_2) {address = 27648 : i32, sym_name = "AC6_cons_buff_1"} : memref<32x64xi8> 
    %AC6_cons_prod_lock_0 = aie.lock(%tile_6_2, 0) {init = 2 : i32, sym_name = "AC6_cons_prod_lock_0"}
    %AC6_cons_cons_lock_0 = aie.lock(%tile_6_2, 1) {init = 0 : i32, sym_name = "AC6_cons_cons_lock_0"}
    %AS6_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 0 : i32, sym_name = "AS6_cons_buff_0"} : memref<32x512xi8> 
    %AS6_cons_buff_1 = aie.buffer(%mem_tile_6_1) {address = 16384 : i32, sym_name = "AS6_cons_buff_1"} : memref<32x512xi8> 
    %AS6_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 0) {init = 2 : i32, sym_name = "AS6_cons_prod_lock_0"}
    %AS6_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 1) {init = 0 : i32, sym_name = "AS6_cons_cons_lock_0"}
    %AS6_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "AS6_prod_lock_0"}
    %AS6_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "AS6_cons_lock_0"}
    %CS5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 4) {init = 0 : i32, sym_name = "CS5_cons_prod_lock_0"}
    %CS5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 5) {init = 0 : i32, sym_name = "CS5_cons_cons_lock_0"}
    %CC5_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 32768 : i32, sym_name = "CC5_cons_buff_0"} : memref<32x128xi16> 
    %CC5_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 4) {init = 1 : i32, sym_name = "CC5_cons_prod_lock_0"}
    %CC5_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 5) {init = 0 : i32, sym_name = "CC5_cons_cons_lock_0"}
    %CC5_buff_0 = aie.buffer(%tile_5_2) {address = 1024 : i32, sym_name = "CC5_buff_0"} : memref<32x128xi16> 
    %CC5_prod_lock_0 = aie.lock(%tile_5_2, 4) {init = 1 : i32, sym_name = "CC5_prod_lock_0"}
    %CC5_cons_lock_0 = aie.lock(%tile_5_2, 5) {init = 0 : i32, sym_name = "CC5_cons_lock_0"}
    %BC5_cons_buff_0 = aie.buffer(%tile_5_2) {address = 9216 : i32, sym_name = "BC5_cons_buff_0"} : memref<64x128xi8> 
    %BC5_cons_buff_1 = aie.buffer(%tile_5_2) {address = 17408 : i32, sym_name = "BC5_cons_buff_1"} : memref<64x128xi8> 
    %BC5_cons_prod_lock_0 = aie.lock(%tile_5_2, 2) {init = 2 : i32, sym_name = "BC5_cons_prod_lock_0"}
    %BC5_cons_cons_lock_0 = aie.lock(%tile_5_2, 3) {init = 0 : i32, sym_name = "BC5_cons_cons_lock_0"}
    %BS5_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 40960 : i32, sym_name = "BS5_cons_buff_0"} : memref<64x128xi8> 
    %BS5_cons_buff_1 = aie.buffer(%mem_tile_5_1) {address = 49152 : i32, sym_name = "BS5_cons_buff_1"} : memref<64x128xi8> 
    %BS5_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 2) {init = 2 : i32, sym_name = "BS5_cons_prod_lock_0"}
    %BS5_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 3) {init = 0 : i32, sym_name = "BS5_cons_cons_lock_0"}
    %BS5_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "BS5_prod_lock_0"}
    %BS5_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "BS5_cons_lock_0"}
    %AC5_cons_buff_0 = aie.buffer(%tile_5_2) {address = 25600 : i32, sym_name = "AC5_cons_buff_0"} : memref<32x64xi8> 
    %AC5_cons_buff_1 = aie.buffer(%tile_5_2) {address = 27648 : i32, sym_name = "AC5_cons_buff_1"} : memref<32x64xi8> 
    %AC5_cons_prod_lock_0 = aie.lock(%tile_5_2, 0) {init = 2 : i32, sym_name = "AC5_cons_prod_lock_0"}
    %AC5_cons_cons_lock_0 = aie.lock(%tile_5_2, 1) {init = 0 : i32, sym_name = "AC5_cons_cons_lock_0"}
    %AS5_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 0 : i32, sym_name = "AS5_cons_buff_0"} : memref<32x512xi8> 
    %AS5_cons_buff_1 = aie.buffer(%mem_tile_5_1) {address = 16384 : i32, sym_name = "AS5_cons_buff_1"} : memref<32x512xi8> 
    %AS5_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 0) {init = 2 : i32, sym_name = "AS5_cons_prod_lock_0"}
    %AS5_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 1) {init = 0 : i32, sym_name = "AS5_cons_cons_lock_0"}
    %AS5_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "AS5_prod_lock_0"}
    %AS5_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "AS5_cons_lock_0"}
    %CS4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 4) {init = 0 : i32, sym_name = "CS4_cons_prod_lock_0"}
    %CS4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 5) {init = 0 : i32, sym_name = "CS4_cons_cons_lock_0"}
    %CC4_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 32768 : i32, sym_name = "CC4_cons_buff_0"} : memref<32x128xi16> 
    %CC4_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 4) {init = 1 : i32, sym_name = "CC4_cons_prod_lock_0"}
    %CC4_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 5) {init = 0 : i32, sym_name = "CC4_cons_cons_lock_0"}
    %CC4_buff_0 = aie.buffer(%tile_4_2) {address = 1024 : i32, sym_name = "CC4_buff_0"} : memref<32x128xi16> 
    %CC4_prod_lock_0 = aie.lock(%tile_4_2, 4) {init = 1 : i32, sym_name = "CC4_prod_lock_0"}
    %CC4_cons_lock_0 = aie.lock(%tile_4_2, 5) {init = 0 : i32, sym_name = "CC4_cons_lock_0"}
    %BC4_cons_buff_0 = aie.buffer(%tile_4_2) {address = 9216 : i32, sym_name = "BC4_cons_buff_0"} : memref<64x128xi8> 
    %BC4_cons_buff_1 = aie.buffer(%tile_4_2) {address = 17408 : i32, sym_name = "BC4_cons_buff_1"} : memref<64x128xi8> 
    %BC4_cons_prod_lock_0 = aie.lock(%tile_4_2, 2) {init = 2 : i32, sym_name = "BC4_cons_prod_lock_0"}
    %BC4_cons_cons_lock_0 = aie.lock(%tile_4_2, 3) {init = 0 : i32, sym_name = "BC4_cons_cons_lock_0"}
    %BS4_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 40960 : i32, sym_name = "BS4_cons_buff_0"} : memref<64x128xi8> 
    %BS4_cons_buff_1 = aie.buffer(%mem_tile_4_1) {address = 49152 : i32, sym_name = "BS4_cons_buff_1"} : memref<64x128xi8> 
    %BS4_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 2) {init = 2 : i32, sym_name = "BS4_cons_prod_lock_0"}
    %BS4_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 3) {init = 0 : i32, sym_name = "BS4_cons_cons_lock_0"}
    %BS4_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "BS4_prod_lock_0"}
    %BS4_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "BS4_cons_lock_0"}
    %AC4_cons_buff_0 = aie.buffer(%tile_4_2) {address = 25600 : i32, sym_name = "AC4_cons_buff_0"} : memref<32x64xi8> 
    %AC4_cons_buff_1 = aie.buffer(%tile_4_2) {address = 27648 : i32, sym_name = "AC4_cons_buff_1"} : memref<32x64xi8> 
    %AC4_cons_prod_lock_0 = aie.lock(%tile_4_2, 0) {init = 2 : i32, sym_name = "AC4_cons_prod_lock_0"}
    %AC4_cons_cons_lock_0 = aie.lock(%tile_4_2, 1) {init = 0 : i32, sym_name = "AC4_cons_cons_lock_0"}
    %AS4_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 0 : i32, sym_name = "AS4_cons_buff_0"} : memref<32x512xi8> 
    %AS4_cons_buff_1 = aie.buffer(%mem_tile_4_1) {address = 16384 : i32, sym_name = "AS4_cons_buff_1"} : memref<32x512xi8> 
    %AS4_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 0) {init = 2 : i32, sym_name = "AS4_cons_prod_lock_0"}
    %AS4_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 1) {init = 0 : i32, sym_name = "AS4_cons_cons_lock_0"}
    %AS4_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "AS4_prod_lock_0"}
    %AS4_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "AS4_cons_lock_0"}
    %CS3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 4) {init = 0 : i32, sym_name = "CS3_cons_prod_lock_0"}
    %CS3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 5) {init = 0 : i32, sym_name = "CS3_cons_cons_lock_0"}
    %CC3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 32768 : i32, sym_name = "CC3_cons_buff_0"} : memref<32x128xi16> 
    %CC3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 4) {init = 1 : i32, sym_name = "CC3_cons_prod_lock_0"}
    %CC3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 5) {init = 0 : i32, sym_name = "CC3_cons_cons_lock_0"}
    %CC3_buff_0 = aie.buffer(%tile_3_2) {address = 1024 : i32, sym_name = "CC3_buff_0"} : memref<32x128xi16> 
    %CC3_prod_lock_0 = aie.lock(%tile_3_2, 4) {init = 1 : i32, sym_name = "CC3_prod_lock_0"}
    %CC3_cons_lock_0 = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "CC3_cons_lock_0"}
    %BC3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 9216 : i32, sym_name = "BC3_cons_buff_0"} : memref<64x128xi8> 
    %BC3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 17408 : i32, sym_name = "BC3_cons_buff_1"} : memref<64x128xi8> 
    %BC3_cons_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "BC3_cons_prod_lock_0"}
    %BC3_cons_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "BC3_cons_cons_lock_0"}
    %BS3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 40960 : i32, sym_name = "BS3_cons_buff_0"} : memref<64x128xi8> 
    %BS3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 49152 : i32, sym_name = "BS3_cons_buff_1"} : memref<64x128xi8> 
    %BS3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 2) {init = 2 : i32, sym_name = "BS3_cons_prod_lock_0"}
    %BS3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "BS3_cons_cons_lock_0"}
    %BS3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "BS3_prod_lock_0"}
    %BS3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "BS3_cons_lock_0"}
    %AC3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 25600 : i32, sym_name = "AC3_cons_buff_0"} : memref<32x64xi8> 
    %AC3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 27648 : i32, sym_name = "AC3_cons_buff_1"} : memref<32x64xi8> 
    %AC3_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "AC3_cons_prod_lock_0"}
    %AC3_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "AC3_cons_cons_lock_0"}
    %AS3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, sym_name = "AS3_cons_buff_0"} : memref<32x512xi8> 
    %AS3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 16384 : i32, sym_name = "AS3_cons_buff_1"} : memref<32x512xi8> 
    %AS3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "AS3_cons_prod_lock_0"}
    %AS3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "AS3_cons_cons_lock_0"}
    %AS3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "AS3_prod_lock_0"}
    %AS3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "AS3_cons_lock_0"}
    %CS2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "CS2_cons_prod_lock_0"}
    %CS2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "CS2_cons_cons_lock_0"}
    %CC2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 32768 : i32, sym_name = "CC2_cons_buff_0"} : memref<32x128xi16> 
    %CC2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 4) {init = 1 : i32, sym_name = "CC2_cons_prod_lock_0"}
    %CC2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 5) {init = 0 : i32, sym_name = "CC2_cons_cons_lock_0"}
    %CC2_buff_0 = aie.buffer(%tile_2_2) {address = 1024 : i32, sym_name = "CC2_buff_0"} : memref<32x128xi16> 
    %CC2_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 1 : i32, sym_name = "CC2_prod_lock_0"}
    %CC2_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "CC2_cons_lock_0"}
    %BC2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 9216 : i32, sym_name = "BC2_cons_buff_0"} : memref<64x128xi8> 
    %BC2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 17408 : i32, sym_name = "BC2_cons_buff_1"} : memref<64x128xi8> 
    %BC2_cons_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "BC2_cons_prod_lock_0"}
    %BC2_cons_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "BC2_cons_cons_lock_0"}
    %BS2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 40960 : i32, sym_name = "BS2_cons_buff_0"} : memref<64x128xi8> 
    %BS2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 49152 : i32, sym_name = "BS2_cons_buff_1"} : memref<64x128xi8> 
    %BS2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 2) {init = 2 : i32, sym_name = "BS2_cons_prod_lock_0"}
    %BS2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "BS2_cons_cons_lock_0"}
    %BS2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "BS2_prod_lock_0"}
    %BS2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "BS2_cons_lock_0"}
    %AC2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 25600 : i32, sym_name = "AC2_cons_buff_0"} : memref<32x64xi8> 
    %AC2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 27648 : i32, sym_name = "AC2_cons_buff_1"} : memref<32x64xi8> 
    %AC2_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "AC2_cons_prod_lock_0"}
    %AC2_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "AC2_cons_cons_lock_0"}
    %AS2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "AS2_cons_buff_0"} : memref<32x512xi8> 
    %AS2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 16384 : i32, sym_name = "AS2_cons_buff_1"} : memref<32x512xi8> 
    %AS2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "AS2_cons_prod_lock_0"}
    %AS2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "AS2_cons_cons_lock_0"}
    %AS2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "AS2_prod_lock_0"}
    %AS2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "AS2_cons_lock_0"}
    %CS1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "CS1_cons_prod_lock_0"}
    %CS1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "CS1_cons_cons_lock_0"}
    %CC1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 32768 : i32, sym_name = "CC1_cons_buff_0"} : memref<32x128xi16> 
    %CC1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 4) {init = 1 : i32, sym_name = "CC1_cons_prod_lock_0"}
    %CC1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "CC1_cons_cons_lock_0"}
    %CC1_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, sym_name = "CC1_buff_0"} : memref<32x128xi16> 
    %CC1_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 1 : i32, sym_name = "CC1_prod_lock_0"}
    %CC1_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "CC1_cons_lock_0"}
    %BC1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 9216 : i32, sym_name = "BC1_cons_buff_0"} : memref<64x128xi8> 
    %BC1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 17408 : i32, sym_name = "BC1_cons_buff_1"} : memref<64x128xi8> 
    %BC1_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "BC1_cons_prod_lock_0"}
    %BC1_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "BC1_cons_cons_lock_0"}
    %BS1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 40960 : i32, sym_name = "BS1_cons_buff_0"} : memref<64x128xi8> 
    %BS1_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 49152 : i32, sym_name = "BS1_cons_buff_1"} : memref<64x128xi8> 
    %BS1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "BS1_cons_prod_lock_0"}
    %BS1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "BS1_cons_cons_lock_0"}
    %BS1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "BS1_prod_lock_0"}
    %BS1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "BS1_cons_lock_0"}
    %AC1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 25600 : i32, sym_name = "AC1_cons_buff_0"} : memref<32x64xi8> 
    %AC1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 27648 : i32, sym_name = "AC1_cons_buff_1"} : memref<32x64xi8> 
    %AC1_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "AC1_cons_prod_lock_0"}
    %AC1_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "AC1_cons_cons_lock_0"}
    %AS1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "AS1_cons_buff_0"} : memref<32x512xi8> 
    %AS1_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 16384 : i32, sym_name = "AS1_cons_buff_1"} : memref<32x512xi8> 
    %AS1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "AS1_cons_prod_lock_0"}
    %AS1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "AS1_cons_cons_lock_0"}
    %AS1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "AS1_prod_lock_0"}
    %AS1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "AS1_cons_lock_0"}
    %CS0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "CS0_cons_prod_lock_0"}
    %CS0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "CS0_cons_cons_lock_0"}
    %CC0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 32768 : i32, sym_name = "CC0_cons_buff_0"} : memref<32x128xi16> 
    %CC0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32, sym_name = "CC0_cons_prod_lock_0"}
    %CC0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "CC0_cons_cons_lock_0"}
    %CC0_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "CC0_buff_0"} : memref<32x128xi16> 
    %CC0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "CC0_prod_lock_0"}
    %CC0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "CC0_cons_lock_0"}
    %BC0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 9216 : i32, sym_name = "BC0_cons_buff_0"} : memref<64x128xi8> 
    %BC0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 17408 : i32, sym_name = "BC0_cons_buff_1"} : memref<64x128xi8> 
    %BC0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "BC0_cons_prod_lock_0"}
    %BC0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "BC0_cons_cons_lock_0"}
    %BS0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 40960 : i32, sym_name = "BS0_cons_buff_0"} : memref<64x128xi8> 
    %BS0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 49152 : i32, sym_name = "BS0_cons_buff_1"} : memref<64x128xi8> 
    %BS0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "BS0_cons_prod_lock_0"}
    %BS0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "BS0_cons_cons_lock_0"}
    %BS0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "BS0_prod_lock_0"}
    %BS0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "BS0_cons_lock_0"}
    %AC0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 25600 : i32, sym_name = "AC0_cons_buff_0"} : memref<32x64xi8> 
    %AC0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 27648 : i32, sym_name = "AC0_cons_buff_1"} : memref<32x64xi8> 
    %AC0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "AC0_cons_prod_lock_0"}
    %AC0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "AC0_cons_cons_lock_0"}
    %AS0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "AS0_cons_buff_0"} : memref<32x512xi8> 
    %AS0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 16384 : i32, sym_name = "AS0_cons_buff_1"} : memref<32x512xi8> 
    %AS0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "AS0_cons_prod_lock_0"}
    %AS0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "AS0_cons_cons_lock_0"}
    %AS0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "AS0_prod_lock_0"}
    %AS0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "AS0_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %mem_tile_1_1, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 2)
    aie.flow(%mem_tile_1_1, DMA : 2, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %mem_tile_2_1, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_2, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 2)
    aie.flow(%mem_tile_2_1, DMA : 2, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %mem_tile_3_1, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_2, DMA : 1)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 2)
    aie.flow(%mem_tile_3_1, DMA : 2, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_4_2, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 1, %mem_tile_4_1, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 1, %tile_4_2, DMA : 1)
    aie.flow(%tile_4_2, DMA : 0, %mem_tile_4_1, DMA : 2)
    aie.flow(%mem_tile_4_1, DMA : 2, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_5_2, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 1, %mem_tile_5_1, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 1, %tile_5_2, DMA : 1)
    aie.flow(%tile_5_2, DMA : 0, %mem_tile_5_1, DMA : 2)
    aie.flow(%mem_tile_5_1, DMA : 2, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_6_2, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 1, %mem_tile_6_1, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_6_2, DMA : 1)
    aie.flow(%tile_6_2, DMA : 0, %mem_tile_6_1, DMA : 2)
    aie.flow(%mem_tile_6_1, DMA : 2, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_7_2, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 1, %mem_tile_7_1, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_7_2, DMA : 1)
    aie.flow(%tile_7_2, DMA : 0, %mem_tile_7_1, DMA : 2)
    aie.flow(%mem_tile_7_1, DMA : 2, %shim_noc_tile_7_0, DMA : 0)
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 29696 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb46
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb47
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
    ^bb4:  // pred: ^bb3
      aie.use_lock(%AC0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%AC0_cons_buff_0 : memref<32x64xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%AC0_cons_buff_1 : memref<32x64xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%AC0_cons_buff_0 : memref<32x64xi8>)
    ^bb8(%7: memref<32x64xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%BC0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%BC0_cons_buff_0 : memref<64x128xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%BC0_cons_buff_1 : memref<64x128xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%BC0_cons_buff_0 : memref<64x128xi8>)
    ^bb12(%11: memref<64x128xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_scalar_i8_i16(%7, %11, %CC0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC0_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%BC0_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous0[%c2] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%CC0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%CC0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%AC0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%AC0_cons_buff_0 : memref<32x64xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%AC0_cons_buff_1 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%AC0_cons_buff_0 : memref<32x64xi8>)
    ^bb19(%32: memref<32x64xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%BC0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%BC0_cons_buff_0 : memref<64x128xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%BC0_cons_buff_1 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%BC0_cons_buff_0 : memref<64x128xi8>)
    ^bb23(%36: memref<64x128xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_scalar_i8_i16(%32, %36, %CC0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC0_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%BC0_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous0[%c2] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%CC0_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%CC0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%AC0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%AC0_cons_buff_0 : memref<32x64xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%AC0_cons_buff_1 : memref<32x64xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%AC0_cons_buff_0 : memref<32x64xi8>)
    ^bb30(%57: memref<32x64xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%BC0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%BC0_cons_buff_0 : memref<64x128xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%BC0_cons_buff_1 : memref<64x128xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%BC0_cons_buff_0 : memref<64x128xi8>)
    ^bb34(%61: memref<64x128xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_scalar_i8_i16(%57, %61, %CC0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC0_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%BC0_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous0[%c2] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%CC0_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%CC0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC0_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%AC0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%AC0_cons_buff_0 : memref<32x64xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%AC0_cons_buff_1 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%AC0_cons_buff_0 : memref<32x64xi8>)
    ^bb41(%82: memref<32x64xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%BC0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%BC0_cons_buff_0 : memref<64x128xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%BC0_cons_buff_1 : memref<64x128xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%BC0_cons_buff_0 : memref<64x128xi8>)
    ^bb45(%86: memref<64x128xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_scalar_i8_i16(%82, %86, %CC0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC0_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%BC0_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous0[%c2] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%CC0_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous0[%c0] : memref<3xi32>
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb47:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"]}
    %_anonymous1 = aie.buffer(%tile_1_2) {address = 29696 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb46
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb47
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC1_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
    ^bb4:  // pred: ^bb3
      aie.use_lock(%AC1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%AC1_cons_buff_0 : memref<32x64xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%AC1_cons_buff_1 : memref<32x64xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%AC1_cons_buff_0 : memref<32x64xi8>)
    ^bb8(%7: memref<32x64xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%BC1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%BC1_cons_buff_0 : memref<64x128xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%BC1_cons_buff_1 : memref<64x128xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%BC1_cons_buff_0 : memref<64x128xi8>)
    ^bb12(%11: memref<64x128xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_scalar_i8_i16(%7, %11, %CC1_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC1_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%BC1_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous1[%c2] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%CC1_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%CC1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC1_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%AC1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%AC1_cons_buff_0 : memref<32x64xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%AC1_cons_buff_1 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%AC1_cons_buff_0 : memref<32x64xi8>)
    ^bb19(%32: memref<32x64xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%BC1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%BC1_cons_buff_0 : memref<64x128xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%BC1_cons_buff_1 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%BC1_cons_buff_0 : memref<64x128xi8>)
    ^bb23(%36: memref<64x128xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_scalar_i8_i16(%32, %36, %CC1_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC1_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%BC1_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous1[%c2] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%CC1_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%CC1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC1_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%AC1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%AC1_cons_buff_0 : memref<32x64xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%AC1_cons_buff_1 : memref<32x64xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%AC1_cons_buff_0 : memref<32x64xi8>)
    ^bb30(%57: memref<32x64xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%BC1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%BC1_cons_buff_0 : memref<64x128xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%BC1_cons_buff_1 : memref<64x128xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%BC1_cons_buff_0 : memref<64x128xi8>)
    ^bb34(%61: memref<64x128xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_scalar_i8_i16(%57, %61, %CC1_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC1_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%BC1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous1[%c2] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%CC1_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%CC1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC1_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%AC1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%AC1_cons_buff_0 : memref<32x64xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%AC1_cons_buff_1 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%AC1_cons_buff_0 : memref<32x64xi8>)
    ^bb41(%82: memref<32x64xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%BC1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%BC1_cons_buff_0 : memref<64x128xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%BC1_cons_buff_1 : memref<64x128xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%BC1_cons_buff_0 : memref<64x128xi8>)
    ^bb45(%86: memref<64x128xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_scalar_i8_i16(%82, %86, %CC1_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC1_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%BC1_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous1[%c2] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%CC1_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous1[%c0] : memref<3xi32>
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb47:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"]}
    %_anonymous2 = aie.buffer(%tile_2_2) {address = 29696 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb46
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb47
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
    ^bb4:  // pred: ^bb3
      aie.use_lock(%AC2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%AC2_cons_buff_0 : memref<32x64xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%AC2_cons_buff_1 : memref<32x64xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%AC2_cons_buff_0 : memref<32x64xi8>)
    ^bb8(%7: memref<32x64xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%BC2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%BC2_cons_buff_0 : memref<64x128xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%BC2_cons_buff_1 : memref<64x128xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%BC2_cons_buff_0 : memref<64x128xi8>)
    ^bb12(%11: memref<64x128xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_scalar_i8_i16(%7, %11, %CC2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC2_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous2[%c1] : memref<3xi32>
      aie.use_lock(%BC2_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous2[%c2] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%CC2_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%CC2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%AC2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%AC2_cons_buff_0 : memref<32x64xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%AC2_cons_buff_1 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%AC2_cons_buff_0 : memref<32x64xi8>)
    ^bb19(%32: memref<32x64xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%BC2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%BC2_cons_buff_0 : memref<64x128xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%BC2_cons_buff_1 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%BC2_cons_buff_0 : memref<64x128xi8>)
    ^bb23(%36: memref<64x128xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_scalar_i8_i16(%32, %36, %CC2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC2_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous2[%c1] : memref<3xi32>
      aie.use_lock(%BC2_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous2[%c2] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%CC2_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%CC2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%AC2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%AC2_cons_buff_0 : memref<32x64xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%AC2_cons_buff_1 : memref<32x64xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%AC2_cons_buff_0 : memref<32x64xi8>)
    ^bb30(%57: memref<32x64xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%BC2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%BC2_cons_buff_0 : memref<64x128xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%BC2_cons_buff_1 : memref<64x128xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%BC2_cons_buff_0 : memref<64x128xi8>)
    ^bb34(%61: memref<64x128xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_scalar_i8_i16(%57, %61, %CC2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC2_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous2[%c1] : memref<3xi32>
      aie.use_lock(%BC2_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous2[%c2] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%CC2_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%CC2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC2_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%AC2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%AC2_cons_buff_0 : memref<32x64xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%AC2_cons_buff_1 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%AC2_cons_buff_0 : memref<32x64xi8>)
    ^bb41(%82: memref<32x64xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%BC2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%BC2_cons_buff_0 : memref<64x128xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%BC2_cons_buff_1 : memref<64x128xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%BC2_cons_buff_0 : memref<64x128xi8>)
    ^bb45(%86: memref<64x128xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_scalar_i8_i16(%82, %86, %CC2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC2_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous2[%c1] : memref<3xi32>
      aie.use_lock(%BC2_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous2[%c2] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%CC2_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous2[%c0] : memref<3xi32>
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb47:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"]}
    %_anonymous3 = aie.buffer(%tile_3_2) {address = 29696 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb46
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb47
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC3_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
    ^bb4:  // pred: ^bb3
      aie.use_lock(%AC3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%AC3_cons_buff_0 : memref<32x64xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%AC3_cons_buff_1 : memref<32x64xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%AC3_cons_buff_0 : memref<32x64xi8>)
    ^bb8(%7: memref<32x64xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%BC3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%BC3_cons_buff_0 : memref<64x128xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%BC3_cons_buff_1 : memref<64x128xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%BC3_cons_buff_0 : memref<64x128xi8>)
    ^bb12(%11: memref<64x128xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_scalar_i8_i16(%7, %11, %CC3_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC3_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous3[%c1] : memref<3xi32>
      aie.use_lock(%BC3_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous3[%c2] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%CC3_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%CC3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC3_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%AC3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%AC3_cons_buff_0 : memref<32x64xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%AC3_cons_buff_1 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%AC3_cons_buff_0 : memref<32x64xi8>)
    ^bb19(%32: memref<32x64xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%BC3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%BC3_cons_buff_0 : memref<64x128xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%BC3_cons_buff_1 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%BC3_cons_buff_0 : memref<64x128xi8>)
    ^bb23(%36: memref<64x128xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_scalar_i8_i16(%32, %36, %CC3_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC3_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous3[%c1] : memref<3xi32>
      aie.use_lock(%BC3_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous3[%c2] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%CC3_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%CC3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC3_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%AC3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%AC3_cons_buff_0 : memref<32x64xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%AC3_cons_buff_1 : memref<32x64xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%AC3_cons_buff_0 : memref<32x64xi8>)
    ^bb30(%57: memref<32x64xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%BC3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%BC3_cons_buff_0 : memref<64x128xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%BC3_cons_buff_1 : memref<64x128xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%BC3_cons_buff_0 : memref<64x128xi8>)
    ^bb34(%61: memref<64x128xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_scalar_i8_i16(%57, %61, %CC3_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC3_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous3[%c1] : memref<3xi32>
      aie.use_lock(%BC3_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous3[%c2] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%CC3_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%CC3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC3_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%AC3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%AC3_cons_buff_0 : memref<32x64xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%AC3_cons_buff_1 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%AC3_cons_buff_0 : memref<32x64xi8>)
    ^bb41(%82: memref<32x64xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%BC3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%BC3_cons_buff_0 : memref<64x128xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%BC3_cons_buff_1 : memref<64x128xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%BC3_cons_buff_0 : memref<64x128xi8>)
    ^bb45(%86: memref<64x128xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_scalar_i8_i16(%82, %86, %CC3_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC3_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous3[%c1] : memref<3xi32>
      aie.use_lock(%BC3_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous3[%c2] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%CC3_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous3[%c0] : memref<3xi32>
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb47:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"]}
    %_anonymous4 = aie.buffer(%tile_4_2) {address = 29696 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb46
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb47
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC4_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
    ^bb4:  // pred: ^bb3
      aie.use_lock(%AC4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%AC4_cons_buff_0 : memref<32x64xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%AC4_cons_buff_1 : memref<32x64xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%AC4_cons_buff_0 : memref<32x64xi8>)
    ^bb8(%7: memref<32x64xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%BC4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%BC4_cons_buff_0 : memref<64x128xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%BC4_cons_buff_1 : memref<64x128xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%BC4_cons_buff_0 : memref<64x128xi8>)
    ^bb12(%11: memref<64x128xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_scalar_i8_i16(%7, %11, %CC4_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC4_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous4[%c1] : memref<3xi32>
      aie.use_lock(%BC4_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous4[%c2] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%CC4_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%CC4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC4_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%AC4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%AC4_cons_buff_0 : memref<32x64xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%AC4_cons_buff_1 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%AC4_cons_buff_0 : memref<32x64xi8>)
    ^bb19(%32: memref<32x64xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%BC4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%BC4_cons_buff_0 : memref<64x128xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%BC4_cons_buff_1 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%BC4_cons_buff_0 : memref<64x128xi8>)
    ^bb23(%36: memref<64x128xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_scalar_i8_i16(%32, %36, %CC4_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC4_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous4[%c1] : memref<3xi32>
      aie.use_lock(%BC4_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous4[%c2] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%CC4_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%CC4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC4_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%AC4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%AC4_cons_buff_0 : memref<32x64xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%AC4_cons_buff_1 : memref<32x64xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%AC4_cons_buff_0 : memref<32x64xi8>)
    ^bb30(%57: memref<32x64xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%BC4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%BC4_cons_buff_0 : memref<64x128xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%BC4_cons_buff_1 : memref<64x128xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%BC4_cons_buff_0 : memref<64x128xi8>)
    ^bb34(%61: memref<64x128xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_scalar_i8_i16(%57, %61, %CC4_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC4_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous4[%c1] : memref<3xi32>
      aie.use_lock(%BC4_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous4[%c2] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%CC4_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%CC4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC4_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%AC4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%AC4_cons_buff_0 : memref<32x64xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%AC4_cons_buff_1 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%AC4_cons_buff_0 : memref<32x64xi8>)
    ^bb41(%82: memref<32x64xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%BC4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%BC4_cons_buff_0 : memref<64x128xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%BC4_cons_buff_1 : memref<64x128xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%BC4_cons_buff_0 : memref<64x128xi8>)
    ^bb45(%86: memref<64x128xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_scalar_i8_i16(%82, %86, %CC4_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC4_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous4[%c1] : memref<3xi32>
      aie.use_lock(%BC4_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous4[%c2] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%CC4_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous4[%c0] : memref<3xi32>
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb47:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"]}
    %_anonymous5 = aie.buffer(%tile_5_2) {address = 29696 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb46
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb47
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC5_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC5_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
    ^bb4:  // pred: ^bb3
      aie.use_lock(%AC5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%AC5_cons_buff_0 : memref<32x64xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%AC5_cons_buff_1 : memref<32x64xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%AC5_cons_buff_0 : memref<32x64xi8>)
    ^bb8(%7: memref<32x64xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%BC5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%BC5_cons_buff_0 : memref<64x128xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%BC5_cons_buff_1 : memref<64x128xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%BC5_cons_buff_0 : memref<64x128xi8>)
    ^bb12(%11: memref<64x128xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_scalar_i8_i16(%7, %11, %CC5_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC5_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous5[%c1] : memref<3xi32>
      aie.use_lock(%BC5_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous5[%c2] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%CC5_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%CC5_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC5_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%AC5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%AC5_cons_buff_0 : memref<32x64xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%AC5_cons_buff_1 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%AC5_cons_buff_0 : memref<32x64xi8>)
    ^bb19(%32: memref<32x64xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%BC5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%BC5_cons_buff_0 : memref<64x128xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%BC5_cons_buff_1 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%BC5_cons_buff_0 : memref<64x128xi8>)
    ^bb23(%36: memref<64x128xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_scalar_i8_i16(%32, %36, %CC5_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC5_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous5[%c1] : memref<3xi32>
      aie.use_lock(%BC5_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous5[%c2] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%CC5_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%CC5_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC5_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%AC5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%AC5_cons_buff_0 : memref<32x64xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%AC5_cons_buff_1 : memref<32x64xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%AC5_cons_buff_0 : memref<32x64xi8>)
    ^bb30(%57: memref<32x64xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%BC5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%BC5_cons_buff_0 : memref<64x128xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%BC5_cons_buff_1 : memref<64x128xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%BC5_cons_buff_0 : memref<64x128xi8>)
    ^bb34(%61: memref<64x128xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_scalar_i8_i16(%57, %61, %CC5_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC5_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous5[%c1] : memref<3xi32>
      aie.use_lock(%BC5_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous5[%c2] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%CC5_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%CC5_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC5_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%AC5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%AC5_cons_buff_0 : memref<32x64xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%AC5_cons_buff_1 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%AC5_cons_buff_0 : memref<32x64xi8>)
    ^bb41(%82: memref<32x64xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%BC5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%BC5_cons_buff_0 : memref<64x128xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%BC5_cons_buff_1 : memref<64x128xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%BC5_cons_buff_0 : memref<64x128xi8>)
    ^bb45(%86: memref<64x128xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_scalar_i8_i16(%82, %86, %CC5_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC5_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous5[%c1] : memref<3xi32>
      aie.use_lock(%BC5_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous5[%c2] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%CC5_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous5[%c0] : memref<3xi32>
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb47:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"]}
    %_anonymous6 = aie.buffer(%tile_6_2) {address = 29696 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb46
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb47
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC6_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
    ^bb4:  // pred: ^bb3
      aie.use_lock(%AC6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%AC6_cons_buff_0 : memref<32x64xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%AC6_cons_buff_1 : memref<32x64xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%AC6_cons_buff_0 : memref<32x64xi8>)
    ^bb8(%7: memref<32x64xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%BC6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%BC6_cons_buff_0 : memref<64x128xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%BC6_cons_buff_1 : memref<64x128xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%BC6_cons_buff_0 : memref<64x128xi8>)
    ^bb12(%11: memref<64x128xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_scalar_i8_i16(%7, %11, %CC6_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC6_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous6[%c1] : memref<3xi32>
      aie.use_lock(%BC6_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous6[%c2] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%CC6_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%CC6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC6_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%AC6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%AC6_cons_buff_0 : memref<32x64xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%AC6_cons_buff_1 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%AC6_cons_buff_0 : memref<32x64xi8>)
    ^bb19(%32: memref<32x64xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%BC6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%BC6_cons_buff_0 : memref<64x128xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%BC6_cons_buff_1 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%BC6_cons_buff_0 : memref<64x128xi8>)
    ^bb23(%36: memref<64x128xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_scalar_i8_i16(%32, %36, %CC6_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC6_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous6[%c1] : memref<3xi32>
      aie.use_lock(%BC6_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous6[%c2] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%CC6_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%CC6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC6_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%AC6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%AC6_cons_buff_0 : memref<32x64xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%AC6_cons_buff_1 : memref<32x64xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%AC6_cons_buff_0 : memref<32x64xi8>)
    ^bb30(%57: memref<32x64xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%BC6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%BC6_cons_buff_0 : memref<64x128xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%BC6_cons_buff_1 : memref<64x128xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%BC6_cons_buff_0 : memref<64x128xi8>)
    ^bb34(%61: memref<64x128xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_scalar_i8_i16(%57, %61, %CC6_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC6_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous6[%c1] : memref<3xi32>
      aie.use_lock(%BC6_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous6[%c2] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%CC6_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%CC6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC6_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%AC6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%AC6_cons_buff_0 : memref<32x64xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%AC6_cons_buff_1 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%AC6_cons_buff_0 : memref<32x64xi8>)
    ^bb41(%82: memref<32x64xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%BC6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%BC6_cons_buff_0 : memref<64x128xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%BC6_cons_buff_1 : memref<64x128xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%BC6_cons_buff_0 : memref<64x128xi8>)
    ^bb45(%86: memref<64x128xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_scalar_i8_i16(%82, %86, %CC6_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC6_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous6[%c1] : memref<3xi32>
      aie.use_lock(%BC6_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous6[%c2] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%CC6_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous6[%c0] : memref<3xi32>
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb47:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"]}
    %_anonymous7 = aie.buffer(%tile_7_2) {address = 29696 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb46
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb47
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC7_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb12
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb13
    ^bb4:  // pred: ^bb3
      aie.use_lock(%AC7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%AC7_cons_buff_0 : memref<32x64xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%AC7_cons_buff_1 : memref<32x64xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%AC7_cons_buff_0 : memref<32x64xi8>)
    ^bb8(%7: memref<32x64xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      aie.use_lock(%BC7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%BC7_cons_buff_0 : memref<64x128xi8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%BC7_cons_buff_1 : memref<64x128xi8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%BC7_cons_buff_0 : memref<64x128xi8>)
    ^bb12(%11: memref<64x128xi8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      func.call @matmul_scalar_i8_i16(%7, %11, %CC7_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC7_cons_prod_lock_0, Release, 1)
      %12 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c2_i32 : i32
      %15 = arith.subi %13, %c2_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous7[%c1] : memref<3xi32>
      aie.use_lock(%BC7_cons_prod_lock_0, Release, 1)
      %17 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c2_i32 : i32
      %20 = arith.subi %18, %c2_i32 : i32
      %21 = arith.select %19, %20, %18 : i32
      memref.store %21, %_anonymous7[%c2] : memref<3xi32>
      %22 = arith.addi %2, %c1 : index
      cf.br ^bb3(%22 : index)
    ^bb13:  // pred: ^bb3
      aie.use_lock(%CC7_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c1_i32 : i32
      %26 = arith.select %25, %23, %24 : i32
      memref.store %26, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%CC7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC7_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb14(%c0 : index)
    ^bb14(%27: index):  // 2 preds: ^bb13, ^bb23
      %28 = arith.cmpi slt, %27, %c16 : index
      cf.cond_br %28, ^bb15, ^bb24
    ^bb15:  // pred: ^bb14
      aie.use_lock(%AC7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %29 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %30 : index to i32
      cf.switch %31 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%AC7_cons_buff_0 : memref<32x64xi8>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%AC7_cons_buff_1 : memref<32x64xi8>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%AC7_cons_buff_0 : memref<32x64xi8>)
    ^bb19(%32: memref<32x64xi8>):  // 3 preds: ^bb16, ^bb17, ^bb18
      aie.use_lock(%BC7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %33 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.index_cast %34 : index to i32
      cf.switch %35 : i32, [
        default: ^bb22,
        0: ^bb20,
        1: ^bb21
      ]
    ^bb20:  // pred: ^bb19
      cf.br ^bb23(%BC7_cons_buff_0 : memref<64x128xi8>)
    ^bb21:  // pred: ^bb19
      cf.br ^bb23(%BC7_cons_buff_1 : memref<64x128xi8>)
    ^bb22:  // pred: ^bb19
      cf.br ^bb23(%BC7_cons_buff_0 : memref<64x128xi8>)
    ^bb23(%36: memref<64x128xi8>):  // 3 preds: ^bb20, ^bb21, ^bb22
      func.call @matmul_scalar_i8_i16(%32, %36, %CC7_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC7_cons_prod_lock_0, Release, 1)
      %37 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %38 = arith.addi %37, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.subi %38, %c2_i32 : i32
      %41 = arith.select %39, %40, %38 : i32
      memref.store %41, %_anonymous7[%c1] : memref<3xi32>
      aie.use_lock(%BC7_cons_prod_lock_0, Release, 1)
      %42 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %43 = arith.addi %42, %c1_i32 : i32
      %44 = arith.cmpi sge, %43, %c2_i32 : i32
      %45 = arith.subi %43, %c2_i32 : i32
      %46 = arith.select %44, %45, %43 : i32
      memref.store %46, %_anonymous7[%c2] : memref<3xi32>
      %47 = arith.addi %27, %c1 : index
      cf.br ^bb14(%47 : index)
    ^bb24:  // pred: ^bb14
      aie.use_lock(%CC7_cons_lock_0, Release, 1)
      %48 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %49 = arith.addi %48, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c1_i32 : i32
      %51 = arith.select %50, %48, %49 : i32
      memref.store %51, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%CC7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC7_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb25(%c0 : index)
    ^bb25(%52: index):  // 2 preds: ^bb24, ^bb34
      %53 = arith.cmpi slt, %52, %c16 : index
      cf.cond_br %53, ^bb26, ^bb35
    ^bb26:  // pred: ^bb25
      aie.use_lock(%AC7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %54 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.index_cast %55 : index to i32
      cf.switch %56 : i32, [
        default: ^bb29,
        0: ^bb27,
        1: ^bb28
      ]
    ^bb27:  // pred: ^bb26
      cf.br ^bb30(%AC7_cons_buff_0 : memref<32x64xi8>)
    ^bb28:  // pred: ^bb26
      cf.br ^bb30(%AC7_cons_buff_1 : memref<32x64xi8>)
    ^bb29:  // pred: ^bb26
      cf.br ^bb30(%AC7_cons_buff_0 : memref<32x64xi8>)
    ^bb30(%57: memref<32x64xi8>):  // 3 preds: ^bb27, ^bb28, ^bb29
      aie.use_lock(%BC7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %58 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.index_cast %59 : index to i32
      cf.switch %60 : i32, [
        default: ^bb33,
        0: ^bb31,
        1: ^bb32
      ]
    ^bb31:  // pred: ^bb30
      cf.br ^bb34(%BC7_cons_buff_0 : memref<64x128xi8>)
    ^bb32:  // pred: ^bb30
      cf.br ^bb34(%BC7_cons_buff_1 : memref<64x128xi8>)
    ^bb33:  // pred: ^bb30
      cf.br ^bb34(%BC7_cons_buff_0 : memref<64x128xi8>)
    ^bb34(%61: memref<64x128xi8>):  // 3 preds: ^bb31, ^bb32, ^bb33
      func.call @matmul_scalar_i8_i16(%57, %61, %CC7_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC7_cons_prod_lock_0, Release, 1)
      %62 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %63 = arith.addi %62, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c2_i32 : i32
      %65 = arith.subi %63, %c2_i32 : i32
      %66 = arith.select %64, %65, %63 : i32
      memref.store %66, %_anonymous7[%c1] : memref<3xi32>
      aie.use_lock(%BC7_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c2_i32 : i32
      %70 = arith.subi %68, %c2_i32 : i32
      %71 = arith.select %69, %70, %68 : i32
      memref.store %71, %_anonymous7[%c2] : memref<3xi32>
      %72 = arith.addi %52, %c1 : index
      cf.br ^bb25(%72 : index)
    ^bb35:  // pred: ^bb25
      aie.use_lock(%CC7_cons_lock_0, Release, 1)
      %73 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c1_i32 : i32
      %76 = arith.select %75, %73, %74 : i32
      memref.store %76, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%CC7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC7_buff_0) : (memref<32x128xi16>) -> ()
      cf.br ^bb36(%c0 : index)
    ^bb36(%77: index):  // 2 preds: ^bb35, ^bb45
      %78 = arith.cmpi slt, %77, %c16 : index
      cf.cond_br %78, ^bb37, ^bb46
    ^bb37:  // pred: ^bb36
      aie.use_lock(%AC7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %79 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = arith.index_cast %80 : index to i32
      cf.switch %81 : i32, [
        default: ^bb40,
        0: ^bb38,
        1: ^bb39
      ]
    ^bb38:  // pred: ^bb37
      cf.br ^bb41(%AC7_cons_buff_0 : memref<32x64xi8>)
    ^bb39:  // pred: ^bb37
      cf.br ^bb41(%AC7_cons_buff_1 : memref<32x64xi8>)
    ^bb40:  // pred: ^bb37
      cf.br ^bb41(%AC7_cons_buff_0 : memref<32x64xi8>)
    ^bb41(%82: memref<32x64xi8>):  // 3 preds: ^bb38, ^bb39, ^bb40
      aie.use_lock(%BC7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %83 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %84 = arith.index_cast %83 : i32 to index
      %85 = arith.index_cast %84 : index to i32
      cf.switch %85 : i32, [
        default: ^bb44,
        0: ^bb42,
        1: ^bb43
      ]
    ^bb42:  // pred: ^bb41
      cf.br ^bb45(%BC7_cons_buff_0 : memref<64x128xi8>)
    ^bb43:  // pred: ^bb41
      cf.br ^bb45(%BC7_cons_buff_1 : memref<64x128xi8>)
    ^bb44:  // pred: ^bb41
      cf.br ^bb45(%BC7_cons_buff_0 : memref<64x128xi8>)
    ^bb45(%86: memref<64x128xi8>):  // 3 preds: ^bb42, ^bb43, ^bb44
      func.call @matmul_scalar_i8_i16(%82, %86, %CC7_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
      aie.use_lock(%AC7_cons_prod_lock_0, Release, 1)
      %87 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %88 = arith.addi %87, %c1_i32 : i32
      %89 = arith.cmpi sge, %88, %c2_i32 : i32
      %90 = arith.subi %88, %c2_i32 : i32
      %91 = arith.select %89, %90, %88 : i32
      memref.store %91, %_anonymous7[%c1] : memref<3xi32>
      aie.use_lock(%BC7_cons_prod_lock_0, Release, 1)
      %92 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %93 = arith.addi %92, %c1_i32 : i32
      %94 = arith.cmpi sge, %93, %c2_i32 : i32
      %95 = arith.subi %93, %c2_i32 : i32
      %96 = arith.select %94, %95, %93 : i32
      memref.store %96, %_anonymous7[%c2] : memref<3xi32>
      %97 = arith.addi %77, %c1 : index
      cf.br ^bb36(%97 : index)
    ^bb46:  // pred: ^bb36
      aie.use_lock(%CC7_cons_lock_0, Release, 1)
      %98 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %99 = arith.addi %98, %c1_i32 : i32
      %100 = arith.cmpi sge, %99, %c1_i32 : i32
      %101 = arith.select %100, %98, %99 : i32
      memref.store %101, %_anonymous7[%c0] : memref<3xi32>
      %102 = arith.addi %0, %c1 : index
      cf.br ^bb1(%102 : index)
    ^bb47:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_32x64x128.o"]}
    aie.runtime_sequence @seq(%arg0: memref<131072xi8>, %arg1: memref<1048576xi8>, %arg2: memref<131072xi16>) {
      %0 = aiex.dma_configure_task_for @AS0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @BS0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @AS1_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @BS1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @AS2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @BS2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @AS3_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @BS3_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @AS4_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @BS4_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @AS5_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @BS5_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @AS6_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @BS6_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @AS7_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @BS7_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @CS0_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @CS1_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 128, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @CS2_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 256, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @CS3_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 384, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @CS4_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 512, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @CS5_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 640, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @CS6_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @CS7_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @AS0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @BS0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @AS1_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @BS1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @AS2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @BS2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @AS3_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @BS3_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @AS4_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @BS4_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @AS5_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @BS5_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @AS6_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @BS6_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @AS7_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @BS7_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @CS0_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 32768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @CS1_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 32896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @CS2_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33024, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @CS3_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33152, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @CS4_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33280, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @CS5_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33408, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @CS6_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @CS7_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @AS0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @BS0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @AS1_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @BS1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @AS2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @BS2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @AS3_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @BS3_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @AS4_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @BS4_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @AS5_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @BS5_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @AS6_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @BS6_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @AS7_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @BS7_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @CS0_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @CS1_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @CS2_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65792, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @CS3_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65920, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @CS4_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66048, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @CS5_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66176, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @CS6_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @CS7_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @AS0_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @BS0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @AS1_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @BS1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @AS2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @BS2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @AS3_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @BS3_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @AS4_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @BS4_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @AS5_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @BS5_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @AS6_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @BS6_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @AS7_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @BS7_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @CS0_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @CS1_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @CS2_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98560, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @CS3_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98688, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @CS4_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98816, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @CS5_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98944, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @CS6_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 99072, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @CS7_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 99200, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%95)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
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
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
      aiex.dma_await_task(%40)
      aiex.dma_await_task(%41)
      aiex.dma_await_task(%42)
      aiex.dma_await_task(%43)
      aiex.dma_await_task(%44)
      aiex.dma_await_task(%45)
      aiex.dma_await_task(%46)
      aiex.dma_await_task(%47)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%25)
      aiex.dma_free_task(%26)
      aiex.dma_free_task(%27)
      aiex.dma_free_task(%28)
      aiex.dma_free_task(%29)
      aiex.dma_free_task(%30)
      aiex.dma_free_task(%31)
      aiex.dma_free_task(%32)
      aiex.dma_free_task(%33)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%39)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%65)
      aiex.dma_await_task(%66)
      aiex.dma_await_task(%67)
      aiex.dma_await_task(%68)
      aiex.dma_await_task(%69)
      aiex.dma_await_task(%70)
      aiex.dma_await_task(%71)
      aiex.dma_free_task(%48)
      aiex.dma_free_task(%49)
      aiex.dma_free_task(%50)
      aiex.dma_free_task(%51)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%55)
      aiex.dma_free_task(%56)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%58)
      aiex.dma_free_task(%59)
      aiex.dma_free_task(%60)
      aiex.dma_free_task(%61)
      aiex.dma_free_task(%62)
      aiex.dma_free_task(%63)
      aiex.dma_await_task(%88)
      aiex.dma_await_task(%89)
      aiex.dma_await_task(%90)
      aiex.dma_await_task(%91)
      aiex.dma_await_task(%92)
      aiex.dma_await_task(%93)
      aiex.dma_await_task(%94)
      aiex.dma_await_task(%95)
      aiex.dma_free_task(%72)
      aiex.dma_free_task(%73)
      aiex.dma_free_task(%74)
      aiex.dma_free_task(%75)
      aiex.dma_free_task(%76)
      aiex.dma_free_task(%77)
      aiex.dma_free_task(%78)
      aiex.dma_free_task(%79)
      aiex.dma_free_task(%80)
      aiex.dma_free_task(%81)
      aiex.dma_free_task(%82)
      aiex.dma_free_task(%83)
      aiex.dma_free_task(%84)
      aiex.dma_free_task(%85)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%87)
    }
    aie.shim_dma_allocation @AS0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AS0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS0_cons_buff_0 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AS0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AS0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS0_cons_buff_1 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AS0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AS0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS0_cons_buff_0 : memref<32x512xi8>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%AS0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AS0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS0_cons_buff_1 : memref<32x512xi8>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%AS0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BS0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS0_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%BS0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BS0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS0_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%BS0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BS0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS0_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BS0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BS0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS0_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BS0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%CC0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%CC0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC0_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CC0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AC0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC0_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AC0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AC0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC0_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AC0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BC0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC0_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BC0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BC0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC0_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BC0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC0_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @BS0_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @CS0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @AS1_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AS1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS1_cons_buff_0 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AS1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AS1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS1_cons_buff_1 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AS1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AS1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS1_cons_buff_0 : memref<32x512xi8>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%AS1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AS1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS1_cons_buff_1 : memref<32x512xi8>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%AS1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BS1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS1_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%BS1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BS1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS1_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%BS1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BS1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS1_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BS1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BS1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS1_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BS1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%CC1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC1_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%CC1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC1_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CC1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AC1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC1_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AC1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AC1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC1_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AC1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BC1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC1_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BC1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BC1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC1_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BC1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC1_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @BS1_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @CS1_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @AS2_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AS2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS2_cons_buff_0 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AS2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AS2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS2_cons_buff_1 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AS2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AS2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS2_cons_buff_0 : memref<32x512xi8>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%AS2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AS2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS2_cons_buff_1 : memref<32x512xi8>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%AS2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BS2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%BS2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BS2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%BS2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BS2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BS2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BS2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BS2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%CC2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC2_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%CC2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC2_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CC2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AC2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC2_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AC2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AC2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC2_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AC2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BC2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BC2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BC2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BC2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC2_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @BS2_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @CS2_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @AS3_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AS3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS3_cons_buff_0 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AS3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AS3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS3_cons_buff_1 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AS3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AS3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS3_cons_buff_0 : memref<32x512xi8>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%AS3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AS3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS3_cons_buff_1 : memref<32x512xi8>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%AS3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BS3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS3_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%BS3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BS3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS3_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%BS3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BS3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS3_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BS3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BS3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS3_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BS3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%CC3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC3_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%CC3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC3_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CC3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AC3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC3_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AC3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AC3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC3_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AC3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BC3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC3_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BC3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BC3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC3_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BC3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC3_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @BS3_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    aie.shim_dma_allocation @CS3_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @AS4_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AS4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS4_cons_buff_0 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AS4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AS4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS4_cons_buff_1 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AS4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AS4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS4_cons_buff_0 : memref<32x512xi8>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%AS4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AS4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS4_cons_buff_1 : memref<32x512xi8>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%AS4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BS4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS4_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%BS4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BS4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS4_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%BS4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BS4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS4_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BS4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BS4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS4_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BS4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%CC4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC4_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%CC4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC4_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CC4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AC4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC4_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AC4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AC4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC4_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AC4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BC4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC4_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BC4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BC4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC4_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BC4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC4_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC4_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @BS4_shim_alloc(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @CS4_shim_alloc(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @AS5_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AS5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS5_cons_buff_0 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AS5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AS5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS5_cons_buff_1 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AS5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AS5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS5_cons_buff_0 : memref<32x512xi8>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%AS5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AS5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS5_cons_buff_1 : memref<32x512xi8>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%AS5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BS5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS5_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%BS5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BS5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS5_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%BS5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BS5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS5_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BS5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BS5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS5_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BS5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%CC5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC5_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%CC5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC5_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CC5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AC5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC5_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AC5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AC5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC5_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AC5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BC5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC5_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BC5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BC5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC5_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BC5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC5_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC5_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @BS5_shim_alloc(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @CS5_shim_alloc(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @AS6_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AS6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS6_cons_buff_0 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AS6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AS6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS6_cons_buff_1 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AS6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AS6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS6_cons_buff_0 : memref<32x512xi8>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%AS6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AS6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS6_cons_buff_1 : memref<32x512xi8>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%AS6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BS6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS6_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%BS6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BS6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS6_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%BS6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BS6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS6_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BS6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BS6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS6_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BS6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%CC6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC6_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%CC6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC6_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CC6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AC6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC6_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AC6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AC6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC6_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AC6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BC6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC6_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BC6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BC6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC6_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BC6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC6_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC6_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @BS6_shim_alloc(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @CS6_shim_alloc(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @AS7_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AS7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS7_cons_buff_0 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AS7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AS7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS7_cons_buff_1 : memref<32x512xi8>, 0, 16384, [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AS7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%AS7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS7_cons_buff_0 : memref<32x512xi8>, 0, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%AS7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%AS7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AS7_cons_buff_1 : memref<32x512xi8>, 0, 16384) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%AS7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%BS7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS7_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%BS7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%BS7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS7_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%BS7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%BS7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS7_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%BS7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%BS7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS7_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%BS7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb13:  // 2 preds: ^bb12, ^bb13
      aie.use_lock(%CC7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC7_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%CC7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC7_cons_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CC7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%AC7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC7_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%AC7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%AC7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%AC7_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%AC7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BC7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC7_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BC7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BC7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BC7_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BC7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC7_buff_0 : memref<32x128xi16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC7_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @BS7_shim_alloc(%shim_noc_tile_7_0, MM2S, 1)
    aie.shim_dma_allocation @CS7_shim_alloc(%shim_noc_tile_7_0, S2MM, 0)
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
