module {
  aie.device(npu2) {
    func.func private @zero_scalar_i16(memref<128x128xi16>) attributes {link_with = "mm_128x64x128.o"}
    func.func private @matmul_scalar_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) attributes {link_with = "mm_128x64x128.o"}
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
    %CS7_cons_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "CS7_cons_prod_lock_0"}
    %CS7_cons_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "CS7_cons_cons_lock_0"}
    %CC7_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 0 : i32, sym_name = "CC7_cons_buff_0"} : memref<128x128xi16> 
    %CC7_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 2) {init = 1 : i32, sym_name = "CC7_cons_prod_lock_0"}
    %CC7_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 3) {init = 0 : i32, sym_name = "CC7_cons_cons_lock_0"}
    %CC7_buff_0 = aie.buffer(%tile_7_2) {address = 1024 : i32, sym_name = "CC7_buff_0"} : memref<128x128xi16> 
    %CC7_prod_lock_0 = aie.lock(%tile_7_2, 4) {init = 1 : i32, sym_name = "CC7_prod_lock_0"}
    %CC7_cons_lock_0 = aie.lock(%tile_7_2, 5) {init = 0 : i32, sym_name = "CC7_cons_lock_0"}
    %BC7_cons_buff_0 = aie.buffer(%tile_7_2) {address = 33792 : i32, sym_name = "BC7_cons_buff_0"} : memref<64x128xi8> 
    %BC7_cons_buff_1 = aie.buffer(%tile_7_2) {address = 41984 : i32, sym_name = "BC7_cons_buff_1"} : memref<64x128xi8> 
    %BC7_cons_prod_lock_0 = aie.lock(%tile_7_2, 2) {init = 2 : i32, sym_name = "BC7_cons_prod_lock_0"}
    %BC7_cons_cons_lock_0 = aie.lock(%tile_7_2, 3) {init = 0 : i32, sym_name = "BC7_cons_cons_lock_0"}
    %BS7_cons_buff_0 = aie.buffer(%mem_tile_7_1) {address = 32768 : i32, sym_name = "BS7_cons_buff_0"} : memref<64x128xi8> 
    %BS7_cons_buff_1 = aie.buffer(%mem_tile_7_1) {address = 40960 : i32, sym_name = "BS7_cons_buff_1"} : memref<64x128xi8> 
    %BS7_cons_prod_lock_0 = aie.lock(%mem_tile_7_1, 0) {init = 2 : i32, sym_name = "BS7_cons_prod_lock_0"}
    %BS7_cons_cons_lock_0 = aie.lock(%mem_tile_7_1, 1) {init = 0 : i32, sym_name = "BS7_cons_cons_lock_0"}
    %BS7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "BS7_prod_lock_0"}
    %BS7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "BS7_cons_lock_0"}
    %CS6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "CS6_cons_prod_lock_0"}
    %CS6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "CS6_cons_cons_lock_0"}
    %CC6_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 0 : i32, sym_name = "CC6_cons_buff_0"} : memref<128x128xi16> 
    %CC6_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 2) {init = 1 : i32, sym_name = "CC6_cons_prod_lock_0"}
    %CC6_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 3) {init = 0 : i32, sym_name = "CC6_cons_cons_lock_0"}
    %CC6_buff_0 = aie.buffer(%tile_6_2) {address = 1024 : i32, sym_name = "CC6_buff_0"} : memref<128x128xi16> 
    %CC6_prod_lock_0 = aie.lock(%tile_6_2, 4) {init = 1 : i32, sym_name = "CC6_prod_lock_0"}
    %CC6_cons_lock_0 = aie.lock(%tile_6_2, 5) {init = 0 : i32, sym_name = "CC6_cons_lock_0"}
    %BC6_cons_buff_0 = aie.buffer(%tile_6_2) {address = 33792 : i32, sym_name = "BC6_cons_buff_0"} : memref<64x128xi8> 
    %BC6_cons_buff_1 = aie.buffer(%tile_6_2) {address = 41984 : i32, sym_name = "BC6_cons_buff_1"} : memref<64x128xi8> 
    %BC6_cons_prod_lock_0 = aie.lock(%tile_6_2, 2) {init = 2 : i32, sym_name = "BC6_cons_prod_lock_0"}
    %BC6_cons_cons_lock_0 = aie.lock(%tile_6_2, 3) {init = 0 : i32, sym_name = "BC6_cons_cons_lock_0"}
    %BS6_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 32768 : i32, sym_name = "BS6_cons_buff_0"} : memref<64x128xi8> 
    %BS6_cons_buff_1 = aie.buffer(%mem_tile_6_1) {address = 40960 : i32, sym_name = "BS6_cons_buff_1"} : memref<64x128xi8> 
    %BS6_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 0) {init = 2 : i32, sym_name = "BS6_cons_prod_lock_0"}
    %BS6_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 1) {init = 0 : i32, sym_name = "BS6_cons_cons_lock_0"}
    %BS6_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "BS6_prod_lock_0"}
    %BS6_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "BS6_cons_lock_0"}
    %CS5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "CS5_cons_prod_lock_0"}
    %CS5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "CS5_cons_cons_lock_0"}
    %CC5_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 0 : i32, sym_name = "CC5_cons_buff_0"} : memref<128x128xi16> 
    %CC5_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 2) {init = 1 : i32, sym_name = "CC5_cons_prod_lock_0"}
    %CC5_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 3) {init = 0 : i32, sym_name = "CC5_cons_cons_lock_0"}
    %CC5_buff_0 = aie.buffer(%tile_5_2) {address = 1024 : i32, sym_name = "CC5_buff_0"} : memref<128x128xi16> 
    %CC5_prod_lock_0 = aie.lock(%tile_5_2, 4) {init = 1 : i32, sym_name = "CC5_prod_lock_0"}
    %CC5_cons_lock_0 = aie.lock(%tile_5_2, 5) {init = 0 : i32, sym_name = "CC5_cons_lock_0"}
    %BC5_cons_buff_0 = aie.buffer(%tile_5_2) {address = 33792 : i32, sym_name = "BC5_cons_buff_0"} : memref<64x128xi8> 
    %BC5_cons_buff_1 = aie.buffer(%tile_5_2) {address = 41984 : i32, sym_name = "BC5_cons_buff_1"} : memref<64x128xi8> 
    %BC5_cons_prod_lock_0 = aie.lock(%tile_5_2, 2) {init = 2 : i32, sym_name = "BC5_cons_prod_lock_0"}
    %BC5_cons_cons_lock_0 = aie.lock(%tile_5_2, 3) {init = 0 : i32, sym_name = "BC5_cons_cons_lock_0"}
    %BS5_cons_buff_0 = aie.buffer(%mem_tile_5_1) {address = 32768 : i32, sym_name = "BS5_cons_buff_0"} : memref<64x128xi8> 
    %BS5_cons_buff_1 = aie.buffer(%mem_tile_5_1) {address = 40960 : i32, sym_name = "BS5_cons_buff_1"} : memref<64x128xi8> 
    %BS5_cons_prod_lock_0 = aie.lock(%mem_tile_5_1, 0) {init = 2 : i32, sym_name = "BS5_cons_prod_lock_0"}
    %BS5_cons_cons_lock_0 = aie.lock(%mem_tile_5_1, 1) {init = 0 : i32, sym_name = "BS5_cons_cons_lock_0"}
    %BS5_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "BS5_prod_lock_0"}
    %BS5_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "BS5_cons_lock_0"}
    %CS4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "CS4_cons_prod_lock_0"}
    %CS4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "CS4_cons_cons_lock_0"}
    %CC4_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 0 : i32, sym_name = "CC4_cons_buff_0"} : memref<128x128xi16> 
    %CC4_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 2) {init = 1 : i32, sym_name = "CC4_cons_prod_lock_0"}
    %CC4_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 3) {init = 0 : i32, sym_name = "CC4_cons_cons_lock_0"}
    %CC4_buff_0 = aie.buffer(%tile_4_2) {address = 1024 : i32, sym_name = "CC4_buff_0"} : memref<128x128xi16> 
    %CC4_prod_lock_0 = aie.lock(%tile_4_2, 4) {init = 1 : i32, sym_name = "CC4_prod_lock_0"}
    %CC4_cons_lock_0 = aie.lock(%tile_4_2, 5) {init = 0 : i32, sym_name = "CC4_cons_lock_0"}
    %BC4_cons_buff_0 = aie.buffer(%tile_4_2) {address = 33792 : i32, sym_name = "BC4_cons_buff_0"} : memref<64x128xi8> 
    %BC4_cons_buff_1 = aie.buffer(%tile_4_2) {address = 41984 : i32, sym_name = "BC4_cons_buff_1"} : memref<64x128xi8> 
    %BC4_cons_prod_lock_0 = aie.lock(%tile_4_2, 2) {init = 2 : i32, sym_name = "BC4_cons_prod_lock_0"}
    %BC4_cons_cons_lock_0 = aie.lock(%tile_4_2, 3) {init = 0 : i32, sym_name = "BC4_cons_cons_lock_0"}
    %BS4_cons_buff_0 = aie.buffer(%mem_tile_4_1) {address = 32768 : i32, sym_name = "BS4_cons_buff_0"} : memref<64x128xi8> 
    %BS4_cons_buff_1 = aie.buffer(%mem_tile_4_1) {address = 40960 : i32, sym_name = "BS4_cons_buff_1"} : memref<64x128xi8> 
    %BS4_cons_prod_lock_0 = aie.lock(%mem_tile_4_1, 0) {init = 2 : i32, sym_name = "BS4_cons_prod_lock_0"}
    %BS4_cons_cons_lock_0 = aie.lock(%mem_tile_4_1, 1) {init = 0 : i32, sym_name = "BS4_cons_cons_lock_0"}
    %BS4_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "BS4_prod_lock_0"}
    %BS4_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "BS4_cons_lock_0"}
    %CS3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "CS3_cons_prod_lock_0"}
    %CS3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "CS3_cons_cons_lock_0"}
    %CC3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, sym_name = "CC3_cons_buff_0"} : memref<128x128xi16> 
    %CC3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 2) {init = 1 : i32, sym_name = "CC3_cons_prod_lock_0"}
    %CC3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "CC3_cons_cons_lock_0"}
    %CC3_buff_0 = aie.buffer(%tile_3_2) {address = 1024 : i32, sym_name = "CC3_buff_0"} : memref<128x128xi16> 
    %CC3_prod_lock_0 = aie.lock(%tile_3_2, 4) {init = 1 : i32, sym_name = "CC3_prod_lock_0"}
    %CC3_cons_lock_0 = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "CC3_cons_lock_0"}
    %BC3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 33792 : i32, sym_name = "BC3_cons_buff_0"} : memref<64x128xi8> 
    %BC3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 41984 : i32, sym_name = "BC3_cons_buff_1"} : memref<64x128xi8> 
    %BC3_cons_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "BC3_cons_prod_lock_0"}
    %BC3_cons_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "BC3_cons_cons_lock_0"}
    %BS3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 32768 : i32, sym_name = "BS3_cons_buff_0"} : memref<64x128xi8> 
    %BS3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 40960 : i32, sym_name = "BS3_cons_buff_1"} : memref<64x128xi8> 
    %BS3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "BS3_cons_prod_lock_0"}
    %BS3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "BS3_cons_cons_lock_0"}
    %BS3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "BS3_prod_lock_0"}
    %BS3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "BS3_cons_lock_0"}
    %CS2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "CS2_cons_prod_lock_0"}
    %CS2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "CS2_cons_cons_lock_0"}
    %CC2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "CC2_cons_buff_0"} : memref<128x128xi16> 
    %CC2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 2) {init = 1 : i32, sym_name = "CC2_cons_prod_lock_0"}
    %CC2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "CC2_cons_cons_lock_0"}
    %CC2_buff_0 = aie.buffer(%tile_2_2) {address = 1024 : i32, sym_name = "CC2_buff_0"} : memref<128x128xi16> 
    %CC2_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 1 : i32, sym_name = "CC2_prod_lock_0"}
    %CC2_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "CC2_cons_lock_0"}
    %BC2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 33792 : i32, sym_name = "BC2_cons_buff_0"} : memref<64x128xi8> 
    %BC2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 41984 : i32, sym_name = "BC2_cons_buff_1"} : memref<64x128xi8> 
    %BC2_cons_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "BC2_cons_prod_lock_0"}
    %BC2_cons_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "BC2_cons_cons_lock_0"}
    %BS2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 32768 : i32, sym_name = "BS2_cons_buff_0"} : memref<64x128xi8> 
    %BS2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 40960 : i32, sym_name = "BS2_cons_buff_1"} : memref<64x128xi8> 
    %BS2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "BS2_cons_prod_lock_0"}
    %BS2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "BS2_cons_cons_lock_0"}
    %BS2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "BS2_prod_lock_0"}
    %BS2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "BS2_cons_lock_0"}
    %CS1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "CS1_cons_prod_lock_0"}
    %CS1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "CS1_cons_cons_lock_0"}
    %CC1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "CC1_cons_buff_0"} : memref<128x128xi16> 
    %CC1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 2) {init = 1 : i32, sym_name = "CC1_cons_prod_lock_0"}
    %CC1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "CC1_cons_cons_lock_0"}
    %CC1_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, sym_name = "CC1_buff_0"} : memref<128x128xi16> 
    %CC1_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 1 : i32, sym_name = "CC1_prod_lock_0"}
    %CC1_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "CC1_cons_lock_0"}
    %BC1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 33792 : i32, sym_name = "BC1_cons_buff_0"} : memref<64x128xi8> 
    %BC1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 41984 : i32, sym_name = "BC1_cons_buff_1"} : memref<64x128xi8> 
    %BC1_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "BC1_cons_prod_lock_0"}
    %BC1_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "BC1_cons_cons_lock_0"}
    %BS1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 32768 : i32, sym_name = "BS1_cons_buff_0"} : memref<64x128xi8> 
    %BS1_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 40960 : i32, sym_name = "BS1_cons_buff_1"} : memref<64x128xi8> 
    %BS1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "BS1_cons_prod_lock_0"}
    %BS1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "BS1_cons_cons_lock_0"}
    %BS1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "BS1_prod_lock_0"}
    %BS1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "BS1_cons_lock_0"}
    %CS0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "CS0_cons_prod_lock_0"}
    %CS0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "CS0_cons_cons_lock_0"}
    %CC0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "CC0_cons_buff_0"} : memref<128x128xi16> 
    %CC0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 6) {init = 1 : i32, sym_name = "CC0_cons_prod_lock_0"}
    %CC0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "CC0_cons_cons_lock_0"}
    %CC0_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "CC0_buff_0"} : memref<128x128xi16> 
    %CC0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "CC0_prod_lock_0"}
    %CC0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "CC0_cons_lock_0"}
    %BC0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 33792 : i32, sym_name = "BC0_cons_buff_0"} : memref<64x128xi8> 
    %BC0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 41984 : i32, sym_name = "BC0_cons_buff_1"} : memref<64x128xi8> 
    %BC0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "BC0_cons_prod_lock_0"}
    %BC0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "BC0_cons_cons_lock_0"}
    %BS0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 32768 : i32, sym_name = "BS0_cons_buff_0"} : memref<64x128xi8> 
    %BS0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 40960 : i32, sym_name = "BS0_cons_buff_1"} : memref<64x128xi8> 
    %BS0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "BS0_cons_prod_lock_0"}
    %BS0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "BS0_cons_cons_lock_0"}
    %BS0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "BS0_prod_lock_0"}
    %BS0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "BS0_cons_lock_0"}
    %A_L2L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 50176 : i32, sym_name = "A_L2L1_0_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 52224 : i32, sym_name = "A_L2L1_0_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L2L1_0_cons_prod_lock_0"}
    %A_L2L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L2L1_0_cons_cons_lock_0"}
    %A_L2L1_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 50176 : i32, sym_name = "A_L2L1_1_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 52224 : i32, sym_name = "A_L2L1_1_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "A_L2L1_1_cons_prod_lock_0"}
    %A_L2L1_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "A_L2L1_1_cons_cons_lock_0"}
    %A_L2L1_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 50176 : i32, sym_name = "A_L2L1_2_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 52224 : i32, sym_name = "A_L2L1_2_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "A_L2L1_2_cons_prod_lock_0"}
    %A_L2L1_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "A_L2L1_2_cons_cons_lock_0"}
    %A_L2L1_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 50176 : i32, sym_name = "A_L2L1_3_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 52224 : i32, sym_name = "A_L2L1_3_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "A_L2L1_3_cons_prod_lock_0"}
    %A_L2L1_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "A_L2L1_3_cons_cons_lock_0"}
    %A_L2L1_4_cons_buff_0 = aie.buffer(%tile_4_2) {address = 50176 : i32, sym_name = "A_L2L1_4_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_4_cons_buff_1 = aie.buffer(%tile_4_2) {address = 52224 : i32, sym_name = "A_L2L1_4_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_4_cons_prod_lock_0 = aie.lock(%tile_4_2, 0) {init = 2 : i32, sym_name = "A_L2L1_4_cons_prod_lock_0"}
    %A_L2L1_4_cons_cons_lock_0 = aie.lock(%tile_4_2, 1) {init = 0 : i32, sym_name = "A_L2L1_4_cons_cons_lock_0"}
    %A_L2L1_5_cons_buff_0 = aie.buffer(%tile_5_2) {address = 50176 : i32, sym_name = "A_L2L1_5_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_5_cons_buff_1 = aie.buffer(%tile_5_2) {address = 52224 : i32, sym_name = "A_L2L1_5_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_5_cons_prod_lock_0 = aie.lock(%tile_5_2, 0) {init = 2 : i32, sym_name = "A_L2L1_5_cons_prod_lock_0"}
    %A_L2L1_5_cons_cons_lock_0 = aie.lock(%tile_5_2, 1) {init = 0 : i32, sym_name = "A_L2L1_5_cons_cons_lock_0"}
    %A_L2L1_6_cons_buff_0 = aie.buffer(%tile_6_2) {address = 50176 : i32, sym_name = "A_L2L1_6_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_6_cons_buff_1 = aie.buffer(%tile_6_2) {address = 52224 : i32, sym_name = "A_L2L1_6_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_6_cons_prod_lock_0 = aie.lock(%tile_6_2, 0) {init = 2 : i32, sym_name = "A_L2L1_6_cons_prod_lock_0"}
    %A_L2L1_6_cons_cons_lock_0 = aie.lock(%tile_6_2, 1) {init = 0 : i32, sym_name = "A_L2L1_6_cons_cons_lock_0"}
    %A_L2L1_7_cons_buff_0 = aie.buffer(%tile_7_2) {address = 50176 : i32, sym_name = "A_L2L1_7_cons_buff_0"} : memref<32x64xi8> 
    %A_L2L1_7_cons_buff_1 = aie.buffer(%tile_7_2) {address = 52224 : i32, sym_name = "A_L2L1_7_cons_buff_1"} : memref<32x64xi8> 
    %A_L2L1_7_cons_prod_lock_0 = aie.lock(%tile_7_2, 0) {init = 2 : i32, sym_name = "A_L2L1_7_cons_prod_lock_0"}
    %A_L2L1_7_cons_cons_lock_0 = aie.lock(%tile_7_2, 1) {init = 0 : i32, sym_name = "A_L2L1_7_cons_cons_lock_0"}
    %A_L2L1_buff_0 = aie.buffer(%mem_tile_0_1) {address = 49152 : i32, sym_name = "A_L2L1_buff_0"} : memref<32x64xi8> 
    %A_L2L1_buff_1 = aie.buffer(%mem_tile_0_1) {address = 51200 : i32, sym_name = "A_L2L1_buff_1"} : memref<32x64xi8> 
    %A_L2L1_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "A_L2L1_prod_lock_0"}
    %A_L2L1_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "A_L2L1_cons_lock_0"}
    %A_L3L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 53248 : i32, sym_name = "A_L3L2_cons_buff_0"} : memref<32x64xi8> 
    %A_L3L2_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 55296 : i32, sym_name = "A_L3L2_cons_buff_1"} : memref<32x64xi8> 
    %A_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "A_L3L2_cons_prod_lock_0"}
    %A_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "A_L3L2_cons_cons_lock_0"}
    %A_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L2_prod_lock_0"}
    %A_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L2_cons_lock_0"}
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
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_2, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_2, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_2, DMA : 1)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_4_2, DMA : 1)
    aie.flow(%tile_4_2, DMA : 0, %mem_tile_4_1, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 1, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_5_2, DMA : 1)
    aie.flow(%tile_5_2, DMA : 0, %mem_tile_5_1, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 1, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_6_2, DMA : 1)
    aie.flow(%tile_6_2, DMA : 0, %mem_tile_6_1, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_7_2, DMA : 1)
    aie.flow(%tile_7_2, DMA : 0, %mem_tile_7_1, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %shim_noc_tile_7_0, DMA : 0)
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 54272 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC0_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%BC0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%BC0_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%BC0_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%BC0_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb14
      %9 = arith.cmpi slt, %8, %c4 : index
      cf.cond_br %9, ^bb10, ^bb15
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_0_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %7, %CC0_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_0_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous0[%c2] : memref<3xi32>
      %19 = arith.addi %8, %c1 : index
      cf.br ^bb9(%19 : index)
    ^bb15:  // pred: ^bb9
      aie.use_lock(%BC0_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous0[%c1] : memref<3xi32>
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%CC0_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous0[%c0] : memref<3xi32>
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"]}
    %_anonymous1 = aie.buffer(%tile_1_2) {address = 54272 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
    %core_1_2 = aie.core(%tile_1_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC1_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%BC1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%BC1_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%BC1_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%BC1_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb14
      %9 = arith.cmpi slt, %8, %c4 : index
      cf.cond_br %9, ^bb10, ^bb15
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_1_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %7, %CC1_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_1_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous1[%c2] : memref<3xi32>
      %19 = arith.addi %8, %c1 : index
      cf.br ^bb9(%19 : index)
    ^bb15:  // pred: ^bb9
      aie.use_lock(%BC1_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous1[%c1] : memref<3xi32>
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%CC1_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous1[%c0] : memref<3xi32>
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"]}
    %_anonymous2 = aie.buffer(%tile_2_2) {address = 54272 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
    %core_2_2 = aie.core(%tile_2_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC2_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%BC2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%BC2_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%BC2_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%BC2_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb14
      %9 = arith.cmpi slt, %8, %c4 : index
      cf.cond_br %9, ^bb10, ^bb15
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_2_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %7, %CC2_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_2_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous2[%c2] : memref<3xi32>
      %19 = arith.addi %8, %c1 : index
      cf.br ^bb9(%19 : index)
    ^bb15:  // pred: ^bb9
      aie.use_lock(%BC2_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous2[%c1] : memref<3xi32>
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%CC2_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous2[%c0] : memref<3xi32>
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"]}
    %_anonymous3 = aie.buffer(%tile_3_2) {address = 54272 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
    %core_3_2 = aie.core(%tile_3_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC3_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%BC3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%BC3_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%BC3_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%BC3_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb14
      %9 = arith.cmpi slt, %8, %c4 : index
      cf.cond_br %9, ^bb10, ^bb15
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_3_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %7, %CC3_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_3_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous3[%c2] : memref<3xi32>
      %19 = arith.addi %8, %c1 : index
      cf.br ^bb9(%19 : index)
    ^bb15:  // pred: ^bb9
      aie.use_lock(%BC3_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous3[%c1] : memref<3xi32>
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%CC3_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous3[%c0] : memref<3xi32>
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"]}
    %_anonymous4 = aie.buffer(%tile_4_2) {address = 54272 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
    %core_4_2 = aie.core(%tile_4_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC4_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%BC4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%BC4_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%BC4_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%BC4_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb14
      %9 = arith.cmpi slt, %8, %c4 : index
      cf.cond_br %9, ^bb10, ^bb15
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_4_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_4_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_4_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %7, %CC4_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_4_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous4[%c2] : memref<3xi32>
      %19 = arith.addi %8, %c1 : index
      cf.br ^bb9(%19 : index)
    ^bb15:  // pred: ^bb9
      aie.use_lock(%BC4_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous4[%c1] : memref<3xi32>
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%CC4_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous4[%c0] : memref<3xi32>
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"]}
    %_anonymous5 = aie.buffer(%tile_5_2) {address = 54272 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
    %core_5_2 = aie.core(%tile_5_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC5_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC5_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%BC5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%BC5_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%BC5_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%BC5_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb14
      %9 = arith.cmpi slt, %8, %c4 : index
      cf.cond_br %9, ^bb10, ^bb15
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_5_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_5_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_5_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %7, %CC5_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_5_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous5[%c2] : memref<3xi32>
      %19 = arith.addi %8, %c1 : index
      cf.br ^bb9(%19 : index)
    ^bb15:  // pred: ^bb9
      aie.use_lock(%BC5_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous5[%c1] : memref<3xi32>
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%CC5_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous5[%c0] : memref<3xi32>
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"]}
    %_anonymous6 = aie.buffer(%tile_6_2) {address = 54272 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
    %core_6_2 = aie.core(%tile_6_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC6_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%BC6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%BC6_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%BC6_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%BC6_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb14
      %9 = arith.cmpi slt, %8, %c4 : index
      cf.cond_br %9, ^bb10, ^bb15
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_6_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_6_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_6_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %7, %CC6_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_6_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous6[%c2] : memref<3xi32>
      %19 = arith.addi %8, %c1 : index
      cf.br ^bb9(%19 : index)
    ^bb15:  // pred: ^bb9
      aie.use_lock(%BC6_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous6[%c1] : memref<3xi32>
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%CC6_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous6[%c0] : memref<3xi32>
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"]}
    %_anonymous7 = aie.buffer(%tile_7_2) {address = 54272 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
    %core_7_2 = aie.core(%tile_7_2) {
      %c4294967295 = arith.constant 4294967295 : index
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%CC7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_scalar_i16(%CC7_buff_0) : (memref<128x128xi16>) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c16 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%BC7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%BC7_cons_buff_0 : memref<64x128xi8>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%BC7_cons_buff_1 : memref<64x128xi8>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%BC7_cons_buff_0 : memref<64x128xi8>)
    ^bb8(%7: memref<64x128xi8>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb14
      %9 = arith.cmpi slt, %8, %c4 : index
      cf.cond_br %9, ^bb10, ^bb15
    ^bb10:  // pred: ^bb9
      aie.use_lock(%A_L2L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_7_cons_buff_0 : memref<32x64xi8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_7_cons_buff_1 : memref<32x64xi8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%A_L2L1_7_cons_buff_0 : memref<32x64xi8>)
    ^bb14(%13: memref<32x64xi8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_scalar_i8_i16(%13, %7, %CC7_buff_0) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
      aie.use_lock(%A_L2L1_7_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous7[%c2] : memref<3xi32>
      %19 = arith.addi %8, %c1 : index
      cf.br ^bb9(%19 : index)
    ^bb15:  // pred: ^bb9
      aie.use_lock(%BC7_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c2_i32 : i32
      %23 = arith.subi %21, %c2_i32 : i32
      %24 = arith.select %22, %23, %21 : i32
      memref.store %24, %_anonymous7[%c1] : memref<3xi32>
      %25 = arith.addi %2, %c1 : index
      cf.br ^bb3(%25 : index)
    ^bb16:  // pred: ^bb3
      aie.use_lock(%CC7_cons_lock_0, Release, 1)
      %26 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %27 = arith.addi %26, %c1_i32 : i32
      %28 = arith.cmpi sge, %27, %c1_i32 : i32
      %29 = arith.select %28, %26, %27 : i32
      memref.store %29, %_anonymous7[%c0] : memref<3xi32>
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["mm_128x64x128.o"]}
    aie.runtime_sequence @seq(%arg0: memref<131072xi8>, %arg1: memref<1048576xi8>, %arg2: memref<131072xi16>) {
      %0 = aiex.dma_configure_task_for @BS0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @BS1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @BS2_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @BS3_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @BS4_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @BS5_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @BS6_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @BS7_shim_alloc {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @A_L3L2_shim_alloc {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @CS0_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @CS1_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 128, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @CS2_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 256, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @CS3_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @CS4_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 512, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @CS5_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 640, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @CS6_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @CS7_shim_alloc {
        aie.dma_bd(%arg2 : memref<131072xi16>, 896, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
    }
    aie.shim_dma_allocation @A_L3L2_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L2L1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L2L1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L2L1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L2L1_prod_lock_0, Release, 1)
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
      aie.dma_bd(%CC0_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb14:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%CC0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC0_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%CC0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_cons_cons_lock_0, Release, 1)
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
      aie.dma_bd(%CC0_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_cons_cons_lock_0, Release, 1)
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
      aie.dma_bd(%CC1_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_cons_cons_lock_0, Release, 1)
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
      aie.dma_bd(%CC2_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_cons_cons_lock_0, Release, 1)
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
      aie.dma_bd(%CC3_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_4_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_4_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_4_cons_cons_lock_0, Release, 1)
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
      aie.dma_bd(%CC4_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC4_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_5_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_5_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_5_cons_cons_lock_0, Release, 1)
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
      aie.dma_bd(%CC5_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC5_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_6_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_6_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_6_cons_cons_lock_0, Release, 1)
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
      aie.dma_bd(%CC6_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC6_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_7_cons_buff_0 : memref<32x64xi8>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_7_cons_buff_1 : memref<32x64xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_7_cons_cons_lock_0, Release, 1)
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
      aie.dma_bd(%CC7_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%CC7_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @BS0_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @CS0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @BS1_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%BS1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS1_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%BS1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%BS1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS1_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%BS1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BS1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS1_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BS1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BS1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS1_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BS1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC1_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%CC1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%CC1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC1_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%CC1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @CS1_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @BS2_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%BS2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%BS2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%BS2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%BS2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BS2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS2_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BS2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BS2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS2_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BS2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC2_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%CC2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%CC2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC2_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%CC2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @CS2_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @BS3_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%BS3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS3_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%BS3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%BS3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS3_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%BS3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BS3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS3_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BS3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BS3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS3_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BS3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC3_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%CC3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%CC3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC3_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%CC3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @CS3_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @BS4_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%BS4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS4_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%BS4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%BS4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS4_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%BS4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BS4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS4_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BS4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BS4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS4_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BS4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC4_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%CC4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%CC4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC4_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%CC4_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @CS4_shim_alloc(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @BS5_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%BS5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS5_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%BS5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%BS5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS5_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%BS5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BS5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS5_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BS5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BS5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS5_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BS5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC5_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%CC5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%CC5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC5_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%CC5_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @CS5_shim_alloc(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @BS6_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%BS6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS6_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%BS6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%BS6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS6_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%BS6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BS6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS6_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BS6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BS6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS6_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BS6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC6_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%CC6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%CC6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC6_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%CC6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @CS6_shim_alloc(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @BS7_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%BS7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS7_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%BS7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%BS7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS7_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%BS7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%BS7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS7_cons_buff_0 : memref<64x128xi8>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%BS7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%BS7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%BS7_cons_buff_1 : memref<64x128xi8>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%BS7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%CC7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC7_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%CC7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%CC7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%CC7_cons_buff_0 : memref<128x128xi16>, 0, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%CC7_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      aie.end
    }
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
