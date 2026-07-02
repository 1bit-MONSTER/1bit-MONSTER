module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %mem_tile_6_1 = aie.tile(6, 1)
    %mem_tile_7_1 = aie.tile(7, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    aie.objectfifo @A_S0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo @A_C0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_S0] -> [@A_C0]([] [])
    aie.objectfifo @A_S1(%shim_noc_tile_0_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo @A_C1(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_S1] -> [@A_C1]([] [])
    aie.objectfifo @A_S2(%shim_noc_tile_0_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo @A_C2(%mem_tile_2_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_S2] -> [@A_C2]([] [])
    aie.objectfifo @A_S3(%shim_noc_tile_0_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo @A_C3(%mem_tile_3_1, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_S3] -> [@A_C3]([] [])
    aie.objectfifo @A_S4(%shim_noc_tile_0_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo @A_C4(%mem_tile_4_1, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_S4] -> [@A_C4]([] [])
    aie.objectfifo @A_S5(%shim_noc_tile_0_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo @A_C5(%mem_tile_5_1, {%tile_5_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_S5] -> [@A_C5]([] [])
    aie.objectfifo @A_S6(%shim_noc_tile_0_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo @A_C6(%mem_tile_6_1, {%tile_6_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_S6] -> [@A_C6]([] [])
    aie.objectfifo @A_S7(%shim_noc_tile_0_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo @A_C7(%mem_tile_7_1, {%tile_7_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_S7] -> [@A_C7]([] [])
  }
}

