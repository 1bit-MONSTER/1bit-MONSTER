module {
  aie.device(npu2) {
    func.func private @zero_kernel_bf16(memref<128x128xbf16>) attributes {link_with = "mm_bf16_direct.o"}
    func.func private @matmul_vectorized_bf16(memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) attributes {link_with = "mm_bf16_direct.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
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
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_4_3 = aie.tile(4, 3)
    %tile_5_3 = aie.tile(5, 3)
    %tile_6_3 = aie.tile(6, 3)
    %tile_7_3 = aie.tile(7, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_4_4 = aie.tile(4, 4)
    %tile_5_4 = aie.tile(5, 4)
    %tile_6_4 = aie.tile(6, 4)
    %tile_7_4 = aie.tile(7, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %tile_4_5 = aie.tile(4, 5)
    %tile_5_5 = aie.tile(5, 5)
    %tile_6_5 = aie.tile(6, 5)
    %tile_7_5 = aie.tile(7, 5)
    aie.objectfifo @AL3_0(%shim_noc_tile_0_0, {%mem_tile_0_1 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>> 
    aie.objectfifo @AL1_0(%mem_tile_0_1 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_2 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_1_2 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_2_2 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_3_2 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_4_2 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_5_2 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_6_2 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_7_2 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@AL3_0] -> [@AL1_0]([] [])
    aie.objectfifo @AL3_1(%shim_noc_tile_2_0, {%mem_tile_2_1 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>> 
    aie.objectfifo @AL1_1(%mem_tile_2_1 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_3 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_1_3 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_2_3 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_3_3 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_4_3 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_5_3 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_6_3 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_7_3 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@AL3_1] -> [@AL1_1]([] [])
    aie.objectfifo @AL3_2(%shim_noc_tile_4_0, {%mem_tile_4_1 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>> 
    aie.objectfifo @AL1_2(%mem_tile_4_1 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_4 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_1_4 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_2_4 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_3_4 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_4_4 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_5_4 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_6_4 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_7_4 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@AL3_2] -> [@AL1_2]([] [])
    aie.objectfifo @AL3_3(%shim_noc_tile_6_0, {%mem_tile_6_1 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>> 
    aie.objectfifo @AL1_3(%mem_tile_6_1 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_5 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_1_5 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_2_5 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_3_5 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_4_5 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_5_5 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_6_5 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>], %tile_7_5 dimensionsFromStream [<size = 8, stride = 16>, <size = 16, stride = 128>, <size = 16, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@AL3_3] -> [@AL1_3]([] [])
    aie.objectfifo @BL3_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo @BL1_0(%mem_tile_0_1, {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 1 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo.link [@BL3_0] -> [@BL1_0]([] [])
    aie.objectfifo @BL3_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo @BL1_1(%mem_tile_1_1, {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 1 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo.link [@BL3_1] -> [@BL1_1]([] [])
    aie.objectfifo @BL3_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo @BL1_2(%mem_tile_2_1, {%tile_2_2, %tile_2_3, %tile_2_4, %tile_2_5}, 1 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo.link [@BL3_2] -> [@BL1_2]([] [])
    aie.objectfifo @BL3_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo @BL1_3(%mem_tile_3_1, {%tile_3_2, %tile_3_3, %tile_3_4, %tile_3_5}, 1 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo.link [@BL3_3] -> [@BL1_3]([] [])
    aie.objectfifo @BL3_4(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo @BL1_4(%mem_tile_4_1, {%tile_4_2, %tile_4_3, %tile_4_4, %tile_4_5}, 1 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo.link [@BL3_4] -> [@BL1_4]([] [])
    aie.objectfifo @BL3_5(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo @BL1_5(%mem_tile_5_1, {%tile_5_2, %tile_5_3, %tile_5_4, %tile_5_5}, 1 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo.link [@BL3_5] -> [@BL1_5]([] [])
    aie.objectfifo @BL3_6(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo @BL1_6(%mem_tile_6_1, {%tile_6_2, %tile_6_3, %tile_6_4, %tile_6_5}, 1 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo.link [@BL3_6] -> [@BL1_6]([] [])
    aie.objectfifo @BL3_7(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo @BL1_7(%mem_tile_7_1, {%tile_7_2, %tile_7_3, %tile_7_4, %tile_7_5}, 1 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo.link [@BL3_7] -> [@BL1_7]([] [])
    aie.objectfifo @CL1_0_0(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_0_1(%tile_0_3, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_0_2(%tile_0_4, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_0_3(%tile_0_5, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL3_0(%mem_tile_0_1 dimensionsToStream [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@CL1_0_0, @CL1_0_1, @CL1_0_2, @CL1_0_3] -> [@CL3_0]([0, 16384, 32768, 49152] [])
    aie.objectfifo @CL1_1_0(%tile_1_2, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_1_1(%tile_1_3, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_1_2(%tile_1_4, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_1_3(%tile_1_5, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL3_1(%mem_tile_1_1 dimensionsToStream [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@CL1_1_0, @CL1_1_1, @CL1_1_2, @CL1_1_3] -> [@CL3_1]([0, 16384, 32768, 49152] [])
    aie.objectfifo @CL1_2_0(%tile_2_2, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_2_1(%tile_2_3, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_2_2(%tile_2_4, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_2_3(%tile_2_5, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL3_2(%mem_tile_2_1 dimensionsToStream [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@CL1_2_0, @CL1_2_1, @CL1_2_2, @CL1_2_3] -> [@CL3_2]([0, 16384, 32768, 49152] [])
    aie.objectfifo @CL1_3_0(%tile_3_2, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_3_1(%tile_3_3, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_3_2(%tile_3_4, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_3_3(%tile_3_5, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL3_3(%mem_tile_3_1 dimensionsToStream [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>], {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@CL1_3_0, @CL1_3_1, @CL1_3_2, @CL1_3_3] -> [@CL3_3]([0, 16384, 32768, 49152] [])
    aie.objectfifo @CL1_4_0(%tile_4_2, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_4_1(%tile_4_3, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_4_2(%tile_4_4, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_4_3(%tile_4_5, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL3_4(%mem_tile_4_1 dimensionsToStream [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>], {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@CL1_4_0, @CL1_4_1, @CL1_4_2, @CL1_4_3] -> [@CL3_4]([0, 16384, 32768, 49152] [])
    aie.objectfifo @CL1_5_0(%tile_5_2, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_5_1(%tile_5_3, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_5_2(%tile_5_4, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_5_3(%tile_5_5, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL3_5(%mem_tile_5_1 dimensionsToStream [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>], {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@CL1_5_0, @CL1_5_1, @CL1_5_2, @CL1_5_3] -> [@CL3_5]([0, 16384, 32768, 49152] [])
    aie.objectfifo @CL1_6_0(%tile_6_2, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_6_1(%tile_6_3, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_6_2(%tile_6_4, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_6_3(%tile_6_5, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL3_6(%mem_tile_6_1 dimensionsToStream [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>], {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@CL1_6_0, @CL1_6_1, @CL1_6_2, @CL1_6_3] -> [@CL3_6]([0, 16384, 32768, 49152] [])
    aie.objectfifo @CL1_7_0(%tile_7_2, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_7_1(%tile_7_3, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_7_2(%tile_7_4, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL1_7_3(%tile_7_5, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @CL3_7(%mem_tile_7_1 dimensionsToStream [<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>], {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@CL1_7_0, @CL1_7_1, @CL1_7_2, @CL1_7_3] -> [@CL3_7]([0, 16384, 32768, 49152] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @BL1_0(Consume, 1)
          }
          aie.objectfifo.release @CL1_0_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @BL1_1(Consume, 1)
          }
          aie.objectfifo.release @CL1_1_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @BL1_2(Consume, 1)
          }
          aie.objectfifo.release @CL1_2_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @BL1_3(Consume, 1)
          }
          aie.objectfifo.release @CL1_3_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @BL1_4(Consume, 1)
          }
          aie.objectfifo.release @CL1_4_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_5_2 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @BL1_5(Consume, 1)
          }
          aie.objectfifo.release @CL1_5_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @BL1_6(Consume, 1)
          }
          aie.objectfifo.release @CL1_6_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_0(Consume, 1)
            aie.objectfifo.release @BL1_7(Consume, 1)
          }
          aie.objectfifo.release @CL1_7_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            aie.objectfifo.release @BL1_0(Consume, 1)
          }
          aie.objectfifo.release @CL1_0_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            aie.objectfifo.release @BL1_1(Consume, 1)
          }
          aie.objectfifo.release @CL1_1_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            aie.objectfifo.release @BL1_2(Consume, 1)
          }
          aie.objectfifo.release @CL1_2_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            aie.objectfifo.release @BL1_3(Consume, 1)
          }
          aie.objectfifo.release @CL1_3_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_4_3 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            aie.objectfifo.release @BL1_4(Consume, 1)
          }
          aie.objectfifo.release @CL1_4_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_5_3 = aie.core(%tile_5_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            aie.objectfifo.release @BL1_5(Consume, 1)
          }
          aie.objectfifo.release @CL1_5_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_6_3 = aie.core(%tile_6_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            aie.objectfifo.release @BL1_6(Consume, 1)
          }
          aie.objectfifo.release @CL1_6_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_7_3 = aie.core(%tile_7_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_1(Consume, 1)
            aie.objectfifo.release @BL1_7(Consume, 1)
          }
          aie.objectfifo.release @CL1_7_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            aie.objectfifo.release @BL1_0(Consume, 1)
          }
          aie.objectfifo.release @CL1_0_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            aie.objectfifo.release @BL1_1(Consume, 1)
          }
          aie.objectfifo.release @CL1_1_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            aie.objectfifo.release @BL1_2(Consume, 1)
          }
          aie.objectfifo.release @CL1_2_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            aie.objectfifo.release @BL1_3(Consume, 1)
          }
          aie.objectfifo.release @CL1_3_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_4_4 = aie.core(%tile_4_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            aie.objectfifo.release @BL1_4(Consume, 1)
          }
          aie.objectfifo.release @CL1_4_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_5_4 = aie.core(%tile_5_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            aie.objectfifo.release @BL1_5(Consume, 1)
          }
          aie.objectfifo.release @CL1_5_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_6_4 = aie.core(%tile_6_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            aie.objectfifo.release @BL1_6(Consume, 1)
          }
          aie.objectfifo.release @CL1_6_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_7_4 = aie.core(%tile_7_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_2(Consume, 1)
            aie.objectfifo.release @BL1_7(Consume, 1)
          }
          aie.objectfifo.release @CL1_7_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            aie.objectfifo.release @BL1_0(Consume, 1)
          }
          aie.objectfifo.release @CL1_0_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            aie.objectfifo.release @BL1_1(Consume, 1)
          }
          aie.objectfifo.release @CL1_1_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            aie.objectfifo.release @BL1_2(Consume, 1)
          }
          aie.objectfifo.release @CL1_2_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            aie.objectfifo.release @BL1_3(Consume, 1)
          }
          aie.objectfifo.release @CL1_3_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_4_5 = aie.core(%tile_4_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            aie.objectfifo.release @BL1_4(Consume, 1)
          }
          aie.objectfifo.release @CL1_4_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_5_5 = aie.core(%tile_5_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            aie.objectfifo.release @BL1_5(Consume, 1)
          }
          aie.objectfifo.release @CL1_5_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            aie.objectfifo.release @BL1_6(Consume, 1)
          }
          aie.objectfifo.release @CL1_6_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          %0 = aie.objectfifo.acquire @CL1_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16 step %c1_3 {
            %2 = aie.objectfifo.acquire @BL1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
            %4 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%5, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %6 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%7, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %8 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%9, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            %10 = aie.objectfifo.acquire @AL1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @matmul_vectorized_bf16(%11, %3, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @AL1_3(Consume, 1)
            aie.objectfifo.release @BL1_7(Consume, 1)
          }
          aie.objectfifo.release @CL1_7_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 2816 : i32}
    aie.runtime_sequence @seq(%arg0: memref<1048576xbf16>, %arg1: memref<1048576xbf16>, %arg2: memref<1048576xbf16>) {
      %0 = aiex.dma_configure_task_for @AL3_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @AL3_1 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 131072, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @AL3_2 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 262144, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @AL3_3 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 393216, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @BL3_0 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @BL3_1 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 128, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @BL3_2 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 256, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @BL3_3 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 384, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @BL3_4 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 512, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @BL3_5 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 640, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @BL3_6 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 768, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @BL3_7 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 896, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @CL3_0 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 0, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @CL3_1 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 128, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @CL3_2 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 256, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @CL3_3 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 384, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @CL3_4 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 512, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @CL3_5 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 640, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @CL3_6 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 768, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @CL3_7 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 896, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @AL3_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 524288, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @AL3_1 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 655360, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @AL3_2 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 786432, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @AL3_3 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 917504, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @BL3_0 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @BL3_1 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 128, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @BL3_2 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 256, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @BL3_3 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 384, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @BL3_4 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 512, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @BL3_5 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 640, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @BL3_6 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 768, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @BL3_7 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 896, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @CL3_0 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524288, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @CL3_1 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524416, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @CL3_2 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524544, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @CL3_3 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524672, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @CL3_4 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524800, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @CL3_5 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524928, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @CL3_6 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 525056, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @CL3_7 {
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
  }
}
