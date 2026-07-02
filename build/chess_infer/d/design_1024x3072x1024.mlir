module {
  aie.device(npu2) {
    func.func private @zero_kernel_bf16(memref<128x128xbf16>) attributes {link_with = "mm_128x64x128.o"}
    func.func private @matmul_vectorized_different_datatypes(memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) attributes {link_with = "mm_128x64x128.o"}
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
    %shim_noc_tile_7_0_0 = aie.tile(7, 0)
    %mem_tile_7_1_1 = aie.tile(7, 1)
    aie.objectfifo @A_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>> 
    aie.objectfifo @A_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_1_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_2_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_3_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_4_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_5_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_6_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_7_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0]([] [])
    aie.objectfifo @A_L3L2_1(%shim_noc_tile_2_0, {%mem_tile_2_1 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>> 
    aie.objectfifo @A_L2L1_1(%mem_tile_2_1 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_3 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_1_3 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_2_3 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_3_3 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_4_3 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_5_3 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_6_3 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_7_3 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_1]([] [])
    aie.objectfifo @A_L3L2_2(%shim_noc_tile_4_0, {%mem_tile_4_1 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>> 
    aie.objectfifo @A_L2L1_2(%mem_tile_4_1 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_1_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_2_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_3_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_4_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_5_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_6_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_7_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@A_L3L2_2] -> [@A_L2L1_2]([] [])
    aie.objectfifo @A_L3L2_3(%shim_noc_tile_6_0, {%mem_tile_6_1 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>> 
    aie.objectfifo @A_L2L1_3(%mem_tile_6_1 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_5 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_1_5 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_2_5 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_3_5 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_4_5 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_5_5 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_6_5 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_7_5 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@A_L3L2_3] -> [@A_L2L1_3]([] [])
    aie.objectfifo @B_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo @B_L2L1_0(%mem_tile_0_1, {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [])
    aie.objectfifo @B_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo @B_L2L1_1(%mem_tile_1_1, {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [])
    aie.objectfifo @B_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo @B_L2L1_2(%mem_tile_2_1, {%tile_2_2, %tile_2_3, %tile_2_4, %tile_2_5}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2]([] [])
    aie.objectfifo @B_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo @B_L2L1_3(%mem_tile_3_1, {%tile_3_2, %tile_3_3, %tile_3_4, %tile_3_5}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3]([] [])
    aie.objectfifo @B_L3L2_4(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo @B_L2L1_4(%mem_tile_4_1, {%tile_4_2, %tile_4_3, %tile_4_4, %tile_4_5}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo.link [@B_L3L2_4] -> [@B_L2L1_4]([] [])
    aie.objectfifo @B_L3L2_5(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo @B_L2L1_5(%mem_tile_5_1, {%tile_5_2, %tile_5_3, %tile_5_4, %tile_5_5}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo.link [@B_L3L2_5] -> [@B_L2L1_5]([] [])
    aie.objectfifo @B_L3L2_6(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo @B_L2L1_6(%mem_tile_6_1, {%tile_6_2, %tile_6_3, %tile_6_4, %tile_6_5}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo.link [@B_L3L2_6] -> [@B_L2L1_6]([] [])
    aie.objectfifo @B_L3L2_7(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo @B_L2L1_7(%mem_tile_7_1, {%tile_7_2, %tile_7_3, %tile_7_4, %tile_7_5}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> 
    aie.objectfifo.link [@B_L3L2_7] -> [@B_L2L1_7]([] [])
    aie.objectfifo @C_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_0_1(%tile_0_3, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_0_2(%tile_0_4, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_0_3(%tile_0_5, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L2L3_0(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_1_0(%tile_1_2, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_1_1(%tile_1_3, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_1_2(%tile_1_4, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_1_3(%tile_1_5, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L2L3_1(%mem_tile_1_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_2_0(%tile_2_2, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_2_1(%tile_2_3, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_2_2(%tile_2_4, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_2_3(%tile_2_5, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L2L3_2(%mem_tile_2_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@C_L1L2_2_0, @C_L1L2_2_1, @C_L1L2_2_2, @C_L1L2_2_3] -> [@C_L2L3_2]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_3_0(%tile_3_2, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_3_1(%tile_3_3, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_3_2(%tile_3_4, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_3_3(%tile_3_5, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L2L3_3(%mem_tile_3_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@C_L1L2_3_0, @C_L1L2_3_1, @C_L1L2_3_2, @C_L1L2_3_3] -> [@C_L2L3_3]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_4_0(%tile_4_2, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_4_1(%tile_4_3, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_4_2(%tile_4_4, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_4_3(%tile_4_5, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L2L3_4(%mem_tile_4_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@C_L1L2_4_0, @C_L1L2_4_1, @C_L1L2_4_2, @C_L1L2_4_3] -> [@C_L2L3_4]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_5_0(%tile_5_2, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_5_1(%tile_5_3, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_5_2(%tile_5_4, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_5_3(%tile_5_5, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L2L3_5(%mem_tile_5_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@C_L1L2_5_0, @C_L1L2_5_1, @C_L1L2_5_2, @C_L1L2_5_3] -> [@C_L2L3_5]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_6_0(%tile_6_2, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_6_1(%tile_6_3, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_6_2(%tile_6_4, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_6_3(%tile_6_5, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L2L3_6(%mem_tile_6_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@C_L1L2_6_0, @C_L1L2_6_1, @C_L1L2_6_2, @C_L1L2_6_3] -> [@C_L2L3_6]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_7_0(%tile_7_2, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_7_1(%tile_7_3, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_7_2(%tile_7_4, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L1L2_7_3(%tile_7_5, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>> 
    aie.objectfifo @C_L2L3_7(%mem_tile_7_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>> 
    aie.objectfifo.link [@C_L1L2_7_0, @C_L1L2_7_1, @C_L1L2_7_2, @C_L1L2_7_3] -> [@C_L2L3_7]([0, 16384, 32768, 49152] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_2 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_3 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_3 = aie.core(%tile_5_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_3 = aie.core(%tile_6_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_3 = aie.core(%tile_7_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_4 = aie.core(%tile_4_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_4 = aie.core(%tile_5_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_4 = aie.core(%tile_6_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_4 = aie.core(%tile_7_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_5 = aie.core(%tile_4_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_5 = aie.core(%tile_5_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%1) : (memref<128x128xbf16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c48 = arith.constant 48 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c48 step %c1_3 {
          %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c48_5 = arith.constant 48 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c48_5 step %c1_6 {
          %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %6 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%7, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %8 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%9, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %10 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%11, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %12 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%13, %5, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<3145728xbf16>, %arg1: memref<393216x!aiex.bfp<"v8bfp16ebs8">>, %arg2: memref<1048576xbf16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<3145728xbf16>, 0, 393216, [<size = 1, stride = 0>, <size = 6, stride = 512>, <size = 128, stride = 3072>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<3145728xbf16>, 393216, 393216, [<size = 1, stride = 0>, <size = 6, stride = 512>, <size = 128, stride = 3072>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<3145728xbf16>, 786432, 393216, [<size = 1, stride = 0>, <size = 6, stride = 512>, <size = 128, stride = 3072>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<3145728xbf16>, 1179648, 393216, [<size = 1, stride = 0>, <size = 6, stride = 512>, <size = 128, stride = 3072>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 0, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 49152, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 98304, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 147456, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 196608, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 245760, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 294912, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 344064, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 0, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 128, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 256, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 384, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 512, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 640, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 768, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 896, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<3145728xbf16>, 1572864, 393216, [<size = 1, stride = 0>, <size = 6, stride = 512>, <size = 128, stride = 3072>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<3145728xbf16>, 1966080, 393216, [<size = 1, stride = 0>, <size = 6, stride = 512>, <size = 128, stride = 3072>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<3145728xbf16>, 2359296, 393216, [<size = 1, stride = 0>, <size = 6, stride = 512>, <size = 128, stride = 3072>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<3145728xbf16>, 2752512, 393216, [<size = 1, stride = 0>, <size = 6, stride = 512>, <size = 128, stride = 3072>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 0, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 49152, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 98304, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 147456, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 196608, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 245760, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 294912, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<393216x!aiex.bfp<"v8bfp16ebs8">>, 344064, 49152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 49152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524288, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524416, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524544, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524672, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524800, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 524928, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 525056, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<1048576xbf16>, 525184, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%39)
    }
  }
}

