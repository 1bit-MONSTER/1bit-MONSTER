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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_1(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_2(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
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
        %c16 = arith.constant 16 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16 step %c1_3 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %1) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%3) : (memref<128x128xbf16>) -> ()
        %c0_4 = arith.constant 0 : index
        %c16_5 = arith.constant 16 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c16_5 step %c1_6 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %3) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%5) : (memref<128x128xbf16>) -> ()
        %c0_7 = arith.constant 0 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c16_8 step %c1_9 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %5) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%7) : (memref<128x128xbf16>) -> ()
        %c0_10 = arith.constant 0 : index
        %c16_11 = arith.constant 16 : index
        %c1_12 = arith.constant 1 : index
        scf.for %arg1 = %c0_10 to %c16_11 step %c1_12 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %7) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%9) : (memref<128x128xbf16>) -> ()
        %c0_13 = arith.constant 0 : index
        %c16_14 = arith.constant 16 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg1 = %c0_13 to %c16_14 step %c1_15 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %9) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%11) : (memref<128x128xbf16>) -> ()
        %c0_16 = arith.constant 0 : index
        %c16_17 = arith.constant 16 : index
        %c1_18 = arith.constant 1 : index
        scf.for %arg1 = %c0_16 to %c16_17 step %c1_18 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %11) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%13) : (memref<128x128xbf16>) -> ()
        %c0_19 = arith.constant 0 : index
        %c16_20 = arith.constant 16 : index
        %c1_21 = arith.constant 1 : index
        scf.for %arg1 = %c0_19 to %c16_20 step %c1_21 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %13) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
        func.call @zero_kernel_bf16(%15) : (memref<128x128xbf16>) -> ()
        %c0_22 = arith.constant 0 : index
        %c16_23 = arith.constant 16 : index
        %c1_24 = arith.constant 1 : index
        scf.for %arg1 = %c0_22 to %c16_23 step %c1_24 {
          %16 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          %18 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%19, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%21, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %22 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%23, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          %24 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @matmul_vectorized_different_datatypes(%25, %17, %15) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @A_L2L1_3(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<1048576xbf16>, %arg1: memref<524288x!aiex.bfp<"v8bfp16ebs8">>, %arg2: memref<4194304xbf16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 131072, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 262144, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 393216, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 16384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 32768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 49152, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 65536, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 81920, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 98304, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 114688, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 0, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 128, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 256, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 384, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 512, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 640, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 768, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 896, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 131072, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 262144, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 393216, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 131072, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 147456, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 163840, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 180224, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 196608, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 212992, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 229376, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 245760, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 1024, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 1152, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 1280, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 1408, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 1536, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 1664, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 1792, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 1920, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 131072, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 262144, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 393216, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 262144, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 278528, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 294912, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 311296, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 327680, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 344064, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 360448, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 376832, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2048, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2176, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2304, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2432, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2560, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2688, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2816, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2944, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 131072, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 262144, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 393216, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 393216, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 409600, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 425984, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 442368, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 458752, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 475136, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 491520, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 507904, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 3072, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 3200, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 3328, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 3456, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 3584, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 3712, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 3840, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 3968, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
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
      %80 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 524288, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 655360, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 786432, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 917504, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 16384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 32768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 49152, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 65536, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 81920, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 98304, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 114688, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2097152, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2097280, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2097408, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2097536, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2097664, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2097792, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2097920, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2098048, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 524288, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 655360, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 786432, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 917504, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 131072, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 147456, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 163840, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 180224, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 196608, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 212992, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 229376, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 245760, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2098176, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2098304, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2098432, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2098560, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2098688, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2098816, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2098944, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2099072, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%119)
      aiex.dma_await_task(%52)
      aiex.dma_await_task(%53)
      aiex.dma_await_task(%54)
      aiex.dma_await_task(%55)
      aiex.dma_await_task(%56)
      aiex.dma_await_task(%57)
      aiex.dma_await_task(%58)
      aiex.dma_await_task(%59)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%41)
      aiex.dma_free_task(%42)
      aiex.dma_free_task(%43)
      aiex.dma_free_task(%44)
      aiex.dma_free_task(%45)
      aiex.dma_free_task(%46)
      aiex.dma_free_task(%47)
      aiex.dma_free_task(%48)
      aiex.dma_free_task(%49)
      aiex.dma_free_task(%50)
      aiex.dma_free_task(%51)
      aiex.dma_await_task(%72)
      aiex.dma_await_task(%73)
      aiex.dma_await_task(%74)
      aiex.dma_await_task(%75)
      aiex.dma_await_task(%76)
      aiex.dma_await_task(%77)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%79)
      aiex.dma_free_task(%60)
      aiex.dma_free_task(%61)
      aiex.dma_free_task(%62)
      aiex.dma_free_task(%63)
      aiex.dma_free_task(%64)
      aiex.dma_free_task(%65)
      aiex.dma_free_task(%66)
      aiex.dma_free_task(%67)
      aiex.dma_free_task(%68)
      aiex.dma_free_task(%69)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%71)
      %120 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 524288, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 655360, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 786432, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 917504, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 262144, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 278528, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 294912, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 311296, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 327680, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 344064, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 360448, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 376832, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2099200, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2099328, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2099456, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2099584, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%135)
      %136 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2099712, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%136)
      %137 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2099840, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2099968, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2100096, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%139)
      %140 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 524288, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 655360, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 786432, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%142)
      %143 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 917504, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%143)
      %144 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 393216, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 409600, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 425984, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%146)
      %147 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 442368, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 458752, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 475136, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%149)
      %150 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 491520, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<524288x!aiex.bfp<"v8bfp16ebs8">>, 507904, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2100224, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%152)
      %153 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2100352, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2100480, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2100608, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%155)
      %156 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2100736, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%156)
      %157 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2100864, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2100992, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xbf16>, 2101120, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 4096>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%159)
      aiex.dma_await_task(%92)
      aiex.dma_await_task(%93)
      aiex.dma_await_task(%94)
      aiex.dma_await_task(%95)
      aiex.dma_await_task(%96)
      aiex.dma_await_task(%97)
      aiex.dma_await_task(%98)
      aiex.dma_await_task(%99)
      aiex.dma_free_task(%80)
      aiex.dma_free_task(%81)
      aiex.dma_free_task(%82)
      aiex.dma_free_task(%83)
      aiex.dma_free_task(%84)
      aiex.dma_free_task(%85)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%87)
      aiex.dma_free_task(%88)
      aiex.dma_free_task(%89)
      aiex.dma_free_task(%90)
      aiex.dma_free_task(%91)
      aiex.dma_await_task(%112)
      aiex.dma_await_task(%113)
      aiex.dma_await_task(%114)
      aiex.dma_await_task(%115)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%117)
      aiex.dma_await_task(%118)
      aiex.dma_await_task(%119)
      aiex.dma_free_task(%100)
      aiex.dma_free_task(%101)
      aiex.dma_free_task(%102)
      aiex.dma_free_task(%103)
      aiex.dma_free_task(%104)
      aiex.dma_free_task(%105)
      aiex.dma_free_task(%106)
      aiex.dma_free_task(%107)
      aiex.dma_free_task(%108)
      aiex.dma_free_task(%109)
      aiex.dma_free_task(%110)
      aiex.dma_free_task(%111)
      aiex.dma_await_task(%132)
      aiex.dma_await_task(%133)
      aiex.dma_await_task(%134)
      aiex.dma_await_task(%135)
      aiex.dma_await_task(%136)
      aiex.dma_await_task(%137)
      aiex.dma_await_task(%138)
      aiex.dma_await_task(%139)
      aiex.dma_free_task(%120)
      aiex.dma_free_task(%121)
      aiex.dma_free_task(%122)
      aiex.dma_free_task(%123)
      aiex.dma_free_task(%124)
      aiex.dma_free_task(%125)
      aiex.dma_free_task(%126)
      aiex.dma_free_task(%127)
      aiex.dma_free_task(%128)
      aiex.dma_free_task(%129)
      aiex.dma_free_task(%130)
      aiex.dma_free_task(%131)
      aiex.dma_await_task(%152)
      aiex.dma_await_task(%153)
      aiex.dma_await_task(%154)
      aiex.dma_await_task(%155)
      aiex.dma_await_task(%156)
      aiex.dma_await_task(%157)
      aiex.dma_await_task(%158)
      aiex.dma_await_task(%159)
      aiex.dma_free_task(%140)
      aiex.dma_free_task(%141)
      aiex.dma_free_task(%142)
      aiex.dma_free_task(%143)
      aiex.dma_free_task(%144)
      aiex.dma_free_task(%145)
      aiex.dma_free_task(%146)
      aiex.dma_free_task(%147)
      aiex.dma_free_task(%148)
      aiex.dma_free_task(%149)
      aiex.dma_free_task(%150)
      aiex.dma_free_task(%151)
    }
  }
}

