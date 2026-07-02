module {
  aie.device(npu2) {
    func.func private @zero_scalar_i16(memref<128x128xi16>) attributes {link_with = "mm_i8.o"}
    func.func private @matmul_scalar_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) attributes {link_with = "mm_i8.o"}
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
    aie.objectfifo @A_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xi8>> 
    aie.objectfifo @A_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%tile_0_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_1_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_2_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_3_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_4_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_5_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_6_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %tile_7_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0]([] [])
    aie.objectfifo @B_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @B_L2L1_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [])
    aie.objectfifo @B_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @B_L2L1_1(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [])
    aie.objectfifo @B_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @B_L2L1_2(%mem_tile_2_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2]([] [])
    aie.objectfifo @B_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @B_L2L1_3(%mem_tile_3_1, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3]([] [])
    aie.objectfifo @B_L3L2_4(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @B_L2L1_4(%mem_tile_4_1, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@B_L3L2_4] -> [@B_L2L1_4]([] [])
    aie.objectfifo @B_L3L2_5(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @B_L2L1_5(%mem_tile_5_1, {%tile_5_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@B_L3L2_5] -> [@B_L2L1_5]([] [])
    aie.objectfifo @B_L3L2_6(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @B_L2L1_6(%mem_tile_6_1, {%tile_6_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@B_L3L2_6] -> [@B_L2L1_6]([] [])
    aie.objectfifo @B_L3L2_7(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @B_L2L1_7(%mem_tile_7_1, {%tile_7_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@B_L3L2_7] -> [@B_L2L1_7]([] [])
    aie.objectfifo @C_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @C_L2L3_0(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@C_L1L2_0_0] -> [@C_L2L3_0]([0] [])
    aie.objectfifo @C_L1L2_1_0(%tile_1_2, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @C_L2L3_1(%mem_tile_1_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@C_L1L2_1_0] -> [@C_L2L3_1]([0] [])
    aie.objectfifo @C_L1L2_2_0(%tile_2_2, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @C_L2L3_2(%mem_tile_2_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@C_L1L2_2_0] -> [@C_L2L3_2]([0] [])
    aie.objectfifo @C_L1L2_3_0(%tile_3_2, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @C_L2L3_3(%mem_tile_3_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@C_L1L2_3_0] -> [@C_L2L3_3]([0] [])
    aie.objectfifo @C_L1L2_4_0(%tile_4_2, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @C_L2L3_4(%mem_tile_4_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@C_L1L2_4_0] -> [@C_L2L3_4]([0] [])
    aie.objectfifo @C_L1L2_5_0(%tile_5_2, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @C_L2L3_5(%mem_tile_5_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@C_L1L2_5_0] -> [@C_L2L3_5]([0] [])
    aie.objectfifo @C_L1L2_6_0(%tile_6_2, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @C_L2L3_6(%mem_tile_6_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@C_L1L2_6_0] -> [@C_L2L3_6]([0] [])
    aie.objectfifo @C_L1L2_7_0(%tile_7_2, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo @C_L2L3_7(%mem_tile_7_1 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<128x128xi16>> 
    aie.objectfifo.link [@C_L1L2_7_0] -> [@C_L2L3_7]([0] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %12 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<128x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %12 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<128x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %12 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<128x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %12 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<128x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %12 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<128x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %12 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %12 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<128x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %12 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<128x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %12 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<128x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %12 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<128x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %12 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<128x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %12 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %12 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<128x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %12 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<128x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %12 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<128x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %12 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<128x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %12 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<128x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %12 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %12 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<128x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %12 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<128x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %12 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<128x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %12 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<128x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %12 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<128x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %12 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %12 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<128x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %12 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<128x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %12 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<128x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %12 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<128x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %12 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<128x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %12 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %12 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<128x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %12 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<128x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %12 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<128x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %12 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<128x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %12 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<128x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %12 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %12 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<128x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %12 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<128x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %12 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<128x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %12 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<128x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %12 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<128x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %12 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<128x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %12 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<128x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %12 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<128x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %12 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<128x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %12 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<128x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %12 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<128x128xi16>> -> memref<128x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<128x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %12 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %14 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%15, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %16 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%17, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %18 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%19, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          %20 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%21, %13, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<128x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<131072xi8>, %arg1: memref<6291456xi8>, %arg2: memref<786432xi16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 16384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 32768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 49152, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 65536, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 81920, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 98304, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 114688, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 128, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 256, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 384, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 512, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 640, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 768, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 896, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 131072, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 147456, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 163840, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 180224, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 196608, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 212992, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 229376, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 245760, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1024, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1152, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1280, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1408, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1536, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1664, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1792, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1920, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 262144, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 278528, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 294912, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 311296, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 327680, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 344064, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 360448, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 376832, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2048, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2176, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2304, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2432, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2560, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2688, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2816, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2944, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 393216, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 409600, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 425984, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 442368, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 458752, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 475136, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 491520, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 507904, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3072, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3200, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3328, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3456, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3584, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3712, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3840, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3968, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
      %68 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 524288, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 540672, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 557056, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 573440, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 589824, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 606208, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 622592, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 638976, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4096, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4224, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4352, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4480, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4608, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4736, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4864, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4992, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 128, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 655360, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 671744, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 688128, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 704512, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 720896, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 737280, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 753664, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 770048, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5120, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5248, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5376, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5504, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5632, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5760, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5888, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 6016, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%101)
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
    }
  }
}

