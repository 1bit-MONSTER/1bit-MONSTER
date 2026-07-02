module {
  aie.device(npu2) {
    func.func private @zero_scalar_i16(memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
    func.func private @matmul_scalar_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
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
    aie.objectfifo @A_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1 dimensionsFromStream [<size = 32, stride = 64>, <size = 8, stride = 2048>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x512xi8>> 
    aie.objectfifo @A_L2L1_0(%mem_tile_0_1, {%tile_0_2, %tile_1_2, %tile_2_2, %tile_3_2, %tile_4_2, %tile_5_2, %tile_6_2, %tile_7_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
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
    aie.objectfifo @C_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @C_L2L3_0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@C_L1L2_0_0] -> [@C_L2L3_0]([0] [])
    aie.objectfifo @C_L1L2_1_0(%tile_1_2, {%mem_tile_1_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @C_L2L3_1(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@C_L1L2_1_0] -> [@C_L2L3_1]([0] [])
    aie.objectfifo @C_L1L2_2_0(%tile_2_2, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @C_L2L3_2(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@C_L1L2_2_0] -> [@C_L2L3_2]([0] [])
    aie.objectfifo @C_L1L2_3_0(%tile_3_2, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @C_L2L3_3(%mem_tile_3_1, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@C_L1L2_3_0] -> [@C_L2L3_3]([0] [])
    aie.objectfifo @C_L1L2_4_0(%tile_4_2, {%mem_tile_4_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @C_L2L3_4(%mem_tile_4_1, {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@C_L1L2_4_0] -> [@C_L2L3_4]([0] [])
    aie.objectfifo @C_L1L2_5_0(%tile_5_2, {%mem_tile_5_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @C_L2L3_5(%mem_tile_5_1, {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@C_L1L2_5_0] -> [@C_L2L3_5]([0] [])
    aie.objectfifo @C_L1L2_6_0(%tile_6_2, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @C_L2L3_6(%mem_tile_6_1, {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@C_L1L2_6_0] -> [@C_L2L3_6]([0] [])
    aie.objectfifo @C_L1L2_7_0(%tile_7_2, {%mem_tile_7_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @C_L2L3_7(%mem_tile_7_1, {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@C_L1L2_7_0] -> [@C_L2L3_7]([0] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<32x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<32x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%13) : (memref<32x128xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c16_18 = arith.constant 16 : index
        %c1_19 = arith.constant 1 : index
        scf.for %arg1 = %c0_17 to %c16_18 step %c1_19 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %13) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%15) : (memref<32x128xi16>) -> ()
        %c0_20 = arith.constant 0 : index
        %c16_21 = arith.constant 16 : index
        %c1_22 = arith.constant 1 : index
        scf.for %arg1 = %c0_20 to %c16_21 step %c1_22 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %15) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %16 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%17) : (memref<32x128xi16>) -> ()
        %c0_23 = arith.constant 0 : index
        %c16_24 = arith.constant 16 : index
        %c1_25 = arith.constant 1 : index
        scf.for %arg1 = %c0_23 to %c16_24 step %c1_25 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %17) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %18 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%19) : (memref<32x128xi16>) -> ()
        %c0_26 = arith.constant 0 : index
        %c16_27 = arith.constant 16 : index
        %c1_28 = arith.constant 1 : index
        scf.for %arg1 = %c0_26 to %c16_27 step %c1_28 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %19) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %20 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%21) : (memref<32x128xi16>) -> ()
        %c0_29 = arith.constant 0 : index
        %c16_30 = arith.constant 16 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c16_30 step %c1_31 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %21) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %22 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%23) : (memref<32x128xi16>) -> ()
        %c0_32 = arith.constant 0 : index
        %c16_33 = arith.constant 16 : index
        %c1_34 = arith.constant 1 : index
        scf.for %arg1 = %c0_32 to %c16_33 step %c1_34 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %23) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %24 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%25) : (memref<32x128xi16>) -> ()
        %c0_35 = arith.constant 0 : index
        %c16_36 = arith.constant 16 : index
        %c1_37 = arith.constant 1 : index
        scf.for %arg1 = %c0_35 to %c16_36 step %c1_37 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %25) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %26 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%27) : (memref<32x128xi16>) -> ()
        %c0_38 = arith.constant 0 : index
        %c16_39 = arith.constant 16 : index
        %c1_40 = arith.constant 1 : index
        scf.for %arg1 = %c0_38 to %c16_39 step %c1_40 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %27) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %28 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%29) : (memref<32x128xi16>) -> ()
        %c0_41 = arith.constant 0 : index
        %c16_42 = arith.constant 16 : index
        %c1_43 = arith.constant 1 : index
        scf.for %arg1 = %c0_41 to %c16_42 step %c1_43 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %29) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %30 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%31) : (memref<32x128xi16>) -> ()
        %c0_44 = arith.constant 0 : index
        %c16_45 = arith.constant 16 : index
        %c1_46 = arith.constant 1 : index
        scf.for %arg1 = %c0_44 to %c16_45 step %c1_46 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %31) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %32 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%33) : (memref<32x128xi16>) -> ()
        %c0_47 = arith.constant 0 : index
        %c16_48 = arith.constant 16 : index
        %c1_49 = arith.constant 1 : index
        scf.for %arg1 = %c0_47 to %c16_48 step %c1_49 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %33) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %34 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%35) : (memref<32x128xi16>) -> ()
        %c0_50 = arith.constant 0 : index
        %c16_51 = arith.constant 16 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c16_51 step %c1_52 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %35) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %36 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%37) : (memref<32x128xi16>) -> ()
        %c0_53 = arith.constant 0 : index
        %c16_54 = arith.constant 16 : index
        %c1_55 = arith.constant 1 : index
        scf.for %arg1 = %c0_53 to %c16_54 step %c1_55 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %37) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %38 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%39) : (memref<32x128xi16>) -> ()
        %c0_56 = arith.constant 0 : index
        %c16_57 = arith.constant 16 : index
        %c1_58 = arith.constant 1 : index
        scf.for %arg1 = %c0_56 to %c16_57 step %c1_58 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %39) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %40 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%41) : (memref<32x128xi16>) -> ()
        %c0_59 = arith.constant 0 : index
        %c16_60 = arith.constant 16 : index
        %c1_61 = arith.constant 1 : index
        scf.for %arg1 = %c0_59 to %c16_60 step %c1_61 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %41) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %42 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%43) : (memref<32x128xi16>) -> ()
        %c0_62 = arith.constant 0 : index
        %c16_63 = arith.constant 16 : index
        %c1_64 = arith.constant 1 : index
        scf.for %arg1 = %c0_62 to %c16_63 step %c1_64 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %43) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %44 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%45) : (memref<32x128xi16>) -> ()
        %c0_65 = arith.constant 0 : index
        %c16_66 = arith.constant 16 : index
        %c1_67 = arith.constant 1 : index
        scf.for %arg1 = %c0_65 to %c16_66 step %c1_67 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %45) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        %46 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%47) : (memref<32x128xi16>) -> ()
        %c0_68 = arith.constant 0 : index
        %c16_69 = arith.constant 16 : index
        %c1_70 = arith.constant 1 : index
        scf.for %arg1 = %c0_68 to %c16_69 step %c1_70 {
          %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %47) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<32x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<32x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%13) : (memref<32x128xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c16_18 = arith.constant 16 : index
        %c1_19 = arith.constant 1 : index
        scf.for %arg1 = %c0_17 to %c16_18 step %c1_19 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %13) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%15) : (memref<32x128xi16>) -> ()
        %c0_20 = arith.constant 0 : index
        %c16_21 = arith.constant 16 : index
        %c1_22 = arith.constant 1 : index
        scf.for %arg1 = %c0_20 to %c16_21 step %c1_22 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %15) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %16 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%17) : (memref<32x128xi16>) -> ()
        %c0_23 = arith.constant 0 : index
        %c16_24 = arith.constant 16 : index
        %c1_25 = arith.constant 1 : index
        scf.for %arg1 = %c0_23 to %c16_24 step %c1_25 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %17) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %18 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%19) : (memref<32x128xi16>) -> ()
        %c0_26 = arith.constant 0 : index
        %c16_27 = arith.constant 16 : index
        %c1_28 = arith.constant 1 : index
        scf.for %arg1 = %c0_26 to %c16_27 step %c1_28 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %19) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %20 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%21) : (memref<32x128xi16>) -> ()
        %c0_29 = arith.constant 0 : index
        %c16_30 = arith.constant 16 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c16_30 step %c1_31 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %21) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %22 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%23) : (memref<32x128xi16>) -> ()
        %c0_32 = arith.constant 0 : index
        %c16_33 = arith.constant 16 : index
        %c1_34 = arith.constant 1 : index
        scf.for %arg1 = %c0_32 to %c16_33 step %c1_34 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %23) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %24 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%25) : (memref<32x128xi16>) -> ()
        %c0_35 = arith.constant 0 : index
        %c16_36 = arith.constant 16 : index
        %c1_37 = arith.constant 1 : index
        scf.for %arg1 = %c0_35 to %c16_36 step %c1_37 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %25) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %26 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%27) : (memref<32x128xi16>) -> ()
        %c0_38 = arith.constant 0 : index
        %c16_39 = arith.constant 16 : index
        %c1_40 = arith.constant 1 : index
        scf.for %arg1 = %c0_38 to %c16_39 step %c1_40 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %27) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %28 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%29) : (memref<32x128xi16>) -> ()
        %c0_41 = arith.constant 0 : index
        %c16_42 = arith.constant 16 : index
        %c1_43 = arith.constant 1 : index
        scf.for %arg1 = %c0_41 to %c16_42 step %c1_43 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %29) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %30 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%31) : (memref<32x128xi16>) -> ()
        %c0_44 = arith.constant 0 : index
        %c16_45 = arith.constant 16 : index
        %c1_46 = arith.constant 1 : index
        scf.for %arg1 = %c0_44 to %c16_45 step %c1_46 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %31) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %32 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%33) : (memref<32x128xi16>) -> ()
        %c0_47 = arith.constant 0 : index
        %c16_48 = arith.constant 16 : index
        %c1_49 = arith.constant 1 : index
        scf.for %arg1 = %c0_47 to %c16_48 step %c1_49 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %33) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %34 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%35) : (memref<32x128xi16>) -> ()
        %c0_50 = arith.constant 0 : index
        %c16_51 = arith.constant 16 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c16_51 step %c1_52 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %35) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %36 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%37) : (memref<32x128xi16>) -> ()
        %c0_53 = arith.constant 0 : index
        %c16_54 = arith.constant 16 : index
        %c1_55 = arith.constant 1 : index
        scf.for %arg1 = %c0_53 to %c16_54 step %c1_55 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %37) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %38 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%39) : (memref<32x128xi16>) -> ()
        %c0_56 = arith.constant 0 : index
        %c16_57 = arith.constant 16 : index
        %c1_58 = arith.constant 1 : index
        scf.for %arg1 = %c0_56 to %c16_57 step %c1_58 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %39) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %40 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%41) : (memref<32x128xi16>) -> ()
        %c0_59 = arith.constant 0 : index
        %c16_60 = arith.constant 16 : index
        %c1_61 = arith.constant 1 : index
        scf.for %arg1 = %c0_59 to %c16_60 step %c1_61 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %41) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %42 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%43) : (memref<32x128xi16>) -> ()
        %c0_62 = arith.constant 0 : index
        %c16_63 = arith.constant 16 : index
        %c1_64 = arith.constant 1 : index
        scf.for %arg1 = %c0_62 to %c16_63 step %c1_64 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %43) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %44 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%45) : (memref<32x128xi16>) -> ()
        %c0_65 = arith.constant 0 : index
        %c16_66 = arith.constant 16 : index
        %c1_67 = arith.constant 1 : index
        scf.for %arg1 = %c0_65 to %c16_66 step %c1_67 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %45) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        %46 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%47) : (memref<32x128xi16>) -> ()
        %c0_68 = arith.constant 0 : index
        %c16_69 = arith.constant 16 : index
        %c1_70 = arith.constant 1 : index
        scf.for %arg1 = %c0_68 to %c16_69 step %c1_70 {
          %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %47) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<32x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<32x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%13) : (memref<32x128xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c16_18 = arith.constant 16 : index
        %c1_19 = arith.constant 1 : index
        scf.for %arg1 = %c0_17 to %c16_18 step %c1_19 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %13) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%15) : (memref<32x128xi16>) -> ()
        %c0_20 = arith.constant 0 : index
        %c16_21 = arith.constant 16 : index
        %c1_22 = arith.constant 1 : index
        scf.for %arg1 = %c0_20 to %c16_21 step %c1_22 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %15) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %16 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%17) : (memref<32x128xi16>) -> ()
        %c0_23 = arith.constant 0 : index
        %c16_24 = arith.constant 16 : index
        %c1_25 = arith.constant 1 : index
        scf.for %arg1 = %c0_23 to %c16_24 step %c1_25 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %17) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %18 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%19) : (memref<32x128xi16>) -> ()
        %c0_26 = arith.constant 0 : index
        %c16_27 = arith.constant 16 : index
        %c1_28 = arith.constant 1 : index
        scf.for %arg1 = %c0_26 to %c16_27 step %c1_28 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %19) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %20 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%21) : (memref<32x128xi16>) -> ()
        %c0_29 = arith.constant 0 : index
        %c16_30 = arith.constant 16 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c16_30 step %c1_31 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %21) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %22 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%23) : (memref<32x128xi16>) -> ()
        %c0_32 = arith.constant 0 : index
        %c16_33 = arith.constant 16 : index
        %c1_34 = arith.constant 1 : index
        scf.for %arg1 = %c0_32 to %c16_33 step %c1_34 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %23) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %24 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%25) : (memref<32x128xi16>) -> ()
        %c0_35 = arith.constant 0 : index
        %c16_36 = arith.constant 16 : index
        %c1_37 = arith.constant 1 : index
        scf.for %arg1 = %c0_35 to %c16_36 step %c1_37 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %25) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %26 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%27) : (memref<32x128xi16>) -> ()
        %c0_38 = arith.constant 0 : index
        %c16_39 = arith.constant 16 : index
        %c1_40 = arith.constant 1 : index
        scf.for %arg1 = %c0_38 to %c16_39 step %c1_40 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %27) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %28 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%29) : (memref<32x128xi16>) -> ()
        %c0_41 = arith.constant 0 : index
        %c16_42 = arith.constant 16 : index
        %c1_43 = arith.constant 1 : index
        scf.for %arg1 = %c0_41 to %c16_42 step %c1_43 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %29) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %30 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%31) : (memref<32x128xi16>) -> ()
        %c0_44 = arith.constant 0 : index
        %c16_45 = arith.constant 16 : index
        %c1_46 = arith.constant 1 : index
        scf.for %arg1 = %c0_44 to %c16_45 step %c1_46 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %31) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %32 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%33) : (memref<32x128xi16>) -> ()
        %c0_47 = arith.constant 0 : index
        %c16_48 = arith.constant 16 : index
        %c1_49 = arith.constant 1 : index
        scf.for %arg1 = %c0_47 to %c16_48 step %c1_49 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %33) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %34 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%35) : (memref<32x128xi16>) -> ()
        %c0_50 = arith.constant 0 : index
        %c16_51 = arith.constant 16 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c16_51 step %c1_52 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %35) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %36 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%37) : (memref<32x128xi16>) -> ()
        %c0_53 = arith.constant 0 : index
        %c16_54 = arith.constant 16 : index
        %c1_55 = arith.constant 1 : index
        scf.for %arg1 = %c0_53 to %c16_54 step %c1_55 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %37) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %38 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%39) : (memref<32x128xi16>) -> ()
        %c0_56 = arith.constant 0 : index
        %c16_57 = arith.constant 16 : index
        %c1_58 = arith.constant 1 : index
        scf.for %arg1 = %c0_56 to %c16_57 step %c1_58 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %39) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %40 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%41) : (memref<32x128xi16>) -> ()
        %c0_59 = arith.constant 0 : index
        %c16_60 = arith.constant 16 : index
        %c1_61 = arith.constant 1 : index
        scf.for %arg1 = %c0_59 to %c16_60 step %c1_61 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %41) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %42 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%43) : (memref<32x128xi16>) -> ()
        %c0_62 = arith.constant 0 : index
        %c16_63 = arith.constant 16 : index
        %c1_64 = arith.constant 1 : index
        scf.for %arg1 = %c0_62 to %c16_63 step %c1_64 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %43) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %44 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%45) : (memref<32x128xi16>) -> ()
        %c0_65 = arith.constant 0 : index
        %c16_66 = arith.constant 16 : index
        %c1_67 = arith.constant 1 : index
        scf.for %arg1 = %c0_65 to %c16_66 step %c1_67 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %45) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        %46 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%47) : (memref<32x128xi16>) -> ()
        %c0_68 = arith.constant 0 : index
        %c16_69 = arith.constant 16 : index
        %c1_70 = arith.constant 1 : index
        scf.for %arg1 = %c0_68 to %c16_69 step %c1_70 {
          %48 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %47) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<32x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<32x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%13) : (memref<32x128xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c16_18 = arith.constant 16 : index
        %c1_19 = arith.constant 1 : index
        scf.for %arg1 = %c0_17 to %c16_18 step %c1_19 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %13) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%15) : (memref<32x128xi16>) -> ()
        %c0_20 = arith.constant 0 : index
        %c16_21 = arith.constant 16 : index
        %c1_22 = arith.constant 1 : index
        scf.for %arg1 = %c0_20 to %c16_21 step %c1_22 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %15) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %16 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%17) : (memref<32x128xi16>) -> ()
        %c0_23 = arith.constant 0 : index
        %c16_24 = arith.constant 16 : index
        %c1_25 = arith.constant 1 : index
        scf.for %arg1 = %c0_23 to %c16_24 step %c1_25 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %17) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %18 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%19) : (memref<32x128xi16>) -> ()
        %c0_26 = arith.constant 0 : index
        %c16_27 = arith.constant 16 : index
        %c1_28 = arith.constant 1 : index
        scf.for %arg1 = %c0_26 to %c16_27 step %c1_28 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %19) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %20 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%21) : (memref<32x128xi16>) -> ()
        %c0_29 = arith.constant 0 : index
        %c16_30 = arith.constant 16 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c16_30 step %c1_31 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %21) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %22 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%23) : (memref<32x128xi16>) -> ()
        %c0_32 = arith.constant 0 : index
        %c16_33 = arith.constant 16 : index
        %c1_34 = arith.constant 1 : index
        scf.for %arg1 = %c0_32 to %c16_33 step %c1_34 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %23) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %24 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%25) : (memref<32x128xi16>) -> ()
        %c0_35 = arith.constant 0 : index
        %c16_36 = arith.constant 16 : index
        %c1_37 = arith.constant 1 : index
        scf.for %arg1 = %c0_35 to %c16_36 step %c1_37 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %25) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %26 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%27) : (memref<32x128xi16>) -> ()
        %c0_38 = arith.constant 0 : index
        %c16_39 = arith.constant 16 : index
        %c1_40 = arith.constant 1 : index
        scf.for %arg1 = %c0_38 to %c16_39 step %c1_40 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %27) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %28 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%29) : (memref<32x128xi16>) -> ()
        %c0_41 = arith.constant 0 : index
        %c16_42 = arith.constant 16 : index
        %c1_43 = arith.constant 1 : index
        scf.for %arg1 = %c0_41 to %c16_42 step %c1_43 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %29) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %30 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%31) : (memref<32x128xi16>) -> ()
        %c0_44 = arith.constant 0 : index
        %c16_45 = arith.constant 16 : index
        %c1_46 = arith.constant 1 : index
        scf.for %arg1 = %c0_44 to %c16_45 step %c1_46 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %31) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %32 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%33) : (memref<32x128xi16>) -> ()
        %c0_47 = arith.constant 0 : index
        %c16_48 = arith.constant 16 : index
        %c1_49 = arith.constant 1 : index
        scf.for %arg1 = %c0_47 to %c16_48 step %c1_49 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %33) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %34 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%35) : (memref<32x128xi16>) -> ()
        %c0_50 = arith.constant 0 : index
        %c16_51 = arith.constant 16 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c16_51 step %c1_52 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %35) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %36 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%37) : (memref<32x128xi16>) -> ()
        %c0_53 = arith.constant 0 : index
        %c16_54 = arith.constant 16 : index
        %c1_55 = arith.constant 1 : index
        scf.for %arg1 = %c0_53 to %c16_54 step %c1_55 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %37) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %38 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%39) : (memref<32x128xi16>) -> ()
        %c0_56 = arith.constant 0 : index
        %c16_57 = arith.constant 16 : index
        %c1_58 = arith.constant 1 : index
        scf.for %arg1 = %c0_56 to %c16_57 step %c1_58 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %39) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %40 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%41) : (memref<32x128xi16>) -> ()
        %c0_59 = arith.constant 0 : index
        %c16_60 = arith.constant 16 : index
        %c1_61 = arith.constant 1 : index
        scf.for %arg1 = %c0_59 to %c16_60 step %c1_61 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %41) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %42 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%43) : (memref<32x128xi16>) -> ()
        %c0_62 = arith.constant 0 : index
        %c16_63 = arith.constant 16 : index
        %c1_64 = arith.constant 1 : index
        scf.for %arg1 = %c0_62 to %c16_63 step %c1_64 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %43) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %44 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%45) : (memref<32x128xi16>) -> ()
        %c0_65 = arith.constant 0 : index
        %c16_66 = arith.constant 16 : index
        %c1_67 = arith.constant 1 : index
        scf.for %arg1 = %c0_65 to %c16_66 step %c1_67 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %45) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        %46 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%47) : (memref<32x128xi16>) -> ()
        %c0_68 = arith.constant 0 : index
        %c16_69 = arith.constant 16 : index
        %c1_70 = arith.constant 1 : index
        scf.for %arg1 = %c0_68 to %c16_69 step %c1_70 {
          %48 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %47) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<32x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<32x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%13) : (memref<32x128xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c16_18 = arith.constant 16 : index
        %c1_19 = arith.constant 1 : index
        scf.for %arg1 = %c0_17 to %c16_18 step %c1_19 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %13) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%15) : (memref<32x128xi16>) -> ()
        %c0_20 = arith.constant 0 : index
        %c16_21 = arith.constant 16 : index
        %c1_22 = arith.constant 1 : index
        scf.for %arg1 = %c0_20 to %c16_21 step %c1_22 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %15) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %16 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%17) : (memref<32x128xi16>) -> ()
        %c0_23 = arith.constant 0 : index
        %c16_24 = arith.constant 16 : index
        %c1_25 = arith.constant 1 : index
        scf.for %arg1 = %c0_23 to %c16_24 step %c1_25 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %17) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %18 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%19) : (memref<32x128xi16>) -> ()
        %c0_26 = arith.constant 0 : index
        %c16_27 = arith.constant 16 : index
        %c1_28 = arith.constant 1 : index
        scf.for %arg1 = %c0_26 to %c16_27 step %c1_28 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %19) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %20 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%21) : (memref<32x128xi16>) -> ()
        %c0_29 = arith.constant 0 : index
        %c16_30 = arith.constant 16 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c16_30 step %c1_31 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %21) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %22 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%23) : (memref<32x128xi16>) -> ()
        %c0_32 = arith.constant 0 : index
        %c16_33 = arith.constant 16 : index
        %c1_34 = arith.constant 1 : index
        scf.for %arg1 = %c0_32 to %c16_33 step %c1_34 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %23) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %24 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%25) : (memref<32x128xi16>) -> ()
        %c0_35 = arith.constant 0 : index
        %c16_36 = arith.constant 16 : index
        %c1_37 = arith.constant 1 : index
        scf.for %arg1 = %c0_35 to %c16_36 step %c1_37 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %25) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %26 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%27) : (memref<32x128xi16>) -> ()
        %c0_38 = arith.constant 0 : index
        %c16_39 = arith.constant 16 : index
        %c1_40 = arith.constant 1 : index
        scf.for %arg1 = %c0_38 to %c16_39 step %c1_40 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %27) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %28 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%29) : (memref<32x128xi16>) -> ()
        %c0_41 = arith.constant 0 : index
        %c16_42 = arith.constant 16 : index
        %c1_43 = arith.constant 1 : index
        scf.for %arg1 = %c0_41 to %c16_42 step %c1_43 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %29) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %30 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%31) : (memref<32x128xi16>) -> ()
        %c0_44 = arith.constant 0 : index
        %c16_45 = arith.constant 16 : index
        %c1_46 = arith.constant 1 : index
        scf.for %arg1 = %c0_44 to %c16_45 step %c1_46 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %31) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %32 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%33) : (memref<32x128xi16>) -> ()
        %c0_47 = arith.constant 0 : index
        %c16_48 = arith.constant 16 : index
        %c1_49 = arith.constant 1 : index
        scf.for %arg1 = %c0_47 to %c16_48 step %c1_49 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %33) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %34 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%35) : (memref<32x128xi16>) -> ()
        %c0_50 = arith.constant 0 : index
        %c16_51 = arith.constant 16 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c16_51 step %c1_52 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %35) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %36 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%37) : (memref<32x128xi16>) -> ()
        %c0_53 = arith.constant 0 : index
        %c16_54 = arith.constant 16 : index
        %c1_55 = arith.constant 1 : index
        scf.for %arg1 = %c0_53 to %c16_54 step %c1_55 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %37) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %38 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%39) : (memref<32x128xi16>) -> ()
        %c0_56 = arith.constant 0 : index
        %c16_57 = arith.constant 16 : index
        %c1_58 = arith.constant 1 : index
        scf.for %arg1 = %c0_56 to %c16_57 step %c1_58 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %39) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %40 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%41) : (memref<32x128xi16>) -> ()
        %c0_59 = arith.constant 0 : index
        %c16_60 = arith.constant 16 : index
        %c1_61 = arith.constant 1 : index
        scf.for %arg1 = %c0_59 to %c16_60 step %c1_61 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %41) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %42 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%43) : (memref<32x128xi16>) -> ()
        %c0_62 = arith.constant 0 : index
        %c16_63 = arith.constant 16 : index
        %c1_64 = arith.constant 1 : index
        scf.for %arg1 = %c0_62 to %c16_63 step %c1_64 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %43) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %44 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%45) : (memref<32x128xi16>) -> ()
        %c0_65 = arith.constant 0 : index
        %c16_66 = arith.constant 16 : index
        %c1_67 = arith.constant 1 : index
        scf.for %arg1 = %c0_65 to %c16_66 step %c1_67 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %45) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        %46 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%47) : (memref<32x128xi16>) -> ()
        %c0_68 = arith.constant 0 : index
        %c16_69 = arith.constant 16 : index
        %c1_70 = arith.constant 1 : index
        scf.for %arg1 = %c0_68 to %c16_69 step %c1_70 {
          %48 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %47) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<32x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<32x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%13) : (memref<32x128xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c16_18 = arith.constant 16 : index
        %c1_19 = arith.constant 1 : index
        scf.for %arg1 = %c0_17 to %c16_18 step %c1_19 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %13) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%15) : (memref<32x128xi16>) -> ()
        %c0_20 = arith.constant 0 : index
        %c16_21 = arith.constant 16 : index
        %c1_22 = arith.constant 1 : index
        scf.for %arg1 = %c0_20 to %c16_21 step %c1_22 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %15) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %16 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%17) : (memref<32x128xi16>) -> ()
        %c0_23 = arith.constant 0 : index
        %c16_24 = arith.constant 16 : index
        %c1_25 = arith.constant 1 : index
        scf.for %arg1 = %c0_23 to %c16_24 step %c1_25 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %17) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %18 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%19) : (memref<32x128xi16>) -> ()
        %c0_26 = arith.constant 0 : index
        %c16_27 = arith.constant 16 : index
        %c1_28 = arith.constant 1 : index
        scf.for %arg1 = %c0_26 to %c16_27 step %c1_28 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %19) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %20 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%21) : (memref<32x128xi16>) -> ()
        %c0_29 = arith.constant 0 : index
        %c16_30 = arith.constant 16 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c16_30 step %c1_31 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %21) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %22 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%23) : (memref<32x128xi16>) -> ()
        %c0_32 = arith.constant 0 : index
        %c16_33 = arith.constant 16 : index
        %c1_34 = arith.constant 1 : index
        scf.for %arg1 = %c0_32 to %c16_33 step %c1_34 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %23) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %24 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%25) : (memref<32x128xi16>) -> ()
        %c0_35 = arith.constant 0 : index
        %c16_36 = arith.constant 16 : index
        %c1_37 = arith.constant 1 : index
        scf.for %arg1 = %c0_35 to %c16_36 step %c1_37 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %25) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %26 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%27) : (memref<32x128xi16>) -> ()
        %c0_38 = arith.constant 0 : index
        %c16_39 = arith.constant 16 : index
        %c1_40 = arith.constant 1 : index
        scf.for %arg1 = %c0_38 to %c16_39 step %c1_40 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %27) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %28 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%29) : (memref<32x128xi16>) -> ()
        %c0_41 = arith.constant 0 : index
        %c16_42 = arith.constant 16 : index
        %c1_43 = arith.constant 1 : index
        scf.for %arg1 = %c0_41 to %c16_42 step %c1_43 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %29) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %30 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%31) : (memref<32x128xi16>) -> ()
        %c0_44 = arith.constant 0 : index
        %c16_45 = arith.constant 16 : index
        %c1_46 = arith.constant 1 : index
        scf.for %arg1 = %c0_44 to %c16_45 step %c1_46 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %31) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %32 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%33) : (memref<32x128xi16>) -> ()
        %c0_47 = arith.constant 0 : index
        %c16_48 = arith.constant 16 : index
        %c1_49 = arith.constant 1 : index
        scf.for %arg1 = %c0_47 to %c16_48 step %c1_49 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %33) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %34 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%35) : (memref<32x128xi16>) -> ()
        %c0_50 = arith.constant 0 : index
        %c16_51 = arith.constant 16 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c16_51 step %c1_52 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %35) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %36 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%37) : (memref<32x128xi16>) -> ()
        %c0_53 = arith.constant 0 : index
        %c16_54 = arith.constant 16 : index
        %c1_55 = arith.constant 1 : index
        scf.for %arg1 = %c0_53 to %c16_54 step %c1_55 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %37) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %38 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%39) : (memref<32x128xi16>) -> ()
        %c0_56 = arith.constant 0 : index
        %c16_57 = arith.constant 16 : index
        %c1_58 = arith.constant 1 : index
        scf.for %arg1 = %c0_56 to %c16_57 step %c1_58 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %39) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %40 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%41) : (memref<32x128xi16>) -> ()
        %c0_59 = arith.constant 0 : index
        %c16_60 = arith.constant 16 : index
        %c1_61 = arith.constant 1 : index
        scf.for %arg1 = %c0_59 to %c16_60 step %c1_61 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %41) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %42 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%43) : (memref<32x128xi16>) -> ()
        %c0_62 = arith.constant 0 : index
        %c16_63 = arith.constant 16 : index
        %c1_64 = arith.constant 1 : index
        scf.for %arg1 = %c0_62 to %c16_63 step %c1_64 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %43) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %44 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%45) : (memref<32x128xi16>) -> ()
        %c0_65 = arith.constant 0 : index
        %c16_66 = arith.constant 16 : index
        %c1_67 = arith.constant 1 : index
        scf.for %arg1 = %c0_65 to %c16_66 step %c1_67 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %45) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        %46 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%47) : (memref<32x128xi16>) -> ()
        %c0_68 = arith.constant 0 : index
        %c16_69 = arith.constant 16 : index
        %c1_70 = arith.constant 1 : index
        scf.for %arg1 = %c0_68 to %c16_69 step %c1_70 {
          %48 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %47) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<32x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<32x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%13) : (memref<32x128xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c16_18 = arith.constant 16 : index
        %c1_19 = arith.constant 1 : index
        scf.for %arg1 = %c0_17 to %c16_18 step %c1_19 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %13) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%15) : (memref<32x128xi16>) -> ()
        %c0_20 = arith.constant 0 : index
        %c16_21 = arith.constant 16 : index
        %c1_22 = arith.constant 1 : index
        scf.for %arg1 = %c0_20 to %c16_21 step %c1_22 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %15) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %16 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%17) : (memref<32x128xi16>) -> ()
        %c0_23 = arith.constant 0 : index
        %c16_24 = arith.constant 16 : index
        %c1_25 = arith.constant 1 : index
        scf.for %arg1 = %c0_23 to %c16_24 step %c1_25 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %17) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %18 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%19) : (memref<32x128xi16>) -> ()
        %c0_26 = arith.constant 0 : index
        %c16_27 = arith.constant 16 : index
        %c1_28 = arith.constant 1 : index
        scf.for %arg1 = %c0_26 to %c16_27 step %c1_28 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %19) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %20 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%21) : (memref<32x128xi16>) -> ()
        %c0_29 = arith.constant 0 : index
        %c16_30 = arith.constant 16 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c16_30 step %c1_31 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %21) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %22 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%23) : (memref<32x128xi16>) -> ()
        %c0_32 = arith.constant 0 : index
        %c16_33 = arith.constant 16 : index
        %c1_34 = arith.constant 1 : index
        scf.for %arg1 = %c0_32 to %c16_33 step %c1_34 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %23) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %24 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%25) : (memref<32x128xi16>) -> ()
        %c0_35 = arith.constant 0 : index
        %c16_36 = arith.constant 16 : index
        %c1_37 = arith.constant 1 : index
        scf.for %arg1 = %c0_35 to %c16_36 step %c1_37 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %25) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %26 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%27) : (memref<32x128xi16>) -> ()
        %c0_38 = arith.constant 0 : index
        %c16_39 = arith.constant 16 : index
        %c1_40 = arith.constant 1 : index
        scf.for %arg1 = %c0_38 to %c16_39 step %c1_40 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %27) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %28 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%29) : (memref<32x128xi16>) -> ()
        %c0_41 = arith.constant 0 : index
        %c16_42 = arith.constant 16 : index
        %c1_43 = arith.constant 1 : index
        scf.for %arg1 = %c0_41 to %c16_42 step %c1_43 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %29) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %30 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%31) : (memref<32x128xi16>) -> ()
        %c0_44 = arith.constant 0 : index
        %c16_45 = arith.constant 16 : index
        %c1_46 = arith.constant 1 : index
        scf.for %arg1 = %c0_44 to %c16_45 step %c1_46 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %31) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %32 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%33) : (memref<32x128xi16>) -> ()
        %c0_47 = arith.constant 0 : index
        %c16_48 = arith.constant 16 : index
        %c1_49 = arith.constant 1 : index
        scf.for %arg1 = %c0_47 to %c16_48 step %c1_49 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %33) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %34 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%35) : (memref<32x128xi16>) -> ()
        %c0_50 = arith.constant 0 : index
        %c16_51 = arith.constant 16 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c16_51 step %c1_52 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %35) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %36 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%37) : (memref<32x128xi16>) -> ()
        %c0_53 = arith.constant 0 : index
        %c16_54 = arith.constant 16 : index
        %c1_55 = arith.constant 1 : index
        scf.for %arg1 = %c0_53 to %c16_54 step %c1_55 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %37) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %38 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%39) : (memref<32x128xi16>) -> ()
        %c0_56 = arith.constant 0 : index
        %c16_57 = arith.constant 16 : index
        %c1_58 = arith.constant 1 : index
        scf.for %arg1 = %c0_56 to %c16_57 step %c1_58 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %39) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %40 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%41) : (memref<32x128xi16>) -> ()
        %c0_59 = arith.constant 0 : index
        %c16_60 = arith.constant 16 : index
        %c1_61 = arith.constant 1 : index
        scf.for %arg1 = %c0_59 to %c16_60 step %c1_61 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %41) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %42 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%43) : (memref<32x128xi16>) -> ()
        %c0_62 = arith.constant 0 : index
        %c16_63 = arith.constant 16 : index
        %c1_64 = arith.constant 1 : index
        scf.for %arg1 = %c0_62 to %c16_63 step %c1_64 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %43) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %44 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%45) : (memref<32x128xi16>) -> ()
        %c0_65 = arith.constant 0 : index
        %c16_66 = arith.constant 16 : index
        %c1_67 = arith.constant 1 : index
        scf.for %arg1 = %c0_65 to %c16_66 step %c1_67 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %45) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        %46 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%47) : (memref<32x128xi16>) -> ()
        %c0_68 = arith.constant 0 : index
        %c16_69 = arith.constant 16 : index
        %c1_70 = arith.constant 1 : index
        scf.for %arg1 = %c0_68 to %c16_69 step %c1_70 {
          %48 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %47) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
        %0 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<32x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<32x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%13) : (memref<32x128xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c16_18 = arith.constant 16 : index
        %c1_19 = arith.constant 1 : index
        scf.for %arg1 = %c0_17 to %c16_18 step %c1_19 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %13) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%15) : (memref<32x128xi16>) -> ()
        %c0_20 = arith.constant 0 : index
        %c16_21 = arith.constant 16 : index
        %c1_22 = arith.constant 1 : index
        scf.for %arg1 = %c0_20 to %c16_21 step %c1_22 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %15) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %16 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%17) : (memref<32x128xi16>) -> ()
        %c0_23 = arith.constant 0 : index
        %c16_24 = arith.constant 16 : index
        %c1_25 = arith.constant 1 : index
        scf.for %arg1 = %c0_23 to %c16_24 step %c1_25 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %17) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %18 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%19) : (memref<32x128xi16>) -> ()
        %c0_26 = arith.constant 0 : index
        %c16_27 = arith.constant 16 : index
        %c1_28 = arith.constant 1 : index
        scf.for %arg1 = %c0_26 to %c16_27 step %c1_28 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %19) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %20 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%21) : (memref<32x128xi16>) -> ()
        %c0_29 = arith.constant 0 : index
        %c16_30 = arith.constant 16 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c16_30 step %c1_31 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %21) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %22 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%23) : (memref<32x128xi16>) -> ()
        %c0_32 = arith.constant 0 : index
        %c16_33 = arith.constant 16 : index
        %c1_34 = arith.constant 1 : index
        scf.for %arg1 = %c0_32 to %c16_33 step %c1_34 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %23) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %24 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%25) : (memref<32x128xi16>) -> ()
        %c0_35 = arith.constant 0 : index
        %c16_36 = arith.constant 16 : index
        %c1_37 = arith.constant 1 : index
        scf.for %arg1 = %c0_35 to %c16_36 step %c1_37 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %25) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %26 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%27) : (memref<32x128xi16>) -> ()
        %c0_38 = arith.constant 0 : index
        %c16_39 = arith.constant 16 : index
        %c1_40 = arith.constant 1 : index
        scf.for %arg1 = %c0_38 to %c16_39 step %c1_40 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %27) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %28 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%29) : (memref<32x128xi16>) -> ()
        %c0_41 = arith.constant 0 : index
        %c16_42 = arith.constant 16 : index
        %c1_43 = arith.constant 1 : index
        scf.for %arg1 = %c0_41 to %c16_42 step %c1_43 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %29) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %30 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%31) : (memref<32x128xi16>) -> ()
        %c0_44 = arith.constant 0 : index
        %c16_45 = arith.constant 16 : index
        %c1_46 = arith.constant 1 : index
        scf.for %arg1 = %c0_44 to %c16_45 step %c1_46 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %31) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %32 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%33) : (memref<32x128xi16>) -> ()
        %c0_47 = arith.constant 0 : index
        %c16_48 = arith.constant 16 : index
        %c1_49 = arith.constant 1 : index
        scf.for %arg1 = %c0_47 to %c16_48 step %c1_49 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %33) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %34 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%35) : (memref<32x128xi16>) -> ()
        %c0_50 = arith.constant 0 : index
        %c16_51 = arith.constant 16 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c16_51 step %c1_52 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %35) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %36 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%37) : (memref<32x128xi16>) -> ()
        %c0_53 = arith.constant 0 : index
        %c16_54 = arith.constant 16 : index
        %c1_55 = arith.constant 1 : index
        scf.for %arg1 = %c0_53 to %c16_54 step %c1_55 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %37) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %38 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%39) : (memref<32x128xi16>) -> ()
        %c0_56 = arith.constant 0 : index
        %c16_57 = arith.constant 16 : index
        %c1_58 = arith.constant 1 : index
        scf.for %arg1 = %c0_56 to %c16_57 step %c1_58 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %39) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %40 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%41) : (memref<32x128xi16>) -> ()
        %c0_59 = arith.constant 0 : index
        %c16_60 = arith.constant 16 : index
        %c1_61 = arith.constant 1 : index
        scf.for %arg1 = %c0_59 to %c16_60 step %c1_61 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %41) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %42 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%43) : (memref<32x128xi16>) -> ()
        %c0_62 = arith.constant 0 : index
        %c16_63 = arith.constant 16 : index
        %c1_64 = arith.constant 1 : index
        scf.for %arg1 = %c0_62 to %c16_63 step %c1_64 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %43) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %44 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%45) : (memref<32x128xi16>) -> ()
        %c0_65 = arith.constant 0 : index
        %c16_66 = arith.constant 16 : index
        %c1_67 = arith.constant 1 : index
        scf.for %arg1 = %c0_65 to %c16_66 step %c1_67 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %45) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        %46 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%47) : (memref<32x128xi16>) -> ()
        %c0_68 = arith.constant 0 : index
        %c16_69 = arith.constant 16 : index
        %c1_70 = arith.constant 1 : index
        scf.for %arg1 = %c0_68 to %c16_69 step %c1_70 {
          %48 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %50 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%51, %49, %47) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<131072xi8>, %arg1: memref<6291456xi8>, %arg2: memref<786432xi16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 128, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 256, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 384, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 512, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 640, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1048576, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1179648, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1310720, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1441792, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1572864, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1703936, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1835008, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1966080, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1024, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1152, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1280, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1408, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1792, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 1920, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2097152, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2228224, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2359296, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2490368, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2621440, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2752512, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2883584, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3014656, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2048, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2176, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2560, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2688, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2816, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 2944, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3145728, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3276800, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3407872, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3538944, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3670016, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3801088, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3932160, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4063232, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3072, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3200, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3328, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3456, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3584, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3712, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3840, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 3968, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
      %68 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4194304, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4325376, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4456448, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4587520, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4718592, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4849664, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4980736, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5111808, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4096, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4224, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4352, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4480, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4608, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4736, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4864, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 4992, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5242880, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5373952, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5505024, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5636096, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5767168, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5898240, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6029312, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6160384, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5120, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5248, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5376, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5504, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5632, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5760, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 5888, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 6016, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
      aiex.dma_await_task(%60)
      aiex.dma_await_task(%61)
      aiex.dma_await_task(%62)
      aiex.dma_await_task(%63)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%65)
      aiex.dma_await_task(%66)
      aiex.dma_await_task(%67)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%39)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%41)
      aiex.dma_free_task(%42)
      aiex.dma_free_task(%51)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%55)
      aiex.dma_free_task(%56)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%58)
      aiex.dma_free_task(%59)
      %102 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 196608, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 196736, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 196864, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 196992, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197120, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197248, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197376, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197504, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1048576, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1179648, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1310720, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1441792, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1572864, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1703936, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1835008, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1966080, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197632, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197760, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 197888, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198016, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198144, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198272, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198400, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198528, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%135)
      aiex.dma_await_task(%77)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%80)
      aiex.dma_await_task(%81)
      aiex.dma_await_task(%82)
      aiex.dma_await_task(%83)
      aiex.dma_await_task(%84)
      aiex.dma_await_task(%94)
      aiex.dma_await_task(%95)
      aiex.dma_await_task(%96)
      aiex.dma_await_task(%97)
      aiex.dma_await_task(%98)
      aiex.dma_await_task(%99)
      aiex.dma_await_task(%100)
      aiex.dma_await_task(%101)
      aiex.dma_free_task(%68)
      aiex.dma_free_task(%69)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%71)
      aiex.dma_free_task(%72)
      aiex.dma_free_task(%73)
      aiex.dma_free_task(%74)
      aiex.dma_free_task(%75)
      aiex.dma_free_task(%76)
      aiex.dma_free_task(%85)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%87)
      aiex.dma_free_task(%88)
      aiex.dma_free_task(%89)
      aiex.dma_free_task(%90)
      aiex.dma_free_task(%91)
      aiex.dma_free_task(%92)
      aiex.dma_free_task(%93)
      %136 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%136)
      %137 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2097152, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2228224, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2359296, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%139)
      %140 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2490368, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2621440, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2752512, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%142)
      %143 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2883584, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%143)
      %144 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3014656, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198656, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198784, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%146)
      %147 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 198912, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199040, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199168, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%149)
      %150 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199296, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199424, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199552, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%152)
      %153 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3145728, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3276800, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%155)
      %156 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3407872, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%156)
      %157 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3538944, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3670016, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3801088, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%159)
      %160 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3932160, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%160)
      %161 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4063232, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%161)
      %162 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199680, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%162)
      %163 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199808, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%163)
      %164 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 199936, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%164)
      %165 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200064, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%165)
      %166 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200192, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%166)
      %167 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200320, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%167)
      %168 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200448, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%168)
      %169 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200576, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%169)
      aiex.dma_await_task(%111)
      aiex.dma_await_task(%112)
      aiex.dma_await_task(%113)
      aiex.dma_await_task(%114)
      aiex.dma_await_task(%115)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%117)
      aiex.dma_await_task(%118)
      aiex.dma_await_task(%128)
      aiex.dma_await_task(%129)
      aiex.dma_await_task(%130)
      aiex.dma_await_task(%131)
      aiex.dma_await_task(%132)
      aiex.dma_await_task(%133)
      aiex.dma_await_task(%134)
      aiex.dma_await_task(%135)
      aiex.dma_free_task(%102)
      aiex.dma_free_task(%103)
      aiex.dma_free_task(%104)
      aiex.dma_free_task(%105)
      aiex.dma_free_task(%106)
      aiex.dma_free_task(%107)
      aiex.dma_free_task(%108)
      aiex.dma_free_task(%109)
      aiex.dma_free_task(%110)
      aiex.dma_free_task(%119)
      aiex.dma_free_task(%120)
      aiex.dma_free_task(%121)
      aiex.dma_free_task(%122)
      aiex.dma_free_task(%123)
      aiex.dma_free_task(%124)
      aiex.dma_free_task(%125)
      aiex.dma_free_task(%126)
      aiex.dma_free_task(%127)
      %170 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%170)
      %171 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4194304, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%171)
      %172 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4325376, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%172)
      %173 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4456448, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%173)
      %174 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4587520, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%174)
      %175 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4718592, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%175)
      %176 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4849664, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%176)
      %177 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4980736, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%177)
      %178 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5111808, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%178)
      %179 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200704, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%179)
      %180 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200832, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%180)
      %181 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 200960, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%181)
      %182 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201088, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%182)
      %183 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201216, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%183)
      %184 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201344, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%184)
      %185 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201472, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%185)
      %186 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201600, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%186)
      %187 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%187)
      %188 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5242880, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%188)
      %189 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5373952, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%189)
      %190 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5505024, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%190)
      %191 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5636096, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%191)
      %192 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5767168, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%192)
      %193 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5898240, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%193)
      %194 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6029312, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%194)
      %195 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6160384, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%195)
      %196 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201728, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%196)
      %197 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201856, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%197)
      %198 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 201984, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%198)
      %199 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 202112, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%199)
      %200 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 202240, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%200)
      %201 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 202368, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%201)
      %202 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 202496, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%202)
      %203 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 202624, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%203)
      aiex.dma_await_task(%145)
      aiex.dma_await_task(%146)
      aiex.dma_await_task(%147)
      aiex.dma_await_task(%148)
      aiex.dma_await_task(%149)
      aiex.dma_await_task(%150)
      aiex.dma_await_task(%151)
      aiex.dma_await_task(%152)
      aiex.dma_await_task(%162)
      aiex.dma_await_task(%163)
      aiex.dma_await_task(%164)
      aiex.dma_await_task(%165)
      aiex.dma_await_task(%166)
      aiex.dma_await_task(%167)
      aiex.dma_await_task(%168)
      aiex.dma_await_task(%169)
      aiex.dma_free_task(%136)
      aiex.dma_free_task(%137)
      aiex.dma_free_task(%138)
      aiex.dma_free_task(%139)
      aiex.dma_free_task(%140)
      aiex.dma_free_task(%141)
      aiex.dma_free_task(%142)
      aiex.dma_free_task(%143)
      aiex.dma_free_task(%144)
      aiex.dma_free_task(%153)
      aiex.dma_free_task(%154)
      aiex.dma_free_task(%155)
      aiex.dma_free_task(%156)
      aiex.dma_free_task(%157)
      aiex.dma_free_task(%158)
      aiex.dma_free_task(%159)
      aiex.dma_free_task(%160)
      aiex.dma_free_task(%161)
      %204 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%204)
      %205 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%205)
      %206 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%206)
      %207 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%207)
      %208 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%208)
      %209 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%209)
      %210 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%210)
      %211 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%211)
      %212 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%212)
      %213 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393216, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%213)
      %214 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393344, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%214)
      %215 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393472, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%215)
      %216 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393600, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%216)
      %217 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393728, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%217)
      %218 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393856, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%218)
      %219 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 393984, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%219)
      %220 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394112, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%220)
      %221 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%221)
      %222 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1048576, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%222)
      %223 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1179648, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%223)
      %224 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1310720, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%224)
      %225 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1441792, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%225)
      %226 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1572864, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%226)
      %227 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1703936, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%227)
      %228 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1835008, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%228)
      %229 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1966080, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%229)
      %230 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394240, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%230)
      %231 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394368, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%231)
      %232 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394496, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%232)
      %233 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394624, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%233)
      %234 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394752, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%234)
      %235 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 394880, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%235)
      %236 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395008, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%236)
      %237 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395136, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%237)
      aiex.dma_await_task(%179)
      aiex.dma_await_task(%180)
      aiex.dma_await_task(%181)
      aiex.dma_await_task(%182)
      aiex.dma_await_task(%183)
      aiex.dma_await_task(%184)
      aiex.dma_await_task(%185)
      aiex.dma_await_task(%186)
      aiex.dma_await_task(%196)
      aiex.dma_await_task(%197)
      aiex.dma_await_task(%198)
      aiex.dma_await_task(%199)
      aiex.dma_await_task(%200)
      aiex.dma_await_task(%201)
      aiex.dma_await_task(%202)
      aiex.dma_await_task(%203)
      aiex.dma_free_task(%170)
      aiex.dma_free_task(%171)
      aiex.dma_free_task(%172)
      aiex.dma_free_task(%173)
      aiex.dma_free_task(%174)
      aiex.dma_free_task(%175)
      aiex.dma_free_task(%176)
      aiex.dma_free_task(%177)
      aiex.dma_free_task(%178)
      aiex.dma_free_task(%187)
      aiex.dma_free_task(%188)
      aiex.dma_free_task(%189)
      aiex.dma_free_task(%190)
      aiex.dma_free_task(%191)
      aiex.dma_free_task(%192)
      aiex.dma_free_task(%193)
      aiex.dma_free_task(%194)
      aiex.dma_free_task(%195)
      %238 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%238)
      %239 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2097152, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%239)
      %240 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2228224, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%240)
      %241 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2359296, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%241)
      %242 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2490368, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%242)
      %243 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2621440, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%243)
      %244 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2752512, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%244)
      %245 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2883584, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%245)
      %246 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3014656, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%246)
      %247 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395264, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%247)
      %248 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395392, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%248)
      %249 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395520, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%249)
      %250 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395648, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%250)
      %251 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395776, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%251)
      %252 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 395904, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%252)
      %253 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396032, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%253)
      %254 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396160, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%254)
      %255 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%255)
      %256 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3145728, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%256)
      %257 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3276800, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%257)
      %258 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3407872, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%258)
      %259 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3538944, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%259)
      %260 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3670016, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%260)
      %261 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3801088, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%261)
      %262 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3932160, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%262)
      %263 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4063232, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%263)
      %264 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396288, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%264)
      %265 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396416, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%265)
      %266 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396544, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%266)
      %267 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396672, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%267)
      %268 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396800, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%268)
      %269 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 396928, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%269)
      %270 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397056, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%270)
      %271 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397184, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%271)
      aiex.dma_await_task(%213)
      aiex.dma_await_task(%214)
      aiex.dma_await_task(%215)
      aiex.dma_await_task(%216)
      aiex.dma_await_task(%217)
      aiex.dma_await_task(%218)
      aiex.dma_await_task(%219)
      aiex.dma_await_task(%220)
      aiex.dma_await_task(%230)
      aiex.dma_await_task(%231)
      aiex.dma_await_task(%232)
      aiex.dma_await_task(%233)
      aiex.dma_await_task(%234)
      aiex.dma_await_task(%235)
      aiex.dma_await_task(%236)
      aiex.dma_await_task(%237)
      aiex.dma_free_task(%204)
      aiex.dma_free_task(%205)
      aiex.dma_free_task(%206)
      aiex.dma_free_task(%207)
      aiex.dma_free_task(%208)
      aiex.dma_free_task(%209)
      aiex.dma_free_task(%210)
      aiex.dma_free_task(%211)
      aiex.dma_free_task(%212)
      aiex.dma_free_task(%221)
      aiex.dma_free_task(%222)
      aiex.dma_free_task(%223)
      aiex.dma_free_task(%224)
      aiex.dma_free_task(%225)
      aiex.dma_free_task(%226)
      aiex.dma_free_task(%227)
      aiex.dma_free_task(%228)
      aiex.dma_free_task(%229)
      %272 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%272)
      %273 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4194304, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%273)
      %274 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4325376, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%274)
      %275 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4456448, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%275)
      %276 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4587520, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%276)
      %277 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4718592, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%277)
      %278 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4849664, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%278)
      %279 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4980736, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%279)
      %280 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5111808, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%280)
      %281 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397312, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%281)
      %282 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397440, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%282)
      %283 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397568, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%283)
      %284 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397696, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%284)
      %285 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397824, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%285)
      %286 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 397952, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%286)
      %287 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398080, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%287)
      %288 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398208, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%288)
      %289 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%289)
      %290 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5242880, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%290)
      %291 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5373952, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%291)
      %292 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5505024, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%292)
      %293 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5636096, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%293)
      %294 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5767168, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%294)
      %295 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5898240, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%295)
      %296 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6029312, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%296)
      %297 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6160384, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%297)
      %298 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398336, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%298)
      %299 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398464, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%299)
      %300 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398592, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%300)
      %301 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398720, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%301)
      %302 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398848, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%302)
      %303 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 398976, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%303)
      %304 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 399104, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%304)
      %305 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 399232, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%305)
      aiex.dma_await_task(%247)
      aiex.dma_await_task(%248)
      aiex.dma_await_task(%249)
      aiex.dma_await_task(%250)
      aiex.dma_await_task(%251)
      aiex.dma_await_task(%252)
      aiex.dma_await_task(%253)
      aiex.dma_await_task(%254)
      aiex.dma_await_task(%264)
      aiex.dma_await_task(%265)
      aiex.dma_await_task(%266)
      aiex.dma_await_task(%267)
      aiex.dma_await_task(%268)
      aiex.dma_await_task(%269)
      aiex.dma_await_task(%270)
      aiex.dma_await_task(%271)
      aiex.dma_free_task(%238)
      aiex.dma_free_task(%239)
      aiex.dma_free_task(%240)
      aiex.dma_free_task(%241)
      aiex.dma_free_task(%242)
      aiex.dma_free_task(%243)
      aiex.dma_free_task(%244)
      aiex.dma_free_task(%245)
      aiex.dma_free_task(%246)
      aiex.dma_free_task(%255)
      aiex.dma_free_task(%256)
      aiex.dma_free_task(%257)
      aiex.dma_free_task(%258)
      aiex.dma_free_task(%259)
      aiex.dma_free_task(%260)
      aiex.dma_free_task(%261)
      aiex.dma_free_task(%262)
      aiex.dma_free_task(%263)
      %306 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%306)
      %307 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%307)
      %308 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%308)
      %309 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%309)
      %310 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%310)
      %311 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%311)
      %312 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%312)
      %313 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%313)
      %314 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%314)
      %315 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 589824, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%315)
      %316 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 589952, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%316)
      %317 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590080, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%317)
      %318 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590208, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%318)
      %319 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590336, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%319)
      %320 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590464, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%320)
      %321 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590592, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%321)
      %322 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590720, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%322)
      %323 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%323)
      %324 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1048576, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%324)
      %325 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1179648, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%325)
      %326 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1310720, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%326)
      %327 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1441792, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%327)
      %328 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1572864, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%328)
      %329 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1703936, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%329)
      %330 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1835008, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%330)
      %331 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 1966080, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%331)
      %332 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590848, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%332)
      %333 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 590976, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%333)
      %334 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591104, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%334)
      %335 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591232, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%335)
      %336 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591360, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%336)
      %337 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591488, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%337)
      %338 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591616, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%338)
      %339 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591744, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%339)
      aiex.dma_await_task(%281)
      aiex.dma_await_task(%282)
      aiex.dma_await_task(%283)
      aiex.dma_await_task(%284)
      aiex.dma_await_task(%285)
      aiex.dma_await_task(%286)
      aiex.dma_await_task(%287)
      aiex.dma_await_task(%288)
      aiex.dma_await_task(%298)
      aiex.dma_await_task(%299)
      aiex.dma_await_task(%300)
      aiex.dma_await_task(%301)
      aiex.dma_await_task(%302)
      aiex.dma_await_task(%303)
      aiex.dma_await_task(%304)
      aiex.dma_await_task(%305)
      aiex.dma_free_task(%272)
      aiex.dma_free_task(%273)
      aiex.dma_free_task(%274)
      aiex.dma_free_task(%275)
      aiex.dma_free_task(%276)
      aiex.dma_free_task(%277)
      aiex.dma_free_task(%278)
      aiex.dma_free_task(%279)
      aiex.dma_free_task(%280)
      aiex.dma_free_task(%289)
      aiex.dma_free_task(%290)
      aiex.dma_free_task(%291)
      aiex.dma_free_task(%292)
      aiex.dma_free_task(%293)
      aiex.dma_free_task(%294)
      aiex.dma_free_task(%295)
      aiex.dma_free_task(%296)
      aiex.dma_free_task(%297)
      %340 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%340)
      %341 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2097152, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%341)
      %342 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2228224, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%342)
      %343 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2359296, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%343)
      %344 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2490368, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%344)
      %345 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2621440, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%345)
      %346 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2752512, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%346)
      %347 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 2883584, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%347)
      %348 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3014656, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%348)
      %349 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 591872, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%349)
      %350 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592000, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%350)
      %351 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592128, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%351)
      %352 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592256, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%352)
      %353 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592384, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%353)
      %354 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592512, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%354)
      %355 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592640, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%355)
      %356 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%356)
      %357 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%357)
      %358 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3145728, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%358)
      %359 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3276800, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%359)
      %360 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3407872, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%360)
      %361 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3538944, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%361)
      %362 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3670016, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%362)
      %363 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3801088, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%363)
      %364 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 3932160, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%364)
      %365 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4063232, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%365)
      %366 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 592896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%366)
      %367 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593024, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%367)
      %368 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593152, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%368)
      %369 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593280, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%369)
      %370 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593408, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%370)
      %371 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%371)
      %372 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%372)
      %373 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593792, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%373)
      aiex.dma_await_task(%315)
      aiex.dma_await_task(%316)
      aiex.dma_await_task(%317)
      aiex.dma_await_task(%318)
      aiex.dma_await_task(%319)
      aiex.dma_await_task(%320)
      aiex.dma_await_task(%321)
      aiex.dma_await_task(%322)
      aiex.dma_await_task(%332)
      aiex.dma_await_task(%333)
      aiex.dma_await_task(%334)
      aiex.dma_await_task(%335)
      aiex.dma_await_task(%336)
      aiex.dma_await_task(%337)
      aiex.dma_await_task(%338)
      aiex.dma_await_task(%339)
      aiex.dma_free_task(%306)
      aiex.dma_free_task(%307)
      aiex.dma_free_task(%308)
      aiex.dma_free_task(%309)
      aiex.dma_free_task(%310)
      aiex.dma_free_task(%311)
      aiex.dma_free_task(%312)
      aiex.dma_free_task(%313)
      aiex.dma_free_task(%314)
      aiex.dma_free_task(%323)
      aiex.dma_free_task(%324)
      aiex.dma_free_task(%325)
      aiex.dma_free_task(%326)
      aiex.dma_free_task(%327)
      aiex.dma_free_task(%328)
      aiex.dma_free_task(%329)
      aiex.dma_free_task(%330)
      aiex.dma_free_task(%331)
      %374 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%374)
      %375 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4194304, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%375)
      %376 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4325376, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%376)
      %377 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4456448, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%377)
      %378 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4587520, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%378)
      %379 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4718592, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%379)
      %380 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4849664, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%380)
      %381 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 4980736, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%381)
      %382 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5111808, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%382)
      %383 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 593920, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%383)
      %384 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594048, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%384)
      %385 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594176, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%385)
      %386 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%386)
      %387 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%387)
      %388 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594560, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%388)
      %389 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594688, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%389)
      %390 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594816, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%390)
      %391 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%391)
      %392 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5242880, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%392)
      %393 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5373952, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%393)
      %394 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5505024, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%394)
      %395 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5636096, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%395)
      %396 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5767168, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%396)
      %397 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 5898240, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%397)
      %398 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6029312, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%398)
      %399 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<6291456xi8>, 6160384, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%399)
      %400 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 594944, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%400)
      %401 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595072, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%401)
      %402 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595200, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%402)
      %403 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595328, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%403)
      %404 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595456, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%404)
      %405 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595584, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%405)
      %406 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595712, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%406)
      %407 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<786432xi16>, 595840, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 6144>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%407)
      aiex.dma_await_task(%349)
      aiex.dma_await_task(%350)
      aiex.dma_await_task(%351)
      aiex.dma_await_task(%352)
      aiex.dma_await_task(%353)
      aiex.dma_await_task(%354)
      aiex.dma_await_task(%355)
      aiex.dma_await_task(%356)
      aiex.dma_await_task(%366)
      aiex.dma_await_task(%367)
      aiex.dma_await_task(%368)
      aiex.dma_await_task(%369)
      aiex.dma_await_task(%370)
      aiex.dma_await_task(%371)
      aiex.dma_await_task(%372)
      aiex.dma_await_task(%373)
      aiex.dma_free_task(%340)
      aiex.dma_free_task(%341)
      aiex.dma_free_task(%342)
      aiex.dma_free_task(%343)
      aiex.dma_free_task(%344)
      aiex.dma_free_task(%345)
      aiex.dma_free_task(%346)
      aiex.dma_free_task(%347)
      aiex.dma_free_task(%348)
      aiex.dma_free_task(%357)
      aiex.dma_free_task(%358)
      aiex.dma_free_task(%359)
      aiex.dma_free_task(%360)
      aiex.dma_free_task(%361)
      aiex.dma_free_task(%362)
      aiex.dma_free_task(%363)
      aiex.dma_free_task(%364)
      aiex.dma_free_task(%365)
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
  }
}

