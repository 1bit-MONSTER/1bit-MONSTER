module {
  aie.device(npu2) {
    func.func private @zero_scalar_i16(memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
    func.func private @matmul_scalar_i8_i16(memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) attributes {link_with = "mm_32x64x128.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @A_L3L2(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x512xi8>> 
    aie.objectfifo @A_L2L1(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
    aie.objectfifo.link [@A_L3L2] -> [@A_L2L1]([] [])
    aie.objectfifo @B_L3L2(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo @B_L2L1(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x128xi8>> 
    aie.objectfifo.link [@B_L3L2] -> [@B_L2L1]([] [])
    aie.objectfifo @C_L1L2(%tile_0_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo @C_L2L3(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32x128xi16>> 
    aie.objectfifo.link [@C_L1L2] -> [@C_L2L3]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%1) : (memref<32x128xi16>) -> ()
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %2 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%3) : (memref<32x128xi16>) -> ()
        %c0_2 = arith.constant 0 : index
        %c16_3 = arith.constant 16 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c16_3 step %c1_4 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %3) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %4 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%5) : (memref<32x128xi16>) -> ()
        %c0_5 = arith.constant 0 : index
        %c16_6 = arith.constant 16 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c16_6 step %c1_7 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %5) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %6 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%7) : (memref<32x128xi16>) -> ()
        %c0_8 = arith.constant 0 : index
        %c16_9 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg1 = %c0_8 to %c16_9 step %c1_10 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %7) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %8 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%9) : (memref<32x128xi16>) -> ()
        %c0_11 = arith.constant 0 : index
        %c16_12 = arith.constant 16 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg1 = %c0_11 to %c16_12 step %c1_13 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %9) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %10 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%11) : (memref<32x128xi16>) -> ()
        %c0_14 = arith.constant 0 : index
        %c16_15 = arith.constant 16 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg1 = %c0_14 to %c16_15 step %c1_16 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %11) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %12 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%13) : (memref<32x128xi16>) -> ()
        %c0_17 = arith.constant 0 : index
        %c16_18 = arith.constant 16 : index
        %c1_19 = arith.constant 1 : index
        scf.for %arg1 = %c0_17 to %c16_18 step %c1_19 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %13) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %14 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%15) : (memref<32x128xi16>) -> ()
        %c0_20 = arith.constant 0 : index
        %c16_21 = arith.constant 16 : index
        %c1_22 = arith.constant 1 : index
        scf.for %arg1 = %c0_20 to %c16_21 step %c1_22 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %15) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %16 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%17) : (memref<32x128xi16>) -> ()
        %c0_23 = arith.constant 0 : index
        %c16_24 = arith.constant 16 : index
        %c1_25 = arith.constant 1 : index
        scf.for %arg1 = %c0_23 to %c16_24 step %c1_25 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %17) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %18 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%19) : (memref<32x128xi16>) -> ()
        %c0_26 = arith.constant 0 : index
        %c16_27 = arith.constant 16 : index
        %c1_28 = arith.constant 1 : index
        scf.for %arg1 = %c0_26 to %c16_27 step %c1_28 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %19) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %20 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%21) : (memref<32x128xi16>) -> ()
        %c0_29 = arith.constant 0 : index
        %c16_30 = arith.constant 16 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c16_30 step %c1_31 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %21) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %22 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%23) : (memref<32x128xi16>) -> ()
        %c0_32 = arith.constant 0 : index
        %c16_33 = arith.constant 16 : index
        %c1_34 = arith.constant 1 : index
        scf.for %arg1 = %c0_32 to %c16_33 step %c1_34 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %23) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %24 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%25) : (memref<32x128xi16>) -> ()
        %c0_35 = arith.constant 0 : index
        %c16_36 = arith.constant 16 : index
        %c1_37 = arith.constant 1 : index
        scf.for %arg1 = %c0_35 to %c16_36 step %c1_37 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %25) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %26 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%27) : (memref<32x128xi16>) -> ()
        %c0_38 = arith.constant 0 : index
        %c16_39 = arith.constant 16 : index
        %c1_40 = arith.constant 1 : index
        scf.for %arg1 = %c0_38 to %c16_39 step %c1_40 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %27) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %28 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%29) : (memref<32x128xi16>) -> ()
        %c0_41 = arith.constant 0 : index
        %c16_42 = arith.constant 16 : index
        %c1_43 = arith.constant 1 : index
        scf.for %arg1 = %c0_41 to %c16_42 step %c1_43 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %29) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %30 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%31) : (memref<32x128xi16>) -> ()
        %c0_44 = arith.constant 0 : index
        %c16_45 = arith.constant 16 : index
        %c1_46 = arith.constant 1 : index
        scf.for %arg1 = %c0_44 to %c16_45 step %c1_46 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %31) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %32 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%33) : (memref<32x128xi16>) -> ()
        %c0_47 = arith.constant 0 : index
        %c16_48 = arith.constant 16 : index
        %c1_49 = arith.constant 1 : index
        scf.for %arg1 = %c0_47 to %c16_48 step %c1_49 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %33) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %34 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%35) : (memref<32x128xi16>) -> ()
        %c0_50 = arith.constant 0 : index
        %c16_51 = arith.constant 16 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c16_51 step %c1_52 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %35) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %36 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%37) : (memref<32x128xi16>) -> ()
        %c0_53 = arith.constant 0 : index
        %c16_54 = arith.constant 16 : index
        %c1_55 = arith.constant 1 : index
        scf.for %arg1 = %c0_53 to %c16_54 step %c1_55 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %37) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %38 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%39) : (memref<32x128xi16>) -> ()
        %c0_56 = arith.constant 0 : index
        %c16_57 = arith.constant 16 : index
        %c1_58 = arith.constant 1 : index
        scf.for %arg1 = %c0_56 to %c16_57 step %c1_58 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %39) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %40 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%41) : (memref<32x128xi16>) -> ()
        %c0_59 = arith.constant 0 : index
        %c16_60 = arith.constant 16 : index
        %c1_61 = arith.constant 1 : index
        scf.for %arg1 = %c0_59 to %c16_60 step %c1_61 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %41) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %42 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%43) : (memref<32x128xi16>) -> ()
        %c0_62 = arith.constant 0 : index
        %c16_63 = arith.constant 16 : index
        %c1_64 = arith.constant 1 : index
        scf.for %arg1 = %c0_62 to %c16_63 step %c1_64 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %43) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %44 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%45) : (memref<32x128xi16>) -> ()
        %c0_65 = arith.constant 0 : index
        %c16_66 = arith.constant 16 : index
        %c1_67 = arith.constant 1 : index
        scf.for %arg1 = %c0_65 to %c16_66 step %c1_67 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %45) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %46 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%47) : (memref<32x128xi16>) -> ()
        %c0_68 = arith.constant 0 : index
        %c16_69 = arith.constant 16 : index
        %c1_70 = arith.constant 1 : index
        scf.for %arg1 = %c0_68 to %c16_69 step %c1_70 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %47) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %48 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%49) : (memref<32x128xi16>) -> ()
        %c0_71 = arith.constant 0 : index
        %c16_72 = arith.constant 16 : index
        %c1_73 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %c16_72 step %c1_73 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %49) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %50 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%51) : (memref<32x128xi16>) -> ()
        %c0_74 = arith.constant 0 : index
        %c16_75 = arith.constant 16 : index
        %c1_76 = arith.constant 1 : index
        scf.for %arg1 = %c0_74 to %c16_75 step %c1_76 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %51) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %52 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%53) : (memref<32x128xi16>) -> ()
        %c0_77 = arith.constant 0 : index
        %c16_78 = arith.constant 16 : index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_77 to %c16_78 step %c1_79 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %53) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %54 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%55) : (memref<32x128xi16>) -> ()
        %c0_80 = arith.constant 0 : index
        %c16_81 = arith.constant 16 : index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_80 to %c16_81 step %c1_82 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %55) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %56 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%57) : (memref<32x128xi16>) -> ()
        %c0_83 = arith.constant 0 : index
        %c16_84 = arith.constant 16 : index
        %c1_85 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %c16_84 step %c1_85 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %57) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %58 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%59) : (memref<32x128xi16>) -> ()
        %c0_86 = arith.constant 0 : index
        %c16_87 = arith.constant 16 : index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_86 to %c16_87 step %c1_88 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %59) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %60 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%61) : (memref<32x128xi16>) -> ()
        %c0_89 = arith.constant 0 : index
        %c16_90 = arith.constant 16 : index
        %c1_91 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %c16_90 step %c1_91 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %61) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
        %62 = aie.objectfifo.acquire @C_L1L2(Produce, 1) : !aie.objectfifosubview<memref<32x128xi16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<32x128xi16>> -> memref<32x128xi16>
        func.call @zero_scalar_i16(%63) : (memref<32x128xi16>) -> ()
        %c0_92 = arith.constant 0 : index
        %c16_93 = arith.constant 16 : index
        %c1_94 = arith.constant 1 : index
        scf.for %arg1 = %c0_92 to %c16_93 step %c1_94 {
          %64 = aie.objectfifo.acquire @A_L2L1(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          %66 = aie.objectfifo.acquire @B_L2L1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          func.call @matmul_scalar_i8_i16(%65, %67, %63) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1(Consume, 1)
          aie.objectfifo.release @B_L2L1(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence @seq(%arg0: memref<131072xi8>, %arg1: memref<1048576xi8>, %arg2: memref<131072xi16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      %3 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 128, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%5)
      aiex.dma_await_task(%3)
      aiex.dma_await_task(%4)
      aiex.dma_await_task(%5)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      %6 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 256, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%8)
      aiex.dma_await_task(%6)
      aiex.dma_await_task(%7)
      aiex.dma_await_task(%8)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      %9 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 384, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      aiex.dma_await_task(%9)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%11)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      %12 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 512, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%14)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      %15 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 640, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_free_task(%15)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      %18 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%20)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%20)
      %21 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%23)
      %24 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 32768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      aiex.dma_await_task(%24)
      aiex.dma_await_task(%25)
      aiex.dma_await_task(%26)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%25)
      aiex.dma_free_task(%26)
      %27 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 32896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      aiex.dma_await_task(%27)
      aiex.dma_await_task(%28)
      aiex.dma_await_task(%29)
      aiex.dma_free_task(%27)
      aiex.dma_free_task(%28)
      aiex.dma_free_task(%29)
      %30 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33024, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%32)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%31)
      aiex.dma_await_task(%32)
      aiex.dma_free_task(%30)
      aiex.dma_free_task(%31)
      aiex.dma_free_task(%32)
      %33 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33152, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%35)
      aiex.dma_await_task(%33)
      aiex.dma_await_task(%34)
      aiex.dma_await_task(%35)
      aiex.dma_free_task(%33)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      %36 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33280, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%38)
      aiex.dma_await_task(%36)
      aiex.dma_await_task(%37)
      aiex.dma_await_task(%38)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%38)
      %39 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33408, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%41)
      aiex.dma_await_task(%39)
      aiex.dma_await_task(%40)
      aiex.dma_await_task(%41)
      aiex.dma_free_task(%39)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%41)
      %42 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      aiex.dma_await_task(%42)
      aiex.dma_await_task(%43)
      aiex.dma_await_task(%44)
      aiex.dma_free_task(%42)
      aiex.dma_free_task(%43)
      aiex.dma_free_task(%44)
      %45 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 32768, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 33664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      aiex.dma_await_task(%45)
      aiex.dma_await_task(%46)
      aiex.dma_await_task(%47)
      aiex.dma_free_task(%45)
      aiex.dma_free_task(%46)
      aiex.dma_free_task(%47)
      %48 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65536, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%50)
      aiex.dma_await_task(%48)
      aiex.dma_await_task(%49)
      aiex.dma_await_task(%50)
      aiex.dma_free_task(%48)
      aiex.dma_free_task(%49)
      aiex.dma_free_task(%50)
      %51 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65664, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%53)
      aiex.dma_await_task(%51)
      aiex.dma_await_task(%52)
      aiex.dma_await_task(%53)
      aiex.dma_free_task(%51)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%53)
      %54 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65792, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%56)
      aiex.dma_await_task(%54)
      aiex.dma_await_task(%55)
      aiex.dma_await_task(%56)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%55)
      aiex.dma_free_task(%56)
      %57 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 65920, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%59)
      aiex.dma_await_task(%57)
      aiex.dma_await_task(%58)
      aiex.dma_await_task(%59)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%58)
      aiex.dma_free_task(%59)
      %60 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66048, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      aiex.dma_await_task(%60)
      aiex.dma_await_task(%61)
      aiex.dma_await_task(%62)
      aiex.dma_free_task(%60)
      aiex.dma_free_task(%61)
      aiex.dma_free_task(%62)
      %63 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66176, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      aiex.dma_await_task(%63)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%65)
      aiex.dma_free_task(%63)
      aiex.dma_free_task(%64)
      aiex.dma_free_task(%65)
      %66 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%68)
      aiex.dma_await_task(%66)
      aiex.dma_await_task(%67)
      aiex.dma_await_task(%68)
      aiex.dma_free_task(%66)
      aiex.dma_free_task(%67)
      aiex.dma_free_task(%68)
      %69 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 65536, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 66432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%71)
      aiex.dma_await_task(%69)
      aiex.dma_await_task(%70)
      aiex.dma_await_task(%71)
      aiex.dma_free_task(%69)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%71)
      %72 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98304, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%74)
      aiex.dma_await_task(%72)
      aiex.dma_await_task(%73)
      aiex.dma_await_task(%74)
      aiex.dma_free_task(%72)
      aiex.dma_free_task(%73)
      aiex.dma_free_task(%74)
      %75 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98432, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      aiex.dma_await_task(%75)
      aiex.dma_await_task(%76)
      aiex.dma_await_task(%77)
      aiex.dma_free_task(%75)
      aiex.dma_free_task(%76)
      aiex.dma_free_task(%77)
      %78 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98560, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%80)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%80)
      aiex.dma_free_task(%78)
      aiex.dma_free_task(%79)
      aiex.dma_free_task(%80)
      %81 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98688, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%83)
      aiex.dma_await_task(%81)
      aiex.dma_await_task(%82)
      aiex.dma_await_task(%83)
      aiex.dma_free_task(%81)
      aiex.dma_free_task(%82)
      aiex.dma_free_task(%83)
      %84 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98816, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%86)
      aiex.dma_await_task(%84)
      aiex.dma_await_task(%85)
      aiex.dma_await_task(%86)
      aiex.dma_free_task(%84)
      aiex.dma_free_task(%85)
      aiex.dma_free_task(%86)
      %87 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 98944, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%89)
      aiex.dma_await_task(%87)
      aiex.dma_await_task(%88)
      aiex.dma_await_task(%89)
      aiex.dma_free_task(%87)
      aiex.dma_free_task(%88)
      aiex.dma_free_task(%89)
      %90 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 99072, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%92)
      aiex.dma_await_task(%90)
      aiex.dma_await_task(%91)
      aiex.dma_await_task(%92)
      aiex.dma_free_task(%90)
      aiex.dma_free_task(%91)
      aiex.dma_free_task(%92)
      %93 = aiex.dma_configure_task_for @A_L3L2 {
        aie.dma_bd(%arg0 : memref<131072xi8>, 98304, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @C_L2L3 {
        aie.dma_bd(%arg2 : memref<131072xi16>, 99200, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%95)
      aiex.dma_await_task(%93)
      aiex.dma_await_task(%94)
      aiex.dma_await_task(%95)
      aiex.dma_free_task(%93)
      aiex.dma_free_task(%94)
      aiex.dma_free_task(%95)
    }
  }
}

