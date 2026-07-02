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
    aie.objectfifo @A_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1 dimensionsFromStream [<size = 32, stride = 512>, <size = 512, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x512xi8>> 
    aie.objectfifo @A_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 32, stride = 512>, <size = 64, stride = 1>], {%tile_0_2 dimensionsFromStream [<size = 32, stride = 64>, <size = 64, stride = 1>], %tile_1_2 dimensionsFromStream [<size = 32, stride = 64>, <size = 64, stride = 1>], %tile_2_2 dimensionsFromStream [<size = 32, stride = 64>, <size = 64, stride = 1>], %tile_3_2 dimensionsFromStream [<size = 32, stride = 64>, <size = 64, stride = 1>], %tile_4_2 dimensionsFromStream [<size = 32, stride = 64>, <size = 64, stride = 1>], %tile_5_2 dimensionsFromStream [<size = 32, stride = 64>, <size = 64, stride = 1>], %tile_6_2 dimensionsFromStream [<size = 32, stride = 64>, <size = 64, stride = 1>], %tile_7_2 dimensionsFromStream [<size = 32, stride = 64>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xi8>> 
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
          %2 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %4 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
          %2 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %4 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
          %2 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %4 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
          %2 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %4 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
          %2 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %4 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
          %2 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %4 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
          %2 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %4 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
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
          %2 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x128xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x128xi8>> -> memref<64x128xi8>
          %4 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi8>> -> memref<32x64xi8>
          func.call @matmul_scalar_i8_i16(%5, %3, %1) : (memref<32x64xi8>, memref<64x128xi8>, memref<32x128xi16>) -> ()
          aie.objectfifo.release @A_L2L1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
        }
        aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<32768xi8>, %arg1: memref<1048576xi8>, %arg2: memref<32768xi16>) {
      %0 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<32768xi8>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 512>, <size = 32, stride = 1024>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<32768xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<32768xi16>, 128, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<32768xi16>, 256, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<32768xi16>, 384, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<32768xi16>, 512, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<32768xi16>, 640, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<32768xi16>, 768, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<32768xi16>, 896, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1024>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
  }
}

